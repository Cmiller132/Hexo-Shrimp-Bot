"""The deck's HTTP contract over telemetry and engine state."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from mantisnet import MantisConfig, MantisNet
from mantisnet.deck.app import create_app
from mantisnet.klent import telemetry
from mantisnet.klent.inspect import inspect_position
from mantisnet.klent.run import _versions, save_checkpoint

from .test_telemetry import _fixture_db


@pytest.fixture
def deck_run(tmp_path):
    runs = tmp_path / "runs"
    run = runs / "fixture"
    run.mkdir(parents=True)
    writer = _fixture_db(run)
    writer.close()
    config = {
        "iterations": 1,
        "checkpoint_every": 1,
        "klent": {"tau": 0.1, "lam": 0.03},
        "model": {},
        "versions": {"fixture": True},
    }
    (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run / "invocations.jsonl").write_text(
        json.dumps({"start_iteration": 0, "versions": config["versions"]}) + "\n",
        encoding="utf-8",
    )
    return runs, run


def test_registry_and_every_query_endpoint(deck_run, tmp_path):
    runs, _run = deck_run
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<div id=root></div>", encoding="utf-8")
    with TestClient(create_app(runs, dist, device="cpu")) as client:
        listed = client.get("/api/runs").json()
        assert listed[0]["name"] == "fixture"
        assert listed[0]["state"] == "completed"
        assert client.get("/api/runs/fixture").status_code == 200
        assert client.get("/api/runs/fixture/manifest").json()["invocations"]
        assert client.get("/api/runs/fixture/summary").json()["iterations"] == 1
        event_snapshot = client.get("/api/runs/fixture/events?once=true")
        assert "event: iteration" in event_snapshot.text
        assert '"iteration":0' in event_snapshot.text

        endpoints = [
            "/api/runs/fixture/iterations?columns=games,plies",
            "/api/runs/fixture/games",
            "/api/runs/fixture/games/0",
            "/api/runs/fixture/calibration",
            "/api/runs/fixture/blunders?threshold=0",
            "/api/runs/fixture/openings?plies=1",
            "/api/runs/fixture/strength",
            "/api/runs/fixture/crossplay",
        ]
        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200, (endpoint, response.text)
            assert isinstance(response.json(), (list, dict))
        game = client.get("/api/runs/fixture/games/0").json()
        assert game["moves"] and game["plies"]
        assert client.put(
            "/api/runs/fixture/games/0/review",
            json={"tags": ["inspect"], "note": "value swing"},
        ).json()["note"] == "value swing"
        assert client.get("/api/runs/fixture/games/0").json()["review"]["tags"] == [
            "inspect"
        ]

        assert client.post("/api/runs/fixture/checkpoint").status_code == 202
        assert (runs / "fixture" / "CHECKPOINT").exists()
        assert client.post("/api/runs/fixture/stop").status_code == 202
        assert (runs / "fixture" / "STOP").exists()
        assert client.get("/missing/deep/link").text == "<div id=root></div>"


def test_play_session_engine_rejects_an_occupied_cell(deck_run, tmp_path):
    runs, _run = deck_run
    with TestClient(create_app(runs, tmp_path / "missing", device="cpu")) as client:
        session = client.post(
            "/api/play", json={"seats": [{"kind": "human"}, {"kind": "human"}]}
        ).json()
        assert session["legal_moves"] == [[0, 0]]
        advanced = client.post(
            f"/api/play/{session['session_id']}/moves", json={"move": [0, 0]}
        )
        assert advanced.status_code == 200
        illegal = client.post(
            f"/api/play/{session['session_id']}/moves", json={"move": [0, 0]}
        )
        assert illegal.status_code == 400
        assert illegal.json()["error"]["code"] == "illegal_move"
        assert illegal.json()["error"]["message"]


def test_checkpoint_match_plays_paired_openings_and_records_both_seats(deck_run):
    runs, run = deck_run
    cfg = MantisConfig(
        h=32, blocks=1, heads=2, value_queries=2, value_bins=5,
        policy_hidden=32, value_hidden=32,
    )
    paths = []
    for index, seed in enumerate((7, 8), start=1):
        torch.manual_seed(seed)
        model = MantisNet(cfg)
        path = run / f"checkpoint_00000{index}.pt"
        save_checkpoint(
            path, model, torch.optim.Adam(model.parameters()), index,
            np.random.default_rng(seed),
        )
        paths.append(path)

    with TestClient(create_app(runs, device="cpu")) as client:
        launched = client.post(
            "/api/matches",
            json={
                "checkpoint_a": str(paths[0]),
                "checkpoint_b": str(paths[1]),
                "opponent": "checkpoint",
                "games": 4,
                "ply_cap": 20,
            },
        )
        assert launched.status_code == 202, launched.text
        job_id = launched.json()["job_id"]
        for _attempt in range(600):
            job = client.get(f"/api/matches/{job_id}").json()
            if job["status"] in {"completed", "failed"}:
                break
            time.sleep(0.1)
        assert job["status"] == "completed", job.get("error")
        result = job["result"]
        assert result["games"] == 4
        # The seat split partitions the total score, and every game has plies.
        assert result["score_as_p0"] + result["score_as_p1"] == result["score_a"]
        assert result["avg_plies"] > 0
        assert result["match_id"] == 1

    # An odd game count cannot be seat-paired.
    with TestClient(create_app(runs, device="cpu")) as client:
        refused = client.post(
            "/api/matches",
            json={
                "checkpoint_a": str(paths[0]),
                "checkpoint_b": str(paths[1]),
                "opponent": "checkpoint",
                "games": 3,
            },
        )
        assert refused.status_code == 400
        assert refused.json()["error"]["code"] == "invalid_match"


def test_inspect_endpoint_equals_direct_inspection(deck_run):
    runs, run = deck_run
    torch.manual_seed(7)
    cfg = MantisConfig(
        h=32, blocks=1, heads=2, value_queries=2, value_bins=5,
        policy_hidden=32, value_hidden=32,
    )
    model = MantisNet(cfg)
    optimizer = torch.optim.Adam(model.parameters())
    checkpoint = run / "checkpoint_000001.pt"
    save_checkpoint(checkpoint, model, optimizer, 1, np.random.default_rng(4))
    moves = [(0, 0), (1, 0), (0, 1)]
    expected = inspect_position(model, moves, 3, 0.1, 0.03)

    with TestClient(create_app(runs, device="cpu")) as client:
        response = client.post(
            "/api/inspect",
            json={"checkpoint": str(checkpoint), "moves": moves},
        )
        assert response.status_code == 200, response.text
        actual = response.json()
        assert actual["v_hat"] == pytest.approx(expected["v_hat"], abs=1e-6)
        assert actual["kl"] == pytest.approx(expected["kl"], abs=1e-6)
        assert actual["norm_entropy"] == pytest.approx(
            expected["norm_entropy"], abs=1e-6
        )
        assert [row["move"] for row in actual["legal"]] == [
            list(row["move"]) for row in expected["legal"]
        ]
        for got, want in zip(actual["legal"], expected["legal"]):
            assert got["policy"] == pytest.approx(want["policy"], abs=1e-6)
            assert got["q"] == pytest.approx(want["q"], abs=1e-6)
            assert got["improved"] == pytest.approx(want["improved"], abs=1e-6)

        d6 = client.post(
            "/api/inspect/d6", json={"checkpoint": str(checkpoint), "moves": moves}
        )
        assert d6.status_code == 200, d6.text
        assert len(d6.json()["transforms"]) == 12
        capture = client.post(
            "/api/inspect/attention",
            json={"checkpoint": str(checkpoint), "moves": moves},
        )
        assert capture.status_code == 200, capture.text
        assert len(capture.json()["layers"]) == 1
        assert len(capture.json()["layers"][0]["heads"]) == 2


def test_inspect_refuses_a_multi_output_checkpoint_with_structured_error(deck_run):
    runs, run = deck_run
    model = MantisNet(MantisConfig())
    state = model.state_dict()
    state["mlp_q.out.weight"] = state["mlp_q.out.weight"].expand(2, -1).clone()
    state["mlp_q.out.bias"] = state["mlp_q.out.bias"].expand(2).clone()
    checkpoint = run / "checkpoint_000002.pt"
    torch.save(
        {
            "model": state,
            "versions": _versions(),
        },
        checkpoint,
    )

    with TestClient(create_app(runs, device="cpu")) as client:
        response = client.post(
            "/api/inspect",
            json={"checkpoint": str(checkpoint), "moves": []},
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "inspect_failed"
    assert "\x66actored critic checkpoint" in response.json()["error"]["message"]

    with TestClient(create_app(runs, device="cpu")) as client:
        response = client.post(
            "/api/play",
            json={
                "seats": [
                    {"kind": "checkpoint", "checkpoint": str(checkpoint)},
                    {"kind": "human"},
                ]
            },
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "play_failed"
    assert "\x66actored critic checkpoint" in response.json()["error"]["message"]


def test_deck_database_refuses_a_version_mismatch(tmp_path):
    path = tmp_path / "runs"
    path.mkdir()
    import sqlite3
    conn = sqlite3.connect(path / "deck.db")
    conn.execute("CREATE TABLE schema_version(version INTEGER NOT NULL)")
    conn.execute("INSERT INTO schema_version VALUES (999)")
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError, match="there are no migrations"):
        with TestClient(create_app(path)):
            pass
