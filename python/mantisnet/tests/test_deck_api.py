"""The deck's HTTP contract over telemetry and engine state."""

from __future__ import annotations

import json
import sqlite3
import time

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from mantisnet import MantisConfig, MantisNet
from mantisnet.deck.app import create_app
from mantisnet.deck.service import _SNAPSHOT_ROOT, telemetry_connection
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
        "klent": {"tau": 0.1, "lam": 0.03, "mass_floor": 0.2},
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
    expected = inspect_position(model, moves, 3, 0.1, 0.03, 0.2)

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


def test_inspect_refuses_a_scalar_critic_checkpoint_with_structured_error(deck_run):
    runs, run = deck_run
    model = MantisNet(MantisConfig())
    state = model.state_dict()
    state["mlp_q.out.weight"] = state["mlp_q.out.weight"][:1].clone()
    state["mlp_q.out.bias"] = state["mlp_q.out.bias"][:1].clone()
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
    assert "unsupported critic readout width 1" in response.json()["error"]["message"]

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
    assert "unsupported critic readout width 1" in response.json()["error"]["message"]


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


def test_telemetry_connection_closes_when_its_block_exits(deck_run):
    """``with sqlite3.connect(...)`` opens a transaction; it does not close.

    The assertion is that the handle is dead once the block ends, not merely
    that the query inside it worked.
    """
    _runs, run = deck_run
    with telemetry_connection(run) as conn:
        conn.execute("SELECT version FROM schema_version").fetchone()
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT version FROM schema_version")


def test_repeated_queries_leave_no_connection_open(deck_run, monkeypatch):
    """One request must not cost a descriptor.

    Counting live connections rather than file descriptors keeps the check
    meaningful on every platform the suite runs on.
    """
    runs, _run = deck_run
    live: list[sqlite3.Connection] = []
    real_connect = sqlite3.connect

    def tracking_connect(target, *args, **kwargs):
        conn = real_connect(target, *args, **kwargs)
        # The deck's own review database is opened once for the app's life;
        # only the per-request telemetry handles are under test here. Those
        # handles target the run's local snapshot, whose file is named after
        # the run rather than after telemetry.DB_NAME.
        if telemetry.DB_NAME in str(target) or _SNAPSHOT_ROOT.name in str(target):
            live.append(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", tracking_connect)
    requests = 8
    with TestClient(create_app(runs, device="cpu")) as client:
        for _ in range(requests):
            assert client.get("/api/runs/fixture/summary").status_code == 200
    assert len(live) == requests, (
        f"expected one telemetry connection per request, got {len(live)}"
    )

    def closed(conn: sqlite3.Connection) -> bool:
        try:
            conn.execute("SELECT 1")
        except sqlite3.ProgrammingError:
            return True
        return False

    assert [c for c in live if not closed(c)] == []


def test_a_missing_telemetry_database_still_maps_to_404(deck_run, tmp_path):
    """Entering the connection inside the request must not cost the status
    codes: a run without a telemetry database still maps to 404."""
    runs, _run = deck_run
    (runs / "empty").mkdir()
    with TestClient(create_app(runs, device="cpu")) as client:
        response = client.get("/api/runs/empty/summary")
    assert response.status_code == 404, response.text
    assert response.json()["error"]["code"] == "telemetry_not_found"


def test_iteration_endpoint_merges_dynamic_metrics_and_refuses_collisions(
    deck_run, tmp_path
):
    runs, run = deck_run
    conn = sqlite3.connect(run / telemetry.DB_NAME)
    conn.execute(
        "UPDATE iterations SET metrics_json=? WHERE iteration=0",
        (json.dumps({"critic_ce": 0.25}),),
    )
    conn.commit()
    conn.close()
    with TestClient(create_app(runs, tmp_path / "missing", device="cpu")) as client:
        response = client.get(
            "/api/runs/fixture/iterations?columns=games,critic_ce"
        )
        assert response.status_code == 200, response.text
        assert response.json() == [{"iteration": 0, "games": 3, "critic_ce": 0.25}]

    conn = sqlite3.connect(run / telemetry.DB_NAME)
    conn.execute(
        "UPDATE iterations SET metrics_json=? WHERE iteration=0",
        (json.dumps({"games": 999}),),
    )
    conn.commit()
    conn.close()
    with TestClient(create_app(runs, tmp_path / "missing", device="cpu")) as client:
        response = client.get("/api/runs/fixture/iterations?columns=games")
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_query"
        assert "collides with a fixed iterations column" in response.text


def test_horizon_endpoint_has_known_buckets_and_last_six_default(tmp_path):
    runs = tmp_path / "runs"
    run = runs / "horizon"
    run.mkdir(parents=True)
    writer = telemetry.open_telemetry(run)
    writer.begin_run({"iterations": 8}, {"fixture": True}, 0)
    conn = writer._conn
    for iteration in range(8):
        conn.execute(
            "INSERT INTO iterations (iteration, run, games, plies, metrics_json) "
            "VALUES (?, 1, 0, 0, '{}')",
            (iteration,),
        )
    for game_id, iteration in ((1, 0), (2, 7)):
        conn.execute(
            "INSERT INTO games (game_id, kind, iteration, game_index, winner, "
            "length, capped, moves) VALUES (?, 'selfplay', ?, ?, 0, 70, 0, ?)",
            (game_id, iteration, game_id, telemetry.pack_moves([])),
        )
    plies = [
        # The iteration-0 row proves that the default window is iterations 2..7.
        (1, 69, 0, 9000),
        # k=1 winner correct; k=2 loser correct; k=10 zero is wrong; k=70 loser wrong.
        (2, 69, 0, 1000),
        (2, 68, 1, -2000),
        (2, 60, 0, 0),
        (2, 0, 1, 1000),
    ]
    conn.executemany(
        "INSERT INTO plies (game_id, t, mover, moves_remaining, legal_count, "
        "rank, v_hat, kl, norm_entropy, pi_top1, pi_chosen) "
        "VALUES (?, ?, ?, 1, 7, 0, ?, 0, 0, 0, 0)",
        plies,
    )
    conn.commit()
    writer.close()
    (run / "config.json").write_text(json.dumps({"iterations": 8}), encoding="utf-8")

    with TestClient(create_app(runs, tmp_path / "missing", device="cpu")) as client:
        default = client.get("/api/runs/horizon/horizon")
        assert default.status_code == 200, default.text
        payload = default.json()
        assert (payload["lo"], payload["hi"]) == (2, 7)
        by_key = {(row["k_min"], row["outcome"]): row for row in payload["buckets"]}
        assert by_key[(1, "won")] == {
            "k_min": 1, "k_max": 4, "bucket": "1–4", "outcome": "won",
            "count": 1, "sign_accuracy": 1.0, "mean_abs_v_hat": 0.1,
        }
        assert by_key[(1, "lost")]["sign_accuracy"] == 1.0
        assert by_key[(1, "lost")]["mean_abs_v_hat"] == 0.2
        assert by_key[(9, "won")]["sign_accuracy"] == 0.0
        assert by_key[(65, "lost")]["sign_accuracy"] == 0.0
        explicit = client.get("/api/runs/horizon/horizon?lo=0&hi=7").json()
        explicit_first = next(
            row for row in explicit["buckets"]
            if row["k_min"] == 1 and row["outcome"] == "won"
        )
        assert explicit_first["count"] == 2
        assert explicit_first["mean_abs_v_hat"] == pytest.approx(0.5)


def _h2h_run(tmp_path):
    runs = tmp_path / "runs"
    run = runs / "paired"
    run.mkdir(parents=True)
    writer = telemetry.open_telemetry(run)
    writer.begin_run({"iterations": 1}, {"fixture": True}, 0)
    conn = writer._conn
    opponent = writer.opponent("h2h:reference/checkpoint_000100", {})
    conn.execute(
        "INSERT INTO eval_matches (match_id, created, source, opponent, iteration, "
        "games, score, win_rate, capped, elo, elo_lo, elo_hi) "
        "VALUES (1, '2026-08-03T00:00:00+00:00', 'driver', ?, 1, "
        "8, 6, .75, 0, 190, 20, 360)",
        (opponent,),
    )
    winners = [(0, 1), (0, 1), (0, 1), (1, 0)]
    for pair_index, pair_winners in enumerate(winners):
        for within_pair, winner in enumerate(pair_winners):
            game_index = pair_index * 2 + within_pair
            conn.execute(
                "INSERT INTO games (kind, iteration, match, game_index, winner, "
                "length, capped, model_seat, opening_len, moves) "
                "VALUES ('eval', 1, 1, ?, ?, 1, 0, ?, 1, ?)",
                (game_index, winner, within_pair, telemetry.pack_moves([(0, 0)])),
            )
    conn.commit()
    writer.close()
    (run / "config.json").write_text(json.dumps({"iterations": 1}), encoding="utf-8")
    return runs, run


def test_h2h_sign_test_is_recomputed_and_unclean_pairs_are_refused(tmp_path):
    runs, run = _h2h_run(tmp_path)
    with TestClient(create_app(runs, tmp_path / "missing", device="cpu")) as client:
        response = client.get("/api/runs/paired/strength")
        assert response.status_code == 200, response.text
        row = response.json()[0]
        assert row["family"] == "h2h"
        assert row["sign_test_p"] == pytest.approx(0.625)
        assert row["decisive_pairs"] == 4
        assert row["pair_counts"] == {
            "model_both": 3, "split": 0, "reference_both": 1, "capped": 0,
        }

    conn = sqlite3.connect(run / telemetry.DB_NAME)
    conn.execute("UPDATE games SET model_seat=0 WHERE match=1 AND game_index=1")
    conn.commit()
    conn.close()
    with TestClient(create_app(runs, tmp_path / "missing", device="cpu")) as client:
        response = client.get("/api/runs/paired/strength")
        assert response.status_code == 500
        assert response.json()["error"]["code"] == "invalid_eval_match"
        assert "h2h match 1 does not pair cleanly" in response.text
        assert "expected [0, 1]" in response.text
