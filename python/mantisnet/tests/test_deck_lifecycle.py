"""Child-process, sentinel, conflict, and SSE behavior."""

from __future__ import annotations

import json
import sys
import textwrap
import time

from fastapi.testclient import TestClient

from mantisnet.deck.app import create_app
from mantisnet.deck.app import _sse
from mantisnet.deck.service import RunRegistry


def _fake_driver(path):
    path.write_text(
        textwrap.dedent(
            """
            import argparse, json, time
            from pathlib import Path

            parser = argparse.ArgumentParser(add_help=False)
            parser.add_argument("--out", type=Path)
            parser.add_argument("--iterations", type=int)
            args, _ = parser.parse_known_args()
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / "config.json").write_text(json.dumps({
                "iterations": args.iterations, "klent": {"tau": .1, "lam": .03}
            }))
            while not (args.out / "STOP").exists():
                (args.out / "status.json").write_text(json.dumps({
                    "updated": "now", "iteration": 0, "collect": None,
                    "fit": None, "eval": None
                }))
                time.sleep(.02)
            (args.out / "STOP").unlink()
            print("stopping: STOP sentinel honored", flush=True)
            """
        ),
        encoding="utf-8",
    )


def test_a_launched_run_is_listed_before_its_directory_exists(tmp_path):
    """A registered child is listed before its run artifacts exist."""
    driver = tmp_path / "slow_driver.py"
    driver.write_text("import time; time.sleep(30)", encoding="utf-8")
    runs = tmp_path / "runs"
    with TestClient(
        create_app(runs, tmp_path / "dist", device="cpu", command_prefix=[sys.executable, str(driver)])
    ) as client:
        launched = client.post(
            "/api/runs",
            json={"name": "slow", "iterations": 5, "games": 2, "envs": 2, "device": "cpu"},
        )
        assert launched.status_code == 201, launched.text
        listed = client.get("/api/runs").json()
        row = next(r for r in listed if r["name"] == "slow")
        assert row["state"] == "active" and row["checkpoints"] == []
        assert row["iterations"] is None  # No config.json exists yet.
        killed = client.post("/api/runs/slow/kill", json={"confirm": True})
        assert killed.status_code == 202, killed.text


def test_launch_defaults_to_the_uncapped_eval_ladder(tmp_path):
    """Launch defaults select time-limited SealBot and Gumbel search without
    adding an evaluation depth cap."""
    driver = tmp_path / "fake_driver.py"
    _fake_driver(driver)
    runs = tmp_path / "runs"
    with TestClient(
        create_app(runs, tmp_path / "dist", device="cpu", command_prefix=[sys.executable, str(driver)])
    ) as client:
        launched = client.post(
            "/api/runs",
            json={
                "name": "ladder", "iterations": 20, "games": 2, "envs": 2,
                "device": "cpu", "eval_every": 5, "sealbot": str(tmp_path),
            },
        )
        assert launched.status_code == 201, launched.text
        command = launched.json()["command"]
        assert "--eval-depth" not in command
        assert command[command.index("--eval-time") + 1] == "0.1"
        assert command[command.index("--eval-sims") + 1] == "32"
        assert client.post("/api/runs/ladder/stop").status_code == 202

        _wait_for_exit(client, "ladder")
        capped = client.post(
            "/api/runs",
            json={
                "name": "rung", "iterations": 20, "games": 2, "envs": 2,
                "device": "cpu", "eval_every": 5, "sealbot": str(tmp_path),
                "eval_depth": 3,
            },
        )
        assert capped.status_code == 201, capped.text
        command = capped.json()["command"]
        assert command[command.index("--eval-depth") + 1] == "3"
        assert client.post("/api/runs/rung/stop").status_code == 202


def _wait_for_exit(client, name):
    deadline = time.time() + 3
    while time.time() < deadline:
        if client.get(f"/api/runs/{name}").json()["exit_code"] is not None:
            return
        time.sleep(.02)
    raise AssertionError(f"{name} did not exit")


def test_lifecycle_one_active_run_and_sentinels(tmp_path):
    driver = tmp_path / "fake_driver.py"
    _fake_driver(driver)
    runs = tmp_path / "runs"
    with TestClient(
        create_app(runs, tmp_path / "dist", device="cpu", command_prefix=[sys.executable, str(driver)])
    ) as client:
        launched = client.post(
            "/api/runs",
            json={"name": "fake", "iterations": 20, "games": 2, "envs": 2, "device": "cpu"},
        )
        assert launched.status_code == 201, launched.text
        conflict = client.post(
            "/api/runs",
            json={"name": "second", "iterations": 20, "games": 2, "envs": 2, "device": "cpu"},
        )
        assert conflict.status_code == 409
        assert client.post("/api/runs/fake/checkpoint").status_code == 202
        assert (runs / "fake" / "CHECKPOINT").exists()
        assert client.post("/api/runs/fake/kill", json={"confirm": False}).status_code == 400
        assert client.post("/api/runs/fake/stop").status_code == 202
        deadline = time.time() + 3
        while time.time() < deadline:
            state = client.get("/api/runs/fake").json()
            if state["exit_code"] is not None:
                break
            time.sleep(.02)
        assert state["exit_code"] == 0
        assert state["state"] == "stopped"
        assert "STOP sentinel honored" in (
            runs / "fake" / "deck-console.log"
        ).read_text(encoding="utf-8")


def test_sse_wire_format_is_typed_and_strict_json():
    assert _sse("iteration", {"iteration": 4}) == (
        'event: iteration\ndata: {"iteration":4}\n\n'
    )
    assert "NaN" not in _sse("eval", {"elo": float("inf")})


def test_registry_distinguishes_external_active_completed_and_starved(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    active = root / "external"
    active.mkdir()
    (active / "config.json").write_text('{"iterations": 10}', encoding="utf-8")
    (active / "status.json").write_text(
        '{"updated":"now","iteration":2,"collect":null,"fit":null,"eval":null}',
        encoding="utf-8",
    )
    completed = root / "done"
    completed.mkdir()
    (completed / "config.json").write_text('{"iterations": 0}', encoding="utf-8")
    starved = root / "collapsed"
    starved.mkdir()
    (starved / "config.json").write_text('{"iterations": 10}', encoding="utf-8")
    (starved / "deck-console.log").write_text(
        "stopping starved: games are not finishing", encoding="utf-8"
    )

    registry = RunRegistry(root)
    assert registry.describe("external")["state"] == "active"
    assert registry.describe("done")["state"] == "completed"
    assert registry.describe("collapsed")["state"] == "starved"
