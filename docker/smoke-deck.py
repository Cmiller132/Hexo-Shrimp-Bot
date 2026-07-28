"""Assert the representative end-to-end deck surface over HTTP."""

from __future__ import annotations

import json
import time

import httpx

BASE = "http://127.0.0.1:8000"


def show(label, response):
    response.raise_for_status()
    value = response.json()
    print(f"{label}: {json.dumps(value, separators=(',', ':'))[:300]}")
    return value


with httpx.Client(base_url=BASE, timeout=30) as client:
    health = show("health", client.get("/api/health"))
    assert health["device"] == "cpu"
    launched = show("launch", client.post("/api/runs", json={
        "name": "deck-smoke", "iterations": 1, "games": 2, "envs": 2,
        "seed": 41, "checkpoint_every": 1, "device": "cpu",
    }))
    deadline = time.time() + 120
    while time.time() < deadline:
        runs = show("runs", client.get("/api/runs"))
        run = next((row for row in runs if row["name"] == launched["name"]), None)
        assert run is not None, "a launched run must appear in the list at once"
        if run["state"] == "completed":
            break
        time.sleep(.25)
    else:
        raise AssertionError("heuristic run did not complete")
    series = show("iterations", client.get(
        "/api/runs/deck-smoke/iterations?columns=f,acting_kl,seconds"
    ))
    assert len(series) == 1 and series[0]["iteration"] == 0
    games = show("games", client.get("/api/runs/deck-smoke/games"))
    game = show("game", client.get(f"/api/runs/deck-smoke/games/{games[0]['game_id']}"))
    assert game["moves"] and game["plies"]
    events = client.get("/api/runs/deck-smoke/events?once=true")
    events.raise_for_status()
    print("events:", events.text.replace("\n", " | ")[:300])
    assert "event: iteration" in events.text and "event: checkpoint" in events.text
    play = show("play", client.post("/api/play", json={
        "seats": [{"kind": "human"}, {"kind": "random"}],
    }))
    play = show("move", client.post(
        f"/api/play/{play['session_id']}/moves", json={"move": [0, 0]}
    ))
    assert len(play["moves"]) >= 2
    checkpoint = run["checkpoints"][-1]["path"]
    inspected = show("inspect", client.post("/api/inspect", json={
        "checkpoint": checkpoint, "moves": game["moves"][:3],
    }))
    assert inspected["legal_count"] == len(inspected["legal"])
    spa = client.get("/")
    spa.raise_for_status()
    assert '<div id="root"></div>' in spa.text
    print("spa: index.html mounted")
    print("SMOKE PASS")
