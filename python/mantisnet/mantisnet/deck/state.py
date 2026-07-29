"""Deck-owned persistence, separate from run telemetry."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE schema_version (version INTEGER NOT NULL);
CREATE TABLE game_reviews (
    run TEXT NOT NULL, game_id INTEGER NOT NULL, tags_json TEXT NOT NULL,
    note TEXT NOT NULL, updated TEXT NOT NULL, PRIMARY KEY (run, game_id)
);
CREATE TABLE probes (
    probe_id INTEGER PRIMARY KEY, name TEXT NOT NULL, checkpoint TEXT NOT NULL,
    moves_json TEXT NOT NULL, module TEXT NOT NULL, created TEXT NOT NULL
);
CREATE TABLE presets (
    preset_id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE,
    config_json TEXT NOT NULL, updated TEXT NOT NULL
);
CREATE TABLE match_jobs (
    job_id INTEGER PRIMARY KEY, status TEXT NOT NULL, request_json TEXT NOT NULL,
    result_json TEXT, error TEXT, created TEXT NOT NULL, updated TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DeckState:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        row = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
        ).fetchone()
        if row is None:
            self.conn.executescript(_SCHEMA)
            self.conn.execute("INSERT INTO schema_version VALUES (?)", (SCHEMA_VERSION,))
            self.conn.commit()
        version = self.conn.execute("SELECT version FROM schema_version").fetchone()[0]
        if version != SCHEMA_VERSION:
            self.conn.close()
            raise RuntimeError(
                f"deck database schema {version} != this build {SCHEMA_VERSION}; "
                "there are no migrations"
            )

    def review(self, run: str, game_id: int) -> dict:
        row = self.conn.execute(
            "SELECT * FROM game_reviews WHERE run=? AND game_id=?", (run, game_id)
        ).fetchone()
        return (
            dict(row) | {"tags": json.loads(row["tags_json"])}
            if row
            else {"run": run, "game_id": game_id, "tags": [], "note": ""}
        )

    def put_review(self, run: str, game_id: int, tags: list[str], note: str) -> dict:
        if len(tags) > 32 or any(not tag.strip() or len(tag) > 64 for tag in tags):
            raise ValueError("tags must be 1-64 characters, at most 32 tags")
        now = _now()
        self.conn.execute(
            "INSERT INTO game_reviews VALUES (?,?,?,?,?) "
            "ON CONFLICT(run,game_id) DO UPDATE SET tags_json=excluded.tags_json,"
            "note=excluded.note,updated=excluded.updated",
            (run, game_id, json.dumps(tags), note, now),
        )
        self.conn.commit()
        return self.review(run, game_id)

    def probes(self) -> list[dict]:
        return [
            dict(row) | {"moves": json.loads(row["moves_json"])}
            for row in self.conn.execute("SELECT * FROM probes ORDER BY probe_id DESC")
        ]

    def add_probe(self, name: str, checkpoint: str, moves, module: str) -> dict:
        cur = self.conn.execute(
            "INSERT INTO probes(name,checkpoint,moves_json,module,created) VALUES (?,?,?,?,?)",
            (name, checkpoint, json.dumps(moves), module, _now()),
        )
        self.conn.commit()
        return next(p for p in self.probes() if p["probe_id"] == cur.lastrowid)

    def delete_probe(self, probe_id: int) -> None:
        if self.conn.execute("DELETE FROM probes WHERE probe_id=?", (probe_id,)).rowcount == 0:
            raise KeyError(f"no probe {probe_id}")
        self.conn.commit()

    def presets(self) -> list[dict]:
        return [
            dict(row) | {"config": json.loads(row["config_json"])}
            for row in self.conn.execute("SELECT * FROM presets ORDER BY name")
        ]

    def put_preset(self, name: str, config: dict) -> dict:
        now = _now()
        self.conn.execute(
            "INSERT INTO presets(name,config_json,updated) VALUES (?,?,?) "
            "ON CONFLICT(name) DO UPDATE SET config_json=excluded.config_json,"
            "updated=excluded.updated",
            (name, json.dumps(config), now),
        )
        self.conn.commit()
        return next(p for p in self.presets() if p["name"] == name)

    def new_match(self, request: dict) -> int:
        now = _now()
        cur = self.conn.execute(
            "INSERT INTO match_jobs(status,request_json,created,updated) VALUES (?,?,?,?)",
            ("queued", json.dumps(request), now, now),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def update_match(self, job_id: int, status: str, result=None, error=None) -> None:
        self.conn.execute(
            "UPDATE match_jobs SET status=?,result_json=?,error=?,updated=? WHERE job_id=?",
            (status, json.dumps(result) if result is not None else None, error, _now(), job_id),
        )
        self.conn.commit()

    def matches(self) -> list[dict]:
        out = []
        for row in self.conn.execute("SELECT * FROM match_jobs ORDER BY job_id DESC"):
            item = dict(row)
            item["request"] = json.loads(item.pop("request_json"))
            item["result"] = (
                json.loads(item.pop("result_json")) if item["result_json"] else None
            )
            out.append(item)
        return out

    def match(self, job_id: int) -> dict:
        try:
            return next(m for m in self.matches() if m["job_id"] == job_id)
        except StopIteration:
            raise KeyError(f"no match job {job_id}") from None
