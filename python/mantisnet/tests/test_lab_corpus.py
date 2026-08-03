"""Frozen-corpus format and replay contracts from LAB-BUILD-SPEC §2."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

import hexo_py
import numpy as np
import pytest

from mantisnet.klent import telemetry
from mantisnet.lab import corpus as corpus_module
from mantisnet.lab.corpus import SPLIT_NAMES, freeze, load_corpus

from .test_klent_returns import FIRST_STONE_WIN, SECOND_STONE_WIN


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _completed_games(count: int) -> list[list[tuple[int, int]]]:
    games: list[list[tuple[int, int]]] = []
    for base in (FIRST_STONE_WIN, SECOND_STONE_WIN):
        for transform in telemetry.D6_TRANSFORMS:
            moves = [transform(move) for move in base]
            if moves not in games:
                pos = hexo_py.Position.replay(moves)
                assert pos.is_terminal and pos.winner == 0
                games.append(moves)
    assert len(games) >= count
    return games[:count]


def _write_db(
    run_dir: Path,
    rows: list[tuple[int, str, int, int | None, int, list[tuple[int, int]]]],
    *,
    schema_version: int = telemetry.SCHEMA_VERSION,
) -> None:
    run_dir.mkdir(parents=True)
    with sqlite3.connect(run_dir / telemetry.DB_NAME) as conn:
        conn.executescript(
            """
            CREATE TABLE schema_version (version INTEGER NOT NULL);
            CREATE TABLE games (
                game_id INTEGER PRIMARY KEY,
                kind TEXT NOT NULL,
                iteration INTEGER,
                winner INTEGER,
                length INTEGER NOT NULL,
                capped INTEGER NOT NULL,
                moves BLOB NOT NULL
            );
            """
        )
        conn.execute("INSERT INTO schema_version VALUES (?)", (schema_version,))
        conn.executemany(
            "INSERT INTO games "
            "(game_id, kind, iteration, winner, length, capped, moves) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    game_id,
                    kind,
                    iteration,
                    winner,
                    len(moves),
                    capped,
                    telemetry.pack_moves(moves),
                )
                for game_id, kind, iteration, winner, capped, moves in rows
            ],
        )


@pytest.fixture(scope="module")
def frozen_pair(tmp_path_factory):
    root = tmp_path_factory.mktemp("lab-corpus")
    run = root / "source-run"
    selected = _completed_games(20)
    rows = [
        (100 + index, "selfplay", 4, 0, 0, moves)
        for index, moves in enumerate(selected)
    ]
    # These exercise every exclusion: outcome-less self-play, completed eval,
    # and completed self-play outside the inclusive iteration window.
    rows.extend(
        [
            (900, "selfplay", 4, None, 1, FIRST_STONE_WIN[:5]),
            (901, "eval", 4, 0, 0, FIRST_STONE_WIN),
            (902, "selfplay", 3, 0, 0, FIRST_STONE_WIN),
            (903, "selfplay", 6, 0, 0, SECOND_STONE_WIN),
        ]
    )
    _write_db(run, rows)

    kwargs = dict(
        name="same-corpus",
        train_samples=37,
        val_samples=5,
        test_samples=6,
        seed=1729,
    )
    first = root / "frozen-a"
    second = root / "frozen-b"
    first_manifest = freeze(run, first, (4, 5), **kwargs)
    second_manifest = freeze(run, second, (4, 5), **kwargs)
    return root, run, first, second, first_manifest, second_manifest


def test_freeze_is_byte_deterministic_and_records_selection(frozen_pair):
    _root, _run, first, second, first_manifest, second_manifest = frozen_pair
    first_bytes = (first / "corpus.npz").read_bytes()
    second_bytes = (second / "corpus.npz").read_bytes()
    assert first_bytes == second_bytes
    assert first_manifest["corpus_sha256"] == second_manifest["corpus_sha256"]
    assert first_manifest["corpus_sha256"] == hashlib.sha256(first_bytes).hexdigest()

    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format_version"] == 1
    assert manifest["RULES_VERSION"] == hexo_py.RULES_VERSION
    assert manifest["ACTION_ORDER_VERSION"] == hexo_py.ACTION_ORDER_VERSION
    assert manifest["source"]["telemetry_schema_version"] == telemetry.SCHEMA_VERSION
    assert manifest["source"]["iteration_window"] == [4, 5]
    assert manifest["source"]["selection_predicate"] == (
        "kind = 'selfplay' AND winner IS NOT NULL"
    )
    assert sum(manifest["split"]["games"].values()) == 20
    assert manifest["samples"]["requested"] == {"train": 37, "val": 5, "test": 6}
    assert manifest["samples"]["available"] == manifest["split"]["plies"]
    assert manifest["samples"]["realized"] == {"train": 37, "val": 5, "test": 6}


def test_splits_are_disjoint_and_excluded_games_stay_out(frozen_pair):
    _root, _run, first, _second, _first_manifest, _second_manifest = frozen_pair
    corpus = load_corpus(first)
    assert corpus.n_games == 20
    assert set(corpus.source_game_id.tolist()) == set(range(100, 120))
    assert not ({900, 901, 902, 903} & set(corpus.source_game_id.tolist()))

    split_games = {
        name: set(corpus.split_samples(name).game.tolist()) for name in SPLIT_NAMES
    }
    assert split_games["train"].isdisjoint(split_games["val"])
    assert split_games["train"].isdisjoint(split_games["test"])
    assert split_games["val"].isdisjoint(split_games["test"])
    for name, split_id in zip(SPLIT_NAMES, range(3), strict=True):
        samples = corpus.split_samples(name)
        assert len({(int(g), int(t)) for g, t in zip(samples.game, samples.t)}) == len(
            samples
        )
        assert np.all(corpus.split[samples.game] == split_id)


def test_replay_verified_rank_mover_outcome_and_distance(frozen_pair):
    _root, _run, first, _second, _first_manifest, _second_manifest = frozen_pair
    corpus = load_corpus(first)
    for split_name in SPLIT_NAMES:
        samples = corpus.split_samples(split_name)
        for game, t, rank, mover, z, dist in zip(
            samples.game,
            samples.t,
            samples.rank,
            samples.mover,
            samples.z,
            samples.dist,
            strict=True,
        ):
            game_index = int(game)
            ply = int(t)
            moves = corpus.moves_for(game_index)
            pos = hexo_py.Position.replay(moves[:ply])
            assert pos.current_player == int(mover)
            assert pos.legal_moves()[int(rank)] == moves[ply]
            assert int(z) == (1 if pos.current_player == corpus.winner[game_index] else -1)
            assert int(dist) == len(moves) - ply

    with pytest.raises(ValueError, match="corpus split"):
        corpus.split_samples("holdout")
    with pytest.raises(IndexError, match="game index"):
        corpus.moves_for(corpus.n_games)


def test_dry_run_uses_read_only_uri_and_writes_nothing(tmp_path, monkeypatch):
    run = tmp_path / "run"
    _write_db(run, [(7, "selfplay", 2, 0, 0, FIRST_STONE_WIN)])
    database = run / telemetry.DB_NAME
    before = _sha256(database)
    connections = []
    original_connect = corpus_module.sqlite3.connect

    def recording_connect(path, *args, **kwargs):
        connections.append((path, kwargs.copy()))
        return original_connect(path, *args, **kwargs)

    monkeypatch.setattr(corpus_module.sqlite3, "connect", recording_connect)
    output = tmp_path / "must-not-exist"
    plan = freeze(run, output, (2, 2), seed=3, dry_run=True)
    assert plan["dry_run"] is True
    assert sum(plan["split"]["games"].values()) == 1
    assert not output.exists()
    assert _sha256(database) == before
    assert set(run.iterdir()) == {database}
    assert len(connections) == 1
    uri, kwargs = connections[0]
    assert str(uri).startswith("file:") and str(uri).endswith("?mode=ro")
    assert kwargs["uri"] is True


def test_loader_refuses_sha_mismatch(frozen_pair):
    root, _run, first, _second, _first_manifest, _second_manifest = frozen_pair
    damaged = root / "damaged-sha"
    shutil.copytree(first, damaged)
    with (damaged / "corpus.npz").open("ab") as stream:
        stream.write(b"damage")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_corpus(damaged)


@pytest.mark.parametrize("field", ["RULES_VERSION", "ACTION_ORDER_VERSION"])
def test_loader_refuses_engine_version_mismatch(frozen_pair, field):
    root, _run, first, _second, _first_manifest, _second_manifest = frozen_pair
    damaged = root / f"damaged-{field.lower()}"
    shutil.copytree(first, damaged)
    path = damaged / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest[field] += 1
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=field):
        load_corpus(damaged)


def test_wrong_telemetry_schema_and_empty_selection_are_refused(tmp_path):
    wrong = tmp_path / "wrong-schema"
    _write_db(
        wrong,
        [(1, "selfplay", 0, 0, 0, FIRST_STONE_WIN)],
        schema_version=telemetry.SCHEMA_VERSION + 1,
    )
    with pytest.raises(ValueError, match="telemetry schema"):
        freeze(wrong, tmp_path / "unused-wrong", (0, 0), dry_run=True)

    empty = tmp_path / "empty-selection"
    _write_db(empty, [(1, "eval", 0, 0, 0, FIRST_STONE_WIN)])
    with pytest.raises(ValueError, match="empty corpus selection"):
        freeze(empty, tmp_path / "unused-empty", (0, 0), dry_run=True)


def test_replay_winner_mismatch_names_the_source_game(tmp_path):
    run = tmp_path / "bad-replay"
    # The engine says P0, while this deliberately malformed telemetry row says
    # P1.  Freeze must finish the replay and diagnose the source primary key.
    _write_db(run, [(77, "selfplay", 0, 1, 0, FIRST_STONE_WIN)])
    with pytest.raises(ValueError, match=r"game 77 replay winner mismatch"):
        freeze(
            run,
            tmp_path / "unused-bad-replay",
            (0, 0),
            train_samples=len(FIRST_STONE_WIN),
            val_samples=0,
            test_samples=0,
        )
