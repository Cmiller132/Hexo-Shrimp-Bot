"""Telemetry storage, isolation, resume, query, and inspection contracts.

The writer must not change training RNG or outputs. Resume replaces the
database tail from the restored iteration. Calibration agrees with direct
arithmetic. ``inspect_position`` agrees with collection when both receive the
same actor weights, improvement parameters, device, and inference precision.
"""

from __future__ import annotations

import json
import sqlite3

import hexo_py
import numpy as np
import pytest
import torch

from mantisnet import MantisConfig, MantisNet
from mantisnet.klent import run as run_mod
from mantisnet.klent import telemetry as tel
from mantisnet.klent.inspect import inspect_position
from mantisnet.klent.run import save_checkpoint
from mantisnet.klent.train import KlentConfig

from .heuristic import heuristic_game


def _tiny_model():
    torch.manual_seed(5)
    return MantisNet(
        MantisConfig(
            h=32, blocks=1, heads=2, value_queries=2, value_bins=5,
            policy_hidden=32, value_hidden=32,
        )
    )


def _train(out_dir, iterations, start_iteration=0, seed=5):
    torch.manual_seed(2)
    model = MantisNet(MantisConfig(h=32, blocks=1, heads=2, value_queries=2, value_bins=5))
    optimizer = torch.optim.Adam(model.parameters())
    run_mod.run_training(
        model,
        optimizer,
        KlentConfig(games_per_iteration=2, envs=2, ply_cap=24, batch_size=64),
        iterations=iterations,
        out_dir=out_dir,
        rng=np.random.default_rng(seed),
        checkpoint_every=iterations,
        start_iteration=start_iteration,
    )
    return model, optimizer


# ---------------------------------------------------------------------------
# The packing and the symmetry group


def test_move_packing_round_trips_and_refuses_the_impossible():
    moves = [(0, 0), (1, -1), (-7, 12), (30000, -30000)]
    blob = tel.pack_moves(moves)
    assert len(blob) == 4 * len(moves)
    assert tel.unpack_moves(blob) == moves
    assert tel.unpack_moves(tel.pack_moves([])) == []

    with pytest.raises(ValueError, match="int16"):
        tel.pack_moves([(40000, 0)])
    with pytest.raises(ValueError, match="whole"):
        tel.unpack_moves(b"\x00\x00\x00")


def test_d6_transforms_are_symmetries_of_the_rules():
    """Every opening-atlas transform preserves engine legality and outcome."""
    rng = np.random.default_rng(4)
    finished = 0
    for _ in range(6):
        moves, winner = heuristic_game(rng, noise=0.1)
        moves = [tuple(m) for m in moves]
        finished += winner is not None
        for transform in tel.D6_TRANSFORMS:
            pos = hexo_py.Position()
            for move in (transform(m) for m in moves):
                assert move in set(pos.legal_moves()), f"{transform} broke legality"
                pos.advance(*move)
            assert pos.winner == winner
            assert pos.is_terminal == (winner is not None)
    assert finished >= 3, "the heuristic player should usually finish games"


def test_canonical_opening_folds_the_orbit_to_one_key():
    moves, _ = heuristic_game(np.random.default_rng(9), noise=0.1)
    moves = [tuple(m) for m in moves]
    keys = {tel.canonical_opening([t(m) for m in moves], 4) for t in tel.D6_TRANSFORMS}
    assert len(keys) == 1
    # The key is one of the orbit's own members, and truncation is a prefix.
    key = keys.pop()
    assert len(key) == 4
    assert key in {tuple(t(m) for m in moves[:4]) for t in tel.D6_TRANSFORMS}


# ---------------------------------------------------------------------------
# The schema


def test_schema_round_trip(tmp_path):
    """A run's iteration goes in and comes back out: metrics as columns,
    games with their moves, plies with their acting-time scalars."""
    _train(tmp_path, iterations=1)
    conn = tel.connect(tmp_path)

    assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == (
        tel.SCHEMA_VERSION
    )
    row = conn.execute("SELECT * FROM iterations").fetchone()
    metrics = json.loads((tmp_path / "metrics.jsonl").read_text().splitlines()[0])
    for key, value in metrics.items():
        assert row[key] == value, key
    assert json.loads(row["metrics_json"]) == metrics
    assert row["games"] == 2 and row["plies"] > 0
    # The process's own counters are always there; the card's only when
    # there is a card, and this run is on the CPU.
    assert row["hw_samples"] >= 1 and row["rss_max"] > 0
    assert row["gpu_util_mean"] is None

    games = tel.search_games(conn)
    assert len(games) == 2
    assert {g["kind"] for g in games} == {"selfplay"}
    total = 0
    for g in games:
        full = tel.fetch_game(conn, g["game_id"])
        assert len(full["moves"]) == g["length"]
        assert hexo_py.Position.replay(full["moves"]).winner == g["winner"]
        assert (g["winner"] is None) == bool(g["capped"])
        assert [p["t"] for p in full["plies"]] == list(range(g["length"]))
        for t, ply in enumerate(full["plies"]):
            position = hexo_py.Position.replay(full["moves"][:t])
            assert ply["legal_count"] == position.legal_count
            assert ply["mover"] == position.current_player
            assert ply["moves_remaining"] == position.moves_remaining
            assert position.legal_moves()[ply["rank"]] == full["moves"][t]
            assert 0.0 < ply["pi_chosen"] <= ply["pi_top1"] <= 1.0
            # H/log|A| is a ratio of float32 reductions over thousands of
            # cells: a near-uniform π′ lands on 1 from either side.
            assert -1e-4 <= ply["norm_entropy"] <= 1 + 1e-4
            assert ply["kl"] >= -1e-6
        total += g["length"]
    assert row["plies"] == total
    conn.close()


def test_schema_version_mismatch_is_refused(tmp_path):
    _train(tmp_path, iterations=1)
    raw = sqlite3.connect(tel.db_path(tmp_path))
    raw.execute("UPDATE schema_version SET version = 999")
    raw.commit()
    raw.close()
    with pytest.raises(ValueError, match="no migrations"):
        tel.connect(tmp_path)


def test_missing_database_is_an_error_not_an_empty_result(tmp_path):
    with pytest.raises(FileNotFoundError):
        tel.connect(tmp_path)


def test_ply_records_of_unequal_length_are_refused(tmp_path):
    from mantisnet.klent import Episode

    with tel.open_telemetry(tmp_path) as writer:
        writer.begin_run({"iterations": 1}, {"v": 1}, 0)
        episode = Episode(
            moves=[(0, 0)], winner=0, moves_remaining=[1], movers=[0], ranks=[0],
            improved=[np.ones(1, dtype=np.float32)], v_hats=[0.5], kls=[0.1],
            norm_entropies=[0.2], pi_top1=[1.0],  # pi_chosen missing
        )
        with pytest.raises(ValueError, match="disagree on length"):
            writer.write_iteration({"iteration": 0}, [episode], {})


# ---------------------------------------------------------------------------
# Isolation and resume


def test_telemetry_leaves_training_untouched(tmp_path, monkeypatch):
    """Telemetry leaves training RNG state, metric rows, and model outputs
    unchanged."""

    class Silent:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def begin_run(self, *a, **k):
            return None

        def write_iteration(self, *a, **k):
            return None

    _train(tmp_path / "captured", iterations=2)
    monkeypatch.setattr(run_mod, "open_telemetry", lambda out_dir: Silent())
    _train(tmp_path / "silent", iterations=2)

    read = lambda name: [  # noqa: E731
        json.loads(line)
        for line in (tmp_path / name / "metrics.jsonl").read_text().splitlines()
    ]
    captured, silent = read("captured"), read("silent")
    assert len(captured) == 2
    for a, b in zip(captured, silent, strict=True):
        assert a.keys() == b.keys()
        for key in a:
            if key != "seconds":  # wall clock, not a property of the run
                assert a[key] == b[key], key
    assert not tel.db_path(tmp_path / "silent").exists()


def test_resume_supersedes_its_replayed_tail(tmp_path):
    """Resume replaces rows from the restored iteration onward and preserves
    earlier rows with unique live game identifiers."""
    _train(tmp_path, iterations=3)
    conn = tel.connect(tmp_path)
    kept = {
        g["game_id"]: tel.fetch_game(conn, g["game_id"])["moves"]
        for g in tel.search_games(conn, iterations=(0, 0))
    }
    assert len(tel.search_games(conn, limit=1000)) == 6
    conn.close()

    _train(tmp_path, iterations=4, start_iteration=1)  # replays 1 and 2

    conn = tel.connect(tmp_path)
    assert [r["iteration"] for r in tel.iteration_series(conn, ["f"])] == [0, 1, 2, 3]
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 2
    games = tel.search_games(conn, limit=1000)
    assert len({g["game_id"] for g in games}) == len(games) == 8
    assert len({(g["iteration"], g["game_index"]) for g in games}) == 8
    # Reallocated game identifiers leave no ply referencing a deleted row.
    for game_id, moves in kept.items():
        assert tel.fetch_game(conn, game_id)["moves"] == moves
    orphans = conn.execute(
        "SELECT COUNT(*) FROM plies WHERE game_id NOT IN (SELECT game_id FROM games)"
    ).fetchone()[0]
    assert orphans == 0
    conn.close()


# ---------------------------------------------------------------------------
# The read layer


def _fixture_db(tmp_path):
    """A hand-built database: two finished games and one capped, with v̂
    chosen so every bucket's arithmetic is checkable by hand."""
    writer = tel.open_telemetry(tmp_path)
    writer.begin_run({"iterations": 0}, {"v": 1}, 0)
    conn = writer._conn
    conn.execute(
        "INSERT INTO iterations (iteration, run, games, plies, metrics_json)"
        " VALUES (0, 1, 3, 5, '{}')"
    )
    games = [
        # (game_id, winner, capped, moves)
        (0, 0, 0, [(0, 0), (1, 0), (2, 0)]),
        (1, 1, 0, [(0, 0), (1, 0)]),
        (2, None, 1, [(0, 0)]),
    ]
    for game_id, winner, capped, moves in games:
        conn.execute(
            "INSERT INTO games (game_id, kind, iteration, game_index, winner,"
            " length, capped, moves) VALUES (?, 'selfplay', 0, ?, ?, ?, ?, ?)",
            (game_id, game_id, winner, len(moves), capped, tel.pack_moves(moves)),
        )
    plies = [
        # game 0 (P0 wins): movers 0, 1, 0 → outcomes +1, -1, +1
        (0, 0, 0, 0.30), (0, 1, 1, -0.90), (0, 2, 0, 0.95),
        # game 1 (P1 wins): movers 0, 1 → outcomes -1, +1
        (1, 0, 0, 0.50), (1, 1, 1, 0.10),
        # game 2 is capped: no outcome, so calibration must ignore it
        (2, 0, 0, 9.00),
    ]
    conn.executemany(
        "INSERT INTO plies (game_id, t, mover, moves_remaining, legal_count,"
        " rank, v_hat, kl, norm_entropy, pi_top1, pi_chosen)"
        " VALUES (?, ?, ?, 1, 7, 0, ?, 0, 5000, 5000, 5000)",
        [(g, t, m, round(v * tel._Q)) for g, t, m, v in plies],
    )
    conn.commit()
    return writer


# Schema-v1 input fixture for conversion, including REAL ply scalars and no
# forfeit columns.
_V1_SCHEMA = """
CREATE TABLE schema_version (version INTEGER NOT NULL);
CREATE TABLE runs (id INTEGER PRIMARY KEY, created TEXT NOT NULL,
    start_iteration INTEGER NOT NULL, iterations INTEGER NOT NULL,
    config_json TEXT NOT NULL, versions_json TEXT NOT NULL);
CREATE TABLE iterations (iteration INTEGER PRIMARY KEY,
    run INTEGER NOT NULL REFERENCES runs(id), f REAL, acting_kl REAL,
    acting_norm_entropy REAL, won_length_mean REAL, p0_win_rate REAL,
    first_stone_win_rate REAL, v_hat_winner_mean REAL, v_hat_loser_mean REAL,
    v_hat_mae REAL, buffer_samples INTEGER, policy_loss REAL, q_loss REAL,
    fit_steps INTEGER, seconds REAL, eval_score REAL, eval_capped INTEGER,
    eval_games INTEGER, eval_seconds REAL, samples_per_s REAL,
    games INTEGER NOT NULL, plies INTEGER NOT NULL, hw_samples INTEGER,
    cpu_percent_mean REAL, cpu_percent_max REAL, threads_mean REAL,
    threads_max INTEGER, rss_mean REAL, rss_max INTEGER,
    sys_ram_used_mean REAL, sys_ram_used_max INTEGER, gpu_util_mean REAL,
    gpu_util_max REAL, gpu_power_w_mean REAL, gpu_power_w_max REAL,
    gpu_mem_used_mean REAL, gpu_mem_used_max INTEGER, gpu_temp_mean REAL,
    gpu_temp_max REAL, torch_alloc_max INTEGER, torch_reserved_max INTEGER,
    metrics_json TEXT NOT NULL);
CREATE TABLE opponents (opponent_id INTEGER PRIMARY KEY, name TEXT NOT NULL,
    config_json TEXT NOT NULL, UNIQUE (name, config_json));
CREATE TABLE eval_matches (match_id INTEGER PRIMARY KEY, created TEXT NOT NULL,
    source TEXT NOT NULL, opponent INTEGER NOT NULL, iteration INTEGER,
    checkpoint TEXT, games INTEGER NOT NULL, score REAL NOT NULL,
    win_rate REAL NOT NULL, capped INTEGER NOT NULL,
    ci_lo REAL, ci_hi REAL, elo REAL, elo_lo REAL, elo_hi REAL,
    score_as_p0 REAL, score_as_p1 REAL, opponent_depth_mean REAL,
    avg_plies REAL, seconds REAL);
CREATE TABLE games (game_id INTEGER PRIMARY KEY, kind TEXT NOT NULL,
    iteration INTEGER, match INTEGER, game_index INTEGER NOT NULL,
    winner INTEGER, length INTEGER NOT NULL, capped INTEGER NOT NULL,
    model_seat INTEGER, opening_len INTEGER, opponent_depth_mean REAL,
    moves BLOB NOT NULL);
CREATE TABLE plies (game_id INTEGER NOT NULL, t INTEGER NOT NULL,
    mover INTEGER NOT NULL, moves_remaining INTEGER NOT NULL,
    legal_count INTEGER NOT NULL, rank INTEGER NOT NULL, v_hat REAL NOT NULL,
    kl REAL NOT NULL, norm_entropy REAL NOT NULL, pi_top1 REAL NOT NULL,
    pi_chosen REAL NOT NULL, PRIMARY KEY (game_id, t)) WITHOUT ROWID;
CREATE TABLE crossplay (checkpoint_a TEXT NOT NULL, checkpoint_b TEXT NOT NULL,
    games INTEGER NOT NULL, score_a REAL NOT NULL, capped INTEGER NOT NULL,
    ply_cap INTEGER NOT NULL, seed INTEGER NOT NULL, created TEXT NOT NULL,
    PRIMARY KEY (checkpoint_a, checkpoint_b));
"""


def test_convert_regenerates_a_v1_database(tmp_path):
    import sqlite3

    path = tmp_path / tel.DB_NAME
    v1 = sqlite3.connect(path)
    with v1:
        v1.executescript(_V1_SCHEMA)
        v1.execute("INSERT INTO schema_version VALUES (1)")
        v1.execute("INSERT INTO runs VALUES (1, 'then', 0, 5, '{}', '{}')")
        v1.execute(
            "INSERT INTO iterations (iteration, run, f, games, plies,"
            " metrics_json) VALUES (7, 1, 0.5, 3, 40, '{}')"
        )
        v1.execute("INSERT INTO opponents VALUES (1, 'sealbot', '{}')")
        v1.execute(
            "INSERT INTO eval_matches VALUES (1, 'then', 'driver', 1, 5, NULL,"
            " 2, 1.5, 0.75, 0, 0.3, 0.9, 190.8, NULL, NULL, 1.0, 0.5, 3.5,"
            " 40.0, 12.0)"
        )
        v1.execute(
            "INSERT INTO games VALUES (0, 'selfplay', 4, NULL, 0, 1, 2, 0,"
            " NULL, NULL, NULL, ?)",
            (tel.pack_moves([(0, 0), (1, 0)]),),
        )
        v1.executemany(
            "INSERT INTO plies VALUES (0, ?, ?, 1, 9, 0, ?, ?, ?, ?, ?)",
            [
                (0, 0, 0.31418, 0.021, 0.5, 0.25, 0.125),
                (1, 1, -0.90071, 0.002, 0.9, 0.75, 0.5),
            ],
        )
    v1.close()

    tel.convert_v1(tmp_path)
    assert (tmp_path / "telemetry.db.v1.bak").exists()

    conn = tel.connect(tmp_path)  # a current-schema open proves the version
    game = tel.fetch_game(conn, 0)
    assert game["moves"] == [(0, 0), (1, 0)]
    assert game["plies"][0]["v_hat"] == pytest.approx(0.31418, abs=0.5 / tel._Q)
    assert game["plies"][1]["v_hat"] == pytest.approx(-0.90071, abs=0.5 / tel._Q)
    assert game["plies"][1]["pi_chosen"] == pytest.approx(0.5)
    row = dict(conn.execute("SELECT * FROM eval_matches").fetchone())
    assert row["win_rate"] == 0.75 and row["forfeits"] is None
    assert dict(conn.execute("SELECT * FROM runs").fetchone())["iterations"] == 5
    assert conn.execute(
        "SELECT iteration FROM iterations"
    ).fetchone()[0] == 7

    # Conversion preserves rows from every schema-v1 table.
    backup = sqlite3.connect(tmp_path / "telemetry.db.v1.bak")
    tables = [
        row[0]
        for row in backup.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
        if row[0] != "schema_version"
    ]
    for table in tables:
        expected = backup.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        carried = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        assert carried == expected, f"{table}: {carried} of {expected} rows survived"
    backup.close()
    conn.close()

    with pytest.raises(ValueError, match="v1 databases only"):
        tel.convert_v1(tmp_path)  # already the current schema


def test_calibration_matches_arithmetic_by_hand(tmp_path):
    writer = _fixture_db(tmp_path)
    conn = tel.connect(tmp_path)

    rows = {r["bucket"]: r for r in tel.calibration(conn, by="v_hat", bucket=0.5)}
    # Buckets are floor(v̂ / 0.5): -0.90 → -2, 0.10 and 0.30 → 0, 0.50 and
    # 0.95 → 1. The capped game's 9.0 is absent, having no outcome.
    assert set(rows) == {-2, 0, 1}
    assert rows[-2]["plies"] == 1
    assert rows[-2]["bucket_lo"] == pytest.approx(-1.0)
    assert rows[-2]["outcome_mean"] == pytest.approx(-1.0)  # the loser's ply
    assert rows[-2]["mae"] == pytest.approx(abs(-0.90 - -1.0))

    assert rows[0]["plies"] == 2  # v̂ 0.30 (outcome +1) and 0.10 (outcome +1)
    assert rows[0]["v_hat_mean"] == pytest.approx(0.20)
    assert rows[0]["outcome_mean"] == pytest.approx(1.0)
    assert rows[0]["mae"] == pytest.approx((0.70 + 0.90) / 2)

    assert rows[1]["plies"] == 2  # v̂ 0.50 (outcome -1) and 0.95 (outcome +1)
    assert rows[1]["outcome_mean"] == pytest.approx(0.0)
    assert rows[1]["mae"] == pytest.approx((1.50 + 0.05) / 2)

    by_ply = {r["bucket"]: r for r in tel.calibration(conn, by="ply", bucket=2)}
    assert by_ply[0]["plies"] == 4  # t = 0, 1 of both finished games
    assert by_ply[1]["plies"] == 1  # t = 2 of the first
    assert sum(r["plies"] for r in tel.calibration(conn, by="length", bucket=1)) == 5
    conn.close()
    writer.close()


def test_calibration_buckets_floor_exactly_across_the_sign(tmp_path):
    """Reliability buckets use mathematical floor for signed v̂ values,
    including negative values on and off bucket edges."""
    import math

    writer = tel.open_telemetry(tmp_path)
    writer.begin_run({"iterations": 0}, {"v": 1}, 0)
    conn = writer._conn
    conn.execute(
        "INSERT INTO iterations (iteration, run, games, plies, metrics_json)"
        " VALUES (0, 1, 1, 0, '{}')"
    )
    conn.execute(
        "INSERT INTO games (game_id, kind, iteration, game_index, winner, length,"
        " capped, moves) VALUES (0, 'selfplay', 0, 0, 0, 1, 0, ?)",
        (tel.pack_moves([(0, 0)]),),
    )
    values = [-2.0, -1.75, -1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.25]
    conn.executemany(
        "INSERT INTO plies (game_id, t, mover, moves_remaining, legal_count, rank,"
        " v_hat, kl, norm_entropy, pi_top1, pi_chosen)"
        " VALUES (0, ?, 0, 1, 1, 0, ?, 0, 0, 10000, 10000)",
        [(t, round(v * tel._Q)) for t, v in enumerate(values)],
    )
    conn.commit()
    writer.close()

    conn = tel.connect(tmp_path)
    for width in (0.25, 0.5, 1.0):
        got = {r["bucket"]: r["plies"] for r in tel.calibration(conn, bucket=width)}
        want: dict[int, int] = {}
        for v in values:
            want[math.floor(v / width)] = want.get(math.floor(v / width), 0) + 1
        assert got == want, width
        for row in tel.calibration(conn, bucket=width):
            assert row["bucket_lo"] == pytest.approx(row["bucket"] * width)
    conn.close()


def test_blunder_query_reads_swings_in_the_mover_s_frame(tmp_path):
    writer = _fixture_db(tmp_path)
    conn = tel.connect(tmp_path)

    rows = tel.blunders(conn, threshold=0.1)
    swings = {(r["game_id"], r["t"]): r["swing"] for r in rows}
    # Game 0, ply 0 → 1: the seat changes, so v̂ = -0.90 for the opponent is
    # +0.90 for this mover — a swing of +0.60, not the -1.20 a raw
    # subtraction would report.
    assert swings[(0, 0)] == pytest.approx(0.90 - 0.30)
    assert swings[(1, 0)] == pytest.approx(-0.10 - 0.50)
    assert [r["swing"] for r in rows] == sorted(swings.values(), key=lambda s: -abs(s))

    # Game 0's second ply also changes seat, and in that frame the two
    # assessments agree to 0.05 — a steady position, below the threshold.
    quiet = {(r["game_id"], r["t"]): r["swing"] for r in tel.blunders(conn, threshold=0.0)}
    assert quiet[(0, 1)] == pytest.approx(-0.95 - -0.90)
    assert set(quiet) - set(swings) == {(0, 1)}
    assert tel.blunders(conn, threshold=1.0) == []
    assert tel.blunders(conn, threshold=0.1, limit=1) == rows[:1]
    conn.close()
    writer.close()


def test_search_and_series_filter_as_asked(tmp_path):
    writer = _fixture_db(tmp_path)
    conn = tel.connect(tmp_path)

    assert len(tel.search_games(conn)) == 3
    assert [g["game_id"] for g in tel.search_games(conn, winner=1)] == [1]
    assert [g["game_id"] for g in tel.search_games(conn, capped=True)] == [2]
    assert [g["game_id"] for g in tel.search_games(conn, min_length=2, max_length=2)] == [1]
    assert tel.search_games(conn, kind="eval") == []
    assert tel.search_games(conn, iterations=(1, None)) == []
    assert [g["length"] for g in tel.search_games(conn, order="longest")] == [3, 2, 1]
    assert [g["length"] for g in tel.search_games(conn, order="shortest")] == [1, 2, 3]
    with pytest.raises(ValueError, match="order must be one of"):
        tel.search_games(conn, order="length; DROP TABLE games")

    assert tel.iteration_series(conn, ["games", "plies"]) == [
        {"iteration": 0, "games": 3, "plies": 5}
    ]
    with pytest.raises(ValueError, match="no such iteration columns"):
        tel.iteration_series(conn, ["f", "not_a_metric"])

    atlas = tel.opening_atlas(conn, plies=2)
    assert len(atlas) == 1  # both 2+ ply games open the same way up to symmetry
    assert atlas[0]["games"] == 2 and atlas[0]["p0_wins"] == 1 and atlas[0]["p1_wins"] == 1
    conn.close()
    writer.close()


def test_every_browse_order_comes_off_an_index(tmp_path):
    """Every ``GAME_ORDERS`` entry has a matching index and query plans use
    no temporary sort B-tree."""

    class Spy:
        """Records the SQL `search_games` runs, unexpanded: the plan for a
        query with `kind = ?` is not the plan for one with the value
        inlined, and it is the parameterized one the deck issues."""

        def __init__(self, conn):
            self.conn, self.calls = conn, []

        def execute(self, sql, params=()):
            self.calls.append((sql, list(params)))
            return self.conn.execute(sql, params)

    writer = _fixture_db(tmp_path)
    conn = tel.connect(tmp_path)

    def plan_of(**kwargs) -> str:
        spy = Spy(conn)
        tel.search_games(spy, **kwargs)
        (sql, params), = spy.calls
        return " / ".join(
            row[3] for row in conn.execute("EXPLAIN QUERY PLAN " + sql, params)
        )

    for order in tel.GAME_ORDERS:
        plan = plan_of(kind="selfplay", order=order)
        assert f"games_browse_{order}" in plan, (order, plan)
        assert "TEMP B-TREE" not in plan, (order, plan)

    # An iteration floor and default order use a range seek on one index.
    plan = plan_of(kind="selfplay", iterations=(1, None))
    assert "games_browse_recent" in plan and "iteration>" in plan, plan
    assert "TEMP B-TREE FOR ORDER BY" not in plan, plan
    conn.close()
    writer.close()


def test_a_database_without_the_browse_indexes_gains_them_from_a_writer(tmp_path):
    """A writer adds missing derived browse indexes without changing
    ``SCHEMA_VERSION``; read-only opens do not add them."""
    _fixture_db(tmp_path).close()
    browse = {f"games_browse_{order}" for order in tel.GAME_ORDERS}

    raw = sqlite3.connect(tel.db_path(tmp_path))
    with raw:  # Construct schema v2 without the required derived indexes.
        for name in browse:
            raw.execute(f"DROP INDEX {name}")
        raw.execute("CREATE INDEX games_search ON games(kind, winner, length)")
    version = raw.execute("SELECT version FROM schema_version").fetchone()[0]
    raw.close()

    with tel.open_telemetry(tmp_path):
        pass

    conn = tel.connect(tmp_path)
    indexes = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
            " AND tbl_name = 'games'"
        )
    }
    # One per order, named after it: an order with no index behind it is a
    # page that sorts the whole run.
    assert browse <= indexes
    assert "games_search" not in indexes  # The writer removes non-contract indexes.
    assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == version
    assert [g["game_id"] for g in tel.search_games(conn)] == [0, 1, 2]
    conn.close()


def test_summary_and_cli_read_a_real_run(tmp_path, capsys):
    _train(tmp_path, iterations=1)
    conn = tel.connect(tmp_path)
    s = tel.summary(conn)
    assert s["invocations"] == 1 and s["iterations"] == 1
    assert s["selfplay_games"] == 2 and s["plies"] > 0
    assert s["eval_matches"] == 0 and s["opponents"] == []
    assert s["latest"]["iteration"] == 0
    conn.close()

    tel.main(["--run", str(tmp_path), "summary"])
    tel.main(["--run", str(tmp_path), "games", "--limit", "5"])
    out = capsys.readouterr().out
    assert "self-play games" in out and "selfplay" in out


# ---------------------------------------------------------------------------
# Evaluation and cross-play records


def test_eval_matches_key_on_the_opponent_not_on_sealbot(tmp_path):
    """Opponents are rows, not columns: two depth caps are two opponents,
    the same cap twice is one, and strength curves query by opponent id."""
    from mantisnet.klent.sealbot import record_match

    def match(score, depth, iteration):
        result = {
            "score": score, "games": 2, "capped": 0, "win_rate": score / 2,
            "ci_lo": 0.0, "ci_hi": 1.0, "elo": 0.0, "elo_lo": -1.0, "elo_hi": 1.0,
            "score_as_p0": score, "score_as_p1": 0.0, "avg_plies": 12.0,
            "seconds": 1.0, "opponent_depth_mean": 1.5,
            "opponent_name": "sealbot",
            "opponent_config": {
                "variant": "current", "time_limit": 0.05, "max_depth": depth
            },
        }
        games = [
            {"seat": 0, "winner": 0, "capped": False, "score": 1.0, "opening_len": 2,
             "depth_mean": 1.5, "moves": [(0, 0), (1, 0)]},
            {"seat": 1, "winner": 0, "capped": False, "score": 0.0, "opening_len": 2,
             "depth_mean": 1.5, "moves": [(0, 0), (1, 0), (2, 0)]},
        ]
        return result, games, iteration

    with tel.open_telemetry(tmp_path) as writer:
        writer.begin_run({"iterations": 0}, {"v": 1}, 0)
        for score, depth, iteration in ((1.0, 1, 10), (2.0, 1, 20), (0.0, 3, 20)):
            result, games, it = match(score, depth, iteration)
            record_match(
                writer, result, games, source="cli",
                iteration=it, checkpoint=f"checkpoint_{it:06d}.pt",
            )

    conn = tel.connect(tmp_path)
    opponents = {o["opponent_id"]: json.loads(o["config_json"]) for o in
                 conn.execute("SELECT * FROM opponents")}
    assert len(opponents) == 2  # depth 1 twice is one opponent, depth 3 another
    depth_1 = next(i for i, c in opponents.items() if c["max_depth"] == 1)

    curve = tel.strength_curve(conn, opponent_id=depth_1)
    assert [(m["iteration"], m["win_rate"]) for m in curve] == [(10, 0.5), (20, 1.0)]
    assert {m["opponent_name"] for m in curve} == {"sealbot"}
    assert all(m["source"] == "cli" for m in curve)

    eval_games = tel.search_games(conn, kind="eval")
    assert len(eval_games) == 6
    assert {g["model_seat"] for g in eval_games} == {0, 1}
    assert all(g["opening_len"] == 2 for g in eval_games)
    # Evaluation games carry moves but no training-time per-ply trace.
    assert tel.fetch_game(conn, eval_games[0]["game_id"])["plies"] == []
    conn.close()


def test_crossplay_rows_replace_the_previous_matrix(tmp_path):
    rows = [{"a": "checkpoint_000001.pt", "b": "checkpoint_000002.pt",
             "score_a": 0.25, "capped": 1, "games": 4}]
    with tel.open_telemetry(tmp_path) as writer:
        writer.begin_run({"iterations": 0}, {"v": 1}, 0)
        writer.write_crossplay(rows, ply_cap=12, seed=0)
        writer.write_crossplay(
            [dict(rows[0], score_a=0.75)] + [
                {"a": "checkpoint_000001.pt", "b": "checkpoint_000003.pt",
                 "score_a": 0.5, "capped": 0, "games": 4}
            ],
            ply_cap=12, seed=0,
        )
    conn = tel.connect(tmp_path)
    matrix = tel.crossplay_matrix(conn)
    assert len(matrix) == 2
    assert matrix[0]["score_a"] == 0.75 and matrix[0]["ply_cap"] == 12
    conn.close()


# ---------------------------------------------------------------------------
# The policy debugger


def test_inspect_position_matches_a_direct_forward():
    """``inspect_position`` matches direct policy, Q, and closed-form
    improvement over engine legal order."""
    from mantisnet.builder import collate_positions
    from mantisnet.klent.improve import improved_policy

    model = _tiny_model().eval()
    moves = [(0, 0), (1, 1), (2, 0), (-1, 0), (0, 2)]
    tau, lam, floor = 0.1, 0.03, 0.2
    out = inspect_position(model, moves, t=3, tau=tau, lam=lam, mass_floor=floor)

    position = hexo_py.Position.replay(moves[:3])
    batch = collate_positions([position])
    with torch.no_grad():
        result = model(batch, floor)
    imp = improved_policy(
        result.policy_logits, result.q_score, result.q_values,
        batch.legal_offsets, tau, lam,
    )
    policy = torch.softmax(result.policy_logits, 0)

    assert out["legal_count"] == position.legal_count
    assert out["mover"] == position.current_player
    assert out["moves_remaining"] == position.moves_remaining
    assert out["played"] == moves[3]
    assert out["v_hat"] == pytest.approx(float(imp.v_hat[0]), abs=1e-6)
    assert out["kl"] == pytest.approx(float(imp.kl[0]), abs=1e-6)
    assert out["norm_entropy"] == pytest.approx(float(imp.norm_entropy[0]), abs=1e-6)

    assert [e["move"] for e in out["legal"]] == position.legal_moves()
    for rank, entry in enumerate(out["legal"]):
        assert entry["rank"] == rank
        assert entry["logit"] == pytest.approx(float(result.policy_logits[rank]), abs=1e-5)
        assert entry["policy"] == pytest.approx(float(policy[rank]), abs=1e-6)
        assert entry["q"] == pytest.approx(float(result.q_values[rank]), abs=1e-5)
        assert entry["improved"] == pytest.approx(float(imp.probs[rank]), abs=1e-6)
    assert sum(e["improved"] for e in out["legal"]) == pytest.approx(1.0, abs=1e-5)
    assert sum(e["policy"] for e in out["legal"]) == pytest.approx(1.0, abs=1e-5)


def test_inspect_position_loads_a_checkpoint_by_path(tmp_path, model):
    """Path-based inspection matches inspection of the loaded model object."""
    path = tmp_path / "checkpoint_000001.pt"
    save_checkpoint(
        path, model, torch.optim.Adam(model.parameters()), 1, np.random.default_rng(0)
    )
    moves = [(0, 0), (1, 1), (2, 0)]
    by_path = inspect_position(path, moves, 2, 0.1, 0.03, 0.2)
    by_object = inspect_position(model, moves, 2, 0.1, 0.03, 0.2)
    assert by_path == by_object


def test_inspect_position_walks_a_line_one_move_at_a_time():
    """Inspection accepts a growing prefix and identifies the next played
    move at every ply."""
    model = _tiny_model().eval()
    moves = [(0, 0), (1, 1), (2, 0), (-1, 0), (0, 2)]
    for t in range(len(moves) + 1):
        out = inspect_position(model, moves, t, 0.1, 0.03, 0.2)
        assert out["stone_count"] == t
        assert out["played"] == (moves[t] if t < len(moves) else None)
        assert out["legal_count"] == len(out["legal"]) > 0

    with pytest.raises(ValueError, match="outside"):
        inspect_position(model, moves, len(moves) + 1, 0.1, 0.03, 0.2)


def test_inspect_position_reproduces_a_recorded_ply(tmp_path):
    """Inspection with the actor weights matches the stored policy reductions."""
    torch.manual_seed(2)
    model = MantisNet(MantisConfig(h=32, blocks=1, heads=2, value_queries=2, value_bins=5))
    optimizer = torch.optim.Adam(model.parameters())
    cfg = KlentConfig(games_per_iteration=2, envs=2, ply_cap=24, batch_size=64)
    run_mod.run_training(
        model, optimizer, cfg, iterations=1, out_dir=tmp_path,
        rng=np.random.default_rng(5), checkpoint_every=1,
    )
    # Recreate the pre-fit actor weights used to collect the stored plies.
    torch.manual_seed(2)
    actor = MantisNet(MantisConfig(h=32, blocks=1, heads=2, value_queries=2, value_bins=5))
    actor.eval()

    conn = tel.connect(tmp_path)
    game = tel.fetch_game(conn, tel.search_games(conn)[0]["game_id"])
    conn.close()
    assert game["plies"]
    # The tolerance covers the quantization half-step plus float noise.
    q = 0.5 / tel._Q + 1e-6
    for ply in game["plies"][:3]:
        out = inspect_position(
            actor, game["moves"], ply["t"], cfg.tau, cfg.lam, cfg.mass_floor
        )
        probs = [e["improved"] for e in out["legal"]]
        assert out["legal_count"] == ply["legal_count"]
        assert out["v_hat"] == pytest.approx(ply["v_hat"], abs=q)
        assert out["kl"] == pytest.approx(ply["kl"], abs=q)
        assert out["norm_entropy"] == pytest.approx(ply["norm_entropy"], abs=q)
        assert max(probs) == pytest.approx(ply["pi_top1"], abs=q)
        assert probs[ply["rank"]] == pytest.approx(ply["pi_chosen"], abs=q)
