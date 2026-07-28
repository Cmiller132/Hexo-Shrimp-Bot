"""The capture layer: what the database holds, and what it must not disturb.

The obligations here are the ones nothing else can see. Training must be
bit-identical with the writer removed — telemetry that perturbed the corpus
would corrupt the experiment it exists to describe. A resume must supersede
its replayed tail rather than duplicate it. The calibration query must agree
with arithmetic done by hand, since it is the instrument for the design's
§9 bias. And `inspect_position` must reproduce collection's own numbers,
because it stands in for the π′ the database deliberately does not store.
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
        KlentConfig(games_per_iteration=2, ply_cap=24, batch_size=64),
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
    """The group the opening atlas folds by, held to the engine: a
    transformed game is legal placement for placement, and ends the same
    way. A wrong sign would fold two different openings together and pass
    every test that only checks the folding is consistent."""
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
    """The capture layer draws nothing from the training RNG and adds
    nothing to the metrics row: the same run with the writer stubbed out is
    identical line for line. If telemetry ever consumed a draw, the fit
    permutation would move and the losses with it."""

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
    """A resume restarts at its checkpoint, which may be behind what was
    recorded. Those iterations are redone, so their old rows go: one row per
    iteration afterwards, ids unique among what is there, and nothing from
    before the resume point disturbed."""
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
    # Iteration 0 is behind the resume point: same games, same ids, same
    # moves. A superseded id may be handed out again — the row it named is
    # gone — but no ply is left pointing at one.
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
        " VALUES (?, ?, ?, 1, 7, 0, ?, 0.0, 0.5, 0.5, 0.5)",
        plies,
    )
    conn.commit()
    return writer


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
    """v̂ is signed and SQLite's CAST truncates toward zero, so the
    reliability axis floors by hand. An off-by-one on the negative side
    would shift half the diagram by a bucket and look entirely plausible —
    so it is checked against `math.floor` on and off the edges."""
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
        " VALUES (0, ?, 0, 1, 1, 0, ?, 0, 0, 1, 1)",
        list(enumerate(values)),
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
    the same cap twice is one, and a strength curve is a query over an
    opponent id — which is what lets a stronger engine arrive without a
    schema change."""
    from mantisnet.klent.sealbot import record_match

    def match(score, depth, iteration):
        result = {
            "score": score, "games": 2, "capped": 0, "win_rate": score / 2,
            "ci_lo": 0.0, "ci_hi": 1.0, "elo": 0.0, "elo_lo": -1.0, "elo_hi": 1.0,
            "score_as_p0": score, "score_as_p1": 0.0, "avg_plies": 12.0,
            "seconds": 1.0, "sealbot_depth_mean": 1.5,
            "sealbot_time_limit": 0.05, "sealbot_max_depth": depth,
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
                writer, result, games, variant="current", source="cli",
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
    # Evaluation games carry moves but no plies: argmax play has no π′.
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
    """`inspect_position` is what stands in for the π′ the database does not
    store, so it has to be the same arithmetic collection ran: policy, Q, and
    the closed-form improvement over the engine's own legal order."""
    from mantisnet.builder import collate_positions
    from mantisnet.klent.improve import improved_policy

    model = _tiny_model().eval()
    moves = [(0, 0), (1, 1), (2, 0), (-1, 0), (0, 2)]
    tau, lam = 0.1, 0.03
    out = inspect_position(model, moves, t=3, tau=tau, lam=lam)

    position = hexo_py.Position.replay(moves[:3])
    batch = collate_positions([position])
    with torch.no_grad():
        result = model(batch)
    imp = improved_policy(
        result.policy_logits, result.q_values, batch.legal_offsets, tau, lam
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
    """The other half of the seam: a viewer names a checkpoint file, and the
    version-checked loader is what reads it — the same numbers as the model
    it holds."""
    path = tmp_path / "checkpoint_000001.pt"
    save_checkpoint(
        path, model, torch.optim.Adam(model.parameters()), 1, np.random.default_rng(0)
    )
    moves = [(0, 0), (1, 1), (2, 0)]
    by_path = inspect_position(path, moves, 2, 0.1, 0.03)
    by_object = inspect_position(model, moves, 2, 0.1, 0.03)
    assert by_path == by_object


def test_inspect_position_walks_a_line_one_move_at_a_time():
    """The branch-and-play call pattern: one loaded model, a prefix that
    grows by a move. Each step's `played` is the next move of the line and
    the ply count follows the engine."""
    model = _tiny_model().eval()
    moves = [(0, 0), (1, 1), (2, 0), (-1, 0), (0, 2)]
    for t in range(len(moves) + 1):
        out = inspect_position(model, moves, t, 0.1, 0.03)
        assert out["stone_count"] == t
        assert out["played"] == (moves[t] if t < len(moves) else None)
        assert out["legal_count"] == len(out["legal"]) > 0

    with pytest.raises(ValueError, match="outside"):
        inspect_position(model, moves, len(moves) + 1, 0.1, 0.03)


def test_inspect_position_reproduces_a_recorded_ply(tmp_path):
    """The stored scalars are a summary of what this recomputes: for every
    recorded ply, π′'s stored maximum, its value at the taken rank, its KL
    and its entropy must come back out of the checkpoint."""
    torch.manual_seed(2)
    model = MantisNet(MantisConfig(h=32, blocks=1, heads=2, value_queries=2, value_bins=5))
    optimizer = torch.optim.Adam(model.parameters())
    cfg = KlentConfig(games_per_iteration=2, ply_cap=24, batch_size=64)
    run_mod.run_training(
        model, optimizer, cfg, iterations=1, out_dir=tmp_path,
        rng=np.random.default_rng(5), checkpoint_every=1,
    )
    # The checkpoint is written after the fit that consumed the corpus, so
    # reproduce with the weights collection actually acted through.
    torch.manual_seed(2)
    actor = MantisNet(MantisConfig(h=32, blocks=1, heads=2, value_queries=2, value_bins=5))
    actor.eval()

    conn = tel.connect(tmp_path)
    game = tel.fetch_game(conn, tel.search_games(conn)[0]["game_id"])
    conn.close()
    assert game["plies"]
    for ply in game["plies"][:3]:
        out = inspect_position(actor, game["moves"], ply["t"], cfg.tau, cfg.lam)
        probs = [e["improved"] for e in out["legal"]]
        assert out["legal_count"] == ply["legal_count"]
        assert out["v_hat"] == pytest.approx(ply["v_hat"], abs=1e-5)
        assert out["kl"] == pytest.approx(ply["kl"], abs=1e-5)
        assert out["norm_entropy"] == pytest.approx(ply["norm_entropy"], abs=1e-5)
        assert max(probs) == pytest.approx(ply["pi_top1"], abs=1e-5)
        assert probs[ply["rank"]] == pytest.approx(ply["pi_chosen"], abs=1e-5)
