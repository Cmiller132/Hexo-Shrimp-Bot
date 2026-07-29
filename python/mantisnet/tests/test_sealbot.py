"""SealBot integration: two independent rule implementations must agree.

These tests need a SealBot checkout with a built ``minimax_cpp``; point
``SEALBOT_ROOT`` at one to enable them. They skip when that external checkout
is unavailable.
"""

from __future__ import annotations

import json
import os

import hexo_py
import numpy as np
import pytest

SEALBOT = os.environ.get("SEALBOT_ROOT")
pytestmark = pytest.mark.skipif(
    SEALBOT is None, reason="SEALBOT_ROOT not set (external SealBot checkout)"
)


def test_rules_agree_over_heuristic_games():
    """Every hexo-engine game replays into SealBot's HexGame placement for
    placement with the same legality, turn structure, and winner."""
    from mantisnet.klent.sealbot import _mirror, load_sealbot

    from .heuristic import heuristic_game

    game_mod, _ = load_sealbot(SEALBOT)
    rng = np.random.default_rng(2)
    finished = 0
    for _ in range(15):
        moves, winner = heuristic_game(rng, noise=0.1)
        g = _mirror(game_mod, [tuple(m) for m in moves])
        if winner is None:
            assert not g.game_over
            continue
        finished += 1
        assert g.game_over
        assert g.winner.value - 1 == winner
    assert finished >= 5, "the heuristic player should usually finish games"


def test_sealbot_match_smoke():
    """A tiny untrained model completes a paired match with legal moves,
    agreed winners, and internally consistent accounting."""
    import torch

    from mantisnet import MantisConfig, MantisNet
    from mantisnet.klent.sealbot import sealbot_match

    torch.manual_seed(0)
    model = MantisNet(
        MantisConfig(h=32, blocks=1, heads=2, value_queries=2, value_bins=5)
    )
    # Random policy-head weights avoid the zero head's first-legal sequence.
    torch.nn.init.normal_(model.mlp_p.out.weight, std=0.1)
    result, per_game = sealbot_match(
        model, "cpu", games=2, ply_cap=80, rng=np.random.default_rng(0),
        time_limit=0.02, sealbot_root=SEALBOT,
    )
    assert result["games"] == 2
    assert 0.0 <= result["score"] <= 2.0
    assert result["score_as_p0"] + result["score_as_p1"] == result["score"]
    assert result["capped"] + result["avg_plies"] >= 0
    assert result["ci_lo"] <= result["win_rate"] <= result["ci_hi"]
    assert result["opponent_config"] == {
        "variant": "current", "time_limit": 0.02, "max_depth": None
    }

    # The per-game detail is the summary, unaggregated: same seats, same
    # score, and each game's moves start with the opening it was paired on.
    assert [g["seat"] for g in per_game] == [0, 1]
    assert sum(g["score"] for g in per_game) == result["score"]
    for g in per_game:
        assert len(g["moves"]) == len(set(g["moves"])) <= 80
        assert g["opening_len"] <= len(g["moves"])
        assert (g["winner"] is None) == g["capped"]


def test_a_real_match_lands_in_the_telemetry_database(tmp_path):
    """A match produces one opponent row, one match row, and replayable games
    keyed by opponent identity."""
    import torch

    from mantisnet import MantisConfig, MantisNet
    from mantisnet.klent import telemetry as tel
    from mantisnet.klent.sealbot import _checkpoint_iteration, record_match, sealbot_match

    torch.manual_seed(0)
    model = MantisNet(
        MantisConfig(h=32, blocks=1, heads=2, value_queries=2, value_bins=5)
    )
    torch.nn.init.normal_(model.mlp_p.out.weight, std=0.1)
    result, per_game = sealbot_match(
        model, "cpu", games=2, ply_cap=60, rng=np.random.default_rng(1),
        time_limit=0.02, sealbot_root=SEALBOT, max_depth=1,
    )
    with tel.open_telemetry(tmp_path) as writer:
        writer.begin_run({"iterations": 0}, {"v": 1}, 0)
        record_match(
            writer, result, per_game, source="driver", iteration=7
        )

    conn = tel.connect(tmp_path)
    (opponent,) = conn.execute("SELECT * FROM opponents").fetchall()
    assert opponent["name"] == "sealbot"
    assert json.loads(opponent["config_json"]) == {
        "max_depth": 1, "time_limit": 0.02, "variant": "current"
    }
    (match,) = tel.strength_curve(conn)
    assert match["iteration"] == 7 and match["checkpoint"] is None
    assert match["source"] == "driver"
    assert match["games"] == 2 and match["win_rate"] == result["win_rate"]

    games = tel.search_games(conn, kind="eval")
    assert len(games) == 2
    for game, detail in zip(games, per_game, strict=True):
        full = tel.fetch_game(conn, game["game_id"])
        assert full["moves"] == detail["moves"]
        assert full["match"] == match["match_id"]
        assert full["model_seat"] == detail["seat"]
        # The moves replay through the engine to the winner that was stored.
        assert hexo_py.Position.replay(full["moves"]).winner == game["winner"]
    conn.close()

    assert _checkpoint_iteration(tmp_path / "checkpoint_000250.pt") == 250
    assert _checkpoint_iteration(tmp_path / "best.pt") is None
