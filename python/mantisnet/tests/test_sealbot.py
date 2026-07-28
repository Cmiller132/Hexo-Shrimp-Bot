"""SealBot integration: two independent rule implementations must agree.

These tests need a SealBot checkout with a built ``minimax_cpp``; point
``SEALBOT_ROOT`` at one to enable them. Elsewhere they skip with this reason
visible — the checkout is machine-local, deliberately not vendored here.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

SEALBOT = os.environ.get("SEALBOT_ROOT")
pytestmark = pytest.mark.skipif(
    SEALBOT is None, reason="SEALBOT_ROOT not set (external SealBot checkout)"
)


def test_rules_agree_over_line_builder_games():
    """Every hexo-engine game replays into SealBot's HexGame placement for
    placement — same legality, same turn structure, same winner. A second,
    unrelated rules implementation acting as an oracle."""
    from mantisnet.klent.sealbot import _mirror, load_sealbot
    from mantisnet.klent.seeds import line_builder_game

    game_mod, _ = load_sealbot(SEALBOT)
    rng = np.random.default_rng(2)
    finished = 0
    for _ in range(15):
        moves, winner = line_builder_game(rng, noise=0.1)
        g = _mirror(game_mod, [tuple(m) for m in moves])
        if winner is None:
            assert not g.game_over
            continue
        finished += 1
        assert g.game_over
        assert g.winner.value - 1 == winner
    assert finished >= 5, "the line builder should usually finish games"


def test_sealbot_opponent_grounds_collection():
    """Real SealBot as the grounding opponent: legal whole turns, finished
    games, records on the model side only."""
    from mantisnet.klent import play_episodes
    from mantisnet.klent.sealbot import sealbot_opponent

    from .test_klent_pipeline import heuristic_evaluate

    opponent = sealbot_opponent(SEALBOT, depth=1, time_limit=0.05)
    episodes, _ = play_episodes(
        heuristic_evaluate, [[], []], 200, 0.1, 0.03, np.random.default_rng(3),
        opponent=opponent, opponent_seats=[0, 1],
    )
    for ep, seat in zip(episodes, (0, 1)):
        assert ep.winner is not None, "SealBot finishes games"
        assert all(m == 1 - seat for m in ep.movers)
        assert len(ep.ts) < len(ep.moves)


def test_sealbot_match_smoke():
    """A tiny untrained model survives a real paired match: legal moves
    only, agreed winners, sane accounting."""
    import torch

    from mantisnet import MantisConfig, MantisNet
    from mantisnet.klent.sealbot import sealbot_match

    torch.manual_seed(0)
    model = MantisNet(
        MantisConfig(h=32, blocks=1, heads=2, value_queries=2, value_bins=5)
    )
    result = sealbot_match(
        model, "cpu", games=2, ply_cap=80, rng=np.random.default_rng(0),
        time_limit=0.02, sealbot_root=SEALBOT,
    )
    assert result["games"] == 2
    assert 0.0 <= result["score"] <= 2.0
    assert result["score_as_p0"] + result["score_as_p1"] == result["score"]
    assert result["capped"] + result["avg_plies"] >= 0
    assert result["ci_lo"] <= result["win_rate"] <= result["ci_hi"]
