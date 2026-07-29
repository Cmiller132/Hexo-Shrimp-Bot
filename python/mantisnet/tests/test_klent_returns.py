"""``KLENT_FOR_HEXO.md`` §1.2/§1.3: the sign function and the λ-return,
pinned per ``KLENT_FOR_HEXO.md`` §1.4.

The engine cross-check is the one test that catches a parity implementation
(K1): the sign derived from the phase is compared against the mover actually
changing, using the engine's own reported movers.
"""

from __future__ import annotations

import hexo_py
import numpy as np
import pytest

from mantisnet.klent import lambda_returns, signs_from_moves_remaining

from .conftest import random_moves

# P0 completes (0,0)..(5,0) with (5,0) as its turn's FIRST stone (T = 11).
FIRST_STONE_WIN = [
    (0, 0),
    (-8, 8), (-8, 9),
    (1, 0), (2, 0),
    (-8, 10), (-6, 8),
    (3, 0), (4, 0),
    (-6, 9), (-6, 10),
    (5, 0),
]

# P0 spends its turn's first stone at (0,5) and wins on the SECOND (T = 12).
SECOND_STONE_WIN = [
    (0, 0),
    (-8, 8), (-8, 9),
    (1, 0), (2, 0),
    (-8, 10), (-6, 8),
    (3, 0), (4, 0),
    (-6, 9), (-6, 10),
    (0, 5), (5, 0),
]


def _walk(moves):
    """Replay, recording (moves_remaining, mover) at each acting ply."""
    pos = hexo_py.Position()
    mr, movers = [], []
    for q, r in moves:
        mr.append(pos.moves_remaining)
        movers.append(pos.current_player)
        pos.advance(q, r)
    return np.array(mr), np.array(movers), pos


def test_signs_follow_the_design_table():
    # KLENT_FOR_HEXO.md §1.2's walk: Opening, then P1's two, then P0's two.
    assert signs_from_moves_remaining([1, 2, 1, 2, 1]).tolist() == [-1, 1, -1, 1, -1]
    with pytest.raises(ValueError):
        signs_from_moves_remaining([1, 3])


def test_sign_equals_mover_change_by_the_engines_own_movers():
    plies = 0
    for seed in range(30):
        moves = random_moves(60, seed=seed)
        mr, movers, _ = _walk(moves)
        signs = signs_from_moves_remaining(mr)
        for t in range(len(moves) - 1):
            expected = 1 if movers[t + 1] == movers[t] else -1
            assert signs[t] == expected, f"seed {seed} ply {t}"
            plies += 1
    assert plies > 1500


@pytest.mark.parametrize("moves", [FIRST_STONE_WIN, SECOND_STONE_WIN])
def test_monte_carlo_identity_at_lambda_one(moves):
    # λ = 1: G_t = +1 on the winner's plies and −1 on the loser's — for a win
    # on either half of a turn (K2).
    mr, movers, pos = _walk(moves)
    assert pos.is_terminal and pos.winner == 0
    g = lambda_returns(signs_from_moves_remaining(mr), np.zeros(len(moves)), 1.0, 1.0)
    expected = np.where(movers == pos.winner, 1.0, -1.0)
    assert np.allclose(g, expected)


@pytest.mark.parametrize("moves", [FIRST_STONE_WIN, SECOND_STONE_WIN])
def test_gamma_discounts_by_distance_to_terminal(moves):
    # λ = 1, γ < 1: |G_t| = γ^(T−t), the sign still the mover's frame.
    mr, movers, pos = _walk(moves)
    assert pos.is_terminal and pos.winner == 0
    g = lambda_returns(signs_from_moves_remaining(mr), np.zeros(len(moves)), 1.0, 0.9)
    steps = len(moves) - 1 - np.arange(len(moves))
    expected = np.where(movers == pos.winner, 1.0, -1.0) * 0.9**steps
    assert np.allclose(g, expected)


@pytest.mark.parametrize("moves", [FIRST_STONE_WIN, SECOND_STONE_WIN])
def test_one_step_bootstrap_identity_at_lambda_zero(moves):
    mr, _movers, _pos = _walk(moves)
    rng = np.random.default_rng(4)
    v = rng.uniform(-1, 1, len(moves))
    signs = signs_from_moves_remaining(mr)
    g = lambda_returns(signs, v, 0.0, 1.0)
    assert g[-1] == 1.0
    for t in range(len(moves) - 1):
        assert g[t] == pytest.approx(signs[t] * v[t + 1])


def test_hand_computed_intermediate_lambda():
    signs = signs_from_moves_remaining([1, 2, 1, 2])
    g = lambda_returns(signs, [0.1, 0.2, 0.3, 0.4], 0.5, 1.0)
    assert np.allclose(g, [0.0, -0.2, -0.7, 1.0])


def test_validation():
    with pytest.raises(ValueError, match="lam_ret"):
        lambda_returns(np.array([1.0]), [0.0], 1.5, 1.0)
    with pytest.raises(ValueError, match="gamma"):
        lambda_returns(np.array([1.0]), [0.0], 1.0, 0.0)
    with pytest.raises(ValueError, match="gamma"):
        lambda_returns(np.array([1.0]), [0.0], 1.0, 1.5)
    with pytest.raises(ValueError, match="equal-length"):
        lambda_returns(np.array([1.0, -1.0]), [0.0], 0.9, 1.0)
