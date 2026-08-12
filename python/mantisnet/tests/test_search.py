"""The evaluation-only Gumbel line search, driven through scripted evaluators."""

from __future__ import annotations

import numpy as np
import torch

import hexo_py

from mantisnet import MantisConfig, MantisNet
from mantisnet.klent.evaluate import argmax_choose
from mantisnet.klent.search import _halving_schedule, gumbel_choose
from mantisnet.klent.train import KlentConfig, network_evaluate

from .heuristic import heuristic_evaluate


def _winning_position():
    # P0 owns q=0..4 on r=0 and is to move; (5, 0) completes six. P1's six
    # stones sit on r=3 with a gap at q=5..6, so no line of theirs closes.
    moves = [
        (0, 0),
        (0, 3), (1, 3),
        (1, 0), (2, 0),
        (2, 3), (3, 3),
        (3, 0), (4, 0),
        (4, 3), (7, 3),
    ]
    pos = hexo_py.Position.replay(moves)
    assert pos.current_player == 0 and not pos.is_terminal
    return pos, (5, 0)


def _flat(batch, fill=0.0):
    return torch.full((batch.n_cells,), float(fill))


def test_immediate_win_overrides_a_better_policy_logit():
    pos, winning_move = _winning_position()
    win_rank = pos.legal_moves().index(winning_move)
    decoy_rank = 0 if win_rank != 0 else 1
    calls = 0

    def evaluate(batch):
        nonlocal calls
        logits, q = _flat(batch, -100.0), _flat(batch)
        if calls == 0:
            logits[win_rank] = 100.0
            logits[decoy_rank] = 101.0
        calls += 1
        return logits, q, q

    choose = gumbel_choose(evaluate, tau=0.1, lam=0.03, sims=32)
    move = choose([pos], np.random.default_rng(4))[0]
    assert pos.nth_legal(decoy_rank) != winning_move
    assert move == winning_move


def test_batching_matches_ordered_singleton_calls():
    positions = [
        hexo_py.Position.replay([(0, 0)]),
        hexo_py.Position.replay([(0, 0), (2, 0), (1, 2)]),
        hexo_py.Position.replay([(0, 0), (-2, 1), (2, -1), (1, 1)]),
    ]
    batch_rng = np.random.default_rng(71)
    single_rng = np.random.default_rng(71)
    choose = gumbel_choose(heuristic_evaluate, 0.1, 0.03, 16)
    together = choose(positions, batch_rng)
    separately = [choose([position], single_rng)[0] for position in positions]
    assert together == separately


def test_same_seed_is_deterministic():
    positions = [
        hexo_py.Position.replay([(0, 0)]),
        hexo_py.Position.replay([(0, 0), (1, 2), (-2, 1)]),
    ]
    choose = gumbel_choose(heuristic_evaluate, 0.1, 0.03, 32)
    first = choose(positions, np.random.default_rng(92))
    second = choose(positions, np.random.default_rng(92))
    assert first == second


def test_zero_simulations_is_exact_policy_argmax():
    torch.manual_seed(3)
    model = MantisNet(
        MantisConfig(h=32, blocks=1, heads=2, value_queries=2, value_bins=5)
    ).eval()
    torch.nn.init.normal_(model.mlp_p.out.weight, std=0.1)
    cfg = KlentConfig(device="cpu")
    positions = [
        hexo_py.Position.replay([(0, 0)]),
        hexo_py.Position.replay([(0, 0), (1, 1), (2, 0)]),
    ]
    searched = gumbel_choose(
        network_evaluate(model, cfg), cfg.tau, cfg.lam, sims=0
    )
    search_rng = np.random.default_rng(8)
    untouched_peer = np.random.default_rng(8)
    assert searched(positions, search_rng) == argmax_choose(model)(
        positions, np.random.default_rng(999)
    )
    assert search_rng.integers(2**63) == untouched_peer.integers(2**63)


def test_depth_exposes_an_opponent_reply_trap():
    root = hexo_py.Position.replay([(0, 0)])
    trap_rank, safe_rank = 0, 1
    trap_move = root.nth_legal(trap_rank)
    safe_move = root.nth_legal(safe_rank)
    root_stones = len(root.stones())

    def evaluate(batch):
        logits, q = _flat(batch, -100.0), _flat(batch)
        offsets = batch.legal_offsets.tolist()
        for row in range(batch.n_pos):
            lo, hi = offsets[row], offsets[row + 1]
            stones = int(batch.attn_valid[row].sum()) - 4
            depth = stones - root_stones
            if depth == 0:
                # This gap fixes candidate order for the seeded Gumbels while
                # remaining small enough for searched value to overturn it.
                logits[lo + trap_rank] = 20.0
                logits[lo + safe_rank] = 0.0
                continue
            logits[lo] = 10.0  # every interior line follows legal rank zero
            # Pending rows preserve candidate order. The first wave is
            # trap, safe, then the other candidates; value sorting preserves
            # that order into both waves of the second round.
            trap = row == 0
            safe = row == 1
            if depth == 1:
                root_value = 0.8 if trap else (0.2 if safe else -0.5)
                q[lo:hi] = root_value
            else:
                # After P1's second stone the mover is P0, so own-frame Q has
                # the opposite sign from the root P1 frame.
                root_value = -1.0 if trap else (0.2 if safe else -0.5)
                q[lo:hi] = -root_value
        return logits, q, q

    argmax = gumbel_choose(evaluate, 0.1, 0.03, sims=0)
    shallow = gumbel_choose(evaluate, 0.1, 0.03, sims=2)
    searched = gumbel_choose(evaluate, 0.1, 0.03, sims=32)
    assert argmax([root], np.random.default_rng(0))[0] == trap_move
    assert shallow([root], np.random.default_rng(0))[0] == trap_move
    assert searched([root], np.random.default_rng(0))[0] == safe_move


def test_network_expansion_budget_is_never_exceeded():
    positions = [
        hexo_py.Position.replay([(0, 0)]),
        hexo_py.Position.replay([(0, 0), (1, 2), (-2, 1)]),
        hexo_py.Position.replay([(0, 0), (2, 0), (0, 2), (-2, 2)]),
    ]
    rows = []

    def spy(batch):
        rows.append(batch.n_pos)
        return heuristic_evaluate(batch)

    sims = 32
    choose = gumbel_choose(spy, 0.1, 0.03, sims)
    choose(positions, np.random.default_rng(19))
    assert rows[0] == len(positions)  # the shared root forward is not a sim
    assert sum(rows[1:]) <= sims * len(positions)


def test_halving_schedule_receipts():
    assert _halving_schedule(16, 8) == [(8, 1), (4, 2)]
    assert _halving_schedule(16, 16) == [(16, 1)]
    assert _halving_schedule(32, 8) == [(8, 1), (4, 2), (2, 8)]
    assert _halving_schedule(32, 16) == [(16, 1), (8, 2)]


def _spread(choose, position, seeds):
    """The distinct moves ``choose`` returns for one position over ``seeds``."""
    return {choose([position], np.random.default_rng(s))[0] for s in seeds}


def test_unit_temperature_is_the_unscaled_draw():
    # The cross-language Gumbel fixtures are generated through this chooser, so
    # the default has to be the old behaviour exactly and not merely closely.
    positions = [
        hexo_py.Position.replay([(0, 0)]),
        hexo_py.Position.replay([(0, 0), (1, 2), (-2, 1)]),
    ]
    default = gumbel_choose(heuristic_evaluate, 0.1, 0.03, 32)
    explicit = gumbel_choose(heuristic_evaluate, 0.1, 0.03, 32, temperature=1.0)
    assert default(positions, np.random.default_rng(55)) == explicit(
        positions, np.random.default_rng(55)
    )


def test_zero_temperature_is_deterministic_across_seeds():
    position = hexo_py.Position.replay([(0, 0), (1, 2), (-2, 1)])
    cold = gumbel_choose(heuristic_evaluate, 0.1, 0.03, 32, temperature=0.0)
    assert len(_spread(cold, position, range(12))) == 1


def test_temperature_widens_the_move_distribution():
    position = hexo_py.Position.replay([(0, 0), (1, 2), (-2, 1)])
    seeds = range(24)
    cold = _spread(gumbel_choose(heuristic_evaluate, 0.1, 0.03, 32, temperature=0.0), position, seeds)
    warm = _spread(gumbel_choose(heuristic_evaluate, 0.1, 0.03, 32, temperature=1.0), position, seeds)
    hot = _spread(gumbel_choose(heuristic_evaluate, 0.1, 0.03, 32, temperature=4.0), position, seeds)
    # Monotone rather than a threshold on any one count: the point of the knob
    # is that it orders these, not that it hits a particular spread.
    assert len(cold) == 1 < len(warm) <= len(hot)


def test_the_rng_stream_does_not_depend_on_temperature():
    # The Gumbel is drawn and then scaled, so two temperatures consume the same
    # draws and a chooser swap cannot silently shift a shared generator.
    position = hexo_py.Position.replay([(0, 0), (1, 2)])
    streams = []
    for temperature in (0.0, 0.25, 1.0):
        rng = np.random.default_rng(30)
        gumbel_choose(
            heuristic_evaluate, 0.1, 0.03, 32, temperature=temperature
        )([position], rng)
        streams.append(rng.integers(2**63))
    assert len(set(streams)) == 1


def test_a_temperature_without_a_budget_is_refused():
    # sims == 0 draws no Gumbel, so honouring the request is impossible and
    # ignoring it would hand back a deterministic chooser to a caller who asked
    # for a stochastic one.
    for kwargs in (
        {"sims": 0, "temperature": 0.5},
        {"sims": 32, "temperature": -0.1},
        {"sims": 32, "temperature": float("nan")},
        {"sims": 32, "temperature": float("inf")},
    ):
        try:
            gumbel_choose(heuristic_evaluate, 0.1, 0.03, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"gumbel_choose accepted {kwargs}")
