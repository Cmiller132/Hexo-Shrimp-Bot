"""Self-play collection, the buffer rules, fitting, and the whole iteration.

The collection loop is exercised through the evaluator seam with a scripted
line-extending evaluator — the same seam the network uses — so games
reliably terminate and every buffer rule is observable without a trained
model.
"""

from __future__ import annotations

import hexo_py
import numpy as np
import torch

from mantisnet import MantisConfig, MantisNet
from mantisnet.klent import (
    KlentConfig,
    episode_samples,
    iterate,
    play_episodes,
    play_match,
)
from mantisnet.klent.evaluate import argmax_choose
from mantisnet.klent.seeds import line_evaluate, line_scores, seed_prefix
from mantisnet.klent.train import _pack, fit, iterate, network_evaluate

# The line builder's scoring through the evaluator seam, so games terminate
# and the buffer rules are observable without a trained model — the same
# evaluator the warm start acts through.
heuristic_evaluate = line_evaluate


def _tiny_model():
    torch.manual_seed(5)
    return MantisNet(
        MantisConfig(
            h=32, blocks=1, heads=2, value_queries=2, value_bins=5,
            policy_hidden=32, value_hidden=32,
        )
    )


def test_warm_iteration_trains_through_the_line_evaluator():
    """The warm start: collection acts through the line builder, games
    finish, and both heads fit on Monte-Carlo outcomes."""
    model = _tiny_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    cfg = KlentConfig(games_per_iteration=8, ply_cap=200, batch_size=256)
    metrics = iterate(model, opt, cfg, np.random.default_rng(9), warm=True)
    assert metrics["warm"] == 1
    assert metrics["buffer_samples"] > 0, "line play must terminate games"
    assert metrics["fit_steps"] >= 1
    assert any(
        p.grad is not None and float(p.grad.abs().sum()) > 0
        for p in model.mlp_q.parameters()
    ), "the zero-initialized Q head must receive warm gradients"


def test_fit_packing_respects_budgets_and_loses_nothing():
    from types import SimpleNamespace

    rng = np.random.default_rng(0)
    samples = [
        SimpleNamespace(t=int(t), improved=np.zeros(int(c), dtype=np.float32))
        for t, c in zip(rng.integers(1, 500, 300), rng.integers(1, 8000, 300))
    ]
    cfg = KlentConfig(batch_size=32, pair_budget=2_000_000, cell_budget=30_000)
    chunks = _pack(samples, rng.permutation(len(samples)), cfg)

    assert sorted(i for c in chunks for i in c) == list(range(len(samples)))
    for chunk in chunks:
        assert len(chunk) <= cfg.batch_size
        t_pad = max(samples[i].t for i in chunk) + 1
        assert t_pad == samples[chunk[0]].t + 1  # descending: first is widest
        if len(chunk) > 1:  # a lone oversized sample is kept, never dropped
            assert len(chunk) * t_pad * t_pad <= cfg.pair_budget
            assert sum(len(samples[i].improved) for i in chunk) <= cfg.cell_budget


def test_accumulated_gradients_match_one_big_batch():
    """Chunking is memory, not optimization: an epoch forced into many tiny
    chunks accumulates to the same gradients as one whole-buffer batch."""
    rng = np.random.default_rng(3)
    episodes, _ = play_episodes(
        heuristic_evaluate, [[] for _ in range(4)], 100, 0.1, 0.03, rng
    )
    samples = [s for e in episodes for s in episode_samples(e, 0.939)][:24]
    assert len(samples) >= 12

    grads = []
    for cell_budget in (10**9, 1):  # one big chunk vs all-singleton chunks
        torch.manual_seed(0)
        model = _tiny_model()
        cfg = KlentConfig(
            batch_size=len(samples), pair_budget=10**9, cell_budget=cell_budget
        )
        opt = torch.optim.SGD(model.parameters(), lr=0.0)  # keep grads readable
        out = fit(model, samples, opt, cfg, np.random.default_rng(0))
        assert out["fit_steps"] == 1
        grads.append({n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None})

    assert grads[0].keys() == grads[1].keys()
    for name in grads[0]:
        assert torch.allclose(grads[0][name], grads[1][name], atol=1e-4), name


def test_collection_is_chunking_invariant():
    """Budgets change how the cohort is split across evaluate calls, and
    nothing else: with a per-position evaluator the episodes are identical."""
    outcomes = []
    for budgets in ({}, {"pair_budget": 2_000, "cell_budget": 400}):
        rng = np.random.default_rng(11)
        episodes, _ = play_episodes(
            heuristic_evaluate, [[] for _ in range(5)], 60, 0.1, 0.03, rng, **budgets
        )
        outcomes.append([(e.moves, e.winner) for e in episodes])
    assert outcomes[0] == outcomes[1]


def test_heuristic_selfplay_terminates_and_buffers_correctly():
    rng = np.random.default_rng(7)
    episodes, metrics = play_episodes(
        heuristic_evaluate, [[] for _ in range(8)], ply_cap=200, tau=0.03, lam=0.1, rng=rng
    )
    won = [e for e in episodes if e.winner is not None]
    assert len(won) >= 4, "the line-extending evaluator should usually finish games"
    assert metrics["acting_kl"] >= 0 and 0 <= metrics["acting_norm_entropy"] <= 1

    for ep in episodes:
        samples = episode_samples(ep, lam_ret=0.883)
        if ep.winner is None:
            assert samples == []  # K4: capped episodes contribute nothing
            continue
        # Unseeded: every ply is a sample, the win included, and G_T = +1.
        assert len(samples) == len(ep.moves)
        assert samples[-1].g == 1.0
        assert all(np.isfinite(s.g) for s in samples)
        # At the Monte Carlo endpoint the returns are exactly ±1, whatever
        # the evaluator's (here deliberately unbounded) v̂ said.
        assert all(s.g in (1.0, -1.0) for s in episode_samples(ep, 1.0))


def test_samples_replay_to_their_positions():
    rng = np.random.default_rng(8)
    episodes, _ = play_episodes(
        heuristic_evaluate, [[]], ply_cap=200, tau=0.03, lam=0.1, rng=rng
    )
    samples = episode_samples(episodes[0], lam_ret=0.9)
    assert samples, "seed 8 should produce a finished game"
    for s in samples[:: max(len(samples) // 6, 1)]:
        pos = hexo_py.Position.replay(list(s.moves[: s.t]))
        legal = pos.legal_moves()
        assert len(legal) == len(s.improved)
        assert legal[s.rank] == tuple(s.moves[s.t])
        assert np.isclose(s.improved.sum(), 1.0, atol=1e-5)


def _scripted_opponent(pos, moves):
    """A whole-turn opponent for the grounding tests: the first legal move,
    up to the turn's remaining placements — deterministic and rules-honest."""
    p = pos.copy()
    turn = []
    for _ in range(p.moves_remaining):
        move = p.nth_legal(0)
        turn.append(move)
        p.advance(*move)
        if p.is_terminal:
            break
    return turn


def test_grounded_games_record_model_plies_only():
    """Grounded games: the opponent's plies enter the move list but never
    the records, samples replay to model-to-move positions, and both seats
    appear across games."""
    rng = np.random.default_rng(14)
    seats = [0, 1, None]
    episodes, _ = play_episodes(
        heuristic_evaluate, [[], [], []], 200, 0.1, 0.03, rng,
        opponent=_scripted_opponent, opponent_seats=seats,
    )
    for ep, seat in zip(episodes, seats):
        assert ep.opponent_seat == seat
        if seat is None:
            continue
        assert ep.winner is not None, "a line-scorer vs the first-legal bot finishes"
        model_seat = 1 - seat
        assert all(m == model_seat for m in ep.movers)
        assert len(ep.ts) == len(ep.ranks) < len(ep.moves)
        for j in (0, len(ep.ts) - 1):
            pos = hexo_py.Position.replay(list(ep.moves[: ep.ts[j]]))
            assert pos.current_player == model_seat
            assert pos.nth_legal(ep.ranks[j]) == tuple(ep.moves[ep.ts[j]])


def test_grounded_returns_are_outcomes_and_caps_are_draws():
    """Grounded returns are Monte-Carlo whatever lam_ret says — the bootstrap
    chain breaks at unrecorded opponent plies — and a capped grounded episode
    yields g = 0 samples where a capped self-play one yields none."""
    rng = np.random.default_rng(15)
    episodes, _ = play_episodes(
        heuristic_evaluate, [[]], 200, 0.1, 0.03, rng,
        opponent=_scripted_opponent, opponent_seats=[0],
    )
    ep = episodes[0]
    assert ep.winner is not None
    z = 1.0 if ep.winner == 1 else -1.0
    for lam_ret in (1.0, 0.5):
        samples = episode_samples(ep, lam_ret)
        assert len(samples) == len(ep.ranks)
        assert all(s.g == z for s in samples)

    capped_ground, _ = play_episodes(
        heuristic_evaluate, [[]], 8, 0.1, 0.03, np.random.default_rng(15),
        opponent=_scripted_opponent, opponent_seats=[0],
    )
    capped_self, _ = play_episodes(
        heuristic_evaluate, [[]], 8, 0.1, 0.03, np.random.default_rng(15),
    )
    assert capped_ground[0].winner is None and capped_self[0].winner is None
    draws = episode_samples(capped_ground[0], 0.939)
    assert draws and all(s.g == 0.0 for s in draws)
    assert episode_samples(capped_self[0], 0.939) == []


def test_iterate_grounds_a_fraction_and_reports_it():
    model = _tiny_model()
    cfg = KlentConfig(
        games_per_iteration=6, ground_fraction=0.5, ply_cap=200,
        batch_size=64, seed_fraction=1.0, seed_cut=(1, 3),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    metrics = iterate(
        model, optimizer, cfg, np.random.default_rng(16), opponent=_scripted_opponent
    )
    assert metrics["f_grounded"] == metrics["f_grounded"]  # grounded games exist
    assert 0.0 <= metrics["grounded_score"] <= 1.0
    assert metrics["buffer_samples"] > 0

    import pytest

    with pytest.raises(ValueError, match="opponent"):
        iterate(model, optimizer, cfg, np.random.default_rng(16))


def test_seeded_prefix_plies_stay_out_of_the_buffer():
    rng = np.random.default_rng(9)
    prefix = seed_prefix(rng, (1, 4))
    assert prefix, "a line-builder game is longer than the cut"
    episodes, _ = play_episodes(
        heuristic_evaluate, [prefix], ply_cap=200, tau=0.03, lam=0.1, rng=rng
    )
    ep = episodes[0]
    assert ep.seed_len == len(prefix)
    assert [tuple(m) for m in ep.moves[: ep.seed_len]] == [tuple(m) for m in prefix]
    for s in episode_samples(ep, lam_ret=0.883):
        assert s.t >= ep.seed_len


def test_fit_trains_policy_and_q_and_never_the_value_head():
    rng = np.random.default_rng(10)
    episodes, _ = play_episodes(
        heuristic_evaluate, [[] for _ in range(4)], ply_cap=200, tau=0.03, lam=0.1, rng=rng
    )
    samples = [s for e in episodes for s in episode_samples(e, 0.883)]
    assert samples

    model = _tiny_model()
    cfg = KlentConfig(batch_size=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    metrics = fit(model, samples, optimizer, cfg, rng)
    assert np.isfinite(metrics["policy_loss"]) and np.isfinite(metrics["q_loss"])
    assert metrics["fit_steps"] == (len(samples) + 63) // 64

    value_only = {"value_queries", "ln_value", "mlp_v"}
    for name, p in model.named_parameters():
        head = name.split(".")[0]
        if head in value_only:
            assert p.grad is None, f"value head parameter {name} was trained"
        else:
            assert p.grad is not None, f"{name} received no gradient"


def test_iterate_runs_end_to_end():
    model = _tiny_model()
    cfg = KlentConfig(
        games_per_iteration=4, ply_cap=60, batch_size=64, seed_fraction=1.0, seed_cut=(1, 3)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    metrics = iterate(model, optimizer, cfg, np.random.default_rng(12))
    for key in ("f_seeded", "f_unseeded", "acting_kl", "acting_norm_entropy", "buffer_samples"):
        assert key in metrics
    assert metrics["buffer_samples"] >= 0
    if metrics["buffer_samples"]:
        assert "policy_loss" in metrics


def test_network_evaluate_matches_forward():
    model = _tiny_model().eval()
    from mantisnet import collate, from_position

    pos = hexo_py.Position.replay([(0, 0), (1, 1), (2, 0)])
    batch = collate([from_position(pos)])
    logits, q = network_evaluate(model, KlentConfig())(batch)
    with torch.no_grad():
        out = model(batch)
    assert torch.allclose(logits, out.policy_logits, atol=1e-6)
    assert torch.allclose(q, out.q_values, atol=1e-6)


def test_play_match_is_seat_balanced_and_scores_caps_as_half():
    from mantisnet.klent.seeds import line_builder_choose_batch

    rng = np.random.default_rng(13)
    builder = lambda ps, r: line_builder_choose_batch(ps, r, noise=0.0)  # noqa: E731
    result = play_match(builder, builder, games=4, ply_cap=300, rng=rng)
    assert result["games"] == 4
    assert 0.0 <= result["score_a"] <= 4.0
    assert result["capped"] * 0.5 <= result["score_a"] <= 4 - result["capped"] * 0.5

    model = _tiny_model().eval()
    quick = play_match(argmax_choose(model), builder, games=2, ply_cap=30, rng=rng)
    assert quick["games"] == 2
    assert 0.0 <= quick["score_a"] <= 2.0
