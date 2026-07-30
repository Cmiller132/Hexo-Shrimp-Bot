"""Self-play collection, buffer, fitting, and iteration contracts.

The collection tests use the evaluator seam with the scripted line-extending
evaluator in ``tests/heuristic.py``.
"""

from __future__ import annotations

import hexo_py
import numpy as np
import torch

from mantisnet import MantisConfig, MantisNet
from mantisnet.klent import (
    Collector,
    KlentConfig,
    collect_episodes,
    episode_samples,
    play_match,
)
from mantisnet.klent.evaluate import argmax_choose
from mantisnet.klent.opponents import shared_openings
from mantisnet.klent.train import _pack, fit, network_evaluate

from .heuristic import heuristic_choose, heuristic_evaluate


def _tiny_model(seed: int = 5):
    torch.manual_seed(seed)
    return MantisNet(
        MantisConfig(
            h=32, blocks=1, heads=2, value_queries=2, value_bins=5,
            policy_hidden=32, value_hidden=32,
        )
    )


def _collect(evaluate, games, ply_cap, tau, lam, rng, **budgets):
    """One fresh-cohort collect call: as many slots as the quota."""
    return Collector(games, ply_cap, tau, lam, rng, **budgets).collect(evaluate, games)


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
    episodes, _ = _collect(heuristic_evaluate, 4, 100, 0.1, 0.03, rng)
    samples = [s for e in episodes for s in episode_samples(e, 0.939, 1.0)][:24]
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
        episodes, _ = _collect(heuristic_evaluate, 5, 60, 0.1, 0.03, rng, **budgets)
        outcomes.append([(e.moves, e.winner) for e in episodes])
    assert outcomes[0] == outcomes[1]


def test_collector_carries_games_across_calls():
    """Auto-reset and carry: a quota smaller than the cohort leaves games in
    flight, the next call finishes them, and every returned record is a
    whole game from the empty board."""
    rng = np.random.default_rng(10)
    collector = Collector(4, 200, 0.1, 0.03, rng)
    first, _ = collector.collect(heuristic_evaluate, 1)
    assert len(first) >= 1
    assert sum(len(e.moves) for e in collector.episodes) > 0, "no game in flight"
    second, _ = collector.collect(heuristic_evaluate, 3)
    assert len(second) >= 3
    for ep in first + second:
        pos = hexo_py.Position.replay([tuple(m) for m in ep.moves])
        if ep.winner is not None:
            assert pos.is_terminal and pos.winner == ep.winner


def test_collector_progress_observer():
    """The per-step observer sees every lockstep step: monotone finished
    counts toward the quota and one live ply count per slot — and, being an
    observer, changes nothing about what collection returns."""
    episodes, _ = Collector(6, 200, 0.1, 0.03, np.random.default_rng(21)).collect(
        heuristic_evaluate, 6
    )
    calls = []
    observed, _ = Collector(6, 200, 0.1, 0.03, np.random.default_rng(21)).collect(
        heuristic_evaluate, 6,
        progress=lambda done, quota, plies: calls.append((done, quota, list(plies))),
    )
    assert calls and calls[-1][0] >= 6
    assert all(quota == 6 and len(plies) == 6 for _, quota, plies in calls)
    assert all(a[0] <= b[0] for a, b in zip(calls, calls[1:]))
    assert [e.moves for e in observed] == [e.moves for e in episodes]


def test_heuristic_selfplay_terminates_and_buffers_correctly():
    rng = np.random.default_rng(7)
    episodes, metrics = _collect(
        heuristic_evaluate, 8, ply_cap=200, tau=0.03, lam=0.1, rng=rng
    )
    won = [e for e in episodes if e.winner is not None]
    assert len(won) >= 4, "the line-extending evaluator should usually finish games"
    assert metrics["acting_kl"] >= 0 and 0 <= metrics["acting_norm_entropy"] <= 1

    for ep in episodes:
        samples = episode_samples(ep, lam_ret=0.883, gamma=1.0)
        if ep.winner is None:
            assert samples == []  # K4: capped episodes contribute nothing
            continue
        # Every ply is a sample, the win included, and G_T = +1.
        assert len(samples) == len(ep.moves)
        assert samples[-1].g == 1.0
        assert all(np.isfinite(s.g) for s in samples)
        # At the Monte Carlo endpoint the returns are ±1 independent of v̂.
        assert all(s.g in (1.0, -1.0) for s in episode_samples(ep, 1.0, 1.0))


def test_samples_replay_to_their_positions():
    rng = np.random.default_rng(8)
    episodes, _ = _collect(heuristic_evaluate, 1, ply_cap=200, tau=0.03, lam=0.1, rng=rng)
    samples = episode_samples(episodes[0], lam_ret=0.9, gamma=1.0)
    assert samples, "seed 8 should produce a finished game"
    for s in samples[:: max(len(samples) // 6, 1)]:
        pos = hexo_py.Position.replay(list(s.moves[: s.t]))
        legal = pos.legal_moves()
        assert len(legal) == len(s.improved)
        assert legal[s.rank] == tuple(s.moves[s.t])
        assert np.isclose(s.improved.sum(), 1.0, atol=1e-5)


def test_fit_trains_policy_and_q_and_never_the_value_head():
    rng = np.random.default_rng(10)
    episodes, _ = _collect(heuristic_evaluate, 4, ply_cap=200, tau=0.03, lam=0.1, rng=rng)
    samples = [s for e in episodes for s in episode_samples(e, 0.883, 1.0)]
    assert samples

    model = _tiny_model()
    cfg = KlentConfig(batch_size=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    metrics = fit(model, samples, optimizer, cfg, rng)
    assert np.isfinite(metrics["policy_loss"]) and np.isfinite(metrics["q_loss"])
    # Groups close at >= batch_size samples and may overshoot by one chunk,
    # so the step count is bounded by, not equal to, ceil(n / batch_size).
    assert 1 <= metrics["fit_steps"] <= (len(samples) + 63) // 64

    value_only = {"value_queries", "ln_value", "mlp_v"}
    for name, p in model.named_parameters():
        head = name.split(".")[0]
        if head in value_only:
            assert p.grad is None, f"value head parameter {name} was trained"
        else:
            assert p.grad is not None, f"{name} received no gradient"


def test_collect_and_fit_end_to_end():
    """An untrained network collects complete metrics and fits any resulting samples."""
    model = _tiny_model()
    cfg = KlentConfig(games_per_iteration=4, envs=4, ply_cap=60, batch_size=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    collector = Collector(
        cfg.envs, cfg.ply_cap, cfg.tau, cfg.lam, np.random.default_rng(12)
    )
    episodes, metrics = collect_episodes(model, collector, cfg)
    assert len(episodes) >= 4
    for key in ("f", "acting_kl", "acting_norm_entropy", "v_hat_mae"):
        assert key in metrics
    samples = [s for e in episodes for s in episode_samples(e, cfg.lam_ret, cfg.gamma)]
    if samples:
        out = fit(model, samples, optimizer, cfg, np.random.default_rng(12))
        assert np.isfinite(out["policy_loss"])


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
    rng = np.random.default_rng(13)
    result, rows = play_match(
        heuristic_choose, heuristic_choose, shared_openings(rng, 2), 300, rng
    )
    assert result["games"] == 4
    assert 0.0 <= result["score_a"] <= 4.0
    assert result["capped"] * 0.5 <= result["score_a"] <= 4 - result["capped"] * 0.5
    assert result["score_a_as_p0"] + result["score_a_as_p1"] == result["score_a"]
    assert [row["seat"] for row in rows] == [0, 1, 0, 1]

    model = _tiny_model().eval()
    quick, _rows = play_match(
        argmax_choose(model), heuristic_choose, shared_openings(rng, 1), 30, rng
    )
    assert quick["games"] == 2
    assert 0.0 <= quick["score_a"] <= 2.0


def _distinguishable_model(seed: int):
    """A tiny model whose policy is not constant: the output layers initialize to
    zero, so two fresh models both argmax to legal rank 0 and play one line."""
    model = _tiny_model(seed).eval()
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 0.1)
    return model


def test_play_match_plays_a_distinct_game_per_schedule_entry():
    """The schedule's openings are the whole source of a match's diversity: two
    deterministic choosers from one start play one game however often they are
    asked to. Both choosers here ignore the generator, so every distinction below
    is the schedule's."""
    schedule = shared_openings(np.random.default_rng(4), 4)
    _summary, rows = play_match(
        argmax_choose(_distinguishable_model(5)),
        argmax_choose(_distinguishable_model(9)),
        schedule,
        60,
        np.random.default_rng(0),
    )
    assert len({tuple(row["opening"]) for row in rows}) == 4
    assert len({tuple(row["moves"]) for row in rows}) == 8
