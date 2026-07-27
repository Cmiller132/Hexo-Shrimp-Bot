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
    line_builder_choose,
    play_episodes,
    play_match,
)
from mantisnet.klent.evaluate import argmax_choose
from mantisnet.klent.seeds import line_scores, seed_prefix
from mantisnet.klent.train import fit, network_evaluate


def heuristic_evaluate(batch):
    """The line builder's scoring through the evaluator seam, so games
    terminate and the buffer rules are observable without a trained model."""
    score = line_scores(batch)
    return score.clone(), score.clone()


def _tiny_model():
    torch.manual_seed(5)
    return MantisNet(
        MantisConfig(
            h=32, blocks=1, heads=2, value_queries=2, value_bins=5,
            policy_hidden=32, value_hidden=32,
        )
    )


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
    rng = np.random.default_rng(13)
    builder = lambda pos, r: line_builder_choose(pos, r, noise=0.0)  # noqa: E731
    result = play_match(builder, builder, games=4, ply_cap=300, rng=rng)
    assert result["games"] == 4
    assert 0.0 <= result["score_a"] <= 4.0
    assert result["capped"] * 0.5 <= result["score_a"] <= 4 - result["capped"] * 0.5

    model = _tiny_model().eval()
    quick = play_match(argmax_choose(model), builder, games=2, ply_cap=30, rng=rng)
    assert quick["games"] == 2
    assert 0.0 <= quick["score_a"] <= 2.0
