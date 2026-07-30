"""The closed-form operator against a dense per-segment reference."""

from __future__ import annotations

import math

import pytest
import torch

from mantisnet.klent import improved_policy


def _random_ragged(seed, counts):
    torch.manual_seed(seed)
    n = sum(counts)
    offsets = torch.tensor([0] + list(torch.tensor(counts).cumsum(0)))
    return torch.randn(n), torch.rand(n) * 2 - 1, offsets


def test_matches_dense_reference():
    logits, q, offsets = _random_ragged(0, [1, 3, 17, 240, 2])
    tau, lam = 0.03, 0.1
    imp = improved_policy(logits, q, offsets, tau, lam)
    for i in range(len(offsets) - 1):
        a, b = offsets[i], offsets[i + 1]
        log_pi = logits[a:b].log_softmax(0)
        ref = ((q[a:b] + tau * log_pi) / (tau + lam)).softmax(0)
        assert torch.allclose(imp.probs[a:b], ref, atol=1e-6)
        assert torch.allclose(imp.v_hat[i], (ref * q[a:b]).sum(), atol=1e-6)
        ref_kl = (ref * (ref.log() - log_pi)).sum()
        assert torch.allclose(imp.kl[i], ref_kl, atol=1e-5)
        n = int(b - a)
        ref_ent = -(ref * ref.log()).sum()
        expected = ref_ent / math.log(n) if n > 1 else 0.0
        assert torch.allclose(imp.norm_entropy[i], torch.tensor(float(expected)), atol=1e-5)


def test_probabilities_sum_to_one_per_segment():
    logits, q, offsets = _random_ragged(1, [5, 700, 1])
    imp = improved_policy(logits, q, offsets, 0.03, 0.1)
    for i in range(len(offsets) - 1):
        assert torch.allclose(imp.probs[offsets[i] : offsets[i + 1]].sum(), torch.tensor(1.0))


def test_tau_zero_ignores_the_prior():
    # τ = 0: π′ = softmax(Q/λ), whatever the policy logits say.
    logits, q, offsets = _random_ragged(2, [9])
    imp = improved_policy(logits, q, offsets, tau=0.0, lam=0.1)
    assert torch.allclose(imp.probs, (q / 0.1).softmax(0), atol=1e-6)


def test_constant_q_flattens_to_the_tempered_prior():
    # Constant Q: π′ = softmax(τ·log π_θ / (τ+λ)) — the prior to the power
    # τ/(τ+λ), the flattening KLENT_FOR_HEXO.md §2 is about.
    logits, _q, offsets = _random_ragged(3, [11])
    tau, lam = 0.03, 0.1
    imp = improved_policy(logits, torch.zeros(11), offsets, tau, lam)
    ref = (tau * logits.log_softmax(0) / (tau + lam)).softmax(0)
    assert torch.allclose(imp.probs, ref, atol=1e-6)


def test_single_action_segment_is_a_point_mass():
    imp = improved_policy(torch.tensor([2.5]), torch.tensor([-0.3]), torch.tensor([0, 1]), 0.03, 0.1)
    assert torch.allclose(imp.probs, torch.tensor([1.0]))
    assert torch.allclose(imp.v_hat, torch.tensor([-0.3]))
    assert imp.norm_entropy.item() == 0.0
    assert abs(imp.kl.item()) < 1e-7


def test_refuses_a_degenerate_temperature():
    with pytest.raises(ValueError, match="tau"):
        improved_policy(torch.zeros(2), torch.zeros(2), torch.tensor([0, 2]), 0.0, 0.0)


def test_expectations_stay_inside_the_q_range_when_every_action_is_lost():
    """A lost endgame has every legal move at exactly Q = -1, so the fp32
    softmax's mass error reaches E[Q] with nothing to cancel it. The operator
    divides by the segment's own mass, so |v_hat| cannot leave |Q|'s range at
    any width — the width is what made this a live failure at 3000 cells."""
    for n in (1, 2, 1000, 3000):
        offsets = torch.tensor([0, n])
        q = -torch.ones(n)
        logits = torch.randn(n) * 3.0  # a real policy, so pi' is not uniform
        out = improved_policy(logits, q, offsets, 0.1, 0.01)
        assert abs(float(out.v_hat[0])) <= 1.0, (n, float(out.v_hat[0]))
        assert float(out.kl[0]) >= -1e-6 and float(out.norm_entropy[0]) >= -1e-6

    # And the ordinary case is unchanged: a uniform policy with two-valued Q
    # still averages to the plain mean.
    offsets = torch.tensor([0, 4])
    out = improved_policy(torch.zeros(4), torch.tensor([1.0, 1.0, -1.0, -1.0]), offsets, 1e9, 1e9)
    assert abs(float(out.v_hat[0])) < 1e-5
