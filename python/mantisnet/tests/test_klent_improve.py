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
    imp = improved_policy(logits, q, offsets, tau, lam, 1.0)
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
    imp = improved_policy(logits, q, offsets, 0.03, 0.1, 1.0)
    for i in range(len(offsets) - 1):
        assert torch.allclose(imp.probs[offsets[i] : offsets[i + 1]].sum(), torch.tensor(1.0))


def test_tau_zero_ignores_the_prior():
    # τ = 0: π′ = softmax(Q/λ), whatever the policy logits say.
    logits, q, offsets = _random_ragged(2, [9])
    imp = improved_policy(logits, q, offsets, tau=0.0, lam=0.1, q_scale=1.0)
    assert torch.allclose(imp.probs, (q / 0.1).softmax(0), atol=1e-6)


def test_constant_q_flattens_to_the_tempered_prior():
    # Constant Q: π′ = softmax(τ·log π_θ / (τ+λ)) — the prior to the power
    # τ/(τ+λ), the flattening §8 is about.
    logits, _q, offsets = _random_ragged(3, [11])
    tau, lam = 0.03, 0.1
    imp = improved_policy(logits, torch.zeros(11), offsets, tau, lam, 1.0)
    ref = (tau * logits.log_softmax(0) / (tau + lam)).softmax(0)
    assert torch.allclose(imp.probs, ref, atol=1e-6)


def test_single_action_segment_is_a_point_mass():
    imp = improved_policy(torch.tensor([2.5]), torch.tensor([-0.3]), torch.tensor([0, 1]), 0.03, 0.1, 1.0)
    assert torch.allclose(imp.probs, torch.tensor([1.0]))
    assert torch.allclose(imp.v_hat, torch.tensor([-0.3]))
    assert imp.norm_entropy.item() == 0.0
    assert abs(imp.kl.item()) < 1e-7


def test_refuses_a_degenerate_temperature():
    with pytest.raises(ValueError, match="tau"):
        improved_policy(torch.zeros(2), torch.zeros(2), torch.tensor([0, 2]), 0.0, 0.0, 1.0)


def test_q_scale_matches_dense_reference_and_leaves_v_hat_unscaled():
    # The gain applies inside the softmax only: π′ follows s·Q, while v̂
    # averages the *unscaled* Q under that π′ — so returns built from v̂
    # stay in (−1, 1) whatever the gain.
    logits, q, offsets = _random_ragged(4, [2, 7, 31])
    tau, lam, s = 0.1, 0.01, 2.0
    imp = improved_policy(logits, q, offsets, tau, lam, s)
    for i in range(len(offsets) - 1):
        a, b = offsets[i], offsets[i + 1]
        log_pi = logits[a:b].log_softmax(0)
        ref = ((s * q[a:b] + tau * log_pi) / (tau + lam)).softmax(0)
        assert torch.allclose(imp.probs[a:b], ref, atol=1e-6)
        assert torch.allclose(imp.v_hat[i], (ref * q[a:b]).sum(), atol=1e-6)


def test_q_scale_sharpens_toward_the_higher_q_action():
    # Uniform prior, small Q gap: raising the gain must move π′ mass toward
    # the better action — the whole point of the knob.
    logits = torch.zeros(2)
    q = torch.tensor([0.04, 0.0])
    offsets = torch.tensor([0, 2])
    top1 = [
        improved_policy(logits, q, offsets, 0.1, 0.01, s).probs[0]
        for s in (1.0, 2.0, 4.0)
    ]
    assert top1[0] < top1[1] < top1[2]


def test_refuses_a_non_positive_q_scale():
    for s in (0.0, -1.0):
        with pytest.raises(ValueError, match="q_scale"):
            improved_policy(torch.zeros(2), torch.zeros(2), torch.tensor([0, 2]), 0.1, 0.03, s)
