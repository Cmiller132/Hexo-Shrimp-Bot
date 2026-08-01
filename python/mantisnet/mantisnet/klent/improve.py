"""The closed-form improvement step (``KLENT_FOR_HEXO.md`` §2, paper eq. 3), ragged.

    π′(a|s) ∝ exp[ (Q̃(s,a) + τ·log π_θ(a|s)) / (τ + λ) ]
    v̂(s)    = E_{A~π′(·|s)}[ Q(s, A) ]

Q̃ is Q divided by the largest total return mass in the legal set, so the
temperature is measured against what the critic has committed here rather than
against the ±1 the outcome could reach; v̂ stays on the unscaled Q because the
λ-return bootstraps on it. One divisor per position leaves the order over legal
cells untouched, so this is the same preference at a local temperature.

τ weighs reverse KL to the current policy, and λ weighs entropy of π′. The
result also contains per-position KL(π′‖π_θ) and entropy normalized by
log|A_legal| as specified by ``KLENT_FOR_HEXO.md`` §2.1.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..segments import segment_ids, segment_log_softmax, segment_sum


@dataclass
class ImprovedPolicy:
    """π′ and its per-position diagnostics, flat over the legal cells."""

    probs: Tensor  # (N,) π′, sums to 1 within each segment
    v_hat: Tensor  # (P,) E_{π′}[Q]
    kl: Tensor  # (P,) D_KL(π′ ‖ π_θ)
    norm_entropy: Tensor  # (P,) H(π′) / log|A|; defined as 0 where |A| = 1


@torch.no_grad()
def improved_policy(
    policy_logits: Tensor,
    q_score: Tensor,
    q_value: Tensor,
    offsets: Tensor,
    tau: float,
    lam: float,
) -> ImprovedPolicy:
    """Apply eq. 3 within each position of a ragged batch.

    The operator uses Q in two roles and they are not the same tensor: π′ ranks
    by ``q_score``, which the critic states in units of the position's own
    committed return mass, while v̂ averages ``q_value`` on the outcome's ±1
    scale, because that is what the λ-return bootstraps on.
    """
    if tau < 0 or lam < 0 or tau + lam <= 0:
        raise ValueError(f"need tau, lam >= 0 with tau + lam > 0, got ({tau}, {lam})")
    if q_score.shape != q_value.shape:
        raise ValueError(
            f"the two Q roles disagree on shape: score {tuple(q_score.shape)} "
            f"vs value {tuple(q_value.shape)}"
        )
    p = offsets.shape[0] - 1
    seg = segment_ids(offsets)

    # At least fp32, so bf16 acting logits do not round π′ and its diagnostics;
    # float64 when both inputs are, so a caller comparing two improved policies
    # keeps a difference the operator would otherwise round away.
    dtype = torch.promote_types(
        torch.promote_types(policy_logits.dtype, q_score.dtype), torch.float32
    )
    log_pi = segment_log_softmax(policy_logits.to(dtype), offsets)
    q = q_value.to(dtype)
    log_improved = segment_log_softmax(
        (q_score.to(dtype) + tau * log_pi) / (tau + lam), offsets
    )
    probs = log_improved.exp()

    # Every expectation divides by the segment's own probability mass. π′ sums
    # to one by definition, but an fp32 segment softmax sums to one only to a
    # few ulps, and that error is not small where it matters: a lost endgame
    # position has every legal move at exactly Q = -1, so nothing cancels and
    # E[Q] inherits the whole of it. Measured at 1.1e-4 outside [-1, 1] on a
    # 159-ply episode, which is what the λ-return's range check refuses.
    # Dividing restores |E[Q]| <= max|Q| to a few ulps at any segment width.
    mass = segment_sum(probs, seg, p)
    v_hat = segment_sum(probs * q, seg, p) / mass
    kl = segment_sum(probs * (log_improved - log_pi), seg, p) / mass
    entropy = segment_sum(-probs * log_improved, seg, p) / mass
    counts = (offsets[1:] - offsets[:-1]).to(dtype)
    norm_entropy = torch.where(counts > 1, entropy / counts.clamp(min=2).log(), entropy.new_zeros(p))
    return ImprovedPolicy(probs=probs, v_hat=v_hat, kl=kl, norm_entropy=norm_entropy)
