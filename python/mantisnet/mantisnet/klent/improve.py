"""The closed-form improvement step (``KLENT_FOR_HEXO.md`` §2, paper eq. 3), ragged.

    π′(a|s) ∝ exp[ (Q(s,a) + τ·log π_θ(a|s)) / (τ + λ) ]
    v̂(s)    = E_{A~π′(·|s)}[ Q(s, A) ]

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
    q_values: Tensor,
    offsets: Tensor,
    tau: float,
    lam: float,
) -> ImprovedPolicy:
    """Apply eq. 3 within each position of a ragged batch."""
    if tau < 0 or lam < 0 or tau + lam <= 0:
        raise ValueError(f"need tau, lam >= 0 with tau + lam > 0, got ({tau}, {lam})")
    p = offsets.shape[0] - 1
    seg = segment_ids(offsets)

    # At least fp32, so bf16 acting logits do not round π′ and its diagnostics;
    # float64 when both inputs are, so a caller comparing two improved policies
    # keeps a difference the operator would otherwise round away.
    dtype = torch.promote_types(
        torch.promote_types(policy_logits.dtype, q_values.dtype), torch.float32
    )
    log_pi = segment_log_softmax(policy_logits.to(dtype), offsets)
    q = q_values.to(dtype)
    log_improved = segment_log_softmax((q + tau * log_pi) / (tau + lam), offsets)
    probs = log_improved.exp()

    v_hat = segment_sum(probs * q, seg, p)
    kl = segment_sum(probs * (log_improved - log_pi), seg, p)
    entropy = segment_sum(-probs * log_improved, seg, p)
    counts = (offsets[1:] - offsets[:-1]).to(dtype)
    norm_entropy = torch.where(counts > 1, entropy / counts.clamp(min=2).log(), entropy.new_zeros(p))
    return ImprovedPolicy(probs=probs, v_hat=v_hat, kl=kl, norm_entropy=norm_entropy)
