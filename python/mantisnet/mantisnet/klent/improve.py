"""The closed-form improvement step (design doc §1, paper eq. 3), ragged.

    π′(a|s) ∝ exp[ (s·Q(s,a) + τ·log π_θ(a|s)) / (τ + λ) ]
    v̂(s)    = E_{A~π′(·|s)}[ Q(s, A) ]

τ weighs the reverse KL to the current policy, λ the entropy of π′. Both
diagnostics of design doc §8 — per-position KL(π′‖π_θ) and entropy normalised
by log|A_legal| — come out of the same pass, because they are the §13 metrics
that decide whether the run is working at all.

``s`` is ``q_scale``, the critic's gain inside the softmax; 1 is eq. 3
verbatim. It exists because the operator's sharpening must outweigh the
flattening prior exponent τ/(τ+λ) < 1 or the fit trains π_θ toward a flatter
copy of itself each iteration — and whether it outweighs it depends on the
*spread* of Q across a position's moves measured against τ+λ. The scalar
tanh head's overconfident magnitudes cleared that bar implicitly; the
factored critic's calibrated magnitudes in contested positions do not
(measured: entropy runaway and eval collapse at s = 1, factored-939
iterations 50-79). The gain applies only inside the softmax: v̂ averages the
*unscaled* Q under π′, so returns and the magnitude target stay in (−1, 1).
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
    q_scale: float,
) -> ImprovedPolicy:
    """Apply eq. 3 within each position of a ragged batch.

    ``q_scale`` has no default for the same reason a caller must state τ and
    λ: π′ is a function of it, and a reader that assumed 1 would silently
    misreport a run that acted at another gain.
    """
    if tau < 0 or lam < 0 or tau + lam <= 0:
        raise ValueError(f"need tau, lam >= 0 with tau + lam > 0, got ({tau}, {lam})")
    if q_scale <= 0:
        raise ValueError(f"need q_scale > 0, got {q_scale}")
    p = offsets.shape[0] - 1
    seg = segment_ids(offsets)

    log_pi = segment_log_softmax(policy_logits.float(), offsets)
    q = q_values.float()
    log_improved = segment_log_softmax((q_scale * q + tau * log_pi) / (tau + lam), offsets)
    probs = log_improved.exp()

    v_hat = segment_sum(probs * q, seg, p)
    kl = segment_sum(probs * (log_improved - log_pi), seg, p)
    entropy = segment_sum(-probs * log_improved, seg, p)
    counts = (offsets[1:] - offsets[:-1]).float()
    norm_entropy = torch.where(counts > 1, entropy / counts.clamp(min=2).log(), entropy.new_zeros(p))
    return ImprovedPolicy(probs=probs, v_hat=v_hat, kl=kl, norm_entropy=norm_entropy)
