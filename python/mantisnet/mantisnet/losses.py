"""Training targets and losses: what the outputs *mean* (MODEL_SPEC §6, §7, §10).

The model emits raw policy logits and a binned value distribution; this module
pins the targets they are trained against. Nothing here is used at inference.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .segments import segment_ids, segment_log_softmax, segment_sum


def value_target(z: Tensor, bins: int) -> Tensor:
    """Project outcomes ``z`` in [-1, 1] onto the bins, exactly in expectation.

    Two-hot: mass split between the neighbouring bin centers so that the
    decoded expectation equals ``z`` (§7). Shape (P,) to (P, bins).
    """
    if torch.any((z < -1) | (z > 1)):
        raise ValueError("value targets must lie in [-1, 1]")
    z = z.float()
    delta = 2.0 / (bins - 1)
    lo = ((z + 1.0) / delta).floor().long().clamp(max=bins - 2)
    hi_weight = (z + 1.0) / delta - lo.float()
    target = z.new_zeros(*z.shape, bins)
    target.scatter_(-1, lo.unsqueeze(-1), (1.0 - hi_weight).unsqueeze(-1))
    target.scatter_add_(-1, (lo + 1).unsqueeze(-1), hi_weight.unsqueeze(-1))
    return target


def value_loss(v_logits: Tensor, z: Tensor) -> Tensor:
    """Cross-entropy of the bin logits against the two-hot projection of ``z``."""
    target = value_target(z, v_logits.shape[-1])
    return -(target * F.log_softmax(v_logits.float(), dim=-1)).sum(-1).mean()


def policy_loss(logits: Tensor, offsets: Tensor, target: Tensor) -> Tensor:
    """Cross-entropy over ragged per-position policies, mean over positions.

    ``logits`` and ``target`` are flat over every legal cell of every position
    (engine order, the model's output layout); ``offsets`` is the (P + 1,)
    CSR boundary from the batch. Each position's target must sum to one.
    """
    p = offsets.shape[0] - 1
    seg = segment_ids(offsets)

    target = target.float()
    # Accumulating in f64 keeps fp32 rounding below the normalization tolerance.
    sums = segment_sum(target.double(), seg, p)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-4):
        dev = (sums - 1.0).abs()
        worst = int(dev.argmax())
        counts = offsets[1:] - offsets[:-1]
        raise ValueError(
            "each position's policy target must sum to 1: "
            f"{int((dev > 1e-4).sum())}/{p} positions off, worst |sum-1|="
            f"{float(dev[worst]):.3e} at width {int(counts[worst])}, "
            f"{int(target.isnan().sum())} NaN / {int(target.isinf().sum())} Inf entries"
        )

    log_probs = segment_log_softmax(logits.float(), offsets)
    return segment_sum(-(target * log_probs), seg, p).mean()


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """§10 optimizer grouping: decay matrices; spare vectors, embedding
    tables, and the attention-bias tables."""
    no_decay_ids = {id(m.weight) for m in model.modules() if isinstance(m, nn.Embedding)}
    for name, p in model.named_parameters():
        if p.ndim <= 1 or name.endswith("dist_bias"):
            no_decay_ids.add(id(p))
    params = list(model.parameters())
    return [
        {"params": [p for p in params if id(p) not in no_decay_ids], "weight_decay": weight_decay},
        {"params": [p for p in params if id(p) in no_decay_ids], "weight_decay": 0.0},
    ]
