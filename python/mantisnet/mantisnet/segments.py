"""Segmented reductions over ragged per-position rows.

Model outputs are flat over every legal cell of every position, bounded by a
(P + 1,) CSR offset tensor. Everything that normalises or reduces within a
position — the policy loss, KLENT's improvement operator, the critic's
policy-centered advantage — goes through these three helpers, so the ragged
arithmetic exists once.

The reductions walk the CSR offsets rather than scatter-adding over a
row-to-segment index. Both express the same sum, but the walk needs no atomics
and is the one that survives ``torch.compile``: the scatter lowering drops its
bounds mask once it can prove the destination has a single row, so a
single-position batch silently sums padding lanes as well.
"""

from __future__ import annotations

import torch
from torch import Tensor


def segment_ids(offsets: Tensor) -> Tensor:
    """(P + 1,) CSR offsets to a flat (N,) tensor of position ids.

    Only for broadcasting a per-position result back over its rows. The
    builder already emits this index for a batch's legal cells as ``cell_pos``;
    callers that hold one pass it rather than rebuilding it, which also keeps
    this data-dependent output shape out of compiled graphs.
    """
    p = offsets.shape[0] - 1
    counts = offsets[1:] - offsets[:-1]
    return torch.repeat_interleave(torch.arange(p, device=offsets.device), counts)


def segment_sum(values: Tensor, offsets: Tensor) -> Tensor:
    """Per-segment sum of a flat (N,) tensor into (P,)."""
    return torch.segment_reduce(values, "sum", offsets=offsets, axis=0)


def segment_log_softmax(values: Tensor, offsets: Tensor, seg: Tensor) -> Tensor:
    """log-softmax within each segment, numerically shifted per segment.

    ``seg`` is the row-to-segment index of ``offsets`` (:func:`segment_ids`),
    which broadcasts the two per-segment reductions back over the rows.
    """
    seg_max = torch.segment_reduce(values, "max", offsets=offsets, axis=0)
    shifted = values - seg_max.index_select(0, seg)
    lse = segment_sum(shifted.exp(), offsets).log()
    return shifted - lse.index_select(0, seg)
