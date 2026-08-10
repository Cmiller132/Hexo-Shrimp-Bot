"""Ordered reductions for traceable gathers over collate-built CSR plans.

Inductor lowers the ordinary backward of ``index_select`` and ``Embedding`` to
destination scatters.  Repeated indices then use atomics, which violates ACT's
fp32 bitwise-determinism contract.  The packed batch already owns stable CSR
views for every repeated gather.  These helpers leave the gather visible to the
outer compiler and replace only its backward with a contiguous run reduction.

Every sum promotes reduced precision to fp32 while preserving fp64 oracles
(section 27).  Empty runs produce exact zeros and no destination has multiple
writers.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _accumulate_dtype(tensor: Tensor) -> torch.dtype:
    return torch.promote_types(tensor.dtype, torch.float32)


def ordered_segment_sum(values: Tensor, offsets: Tensor) -> Tensor:
    """Sum contiguous rows according to ``offsets``, in at least fp32.

    ``offsets`` comes from a collate-built plan whose constructor already
    validated monotonicity and bounds; the ``lengths=`` form would rebuild
    them with a scan and revalidate on every call.
    """
    if values.ndim < 1:
        raise ValueError(f"values must have a row dimension, got {values.shape}")
    if offsets.ndim != 1:
        raise ValueError(f"offsets must be one-dimensional, got {offsets.shape}")
    reduced = torch.segment_reduce(
        values.to(_accumulate_dtype(values)),
        "sum",
        offsets=offsets.long(),
        initial=0.0,
    )
    if reduced.shape[0] != offsets.shape[0] - 1:
        raise ValueError("the ordered segment sum produced the wrong row count")
    return reduced


def ordered_two_stage_segment_sum(
    values: Tensor, block_offsets: Tensor, block_lengths: Tensor
) -> Tensor:
    """Reduce compact contiguous row blocks, then their owning segments.

    ``values`` is already in owner-major order.  ``block_lengths`` partitions
    that row order into nonempty fixed-size blocks, while
    ``block_offsets[c]:block_offsets[c + 1]`` names class ``c``'s consecutive
    block partials.  Both stages therefore have a fixed association, including
    empty classes, and need neither a scatter nor an atomic destination update.

    This traceable formulation is the CPU/oracle counterpart of the hand
    kernels used by the high-volume action-table backwards.
    """
    if values.ndim < 1:
        raise ValueError(f"values must have a row dimension, got {values.shape}")
    for name, tensor in (
        ("block_offsets", block_offsets),
        ("block_lengths", block_lengths),
    ):
        if tensor.ndim != 1 or tensor.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"{name} must be a one-dimensional integer tensor")
        if tensor.device != values.device:
            raise ValueError(f"{name} and values must share a device")

    if block_lengths.shape[0] == 0:
        return values.new_zeros(
            (block_offsets.shape[0] - 1, *values.shape[1:]),
            dtype=_accumulate_dtype(values),
        )

    block_row_offsets = torch.cat(
        [block_lengths.new_zeros(1), block_lengths.cumsum(0)]
    )
    partials = torch.segment_reduce(
        values.to(_accumulate_dtype(values)),
        "sum",
        offsets=block_row_offsets.long(),
        initial=0.0,
    )
    return ordered_segment_sum(partials, block_offsets)


class _OrderedIndexSelect(torch.autograd.Function):
    """A traceable gather whose table gradient follows a stable class CSR."""

    @staticmethod
    def forward(
        ctx,
        table: Tensor,
        index: Tensor,
        offsets: Tensor,
        ordered_rows: Tensor,
    ) -> Tensor:
        ctx.save_for_backward(offsets, ordered_rows)
        ctx.table_dtype = table.dtype
        return table.index_select(0, index.long())

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        offsets, ordered_rows = ctx.saved_tensors
        ordered = grad_out.index_select(0, ordered_rows.long())
        grad_table = ordered_segment_sum(ordered, offsets).to(ctx.table_dtype)
        return grad_table, None, None, None


def ordered_index_select(
    table: Tensor,
    index: Tensor,
    offsets: Tensor,
    ordered_rows: Tensor,
) -> Tensor:
    """Select ``table[index]`` with an ordered, scatter-free table backward.

    ``ordered_rows[offsets[c]:offsets[c+1]]`` must name, in original row
    order, every output row whose ``index`` is ``c``.  Plans validate that
    invariant on the CPU before transport.
    """
    if table.ndim < 1:
        raise ValueError(f"table must have a row dimension, got {table.shape}")
    if index.ndim != 1:
        raise ValueError(f"index must be one-dimensional, got {index.shape}")
    if index.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"index must be int32 or int64, got {index.dtype}")
    if offsets.ndim != 1 or offsets.dtype not in (torch.int32, torch.int64):
        raise TypeError("offsets must be a one-dimensional integer tensor")
    if ordered_rows.ndim != 1 or ordered_rows.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise TypeError("ordered_rows must be a one-dimensional integer tensor")
    if table.shape[0] != offsets.shape[0] - 1:
        raise ValueError(
            f"offsets describe {offsets.shape[0] - 1} table rows against "
            f"the table's {table.shape[0]}"
        )
    if index.shape[0] != ordered_rows.shape[0]:
        raise ValueError(
            f"index has {index.shape[0]} rows against the plan's "
            f"{ordered_rows.shape[0]}"
        )
    if not (table.device == index.device == offsets.device == ordered_rows.device):
        raise ValueError("table, index, offsets, and ordered_rows must share a device")
    return _OrderedIndexSelect.apply(table, index, offsets, ordered_rows)


class _OrderedRowBroadcast(torch.autograd.Function):
    """A position-to-row gather whose rows are already position-major."""

    @staticmethod
    def forward(
        ctx, table: Tensor, row_position: Tensor, offsets: Tensor
    ) -> Tensor:
        ctx.save_for_backward(offsets)
        ctx.table_dtype = table.dtype
        return table.index_select(0, row_position.long())

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        (offsets,) = ctx.saved_tensors
        grad_table = ordered_segment_sum(grad_out, offsets).to(ctx.table_dtype)
        return grad_table, None, None


def ordered_row_broadcast(
    table: Tensor, row_position: Tensor, offsets: Tensor
) -> Tensor:
    """Broadcast position rows with a contiguous segment-reduce backward."""
    if table.ndim < 1:
        raise ValueError(f"table must have a row dimension, got {table.shape}")
    if row_position.ndim != 1 or row_position.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise TypeError("row_position must be a one-dimensional integer tensor")
    if offsets.ndim != 1 or offsets.dtype not in (torch.int32, torch.int64):
        raise TypeError("offsets must be a one-dimensional integer tensor")
    if table.shape[0] != offsets.shape[0] - 1:
        raise ValueError(
            f"offsets describe {offsets.shape[0] - 1} positions against "
            f"the table's {table.shape[0]}"
        )
    if table.device != row_position.device or table.device != offsets.device:
        raise ValueError("table, row_position, and offsets must share a device")
    return _OrderedRowBroadcast.apply(table, row_position, offsets)


def ordered_segment_max(values: Tensor, offsets: Tensor) -> Tensor:
    """Maximum of contiguous fp32 row runs, with the dtype minimum identity."""
    if values.ndim != 1:
        raise ValueError(f"values must be one-dimensional, got {values.shape}")
    if not values.is_floating_point():
        raise TypeError(f"values must be floating point, got {values.dtype}")
    if offsets.ndim != 1 or offsets.dtype not in (torch.int32, torch.int64):
        raise TypeError("offsets must be a one-dimensional integer tensor")
    return torch.segment_reduce(
        values,
        "max",
        offsets=offsets.long(),
        initial=torch.finfo(values.dtype).min,
    )


__all__ = [
    "ordered_index_select",
    "ordered_row_broadcast",
    "ordered_segment_max",
    "ordered_segment_sum",
    "ordered_two_stage_segment_sum",
]
