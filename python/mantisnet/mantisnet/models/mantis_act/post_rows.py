"""Section 19.2's eighteen post-placement rows, fused into two Triton ops.

``sentinel_gather`` gathers a window row per slot (sentinel index ``-1`` reads
a shared base in-kernel).  ``row_gate`` fuses LN, projections, sigmoid, and
product into registers.  The backward recomputes the forward from saved
inputs; parameter gradients use a two-stage deterministic reduction.
Accumulators are fp32 (§27).  CPU: torch parity reference of §36.
"""

from __future__ import annotations

import warnings

import torch
from torch import Tensor

from .equivariant import AXIS_CHANNELS

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only without triton
    triton = None
    tl = None


# One program covers this many rows, wide enough for `tl.dot` to reach the
# tensor cores — its contraction dimension in the parameter gradients is the
# row tile itself.
_BLOCK_M = 32

# The shared sentinel base is read by most post rows.  Giving that reduction
# one owner program made its backward deterministic, but also serialized tens
# of thousands of vector additions on one CTA.  The first stage below gives
# each fixed, contiguous slice of the already-sorted sentinel plan one owner;
# the second stage combines those partials in ascending slice order.  The
# association is therefore fixed without sacrificing grid parallelism.
_GATHER_BASE_BLOCK_ROWS = 256

# The table gradients are summed over a fixed number of programs rather than
# over the row count, so the partial buffer does not grow with the batch and
# the second stage is a reduction over a dimension the grid fixes.
_ROW_PROGRAMS = 256
_TABLE_PROGRAMS = 256
_ROW_WARPS = 4
_TABLE_WARPS = 4

# How many output columns one table-gradient program owns, which decides the
# live accumulator set: a narrower slice fits the register file but re-derives
# the row's LayerNorm more times; 32 is the measured balance.
_BLOCK_O = 32

_FAILED_SHAPES: dict[tuple[object, ...], str] = {}
_FAILED_BACKWARD_SHAPES: dict[tuple[object, ...], str] = {}

_SUPPORTED_DTYPES = (torch.float32, torch.bfloat16, torch.float16)


def _accumulate_dtype(tensor: Tensor) -> torch.dtype:
    return torch.promote_types(tensor.dtype, torch.float32)


# --------------------------------------------------------------------------
# Kernels


if triton is not None:

    @triton.jit
    def _gather_kernel(
        source_ptr,
        base_ptr,
        index_ptr,
        out_ptr,
        M,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        CHANNELS: tl.constexpr,
        SLOTS: tl.constexpr,
        AXES: tl.constexpr,
    ):
        # One tile of the flattened (N, AXES, SLOTS) row grid. A row's source
        # slot is its window's row times the channel count plus the axis its own
        # grid position names; a negative window index reads the shared base
        # instead, and the masked load is what keeps that from being an address.
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        live_m = rows < M
        offs = tl.arange(0, BLOCK_D)
        live_d = offs < D

        index = tl.load(index_ptr + rows, mask=live_m, other=-1)
        present = index >= 0
        axis = (rows // SLOTS) % AXES if CHANNELS > 1 else rows * 0
        slot = index * CHANNELS + axis
        gathered = tl.load(
            source_ptr + slot[:, None] * D + offs[None, :],
            mask=present[:, None] & live_m[:, None] & live_d[None, :],
            other=0.0,
        )
        shared = tl.load(base_ptr + offs, mask=live_d, other=0.0)
        value = tl.where(present[:, None], gathered, shared[None, :])
        tl.store(
            out_ptr + rows[:, None] * D + offs[None, :],
            value,
            mask=live_m[:, None] & live_d[None, :],
        )

    @triton.jit
    def _gather_source_backward_kernel(
        window_ptr,
        window_rows,
        grad_ptr,
        d_source_ptr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
        CHANNELS: tl.constexpr,
        SLOTS: tl.constexpr,
        AXES: tl.constexpr,
    ):
        # One program owns one source slot. The CPU plan groups live flattened
        # action rows by persistent window in their original order; an axis
        # slot keeps only rows whose grid axis names that slot. Two programs
        # therefore never write one destination, and every sum has a fixed
        # association without an atomic scatter.
        slot = tl.program_id(0)
        window = slot // CHANNELS
        wanted_axis = slot % CHANNELS
        offs = tl.arange(0, BLOCK_D)
        live_d = offs < D
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        start = tl.load(window_ptr + window)
        end = tl.load(window_ptr + window + 1)
        for entry in tl.range(start, end):
            row = tl.load(window_rows + entry)
            axis = (row // SLOTS) % AXES
            selected = (axis == wanted_axis) if CHANNELS > 1 else True
            grad = tl.load(
                grad_ptr + row * D + offs,
                mask=selected & live_d,
                other=0.0,
            ).to(tl.float32)
            acc += grad
        tl.store(d_source_ptr + slot * D + offs, acc, mask=live_d)

    @triton.jit
    def _gather_base_backward_partial_kernel(
        sentinel_rows,
        grad_ptr,
        partial_ptr,
        sentinel_count,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        # The sentinel list is ascending flattened-row order.  One program
        # owns one contiguous slice and writes one fp32 partial; no two
        # programs ever write the same address.
        block = tl.program_id(0)
        start = block * BLOCK_ROWS
        end = tl.minimum(start + BLOCK_ROWS, sentinel_count)
        offs = tl.arange(0, BLOCK_D)
        live_d = offs < D
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for entry in tl.range(start, end):
            row = tl.load(sentinel_rows + entry)
            acc += tl.load(
                grad_ptr + row * D + offs, mask=live_d, other=0.0
            ).to(tl.float32)
        tl.store(partial_ptr + block * D + offs, acc, mask=live_d)

    @triton.jit
    def _gather_base_backward_combine_kernel(
        partial_ptr,
        d_base_ptr,
        partial_count,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        # A single owner visits the stage-one partials in their fixed ascending
        # block order.  This is the only serialization left, and its trip count
        # is reduced by GATHER_BASE_BLOCK_ROWS.
        offs = tl.arange(0, BLOCK_D)
        live_d = offs < D
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for block in tl.range(0, partial_count):
            acc += tl.load(
                partial_ptr + block * D + offs,
                mask=live_d,
                other=0.0,
            ).to(tl.float32)
        tl.store(d_base_ptr + offs, acc, mask=live_d)

    @triton.jit
    def _row_gate_kernel(
        source_ptr,
        ln_weight_ptr,
        ln_bias_ptr,
        wv_ptr,
        relation_ptr,
        wb_ptr,
        bb_ptr,
        wg_ptr,
        bg_ptr,
        out_ptr,
        M,
        EPS,
        D: tl.constexpr,
        R: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        # §14's gated combination on one tile of rows. Everything between the
        # gathered row and the product lives in registers: the normalised row,
        # the projected value, the relation's bias and the gate never reach
        # memory, and neither do their gradients' saved copies.
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        live_m = rows < M
        d = tl.arange(0, BLOCK_D)
        live_d = d < D
        r = tl.arange(0, BLOCK_R)
        live_r = r < R

        source = tl.load(
            source_ptr + rows[:, None] * D + d[None, :],
            mask=live_m[:, None] & live_d[None, :],
            other=0.0,
        ).to(tl.float32)
        relation = tl.load(
            relation_ptr + rows[:, None] * R + r[None, :],
            mask=live_m[:, None] & live_r[None, :],
            other=0.0,
        ).to(tl.float32)
        ln_weight = tl.load(ln_weight_ptr + d, mask=live_d, other=0.0).to(tl.float32)
        ln_bias = tl.load(ln_bias_ptr + d, mask=live_d, other=0.0).to(tl.float32)
        wv = tl.load(
            wv_ptr + d[:, None] * D + d[None, :],
            mask=live_d[:, None] & live_d[None, :],
            other=0.0,
        ).to(tl.float32)
        wb = tl.load(
            wb_ptr + d[:, None] * R + r[None, :],
            mask=live_d[:, None] & live_r[None, :],
            other=0.0,
        ).to(tl.float32)
        wg = tl.load(
            wg_ptr + d[:, None] * R + r[None, :],
            mask=live_d[:, None] & live_r[None, :],
            other=0.0,
        ).to(tl.float32)
        bb = tl.load(bb_ptr + d, mask=live_d, other=0.0).to(tl.float32)
        bg = tl.load(bg_ptr + d, mask=live_d, other=0.0).to(tl.float32)

        mean = tl.sum(source, axis=1) / D
        centred = tl.where(live_d[None, :], source - mean[:, None], 0.0)
        rstd = 1.0 / tl.sqrt(tl.sum(centred * centred, axis=1) / D + EPS)
        normed = centred * rstd[:, None]
        u = tl.where(
            live_d[None, :], normed * ln_weight[None, :] + ln_bias[None, :], 0.0
        )

        value = tl.dot(u, tl.trans(wv), input_precision="ieee")
        bias = tl.dot(relation, tl.trans(wb), input_precision="ieee") + bb[None, :]
        gate = tl.sigmoid(
            tl.dot(relation, tl.trans(wg), input_precision="ieee") + bg[None, :]
        )
        tl.store(
            out_ptr + rows[:, None] * D + d[None, :],
            gate * (value + bias),
            mask=live_m[:, None] & live_d[None, :],
        )

    @triton.jit
    def _row_gate_dinput_kernel(
        source_ptr,
        ln_weight_ptr,
        ln_bias_ptr,
        wv_ptr,
        relation_ptr,
        wb_ptr,
        bb_ptr,
        wg_ptr,
        bg_ptr,
        grad_ptr,
        d_source_ptr,
        d_relation_ptr,
        partial_ptr,
        M,
        EPS,
        D: tl.constexpr,
        R: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        # The per-row half of the backward: the forward re-derived from the
        # inputs, then the gradient of the row's own source and relation. The
        # only accumulators here are the LayerNorm's two vectors, since they
        # are the only table gradients whose summand needs the whole output
        # row; every other one is column-local and belongs to the second
        # kernel, where a slice of the output columns keeps the live set
        # inside the register file.
        program = tl.program_id(0)
        d = tl.arange(0, BLOCK_D)
        live_d = d < D
        r = tl.arange(0, BLOCK_R)
        live_r = r < R

        ln_weight = tl.load(ln_weight_ptr + d, mask=live_d, other=0.0).to(tl.float32)
        ln_bias = tl.load(ln_bias_ptr + d, mask=live_d, other=0.0).to(tl.float32)
        wv = tl.load(
            wv_ptr + d[:, None] * D + d[None, :],
            mask=live_d[:, None] & live_d[None, :],
            other=0.0,
        ).to(tl.float32)
        wb = tl.load(
            wb_ptr + d[:, None] * R + r[None, :],
            mask=live_d[:, None] & live_r[None, :],
            other=0.0,
        ).to(tl.float32)
        wg = tl.load(
            wg_ptr + d[:, None] * R + r[None, :],
            mask=live_d[:, None] & live_r[None, :],
            other=0.0,
        ).to(tl.float32)
        bb = tl.load(bb_ptr + d, mask=live_d, other=0.0).to(tl.float32)
        bg = tl.load(bg_ptr + d, mask=live_d, other=0.0).to(tl.float32)

        acc_lnw = tl.zeros([BLOCK_D], dtype=tl.float32)
        acc_lnb = tl.zeros([BLOCK_D], dtype=tl.float32)

        start = program * BLOCK_M
        stride = tl.num_programs(0) * BLOCK_M
        for tile in tl.range(start, M, stride):
            rows = tile + tl.arange(0, BLOCK_M)
            live_m = rows < M
            keep = live_m[:, None] & live_d[None, :]
            source = tl.load(
                source_ptr + rows[:, None] * D + d[None, :], mask=keep, other=0.0
            ).to(tl.float32)
            relation = tl.load(
                relation_ptr + rows[:, None] * R + r[None, :],
                mask=live_m[:, None] & live_r[None, :],
                other=0.0,
            ).to(tl.float32)
            grad = tl.load(
                grad_ptr + rows[:, None] * D + d[None, :], mask=keep, other=0.0
            ).to(tl.float32)

            mean = tl.sum(source, axis=1) / D
            centred = tl.where(live_d[None, :], source - mean[:, None], 0.0)
            rstd = 1.0 / tl.sqrt(tl.sum(centred * centred, axis=1) / D + EPS)
            normed = centred * rstd[:, None]
            u = tl.where(
                live_d[None, :], normed * ln_weight[None, :] + ln_bias[None, :], 0.0
            )
            value = tl.dot(u, tl.trans(wv), input_precision="ieee")
            bias = tl.dot(relation, tl.trans(wb), input_precision="ieee") + bb[None, :]
            gate = tl.sigmoid(
                tl.dot(relation, tl.trans(wg), input_precision="ieee") + bg[None, :]
            )

            # x = gate * (value + bias): bilinear, so both factors come back out
            # of the two tensors already in registers.
            d_sum = tl.where(keep, gate * grad, 0.0)
            d_pre_gate = tl.where(
                keep, (value + bias) * grad * gate * (1.0 - gate), 0.0
            )
            tl.store(
                d_relation_ptr + rows[:, None] * R + r[None, :],
                tl.dot(d_sum, wb, input_precision="ieee")
                + tl.dot(d_pre_gate, wg, input_precision="ieee"),
                mask=live_m[:, None] & live_r[None, :],
            )

            d_u = tl.where(keep, tl.dot(d_sum, wv, input_precision="ieee"), 0.0)
            acc_lnw += tl.sum(d_u * normed, axis=0)
            acc_lnb += tl.sum(d_u, axis=0)
            d_normed = tl.where(keep, d_u * ln_weight[None, :], 0.0)
            tl.store(
                d_source_ptr + rows[:, None] * D + d[None, :],
                rstd[:, None]
                * (
                    d_normed
                    - tl.sum(d_normed, axis=1)[:, None] / D
                    - normed * (tl.sum(d_normed * normed, axis=1)[:, None] / D)
                ),
                mask=keep,
            )

        tl.store(partial_ptr + program * 2 * BLOCK_D + d, acc_lnw, mask=live_d)
        tl.store(
            partial_ptr + program * 2 * BLOCK_D + BLOCK_D + d, acc_lnb, mask=live_d
        )

    @triton.jit
    def _row_gate_dtable_kernel(
        source_ptr,
        ln_weight_ptr,
        ln_bias_ptr,
        wv_ptr,
        relation_ptr,
        wb_ptr,
        bb_ptr,
        wg_ptr,
        bg_ptr,
        grad_ptr,
        partial_ptr,
        M,
        EPS,
        D: tl.constexpr,
        R: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_O: tl.constexpr,
        PARTIAL: tl.constexpr,
    ):
        # The five column-local table gradients, over a slice of the output
        # columns. Column ``o`` of ``d_sum`` and ``d_pre_gate`` needs only row
        # ``o`` of each weight, so a slice of ``BLOCK_O`` columns holds
        # ``BLOCK_O * (D + 2R + 2)`` accumulators live across the whole sweep
        # where the whole row would hold ``D`` times as many.
        program = tl.program_id(0)
        column = tl.program_id(1)
        d = tl.arange(0, BLOCK_D)
        live_d = d < D
        r = tl.arange(0, BLOCK_R)
        live_r = r < R
        o = column * BLOCK_O + tl.arange(0, BLOCK_O)
        live_o = o < D

        ln_weight = tl.load(ln_weight_ptr + d, mask=live_d, other=0.0).to(tl.float32)
        ln_bias = tl.load(ln_bias_ptr + d, mask=live_d, other=0.0).to(tl.float32)
        wv = tl.load(
            wv_ptr + o[:, None] * D + d[None, :],
            mask=live_o[:, None] & live_d[None, :],
            other=0.0,
        ).to(tl.float32)
        wb = tl.load(
            wb_ptr + o[:, None] * R + r[None, :],
            mask=live_o[:, None] & live_r[None, :],
            other=0.0,
        ).to(tl.float32)
        wg = tl.load(
            wg_ptr + o[:, None] * R + r[None, :],
            mask=live_o[:, None] & live_r[None, :],
            other=0.0,
        ).to(tl.float32)
        bb = tl.load(bb_ptr + o, mask=live_o, other=0.0).to(tl.float32)
        bg = tl.load(bg_ptr + o, mask=live_o, other=0.0).to(tl.float32)

        acc_wv = tl.zeros([BLOCK_O, BLOCK_D], dtype=tl.float32)
        acc_wb = tl.zeros([BLOCK_O, BLOCK_R], dtype=tl.float32)
        acc_wg = tl.zeros([BLOCK_O, BLOCK_R], dtype=tl.float32)
        acc_bb = tl.zeros([BLOCK_O], dtype=tl.float32)
        acc_bg = tl.zeros([BLOCK_O], dtype=tl.float32)

        start = program * BLOCK_M
        stride = tl.num_programs(0) * BLOCK_M
        for tile in tl.range(start, M, stride):
            rows = tile + tl.arange(0, BLOCK_M)
            live_m = rows < M
            source = tl.load(
                source_ptr + rows[:, None] * D + d[None, :],
                mask=live_m[:, None] & live_d[None, :],
                other=0.0,
            ).to(tl.float32)
            relation = tl.load(
                relation_ptr + rows[:, None] * R + r[None, :],
                mask=live_m[:, None] & live_r[None, :],
                other=0.0,
            ).to(tl.float32)
            grad = tl.load(
                grad_ptr + rows[:, None] * D + o[None, :],
                mask=live_m[:, None] & live_o[None, :],
                other=0.0,
            ).to(tl.float32)

            mean = tl.sum(source, axis=1) / D
            centred = tl.where(live_d[None, :], source - mean[:, None], 0.0)
            rstd = 1.0 / tl.sqrt(tl.sum(centred * centred, axis=1) / D + EPS)
            u = tl.where(
                live_d[None, :],
                centred * rstd[:, None] * ln_weight[None, :] + ln_bias[None, :],
                0.0,
            )
            value = tl.dot(u, tl.trans(wv), input_precision="ieee")
            bias = tl.dot(relation, tl.trans(wb), input_precision="ieee") + bb[None, :]
            gate = tl.sigmoid(
                tl.dot(relation, tl.trans(wg), input_precision="ieee") + bg[None, :]
            )
            keep = live_m[:, None] & live_o[None, :]
            d_sum = tl.where(keep, gate * grad, 0.0)
            d_pre_gate = tl.where(
                keep, (value + bias) * grad * gate * (1.0 - gate), 0.0
            )
            acc_wv += tl.dot(tl.trans(d_sum), u, input_precision="ieee")
            acc_wb += tl.dot(tl.trans(d_sum), relation, input_precision="ieee")
            acc_wg += tl.dot(tl.trans(d_pre_gate), relation, input_precision="ieee")
            acc_bb += tl.sum(d_sum, axis=0)
            acc_bg += tl.sum(d_pre_gate, axis=0)

        # This program owns rows ``o`` of every table in its own flat slice, and
        # the column slices are disjoint, so two programs never write one
        # address and the partial buffer needs no atomic.
        slot = partial_ptr + program * PARTIAL
        tl.store(
            slot + o[:, None] * BLOCK_D + d[None, :],
            acc_wv,
            mask=live_o[:, None] & live_d[None, :],
        )
        base = BLOCK_D * BLOCK_D
        tl.store(
            slot + base + o[:, None] * BLOCK_R + r[None, :],
            acc_wb,
            mask=live_o[:, None] & live_r[None, :],
        )
        base += BLOCK_D * BLOCK_R
        tl.store(
            slot + base + o[:, None] * BLOCK_R + r[None, :],
            acc_wg,
            mask=live_o[:, None] & live_r[None, :],
        )
        base += BLOCK_D * BLOCK_R
        tl.store(slot + base + o, acc_bb, mask=live_o)
        base += BLOCK_D
        tl.store(slot + base + o, acc_bg, mask=live_o)


# --------------------------------------------------------------------------
# The torch reference (§36) — CPU, unsupported signatures, and parity


def _gather_reference(
    source: Tensor, base: Tensor, index: Tensor, channels: int
) -> Tensor:
    """§19.2's sentinel-padded gather, written as the table concatenation.

    Literal: pad the source table with the base, send every sentinel row to
    the pad, and gather. This is the formulation the kernel exists to avoid —
    the pad puts many rows on one shared destination in the backward — and
    keeping it is what makes a parity test mean anything.
    """
    rows, axes, slots = index.shape
    width = int(source.shape[-1])
    padded = torch.cat([source, base.to(source.dtype).reshape(1, width)])
    pad_row = int(source.shape[0])
    if channels == 1:
        flat = torch.where(index >= 0, index, torch.full_like(index, pad_row))
    else:
        axis = torch.arange(axes, device=index.device).view(1, -1, 1)
        flat = torch.where(
            index >= 0, index * channels + axis, torch.full_like(index, pad_row)
        )
    return padded.index_select(0, flat.reshape(-1)).view(rows, axes, slots, width)


def _row_gate_reference(
    source: Tensor,
    ln_weight: Tensor,
    ln_bias: Tensor,
    wv: Tensor,
    relation: Tensor,
    wb: Tensor,
    bb: Tensor,
    wg: Tensor,
    bg: Tensor,
    eps: float,
) -> Tensor:
    """§14's combination as `PostPlacementEncoder` writes it, term for term.

    ``sigmoid(W_g e) * (W_v LN(w) + W_b e)`` over whole tensors, with the
    product taken at no less than fp32 (§27). Every intermediate here is one of
    the ``(M, d)`` tensors the fused path keeps in registers.
    """
    width = int(source.shape[-1])
    accumulate = _accumulate_dtype(source)
    normed = torch.nn.functional.layer_norm(
        source, (width,), ln_weight, ln_bias, eps
    )
    value = torch.nn.functional.linear(normed, wv)
    bias = torch.nn.functional.linear(relation, wb, bb)
    gate = torch.sigmoid(torch.nn.functional.linear(relation, wg, bg))
    return gate.to(accumulate) * (value.to(accumulate) + bias.to(accumulate))


def _gather_reference_backward(
    source: Tensor, base: Tensor, index: Tensor, channels: int, grad_out: Tensor
) -> tuple[Tensor, Tensor]:
    """:func:`_gather_reference`'s gradient over the same padded table.

    One ``index_add_`` into ``(n_rows * channels + 1, D)`` and a split, which
    is what autograd does to the concatenation, written where ``gradcheck``
    can reach it.
    """
    width = int(source.shape[-1])
    accumulate = _accumulate_dtype(source)
    pad_row = int(source.shape[0])
    axes = int(index.shape[1])
    if channels == 1:
        flat = torch.where(index >= 0, index, torch.full_like(index, pad_row))
    else:
        axis = torch.arange(axes, device=index.device).view(1, -1, 1)
        flat = torch.where(
            index >= 0, index * channels + axis, torch.full_like(index, pad_row)
        )
    d_padded = torch.zeros(
        pad_row + 1, width, dtype=accumulate, device=source.device
    ).index_add_(
        0, flat.reshape(-1), grad_out.reshape(-1, width).to(accumulate)
    )
    return (
        d_padded[:pad_row].to(source.dtype).clone(),
        d_padded[pad_row].to(base.dtype).clone(),
    )


def _row_gate_reference_backward(
    source: Tensor,
    ln_weight: Tensor,
    ln_bias: Tensor,
    wv: Tensor,
    relation: Tensor,
    wb: Tensor,
    bb: Tensor,
    wg: Tensor,
    bg: Tensor,
    eps: float,
    grad_out: Tensor,
) -> tuple[
    Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor
]:
    """:func:`_row_gate_reference`'s gradient, recomputed from the same inputs.

    The forward is re-derived rather than stored, which is the trade the kernel
    makes; writing it here is what lets ``gradcheck`` check the derivation by
    finite differences instead of against a second copy of the same algebra.
    """
    accumulate = _accumulate_dtype(source)
    w = source.to(accumulate)
    e = relation.to(accumulate)
    grad = grad_out.to(accumulate)

    mean = w.mean(dim=-1, keepdim=True)
    centred = w - mean
    rstd = torch.rsqrt(centred.square().mean(dim=-1, keepdim=True) + eps)
    normed = centred * rstd
    u = normed * ln_weight.to(accumulate) + ln_bias.to(accumulate)
    value = u @ wv.to(accumulate).t()
    bias = e @ wb.to(accumulate).t() + bb.to(accumulate)
    gate = torch.sigmoid(e @ wg.to(accumulate).t() + bg.to(accumulate))

    d_sum = gate * grad
    d_pre_gate = (value + bias) * grad * gate * (1.0 - gate)
    d_relation = d_sum @ wb.to(accumulate) + d_pre_gate @ wg.to(accumulate)
    d_u = d_sum @ wv.to(accumulate)
    d_normed = d_u * ln_weight.to(accumulate)
    d_source = rstd * (
        d_normed
        - d_normed.mean(dim=-1, keepdim=True)
        - normed * (d_normed * normed).mean(dim=-1, keepdim=True)
    )
    return (
        d_source.to(source.dtype),
        (d_u * normed).sum(dim=0).to(ln_weight.dtype),
        d_u.sum(dim=0).to(ln_bias.dtype),
        (d_sum.t() @ u).to(wv.dtype),
        d_relation.to(relation.dtype),
        (d_sum.t() @ e).to(wb.dtype),
        d_sum.sum(dim=0).to(bb.dtype),
        (d_pre_gate.t() @ e).to(wg.dtype),
        d_pre_gate.sum(dim=0).to(bg.dtype),
    )


# --------------------------------------------------------------------------
# Guards, launches, and the custom ops


def _validate_gather(
    source: Tensor,
    base: Tensor,
    index: Tensor,
    channels: int,
    window_ptr: Tensor,
    window_rows: Tensor,
    sentinel_rows: Tensor,
) -> None:
    if channels not in (1, AXIS_CHANNELS):
        raise ValueError(
            f"channels must be scalar or the {AXIS_CHANNELS} ACT axes, "
            f"got {channels}"
        )
    if source.ndim != 2:
        raise ValueError(
            f"source must be (n_rows * channels, D), got {tuple(source.shape)}"
        )
    if source.shape[0] % channels:
        raise ValueError(
            f"source has {source.shape[0]} rows, not a multiple of {channels} channels"
        )
    if base.ndim != 1 or base.shape[0] != source.shape[1]:
        raise ValueError(
            f"base must be ({source.shape[1]},) beside the source's width, got "
            f"{tuple(base.shape)}"
        )
    if index.ndim != 3 or index.shape[1] != AXIS_CHANNELS:
        raise ValueError(
            f"index must be (N, {AXIS_CHANNELS}, slots), got "
            f"{tuple(index.shape)}"
        )
    if index.dtype != torch.long:
        raise ValueError(f"index must be int64, got {index.dtype}")
    if base.device != source.device or index.device != source.device:
        raise ValueError("every post-row gather input must be on one device")
    windows = int(source.shape[0]) // channels
    total = int(index.numel())
    for name, value, rows in (
        ("window_ptr", window_ptr, windows + 1),
        ("window_rows", window_rows, None),
        ("sentinel_rows", sentinel_rows, None),
    ):
        if value.dtype != torch.int32:
            raise TypeError(f"{name} must be int32, got {value.dtype}")
        if value.ndim != 1 or (rows is not None and int(value.shape[0]) != rows):
            expected = "1-D" if rows is None else f"({rows},)"
            raise ValueError(f"{name} must be {expected}, got {tuple(value.shape)}")
        if not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if value.device != source.device:
            raise ValueError("every post-row gather input must be on one device")
    if int(window_rows.numel()) + int(sentinel_rows.numel()) != total:
        raise ValueError(
            "window_rows and sentinel_rows must cover every flattened action row"
        )


def _validate_row_gate(
    source: Tensor,
    ln_weight: Tensor,
    ln_bias: Tensor,
    wv: Tensor,
    relation: Tensor,
    wb: Tensor,
    bb: Tensor,
    wg: Tensor,
    bg: Tensor,
) -> None:
    if source.ndim != 2 or relation.ndim != 2:
        raise ValueError(
            f"source must be (M, D) and relation (M, R), got "
            f"{tuple(source.shape)} and {tuple(relation.shape)}"
        )
    if source.shape[0] != relation.shape[0]:
        raise ValueError(
            f"source carries {source.shape[0]} rows against relation's "
            f"{relation.shape[0]}"
        )
    width, rel_width = int(source.shape[1]), int(relation.shape[1])
    for name, tensor, shape in (
        ("ln_weight", ln_weight, (width,)),
        ("ln_bias", ln_bias, (width,)),
        ("wv", wv, (width, width)),
        ("wb", wb, (width, rel_width)),
        ("bb", bb, (width,)),
        ("wg", wg, (width, rel_width)),
        ("bg", bg, (width,)),
    ):
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"{name} must be {shape} for a {width}-wide row over a "
                f"{rel_width}-wide relation, got {tuple(tensor.shape)}"
            )
    tensors = (ln_weight, ln_bias, wv, relation, wb, bb, wg, bg)
    if any(tensor.device != source.device for tensor in tensors):
        raise ValueError("every post-row gate input must be on one device")


# A row's weights are resident for the whole sweep, so the widest row that
# fits is set by shared memory, and it is the *padded* width that costs — past
# this, the launch is refused by the device. Every preset is 64 wide invariant
# and 24 axis; a wider one falls back to the reference by signature, which is
# a supported shape answering correctly rather than a caught OutOfResources.
_MAX_BLOCK_WIDTH = 64


def _supported(sample: Tensor, width: int, rows: int) -> bool:
    return (
        triton is not None
        and sample.is_cuda
        and sample.dtype in _SUPPORTED_DTYPES
        and 0 < width
        and triton.next_power_of_2(width) <= _MAX_BLOCK_WIDTH
        and rows > 0
    )


def _shape_key(sample: Tensor, *rest: object) -> tuple[object, ...]:
    return (sample.device.type, sample.device.index, sample.dtype, *rest)


def _launch_gather(
    source: Tensor, base: Tensor, index: Tensor, channels: int
) -> Tensor:
    rows, axes, slots = index.shape
    width = int(source.shape[1])
    total = rows * axes * slots
    out = torch.empty(total, width, dtype=source.dtype, device=source.device)
    _gather_kernel[(triton.cdiv(total, _BLOCK_M),)](
        source,
        base,
        index.reshape(-1),
        out,
        total,
        D=width,
        BLOCK_D=triton.next_power_of_2(width),
        BLOCK_M=_BLOCK_M,
        CHANNELS=channels,
        SLOTS=slots,
        AXES=axes,
    )
    return out.view(rows, axes, slots, width)


def _launch_gather_backward(
    source: Tensor,
    base: Tensor,
    index: Tensor,
    channels: int,
    window_ptr: Tensor,
    window_rows: Tensor,
    sentinel_rows: Tensor,
    grad_out: Tensor,
) -> tuple[Tensor, Tensor]:
    width = int(source.shape[1])
    _rows, axes, slots = index.shape
    block_d = triton.next_power_of_2(width)
    d_source = torch.empty_like(source)
    if source.shape[0]:
        _gather_source_backward_kernel[(int(source.shape[0]),)](
            window_ptr,
            window_rows,
            grad_out,
            d_source,
            D=width,
            BLOCK_D=block_d,
            CHANNELS=channels,
            SLOTS=slots,
            AXES=axes,
        )
    d_base = torch.empty_like(base)
    sentinel_count = int(sentinel_rows.numel())
    partial_count = max(1, triton.cdiv(sentinel_count, _GATHER_BASE_BLOCK_ROWS))
    base_partials = torch.empty(
        partial_count,
        width,
        dtype=torch.float32,
        device=base.device,
    )
    _gather_base_backward_partial_kernel[(partial_count,)](
        sentinel_rows,
        grad_out,
        base_partials,
        sentinel_count,
        D=width,
        BLOCK_D=block_d,
        BLOCK_ROWS=_GATHER_BASE_BLOCK_ROWS,
    )
    _gather_base_backward_combine_kernel[(1,)](
        base_partials,
        d_base,
        partial_count,
        D=width,
        BLOCK_D=block_d,
    )
    return d_source, d_base


def _row_gate_partial_width(width: int, rel_width: int) -> tuple[int, int, int]:
    block_d = triton.next_power_of_2(width)
    block_r = max(16, triton.next_power_of_2(rel_width))
    return block_d, block_r, block_d * block_d + 2 * block_d * block_r + 4 * block_d


def _launch_row_gate(
    source: Tensor,
    ln_weight: Tensor,
    ln_bias: Tensor,
    wv: Tensor,
    relation: Tensor,
    wb: Tensor,
    bb: Tensor,
    wg: Tensor,
    bg: Tensor,
    eps: float,
) -> Tensor:
    total, width = source.shape
    rel_width = int(relation.shape[1])
    block_d, block_r, _ = _row_gate_partial_width(width, rel_width)
    out = torch.empty(
        total, width, dtype=_accumulate_dtype(source), device=source.device
    )
    _row_gate_kernel[(triton.cdiv(total, _BLOCK_M),)](
        source,
        ln_weight,
        ln_bias,
        wv,
        relation,
        wb,
        bb,
        wg,
        bg,
        out,
        total,
        eps,
        D=width,
        R=rel_width,
        BLOCK_D=block_d,
        BLOCK_R=block_r,
        BLOCK_M=_BLOCK_M,
    )
    return out


def _launch_row_gate_backward(
    source: Tensor,
    ln_weight: Tensor,
    ln_bias: Tensor,
    wv: Tensor,
    relation: Tensor,
    wb: Tensor,
    bb: Tensor,
    wg: Tensor,
    bg: Tensor,
    eps: float,
    grad_out: Tensor,
) -> tuple[Tensor, ...]:
    total, width = source.shape
    rel_width = int(relation.shape[1])
    block_d, block_r, partial_width = _row_gate_partial_width(width, rel_width)
    weights = (ln_weight, ln_bias, wv, relation, wb, bb, wg, bg)

    d_source = torch.empty(
        total, width, dtype=torch.float32, device=source.device
    )
    d_relation = torch.empty(
        total, rel_width, dtype=torch.float32, device=source.device
    )
    rows = min(_ROW_PROGRAMS, max(1, triton.cdiv(total, _BLOCK_M)))
    partial_ln = torch.empty(
        rows, 2 * block_d, dtype=torch.float32, device=source.device
    )
    _row_gate_dinput_kernel[(rows,)](
        source,
        *weights,
        grad_out,
        d_source,
        d_relation,
        partial_ln,
        total,
        eps,
        D=width,
        R=rel_width,
        BLOCK_D=block_d,
        BLOCK_R=block_r,
        BLOCK_M=_BLOCK_M,
        num_warps=_ROW_WARPS,
    )

    columns = triton.cdiv(width, _BLOCK_O)
    tables = min(_TABLE_PROGRAMS, max(1, triton.cdiv(total, _BLOCK_M)))
    partial = torch.zeros(
        tables, partial_width, dtype=torch.float32, device=source.device
    )
    _row_gate_dtable_kernel[(tables, columns)](
        source,
        *weights,
        grad_out,
        partial,
        total,
        eps,
        D=width,
        R=rel_width,
        BLOCK_D=block_d,
        BLOCK_R=block_r,
        BLOCK_M=_BLOCK_M,
        BLOCK_O=_BLOCK_O,
        PARTIAL=partial_width,
        num_warps=_TABLE_WARPS,
    )

    # The second stage of the reduction: every table's partials are summed
    # along dimension zero, in program order, so the remainder is reassociation
    # noise fixed by the grid rather than run-to-run noise. Each sum allocates
    # its own result, because a custom op may not answer with several views of
    # one buffer.
    def _table(buffer: Tensor, start: int, *shape: int) -> tuple[Tensor, int]:
        span = shape[0] * (shape[1] if len(shape) > 1 else 1)
        view = buffer[:, start : start + span].view(buffer.shape[0], *shape)
        if len(shape) == 1:
            return view[:, :width].sum(dim=0), start + span
        return (
            view[:, : shape[0], : shape[1]].sum(dim=0),
            start + span,
        )

    d_wv, cut = _table(partial, 0, block_d, block_d)
    d_wv = d_wv[:width, :width].contiguous()
    d_wb, cut = _table(partial, cut, block_d, block_r)
    d_wb = d_wb[:width, :rel_width].contiguous()
    d_wg, cut = _table(partial, cut, block_d, block_r)
    d_wg = d_wg[:width, :rel_width].contiguous()
    d_bb, cut = _table(partial, cut, block_d)
    d_bg, _cut = _table(partial, cut, block_d)
    d_ln_weight, _ = _table(partial_ln, 0, block_d)
    d_ln_bias, _ = _table(partial_ln, block_d, block_d)
    return (
        d_source.to(source.dtype),
        d_ln_weight.to(ln_weight.dtype),
        d_ln_bias.to(ln_bias.dtype),
        d_wv.to(wv.dtype),
        d_relation.to(relation.dtype),
        d_wb.to(wb.dtype),
        d_bb.to(bb.dtype),
        d_wg.to(wg.dtype),
        d_bg.to(bg.dtype),
    )


@torch.library.custom_op("mantisnet::act_post_gather", mutates_args=())
def _gather_op(
    source: Tensor,
    base: Tensor,
    index: Tensor,
    channels: int,
    window_ptr: Tensor,
    window_rows: Tensor,
    sentinel_rows: Tensor,
) -> Tensor:
    _validate_gather(
        source,
        base,
        index,
        channels,
        window_ptr,
        window_rows,
        sentinel_rows,
    )
    reference = lambda: _gather_reference(source, base, index, channels)  # noqa: E731
    total = int(index.numel())
    if not _supported(source, int(source.shape[1]), total):
        return reference()
    key = _shape_key(source, int(source.shape[1]), channels)
    if key in _FAILED_SHAPES:
        return reference()
    try:
        return _launch_gather(source, base, index, channels)
    except Exception as exc:
        _FAILED_SHAPES[key] = f"{type(exc).__name__}: {exc}"
        warnings.warn(
            f"fused post-row gather failed for D={source.shape[1]}, "
            f"channels={channels}; padding the table instead for this shape: "
            f"{_FAILED_SHAPES[key]}",
            RuntimeWarning,
            stacklevel=2,
        )
        return reference()


@_gather_op.register_fake
def _(
    source: Tensor,
    base: Tensor,
    index: Tensor,
    channels: int,
    window_ptr: Tensor,
    window_rows: Tensor,
    sentinel_rows: Tensor,
) -> Tensor:
    rows, axes, slots = index.shape
    return source.new_empty((rows, axes, slots, source.shape[1]))


@torch.library.custom_op("mantisnet::act_post_gather_backward", mutates_args=())
def _gather_backward_op(
    source: Tensor,
    base: Tensor,
    index: Tensor,
    channels: int,
    window_ptr: Tensor,
    window_rows: Tensor,
    sentinel_rows: Tensor,
    grad_out: Tensor,
) -> tuple[Tensor, Tensor]:
    reference = lambda: _gather_reference_backward(  # noqa: E731
        source, base, index, channels, grad_out
    )
    if not _supported(source, int(source.shape[1]), int(index.numel())):
        return reference()
    key = _shape_key(source, int(source.shape[1]), channels, "backward")
    if key in _FAILED_BACKWARD_SHAPES:
        return reference()
    try:
        return _launch_gather_backward(
            source,
            base,
            index,
            channels,
            window_ptr,
            window_rows,
            sentinel_rows,
            grad_out.reshape(-1, int(source.shape[1])).contiguous().float(),
        )
    except Exception as exc:
        _FAILED_BACKWARD_SHAPES[key] = f"{type(exc).__name__}: {exc}"
        warnings.warn(
            f"fused post-row gather backward failed for D={source.shape[1]}, "
            f"channels={channels}; scattering instead for this shape: "
            f"{_FAILED_BACKWARD_SHAPES[key]}",
            RuntimeWarning,
            stacklevel=2,
        )
        return reference()


@_gather_backward_op.register_fake
def _(
    source: Tensor,
    base: Tensor,
    index: Tensor,
    channels: int,
    window_ptr: Tensor,
    window_rows: Tensor,
    sentinel_rows: Tensor,
    grad_out: Tensor,
) -> tuple[Tensor, Tensor]:
    return torch.empty_like(source), torch.empty_like(base)


def _gather_setup(ctx, inputs, output) -> None:
    ctx.channels = inputs[3]
    ctx.save_for_backward(inputs[0], inputs[1], inputs[2], inputs[4], inputs[5], inputs[6])


def _gather_dispatch(ctx, grad_out: Tensor):
    source, base, index, window_ptr, window_rows, sentinel_rows = ctx.saved_tensors
    d_source, d_base = _gather_backward_op(
        source,
        base,
        index,
        ctx.channels,
        window_ptr,
        window_rows,
        sentinel_rows,
        grad_out,
    )
    return d_source, d_base, None, None, None, None, None


_gather_op.register_autograd(_gather_dispatch, setup_context=_gather_setup)


@torch.library.custom_op("mantisnet::act_post_row_gate", mutates_args=())
def _row_gate_op(
    source: Tensor,
    ln_weight: Tensor,
    ln_bias: Tensor,
    wv: Tensor,
    relation: Tensor,
    wb: Tensor,
    bb: Tensor,
    wg: Tensor,
    bg: Tensor,
    eps: float,
) -> Tensor:
    _validate_row_gate(source, ln_weight, ln_bias, wv, relation, wb, bb, wg, bg)
    reference = lambda: _row_gate_reference(  # noqa: E731
        source, ln_weight, ln_bias, wv, relation, wb, bb, wg, bg, eps
    )
    if not _supported(source, int(source.shape[1]), int(source.shape[0])):
        return reference()
    key = _shape_key(source, int(source.shape[1]), int(relation.shape[1]))
    if key in _FAILED_SHAPES:
        return reference()
    try:
        return _launch_row_gate(
            source, ln_weight, ln_bias, wv, relation, wb, bb, wg, bg, eps
        )
    except Exception as exc:
        _FAILED_SHAPES[key] = f"{type(exc).__name__}: {exc}"
        warnings.warn(
            f"fused post-row gate failed for D={source.shape[1]}, "
            f"R={relation.shape[1]}; gathering instead for this shape: "
            f"{_FAILED_SHAPES[key]}",
            RuntimeWarning,
            stacklevel=2,
        )
        return reference()


@_row_gate_op.register_fake
def _(
    source: Tensor,
    ln_weight: Tensor,
    ln_bias: Tensor,
    wv: Tensor,
    relation: Tensor,
    wb: Tensor,
    bb: Tensor,
    wg: Tensor,
    bg: Tensor,
    eps: float,
) -> Tensor:
    return source.new_empty(source.shape, dtype=_accumulate_dtype(source))


_ROW_GATE_ARITY = 9


@torch.library.custom_op("mantisnet::act_post_row_gate_backward", mutates_args=())
def _row_gate_backward_op(
    source: Tensor,
    ln_weight: Tensor,
    ln_bias: Tensor,
    wv: Tensor,
    relation: Tensor,
    wb: Tensor,
    bb: Tensor,
    wg: Tensor,
    bg: Tensor,
    eps: float,
    grad_out: Tensor,
) -> tuple[
    Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor
]:
    reference = lambda: _row_gate_reference_backward(  # noqa: E731
        source, ln_weight, ln_bias, wv, relation, wb, bb, wg, bg, eps, grad_out
    )
    if not _supported(source, int(source.shape[1]), int(source.shape[0])):
        return reference()
    key = _shape_key(
        source, int(source.shape[1]), int(relation.shape[1]), "backward"
    )
    if key in _FAILED_BACKWARD_SHAPES:
        return reference()
    try:
        return _launch_row_gate_backward(
            source,
            ln_weight,
            ln_bias,
            wv,
            relation,
            wb,
            bb,
            wg,
            bg,
            eps,
            grad_out.contiguous().float(),
        )
    except Exception as exc:
        _FAILED_BACKWARD_SHAPES[key] = f"{type(exc).__name__}: {exc}"
        warnings.warn(
            f"fused post-row gate backward failed for D={source.shape[1]}, "
            f"R={relation.shape[1]}; differentiating the reference instead for "
            f"this shape: {_FAILED_BACKWARD_SHAPES[key]}",
            RuntimeWarning,
            stacklevel=2,
        )
        return reference()


@_row_gate_backward_op.register_fake
def _(
    source: Tensor,
    ln_weight: Tensor,
    ln_bias: Tensor,
    wv: Tensor,
    relation: Tensor,
    wb: Tensor,
    bb: Tensor,
    wg: Tensor,
    bg: Tensor,
    eps: float,
    grad_out: Tensor,
) -> tuple[
    Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor
]:
    return tuple(
        torch.empty_like(tensor)
        for tensor in (source, ln_weight, ln_bias, wv, relation, wb, bb, wg, bg)
    )


def _row_gate_setup(ctx, inputs, output) -> None:
    ctx.eps = inputs[_ROW_GATE_ARITY]
    ctx.save_for_backward(*inputs[:_ROW_GATE_ARITY])


def _row_gate_dispatch(ctx, grad_out: Tensor):
    grads = _row_gate_backward_op(*ctx.saved_tensors, ctx.eps, grad_out)
    return (*grads, None)


_row_gate_op.register_autograd(_row_gate_dispatch, setup_context=_row_gate_setup)


# --------------------------------------------------------------------------
# What `action_encoder` calls


def sentinel_gather(
    source: Tensor,
    base: Tensor,
    index: Tensor,
    channels: int,
    window_ptr: Tensor,
    window_rows: Tensor,
    sentinel_rows: Tensor,
) -> Tensor:
    """§19.2's row states: ``source[index]``, or ``base`` where ``index < 0``.

    ``index`` is the ``(N, 3, 6)`` row grid, ``source`` is ``(n_rows, D)`` for
    the invariant stream and ``(n_rows * 3, D)`` flattened over the window's own
    channels for the axis stream, and ``channels`` says which. An axis row reads
    channel ``a`` of its window, ``a`` being the axis its grid position names,
    so the source slot is ``index * 3 + a`` — §12.3's "route line messages into
    the structural native axis", computed in the kernel rather than assembled
    from an ``arange``.

    ``window_ptr``/``window_rows`` are the CPU-planned stable CSR of present
    flattened rows by source window; ``sentinel_rows`` is the ascending list
    of absent rows. The forward needs only ``index``, but carrying these views
    through the registered op lets its backward reduce every repeated source
    deterministically without atomics.

    The result is ``(N, 3, 6, D)`` in ``source``'s dtype.
    """
    return _gather_op(
        source.contiguous(),
        base.to(source.dtype).contiguous(),
        index,
        channels,
        window_ptr,
        window_rows,
        sentinel_rows,
    )


def row_gate(
    source: Tensor,
    ln_weight: Tensor,
    ln_bias: Tensor,
    wv: Tensor,
    relation: Tensor,
    wb: Tensor,
    bb: Tensor,
    wg: Tensor,
    bg: Tensor,
    eps: float,
) -> Tensor:
    """``sigmoid(W_g e + b_g) * (W_v LN(w) + W_b e + b_b)`` over a row grid.

    ``source`` is the gathered rows flattened to ``(M, D)`` and ``relation``
    their ``(M, R)`` relation vectors; every weight is the module's own. The
    result is ``(M, D)`` at no less than fp32 (§27), and nothing between the
    two inputs and it is ever a tensor.
    """
    return _row_gate_op(
        source.contiguous(),
        ln_weight.contiguous(),
        ln_bias.contiguous(),
        wv.contiguous(),
        relation.contiguous(),
        wb.contiguous(),
        bb.contiguous(),
        wg.contiguous(),
        bg.contiguous(),
        float(eps),
    )


__all__ = ["row_gate", "sentinel_gather"]
