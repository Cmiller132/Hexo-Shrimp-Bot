"""The incidence pass both cell heads read.

MODEL_SPEC §6's policy decoder and appendix B's action-value decoder walk the
same table: a legal cell's live windows, the slot class the cell holds in each,
and — for a cell lying in no live window — its nearest-stone bucket. Only their
parameters differ. Their per-cell decoder input is

    covered cell:      h[c] = M · Σ_e w[dec_window[e]] + Σ_e e_class[dec_class[e]]
    background cell:   h[c] = e_bg[bg_bucket[c]]

over the cell's incidence entries ``e``. A linear map commutes with a sum, so
``M`` can be applied *after* the aggregation rather than under it. Everything
remaining under the sum is head-independent, and one pass over the incidence
then serves both heads instead of two.

`aggregate` is that pass. It returns the coefficients of ``h``, one row per
cell:

    [0, H)            Σ_e w[dec_window[e]]
    [H, H+3)          how many entries carried each slot class
    [H+3, H+11)       one-hot of the background bucket, background cells only
    [H+11, H+16)      zero — the row is rounded to a multiple of 16 so the head
                      GEMM keeps a tensor-core-friendly K

`head_matrix` folds a head's projection, both its embedding tables, and its
first MLP layer into the single matrix that reads a whole row, so the per-cell
work is exactly the GEMM the head already ran and the projection GEMM over
windows disappears. Neither function holds parameters: they read the ones
`model` owns, and the checkpoint layout is untouched.

The background block adds where the spec overwrites. A background cell is by
construction in no live window, so its window and class coefficients are zero
and the two agree.

The CUDA path is a Triton segment reduction over ``dec_cell``'s runs, which
the builder emits in cell order: each cell's entries are summed in registers
and its row is stored once, so the aggregation needs neither a zeroed
accumulator nor atomics. The torch path sums by ``index_add_`` and is
order-independent, which is what makes the parity tests between the two a
detector for a builder that stops emitting entries in cell order.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F
from torch import Tensor

from .builder import NEAREST_BUCKETS

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


CLASS_SLOTS = 3  # slot classes: end, near-end, centre (MODEL_SPEC §4.3)
BG_SLOTS = NEAREST_BUCKETS
# Rounded up from the 11 coefficients in use. K stays a multiple of 16 for the
# head GEMM, and the slack columns are stored as zero so they contribute
# nothing whatever the head matrix holds there.
COEF_WIDTH = 16

if CLASS_SLOTS + BG_SLOTS > COEF_WIDTH:
    raise RuntimeError(
        f"{CLASS_SLOTS} class + {BG_SLOTS} background coefficients exceed the "
        f"{COEF_WIDTH}-wide block"
    )

# One program per cell, one row of width H per program. A single warp beat
# {2, 4, 8} on the target RTX 4070 Ti — kernel milliseconds at cohort 256 were
# 0.68 / 0.72 / 0.84 / 1.40 at 50 stones and 3.36 / 3.40 / 4.25 / 7.21 at 400.
# Each of the 32 threads then covers four of the 128 columns, which is one
# vectorised load per thread against a row the warp reads whole. The geometry
# is fixed rather than autotuned so symbolic shape changes stay out of
# Triton's tuning cache.
_NUM_WARPS = 1

_FAILED_SHAPES: dict[tuple[object, ...], str] = {}


if triton is not None:

    @triton.jit
    def _aggregate_kernel(
        w_ptr,
        window_ptr,
        class_ptr,
        row_ptr,
        out_ptr,
        stride_ww,
        stride_wh,
        stride_oc,
        stride_oh,
        H: tl.constexpr,
        BLOCK_H: tl.constexpr,
        COEF: tl.constexpr,
    ):
        cell = tl.program_id(0)
        offs_h = tl.arange(0, BLOCK_H)
        live = offs_h < H
        offs_c = tl.arange(0, COEF)

        # The cell's run in the incidence arrays. Entries arrive in cell order,
        # so a run is contiguous and the sum needs no atomics.
        start = tl.load(row_ptr + cell)
        end = tl.load(row_ptr + cell + 1)

        acc = tl.zeros([BLOCK_H], dtype=tl.float32)
        coef = tl.zeros([COEF], dtype=tl.float32)
        for entry in tl.range(start, end):
            window = tl.load(window_ptr + entry)
            acc += tl.load(
                w_ptr + window * stride_ww + offs_h * stride_wh,
                mask=live,
                other=0.0,
            ).to(tl.float32)
            slot_class = tl.load(class_ptr + entry)
            coef += tl.where(offs_c == slot_class, 1.0, 0.0)

        out_row = out_ptr + cell * stride_oc
        element = out_ptr.dtype.element_ty
        tl.store(out_row + offs_h * stride_oh, acc.to(element), mask=live)
        tl.store(out_row + (H + offs_c) * stride_oh, coef.to(element))


def _aggregate_reference(
    w: Tensor,
    dec_window: Tensor,
    dec_class: Tensor,
    dec_cell: Tensor,
    n_cells: int,
) -> Tensor:
    """The scatter formulation used by CPU, failed launches, and recompute.

    Accumulation is fp32 whatever ``w``'s dtype, matching the kernel's
    registers: a cell can carry up to eighteen entries, and rounding each
    partial sum to bf16 would cost more than the aggregation is worth.
    """
    h = w.shape[1]
    acc = torch.zeros(n_cells, h + COEF_WIDTH, dtype=torch.float32, device=w.device)
    acc[:, :h].index_add_(0, dec_cell, w.float().index_select(0, dec_window))
    acc[:, h : h + CLASS_SLOTS].index_add_(
        0, dec_cell, F.one_hot(dec_class, CLASS_SLOTS).float()
    )
    return acc.to(w.dtype)


def _shape_key(w: Tensor, dec_window: Tensor, dec_class: Tensor, n_cells: int):
    return (
        w.device.type,
        w.device.index,
        w.dtype,
        w.shape[1],
        tuple(w.stride()),
        tuple(dec_window.stride()),
        tuple(dec_class.stride()),
    )


def _validate(
    w: Tensor,
    dec_window: Tensor,
    dec_class: Tensor,
    dec_cell: Tensor,
    bg_cell: Tensor,
    bg_bucket: Tensor,
    n_cells: int,
) -> None:
    if w.ndim != 2:
        raise ValueError("w must have shape (N_w, H)")
    entries = dec_window.shape
    if dec_class.shape != entries or dec_cell.shape != entries:
        raise ValueError("dec_window, dec_class, and dec_cell must be one length")
    if bg_bucket.shape != bg_cell.shape:
        raise ValueError("bg_cell and bg_bucket must be one length")
    others = (dec_window, dec_class, dec_cell, bg_cell, bg_bucket)
    if any(x.device != w.device for x in others):
        raise ValueError("all decoder inputs must be on one device")


def _launch_triton(
    w: Tensor,
    dec_window: Tensor,
    dec_class: Tensor,
    dec_cell: Tensor,
    n_cells: int,
) -> Tensor:
    h = w.shape[1]
    out = torch.empty(n_cells, h + COEF_WIDTH, dtype=w.dtype, device=w.device)
    # Run boundaries: entry ``e`` belongs to cell ``dec_cell[e]`` and the
    # entries arrive in cell order, so the leftmost insertion point of each
    # cell index is that cell's run start and its successor's run end.
    queries = torch.arange(n_cells + 1, device=w.device, dtype=dec_cell.dtype)
    row_ptr = torch.searchsorted(dec_cell, queries)
    _aggregate_kernel[(n_cells,)](
        w,
        dec_window,
        dec_class,
        row_ptr,
        out,
        *w.stride(),
        *out.stride(),
        H=h,
        BLOCK_H=triton.next_power_of_2(h),
        COEF=COEF_WIDTH,
        num_warps=_NUM_WARPS,
    )
    return out


@torch.library.custom_op("mantisnet::decoder_aggregate", mutates_args=())
def _aggregate_op(
    w: Tensor,
    dec_window: Tensor,
    dec_class: Tensor,
    dec_cell: Tensor,
    bg_cell: Tensor,
    bg_bucket: Tensor,
    n_cells: int,
) -> Tensor:
    _validate(w, dec_window, dec_class, dec_cell, bg_cell, bg_bucket, n_cells)
    supported = (
        triton is not None
        and w.is_cuda
        and w.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and n_cells > 0
    )
    key = _shape_key(w, dec_window, dec_class, n_cells)
    if not supported or key in _FAILED_SHAPES:
        out = _aggregate_reference(w, dec_window, dec_class, dec_cell, n_cells)
    else:
        try:
            out = _launch_triton(w, dec_window, dec_class, dec_cell, n_cells)
        except Exception as exc:
            _FAILED_SHAPES[key] = f"{type(exc).__name__}: {exc}"
            warnings.warn(
                "decoder aggregation failed for "
                f"H={w.shape[1]}, dtype={w.dtype}; scattering instead for this "
                f"shape: {_FAILED_SHAPES[key]}",
                RuntimeWarning,
                stacklevel=2,
            )
            out = _aggregate_reference(w, dec_window, dec_class, dec_cell, n_cells)
    if bg_cell.numel():
        out[bg_cell, w.shape[1] + CLASS_SLOTS + bg_bucket] = 1
    return out


@_aggregate_op.register_fake
def _(
    w: Tensor,
    dec_window: Tensor,
    dec_class: Tensor,
    dec_cell: Tensor,
    bg_cell: Tensor,
    bg_bucket: Tensor,
    n_cells: int,
) -> Tensor:
    return w.new_empty(n_cells, w.shape[1] + COEF_WIDTH)


def _setup_context(ctx, inputs, output) -> None:
    w, dec_window, _dec_class, dec_cell, _bg_cell, _bg_bucket, _n_cells = inputs
    ctx.save_for_backward(w, dec_window, dec_cell)


def _backward(ctx, grad_out: Tensor):
    """Only ``w`` carries a gradient: the coefficient blocks count entries and
    mark buckets. Each window's gradient is the sum over the cells its rows
    reached — the transpose of the aggregation's gather."""
    w, dec_window, dec_cell = ctx.saved_tensors
    reached = grad_out[:, : w.shape[1]].index_select(0, dec_cell)
    grad_w = torch.zeros_like(w).index_add_(0, dec_window, reached)
    return grad_w, None, None, None, None, None, None


_aggregate_op.register_autograd(_backward, setup_context=_setup_context)


def aggregate(
    w: Tensor,
    dec_window: Tensor,
    dec_class: Tensor,
    dec_cell: Tensor,
    bg_cell: Tensor,
    bg_bucket: Tensor,
    n_cells: int,
) -> Tensor:
    """The (N_c, H + COEF_WIDTH) coefficient rows both cell heads read."""
    return _aggregate_op(
        w, dec_window, dec_class, dec_cell, bg_cell, bg_bucket, n_cells
    )


def head_matrix(
    proj: Tensor, e_class: Tensor, e_bg: Tensor, lin_a: Tensor
) -> Tensor:
    """One head's whole read of an aggregate row, as a (P_H, H + COEF_WIDTH)
    matrix: its first MLP layer folded through the decoder projection for the
    window block, and applied to each embedding row for the coefficient
    blocks."""
    pad = lin_a.new_zeros(lin_a.shape[0], COEF_WIDTH - CLASS_SLOTS - BG_SLOTS)
    return torch.cat([lin_a @ proj, lin_a @ e_class.t(), lin_a @ e_bg.t(), pad], dim=1)
