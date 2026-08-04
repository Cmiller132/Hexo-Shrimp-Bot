"""Fused incidence aggregation for the two trunk message-passing passes.

Both directions consume the builder's one window-major incidence table.  The
torch formulation is deliberately literal: gather a projected entity row and
its three-way slot-class row for every incidence, add them, then ``index_add_``
the messages into their destinations.  It is the CPU implementation, the
autograd reference, and the fallback for unsupported CUDA signatures.  The
window pass stores in the projected value's dtype: its consuming autocast
linear immediately performs the same cast, and a direct segment reduction can
fold it without allocating an intermediate fp32 table.  The contended stone
scatter retains the reference's fp32 output so small differences in atomic
arrival order are not magnified by an early low-precision rounding boundary.

The CUDA window kernel uses the table's stronger ordering contract.  Each live
window's one-to-six entries are contiguous and slot-ascending, so a program at
a run head reduces directly into that window without materializing the
``(E, H)`` message table.  The reverse pass deliberately remains the literal
torch scatter: Inductor fuses it with the surrounding compiled block more
effectively than an explicit kernel, even though the explicit kernel is faster
in isolation.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.library import triton_op, wrap_triton

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


# A live window has at most six occupied slots.  Non-terminal model inputs have
# at most five, but covering all six keeps the kernel correct at the structural
# boundary without changing its fixed launch geometry.
_WINDOW_LEN = 6
_WINDOW_WARPS = 1

if triton is not None:

    @triton.jit(do_not_specialize_on_alignment=("n_entries",))
    def _aggregate_to_windows_kernel(
        values_ptr,
        class_ptr,
        stone_ptr,
        window_ptr,
        inc_class_ptr,
        out_ptr,
        stride_vr: tl.constexpr,
        stride_vh: tl.constexpr,
        stride_cr: tl.constexpr,
        stride_ch: tl.constexpr,
        stride_is: tl.constexpr,
        stride_iw: tl.constexpr,
        stride_ic: tl.constexpr,
        stride_or: tl.constexpr,
        stride_oh: tl.constexpr,
        n_entries,
        H: tl.constexpr,
        BLOCK_H: tl.constexpr,
        WINDOW_LEN: tl.constexpr,
    ):
        entry = tl.program_id(0)
        window = tl.load(window_ptr + entry * stride_iw)
        previous = tl.load(
            window_ptr + (entry - 1) * stride_iw,
            mask=entry > 0,
            other=-1,
        )
        run_head = (entry == 0) | (previous != window)

        offs_h = tl.arange(0, BLOCK_H)
        live_h = offs_h < H
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)

        for relative in tl.static_range(0, WINDOW_LEN):
            current = entry + relative
            in_bounds = current < n_entries
            current_window = tl.load(
                window_ptr + current * stride_iw,
                mask=run_head & in_bounds,
                other=-1,
            )
            same_window = run_head & in_bounds & (current_window == window)
            stone = tl.load(
                stone_ptr + current * stride_is,
                mask=same_window,
                other=0,
            )
            slot_class = tl.load(
                inc_class_ptr + current * stride_ic,
                mask=same_window,
                other=0,
            )
            value = tl.load(
                values_ptr + stone * stride_vr + offs_h * stride_vh,
                mask=same_window & live_h,
                other=0.0,
            ).to(tl.float32)
            class_value = tl.load(
                class_ptr + slot_class * stride_cr + offs_h * stride_ch,
                mask=same_window & live_h,
                other=0.0,
            ).to(tl.float32)
            acc += value + class_value

        tl.store(
            out_ptr + window * stride_or + offs_h * stride_oh,
            acc,
            mask=run_head & live_h,
        )


def _aggregate_reference(
    values: Tensor,
    class_weight: Tensor,
    source_index: Tensor,
    dest_index: Tensor,
    inc_class: Tensor,
    n_dest: int,
) -> Tensor:
    """Literal gather/add/scatter formulation for either direction."""
    messages = values.index_select(0, source_index) + class_weight.index_select(
        0, inc_class
    )
    return messages.new_zeros((n_dest, values.shape[1])).index_add_(
        0, dest_index, messages
    )


def _window_reference_output(
    values: Tensor,
    class_weight: Tensor,
    source_index: Tensor,
    dest_index: Tensor,
    inc_class: Tensor,
    n_dest: int,
) -> Tensor:
    out = _aggregate_reference(
        values, class_weight, source_index, dest_index, inc_class, n_dest
    )
    return out.to(values.dtype)


def _validate(
    values: Tensor,
    class_weight: Tensor,
    inc_stone: Tensor,
    inc_window: Tensor,
    inc_class: Tensor,
    n_dest: int,
) -> None:
    if values.ndim != 2:
        raise ValueError("values must have shape (N, H)")
    # The kernels are generic in the table height: three rows for a stock
    # model's folded classes, OCC_CLASSES for a joint_incidence one (§4.3).
    if class_weight.ndim != 2 or class_weight.shape[1] != values.shape[1]:
        raise ValueError("class_weight must have shape (classes, H)")
    entries = inc_stone.shape
    if inc_stone.ndim != 1 or inc_window.shape != entries or inc_class.shape != entries:
        raise ValueError("inc_stone, inc_window, and inc_class must be one length")
    indices = (inc_stone, inc_window, inc_class)
    if any(index.dtype != torch.int64 for index in indices):
        raise ValueError("incidence tensors must have dtype int64")
    if any(index.device != values.device for index in indices):
        raise ValueError("all incidence inputs must be on the values device")
    if class_weight.device != values.device:
        raise ValueError("values and class_weight must be on one device")
    if not values.is_floating_point() or not class_weight.is_floating_point():
        raise ValueError("values and class_weight must have floating-point dtype")
    if n_dest < 0:
        raise ValueError(f"destination count must be nonnegative, got {n_dest}")


def _supported(values: Tensor, class_weight: Tensor, n_dest: int) -> bool:
    promoted = torch.promote_types(values.dtype, class_weight.dtype)
    return (
        triton is not None
        and values.is_cuda
        and promoted == torch.float32
        and values.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and class_weight.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and values.shape[1] > 0
        and n_dest > 0
    )


def _launch_to_windows(
    values: Tensor,
    class_weight: Tensor,
    inc_stone: Tensor,
    inc_window: Tensor,
    inc_class: Tensor,
    n_windows: int,
) -> Tensor:
    h = values.shape[1]
    n_entries = inc_stone.shape[0]
    # Every live window owns a nonempty incidence run, so the run-head programs
    # overwrite every row.  Avoiding a full zero fill matters at N_w x H scale.
    out = torch.empty(n_windows, h, dtype=values.dtype, device=values.device)
    wrap_triton(_aggregate_to_windows_kernel)[(n_entries,)](
        values,
        class_weight,
        inc_stone,
        inc_window,
        inc_class,
        out,
        *values.stride(),
        *class_weight.stride(),
        *inc_stone.stride(),
        *inc_window.stride(),
        *inc_class.stride(),
        *out.stride(),
        n_entries,
        H=h,
        BLOCK_H=triton.next_power_of_2(h),
        WINDOW_LEN=_WINDOW_LEN,
        num_warps=_WINDOW_WARPS,
    )
    return out


def _dispatch_to_windows(
    values: Tensor,
    class_weight: Tensor,
    inc_stone: Tensor,
    inc_window: Tensor,
    inc_class: Tensor,
    n_dest: int,
) -> Tensor:
    _validate(values, class_weight, inc_stone, inc_window, inc_class, n_dest)
    # An empty launch is invalid in Triton.  The reference also covers every
    # non-CUDA or unsupported signature.
    if not _supported(values, class_weight, n_dest) or inc_stone.numel() == 0:
        return _window_reference_output(
            values,
            class_weight,
            inc_stone,
            inc_window,
            inc_class,
            n_dest,
        )

    if _aggregate_to_windows_op is None:  # pragma: no cover - implied above
        raise AssertionError("a supported signature requires Triton")
    return _aggregate_to_windows_op(
        values,
        class_weight,
        inc_stone,
        inc_window,
        inc_class,
        n_dest,
    )


if triton is not None:

    @triton_op("mantisnet::aggregate_to_windows", mutates_args={})
    def _aggregate_to_windows_op(
        values: Tensor,
        class_weight: Tensor,
        inc_stone: Tensor,
        inc_window: Tensor,
        inc_class: Tensor,
        n_windows: int,
    ) -> Tensor:
        return _launch_to_windows(
            values,
            class_weight,
            inc_stone,
            inc_window,
            inc_class,
            n_windows,
        )

else:  # pragma: no cover - exercised only by installations without Triton
    _aggregate_to_windows_op = None


def _setup_context(ctx, inputs, output) -> None:
    del output
    values, class_weight, inc_stone, inc_window, inc_class, _n_dest = inputs
    ctx.save_for_backward(
        values, class_weight, inc_stone, inc_window, inc_class
    )


def _backward_to_windows(ctx, grad_out: Tensor):
    values, class_weight, inc_stone, inc_window, inc_class = ctx.saved_tensors
    reached = grad_out.index_select(0, inc_window)
    grad_values = torch.zeros_like(values).index_add_(
        0, inc_stone, reached.to(values.dtype)
    )
    grad_class = torch.zeros_like(class_weight).index_add_(
        0, inc_class, reached.to(class_weight.dtype)
    )
    return grad_values, grad_class, None, None, None, None


if _aggregate_to_windows_op is not None:
    _aggregate_to_windows_op.register_autograd(
        _backward_to_windows, setup_context=_setup_context
    )


def aggregate_to_windows(
    values: Tensor,
    class_weight: Tensor,
    inc_stone: Tensor,
    inc_window: Tensor,
    inc_class: Tensor,
    n_windows: int,
) -> Tensor:
    """Sum projected stone and slot-class rows into live-window rows."""
    return _dispatch_to_windows(
        values,
        class_weight,
        inc_stone,
        inc_window,
        inc_class,
        n_windows,
    )


def aggregate_to_stones(
    values: Tensor,
    class_weight: Tensor,
    inc_stone: Tensor,
    inc_window: Tensor,
    inc_class: Tensor,
    n_stones: int,
) -> Tensor:
    """Sum projected window and slot-class rows into stone rows."""
    _validate(values, class_weight, inc_stone, inc_window, inc_class, n_stones)
    return _aggregate_reference(
        values,
        class_weight,
        inc_window,
        inc_stone,
        inc_class,
        n_stones,
    )
