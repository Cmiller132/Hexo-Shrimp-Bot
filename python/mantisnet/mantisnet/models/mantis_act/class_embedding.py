"""Deterministic lookup and backward for §19.2's two relation tables.

The post-placement row relation is

``E_post1[action_post1_class] + E_status[action_pre_status]``.

The forward gathers and adds both rows in one kernel.  The backward consumes
the class-major CSR views built by :mod:`plans`: each class owns one contiguous
run of original row ids, so one program sums that run in row order.  No sort or
atomic scatter occurs in the model step, and empty classes produce exact zero
gradients.  CPU and unsupported signatures use the literal torch formulation
held against the kernel by the §36 parity tests.
"""

from __future__ import annotations

import warnings

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only without Triton
    triton = None
    tl = None


_BLOCK_M = 128
_MAX_BLOCK_D = 256
_WARPS = 4

_FAILED_SHAPES: dict[tuple[object, ...], str] = {}
_FAILED_BACKWARD_SHAPES: dict[tuple[object, ...], str] = {}


def _accumulate_dtype(tensor: Tensor) -> torch.dtype:
    return torch.promote_types(tensor.dtype, torch.float32)


def _reference(
    post_weight: Tensor,
    status_weight: Tensor,
    post_index: Tensor,
    status_index: Tensor,
) -> Tensor:
    """The two embedding lookups and their sum, stated literally."""
    return post_weight.index_select(0, post_index.long()) + status_weight.index_select(
        0, status_index.long()
    )


def _class_sum_reference(
    grad_out: Tensor, ptr: Tensor, rows: Tensor, classes: int, dtype: torch.dtype
) -> Tensor:
    """Ordered class reductions from a class-major CSR view.

    ``torch.segment_reduce`` is the reference counterpart of the kernel's
    contiguous walk.  Unlike ``EmbeddingBackward``/``index_add_`` it neither
    sorts in the step nor scatters duplicate classes through atomics.
    """
    ordered = grad_out.index_select(0, rows.long()).to(_accumulate_dtype(grad_out))
    lengths = (ptr[1:] - ptr[:-1]).long()
    reduced = torch.segment_reduce(ordered, "sum", lengths=lengths, initial=0.0)
    if reduced.shape[0] != classes:
        raise ValueError(
            f"class CSR produced {reduced.shape[0]} rows for a {classes}-row table"
        )
    return reduced.to(dtype)


def _reference_backward(
    post_weight: Tensor,
    status_weight: Tensor,
    post_ptr: Tensor,
    post_rows: Tensor,
    status_ptr: Tensor,
    status_rows: Tensor,
    grad_out: Tensor,
) -> tuple[Tensor, Tensor]:
    """The lookup gradients, reduced through the supplied stable CSR views."""
    return (
        _class_sum_reference(
            grad_out, post_ptr, post_rows, int(post_weight.shape[0]), post_weight.dtype
        ),
        _class_sum_reference(
            grad_out,
            status_ptr,
            status_rows,
            int(status_weight.shape[0]),
            status_weight.dtype,
        ),
    )


if triton is not None:

    @triton.jit
    def _pair_lookup_kernel(
        post_weight_ptr,
        status_weight_ptr,
        post_index_ptr,
        status_index_ptr,
        out_ptr,
        M,
        D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        columns = tl.arange(0, BLOCK_D)
        live = (rows[:, None] < M) & (columns[None, :] < D)
        post = tl.load(post_index_ptr + rows, mask=rows < M, other=0)
        status = tl.load(status_index_ptr + rows, mask=rows < M, other=0)
        value = tl.load(
            post_weight_ptr + post[:, None] * D + columns[None, :],
            mask=live,
            other=0.0,
        )
        value += tl.load(
            status_weight_ptr + status[:, None] * D + columns[None, :],
            mask=live,
            other=0.0,
        )
        tl.store(out_ptr + rows[:, None] * D + columns[None, :], value, mask=live)

    @triton.jit
    def _class_sum_kernel(
        grad_ptr,
        ptr,
        class_rows,
        out_ptr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        # One program owns one class. Its loop follows the stable original-row
        # order encoded by the CSR, so both stages of the reduction have a
        # fixed association and no destination is written by two programs.
        cls = tl.program_id(0)
        columns = tl.arange(0, BLOCK_D)
        live_d = columns < D
        start = tl.load(ptr + cls)
        end = tl.load(ptr + cls + 1)
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for entry in tl.range(start, end):
            row = tl.load(class_rows + entry)
            acc += tl.load(
                grad_ptr + row * D + columns, mask=live_d, other=0.0
            ).to(tl.float32)
        tl.store(out_ptr + cls * D + columns, acc, mask=live_d)


def _validate(
    post_weight: Tensor,
    status_weight: Tensor,
    post_index: Tensor,
    status_index: Tensor,
    post_ptr: Tensor,
    post_rows: Tensor,
    status_ptr: Tensor,
    status_rows: Tensor,
) -> None:
    if post_weight.ndim != 2 or status_weight.ndim != 2:
        raise ValueError("relation weights must both be (classes, width)")
    if post_weight.shape[1] != status_weight.shape[1]:
        raise ValueError(
            f"post1 is {post_weight.shape[1]} wide against pre_status' "
            f"{status_weight.shape[1]}"
        )
    if post_weight.dtype != status_weight.dtype:
        raise ValueError(
            f"relation weights must share a dtype, got {post_weight.dtype} and "
            f"{status_weight.dtype}"
        )
    if not post_weight.is_floating_point():
        raise ValueError(f"relation weights must be floating point, got {post_weight.dtype}")

    for name, value in (
        ("post_index", post_index),
        ("status_index", status_index),
        ("post_ptr", post_ptr),
        ("post_rows", post_rows),
        ("status_ptr", status_ptr),
        ("status_rows", status_rows),
    ):
        if value.ndim != 1:
            raise ValueError(f"{name} must be 1-D, got {tuple(value.shape)}")
        if value.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"{name} must be int32 or int64, got {value.dtype}")

    rows = int(post_index.numel())
    if status_index.numel() != rows:
        raise ValueError(
            f"post_index has {rows} rows against status_index's {status_index.numel()}"
        )
    if post_rows.numel() != rows or status_rows.numel() != rows:
        raise ValueError(
            "each class-major row permutation must cover every flattened action row"
        )
    if post_ptr.numel() != post_weight.shape[0] + 1:
        raise ValueError(
            f"post_ptr must have {post_weight.shape[0] + 1} entries, got "
            f"{post_ptr.numel()}"
        )
    if status_ptr.numel() != status_weight.shape[0] + 1:
        raise ValueError(
            f"status_ptr must have {status_weight.shape[0] + 1} entries, got "
            f"{status_ptr.numel()}"
        )
    tensors = (
        post_weight,
        status_weight,
        post_index,
        status_index,
        post_ptr,
        post_rows,
        status_ptr,
        status_rows,
    )
    if any(t.device != post_weight.device for t in tensors):
        raise ValueError("every class-pair embedding input must be on one device")


def _supported(sample: Tensor, width: int, rows: int) -> bool:
    return (
        triton is not None
        and sample.is_cuda
        and sample.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and 0 < width
        and triton.next_power_of_2(width) <= _MAX_BLOCK_D
        and rows > 0
    )


def _shape_key(sample: Tensor, *rest: object) -> tuple[object, ...]:
    return (sample.device.type, sample.device.index, sample.dtype, *rest)


def _launch_forward(
    post_weight: Tensor,
    status_weight: Tensor,
    post_index: Tensor,
    status_index: Tensor,
) -> Tensor:
    rows = int(post_index.numel())
    width = int(post_weight.shape[1])
    out = torch.empty(rows, width, dtype=post_weight.dtype, device=post_weight.device)
    _pair_lookup_kernel[(triton.cdiv(rows, _BLOCK_M),)](
        post_weight,
        status_weight,
        post_index,
        status_index,
        out,
        rows,
        D=width,
        BLOCK_M=_BLOCK_M,
        BLOCK_D=triton.next_power_of_2(width),
        num_warps=_WARPS,
    )
    return out


def _launch_backward(
    post_weight: Tensor,
    status_weight: Tensor,
    post_ptr: Tensor,
    post_rows: Tensor,
    status_ptr: Tensor,
    status_rows: Tensor,
    grad_out: Tensor,
) -> tuple[Tensor, Tensor]:
    width = int(post_weight.shape[1])
    block_d = triton.next_power_of_2(width)
    d_post = torch.empty_like(post_weight)
    d_status = torch.empty_like(status_weight)
    _class_sum_kernel[(int(post_weight.shape[0]),)](
        grad_out,
        post_ptr,
        post_rows,
        d_post,
        D=width,
        BLOCK_D=block_d,
        num_warps=_WARPS,
    )
    _class_sum_kernel[(int(status_weight.shape[0]),)](
        grad_out,
        status_ptr,
        status_rows,
        d_status,
        D=width,
        BLOCK_D=block_d,
        num_warps=_WARPS,
    )
    return d_post, d_status


@torch.library.custom_op("mantisnet::act_class_pair_embedding", mutates_args=())
def _class_pair_op(
    post_weight: Tensor,
    status_weight: Tensor,
    post_index: Tensor,
    status_index: Tensor,
    post_ptr: Tensor,
    post_rows: Tensor,
    status_ptr: Tensor,
    status_rows: Tensor,
) -> Tensor:
    _validate(
        post_weight,
        status_weight,
        post_index,
        status_index,
        post_ptr,
        post_rows,
        status_ptr,
        status_rows,
    )
    reference = lambda: _reference(  # noqa: E731
        post_weight, status_weight, post_index, status_index
    )
    rows, width = int(post_index.numel()), int(post_weight.shape[1])
    if not _supported(post_weight, width, rows):
        return reference()
    key = _shape_key(
        post_weight,
        width,
        int(post_weight.shape[0]),
        int(status_weight.shape[0]),
    )
    if key in _FAILED_SHAPES:
        return reference()
    try:
        return _launch_forward(post_weight, status_weight, post_index, status_index)
    except Exception as exc:
        _FAILED_SHAPES[key] = f"{type(exc).__name__}: {exc}"
        warnings.warn(
            f"fused class-pair lookup failed for width={width}; using the two "
            f"torch lookups for this shape: {_FAILED_SHAPES[key]}",
            RuntimeWarning,
            stacklevel=2,
        )
        return reference()


@_class_pair_op.register_fake
def _(
    post_weight: Tensor,
    status_weight: Tensor,
    post_index: Tensor,
    status_index: Tensor,
    post_ptr: Tensor,
    post_rows: Tensor,
    status_ptr: Tensor,
    status_rows: Tensor,
) -> Tensor:
    return post_weight.new_empty((post_index.numel(), post_weight.shape[1]))


@torch.library.custom_op(
    "mantisnet::act_class_pair_embedding_backward", mutates_args=()
)
def _class_pair_backward_op(
    post_weight: Tensor,
    status_weight: Tensor,
    post_ptr: Tensor,
    post_rows: Tensor,
    status_ptr: Tensor,
    status_rows: Tensor,
    grad_out: Tensor,
) -> tuple[Tensor, Tensor]:
    reference = lambda: _reference_backward(  # noqa: E731
        post_weight,
        status_weight,
        post_ptr,
        post_rows,
        status_ptr,
        status_rows,
        grad_out,
    )
    rows, width = int(grad_out.shape[0]), int(post_weight.shape[1])
    if not _supported(post_weight, width, rows):
        return reference()
    key = _shape_key(
        post_weight,
        width,
        int(post_weight.shape[0]),
        int(status_weight.shape[0]),
    )
    if key in _FAILED_BACKWARD_SHAPES:
        return reference()
    try:
        return _launch_backward(
            post_weight,
            status_weight,
            post_ptr,
            post_rows,
            status_ptr,
            status_rows,
            grad_out.contiguous(),
        )
    except Exception as exc:
        _FAILED_BACKWARD_SHAPES[key] = f"{type(exc).__name__}: {exc}"
        warnings.warn(
            f"fused class-pair lookup backward failed for width={width}; using "
            f"ordered torch segment reductions for this shape: "
            f"{_FAILED_BACKWARD_SHAPES[key]}",
            RuntimeWarning,
            stacklevel=2,
        )
        return reference()


@_class_pair_backward_op.register_fake
def _(
    post_weight: Tensor,
    status_weight: Tensor,
    post_ptr: Tensor,
    post_rows: Tensor,
    status_ptr: Tensor,
    status_rows: Tensor,
    grad_out: Tensor,
) -> tuple[Tensor, Tensor]:
    return torch.empty_like(post_weight), torch.empty_like(status_weight)


def _setup_context(ctx, inputs, output) -> None:
    ctx.save_for_backward(
        inputs[0], inputs[1], inputs[4], inputs[5], inputs[6], inputs[7]
    )


def _dispatch_backward(ctx, grad_out: Tensor):
    d_post, d_status = _class_pair_backward_op(*ctx.saved_tensors, grad_out)
    return d_post, d_status, None, None, None, None, None, None


_class_pair_op.register_autograd(_dispatch_backward, setup_context=_setup_context)


def class_pair_embedding(
    post_weight: Tensor,
    status_weight: Tensor,
    post_index: Tensor,
    status_index: Tensor,
    post_ptr: Tensor,
    post_rows: Tensor,
    status_ptr: Tensor,
    status_rows: Tensor,
) -> Tensor:
    """Return both relation lookups summed, with ordered table gradients.

    ``post_index`` and ``status_index`` are the flattened ``[legal, 3, 6]``
    class grids.  Each ``ptr``/``rows`` pair is the matching class-major CSR
    plan: ``rows[ptr[c]:ptr[c+1]]`` names every flattened row of class ``c`` in
    original row order.
    """
    return _class_pair_op(
        post_weight.contiguous(),
        status_weight.contiguous(),
        post_index.contiguous(),
        status_index.contiguous(),
        post_ptr.contiguous(),
        post_rows.contiguous(),
        status_ptr.contiguous(),
        status_rows.contiguous(),
    )


__all__ = ["class_pair_embedding"]
