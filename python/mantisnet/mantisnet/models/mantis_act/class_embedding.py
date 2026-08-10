"""Deterministic lookup and backward for §19.2's two relation tables.

The post-placement row relation is

``E_post1[action_post1_class] + E_status[action_pre_status]``.

The forward is the literal pair of table lookups, left visible to the enclosing
``torch.compile`` region.  The backward consumes the class-major CSR views and
their compact fixed-size block partition built by :mod:`plans`.  Grid-parallel
programs reduce contiguous row blocks first; one owner per class then combines
those partials in their fixed order.  No sort or atomic scatter occurs in the
model step, and empty classes produce exact zero gradients.  CPU and unsupported
signatures use the literal two-stage ordered reduction held against the kernel
by the §36 parity tests.
"""

from __future__ import annotations

import warnings

import torch
from torch import Tensor

from .ordered_reductions import ordered_two_stage_segment_sum
from .plans import CLASS_REDUCTION_BLOCK_ROWS

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only without Triton
    triton = None
    tl = None


_MAX_BLOCK_D = 256
_WARPS = 4

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
    grad_out: Tensor,
    ptr: Tensor,
    rows: Tensor,
    block_ptr: Tensor,
    block_lengths: Tensor,
    classes: int,
    dtype: torch.dtype,
) -> Tensor:
    """Two-stage ordered class reductions from a blocked class-major CSR.

    The first segment reduction forms the same contiguous block partials as the
    grid-parallel kernel; the second combines each class's consecutive blocks.
    Unlike ``EmbeddingBackward``/``index_add_`` neither stage sorts in the step
    nor scatters duplicate classes through atomics.
    """
    ordered = grad_out.index_select(0, rows.long()).to(_accumulate_dtype(grad_out))
    reduced = ordered_two_stage_segment_sum(ordered, block_ptr, block_lengths)
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
    post_block_ptr: Tensor,
    post_block_starts: Tensor,
    post_block_lengths: Tensor,
    status_block_ptr: Tensor,
    status_block_starts: Tensor,
    status_block_lengths: Tensor,
    grad_out: Tensor,
) -> tuple[Tensor, Tensor]:
    """The lookup gradients, reduced through the supplied stable CSR views."""
    return (
        _class_sum_reference(
            grad_out,
            post_ptr,
            post_rows,
            post_block_ptr,
            post_block_lengths,
            int(post_weight.shape[0]),
            post_weight.dtype,
        ),
        _class_sum_reference(
            grad_out,
            status_ptr,
            status_rows,
            status_block_ptr,
            status_block_lengths,
            int(status_weight.shape[0]),
            status_weight.dtype,
        ),
    )


if triton is not None:

    @triton.jit
    def _class_block_sum_kernel(
        grad_ptr,
        class_rows,
        block_starts,
        block_lengths,
        partial_ptr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        # One program owns one compact slice of one class run. Its scalar row
        # loop follows the stable original-row order exactly; many such slices
        # fill the GPU even for the four-class pre-status table.
        block = tl.program_id(0)
        columns = tl.arange(0, BLOCK_D)
        live_d = columns < D
        start = tl.load(block_starts + block)
        length = tl.load(block_lengths + block)
        end = tl.minimum(length, BLOCK_ROWS)
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for offset in tl.range(0, end):
            row = tl.load(class_rows + start + offset)
            acc += tl.load(
                grad_ptr + row * D + columns,
                mask=live_d,
                other=0.0,
            ).to(tl.float32)
        tl.store(partial_ptr + block * D + columns, acc, mask=live_d)

    @triton.jit
    def _class_block_combine_kernel(
        partial_ptr,
        block_ptr,
        out_ptr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        # One program owns one final class row and visits its contiguous block
        # partials in ascending order. Empty classes retain the exact zero
        # identity and no output has multiple writers.
        cls = tl.program_id(0)
        columns = tl.arange(0, BLOCK_D)
        live_d = columns < D
        start = tl.load(block_ptr + cls)
        end = tl.load(block_ptr + cls + 1)
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for block in tl.range(start, end):
            acc += tl.load(
                partial_ptr + block * D + columns, mask=live_d, other=0.0
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
    post_block_ptr: Tensor,
    post_block_starts: Tensor,
    post_block_lengths: Tensor,
    status_block_ptr: Tensor,
    status_block_starts: Tensor,
    status_block_lengths: Tensor,
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
        ("post_block_ptr", post_block_ptr),
        ("post_block_starts", post_block_starts),
        ("post_block_lengths", post_block_lengths),
        ("status_block_ptr", status_block_ptr),
        ("status_block_starts", status_block_starts),
        ("status_block_lengths", status_block_lengths),
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
    if post_block_ptr.numel() != post_weight.shape[0] + 1:
        raise ValueError(
            f"post_block_ptr must have {post_weight.shape[0] + 1} entries, got "
            f"{post_block_ptr.numel()}"
        )
    if status_block_ptr.numel() != status_weight.shape[0] + 1:
        raise ValueError(
            f"status_block_ptr must have {status_weight.shape[0] + 1} entries, got "
            f"{status_block_ptr.numel()}"
        )
    if post_block_starts.numel() != post_block_lengths.numel():
        raise ValueError("post block starts and lengths must have equal size")
    if status_block_starts.numel() != status_block_lengths.numel():
        raise ValueError("status block starts and lengths must have equal size")
    tensors = (
        post_weight,
        status_weight,
        post_index,
        status_index,
        post_ptr,
        post_rows,
        status_ptr,
        status_rows,
        post_block_ptr,
        post_block_starts,
        post_block_lengths,
        status_block_ptr,
        status_block_starts,
        status_block_lengths,
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


def _launch_backward(
    post_weight: Tensor,
    status_weight: Tensor,
    post_ptr: Tensor,
    post_rows: Tensor,
    status_ptr: Tensor,
    status_rows: Tensor,
    post_block_ptr: Tensor,
    post_block_starts: Tensor,
    post_block_lengths: Tensor,
    status_block_ptr: Tensor,
    status_block_starts: Tensor,
    status_block_lengths: Tensor,
    grad_out: Tensor,
) -> tuple[Tensor, Tensor]:
    width = int(post_weight.shape[1])
    block_d = triton.next_power_of_2(width)

    def table_gradient(
        weight: Tensor,
        rows: Tensor,
        block_ptr: Tensor,
        block_starts: Tensor,
        block_lengths: Tensor,
    ) -> Tensor:
        block_count = int(block_starts.numel())
        if block_count < 1:
            raise ValueError("a nonempty class row grid needs at least one reduction block")
        partial = torch.empty(
            block_count, width, dtype=torch.float32, device=grad_out.device
        )
        _class_block_sum_kernel[(block_count,)](
            grad_out,
            rows,
            block_starts,
            block_lengths,
            partial,
            D=width,
            BLOCK_D=block_d,
            BLOCK_ROWS=CLASS_REDUCTION_BLOCK_ROWS,
            num_warps=_WARPS,
        )
        result = torch.empty_like(weight)
        _class_block_combine_kernel[(int(weight.shape[0]),)](
            partial,
            block_ptr,
            result,
            D=width,
            BLOCK_D=block_d,
            num_warps=_WARPS,
        )
        return result

    d_post = table_gradient(
        post_weight, post_rows, post_block_ptr, post_block_starts, post_block_lengths
    )
    d_status = table_gradient(
        status_weight,
        status_rows,
        status_block_ptr,
        status_block_starts,
        status_block_lengths,
    )
    return d_post, d_status


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
    post_block_ptr: Tensor,
    post_block_starts: Tensor,
    post_block_lengths: Tensor,
    status_block_ptr: Tensor,
    status_block_starts: Tensor,
    status_block_lengths: Tensor,
    grad_out: Tensor,
) -> tuple[Tensor, Tensor]:
    reference = lambda: _reference_backward(  # noqa: E731
        post_weight,
        status_weight,
        post_ptr,
        post_rows,
        status_ptr,
        status_rows,
        post_block_ptr,
        post_block_starts,
        post_block_lengths,
        status_block_ptr,
        status_block_starts,
        status_block_lengths,
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
            post_block_ptr,
            post_block_starts,
            post_block_lengths,
            status_block_ptr,
            status_block_starts,
            status_block_lengths,
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
    post_block_ptr: Tensor,
    post_block_starts: Tensor,
    post_block_lengths: Tensor,
    status_block_ptr: Tensor,
    status_block_starts: Tensor,
    status_block_lengths: Tensor,
    grad_out: Tensor,
) -> tuple[Tensor, Tensor]:
    return torch.empty_like(post_weight), torch.empty_like(status_weight)


class _ClassPairEmbedding(torch.autograd.Function):
    """Traceable lookup forward with the ordered reduction as its only op wall."""

    @staticmethod
    def forward(
        ctx,
        post_weight: Tensor,
        status_weight: Tensor,
        post_index: Tensor,
        status_index: Tensor,
        post_ptr: Tensor,
        post_rows: Tensor,
        status_ptr: Tensor,
        status_rows: Tensor,
        post_block_ptr: Tensor,
        post_block_starts: Tensor,
        post_block_lengths: Tensor,
        status_block_ptr: Tensor,
        status_block_starts: Tensor,
        status_block_lengths: Tensor,
    ) -> Tensor:
        ctx.save_for_backward(
            post_weight,
            status_weight,
            post_ptr,
            post_rows,
            status_ptr,
            status_rows,
            post_block_ptr,
            post_block_starts,
            post_block_lengths,
            status_block_ptr,
            status_block_starts,
            status_block_lengths,
        )
        return _reference(post_weight, status_weight, post_index, status_index)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        d_post, d_status = _class_pair_backward_op(*ctx.saved_tensors, grad_out)
        return d_post, d_status, *(None for _ in range(12))


def class_pair_embedding(
    post_weight: Tensor,
    status_weight: Tensor,
    post_index: Tensor,
    status_index: Tensor,
    post_ptr: Tensor,
    post_rows: Tensor,
    status_ptr: Tensor,
    status_rows: Tensor,
    post_block_ptr: Tensor,
    post_block_starts: Tensor,
    post_block_lengths: Tensor,
    status_block_ptr: Tensor,
    status_block_starts: Tensor,
    status_block_lengths: Tensor,
) -> Tensor:
    """Return both relation lookups summed, with ordered table gradients.

    ``post_index`` and ``status_index`` are the flattened ``[legal, 3, 6]``
    class grids.  Each ``ptr``/``rows`` pair is the matching class-major CSR
    plan: ``rows[ptr[c]:ptr[c+1]]`` names every flattened row of class ``c`` in
    original row order.  Each appended ``block_ptr``/``block_starts``/
    ``block_lengths`` triple is the compact, class-major partition used by the
    two-stage deterministic table-gradient reduction.
    """
    inputs = (
        post_weight.contiguous(),
        status_weight.contiguous(),
        post_index.contiguous(),
        status_index.contiguous(),
        post_ptr.contiguous(),
        post_rows.contiguous(),
        status_ptr.contiguous(),
        status_rows.contiguous(),
        post_block_ptr.contiguous(),
        post_block_starts.contiguous(),
        post_block_lengths.contiguous(),
        status_block_ptr.contiguous(),
        status_block_starts.contiguous(),
        status_block_lengths.contiguous(),
    )
    _validate(*inputs)
    return _ClassPairEmbedding.apply(*inputs)


__all__ = ["class_pair_embedding"]
