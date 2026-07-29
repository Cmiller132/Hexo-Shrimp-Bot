"""Fused coordinate-biased attention for the stone rows."""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


_PAD_BIAS = -3.0e4

# Fixed launch geometry keeps symbolic shape changes out of Triton's tuning cache.
_BLOCK_M = 64
_BLOCK_N = 64
_NUM_WARPS = 4
_NUM_STAGES = 3

_FAILED_SHAPES: dict[tuple[object, ...], str] = {}


if triton is not None:

    @triton.jit
    def _fused_attention_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        coords_ptr,
        seq_lens_ptr,
        table_ptr,
        out_ptr,
        stride_qp,
        stride_qh,
        stride_qt,
        stride_qd,
        stride_kp,
        stride_kh,
        stride_kt,
        stride_kd,
        stride_vp,
        stride_vh,
        stride_vt,
        stride_vd,
        stride_cp,
        stride_ct,
        stride_cc,
        stride_lp,
        stride_th,
        stride_tb,
        stride_op,
        stride_oh,
        stride_ot,
        stride_od,
        n_heads,
        n_ctx,
        sm_scale,
        D_MAX: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        start_m = tl.program_id(0) * BLOCK_M
        off_ph = tl.program_id(1)
        off_p = off_ph // n_heads
        off_h = off_ph - off_p * n_heads

        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)
        live_len = tl.load(seq_lens_ptr + off_p * stride_lp)
        live_len = tl.minimum(tl.maximum(live_len, 0), n_ctx)
        out_ptrs = (
            out_ptr
            + off_p * stride_op
            + off_h * stride_oh
            + offs_m[:, None] * stride_ot
            + offs_d[None, :] * stride_od
        )

        # A row whose query tile starts after its live prefix performs no key
        # work. The zeros also make padding deterministic for direct op users.
        if start_m >= live_len:
            zeros = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
            tl.store(out_ptrs, zeros, mask=offs_m[:, None] < n_ctx)
        else:
            q_ptrs = (
                q_ptr
                + off_p * stride_qp
                + off_h * stride_qh
                + offs_m[:, None] * stride_qt
                + offs_d[None, :] * stride_qd
            )
            q_live = offs_m < live_len
            q = tl.load(q_ptrs, mask=q_live[:, None], other=0.0)

            coords_base = coords_ptr + off_p * stride_cp
            q_q = tl.load(
                coords_base + offs_m * stride_ct,
                mask=offs_m < n_ctx,
                other=0,
            )
            q_r = tl.load(
                coords_base + offs_m * stride_ct + stride_cc,
                mask=offs_m < n_ctx,
                other=0,
            )

            m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
            l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
            acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

            # The dynamic upper bound is the row's live prefix, so no complete
            # key tile beyond it is loaded or multiplied.
            for start_n in tl.range(0, live_len, BLOCK_N):
                start_n = tl.multiple_of(start_n, BLOCK_N)
                offs_n = start_n + tl.arange(0, BLOCK_N)
                k_live = offs_n < live_len

                k_ptrs = (
                    k_ptr
                    + off_p * stride_kp
                    + off_h * stride_kh
                    + offs_n[:, None] * stride_kt
                    + offs_d[None, :] * stride_kd
                )
                v_ptrs = (
                    v_ptr
                    + off_p * stride_vp
                    + off_h * stride_vh
                    + offs_n[:, None] * stride_vt
                    + offs_d[None, :] * stride_vd
                )
                k = tl.load(k_ptrs, mask=k_live[:, None], other=0.0)
                v = tl.load(v_ptrs, mask=k_live[:, None], other=0.0)

                k_q = tl.load(
                    coords_base + offs_n * stride_ct,
                    mask=k_live,
                    other=0,
                )
                k_r = tl.load(
                    coords_base + offs_n * stride_ct + stride_cc,
                    mask=k_live,
                    other=0,
                )
                dq = q_q[:, None] - k_q[None, :]
                dr = q_r[:, None] - k_r[None, :]
                distance = tl.maximum(
                    tl.abs(dq),
                    tl.maximum(tl.abs(dr), tl.abs(dq + dr)),
                )
                bucket = tl.minimum(tl.maximum(distance, 1), D_MAX) - 1
                bucket = tl.where(
                    offs_m[:, None] == offs_n[None, :],
                    D_MAX,
                    bucket,
                )
                bucket = tl.where(
                    (offs_m[:, None] == 0) | (offs_n[None, :] == 0),
                    D_MAX + 1,
                    bucket,
                )
                bucket = tl.where(k_live[None, :], bucket, D_MAX + 2)
                bias = tl.load(
                    table_ptr + off_h * stride_th + bucket * stride_tb,
                    cache_modifier=".ca",
                )

                scores = tl.dot(q, tl.trans(k)) * sm_scale + bias
                m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
                p = tl.math.exp2((scores - m_ij[:, None]) * 1.4426950408889634)
                alpha = tl.math.exp2((m_i - m_ij) * 1.4426950408889634)
                acc *= alpha[:, None]
                acc += tl.dot(p.to(q_ptr.dtype.element_ty), v)
                l_i = l_i * alpha + tl.sum(p, axis=1)
                m_i = m_ij

            out = acc / l_i[:, None]
            out = tl.where(q_live[:, None], out, 0.0)
            tl.store(out_ptrs, out, mask=offs_m[:, None] < n_ctx)


def _bias_table(q: Tensor, dist_bias: Tensor) -> Tensor:
    """Cast the learned rows once and append the finite PAD sentinel."""
    table = dist_bias.to(q.dtype)
    pad = table.new_full((table.shape[0], 1), _PAD_BIAS)
    return torch.cat((table, pad), dim=1)


def _attention_reference_table(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    coords: Tensor,
    seq_lens: Tensor,
    table: Tensor,
) -> Tensor:
    """The dense formulation used by CPU, failed launches, and recompute."""
    _, _, t, _ = q.shape
    d_max = table.shape[1] - 3
    dq = coords[:, :, None, 0] - coords[:, None, :, 0]
    dr = coords[:, :, None, 1] - coords[:, None, :, 1]
    distance = torch.maximum(dq.abs(), torch.maximum(dr.abs(), (dq + dr).abs()))
    bucket = distance.clamp(1, d_max) - 1

    rows = torch.arange(t, device=q.device)
    bucket = torch.where(rows[:, None] == rows[None, :], d_max, bucket)
    token = (rows[:, None] == 0) | (rows[None, :] == 0)
    bucket = torch.where(token, d_max + 1, bucket)
    valid = rows[None, :] < seq_lens[:, None]
    bucket = torch.where(valid[:, None, :], bucket, d_max + 2)

    mask = table[:, bucket.long()].permute(1, 0, 2, 3)
    result = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    result = result.masked_fill(~valid[:, None, :, None], 0)

    # Match empty_like(q)'s preserved strides in every dispatch path. The
    # model's following head-to-row transpose can then remain a view.
    out = torch.empty_like(q)
    out.copy_(result)
    return out


def _attention_reference(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    coords: Tensor,
    seq_lens: Tensor,
    dist_bias: Tensor,
) -> Tensor:
    """Reference attention with the checkpoint-compatible bias parameter."""
    return _attention_reference_table(
        q, k, v, coords, seq_lens, _bias_table(q, dist_bias)
    )


def _shape_key(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    coords: Tensor,
    seq_lens: Tensor,
    table: Tensor,
) -> tuple[object, ...]:
    return (
        q.device.type,
        q.device.index,
        q.dtype,
        tuple(q.shape),
        tuple(q.stride()),
        tuple(k.stride()),
        tuple(v.stride()),
        tuple(coords.stride()),
        tuple(seq_lens.stride()),
        tuple(table.shape),
        tuple(table.stride()),
    )


def _validate_inputs(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    coords: Tensor,
    seq_lens: Tensor,
    table: Tensor,
) -> None:
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must have the same (P, A, T, D) shape")
    p, heads, t, _ = q.shape
    if coords.shape != (p, t, 2) or coords.dtype != torch.int32:
        raise ValueError("coords must be int32 with shape (P, T, 2)")
    if seq_lens.shape != (p,) or seq_lens.dtype != torch.int32:
        raise ValueError("seq_lens must be int32 with shape (P,)")
    if table.ndim != 2 or table.shape[0] != heads or table.shape[1] < 4:
        raise ValueError("bias table must have shape (A, d_max + 3)")
    tensors = (k, v, coords, seq_lens, table)
    if any(x.device != q.device for x in tensors):
        raise ValueError("all attention inputs must be on one device")
    if k.dtype != q.dtype or v.dtype != q.dtype or table.dtype != q.dtype:
        raise ValueError("q, k, v, and the bias table must have one dtype")


def _launch_triton(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    coords: Tensor,
    seq_lens: Tensor,
    table: Tensor,
) -> Tensor:
    p, heads, t, head_dim = q.shape
    out = torch.empty_like(q)
    grid = (triton.cdiv(t, _BLOCK_M), p * heads)
    _fused_attention_kernel[grid](
        q,
        k,
        v,
        coords,
        seq_lens,
        table,
        out,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *coords.stride(),
        *seq_lens.stride(),
        *table.stride(),
        *out.stride(),
        heads,
        t,
        1.0 / math.sqrt(head_dim),
        D_MAX=table.shape[1] - 3,
        HEAD_DIM=head_dim,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    return out


@torch.library.custom_op("mantisnet::fused_attention", mutates_args=())
def _fused_attention_op(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    coords: Tensor,
    seq_lens: Tensor,
    table: Tensor,
) -> Tensor:
    _validate_inputs(q, k, v, coords, seq_lens, table)
    supported = (
        triton is not None
        and q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16)
        and q.shape[-1] in (16, 32, 64)
    )
    if not supported:
        return _attention_reference_table(q, k, v, coords, seq_lens, table)

    key = _shape_key(q, k, v, coords, seq_lens, table)
    if key in _FAILED_SHAPES:
        return _attention_reference_table(q, k, v, coords, seq_lens, table)
    try:
        return _launch_triton(q, k, v, coords, seq_lens, table)
    except Exception as exc:
        _FAILED_SHAPES[key] = f"{type(exc).__name__}: {exc}"
        warnings.warn(
            "fused attention failed for "
            f"shape={tuple(q.shape)}, dtype={q.dtype}; using SDPA for this "
            f"shape: {_FAILED_SHAPES[key]}",
            RuntimeWarning,
            stacklevel=2,
        )
        return _attention_reference_table(q, k, v, coords, seq_lens, table)


@_fused_attention_op.register_fake
def _(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    coords: Tensor,
    seq_lens: Tensor,
    table: Tensor,
) -> Tensor:
    return torch.empty_like(q)


def _setup_context(ctx, inputs, output) -> None:
    q, k, v, coords, seq_lens, table = inputs
    ctx.save_for_backward(q, k, v, coords, seq_lens, table)


def _backward(ctx, grad_out: Tensor):
    q, k, v, coords, seq_lens, table = ctx.saved_tensors
    with torch.enable_grad():
        q_ = q.detach().requires_grad_(True)
        k_ = k.detach().requires_grad_(True)
        v_ = v.detach().requires_grad_(True)
        table_ = table.detach().requires_grad_(True)
        out = _attention_reference_table(q_, k_, v_, coords, seq_lens, table_)
        dq, dk, dv, dtable = torch.autograd.grad(
            out,
            (q_, k_, v_, table_),
            grad_out,
        )
    return dq, dk, dv, None, None, dtable


_fused_attention_op.register_autograd(_backward, setup_context=_setup_context)


def fused_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    coords: Tensor,
    seq_lens: Tensor,
    dist_bias: Tensor,
) -> Tensor:
    """Apply fused attention while retaining the checkpoint bias layout."""
    return _fused_attention_op(q, k, v, coords, seq_lens, _bias_table(q, dist_bias))
