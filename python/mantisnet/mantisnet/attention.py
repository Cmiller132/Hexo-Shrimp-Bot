"""Fused block-diagonal attention for the stone rows (MODEL_SPEC §5.3).

Each position attends over its live ``[4 state latents; stones]`` prefix and
nothing else: keys past the prefix are masked hard, and a query tile past the
prefix performs no key work. There is no learned pair bias — the Step-11
knock-out measured the geometric bias channel decorative in the trained
function, and its removal also removed the one nondeterministic accumulation
(the bias-table gradient's atomic adds) from the backward.
"""

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


# Fixed launch geometry keeps symbolic shape changes out of Triton's tuning cache.
_BLOCK_M = 64
_BLOCK_N = 64
_NUM_WARPS = 4
_NUM_STAGES = 3

_FAILED_SHAPES: dict[tuple[object, ...], str] = {}
_FAILED_BACKWARD_SHAPES: dict[tuple[object, ...], str] = {}


# A custom-op implementation runs below the Autograd dispatch keys. Restore
# the normal thread-local keysets only for its rare dense-backward fallback.
_DENSE_BACKWARD_INCLUDE_KEYS = torch._C._dispatch_tls_local_include_set()
_DENSE_BACKWARD_EXCLUDE_KEYS = torch._C._dispatch_tls_local_exclude_set()


if triton is not None:

    @triton.jit
    def _fused_attention_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        seq_lens_ptr,
        out_ptr,
        lse_ptr,
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
        stride_lp,
        stride_op,
        stride_oh,
        stride_ot,
        stride_od,
        stride_lsep,
        stride_lseh,
        stride_lset,
        n_heads,
        n_ctx,
        sm_scale,
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
        lse_ptrs = (
            lse_ptr
            + off_p * stride_lsep
            + off_h * stride_lseh
            + offs_m * stride_lset
        )

        # A row whose query tile starts after its live prefix performs no key
        # work. The zeros also make padding deterministic for direct op users.
        if start_m >= live_len:
            zeros = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
            tl.store(out_ptrs, zeros, mask=offs_m[:, None] < n_ctx)
            tl.store(lse_ptrs, 0.0, mask=offs_m < n_ctx)
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

                scores = tl.dot(q, tl.trans(k)) * sm_scale
                scores = tl.where(k_live[None, :], scores, -float("inf"))
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
            lse = m_i + tl.log(l_i)
            lse = tl.where(q_live, lse, 0.0)
            tl.store(lse_ptrs, lse, mask=offs_m < n_ctx)


    @triton.jit
    def _fused_attention_delta_kernel(
        out_ptr,
        do_ptr,
        seq_lens_ptr,
        delta_ptr,
        stride_op,
        stride_oh,
        stride_ot,
        stride_od,
        stride_dop,
        stride_doh,
        stride_dot,
        stride_dod,
        stride_lp,
        stride_dp,
        stride_dh,
        stride_dt,
        n_heads,
        n_ctx,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        start_m = tl.program_id(0) * BLOCK_M
        off_ph = tl.program_id(1)
        off_p = off_ph // n_heads
        off_h = off_ph - off_p * n_heads

        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)
        live_len = tl.load(seq_lens_ptr + off_p * stride_lp)
        live_len = tl.minimum(tl.maximum(live_len, 0), n_ctx)
        row_live = offs_m < live_len

        out = tl.load(
            out_ptr
            + off_p * stride_op
            + off_h * stride_oh
            + offs_m[:, None] * stride_ot
            + offs_d[None, :] * stride_od,
            mask=row_live[:, None],
            other=0.0,
        ).to(tl.float32)
        do = tl.load(
            do_ptr
            + off_p * stride_dop
            + off_h * stride_doh
            + offs_m[:, None] * stride_dot
            + offs_d[None, :] * stride_dod,
            mask=row_live[:, None],
            other=0.0,
        ).to(tl.float32)
        delta = tl.sum(out * do, axis=1)
        delta = tl.where(row_live, delta, 0.0)
        tl.store(
            delta_ptr
            + off_p * stride_dp
            + off_h * stride_dh
            + offs_m * stride_dt,
            delta,
            mask=offs_m < n_ctx,
        )


    @triton.jit
    def _fused_attention_dq_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        seq_lens_ptr,
        lse_ptr,
        delta_ptr,
        do_ptr,
        dq_ptr,
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
        stride_lp,
        stride_lsep,
        stride_lseh,
        stride_lset,
        stride_delp,
        stride_delh,
        stride_delt,
        stride_dop,
        stride_doh,
        stride_dot,
        stride_dod,
        stride_dqp,
        stride_dqh,
        stride_dqt,
        stride_dqd,
        n_heads,
        n_ctx,
        sm_scale,
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
        q_live = offs_m < live_len
        dq_ptrs = (
            dq_ptr
            + off_p * stride_dqp
            + off_h * stride_dqh
            + offs_m[:, None] * stride_dqt
            + offs_d[None, :] * stride_dqd
        )

        if start_m >= live_len:
            zeros = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
            tl.store(dq_ptrs, zeros, mask=offs_m[:, None] < n_ctx)
        else:
            q = tl.load(
                q_ptr
                + off_p * stride_qp
                + off_h * stride_qh
                + offs_m[:, None] * stride_qt
                + offs_d[None, :] * stride_qd,
                mask=q_live[:, None],
                other=0.0,
            )
            do = tl.load(
                do_ptr
                + off_p * stride_dop
                + off_h * stride_doh
                + offs_m[:, None] * stride_dot
                + offs_d[None, :] * stride_dod,
                mask=q_live[:, None],
                other=0.0,
            )
            lse = tl.load(
                lse_ptr
                + off_p * stride_lsep
                + off_h * stride_lseh
                + offs_m * stride_lset,
                mask=q_live,
                other=0.0,
            )
            delta = tl.load(
                delta_ptr
                + off_p * stride_delp
                + off_h * stride_delh
                + offs_m * stride_delt,
                mask=q_live,
                other=0.0,
            )

            dq_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

            for start_n in tl.range(0, live_len, BLOCK_N):
                start_n = tl.multiple_of(start_n, BLOCK_N)
                offs_n = start_n + tl.arange(0, BLOCK_N)
                k_live = offs_n < live_len

                k = tl.load(
                    k_ptr
                    + off_p * stride_kp
                    + off_h * stride_kh
                    + offs_n[:, None] * stride_kt
                    + offs_d[None, :] * stride_kd,
                    mask=k_live[:, None],
                    other=0.0,
                )
                v = tl.load(
                    v_ptr
                    + off_p * stride_vp
                    + off_h * stride_vh
                    + offs_n[:, None] * stride_vt
                    + offs_d[None, :] * stride_vd,
                    mask=k_live[:, None],
                    other=0.0,
                )

                scores = tl.dot(q, tl.trans(k)) * sm_scale
                p = tl.math.exp2(
                    (scores - lse[:, None]) * 1.4426950408889634
                )
                pair_live = q_live[:, None] & k_live[None, :]
                p = tl.where(pair_live, p, 0.0)
                dp = tl.dot(do, tl.trans(v))
                ds = p * (dp - delta[:, None])
                dq_acc += tl.dot(ds.to(k_ptr.dtype.element_ty), k) * sm_scale

            dq_acc = tl.where(q_live[:, None], dq_acc, 0.0)
            tl.store(dq_ptrs, dq_acc, mask=offs_m[:, None] < n_ctx)


    @triton.jit
    def _fused_attention_dkdv_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        seq_lens_ptr,
        lse_ptr,
        delta_ptr,
        do_ptr,
        dk_ptr,
        dv_ptr,
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
        stride_lp,
        stride_lsep,
        stride_lseh,
        stride_lset,
        stride_delp,
        stride_delh,
        stride_delt,
        stride_dop,
        stride_doh,
        stride_dot,
        stride_dod,
        stride_dkp,
        stride_dkh,
        stride_dkt,
        stride_dkd,
        stride_dvp,
        stride_dvh,
        stride_dvt,
        stride_dvd,
        n_heads,
        n_ctx,
        sm_scale,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        start_n = tl.program_id(0) * BLOCK_N
        off_ph = tl.program_id(1)
        off_p = off_ph // n_heads
        off_h = off_ph - off_p * n_heads

        offs_n = start_n + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, HEAD_DIM)
        live_len = tl.load(seq_lens_ptr + off_p * stride_lp)
        live_len = tl.minimum(tl.maximum(live_len, 0), n_ctx)
        k_live = offs_n < live_len
        dk_ptrs = (
            dk_ptr
            + off_p * stride_dkp
            + off_h * stride_dkh
            + offs_n[:, None] * stride_dkt
            + offs_d[None, :] * stride_dkd
        )
        dv_ptrs = (
            dv_ptr
            + off_p * stride_dvp
            + off_h * stride_dvh
            + offs_n[:, None] * stride_dvt
            + offs_d[None, :] * stride_dvd
        )

        if start_n >= live_len:
            zeros = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
            tl.store(dk_ptrs, zeros, mask=offs_n[:, None] < n_ctx)
            tl.store(dv_ptrs, zeros, mask=offs_n[:, None] < n_ctx)
        else:
            k = tl.load(
                k_ptr
                + off_p * stride_kp
                + off_h * stride_kh
                + offs_n[:, None] * stride_kt
                + offs_d[None, :] * stride_kd,
                mask=k_live[:, None],
                other=0.0,
            )
            v = tl.load(
                v_ptr
                + off_p * stride_vp
                + off_h * stride_vh
                + offs_n[:, None] * stride_vt
                + offs_d[None, :] * stride_vd,
                mask=k_live[:, None],
                other=0.0,
            )

            dk_acc = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
            dv_acc = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

            for start_m in tl.range(0, live_len, BLOCK_M):
                start_m = tl.multiple_of(start_m, BLOCK_M)
                offs_m = start_m + tl.arange(0, BLOCK_M)
                q_live = offs_m < live_len

                q = tl.load(
                    q_ptr
                    + off_p * stride_qp
                    + off_h * stride_qh
                    + offs_m[:, None] * stride_qt
                    + offs_d[None, :] * stride_qd,
                    mask=q_live[:, None],
                    other=0.0,
                )
                do = tl.load(
                    do_ptr
                    + off_p * stride_dop
                    + off_h * stride_doh
                    + offs_m[:, None] * stride_dot
                    + offs_d[None, :] * stride_dod,
                    mask=q_live[:, None],
                    other=0.0,
                )
                lse = tl.load(
                    lse_ptr
                    + off_p * stride_lsep
                    + off_h * stride_lseh
                    + offs_m * stride_lset,
                    mask=q_live,
                    other=0.0,
                )
                delta = tl.load(
                    delta_ptr
                    + off_p * stride_delp
                    + off_h * stride_delh
                    + offs_m * stride_delt,
                    mask=q_live,
                    other=0.0,
                )

                scores = tl.dot(q, tl.trans(k)) * sm_scale
                p = tl.math.exp2(
                    (scores - lse[:, None]) * 1.4426950408889634
                )
                pair_live = q_live[:, None] & k_live[None, :]
                p = tl.where(pair_live, p, 0.0)
                dp = tl.dot(do, tl.trans(v))
                ds = p * (dp - delta[:, None])
                dk_acc += (
                    tl.dot(tl.trans(ds.to(q_ptr.dtype.element_ty)), q)
                    * sm_scale
                )
                dv_acc += tl.dot(
                    tl.trans(p.to(do_ptr.dtype.element_ty)),
                    do,
                )

            dk_acc = tl.where(k_live[:, None], dk_acc, 0.0)
            dv_acc = tl.where(k_live[:, None], dv_acc, 0.0)
            tl.store(dk_ptrs, dk_acc, mask=offs_n[:, None] < n_ctx)
            tl.store(dv_ptrs, dv_acc, mask=offs_n[:, None] < n_ctx)


def _key_valid(seq_lens: Tensor, t: int) -> Tensor:
    """(P, T) bool: which rows of each position are live."""
    rows = torch.arange(t, device=seq_lens.device)
    return rows[None, :] < seq_lens[:, None]


def _apply_reference(q: Tensor, k: Tensor, v: Tensor, valid: Tensor) -> Tensor:
    result = F.scaled_dot_product_attention(
        q, k, v, attn_mask=valid[:, None, None, :]
    )
    result = result.masked_fill(~valid[:, None, :, None], 0)

    # Match empty_like(q)'s preserved strides in every dispatch path. The
    # model's following head-to-row transpose can then remain a view.
    out = torch.empty_like(q)
    out.copy_(result)
    return out


def _attention_reference(q: Tensor, k: Tensor, v: Tensor, seq_lens: Tensor) -> Tensor:
    """The dense formulation used by CPU, failed launches, and recompute."""
    return _apply_reference(q, k, v, _key_valid(seq_lens, q.shape[2]))


def _shape_key(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    seq_lens: Tensor,
) -> tuple[object, ...]:
    return (
        q.device.type,
        q.device.index,
        q.dtype,
        tuple(q.shape),
        tuple(q.stride()),
        tuple(k.stride()),
        tuple(v.stride()),
        tuple(seq_lens.stride()),
    )


def _validate_inputs(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    seq_lens: Tensor,
) -> None:
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must have the same (P, A, T, D) shape")
    p = q.shape[0]
    if seq_lens.shape != (p,) or seq_lens.dtype != torch.int32:
        raise ValueError("seq_lens must be int32 with shape (P,)")
    tensors = (k, v, seq_lens)
    if any(x.device != q.device for x in tensors):
        raise ValueError("all attention inputs must be on one device")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q, k, and v must have one dtype")


def _launch_triton(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    seq_lens: Tensor,
) -> tuple[Tensor, Tensor]:
    p, heads, t, head_dim = q.shape
    out = torch.empty_like(q)
    lse = torch.empty((p, heads, t), dtype=torch.float32, device=q.device)
    grid = (triton.cdiv(t, _BLOCK_M), p * heads)
    _fused_attention_kernel[grid](
        q,
        k,
        v,
        seq_lens,
        out,
        lse,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *seq_lens.stride(),
        *out.stride(),
        *lse.stride(),
        heads,
        t,
        1.0 / math.sqrt(head_dim),
        HEAD_DIM=head_dim,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    return out, lse


def _launch_triton_backward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    seq_lens: Tensor,
    out: Tensor,
    lse: Tensor,
    grad_out: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    p, heads, t, head_dim = q.shape
    delta = torch.empty((p, heads, t), dtype=torch.float32, device=q.device)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    row_grid = (triton.cdiv(t, _BLOCK_M), p * heads)
    key_grid = (triton.cdiv(t, _BLOCK_N), p * heads)

    _fused_attention_delta_kernel[row_grid](
        out,
        grad_out,
        seq_lens,
        delta,
        *out.stride(),
        *grad_out.stride(),
        *seq_lens.stride(),
        *delta.stride(),
        heads,
        t,
        HEAD_DIM=head_dim,
        BLOCK_M=_BLOCK_M,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    _fused_attention_dq_kernel[row_grid](
        q,
        k,
        v,
        seq_lens,
        lse,
        delta,
        grad_out,
        dq,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *seq_lens.stride(),
        *lse.stride(),
        *delta.stride(),
        *grad_out.stride(),
        *dq.stride(),
        heads,
        t,
        1.0 / math.sqrt(head_dim),
        HEAD_DIM=head_dim,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    _fused_attention_dkdv_kernel[key_grid](
        q,
        k,
        v,
        seq_lens,
        lse,
        delta,
        grad_out,
        dk,
        dv,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *seq_lens.stride(),
        *lse.stride(),
        *delta.stride(),
        *grad_out.stride(),
        *dk.stride(),
        *dv.stride(),
        heads,
        t,
        1.0 / math.sqrt(head_dim),
        HEAD_DIM=head_dim,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    return dq, dk, dv


@torch.library.custom_op("mantisnet::fused_attention", mutates_args=())
def _fused_attention_op(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    seq_lens: Tensor,
) -> tuple[Tensor, Tensor]:
    _validate_inputs(q, k, v, seq_lens)
    supported = (
        triton is not None
        and q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16)
        and q.shape[-1] in (16, 32, 64)
    )
    if not supported:
        out = _attention_reference(q, k, v, seq_lens)
        return out, torch.empty(0, dtype=torch.float32, device=q.device)

    key = _shape_key(q, k, v, seq_lens)
    if key in _FAILED_SHAPES:
        out = _attention_reference(q, k, v, seq_lens)
        return out, torch.empty(0, dtype=torch.float32, device=q.device)
    try:
        return _launch_triton(q, k, v, seq_lens)
    except Exception as exc:
        _FAILED_SHAPES[key] = f"{type(exc).__name__}: {exc}"
        warnings.warn(
            "fused attention failed for "
            f"shape={tuple(q.shape)}, dtype={q.dtype}; using SDPA for this "
            f"shape: {_FAILED_SHAPES[key]}",
            RuntimeWarning,
            stacklevel=2,
        )
        out = _attention_reference(q, k, v, seq_lens)
        return out, torch.empty(0, dtype=torch.float32, device=q.device)


@_fused_attention_op.register_fake
def _(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    seq_lens: Tensor,
) -> tuple[Tensor, Tensor]:
    supported = (
        triton is not None
        and q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16)
        and q.shape[-1] in (16, 32, 64)
    )
    if supported:
        lse = torch.empty(q.shape[:3], dtype=torch.float32, device=q.device)
    else:
        lse = torch.empty(0, dtype=torch.float32, device=q.device)
    return torch.empty_like(q), lse


def _setup_context(ctx, inputs, output) -> None:
    q, k, v, seq_lens = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, seq_lens, out, lse)
    ctx.mark_non_differentiable(lse)


def _backward(ctx, grad_out: Tensor):
    q, k, v, seq_lens = ctx.saved_tensors[:4]
    valid = _key_valid(seq_lens, q.shape[2])
    with torch.enable_grad():
        q_ = q.detach().requires_grad_(True)
        k_ = k.detach().requires_grad_(True)
        v_ = v.detach().requires_grad_(True)
        out = _apply_reference(q_, k_, v_, valid)
        dq, dk, dv = torch.autograd.grad(out, (q_, k_, v_), grad_out)
    return dq, dk, dv, None


class _DenseBackwardContext:
    def __init__(self, saved_tensors: tuple[Tensor, ...]) -> None:
        self.saved_tensors = saved_tensors


def _dense_backward_below_autograd(
    saved_tensors: tuple[Tensor, ...],
    grad_out: Tensor,
):
    with torch._C._ForceDispatchKeyGuard(
        _DENSE_BACKWARD_INCLUDE_KEYS, _DENSE_BACKWARD_EXCLUDE_KEYS
    ):
        return _backward(_DenseBackwardContext(saved_tensors), grad_out)


def _backward_shape_key(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    seq_lens: Tensor,
    grad_out: Tensor,
) -> tuple[object, ...]:
    return _shape_key(q, k, v, seq_lens) + (
        grad_out.dtype,
        tuple(grad_out.stride()),
    )


@torch.library.custom_op("mantisnet::fused_attention_backward", mutates_args=())
def _fused_attention_backward_op(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    seq_lens: Tensor,
    out: Tensor,
    lse: Tensor,
    grad_out: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    _validate_inputs(q, k, v, seq_lens)
    saved = (q, k, v, seq_lens)
    # The outer autograd formula handles this branch in eager mode. Keep the
    # sentinel check inside the opaque op as well: a compiled graph may have
    # been traced for Triton before a runtime forward launch marks the shape
    # failed and returns the dense sentinel.
    if lse.numel() == 0:
        dq, dk, dv, _ = _dense_backward_below_autograd(saved, grad_out)
        return dq, dk, dv

    key = _backward_shape_key(q, k, v, seq_lens, grad_out)
    if key in _FAILED_BACKWARD_SHAPES:
        dq, dk, dv, _ = _dense_backward_below_autograd(saved, grad_out)
        return dq, dk, dv
    try:
        return _launch_triton_backward(q, k, v, seq_lens, out, lse, grad_out)
    except Exception as exc:
        _FAILED_BACKWARD_SHAPES[key] = f"{type(exc).__name__}: {exc}"
        warnings.warn(
            "fused attention backward failed for "
            f"shape={tuple(q.shape)}, dtype={q.dtype}; using dense backward "
            f"for this shape: {_FAILED_BACKWARD_SHAPES[key]}",
            RuntimeWarning,
            stacklevel=2,
        )
        dq, dk, dv, _ = _dense_backward_below_autograd(saved, grad_out)
        return dq, dk, dv


@_fused_attention_backward_op.register_fake
def _(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    seq_lens: Tensor,
    out: Tensor,
    lse: Tensor,
    grad_out: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    return (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
    )


def _dispatch_backward(ctx, grad_out: Tensor, _grad_lse: Tensor | None):
    q, k, v, seq_lens, out, lse = ctx.saved_tensors
    if lse.numel() == 0:
        return _backward(_DenseBackwardContext((q, k, v, seq_lens)), grad_out)
    dq, dk, dv = _fused_attention_backward_op(q, k, v, seq_lens, out, lse, grad_out)
    return dq, dk, dv, None


_fused_attention_op.register_autograd(
    _dispatch_backward, setup_context=_setup_context
)


def fused_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    seq_lens: Tensor,
) -> Tensor:
    """Attention over ``[latent rows; stones]`` per position.

    Keys past each position's live prefix (``seq_lens``) are masked hard;
    padding query rows return exact zeros.
    """
    out, _ = _fused_attention_op(q, k, v, seq_lens)
    return out
