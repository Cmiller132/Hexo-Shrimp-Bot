"""Fused orbit-biased attention for the stone rows (MODEL_SPEC §4.1, §5.3).

Every query-key pair is typed by the D6 orbit of its displacement: the frozen
48-class vocabulary of §4.2 within hex radius 12, then FAR beyond it, SELF on
the diagonal, TOKEN for any pair touching a latent row, and a finite PAD
sentinel for keys past the live prefix. PAD is appended at compute time.

The bias the kernels read comes in two widths, selected by the model's
``orbit_vectors`` knob and told apart by rank at the seam:

* ``(A, BIAS_ROWS)`` — the static table of 51 orbit/FAR/SELF/TOKEN rows per
  head (:func:`compose_bias_table`), one opinion of a displacement per head;
* ``(P, A, T, BIAS_ROWS)`` — the same rows plus a content term the querying
  row projects out of a learned per-orbit vector
  (:func:`compose_row_bias_table`), so what a head makes of a displacement
  depends on the stone that is asking.

Each width has its own kernel specialization: the static one broadcasts its
row over the query tile and reduces its gradient with an atomic, the per-row
one gathers and stores by query row. The seam knows only "a table indexed by
(row, bucket)"; the parametrisation is the model's.
"""

from __future__ import annotations

import contextlib
import math
import warnings

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

from .builder import orbit48_id

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


_PAD_BIAS = -3.0e4

# Bias-table layout, per head: orbits 0..47, then FAR, SELF, TOKEN (learned),
# then PAD (sentinel, not a parameter).
ORBIT_RADIUS = 12
ORBIT_CLASSES = 48
FAR_BUCKET = ORBIT_CLASSES
SELF_BUCKET = ORBIT_CLASSES + 1
TOKEN_BUCKET = ORBIT_CLASSES + 2
PAD_BUCKET = ORBIT_CLASSES + 3
BIAS_ROWS = PAD_BUCKET  # learned rows: orbits + FAR + SELF + TOKEN
TABLE_WIDTH = PAD_BUCKET + 1  # compute-time width, PAD appended
_LUT_SIDE = 2 * ORBIT_RADIUS + 1

_ORBIT_LUTS: dict[torch.device, Tensor] = {}


def _orbit_lut_cpu() -> Tensor:
    """The (25*25,) int32 displacement → bucket table, row-major in
    ``(dq + 12, dr + 12)``.

    Generated from the builder's frozen orbit function, never hand-written.
    Displacements of hex distance 1..12 map to their orbit, the zero
    displacement to SELF (only ever reached on the diagonal, which the
    kernels override anyway), and the square's corners past radius 12 to
    FAR, so any clamped index is a valid bucket."""
    table = torch.full((_LUT_SIDE * _LUT_SIDE,), FAR_BUCKET, dtype=torch.int32)
    for dq in range(-ORBIT_RADIUS, ORBIT_RADIUS + 1):
        for dr in range(-ORBIT_RADIUS, ORBIT_RADIUS + 1):
            distance = max(abs(dq), abs(dr), abs(dq + dr))
            index = (dq + ORBIT_RADIUS) * _LUT_SIDE + (dr + ORBIT_RADIUS)
            if distance == 0:
                table[index] = SELF_BUCKET
            elif distance <= ORBIT_RADIUS:
                table[index] = orbit48_id(dq, dr)
    return table


def orbit_lut(device: torch.device | str) -> Tensor:
    """The displacement → bucket table resident on ``device``, built once."""
    device = torch.device(device)
    lut = _ORBIT_LUTS.get(device)
    if lut is None:
        lut = _orbit_lut_cpu().to(device)
        _ORBIT_LUTS[device] = lut
    return lut


# Coarse (distance, on-axis) rows — the vocabulary the orbit rows refine.
# Distance 1 is all on-axis, so distances 1..12 give 23 classes; FAR, SELF
# and TOKEN follow in the orbit table's order.
AXIS_CLASSES = 23
AXIS_ROWS = AXIS_CLASSES + 3

_AXIS_INDICES: dict[torch.device, Tensor] = {}


def _axis_index_cpu() -> Tensor:
    """(BIAS_ROWS,) int64: each bias row's coarse row — an orbit's
    (distance, on-axis) class, FAR/SELF/TOKEN their own rows.

    Derived from the builder's orbit function: every displacement of an orbit
    must agree on its class (D6 preserves distance and axis membership), and
    every class must be reached — both checked here, never assumed."""
    keys = [(1, True)] + [(d, on) for d in range(2, ORBIT_RADIUS + 1) for on in (True, False)]
    classes = {key: i for i, key in enumerate(keys)}
    assert len(classes) == AXIS_CLASSES
    index = torch.full((BIAS_ROWS,), -1, dtype=torch.int64)
    for dq in range(-ORBIT_RADIUS, ORBIT_RADIUS + 1):
        for dr in range(-ORBIT_RADIUS, ORBIT_RADIUS + 1):
            distance = max(abs(dq), abs(dr), abs(dq + dr))
            if distance == 0 or distance > ORBIT_RADIUS:
                continue
            on_axis = dq == 0 or dr == 0 or dq + dr == 0
            orbit = orbit48_id(dq, dr)
            cls = classes[(distance, on_axis)]
            if index[orbit] == -1:
                index[orbit] = cls
            elif int(index[orbit]) != cls:
                raise AssertionError(
                    f"orbit {orbit} straddles (distance, on-axis) classes {int(index[orbit])} and {cls}"
                )
    if int((index[:ORBIT_CLASSES] < 0).sum()) != 0:
        raise AssertionError("an orbit has no displacement within the LUT radius")
    if len(set(index[:ORBIT_CLASSES].tolist())) != AXIS_CLASSES:
        raise AssertionError("the orbits do not cover every (distance, on-axis) class")
    index[FAR_BUCKET] = AXIS_CLASSES
    index[SELF_BUCKET] = AXIS_CLASSES + 1
    index[TOKEN_BUCKET] = AXIS_CLASSES + 2
    return index


def axis_index(device: torch.device | str) -> Tensor:
    """The bias-row → coarse-row table resident on ``device``, built once."""
    device = torch.device(device)
    index = _AXIS_INDICES.get(device)
    if index is None:
        index = _axis_index_cpu().to(device)
        _AXIS_INDICES[device] = index
    return index


def compose_bias_table(axis_bias: Tensor, orbit_bias: Tensor, index: Tensor) -> Tensor:
    """The ``(A, BIAS_ROWS)`` table the kernels consume.

    Every orbit row is its coarse (distance, on-axis) row plus the orbit's
    learned residual; FAR, SELF and TOKEN are their coarse rows alone. The
    coarse rows pool gradient across every displacement of a distance, so the
    radial profile learns at the rate of a (distance, on-axis) table while the
    residual adds orbit resolution only where the data asks for it.
    """
    if axis_bias.ndim != 2 or axis_bias.shape[1] != AXIS_ROWS:
        raise ValueError(
            f"axis_bias must have shape (A, {AXIS_ROWS}), got {tuple(axis_bias.shape)}"
        )
    if orbit_bias.shape != (axis_bias.shape[0], ORBIT_CLASSES):
        raise ValueError(
            f"orbit_bias must have shape ({axis_bias.shape[0]}, {ORBIT_CLASSES}), "
            f"got {tuple(orbit_bias.shape)}"
        )
    coarse = axis_bias.index_select(1, index)
    return torch.cat((coarse[:, :ORBIT_CLASSES] + orbit_bias, coarse[:, ORBIT_CLASSES:]), dim=1)


def compose_row_bias_table(bias_table: Tensor, orbit_vec: Tensor, q: Tensor) -> Tensor:
    """The ``(P, A, T, BIAS_ROWS)`` per-query-row table of the
    ``orbit_vectors`` knob.

    Row ``m`` of head ``h`` is the static row ``bias_table[h]`` plus
    ``q[m] @ orbit_vec[h]^T``, so a head's opinion about a displacement is a
    function of the querying stone rather than a constant of the geometry.
    Only the query enters: the key's content already reaches the score through
    ``q·k``. The content term carries no scale factor — ``orbit_vec`` is
    learned and zero-initialised, so training starts exactly at the static
    table and the vectors set their own magnitude from there.

    Composed here rather than inside the kernel because the kernel seam is "a
    table indexed by (row, bucket)": autograd routes the table's gradient back
    to ``orbit_vec``, to the static parts, and to ``q`` with no kernel of its
    own.
    """
    if bias_table.ndim != 2 or bias_table.shape[1] != BIAS_ROWS:
        raise ValueError(
            f"bias_table must have shape (A, {BIAS_ROWS}), got {tuple(bias_table.shape)}"
        )
    heads = bias_table.shape[0]
    if q.ndim != 4 or q.shape[1] != heads:
        raise ValueError(
            f"q must have shape (P, {heads}, T, D), got {tuple(q.shape)}"
        )
    if orbit_vec.shape != (heads, BIAS_ROWS, q.shape[3]):
        raise ValueError(
            f"orbit_vec must have shape ({heads}, {BIAS_ROWS}, {q.shape[3]}), "
            f"got {tuple(orbit_vec.shape)}"
        )
    content = torch.matmul(q, orbit_vec.to(q.dtype).transpose(-2, -1))
    return bias_table.to(q.dtype)[None, :, None, :] + content


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
    def _pair_buckets(
        q_q,
        q_r,
        k_q,
        k_r,
        offs_m,
        offs_n,
        k_live,
        global_rows,
        lut_ptr,
        RADIUS: tl.constexpr,
        SIDE: tl.constexpr,
        FAR: tl.constexpr,
        SELF: tl.constexpr,
        TOKEN: tl.constexpr,
        PAD: tl.constexpr,
    ):
        # The (BLOCK_M, BLOCK_N) bias-bucket tile shared by the forward and
        # both backward sweeps: orbit of the displacement within RADIUS
        # (one gather into the 25x25 table), FAR past it, then the SELF,
        # TOKEN, and PAD overrides in increasing precedence.
        dq = q_q[:, None] - k_q[None, :]
        dr = q_r[:, None] - k_r[None, :]
        distance = tl.maximum(tl.abs(dq), tl.maximum(tl.abs(dr), tl.abs(dq + dr)))
        cq = tl.minimum(tl.maximum(dq, -RADIUS), RADIUS) + RADIUS
        cr = tl.minimum(tl.maximum(dr, -RADIUS), RADIUS) + RADIUS
        orbit = tl.load(lut_ptr + cq * SIDE + cr, cache_modifier=".ca")
        bucket = tl.where(distance <= RADIUS, orbit, FAR)
        bucket = tl.where(offs_m[:, None] == offs_n[None, :], SELF, bucket)
        bucket = tl.where(
            (offs_m[:, None] < global_rows) | (offs_n[None, :] < global_rows),
            TOKEN,
            bucket,
        )
        return tl.where(k_live[None, :], bucket, PAD)

    @triton.jit
    def _bias_tile(
        table_base,
        bucket,
        offs_m,
        q_live,
        stride_tm,
        stride_tb,
        ROW_TABLE: tl.constexpr,
    ):
        # The (BLOCK_M, BLOCK_N) bias tile, one load per pair either way.
        # ROW_TABLE roots each pair's gather at its own query row; without it
        # the head's single row broadcasts over the tile. Dead query rows read
        # nothing under ROW_TABLE: their table rows may sit past the tensor,
        # and their output is zeroed regardless.
        if ROW_TABLE:
            return tl.load(
                table_base + offs_m[:, None] * stride_tm + bucket * stride_tb,
                mask=q_live[:, None],
                other=0.0,
                cache_modifier=".ca",
            )
        return tl.load(table_base + bucket * stride_tb, cache_modifier=".ca")

    @triton.jit
    def _fused_attention_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        coords_ptr,
        seq_lens_ptr,
        table_ptr,
        lut_ptr,
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
        stride_cp,
        stride_ct,
        stride_cc,
        stride_lp,
        stride_tp,
        stride_th,
        stride_tm,
        stride_tb,
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
        global_rows,
        RADIUS: tl.constexpr,
        SIDE: tl.constexpr,
        FAR: tl.constexpr,
        SELF: tl.constexpr,
        TOKEN: tl.constexpr,
        PAD: tl.constexpr,
        ROW_TABLE: tl.constexpr,
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

            table_base = table_ptr + off_p * stride_tp + off_h * stride_th
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
                bucket = _pair_buckets(
                    q_q, q_r, k_q, k_r, offs_m, offs_n, k_live, global_rows,
                    lut_ptr, RADIUS, SIDE, FAR, SELF, TOKEN, PAD,
                )
                bias = _bias_tile(
                    table_base, bucket, offs_m, q_live, stride_tm, stride_tb,
                    ROW_TABLE,
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
        coords_ptr,
        seq_lens_ptr,
        table_ptr,
        lut_ptr,
        lse_ptr,
        delta_ptr,
        do_ptr,
        dq_ptr,
        dtable_ptr,
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
        stride_tp,
        stride_th,
        stride_tm,
        stride_tb,
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
        stride_dtp,
        stride_dth,
        stride_dtm,
        stride_dtb,
        n_heads,
        n_ctx,
        sm_scale,
        global_rows,
        RADIUS: tl.constexpr,
        SIDE: tl.constexpr,
        FAR: tl.constexpr,
        SELF: tl.constexpr,
        TOKEN: tl.constexpr,
        PAD: tl.constexpr,
        ROW_TABLE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_BUCKETS: tl.constexpr,
        TABLE_COLS: tl.constexpr,
    ):
        start_m = tl.program_id(0) * BLOCK_M
        off_ph = tl.program_id(1)
        off_p = off_ph // n_heads
        off_h = off_ph - off_p * n_heads

        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)
        offs_b = tl.arange(0, BLOCK_BUCKETS)
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
        if ROW_TABLE:
            # This program owns every table row of its query tile, so the row
            # histogram is a plain store — no atomics, and the result is
            # independent of block scheduling.
            dtable_ptrs = (
                dtable_ptr
                + off_p * stride_dtp
                + off_h * stride_dth
                + offs_m[:, None] * stride_dtm
                + offs_b[None, :] * stride_dtb
            )
            dtable_mask = (offs_m[:, None] < n_ctx) & (offs_b[None, :] < TABLE_COLS)

        if start_m >= live_len:
            zeros = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
            tl.store(dq_ptrs, zeros, mask=offs_m[:, None] < n_ctx)
            if ROW_TABLE:
                tl.store(
                    dtable_ptrs,
                    tl.zeros([BLOCK_M, BLOCK_BUCKETS], dtype=tl.float32),
                    mask=dtable_mask,
                )
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

            table_base = table_ptr + off_p * stride_tp + off_h * stride_th
            dq_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
            if ROW_TABLE:
                dtable_acc = tl.zeros([BLOCK_M, BLOCK_BUCKETS], dtype=tl.float32)
            else:
                dtable_acc = tl.zeros([BLOCK_BUCKETS], dtype=tl.float32)

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

                bucket = _pair_buckets(
                    q_q, q_r, k_q, k_r, offs_m, offs_n, k_live, global_rows,
                    lut_ptr, RADIUS, SIDE, FAR, SELF, TOKEN, PAD,
                )
                bias = _bias_tile(
                    table_base, bucket, offs_m, q_live, stride_tm, stride_tb,
                    ROW_TABLE,
                )

                scores = tl.dot(q, tl.trans(k)) * sm_scale + bias
                p = tl.math.exp2(
                    (scores - lse[:, None]) * 1.4426950408889634
                )
                pair_live = q_live[:, None] & k_live[None, :]
                p = tl.where(pair_live, p, 0.0)
                dp = tl.dot(do, tl.trans(v))
                ds = p * (dp - delta[:, None])
                dq_acc += tl.dot(ds.to(k_ptr.dtype.element_ty), k) * sm_scale

                # The table enters each score additively through one gather, so
                # its gradient is the score gradient's bucket histogram — kept
                # per query row when each row owns a table row, pooled over the
                # tile when they all share one.
                if ROW_TABLE:
                    for b in tl.static_range(0, TABLE_COLS):
                        bucket_grad = tl.sum(tl.where(bucket == b, ds, 0.0), axis=1)
                        dtable_acc += tl.where(
                            offs_b[None, :] == b, bucket_grad[:, None], 0.0
                        )
                else:
                    for b in tl.static_range(0, TABLE_COLS):
                        bucket_grad = tl.sum(
                            tl.sum(tl.where(bucket == b, ds, 0.0), axis=1),
                            axis=0,
                        )
                        dtable_acc += tl.where(offs_b == b, bucket_grad, 0.0)

            dq_acc = tl.where(q_live[:, None], dq_acc, 0.0)
            tl.store(dq_ptrs, dq_acc, mask=offs_m[:, None] < n_ctx)
            if ROW_TABLE:
                tl.store(dtable_ptrs, dtable_acc, mask=dtable_mask)
            else:
                tl.atomic_add(
                    dtable_ptr
                    + off_h * stride_dth
                    + offs_b * stride_dtb,
                    dtable_acc,
                    mask=offs_b < TABLE_COLS,
                )


    @triton.jit
    def _fused_attention_dkdv_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        coords_ptr,
        seq_lens_ptr,
        table_ptr,
        lut_ptr,
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
        stride_cp,
        stride_ct,
        stride_cc,
        stride_lp,
        stride_tp,
        stride_th,
        stride_tm,
        stride_tb,
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
        global_rows,
        RADIUS: tl.constexpr,
        SIDE: tl.constexpr,
        FAR: tl.constexpr,
        SELF: tl.constexpr,
        TOKEN: tl.constexpr,
        PAD: tl.constexpr,
        ROW_TABLE: tl.constexpr,
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
            table_base = table_ptr + off_p * stride_tp + off_h * stride_th
            coords_base = coords_ptr + off_p * stride_cp
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

                bucket = _pair_buckets(
                    q_q, q_r, k_q, k_r, offs_m, offs_n, k_live, global_rows,
                    lut_ptr, RADIUS, SIDE, FAR, SELF, TOKEN, PAD,
                )
                bias = _bias_tile(
                    table_base, bucket, offs_m, q_live, stride_tm, stride_tb,
                    ROW_TABLE,
                )

                scores = tl.dot(q, tl.trans(k)) * sm_scale + bias
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


def _bias_table(q: Tensor, bias_table: Tensor) -> Tensor:
    """Cast the learned rows once and append the finite PAD sentinel column.

    Layout of the last axis: orbits 0..47, FAR, SELF, TOKEN, then PAD —
    ``TABLE_WIDTH`` wide. Rank tells the two biases apart: ``(A, BIAS_ROWS)``
    is the static table, ``(P, A, T, BIAS_ROWS)`` the per-query-row one.
    """
    if bias_table.ndim not in (2, 4) or bias_table.shape[-1] != BIAS_ROWS:
        raise ValueError(
            f"bias_table must have shape (A, {BIAS_ROWS}) or "
            f"(P, A, T, {BIAS_ROWS}), got {tuple(bias_table.shape)}"
        )
    table = bias_table.to(q.dtype)
    pad = table.new_full(table.shape[:-1] + (1,), _PAD_BIAS)
    return torch.cat((table, pad), dim=-1)


def _bucket_index(
    coords: Tensor,
    seq_lens: Tensor,
    t: int,
    lut: Tensor,
    global_rows: int = 1,
):
    """The (P, T, T) bias-bucket index and (P, T) key validity — the dense
    twin of the kernels' ``_pair_buckets``."""
    dq = coords[:, :, None, 0] - coords[:, None, :, 0]
    dr = coords[:, :, None, 1] - coords[:, None, :, 1]
    distance = torch.maximum(dq.abs(), torch.maximum(dr.abs(), (dq + dr).abs()))
    cq = dq.clamp(-ORBIT_RADIUS, ORBIT_RADIUS) + ORBIT_RADIUS
    cr = dr.clamp(-ORBIT_RADIUS, ORBIT_RADIUS) + ORBIT_RADIUS
    orbit = lut.to(torch.long)[(cq * _LUT_SIDE + cr).long()]
    bucket = torch.where(distance <= ORBIT_RADIUS, orbit, FAR_BUCKET)

    rows = torch.arange(t, device=coords.device)
    bucket = torch.where(rows[:, None] == rows[None, :], SELF_BUCKET, bucket)
    token = (rows[:, None] < global_rows) | (rows[None, :] < global_rows)
    bucket = torch.where(token, TOKEN_BUCKET, bucket)
    valid = rows[None, :] < seq_lens[:, None]
    bucket = torch.where(valid[:, None, :], bucket, PAD_BUCKET)
    return bucket, valid


def _apply_reference(q: Tensor, k: Tensor, v: Tensor, mask: Tensor, valid: Tensor) -> Tensor:
    result = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    result = result.masked_fill(~valid[:, None, :, None], 0)

    # Match empty_like(q)'s preserved strides in every dispatch path. The
    # model's following head-to-row transpose can then remain a view.
    out = torch.empty_like(q)
    out.copy_(result)
    return out


def _table_mask(table: Tensor, bucket: Tensor) -> Tensor:
    """The dense ``(P, A, T, T)`` bias: each pair reads its bucket, from the
    head's one row or — for a per-row table — from its own query row."""
    if table.ndim == 2:
        return table[:, bucket].permute(1, 0, 2, 3)
    heads = table.shape[1]
    return torch.gather(table, 3, bucket.unsqueeze(1).expand(-1, heads, -1, -1))


def _attention_reference_table(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    coords: Tensor,
    seq_lens: Tensor,
    table: Tensor,
    lut: Tensor,
    global_rows: int,
) -> Tensor:
    """The dense formulation used by CPU, failed launches, and recompute."""
    _, _, t, _ = q.shape
    if table.shape[-1] != TABLE_WIDTH:
        raise ValueError(f"bias table width {table.shape[-1]} != {TABLE_WIDTH}")
    bucket, valid = _bucket_index(coords, seq_lens, t, lut, global_rows)
    return _apply_reference(q, k, v, _table_mask(table, bucket), valid)


def _attention_reference(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    coords: Tensor,
    seq_lens: Tensor,
    bias_table: Tensor,
    lut: Tensor,
    global_rows: int = 1,
) -> Tensor:
    """Reference attention over either bias width — the static ``(A,
    BIAS_ROWS)`` rows or the per-query-row ``(P, A, T, BIAS_ROWS)`` table."""
    return _attention_reference_table(
        q, k, v, coords, seq_lens, _bias_table(q, bias_table), lut, global_rows
    )


def _shape_key(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    coords: Tensor,
    seq_lens: Tensor,
    table: Tensor,
    global_rows: int,
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
        global_rows,
    )


def _validate_inputs(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    coords: Tensor,
    seq_lens: Tensor,
    table: Tensor,
    lut: Tensor,
    global_rows: int,
) -> None:
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must have the same (P, A, T, D) shape")
    p, heads, t, _ = q.shape
    if coords.shape != (p, t, 2) or coords.dtype != torch.int32:
        raise ValueError("coords must be int32 with shape (P, T, 2)")
    if seq_lens.shape != (p,) or seq_lens.dtype != torch.int32:
        raise ValueError("seq_lens must be int32 with shape (P,)")
    if global_rows < 1 or global_rows > t:
        raise ValueError(
            f"global_rows must be in [1, {t}], got {global_rows}"
        )
    if table.shape not in ((heads, TABLE_WIDTH), (p, heads, t, TABLE_WIDTH)):
        raise ValueError(
            f"bias table must have shape ({heads}, {TABLE_WIDTH}) or "
            f"({p}, {heads}, {t}, {TABLE_WIDTH}), got {tuple(table.shape)}"
        )
    if lut.shape != (_LUT_SIDE * _LUT_SIDE,) or lut.dtype != torch.int32:
        raise ValueError(
            f"orbit lut must be int32 with shape ({_LUT_SIDE * _LUT_SIDE},)"
        )
    tensors = (k, v, coords, seq_lens, table, lut)
    if any(x.device != q.device for x in tensors):
        raise ValueError("all attention inputs must be on one device")
    if k.dtype != q.dtype or v.dtype != q.dtype or table.dtype != q.dtype:
        raise ValueError("q, k, v, and the bias table must have one dtype")


def _orbit_constexprs(table: Tensor) -> dict:
    return dict(
        RADIUS=ORBIT_RADIUS,
        SIDE=_LUT_SIDE,
        FAR=FAR_BUCKET,
        SELF=SELF_BUCKET,
        TOKEN=TOKEN_BUCKET,
        PAD=PAD_BUCKET,
        ROW_TABLE=table.ndim == 4,
    )


def _table_strides(table: Tensor) -> tuple[int, ...]:
    """``(P, A, T, R)`` strides for either table rank. The static table is one
    row per head, broadcast over positions and query rows, so those two
    strides are zero — and ``ROW_TABLE=False`` compiles their loads away."""
    if table.ndim == 2:
        return (0, table.stride(0), 0, table.stride(1))
    return tuple(table.stride())


def _launch_triton(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    coords: Tensor,
    seq_lens: Tensor,
    table: Tensor,
    lut: Tensor,
    global_rows: int,
) -> tuple[Tensor, Tensor]:
    p, heads, t, head_dim = q.shape
    out = torch.empty_like(q)
    lse = torch.empty((p, heads, t), dtype=torch.float32, device=q.device)
    grid = (triton.cdiv(t, _BLOCK_M), p * heads)
    _fused_attention_kernel[grid](
        q,
        k,
        v,
        coords,
        seq_lens,
        table,
        lut,
        out,
        lse,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *coords.stride(),
        *seq_lens.stride(),
        *_table_strides(table),
        *out.stride(),
        *lse.stride(),
        heads,
        t,
        1.0 / math.sqrt(head_dim),
        global_rows,
        **_orbit_constexprs(table),
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
    coords: Tensor,
    seq_lens: Tensor,
    table: Tensor,
    lut: Tensor,
    out: Tensor,
    lse: Tensor,
    grad_out: Tensor,
    global_rows: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    p, heads, t, head_dim = q.shape
    delta = torch.empty((p, heads, t), dtype=torch.float32, device=q.device)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    if table.ndim == 4:
        # Every entry is written by the one dq program that owns its query row.
        dtable = torch.empty(table.shape, dtype=table.dtype, device=table.device)
    else:
        # Every dq program adds its tile's histogram into the head's one row.
        dtable = torch.zeros(table.shape, dtype=torch.float32, device=table.device)
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
        coords,
        seq_lens,
        table,
        lut,
        lse,
        delta,
        grad_out,
        dq,
        dtable,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *coords.stride(),
        *seq_lens.stride(),
        *_table_strides(table),
        *lse.stride(),
        *delta.stride(),
        *grad_out.stride(),
        *dq.stride(),
        *_table_strides(dtable),
        heads,
        t,
        1.0 / math.sqrt(head_dim),
        global_rows,
        **_orbit_constexprs(table),
        HEAD_DIM=head_dim,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_BUCKETS=triton.next_power_of_2(table.shape[-1]),
        TABLE_COLS=table.shape[-1],
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    _fused_attention_dkdv_kernel[key_grid](
        q,
        k,
        v,
        coords,
        seq_lens,
        table,
        lut,
        lse,
        delta,
        grad_out,
        dk,
        dv,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *coords.stride(),
        *seq_lens.stride(),
        *_table_strides(table),
        *lse.stride(),
        *delta.stride(),
        *grad_out.stride(),
        *dk.stride(),
        *dv.stride(),
        heads,
        t,
        1.0 / math.sqrt(head_dim),
        global_rows,
        **_orbit_constexprs(table),
        HEAD_DIM=head_dim,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    return dq, dk, dv, dtable.to(table.dtype)


@torch.library.custom_op("mantisnet::fused_attention", mutates_args=())
def _fused_attention_op(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    coords: Tensor,
    seq_lens: Tensor,
    table: Tensor,
    lut: Tensor,
    global_rows: int,
) -> tuple[Tensor, Tensor]:
    _validate_inputs(q, k, v, coords, seq_lens, table, lut, global_rows)
    supported = (
        triton is not None
        and q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16)
        and q.shape[-1] in (16, 32, 64)
    )
    if not supported:
        out = _attention_reference_table(
            q, k, v, coords, seq_lens, table, lut, global_rows
        )
        return out, torch.empty(0, dtype=torch.float32, device=q.device)

    key = _shape_key(q, k, v, coords, seq_lens, table, global_rows)
    if key in _FAILED_SHAPES:
        out = _attention_reference_table(
            q, k, v, coords, seq_lens, table, lut, global_rows
        )
        return out, torch.empty(0, dtype=torch.float32, device=q.device)
    try:
        return _launch_triton(q, k, v, coords, seq_lens, table, lut, global_rows)
    except Exception as exc:
        _FAILED_SHAPES[key] = f"{type(exc).__name__}: {exc}"
        warnings.warn(
            "fused attention failed for "
            f"shape={tuple(q.shape)}, dtype={q.dtype}; using SDPA for this "
            f"shape: {_FAILED_SHAPES[key]}",
            RuntimeWarning,
            stacklevel=2,
        )
        out = _attention_reference_table(
            q, k, v, coords, seq_lens, table, lut, global_rows
        )
        return out, torch.empty(0, dtype=torch.float32, device=q.device)


@_fused_attention_op.register_fake
def _(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    coords: Tensor,
    seq_lens: Tensor,
    table: Tensor,
    lut: Tensor,
    global_rows: int,
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
    q, k, v, coords, seq_lens, table, lut, global_rows = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, coords, seq_lens, table, lut, out, lse)
    ctx.global_rows = global_rows
    ctx.mark_non_differentiable(lse)


def _backward(ctx, grad_out: Tensor):
    q, k, v, coords, seq_lens, table, lut = ctx.saved_tensors
    t = q.shape[2]
    global_rows = ctx.global_rows
    row_table = table.ndim == 4
    bucket, valid = _bucket_index(coords, seq_lens, t, lut, global_rows)
    with torch.enable_grad():
        q_ = q.detach().requires_grad_(True)
        k_ = k.detach().requires_grad_(True)
        v_ = v.detach().requires_grad_(True)
        if row_table:
            # Over a table this wide the gather's scatter is per query row and
            # no longer serializes, so differentiating the gather itself *is*
            # the per-row bucket histogram the Triton path accumulates.
            # Gathering out of an fp32 copy keeps a long key run from rounding
            # away in the compute dtype.
            wrt = table.detach().float().requires_grad_(True)
            mask = _table_mask(wrt, bucket).to(q.dtype)
        else:
            # The dense bias enters the scores additively, so its gradient is
            # the per-pair score gradient. Differentiating the mask directly
            # (instead of the table gather) keeps the scatter out of autograd:
            # reducing P*A*T*T gradients into a table this small serializes on
            # atomics.
            wrt = _table_mask(table.detach(), bucket).requires_grad_(True)
            mask = wrt
        # The per-row gather leaves the mask contiguous, which sends SDPA's
        # heuristics to the memory-efficient backend — whose backward refuses
        # to differentiate a dense bias at some sequence lengths. Pin the row
        # case to the math backend, the formula itself; a fallback must not
        # rest on an accident of the mask's strides. The static case keeps the
        # dispatch it has always had, so its gradients are unchanged.
        with sdpa_kernel(SDPBackend.MATH) if row_table else contextlib.nullcontext():
            out = _apply_reference(q_, k_, v_, mask, valid)
            dq, dk, dv, dwrt = torch.autograd.grad(out, (q_, k_, v_, wrt), grad_out)
    if row_table:
        return dq, dk, dv, None, None, dwrt.to(table.dtype), None, None
    grads = dwrt.float()
    dtable = torch.stack(
        [
            (grads * (bucket == b).unsqueeze(1)).sum(dim=(0, 2, 3))
            for b in range(table.shape[1])
        ],
        dim=1,
    ).to(table.dtype)
    return dq, dk, dv, None, None, dtable, None, None


class _DenseBackwardContext:
    def __init__(self, saved_tensors: tuple[Tensor, ...], global_rows: int) -> None:
        self.saved_tensors = saved_tensors
        self.global_rows = global_rows


def _dense_backward_below_autograd(
    saved_tensors: tuple[Tensor, ...],
    global_rows: int,
    grad_out: Tensor,
):
    with torch._C._ForceDispatchKeyGuard(
        _DENSE_BACKWARD_INCLUDE_KEYS, _DENSE_BACKWARD_EXCLUDE_KEYS
    ):
        return _backward(_DenseBackwardContext(saved_tensors, global_rows), grad_out)


def _backward_shape_key(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    coords: Tensor,
    seq_lens: Tensor,
    table: Tensor,
    grad_out: Tensor,
    global_rows: int,
) -> tuple[object, ...]:
    return _shape_key(q, k, v, coords, seq_lens, table, global_rows) + (
        grad_out.dtype,
        tuple(grad_out.stride()),
    )


@torch.library.custom_op("mantisnet::fused_attention_backward", mutates_args=())
def _fused_attention_backward_op(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    coords: Tensor,
    seq_lens: Tensor,
    table: Tensor,
    lut: Tensor,
    out: Tensor,
    lse: Tensor,
    grad_out: Tensor,
    global_rows: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    _validate_inputs(q, k, v, coords, seq_lens, table, lut, global_rows)
    saved = (q, k, v, coords, seq_lens, table, lut)
    # The outer autograd formula handles this branch in eager mode. Keep the
    # sentinel check inside the opaque op as well: a compiled graph may have
    # been traced for Triton before a runtime forward launch marks the shape
    # failed and returns the dense sentinel.
    if lse.numel() == 0:
        dq, dk, dv, _, _, dtable, _, _ = _dense_backward_below_autograd(
            saved, global_rows, grad_out
        )
        return dq, dk, dv, dtable

    key = _backward_shape_key(q, k, v, coords, seq_lens, table, grad_out, global_rows)
    if key in _FAILED_BACKWARD_SHAPES:
        dq, dk, dv, _, _, dtable, _, _ = _dense_backward_below_autograd(
            saved, global_rows, grad_out
        )
        return dq, dk, dv, dtable
    try:
        return _launch_triton_backward(
            q, k, v, coords, seq_lens, table, lut, out, lse, grad_out, global_rows
        )
    except Exception as exc:
        _FAILED_BACKWARD_SHAPES[key] = f"{type(exc).__name__}: {exc}"
        warnings.warn(
            "fused attention backward failed for "
            f"shape={tuple(q.shape)}, dtype={q.dtype}; using dense backward "
            f"for this shape: {_FAILED_BACKWARD_SHAPES[key]}",
            RuntimeWarning,
            stacklevel=2,
        )
        dq, dk, dv, _, _, dtable, _, _ = _dense_backward_below_autograd(
            saved, global_rows, grad_out
        )
        return dq, dk, dv, dtable


@_fused_attention_backward_op.register_fake
def _(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    coords: Tensor,
    seq_lens: Tensor,
    table: Tensor,
    lut: Tensor,
    out: Tensor,
    lse: Tensor,
    grad_out: Tensor,
    global_rows: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
        torch.empty(table.shape, dtype=table.dtype, device=table.device),
    )


def _dispatch_backward(ctx, grad_out: Tensor, _grad_lse: Tensor | None):
    q, k, v, coords, seq_lens, table, lut, out, lse = ctx.saved_tensors
    global_rows = ctx.global_rows
    if lse.numel() == 0:
        return _backward(
            _DenseBackwardContext((q, k, v, coords, seq_lens, table, lut), global_rows),
            grad_out,
        )
    dq, dk, dv, dtable = _fused_attention_backward_op(
        q, k, v, coords, seq_lens, table, lut, out, lse, grad_out, global_rows
    )
    return dq, dk, dv, None, None, dtable, None, None


_fused_attention_op.register_autograd(
    _dispatch_backward, setup_context=_setup_context
)


def fused_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    coords: Tensor,
    seq_lens: Tensor,
    bias_table: Tensor,
    lut: Tensor,
    global_rows: int = 1,
) -> Tensor:
    """Orbit-biased attention over ``[latent rows; stones]`` per position.

    ``bias_table`` is either the ``(A, BIAS_ROWS)`` table of per-head bias
    rows (:func:`compose_bias_table`) or, with the model's ``orbit_vectors``
    knob on, the ``(P, A, T, BIAS_ROWS)`` table of each query row
    (:func:`compose_row_bias_table`). Rank selects the kernel; ``lut`` is the
    device-resident displacement table from :func:`orbit_lut`.
    """
    out, _ = _fused_attention_op(
        q, k, v, coords, seq_lens, _bias_table(q, bias_table), lut, global_rows
    )
    return out
