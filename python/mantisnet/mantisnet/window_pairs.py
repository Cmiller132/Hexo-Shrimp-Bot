"""Window-pair relation tables and attention for §5.1c.

Like the cell-pass relay tables, these are derived once at collation and the
forward performs no index discovery. Unlike them, the derivation is opt-in
(``pairs=True`` on the collators): only a ``window_attention`` model consumes
the tables, and the join is heavy enough that arms which never read it should
not pay for it at every chunk.

Two windows relate in exactly one of two game-mechanical ways, and both are
functions of the identity triples ``(axis, start_q, start_r)`` alone:

- **Colinear** (classes ``0..10``): the same line, signed start offset ``o``.
  A reflection reverses the line, flipping the sign, so the class is
  ``|o| - 1`` for ``|o| <= 11`` — overlap at 1..5, a gap of at most five
  cells at 6..11. Beyond that the spans cannot participate in one forcing
  sequence, and there is no edge.
- **Crossing** (classes ``11..46``): two non-parallel hex-axis lines always
  meet at exactly one lattice cell (every axis-pair determinant is ±1). The
  relation is where that cell sits in each window's span parameter — slot
  ``t`` in the destination, ``u`` in the source — including up to five cells
  beyond an end, mirroring the colinear reach. Each side folds to
  ``{in0, in1, in2, out1, out2, out3+}``; the class is
  ``11 + fold(t) * 6 + fold(u)``.
- **SELF** (class ``47``): one loop per window, so every softmax segment is
  nonempty.

D6-invariance of the crossing fold: a board symmetry permutes the three axes
and may reverse the two lines' slot parameterizations *independently* (a 60°
rotation maps the axes with mixed direction flips), so an invariant class may
not couple the two sides' orientations. Reversal maps ``t`` to ``5 - t``,
which preserves the in-span fold ``min(t, 5 - t)`` and the out-of-span
distance ``max(-t, t - 5)`` — each side's fold is invariant on its own, and
the product therefore under every group element.

Edge enumeration is a sorted join, cell-key style: each window claims its 16
line cells ``t ∈ -5..10``; two windows cross with mutual reach exactly when
they claim one common cell with different axes, and they do so at exactly one
cell, so the join yields each directed edge once. Colinear edges join on the
line key instead. Both joins are per-position by construction (the keys pack
the position index).

The edge set is served in three CSR views, like the relay's: by destination
(forward and ``dq``), by source (``dk``/``dv``), and by class (``dbias``).
The CUDA kernels are flash-style: the forward keeps one destination segment's
online softmax in registers and saves only the per-``(window, head)`` max and
denominator; the backward recomputes every per-edge quantity from those stats
inside deterministic segment sweeps — no atomics, and nothing of size
``(E, head_dim)`` ever reaches memory. The eager slice composition over the
destination view is the parity reference and serves CPU tensors and failed
launches; its CUDA ``index_add_`` accumulation is atomic and nondeterministic,
which is acceptable for a fallback and wrong for production.
"""

from __future__ import annotations

import math
import warnings
from typing import NamedTuple

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

# 11 colinear + 36 crossing + SELF.
WA_CLASSES = 48
_SELF = 47
_REACH = 5  # cells beyond a span end, matching the colinear gap of <= 5
_MAX_OFFSET = 11

# fold(t) for t in -_REACH..5+_REACH, indexed by t + _REACH: in-span slots to
# min(t, 5 - t), out-of-span to 2 + min(distance, 3).
_FOLD = torch.tensor(
    [2 + min(d, 3) for d in range(_REACH, 0, -1)]
    + [min(t, 5 - t) for t in range(6)]
    + [2 + min(d, 3) for d in range(1, _REACH + 1)],
    dtype=torch.long,
)

# Unit steps of the engine's axes, canonical order Q, R, QR (builder.AXES).
_AXES = torch.tensor([[1, 0], [0, 1], [1, -1]], dtype=torch.long)

# Key packing: coordinates are i16-bounded, so 17 bits per component after an
# offset is collision-free, and the position index rides above them.
_KOFF = 1 << 16
_KSPAN = 1 << 17


class PairTables(NamedTuple):
    """The three CSR views of one batch's directed §5.1c edges."""

    ptr: Tensor  # (N_w + 1,) destination-major run starts
    src: Tensor  # (E,) source window per edge, destination order
    cls: Tensor  # (E,) relation class per edge, destination order
    sptr: Tensor  # (N_w + 1,) source-major run starts
    sdst: Tensor  # (E,) destination window per edge, source order
    scls: Tensor  # (E,) relation class per edge, source order
    cptr: Tensor  # (WA_CLASSES + 1,) class-major run starts
    cedge: Tensor  # (E,) destination-order edge ids, class order


def pair_tables(window_id: Tensor, window_pos: Tensor) -> PairTables:
    """Derive every §5.1c edge view from the batch's window identities.

    ``window_id`` is the batch's ``(N_w, 3)`` identity table and
    ``window_pos`` the ``(N_w,)`` position of each window.
    """
    if window_id.ndim != 2 or window_id.shape[1] != 3:
        raise ValueError("window_id must have shape (N_w, 3)")
    if window_pos.shape != window_id.shape[:1]:
        raise ValueError("window_pos must have one entry per window")
    n_w = window_id.shape[0]
    axis, sq, sr = window_id.unbind(1)

    dsts, srcs, classes = [], [], []

    # Colinear: sort by (position, axis, line, position-on-line); partners sit
    # within 11 slots of one another, and starts on a line are distinct, so
    # eleven shifted comparisons enumerate every pair once.
    line = torch.where(axis == 0, sr, torch.where(axis == 1, sq, sq + sr))
    pos_on = torch.where(axis == 1, sr, sq)
    key = ((window_pos * 4 + axis) * _KSPAN + (line + _KOFF)) * _KSPAN + (
        pos_on + _KOFF
    )
    order = torch.argsort(key)
    skey = key[order]
    group = skey // _KSPAN
    spos = skey % _KSPAN
    for shift in range(1, _MAX_OFFSET + 1):
        if shift >= n_w:
            break
        near, far = order[:-shift], order[shift:]
        delta = spos[shift:] - spos[:-shift]
        ok = (group[:-shift] == group[shift:]) & (delta <= _MAX_OFFSET)
        near, far, cls = near[ok], far[ok], delta[ok] - 1
        dsts.append(torch.cat([near, far]))
        srcs.append(torch.cat([far, near]))
        classes.append(torch.cat([cls, cls]))

    # Crossing: join the claimed line cells. Runs share a (position, cell)
    # key; a pair with different axes in one run is a crossing within reach.
    t_ext = torch.arange(-_REACH, 6 + _REACH, device=window_id.device)
    vec = _AXES.to(window_id.device)[axis]  # (N_w, 2)
    cq = sq[:, None] + t_ext[None, :] * vec[:, 0:1]
    cr = sr[:, None] + t_ext[None, :] * vec[:, 1:2]
    span = t_ext.shape[0]
    ckey = (
        (window_pos[:, None] * _KSPAN + (cq + _KOFF)) * _KSPAN + (cr + _KOFF)
    ).reshape(-1)
    cwin = (
        torch.arange(n_w, device=window_id.device)[:, None].expand(-1, span).reshape(-1)
    )
    ct = t_ext[None, :].expand(n_w, -1).reshape(-1)
    corder = torch.argsort(ckey)
    rkey = ckey[corder]
    rwin = cwin[corder]
    rt = ct[corder]
    raxis = axis[rwin]
    fold = _FOLD.to(window_id.device)
    if rkey.numel():
        run_lengths = torch.unique_consecutive(rkey, return_counts=True)[1]
        for shift in range(1, int(run_lengths.max())):
            ok = (rkey[:-shift] == rkey[shift:]) & (raxis[:-shift] != raxis[shift:])
            wi, wj = rwin[:-shift][ok], rwin[shift:][ok]
            fi, fj = fold[rt[:-shift][ok] + _REACH], fold[rt[shift:][ok] + _REACH]
            dsts.append(torch.cat([wi, wj]))
            srcs.append(torch.cat([wj, wi]))
            classes.append(torch.cat([11 + fi * 6 + fj, 11 + fj * 6 + fi]))

    # SELF.
    loop = torch.arange(n_w, device=window_id.device)
    dsts.append(loop)
    srcs.append(loop)
    classes.append(torch.full((n_w,), _SELF, dtype=torch.long, device=window_id.device))

    dst = torch.cat(dsts)
    src = torch.cat(srcs)
    cls = torch.cat(classes)
    order = torch.argsort(dst, stable=True)
    dst, src, cls = dst[order], src[order], cls[order]
    steps = torch.arange(n_w + 1, device=window_id.device)
    ptr = torch.searchsorted(dst, steps)

    sorder = torch.argsort(src, stable=True)
    sptr = torch.searchsorted(src[sorder], steps)
    sdst, scls = dst[sorder], cls[sorder]

    corder = torch.argsort(cls, stable=True)
    cptr = torch.searchsorted(
        cls[corder], torch.arange(WA_CLASSES + 1, device=window_id.device)
    )
    return PairTables(ptr, src, cls, sptr, sdst, scls, cptr, corder)


# The eager fallback walks the destination view in fixed slices so nothing of
# size (E, head_dim) is ever materialized whole.
_EDGE_SLICE = 2_000_000

# Kernel launch geometry: segments average a few dozen edges, so one 32-edge
# tile per iteration; the class gradient is the relay's split-partial shape.
_WA_BLOCK_E = 32
_WA_NUM_WARPS = 2
_BIAS_SPLITS = 64
_BIAS_BLOCK_E = 128

_FAILED_SHAPES: dict[tuple[object, ...], str] = {}
_FAILED_BACKWARD_SHAPES: dict[tuple[object, ...], str] = {}


if triton is not None:

    @triton.jit
    def _wa_forward_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        bias_ptr,
        ptr_ptr,
        src_ptr,
        cls_ptr,
        out_ptr,
        m_ptr,
        l_ptr,
        scale,
        stride_bias,
        HEADS: tl.constexpr,
        HD: tl.constexpr,
        BLOCK_HD: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        # One (destination window, head) per program: an online softmax over
        # the destination's edge run, saving only the max and denominator.
        w = tl.program_id(0)
        a = tl.program_id(1)
        offs = tl.arange(0, BLOCK_HD)
        live = offs < HD
        row = (w * HEADS + a) * HD
        q_row = tl.load(q_ptr + row + offs, mask=live, other=0.0).to(tl.float32)
        start = tl.load(ptr_ptr + w)
        end = tl.load(ptr_ptr + w + 1)
        m = -float("inf")
        l = 0.0
        acc = tl.zeros([BLOCK_HD], dtype=tl.float32)
        for lo in tl.range(start, end, BLOCK_E):
            eids = lo + tl.arange(0, BLOCK_E)
            ok = eids < end
            s_idx = tl.load(src_ptr + eids, mask=ok, other=0)
            c_idx = tl.load(cls_ptr + eids, mask=ok, other=0)
            k_tile = tl.load(
                k_ptr + (s_idx[:, None] * HEADS + a) * HD + offs[None, :],
                mask=ok[:, None] & live[None, :],
                other=0.0,
            ).to(tl.float32)
            score = tl.sum(q_row[None, :] * k_tile, axis=1) * scale
            score += tl.load(
                bias_ptr + a * stride_bias + c_idx, mask=ok, other=0.0
            ).to(tl.float32)
            score = tl.where(ok, score, -float("inf"))
            m_new = tl.maximum(m, tl.max(score, axis=0))
            rescale = tl.exp(m - m_new)
            p = tl.exp(score - m_new)
            l = l * rescale + tl.sum(p, axis=0)
            v_tile = tl.load(
                v_ptr + (s_idx[:, None] * HEADS + a) * HD + offs[None, :],
                mask=ok[:, None] & live[None, :],
                other=0.0,
            ).to(tl.float32)
            acc = acc * rescale + tl.sum(p[:, None] * v_tile, axis=0)
            m = m_new
        tl.store(out_ptr + row + offs, acc / l, mask=live)
        tl.store(m_ptr + w * HEADS + a, m)
        tl.store(l_ptr + w * HEADS + a, l)

    @triton.jit
    def _wa_dq_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        bias_ptr,
        go_ptr,
        ptr_ptr,
        src_ptr,
        cls_ptr,
        m_ptr,
        l_ptr,
        delta_ptr,
        dq_ptr,
        dscore_ptr,
        scale,
        stride_bias,
        HEADS: tl.constexpr,
        HD: tl.constexpr,
        BLOCK_HD: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        # Destination sweep: recompute alpha from the saved stats, emit the
        # per-edge dscore for the bias gradient, and accumulate dq in
        # registers — one deterministic pass, no atomics.
        w = tl.program_id(0)
        a = tl.program_id(1)
        offs = tl.arange(0, BLOCK_HD)
        live = offs < HD
        row = (w * HEADS + a) * HD
        q_row = tl.load(q_ptr + row + offs, mask=live, other=0.0).to(tl.float32)
        go_row = tl.load(go_ptr + row + offs, mask=live, other=0.0).to(tl.float32)
        m = tl.load(m_ptr + w * HEADS + a)
        l = tl.load(l_ptr + w * HEADS + a)
        delta = tl.load(delta_ptr + w * HEADS + a)
        start = tl.load(ptr_ptr + w)
        end = tl.load(ptr_ptr + w + 1)
        acc = tl.zeros([BLOCK_HD], dtype=tl.float32)
        for lo in tl.range(start, end, BLOCK_E):
            eids = lo + tl.arange(0, BLOCK_E)
            ok = eids < end
            s_idx = tl.load(src_ptr + eids, mask=ok, other=0)
            c_idx = tl.load(cls_ptr + eids, mask=ok, other=0)
            k_tile = tl.load(
                k_ptr + (s_idx[:, None] * HEADS + a) * HD + offs[None, :],
                mask=ok[:, None] & live[None, :],
                other=0.0,
            ).to(tl.float32)
            score = tl.sum(q_row[None, :] * k_tile, axis=1) * scale
            score += tl.load(
                bias_ptr + a * stride_bias + c_idx, mask=ok, other=0.0
            ).to(tl.float32)
            alpha = tl.exp(score - m) / l
            v_tile = tl.load(
                v_ptr + (s_idx[:, None] * HEADS + a) * HD + offs[None, :],
                mask=ok[:, None] & live[None, :],
                other=0.0,
            ).to(tl.float32)
            dalpha = tl.sum(go_row[None, :] * v_tile, axis=1)
            ds = tl.where(ok, alpha * (dalpha - delta), 0.0)
            tl.store(dscore_ptr + eids * HEADS + a, ds, mask=ok)
            acc += tl.sum(ds[:, None] * k_tile, axis=0)
        element = dq_ptr.dtype.element_ty
        tl.store(dq_ptr + row + offs, (acc * scale).to(element), mask=live)

    @triton.jit
    def _wa_dkdv_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        bias_ptr,
        go_ptr,
        sptr_ptr,
        sdst_ptr,
        scls_ptr,
        m_ptr,
        l_ptr,
        delta_ptr,
        dk_ptr,
        dv_ptr,
        scale,
        stride_bias,
        HEADS: tl.constexpr,
        HD: tl.constexpr,
        BLOCK_HD: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        # Source sweep over the same edges: this program's k and v rows are
        # fixed, the destinations' rows and stats are gathered per edge.
        s = tl.program_id(0)
        a = tl.program_id(1)
        offs = tl.arange(0, BLOCK_HD)
        live = offs < HD
        row = (s * HEADS + a) * HD
        k_row = tl.load(k_ptr + row + offs, mask=live, other=0.0).to(tl.float32)
        v_row = tl.load(v_ptr + row + offs, mask=live, other=0.0).to(tl.float32)
        start = tl.load(sptr_ptr + s)
        end = tl.load(sptr_ptr + s + 1)
        acc_k = tl.zeros([BLOCK_HD], dtype=tl.float32)
        acc_v = tl.zeros([BLOCK_HD], dtype=tl.float32)
        for lo in tl.range(start, end, BLOCK_E):
            eids = lo + tl.arange(0, BLOCK_E)
            ok = eids < end
            d_idx = tl.load(sdst_ptr + eids, mask=ok, other=0)
            c_idx = tl.load(scls_ptr + eids, mask=ok, other=0)
            q_tile = tl.load(
                q_ptr + (d_idx[:, None] * HEADS + a) * HD + offs[None, :],
                mask=ok[:, None] & live[None, :],
                other=0.0,
            ).to(tl.float32)
            go_tile = tl.load(
                go_ptr + (d_idx[:, None] * HEADS + a) * HD + offs[None, :],
                mask=ok[:, None] & live[None, :],
                other=0.0,
            ).to(tl.float32)
            m_e = tl.load(m_ptr + d_idx * HEADS + a, mask=ok, other=0.0)
            l_e = tl.load(l_ptr + d_idx * HEADS + a, mask=ok, other=1.0)
            delta_e = tl.load(delta_ptr + d_idx * HEADS + a, mask=ok, other=0.0)
            score = tl.sum(q_tile * k_row[None, :], axis=1) * scale
            score += tl.load(
                bias_ptr + a * stride_bias + c_idx, mask=ok, other=0.0
            ).to(tl.float32)
            alpha = tl.where(ok, tl.exp(score - m_e) / l_e, 0.0)
            dalpha = tl.sum(go_tile * v_row[None, :], axis=1)
            ds = alpha * (dalpha - delta_e)
            acc_k += tl.sum(ds[:, None] * q_tile, axis=0)
            acc_v += tl.sum(alpha[:, None] * go_tile, axis=0)
        element = dk_ptr.dtype.element_ty
        tl.store(dk_ptr + row + offs, (acc_k * scale).to(element), mask=live)
        tl.store(dv_ptr + row + offs, acc_v.to(element), mask=live)

    @triton.jit
    def _wa_bias_partial_kernel(
        dscore_ptr,
        cptr_ptr,
        cedge_ptr,
        partial_ptr,
        SPLITS: tl.constexpr,
        HEADS: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        # The relay's split-partial shape: a class run can own most of the
        # edge set, so each program sums one fixed slice of one run.
        cls = tl.program_id(0)
        part = tl.program_id(1)
        offs = tl.arange(0, BLOCK_H)
        live = offs < HEADS
        start = tl.load(cptr_ptr + cls)
        end = tl.load(cptr_ptr + cls + 1)
        per = (end - start + SPLITS - 1) // SPLITS
        lo = start + part * per
        hi = tl.minimum(lo + per, end)
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)
        for base in tl.range(lo, hi, BLOCK_E):
            entries = base + tl.arange(0, BLOCK_E)
            inside = entries < hi
            rows = tl.load(cedge_ptr + entries, mask=inside, other=0)
            acc += tl.sum(
                tl.load(
                    dscore_ptr + rows[:, None] * HEADS + offs[None, :],
                    mask=inside[:, None] & live[None, :],
                    other=0.0,
                ),
                axis=0,
            )
        tl.store(partial_ptr + (cls * SPLITS + part) * HEADS + offs, acc, mask=live)


def _edge_dst(ptr: Tensor) -> Tensor:
    n_w = ptr.shape[0] - 1
    return torch.repeat_interleave(
        torch.arange(n_w, device=ptr.device), ptr[1:] - ptr[:-1]
    )


def _validate_attention(q, k, v, bias, tables) -> None:
    ptr, src, cls, sptr, sdst, scls, cptr, cedge = tables
    if q.ndim != 3 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, v must share shape (N_w, heads, head_dim)")
    if bias.ndim != 2 or bias.shape[0] != q.shape[1]:
        raise ValueError("bias must have shape (heads, classes)")
    if ptr.shape[0] != q.shape[0] + 1 or sptr.shape[0] != q.shape[0] + 1:
        raise ValueError("ptr and sptr must have one row per window plus one")
    if cptr.shape[0] != bias.shape[1] + 1:
        raise ValueError("cptr must have one row per class plus one")
    edges = src.shape
    if any(t.shape != edges for t in (cls, sdst, scls, cedge)) or src.ndim != 1:
        raise ValueError("the three edge views must have one length")


def _reference_forward(q, k, v, bias, ptr, src, cls):
    """Sliced eager composition: the parity reference and the CPU path."""
    n_w, heads, hd = q.shape
    scale = 1.0 / math.sqrt(hd)
    dst = _edge_dst(ptr)
    e = src.shape[0]

    score = bias.t().float().index_select(0, cls)  # (E, heads)
    for a in range(heads):
        q_a, k_a = q[:, a], k[:, a]
        for lo in range(0, e, _EDGE_SLICE):
            sl = slice(lo, min(lo + _EDGE_SLICE, e))
            score[sl, a] += scale * (
                q_a.index_select(0, dst[sl]).float()
                * k_a.index_select(0, src[sl]).float()
            ).sum(-1)

    # Segment softmax per (destination, head); SELF edges keep every segment
    # nonempty, so neither the max identity nor a zero denominator can leak.
    m = score.new_full((n_w, heads), torch.finfo(torch.float32).min)
    m.index_reduce_(0, dst, score, "amax", include_self=True)
    alpha = (score - m.index_select(0, dst)).exp_()
    l = score.new_zeros((n_w, heads)).index_add_(0, dst, alpha)

    out = q.new_zeros((n_w, heads, hd), dtype=torch.float32)
    for a in range(heads):
        v_a = v[:, a]
        for lo in range(0, e, _EDGE_SLICE):
            sl = slice(lo, min(lo + _EDGE_SLICE, e))
            out[:, a].index_add_(
                0,
                dst[sl],
                alpha[sl, a, None] * v_a.index_select(0, src[sl]).float(),
            )
    out /= l.unsqueeze(-1)
    return out, m, l


def _reference_backward(q, k, v, bias, ptr, src, cls, m, l, delta, go):
    """Recompute alpha from the saved stats, then the four gradients."""
    n_w, heads, hd = q.shape
    scale = 1.0 / math.sqrt(hd)
    dst = _edge_dst(ptr)
    e = src.shape[0]

    alpha = bias.t().float().index_select(0, cls)
    for a in range(heads):
        q_a, k_a = q[:, a], k[:, a]
        for lo in range(0, e, _EDGE_SLICE):
            sl = slice(lo, min(lo + _EDGE_SLICE, e))
            alpha[sl, a] += scale * (
                q_a.index_select(0, dst[sl]).float()
                * k_a.index_select(0, src[sl]).float()
            ).sum(-1)
    alpha = (alpha - m.index_select(0, dst)).exp_()
    alpha /= l.index_select(0, dst)

    # dalpha, dv: one sliced sweep re-gathering v and the upstream rows.
    dalpha = alpha.new_empty((e, heads))
    dv = torch.zeros((n_w, heads, hd), dtype=torch.float32, device=q.device)
    for a in range(heads):
        v_a, go_a = v[:, a], go[:, a]
        for lo in range(0, e, _EDGE_SLICE):
            sl = slice(lo, min(lo + _EDGE_SLICE, e))
            go_rows = go_a.index_select(0, dst[sl])
            dalpha[sl, a] = (
                go_rows * v_a.index_select(0, src[sl]).float()
            ).sum(-1)
            dv[:, a].index_add_(0, src[sl], alpha[sl, a, None] * go_rows)

    # Softmax backward: dscore = alpha * (dalpha - delta[dst]), with delta
    # the (go · out) row sums — algebraically Σ alpha dalpha per segment.
    dscore = alpha * (dalpha - delta.index_select(0, dst))

    dbias = torch.zeros(
        (bias.shape[1], heads), dtype=torch.float32, device=bias.device
    ).index_add_(0, cls, dscore).t()

    dq = torch.zeros((n_w, heads, hd), dtype=torch.float32, device=q.device)
    dk = torch.zeros((n_w, heads, hd), dtype=torch.float32, device=q.device)
    for a in range(heads):
        q_a, k_a = q[:, a], k[:, a]
        for lo in range(0, e, _EDGE_SLICE):
            sl = slice(lo, min(lo + _EDGE_SLICE, e))
            weight = scale * dscore[sl, a, None]
            dq[:, a].index_add_(
                0, dst[sl], weight * k_a.index_select(0, src[sl]).float()
            )
            dk[:, a].index_add_(
                0, src[sl], weight * q_a.index_select(0, dst[sl]).float()
            )
    return (
        dq.to(q.dtype),
        dk.to(k.dtype),
        dv.to(v.dtype),
        dbias.contiguous().to(bias.dtype),
    )


def _shape_key(x: Tensor) -> tuple[object, ...]:
    return (x.device.type, x.device.index, x.dtype, x.shape[1], x.shape[2])


def _supported(q: Tensor, src: Tensor) -> bool:
    return (
        triton is not None
        and q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and q.shape[0] > 0
        and src.numel() > 0
    )


def _launch_forward(q, k, v, bias, ptr, src, cls):
    n_w, heads, hd = q.shape
    out = torch.empty((n_w, heads, hd), dtype=torch.float32, device=q.device)
    m = torch.empty((n_w, heads), dtype=torch.float32, device=q.device)
    l = torch.empty((n_w, heads), dtype=torch.float32, device=q.device)
    _wa_forward_kernel[(n_w, heads)](
        q,
        k,
        v,
        bias,
        ptr,
        src,
        cls,
        out,
        m,
        l,
        1.0 / math.sqrt(hd),
        bias.stride(0),
        HEADS=heads,
        HD=hd,
        BLOCK_HD=triton.next_power_of_2(hd),
        BLOCK_E=_WA_BLOCK_E,
        num_warps=_WA_NUM_WARPS,
    )
    return out, m, l


def _launch_backward(q, k, v, bias, tables, m, l, delta, go):
    ptr, src, cls, sptr, sdst, scls, cptr, cedge = tables
    n_w, heads, hd = q.shape
    scale = 1.0 / math.sqrt(hd)
    block_hd = triton.next_power_of_2(hd)
    dq = torch.empty_like(q)
    dscore = torch.empty((src.shape[0], heads), dtype=torch.float32, device=q.device)
    _wa_dq_kernel[(n_w, heads)](
        q,
        k,
        v,
        bias,
        go,
        ptr,
        src,
        cls,
        m,
        l,
        delta,
        dq,
        dscore,
        scale,
        bias.stride(0),
        HEADS=heads,
        HD=hd,
        BLOCK_HD=block_hd,
        BLOCK_E=_WA_BLOCK_E,
        num_warps=_WA_NUM_WARPS,
    )
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    _wa_dkdv_kernel[(n_w, heads)](
        q,
        k,
        v,
        bias,
        go,
        sptr,
        sdst,
        scls,
        m,
        l,
        delta,
        dk,
        dv,
        scale,
        bias.stride(0),
        HEADS=heads,
        HD=hd,
        BLOCK_HD=block_hd,
        BLOCK_E=_WA_BLOCK_E,
        num_warps=_WA_NUM_WARPS,
    )
    classes = bias.shape[1]
    partial = torch.empty(
        (classes * _BIAS_SPLITS, heads), dtype=torch.float32, device=q.device
    )
    _wa_bias_partial_kernel[(classes, _BIAS_SPLITS)](
        dscore,
        cptr,
        cedge,
        partial,
        SPLITS=_BIAS_SPLITS,
        HEADS=heads,
        BLOCK_H=triton.next_power_of_2(heads),
        BLOCK_E=_BIAS_BLOCK_E,
        num_warps=_WA_NUM_WARPS,
    )
    dbias = (
        partial.view(classes, _BIAS_SPLITS, heads).sum(dim=1).t().contiguous()
    ).to(bias.dtype)
    return dq, dk, dv, dbias


@torch.library.custom_op("mantisnet::window_attention", mutates_args=())
def _wa_op(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    bias: Tensor,
    ptr: Tensor,
    src: Tensor,
    cls: Tensor,
    sptr: Tensor,
    sdst: Tensor,
    scls: Tensor,
    cptr: Tensor,
    cedge: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    tables = (ptr, src, cls, sptr, sdst, scls, cptr, cedge)
    _validate_attention(q, k, v, bias, tables)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    if not _supported(q, src):
        return _reference_forward(q, k, v, bias, ptr, src, cls)
    key = _shape_key(q)
    if key in _FAILED_SHAPES:
        return _reference_forward(q, k, v, bias, ptr, src, cls)
    try:
        return _launch_forward(q, k, v, bias, ptr, src, cls)
    except Exception as exc:
        _FAILED_SHAPES[key] = f"{type(exc).__name__}: {exc}"
        warnings.warn(
            "window attention failed for "
            f"heads={q.shape[1]}, hd={q.shape[2]}, dtype={q.dtype}; slicing "
            f"instead for this shape: {_FAILED_SHAPES[key]}",
            RuntimeWarning,
            stacklevel=2,
        )
        return _reference_forward(q, k, v, bias, ptr, src, cls)


@_wa_op.register_fake
def _(q, k, v, bias, ptr, src, cls, sptr, sdst, scls, cptr, cedge):
    n_w, heads, hd = q.shape
    out = q.new_empty((n_w, heads, hd), dtype=torch.float32)
    m = q.new_empty((n_w, heads), dtype=torch.float32)
    l = q.new_empty((n_w, heads), dtype=torch.float32)
    return out, m, l


@torch.library.custom_op("mantisnet::window_attention_backward", mutates_args=())
def _wa_backward_op(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    bias: Tensor,
    ptr: Tensor,
    src: Tensor,
    cls: Tensor,
    sptr: Tensor,
    sdst: Tensor,
    scls: Tensor,
    cptr: Tensor,
    cedge: Tensor,
    out: Tensor,
    m: Tensor,
    l: Tensor,
    grad_out: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    go = grad_out.contiguous().float()
    # delta = Σ_hd go · out per (window, head) — algebraically the segment's
    # Σ alpha dalpha, so the softmax backward needs no first edge sweep.
    delta = (go * out).sum(-1)
    if not _supported(q, src):
        return _reference_backward(q, k, v, bias, ptr, src, cls, m, l, delta, go)
    key = _shape_key(q) + (grad_out.dtype,)
    if key in _FAILED_BACKWARD_SHAPES:
        return _reference_backward(q, k, v, bias, ptr, src, cls, m, l, delta, go)
    tables = (ptr, src, cls, sptr, sdst, scls, cptr, cedge)
    try:
        return _launch_backward(q, k, v, bias, tables, m, l, delta, go)
    except Exception as exc:
        _FAILED_BACKWARD_SHAPES[key] = f"{type(exc).__name__}: {exc}"
        warnings.warn(
            "window attention backward failed for "
            f"heads={q.shape[1]}, hd={q.shape[2]}, dtype={q.dtype}; slicing "
            f"instead for this shape: {_FAILED_BACKWARD_SHAPES[key]}",
            RuntimeWarning,
            stacklevel=2,
        )
        return _reference_backward(q, k, v, bias, ptr, src, cls, m, l, delta, go)


@_wa_backward_op.register_fake
def _(q, k, v, bias, ptr, src, cls, sptr, sdst, scls, cptr, cedge, out, m, l, grad_out):
    return (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
        torch.empty_like(bias),
    )


def _wa_setup_context(ctx, inputs, output) -> None:
    q, k, v, bias, ptr, src, cls, sptr, sdst, scls, cptr, cedge = inputs
    out, m, l = output
    ctx.save_for_backward(
        q, k, v, bias, ptr, src, cls, sptr, sdst, scls, cptr, cedge, out, m, l
    )


def _wa_dispatch_backward(ctx, grad_out, _grad_m, _grad_l):
    dq, dk, dv, dbias = _wa_backward_op(*ctx.saved_tensors, grad_out)
    return (dq, dk, dv, dbias) + (None,) * 8


_wa_op.register_autograd(_wa_dispatch_backward, setup_context=_wa_setup_context)


def edge_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    bias: Tensor,
    ptr: Tensor,
    src: Tensor,
    cls: Tensor,
    sptr: Tensor,
    sdst: Tensor,
    scls: Tensor,
    cptr: Tensor,
    cedge: Tensor,
) -> Tensor:
    """§5.1c attention over the edge views: fp32 ``(N_w, heads, head_dim)``."""
    out, _m, _l = _wa_op(q, k, v, bias, ptr, src, cls, sptr, sdst, scls, cptr, cedge)
    return out
