"""Window-pair relations and attention for section 5.1c, cell-mediated.

Relation classes from window identity triples ``(axis, start_q, start_r)``:

- **Colinear** (``0..10``): same line, class ``|offset| - 1``.
  Overlap at 1..5, gap at 6..11.
- **Crossing** (``11..46``): non-parallel lines meeting at one cell.
  Each side folds to ``{in0, in1, in2, out1, out2, out3+}``;
  class = ``11 + fold(t) * 6 + fold(u)``.  D6-invariant because
  each side's fold is invariant under line reversal independently.
- **SELF** (``47``): one loop per window.

The crossing set is never materialized as edges.  Two non-parallel lines
meet at exactly one cell, so a window's crossing partners are exactly the
other-axis claimants of its sixteen claimed cells (the six slots plus
``_REACH`` on each end).  ``wa_tables`` therefore derives a cell-major
claimant CSR — ``n_w * 16`` entries however dense the scope — plus a small
explicit colinear/self edge list, and the kernels walk claims instead of
edges: one program per window covering every head at once, the colinear
run (degree <= 23) and sixteen claim runs (claimants per cell <= 48:
sixteen spans on each of three axes), one online softmax across all of
them.  Under the mixed-window
scope this replaces tens of millions of
materialized edges with a few hundred MB of claims, and the per-edge
``dscore`` array of the old backward disappears: the bias gradient
accumulates per-program partial rows summed by a fixed torch reduction, so
every output stays run-to-run deterministic.

CUDA: flash-style kernels with online softmax in registers.  CPU and the
failure fallback: expand the claims to an explicit edge list and run the
sliced eager composition as the parity reference.
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
_CLAIM_SPAN = 6 + 2 * _REACH  # claimed cells per window

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


def _segment_pairs(counts: Tensor, slot: Tensor) -> tuple[Tensor, Tensor]:
    """Enumerate ``(i, j)`` with ``j`` in ``(i, i + counts[i]]`` per element."""
    first = torch.repeat_interleave(slot, counts)
    rank = torch.arange(first.shape[0], device=slot.device) - (
        counts.cumsum(0) - counts
    ).index_select(0, first)
    return first, first + 1 + rank


class WaTables(NamedTuple):
    """One batch's §5.1c structure: colinear edges and cell claims."""

    col_ptr: Tensor  # (N_w + 1,) destination-major colinear/self run starts
    col_src: Tensor  # (E_col,) source window per colinear edge
    col_cls: Tensor  # (E_col,) class per colinear edge (0..10, SELF)
    claim_lo: Tensor  # (N_w * 16,) claimant-run start of window w's k-th cell
    claim_hi: Tensor  # (N_w * 16,) claimant-run end
    cl_win: Tensor  # (N_w * 16,) claimant window, cell-major order
    cl_fold: Tensor  # (N_w * 16,) claimant's fold at that cell
    cl_axis: Tensor  # (N_w * 16,) claimant's axis
    waxis: Tensor  # (N_w,) each window's axis


def wa_tables(window_id: Tensor, window_pos: Tensor) -> WaTables:
    """Derive the §5.1c structure from the batch's window identities.

    ``window_id`` is the batch's ``(N_w, 3)`` identity table and
    ``window_pos`` the ``(N_w,)`` position of each window.
    """
    if window_id.ndim != 2 or window_id.shape[1] != 3:
        raise ValueError("window_id must have shape (N_w, 3)")
    if window_pos.shape != window_id.shape[:1]:
        raise ValueError("window_pos must have one entry per window")
    n_w = window_id.shape[0]
    device = window_id.device
    axis, sq, sr = window_id.unbind(1)

    # Colinear: sort by (position, axis, line, position-on-line); starts on a
    # line are distinct, so within a sorted group every pair at most 11 slots
    # apart is an edge. searchsorted bounds each start's partner run — the
    # offset cannot escape its group because the position-on-line rides the
    # low key bits with margin — and the runs enumerate without per-shift
    # rescans or compaction syncs.
    dsts, srcs, classes = [], [], []
    line = torch.where(axis == 0, sr, torch.where(axis == 1, sq, sq + sr))
    pos_on = torch.where(axis == 1, sr, sq)
    key = ((window_pos * 4 + axis) * _KSPAN + (line + _KOFF)) * _KSPAN + (
        pos_on + _KOFF
    )
    order = torch.argsort(key)
    skey = key[order]
    if n_w:
        slot = torch.arange(n_w, device=device)
        hi = torch.searchsorted(skey, skey + _MAX_OFFSET, right=True)
        first, second = _segment_pairs(hi - slot - 1, slot)
        near = order.index_select(0, first)
        far = order.index_select(0, second)
        cls = skey.index_select(0, second) - skey.index_select(0, first) - 1
        dsts.append(torch.cat([near, far]))
        srcs.append(torch.cat([far, near]))
        classes.append(torch.cat([cls, cls]))

    # SELF keeps every destination's softmax segment nonempty.
    loop = torch.arange(n_w, device=device)
    dsts.append(loop)
    srcs.append(loop)
    classes.append(torch.full((n_w,), _SELF, dtype=torch.long, device=device))

    dst = torch.cat(dsts)
    col_src = torch.cat(srcs)
    col_cls = torch.cat(classes)
    # Window ids fit int32, and radix passes scale with key width.
    corder = torch.argsort(dst.to(torch.int32), stable=True)
    dst = dst[corder]
    col_src = col_src[corder].contiguous()
    col_cls = col_cls[corder].contiguous()
    steps = torch.arange(n_w + 1, device=device)
    col_ptr = torch.searchsorted(dst, steps)

    # Claims: each window claims its six slots plus _REACH beyond each end.
    # Windows claiming one (position, cell) key form a sorted run; a crossing
    # pair shares exactly one cell, so the other-axis members of a claim's
    # run are exactly that claim's crossing partners — no edge enumeration.
    t_ext = torch.arange(-_REACH, 6 + _REACH, device=device)
    vec = _AXES.to(device)[axis]  # (N_w, 2)
    cq = sq[:, None] + t_ext[None, :] * vec[:, 0:1]
    cr = sr[:, None] + t_ext[None, :] * vec[:, 1:2]
    ckey = (
        (window_pos[:, None] * _KSPAN + (cq + _KOFF)) * _KSPAN + (cr + _KOFF)
    ).reshape(-1)
    n_claims = ckey.shape[0]
    cwin = loop[:, None].expand(-1, _CLAIM_SPAN).reshape(-1)
    ct = t_ext[None, :].expand(n_w, -1).reshape(-1)
    sorted_order = torch.argsort(ckey)
    rkey = ckey[sorted_order]
    cl_win = cwin[sorted_order].contiguous()
    cl_fold = _FOLD.to(device)[ct[sorted_order] + _REACH].contiguous()
    cl_axis = axis.index_select(0, cl_win).contiguous()

    claim_lo = torch.empty(n_claims, dtype=torch.long, device=device)
    claim_hi = torch.empty(n_claims, dtype=torch.long, device=device)
    if n_claims:
        starts = torch.ones(n_claims, dtype=torch.bool, device=device)
        starts[1:] = rkey[1:] != rkey[:-1]
        run = starts.cumsum(0) - 1
        bounds = torch.cat(
            [
                starts.nonzero().squeeze(1),
                torch.tensor([n_claims], device=device),
            ]
        )
        claim_lo[sorted_order] = bounds.index_select(0, run)
        claim_hi[sorted_order] = bounds.index_select(0, run + 1)

    return WaTables(
        col_ptr,
        col_src,
        col_cls,
        claim_lo,
        claim_hi,
        cl_win,
        cl_fold,
        cl_axis,
        # clone, not contiguous: a 0/1-window unbind view is already
        # contiguous and custom ops may not return input aliases.
        axis.clone(),
    )


@torch.library.custom_op("mantisnet::wa_tables", mutates_args=())
def derive_wa_tables(
    window_id: Tensor, window_pos: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """``wa_tables``' tensors as an opaque op for the trunk.

    The joins are data-dependent and cannot trace; as a graph break they
    would spill the surrounding message passing out of the compiled graph
    to eager. As a custom op with an unbacked colinear edge size the
    derivation sits inside the graph like the attention op it feeds.
    """
    return tuple(wa_tables(window_id, window_pos))


@derive_wa_tables.register_fake
def _(window_id, window_pos):
    n_w = window_id.shape[0]
    e = torch.library.get_ctx().new_dynamic_size()

    def col():
        return window_id.new_empty((e,), dtype=torch.long)

    def claims():
        return window_id.new_empty((n_w * _CLAIM_SPAN,), dtype=torch.long)

    return (
        window_id.new_empty((n_w + 1,), dtype=torch.long),
        col(),
        col(),
        claims(),
        claims(),
        claims(),
        claims(),
        claims(),
        window_id.new_empty((n_w,), dtype=torch.long),
    )


def _expanded_edges(tables: WaTables) -> tuple[Tensor, Tensor, Tensor]:
    """The explicit destination-major edge list the claims encode.

    The reference path and the CUDA failure fallback compose attention over
    these expanded views; the kernels never build them.
    """
    (
        col_ptr,
        col_src,
        col_cls,
        claim_lo,
        claim_hi,
        cl_win,
        cl_fold,
        cl_axis,
        waxis,
    ) = tables
    n_w = waxis.shape[0]
    device = waxis.device
    lens = claim_hi - claim_lo
    total = int(lens.sum())
    claim = torch.repeat_interleave(
        torch.arange(lens.shape[0], device=device), lens
    )
    within = torch.arange(total, device=device) - (
        lens.cumsum(0) - lens
    ).index_select(0, claim)
    entry = claim_lo.index_select(0, claim) + within
    owner = claim // _CLAIM_SPAN
    partner = cl_win.index_select(0, entry)
    keep = cl_axis.index_select(0, entry) != waxis.index_select(0, owner)
    fold_own = _FOLD.to(device)[claim % _CLAIM_SPAN]
    cross_cls = 11 + fold_own * 6 + cl_fold.index_select(0, entry)

    dst = torch.cat([owner[keep], _edge_dst(col_ptr)])
    src = torch.cat([partner[keep], col_src])
    cls = torch.cat([cross_cls[keep], col_cls])
    order = torch.argsort(dst.to(torch.int32), stable=True)
    dst, src, cls = dst[order], src[order], cls[order]
    ptr = torch.searchsorted(dst, torch.arange(n_w + 1, device=device))
    return ptr, src, cls


# The eager fallback walks the destination view in fixed slices so nothing of
# size (E, head_dim) is ever materialized whole.
_EDGE_SLICE = 2_000_000

# Kernel launch geometry: a 32-edge tile covers the colinear run (<= 22
# partners plus SELF) in one pass and each claim run (<= 48 claimants:
# sixteen spans on each of three axes) in two.
_WA_BLOCK_E = 32
_WA_BLOCK_CLS = 64  # next power of two over WA_CLASSES
_WA_NUM_WARPS = 4

_FAILED_SHAPES: dict[tuple[object, ...], str] = {}
_FAILED_BACKWARD_SHAPES: dict[tuple[object, ...], str] = {}


if triton is not None:

    @triton.jit
    def _wa_forward_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        bias_ptr,
        colptr_ptr,
        colsrc_ptr,
        colcls_ptr,
        clo_ptr,
        chi_ptr,
        clwin_ptr,
        clfold_ptr,
        claxis_ptr,
        waxis_ptr,
        out_ptr,
        m_ptr,
        l_ptr,
        scale,
        stride_bias,
        HEADS: tl.constexpr,
        HD: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_HD: tl.constexpr,
        BLOCK_E: tl.constexpr,
        CLAIM_SPAN: tl.constexpr,
    ):
        # One destination window per program, every head at once: the colinear
        # and claim runs are walked a single time for all heads, and each
        # gathered source row is HEADS*HD consecutive elements, so the index
        # traffic is amortized HEADS-fold and the row loads coalesce. The
        # colinear run always holds SELF, so the first tile seeds the softmax
        # max before any claim run can be empty.
        w = tl.program_id(0)
        h_offs = tl.arange(0, BLOCK_H)
        offs = tl.arange(0, BLOCK_HD)
        h_live = h_offs < HEADS
        live = h_live[:, None] & (offs[None, :] < HD)
        base_row = w * HEADS * HD
        q_tile = tl.load(
            q_ptr + base_row + h_offs[:, None] * HD + offs[None, :],
            mask=live,
            other=0.0,
        ).to(tl.float32)
        ax_w = tl.load(waxis_ptr + w)
        m = tl.full([BLOCK_H], -float("inf"), dtype=tl.float32)
        l = tl.zeros([BLOCK_H], dtype=tl.float32)
        acc = tl.zeros([BLOCK_H, BLOCK_HD], dtype=tl.float32)

        start = tl.load(colptr_ptr + w)
        end = tl.load(colptr_ptr + w + 1)
        for lo in tl.range(start, end, BLOCK_E):
            eids = lo + tl.arange(0, BLOCK_E)
            ok = eids < end
            s_idx = tl.load(colsrc_ptr + eids, mask=ok, other=0)
            c_idx = tl.load(colcls_ptr + eids, mask=ok, other=0)
            k_tile = tl.load(
                k_ptr
                + (s_idx[:, None, None] * HEADS + h_offs[None, :, None]) * HD
                + offs[None, None, :],
                mask=ok[:, None, None] & live[None, :, :],
                other=0.0,
            ).to(tl.float32)
            score = tl.sum(q_tile[None, :, :] * k_tile, axis=2) * scale
            score += tl.load(
                bias_ptr + h_offs[None, :] * stride_bias + c_idx[:, None],
                mask=ok[:, None] & h_live[None, :],
                other=0.0,
            ).to(tl.float32)
            score = tl.where(ok[:, None], score, -float("inf"))
            m_new = tl.maximum(m, tl.max(score, axis=0))
            rescale = tl.exp(m - m_new)
            p = tl.exp(score - m_new[None, :])
            l = l * rescale + tl.sum(p, axis=0)
            v_tile = tl.load(
                v_ptr
                + (s_idx[:, None, None] * HEADS + h_offs[None, :, None]) * HD
                + offs[None, None, :],
                mask=ok[:, None, None] & live[None, :, :],
                other=0.0,
            ).to(tl.float32)
            acc = acc * rescale[:, None] + tl.sum(p[:, :, None] * v_tile, axis=0)
            m = m_new

        for k_slot in tl.static_range(CLAIM_SPAN):
            # fold(t) of this claim slot, folded at trace time: static_range
            # makes k_slot a compile-time constant per unrolled iteration.
            t = k_slot - 5
            if t >= 0 and t <= 5:
                f_own = min(t, 5 - t)
            else:
                f_own = 2 + min(-t if t < 0 else t - 5, 3)
            lo = tl.load(clo_ptr + w * CLAIM_SPAN + k_slot)
            hi = tl.load(chi_ptr + w * CLAIM_SPAN + k_slot)
            for base in tl.range(lo, hi, BLOCK_E):
                eids = base + tl.arange(0, BLOCK_E)
                inside = eids < hi
                u = tl.load(clwin_ptr + eids, mask=inside, other=0)
                ax_u = tl.load(claxis_ptr + eids, mask=inside, other=0)
                ok = inside & (ax_u != ax_w)
                f_u = tl.load(clfold_ptr + eids, mask=ok, other=0)
                c_idx = 11 + f_own * 6 + f_u
                k_tile = tl.load(
                    k_ptr
                    + (u[:, None, None] * HEADS + h_offs[None, :, None]) * HD
                    + offs[None, None, :],
                    mask=ok[:, None, None] & live[None, :, :],
                    other=0.0,
                ).to(tl.float32)
                score = tl.sum(q_tile[None, :, :] * k_tile, axis=2) * scale
                score += tl.load(
                    bias_ptr + h_offs[None, :] * stride_bias + c_idx[:, None],
                    mask=ok[:, None] & h_live[None, :],
                    other=0.0,
                ).to(tl.float32)
                score = tl.where(ok[:, None], score, -float("inf"))
                m_new = tl.maximum(m, tl.max(score, axis=0))
                rescale = tl.exp(m - m_new)
                p = tl.exp(score - m_new[None, :])
                l = l * rescale + tl.sum(p, axis=0)
                v_tile = tl.load(
                    v_ptr
                    + (u[:, None, None] * HEADS + h_offs[None, :, None]) * HD
                    + offs[None, None, :],
                    mask=ok[:, None, None] & live[None, :, :],
                    other=0.0,
                ).to(tl.float32)
                acc = acc * rescale[:, None] + tl.sum(
                    p[:, :, None] * v_tile, axis=0
                )
                m = m_new

        tl.store(
            out_ptr + base_row + h_offs[:, None] * HD + offs[None, :],
            acc / l[:, None],
            mask=live,
        )
        tl.store(m_ptr + w * HEADS + h_offs, m, mask=h_live)
        tl.store(l_ptr + w * HEADS + h_offs, l, mask=h_live)

    @triton.jit
    def _wa_dq_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        bias_ptr,
        go_ptr,
        colptr_ptr,
        colsrc_ptr,
        colcls_ptr,
        clo_ptr,
        chi_ptr,
        clwin_ptr,
        clfold_ptr,
        claxis_ptr,
        waxis_ptr,
        m_ptr,
        l_ptr,
        delta_ptr,
        dq_ptr,
        partial_ptr,
        scale,
        stride_bias,
        HEADS: tl.constexpr,
        HD: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_HD: tl.constexpr,
        BLOCK_E: tl.constexpr,
        BLOCK_CLS: tl.constexpr,
        CLAIM_SPAN: tl.constexpr,
    ):
        # Destination sweep, every head at once: recompute alpha from the
        # saved stats, accumulate dq and this program's per-(head, class)
        # dscore sums in registers — one deterministic pass, no atomics. The
        # per-program bias partials are summed by a fixed torch reduction
        # afterwards.
        w = tl.program_id(0)
        h_offs = tl.arange(0, BLOCK_H)
        offs = tl.arange(0, BLOCK_HD)
        h_live = h_offs < HEADS
        live = h_live[:, None] & (offs[None, :] < HD)
        cls_range = tl.arange(0, BLOCK_CLS)
        base_row = w * HEADS * HD
        q_tile = tl.load(
            q_ptr + base_row + h_offs[:, None] * HD + offs[None, :],
            mask=live,
            other=0.0,
        ).to(tl.float32)
        go_tile = tl.load(
            go_ptr + base_row + h_offs[:, None] * HD + offs[None, :],
            mask=live,
            other=0.0,
        ).to(tl.float32)
        ax_w = tl.load(waxis_ptr + w)
        m = tl.load(m_ptr + w * HEADS + h_offs, mask=h_live, other=0.0)
        l = tl.load(l_ptr + w * HEADS + h_offs, mask=h_live, other=1.0)
        delta = tl.load(delta_ptr + w * HEADS + h_offs, mask=h_live, other=0.0)
        acc = tl.zeros([BLOCK_H, BLOCK_HD], dtype=tl.float32)
        acc_bias = tl.zeros([BLOCK_H, BLOCK_CLS], dtype=tl.float32)

        start = tl.load(colptr_ptr + w)
        end = tl.load(colptr_ptr + w + 1)
        for lo in tl.range(start, end, BLOCK_E):
            eids = lo + tl.arange(0, BLOCK_E)
            ok = eids < end
            s_idx = tl.load(colsrc_ptr + eids, mask=ok, other=0)
            c_idx = tl.load(colcls_ptr + eids, mask=ok, other=0)
            k_tile = tl.load(
                k_ptr
                + (s_idx[:, None, None] * HEADS + h_offs[None, :, None]) * HD
                + offs[None, None, :],
                mask=ok[:, None, None] & live[None, :, :],
                other=0.0,
            ).to(tl.float32)
            score = tl.sum(q_tile[None, :, :] * k_tile, axis=2) * scale
            score += tl.load(
                bias_ptr + h_offs[None, :] * stride_bias + c_idx[:, None],
                mask=ok[:, None] & h_live[None, :],
                other=0.0,
            ).to(tl.float32)
            alpha = tl.where(
                ok[:, None], tl.exp(score - m[None, :]) / l[None, :], 0.0
            )
            v_tile = tl.load(
                v_ptr
                + (s_idx[:, None, None] * HEADS + h_offs[None, :, None]) * HD
                + offs[None, None, :],
                mask=ok[:, None, None] & live[None, :, :],
                other=0.0,
            ).to(tl.float32)
            ds = alpha * (
                tl.sum(go_tile[None, :, :] * v_tile, axis=2) - delta[None, :]
            )
            acc += tl.sum(ds[:, :, None] * k_tile, axis=0)
            acc_bias += tl.sum(
                tl.where(
                    c_idx[:, None, None] == cls_range[None, None, :],
                    ds[:, :, None],
                    0.0,
                ),
                axis=0,
            )

        for k_slot in tl.static_range(CLAIM_SPAN):
            # fold(t) of this claim slot, folded at trace time: static_range
            # makes k_slot a compile-time constant per unrolled iteration.
            t = k_slot - 5
            if t >= 0 and t <= 5:
                f_own = min(t, 5 - t)
            else:
                f_own = 2 + min(-t if t < 0 else t - 5, 3)
            lo = tl.load(clo_ptr + w * CLAIM_SPAN + k_slot)
            hi = tl.load(chi_ptr + w * CLAIM_SPAN + k_slot)
            for base in tl.range(lo, hi, BLOCK_E):
                eids = base + tl.arange(0, BLOCK_E)
                inside = eids < hi
                u = tl.load(clwin_ptr + eids, mask=inside, other=0)
                ax_u = tl.load(claxis_ptr + eids, mask=inside, other=0)
                ok = inside & (ax_u != ax_w)
                f_u = tl.load(clfold_ptr + eids, mask=ok, other=0)
                c_idx = 11 + f_own * 6 + f_u
                k_tile = tl.load(
                    k_ptr
                    + (u[:, None, None] * HEADS + h_offs[None, :, None]) * HD
                    + offs[None, None, :],
                    mask=ok[:, None, None] & live[None, :, :],
                    other=0.0,
                ).to(tl.float32)
                score = tl.sum(q_tile[None, :, :] * k_tile, axis=2) * scale
                score += tl.load(
                    bias_ptr + h_offs[None, :] * stride_bias + c_idx[:, None],
                    mask=ok[:, None] & h_live[None, :],
                    other=0.0,
                ).to(tl.float32)
                alpha = tl.where(
                    ok[:, None], tl.exp(score - m[None, :]) / l[None, :], 0.0
                )
                v_tile = tl.load(
                    v_ptr
                    + (u[:, None, None] * HEADS + h_offs[None, :, None]) * HD
                    + offs[None, None, :],
                    mask=ok[:, None, None] & live[None, :, :],
                    other=0.0,
                ).to(tl.float32)
                ds = alpha * (
                    tl.sum(go_tile[None, :, :] * v_tile, axis=2) - delta[None, :]
                )
                acc += tl.sum(ds[:, :, None] * k_tile, axis=0)
                acc_bias += tl.sum(
                    tl.where(
                        c_idx[:, None, None] == cls_range[None, None, :],
                        ds[:, :, None],
                        0.0,
                    ),
                    axis=0,
                )

        element = dq_ptr.dtype.element_ty
        tl.store(
            dq_ptr + base_row + h_offs[:, None] * HD + offs[None, :],
            (acc * scale).to(element),
            mask=live,
        )
        tl.store(
            partial_ptr
            + (w * HEADS + h_offs[:, None]) * BLOCK_CLS
            + cls_range[None, :],
            acc_bias,
            mask=h_live[:, None],
        )

    @triton.jit
    def _wa_dkdv_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        bias_ptr,
        go_ptr,
        colptr_ptr,
        colsrc_ptr,
        colcls_ptr,
        clo_ptr,
        chi_ptr,
        clwin_ptr,
        clfold_ptr,
        claxis_ptr,
        waxis_ptr,
        m_ptr,
        l_ptr,
        delta_ptr,
        dk_ptr,
        dv_ptr,
        scale,
        stride_bias,
        HEADS: tl.constexpr,
        HD: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_HD: tl.constexpr,
        BLOCK_E: tl.constexpr,
        CLAIM_SPAN: tl.constexpr,
    ):
        # Source sweep over the same relation set, every head at once: this
        # program's k and v rows are fixed, the destinations' rows and stats
        # are gathered per partner. Colinear and SELF are reversal-symmetric
        # so the destination-major run doubles as the source view; a claim's
        # crossing partners see this window with the folds' roles swapped.
        s = tl.program_id(0)
        h_offs = tl.arange(0, BLOCK_H)
        offs = tl.arange(0, BLOCK_HD)
        h_live = h_offs < HEADS
        live = h_live[:, None] & (offs[None, :] < HD)
        base_row = s * HEADS * HD
        k_tile = tl.load(
            k_ptr + base_row + h_offs[:, None] * HD + offs[None, :],
            mask=live,
            other=0.0,
        ).to(tl.float32)
        v_tile = tl.load(
            v_ptr + base_row + h_offs[:, None] * HD + offs[None, :],
            mask=live,
            other=0.0,
        ).to(tl.float32)
        ax_s = tl.load(waxis_ptr + s)
        acc_k = tl.zeros([BLOCK_H, BLOCK_HD], dtype=tl.float32)
        acc_v = tl.zeros([BLOCK_H, BLOCK_HD], dtype=tl.float32)

        start = tl.load(colptr_ptr + s)
        end = tl.load(colptr_ptr + s + 1)
        for lo in tl.range(start, end, BLOCK_E):
            eids = lo + tl.arange(0, BLOCK_E)
            ok = eids < end
            d_idx = tl.load(colsrc_ptr + eids, mask=ok, other=0)
            c_idx = tl.load(colcls_ptr + eids, mask=ok, other=0)
            q_tile = tl.load(
                q_ptr
                + (d_idx[:, None, None] * HEADS + h_offs[None, :, None]) * HD
                + offs[None, None, :],
                mask=ok[:, None, None] & live[None, :, :],
                other=0.0,
            ).to(tl.float32)
            go_tile = tl.load(
                go_ptr
                + (d_idx[:, None, None] * HEADS + h_offs[None, :, None]) * HD
                + offs[None, None, :],
                mask=ok[:, None, None] & live[None, :, :],
                other=0.0,
            ).to(tl.float32)
            stat_mask = ok[:, None] & h_live[None, :]
            m_e = tl.load(
                m_ptr + d_idx[:, None] * HEADS + h_offs[None, :],
                mask=stat_mask,
                other=0.0,
            )
            l_e = tl.load(
                l_ptr + d_idx[:, None] * HEADS + h_offs[None, :],
                mask=stat_mask,
                other=1.0,
            )
            delta_e = tl.load(
                delta_ptr + d_idx[:, None] * HEADS + h_offs[None, :],
                mask=stat_mask,
                other=0.0,
            )
            score = tl.sum(q_tile * k_tile[None, :, :], axis=2) * scale
            score += tl.load(
                bias_ptr + h_offs[None, :] * stride_bias + c_idx[:, None],
                mask=stat_mask,
                other=0.0,
            ).to(tl.float32)
            alpha = tl.where(ok[:, None], tl.exp(score - m_e) / l_e, 0.0)
            ds = alpha * (
                tl.sum(go_tile * v_tile[None, :, :], axis=2) - delta_e
            )
            acc_k += tl.sum(ds[:, :, None] * q_tile, axis=0)
            acc_v += tl.sum(alpha[:, :, None] * go_tile, axis=0)

        for k_slot in tl.static_range(CLAIM_SPAN):
            # fold(t) of this claim slot, folded at trace time: static_range
            # makes k_slot a compile-time constant per unrolled iteration.
            t = k_slot - 5
            if t >= 0 and t <= 5:
                f_own = min(t, 5 - t)
            else:
                f_own = 2 + min(-t if t < 0 else t - 5, 3)
            lo = tl.load(clo_ptr + s * CLAIM_SPAN + k_slot)
            hi = tl.load(chi_ptr + s * CLAIM_SPAN + k_slot)
            for base in tl.range(lo, hi, BLOCK_E):
                eids = base + tl.arange(0, BLOCK_E)
                inside = eids < hi
                d_idx = tl.load(clwin_ptr + eids, mask=inside, other=0)
                ax_d = tl.load(claxis_ptr + eids, mask=inside, other=0)
                ok = inside & (ax_d != ax_s)
                f_d = tl.load(clfold_ptr + eids, mask=ok, other=0)
                c_idx = 11 + f_d * 6 + f_own
                q_tile = tl.load(
                    q_ptr
                    + (d_idx[:, None, None] * HEADS + h_offs[None, :, None]) * HD
                    + offs[None, None, :],
                    mask=ok[:, None, None] & live[None, :, :],
                    other=0.0,
                ).to(tl.float32)
                go_tile = tl.load(
                    go_ptr
                    + (d_idx[:, None, None] * HEADS + h_offs[None, :, None]) * HD
                    + offs[None, None, :],
                    mask=ok[:, None, None] & live[None, :, :],
                    other=0.0,
                ).to(tl.float32)
                stat_mask = ok[:, None] & h_live[None, :]
                m_e = tl.load(
                    m_ptr + d_idx[:, None] * HEADS + h_offs[None, :],
                    mask=stat_mask,
                    other=0.0,
                )
                l_e = tl.load(
                    l_ptr + d_idx[:, None] * HEADS + h_offs[None, :],
                    mask=stat_mask,
                    other=1.0,
                )
                delta_e = tl.load(
                    delta_ptr + d_idx[:, None] * HEADS + h_offs[None, :],
                    mask=stat_mask,
                    other=0.0,
                )
                score = tl.sum(q_tile * k_tile[None, :, :], axis=2) * scale
                score += tl.load(
                    bias_ptr + h_offs[None, :] * stride_bias + c_idx[:, None],
                    mask=stat_mask,
                    other=0.0,
                ).to(tl.float32)
                alpha = tl.where(ok[:, None], tl.exp(score - m_e) / l_e, 0.0)
                ds = alpha * (
                    tl.sum(go_tile * v_tile[None, :, :], axis=2) - delta_e
                )
                acc_k += tl.sum(ds[:, :, None] * q_tile, axis=0)
                acc_v += tl.sum(alpha[:, :, None] * go_tile, axis=0)

        element = dk_ptr.dtype.element_ty
        tl.store(
            dk_ptr + base_row + h_offs[:, None] * HD + offs[None, :],
            (acc_k * scale).to(element),
            mask=live,
        )
        tl.store(
            dv_ptr + base_row + h_offs[:, None] * HD + offs[None, :],
            acc_v.to(element),
            mask=live,
        )


def _edge_dst(ptr: Tensor) -> Tensor:
    n_w = ptr.shape[0] - 1
    return torch.repeat_interleave(
        torch.arange(n_w, device=ptr.device), ptr[1:] - ptr[:-1]
    )


def _validate_attention(q, k, v, bias, tables: WaTables) -> None:
    if q.ndim != 3 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, v must share shape (N_w, heads, head_dim)")
    if bias.ndim != 2 or bias.shape[0] != q.shape[1]:
        raise ValueError("bias must have shape (heads, classes)")
    n_w = q.shape[0]
    if tables.col_ptr.shape[0] != n_w + 1 or tables.waxis.shape[0] != n_w:
        raise ValueError("col_ptr and waxis must have one row per window")
    claims = (n_w * _CLAIM_SPAN,)
    claim_views = (
        tables.claim_lo,
        tables.claim_hi,
        tables.cl_win,
        tables.cl_fold,
        tables.cl_axis,
    )
    if any(view.shape != claims for view in claim_views):
        raise ValueError("the claim views must have sixteen entries per window")
    if tables.col_src.shape != tables.col_cls.shape or tables.col_src.ndim != 1:
        raise ValueError("the colinear views must be one length")


def _reference_forward(q, k, v, bias, ptr, src, cls):
    """Sliced eager composition over expanded edges: parity and CPU path."""
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


def _supported(q: Tensor) -> bool:
    return (
        triton is not None
        and q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and q.shape[0] > 0
    )


def _table_args(tables: WaTables) -> tuple[Tensor, ...]:
    return (
        tables.col_ptr,
        tables.col_src,
        tables.col_cls,
        tables.claim_lo,
        tables.claim_hi,
        tables.cl_win,
        tables.cl_fold,
        tables.cl_axis,
        tables.waxis,
    )


def _launch_forward(q, k, v, bias, tables: WaTables):
    n_w, heads, hd = q.shape
    out = torch.empty((n_w, heads, hd), dtype=torch.float32, device=q.device)
    m = torch.empty((n_w, heads), dtype=torch.float32, device=q.device)
    l = torch.empty((n_w, heads), dtype=torch.float32, device=q.device)
    _wa_forward_kernel[(n_w,)](
        q,
        k,
        v,
        bias,
        *_table_args(tables),
        out,
        m,
        l,
        1.0 / math.sqrt(hd),
        bias.stride(0),
        HEADS=heads,
        HD=hd,
        BLOCK_H=triton.next_power_of_2(heads),
        BLOCK_HD=triton.next_power_of_2(hd),
        BLOCK_E=_WA_BLOCK_E,
        CLAIM_SPAN=_CLAIM_SPAN,
        num_warps=_WA_NUM_WARPS,
    )
    return out, m, l


def _launch_backward(q, k, v, bias, tables: WaTables, m, l, delta, go):
    n_w, heads, hd = q.shape
    scale = 1.0 / math.sqrt(hd)
    block_hd = triton.next_power_of_2(hd)
    dq = torch.empty_like(q)
    # One partial (head, class) block per window program keeps the bias
    # gradient atomics-free and run-to-run deterministic; the fixed-tree
    # torch sum finishes it. The buffer is (N_w * heads, 64) fp32 — far
    # below the per-edge dscore array this design retires.
    partial = torch.empty(
        (n_w * heads, _WA_BLOCK_CLS), dtype=torch.float32, device=q.device
    )
    _wa_dq_kernel[(n_w,)](
        q,
        k,
        v,
        bias,
        go,
        *_table_args(tables),
        m,
        l,
        delta,
        dq,
        partial,
        scale,
        bias.stride(0),
        HEADS=heads,
        HD=hd,
        BLOCK_H=triton.next_power_of_2(heads),
        BLOCK_HD=block_hd,
        BLOCK_E=_WA_BLOCK_E,
        BLOCK_CLS=_WA_BLOCK_CLS,
        CLAIM_SPAN=_CLAIM_SPAN,
        num_warps=_WA_NUM_WARPS,
    )
    dbias = (
        partial.view(n_w, heads, _WA_BLOCK_CLS)
        .sum(dim=0)[:, : bias.shape[1]]
        .contiguous()
    ).to(bias.dtype)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    _wa_dkdv_kernel[(n_w,)](
        q,
        k,
        v,
        bias,
        go,
        *_table_args(tables),
        m,
        l,
        delta,
        dk,
        dv,
        scale,
        bias.stride(0),
        HEADS=heads,
        HD=hd,
        BLOCK_H=triton.next_power_of_2(heads),
        BLOCK_HD=block_hd,
        BLOCK_E=_WA_BLOCK_E,
        CLAIM_SPAN=_CLAIM_SPAN,
        num_warps=_WA_NUM_WARPS,
    )
    return dq, dk, dv, dbias


@torch.library.custom_op("mantisnet::window_attention", mutates_args=())
def _wa_op(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    bias: Tensor,
    col_ptr: Tensor,
    col_src: Tensor,
    col_cls: Tensor,
    claim_lo: Tensor,
    claim_hi: Tensor,
    cl_win: Tensor,
    cl_fold: Tensor,
    cl_axis: Tensor,
    waxis: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    tables = WaTables(
        col_ptr, col_src, col_cls, claim_lo, claim_hi, cl_win, cl_fold, cl_axis, waxis
    )
    _validate_attention(q, k, v, bias, tables)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    if not _supported(q):
        return _reference_forward(q, k, v, bias, *_expanded_edges(tables))
    key = _shape_key(q)
    if key in _FAILED_SHAPES:
        return _reference_forward(q, k, v, bias, *_expanded_edges(tables))
    try:
        return _launch_forward(q, k, v, bias, tables)
    except Exception as exc:
        _FAILED_SHAPES[key] = f"{type(exc).__name__}: {exc}"
        warnings.warn(
            "window attention failed for "
            f"heads={q.shape[1]}, hd={q.shape[2]}, dtype={q.dtype}; slicing "
            f"instead for this shape: {_FAILED_SHAPES[key]}",
            RuntimeWarning,
            stacklevel=2,
        )
        return _reference_forward(q, k, v, bias, *_expanded_edges(tables))


@_wa_op.register_fake
def _(q, k, v, bias, col_ptr, col_src, col_cls, claim_lo, claim_hi, cl_win, cl_fold, cl_axis, waxis):
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
    col_ptr: Tensor,
    col_src: Tensor,
    col_cls: Tensor,
    claim_lo: Tensor,
    claim_hi: Tensor,
    cl_win: Tensor,
    cl_fold: Tensor,
    cl_axis: Tensor,
    waxis: Tensor,
    out: Tensor,
    m: Tensor,
    l: Tensor,
    grad_out: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    tables = WaTables(
        col_ptr, col_src, col_cls, claim_lo, claim_hi, cl_win, cl_fold, cl_axis, waxis
    )
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    go = grad_out.contiguous().float()
    # delta = Σ_hd go · out per (window, head) — algebraically the segment's
    # Σ alpha dalpha, so the softmax backward needs no first edge sweep.
    delta = (go * out).sum(-1)
    if not _supported(q):
        return _reference_backward(
            q, k, v, bias, *_expanded_edges(tables), m, l, delta, go
        )
    key = _shape_key(q) + (grad_out.dtype,)
    if key in _FAILED_BACKWARD_SHAPES:
        return _reference_backward(
            q, k, v, bias, *_expanded_edges(tables), m, l, delta, go
        )
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
        return _reference_backward(
            q, k, v, bias, *_expanded_edges(tables), m, l, delta, go
        )


@_wa_backward_op.register_fake
def _(q, k, v, bias, col_ptr, col_src, col_cls, claim_lo, claim_hi, cl_win, cl_fold, cl_axis, waxis, out, m, l, grad_out):
    return (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
        torch.empty_like(bias),
    )


def _wa_setup_context(ctx, inputs, output) -> None:
    (
        q,
        k,
        v,
        bias,
        col_ptr,
        col_src,
        col_cls,
        claim_lo,
        claim_hi,
        cl_win,
        cl_fold,
        cl_axis,
        waxis,
    ) = inputs
    out, m, l = output
    ctx.save_for_backward(
        q,
        k,
        v,
        bias,
        col_ptr,
        col_src,
        col_cls,
        claim_lo,
        claim_hi,
        cl_win,
        cl_fold,
        cl_axis,
        waxis,
        out,
        m,
        l,
    )


def _wa_dispatch_backward(ctx, grad_out, _grad_m, _grad_l):
    dq, dk, dv, dbias = _wa_backward_op(*ctx.saved_tensors, grad_out)
    return (dq, dk, dv, dbias) + (None,) * 9


_wa_op.register_autograd(_wa_dispatch_backward, setup_context=_wa_setup_context)


def edge_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    bias: Tensor,
    col_ptr: Tensor,
    col_src: Tensor,
    col_cls: Tensor,
    claim_lo: Tensor,
    claim_hi: Tensor,
    cl_win: Tensor,
    cl_fold: Tensor,
    cl_axis: Tensor,
    waxis: Tensor,
) -> Tensor:
    """§5.1c attention over the claim views: fp32 ``(N_w, heads, head_dim)``."""
    out, _m, _l = _wa_op(
        q,
        k,
        v,
        bias,
        col_ptr,
        col_src,
        col_cls,
        claim_lo,
        claim_hi,
        cl_win,
        cl_fold,
        cl_axis,
        waxis,
    )
    return out
