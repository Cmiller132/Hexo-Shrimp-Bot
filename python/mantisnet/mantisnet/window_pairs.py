"""Window-pair relations and line-blocked attention for section 5.1c.

Relation classes are derived from window identities ``(axis, start_q,
start_r)``.  Same-line windows at offsets 1 through 11 use classes 0 through
10.  Crossing windows use the product of their two six-way folded slots,
classes 11 through 46.  Class 47 is SELF.

The CUDA path groups starts on one line into 16-position blocks.  One program
handles every present lane, reusing each neighboring line tile and each
cell-claimant tile across up to sixteen destinations.  Crossing edges are not
materialized: programs walk the cell-major claimant array for their block's
31-cell span.  Backward uses block-local bias partials followed by a fixed
torch reduction, so all reductions are atomics-free and deterministic.

CPU, unsupported dtypes, and the existing failure fallback expand the same
tables into a destination-major edge list and use the sliced eager reference.
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


WA_CLASSES = 48
_SELF = 47
_REACH = 5
_MAX_OFFSET = 11
_CLAIM_SPAN = 16
_BLOCK_LANES = 16
_BLOCK_CELLS = 31

_FOLD = torch.tensor(
    [2 + min(d, 3) for d in range(_REACH, 0, -1)]
    + [min(t, 5 - t) for t in range(6)]
    + [2 + min(d, 3) for d in range(1, _REACH + 1)],
    dtype=torch.long,
)
_AXES = torch.tensor([[1, 0], [0, 1], [1, -1]], dtype=torch.long)

_KOFF = 1 << 16
_KSPAN = 1 << 17


class WaTables(NamedTuple):
    """One batch's line blocks and cell-major claimant structure."""

    lane_win: Tensor
    block_prev: Tensor
    block_next: Tensor
    block_axis: Tensor
    claim_lo: Tensor
    claim_hi: Tensor
    cl_win: Tensor
    cl_fold: Tensor
    cl_axis: Tensor
    w_block: Tensor
    w_lane: Tensor
    waxis: Tensor


def wa_tables(window_id: Tensor, window_pos: Tensor) -> WaTables:
    """Derive line blocks and claimant runs from a batch's window identities."""
    if window_id.ndim != 2 or window_id.shape[1] != 3:
        raise ValueError("window_id must have shape (N_w, 3)")
    if window_pos.shape != window_id.shape[:1]:
        raise ValueError("window_pos must have one entry per window")

    n_w = window_id.shape[0]
    device = window_id.device
    axis, sq, sr = window_id.to(torch.long).unbind(1)
    window_pos = window_pos.to(torch.long)
    line = torch.where(axis == 0, sr, torch.where(axis == 1, sq, sq + sr))
    pos_on = torch.where(axis == 1, sr, sq)
    bucket = torch.floor_divide(pos_on, _BLOCK_LANES)
    lane = pos_on - bucket * _BLOCK_LANES

    block_key_w = (
        ((window_pos * 4 + axis) * _KSPAN + (line + _KOFF)) * _KSPAN
        + (bucket + _KOFF)
    )
    block_key, block_of_w = torch.unique(
        block_key_w, sorted=True, return_inverse=True
    )
    n_blocks = block_key.shape[0]
    loop = torch.arange(n_w, device=device)

    lane_win = torch.full(
        (n_blocks, _BLOCK_LANES), -1, dtype=torch.int32, device=device
    )
    if n_w:
        lane_win[block_of_w, lane] = loop.to(torch.int32)

    def neighbor(delta: int) -> Tensor:
        if not n_blocks:
            return torch.empty((0,), dtype=torch.int32, device=device)
        wanted = block_key + delta
        found = torch.searchsorted(block_key, wanted)
        safe = found.clamp(max=n_blocks - 1)
        valid = (found < n_blocks) & (block_key.index_select(0, safe) == wanted)
        return torch.where(valid, found, -1).to(torch.int32)

    block_prev = neighbor(-1)
    block_next = neighbor(1)

    packed_line = torch.floor_divide(block_key, _KSPAN)
    block_bucket = block_key - packed_line * _KSPAN - _KOFF
    packed_pos_axis = torch.floor_divide(packed_line, _KSPAN)
    block_line = packed_line - packed_pos_axis * _KSPAN - _KOFF
    block_axis64 = packed_pos_axis.remainder(4)
    block_position = torch.floor_divide(packed_pos_axis, 4)

    # Claims remain sorted by packed (position, q, r) cell key.
    t_ext = torch.arange(-_REACH, 6 + _REACH, device=device)
    vec = _AXES.to(device)[axis]
    cq = sq[:, None] + t_ext[None, :] * vec[:, 0:1]
    cr = sr[:, None] + t_ext[None, :] * vec[:, 1:2]
    claim_key_w = (
        (window_pos[:, None] * _KSPAN + (cq + _KOFF)) * _KSPAN
        + (cr + _KOFF)
    ).reshape(-1)
    cwin = loop[:, None].expand(-1, _CLAIM_SPAN).reshape(-1)
    ct = t_ext[None, :].expand(n_w, -1).reshape(-1)
    claim_order = torch.argsort(claim_key_w)
    claim_key = claim_key_w[claim_order]
    cl_win = cwin[claim_order].to(torch.int32).contiguous()
    cl_fold = (
        _FOLD.to(device)[ct[claim_order] + _REACH].to(torch.int32).contiguous()
    )
    cl_axis = (
        axis.index_select(0, cl_win.to(torch.long)).to(torch.int32).contiguous()
    )

    cell_pos = block_bucket[:, None] * _BLOCK_LANES - _REACH + torch.arange(
        _BLOCK_CELLS, device=device
    )[None, :]
    baxis = block_axis64[:, None]
    bline = block_line[:, None]
    cell_q = torch.where(baxis == 1, bline, cell_pos)
    cell_r = torch.where(
        baxis == 0, bline, torch.where(baxis == 1, cell_pos, bline - cell_pos)
    )
    query_key = (
        (block_position[:, None] * _KSPAN + (cell_q + _KOFF)) * _KSPAN
        + (cell_r + _KOFF)
    )
    claim_lo = torch.searchsorted(claim_key, query_key, side="left").to(torch.int32)
    claim_hi = torch.searchsorted(claim_key, query_key, side="right").to(torch.int32)

    return WaTables(
        lane_win,
        block_prev,
        block_next,
        block_axis64.to(torch.int32),
        claim_lo,
        claim_hi,
        cl_win,
        cl_fold,
        cl_axis,
        block_of_w.to(torch.int32),
        lane.to(torch.int32),
        axis.to(torch.int32).clone(),
    )


@torch.library.custom_op("mantisnet::wa_tables", mutates_args=())
def derive_wa_tables(
    window_id: Tensor, window_pos: Tensor
) -> tuple[
    Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
    Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
]:
    """Expose data-dependent table derivation as an opaque trunk op."""
    return tuple(wa_tables(window_id, window_pos))


@derive_wa_tables.register_fake
def _(window_id, window_pos):
    n_w = window_id.shape[0]
    n_blocks = torch.library.get_ctx().new_dynamic_size()

    def block(*tail):
        return window_id.new_empty((n_blocks, *tail), dtype=torch.int32)

    def claims():
        return window_id.new_empty((n_w * _CLAIM_SPAN,), dtype=torch.int32)

    return (
        block(_BLOCK_LANES),
        block(),
        block(),
        block(),
        block(_BLOCK_CELLS),
        block(_BLOCK_CELLS),
        claims(),
        claims(),
        claims(),
        window_id.new_empty((n_w,), dtype=torch.int32),
        window_id.new_empty((n_w,), dtype=torch.int32),
        window_id.new_empty((n_w,), dtype=torch.int32),
    )


def _enumerate_edges(
    window_id: Tensor | WaTables, window_pos: Tensor | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    """Expand relations from identities or tables; used only by the reference."""
    tables = (
        window_id
        if isinstance(window_id, WaTables)
        else wa_tables(window_id, window_pos)
    )
    n_w = tables.waxis.shape[0]
    device = tables.waxis.device
    owner = torch.arange(n_w, device=device)
    lane = tables.w_lane.to(torch.long)
    block = tables.w_block.to(torch.long)

    neighbor_blocks = torch.stack(
        [tables.block_prev[block], tables.w_block, tables.block_next[block]], dim=1
    ).to(torch.long)
    have_block = neighbor_blocks >= 0
    candidates = tables.lane_win[neighbor_blocks.clamp_min(0)].to(torch.long)
    delta = (
        (torch.arange(3, device=device).view(1, 3, 1) - 1) * _BLOCK_LANES
        + torch.arange(_BLOCK_LANES, device=device).view(1, 1, -1)
        - lane.view(-1, 1, 1)
    )
    col_keep = have_block[:, :, None] & (candidates >= 0) & (delta.abs() <= 11)
    col_dst = owner[:, None, None].expand_as(candidates)[col_keep]
    col_src = candidates[col_keep]
    col_delta = delta.expand_as(candidates)[col_keep]
    col_cls = torch.where(col_delta == 0, _SELF, col_delta.abs() - 1)

    # A window at lane b owns block-cell slots b..b+15.
    slots = lane[:, None] + torch.arange(_CLAIM_SPAN, device=device)[None, :]
    flat = block[:, None] * _BLOCK_CELLS + slots
    all_lo = tables.claim_lo.reshape(-1).to(torch.long)
    all_hi = tables.claim_hi.reshape(-1).to(torch.long)
    lo = all_lo.index_select(0, flat.reshape(-1))
    hi = all_hi.index_select(0, flat.reshape(-1))
    lens = hi - lo
    total = int(lens.sum())
    claim = torch.repeat_interleave(torch.arange(lens.numel(), device=device), lens)
    within = torch.arange(total, device=device) - (
        lens.cumsum(0) - lens
    ).index_select(0, claim)
    entry = lo.index_select(0, claim) + within
    cross_dst = torch.floor_divide(claim, _CLAIM_SPAN)
    cross_src = tables.cl_win.index_select(0, entry).to(torch.long)
    keep = tables.cl_axis.index_select(0, entry) != tables.waxis.index_select(
        0, cross_dst
    )
    fold_own = _FOLD.to(device)[claim.remainder(_CLAIM_SPAN)]
    cross_cls = 11 + fold_own * 6 + tables.cl_fold.index_select(0, entry)
    return (
        torch.cat([cross_dst[keep], col_dst]),
        torch.cat([cross_src[keep], col_src]),
        torch.cat([cross_cls[keep].to(torch.long), col_cls]),
    )


def _expanded_edges(tables: WaTables) -> tuple[Tensor, Tensor, Tensor]:
    """Return the explicit destination-major edge list encoded by the tables."""
    n_w = tables.waxis.shape[0]
    device = tables.waxis.device
    dst, src, cls = _enumerate_edges(tables)
    order = torch.argsort(dst.to(torch.int32), stable=True)
    dst, src, cls = dst[order], src[order], cls[order]
    ptr = torch.searchsorted(dst, torch.arange(n_w + 1, device=device))
    return ptr, src, cls


_EDGE_SLICE = 2_000_000
_WA_BLOCK_E = 16
_WA_BLOCK_CLS = 64
_WA_NUM_WARPS = 4
_WA_BLOCK_H = 4
_WA_BLOCK_HD = 32

_FAILED_SHAPES: dict[tuple[object, ...], str] = {}
_FAILED_BACKWARD_SHAPES: dict[tuple[object, ...], str] = {}


if triton is not None:

    @triton.jit
    def _wa_scores_16(left, right, scale, HEADS: tl.constexpr, BLOCK_H: tl.constexpr):
        """Per-head 16xK by Kx16 dots without a four-dimensional product.

        Head slices come out by masked reduction: Triton has no integer
        tensor subscript, and the select-sum keeps every operand a 2-D dot.
        """
        h_offs = tl.arange(0, BLOCK_H)
        score = tl.zeros([16, 16, BLOCK_H], dtype=tl.float32)
        for h in tl.static_range(HEADS):
            sel = (h_offs == h).to(tl.float32)
            left_h = tl.sum(left * sel[None, :, None], axis=1)
            right_h = tl.sum(right * sel[None, :, None], axis=1)
            dot = tl.dot(left_h, tl.trans(right_h), input_precision="ieee")
            score += dot[:, :, None] * sel[None, None, :]
        return score * scale

    @triton.jit
    def _wa_weighted_16(
        weight,
        rows,
        HEADS: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_HD: tl.constexpr,
    ):
        """Per-head 16x16 by 16xK products without a 4-D broadcast."""
        h_offs = tl.arange(0, BLOCK_H)
        out = tl.zeros([16, BLOCK_H, BLOCK_HD], dtype=tl.float32)
        for h in tl.static_range(HEADS):
            sel = (h_offs == h).to(tl.float32)
            w_h = tl.sum(weight * sel[None, None, :], axis=2)
            rows_h = tl.sum(rows * sel[None, :, None], axis=1)
            dot = tl.dot(w_h, rows_h, input_precision="ieee")
            out += dot[:, None, :] * sel[None, :, None]
        return out

    @triton.jit
    def _wa_fold(t):
        inside = (t >= 0) & (t <= 5)
        inner = tl.minimum(t, 5 - t)
        distance = tl.where(t < 0, -t, t - 5)
        return tl.where(inside, inner, 2 + tl.minimum(distance, 3))

    @triton.jit
    def _wa_forward_kernel(
        q_ptr, k_ptr, v_ptr, bias_ptr,
        lane_ptr, prev_ptr, next_ptr, block_axis_ptr,
        clo_ptr, chi_ptr, clwin_ptr, clfold_ptr, claxis_ptr,
        wblock_ptr, wlane_ptr, waxis_ptr,
        out_ptr, m_ptr, l_ptr, scale, stride_bias,
        HEADS: tl.constexpr, HD: tl.constexpr,
        BLOCK_H: tl.constexpr, BLOCK_HD: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        blk = tl.program_id(0).to(tl.int32)
        stride_bias = stride_bias.to(tl.int32)
        b = tl.arange(0, 16)
        e = tl.arange(0, BLOCK_E)
        h = tl.arange(0, BLOCK_H)
        d = tl.arange(0, BLOCK_HD)
        h_live = h < HEADS
        d_live = d < HD

        lanes = tl.load(lane_ptr + blk * 16 + b).to(tl.int32)
        present = lanes >= 0
        safe_lanes = tl.where(present, lanes, 0)
        q = tl.load(
            q_ptr
            + ((safe_lanes[:, None, None] * HEADS + h[None, :, None]) * HD)
            + d[None, None, :],
            mask=present[:, None, None] & h_live[None, :, None] & d_live[None, None, :],
            other=0.0,
        ).to(tl.float32)
        ax_blk = tl.load(block_axis_ptr + blk).to(tl.int32)
        m = tl.full([16, BLOCK_H], -float("inf"), dtype=tl.float32)
        l = tl.zeros([16, BLOCK_H], dtype=tl.float32)
        acc = tl.zeros([16, BLOCK_H, BLOCK_HD], dtype=tl.float32)

        # Previous, own, and next buckets, in that fixed order.
        for tile in tl.static_range(3):
            neighbor = blk
            if tile == 0:
                neighbor = tl.load(prev_ptr + blk).to(tl.int32)
            elif tile == 2:
                neighbor = tl.load(next_ptr + blk).to(tl.int32)
            have_neighbor = neighbor >= 0
            safe_neighbor = tl.where(have_neighbor, neighbor, 0)
            src = tl.load(
                lane_ptr + safe_neighbor * 16 + e,
                mask=have_neighbor,
                other=-1,
            ).to(tl.int32)
            src_present = src >= 0
            safe_src = tl.where(src_present, src, 0)
            delta_pos = (tile - 1) * 16 + e[None, :] - b[:, None]
            pair = (
                present[:, None]
                & src_present[None, :]
                & (tl.abs(delta_pos) <= 11)
            )
            cls = tl.where(delta_pos == 0, 47, tl.abs(delta_pos) - 1).to(tl.int32)
            k_rows = tl.load(
                k_ptr
                + ((safe_src[:, None, None] * HEADS + h[None, :, None]) * HD)
                + d[None, None, :],
                mask=src_present[:, None, None] & h_live[None, :, None] & d_live[None, None, :],
                other=0.0,
            ).to(tl.float32)
            score = _wa_scores_16(q, k_rows, scale, HEADS, BLOCK_H)
            score += tl.load(
                bias_ptr + h[None, None, :] * stride_bias + cls[:, :, None],
                mask=pair[:, :, None] & h_live[None, None, :],
                other=0.0,
            ).to(tl.float32)
            score = tl.where(pair[:, :, None], score, -float("inf"))
            tile_m = tl.max(score, axis=1)
            m_new = tl.maximum(m, tile_m)
            seeded = m_new != -float("inf")
            rescale = tl.where(seeded, tl.exp(m - m_new), 1.0)
            p = tl.where(pair[:, :, None], tl.exp(score - m_new[:, None, :]), 0.0)
            l = l * rescale + tl.sum(p, axis=1)
            v_rows = tl.load(
                v_ptr
                + ((safe_src[:, None, None] * HEADS + h[None, :, None]) * HD)
                + d[None, None, :],
                mask=src_present[:, None, None] & h_live[None, :, None] & d_live[None, None, :],
                other=0.0,
            ).to(tl.float32)
            acc = acc * rescale[:, :, None] + _wa_weighted_16(
                p, v_rows, HEADS, BLOCK_H, BLOCK_HD
            )
            m = m_new

        # Runtime loop: the body is fully tensorized in j, and a static
        # 31-way unroll multiplies the kernel IR past practical compile time.
        for j in tl.range(0, 31):
            slot = j - b
            lane_claims = present & (slot >= 0) & (slot <= 15)
            f_own = _wa_fold(slot - 5).to(tl.int32)
            lo = tl.load(clo_ptr + blk * 31 + j).to(tl.int32)
            hi = tl.load(chi_ptr + blk * 31 + j).to(tl.int32)
            for base in tl.range(lo, hi, BLOCK_E):
                entry = base + e
                inside = entry < hi
                src = tl.load(clwin_ptr + entry, mask=inside, other=0).to(tl.int32)
                src_axis = tl.load(claxis_ptr + entry, mask=inside, other=ax_blk).to(tl.int32)
                src_ok = inside & (src_axis != ax_blk)
                f_src = tl.load(clfold_ptr + entry, mask=src_ok, other=0).to(tl.int32)
                pair = lane_claims[:, None] & src_ok[None, :]
                cls = 11 + f_own[:, None] * 6 + f_src[None, :]
                k_rows = tl.load(
                    k_ptr
                    + ((src[:, None, None] * HEADS + h[None, :, None]) * HD)
                    + d[None, None, :],
                    mask=src_ok[:, None, None] & h_live[None, :, None] & d_live[None, None, :],
                    other=0.0,
                ).to(tl.float32)
                score = _wa_scores_16(q, k_rows, scale, HEADS, BLOCK_H)
                score += tl.load(
                    bias_ptr + h[None, None, :] * stride_bias + cls[:, :, None],
                    mask=pair[:, :, None] & h_live[None, None, :],
                    other=0.0,
                ).to(tl.float32)
                score = tl.where(pair[:, :, None], score, -float("inf"))
                tile_m = tl.max(score, axis=1)
                m_new = tl.maximum(m, tile_m)
                rescale = tl.exp(m - m_new)
                p = tl.where(pair[:, :, None], tl.exp(score - m_new[:, None, :]), 0.0)
                l = l * rescale + tl.sum(p, axis=1)
                v_rows = tl.load(
                    v_ptr
                    + ((src[:, None, None] * HEADS + h[None, :, None]) * HD)
                    + d[None, None, :],
                    mask=src_ok[:, None, None] & h_live[None, :, None] & d_live[None, None, :],
                    other=0.0,
                ).to(tl.float32)
                acc = acc * rescale[:, :, None] + _wa_weighted_16(
                    p, v_rows, HEADS, BLOCK_H, BLOCK_HD
                )
                m = m_new

        row_ptr = ((safe_lanes[:, None, None] * HEADS + h[None, :, None]) * HD) + d[None, None, :]
        live = present[:, None, None] & h_live[None, :, None] & d_live[None, None, :]
        tl.store(out_ptr + row_ptr, acc / l[:, :, None], mask=live)
        stat_ptr = safe_lanes[:, None] * HEADS + h[None, :]
        stat_live = present[:, None] & h_live[None, :]
        tl.store(m_ptr + stat_ptr, m, mask=stat_live)
        tl.store(l_ptr + stat_ptr, l, mask=stat_live)

    @triton.jit
    def _wa_dq_kernel(
        q_ptr, k_ptr, v_ptr, bias_ptr, go_ptr,
        lane_ptr, prev_ptr, next_ptr, block_axis_ptr,
        clo_ptr, chi_ptr, clwin_ptr, clfold_ptr, claxis_ptr,
        wblock_ptr, wlane_ptr, waxis_ptr,
        m_ptr, l_ptr, delta_ptr, dq_ptr, partial_ptr,
        scale, stride_bias,
        HEADS: tl.constexpr, HD: tl.constexpr,
        BLOCK_H: tl.constexpr, BLOCK_HD: tl.constexpr,
        BLOCK_E: tl.constexpr, BLOCK_CLS: tl.constexpr,
    ):
        blk = tl.program_id(0).to(tl.int32)
        stride_bias = stride_bias.to(tl.int32)
        b = tl.arange(0, 16)
        e = tl.arange(0, BLOCK_E)
        h = tl.arange(0, BLOCK_H)
        d = tl.arange(0, BLOCK_HD)
        cls_range = tl.arange(0, BLOCK_CLS)
        h_live = h < HEADS
        d_live = d < HD
        lanes = tl.load(lane_ptr + blk * 16 + b).to(tl.int32)
        present = lanes >= 0
        safe_lanes = tl.where(present, lanes, 0)
        row_ptr = ((safe_lanes[:, None, None] * HEADS + h[None, :, None]) * HD) + d[None, None, :]
        row_live = present[:, None, None] & h_live[None, :, None] & d_live[None, None, :]
        q = tl.load(q_ptr + row_ptr, mask=row_live, other=0.0).to(tl.float32)
        go = tl.load(go_ptr + row_ptr, mask=row_live, other=0.0).to(tl.float32)
        stat_ptr = safe_lanes[:, None] * HEADS + h[None, :]
        stat_live = present[:, None] & h_live[None, :]
        m = tl.load(m_ptr + stat_ptr, mask=stat_live, other=0.0)
        l = tl.load(l_ptr + stat_ptr, mask=stat_live, other=1.0)
        delta = tl.load(delta_ptr + stat_ptr, mask=stat_live, other=0.0)
        ax_blk = tl.load(block_axis_ptr + blk).to(tl.int32)
        acc = tl.zeros([16, BLOCK_H, BLOCK_HD], dtype=tl.float32)
        acc_bias = tl.zeros([BLOCK_H, BLOCK_CLS], dtype=tl.float32)

        for tile in tl.static_range(3):
            neighbor = blk
            if tile == 0:
                neighbor = tl.load(prev_ptr + blk).to(tl.int32)
            elif tile == 2:
                neighbor = tl.load(next_ptr + blk).to(tl.int32)
            have_neighbor = neighbor >= 0
            safe_neighbor = tl.where(have_neighbor, neighbor, 0)
            src = tl.load(lane_ptr + safe_neighbor * 16 + e, mask=have_neighbor, other=-1).to(tl.int32)
            src_present = src >= 0
            safe_src = tl.where(src_present, src, 0)
            delta_pos = (tile - 1) * 16 + e[None, :] - b[:, None]
            pair = present[:, None] & src_present[None, :] & (tl.abs(delta_pos) <= 11)
            cls = tl.where(delta_pos == 0, 47, tl.abs(delta_pos) - 1).to(tl.int32)
            candidate_live = src_present[:, None, None] & h_live[None, :, None] & d_live[None, None, :]
            candidate_ptr = ((safe_src[:, None, None] * HEADS + h[None, :, None]) * HD) + d[None, None, :]
            k_rows = tl.load(k_ptr + candidate_ptr, mask=candidate_live, other=0.0).to(tl.float32)
            v_rows = tl.load(v_ptr + candidate_ptr, mask=candidate_live, other=0.0).to(tl.float32)
            score = _wa_scores_16(q, k_rows, scale, HEADS, BLOCK_H)
            score += tl.load(
                bias_ptr + h[None, None, :] * stride_bias + cls[:, :, None],
                mask=pair[:, :, None] & h_live[None, None, :], other=0.0,
            ).to(tl.float32)
            alpha = tl.where(pair[:, :, None], tl.exp(score - m[:, None, :]) / l[:, None, :], 0.0)
            dalpha = _wa_scores_16(go, v_rows, 1.0, HEADS, BLOCK_H)
            ds = alpha * (dalpha - delta[:, None, :])
            acc += _wa_weighted_16(ds, k_rows, HEADS, BLOCK_H, BLOCK_HD)
            for di in tl.static_range(23):
                delta_class = di - 11
                # Triton's trace namespace has no abs builtin.
                magnitude = delta_class if delta_class > 0 else -delta_class
                class_index = 47 if delta_class == 0 else magnitude - 1
                selected = pair & (delta_pos == delta_class)
                summed = tl.sum(tl.sum(tl.where(selected[:, :, None], ds, 0.0), axis=1), axis=0)
                acc_bias += summed[:, None] * (cls_range == class_index)[None, :]

        # Runtime loop: the body is fully tensorized in j, and a static
        # 31-way unroll multiplies the kernel IR past practical compile time.
        for j in tl.range(0, 31):
            slot = j - b
            lane_claims = present & (slot >= 0) & (slot <= 15)
            f_own = _wa_fold(slot - 5).to(tl.int32)
            lo = tl.load(clo_ptr + blk * 31 + j).to(tl.int32)
            hi = tl.load(chi_ptr + blk * 31 + j).to(tl.int32)
            for base in tl.range(lo, hi, BLOCK_E):
                entry = base + e
                inside = entry < hi
                src = tl.load(clwin_ptr + entry, mask=inside, other=0).to(tl.int32)
                src_axis = tl.load(claxis_ptr + entry, mask=inside, other=ax_blk).to(tl.int32)
                src_ok = inside & (src_axis != ax_blk)
                f_src = tl.load(clfold_ptr + entry, mask=src_ok, other=0).to(tl.int32)
                pair = lane_claims[:, None] & src_ok[None, :]
                cls = 11 + f_own[:, None] * 6 + f_src[None, :]
                candidate_live = src_ok[:, None, None] & h_live[None, :, None] & d_live[None, None, :]
                candidate_ptr = ((src[:, None, None] * HEADS + h[None, :, None]) * HD) + d[None, None, :]
                k_rows = tl.load(k_ptr + candidate_ptr, mask=candidate_live, other=0.0).to(tl.float32)
                v_rows = tl.load(v_ptr + candidate_ptr, mask=candidate_live, other=0.0).to(tl.float32)
                score = _wa_scores_16(q, k_rows, scale, HEADS, BLOCK_H)
                score += tl.load(
                    bias_ptr + h[None, None, :] * stride_bias + cls[:, :, None],
                    mask=pair[:, :, None] & h_live[None, None, :], other=0.0,
                ).to(tl.float32)
                alpha = tl.where(pair[:, :, None], tl.exp(score - m[:, None, :]) / l[:, None, :], 0.0)
                dalpha = _wa_scores_16(go, v_rows, 1.0, HEADS, BLOCK_H)
                ds = alpha * (dalpha - delta[:, None, :])
                acc += _wa_weighted_16(ds, k_rows, HEADS, BLOCK_H, BLOCK_HD)
                for f in tl.static_range(6):
                    by_lane = tl.sum(tl.where(f_src[None, :, None] == f, ds, 0.0), axis=1)
                    for fo in tl.static_range(6):
                        summed = tl.sum(tl.where((f_own == fo)[:, None], by_lane, 0.0), axis=0)
                        acc_bias += summed[:, None] * (cls_range == (11 + fo * 6 + f))[None, :]

        element = dq_ptr.dtype.element_ty
        tl.store(dq_ptr + row_ptr, (acc * scale).to(element), mask=row_live)
        tl.store(
            partial_ptr + (blk * HEADS + h[:, None]) * BLOCK_CLS + cls_range[None, :],
            acc_bias,
            mask=h_live[:, None],
        )

    @triton.jit
    def _wa_dkdv_kernel(
        q_ptr, k_ptr, v_ptr, bias_ptr, go_ptr,
        lane_ptr, prev_ptr, next_ptr, block_axis_ptr,
        clo_ptr, chi_ptr, clwin_ptr, clfold_ptr, claxis_ptr,
        wblock_ptr, wlane_ptr, waxis_ptr,
        m_ptr, l_ptr, delta_ptr, dk_ptr, dv_ptr,
        scale, stride_bias,
        HEADS: tl.constexpr, HD: tl.constexpr,
        BLOCK_H: tl.constexpr, BLOCK_HD: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        blk = tl.program_id(0).to(tl.int32)
        stride_bias = stride_bias.to(tl.int32)
        b = tl.arange(0, 16)
        e = tl.arange(0, BLOCK_E)
        h = tl.arange(0, BLOCK_H)
        d = tl.arange(0, BLOCK_HD)
        h_live = h < HEADS
        d_live = d < HD
        lanes = tl.load(lane_ptr + blk * 16 + b).to(tl.int32)
        present = lanes >= 0
        safe_lanes = tl.where(present, lanes, 0)
        source_ptr = ((safe_lanes[:, None, None] * HEADS + h[None, :, None]) * HD) + d[None, None, :]
        source_live = present[:, None, None] & h_live[None, :, None] & d_live[None, None, :]
        k_source = tl.load(k_ptr + source_ptr, mask=source_live, other=0.0).to(tl.float32)
        v_source = tl.load(v_ptr + source_ptr, mask=source_live, other=0.0).to(tl.float32)
        ax_blk = tl.load(block_axis_ptr + blk).to(tl.int32)
        acc_k = tl.zeros([16, BLOCK_H, BLOCK_HD], dtype=tl.float32)
        acc_v = tl.zeros([16, BLOCK_H, BLOCK_HD], dtype=tl.float32)

        for tile in tl.static_range(3):
            neighbor = blk
            if tile == 0:
                neighbor = tl.load(prev_ptr + blk).to(tl.int32)
            elif tile == 2:
                neighbor = tl.load(next_ptr + blk).to(tl.int32)
            have_neighbor = neighbor >= 0
            safe_neighbor = tl.where(have_neighbor, neighbor, 0)
            dst = tl.load(lane_ptr + safe_neighbor * 16 + e, mask=have_neighbor, other=-1).to(tl.int32)
            dst_present = dst >= 0
            safe_dst = tl.where(dst_present, dst, 0)
            delta_pos = (tile - 1) * 16 + e[None, :] - b[:, None]
            pair = present[:, None] & dst_present[None, :] & (tl.abs(delta_pos) <= 11)
            cls = tl.where(delta_pos == 0, 47, tl.abs(delta_pos) - 1).to(tl.int32)
            dst_ptr = ((safe_dst[:, None, None] * HEADS + h[None, :, None]) * HD) + d[None, None, :]
            dst_live = dst_present[:, None, None] & h_live[None, :, None] & d_live[None, None, :]
            q_dst = tl.load(q_ptr + dst_ptr, mask=dst_live, other=0.0).to(tl.float32)
            go_dst = tl.load(go_ptr + dst_ptr, mask=dst_live, other=0.0).to(tl.float32)
            stat_ptr = safe_dst[:, None] * HEADS + h[None, :]
            stat_live = dst_present[:, None] & h_live[None, :]
            m_dst = tl.load(m_ptr + stat_ptr, mask=stat_live, other=0.0)
            l_dst = tl.load(l_ptr + stat_ptr, mask=stat_live, other=1.0)
            delta_dst = tl.load(delta_ptr + stat_ptr, mask=stat_live, other=0.0)
            score = _wa_scores_16(k_source, q_dst, scale, HEADS, BLOCK_H)
            score += tl.load(
                bias_ptr + h[None, None, :] * stride_bias + cls[:, :, None],
                mask=pair[:, :, None] & h_live[None, None, :], other=0.0,
            ).to(tl.float32)
            alpha = tl.where(pair[:, :, None], tl.exp(score - m_dst[None, :, :]) / l_dst[None, :, :], 0.0)
            dalpha = _wa_scores_16(v_source, go_dst, 1.0, HEADS, BLOCK_H)
            ds = alpha * (dalpha - delta_dst[None, :, :])
            acc_k += _wa_weighted_16(ds, q_dst, HEADS, BLOCK_H, BLOCK_HD)
            acc_v += _wa_weighted_16(alpha, go_dst, HEADS, BLOCK_H, BLOCK_HD)

        # Runtime loop: the body is fully tensorized in j, and a static
        # 31-way unroll multiplies the kernel IR past practical compile time.
        for j in tl.range(0, 31):
            slot = j - b
            lane_claims = present & (slot >= 0) & (slot <= 15)
            f_own = _wa_fold(slot - 5).to(tl.int32)
            lo = tl.load(clo_ptr + blk * 31 + j).to(tl.int32)
            hi = tl.load(chi_ptr + blk * 31 + j).to(tl.int32)
            for base in tl.range(lo, hi, BLOCK_E):
                entry = base + e
                inside = entry < hi
                dst = tl.load(clwin_ptr + entry, mask=inside, other=0).to(tl.int32)
                dst_axis = tl.load(claxis_ptr + entry, mask=inside, other=ax_blk).to(tl.int32)
                dst_ok = inside & (dst_axis != ax_blk)
                f_dst = tl.load(clfold_ptr + entry, mask=dst_ok, other=0).to(tl.int32)
                pair = lane_claims[:, None] & dst_ok[None, :]
                cls = 11 + f_dst[None, :] * 6 + f_own[:, None]
                dst_ptr = ((dst[:, None, None] * HEADS + h[None, :, None]) * HD) + d[None, None, :]
                dst_live = dst_ok[:, None, None] & h_live[None, :, None] & d_live[None, None, :]
                q_dst = tl.load(q_ptr + dst_ptr, mask=dst_live, other=0.0).to(tl.float32)
                go_dst = tl.load(go_ptr + dst_ptr, mask=dst_live, other=0.0).to(tl.float32)
                stat_ptr = dst[:, None] * HEADS + h[None, :]
                stat_live = dst_ok[:, None] & h_live[None, :]
                m_dst = tl.load(m_ptr + stat_ptr, mask=stat_live, other=0.0)
                l_dst = tl.load(l_ptr + stat_ptr, mask=stat_live, other=1.0)
                delta_dst = tl.load(delta_ptr + stat_ptr, mask=stat_live, other=0.0)
                score = _wa_scores_16(k_source, q_dst, scale, HEADS, BLOCK_H)
                score += tl.load(
                    bias_ptr + h[None, None, :] * stride_bias + cls[:, :, None],
                    mask=pair[:, :, None] & h_live[None, None, :], other=0.0,
                ).to(tl.float32)
                alpha = tl.where(pair[:, :, None], tl.exp(score - m_dst[None, :, :]) / l_dst[None, :, :], 0.0)
                dalpha = _wa_scores_16(v_source, go_dst, 1.0, HEADS, BLOCK_H)
                ds = alpha * (dalpha - delta_dst[None, :, :])
                acc_k += _wa_weighted_16(ds, q_dst, HEADS, BLOCK_H, BLOCK_HD)
                acc_v += _wa_weighted_16(alpha, go_dst, HEADS, BLOCK_H, BLOCK_HD)

        element = dk_ptr.dtype.element_ty
        tl.store(dk_ptr + source_ptr, (acc_k * scale).to(element), mask=source_live)
        tl.store(dv_ptr + source_ptr, acc_v.to(element), mask=source_live)


def _edge_dst(ptr: Tensor) -> Tensor:
    n_w = ptr.shape[0] - 1
    return torch.repeat_interleave(
        torch.arange(n_w, device=ptr.device), ptr[1:] - ptr[:-1]
    )


def _validate_attention(q, k, v, bias, tables: WaTables) -> None:
    if q.ndim != 3 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, v must share shape (N_w, heads, head_dim)")
    if bias.ndim != 2 or bias.shape != (q.shape[1], WA_CLASSES):
        raise ValueError("bias must have shape (heads, 48)")
    n_w = q.shape[0]
    n_blocks = tables.lane_win.shape[0]
    if tables.lane_win.shape != (n_blocks, _BLOCK_LANES):
        raise ValueError("lane_win must have shape (n_blocks, 16)")
    for name in ("block_prev", "block_next", "block_axis"):
        if getattr(tables, name).shape != (n_blocks,):
            raise ValueError(f"{name} must have one entry per block")
    if tables.claim_lo.shape != (n_blocks, _BLOCK_CELLS):
        raise ValueError("claim_lo must have shape (n_blocks, 31)")
    if tables.claim_hi.shape != tables.claim_lo.shape:
        raise ValueError("claim_hi must match claim_lo")
    claims = (n_w * _CLAIM_SPAN,)
    for name in ("cl_win", "cl_fold", "cl_axis"):
        if getattr(tables, name).shape != claims:
            raise ValueError(f"{name} must have sixteen entries per window")
    for name in ("w_block", "w_lane", "waxis"):
        if getattr(tables, name).shape != (n_w,):
            raise ValueError(f"{name} must have one entry per window")


def _reference_forward(q, k, v, bias, ptr, src, cls):
    """Sliced eager composition over expanded edges: parity and CPU path."""
    n_w, heads, hd = q.shape
    scale = 1.0 / math.sqrt(hd)
    dst = _edge_dst(ptr)
    e = src.shape[0]
    score = bias.t().float().index_select(0, cls)
    for a in range(heads):
        q_a, k_a = q[:, a], k[:, a]
        for lo in range(0, e, _EDGE_SLICE):
            sl = slice(lo, min(lo + _EDGE_SLICE, e))
            score[sl, a] += scale * (
                q_a.index_select(0, dst[sl]).float()
                * k_a.index_select(0, src[sl]).float()
            ).sum(-1)
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
                0, dst[sl], alpha[sl, a, None] * v_a.index_select(0, src[sl]).float()
            )
    out /= l.unsqueeze(-1)
    return out, m, l


def _reference_backward(q, k, v, bias, ptr, src, cls, m, l, delta, go):
    """Recompute alpha from saved stats, then the four gradients."""
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
        and q.shape[1] <= _WA_BLOCK_H
        and q.shape[2] <= _WA_BLOCK_HD
    )


def _table_args(tables: WaTables) -> tuple[Tensor, ...]:
    return tuple(tables)


def _launch_forward(q, k, v, bias, tables: WaTables):
    n_w, heads, hd = q.shape
    n_blocks = tables.lane_win.shape[0]
    out = torch.empty((n_w, heads, hd), dtype=torch.float32, device=q.device)
    m = torch.empty((n_w, heads), dtype=torch.float32, device=q.device)
    l = torch.empty((n_w, heads), dtype=torch.float32, device=q.device)
    _wa_forward_kernel[(n_blocks,)](
        q, k, v, bias, *_table_args(tables), out, m, l,
        1.0 / math.sqrt(hd), bias.stride(0), HEADS=heads, HD=hd,
        BLOCK_H=_WA_BLOCK_H, BLOCK_HD=_WA_BLOCK_HD, BLOCK_E=_WA_BLOCK_E,
        num_warps=_WA_NUM_WARPS,
    )
    return out, m, l


def _launch_backward(q, k, v, bias, tables: WaTables, m, l, delta, go):
    n_w, heads, hd = q.shape
    n_blocks = tables.lane_win.shape[0]
    scale = 1.0 / math.sqrt(hd)
    dq = torch.empty_like(q)
    partial = torch.empty(
        (n_blocks * heads, _WA_BLOCK_CLS), dtype=torch.float32, device=q.device
    )
    _wa_dq_kernel[(n_blocks,)](
        q, k, v, bias, go, *_table_args(tables), m, l, delta, dq, partial,
        scale, bias.stride(0), HEADS=heads, HD=hd, BLOCK_H=_WA_BLOCK_H,
        BLOCK_HD=_WA_BLOCK_HD, BLOCK_E=_WA_BLOCK_E,
        BLOCK_CLS=_WA_BLOCK_CLS, num_warps=_WA_NUM_WARPS,
    )
    dbias = (
        partial.view(n_blocks, heads, _WA_BLOCK_CLS)
        .sum(dim=0)[:, :WA_CLASSES]
        .contiguous()
    ).to(bias.dtype)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    _wa_dkdv_kernel[(n_blocks,)](
        q, k, v, bias, go, *_table_args(tables), m, l, delta, dk, dv,
        scale, bias.stride(0), HEADS=heads, HD=hd, BLOCK_H=_WA_BLOCK_H,
        BLOCK_HD=_WA_BLOCK_HD, BLOCK_E=_WA_BLOCK_E,
        num_warps=_WA_NUM_WARPS,
    )
    return dq, dk, dv, dbias


@torch.library.custom_op("mantisnet::window_attention", mutates_args=())
def _wa_op(
    q: Tensor, k: Tensor, v: Tensor, bias: Tensor,
    lane_win: Tensor, block_prev: Tensor, block_next: Tensor, block_axis: Tensor,
    claim_lo: Tensor, claim_hi: Tensor, cl_win: Tensor, cl_fold: Tensor,
    cl_axis: Tensor, w_block: Tensor, w_lane: Tensor, waxis: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    tables = WaTables(
        lane_win, block_prev, block_next, block_axis, claim_lo, claim_hi,
        cl_win, cl_fold, cl_axis, w_block, w_lane, waxis,
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
            RuntimeWarning, stacklevel=2,
        )
        return _reference_forward(q, k, v, bias, *_expanded_edges(tables))


@_wa_op.register_fake
def _(
    q, k, v, bias, lane_win, block_prev, block_next, block_axis,
    claim_lo, claim_hi, cl_win, cl_fold, cl_axis, w_block, w_lane, waxis,
):
    n_w, heads, hd = q.shape
    return (
        q.new_empty((n_w, heads, hd), dtype=torch.float32),
        q.new_empty((n_w, heads), dtype=torch.float32),
        q.new_empty((n_w, heads), dtype=torch.float32),
    )


@torch.library.custom_op("mantisnet::window_attention_backward", mutates_args=())
def _wa_backward_op(
    q: Tensor, k: Tensor, v: Tensor, bias: Tensor,
    lane_win: Tensor, block_prev: Tensor, block_next: Tensor, block_axis: Tensor,
    claim_lo: Tensor, claim_hi: Tensor, cl_win: Tensor, cl_fold: Tensor,
    cl_axis: Tensor, w_block: Tensor, w_lane: Tensor, waxis: Tensor,
    out: Tensor, m: Tensor, l: Tensor, grad_out: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    tables = WaTables(
        lane_win, block_prev, block_next, block_axis, claim_lo, claim_hi,
        cl_win, cl_fold, cl_axis, w_block, w_lane, waxis,
    )
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    go = grad_out.contiguous().float()
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
            RuntimeWarning, stacklevel=2,
        )
        return _reference_backward(
            q, k, v, bias, *_expanded_edges(tables), m, l, delta, go
        )


@_wa_backward_op.register_fake
def _(
    q, k, v, bias, lane_win, block_prev, block_next, block_axis,
    claim_lo, claim_hi, cl_win, cl_fold, cl_axis, w_block, w_lane, waxis,
    out, m, l, grad_out,
):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v), torch.empty_like(bias)


def _wa_setup_context(ctx, inputs, output) -> None:
    out, m, l = output
    ctx.save_for_backward(*inputs, out, m, l)


def _wa_dispatch_backward(ctx, grad_out, _grad_m, _grad_l):
    dq, dk, dv, dbias = _wa_backward_op(*ctx.saved_tensors, grad_out)
    return (dq, dk, dv, dbias) + (None,) * 12


_wa_op.register_autograd(_wa_dispatch_backward, setup_context=_wa_setup_context)


def edge_attention(
    q: Tensor, k: Tensor, v: Tensor, bias: Tensor,
    lane_win: Tensor, block_prev: Tensor, block_next: Tensor, block_axis: Tensor,
    claim_lo: Tensor, claim_hi: Tensor, cl_win: Tensor, cl_fold: Tensor,
    cl_axis: Tensor, w_block: Tensor, w_lane: Tensor, waxis: Tensor,
) -> Tensor:
    """Section 5.1c line-blocked attention, returned in fp32."""
    out, _m, _l = _wa_op(
        q, k, v, bias, lane_win, block_prev, block_next, block_axis,
        claim_lo, claim_hi, cl_win, cl_fold, cl_axis, w_block, w_lane, waxis,
    )
    return out
