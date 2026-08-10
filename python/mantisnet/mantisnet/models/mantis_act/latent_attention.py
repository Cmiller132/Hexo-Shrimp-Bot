"""Section 17's latent read and broadcast, fused into flash-style segment kernels.

Read (§17.2): tiles over ragged nodes per position, storing only the
``(P, K, C, heads, head_dim)`` output and softmax stats.  Broadcast (§17.4):
constant key set, entire softmax in registers.  Both backwards recompute
from saved inputs.  ``latent_segments`` records per-family ranges.
Accumulators are fp32 (§27).  CPU: torch parity reference of §36.
"""

from __future__ import annotations

import math
import warnings
from typing import Sequence

import torch
from torch import Tensor

from .equivariant import at_least_fp32
from .plans import LatentSegments

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only without triton
    triton = None
    tl = None


# A node tile of the ragged walk. The read's programs hold a
# ``(K, BLOCK_E, head_dim)`` intermediate in registers, so the tile is sized
# against that product rather than against the segment length.
_BLOCK_E = 32
# The broadcast's row sweep holds ``(BLOCK_E, R, head_dim)`` instead, and ``R``
# is up to eight where ``K`` is four.
_BCAST_BLOCK_E = 16

# How far a small grid is allowed to be split across a position's rows, and the
# program count worth reaching for. A 4070 Ti has 60 multiprocessors; a few
# hundred programs saturate it and more only shortens the tail.
_MAX_SPLITS = 64
_TARGET_PROGRAMS = 2048

# One program covers a head_dim-wide row at a few elements per lane. The read's
# programs carry a three-dimensional intermediate and want the extra warps; the
# per-row kernels are a single row wide and would leave them idle.
_TILE_WARPS = 4
_ROW_WARPS = 1

_FAILED_SHAPES: dict[tuple[object, ...], str] = {}
_FAILED_BACKWARD_SHAPES: dict[tuple[object, ...], str] = {}


def row_positions(offsets: Tensor, n_rows: int) -> Tensor:
    """The position owning each row of a ragged family, from its CSR offsets.

    ``n_rows`` is the family's row count, which every caller already holds as
    a host-side tensor shape. It is required rather than derived, since
    deriving it would mean reading ``offsets[-1]`` back off the device — a
    blocking host stall. ATen refuses an ``output_size`` that disagrees with
    the offsets' own total (``aten/src/ATen/native/Repeat.h``'s
    ``result_size == cumsum_ptr[size - 1]``), so that equality is enforced on
    every call, from the device.
    """
    if offsets.ndim != 1 or offsets.shape[0] < 1:
        raise ValueError(
            f"offsets must be a 1-D (P + 1,) tensor, got {tuple(offsets.shape)}"
        )
    if n_rows < 0:
        raise ValueError(f"n_rows must not be negative, got {n_rows}")
    counts = offsets[1:] - offsets[:-1]
    return torch.repeat_interleave(
        torch.arange(offsets.shape[0] - 1, device=offsets.device),
        counts,
        output_size=n_rows,
    )


# --------------------------------------------------------------------------
# The multi-range view of a read's key rows


def latent_segments(
    offsets: Sequence[Tensor], row_pos: Sequence[Tensor]
) -> LatentSegments:
    """The concatenation of several ragged families, viewed per position.

    ``offsets`` is one ``(P + 1,)`` CSR offset array per family, in the order
    the families' rows are concatenated; ``row_pos`` is each family's
    already-computed :func:`row_positions`. Nothing is sorted: a position's
    rows are already contiguous inside each family, giving exactly ``F``
    contiguous ranges to walk.

    The per-family ``row_pos`` is taken rather than derived, since it depends
    on the batch alone: one vector per family per forward serves every latent
    read and broadcast, rather than being rebuilt per block.

    Every arithmetic step stays on the device: reading a family's total off
    its last offset would be a host synchronisation, which this avoids by
    reading the row count off ``row_pos``'s host-side length instead.
    """
    families = list(offsets)
    rows = list(row_pos)
    if not families:
        raise ValueError("a latent read needs at least one node family")
    if len(rows) != len(families):
        raise ValueError(
            f"{len(families)} families of offsets against {len(rows)} row-position "
            "vectors: each family carries exactly one"
        )
    positions = int(families[0].shape[0]) - 1
    for index, family in enumerate(families):
        if family.ndim != 1 or int(family.shape[0]) - 1 != positions:
            raise ValueError(
                f"family {index} has offsets of shape {tuple(family.shape)} "
                f"against the first family's {positions} positions"
            )
        if rows[index].ndim != 1:
            raise ValueError(
                f"family {index} has row positions of shape "
                f"{tuple(rows[index].shape)}, which is not a 1-D row vector"
            )

    row_pos = torch.cat(rows)
    totals = torch.stack([family[-1] for family in families])
    base = torch.cat((totals.new_zeros(1), totals.cumsum(0)[:-1]))
    starts = torch.stack([family[:-1] for family in families], dim=1) + base
    ends = torch.stack([family[1:] for family in families], dim=1) + base

    lengths = ends - starts
    return LatentSegments(
        ranges=torch.stack((starts, ends), dim=-1).to(torch.int32).contiguous(),
        range_base=(lengths.cumsum(dim=1) - lengths).to(torch.int32).contiguous(),
        counts=lengths.sum(dim=1).to(torch.int32).contiguous(),
        row_pos=row_pos,
        n_rows=int(row_pos.shape[0]),
        positions=positions,
        families=len(families),
    )


def _splits(programs: int, rows_per_position: int) -> int:
    """How many programs share one position's rows (a power of two).

    Bounded three ways: by the program count worth reaching, by how many tiles
    the average position even has to give away, and by a hard cap, because past
    that point the partial buffer and its second-stage reduction cost more than
    the tail they shorten.
    """
    wanted = max(1, rows_per_position // (_BLOCK_E * 2))
    target = max(1, _TARGET_PROGRAMS // max(programs, 1))
    splits = 1
    while splits * 2 <= min(wanted, target, _MAX_SPLITS):
        splits *= 2
    return splits


# --------------------------------------------------------------------------
# Kernels


if triton is not None:

    @triton.jit
    def _read_forward_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        ranges_ptr,
        base_ptr,
        counts_ptr,
        acc_ptr,
        m_ptr,
        l_ptr,
        scale,
        P: tl.constexpr,
        K: tl.constexpr,
        KPAD: tl.constexpr,
        C: tl.constexpr,
        H: tl.constexpr,
        HD: tl.constexpr,
        BLOCK_HD: tl.constexpr,
        F: tl.constexpr,
        SPLITS: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        # One program per (position, channel, head) and slice of that
        # position's rows. It holds every latent slot's running maximum,
        # denominator and accumulator at once, so a node row is loaded once for
        # all K slots instead of once per slot.
        pid = tl.program_id(0)
        part = tl.program_id(1)
        head = pid % H
        chan = (pid // H) % C
        pos = pid // (H * C)

        offs = tl.arange(0, BLOCK_HD)
        live = offs < HD
        slots = tl.arange(0, KPAD)
        held = slots < K

        query = tl.load(
            q_ptr + (((pos * K + slots[:, None]) * C + chan) * H + head) * HD
            + offs[None, :],
            mask=held[:, None] & live[None, :],
            other=0.0,
        ).to(tl.float32)

        total = tl.load(counts_ptr + pos)
        per = (total + SPLITS - 1) // SPLITS
        lo = part * per
        hi = tl.minimum(lo + per, total)

        m = tl.full([KPAD], -float("inf"), tl.float32)
        denominator = tl.zeros([KPAD], tl.float32)
        acc = tl.zeros([KPAD, BLOCK_HD], tl.float32)

        for family in tl.static_range(F):
            span = tl.load(ranges_ptr + (pos * F + family) * 2)
            span_end = tl.load(ranges_ptr + (pos * F + family) * 2 + 1)
            logical = tl.load(base_ptr + pos * F + family)
            begin = span + tl.maximum(lo, logical) - logical
            end = span + tl.minimum(hi, logical + (span_end - span)) - logical
            for tile in tl.range(begin, end, BLOCK_E):
                rows = tile + tl.arange(0, BLOCK_E)
                inside = rows < end
                row_off = ((rows * C + chan) * H + head) * HD
                keys = tl.load(
                    k_ptr + row_off[:, None] + offs[None, :],
                    mask=inside[:, None] & live[None, :],
                    other=0.0,
                ).to(tl.float32)
                score = tl.sum(query[:, None, :] * keys[None, :, :], axis=2) * scale
                score = tl.where(inside[None, :], score, -float("inf"))
                m_new = tl.maximum(m, tl.max(score, axis=1))
                rescale = tl.exp(m - m_new)
                weight = tl.exp(score - m_new[:, None])
                denominator = denominator * rescale + tl.sum(weight, axis=1)
                values = tl.load(
                    v_ptr + row_off[:, None] + offs[None, :],
                    mask=inside[:, None] & live[None, :],
                    other=0.0,
                ).to(tl.float32)
                acc = acc * rescale[:, None] + tl.sum(
                    weight[:, :, None] * values[None, :, :], axis=1
                )
                m = m_new

        stat = (((part * P + pos) * K + slots) * C + chan) * H + head
        tl.store(
            acc_ptr + stat[:, None] * HD + offs[None, :],
            acc,
            mask=held[:, None] & live[None, :],
        )
        tl.store(m_ptr + stat, m, mask=held)
        tl.store(l_ptr + stat, denominator, mask=held)

    @triton.jit
    def _read_dq_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        go_ptr,
        m_ptr,
        l_ptr,
        delta_ptr,
        ranges_ptr,
        base_ptr,
        counts_ptr,
        dq_ptr,
        scale,
        P: tl.constexpr,
        K: tl.constexpr,
        KPAD: tl.constexpr,
        C: tl.constexpr,
        H: tl.constexpr,
        HD: tl.constexpr,
        BLOCK_HD: tl.constexpr,
        F: tl.constexpr,
        SPLITS: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        # The forward's walk again, with the saved statistics in place of the
        # online softmax: every score is recomputed from q and k rather than
        # read back from a tensor the forward would have had to keep.
        pid = tl.program_id(0)
        part = tl.program_id(1)
        head = pid % H
        chan = (pid // H) % C
        pos = pid // (H * C)

        offs = tl.arange(0, BLOCK_HD)
        live = offs < HD
        slots = tl.arange(0, KPAD)
        held = slots < K

        latent = (((pos * K + slots) * C + chan) * H + head) * HD
        query = tl.load(
            q_ptr + latent[:, None] + offs[None, :],
            mask=held[:, None] & live[None, :],
            other=0.0,
        ).to(tl.float32)
        upstream = tl.load(
            go_ptr + latent[:, None] + offs[None, :],
            mask=held[:, None] & live[None, :],
            other=0.0,
        ).to(tl.float32)
        stat = ((pos * K + slots) * C + chan) * H + head
        m = tl.load(m_ptr + stat, mask=held, other=0.0)
        denominator = tl.load(l_ptr + stat, mask=held, other=1.0)
        delta = tl.load(delta_ptr + stat, mask=held, other=0.0)

        total = tl.load(counts_ptr + pos)
        per = (total + SPLITS - 1) // SPLITS
        lo = part * per
        hi = tl.minimum(lo + per, total)
        acc = tl.zeros([KPAD, BLOCK_HD], tl.float32)

        for family in tl.static_range(F):
            span = tl.load(ranges_ptr + (pos * F + family) * 2)
            span_end = tl.load(ranges_ptr + (pos * F + family) * 2 + 1)
            logical = tl.load(base_ptr + pos * F + family)
            begin = span + tl.maximum(lo, logical) - logical
            end = span + tl.minimum(hi, logical + (span_end - span)) - logical
            for tile in tl.range(begin, end, BLOCK_E):
                rows = tile + tl.arange(0, BLOCK_E)
                inside = rows < end
                row_off = ((rows * C + chan) * H + head) * HD
                keys = tl.load(
                    k_ptr + row_off[:, None] + offs[None, :],
                    mask=inside[:, None] & live[None, :],
                    other=0.0,
                ).to(tl.float32)
                values = tl.load(
                    v_ptr + row_off[:, None] + offs[None, :],
                    mask=inside[:, None] & live[None, :],
                    other=0.0,
                ).to(tl.float32)
                score = tl.sum(query[:, None, :] * keys[None, :, :], axis=2) * scale
                score = tl.where(inside[None, :], score, -float("inf"))
                alpha = tl.exp(score - m[:, None]) / denominator[:, None]
                d_alpha = tl.sum(upstream[:, None, :] * values[None, :, :], axis=2)
                d_score = alpha * (d_alpha - delta[:, None])
                acc += tl.sum(d_score[:, :, None] * keys[None, :, :], axis=1)

        out = (((part * P + pos) * K + slots) * C + chan) * H + head
        tl.store(
            dq_ptr + out[:, None] * HD + offs[None, :],
            acc * scale,
            mask=held[:, None] & live[None, :],
        )

    @triton.jit
    def _read_dkv_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        go_ptr,
        m_ptr,
        l_ptr,
        delta_ptr,
        row_pos_ptr,
        dk_ptr,
        dv_ptr,
        scale,
        K: tl.constexpr,
        KPAD: tl.constexpr,
        C: tl.constexpr,
        H: tl.constexpr,
        HD: tl.constexpr,
        BLOCK_HD: tl.constexpr,
    ):
        # One program per (node row, channel, head). A node's key and value
        # gradients are sums over the K latent slots alone — a fixed, tiny
        # reduction that no other program touches — so this side of the
        # backward needs neither a scatter nor a split.
        pid = tl.program_id(0)
        head = pid % H
        chan = (pid // H) % C
        row = pid // (H * C)
        pos = tl.load(row_pos_ptr + row)

        offs = tl.arange(0, BLOCK_HD)
        live = offs < HD
        slots = tl.arange(0, KPAD)
        held = slots < K

        row_off = ((row * C + chan) * H + head) * HD
        key = tl.load(k_ptr + row_off + offs, mask=live, other=0.0).to(tl.float32)
        value = tl.load(v_ptr + row_off + offs, mask=live, other=0.0).to(tl.float32)

        latent = (((pos * K + slots) * C + chan) * H + head) * HD
        query = tl.load(
            q_ptr + latent[:, None] + offs[None, :],
            mask=held[:, None] & live[None, :],
            other=0.0,
        ).to(tl.float32)
        upstream = tl.load(
            go_ptr + latent[:, None] + offs[None, :],
            mask=held[:, None] & live[None, :],
            other=0.0,
        ).to(tl.float32)
        stat = ((pos * K + slots) * C + chan) * H + head
        m = tl.load(m_ptr + stat, mask=held, other=0.0)
        denominator = tl.load(l_ptr + stat, mask=held, other=1.0)
        delta = tl.load(delta_ptr + stat, mask=held, other=0.0)

        score = tl.sum(query * key[None, :], axis=1) * scale
        alpha = tl.where(held, tl.exp(score - m) / denominator, 0.0)
        d_alpha = tl.sum(upstream * value[None, :], axis=1)
        d_score = alpha * (d_alpha - delta)

        element = dk_ptr.dtype.element_ty
        tl.store(
            dk_ptr + row_off + offs,
            (tl.sum(d_score[:, None] * query, axis=0) * scale).to(element),
            mask=live,
        )
        tl.store(
            dv_ptr + row_off + offs,
            tl.sum(alpha[:, None] * upstream, axis=0).to(element),
            mask=live,
        )

    @triton.jit
    def _bcast_forward_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        node_pos_ptr,
        out_ptr,
        scale,
        R: tl.constexpr,
        RPAD: tl.constexpr,
        C: tl.constexpr,
        H: tl.constexpr,
        HD: tl.constexpr,
        BLOCK_HD: tl.constexpr,
    ):
        # One program per (node row, channel, head). The key set is this
        # position's R context rows — a configured constant — so the whole
        # softmax fits in registers and the backward can recompute it from the
        # same three tensors rather than save anything.
        pid = tl.program_id(0)
        head = pid % H
        chan = (pid // H) % C
        row = pid // (H * C)
        pos = tl.load(node_pos_ptr + row)

        offs = tl.arange(0, BLOCK_HD)
        live = offs < HD
        context = tl.arange(0, RPAD)
        present = context < R

        query = tl.load(
            q_ptr + ((row * C + chan) * H + head) * HD + offs, mask=live, other=0.0
        ).to(tl.float32)
        source = (((pos * R + context[:, None]) * C + chan) * H + head) * HD
        keys = tl.load(
            k_ptr + source + offs[None, :],
            mask=present[:, None] & live[None, :],
            other=0.0,
        ).to(tl.float32)
        values = tl.load(
            v_ptr + source + offs[None, :],
            mask=present[:, None] & live[None, :],
            other=0.0,
        ).to(tl.float32)

        score = tl.sum(query[None, :] * keys, axis=1) * scale
        score = tl.where(present, score, -float("inf"))
        weight = tl.exp(score - tl.max(score, axis=0))
        weight = weight / tl.sum(weight, axis=0)
        tl.store(
            out_ptr + ((row * C + chan) * H + head) * HD + offs,
            tl.sum(weight[:, None] * values, axis=0),
            mask=live,
        )

    @triton.jit
    def _bcast_dq_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        go_ptr,
        node_pos_ptr,
        dq_ptr,
        scale,
        R: tl.constexpr,
        RPAD: tl.constexpr,
        C: tl.constexpr,
        H: tl.constexpr,
        HD: tl.constexpr,
        BLOCK_HD: tl.constexpr,
    ):
        # The forward recomputed, then the softmax backward over R terms.
        pid = tl.program_id(0)
        head = pid % H
        chan = (pid // H) % C
        row = pid // (H * C)
        pos = tl.load(node_pos_ptr + row)

        offs = tl.arange(0, BLOCK_HD)
        live = offs < HD
        context = tl.arange(0, RPAD)
        present = context < R

        node = ((row * C + chan) * H + head) * HD
        query = tl.load(q_ptr + node + offs, mask=live, other=0.0).to(tl.float32)
        upstream = tl.load(go_ptr + node + offs, mask=live, other=0.0).to(tl.float32)
        source = (((pos * R + context[:, None]) * C + chan) * H + head) * HD
        keys = tl.load(
            k_ptr + source + offs[None, :],
            mask=present[:, None] & live[None, :],
            other=0.0,
        ).to(tl.float32)
        values = tl.load(
            v_ptr + source + offs[None, :],
            mask=present[:, None] & live[None, :],
            other=0.0,
        ).to(tl.float32)

        score = tl.sum(query[None, :] * keys, axis=1) * scale
        score = tl.where(present, score, -float("inf"))
        alpha = tl.exp(score - tl.max(score, axis=0))
        alpha = alpha / tl.sum(alpha, axis=0)
        d_alpha = tl.sum(upstream[None, :] * values, axis=1)
        d_score = alpha * (d_alpha - tl.sum(alpha * d_alpha, axis=0))
        element = dq_ptr.dtype.element_ty
        tl.store(
            dq_ptr + node + offs,
            (tl.sum(d_score[:, None] * keys, axis=0) * scale).to(element),
            mask=live,
        )

    @triton.jit
    def _bcast_dkv_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        go_ptr,
        offsets_ptr,
        dk_ptr,
        dv_ptr,
        scale,
        P: tl.constexpr,
        R: tl.constexpr,
        RPAD: tl.constexpr,
        C: tl.constexpr,
        H: tl.constexpr,
        HD: tl.constexpr,
        BLOCK_HD: tl.constexpr,
        SPLITS: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        # One program per (position, channel, head) and slice of that
        # position's rows, holding all R context gradients at once: the whole
        # R-wide softmax of a node has to be recomputed anyway, so computing it
        # once for every context row costs nothing extra.
        pid = tl.program_id(0)
        part = tl.program_id(1)
        head = pid % H
        chan = (pid // H) % C
        pos = pid // (H * C)

        offs = tl.arange(0, BLOCK_HD)
        live = offs < HD
        context = tl.arange(0, RPAD)
        present = context < R

        source = (((pos * R + context[:, None]) * C + chan) * H + head) * HD
        keys = tl.load(
            k_ptr + source + offs[None, :],
            mask=present[:, None] & live[None, :],
            other=0.0,
        ).to(tl.float32)
        values = tl.load(
            v_ptr + source + offs[None, :],
            mask=present[:, None] & live[None, :],
            other=0.0,
        ).to(tl.float32)

        start = tl.load(offsets_ptr + pos)
        finish = tl.load(offsets_ptr + pos + 1)
        per = (finish - start + SPLITS - 1) // SPLITS
        lo = start + part * per
        hi = tl.minimum(lo + per, finish)

        acc_k = tl.zeros([RPAD, BLOCK_HD], tl.float32)
        acc_v = tl.zeros([RPAD, BLOCK_HD], tl.float32)
        for tile in tl.range(lo, hi, BLOCK_E):
            rows = tile + tl.arange(0, BLOCK_E)
            inside = rows < hi
            node = ((rows * C + chan) * H + head) * HD
            query = tl.load(
                q_ptr + node[:, None] + offs[None, :],
                mask=inside[:, None] & live[None, :],
                other=0.0,
            ).to(tl.float32)
            upstream = tl.load(
                go_ptr + node[:, None] + offs[None, :],
                mask=inside[:, None] & live[None, :],
                other=0.0,
            ).to(tl.float32)
            score = tl.sum(query[:, None, :] * keys[None, :, :], axis=2) * scale
            score = tl.where(present[None, :], score, -float("inf"))
            alpha = tl.exp(score - tl.max(score, axis=1)[:, None])
            alpha = alpha / tl.sum(alpha, axis=1)[:, None]
            alpha = tl.where(inside[:, None], alpha, 0.0)
            d_alpha = tl.sum(upstream[:, None, :] * values[None, :, :], axis=2)
            d_score = alpha * (d_alpha - tl.sum(alpha * d_alpha, axis=1)[:, None])
            acc_k += tl.sum(d_score[:, :, None] * query[:, None, :], axis=0)
            acc_v += tl.sum(alpha[:, :, None] * upstream[:, None, :], axis=0)

        out = (((part * P + pos) * R + context[:, None]) * C + chan) * H + head
        mask = present[:, None] & live[None, :]
        tl.store(dk_ptr + out * HD + offs[None, :], acc_k * scale, mask=mask)
        tl.store(dv_ptr + out * HD + offs[None, :], acc_v, mask=mask)


# --------------------------------------------------------------------------
# The torch reference (§36) — CPU, unsupported signatures, and parity


def read_reference(
    q: Tensor, k: Tensor, v: Tensor, row_pos: Tensor, position_count: int
) -> tuple[Tensor, Tensor, Tensor]:
    """§17.2's read as the gather the kernel exists to avoid, plus its statistics.

    Returns the ``(P, K, C, heads, head_dim)`` output beside the segment
    maximum and denominator, since the fused backward is defined against
    those statistics.

    Literal: gather every position's queries onto its rows, score, shift by
    the segment maximum, exponentiate, and scatter the weighted values back.
    Scores, softmax, and the weighted sum run at no less than fp32 (§27). The
    queries are promoted before the gather, not after, to avoid a bf16 scatter
    of N row gradients over P latent rows in the backward.

    A position with no rows reads zero — reachable under
    ``full_occupied_cells_only`` on an empty board, and the only finite answer
    a softmax over an empty set can give.
    """
    channels, heads, head_dim = k.shape[1:]
    slots = q.shape[1]
    scale = 1.0 / math.sqrt(head_dim)
    stats = (position_count, slots, channels, heads)

    query = at_least_fp32(q).index_select(0, row_pos)
    score = (query * at_least_fp32(k).unsqueeze(1)).sum(-1) * scale

    # Segment softmax per (position, slot, channel, head), shifted by the
    # segment maximum so a long ragged segment cannot overflow the exponential.
    maxima = score.new_full(stats, torch.finfo(score.dtype).min).scatter_reduce_(
        0, row_pos.view(-1, 1, 1, 1).expand_as(score), score, reduce="amax"
    )
    weight = (score - maxima.index_select(0, row_pos)).exp()
    total = score.new_zeros(stats).index_add_(0, row_pos, weight)
    out = score.new_zeros((*stats, head_dim)).index_add_(
        0, row_pos, weight.unsqueeze(-1) * at_least_fp32(v).unsqueeze(1)
    )
    denominator = torch.where(total > 0, total, torch.ones_like(total))
    return out / denominator.unsqueeze(-1), maxima, total


def broadcast_reference(q: Tensor, k: Tensor, v: Tensor, node_pos: Tensor) -> Tensor:
    """§17.4's broadcast as the gather the kernel exists to avoid.

    ``q`` is ``(N, C, heads, head_dim)`` and ``k`` and ``v`` are
    ``(P, R, C, heads, head_dim)``: every node attends over the ``R`` context
    rows of its own position and nothing else. The context is promoted before
    the gather for the same reason the read's queries are.
    """
    scale = 1.0 / math.sqrt(k.shape[-1])
    keys = at_least_fp32(k).index_select(0, node_pos)
    values = at_least_fp32(v).index_select(0, node_pos)
    score = (at_least_fp32(q).unsqueeze(1) * keys).sum(-1) * scale
    weight = score.softmax(dim=1)
    return (weight.unsqueeze(-1) * values).sum(dim=1)


def read_reference_backward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    row_pos: Tensor,
    m: Tensor,
    denominator: Tensor,
    delta: Tensor,
    grad_out: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """:func:`read_reference`'s gradient, over the same gathered tensors.

    The weights are recomputed from the saved segment statistics rather than
    kept from the forward — the same trade the kernel makes, written where a
    ``gradcheck`` can reach it. ``delta`` is the row sum of the upstream
    gradient against the output, which is algebraically each segment's
    ``Σ alpha * dalpha`` and saves a first sweep over the rows.
    """
    scale = 1.0 / math.sqrt(k.shape[-1])
    query = at_least_fp32(q).index_select(0, row_pos)
    keys = at_least_fp32(k).unsqueeze(1)
    values = at_least_fp32(v).unsqueeze(1)
    upstream = at_least_fp32(grad_out).index_select(0, row_pos)

    score = (query * keys).sum(-1) * scale
    alpha = (score - m.index_select(0, row_pos)).exp() / denominator.index_select(
        0, row_pos
    )
    d_score = alpha * (
        (upstream * values).sum(-1) - delta.index_select(0, row_pos)
    )
    d_query = torch.zeros(
        q.shape, dtype=score.dtype, device=q.device
    ).index_add_(0, row_pos, d_score.unsqueeze(-1) * keys)
    return (
        (d_query * scale).to(q.dtype),
        ((d_score.unsqueeze(-1) * query).sum(1) * scale).to(k.dtype),
        (alpha.unsqueeze(-1) * upstream).sum(1).to(v.dtype),
    )


def broadcast_reference_backward(
    q: Tensor, k: Tensor, v: Tensor, node_pos: Tensor, grad_out: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """:func:`broadcast_reference`'s gradient, over the same gathered tensors.

    The key set is a configured constant, so nothing is saved from the forward
    at all: the softmax is recomputed from ``q`` and ``k``, which is cheaper
    than the ``(N, R, C, heads)`` tensor keeping it would cost.
    """
    scale = 1.0 / math.sqrt(k.shape[-1])
    keys = at_least_fp32(k).index_select(0, node_pos)
    values = at_least_fp32(v).index_select(0, node_pos)
    query = at_least_fp32(q).unsqueeze(1)
    upstream = at_least_fp32(grad_out).unsqueeze(1)

    alpha = ((query * keys).sum(-1) * scale).softmax(dim=1)
    d_alpha = (upstream * values).sum(-1)
    d_score = alpha * (d_alpha - (alpha * d_alpha).sum(dim=1, keepdim=True))

    d_key = torch.zeros(k.shape, dtype=alpha.dtype, device=k.device).index_add_(
        0, node_pos, d_score.unsqueeze(-1) * query
    )
    d_value = torch.zeros(v.shape, dtype=alpha.dtype, device=v.device).index_add_(
        0, node_pos, alpha.unsqueeze(-1) * upstream
    )
    return (
        ((d_score.unsqueeze(-1) * keys).sum(1) * scale).to(q.dtype),
        (d_key * scale).to(k.dtype),
        d_value.to(v.dtype),
    )


# --------------------------------------------------------------------------
# Guards, launches, and the custom ops


def validate_read(q: Tensor, k: Tensor, v: Tensor, row_pos: Tensor) -> None:
    """Refuse a read whose shapes, dtypes or devices do not line up.

    The op's front door, ahead of the dispatch between the fused kernel and
    `read_reference`, so both are reached only through one signature check.
    """
    if q.ndim != 5:
        raise ValueError(f"q must be (P, K, C, heads, head_dim), got {tuple(q.shape)}")
    if k.ndim != 4 or k.shape != v.shape:
        raise ValueError(
            f"k and v must share shape (N, C, heads, head_dim), got "
            f"{tuple(k.shape)} and {tuple(v.shape)}"
        )
    if q.shape[2:] != k.shape[1:]:
        raise ValueError(
            f"q channels/heads/head_dim {tuple(q.shape[2:])} disagree with k's "
            f"{tuple(k.shape[1:])}"
        )
    if row_pos.ndim != 1 or row_pos.shape[0] != k.shape[0]:
        raise ValueError(
            f"row_pos must be ({k.shape[0]},) to match the key rows, got "
            f"{tuple(row_pos.shape)}"
        )
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(
            f"q, k and v must share a dtype, got {q.dtype}, {k.dtype}, {v.dtype}"
        )
    if k.device != q.device or v.device != q.device or row_pos.device != q.device:
        raise ValueError("every latent-read input must be on one device")


def _validate_broadcast(q: Tensor, k: Tensor, v: Tensor, node_pos: Tensor) -> None:
    if q.ndim != 4:
        raise ValueError(f"q must be (N, C, heads, head_dim), got {tuple(q.shape)}")
    if k.ndim != 5 or k.shape != v.shape:
        raise ValueError(
            f"k and v must share shape (P, R, C, heads, head_dim), got "
            f"{tuple(k.shape)} and {tuple(v.shape)}"
        )
    if q.shape[1:] != k.shape[2:]:
        raise ValueError(
            f"q channels/heads/head_dim {tuple(q.shape[1:])} disagree with the "
            f"context's {tuple(k.shape[2:])}"
        )
    if node_pos.ndim != 1 or node_pos.shape[0] != q.shape[0]:
        raise ValueError(
            f"node_pos must be ({q.shape[0]},) to match the query rows, got "
            f"{tuple(node_pos.shape)}"
        )
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(
            f"q, k and v must share a dtype, got {q.dtype}, {k.dtype}, {v.dtype}"
        )
    if k.device != q.device or v.device != q.device or node_pos.device != q.device:
        raise ValueError("every latent-broadcast input must be on one device")


def _shape_key(attention: str, q: Tensor, keys: int) -> tuple[object, ...]:
    """What a launch failure is remembered by: the two attentions share a cache.

    ``keys`` is the read's family count and the broadcast's context width — in
    both cases the one thing the kernel specialises on that the query shape
    does not carry.
    """
    return (
        attention,
        q.device.type,
        q.device.index,
        q.dtype,
        tuple(int(size) for size in q.shape[1:]),
        keys,
    )


def _supported(q: Tensor, n_rows: int) -> bool:
    return (
        triton is not None
        and q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and int(q.shape[-1]) <= 128
        and n_rows > 0
        and q.shape[0] > 0
    )


def _combine(
    acc: Tensor, m: Tensor, denominator: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Fold the splits' partial softmaxes into one, in index order.

    A position whose every slice is empty has ``m = -inf`` everywhere; its
    shift is taken as zero so the weights are zero rather than ``nan``, and it
    reads zero, which is what a softmax over an empty set gives.

    The three partial buffers are this module's own and dead after the fold, so
    the rescaling is in place: a second copy of the accumulator would be the
    largest allocation on the whole path.
    """
    peak = m.amax(dim=0)
    peak = torch.where(peak > -float("inf"), peak, torch.zeros_like(peak))
    weight = m.sub_(peak).exp_()
    total = denominator.mul_(weight).sum(dim=0)
    summed = acc.mul_(weight.unsqueeze(-1)).sum(dim=0)
    return summed / torch.where(total > 0, total, torch.ones_like(total)).unsqueeze(
        -1
    ), peak, total


def _launch_read(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    ranges: Tensor,
    range_base: Tensor,
    counts: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    positions, slots, channels, heads, head_dim = q.shape
    families = int(ranges.shape[1])
    programs = positions * channels * heads
    splits = _splits(programs, int(k.shape[0]) // max(positions, 1))
    pad = triton.next_power_of_2(slots)
    partial = torch.empty(
        (splits, positions, slots, channels, heads, head_dim),
        dtype=torch.float32,
        device=q.device,
    )
    stats = torch.empty(
        (2, splits, positions, slots, channels, heads),
        dtype=torch.float32,
        device=q.device,
    )
    _read_forward_kernel[(programs, splits)](
        q,
        k,
        v,
        ranges,
        range_base,
        counts,
        partial,
        stats[0],
        stats[1],
        1.0 / math.sqrt(head_dim),
        P=positions,
        K=slots,
        KPAD=pad,
        C=channels,
        H=heads,
        HD=head_dim,
        BLOCK_HD=triton.next_power_of_2(head_dim),
        F=families,
        SPLITS=splits,
        BLOCK_E=_BLOCK_E,
        num_warps=_TILE_WARPS,
    )
    return _combine(partial, stats[0], stats[1])


def _launch_read_backward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    ranges: Tensor,
    range_base: Tensor,
    counts: Tensor,
    row_pos: Tensor,
    m: Tensor,
    denominator: Tensor,
    delta: Tensor,
    grad_out: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    positions, slots, channels, heads, head_dim = q.shape
    n_rows = int(k.shape[0])
    families = int(ranges.shape[1])
    programs = positions * channels * heads
    splits = _splits(programs, n_rows // max(positions, 1))
    pad = triton.next_power_of_2(slots)
    block_hd = triton.next_power_of_2(head_dim)
    scale = 1.0 / math.sqrt(head_dim)

    partial = torch.empty(
        (splits, *q.shape), dtype=torch.float32, device=q.device
    )
    _read_dq_kernel[(programs, splits)](
        q,
        k,
        v,
        grad_out,
        m,
        denominator,
        delta,
        ranges,
        range_base,
        counts,
        partial,
        scale,
        P=positions,
        K=slots,
        KPAD=pad,
        C=channels,
        H=heads,
        HD=head_dim,
        BLOCK_HD=block_hd,
        F=families,
        SPLITS=splits,
        BLOCK_E=_BLOCK_E,
        num_warps=_TILE_WARPS,
    )
    d_query = partial.sum(dim=0).to(q.dtype)

    d_key = torch.empty_like(k)
    d_value = torch.empty_like(v)
    _read_dkv_kernel[(n_rows * channels * heads,)](
        q,
        k,
        v,
        grad_out,
        m,
        denominator,
        delta,
        row_pos,
        d_key,
        d_value,
        scale,
        K=slots,
        KPAD=pad,
        C=channels,
        H=heads,
        HD=head_dim,
        BLOCK_HD=block_hd,
        num_warps=_ROW_WARPS,
    )
    return d_query, d_key, d_value


def _launch_broadcast(q: Tensor, k: Tensor, v: Tensor, node_pos: Tensor) -> Tensor:
    n_rows, channels, heads, head_dim = q.shape
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    _bcast_forward_kernel[(n_rows * channels * heads,)](
        q,
        k,
        v,
        node_pos,
        out,
        1.0 / math.sqrt(head_dim),
        R=int(k.shape[1]),
        RPAD=triton.next_power_of_2(int(k.shape[1])),
        C=channels,
        H=heads,
        HD=head_dim,
        BLOCK_HD=triton.next_power_of_2(head_dim),
        num_warps=_ROW_WARPS,
    )
    return out


def _launch_broadcast_backward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    node_pos: Tensor,
    offsets: Tensor,
    grad_out: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    n_rows, channels, heads, head_dim = q.shape
    positions, context = int(k.shape[0]), int(k.shape[1])
    pad = triton.next_power_of_2(context)
    block_hd = triton.next_power_of_2(head_dim)
    scale = 1.0 / math.sqrt(head_dim)

    d_query = torch.empty_like(q)
    _bcast_dq_kernel[(n_rows * channels * heads,)](
        q,
        k,
        v,
        grad_out,
        node_pos,
        d_query,
        scale,
        R=context,
        RPAD=pad,
        C=channels,
        H=heads,
        HD=head_dim,
        BLOCK_HD=block_hd,
        num_warps=_ROW_WARPS,
    )

    programs = positions * channels * heads
    splits = _splits(programs, n_rows // max(positions, 1))
    # Two buffers rather than one stacked pair: a custom op may not return two
    # tensors that view the same storage, and a slice of a stack would.
    partial_key = torch.empty(
        (splits, *k.shape), dtype=torch.float32, device=q.device
    )
    partial_value = torch.empty_like(partial_key)
    _bcast_dkv_kernel[(programs, splits)](
        q,
        k,
        v,
        grad_out,
        offsets,
        partial_key,
        partial_value,
        scale,
        P=positions,
        R=context,
        RPAD=pad,
        C=channels,
        H=heads,
        HD=head_dim,
        BLOCK_HD=block_hd,
        SPLITS=splits,
        BLOCK_E=_BCAST_BLOCK_E,
        num_warps=_TILE_WARPS,
    )
    return (
        d_query,
        partial_key.sum(dim=0).to(k.dtype),
        partial_value.sum(dim=0).to(v.dtype),
    )


@torch.library.custom_op("mantisnet::act_latent_read", mutates_args=())
def _latent_read_op(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    ranges: Tensor,
    range_base: Tensor,
    counts: Tensor,
    row_pos: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    validate_read(q, k, v, row_pos)
    reference = lambda: read_reference(  # noqa: E731
        q, k, v, row_pos, int(q.shape[0])
    )
    if not _supported(q, int(k.shape[0])):
        return reference()
    key = _shape_key("read", q, int(ranges.shape[1]))
    if key in _FAILED_SHAPES:
        return reference()
    try:
        return _launch_read(q, k, v, ranges, range_base, counts)
    except Exception as exc:
        _FAILED_SHAPES[key] = f"{type(exc).__name__}: {exc}"
        warnings.warn(
            f"fused latent read failed for q={tuple(q.shape)}; gathering "
            f"instead for this shape: {_FAILED_SHAPES[key]}",
            RuntimeWarning,
            stacklevel=2,
        )
        return reference()


@_latent_read_op.register_fake
def _(q, k, v, ranges, range_base, counts, row_pos):
    accumulate = torch.promote_types(q.dtype, torch.float32)
    stats = q.new_empty(q.shape[:-1], dtype=accumulate)
    return q.new_empty(q.shape, dtype=accumulate), stats, stats.clone()


@torch.library.custom_op("mantisnet::act_latent_read_backward", mutates_args=())
def _latent_read_backward_op(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    ranges: Tensor,
    range_base: Tensor,
    counts: Tensor,
    row_pos: Tensor,
    out: Tensor,
    m: Tensor,
    denominator: Tensor,
    grad_out: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    grad_out = grad_out.contiguous().float()
    # delta is the row sum of the upstream gradient against the output —
    # algebraically the segment's Σ alpha * dalpha, so the softmax backward
    # needs no first sweep over the rows to find it.
    delta = (grad_out * out).sum(-1).contiguous()
    reference = lambda: read_reference_backward(  # noqa: E731
        q, k, v, row_pos, m, denominator, delta, grad_out
    )
    if not _supported(q, int(k.shape[0])):
        return reference()
    key = _shape_key("read", q, int(ranges.shape[1]))
    if key in _FAILED_BACKWARD_SHAPES:
        return reference()
    try:
        return _launch_read_backward(
            q,
            k,
            v,
            ranges,
            range_base,
            counts,
            row_pos,
            m,
            denominator,
            delta,
            grad_out,
        )
    except Exception as exc:
        _FAILED_BACKWARD_SHAPES[key] = f"{type(exc).__name__}: {exc}"
        warnings.warn(
            f"fused latent read backward failed for q={tuple(q.shape)}; "
            f"gathering instead for this shape: {_FAILED_BACKWARD_SHAPES[key]}",
            RuntimeWarning,
            stacklevel=2,
        )
        return reference()


@_latent_read_backward_op.register_fake
def _(q, k, v, ranges, range_base, counts, row_pos, out, m, denominator, grad_out):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _read_setup_context(ctx, inputs, output) -> None:
    ctx.save_for_backward(*inputs, *output)


def _read_backward(ctx, grad_out, _grad_m, _grad_l):
    # Saved tensors: seven inputs then three outputs, matching backward's args.
    d_query, d_key, d_value = _latent_read_backward_op(*ctx.saved_tensors, grad_out)
    return (d_query, d_key, d_value) + (None,) * 4


_latent_read_op.register_autograd(_read_backward, setup_context=_read_setup_context)


@torch.library.custom_op("mantisnet::act_latent_broadcast", mutates_args=())
def _latent_broadcast_op(
    q: Tensor, k: Tensor, v: Tensor, node_pos: Tensor, offsets: Tensor
) -> Tensor:
    _validate_broadcast(q, k, v, node_pos)
    reference = lambda: broadcast_reference(q, k, v, node_pos)  # noqa: E731
    if not _supported(q, int(q.shape[0])):
        return reference()
    key = _shape_key("broadcast", q, int(k.shape[1]))
    if key in _FAILED_SHAPES:
        return reference()
    try:
        return _launch_broadcast(q, k, v, node_pos)
    except Exception as exc:
        _FAILED_SHAPES[key] = f"{type(exc).__name__}: {exc}"
        warnings.warn(
            f"fused latent broadcast failed for q={tuple(q.shape)}; gathering "
            f"instead for this shape: {_FAILED_SHAPES[key]}",
            RuntimeWarning,
            stacklevel=2,
        )
        return reference()


@_latent_broadcast_op.register_fake
def _(q, k, v, node_pos, offsets):
    return q.new_empty(q.shape, dtype=torch.promote_types(q.dtype, torch.float32))


@torch.library.custom_op("mantisnet::act_latent_broadcast_backward", mutates_args=())
def _latent_broadcast_backward_op(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    node_pos: Tensor,
    offsets: Tensor,
    grad_out: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    grad_out = grad_out.contiguous().float()
    reference = lambda: broadcast_reference_backward(  # noqa: E731
        q, k, v, node_pos, grad_out
    )
    if not _supported(q, int(q.shape[0])):
        return reference()
    key = _shape_key("broadcast", q, int(k.shape[1]))
    if key in _FAILED_BACKWARD_SHAPES:
        return reference()
    try:
        return _launch_broadcast_backward(q, k, v, node_pos, offsets, grad_out)
    except Exception as exc:
        _FAILED_BACKWARD_SHAPES[key] = f"{type(exc).__name__}: {exc}"
        warnings.warn(
            f"fused latent broadcast backward failed for q={tuple(q.shape)}; "
            f"gathering instead for this shape: {_FAILED_BACKWARD_SHAPES[key]}",
            RuntimeWarning,
            stacklevel=2,
        )
        return reference()


@_latent_broadcast_backward_op.register_fake
def _(q, k, v, node_pos, offsets, grad_out):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _broadcast_setup_context(ctx, inputs, output) -> None:
    ctx.save_for_backward(*inputs)


def _broadcast_backward(ctx, grad_out):
    d_query, d_key, d_value = _latent_broadcast_backward_op(
        *ctx.saved_tensors, grad_out
    )
    return (d_query, d_key, d_value, None, None)


_latent_broadcast_op.register_autograd(
    _broadcast_backward, setup_context=_broadcast_setup_context
)


# --------------------------------------------------------------------------
# What `latents.py` calls


def latent_read(
    q: Tensor, k: Tensor, v: Tensor, segments: LatentSegments
) -> Tensor:
    """§17.2's read without a per-node score matrix: fp32 ``(P, K, C, h, hd)``.

    ``q`` is ``(P, K, C, heads, head_dim)`` — ``K`` latent slots per position,
    each holding ``C`` channels the keys share, where ``C`` is 1 for the
    invariant stream and 3 for the axis stream and channel ``a`` of the query
    only ever meets channel ``a`` of the key. That pairing is what makes the
    axis read equivariant rather than merely parameter-shared (§12.1).

    ``k`` and ``v`` are the ``(N, C, heads, head_dim)`` rows of ``segments``'
    concatenated families, in the order those families were given.
    """
    if int(q.shape[0]) != segments.positions:
        raise ValueError(
            f"q holds {q.shape[0]} positions but the segments describe "
            f"{segments.positions}"
        )
    if int(k.shape[0]) != segments.n_rows:
        raise ValueError(
            f"k holds {k.shape[0]} rows but the segments describe "
            f"{segments.n_rows}"
        )
    out, _m, _total = _latent_read_op(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        segments.ranges,
        segments.range_base,
        segments.counts,
        segments.row_pos,
    )
    return out


def latent_broadcast(
    q: Tensor, k: Tensor, v: Tensor, node_pos: Tensor, offsets: Tensor
) -> Tensor:
    """§17.4's broadcast without a per-node context tensor: fp32 ``(N, C, h, hd)``.

    ``q`` is one row per node, ``k`` and ``v`` are the ``(P, R, C, heads,
    head_dim)`` context of each position, and ``offsets`` is the node family's
    ``(P + 1,)`` CSR offsets — the same information as ``node_pos``, in the
    ordering the gradient sweep reduces over.
    """
    if int(offsets.shape[0]) - 1 != int(k.shape[0]):
        raise ValueError(
            f"offsets describe {int(offsets.shape[0]) - 1} positions against "
            f"the context's {int(k.shape[0])}"
        )
    return _latent_broadcast_op(
        q.contiguous(), k.contiguous(), v.contiguous(), node_pos, offsets
    )


__all__ = [
    "LatentSegments",
    "broadcast_reference",
    "broadcast_reference_backward",
    "latent_broadcast",
    "latent_read",
    "latent_segments",
    "read_reference",
    "read_reference_backward",
    "row_positions",
    "validate_read",
]
