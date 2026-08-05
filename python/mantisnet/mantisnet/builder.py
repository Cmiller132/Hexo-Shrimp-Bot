"""The MantisNet input builder: positions to graphs, graphs to batches.

This module implements the representation of ``docs/MODEL_SPEC.md`` §3–§4
and the batching contract of §9 under ``MODEL_REPR_VERSION``. It derives live
windows from the stone list without calling the engine's window walk, which
remains the independent oracle for §12.1.

Index conventions this module fixes (each is part of the representation):

- Window feature: ``colour * NUM_PATTERNS + pattern_rank``, colour ``0`` = own,
  ``1`` = opponent, rank = position of the canonical occupancy mask in the
  sorted list of the ``NUM_PATTERNS`` canonical 6-bit patterns of 1–5 bits.
- Decoder class: the rank of the ``(occupancy mask, candidate slot)`` reversal
  orbit in ascending ``(mask, slot)`` order, one of ``DEC_CLASSES``. Stone
  incidence uses the same joint-orbit construction over occupied rather than
  empty slots, one of ``OCC_CLASSES`` (§4.3).
- Attention distance bucket: hex distance ``d >= 1`` maps to ``d - 1`` clamped
  to ``D_MAX - 1``; ``SELF`` is ``D_MAX``; ``TOKEN`` is ``D_MAX + 1`` and wins
  over ``SELF`` on the token–token pair.
- Nearest-stone bucket: distance ``d`` in ``1..8`` maps to ``d - 1``. The one
  stoneless position (ply 0) has no nearest stone; the clamp sends it to
  bucket ``7``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from hexo_py import MODEL_REPR_VERSION

from .relay import relay_tables

WINDOW_LEN = 6
# Unit steps of the engine's axes, in canonical order Q, R, QR.
AXES = np.array([[1, 0], [0, 1], [1, -1]], dtype=np.int64)
LEGAL_RADIUS = 8
NEAREST_BUCKETS = LEGAL_RADIUS


def _reverse6(m: np.ndarray) -> np.ndarray:
    """Bit-reverse each 6-bit mask."""
    out = np.zeros_like(m)
    for k in range(WINDOW_LEN):
        out |= ((m >> k) & 1) << (WINDOW_LEN - 1 - k)
    return out


_MASKS = np.arange(64, dtype=np.int64)
# canon(m) = min(m, reverse6(m)): a reflection reverses slot order (§3.2).
_CANON = np.minimum(_MASKS, _reverse6(_MASKS))
_CANONICAL = np.unique(_CANON[1:63])  # 1–5 bits set; 0 and 63 are not windows
# Rank of each canonical mask; -1 leaves noncanonical, empty, and full masks
# outside the embedding index range.
_PATTERN_RANK = np.full(64, -1, dtype=np.int64)
_PATTERN_RANK[_CANONICAL] = np.arange(len(_CANONICAL))

# 34: the 62 nonempty, nonfull 6-bit masks fold to (62 + 6 palindromes) / 2
# orbits under reversal. (MODEL_SPEC §3.2.)
NUM_PATTERNS = len(_CANONICAL)

# Stones in each canonical pattern, indexed by rank — reversal preserves the
# count, so this is well-defined per orbit.
PATTERN_STONES = np.array([bin(int(m)).count("1") for m in _CANONICAL])

def _orbit_classes(occupied: bool) -> np.ndarray:
    """A (64, 6) table of joint reversal-orbit classes, by mask and slot.

    A reflection reverses a window's slot order, so it sends the pair
    ``(mask, slot)`` to ``(reverse6(mask), 5 - slot)`` — *jointly*. The orbits of
    that involution are therefore the finest reversal-invariant description of
    where a slot sits among a window's stones (§4.3), and the class is the
    orbit's rank in ascending ``(mask, slot)`` order. ``occupied`` selects which
    slots are classed: the empty ones (the decoder and cell-pass table) or the
    occupied ones (the stone-incidence table).

    Every other entry is ``-1``: the empty and full masks, and any slot whose
    occupancy bit disagrees with ``occupied``. The builders pair decoder entries
    with empty slots and incidence entries with occupied ones by construction,
    so a ``-1`` reaching an index tensor is a builder fault and is refused
    rather than embedded.
    """
    table = np.full((64, WINDOW_LEN), -1, dtype=np.int64)
    nxt = 0
    for mask in range(1, 63):
        rev = int(_reverse6(np.array(mask))[()])
        for slot in range(WINDOW_LEN):
            if bool((mask >> slot) & 1) != occupied:
                continue
            # Ascending order visits each orbit's representative first, so its
            # rank is assigned there and its partner reads it back.
            if (mask, slot) <= (rev, WINDOW_LEN - 1 - slot):
                table[mask, slot] = nxt
                nxt += 1
            else:
                table[mask, slot] = table[rev, WINDOW_LEN - 1 - slot]
    return table


_DEC_CLASS = _orbit_classes(occupied=False)
_OCC_CLASS = _orbit_classes(occupied=True)

# 93 each: both slot selections give 186 (mask, slot) pairs, folding to
# 186 / 2 orbits — the involution has no fixed point, since no slot equals
# its own mirror.
DEC_CLASSES = int(_DEC_CLASS.max()) + 1
OCC_CLASSES = int(_OCC_CLASS.max()) + 1

# The stock model's incidence fold: reversal preserves min(s, 5 - s), so each
# occupied-slot orbit lands on one coarse end / near-end / centre class, and a
# model that embeds the folded classes reproduces the pre-joint three-class
# incidence bit for bit (§4.3).
_occ_mask, _occ_slot = np.nonzero(_OCC_CLASS >= 0)
OCC_FOLD = np.full(OCC_CLASSES, -1, dtype=np.int64)
OCC_FOLD[_OCC_CLASS[_occ_mask, _occ_slot]] = np.minimum(
    _occ_slot, WINDOW_LEN - 1 - _occ_slot
)

# Coordinate packing: q, r fit i16, so 21 bits of headroom per component is
# collision-free. Window identity packs the axis into the low two bits.
_QSHIFT = 1 << 21


def _pack(qr: np.ndarray) -> np.ndarray:
    """(n, 2) coordinates to collision-free int64 keys."""
    return qr[:, 0] * _QSHIFT + qr[:, 1]


@dataclass(frozen=True)
class PositionGraph:
    """One position's entities and index tables, in numpy (§9)."""

    # Stones, in the order given (engine canonical when built from a position).
    stone_own: np.ndarray  # (n_s,) int64: 0 = side to move, 1 = opponent
    stone_qr: np.ndarray  # (n_s, 2) int64, for the distance buckets only
    # Live windows.
    window_feat: np.ndarray  # (n_w,) int64: colour * NUM_PATTERNS + rank
    window_id: np.ndarray  # (n_w, 3) int64: (axis, start_q, start_r), consumed
    # only through reversal-invariant pair classes (§5.1c) and by tests.
    # Stone <-> window incidence with joint occupied-slot classes.
    inc_stone: np.ndarray  # (e,) int64
    inc_window: np.ndarray  # (e,) int64
    inc_class: np.ndarray  # (e,) int64, < OCC_CLASSES
    # Policy decoder table over legal cells, in engine legal order.
    n_legal: int
    dec_cell: np.ndarray  # (e_d,) int64: legal-cell index
    dec_window: np.ndarray  # (e_d,) int64: live window through it
    dec_class: np.ndarray  # (e_d,) int64: the (mask, slot) class there, < DEC_CLASSES
    bg_cell: np.ndarray  # (n_bg,) int64: cells in no live window
    bg_bucket: np.ndarray  # (n_bg,) int64 in 0..7: nearest-stone bucket
    moves_remaining: int  # 1 or 2

    @property
    def n_stones(self) -> int:
        return len(self.stone_own)

    @property
    def n_windows(self) -> int:
        return len(self.window_feat)


def build(
    stone_qr: np.ndarray,
    stone_owner: np.ndarray,
    mover: int,
    legal_qr: np.ndarray,
    moves_remaining: int,
) -> PositionGraph:
    """Build one position's graph from the §11 input list.

    ``stone_qr`` is (n_s, 2) int, ``stone_owner`` (n_s,) int in {0, 1},
    ``mover`` the side to move, ``legal_qr`` (n_legal, 2) int in engine legal
    order. Raises ``ValueError`` for a terminal position (no legal moves):
    terminal positions are a builder error, not a silent default.
    """
    stone_qr = np.asarray(stone_qr, dtype=np.int64).reshape(-1, 2)
    stone_owner = np.asarray(stone_owner, dtype=np.int64).reshape(-1)
    legal_qr = np.asarray(legal_qr, dtype=np.int64).reshape(-1, 2)
    if len(legal_qr) == 0:
        raise ValueError("terminal position: the builder refuses it")
    if moves_remaining not in (1, 2):
        raise ValueError(f"moves_remaining must be 1 or 2, got {moves_remaining}")

    n_s = len(stone_qr)
    stone_own = (stone_owner != mover).astype(np.int64)

    if n_s == 0:
        # Ply 0: no stones, no windows, every legal cell on the background
        # path with the clamp bucket.
        return PositionGraph(
            stone_own=stone_own,
            stone_qr=stone_qr,
            window_feat=np.empty(0, dtype=np.int64),
            window_id=np.empty((0, 3), dtype=np.int64),
            inc_stone=np.empty(0, dtype=np.int64),
            inc_window=np.empty(0, dtype=np.int64),
            inc_class=np.empty(0, dtype=np.int64),
            n_legal=len(legal_qr),
            dec_cell=np.empty(0, dtype=np.int64),
            dec_window=np.empty(0, dtype=np.int64),
            dec_class=np.empty(0, dtype=np.int64),
            bg_cell=np.arange(len(legal_qr), dtype=np.int64),
            bg_bucket=np.full(len(legal_qr), NEAREST_BUCKETS - 1, dtype=np.int64),
            moves_remaining=moves_remaining,
        )

    stone_key = _pack(stone_qr)
    order = np.argsort(stone_key)
    sorted_key = stone_key[order]
    if np.any(sorted_key[1:] == sorted_key[:-1]):
        raise ValueError("duplicate stone coordinates")

    # Candidate windows: every (axis, start) through some stone — 18 per stone,
    # start = stone - k * axis for k in 0..5 (§3.2's builder walk).
    ks = np.arange(WINDOW_LEN, dtype=np.int64)
    # (n_s, 3, 6, 2): stone i, axis a, offset k.
    starts = stone_qr[:, None, None, :] - AXES[None, :, None, :] * ks[None, None, :, None]
    axis_idx = np.broadcast_to(np.arange(3, dtype=np.int64)[None, :, None], starts.shape[:3])
    wkey = _pack(starts.reshape(-1, 2)) * 4 + axis_idx.reshape(-1)
    uniq_key = np.unique(wkey)

    # Occupancy of each candidate: 6 cells, each looked up in the stone set.
    u_axis = uniq_key & 3
    u_start_packed = uniq_key >> 2  # arithmetic shift keeps the sign
    # Invert _pack: floor divmod puts a negative r into the high half of the
    # remainder range, since |r| stays far below _QSHIFT / 2.
    q, rem = np.divmod(u_start_packed, _QSHIFT)
    r = rem.copy()
    high = rem >= _QSHIFT // 2
    r[high] -= _QSHIFT
    q[high] += 1
    u_start = np.stack([q, r], axis=1)

    cells = u_start[:, None, :] + AXES[u_axis][:, None, :] * ks[None, :, None]  # (n_c, 6, 2)
    cell_key = _pack(cells.reshape(-1, 2))
    pos = np.searchsorted(sorted_key, cell_key)
    pos_clip = np.minimum(pos, n_s - 1)
    hit = sorted_key[pos_clip] == cell_key
    occupant = np.where(hit, order[pos_clip], -1).reshape(-1, WINDOW_LEN)  # stone index or -1
    occ_own = (occupant >= 0) & (stone_own[np.maximum(occupant, 0)] == 0)
    occ_opp = (occupant >= 0) & (stone_own[np.maximum(occupant, 0)] == 1)
    own_mask = (occ_own.astype(np.int64) << ks[None, :]).sum(axis=1)
    opp_mask = (occ_opp.astype(np.int64) << ks[None, :]).sum(axis=1)

    # Live: stones of exactly one colour (§3.2). Every candidate has >= 1
    # stone by construction. A full six is a completed win, which a
    # non-terminal position cannot contain — and terminals were refused above.
    live = (own_mask > 0) != (opp_mask > 0)
    live_key = uniq_key[live]
    colour = (opp_mask[live] > 0).astype(np.int64)
    pattern = own_mask[live] | opp_mask[live]
    rank = _PATTERN_RANK[_CANON[pattern]]
    window_feat = colour * NUM_PATTERNS + rank
    window_id = np.column_stack([u_axis[live], u_start[live, 0], u_start[live, 1]])

    # Incidence: one entry per occupied slot of each live window. The class is
    # joint in the window's occupancy and the stone's own slot (§4.3), off the
    # raw mask in slot order — `pattern`, like the decoder classes below.
    l_occupant = occupant[live]  # (n_w, 6)
    w_idx, slot = np.nonzero(l_occupant >= 0)
    inc_stone = l_occupant[w_idx, slot]
    inc_window = w_idx.astype(np.int64)
    inc_class = _OCC_CLASS[pattern[w_idx], slot]
    if inc_class.size and inc_class.min() < 0:
        bad = int(np.argmin(inc_class))
        raise ValueError(
            f"incidence entry {bad} pairs window mask "
            f"{int(pattern[w_idx[bad]]):06b} with slot {int(slot[bad])}, "
            f"which that window does not occupy"
        )

    # Decoder table: each legal cell's live windows, by the same 18-candidate
    # walk matched against the live set.
    n_legal = len(legal_qr)
    c_starts = legal_qr[:, None, None, :] - AXES[None, :, None, :] * ks[None, None, :, None]
    c_axis = np.broadcast_to(np.arange(3, dtype=np.int64)[None, :, None], c_starts.shape[:3])
    c_key = _pack(c_starts.reshape(-1, 2)) * 4 + c_axis.reshape(-1)
    wpos = np.searchsorted(live_key, c_key)
    wpos_clip = np.minimum(wpos, max(len(live_key) - 1, 0))
    c_hit = (live_key[wpos_clip] == c_key) if len(live_key) else np.zeros(len(c_key), bool)
    flat = np.nonzero(c_hit)[0]
    dec_cell = flat // (3 * WINDOW_LEN)
    dec_window = wpos_clip[flat]
    # The class is joint in the window's occupancy and the candidate's own slot
    # (§4.3), so it needs the window's raw mask in slot order — `pattern`, not
    # the canonicalized rank the window embedding carries.
    dec_class = _DEC_CLASS[pattern[dec_window], flat % WINDOW_LEN]
    if dec_class.size and dec_class.min() < 0:
        bad = int(np.argmin(dec_class))
        raise ValueError(
            f"decoder entry {bad} pairs window mask "
            f"{int(pattern[dec_window[bad]]):06b} with slot "
            f"{int(flat[bad] % WINDOW_LEN)}, which that window already occupies"
        )

    covered = np.zeros(n_legal, dtype=bool)
    covered[dec_cell] = True
    bg_cell = np.nonzero(~covered)[0].astype(np.int64)
    if len(bg_cell):
        # Nearest-stone hex distance, vectorised over (background cells, stones).
        dq = legal_qr[bg_cell, 0][:, None] - stone_qr[None, :, 0]
        dr = legal_qr[bg_cell, 1][:, None] - stone_qr[None, :, 1]
        d = np.maximum(np.abs(dq), np.maximum(np.abs(dr), np.abs(dq + dr)))
        nearest = d.min(axis=1)
        bg_bucket = np.minimum(nearest, LEGAL_RADIUS) - 1
    else:
        bg_bucket = np.empty(0, dtype=np.int64)

    return PositionGraph(
        stone_own=stone_own,
        stone_qr=stone_qr,
        window_feat=window_feat,
        window_id=window_id,
        inc_stone=inc_stone,
        inc_window=inc_window,
        inc_class=inc_class,
        n_legal=n_legal,
        dec_cell=dec_cell,
        dec_window=dec_window,
        dec_class=dec_class,
        bg_cell=bg_cell,
        bg_bucket=bg_bucket.astype(np.int64),
        moves_remaining=moves_remaining,
    )


def from_position(pos) -> PositionGraph:
    """Build from a ``hexo_py.Position``. Terminal positions raise."""
    if pos.is_terminal:
        raise ValueError("terminal position: the builder refuses it")
    stones = pos.stones()
    if stones:
        arr = np.asarray(stones, dtype=np.int64)
        stone_qr, stone_owner = arr[:, :2], arr[:, 2]
    else:
        stone_qr = np.empty((0, 2), dtype=np.int64)
        stone_owner = np.empty(0, dtype=np.int64)
    legal = np.asarray(pos.legal_moves(), dtype=np.int64).reshape(-1, 2)
    return build(stone_qr, stone_owner, pos.current_player, legal, pos.moves_remaining)


@dataclass
class Batch:
    """A collated batch: concatenated entities plus padded attention tables.

    Every index tensor is precomputed here; the forward performs no
    data-dependent index discovery (§9). Attention and the value readout use
    per-position padded layouts with the token at slot 0.
    """

    n_pos: int
    # Concatenated entity features.
    stone_own: torch.Tensor  # (N_s,) long
    window_feat: torch.Tensor  # (N_w,) long
    # Window identities (axis, start_q, start_r), each in its position's own
    # frame. The model consumes them only through reversal-invariant pair
    # classes (§5.1c), never as raw coordinates.
    window_id: torch.Tensor  # (N_w, 3) long
    moves_idx: torch.Tensor  # (P,) long: moves_remaining - 1
    # Incidence, with window/stone indices globally offset.
    inc_stone: torch.Tensor  # (E,) long
    inc_window: torch.Tensor  # (E,) long
    inc_class: torch.Tensor  # (E,) long
    # Stone-attention padding: rows [token; stones] per position, width max_t.
    max_t: int
    stone_slot: torch.Tensor  # (N_s,) long, flat index into (P * max_t)
    coords: torch.Tensor  # (P, max_t, 2) int32; row 0 and padding are zeros
    attn_valid: torch.Tensor  # (P, max_t) bool
    # Value-readout padding: rows [token; windows] per position, width max_w.
    max_w: int
    window_slot: torch.Tensor  # (N_w,) long, flat index into (P * max_w)
    value_valid: torch.Tensor  # (P, max_w) bool
    # Policy decoder, cells concatenated in engine order per position.
    n_cells: int
    legal_offsets: torch.Tensor  # (P + 1,) long
    cell_pos: torch.Tensor  # (N_c,) long: position of each cell
    dec_cell: torch.Tensor  # (E_d,) long, global cell index
    dec_window: torch.Tensor  # (E_d,) long, global window index
    dec_class: torch.Tensor  # (E_d,) long, < DEC_CLASSES
    bg_cell: torch.Tensor  # (N_bg,) long, global cell index
    bg_bucket: torch.Tensor  # (N_bg,) long
    # Cell-pass relay (§5.1b): the decoder incidence sorted once at collation
    # into CSR views — by covered cell (relabelled compactly), by window, and
    # by class — so the pass runs as contiguous segment reductions.
    relay_cell_ptr: torch.Tensor  # (covered cells + 1,) long
    relay_window: torch.Tensor  # (E_d,) long: edge windows, cell order
    relay_class: torch.Tensor  # (E_d,) long: edge classes, cell order
    relay_win_ptr: torch.Tensor  # (N_w + 1,) long
    relay_wcell: torch.Tensor  # (E_d,) long: compact edge cells, window order
    relay_cls_ptr: torch.Tensor  # (DEC_CLASSES + 1,) long
    relay_ccell: torch.Tensor  # (E_d,) long: compact edge cells, class order
    # The §5.1c window-pair views are not collated: a window_attention model
    # derives them on its own device from window_id — the edge views cost
    # several times more to ship than to derive beside the model.

    def to(self, device) -> "Batch":
        """The same batch with every tensor on ``device``."""
        moved = {
            name: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for name, v in vars(self).items()
        }
        return Batch(**moved)

    def pin_memory(self) -> "Batch":
        """The same batch in pinned host memory, so ``to`` is a true async DMA.

        ``non_blocking`` silently degrades to a synchronous staged copy from
        pageable memory; a prefetch worker pins ahead of the transfer instead.
        """
        pinned = {
            name: (v.pin_memory() if isinstance(v, torch.Tensor) else v)
            for name, v in vars(self).items()
        }
        return Batch(**pinned)


_RELAY_FIELDS = (
    "relay_cell_ptr",
    "relay_window",
    "relay_class",
    "relay_win_ptr",
    "relay_wcell",
    "relay_cls_ptr",
    "relay_ccell",
)


def _relay_fields(
    dec_cell: torch.Tensor,
    dec_window: torch.Tensor,
    dec_class: torch.Tensor,
    n_windows: int,
) -> dict:
    tables = relay_tables(dec_cell, dec_window, dec_class, n_windows, DEC_CLASSES)
    return dict(zip(_RELAY_FIELDS, tables))


def batch_from_arrays(**fields) -> Batch:
    """A ``Batch`` from per-tensor arrays, with the derived tables built here.

    Both external construction paths land on this one function — the
    ``hexo_py.build_batch*`` array dicts unpacked as kwargs, and the embedded
    Rust forward calling with torch tensors — so the collation-time relay
    derivation lives in exactly one place. The scalar fields are derived from
    tensor shapes; callers may also pass them, and a disagreement is refused.
    """
    scalars = {
        name: int(fields.pop(name))
        for name in ("n_pos", "max_t", "max_w", "n_cells")
        if name in fields
    }
    t = {name: torch.as_tensor(value) for name, value in fields.items()}
    derived = {
        "n_pos": int(t["attn_valid"].shape[0]),
        "max_t": int(t["attn_valid"].shape[1]),
        "max_w": int(t["value_valid"].shape[1]),
        "n_cells": int(t["cell_pos"].shape[0]),
    }
    for name, value in scalars.items():
        if value != derived[name]:
            raise ValueError(f"{name}={value} disagrees with the derived {derived[name]}")
    return Batch(
        **derived,
        **t,
        **_relay_fields(
            t["dec_cell"], t["dec_window"], t["dec_class"], int(t["window_feat"].shape[0])
        ),
    )


def collate_positions(positions) -> Batch:
    """Build and collate positions with the Rust builder.

    ``hexo_py.build_batch`` runs in parallel with the GIL released and returns
    the same fields as ``collate([from_position(p) ...])`` under
    ``MODEL_REPR_VERSION``.
    """
    import hexo_py

    return batch_from_arrays(**hexo_py.build_batch(list(positions)))


def collate_prefixes(games, ts) -> Batch:
    """Move prefixes to one collated batch: replay + build, in parallel.

    Stored fitting positions are move prefixes
    (``docs/KLENT_FOR_HEXO.md`` §4.3).
    """
    import hexo_py

    return batch_from_arrays(**hexo_py.build_batch_prefixes(list(games), list(ts)))


def collate(graphs: list[PositionGraph]) -> Batch:
    """Concatenate position graphs into one batch (§9)."""
    if not graphs:
        raise ValueError("empty batch")
    p = len(graphs)
    ns = np.array([g.n_stones for g in graphs])
    nw = np.array([g.n_windows for g in graphs])
    nl = np.array([g.n_legal for g in graphs])
    stone_off = np.concatenate([[0], np.cumsum(ns)])
    win_off = np.concatenate([[0], np.cumsum(nw)])
    cell_off = np.concatenate([[0], np.cumsum(nl)])

    max_t = int(ns.max()) + 1
    max_w = int(nw.max()) + 1

    coords = np.zeros((p, max_t, 2), dtype=np.int32)
    attn_valid = np.zeros((p, max_t), dtype=bool)
    attn_valid[:, 0] = True
    value_valid = np.zeros((p, max_w), dtype=bool)
    value_valid[:, 0] = True
    for i, g in enumerate(graphs):
        coords[i, 1 : 1 + g.n_stones] = g.stone_qr
        attn_valid[i, 1 : 1 + g.n_stones] = True
        value_valid[i, 1 : 1 + g.n_windows] = True

    def cat(parts, dtype=np.int64):
        return torch.from_numpy(np.concatenate(parts).astype(dtype)) if parts else torch.empty(0, dtype=torch.long)

    stone_slot = cat([i * max_t + 1 + np.arange(g.n_stones) for i, g in enumerate(graphs)])
    window_slot = cat([i * max_w + 1 + np.arange(g.n_windows) for i, g in enumerate(graphs)])
    window_id = cat([g.window_id for g in graphs]).view(-1, 3)
    dec_cell = cat([g.dec_cell + cell_off[i] for i, g in enumerate(graphs)])
    dec_window = cat([g.dec_window + win_off[i] for i, g in enumerate(graphs)])
    dec_class = cat([g.dec_class for g in graphs])

    return Batch(
        n_pos=p,
        stone_own=cat([g.stone_own for g in graphs]),
        window_feat=cat([g.window_feat for g in graphs]),
        window_id=window_id,
        moves_idx=torch.tensor([g.moves_remaining - 1 for g in graphs], dtype=torch.long),
        inc_stone=cat([g.inc_stone + stone_off[i] for i, g in enumerate(graphs)]),
        inc_window=cat([g.inc_window + win_off[i] for i, g in enumerate(graphs)]),
        inc_class=cat([g.inc_class for g in graphs]),
        max_t=max_t,
        stone_slot=stone_slot,
        coords=torch.from_numpy(coords),
        attn_valid=torch.from_numpy(attn_valid),
        max_w=max_w,
        window_slot=window_slot,
        value_valid=torch.from_numpy(value_valid),
        n_cells=int(cell_off[-1]),
        legal_offsets=torch.from_numpy(cell_off.astype(np.int64)),
        cell_pos=cat([np.full(g.n_legal, i) for i, g in enumerate(graphs)]),
        dec_cell=dec_cell,
        dec_window=dec_window,
        dec_class=dec_class,
        bg_cell=cat([g.bg_cell + cell_off[i] for i, g in enumerate(graphs)]),
        bg_bucket=cat([g.bg_bucket for g in graphs]),
        **_relay_fields(dec_cell, dec_window, dec_class, int(win_off[-1])),
    )
