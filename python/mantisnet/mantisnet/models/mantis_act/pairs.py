"""Same-turn second-placement partner evidence (§20).

A first placement is followed by a second one the same mover chooses, so what
an action is worth depends on what it makes available. This module enumerates
those partners and the six-cell windows through which a partner would matter,
as the flat evidence rows of §20.3. The output is still one placement at a
time: these rows only enrich the first placement's action state and its
``Q(s, a1)``.

Nothing here calls the engine. A cell is legal iff it is empty and within
``LEGAL_RADIUS`` hex steps of some occupied cell, and that rule applied to the
board plus the hypothetical first stone is the whole of ``post_action_*``
legality. The engine stays the independent oracle of §30.16, which it stops
being the moment a builder asks it what it is about to be checked against.

Index conventions this module fixes (each is part of the representation):

- Destination: a legal-action index in engine order. Source: the legal-action
  index of the partner cell, or ``-1`` for a partner that only becomes legal
  once the first stone lands (§20.2). ``pair_src_is_current_legal`` states the
  same fact as a flag, because the model reads the two through different
  paths — a current partner contributes its own action embedding, a
  prospective one a shared base.
- Signed offset: a collinear partner is ``a + s * AXES[axis]`` for
  ``s`` in ``-D..-1, 1..D`` with ``D = cfg.pair_max_distance``. The row keeps
  ``|s|`` and the axis; the sign is not representable, since a reflection
  flips it while carrying the position to an equivalent one.
- Evidence kind: ``PAIR_EVIDENCE_SHARED_WINDOW`` rows carry one six-cell
  window containing *both* placements and route to the destination's matching
  axis channel. ``PAIR_EVIDENCE_TACTICAL`` rows (§20.4) carry a window
  containing the partner alone, so they name no shared line and route to the
  invariant stream with ``pair_axis_or_neg1 = -1``.
- Post-two pattern: the ternary class (§9.2) of the evidence window after both
  own stones are written into it, one of the 377 nonempty classes. Reversal
  invariance is what makes it independent of which way the window is read.

Gating, all of it exact (§20):

- rows exist only in the ``FIRST`` phase, where ``moves_remaining == 2``;
- an action that already wins immediately gets no rows at all, since its turn
  ends on the first stone and there is no partner to model;
- ``pair_scope="none"`` emits nothing.

The immediate-win test is the post-placement window pattern of §19.2 read for
a six-run: the mover's stone at ``a`` plus its own stones reaching five steps
each way along one axis. It is computed here rather than taken from the action
tables, because the gate has to hold whether or not those tables were built.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from .config import ENUM_VOCABULARIES, MantisACTConfig
from .packed import PHASE_FIRST, PHASE_OPENING, PHASE_SECOND, SENTINEL, WINDOW_LEN
from .pattern_classes import CLASS_IS_EMPTY, PATTERN_CLASS
from .symmetry import AXES, hex_distance

# A cell is legal iff it is empty and within this many hex steps of some
# occupied cell. The engine holds the same constant; it is restated rather
# than imported so the builder path never consults the oracle it is tested
# against.
LEGAL_RADIUS = 8

# Slot values of a window, and equally the cell occupancy codes of §8.2: the
# two encodings are one ternary digit relative to the side to move.
_EMPTY, _OWN, _OPP = 0, 1, 2
_POWERS = 3 ** np.arange(WINDOW_LEN, dtype=np.int64)

# A window spans five steps, so five is both the largest usable pair distance
# and the furthest a window containing the pair can start from either cell.
_SPAN = WINDOW_LEN - 1

# §20.3 gives the row set no name of its own; these are the two ways a row
# reaches its destination, and they are disjoint by construction.
PAIR_EVIDENCE_SHARED_WINDOW = 0
PAIR_EVIDENCE_TACTICAL = 1
PAIR_EVIDENCE_KINDS = 2

# §20.4's thresholds: after both placements the partner's window is an own
# four-or-better, or it was an opponent four-or-better the partner now kills.
_TACTICAL_OWN_STONES = 4
_TACTICAL_OPP_STONES = 4

# The scopes this module dispatches on. A scope the config accepts and this
# tuple omits would reach the dispatch below and raise at build time, in the
# middle of a run; the import check turns it into an import error instead.
_IMPLEMENTED_SCOPES = (
    "none",
    "current_legal_collinear",
    "post_action_collinear",
    "post_action_tactical",
)
_UNIMPLEMENTED = ENUM_VOCABULARIES["pair_scope"] - set(_IMPLEMENTED_SCOPES)
if _UNIMPLEMENTED:
    raise RuntimeError(
        f"pair scopes the config allows and pairs.py omits: {sorted(_UNIMPLEMENTED)}"
    )

# Coordinate packing: q and r fit i16, so 21 bits per component is
# collision-free, and a packed key orders as (q, r) does.
_QSHIFT = 1 << 21


def _pack(qr: np.ndarray) -> np.ndarray:
    """An ``(..., 2)`` coordinate array as collision-free int64 keys."""
    return qr[..., 0] * _QSHIFT + qr[..., 1]


def _lookup(
    sorted_keys: np.ndarray, values: np.ndarray, keys: np.ndarray, missing: int
) -> np.ndarray:
    """Each key's value in a sorted key table, ``missing`` where absent."""
    if len(sorted_keys) == 0:
        return np.full(keys.shape, missing, dtype=np.int64)
    position = np.minimum(np.searchsorted(sorted_keys, keys), len(sorted_keys) - 1)
    return np.where(sorted_keys[position] == keys, values[position], missing)


@dataclass(frozen=True)
class PairRows:
    """The seven parallel §20.3 arrays, sorted into §7 order.

    The field names are ``ACTGraph``'s, so a builder splices them in with
    ``ACTGraph(**vars(rows), ...)``. The §7 sort key's partner coordinate and
    window identity are deliberately not among them: they are builder metadata
    that would otherwise ride into the model, and every row's content is
    already stated by the seven fields.
    """

    pair_dst_action: np.ndarray  # (e_pair,) legal-action index, engine order
    pair_src_action_or_neg1: np.ndarray  # (e_pair,) -1 for a prospective partner
    pair_axis_or_neg1: np.ndarray  # (e_pair,) -1 on a tactical row
    pair_distance: np.ndarray  # (e_pair,) 1..pair_max_distance
    pair_post2_pattern: np.ndarray  # (e_pair,) nonempty pattern class
    pair_evidence_kind: np.ndarray  # (e_pair,)
    pair_src_is_current_legal: np.ndarray  # (e_pair,) 0/1

    @property
    def n_rows(self) -> int:
        return len(self.pair_dst_action)


class _Evidence(NamedTuple):
    """One enumeration's rows, with the §7 sort keys still attached."""

    dst: np.ndarray  # (n,)
    src: np.ndarray  # (n,) legal-action index of the partner, or -1
    axis: np.ndarray  # (n,)
    distance: np.ndarray  # (n,)
    post2: np.ndarray  # (n,)
    kind: np.ndarray  # (n,)
    partner_qr: np.ndarray  # (n, 2) sort key only
    window_id: np.ndarray  # (n, 3) (native_axis, start_q, start_r), sort key only


def _no_evidence() -> _Evidence:
    """An empty enumeration, in every array's dtype."""
    return _Evidence(
        dst=np.empty(0, dtype=np.int64),
        src=np.empty(0, dtype=np.int64),
        axis=np.empty(0, dtype=np.int64),
        distance=np.empty(0, dtype=np.int64),
        post2=np.empty(0, dtype=np.int64),
        kind=np.empty(0, dtype=np.int64),
        partner_qr=np.empty((0, 2), dtype=np.int64),
        window_id=np.empty((0, 3), dtype=np.int64),
    )


def _check_coordinates(name: str, array) -> np.ndarray:
    """An ``(n, 2)`` int64 coordinate array, or a named failure."""
    out = np.asarray(array, dtype=np.int64)
    if out.ndim != 2 or out.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n, 2), got {out.shape}")
    return out


def _check_phase(moves_remaining: int, phase_id: int) -> None:
    """Refuse a phase and move count that describe no reachable turn (§13.1)."""
    if moves_remaining not in (1, 2):
        raise ValueError(f"moves_remaining must be 1 or 2, got {moves_remaining!r}")
    if phase_id not in (PHASE_OPENING, PHASE_FIRST, PHASE_SECOND):
        raise ValueError(
            f"phase_id must be {PHASE_OPENING}, {PHASE_FIRST}, or {PHASE_SECOND}, "
            f"got {phase_id!r}"
        )
    if (phase_id == PHASE_FIRST) != (moves_remaining == 2):
        raise ValueError(
            f"phase_id {phase_id} disagrees with moves_remaining {moves_remaining} "
            "(§13.1: FIRST means two placements remain)"
        )


def _stone_table(stone_qr: np.ndarray, stone_own: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sorted stone keys and their occupancy codes, relative to the mover."""
    if stone_own.shape != stone_qr.shape[:1]:
        raise ValueError(
            f"stone_own must have one entry per stone: {stone_own.shape} against "
            f"{stone_qr.shape[0]} stones"
        )
    outside = ~np.isin(stone_own, (0, 1))
    if outside.any():
        bad = int(np.flatnonzero(outside)[0])
        raise ValueError(
            f"stone_own must be 0 (mover) or 1 (opponent): stone {bad} is "
            f"{int(stone_own[bad])}"
        )
    packed = _pack(stone_qr)
    order = np.argsort(packed)
    keys = packed[order]
    repeated = keys[1:] == keys[:-1]
    if repeated.any():
        bad = int(np.flatnonzero(repeated)[0]) + 1
        q, r = stone_qr[order][bad]
        raise ValueError(f"duplicate stone at ({int(q)}, {int(r)})")
    return keys, np.where(stone_own[order] == 0, _OWN, _OPP)


def _line_occupancy(
    legal_qr: np.ndarray, stone_keys: np.ndarray, stone_values: np.ndarray
) -> np.ndarray:
    """Occupancy of each action's three lines, five steps each way.

    ``(n_legal, 3, 2 * _SPAN + 1)`` indexed by ``t + _SPAN`` for the cell
    ``a + t * AXES[axis]``. Every window containing the action, and every
    window containing it together with a partner at most ``_SPAN`` steps away,
    lies inside that stretch — the window would otherwise have to start more
    than five steps before the earlier of the two cells.
    """
    steps = np.arange(-_SPAN, _SPAN + 1, dtype=np.int64)
    cells = (
        legal_qr[:, None, None, :] + AXES[None, :, None, :] * steps[None, None, :, None]
    )
    return _lookup(stone_keys, stone_values, _pack(cells), _EMPTY)


def _immediate_wins(line_occupancy: np.ndarray) -> np.ndarray:
    """Which actions win on the spot, from their post-placement lines (§20).

    An action wins iff one of its lines has six own stones in a row through it,
    so the run is its own stone plus the unbroken own stones reaching each way.
    Counting from the centre is exactly the six-cell window test: a run of six
    through the placement is a window, and a window through the placement is a
    run of six only if it is unbroken.
    """
    own = line_occupancy == _OWN
    before = np.logical_and.accumulate(own[:, :, _SPAN - 1 :: -1], axis=2).sum(axis=2)
    after = np.logical_and.accumulate(own[:, :, _SPAN + 1 :], axis=2).sum(axis=2)
    return ((before + after + 1) >= WINDOW_LEN).any(axis=1)


def _collinear_evidence(
    legal_qr: np.ndarray,
    line_occupancy: np.ndarray,
    legal_keys: np.ndarray,
    legal_index: np.ndarray,
    winning: np.ndarray,
    scope: str,
    max_distance: int,
) -> _Evidence:
    """One row per window containing both the action and a collinear partner.

    §20.2's enumeration: every empty cell on the three axes at signed distance
    one through ``max_distance``, kept if it is currently legal, or — in the
    ``post_action_*`` scopes — legal once the first stone lands. Every window
    containing the pair is then an evidence row (§20.3); there are
    ``6 - distance`` of them, since the window must start at most five steps
    before the earlier cell and no later than the earlier cell itself.
    """
    offsets = np.concatenate(
        [
            np.arange(-max_distance, 0, dtype=np.int64),
            np.arange(1, max_distance + 1, dtype=np.int64),
        ]
    )
    partner_qr = (
        legal_qr[:, None, None, :] + AXES[None, :, None, :] * offsets[None, None, :, None]
    )
    partner_occupancy = line_occupancy[:, :, offsets + _SPAN]
    source = _lookup(legal_keys, legal_index, _pack(partner_qr), SENTINEL)

    empty = partner_occupancy == _EMPTY
    current = source >= 0
    if scope == "current_legal_collinear":
        keep = empty & current
    else:
        # Legal after the first stone: empty, and inside the legal radius of
        # some occupied cell — either one already on the board, which is what
        # being currently legal says, or the stone just placed at the action.
        # The config caps the pair distance below LEGAL_RADIUS, so the second
        # disjunct holds for every candidate and the cells the action opens are
        # the whole of what this scope adds over the first (§20.2, §38).
        opened = np.abs(offsets) <= LEGAL_RADIUS
        keep = empty & (current | opened[None, None, :])
    keep &= ~winning[:, None, None]

    action, axis, offset_index = np.nonzero(keep)
    offset = offsets[offset_index]
    partner = partner_qr[action, axis, offset_index]

    # Window starts, as steps from the action along its axis: from five steps
    # before the later cell to the earlier cell. Entries past the last start
    # repeat it and are masked out, which keeps every slot index inside the
    # line stretch.
    candidate = np.arange(_SPAN, dtype=np.int64)
    first_start = np.maximum(offset, 0) - _SPAN
    last_start = np.minimum(offset, 0)
    exists = first_start[:, None] + candidate[None, :] <= last_start[:, None]
    start = np.minimum(first_start[:, None] + candidate[None, :], last_start[:, None])

    slot_step = start[:, :, None] + np.arange(WINDOW_LEN, dtype=np.int64)
    value = line_occupancy[action[:, None, None], axis[:, None, None], slot_step + _SPAN]
    placed = (slot_step == 0) | (slot_step == offset[:, None, None])
    code = (np.where(placed, _OWN, value) * _POWERS).sum(axis=2)

    row, which = np.nonzero(exists)
    window_start = (
        legal_qr[action[row]] + AXES[axis[row]] * start[row, which][:, None]
    )
    return _Evidence(
        dst=action[row],
        src=source[action, axis, offset_index][row],
        axis=axis[row],
        distance=np.abs(offset[row]),
        post2=PATTERN_CLASS[code[row, which]],
        kind=np.full(len(row), PAIR_EVIDENCE_SHARED_WINDOW, dtype=np.int64),
        partner_qr=partner[row],
        window_id=np.column_stack([axis[row], window_start]),
    )


def _tactical_evidence(
    legal_qr: np.ndarray,
    stone_qr: np.ndarray,
    stone_keys: np.ndarray,
    stone_values: np.ndarray,
    legal_keys: np.ndarray,
    legal_index: np.ndarray,
    winning: np.ndarray,
    max_distance: int,
) -> _Evidence:
    """§20.4's partners: a second cell that decides a window of its own.

    A tactical partner ``b`` is a cell legal after the first stone whose own
    window — a window the first stone is *not* part of — is settled by it:
    after ``b`` lands the window holds four or more own stones and no
    opponent's, or it was an opponent four-or-better that ``b`` now kills. One
    row per such window, carrying the same post-two pattern class as any other
    row.

    Windows containing the action are excluded because a window containing
    both cells makes them collinear within five steps, which is precisely a
    shared-window row: including them here would emit each of those twice. So
    a tactical row's window shares no line with the action, its axis is ``-1``
    (§20.4 routes it to the invariant stream), and its distance is the pair's
    hex distance clamped to ``max_distance`` — beyond the window span the exact
    separation says nothing, since the two placements share no window to say it
    about.
    """
    steps = np.arange(WINDOW_LEN, dtype=np.int64)
    # Every window through some stone, deduplicated: a window with four of one
    # colour, or three own, is one of these by definition.
    starts = (
        stone_qr[:, None, None, :] - AXES[None, :, None, :] * steps[None, None, :, None]
    ).reshape(-1, 2)
    axes = np.broadcast_to(
        np.arange(len(AXES), dtype=np.int64)[None, :, None],
        (len(stone_qr), len(AXES), WINDOW_LEN),
    ).reshape(-1)
    first = np.unique(_pack(starts) * len(AXES) + axes, return_index=True)[1]
    window_axis, window_start = axes[first], starts[first]

    cells = window_start[:, None, :] + AXES[window_axis][:, None, :] * steps[None, :, None]
    cell_keys = _pack(cells)
    value = _lookup(stone_keys, stone_values, cell_keys, _EMPTY)
    own_stones = (value == _OWN).sum(axis=1)
    opp_stones = (value == _OPP).sum(axis=1)
    # Own side: the partner is the fourth own stone or better. Opponent side:
    # the window is already an opponent four the partner kills by entering it.
    decisive = ((opp_stones == 0) & (own_stones + 1 >= _TACTICAL_OWN_STONES)) | (
        (own_stones == 0) & (opp_stones >= _TACTICAL_OPP_STONES)
    )

    window, slot = np.nonzero((value == _EMPTY) & decisive[:, None])
    partner_qr = cells[window, slot]
    partner_key = cell_keys[window, slot]
    partner_source = _lookup(legal_keys, legal_index, partner_key, SENTINEL)
    # The partner is the only cell of the window either placement touches, so
    # the post-two code is the window's own code with that slot made own.
    partner_post2 = PATTERN_CLASS[
        (value * _POWERS).sum(axis=1)[window] + _POWERS[slot] * _OWN
    ]

    action = np.flatnonzero(~winning)
    distance = hex_distance(
        partner_qr[None, :, 0] - legal_qr[action][:, None, 0],
        partner_qr[None, :, 1] - legal_qr[action][:, None, 1],
    )
    reachable = (partner_source >= 0)[None, :] | (distance <= LEGAL_RADIUS)
    holds_action = (
        cell_keys[window][None, :, :] == _pack(legal_qr[action])[:, None, None]
    ).any(axis=2)

    row, column = np.nonzero(reachable & ~holds_action)
    return _Evidence(
        dst=action[row],
        src=partner_source[column],
        axis=np.full(len(row), SENTINEL, dtype=np.int64),
        distance=np.minimum(distance[row, column], max_distance),
        post2=partner_post2[column],
        kind=np.full(len(row), PAIR_EVIDENCE_TACTICAL, dtype=np.int64),
        partner_qr=partner_qr[column],
        window_id=np.column_stack(
            [window_axis[window][column], window_start[window][column]]
        ),
    )


def _assemble(parts: list[_Evidence]) -> PairRows:
    """Sort the enumerations into §7 order and drop the sort keys.

    The key is ``(dst_action, partner_coord, evidence_kind, window_identity)``,
    and ``lexsort`` reads its last argument as the primary one.
    """
    joined = _Evidence(
        *(np.concatenate([getattr(p, f) for p in parts]) for f in _Evidence._fields)
    )
    order = np.lexsort(
        (
            joined.window_id[:, 2],
            joined.window_id[:, 1],
            joined.window_id[:, 0],
            joined.kind,
            joined.partner_qr[:, 1],
            joined.partner_qr[:, 0],
            joined.dst,
        )
    )
    post2 = joined.post2[order]
    if CLASS_IS_EMPTY[post2].any():
        bad = int(np.flatnonzero(CLASS_IS_EMPTY[post2])[0])
        raise ValueError(
            f"pair row {bad} carries the all-empty pattern class {int(post2[bad])}, "
            "but both placements are own stones in its window (§20.3)"
        )
    source = joined.src[order]
    return PairRows(
        pair_dst_action=joined.dst[order],
        pair_src_action_or_neg1=source,
        pair_axis_or_neg1=joined.axis[order],
        pair_distance=joined.distance[order],
        pair_post2_pattern=post2,
        pair_evidence_kind=joined.kind[order],
        pair_src_is_current_legal=(source >= 0).astype(np.int64),
    )


def pair_rows(
    window_set: object,
    cell_set: object,
    stone_qr: np.ndarray,
    stone_own: np.ndarray,
    legal_qr: np.ndarray,
    moves_remaining: int,
    phase_id: int,
    cfg: MantisACTConfig,
) -> PairRows:
    """Every §20.3 pair evidence row of one position, in §7 order.

    ``stone_own`` is ``0`` for the mover's stones and ``1`` for the opponent's,
    and ``legal_qr`` is the engine's legal-move list in engine order — the row
    indices are positions in it.

    ``window_set`` and ``cell_set`` are the persistent window and cell node
    sets the rest of the builder has already produced. Neither is read, and
    neither can be: a pair's windows include ones that are not persistent
    nodes, since two empty cells five steps apart share a window with no stone
    in it at all, and legality after the first placement is the radius rule
    rather than a property of the current cell set. They stay in the signature
    so every builder stage takes the same node context.
    """
    max_distance = int(cfg.pair_max_distance)
    if not 1 <= max_distance <= _SPAN:
        raise ValueError(
            f"pair_max_distance must be 1..{_SPAN}, got {max_distance}: cells "
            "further apart share no six-cell window"
        )
    scope = cfg.pair_scope
    if scope not in _IMPLEMENTED_SCOPES:
        raise ValueError(
            f"unknown pair_scope {scope!r}; expected one of {sorted(_IMPLEMENTED_SCOPES)}"
        )

    stone_qr = _check_coordinates("stone_qr", stone_qr)
    legal_qr = _check_coordinates("legal_qr", legal_qr)
    stone_keys, stone_values = _stone_table(stone_qr, np.asarray(stone_own, dtype=np.int64))
    _check_phase(moves_remaining, phase_id)

    if scope == "none" or phase_id != PHASE_FIRST:
        return _assemble([_no_evidence()])
    if len(stone_qr) == 0:
        raise ValueError(
            "phase FIRST with no stones: the first placement of a turn on an "
            "empty board is the OPENING phase"
        )

    legal_order = np.argsort(_pack(legal_qr))
    legal_keys = _pack(legal_qr)[legal_order]
    repeated = legal_keys[1:] == legal_keys[:-1]
    if repeated.any():
        bad = int(np.flatnonzero(repeated)[0]) + 1
        q, r = legal_qr[legal_order][bad]
        raise ValueError(f"duplicate legal move at ({int(q)}, {int(r)})")
    occupied = _lookup(stone_keys, stone_values, legal_keys, _EMPTY) != _EMPTY
    if occupied.any():
        bad = int(np.flatnonzero(occupied)[0])
        q, r = legal_qr[legal_order][bad]
        raise ValueError(f"legal move ({int(q)}, {int(r)}) is an occupied cell")

    line_occupancy = _line_occupancy(legal_qr, stone_keys, stone_values)
    winning = _immediate_wins(line_occupancy)

    parts = [
        _collinear_evidence(
            legal_qr,
            line_occupancy,
            legal_keys,
            legal_order,
            winning,
            scope,
            max_distance,
        )
    ]
    if scope == "post_action_tactical":
        parts.append(
            _tactical_evidence(
                legal_qr,
                stone_qr,
                stone_keys,
                stone_values,
                legal_keys,
                legal_order,
                winning,
                max_distance,
            )
        )
    return _assemble(parts)
