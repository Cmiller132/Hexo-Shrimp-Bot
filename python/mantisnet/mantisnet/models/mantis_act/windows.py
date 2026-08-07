"""Persistent six-cell window enumeration and ternary encoding (§4, §9).

A window is six consecutive cells of one hex line, identified by
``(native_axis, start_q, start_r)`` with cells ``start + k * AXES[a]`` for
slot ``k`` in ``0..5``. Finite scope: windows through a stone, and optionally
through legal cells (``action_relevant``).

Ternary code: ``sum_k v_k * 3**k`` where 0=empty, 1=own, 2=opponent relative
to side to move. Slot order follows the stored axis direction; downstream
reads must canonicalise through ``PATTERN_CLASS``. Numeric features are in
``WINDOW_NUMERIC_NAMES`` order, each in ``[0, 1]`` (§9.3).

Windows are returned in §7 order; ``np.unique`` deduplicates and sorts in one
pass. §9.1 refuses completed-line windows (six own or six opponent stones).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MantisACTConfig
from .pattern_classes import (
    EMPTY,
    EMPTY_COUNT,
    OPP_COUNT,
    OPP_LIVE,
    OPP_MAX_RUN,
    OWN_COUNT,
    OWN_LIVE,
    OWN_MAX_RUN,
    PATTERN_CLASS,
    STATUS,
)
from .symmetry import AXES

WINDOW_LEN = 6
NUM_AXES = len(AXES)

# The ternary slot value of a stone is one more than its owner flag: flag 0,
# the mover's own stone, is slot value 1, and flag 1, the opponent's, is 2.
_STONE_SLOT_BASE = 1

_SLOTS = np.arange(WINDOW_LEN, dtype=np.int64)
_POWERS = 3 ** _SLOTS

# §9.3's counts and runs, in the order the feature vector stores them. The
# tables are `pattern_classes`' own, so the features and the pattern class are
# two readings of one code rather than two counts of one window.
WINDOW_NUMERIC_NAMES = (
    "own_count",
    "opp_count",
    "empty_count",
    "own_max_run",
    "opp_max_run",
)
_NUMERIC_TABLES = (OWN_COUNT, OPP_COUNT, EMPTY_COUNT, OWN_MAX_RUN, OPP_MAX_RUN)
WINDOW_NUMERIC_FEATURES = len(WINDOW_NUMERIC_NAMES)
if len(_NUMERIC_TABLES) != WINDOW_NUMERIC_FEATURES:
    raise RuntimeError("the window numeric names and their tables disagree in length")

# The coordinate range this representation addresses, and the packing that
# fixes it. Both components are offset into `0..2 * WINDOW_COORD_LIMIT` and laid
# out most-significant first — axis, then q, then r — so the packed key orders
# identities exactly as §7 does and unpacks by two exact divmods with no sign
# case. A coordinate reaching the limit is refused rather than wrapped.
#
# This module's own key addresses 2**30 either way; the binding constraint is
# §16's device-side window-pair key, which packs a line's offset from the
# origin by 2**16 into a 2**17 field. A *sum* of two coordinates must stay
# inside 2**16, which puts each component inside 2**15 — the same int16 range
# the engine's own move coordinates live in.
WINDOW_COORD_LIMIT = 1 << 15
_COORD_SPAN = 2 * WINDOW_COORD_LIMIT
_AXIS_STRIDE = _COORD_SPAN * _COORD_SPAN


def _pack_cells(qr: np.ndarray) -> np.ndarray:
    """Pack an ``(n, 2)`` coordinate array into collision-free int64 keys."""
    if qr.size and int(np.abs(qr).max()) >= WINDOW_COORD_LIMIT:
        flat = int(np.abs(qr).argmax())
        row = flat // 2
        raise ValueError(
            f"coordinate ({int(qr[row, 0])}, {int(qr[row, 1])}) lies outside the "
            f"+-{WINDOW_COORD_LIMIT} this representation addresses"
        )
    return (qr[:, 0] + WINDOW_COORD_LIMIT) * _COORD_SPAN + (
        qr[:, 1] + WINDOW_COORD_LIMIT
    )


def _pack_identity(window_id: np.ndarray) -> np.ndarray:
    """Pack ``(n, 3)`` window identities into keys ordered as §7 orders them."""
    axis = window_id[:, 0]
    if window_id.size and (int(axis.min()) < 0 or int(axis.max()) >= NUM_AXES):
        bad = int(np.argmax((axis < 0) | (axis >= NUM_AXES)))
        raise ValueError(
            f"window {bad} names native axis {int(axis[bad])}, not one of 0..{NUM_AXES - 1}"
        )
    return axis * _AXIS_STRIDE + _pack_cells(window_id[:, 1:])


def _unpack_identity(key: np.ndarray) -> np.ndarray:
    """Invert :func:`_pack_identity` into an ``(n, 3)`` identity array."""
    upper, r = np.divmod(key, _COORD_SPAN)
    axis, q = np.divmod(upper, _COORD_SPAN)
    return np.stack(
        [axis, q - WINDOW_COORD_LIMIT, r - WINDOW_COORD_LIMIT], axis=1
    )


def window_cells(window_id: np.ndarray) -> np.ndarray:
    """The six cells of each window in slot order, as an ``(n, 6, 2)`` array.

    This is the whole content of a window identity: the identity is metadata
    the model never sees, and every consumer that needs a window's geometry —
    the cell scope, the incidence table, the action rows — decodes it here so
    the slot-order convention has one statement.
    """
    window_id = np.asarray(window_id, dtype=np.int64).reshape(-1, 3)
    axis = window_id[:, 0]
    if window_id.size and (int(axis.min()) < 0 or int(axis.max()) >= NUM_AXES):
        bad = int(np.argmax((axis < 0) | (axis >= NUM_AXES)))
        raise ValueError(
            f"window {bad} names native axis {int(axis[bad])}, not one of 0..{NUM_AXES - 1}"
        )
    return window_id[:, None, 1:] + AXES[axis][:, None, :] * _SLOTS[None, :, None]


def _candidate_keys(seed_qr: np.ndarray) -> np.ndarray:
    """Packed identities of all 18 windows through each seed cell (§9.1).

    A cell sits in slot ``k`` of the window starting ``k`` steps back along the
    axis, so the three undirected axes and six slots give 18 candidates per
    seed — every window the cell can belong to, and no other.
    """
    starts = seed_qr[:, None, None, :] - AXES[None, :, None, :] * _SLOTS[None, None, :, None]
    axis = np.broadcast_to(
        np.arange(NUM_AXES, dtype=np.int64)[None, :, None], starts.shape[:3]
    )
    return _pack_identity(
        np.concatenate([axis.reshape(-1, 1), starts.reshape(-1, 2)], axis=1)
    )


def _slot_digits(
    cells: np.ndarray, sorted_key: np.ndarray, sorted_own: np.ndarray
) -> np.ndarray:
    """The ternary value of every slot of every window, as an ``(n, 6)`` array.

    One sorted-set membership test per slot: the stone keys are searched for
    each cell key, and a hit contributes the stone's owner flag raised to its
    slot value. A miss is empty.
    """
    if len(sorted_key) == 0:
        return np.zeros(cells.shape[:2], dtype=np.int64)
    key = _pack_cells(cells.reshape(-1, 2))
    at = np.minimum(np.searchsorted(sorted_key, key), len(sorted_key) - 1)
    hit = sorted_key[at] == key
    return np.where(hit, sorted_own[at] + _STONE_SLOT_BASE, 0).reshape(cells.shape[:2])


def _refuse_terminal(window_id: np.ndarray, code: np.ndarray) -> None:
    """Refuse a position that already contains a completed line (§9.1).

    Six stones of one colour in a row is a win, so the state is terminal and
    the builder must not encode it. A full window of both colours is not a win
    and is left alone — it is the mixed dead window the whole architecture
    exists to represent.
    """
    full_own = OWN_COUNT[code] == WINDOW_LEN
    full_opp = OPP_COUNT[code] == WINDOW_LEN
    complete = full_own | full_opp
    if not complete.any():
        return
    row = int(np.flatnonzero(complete)[0])
    colour = "own" if full_own[row] else "opponent"
    axis, start_q, start_r = (int(value) for value in window_id[row])
    raise ValueError(
        f"window (axis={axis}, start=({start_q}, {start_r})) holds six {colour} "
        "stones: the position is terminal and the builder refuses it"
    )


@dataclass(frozen=True)
class WindowSet:
    """One position's persistent windows, in the §7 order (§9.1–§9.3).

    Every array has one row per window and the rows correspond. ``window_id``
    is builder metadata for deduplication, ordering, the cell and action
    tables, and diagnostics; it never reaches the model (§7).
    """

    window_id: np.ndarray  # (n_w, 3) (native_axis, start_q, start_r), metadata only
    code: np.ndarray  # (n_w,) raw ternary code 0..728, in stored slot order
    pattern_class: np.ndarray  # (n_w,) reversal class 0..377
    status: np.ndarray  # (n_w,) EMPTY / OWN_LIVE / OPP_LIVE / MIXED
    axis: np.ndarray  # (n_w,) native axis 0..2
    numeric: np.ndarray  # (n_w, WINDOW_NUMERIC_FEATURES) float32, in [0, 1]

    @property
    def n_windows(self) -> int:
        return len(self.code)

    def cell_coords(self) -> np.ndarray:
        """The ``(n_w, 6, 2)`` cells of every window, in slot order."""
        return window_cells(self.window_id)

    def index_of(self, window_id: np.ndarray) -> np.ndarray:
        """Row of each queried identity, ``-1`` where this set has no such window.

        The queried array is ``(..., 3)`` and the result its leading shape, so
        the action table looks up all 18 candidate identities of every legal
        cell in one call. ``-1`` is the sentinel §19.2 requires for a candidate
        with no persistent pre-action window.
        """
        window_id = np.asarray(window_id, dtype=np.int64)
        if window_id.ndim == 0 or window_id.shape[-1] != 3:
            raise ValueError(
                f"window identities must have shape (..., 3), got {window_id.shape}"
            )
        query = _pack_identity(window_id.reshape(-1, 3))
        if self.n_windows == 0:
            return np.full(window_id.shape[:-1], -1, dtype=np.int64)
        table = _pack_identity(self.window_id)
        at = np.minimum(np.searchsorted(table, query), self.n_windows - 1)
        return np.where(table[at] == query, at, -1).reshape(window_id.shape[:-1])


def enumerate_windows(
    stone_qr: np.ndarray,
    stone_own: np.ndarray,
    legal_qr: np.ndarray,
    cfg: MantisACTConfig,
) -> WindowSet:
    """Enumerate one position's persistent windows under ``cfg.window_scope``.

    ``stone_qr`` is ``(n_s, 2)`` and ``stone_own`` ``(n_s,)`` with ``0`` the
    mover's stones and ``1`` the opponent's; ``legal_qr`` is ``(n_legal, 2)``
    and is read only by ``action_relevant``, though its consistency with the
    stones is checked under every scope. The three scopes of §4 are:

    - ``live``: one-colour windows only;
    - ``nonempty``: every window with at least one stone, mixed included
      (the default);
    - ``action_relevant``: those plus every empty window through a legal cell,
      via the 18-candidate walk over the legal cells as well.

    Raises for a terminal position, for a coordinate outside the packing's
    range, for malformed stone or legal input, and for a scope this module does
    not implement.
    """
    stone_qr = np.asarray(stone_qr, dtype=np.int64).reshape(-1, 2)
    stone_own = np.asarray(stone_own, dtype=np.int64).reshape(-1)
    legal_qr = np.asarray(legal_qr, dtype=np.int64).reshape(-1, 2)
    if len(stone_own) != len(stone_qr):
        raise ValueError(
            f"{len(stone_qr)} stone coordinates against {len(stone_own)} owners"
        )
    wrong = np.flatnonzero((stone_own != 0) & (stone_own != 1))
    if wrong.size:
        bad = int(wrong[0])
        raise ValueError(
            f"stone_own[{bad}] = {int(stone_own[bad])}: owners are relative to the "
            "side to move, 0 its own stone and 1 the opponent's"
        )

    stone_key = _pack_cells(stone_qr)
    order = np.argsort(stone_key, kind="stable")
    sorted_key, sorted_own = stone_key[order], stone_own[order]
    duplicate = np.flatnonzero(sorted_key[1:] == sorted_key[:-1])
    if duplicate.size:
        row = int(order[duplicate[0]])
        raise ValueError(
            f"stone {row} repeats coordinate "
            f"({int(stone_qr[row, 0])}, {int(stone_qr[row, 1])})"
        )
    occupied_legal = np.flatnonzero(np.isin(_pack_cells(legal_qr), stone_key))
    if occupied_legal.size:
        row = int(occupied_legal[0])
        raise ValueError(
            f"legal cell {row} at ({int(legal_qr[row, 0])}, {int(legal_qr[row, 1])}) "
            "is occupied"
        )

    scope = cfg.window_scope
    if scope in ("live", "nonempty"):
        candidate = _candidate_keys(stone_qr)
    elif scope == "action_relevant":
        candidate = np.concatenate(
            [_candidate_keys(stone_qr), _candidate_keys(legal_qr)]
        )
    else:
        raise ValueError(
            f"window_scope={scope!r} is not one of 'live', 'nonempty', "
            "'action_relevant'"
        )

    # One pass deduplicates the walk and puts the survivors in the §7 order.
    window_id = _unpack_identity(np.unique(candidate))
    code = (
        _slot_digits(window_cells(window_id), sorted_key, sorted_own) * _POWERS[None, :]
    ).sum(axis=1)

    # Before any scope filter: a completed line makes the position terminal
    # whatever scope would have kept or dropped the window carrying it.
    _refuse_terminal(window_id, code)

    status = STATUS[code]
    if scope == "live":
        keep = (status == OWN_LIVE) | (status == OPP_LIVE)
    elif scope == "nonempty":
        # Every candidate of the stone walk contains the stone that generated
        # it, so this drops nothing; it states the scope rather than trusting
        # the walk to have already applied it.
        keep = status != EMPTY
    else:
        # The legal walk's candidates are kept precisely because they may be
        # empty: they are the windows an action can still fill.
        keep = np.ones(len(code), dtype=bool)
    window_id, code, status = window_id[keep], code[keep], status[keep]

    numeric = np.stack([table[code] for table in _NUMERIC_TABLES], axis=1)
    return WindowSet(
        window_id=np.ascontiguousarray(window_id),
        code=code,
        pattern_class=PATTERN_CLASS[code],
        status=status,
        axis=np.ascontiguousarray(window_id[:, 0]),
        numeric=(numeric / WINDOW_LEN).astype(np.float32),
    )
