"""Counterfactual legal-action tables and the tactical vector (§19).

Each legal action is encoded from eighteen windows (3 axes x 6 slots) a stone
placed there would join. Action row ``[action, axis, k]``: the window starts
at ``a - k * AXES[axis]``, post-placement code is ``pre_code + 3**k``.

``action_window_index`` is ``-1`` where the row's window is not persistent;
the model substitutes a learned pre-empty-window state there.
``action_post1_class`` is never ``-1``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MantisACTConfig
from .packed import NUM_AXES, POST_ACTION_ROWS, WINDOW_LEN
from .pattern_classes import (
    OPP_COUNT,
    OPP_LIVE,
    OWN_COUNT,
    OWN_LIVE,
    POST1_CLASS,
    STATUS,
)
from .symmetry import AXES
from .windows import WindowSet

# Ternary slot values relative to the side to move (§9.2).
_OWN, _OPP = 1, 2

_SLOTS = np.arange(WINDOW_LEN, dtype=np.int64)
_POWERS = 3**_SLOTS

# The ternary code of a window of six own stones (six in a row wins).
_ALL_OWN_CODE = int((_OWN * _POWERS).sum())

# The cells one action's rows can reach along one axis: five steps either way,
# indexed from ``_LINE_ORIGIN`` at the action cell itself.
_LINE_ORIGIN = WINDOW_LEN - 1
_LINE_OFFSETS = np.arange(-_LINE_ORIGIN, _LINE_ORIGIN + 1, dtype=np.int64)

# Slot ``j`` of the window whose candidate slot is ``k`` sits ``j - k`` steps
# from the action cell, so it is line entry ``j - k + _LINE_ORIGIN``.
_SLOT_TO_LINE = _SLOTS[None, :] - _SLOTS[:, None] + _LINE_ORIGIN

# Coordinate packing: q and r fit i16, so 21 bits per component is
# collision-free. Only the stone lookup uses this; window identities are
# located by ``WindowSet.index_of``.
_R_SPAN = 1 << 21

# The §19.3 vector, in the order the spec lists its fields. Counts over an
# action's eighteen rows are divided by eighteen and per-window counts by six,
# so every entry lands in [0, 1] without a learned scale.
TACTICAL_FEATURE_NAMES: tuple[str, ...] = (
    "immediate_win",
    "max_own_count_after",
    "max_opponent_count_before",
    "own_five_windows_after",
    "own_four_windows_after",
    "opponent_five_windows_hit",
    "opponent_four_windows_hit",
    "opponent_five_windows_remaining",
    "opponent_four_windows_remaining",
    "blocks_all_immediate_threats",
    "mixed_windows_created",
    "nonempty_pre_windows",
)
TACTICAL_FEATURES = len(TACTICAL_FEATURE_NAMES)

# Where the two board-level threat counts saturate. A side facing this many
# live opponent fours at once has lost whatever the exact number is, and the
# five count is decided far below it, so neither needs an unbounded input.
_GLOBAL_THREAT_CAP = 8


@dataclass(frozen=True)
class ActionTables:
    """The §19.2 rows of every legal action, dense and in engine order.

    The three named tables are ``ACTGraph`` fields verbatim. ``pre_code`` and
    ``post_code`` are builder metadata: the tactical vector is a function of
    them and the tests read them, but they never reach the model, which sees
    only reversal classes and statuses (§7).
    """

    action_window_index: np.ndarray  # (n_legal, 3, 6), -1 with no persistent window
    action_post1_class: np.ndarray  # (n_legal, 3, 6) in 0..728
    action_pre_status: np.ndarray  # (n_legal, 3, 6) EMPTY/OWN_LIVE/OPP_LIVE/MIXED
    pre_code: np.ndarray  # (n_legal, 3, 6) raw ternary code before the placement
    post_code: np.ndarray  # (n_legal, 3, 6) raw ternary code after it

    @property
    def n_legal(self) -> int:
        return len(self.action_window_index)


def _coordinate_key(qr: np.ndarray) -> np.ndarray:
    """Pack ``(..., 2)`` coordinates into collision-free int64 keys."""
    return qr[..., 0] * _R_SPAN + qr[..., 1]


def _coords(qr) -> np.ndarray:
    """An ``(n, 2)`` int64 view of a coordinate list."""
    return np.asarray(qr, dtype=np.int64).reshape(-1, 2)


def _board_occupancy(
    stone_qr: np.ndarray, stone_own: np.ndarray, cells: np.ndarray
) -> np.ndarray:
    """Ternary occupancy of ``(..., 2)`` cells: 0 empty, 1 own, 2 opponent.

    ``stone_own`` follows ``windows.py``'s convention: ``0`` the mover's stone,
    ``1`` the opponent's. Any other value raises.
    """
    stone_own = np.asarray(stone_own, dtype=np.int64).reshape(-1)
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
    if len(stone_qr) == 0:
        # The opening position, whose one legal cell lies in eighteen empty
        # windows: a real board state, not a missing input.
        return np.zeros(cells.shape[:-1], dtype=np.int64)

    stone_key = _coordinate_key(stone_qr)
    order = np.argsort(stone_key, kind="stable")
    sorted_key = stone_key[order]
    repeat = np.flatnonzero(sorted_key[1:] == sorted_key[:-1])
    if repeat.size:
        row = int(order[repeat[0]])
        raise ValueError(
            f"stone {row} repeats coordinate "
            f"({int(stone_qr[row, 0])}, {int(stone_qr[row, 1])})"
        )

    key = _coordinate_key(cells)
    at = np.minimum(np.searchsorted(sorted_key, key), len(sorted_key) - 1)
    colour = np.where(stone_own[order] == 0, _OWN, _OPP)
    return np.where(sorted_key[at] == key, colour[at], 0)


def _persistent_by_scope(pre_code: np.ndarray, window_scope: str) -> np.ndarray:
    """Which action rows' windows the scope makes persistent nodes (§4).

    Every window in these tables passes through a legal cell, which is what
    makes ``action_relevant`` total here and leaves the other two scopes as
    predicates on the window's own state.
    """
    if window_scope == "action_relevant":
        return np.ones(pre_code.shape, dtype=bool)
    if window_scope == "nonempty":
        return pre_code != 0
    if window_scope == "live":
        status = STATUS[pre_code]
        return (status == OWN_LIVE) | (status == OPP_LIVE)
    raise ValueError(f"unknown window_scope {window_scope!r}")


def _check_scope_agreement(
    found: np.ndarray, pre_code: np.ndarray, start: np.ndarray, window_scope: str
) -> None:
    """Refuse a window set that disagrees with the scope about these windows."""
    expected = _persistent_by_scope(pre_code, window_scope)
    if np.array_equal(found, expected):
        return
    action, axis, slot = (found != expected).nonzero()
    action, axis, slot = int(action[0]), int(axis[0]), int(slot[0])
    identity = (axis, *(int(c) for c in start[action, axis, slot]))
    verb = "lacks" if expected[action, axis, slot] else "carries"
    raise ValueError(
        f"the {window_scope!r} window set {verb} window {identity}, which is on "
        f"candidate slot {slot} of legal action {action} with pre-placement code "
        f"{int(pre_code[action, axis, slot])}"
    )


def action_tables(
    window_set: WindowSet,
    stone_qr: np.ndarray,
    stone_own: np.ndarray,
    legal_qr: np.ndarray,
    cfg: MantisACTConfig,
) -> ActionTables:
    """Build the eighteen post-placement rows of every legal action (§19.2).

    ``stone_qr`` is ``(n_s, 2)`` and ``stone_own`` ``(n_s,)`` with ``0`` the
    mover's stones and ``1`` the opponent's, as ``enumerate_windows`` takes
    them; ``legal_qr`` is ``(n_legal, 2)`` in engine legal order, which is
    preserved and never sorted (§7).

    The rows are read off eleven cells per action and axis — five steps either
    way — so the eighteen pre-placement codes are six sliding windows of one
    line rather than eighteen independent board walks. Raises ``ValueError``
    for a terminal position, an occupied or repeated legal cell, a repeated
    stone, an owner that is neither colour, or a window set that disagrees with
    ``cfg.window_scope``.
    """
    stone_qr = _coords(stone_qr)
    legal_qr = _coords(legal_qr)
    if len(legal_qr) == 0:
        raise ValueError("terminal position: the action encoder refuses it")

    legal_key = _coordinate_key(legal_qr)
    if len(np.unique(legal_key)) != len(legal_key):
        order = np.argsort(legal_key, kind="stable")
        repeat = order[np.flatnonzero(legal_key[order][1:] == legal_key[order][:-1])[0]]
        raise ValueError(
            f"legal action {int(repeat)} repeats the coordinate "
            f"{tuple(int(c) for c in legal_qr[repeat])}"
        )

    # (n_legal, 3, 11, 2): the line through each action along each axis.
    line_cells = (
        legal_qr[:, None, None, :]
        + AXES[None, :, None, :] * _LINE_OFFSETS[None, None, :, None]
    )
    line = _board_occupancy(stone_qr, stone_own, line_cells)
    occupied = np.flatnonzero(line[:, 0, _LINE_ORIGIN] != 0)
    if occupied.size:
        action = int(occupied[0])
        raise ValueError(
            f"legal action {action} at {tuple(int(c) for c in legal_qr[action])} is "
            f"occupied by colour {int(line[action, 0, _LINE_ORIGIN])}"
        )

    # The gather is (n_legal, 3, 6, 6) — candidate slot by window slot — and
    # summing the powers over the last axis leaves one code per row.
    pre_code = (line[:, :, _SLOT_TO_LINE] * _POWERS).sum(axis=3)
    # Slot k of the pre-code is the action cell, which the check above proved
    # empty, so writing an own stone there is one addition.
    post_code = pre_code + _OWN * _POWERS

    action_post1_class = POST1_CLASS[post_code, _SLOTS]
    unclassed = np.flatnonzero(action_post1_class.reshape(-1) < 0)
    if unclassed.size:
        action, axis, slot = np.unravel_index(unclassed[0], action_post1_class.shape)
        raise ValueError(
            f"post-placement code {int(post_code[action, axis, slot])} does not hold an "
            f"own stone at candidate slot {int(slot)}: the slot-to-power mapping is wrong"
        )

    # The row's window starts a candidate slot back from the action cell, so
    # its identity is the (axis, start) triple ``windows.py`` deduplicates on.
    start = (
        legal_qr[:, None, None, :]
        - AXES[None, :, None, :] * _SLOTS[None, None, :, None]
    )
    axis_of_row = np.broadcast_to(
        np.arange(NUM_AXES, dtype=np.int64)[None, :, None, None], start.shape[:3] + (1,)
    )
    identity = np.concatenate([axis_of_row, start], axis=3)
    action_window_index = window_set.index_of(identity)
    _check_scope_agreement(action_window_index >= 0, pre_code, start, cfg.window_scope)

    return ActionTables(
        action_window_index=action_window_index,
        action_post1_class=action_post1_class,
        action_pre_status=STATUS[pre_code],
        pre_code=pre_code,
        post_code=post_code,
    )


def tactical_features(tables: ActionTables, cfg: MantisACTConfig) -> np.ndarray:
    """The §19.3 deterministic tactical vector, ``[n_legal, T]`` float32.

    Every field is a function of the current state and the hypothetical
    placement, with no search: the eighteen pre- and post-placement codes carry
    the local terms, and the two board-level threat counts are broadcast to
    every action. Disabled, the vector has width zero (§32).

    Two readings not fixed by the spec's field list:

    - A five- or four-window count only counts a window holding one colour: a
      window already holding both colours can never be completed.
    - "nonempty post-windows" counts rows whose window held a stone *before*
      the placement, since every post-placement window trivially holds one.

    A live opponent four or five has at most two empty cells within the legal
    radius, so every such window appears among some legal action's rows. The
    board-level counts are the distinct windows over the whole table, counted
    by persistent window index.
    """
    if not cfg.use_action_tactical_features:
        return np.zeros((tables.n_legal, 0), dtype=np.float32)

    pre_code, post_code = tables.pre_code, tables.post_code
    own_after = OWN_COUNT[post_code]
    opp_before = OPP_COUNT[pre_code]
    own_live_after = STATUS[post_code] == OWN_LIVE
    opp_live_before = STATUS[pre_code] == OPP_LIVE

    rows = (1, 2)
    opponent_five_hit = (opp_live_before & (opp_before == 5)).sum(axis=rows)
    opponent_four_hit = (opp_live_before & (opp_before == 4)).sum(axis=rows)

    def remaining(stones: int) -> int:
        """Distinct live opponent windows on the board holding ``stones``."""
        threat = opp_live_before & (opp_before == stones)
        return len(np.unique(tables.action_window_index[threat]))

    five_remaining = remaining(5)
    four_remaining = remaining(4)

    def broadcast(value) -> np.ndarray:
        """A board-level quantity as one column over the action set."""
        return np.full(tables.n_legal, value)

    def saturate(count: int) -> float:
        return min(count, _GLOBAL_THREAT_CAP) / _GLOBAL_THREAT_CAP

    features = np.stack(
        [
            (post_code == _ALL_OWN_CODE).any(axis=rows),
            own_after.max(axis=rows) / WINDOW_LEN,
            opp_before.max(axis=rows) / WINDOW_LEN,
            (own_live_after & (own_after == 5)).sum(axis=rows) / POST_ACTION_ROWS,
            (own_live_after & (own_after == 4)).sum(axis=rows) / POST_ACTION_ROWS,
            opponent_five_hit / POST_ACTION_ROWS,
            opponent_four_hit / POST_ACTION_ROWS,
            broadcast(saturate(five_remaining)),
            broadcast(saturate(four_remaining)),
            # An action blocks every immediate threat when it sits in all of
            # them. With no threat on the board the flag is false: there is
            # nothing to block, and the count above already says so.
            broadcast(five_remaining > 0) & (opponent_five_hit == five_remaining),
            # Adding an own stone to a window holding only opponent stones is
            # what makes a window mixed, and kills it for both sides.
            opp_live_before.sum(axis=rows) / POST_ACTION_ROWS,
            (pre_code != 0).sum(axis=rows) / POST_ACTION_ROWS,
        ],
        axis=1,
    )
    return features.astype(np.float32, copy=False)
