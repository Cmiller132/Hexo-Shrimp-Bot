"""§30.1–§30.3 for `mantis_act.windows`, against a naive oracle and the engine.

The §30.1 oracle here is deliberately slow and obviously correct: it walks every
window start in a margin around the stones' bounding box, reads each of the six
cells out of a dictionary, and keeps the ones holding a stone. It shares no
coordinate packing, no deduplication, no sorted-set lookup, and no vectorisation
with the enumerator under test, so a symmetric index error — a start computed
one step off on both the enumeration and the classification side — shows up as
disagreeing identity sets rather than cancelling. Every derived quantity is
recomputed here from the oracle's own slot values: the counts, the runs, and the
status come from the six digits directly, not from the tables the enumerator
reads. Only the pattern class is compared through `pattern_classes`, which is a
table with an oracle of its own in `test_act_pattern_classes.py`.

The `live` scope has a second, entirely independent oracle: the engine's own
`windows_through` walk, which the enumerator never calls.
"""

from __future__ import annotations

import itertools

import hexo_py
import numpy as np
import pytest

from mantisnet.models.mantis_act.config import MantisACTConfig
from mantisnet.models.mantis_act.pattern_classes import (
    EMPTY,
    MIXED,
    OPP_LIVE,
    OWN_LIVE,
    PATTERN_CLASS,
)
from mantisnet.models.mantis_act.windows import (
    WINDOW_LEN,
    WINDOW_NUMERIC_NAMES,
    enumerate_windows,
    window_cells,
)

from ..conftest import oracle_live_windows

# The unit steps, spelled out rather than imported: the oracle is not allowed to
# inherit the axis order it is meant to confirm.
UNIT_STEPS = ((1, 0), (0, 1), (1, -1))

# A window through a stone starts at most five steps back along its axis, so
# five would do; six leaves the oracle slack it does not need to justify.
MARGIN = 6

SCOPES = ("live", "nonempty", "action_relevant")


def scoped(scope: str) -> MantisACTConfig:
    """The default configuration at one window scope."""
    return MantisACTConfig(window_scope=scope)


def position_arrays(pos: hexo_py.Position):
    """A position as the ``(stone_qr, stone_own, legal_qr)`` the builder takes."""
    stones = np.asarray(pos.stones(), dtype=np.int64).reshape(-1, 3)
    stone_own = (stones[:, 2] != pos.current_player).astype(np.int64)
    legal = np.asarray(pos.legal_moves(), dtype=np.int64).reshape(-1, 2)
    return stones[:, :2], stone_own, legal


def naive_windows(stones, mover: int) -> dict[tuple[int, int, int], list[int]]:
    """Every window holding a stone, by brute force over a bounding box.

    Returns the identity of each such window mapped to its six slot values,
    ``0`` empty, ``1`` the mover's, ``2`` the opponent's. Nothing is packed,
    sorted, or vectorised: each of the six cells is looked up by coordinate in
    a dictionary, one window at a time.
    """
    occupancy = {(q, r): (1 if player == mover else 2) for q, r, player in stones}
    if not occupancy:
        return {}
    qs = [q for q, _r in occupancy]
    rs = [r for _q, r in occupancy]
    found = {}
    for axis, (step_q, step_r) in enumerate(UNIT_STEPS):
        for start_q in range(min(qs) - MARGIN, max(qs) + MARGIN + 1):
            for start_r in range(min(rs) - MARGIN, max(rs) + MARGIN + 1):
                digits = [
                    occupancy.get((start_q + k * step_q, start_r + k * step_r), 0)
                    for k in range(WINDOW_LEN)
                ]
                if any(digits):
                    found[(axis, start_q, start_r)] = digits
    return found


def longest_run(digits, value: int) -> int:
    """Longest run of ``value`` in a slot list, by grouping consecutive equals."""
    runs = [len(list(g)) for k, g in itertools.groupby(digits) if k == value]
    return max(runs, default=0)


def identities(window_set) -> dict[tuple[int, int, int], int]:
    """The built window identities mapped to their row."""
    return {tuple(int(v) for v in row): i for i, row in enumerate(window_set.window_id)}


def won_position(axis: int) -> hexo_py.Position:
    """An engine replay in which the opener completes six along ``axis``.

    The opener takes the origin alone, then two stones a turn along the axis
    while the other side answers well clear of the line, and wins on the first
    stone of the fourth turn. Random play does not reach a win in any usable
    time on an unbounded board, so the line is played out deliberately.
    """
    step_q, step_r = UNIT_STEPS[axis]
    line = [(k * step_q, k * step_r) for k in range(WINDOW_LEN)]
    # Two parallel columns of three, clear of all three axis lines through the
    # origin, inside the legal radius, and too short to win.
    answers = [(q, r) for r in (3, 4, 5) for q in (2, 3)]
    moves = [line[0]]
    for turn in range(3):
        moves += answers[2 * turn : 2 * turn + 2]
        moves += line[1 + 2 * turn : 3 + 2 * turn]
    pos = hexo_py.Position.replay(moves)
    assert pos.is_terminal, "the deliberate win did not end the game"
    return pos


# --------------------------------------------------------------------------
# §30.1 — the nonempty enumeration against the naive oracle


def test_nonempty_windows_match_the_naive_oracle(positions):
    seen_windows = 0
    for pos in positions:
        stone_qr, stone_own, legal = position_arrays(pos)
        built = enumerate_windows(stone_qr, stone_own, legal, scoped("nonempty"))
        oracle = naive_windows(pos.stones(), pos.current_player)

        rows = identities(built)
        assert set(rows) == set(oracle), f"window identity sets differ at ply {len(pos.stones())}"
        seen_windows += len(oracle)

        cells = window_cells(built.window_id)
        for identity, digits in oracle.items():
            row = rows[identity]
            axis, start_q, start_r = identity
            step_q, step_r = UNIT_STEPS[axis]

            assert [tuple(int(c) for c in cell) for cell in cells[row]] == [
                (start_q + k * step_q, start_r + k * step_r) for k in range(WINDOW_LEN)
            ]
            assert int(built.axis[row]) == axis
            assert int(built.code[row]) == sum(d * 3**k for k, d in enumerate(digits))
            assert int(built.pattern_class[row]) == PATTERN_CLASS[int(built.code[row])]

            own, opp = digits.count(1), digits.count(2)
            assert int(built.status[row]) == (
                MIXED if own and opp else OWN_LIVE if own else OPP_LIVE
            )
            expected = {
                "own_count": own,
                "opp_count": opp,
                "empty_count": digits.count(0),
                "own_max_run": longest_run(digits, 1),
                "opp_max_run": longest_run(digits, 2),
            }
            assert [
                round(float(v) * WINDOW_LEN) for v in built.numeric[row]
            ] == [expected[name] for name in WINDOW_NUMERIC_NAMES]
            assert float(built.numeric[row].max()) <= 1.0
    # The oracle is only a detector if the positions actually carry windows.
    assert seen_windows > 100


def test_windows_come_out_in_the_spec_order(positions):
    for scope in SCOPES:
        for pos in positions:
            built = enumerate_windows(*position_arrays(pos), scoped(scope))
            keys = [tuple(int(v) for v in row) for row in built.window_id]
            assert keys == sorted(keys), f"{scope} windows are not in (axis, q, r) order"
            assert len(set(keys)) == len(keys), f"{scope} windows are not deduplicated"


def test_index_of_finds_every_window_and_only_those(positions):
    for pos in positions:
        built = enumerate_windows(*position_arrays(pos), scoped("nonempty"))
        assert np.array_equal(
            built.index_of(built.window_id), np.arange(built.n_windows)
        )
        # A start shifted off every window in the set: no row may claim it.
        absent = built.window_id + np.array([0, 10_000, 10_000], dtype=np.int64)
        assert np.array_equal(
            built.index_of(absent), np.full(built.n_windows, -1, dtype=np.int64)
        )
        # Shape follows the query, which is how the action table asks.
        assert built.index_of(np.zeros((4, 6, 3), dtype=np.int64)).shape == (4, 6)


# --------------------------------------------------------------------------
# §30.2 — mixed windows kept under nonempty, absent under live


def test_mixed_windows_are_kept_under_nonempty_and_dropped_under_live(positions):
    mixed_seen = 0
    for pos in positions:
        arrays = position_arrays(pos)
        nonempty = enumerate_windows(*arrays, scoped("nonempty"))
        live = enumerate_windows(*arrays, scoped("live"))

        mixed = {
            identity
            for identity, row in identities(nonempty).items()
            if nonempty.status[row] == MIXED
        }
        mixed_seen += len(mixed)

        live_ids = set(identities(live))
        assert live_ids <= set(identities(nonempty))
        assert not (mixed & live_ids), "a mixed window survived the live scope"
        assert live_ids == {
            identity
            for identity, row in identities(nonempty).items()
            if nonempty.status[row] in (OWN_LIVE, OPP_LIVE)
        }
        assert set(live.status.tolist()) <= {OWN_LIVE, OPP_LIVE}
    assert mixed_seen > 0, "no position in the fixture set contains a mixed window"


def test_live_windows_match_the_engine_window_walk(positions):
    for pos in positions:
        built = enumerate_windows(*position_arrays(pos), scoped("live"))
        oracle = oracle_live_windows(pos)
        rows = identities(built)
        assert set(rows) == set(oracle)
        for identity, (colour, mask) in oracle.items():
            row = rows[identity]
            assert int(built.status[row]) == (OWN_LIVE if colour == 0 else OPP_LIVE)
            # The engine's occupancy bitmask against the ternary code's
            # nonempty slots, slot for slot.
            code = int(built.code[row])
            digits = [(code // 3**k) % 3 for k in range(WINDOW_LEN)]
            assert [int(bool(d)) for d in digits] == [(mask >> k) & 1 for k in range(WINDOW_LEN)]


def test_action_relevant_adds_exactly_the_empty_windows_through_legal_cells(positions):
    for pos in positions:
        stone_qr, stone_own, legal = position_arrays(pos)
        nonempty = enumerate_windows(stone_qr, stone_own, legal, scoped("nonempty"))
        relevant = enumerate_windows(stone_qr, stone_own, legal, scoped("action_relevant"))

        base, wider = identities(nonempty), identities(relevant)
        assert set(base) <= set(wider)
        for identity, row in base.items():
            assert relevant.code[wider[identity]] == nonempty.code[row]

        legal_cells = {tuple(int(c) for c in cell) for cell in legal}
        for identity in set(wider) - set(base):
            row = wider[identity]
            assert int(relevant.status[row]) == EMPTY
            cells = {tuple(int(c) for c in cell) for cell in window_cells(relevant.window_id)[row]}
            assert cells & legal_cells

        # Completeness: every one of a legal cell's 18 windows is present.
        for cell in legal:
            for axis, (step_q, step_r) in enumerate(UNIT_STEPS):
                for k in range(WINDOW_LEN):
                    start = (int(cell[0]) - k * step_q, int(cell[1]) - k * step_r)
                    assert (axis, *start) in wider


def test_the_empty_board_has_no_persistent_window():
    empty = np.empty((0, 2), dtype=np.int64)
    for scope in ("live", "nonempty"):
        built = enumerate_windows(empty, np.empty(0, dtype=np.int64), empty, scoped(scope))
        assert built.n_windows == 0
        assert built.window_id.shape == (0, 3)
        assert built.numeric.shape == (0, len(WINDOW_NUMERIC_NAMES))
    # `action_relevant` still persists the windows an opening move can fill.
    opening = enumerate_windows(
        empty,
        np.empty(0, dtype=np.int64),
        np.array([[0, 0]], dtype=np.int64),
        scoped("action_relevant"),
    )
    assert opening.n_windows == 18
    assert set(opening.status.tolist()) == {EMPTY}


# --------------------------------------------------------------------------
# §30.3 — completed lines are refused, full mixed windows are not


@pytest.mark.parametrize("scope", SCOPES)
def test_six_in_a_row_is_refused_under_every_scope(scope):
    line = np.array([[k, 0] for k in range(WINDOW_LEN)], dtype=np.int64)
    no_legal = np.empty((0, 2), dtype=np.int64)
    with pytest.raises(ValueError, match="six own stones"):
        enumerate_windows(line, np.zeros(WINDOW_LEN, dtype=np.int64), no_legal, scoped(scope))
    with pytest.raises(ValueError, match="six opponent stones"):
        enumerate_windows(line, np.ones(WINDOW_LEN, dtype=np.int64), no_legal, scoped(scope))


@pytest.mark.parametrize("axis", range(len(UNIT_STEPS)))
def test_a_real_won_position_is_refused_for_either_mover(axis):
    pos = won_position(axis)
    stones = np.asarray(pos.stones(), dtype=np.int64).reshape(-1, 3)
    legal = np.asarray(pos.legal_moves(), dtype=np.int64).reshape(-1, 2)
    # The winner's six is "own" to one seat and "opponent" to the other, so
    # both refusals are exercised without assuming which seat the engine
    # reports as to move in a finished game.
    for mover in (0, 1):
        stone_own = (stones[:, 2] != mover).astype(np.int64)
        with pytest.raises(ValueError, match="the position is terminal"):
            enumerate_windows(stones[:, :2], stone_own, legal, scoped("nonempty"))


def test_a_full_mixed_window_is_a_node_not_a_refusal():
    line = np.array([[k, 0] for k in range(WINDOW_LEN)], dtype=np.int64)
    own = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    built = enumerate_windows(line, own, np.empty((0, 2), dtype=np.int64), scoped("nonempty"))
    row = identities(built)[(0, 0, 0)]
    assert int(built.status[row]) == MIXED
    counts = dict(zip(WINDOW_NUMERIC_NAMES, built.numeric[row] * WINDOW_LEN))
    assert round(float(counts["own_count"])) == 3
    assert round(float(counts["opp_count"])) == 3
    assert round(float(counts["empty_count"])) == 0
    assert round(float(counts["own_max_run"])) == 3


# --------------------------------------------------------------------------
# Malformed input


def test_malformed_input_is_refused_by_name():
    line = np.array([[k, 0] for k in range(3)], dtype=np.int64)
    no_legal = np.empty((0, 2), dtype=np.int64)
    cfg = scoped("nonempty")

    with pytest.raises(ValueError, match="3 stone coordinates against 2 owners"):
        enumerate_windows(line, np.zeros(2, dtype=np.int64), no_legal, cfg)
    with pytest.raises(ValueError, match=r"stone_own\[1\] = 2"):
        enumerate_windows(line, np.array([0, 2, 1], dtype=np.int64), no_legal, cfg)
    with pytest.raises(ValueError, match=r"repeats coordinate \(1, 0\)"):
        enumerate_windows(
            np.array([[0, 0], [1, 0], [1, 0]], dtype=np.int64),
            np.zeros(3, dtype=np.int64),
            no_legal,
            cfg,
        )
    with pytest.raises(ValueError, match=r"legal cell 1 at \(2, 0\) is occupied"):
        enumerate_windows(
            line,
            np.zeros(3, dtype=np.int64),
            np.array([[9, 9], [2, 0]], dtype=np.int64),
            cfg,
        )
    with pytest.raises(ValueError, match="outside the"):
        enumerate_windows(
            np.array([[1 << 21, 0]], dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            no_legal,
            cfg,
        )


def test_an_unimplemented_window_scope_is_refused():
    # `MantisACTConfig` refuses the value at construction, so the only way to
    # reach the enumerator's own guard is to bypass the frozen dataclass. The
    # guard has to exist regardless: a scope this module does not implement
    # must raise rather than quietly enumerate one it does.
    cfg = scoped("nonempty")
    object.__setattr__(cfg, "window_scope", "everything")
    with pytest.raises(ValueError, match="window_scope='everything'"):
        enumerate_windows(
            np.array([[0, 0]], dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            np.empty((0, 2), dtype=np.int64),
            cfg,
        )
