"""§30.1–§30.3 window tests against naive and engine oracles.

Production enumeration lives in Rust.  The deliberately slow Python oracle
below remains test-side and compares its identities and slot-derived fields to
the public ``ACTGraph`` result.
"""

from __future__ import annotations

import itertools

import hexo_py
import numpy as np
import pytest

from mantisnet.models.mantis_act.builder import build
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
    window_cells,
)

from ..conftest import oracle_live_windows

UNIT_STEPS = ((1, 0), (0, 1), (1, -1))
MARGIN = 6
SCOPES = ("live", "nonempty", "action_relevant")


def scoped(scope: str) -> MantisACTConfig:
    return MantisACTConfig(window_scope=scope)


def naive_windows(stones, mover: int) -> dict[tuple[int, int, int], list[int]]:
    """Every stone-bearing window, by a brute-force bounding-box walk."""
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
    runs = [len(list(group)) for key, group in itertools.groupby(digits) if key == value]
    return max(runs, default=0)


def identities(graph) -> dict[tuple[int, int, int], int]:
    return {
        tuple(int(value) for value in identity): row
        for row, identity in enumerate(graph.window_id)
    }


def won_position(axis: int) -> hexo_py.Position:
    """An engine replay in which the opener completes six along ``axis``."""
    step_q, step_r = UNIT_STEPS[axis]
    line = [(k * step_q, k * step_r) for k in range(WINDOW_LEN)]
    answers = [(q, r) for r in (3, 4, 5) for q in (2, 3)]
    moves = [line[0]]
    for turn in range(3):
        moves += answers[2 * turn : 2 * turn + 2]
        moves += line[1 + 2 * turn : 3 + 2 * turn]
    position = hexo_py.Position.replay(moves)
    assert position.is_terminal
    return position


def test_nonempty_windows_match_the_naive_oracle(positions):
    seen_windows = 0
    for position in positions:
        graph = build(position, scoped("nonempty"))
        oracle = naive_windows(position.stones(), position.current_player)
        rows = identities(graph)
        assert set(rows) == set(oracle)
        seen_windows += len(oracle)

        cells = window_cells(graph.window_id)
        for identity, digits in oracle.items():
            row = rows[identity]
            axis, start_q, start_r = identity
            step_q, step_r = UNIT_STEPS[axis]
            assert [tuple(int(c) for c in cell) for cell in cells[row]] == [
                (start_q + k * step_q, start_r + k * step_r)
                for k in range(WINDOW_LEN)
            ]
            assert int(graph.window_axis[row]) == axis
            code = sum(digit * 3**slot for slot, digit in enumerate(digits))
            assert int(graph.window_pattern_class[row]) == PATTERN_CLASS[code]

            own, opponent = digits.count(1), digits.count(2)
            assert int(graph.window_status[row]) == (
                MIXED if own and opponent else OWN_LIVE if own else OPP_LIVE
            )
            expected = {
                "own_count": own,
                "opp_count": opponent,
                "empty_count": digits.count(0),
                "own_max_run": longest_run(digits, 1),
                "opp_max_run": longest_run(digits, 2),
            }
            assert [round(float(v) * WINDOW_LEN) for v in graph.window_numeric[row]] == [
                expected[name] for name in WINDOW_NUMERIC_NAMES
            ]
    assert seen_windows > 100


def test_windows_come_out_in_the_spec_order(positions):
    for scope in SCOPES:
        for position in positions:
            graph = build(position, scoped(scope))
            keys = [tuple(int(v) for v in row) for row in graph.window_id]
            assert keys == sorted(keys)
            assert len(keys) == len(set(keys))


def test_mixed_windows_are_kept_under_nonempty_and_dropped_under_live(positions):
    mixed_seen = 0
    for position in positions:
        nonempty = build(position, scoped("nonempty"))
        live = build(position, scoped("live"))
        mixed = {
            identity
            for identity, row in identities(nonempty).items()
            if nonempty.window_status[row] == MIXED
        }
        mixed_seen += len(mixed)
        live_rows = identities(live)
        assert set(live_rows) == {
            identity
            for identity, row in identities(nonempty).items()
            if nonempty.window_status[row] in (OWN_LIVE, OPP_LIVE)
        }
        assert not mixed.intersection(live_rows)
    assert mixed_seen > 0


def test_live_windows_match_the_engine_window_walk(positions):
    for position in positions:
        graph = build(position, scoped("live"))
        oracle = oracle_live_windows(position)
        rows = identities(graph)
        assert set(rows) == set(oracle)
        for identity, (colour, mask) in oracle.items():
            row = rows[identity]
            value = 1 if colour == 0 else 2
            code = sum(value * ((mask >> slot) & 1) * 3**slot for slot in range(WINDOW_LEN))
            assert int(graph.window_status[row]) == (
                OWN_LIVE if colour == 0 else OPP_LIVE
            )
            assert int(graph.window_pattern_class[row]) == PATTERN_CLASS[code]


def test_action_relevant_adds_exactly_empty_windows_through_legal_cells(positions):
    for position in positions:
        nonempty = build(position, scoped("nonempty"))
        relevant = build(position, scoped("action_relevant"))
        base, wider = identities(nonempty), identities(relevant)
        assert set(base) <= set(wider)
        for identity, row in base.items():
            other = wider[identity]
            assert relevant.window_pattern_class[other] == nonempty.window_pattern_class[row]
            assert relevant.window_status[other] == nonempty.window_status[row]

        legal = {tuple(int(c) for c in cell) for cell in position.legal_moves()}
        cells = window_cells(relevant.window_id)
        for identity in set(wider) - set(base):
            row = wider[identity]
            assert int(relevant.window_status[row]) == EMPTY
            assert legal.intersection(tuple(int(c) for c in cell) for cell in cells[row])

        for q, r in legal:
            for axis, (step_q, step_r) in enumerate(UNIT_STEPS):
                for slot in range(WINDOW_LEN):
                    assert (axis, q - slot * step_q, r - slot * step_r) in wider


def test_opening_window_scopes():
    opening = hexo_py.Position()
    for scope in ("live", "nonempty"):
        graph = build(opening, scoped(scope))
        assert graph.window_id.shape == (0, 3)
        assert graph.window_numeric.shape == (0, len(WINDOW_NUMERIC_NAMES))
    relevant = build(opening, scoped("action_relevant"))
    assert len(relevant.window_id) == 18
    assert set(relevant.window_status.tolist()) == {EMPTY}


@pytest.mark.parametrize("scope", SCOPES)
@pytest.mark.parametrize("axis", range(len(UNIT_STEPS)))
def test_real_won_positions_are_refused(scope, axis):
    with pytest.raises(ValueError, match="terminal"):
        build(won_position(axis), scoped(scope))


def test_a_full_mixed_window_is_a_node_not_a_refusal():
    # Seat ownership along q=0..5 is P0,P1,P1,P0,P0,P1: full but mixed.
    position = hexo_py.Position.replay([(q, 0) for q in range(WINDOW_LEN)])
    assert not position.is_terminal
    graph = build(position, scoped("nonempty"))
    row = identities(graph)[(0, 0, 0)]
    assert int(graph.window_status[row]) == MIXED
    counts = dict(zip(WINDOW_NUMERIC_NAMES, graph.window_numeric[row] * WINDOW_LEN))
    assert round(float(counts["own_count"])) == 3
    assert round(float(counts["opp_count"])) == 3
    assert round(float(counts["empty_count"])) == 0
    assert round(float(counts["own_max_run"])) == 2


def test_window_cells_refuses_an_unknown_axis():
    with pytest.raises(ValueError, match="native axis 3"):
        window_cells(np.array([[3, 0, 0]], dtype=np.int64))
