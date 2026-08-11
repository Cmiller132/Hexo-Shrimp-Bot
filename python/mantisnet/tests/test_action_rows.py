"""Step 4 action-row tables: class laws, successor-board oracle, D6.

The builder's 18 hypothetical post-placement windows per legal action are
checked against an independent oracle that actually plays each action on a
board copy and reads the successor's windows from the engine walk.
"""

from __future__ import annotations

import numpy as np
import pytest

import hexo_py
from mantisnet.builder import (
    ACTION_EMPTY,
    ACTION_MIXED,
    ACTION_OPP,
    ACTION_OWN,
    DEC_CLASSES,
    POST1_GRAFT_CLASSES,
    TERN_POST1_CLASSES,
    WINDOW_LEN,
    _DEC_CLASS,
    _SLOT_FOLD,
    _TERN_POST1_CLASS,
    _TERN_REV,
    from_position,
)

AXES = ((1, 0), (0, 1), (1, -1))

_GAMES = [
    [],
    [(0, 0)],
    [(0, 0), (1, 0), (2, 0)],
    [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)],
    [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (0, 3), (0, 4), (3, 0)],
    [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, -1), (0, 2), (4, -1), (-1, 2)],
]


def test_ternary_post1_law():
    """729 orbits of own-slot (post, slot) pairs, reversal-invariant."""
    assert TERN_POST1_CLASSES == 729
    assert int(_TERN_POST1_CLASS.max()) + 1 == 729
    for post in range(729):
        rev = int(_TERN_REV[post])
        for s in range(WINDOW_LEN):
            own = (post // 3**s) % 3 == 1
            mirror = _TERN_POST1_CLASS[rev, WINDOW_LEN - 1 - s]
            if own:
                assert _TERN_POST1_CLASS[post, s] == mirror >= 0
            else:
                assert _TERN_POST1_CLASS[post, s] == -1


def test_binary_post1_composite_law():
    """189 = 93 own + 93 opp + 3 empty-slot orbits; each branch is
    reversal-invariant and the branches are disjoint."""
    assert POST1_GRAFT_CLASSES == 2 * DEC_CLASSES + 3 == 189

    def reverse6(m: int) -> int:
        return sum(((m >> k) & 1) << (WINDOW_LEN - 1 - k) for k in range(WINDOW_LEN))

    seen = set()
    for mask in range(1, 63):
        for slot in range(WINDOW_LEN):
            if (mask >> slot) & 1:
                continue
            own = int(_DEC_CLASS[mask, slot])
            assert 0 <= own < DEC_CLASSES
            assert own == _DEC_CLASS[reverse6(mask), WINDOW_LEN - 1 - slot]
            seen.add(own)
            seen.add(DEC_CLASSES + own)
    for slot in range(WINDOW_LEN):
        fold = int(_SLOT_FOLD[slot])
        assert fold == _SLOT_FOLD[WINDOW_LEN - 1 - slot]
        seen.add(2 * DEC_CLASSES + fold)
    assert seen == set(range(POST1_GRAFT_CLASSES))


def _successor_windows(pos, move):
    """Engine oracle: play the move on a copy, return the successor's window
    masks keyed by (axis, start_q, start_r)."""
    succ = pos.copy()
    succ.advance(*move)
    return {
        (axis, sq, sr): (m0, m1)
        for axis, sq, sr, m0, m1 in succ.windows_through(*move)
    }


def _expected_row(pre_own: int, pre_opp: int, slot: int, mixed: bool):
    """The class and status an action row must carry, from PRE-insert masks."""
    has_own, has_opp = pre_own > 0, pre_opp > 0
    if has_own and not has_opp:
        status = ACTION_OWN
    elif has_opp and not has_own:
        status = ACTION_OPP
    elif has_own and has_opp:
        status = ACTION_MIXED
    else:
        status = ACTION_EMPTY
    if mixed:
        post = sum(
            (1 if (pre_own >> j) & 1 else 2 if (pre_opp >> j) & 1 else 0) * 3**j
            for j in range(WINDOW_LEN)
        ) + 3**slot
        return int(_TERN_POST1_CLASS[post, slot]), status
    if status == ACTION_OWN:
        return int(_DEC_CLASS[pre_own, slot]), status
    if status == ACTION_OPP:
        return DEC_CLASSES + int(_DEC_CLASS[pre_opp, slot]), status
    if status == ACTION_EMPTY:
        return 2 * DEC_CLASSES + int(_SLOT_FOLD[slot]), status
    return -1, status


@pytest.mark.parametrize("mixed", (False, True))
def test_action_rows_match_the_successor_board_oracle(mixed):
    """Every emitted row agrees with actually playing the action: the class
    recomputed from the successor's engine windows, the status from the pre
    masks, and the window index from the graph's own kept-window identity."""
    rows_checked = 0
    for moves in _GAMES:
        pos = hexo_py.Position.replay(moves)
        mover = pos.current_player
        graph = from_position(pos, mixed_windows=mixed, action_rows=True)
        window_ids = [tuple(map(int, row)) for row in graph.window_id]
        legal = pos.legal_moves()
        picks = range(len(legal)) if len(legal) <= 40 else range(0, len(legal), 7)
        for a in picks:
            move = legal[a]
            oracle = _successor_windows(pos, move)
            for axis, (dq, dr) in enumerate(AXES):
                for k in range(WINDOW_LEN):
                    start = (move[0] - k * dq, move[1] - k * dr)
                    got_class = int(graph.action_post1_class[a, axis, k])
                    got_status = int(graph.action_pre_status[a, axis, k])
                    got_index = int(graph.action_window_index[a, axis, k])

                    key = (axis, start[0], start[1])
                    if key not in oracle:
                        # Off the engine's valid-coordinate domain: the walk
                        # still emits the row from an all-empty line edge.
                        continue
                    m0, m1 = oracle[key]
                    own_post, opp_post = (m0, m1) if mover == 0 else (m1, m0)
                    assert (own_post >> k) & 1, "the played stone is missing"
                    pre_own = own_post & ~(1 << k)
                    want_class, want_status = _expected_row(pre_own, opp_post, k, mixed)
                    assert got_class == want_class, (moves, move, axis, k)
                    assert got_status == want_status, (moves, move, axis, k)

                    kept = (
                        want_status != ACTION_EMPTY
                        if mixed
                        else want_status in (ACTION_OWN, ACTION_OPP)
                    )
                    assert (got_index >= 0) == kept
                    if got_index >= 0:
                        assert window_ids[got_index] == key
                    rows_checked += 1
    assert rows_checked > 500


@pytest.mark.parametrize("mixed", (False, True))
def test_action_row_tables_are_d6_invariant_as_multisets(mixed):
    """A transformed board's rows are the same multiset of (class, status)
    per action — the classes are reversal orbits, so the multiset survives
    any axis permutation or line reversal the transform induces."""
    from mantisnet.klent import telemetry

    for moves in _GAMES[2:]:
        pos = hexo_py.Position.replay(moves)
        graph = from_position(pos, mixed_windows=mixed, action_rows=True)
        base = {
            move: sorted(
                zip(
                    graph.action_post1_class[a].ravel().tolist(),
                    graph.action_pre_status[a].ravel().tolist(),
                )
            )
            for a, move in enumerate(pos.legal_moves())
        }
        for transform in telemetry.D6_TRANSFORMS[1:3]:
            turned_pos = hexo_py.Position.replay([transform(m) for m in moves])
            turned = from_position(turned_pos, mixed_windows=mixed, action_rows=True)
            turned_rows = {
                move: sorted(
                    zip(
                        turned.action_post1_class[a].ravel().tolist(),
                        turned.action_pre_status[a].ravel().tolist(),
                    )
                )
                for a, move in enumerate(turned_pos.legal_moves())
            }
            for move, rows in base.items():
                assert turned_rows[transform(move)] == rows


def test_ply0_rows_are_empty_inserts():
    graph = from_position(hexo_py.Position(), action_rows=True)
    assert graph.action_window_index.shape == (1, 3, 6)
    assert (graph.action_window_index == -1).all()
    assert (graph.action_pre_status == ACTION_EMPTY).all()
    fold = 2 * DEC_CLASSES + _SLOT_FOLD[np.arange(WINDOW_LEN)]
    assert (graph.action_post1_class == fold[None, None, :]).all()


def test_action_rows_default_off():
    graph = from_position(hexo_py.Position.replay(_GAMES[2]))
    assert graph.action_window_index is None
    assert graph.action_post1_class is None
    assert graph.action_pre_status is None
