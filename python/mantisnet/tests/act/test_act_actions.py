"""§30.13 and §30.14: the eighteen counterfactual rows of every legal action.

Three descriptions of the same windows meet here, and the point of the file is
that none of them is derived from another:

- ``actions.py`` slides six windows along one eleven-cell line per axis and
  reaches the post-placement code by adding ``3 ** k`` to the pre-placement one;
- the oracle below walks three axes, six candidate slots, and six slots each in
  plain Python over a dictionary of the board — including a board that really
  carries the hypothetical stone, obtained by advancing a copy of the position
  (§30.14). It never touches the successor's *pattern*: it rebuilds it;
- the engine's own ``windows_through`` walk supplies the window identities and
  their occupancy masks, so the ``(axis, start)`` convention and the slot order
  are pinned to the rules rather than to either Python enumeration.

An off-by-one in the slot-to-power mapping applies and un-applies identically,
so it survives every round trip; only a rebuild from a board that has the stone
on it can see it, which is why the successor here is the engine's successor and
the codes are recomputed from its stone list.

Positions come from the shared engine-replay fixture. The successor oracle runs
on a strided sample of each position's legal actions, since it rebuilds 18
windows from scratch per action; the pre-placement checks run on all of them.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from mantisnet.models.mantis_act.actions import (
    TACTICAL_FEATURE_NAMES,
    TACTICAL_FEATURES,
    action_tables,
    tactical_features,
)
from mantisnet.models.mantis_act.config import MantisACTConfig
from mantisnet.models.mantis_act.packed import (
    NUM_AXES,
    POST_ACTION_ROWS,
    WINDOW_LEN,
)
from mantisnet.models.mantis_act.pattern_classes import (
    EMPTY,
    MIXED,
    OPP_LIVE,
    OWN_LIVE,
    POST1_CLASS,
    STATUS,
    TERNARY_CODES,
)
from mantisnet.models.mantis_act.windows import enumerate_windows

# The engine's axis order, restated here rather than imported: the oracle must
# not inherit the implementation's idea of which axis is which.
ORACLE_AXES = ((1, 0), (0, 1), (1, -1))

LEGAL_RADIUS = 8

CFG = MantisACTConfig()


# --------------------------------------------------------------------------
# Position plumbing and the independent oracles


def position_arrays(pos) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """``(stone_qr, stone_own, legal_qr, mover)`` for a ``hexo_py.Position``.

    ``stone_own`` is ``0`` for the mover's stones and ``1`` for the opponent's,
    which is the convention ``enumerate_windows`` and ``action_tables`` share.
    """
    mover = pos.current_player
    stones = pos.stones()
    stone_qr = np.array([[q, r] for q, r, _ in stones], dtype=np.int64).reshape(-1, 2)
    stone_own = np.array([0 if o == mover else 1 for _, _, o in stones], dtype=np.int64)
    legal_qr = np.array(pos.legal_moves(), dtype=np.int64).reshape(-1, 2)
    return stone_qr, stone_own, legal_qr, mover


def tables_for(pos, cfg: MantisACTConfig = CFG):
    """The window set and action tables of a position under ``cfg``."""
    stone_qr, stone_own, legal_qr, _ = position_arrays(pos)
    window_set = enumerate_windows(stone_qr, stone_own, legal_qr, cfg)
    return window_set, action_tables(window_set, stone_qr, stone_own, legal_qr, cfg)


def oracle_codes(stones, mover: int, cell: tuple[int, int]) -> dict:
    """The 18 windows through ``cell``, rebuilt from a board by hand.

    ``stones`` is a ``(q, r, owner)`` list and ``mover`` the side the codes are
    relative to. Returns ``{(axis, k): ((axis, start_q, start_r), code)}`` — a
    plain triple loop over axes, candidate slots, and slots, with one dictionary
    lookup per cell and no numpy anywhere.
    """
    board = {(q, r): (1 if owner == mover else 2) for q, r, owner in stones}
    out = {}
    for axis, (aq, ar) in enumerate(ORACLE_AXES):
        for k in range(WINDOW_LEN):
            start = (cell[0] - k * aq, cell[1] - k * ar)
            code = 0
            for j in range(WINDOW_LEN):
                code += board.get((start[0] + j * aq, start[1] + j * ar), 0) * 3**j
            out[(axis, k)] = ((axis, start[0], start[1]), code)
    return out


def engine_codes(pos, mover: int, cell: tuple[int, int]) -> dict:
    """The same 18 windows, from the engine's own walk and occupancy masks.

    Returns ``{(axis, start_q, start_r): code}``. The engine reports each
    window's two colour masks with bit ``j`` the cell ``j`` steps from the
    start, so this fixes the slot order as well as the identities.
    """
    out = {}
    for axis, start_q, start_r, mask_p0, mask_p1 in pos.windows_through(*cell):
        own, opp = (mask_p0, mask_p1) if mover == 0 else (mask_p1, mask_p0)
        code = 0
        for j in range(WINDOW_LEN):
            code += (1 if (own >> j) & 1 else 2 if (opp >> j) & 1 else 0) * 3**j
        out[(axis, start_q, start_r)] = code
    return out


def sampled_actions(n_legal: int, wanted: int = 16) -> list[int]:
    """A spread of action indices, always including the first and the last."""
    stride = max(1, n_legal // wanted)
    return sorted(set(list(range(0, n_legal, stride)) + [n_legal - 1]))


def brute_force_legal(stone_qr: np.ndarray) -> np.ndarray:
    """Every empty cell within ``LEGAL_RADIUS`` of a stone, for built boards.

    A plain loop over each stone's disk. This is test *input*, not an oracle:
    ``actions.py`` never computes legality, so nothing is being confirmed
    against itself here.
    """
    occupied = {(int(q), int(r)) for q, r in stone_qr}
    cells = set()
    for q, r in occupied:
        for dq in range(-LEGAL_RADIUS, LEGAL_RADIUS + 1):
            for dr in range(-LEGAL_RADIUS, LEGAL_RADIUS + 1):
                if max(abs(dq), abs(dr), abs(dq + dr)) > LEGAL_RADIUS:
                    continue
                cell = (q + dq, r + dr)
                if cell not in occupied:
                    cells.add(cell)
    return np.array(sorted(cells), dtype=np.int64).reshape(-1, 2)


# --------------------------------------------------------------------------
# §30.13 — eighteen rows, always


def test_every_action_has_eighteen_rows(positions):
    # 3 axes x 6 candidate slots, dense for every legal action including one in
    # no nonempty window: §19.2's replacement for the background alias.
    assert POST_ACTION_ROWS == NUM_AXES * WINDOW_LEN == 18
    for pos in positions:
        _, tables = tables_for(pos)
        n_legal = pos.legal_count
        assert n_legal > 0
        for name in ("action_window_index", "action_post1_class", "action_pre_status"):
            array = getattr(tables, name)
            assert array.shape == (n_legal, NUM_AXES, WINDOW_LEN), name
            assert array.dtype == np.int64, name
        # Never a sentinel: the candidate slot holds an own stone in every
        # post-placement code by construction.
        assert tables.action_post1_class.min() >= 0
        assert tables.action_post1_class.max() < TERNARY_CODES
        statuses = set(np.unique(tables.action_pre_status))
        assert statuses <= {EMPTY, OWN_LIVE, OPP_LIVE, MIXED}


def test_the_eighteen_windows_of_an_action_are_distinct(positions):
    # The rows are 18 different windows, not one window counted 18 ways: two
    # distinct (axis, candidate slot) pairs give two distinct (axis, start)
    # identities, which is what makes the row-wise counts of §19.3 counts of
    # windows.
    for pos in positions:
        _, _, legal_qr, mover = position_arrays(pos)
        for action in sampled_actions(len(legal_qr)):
            cell = tuple(int(c) for c in legal_qr[action])
            identities = {v[0] for v in oracle_codes(pos.stones(), mover, cell).values()}
            assert len(identities) == POST_ACTION_ROWS
            assert identities == set(engine_codes(pos, mover, cell))


def test_action_order_is_engine_order(positions):
    # §7: legal actions are never re-sorted, so row j is legal_moves()[j].
    for pos in positions:
        _, _, legal_qr, mover = position_arrays(pos)
        assert [tuple(int(c) for c in row) for row in legal_qr] == pos.legal_moves()
        _, tables = tables_for(pos)
        assert tables.n_legal == pos.legal_count


# --------------------------------------------------------------------------
# §30.14 — the successor-board oracle


def test_counterfactual_matches_successor_board(positions):
    # The stone is really placed: a copy of the position advances, and the 18
    # windows are rebuilt from the successor's own stone list. Nothing of the
    # fast path's `pre + 3 ** k` arithmetic is reused, so a slot-to-power
    # mapping off by one — which applies and un-applies identically and so
    # survives every round trip — shows up here as a wrong code.
    for pos in positions:
        _, tables = tables_for(pos)
        _, _, legal_qr, mover = position_arrays(pos)
        for action in sampled_actions(len(legal_qr)):
            cell = tuple(int(c) for c in legal_qr[action])
            successor = pos.copy()
            successor.advance(*cell)
            rebuilt = oracle_codes(successor.stones(), mover, cell)
            engine = engine_codes(successor, mover, cell)
            for axis in range(NUM_AXES):
                for slot in range(WINDOW_LEN):
                    identity, code = rebuilt[(axis, slot)]
                    assert engine[identity] == code
                    assert tables.post_code[action, axis, slot] == code
                    assert (
                        tables.action_post1_class[action, axis, slot]
                        == POST1_CLASS[code, slot]
                    )
                    # The candidate slot is own by construction, which is what
                    # makes the class defined for all 18 rows.
                    assert (code // 3**slot) % 3 == 1


def test_pre_placement_codes_match_the_engine_walk(positions):
    # The pre-placement side of the same statement, over every legal action:
    # the row's window state on the *current* board, whether or not that window
    # is a persistent node.
    for pos in positions:
        _, tables = tables_for(pos)
        _, _, legal_qr, mover = position_arrays(pos)
        for action in range(len(legal_qr)):
            cell = tuple(int(c) for c in legal_qr[action])
            engine = engine_codes(pos, mover, cell)
            for axis in range(NUM_AXES):
                for slot in range(WINDOW_LEN):
                    start = (
                        cell[0] - slot * ORACLE_AXES[axis][0],
                        cell[1] - slot * ORACLE_AXES[axis][1],
                    )
                    code = engine[(axis, *start)]
                    assert tables.pre_code[action, axis, slot] == code
                    assert tables.action_pre_status[action, axis, slot] == STATUS[code]
                    # A legal cell is empty, so its own slot is empty before the
                    # placement — the premise the post-placement addition rests on.
                    assert (code // 3**slot) % 3 == 0


def test_post_code_is_the_pre_code_with_the_candidate_slot_own(positions):
    for pos in positions:
        _, tables = tables_for(pos)
        powers = 3 ** np.arange(WINDOW_LEN, dtype=np.int64)
        assert np.array_equal(tables.post_code, tables.pre_code + powers)
        assert tables.post_code.max() < TERNARY_CODES


# --------------------------------------------------------------------------
# The persistent-window link and the window scopes


def test_window_index_agrees_with_the_persistent_window_set(positions):
    # Two independent enumerations of the same windows: this module's line
    # slide, and windows.py's stone-seeded 18-candidate walk. Where the row
    # names a node, the node's identity and its raw code must be the row's.
    for pos in positions:
        window_set, tables = tables_for(pos)
        _, _, legal_qr, _ = position_arrays(pos)
        index = tables.action_window_index
        named = index >= 0
        # Under `nonempty` a window is a node exactly when it holds a stone.
        assert np.array_equal(named, tables.pre_code != 0)
        assert np.array_equal(window_set.code[index[named]], tables.pre_code[named])

        action, axis, slot = named.nonzero()
        step = np.array(ORACLE_AXES, dtype=np.int64)[axis]
        starts = legal_qr[action] - step * slot[:, None]
        identity = np.column_stack([axis, starts])
        assert np.array_equal(window_set.window_id[index[named]], identity)


def test_an_action_in_no_window_still_has_eighteen_rows(positions):
    # The cell furthest from every stone lies in no nonempty window: all 18
    # rows are sentinels, all 18 post-placement classes are defined, and the
    # only stone in each post window is the new one.
    for pos in positions:
        stone_qr, _, legal_qr, _ = position_arrays(pos)
        if len(stone_qr) == 0:
            continue
        delta = legal_qr[:, None, :] - stone_qr[None, :, :]
        distance = np.maximum(
            np.abs(delta[..., 0]),
            np.maximum(np.abs(delta[..., 1]), np.abs(delta.sum(axis=-1))),
        ).min(axis=1)
        action = int(distance.argmax())
        assert distance[action] >= WINDOW_LEN

        _, tables = tables_for(pos)
        assert (tables.action_window_index[action] == -1).all()
        assert (tables.pre_code[action] == 0).all()
        assert (tables.action_pre_status[action] == EMPTY).all()
        assert (tables.action_post1_class[action] >= 0).all()
        assert np.array_equal(
            tables.post_code[action], np.broadcast_to(3 ** np.arange(WINDOW_LEN), (3, 6))
        )


def test_scope_decides_which_rows_are_sentinels(positions):
    # Which rows may be -1 is a function of window_scope alone, and the builder
    # checks its prediction against the node set rather than reading it off.
    for pos in positions:
        for scope in ("live", "nonempty", "action_relevant"):
            _, tables = tables_for(pos, replace(CFG, window_scope=scope))
            status = STATUS[tables.pre_code]
            named = tables.action_window_index >= 0
            if scope == "action_relevant":
                expected = np.ones_like(named)
            elif scope == "nonempty":
                expected = tables.pre_code != 0
            else:
                expected = (status == OWN_LIVE) | (status == OPP_LIVE)
            assert np.array_equal(named, expected), scope
            # The status is the board's either way: a mixed window absent from
            # the `live` node set still reports MIXED.
            assert np.array_equal(tables.action_pre_status, status)


def test_a_window_set_missing_a_required_window_is_refused(positions):
    # A window the scope requires and the node set lacks would otherwise train
    # the learned pre-empty state on a window that exists.
    pos = positions[-1]
    stone_qr, stone_own, legal_qr, _ = position_arrays(pos)
    window_set = enumerate_windows(stone_qr, stone_own, legal_qr, CFG)
    tables = action_tables(window_set, stone_qr, stone_own, legal_qr, CFG)
    named = tables.action_window_index[tables.action_window_index >= 0]
    keep = np.ones(window_set.n_windows, dtype=bool)
    keep[int(named[0])] = False
    truncated = replace(
        window_set,
        window_id=window_set.window_id[keep],
        code=window_set.code[keep],
        pattern_class=window_set.pattern_class[keep],
        status=window_set.status[keep],
        axis=window_set.axis[keep],
        numeric=window_set.numeric[keep],
    )
    with pytest.raises(ValueError, match="window set lacks window"):
        action_tables(truncated, stone_qr, stone_own, legal_qr, CFG)


# --------------------------------------------------------------------------
# §19.3 — the deterministic tactical vector


def test_tactical_vector_shape_and_range(positions):
    assert len(TACTICAL_FEATURE_NAMES) == TACTICAL_FEATURES == 12
    for pos in positions:
        _, tables = tables_for(pos)
        features = tactical_features(tables, CFG)
        assert features.shape == (tables.n_legal, TACTICAL_FEATURES)
        assert features.dtype == np.float32
        assert np.isfinite(features).all()
        assert features.min() >= 0.0 and features.max() <= 1.0


def test_tactical_vector_is_empty_when_disabled(positions):
    for pos in positions:
        _, tables = tables_for(pos)
        off = replace(CFG, use_action_tactical_features=False)
        features = tactical_features(tables, off)
        assert features.shape == (tables.n_legal, 0)
        assert features.dtype == np.float32


def test_immediate_win_matches_the_engine(positions):
    # A placement ends the game exactly when it wins, so the engine's terminal
    # flag on the successor is an oracle for the first field.
    names = list(TACTICAL_FEATURE_NAMES)
    for pos in positions:
        _, tables = tables_for(pos)
        features = tactical_features(tables, CFG)
        _, _, legal_qr, mover = position_arrays(pos)
        for action in sampled_actions(len(legal_qr)):
            successor = pos.copy()
            successor.advance(*(int(c) for c in legal_qr[action]))
            won = successor.is_terminal and successor.winner == mover
            assert bool(features[action, names.index("immediate_win")]) == won


def opponent_threat_board():
    """Five opponent stones in a row with one end blocked by the mover.

    The Q line holds opponent stones at ``(0..4, 0)`` and a mover's stone at
    ``(5, 0)``. Exactly one opponent five-window survives — the one starting at
    ``(-1, 0)``, whose only empty cell is ``(-1, 0)`` — and exactly one
    opponent four-window, the one starting at ``(-2, 0)``.
    """
    stone_qr = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]], dtype=np.int64)
    stone_own = np.array([1, 1, 1, 1, 1, 0], dtype=np.int64)
    return stone_qr, stone_own, brute_force_legal(stone_qr)


def test_threat_fields_on_a_built_position():
    stone_qr, stone_own, legal_qr = opponent_threat_board()
    window_set = enumerate_windows(stone_qr, stone_own, legal_qr, CFG)
    tables = action_tables(window_set, stone_qr, stone_own, legal_qr, CFG)
    features = tactical_features(tables, CFG)
    names = list(TACTICAL_FEATURE_NAMES)

    # One live opponent five and one live opponent four, board-wide, broadcast
    # to every action. The cap is 8, so the values are 1/8 and 1/8.
    assert set(features[:, names.index("opponent_five_windows_remaining")]) == {1 / 8}
    assert set(features[:, names.index("opponent_four_windows_remaining")]) == {1 / 8}

    blocking = int(np.flatnonzero((legal_qr == np.array([-1, 0])).all(axis=1))[0])
    blocks = features[:, names.index("blocks_all_immediate_threats")]
    # The one cell that takes the opponent's winning square blocks everything;
    # nothing else does, and the flag is false wherever there is nothing to
    # block rather than vacuously true.
    assert blocks[blocking] == 1.0
    assert blocks.sum() == 1.0

    hit_five = features[:, names.index("opponent_five_windows_hit")]
    assert hit_five[blocking] == 1 / POST_ACTION_ROWS
    assert set(np.flatnonzero(hit_five)) == {blocking}

    # Placing at (-1, 0) turns that opponent-only window mixed, and (-2, 0)
    # sits in the four-window without touching the five.
    created = features[:, names.index("mixed_windows_created")]
    assert created[blocking] > 0
    far = int(np.flatnonzero((legal_qr == np.array([0, 8])).all(axis=1))[0])
    assert created[far] == 0.0
    assert features[far, names.index("nonempty_pre_windows")] == 0.0


def test_own_threat_fields_on_a_built_position():
    # Four of the mover's stones in a row: the placement at either end makes a
    # live five, and the two placements that complete six win outright.
    stone_qr = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]], dtype=np.int64)
    stone_own = np.zeros(len(stone_qr), dtype=np.int64)
    legal_qr = brute_force_legal(stone_qr)
    window_set = enumerate_windows(stone_qr, stone_own, legal_qr, CFG)
    tables = action_tables(window_set, stone_qr, stone_own, legal_qr, CFG)
    features = tactical_features(tables, CFG)
    names = list(TACTICAL_FEATURE_NAMES)

    winning = {(-1, 0), (5, 0)}
    won = features[:, names.index("immediate_win")]
    assert {tuple(int(c) for c in legal_qr[i]) for i in np.flatnonzero(won)} == winning
    for cell in winning:
        row = int(np.flatnonzero((legal_qr == np.array(cell)).all(axis=1))[0])
        assert features[row, names.index("max_own_count_after")] == 1.0
    # Six or more steps from every stone, the placement shares no window with
    # one, so the maximum after it is the new stone alone.
    row = int(np.flatnonzero((legal_qr == np.array([0, 7])).all(axis=1))[0])
    assert features[row, names.index("max_own_count_after")] == 1 / WINDOW_LEN
    assert features[row, names.index("own_five_windows_after")] == 0.0


# --------------------------------------------------------------------------
# Loud refusals


def test_malformed_input_is_refused(positions):
    pos = positions[3]
    stone_qr, stone_own, legal_qr, _ = position_arrays(pos)
    window_set = enumerate_windows(stone_qr, stone_own, legal_qr, CFG)

    with pytest.raises(ValueError, match="terminal position"):
        action_tables(window_set, stone_qr, stone_own, np.empty((0, 2), np.int64), CFG)

    occupied = np.vstack([legal_qr, stone_qr[:1]])
    with pytest.raises(ValueError, match="is occupied"):
        action_tables(window_set, stone_qr, stone_own, occupied, CFG)

    repeated = np.vstack([legal_qr, legal_qr[:1]])
    with pytest.raises(ValueError, match="repeats the coordinate"):
        action_tables(window_set, stone_qr, stone_own, repeated, CFG)

    wrong_owner = stone_own.copy()
    wrong_owner[0] = 2
    with pytest.raises(ValueError, match="owners are relative to the side to move"):
        action_tables(window_set, stone_qr, wrong_owner, legal_qr, CFG)

    with pytest.raises(ValueError, match="stone coordinates against"):
        action_tables(window_set, stone_qr, stone_own[:-1], legal_qr, CFG)


def test_a_scope_this_module_does_not_implement_is_refused(positions):
    # `MantisACTConfig` cannot hold an unknown scope, so this guards the other
    # direction: a scope added to the vocabulary and not to the sentinel
    # prediction must raise rather than quietly persist nothing.
    pos = positions[3]
    stone_qr, stone_own, legal_qr, _ = position_arrays(pos)
    window_set = enumerate_windows(stone_qr, stone_own, legal_qr, CFG)
    future = SimpleNamespace(window_scope="window_and_threat_relevant")
    with pytest.raises(ValueError, match="unknown window_scope"):
        action_tables(window_set, stone_qr, stone_own, legal_qr, future)
