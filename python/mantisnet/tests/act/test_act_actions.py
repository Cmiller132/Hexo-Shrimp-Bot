"""§19 action rows and tactical features against independent board walks."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

import hexo_py
from mantisnet.models.mantis_act.actions import (
    TACTICAL_FEATURE_NAMES,
    TACTICAL_FEATURES,
)
from mantisnet.models.mantis_act.builder import build
from mantisnet.models.mantis_act.config import MantisACTConfig
from mantisnet.models.mantis_act.packed import NUM_AXES, POST_ACTION_ROWS, WINDOW_LEN
from mantisnet.models.mantis_act.pattern_classes import (
    EMPTY,
    MIXED,
    OPP_LIVE,
    OWN_LIVE,
    POST1_CLASS,
    STATUS,
    TERNARY_CODES,
)

ORACLE_AXES = ((1, 0), (0, 1), (1, -1))
CFG = MantisACTConfig()
GLOBAL_THREAT_CAP = 8

# Engine-legal positions in which both seats have line threats.  WIN_GAME ends
# after P1's first placement and so still has one placement remaining.
THREAT_GAME = [
    (0, 0),
    (0, 7),
    (1, 7),
    (1, 0),
    (2, 0),
    (2, 7),
    (3, 7),
    (3, 0),
    (3, 3),
]
WIN_GAME = [*THREAT_GAME, (4, 7)]
OPPONENT_FIVE_GAME = [*WIN_GAME, (3, 4)]


def position_arrays(position) -> tuple[np.ndarray, np.ndarray, int]:
    legal = np.array(position.legal_moves(), dtype=np.int64).reshape(-1, 2)
    return legal, np.array(position.stones(), dtype=np.int64).reshape(-1, 3), int(
        position.current_player
    )


def oracle_codes(stones, mover: int, cell: tuple[int, int]) -> dict:
    """The eighteen windows through ``cell``, rebuilt by plain Python loops."""
    board = {(q, r): (1 if owner == mover else 2) for q, r, owner in stones}
    out = {}
    for axis, (step_q, step_r) in enumerate(ORACLE_AXES):
        for candidate_slot in range(WINDOW_LEN):
            start = (
                cell[0] - candidate_slot * step_q,
                cell[1] - candidate_slot * step_r,
            )
            code = sum(
                board.get((start[0] + slot * step_q, start[1] + slot * step_r), 0)
                * 3**slot
                for slot in range(WINDOW_LEN)
            )
            out[(axis, candidate_slot)] = ((axis, *start), code)
    return out


def engine_codes(position, mover: int, cell: tuple[int, int]) -> dict:
    """The same rows from the engine's occupancy masks."""
    out = {}
    for axis, start_q, start_r, mask_p0, mask_p1 in position.windows_through(*cell):
        own, opponent = (mask_p0, mask_p1) if mover == 0 else (mask_p1, mask_p0)
        code = sum(
            (1 if (own >> slot) & 1 else 2 if (opponent >> slot) & 1 else 0)
            * 3**slot
            for slot in range(WINDOW_LEN)
        )
        out[(axis, start_q, start_r)] = code
    return out


def sampled_actions(n_legal: int, wanted: int = 16) -> list[int]:
    stride = max(1, n_legal // wanted)
    return sorted(set(range(0, n_legal, stride)) | {n_legal - 1})


def oracle_rows(position):
    """Identities plus pre/post codes for every action in engine order."""
    legal, stones, mover = position_arrays(position)
    identities = np.empty((len(legal), NUM_AXES, WINDOW_LEN, 3), dtype=np.int64)
    pre = np.empty((len(legal), NUM_AXES, WINDOW_LEN), dtype=np.int64)
    for action, cell in enumerate(legal):
        rows = oracle_codes(stones, mover, tuple(int(c) for c in cell))
        for axis in range(NUM_AXES):
            for slot in range(WINDOW_LEN):
                identity, code = rows[(axis, slot)]
                identities[action, axis, slot] = identity
                pre[action, axis, slot] = code
    powers = 3 ** np.arange(WINDOW_LEN, dtype=np.int64)
    return identities, pre, pre + powers


def test_every_action_has_eighteen_rows(positions):
    assert POST_ACTION_ROWS == NUM_AXES * WINDOW_LEN == 18
    for position in positions:
        graph = build(position, CFG)
        shape = (position.legal_count, NUM_AXES, WINDOW_LEN)
        for name in (
            "action_window_index",
            "action_post1_class",
            "action_pre_status",
        ):
            value = getattr(graph, name)
            assert value.shape == shape
            assert value.dtype == np.int64
        assert graph.action_post1_class.min() >= 0
        assert graph.action_post1_class.max() < TERNARY_CODES
        assert set(np.unique(graph.action_pre_status)) <= {
            EMPTY,
            OWN_LIVE,
            OPP_LIVE,
            MIXED,
        }


def test_the_eighteen_windows_of_an_action_are_distinct(positions):
    for position in positions:
        legal, _stones, mover = position_arrays(position)
        for action in sampled_actions(len(legal)):
            cell = tuple(int(c) for c in legal[action])
            identities = {
                entry[0]
                for entry in oracle_codes(position.stones(), mover, cell).values()
            }
            assert len(identities) == POST_ACTION_ROWS
            assert identities == set(engine_codes(position, mover, cell))


def test_action_rows_match_the_board_and_engine_walk(positions):
    for position in positions:
        graph = build(position, CFG)
        identities, pre, post = oracle_rows(position)
        slots = np.arange(WINDOW_LEN, dtype=np.int64)[None, None, :]
        assert np.array_equal(graph.action_pre_status, STATUS[pre])
        assert np.array_equal(graph.action_post1_class, POST1_CLASS[post, slots])

        window_rows = {
            tuple(int(v) for v in identity): row
            for row, identity in enumerate(graph.window_id)
        }
        expected_index = np.array(
            [
                window_rows.get(tuple(int(v) for v in identity), -1)
                for identity in identities.reshape(-1, 3)
            ],
            dtype=np.int64,
        ).reshape(pre.shape)
        assert np.array_equal(graph.action_window_index, expected_index)

        legal, _stones, mover = position_arrays(position)
        for action in sampled_actions(len(legal)):
            cell = tuple(int(c) for c in legal[action])
            walked = engine_codes(position, mover, cell)
            for axis in range(NUM_AXES):
                for slot in range(WINDOW_LEN):
                    identity = tuple(int(v) for v in identities[action, axis, slot])
                    assert pre[action, axis, slot] == walked[identity]


def test_counterfactual_classes_match_successor_boards(positions):
    for position in positions:
        graph = build(position, CFG)
        legal, _stones, mover = position_arrays(position)
        for action in sampled_actions(len(legal)):
            cell = tuple(int(c) for c in legal[action])
            successor = position.copy()
            successor.advance(*cell)
            rebuilt = oracle_codes(successor.stones(), mover, cell)
            engine = engine_codes(successor, mover, cell)
            for axis in range(NUM_AXES):
                for slot in range(WINDOW_LEN):
                    identity, code = rebuilt[(axis, slot)]
                    assert engine[identity] == code
                    assert graph.action_post1_class[action, axis, slot] == POST1_CLASS[
                        code, slot
                    ]
                    assert (code // 3**slot) % 3 == 1


def test_scope_decides_which_rows_are_sentinels(positions):
    for position in positions:
        _identities, pre, _post = oracle_rows(position)
        status = STATUS[pre]
        for scope in ("live", "nonempty", "action_relevant"):
            graph = build(position, replace(CFG, window_scope=scope))
            named = graph.action_window_index >= 0
            if scope == "action_relevant":
                expected = np.ones_like(named)
            elif scope == "nonempty":
                expected = pre != 0
            else:
                expected = (status == OWN_LIVE) | (status == OPP_LIVE)
            assert np.array_equal(named, expected)
            assert np.array_equal(graph.action_pre_status, status)


def naive_tactical(position) -> np.ndarray:
    """§19.3 from the oracle's raw codes and identity tuples."""
    identities, pre, post = oracle_rows(position)
    own_after = np.count_nonzero(
        ((post[..., None] // (3 ** np.arange(WINDOW_LEN))) % 3) == 1, axis=-1
    )
    opponent_after = np.count_nonzero(
        ((post[..., None] // (3 ** np.arange(WINDOW_LEN))) % 3) == 2, axis=-1
    )
    own_before = np.count_nonzero(
        ((pre[..., None] // (3 ** np.arange(WINDOW_LEN))) % 3) == 1, axis=-1
    )
    opponent_before = np.count_nonzero(
        ((pre[..., None] // (3 ** np.arange(WINDOW_LEN))) % 3) == 2, axis=-1
    )
    own_live_after = opponent_after == 0
    opponent_live_before = (own_before == 0) & (opponent_before > 0)
    rows = (1, 2)
    opponent_five_hit = (opponent_live_before & (opponent_before == 5)).sum(axis=rows)
    opponent_four_hit = (opponent_live_before & (opponent_before == 4)).sum(axis=rows)

    def distinct_threats(stones: int) -> int:
        mask = opponent_live_before & (opponent_before == stones)
        return len({tuple(int(v) for v in identity) for identity in identities[mask]})

    five_remaining = distinct_threats(5)
    four_remaining = distinct_threats(4)
    n_legal = len(pre)
    broadcast = lambda value: np.full(n_legal, value)  # noqa: E731
    saturate = lambda count: min(count, GLOBAL_THREAT_CAP) / GLOBAL_THREAT_CAP  # noqa: E731
    all_own = int((3 ** np.arange(WINDOW_LEN, dtype=np.int64)).sum())
    features = np.stack(
        [
            (post == all_own).any(axis=rows),
            own_after.max(axis=rows) / WINDOW_LEN,
            opponent_before.max(axis=rows) / WINDOW_LEN,
            (own_live_after & (own_after == 5)).sum(axis=rows) / POST_ACTION_ROWS,
            (own_live_after & (own_after == 4)).sum(axis=rows) / POST_ACTION_ROWS,
            opponent_five_hit / POST_ACTION_ROWS,
            opponent_four_hit / POST_ACTION_ROWS,
            broadcast(saturate(five_remaining)),
            broadcast(saturate(four_remaining)),
            broadcast(five_remaining > 0) & (opponent_five_hit == five_remaining),
            opponent_live_before.sum(axis=rows) / POST_ACTION_ROWS,
            (pre != 0).sum(axis=rows) / POST_ACTION_ROWS,
        ],
        axis=1,
    )
    return features.astype(np.float32)


def test_tactical_vector_matches_naive_codes(positions):
    assert len(TACTICAL_FEATURE_NAMES) == TACTICAL_FEATURES == 12
    crafted = [
        hexo_py.Position.replay(moves)
        for moves in (THREAT_GAME, WIN_GAME, OPPONENT_FIVE_GAME)
    ]
    for position in [*positions, *crafted]:
        graph = build(position, CFG)
        expected = naive_tactical(position)
        assert graph.action_tactical_numeric.shape == (
            position.legal_count,
            TACTICAL_FEATURES,
        )
        assert graph.action_tactical_numeric.dtype == np.float32
        assert np.array_equal(graph.action_tactical_numeric, expected)
        assert np.isfinite(expected).all()
        assert expected.min() >= 0.0 and expected.max() <= 1.0


def test_tactical_vector_is_empty_when_disabled(positions):
    config = replace(CFG, use_action_tactical_features=False)
    for position in positions:
        graph = build(position, config)
        assert graph.action_tactical_numeric.shape == (position.legal_count, 0)
        assert graph.action_tactical_numeric.dtype == np.float32


def test_immediate_win_matches_the_engine(positions):
    column = TACTICAL_FEATURE_NAMES.index("immediate_win")
    for position in [*positions, hexo_py.Position.replay(WIN_GAME)]:
        graph = build(position, CFG)
        legal, _stones, mover = position_arrays(position)
        for action in sampled_actions(len(legal)):
            successor = position.copy()
            successor.advance(*(int(c) for c in legal[action]))
            won = successor.is_terminal and successor.winner == mover
            assert bool(graph.action_tactical_numeric[action, column]) == won
