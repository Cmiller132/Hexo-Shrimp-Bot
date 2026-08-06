"""Same-turn partner rows against the engine (§30.15, §30.16, §30.17).

The oracle here is ``hexo_py`` and nothing else: the engine says which cells
are legal after a hypothetical first placement, which first placements end the
game on the spot, and — through ``windows_through``, its own window walk —
which six-cell windows hold both placements and what is in them afterwards.
``pairs.py`` is told none of that; it works from the stone list, the legal-move
list, and the radius rule. Comparing the two as multisets is therefore a real
check rather than a restatement, and a row the builder invents or drops changes
a count.

§30.16 gets a position built for it rather than a sampled one: a legal cell
exactly eight steps from the only stone opens cells no stone covers, so the
newly legal partners the mode exists for are present by construction, and the
test asserts they are nonempty before asserting they are matched.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import hexo_py
import numpy as np
import pytest

from mantisnet.models.mantis_act.config import PRESETS, MantisACTConfig
from mantisnet.models.mantis_act.packed import PHASE_FIRST, PHASE_OPENING, PHASE_SECOND
from mantisnet.models.mantis_act.pairs import (
    PAIR_EVIDENCE_SHARED_WINDOW,
    PAIR_EVIDENCE_TACTICAL,
    PairRows,
    pair_rows,
)
from mantisnet.models.mantis_act.pattern_classes import (
    CLASS_REPRESENTATIVE,
    OPP_COUNT,
    OWN_COUNT,
    PATTERN_CLASS,
)

WINDOW_LEN = 6
AXES = ((1, 0), (0, 1), (1, -1))

# One stone at the origin, the opponent to move with two placements: the plain
# FIRST phase, and the position §30.16 needs — every legal cell eight steps out
# opens cells that no stone reaches.
OPENING_MOVES = [(0, 0)]
# P0 builds a line of three while P1 spreads out; P0 to move with two left.
MIDGAME_MOVES = [(0, 0), (0, 5), (2, 5), (1, 0), (2, 0), (4, 5), (6, 5)]
# The same game two turns later: P0 holds five in a row, so (5, 0) and (-1, 0)
# win on the first placement of the turn.
FIVE_IN_A_ROW_MOVES = MIDGAME_MOVES + [(3, 0), (4, 0), (0, 7), (2, 7)]


def position(moves: list[tuple[int, int]]) -> hexo_py.Position:
    return hexo_py.Position.replay(moves)


def act_inputs(pos: hexo_py.Position) -> tuple:
    """The builder-side inputs of one position, as ``pair_rows`` takes them."""
    stones = pos.stones()
    if stones:
        table = np.asarray(stones, dtype=np.int64)
        stone_qr, owner = table[:, :2], table[:, 2]
    else:
        stone_qr = np.empty((0, 2), dtype=np.int64)
        owner = np.empty(0, dtype=np.int64)
    legal = pos.legal_moves()
    legal_qr = (
        np.asarray(legal, dtype=np.int64)
        if legal
        else np.empty((0, 2), dtype=np.int64)
    )
    if not stones:
        phase = PHASE_OPENING
    elif pos.moves_remaining == 2:
        phase = PHASE_FIRST
    else:
        phase = PHASE_SECOND
    return (
        stone_qr,
        (owner != pos.current_player).astype(np.int64),
        legal_qr,
        pos.moves_remaining,
        phase,
    )


def config(scope: str) -> MantisACTConfig:
    if scope == "none":
        return PRESETS["full_no_pair"]
    return replace(MantisACTConfig(), pair_scope=scope)


def build(pos: hexo_py.Position, scope: str) -> PairRows:
    return pair_rows(None, None, *act_inputs(pos), config(scope))


def counted(rows: PairRows) -> Counter:
    """The rows as a multiset of seven-tuples, which is all §7 order leaves."""
    return Counter(
        zip(
            rows.pair_dst_action.tolist(),
            rows.pair_src_action_or_neg1.tolist(),
            rows.pair_axis_or_neg1.tolist(),
            rows.pair_distance.tolist(),
            rows.pair_post2_pattern.tolist(),
            rows.pair_evidence_kind.tolist(),
            rows.pair_src_is_current_legal.tolist(),
        )
    )


def slot_of(cell: tuple[int, int], axis: int, start_q: int, start_r: int) -> int | None:
    """Which slot of a window a cell occupies, or ``None`` if it is outside."""
    step_q, step_r = AXES[axis]
    for k in range(WINDOW_LEN):
        if (start_q + k * step_q, start_r + k * step_r) == cell:
            return k
    return None


def ternary_code(mask_p0: int, mask_p1: int, mover: int) -> int:
    """A window's ternary code from the engine's two occupancy masks."""
    own_mask, opp_mask = (mask_p0, mask_p1) if mover == 0 else (mask_p1, mask_p0)
    code = 0
    for k in range(WINDOW_LEN):
        if (own_mask >> k) & 1:
            code += 1 * 3**k
        elif (opp_mask >> k) & 1:
            code += 2 * 3**k
    return code


def oracle(pos: hexo_py.Position, scope: str, max_distance: int = 5) -> Counter:
    """Every §20.3 collinear row of a position, straight from the engine.

    Placement legality, immediate wins, the windows holding both placements and
    their contents afterwards all come from ``hexo_py``; the only arithmetic
    here is stepping along an axis.
    """
    if scope == "none" or pos.moves_remaining != 2 or not pos.stones():
        return Counter()
    mover = pos.current_player
    legal = pos.legal_moves()
    index = {cell: i for i, cell in enumerate(legal)}
    occupied = {(q, r) for q, r, _ in pos.stones()}
    offsets = [s for s in range(-max_distance, max_distance + 1) if s]

    rows: Counter = Counter()
    for i, action in enumerate(legal):
        after_action = pos.copy()
        after_action.advance(*action)
        if after_action.is_terminal:
            continue
        allowed = (
            set(legal)
            if scope == "current_legal_collinear"
            else set(after_action.legal_moves())
        )
        for axis, (step_q, step_r) in enumerate(AXES):
            for offset in offsets:
                partner = (action[0] + offset * step_q, action[1] + offset * step_r)
                if partner in occupied or partner not in allowed:
                    continue
                after_partner = after_action.copy()
                after_partner.advance(*partner)
                for window_axis, start_q, start_r, mask_p0, mask_p1 in (
                    after_partner.windows_through(*action)
                ):
                    if slot_of(partner, window_axis, start_q, start_r) is None:
                        continue
                    assert window_axis == axis, "a shared window runs along the pair's axis"
                    rows[
                        (
                            i,
                            index.get(partner, -1),
                            axis,
                            abs(offset),
                            int(PATTERN_CLASS[ternary_code(mask_p0, mask_p1, mover)]),
                            PAIR_EVIDENCE_SHARED_WINDOW,
                            int(partner in index),
                        )
                    ] += 1
    return rows


# --------------------------------------------------------------------------
# §30.15 — empty partner cells, correct distance and axis, correct patterns


@pytest.mark.parametrize("scope", ["current_legal_collinear", "post_action_collinear"])
@pytest.mark.parametrize(
    "moves",
    [OPENING_MOVES, MIDGAME_MOVES, FIVE_IN_A_ROW_MOVES],
    ids=["ply1", "midgame", "five"],
)
def test_collinear_rows_match_the_engine(moves, scope):
    pos = position(moves)
    assert pos.moves_remaining == 2, "these positions are FIRST-phase by construction"
    assert counted(build(pos, scope)) == oracle(pos, scope)


def test_visible_partners_are_empty_cells_at_the_stated_offset():
    """Every row naming a current partner names an empty cell on its axis.

    The prospective partners are checked through the engine multiset above;
    this is the half the row itself can be read against the board.
    """
    pos = position(MIDGAME_MOVES)
    legal = pos.legal_moves()
    occupied = {(q, r) for q, r, _ in pos.stones()}
    rows = build(pos, "post_action_collinear")
    seen = 0
    for dst, src, axis, distance in zip(
        rows.pair_dst_action.tolist(),
        rows.pair_src_action_or_neg1.tolist(),
        rows.pair_axis_or_neg1.tolist(),
        rows.pair_distance.tolist(),
    ):
        if src < 0:
            continue
        seen += 1
        action, partner = legal[dst], legal[src]
        assert partner not in occupied
        step_q, step_r = AXES[axis]
        offset_q = partner[0] - action[0]
        offset_r = partner[1] - action[1]
        assert (abs(offset_q), abs(offset_r)) == (
            distance * abs(step_q),
            distance * abs(step_r),
        )
        assert offset_q * step_r == offset_r * step_q  # collinear with the axis
        assert 1 <= distance <= 5
    assert seen > 0


def test_rows_are_sorted_by_destination_action():
    rows = build(position(MIDGAME_MOVES), "post_action_collinear")
    assert np.all(np.diff(rows.pair_dst_action) >= 0)


def test_row_fields_are_int64_and_in_range():
    rows = build(position(MIDGAME_MOVES), "post_action_collinear")
    n_legal = len(position(MIDGAME_MOVES).legal_moves())
    for name in (
        "pair_dst_action",
        "pair_src_action_or_neg1",
        "pair_axis_or_neg1",
        "pair_distance",
        "pair_post2_pattern",
        "pair_evidence_kind",
        "pair_src_is_current_legal",
    ):
        assert getattr(rows, name).dtype == np.int64, name
    assert rows.pair_dst_action.min() >= 0
    assert rows.pair_dst_action.max() < n_legal
    assert rows.pair_src_action_or_neg1.max() < n_legal
    assert rows.pair_src_action_or_neg1.min() >= -1
    assert set(np.unique(rows.pair_axis_or_neg1).tolist()) <= {0, 1, 2}
    assert rows.pair_distance.min() >= 1 and rows.pair_distance.max() <= 5
    assert rows.pair_post2_pattern.min() >= 1  # class 0 is the all-empty pattern
    assert np.array_equal(
        rows.pair_src_is_current_legal, (rows.pair_src_action_or_neg1 >= 0).astype(np.int64)
    )


def test_pair_max_distance_bounds_the_enumeration():
    pos = position(MIDGAME_MOVES)
    cfg = replace(MantisACTConfig(), pair_max_distance=2)
    rows = pair_rows(None, None, *act_inputs(pos), cfg)
    assert rows.pair_distance.max() == 2
    assert counted(rows) == oracle(pos, "post_action_collinear", max_distance=2)


# --------------------------------------------------------------------------
# §30.16 — newly legal partners whenever the engine has them


def newly_legal(pos: hexo_py.Position, action: tuple[int, int]) -> set[tuple[int, int]]:
    """The cells the engine opens by placing ``action``."""
    after = pos.copy()
    after.advance(*action)
    return set(after.legal_moves()) - set(pos.legal_moves())


def test_an_edge_action_really_opens_new_cells():
    """The §30.16 premise, asserted so the tests below cannot pass vacuously."""
    pos = position(OPENING_MOVES)
    opened = newly_legal(pos, (8, 0))
    assert opened, "a cell eight steps from the only stone must open new ones"
    assert (9, 0) in opened


def test_post_action_collinear_matches_the_engine_on_an_edge_action():
    """Every partner the engine opens on the pair's axes is represented.

    The rows do not carry the partner coordinate, so the count is the check:
    two cells ``d`` apart lie in ``6 - d`` common windows, so the engine's
    newly opened collinear cells fix exactly how many prospective rows the
    action must have.
    """
    pos = position(OPENING_MOVES)
    legal = pos.legal_moves()
    action = (8, 0)
    dst = legal.index(action)
    opened = newly_legal(pos, action)
    occupied = {(q, r) for q, r, _ in pos.stones()}

    expected = 0
    for step_q, step_r in AXES:
        for offset in [s for s in range(-5, 6) if s]:
            partner = (action[0] + offset * step_q, action[1] + offset * step_r)
            if partner in opened and partner not in occupied:
                expected += WINDOW_LEN - abs(offset)
    assert expected > 0

    rows = build(pos, "post_action_collinear")
    prospective = (rows.pair_dst_action == dst) & (rows.pair_src_action_or_neg1 < 0)
    assert int(prospective.sum()) == expected


def test_post_action_collinear_adds_exactly_the_opened_cells():
    """The post scope is the current scope plus prospective rows, nothing else."""
    pos = position(OPENING_MOVES)
    current = counted(build(pos, "current_legal_collinear"))
    post = counted(build(pos, "post_action_collinear"))
    assert all(row[1] >= 0 for row in current), "current partners are legal actions"
    assert current <= post
    added = post - current
    assert added, "the mode exists for the cells the placement opens"
    assert all(row[1] == -1 and row[6] == 0 for row in added)


def test_prospective_rows_are_present_and_flagged():
    rows = build(position(OPENING_MOVES), "post_action_collinear")
    prospective = rows.pair_src_action_or_neg1 < 0
    assert int(prospective.sum()) > 0
    assert not rows.pair_src_is_current_legal[prospective].any()


# --------------------------------------------------------------------------
# §30.17 — no rows outside the first phase, none for a winning action


@pytest.mark.parametrize("moves", [[], [(0, 0), (3, 1)]], ids=["opening", "second"])
def test_no_rows_outside_the_first_phase(moves):
    pos = position(moves)
    assert pos.moves_remaining == 1
    for scope in ("current_legal_collinear", "post_action_collinear", "post_action_tactical"):
        assert build(pos, scope).n_rows == 0


@pytest.mark.parametrize(
    "scope", ["current_legal_collinear", "post_action_collinear", "post_action_tactical"]
)
def test_no_rows_for_an_immediately_winning_action(scope):
    pos = position(FIVE_IN_A_ROW_MOVES)
    legal = pos.legal_moves()
    winners = [(5, 0), (-1, 0)]
    for cell in winners:
        after = pos.copy()
        after.advance(*cell)
        assert after.is_terminal, "the engine agrees this placement wins on the spot"

    rows = build(pos, scope)
    destinations = set(rows.pair_dst_action.tolist())
    for cell in winners:
        assert legal.index(cell) not in destinations
    # A nonwinning action of the same position still has its rows, so the gate
    # is on the action and not on the position.
    assert legal.index((0, 1)) in destinations


def test_pair_scope_none_emits_nothing():
    rows = build(position(MIDGAME_MOVES), "none")
    assert rows.n_rows == 0
    assert rows.pair_dst_action.dtype == np.int64


# --------------------------------------------------------------------------
# §20.4 — the optional tactical scope


def hex_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    dq, dr = b[0] - a[0], b[1] - a[1]
    return max(abs(dq), abs(dr), abs(dq + dr))


def tactical_oracle(pos: hexo_py.Position, max_distance: int = 5) -> Counter:
    """§20.4's rows, from the engine's window walk and its post-action legality.

    Only the threshold rule itself is restated here — which windows count as
    settled by a partner. The windows, their contents, and whether a partner is
    legal after the first placement all come from ``hexo_py``, so an indexing or
    broadcasting fault in the vectorised enumeration shows up as a count.
    """
    mover = pos.current_player
    legal = pos.legal_moves()
    index = {cell: i for i, cell in enumerate(legal)}

    after_action = {}
    for i, action in enumerate(legal):
        successor = pos.copy()
        successor.advance(*action)
        if not successor.is_terminal:
            after_action[i] = set(successor.legal_moves())

    windows = {}
    for q, r, _player in pos.stones():
        for axis, start_q, start_r, mask_p0, mask_p1 in pos.windows_through(q, r):
            windows[(axis, start_q, start_r)] = (mask_p0, mask_p1)

    rows: Counter = Counter()
    for (axis, start_q, start_r), (mask_p0, mask_p1) in windows.items():
        own_mask, opp_mask = (mask_p0, mask_p1) if mover == 0 else (mask_p1, mask_p0)
        own, opp = bin(own_mask).count("1"), bin(opp_mask).count("1")
        if not ((opp == 0 and own >= 3) or (own == 0 and opp >= 4)):
            continue
        step_q, step_r = AXES[axis]
        cells = [(start_q + k * step_q, start_r + k * step_r) for k in range(WINDOW_LEN)]
        for k, partner in enumerate(cells):
            if (own_mask >> k) & 1 or (opp_mask >> k) & 1:
                continue
            post2 = int(PATTERN_CLASS[ternary_code(mask_p0, mask_p1, mover) + 3**k])
            for i, action in enumerate(legal):
                if i not in after_action or action in cells:
                    continue
                if partner not in after_action[i]:
                    continue
                rows[
                    (
                        i,
                        index.get(partner, -1),
                        -1,
                        min(hex_distance(action, partner), max_distance),
                        post2,
                        PAIR_EVIDENCE_TACTICAL,
                        int(partner in index),
                    )
                ] += 1
    return rows


def test_tactical_rows_match_the_engine():
    pos = position(FIVE_IN_A_ROW_MOVES)
    rows = build(pos, "post_action_tactical")
    tactical = Counter(
        row for row in counted(rows).elements() if row[5] == PAIR_EVIDENCE_TACTICAL
    )
    expected = tactical_oracle(pos)
    assert expected, "five own stones in a row leave windows a partner settles"
    assert tactical == expected


def test_tactical_scope_extends_the_collinear_rows():
    pos = position(FIVE_IN_A_ROW_MOVES)
    collinear = counted(build(pos, "post_action_collinear"))
    tactical = counted(build(pos, "post_action_tactical"))
    assert collinear <= tactical
    added = tactical - collinear
    assert added, "five own stones in a row leave windows a partner settles"
    assert all(row[2] == -1 and row[5] == PAIR_EVIDENCE_TACTICAL for row in added)


def test_tactical_rows_carry_a_decided_window():
    """Every tactical row's window is an own four-plus or a killed opponent four."""
    rows = build(position(FIVE_IN_A_ROW_MOVES), "post_action_tactical")
    tactical = rows.pair_evidence_kind == PAIR_EVIDENCE_TACTICAL
    assert int(tactical.sum()) > 0
    codes = CLASS_REPRESENTATIVE[rows.pair_post2_pattern[tactical]]
    own, opp = OWN_COUNT[codes], OPP_COUNT[codes]
    own_side = (own >= 4) & (opp == 0)
    opp_side = (opp >= 4) & (own == 1)
    assert bool(np.all(own_side | opp_side))
    assert bool(np.all(rows.pair_axis_or_neg1[tactical] == -1))
    assert int(rows.pair_distance[tactical].min()) >= 1
    assert int(rows.pair_distance[tactical].max()) <= 5


# --------------------------------------------------------------------------
# Loud failures


def call(inputs: tuple, cfg=None) -> PairRows:
    """``pair_rows`` on an input tuple the caller has damaged on purpose."""
    return pair_rows(None, None, *inputs, cfg or config("post_action_collinear"))


def test_unknown_pair_scope_raises():
    cfg = SimpleNamespace(pair_scope="collinear_ish", pair_max_distance=5)
    with pytest.raises(ValueError, match="collinear_ish"):
        call(act_inputs(position(MIDGAME_MOVES)), cfg)


def test_out_of_range_pair_max_distance_raises():
    cfg = SimpleNamespace(pair_scope="post_action_collinear", pair_max_distance=6)
    with pytest.raises(ValueError, match="pair_max_distance"):
        call(act_inputs(position(MIDGAME_MOVES)), cfg)


def test_phase_and_moves_remaining_must_agree():
    stone_qr, stone_own, legal_qr, _moves, _phase = act_inputs(position(MIDGAME_MOVES))
    with pytest.raises(ValueError, match="disagrees"):
        call((stone_qr, stone_own, legal_qr, 1, PHASE_FIRST))


def test_first_phase_without_stones_raises():
    _stones, _own, legal_qr, moves, phase = act_inputs(position(OPENING_MOVES))
    stoneless = (
        np.empty((0, 2), dtype=np.int64),
        np.empty(0, dtype=np.int64),
        legal_qr,
        moves,
        phase,
    )
    with pytest.raises(ValueError, match="OPENING"):
        call(stoneless)


def test_occupied_legal_cell_raises():
    stone_qr, stone_own, legal_qr, moves, phase = act_inputs(position(MIDGAME_MOVES))
    legal_qr = np.concatenate([legal_qr, stone_qr[:1]])
    with pytest.raises(ValueError, match="occupied cell"):
        call((stone_qr, stone_own, legal_qr, moves, phase))


def test_duplicate_stone_raises():
    stone_qr, stone_own, legal_qr, moves, phase = act_inputs(position(MIDGAME_MOVES))
    stone_qr = np.concatenate([stone_qr, stone_qr[:1]])
    stone_own = np.concatenate([stone_own, stone_own[:1]])
    with pytest.raises(ValueError, match="duplicate stone"):
        call((stone_qr, stone_own, legal_qr, moves, phase))


def test_stone_owner_outside_the_two_colours_raises():
    stone_qr, stone_own, legal_qr, moves, phase = act_inputs(position(MIDGAME_MOVES))
    stone_own = stone_own.copy()
    stone_own[0] = 2
    with pytest.raises(ValueError, match="stone_own"):
        call((stone_qr, stone_own, legal_qr, moves, phase))


def test_malformed_coordinate_array_raises():
    stone_qr, stone_own, legal_qr, moves, phase = act_inputs(position(MIDGAME_MOVES))
    with pytest.raises(ValueError, match=r"legal_qr must have shape"):
        call((stone_qr, stone_own, legal_qr.reshape(-1), moves, phase))
