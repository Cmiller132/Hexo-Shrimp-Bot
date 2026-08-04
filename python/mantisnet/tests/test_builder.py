"""§12.1 builder oracle, §12.2 ordering, §12.6 decoder coverage.

The oracle is the engine's own window walk (`windows_through`), which the
builder never calls — the two enumerate the same set from independent code.
"""

from __future__ import annotations

import hexo_py
import numpy as np
import pytest

from mantisnet import NUM_PATTERNS, OCC_CLASSES, collate, from_position
from mantisnet.builder import DEC_CLASSES, OCC_FOLD, _CANON, _PATTERN_RANK, AXES

from .conftest import (
    JOINT_ORBITS,
    OCC_ORBITS,
    joint_class,
    occ_class,
    oracle_live_windows,
    reverse6,
)


def test_ninety_three_joint_decoder_classes():
    # The 186 (nonempty-nonfull mask, empty slot) pairs fold to 186 / 2 = 93
    # orbits: no slot is its own mirror, so the involution has no fixed point.
    assert DEC_CLASSES == len(JOINT_ORBITS) == 93
    pairs = [(m, s) for m in range(1, 63) for s in range(6) if not (m >> s) & 1]
    assert len(pairs) == 186
    # Exactly reversal-invariant, and no coarser: the two members of an orbit
    # share a class, and members of different orbits never do.
    assert all(joint_class(m, s) == joint_class(reverse6(m), 5 - s) for m, s in pairs)
    assert len({joint_class(m, s) for m, s in pairs}) == 93
    # Strictly finer than the slot class it replaces, at 18 of the 93: a
    # (canonical mask, slot class) key merges the two ways a candidate can sit
    # at mirrored slots of a non-palindromic window — an adjacent extension and
    # a split one get one embedding under it and two under this.
    merged = {}
    for m, s in pairs:
        merged.setdefault(
            (min(m, reverse6(m)), min(s, 5 - s)), set()
        ).add(joint_class(m, s))
    assert sum(len(v) - 1 for v in merged.values()) == 18
    assert all(len(v) <= 2 for v in merged.values())
    assert joint_class(0b000001, 1) != joint_class(0b000001, 4)


def test_ninety_three_joint_incidence_classes():
    # The same fold over occupied slots: complementing the mask maps the
    # decoder's 186 (mask, empty slot) pairs onto these bijectively and
    # commutes with reversal, so the counts repeat — 93 orbits, and 18 more
    # classes than the coarse (canonical mask, slot class) key realizes.
    assert OCC_CLASSES == len(OCC_ORBITS) == 93
    pairs = [(m, s) for m in range(1, 63) for s in range(6) if (m >> s) & 1]
    assert len(pairs) == 186
    assert all(occ_class(m, s) == occ_class(reverse6(m), 5 - s) for m, s in pairs)
    assert len({occ_class(m, s) for m, s in pairs}) == 93
    merged = {}
    for m, s in pairs:
        merged.setdefault(
            (min(m, reverse6(m)), min(s, 5 - s)), set()
        ).add(occ_class(m, s))
    assert sum(len(v) - 1 for v in merged.values()) == 18
    assert all(len(v) <= 2 for v in merged.values())
    # The aliasing the joint classes remove: in 110001 the lone stone and the
    # pair's outer stone are both "end", and the coarse class cannot say which
    # end of the pattern a stone's own state binds to.
    assert occ_class(0b110001, 0) != occ_class(0b110001, 5)
    assert min(0, 5 - 0) == min(5, 5 - 5) == 0


def test_the_incidence_fold_reproduces_the_coarse_slot_class():
    # The §4.3 fold consumed by stock (joint_incidence=False) models: every
    # orbit lands exactly on its members' shared min(s, 5 - s), so folded
    # joint classes reproduce the pre-joint three-class incidence bit for bit.
    pairs = [(m, s) for m in range(1, 63) for s in range(6) if (m >> s) & 1]
    assert all(OCC_FOLD[occ_class(m, s)] == min(s, 5 - s) for m, s in pairs)
    assert sorted(set(OCC_FOLD.tolist())) == [0, 1, 2]


def test_thirty_four_canonical_patterns():
    # 62 nonempty, nonfull 6-bit masks fold to (62 + 6 palindromes) / 2 = 34
    # orbits under reversal.
    assert NUM_PATTERNS == 34
    assert (_PATTERN_RANK[_CANON[1:63]] >= 0).all()
    assert _PATTERN_RANK[0] == -1 and _PATTERN_RANK[63] == -1


def test_windows_match_engine_oracle(positions):
    for pos in positions:
        g = from_position(pos)
        oracle = oracle_live_windows(pos)

        built = {tuple(w): None for w in g.window_id}
        assert set(built) == set(oracle), "live-window identity sets differ"

        # Feature agreement: colour and canonical occupancy pattern.
        for i, wid in enumerate(map(tuple, g.window_id)):
            colour, occ = oracle[wid]
            assert g.window_feat[i] == colour * NUM_PATTERNS + _PATTERN_RANK[_CANON[occ]]


def test_incidence_matches_oracle(positions):
    for pos in positions:
        g = from_position(pos)
        oracle = oracle_live_windows(pos)
        stones = {i: (q, r) for i, (q, r, _p) in enumerate(pos.stones())}

        built = {
            (tuple(g.window_id[w]), stones[s], int(c))
            for s, w, c in zip(g.inc_stone, g.inc_window, g.inc_class)
        }
        expected = set()
        for (axis, sq, sr), (_colour, occ) in oracle.items():
            for k in range(6):
                if occ >> k & 1:
                    cell = (sq + k * int(AXES[axis, 0]), sr + k * int(AXES[axis, 1]))
                    # The class pairs the window's occupancy — the oracle's
                    # own, from the engine's walk — with this stone's slot.
                    expected.add(((axis, sq, sr), cell, occ_class(occ, k)))
        assert built == expected


def test_incidence_entries_arrive_in_window_order(positions):
    # Each window's entries remain contiguous before and after collation.
    graphs = [from_position(pos) for pos in positions]
    for g in graphs:
        assert (np.diff(g.inc_window) >= 0).all()
    batch = collate(graphs)
    assert (batch.inc_window[1:] >= batch.inc_window[:-1]).all()


def test_decoder_table_ordering_and_coverage(positions):
    for pos in positions:
        g = from_position(pos)
        legal = pos.legal_moves()
        assert g.n_legal == len(legal) == pos.legal_count

        # §12.6: every legal cell is scored exactly once — window path and
        # background path partition the index range.
        window_cells = set(g.dec_cell.tolist())
        bg_cells = set(g.bg_cell.tolist())
        assert window_cells.isdisjoint(bg_cells)
        assert window_cells | bg_cells == set(range(len(legal)))

        # §12.2: the table is asserted against `legal_moves[j]` itself — index
        # j's windows are exactly the live windows through the j-th legal cell.
        oracle = oracle_live_windows(pos)
        live = set(oracle)
        rows_by_cell: dict[int, set] = {}
        for cell, w, c in zip(g.dec_cell, g.dec_window, g.dec_class):
            rows_by_cell.setdefault(int(cell), set()).add((tuple(g.window_id[w]), int(c)))
        bucket_by_cell = dict(zip(g.bg_cell.tolist(), g.bg_bucket.tolist()))
        for j, (q, r) in enumerate(legal):
            expected = set()
            for axis, vec in enumerate(AXES):
                for k in range(6):
                    wid = (axis, q - k * int(vec[0]), r - k * int(vec[1]))
                    if wid in live:
                        # The class pairs the window's occupancy — the oracle's
                        # own, from the engine's walk — with this cell's slot.
                        expected.add((wid, joint_class(oracle[wid][1], k)))
            assert rows_by_cell.get(j, set()) == expected, f"decoder row {j} at {(q, r)}"
            if not expected:
                dists = [
                    max(abs(q - sq), abs(r - sr), abs((q - sq) + (r - sr)))
                    for sq, sr, _p in pos.stones()
                ]
                bucket = min(min(dists), 8) - 1 if dists else 7
                assert bucket_by_cell[j] == bucket


def test_decoder_entries_arrive_in_cell_order(positions):
    # Decoder entries for each cell remain contiguous before and after collation.
    graphs = [from_position(pos) for pos in positions]
    for g in graphs:
        assert (np.diff(g.dec_cell) >= 0).all()
    batch = collate(graphs)
    assert (batch.dec_cell[1:] >= batch.dec_cell[:-1]).all()


def test_ply_zero_builds_background_only():
    g = from_position(hexo_py.Position())
    assert g.n_stones == 0 and g.n_windows == 0
    assert g.n_legal == 1
    assert g.bg_cell.tolist() == [0] and g.bg_bucket.tolist() == [7]
    assert g.moves_remaining == 1


def test_terminal_position_is_a_builder_error():
    pos = hexo_py.Position()
    # P0 builds six in a row on the Q axis while P1 plays far away on R.
    pos.advance(0, 0)
    # Two P1 triples on separate columns — never six for P1.
    p1 = [(-8, 8), (-8, 9), (-8, 10), (-6, 8), (-6, 9), (-6, 10)]
    p0 = [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]
    seq = [p1[0], p1[1], p0[0], p0[1], p1[2], p1[3], p0[2], p0[3], p1[4], p1[5], p0[4]]
    for q, r in seq:
        pos.advance(q, r)
    assert pos.is_terminal and pos.winner == 0
    with pytest.raises(ValueError, match="terminal"):
        from_position(pos)


def test_stone_features_are_mover_relative(positions):
    for pos in positions:
        g = from_position(pos)
        mover = pos.current_player
        owners = np.array([p for _q, _r, p in pos.stones()], dtype=np.int64)
        assert np.array_equal(g.stone_own, (owners != mover).astype(np.int64))
