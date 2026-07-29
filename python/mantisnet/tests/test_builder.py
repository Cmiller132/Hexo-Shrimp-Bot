"""§12.1 builder oracle, §12.2 ordering, §12.6 decoder coverage.

The oracle is the engine's own window walk (`windows_through`), which the
builder never calls — the two enumerate the same set from independent code.
"""

from __future__ import annotations

import hexo_py
import numpy as np
import pytest

from mantisnet import NUM_PATTERNS, collate, from_position
from mantisnet.builder import _CANON, _PATTERN_RANK, _SLOT_CLASS, AXES

from .conftest import oracle_live_windows


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
                    expected.add(((axis, sq, sr), cell, int(_SLOT_CLASS[k])))
        assert built == expected


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
                        expected.add((wid, int(_SLOT_CLASS[k])))
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
