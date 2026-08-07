"""§30.11, §30.12, and the rest of the §8/§10/§15 cell builder.

Every oracle here is built from the game's rules rather than from the module
under test: window enumeration walks the 18 windows through each stone in
Python, distances use the cube form ``(|dq| + |dr| + |dq + dr|) / 2`` rather
than the module's maximum form, legality is re-derived as "empty and within
eight steps of a stone", and the joint incidence classes are ranked by sorting
representatives rather than by ``pattern_classes``' scan. The engine supplies
the positions and its own legal-move list, and is never told what the builder
thinks — making either side agree with the other by construction would delete
the detector.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import hexo_py
import numpy as np
import pytest

from mantisnet.models.mantis_act.cells import (
    OCCUPANCY_OPP,
    OCCUPANCY_OWN,
    CellSet,
    adjacency_edges,
    incidence,
    radius_edges,
    relevant_cells,
)
from mantisnet.models.mantis_act.config import PRESETS, MantisACTConfig
from mantisnet.models.mantis_act.packed import LEGAL_RADIUS, NEAREST_UNREACHED
from mantisnet.models.mantis_act.symmetry import (
    D6_TRANSFORMS,
    axis_permutation,
    transform_coords,
)
from mantisnet.models.mantis_act.windows import enumerate_windows

FULL = PRESETS["full_act_v4"]
AXIS_STEPS = ((1, 0), (0, 1), (1, -1))
WINDOW_LEN = 6


# --------------------------------------------------------------------------
# Oracles


def distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Hex distance by the cube metric, summed rather than maximised."""
    dq, dr = a[0] - b[0], a[1] - b[1]
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


def board(pos: hexo_py.Position) -> dict[tuple[int, int], int]:
    """Occupancy by coordinate, ``1`` the mover's stone and ``2`` the opponent's."""
    mover = pos.current_player
    return {(q, r): (1 if p == mover else 2) for q, r, p in pos.stones()}


def arrays(pos: hexo_py.Position) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The ``(stone_qr, stone_own, legal_qr)`` triple of the §11 input list."""
    stones = list(pos.stones())
    mover = pos.current_player
    stone_qr = np.array([(q, r) for q, r, _p in stones], dtype=np.int64).reshape(-1, 2)
    stone_own = np.array([int(p != mover) for _q, _r, p in stones], dtype=np.int64)
    legal_qr = np.array(pos.legal_moves(), dtype=np.int64).reshape(-1, 2)
    return stone_qr, stone_own, legal_qr


@dataclass(frozen=True)
class Windows:
    """What ``cells.incidence`` reads of a window set: identities and codes.

    Deliberately not ``windows.WindowSet``: the identities and codes here are
    enumerated below from the game's rules, so the incidence tables are checked
    against a window set the module under test had no hand in.
    """

    window_id: np.ndarray
    code: np.ndarray


def naive_windows(pos: hexo_py.Position) -> tuple[Windows, np.ndarray]:
    """Every nonempty window and its slot coordinates, enumerated in Python.

    Each stone offers 18 windows — three axes by six slots it could occupy —
    which are deduplicated by ``(native_axis, start_q, start_r)`` and sorted
    into the §7 window order. The ternary code reads the slots in ascending
    slot order relative to the mover, which is §9.2's encoding stated here from
    the game rules rather than imported.
    """
    occupancy = board(pos)
    windows: dict[tuple[int, int, int], int] = {}
    for (q, r), _value in occupancy.items():
        for axis, (sq, sr) in enumerate(AXIS_STEPS):
            for k in range(WINDOW_LEN):
                start = (q - k * sq, r - k * sr)
                code = sum(
                    occupancy.get((start[0] + j * sq, start[1] + j * sr), 0) * 3**j
                    for j in range(WINDOW_LEN)
                )
                windows[(axis, start[0], start[1])] = code
    identities = sorted(windows)
    window_id = np.array(identities, dtype=np.int64).reshape(-1, 3)
    window_code = np.array([windows[i] for i in identities], dtype=np.int64)
    coords = np.array(
        [
            [(sq + k * AXIS_STEPS[a][0], sr + k * AXIS_STEPS[a][1]) for k in range(WINDOW_LEN)]
            for a, sq, sr in identities
        ],
        dtype=np.int64,
    ).reshape(-1, WINDOW_LEN, 2)
    return Windows(window_id, window_code), coords


def reverse_code(code: int) -> int:
    """The ternary code of the same six slots read in the other direction."""
    digits = [(code // 3**k) % 3 for k in range(WINDOW_LEN)]
    return sum(v * 3 ** (WINDOW_LEN - 1 - k) for k, v in enumerate(digits))


_JOINT_RANK = {
    pair: rank
    for rank, pair in enumerate(
        sorted(
            {
                min((code, slot), (reverse_code(code), WINDOW_LEN - 1 - slot))
                for code in range(3**WINDOW_LEN)
                for slot in range(WINDOW_LEN)
            }
        )
    )
}


def joint_class(code: int, slot: int) -> int:
    """The §10.1 class of a ``(pattern, slot)`` pair, ranked from a sorted set."""
    return _JOINT_RANK[min((code, slot), (reverse_code(code), WINDOW_LEN - 1 - slot))]


def naive_nearest(cell: tuple[int, int], stones: list[tuple[int, int]]) -> int:
    """The §8.2 bucket: distance to the closest stone, or the unreached bucket."""
    if not stones:
        return NEAREST_UNREACHED
    closest = min(distance(cell, s) for s in stones)
    return closest if closest <= LEGAL_RADIUS else NEAREST_UNREACHED


def coordinate_set(qr: np.ndarray) -> set[tuple[int, int]]:
    return {(int(q), int(r)) for q, r in np.asarray(qr).reshape(-1, 2)}


def build(pos: hexo_py.Position, cfg: MantisACTConfig = FULL) -> tuple[CellSet, Windows, np.ndarray]:
    """The cell set of a position under ``cfg``, with its naive window set."""
    stone_qr, stone_own, legal_qr = arrays(pos)
    windows, window_coords = naive_windows(pos)
    cells = relevant_cells(stone_qr, stone_own, legal_qr, window_coords, cfg)
    return cells, windows, window_coords


# --------------------------------------------------------------------------
# §8.1 node sets and §8.2 fields


def test_window_and_legal_node_set_is_the_union_of_the_three_sources(positions):
    for pos in positions:
        stone_qr, _own, legal_qr = arrays(pos)
        cells, _windows, window_coords = build(pos)
        expected = (
            coordinate_set(stone_qr) | coordinate_set(legal_qr) | coordinate_set(window_coords)
        )
        assert coordinate_set(cells.qr) == expected
        assert len(cells) == len(expected)


def test_narrower_scopes_drop_exactly_their_sources(positions):
    for pos in positions:
        stone_qr, _own, legal_qr = arrays(pos)
        occupied = build(pos, replace(FULL, cell_scope="occupied_only"))[0]
        both = build(pos, replace(FULL, cell_scope="occupied_and_legal"))[0]
        assert coordinate_set(occupied.qr) == coordinate_set(stone_qr)
        assert coordinate_set(both.qr) == coordinate_set(stone_qr) | coordinate_set(legal_qr)
        # §29: under occupied_only the legal cells have no node at all, so
        # every legal action points at the sentinel.
        assert not occupied.is_legal.any()
        assert (occupied.legal_to_cell_index == -1).all()
        assert len(occupied.legal_to_cell_index) == len(legal_qr)


def test_cells_are_sorted_lexicographically(positions):
    for pos in positions:
        cells = build(pos)[0]
        assert sorted(coordinate_set(cells.qr)) == [
            (int(q), int(r)) for q, r in cells.qr
        ]


def test_occupancy_and_flags_match_the_board(positions):
    for pos in positions:
        cells = build(pos)[0]
        occupancy = board(pos)
        legal = coordinate_set(np.array(pos.legal_moves(), dtype=np.int64).reshape(-1, 2))
        for i, (q, r) in enumerate(cells.qr):
            cell = (int(q), int(r))
            assert int(cells.occupancy[i]) == occupancy.get(cell, 0)
            assert int(cells.is_occupied[i]) == int(cell in occupancy)
            assert int(cells.is_legal[i]) == int(cell in legal)
        own = cells.occupancy == OCCUPANCY_OWN
        opp = cells.occupancy == OCCUPANCY_OPP
        assert int(own.sum()) + int(opp.sum()) == len(occupancy)


def test_nearest_bucket_matches_a_brute_force_distance(positions):
    for pos in positions:
        cells = build(pos)[0]
        stones = sorted(board(pos))
        expected = np.array(
            [naive_nearest((int(q), int(r)), stones) for q, r in cells.qr], dtype=np.int64
        )
        assert np.array_equal(cells.nearest_bucket, expected)
        # Every occupied cell is its own nearest stone, and every legal cell is
        # reached: the legality rule and the bucket agree on real positions.
        assert not cells.nearest_bucket[cells.is_occupied == 1].any()
        if stones:
            assert (cells.nearest_bucket[cells.is_legal == 1] <= LEGAL_RADIUS).all()


def test_empty_board_has_one_node_and_no_stone_to_measure_from(positions):
    empty = hexo_py.Position()
    cells = build(empty)[0]
    assert coordinate_set(cells.qr) == coordinate_set(
        np.array(empty.legal_moves(), dtype=np.int64)
    )
    assert (cells.nearest_bucket == NEAREST_UNREACHED).all()
    assert not cells.is_occupied.any()


# --------------------------------------------------------------------------
# §30.12 legal mapping


def test_every_legal_move_maps_to_exactly_one_cell_node(positions):
    for pos in positions:
        legal = pos.legal_moves()
        cells = build(pos)[0]
        index = cells.legal_to_cell_index
        assert len(index) == len(legal)
        assert len(np.unique(index)) == len(index)
        assert (index >= 0).all()
        assert [(int(q), int(r)) for q, r in cells.qr[index]] == [
            (int(q), int(r)) for q, r in legal
        ]
        assert int(cells.is_legal.sum()) == len(legal)


# --------------------------------------------------------------------------
# §30.11 and §10 incidence


def test_every_window_has_six_slots_with_valid_masks(positions):
    for pos in positions:
        cells, windows, coords = build(pos)
        index, klass, mask = incidence(windows, cells)
        n = len(windows.window_id)
        assert index.shape == klass.shape == mask.shape == (n, WINDOW_LEN)
        # The default scope holds every slot of every persistent window (§10).
        assert mask.all()
        assert np.array_equal(mask, index >= 0)
        assert np.array_equal(cells.qr[index], coords)
        expected = np.array(
            [
                [joint_class(int(code), slot) for slot in range(WINDOW_LEN)]
                for code in windows.code
            ],
            dtype=np.int64,
        ).reshape(n, WINDOW_LEN)
        assert np.array_equal(klass, expected)


def test_occupied_only_masks_the_empty_window_slots(positions):
    for pos in positions:
        cfg = replace(FULL, cell_scope="occupied_only")
        cells, windows, coords = build(pos, cfg)
        index, klass, mask = incidence(windows, cells)
        occupancy = board(pos)
        expected = np.array(
            [[(int(q), int(r)) in occupancy for q, r in window] for window in coords],
            dtype=bool,
        ).reshape(len(windows.window_id), WINDOW_LEN)
        assert np.array_equal(mask, expected)
        assert (klass[~mask] == -1).all()
        assert (klass[mask] >= 0).all()
        if len(windows.window_id):
            # The masking is real, not vacuous: no window is all stones unless
            # the game is over, which these positions are not.
            assert not mask.all()


def test_incidence_reads_the_real_window_set(positions):
    """The seam between the two modules: same tables from the shipped enumerator.

    The oracle above is the detector; this is the check that ``incidence`` reads
    the fields ``windows.WindowSet`` actually carries, which no oracle-built
    stand-in can catch.
    """
    for pos in positions[1:]:
        stone_qr, stone_own, legal_qr = arrays(pos)
        shipped = enumerate_windows(stone_qr, stone_own, legal_qr, FULL)
        cells, naive, _coords = build(pos)
        assert np.array_equal(shipped.window_id, naive.window_id)
        assert np.array_equal(shipped.code, naive.code)
        assert all(
            np.array_equal(a, b)
            for a, b in zip(incidence(shipped, cells), incidence(naive, cells))
        )


def test_incidence_refuses_a_slot_the_default_scope_should_have_held(positions):
    pos = positions[-1]
    stone_qr, stone_own, _legal = arrays(pos)
    windows, _coords = naive_windows(pos)
    # A cell set built from the stones alone, but claiming the default scope,
    # is not this window set's cell set: every empty slot is missing.
    empty = np.empty((0, 2), dtype=np.int64)
    cells = relevant_cells(stone_qr, stone_own, empty, empty, FULL)
    with pytest.raises(ValueError, match="window_and_legal"):
        incidence(windows, cells)


def test_incidence_refuses_a_malformed_window_table(positions):
    cells, windows, _coords = build(positions[3])
    with pytest.raises(ValueError, match=r"window code must be 0\.\.728, got 729"):
        incidence(replace(windows, code=windows.code * 0 + 729), cells)
    bad_axis = windows.window_id.copy()
    bad_axis[0, 0] = 3
    with pytest.raises(ValueError, match="native axis 3"):
        incidence(replace(windows, window_id=bad_axis), cells)
    with pytest.raises(ValueError, match="window code has"):
        incidence(replace(windows, code=windows.code[:-1]), cells)


# --------------------------------------------------------------------------
# §15.1 adjacency


def axis_of(dq: int, dr: int) -> int:
    """The undirected axis a unit displacement runs along."""
    if dr == 0:
        return 0
    if dq == 0:
        return 1
    assert dq + dr == 0
    return 2


def test_adjacency_edges_match_a_brute_force_neighbour_search(positions):
    for pos in positions:
        cells = build(pos)[0]
        src, dst, axis = adjacency_edges(cells)
        coords = [(int(q), int(r)) for q, r in cells.qr]
        rank = {cell: i for i, cell in enumerate(coords)}
        expected = {
            (i, rank[neighbour]): axis_of(*step)
            for i, cell in enumerate(coords)
            for step in [(1, 0), (0, 1), (1, -1), (-1, 0), (0, -1), (-1, 1)]
            if (neighbour := (cell[0] + step[0], cell[1] + step[1])) in rank
        }
        got = {(int(s), int(d)): int(a) for s, d, a in zip(src, dst, axis)}
        assert got == expected
        # Every listed pair really is one step apart, by the cube metric.
        assert all(distance(coords[s], coords[d]) == 1 for s, d in got)


def test_edge_families_are_sorted_by_dst_src_relation(positions):
    for pos in positions:
        cells = build(pos)[0]
        for family in (
            adjacency_edges(cells),
            radius_edges(cells, *arrays(pos)[:2], FULL),
        ):
            src, dst, relation = family[0], family[1], family[2]
            key = list(zip(dst.tolist(), src.tolist(), relation.tolist()))
            assert key == sorted(key)


# --------------------------------------------------------------------------
# §15.2 radius edges


def test_radius_edges_match_a_brute_force_pairing(positions):
    for pos in positions:
        cells = build(pos)[0]
        stone_qr, stone_own, _legal = arrays(pos)
        src, dst, orbit, axis = radius_edges(cells, stone_qr, stone_own, FULL)
        coords = [(int(q), int(r)) for q, r in cells.qr]
        stones = [(int(q), int(r)) for q, r in stone_qr]
        expected = {
            (stone, cell)
            for stone in stones
            for cell in coords
            if 1 <= distance(stone, cell) <= FULL.occupied_radius
        }
        assert {(coords[s], coords[d]) for s, d in zip(src.tolist(), dst.tolist())} == expected
        assert len(src) == len(expected)
        assert (cells.is_occupied[src] == 1).all()
        # §11.3: the route is the axis the displacement lies on, or -1.
        for s, d, a in zip(src.tolist(), dst.tolist(), axis.tolist()):
            dq, dr = coords[d][0] - coords[s][0], coords[d][1] - coords[s][1]
            colinear = [dr == 0, dq == 0, dq + dr == 0]
            assert a == (colinear.index(True) if any(colinear) else -1)
        assert (orbit >= 0).all()


def test_radius_edges_stop_at_the_configured_radius(positions):
    cfg = PRESETS["full_radius6"]
    for pos in positions:
        cells = build(pos, cfg)[0]
        stone_qr, stone_own, _legal = arrays(pos)
        src, dst, _orbit, _axis = radius_edges(cells, stone_qr, stone_own, cfg)
        coords = [(int(q), int(r)) for q, r in cells.qr]
        assert all(
            1 <= distance(coords[s], coords[d]) <= cfg.occupied_radius
            for s, d in zip(src.tolist(), dst.tolist())
        )


def test_radius_edge_relations_are_d6_equivariant(positions):
    """The orbit id is invariant and the axis route permutes (§11.3, §15.3)."""
    pos = positions[6]
    stone_qr, stone_own, legal_qr = arrays(pos)
    windows, coords = naive_windows(pos)

    def edge_map(t: int) -> dict[tuple[tuple[int, int], tuple[int, int]], tuple[int, int]]:
        cells = relevant_cells(
            transform_coords(t, stone_qr),
            stone_own,
            transform_coords(t, legal_qr),
            transform_coords(t, coords),
            FULL,
        )
        src, dst, orbit, axis = radius_edges(
            cells, transform_coords(t, stone_qr), stone_own, FULL
        )
        cell = [(int(q), int(r)) for q, r in cells.qr]
        return {
            (cell[s], cell[d]): (int(o), int(a))
            for s, d, o, a in zip(src.tolist(), dst.tolist(), orbit.tolist(), axis.tolist())
        }

    base = edge_map(0)
    for t in range(1, len(D6_TRANSFORMS)):
        permutation = axis_permutation(t)
        image = edge_map(t)
        expected = {
            (D6_TRANSFORMS[t](s), D6_TRANSFORMS[t](d)): (
                orbit,
                permutation[axis] if axis >= 0 else -1,
            )
            for (s, d), (orbit, axis) in base.items()
        }
        assert image == expected


def test_coarse_relation_mode_replaces_the_orbit_ids(positions):
    cfg = PRESETS["full_coarse_geometry"]
    pos = positions[5]
    cells = build(pos, cfg)[0]
    stone_qr, stone_own, _legal = arrays(pos)
    src, dst, relation, _axis = radius_edges(cells, stone_qr, stone_own, cfg)
    coords = [(int(q), int(r)) for q, r in cells.qr]
    for s, d, r in zip(src.tolist(), dst.tolist(), relation.tolist()):
        dq, dr = coords[d][0] - coords[s][0], coords[d][1] - coords[s][1]
        off_axis = not (dr == 0 or dq == 0 or dq + dr == 0)
        assert r == 2 * (min(distance(coords[s], coords[d]), cfg.d_max) - 1) + off_axis


# --------------------------------------------------------------------------
# Loud refusals


class UncheckedScope:
    """A configuration naming a cell scope the dataclass would have refused."""

    def __init__(self, cell_scope: str):
        self.cell_scope = cell_scope


def test_relevant_cells_refuses_malformed_input(positions):
    pos = positions[4]
    stone_qr, stone_own, legal_qr = arrays(pos)
    window_coords = naive_windows(pos)[1]

    doubled = np.concatenate([stone_qr, stone_qr[:1]])
    with pytest.raises(ValueError, match="stone_qr lists"):
        relevant_cells(doubled, np.append(stone_own, 0), legal_qr, window_coords, FULL)
    with pytest.raises(ValueError, match="stone_own has 1 entries"):
        relevant_cells(stone_qr, stone_own[:1], legal_qr, window_coords, FULL)
    with pytest.raises(ValueError, match="stone_own must be 0"):
        relevant_cells(stone_qr, stone_own + 2, legal_qr, window_coords, FULL)
    with pytest.raises(ValueError, match="holds a stone"):
        relevant_cells(
            stone_qr,
            stone_own,
            np.concatenate([legal_qr, stone_qr[:1]]),
            window_coords,
            FULL,
        )
    far = np.array([[500, -250]], dtype=np.int64)
    with pytest.raises(ValueError, match="hex steps from every stone"):
        relevant_cells(stone_qr, stone_own, far, window_coords, FULL)
    with pytest.raises(ValueError, match="unknown cell_scope 'sometimes'"):
        relevant_cells(
            stone_qr, stone_own, legal_qr, window_coords, UncheckedScope("sometimes")
        )
    with pytest.raises(ValueError, match=r"must be \(\.\.\., 2\)"):
        relevant_cells(stone_qr[:, :1], stone_own, legal_qr, window_coords, FULL)


def test_radius_edges_refuse_a_stone_the_cell_set_does_not_hold(positions):
    pos = positions[6]
    stone_qr, stone_own, legal_qr = arrays(pos)
    window_coords = naive_windows(pos)[1]
    partial = relevant_cells(
        stone_qr[1:],
        stone_own[1:],
        legal_qr,
        window_coords,
        replace(FULL, cell_scope="occupied_only"),
    )
    with pytest.raises(ValueError, match="is not a cell node"):
        radius_edges(partial, stone_qr, stone_own, FULL)

    cells = relevant_cells(stone_qr, stone_own, legal_qr, window_coords, FULL)
    with pytest.raises(ValueError, match="against occupancy"):
        radius_edges(cells, stone_qr, 1 - stone_own, FULL)
    with pytest.raises(ValueError, match="stone_own has"):
        radius_edges(cells, stone_qr, stone_own[:-1], FULL)
