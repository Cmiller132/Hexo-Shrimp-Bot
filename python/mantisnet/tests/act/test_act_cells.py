"""§8/§10/§15 Rust cell-builder tests against naive Python oracles."""

from __future__ import annotations

from dataclasses import dataclass, replace

import hexo_py
import numpy as np

from mantisnet.models.mantis_act.builder import build
from mantisnet.models.mantis_act.cells import OCCUPANCY_OPP, OCCUPANCY_OWN
from mantisnet.models.mantis_act.config import PRESETS, MantisACTConfig
from mantisnet.models.mantis_act.packed import LEGAL_RADIUS, NEAREST_UNREACHED

FULL = PRESETS["full_act_v4"]
AXIS_STEPS = ((1, 0), (0, 1), (1, -1))
WINDOW_LEN = 6


def distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Hex distance through the cube-coordinate sum formula."""
    dq, dr = a[0] - b[0], a[1] - b[1]
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


def board(position: hexo_py.Position) -> dict[tuple[int, int], int]:
    mover = position.current_player
    return {
        (q, r): (OCCUPANCY_OWN if player == mover else OCCUPANCY_OPP)
        for q, r, player in position.stones()
    }


def arrays(position: hexo_py.Position) -> tuple[np.ndarray, np.ndarray]:
    stones = np.array([(q, r) for q, r, _ in position.stones()], dtype=np.int64)
    legal = np.array(position.legal_moves(), dtype=np.int64)
    return stones.reshape(-1, 2), legal.reshape(-1, 2)


@dataclass(frozen=True)
class Windows:
    window_id: np.ndarray
    code: np.ndarray


def naive_windows(position: hexo_py.Position) -> tuple[Windows, np.ndarray]:
    """Every nonempty window, independently walked from each stone."""
    occupancy = board(position)
    windows: dict[tuple[int, int, int], int] = {}
    for q, r in occupancy:
        for axis, (step_q, step_r) in enumerate(AXIS_STEPS):
            for slot in range(WINDOW_LEN):
                start = (q - slot * step_q, r - slot * step_r)
                windows[(axis, *start)] = sum(
                    occupancy.get((start[0] + k * step_q, start[1] + k * step_r), 0)
                    * 3**k
                    for k in range(WINDOW_LEN)
                )
    identities = sorted(windows)
    window_id = np.array(identities, dtype=np.int64).reshape(-1, 3)
    code = np.array([windows[identity] for identity in identities], dtype=np.int64)
    coords = np.array(
        [
            [
                (q + slot * AXIS_STEPS[axis][0], r + slot * AXIS_STEPS[axis][1])
                for slot in range(WINDOW_LEN)
            ]
            for axis, q, r in identities
        ],
        dtype=np.int64,
    ).reshape(-1, WINDOW_LEN, 2)
    return Windows(window_id, code), coords


def reverse_code(code: int) -> int:
    digits = [(code // 3**slot) % 3 for slot in range(WINDOW_LEN)]
    return sum(value * 3 ** (WINDOW_LEN - 1 - slot) for slot, value in enumerate(digits))


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
    return _JOINT_RANK[
        min((code, slot), (reverse_code(code), WINDOW_LEN - 1 - slot))
    ]


def naive_nearest(cell: tuple[int, int], stones: list[tuple[int, int]]) -> int:
    if not stones:
        return NEAREST_UNREACHED
    closest = min(distance(cell, stone) for stone in stones)
    return closest if closest <= LEGAL_RADIUS else NEAREST_UNREACHED


def coordinate_set(qr: np.ndarray) -> set[tuple[int, int]]:
    return {(int(q), int(r)) for q, r in np.asarray(qr).reshape(-1, 2)}


def graph(position: hexo_py.Position, cfg: MantisACTConfig = FULL):
    return build(position, cfg)


def test_window_and_legal_node_set_is_the_union_of_three_sources(positions):
    for position in positions:
        stone_qr, legal_qr = arrays(position)
        built = graph(position)
        _windows, window_coords = naive_windows(position)
        expected = (
            coordinate_set(stone_qr)
            | coordinate_set(legal_qr)
            | coordinate_set(window_coords)
        )
        assert coordinate_set(built.cell_qr) == expected


def test_narrower_scopes_drop_exactly_their_sources(positions):
    for position in positions:
        stone_qr, legal_qr = arrays(position)
        occupied = graph(position, replace(FULL, cell_scope="occupied_only"))
        both = graph(position, replace(FULL, cell_scope="occupied_and_legal"))
        assert coordinate_set(occupied.cell_qr) == coordinate_set(stone_qr)
        assert coordinate_set(both.cell_qr) == coordinate_set(stone_qr) | coordinate_set(
            legal_qr
        )
        assert not occupied.cell_is_legal.any()
        assert (occupied.legal_to_cell_index == -1).all()


def test_cells_are_sorted_lexicographically(positions):
    for position in positions:
        cells = graph(position).cell_qr
        assert [(int(q), int(r)) for q, r in cells] == sorted(coordinate_set(cells))


def test_occupancy_and_flags_match_the_board(positions):
    for position in positions:
        built = graph(position)
        occupancy = board(position)
        legal = {tuple(int(c) for c in move) for move in position.legal_moves()}
        for row, (q, r) in enumerate(built.cell_qr):
            cell = (int(q), int(r))
            assert int(built.cell_occupancy[row]) == occupancy.get(cell, 0)
            assert int(built.cell_is_occupied[row]) == int(cell in occupancy)
            assert int(built.cell_is_legal[row]) == int(cell in legal)
        assert int((built.cell_occupancy == OCCUPANCY_OWN).sum()) + int(
            (built.cell_occupancy == OCCUPANCY_OPP).sum()
        ) == len(occupancy)


def test_nearest_bucket_matches_a_brute_force_distance(positions):
    for position in positions:
        built = graph(position)
        stones = sorted(board(position))
        expected = np.array(
            [naive_nearest((int(q), int(r)), stones) for q, r in built.cell_qr],
            dtype=np.int64,
        )
        assert np.array_equal(built.cell_nearest_bucket, expected)
        assert not built.cell_nearest_bucket[built.cell_is_occupied == 1].any()
        if stones:
            assert (
                built.cell_nearest_bucket[built.cell_is_legal == 1] <= LEGAL_RADIUS
            ).all()


def test_empty_board_has_one_node_and_no_stone_to_measure_from():
    opening = hexo_py.Position()
    built = graph(opening)
    assert coordinate_set(built.cell_qr) == {
        tuple(int(c) for c in move) for move in opening.legal_moves()
    }
    assert (built.cell_nearest_bucket == NEAREST_UNREACHED).all()
    assert not built.cell_is_occupied.any()


def test_every_legal_move_maps_to_exactly_one_cell_node(positions):
    for position in positions:
        built = graph(position)
        index = built.legal_to_cell_index
        assert len(index) == position.legal_count
        assert len(np.unique(index)) == len(index)
        assert (index >= 0).all()
        assert [tuple(int(c) for c in row) for row in built.cell_qr[index]] == [
            tuple(int(c) for c in move) for move in position.legal_moves()
        ]
        assert int(built.cell_is_legal.sum()) == position.legal_count


def test_every_window_has_six_slots_with_valid_masks(positions):
    for position in positions:
        built = graph(position)
        windows, coords = naive_windows(position)
        assert np.array_equal(built.window_id, windows.window_id)
        n_windows = len(windows.window_id)
        assert built.window_cell_index.shape == (n_windows, WINDOW_LEN)
        assert built.window_incidence_class.shape == (n_windows, WINDOW_LEN)
        assert built.window_incidence_mask.shape == (n_windows, WINDOW_LEN)
        assert built.window_incidence_mask.all()
        assert np.array_equal(built.cell_qr[built.window_cell_index], coords)
        expected = np.array(
            [
                [joint_class(int(code), slot) for slot in range(WINDOW_LEN)]
                for code in windows.code
            ],
            dtype=np.int64,
        ).reshape(n_windows, WINDOW_LEN)
        assert np.array_equal(built.window_incidence_class, expected)


def test_occupied_only_masks_empty_window_slots(positions):
    cfg = replace(FULL, cell_scope="occupied_only")
    for position in positions:
        built = graph(position, cfg)
        windows, coords = naive_windows(position)
        occupancy = board(position)
        expected = np.array(
            [
                [(int(q), int(r)) in occupancy for q, r in window]
                for window in coords
            ],
            dtype=bool,
        ).reshape(len(windows.window_id), WINDOW_LEN)
        assert np.array_equal(built.window_incidence_mask, expected)
        assert (built.window_incidence_class[~expected] == -1).all()
        assert (built.window_incidence_class[expected] >= 0).all()


def axis_of(dq: int, dr: int) -> int:
    if dr == 0:
        return 0
    if dq == 0:
        return 1
    assert dq + dr == 0
    return 2


def test_adjacency_edges_match_a_brute_force_neighbour_search(positions):
    steps = ((1, 0), (0, 1), (1, -1), (-1, 0), (0, -1), (-1, 1))
    for position in positions:
        built = graph(position)
        coords = [tuple(int(c) for c in cell) for cell in built.cell_qr]
        rank = {cell: row for row, cell in enumerate(coords)}
        expected = {
            (row, rank[neighbour]): axis_of(*step)
            for row, cell in enumerate(coords)
            for step in steps
            if (neighbour := (cell[0] + step[0], cell[1] + step[1])) in rank
        }
        got = {
            (int(src), int(dst)): int(axis)
            for src, dst, axis in zip(
                built.adjacency_src, built.adjacency_dst, built.adjacency_axis
            )
        }
        assert got == expected


def test_edge_families_are_sorted_by_dst_src_relation(positions):
    for position in positions:
        built = graph(position)
        for src, dst, relation in (
            (built.adjacency_src, built.adjacency_dst, built.adjacency_axis),
            (built.radius_src, built.radius_dst, built.radius_orbit),
        ):
            key = list(zip(dst.tolist(), src.tolist(), relation.tolist()))
            assert key == sorted(key)


def test_radius_edges_match_a_brute_force_pairing(positions):
    for position in positions:
        built = graph(position)
        stone_qr, _legal = arrays(position)
        coords = [tuple(int(c) for c in cell) for cell in built.cell_qr]
        stones = [tuple(int(c) for c in stone) for stone in stone_qr]
        expected = {
            (stone, cell)
            for stone in stones
            for cell in coords
            if 1 <= distance(stone, cell) <= FULL.occupied_radius
        }
        got = {
            (coords[int(src)], coords[int(dst)])
            for src, dst in zip(built.radius_src, built.radius_dst)
        }
        assert got == expected
        assert built.cell_is_occupied[built.radius_src].all()
        for src, dst, axis in zip(
            built.radius_src, built.radius_dst, built.radius_axis_or_neg1
        ):
            dq = coords[int(dst)][0] - coords[int(src)][0]
            dr = coords[int(dst)][1] - coords[int(src)][1]
            colinear = [dr == 0, dq == 0, dq + dr == 0]
            assert int(axis) == (colinear.index(True) if any(colinear) else -1)
        assert (built.radius_orbit >= 0).all()


def test_radius_edges_stop_at_the_configured_radius(positions):
    cfg = PRESETS["full_radius6"]
    for position in positions:
        built = graph(position, cfg)
        coords = [tuple(int(c) for c in cell) for cell in built.cell_qr]
        assert all(
            1 <= distance(coords[int(src)], coords[int(dst)]) <= cfg.occupied_radius
            for src, dst in zip(built.radius_src, built.radius_dst)
        )


def test_coarse_relation_mode_replaces_the_orbit_ids(positions):
    cfg = PRESETS["full_coarse_geometry"]
    built = graph(positions[5], cfg)
    coords = [tuple(int(c) for c in cell) for cell in built.cell_qr]
    for src, dst, relation in zip(
        built.radius_src, built.radius_dst, built.radius_orbit
    ):
        source, target = coords[int(src)], coords[int(dst)]
        dq, dr = target[0] - source[0], target[1] - source[1]
        off_axis = not (dr == 0 or dq == 0 or dq + dr == 0)
        expected = 2 * (min(distance(source, target), cfg.d_max) - 1) + off_axis
        assert int(relation) == expected
