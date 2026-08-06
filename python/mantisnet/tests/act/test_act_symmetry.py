"""§30.8-§30.10 and the orbit id ordering rule of `mantis_act.symmetry`.

The orbit count is checked twice, by two derivations that share nothing: the
module generates orbits by canonicalising every displacement, and Burnside's
lemma counts them from the fixed-point structure of the twelve transforms.
The hex metric is checked against a breadth-first search over the six unit
steps, which knows nothing of the closed form. Making either oracle read the
module's own tables would delete the detector rather than confirm it.
"""

from __future__ import annotations

import itertools

import hexo_py
import numpy as np
import pytest

from mantisnet.klent import telemetry
from mantisnet.models.mantis_act.symmetry import (
    AXES,
    D6_ORBITS_DMAX12,
    D6_TRANSFORMS,
    RELATION_FAR,
    RELATION_LATENT,
    RELATION_PAD,
    RELATION_SELF,
    axis_permutation,
    axis_reverses,
    coarse_relation,
    coarse_relation_count,
    hex_distance,
    on_axis,
    orbit_table,
    transform_coords,
)

D_MAX = 12


def shell(d_max: int = D_MAX) -> list[tuple[int, int]]:
    """Every displacement with hex distance 1..d_max, by the cube metric."""
    return [
        (q, r)
        for q in range(-d_max, d_max + 1)
        for r in range(-d_max, d_max + 1)
        if 1 <= (abs(q) + abs(r) + abs(q + r)) // 2 <= d_max
    ]


def orbit_of(displacement: tuple[int, int]) -> set[tuple[int, int]]:
    """The twelve images of a displacement, deduplicated."""
    return {t(displacement) for t in D6_TRANSFORMS}


# --------------------------------------------------------------------------
# The transform group


def test_transforms_are_twelve_distinct_symmetries_closed_under_composition():
    probe = [(1, 0), (0, 1), (1, -1), (3, -2)]
    images = [tuple(t(m) for m in probe) for t in D6_TRANSFORMS]
    assert len(D6_TRANSFORMS) == 12
    assert len(set(images)) == 12
    assert D6_TRANSFORMS[0]((3, -2)) == (3, -2)
    for f, g in itertools.product(D6_TRANSFORMS, repeat=2):
        assert tuple(f(g(m)) for m in probe) in set(images)


def test_transforms_preserve_the_hex_metric():
    for t in D6_TRANSFORMS:
        for q, r in shell(4):
            assert hex_distance(*t((q, r))) == hex_distance(q, r)


def test_hex_distance_matches_a_breadth_first_search():
    frontier = {(0, 0)}
    seen = {(0, 0): 0}
    for step in range(1, 5):
        frontier = {
            (q + int(dq), r + int(dr))
            for q, r in frontier
            for dq, dr in np.concatenate([AXES, -AXES])
        } - set(seen)
        seen.update(dict.fromkeys(frontier, step))
    for (q, r), distance in seen.items():
        assert int(hex_distance(q, r)) == distance


def test_matrix_transform_agrees_with_the_callable():
    displacement = np.array(shell(5), dtype=np.int64)
    for t, transform in enumerate(D6_TRANSFORMS):
        expected = np.array([transform(tuple(int(c) for c in d)) for d in displacement])
        assert np.array_equal(transform_coords(t, displacement), expected)
    # The shape contract holds through leading dimensions.
    stacked = displacement.reshape(-1, 3, 2)[:4]
    assert transform_coords(5, stacked).shape == stacked.shape


def test_transform_index_is_validated():
    with pytest.raises(ValueError, match="0..11, got 12"):
        transform_coords(12, np.zeros((1, 2), dtype=np.int64))
    with pytest.raises(ValueError, match="0..11, got -1"):
        axis_permutation(-1)
    with pytest.raises(ValueError, match=r"\(\.\.\., 2\)"):
        transform_coords(0, np.zeros((2, 3), dtype=np.int64))


def test_telemetry_shares_this_definition():
    # One definition of the group: the opening atlas applies these transforms,
    # it does not build its own.
    assert telemetry.D6_TRANSFORMS is D6_TRANSFORMS


def test_transforms_are_engine_symmetries(move_lists):
    # The engine is the authority on what a symmetry is: a transformed playout
    # must be legal move for move and reach an equivalent position.
    for moves in move_lists[:4]:
        base = hexo_py.Position.replay(moves)
        for transform in D6_TRANSFORMS:
            image = hexo_py.Position.replay([transform(m) for m in moves])
            assert image.legal_count == base.legal_count
            assert image.is_terminal == base.is_terminal
            assert {transform(m) for m in base.legal_moves()} == set(image.legal_moves())


# --------------------------------------------------------------------------
# §30.9 axis permutations


def test_every_transform_permutes_the_three_undirected_axes():
    for t in range(len(D6_TRANSFORMS)):
        permutation, reverses = axis_permutation(t), axis_reverses(t)
        assert sorted(permutation) == [0, 1, 2]
        for a in range(3):
            image = transform_coords(t, AXES[a])
            sign = -1 if reverses[a] else 1
            assert np.array_equal(image, sign * AXES[permutation[a]])


def test_axis_permutations_cover_s3_twice():
    # D6 acts on the three undirected axes through S3; the kernel is the
    # centre, so each of the six permutations is induced by exactly two
    # transforms and the point reflection reverses all three axes.
    counts: dict[tuple[int, int, int], int] = {}
    for t in range(len(D6_TRANSFORMS)):
        counts[axis_permutation(t)] = counts.get(axis_permutation(t), 0) + 1
    assert len(counts) == 6 and set(counts.values()) == {2}
    point_reflection = next(
        t for t in range(1, 12) if D6_TRANSFORMS[t]((1, 0)) == (-1, 0)
    )
    assert axis_permutation(point_reflection) == (0, 1, 2)
    assert axis_reverses(point_reflection) == (True, True, True)
    assert axis_reverses(0) == (False, False, False)


def test_axis_reversal_matches_window_slot_order():
    # A window is six cells from `start` along its axis, so the permutation and
    # the reversal flag together must say where every slot lands.
    start = np.array([2, -3], dtype=np.int64)
    for t in range(len(D6_TRANSFORMS)):
        permutation, reverses = axis_permutation(t), axis_reverses(t)
        for a in range(3):
            cells = start + np.arange(6)[:, None] * AXES[a]
            image = transform_coords(t, cells)
            step = image[1] - image[0]
            assert np.array_equal(step, (-1 if reverses[a] else 1) * AXES[permutation[a]])
            image_start = image[-1] if reverses[a] else image[0]
            expected = image_start + np.arange(6)[:, None] * AXES[permutation[a]]
            assert np.array_equal(np.sort(image, axis=0), np.sort(expected, axis=0))


def test_on_axis_names_the_axis_and_permutes_with_it():
    for a in range(3):
        for k in (1, -1, 5, -7):
            assert int(on_axis(*(k * AXES[a]))) == a
    assert int(on_axis(0, 0)) == -1
    assert int(on_axis(2, 1)) == -1
    displacement = np.array(shell(6), dtype=np.int64)
    base = on_axis(displacement[:, 0], displacement[:, 1])
    for t in range(len(D6_TRANSFORMS)):
        image = transform_coords(t, displacement)
        routed = on_axis(image[:, 0], image[:, 1])
        permutation = np.array(axis_permutation(t) + (-1,), dtype=np.int64)
        assert np.array_equal(routed, permutation[base])


# --------------------------------------------------------------------------
# §30.8 and §30.10 orbits


def test_there_are_forty_eight_orbits_through_radius_twelve():
    table = orbit_table(D_MAX)
    assert D6_ORBITS_DMAX12 == table.count == 48
    assert sorted(set(map(int, table.grid[table.grid >= 0]))) == list(range(48))
    assert int((table.grid >= 0).sum()) == len(shell()) == 3 * D_MAX * (D_MAX + 1)


def test_burnside_confirms_the_orbit_count_independently():
    cells = set(shell())
    fixed = [sum(1 for c in cells if t(c) == c) for t in D6_TRANSFORMS]
    # Identity fixes every cell, the five other rotations none, three axis
    # reflections a 24-cell line, three diagonal reflections a 12-cell one.
    assert sorted(fixed) == [0] * 5 + [12] * 3 + [24] * 3 + [len(cells)]
    assert sum(fixed) % len(D6_TRANSFORMS) == 0
    assert sum(fixed) // len(D6_TRANSFORMS) == D6_ORBITS_DMAX12


def test_orbit_id_is_invariant_under_every_transform():
    table = orbit_table(D_MAX)
    displacement = np.array(shell(), dtype=np.int64)
    base = table.lookup(displacement[:, 0], displacement[:, 1])
    for t in range(len(D6_TRANSFORMS)):
        image = transform_coords(t, displacement)
        assert np.array_equal(table.lookup(image[:, 0], image[:, 1]), base)


def test_orbits_are_exactly_the_transform_orbits():
    table = orbit_table(D_MAX)
    by_id: dict[int, set[tuple[int, int]]] = {}
    for displacement in shell():
        by_id.setdefault(int(table.lookup(*displacement)), set()).add(displacement)
    assert len(by_id) == 48
    for orbit_id, members in by_id.items():
        member = next(iter(members))
        assert members == orbit_of(member)
        assert tuple(int(c) for c in table.canonical[orbit_id]) == min(members)
        assert int(table.distance[orbit_id]) == int(hex_distance(*member))


def test_orbit_ids_are_ranked_by_distance_then_canonical():
    table = orbit_table(D_MAX)
    keys = [
        (int(d), int(q), int(r))
        for d, (q, r) in zip(table.distance, table.canonical.tolist())
    ]
    assert keys == sorted(keys)
    assert len(set(keys)) == table.count


def test_a_smaller_radius_is_a_prefix_of_the_radius_twelve_table():
    # Distance leads the sort, so radius 6 keeps every id it shares with
    # radius 12 — a radius ablation changes the edge set, not the relation
    # vocabulary.
    table, small = orbit_table(D_MAX), orbit_table(6)
    assert np.array_equal(small.canonical, table.canonical[: small.count])
    assert np.array_equal(small.distance, table.distance[: small.count])
    displacement = np.array(shell(6), dtype=np.int64)
    assert np.array_equal(
        small.lookup(displacement[:, 0], displacement[:, 1]),
        table.lookup(displacement[:, 0], displacement[:, 1]),
    )
    assert small.count < table.count


def test_lookup_refuses_displacements_outside_the_shell():
    table = orbit_table(D_MAX)
    with pytest.raises(ValueError, match=r"displacement \(0, 0\) has hex distance 0"):
        table.lookup(0, 0)
    with pytest.raises(ValueError, match=r"displacement \(13, 0\) has hex distance 13"):
        table.lookup(13, 0)
    with pytest.raises(ValueError, match=r"displacement \(7, 7\) has hex distance 14"):
        table.lookup(np.array([1, 7]), np.array([0, 7]))


def test_orbit_table_radius_is_bounded_by_the_reserved_relation_ids():
    assert (RELATION_FAR, RELATION_SELF, RELATION_LATENT, RELATION_PAD) == (48, 49, 50, 51)
    assert RELATION_FAR == D6_ORBITS_DMAX12
    with pytest.raises(ValueError, match="at most 12, got 13"):
        orbit_table(13)
    with pytest.raises(ValueError, match="at least 1, got 0"):
        orbit_table(0)


def test_table_arrays_are_read_only():
    # One instance is shared by every caller, so a mutation would be global.
    table = orbit_table(D_MAX)
    assert orbit_table(D_MAX) is table
    for array in (table.grid, table.canonical, table.distance, AXES):
        with pytest.raises(ValueError):
            array[0] = 0


# --------------------------------------------------------------------------
# The coarse ablation


def test_coarse_relation_is_invariant_and_coarser_than_the_orbits():
    displacement = np.array(shell(), dtype=np.int64)
    base = coarse_relation(displacement[:, 0], displacement[:, 1])
    for t in range(len(D6_TRANSFORMS)):
        image = transform_coords(t, displacement)
        assert np.array_equal(coarse_relation(image[:, 0], image[:, 1]), base)
    assert coarse_relation_count() == 2 * D_MAX
    assert base.min() >= 0 and base.max() < coarse_relation_count()
    # Every distance has an on-axis class; distance one has no off-axis cell,
    # so the scheme realizes 23 of its 24 classes and 48 orbits collapse onto
    # them.
    assert len(set(map(int, base))) == 23


def test_coarse_relation_buckets_distance_and_axis():
    assert int(coarse_relation(1, 0)) == 0
    assert int(coarse_relation(2, 1)) == 2 * 2 + 1
    assert int(coarse_relation(4, -4)) == 2 * 3
    # The last bucket clamps, the way the older distance buckets do.
    assert int(coarse_relation(20, 0, d_max=6)) == int(coarse_relation(6, 0, d_max=6))
    with pytest.raises(ValueError, match=r"\(0, 0\) has no coarse relation class"):
        coarse_relation(0, 0)
