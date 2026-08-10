"""The ACT containers: ordering, index bounds, offsets, and sentinels.

Every graph here is hand-built rather than produced by a builder, so the checks
are on the container contract alone: §7's orders, §25's packed names, §26's rule
that no edge crosses a position, and the ``-1`` sentinels that must survive
collation unshifted. Two positions of deliberately different sizes make an
offset that is applied to the wrong family, or not applied at all, visible.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from mantisnet.models.mantis_act.config import MantisACTConfig
from mantisnet.models.mantis_act.packed import (
    _refuse_crossing,
    PHASE_FIRST,
    PHASE_OPENING,
    PHASE_SECOND,
    POST_ACTION_ROWS,
    ACTGraph,
    PackedACTBatch,
    collate,
    telemetry,
)

FULL = MantisACTConfig()


def _int(*values) -> np.ndarray:
    return np.array(values, dtype=np.int64)


def graph_a(**overrides) -> ACTGraph:
    """Four cells, two windows, and two legal actions.

    Own stone at (0, 0), opponent at (1, 0); the two empty cells are legal. The
    windows are the Q line and the R line through the origin, each represented
    only in its first two slots so the ``-1`` slots are exercised.
    """
    fields = dict(
        cell_qr=np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64),
        cell_occupancy=_int(1, 0, 2, 0),
        cell_is_legal=_int(0, 1, 0, 1),
        cell_is_occupied=_int(1, 0, 1, 0),
        cell_nearest_bucket=_int(0, 0, 0, 1),
        # Engine order, which is not the cell order.
        legal_to_cell_index=_int(3, 1),
        window_id=np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int64),
        window_pattern_class=_int(12, 7),
        window_status=_int(3, 1),
        window_axis=_int(0, 1),
        window_numeric=np.zeros((2, 3), dtype=np.float32),
        window_cell_index=np.array(
            [[0, 2, -1, -1, -1, -1], [0, 1, -1, -1, -1, -1]], dtype=np.int64
        ),
        window_incidence_class=np.array(
            [[4, 9, -1, -1, -1, -1], [2, 5, -1, -1, -1, -1]], dtype=np.int64
        ),
        window_incidence_mask=np.array(
            [[True, True, False, False, False, False]] * 2, dtype=np.bool_
        ),
        adjacency_dst=_int(0, 0, 1, 2),
        adjacency_src=_int(1, 2, 0, 0),
        adjacency_axis=_int(1, 0, 1, 0),
        radius_dst=_int(1, 1, 3, 3),
        radius_src=_int(0, 2, 0, 2),
        radius_orbit=_int(0, 1, 2, 3),
        radius_axis_or_neg1=_int(1, -1, -1, 2),
        action_window_index=np.full((2, 3, 6), -1, dtype=np.int64),
        action_post1_class=np.arange(36, dtype=np.int64).reshape(2, 3, 6),
        action_pre_status=np.zeros((2, 3, 6), dtype=np.int64),
        action_tactical_numeric=np.zeros((2, 4), dtype=np.float32),
        global_numeric=np.zeros(5, dtype=np.float32),
        moves_remaining=2,
        phase_id=PHASE_FIRST,
    )
    fields["action_window_index"][0, 0, 0] = 0
    fields["action_window_index"][1, 1, 0] = 1
    fields["action_pre_status"][0, 0, 0] = 3
    fields["action_pre_status"][1, 1, 0] = 1
    fields.update(overrides)
    return ACTGraph(**fields)


def graph_b(**overrides) -> ACTGraph:
    """Three cells, one window, and two legal actions."""
    fields = dict(
        cell_qr=np.array([[0, 0], [0, 1], [2, 3]], dtype=np.int64),
        cell_occupancy=_int(2, 0, 0),
        cell_is_legal=_int(0, 1, 1),
        cell_is_occupied=_int(1, 0, 0),
        cell_nearest_bucket=_int(0, 1, 2),
        legal_to_cell_index=_int(1, 2),
        window_id=np.array([[2, 0, 0]], dtype=np.int64),
        window_pattern_class=_int(4),
        window_status=_int(2),
        window_axis=_int(2),
        window_numeric=np.zeros((1, 3), dtype=np.float32),
        window_cell_index=np.array([[0, -1, -1, -1, -1, -1]], dtype=np.int64),
        window_incidence_class=np.array([[3, -1, -1, -1, -1, -1]], dtype=np.int64),
        window_incidence_mask=np.array(
            [[True, False, False, False, False, False]], dtype=np.bool_
        ),
        adjacency_dst=_int(0, 1),
        adjacency_src=_int(1, 0),
        adjacency_axis=_int(1, 1),
        radius_dst=_int(1),
        radius_src=_int(0),
        radius_orbit=_int(5),
        radius_axis_or_neg1=_int(1),
        action_window_index=np.full((2, 3, 6), -1, dtype=np.int64),
        action_post1_class=np.arange(36, dtype=np.int64).reshape(2, 3, 6),
        action_pre_status=np.zeros((2, 3, 6), dtype=np.int64),
        action_tactical_numeric=np.zeros((2, 4), dtype=np.float32),
        global_numeric=np.zeros(5, dtype=np.float32),
        moves_remaining=1,
        phase_id=PHASE_SECOND,
    )
    fields["action_window_index"][0, 2, 3] = 0
    fields.update(overrides)
    return ACTGraph(**fields)


def replaced(array: np.ndarray, index, value) -> np.ndarray:
    """A copy of ``array`` with one entry changed, for the rejection cases."""
    out = array.copy()
    out[index] = value
    return out


def test_hand_built_graphs_are_valid():
    # Construction is the gate, so these two lines are the assertion: a fixture
    # that stopped satisfying §7 could not be built at all.
    graph_a()
    graph_b()


def test_counts_come_from_the_arrays():
    graph = graph_a()
    assert (graph.n_cells, graph.n_windows, graph.n_legal) == (4, 2, 2)
    assert (graph.n_adjacency, graph.n_radius) == (4, 4)
    assert graph.family_sizes() == {
        "cells": 4,
        "windows": 2,
        "legal": 2,
        "adjacency": 4,
        "radius": 4,
    }


# Each case breaks exactly one §7 order; the message must name the family.
@pytest.mark.parametrize(
    "overrides, message",
    [
        (
            {"cell_qr": np.array([[0, 1], [0, 0], [1, 0], [1, 1]], dtype=np.int64)},
            "cell nodes must be sorted",
        ),
        (
            {"cell_qr": np.array([[0, 0], [0, 0], [1, 0], [1, 1]], dtype=np.int64)},
            "cell nodes must be sorted",
        ),
        (
            {"window_id": np.array([[1, 0, 0], [0, 0, 0]], dtype=np.int64)},
            "persistent windows must be sorted",
        ),
        (
            {"window_id": np.array([[0, 0, 0], [0, 0, 0]], dtype=np.int64)},
            "persistent windows must be sorted",
        ),
        ({"adjacency_dst": _int(0, 1, 0, 2)}, "cell adjacency edges must be sorted"),
        ({"adjacency_src": _int(2, 1, 0, 0)}, "cell adjacency edges must be sorted"),
        (
            {"adjacency_src": _int(1, 1, 0, 0), "adjacency_axis": _int(1, 0, 1, 0)},
            "cell adjacency edges must be sorted",
        ),
        ({"radius_dst": _int(1, 1, 1, 3)}, "occupied radius edges must be sorted"),
        (
            {"radius_src": _int(0, 0, 2, 2), "radius_orbit": _int(1, 0, 2, 3)},
            "occupied radius edges must be sorted",
        ),
    ],
)
def test_validate_rejects_each_ordering_violation(overrides, message):
    with pytest.raises(ValueError, match=message):
        graph_a(**overrides)


# Each case puts one index out of the family it points into.
@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"legal_to_cell_index": _int(4, 1)}, r"legal_to_cell_index must be <= 3"),
        ({"adjacency_dst": _int(0, 0, 1, 4)}, r"adjacency_dst must be <= 3"),
        ({"adjacency_src": _int(1, 2, 0, 9)}, r"adjacency_src must be <= 3"),
        ({"radius_src": _int(0, 2, 0, 7)}, r"radius_src must be <= 3"),
        ({"radius_dst": _int(1, 1, 3, 4)}, r"radius_dst must be <= 3"),
        (
            {"window_cell_index": np.full((2, 6), 4, dtype=np.int64)},
            r"window_cell_index must be <= 3",
        ),
        (
            {"action_window_index": np.full((2, 3, 6), 2, dtype=np.int64)},
            r"action_window_index must be <= 1",
        ),
        (
            {"window_cell_index": np.full((2, 6), -2, dtype=np.int64)},
            r"window_cell_index must be >= -1",
        ),
        # A field with no sentinel gets zero for a floor, so `-1` is refused
        # here rather than surviving to be shifted into the previous position's
        # slice by `collate`.
        ({"adjacency_dst": _int(-1, 0, 1, 2)}, r"adjacency_dst must be >= 0"),
    ],
)
def test_validate_rejects_each_out_of_bounds_index(overrides, message):
    with pytest.raises(ValueError, match=message):
        graph_a(**overrides)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"cell_occupancy": _int(1, 0, 3, 0)}, r"cell_occupancy must be <= 2"),
        ({"cell_is_legal": _int(0, 2, 0, 1)}, r"cell_is_legal must be <= 1"),
        ({"cell_nearest_bucket": _int(0, -1, 0, 1)}, r"cell_nearest_bucket must be >= 0"),
        # The ceiling is the §8.2 bucket vocabulary, closed here rather than
        # left to whichever helper emitted the column: an unbounded value walks
        # straight into `CellEmbedding.nearest` as an unnamed IndexError on the
        # host and an uncatchable device-side assert on CUDA.
        (
            {"cell_nearest_bucket": _int(0, 9999, 0, 1)},
            r"cell_nearest_bucket must be <= 9",
        ),
        ({"window_status": _int(4, 1)}, r"window_status must be <= 3"),
        ({"window_axis": _int(3, 1)}, r"window_axis must be <= 2"),
        ({"adjacency_axis": _int(1, 0, 1, 3)}, r"adjacency_axis must be <= 2"),
        ({"radius_axis_or_neg1": _int(1, -2, -1, 2)}, r"radius_axis_or_neg1 must be >= -1"),
        # Class codes are bounded by the tables they index into.
        ({"window_pattern_class": _int(378, 7)}, r"window_pattern_class must be <= 377"),
        (
            {
                "action_post1_class": replaced(
                    np.arange(36, dtype=np.int64).reshape(2, 3, 6), (0, 0, 0), 729
                )
            },
            r"action_post1_class must be <= 728",
        ),
        (
            {
                "window_incidence_class": np.array(
                    [[4, 2187, -1, -1, -1, -1], [2, 5, -1, -1, -1, -1]], dtype=np.int64
                )
            },
            r"window_incidence_class must be <= 2186",
        ),
    ],
)
def test_validate_rejects_each_out_of_range_enum(overrides, message):
    with pytest.raises(ValueError, match=message):
        graph_a(**overrides)


def test_validate_rejects_a_wrong_dtype():
    with pytest.raises(TypeError, match=r"cell_occupancy must be int64, got int32"):
        graph_a(cell_occupancy=np.array([1, 0, 2, 0], dtype=np.int32))


def test_validate_rejects_a_wrong_shape():
    with pytest.raises(ValueError, match=r"window_cell_index must have shape \(2, 6\)"):
        graph_a(window_cell_index=np.zeros((2, 5), dtype=np.int64))


def test_validate_rejects_a_scalar_field_that_is_not_an_array():
    with pytest.raises(TypeError, match=r"global_numeric must be a numpy array"):
        graph_a(global_numeric=[0.0] * 5)


@pytest.mark.parametrize(
    "overrides, message",
    [
        (
            {"window_incidence_mask": np.zeros((2, 6), dtype=np.bool_)},
            r"window_incidence_mask disagrees with window_cell_index",
        ),
        (
            {"window_incidence_class": np.full((2, 6), -1, dtype=np.int64)},
            r"window_incidence_class is -1 at represented window",
        ),
        (
            {"cell_is_occupied": _int(1, 1, 1, 0)},
            r"cell_is_occupied disagrees with cell_occupancy",
        ),
        ({"cell_is_legal": _int(1, 1, 0, 1)}, r"cell 0 is both legal and occupied"),
        ({"cell_is_legal": _int(0, 1, 0, 0)}, r"cells are flagged legal but"),
        ({"legal_to_cell_index": _int(1, 1)}, r"must name each legal cell once"),
        ({"legal_to_cell_index": _int(0, 1)}, r"which is not flagged legal"),
        ({"moves_remaining": 3}, r"moves_remaining must be 1 or 2"),
        ({"phase_id": 7}, r"phase_id must be 0, 1, or 2"),
        ({"phase_id": PHASE_SECOND}, r"disagrees with moves_remaining"),
        # §15.2's edges run from stones. An empty source produces a relation of
        # `2 * orbit + 0` that is in range and means something the position does
        # not contain, which no shape, dtype, index bound, or round trip can
        # see: cell 1 is empty in graph_a.
        (
            {"radius_src": _int(1, 2, 0, 2)},
            r"radius edge 0 has source cell 1, which is empty",
        ),
    ],
)
def test_validate_rejects_each_internal_disagreement(overrides, message):
    with pytest.raises(ValueError, match=message):
        graph_a(**overrides)


def test_validate_rejects_an_opening_phase_with_stones():
    with pytest.raises(ValueError, match=r"OPENING phase with 1 occupied cells"):
        graph_b(phase_id=PHASE_OPENING)


def test_validate_rejects_a_second_phase_with_an_empty_board():
    with pytest.raises(ValueError, match=r"SECOND phase with an empty board"):
        graph_b(
            cell_occupancy=_int(0, 0, 0),
            cell_is_occupied=_int(0, 0, 0),
            cell_is_legal=_int(0, 1, 1),
            # No stone means no radius edge either: §15.2's sources are occupied
            # cells, which `_check_consistency` refuses to see empty.
            radius_dst=_int(),
            radius_src=_int(),
            radius_orbit=_int(),
            radius_axis_or_neg1=_int(),
        )


def test_collate_offsets_are_the_cumulative_counts():
    batch = collate([graph_a(), graph_b()], FULL)
    assert batch.position_count == 2
    assert batch.cell_offsets.tolist() == [0, 4, 7]
    assert batch.window_offsets.tolist() == [0, 2, 3]
    assert batch.legal_offsets.tolist() == [0, 2, 4]
    assert batch.adjacency_offsets.tolist() == [0, 4, 6]
    assert batch.radius_offsets.tolist() == [0, 4, 5]


def test_collate_shifts_every_index_into_the_global_frame():
    a, b = graph_a(), graph_b()
    batch = collate([a, b], FULL)
    cell_offset, window_offset = 4, 2

    assert batch.legal_to_cell_index.tolist() == [3, 1, 1 + cell_offset, 2 + cell_offset]
    assert batch.adjacency_dst.tolist() == [0, 0, 1, 2, 0 + cell_offset, 1 + cell_offset]
    assert batch.adjacency_src.tolist() == [1, 2, 0, 0, 1 + cell_offset, 0 + cell_offset]
    assert batch.radius_dst.tolist() == [1, 1, 3, 3, 1 + cell_offset]
    assert batch.radius_src.tolist() == [0, 2, 0, 2, 0 + cell_offset]
    assert batch.window_cell_index[2].tolist() == [0 + cell_offset, -1, -1, -1, -1, -1]
    assert int(batch.action_window_index[2, 2, 3]) == 0 + window_offset

    # A third position's indices shift by the running total, not by one
    # position's counts.
    three = collate([b, a, a], FULL)
    assert three.legal_to_cell_index.tolist() == [1, 2, 6, 4, 10, 8]


def test_collate_preserves_minus_one_sentinels():
    # B first, so A's sentinels sit behind a nonzero offset: an offset sentinel
    # would read as a real index into B's slice rather than as "no entity".
    batch = collate([graph_b(), graph_a()], FULL)
    a = graph_a()

    for name in ("window_cell_index", "action_window_index"):
        packed = getattr(batch, name).numpy()
        original = getattr(a, name)
        assert int((packed == -1).sum()) == int(
            (original == -1).sum()
        ) + int((getattr(graph_b(), name) == -1).sum())
        assert int(packed.min()) == -1

    # The exact rows: A's windows follow B's one window, and its empty slots
    # must be -1 rather than B's last cell or last window.
    assert batch.window_cell_index[1].tolist() == [3, 5, -1, -1, -1, -1]
    assert batch.window_cell_index[2].tolist() == [3, 4, -1, -1, -1, -1]
    assert int(batch.action_window_index[2, 0, 0]) == 1
    assert int(batch.action_window_index[2, 0, 1]) == -1


def test_no_index_crosses_a_batch_position():
    """Spec §30.18, re-derived from the packed offsets alone."""
    batch = collate([graph_a(), graph_b(), graph_a()], FULL)
    families = {
        "cells": batch.cell_offsets.numpy(),
        "windows": batch.window_offsets.numpy(),
        "legal": batch.legal_offsets.numpy(),
        "adjacency": batch.adjacency_offsets.numpy(),
        "radius": batch.radius_offsets.numpy(),
    }

    def owner(family, index):
        return np.searchsorted(families[family], index, side="right") - 1

    def rows(family):
        counts = np.diff(families[family])
        return np.repeat(np.arange(len(counts)), counts)

    checks = [
        ("legal_to_cell_index", "legal", "cells"),
        ("window_cell_index", "windows", "cells"),
        ("adjacency_src", "adjacency", "cells"),
        ("adjacency_dst", "adjacency", "cells"),
        ("radius_src", "radius", "cells"),
        ("radius_dst", "radius", "cells"),
        ("action_window_index", "legal", "windows"),
    ]
    for name, row_family, target_family in checks:
        values = getattr(batch, name).numpy().reshape(len(rows(row_family)), -1)
        live = values >= 0
        assert np.array_equal(
            owner(target_family, values)[live],
            np.broadcast_to(rows(row_family)[:, None], values.shape)[live],
        ), name


def test_the_cross_position_check_catches_a_shift_by_the_wrong_family():
    """What `_refuse_crossing` is for, now that no graph can arrive malformed.

    Every index is inside its own family's range before the shift — that is
    `_INDEX_FIELDS`, run at construction — so a *correct* shift by that
    family's own offsets can only land inside the row's own position. The fault
    left for this check is collation's own arithmetic: an offset taken from the
    wrong family. It applies and un-applies identically, every value stays in
    range for the batch, and only the row's own position says it is wrong.
    """
    a, b = graph_a(), graph_b()
    batch = collate([a, b], FULL)
    cells = batch.cell_offsets.numpy()
    windows = batch.window_offsets.numpy()
    assert cells[1] != windows[1], "the two families must offset differently"
    rows = np.repeat([0, 1], [a.n_adjacency, b.n_adjacency])

    wrong = np.concatenate([a.adjacency_dst, b.adjacency_dst + windows[1]])
    with pytest.raises(ValueError, match=r"adjacency_dst crosses a batch position"):
        _refuse_crossing("adjacency_dst", wrong, rows, cells)

    right = np.concatenate([a.adjacency_dst, b.adjacency_dst + cells[1]])
    _refuse_crossing("adjacency_dst", right, rows, cells)


def test_a_graph_cannot_reach_collate_unvalidated():
    """The contract `collate` rests on, tested where it is enforced.

    Collation re-checks no dtype, shape or value range, which is sound only
    because there is no way to hold an unvalidated ``ACTGraph``: the gate runs
    from ``__post_init__``, so the builder, a keyword construction and
    ``dataclasses.replace`` all pass it. Every ``test_validate_rejects_*``
    above is therefore a statement about construction, and which producer made
    a graph — something `collate` cannot see — stops mattering.
    """
    good = graph_a()
    with pytest.raises(ValueError, match=r"cell_is_legal must be <= 1"):
        replace(good, cell_is_legal=_int(0, 2, 0, 1))
    with pytest.raises(ValueError, match=r"window_cell_index must be >= -1"):
        replace(good, window_cell_index=np.full((2, 6), -2, dtype=np.int64))


def test_collate_refuses_an_empty_batch():
    with pytest.raises(ValueError, match=r"empty batch"):
        collate([], FULL)


def test_collate_refuses_disagreeing_feature_widths():
    wide = graph_b(window_numeric=np.zeros((1, 4), dtype=np.float32))
    with pytest.raises(ValueError, match=r"window_numeric has inconsistent feature widths"):
        collate([graph_a(), wide], FULL)


def test_collate_keeps_the_packed_dtypes():
    batch = collate([graph_a(), graph_b()], FULL)
    assert batch.cell_occupancy.dtype is torch.int64
    assert batch.window_incidence_mask.dtype is torch.bool
    assert batch.window_numeric.dtype is torch.float32
    assert batch.action_tactical_numeric.dtype is torch.float32
    assert batch.global_numeric.shape == (2, 5)
    assert batch.phase_id.tolist() == [PHASE_FIRST, PHASE_SECOND]
    assert batch.moves_remaining.tolist() == [2, 1]


def test_packed_batch_carries_every_spec_field():
    """The §25 names plus the CSR offsets, so downstream stages can bind them.

    ``window_id`` is the one name past §25's list: §16's typed window↔window
    edges are a join of the window identities, and joining them beside the model
    is what keeps the edges themselves off the bus.
    """
    present = set(vars(collate([graph_a()], FULL)))
    assert {
        "window_id",
        "position_count",
        "cell_offsets",
        "window_offsets",
        "legal_offsets",
        "adjacency_offsets",
        "radius_offsets",
        "cell_occupancy",
        "cell_is_legal",
        "cell_nearest_bucket",
        "legal_to_cell_index",
        "window_pattern_class",
        "window_status",
        "window_axis",
        "window_numeric",
        "window_cell_index",
        "window_incidence_class",
        "window_incidence_mask",
        "adjacency_src",
        "adjacency_dst",
        "adjacency_axis",
        "radius_src",
        "radius_dst",
        "radius_orbit",
        "radius_axis_or_neg1",
        "action_window_index",
        "action_post1_class",
        "action_pre_status",
        "action_tactical_numeric",
        "phase_id",
        "moves_remaining",
        "global_numeric",
        "radius_orbit_bound",
        "plans",
        "builder_fingerprint",
    } == present


def test_collate_records_the_batch_s_own_orbit_ceiling():
    """§11.2's vocabulary is a configuration choice, so the packer records it.

    `_VALUE_RANGES` leaves `radius_orbit` open above deliberately — the
    cardinality belongs to the module that emits it — and the model still has to
    refuse a batch built for a wider relation space than its own. Taking the
    ceiling here, in numpy, is what lets that comparison be between two
    host-side integers instead of a read back off the device.
    """
    batch = collate([graph_a(), graph_b()], FULL)
    assert int(batch.radius_orbit.max()) == 5
    assert batch.radius_orbit_bound == 6

    # A batch with no radius edge at all has no ceiling to state, and 0 is
    # below every vocabulary, so it refuses nothing.
    empty = graph_b(
        radius_dst=_int(), radius_src=_int(), radius_orbit=_int(),
        radius_axis_or_neg1=_int(),
    )
    assert collate([empty], FULL).radius_orbit_bound == 0


def test_batch_moves_to_a_device():
    batch = collate([graph_a(), graph_b()], FULL)
    moved = batch.to("cpu")
    assert isinstance(moved, PackedACTBatch)
    assert moved.position_count == 2
    assert torch.equal(moved.legal_to_cell_index, batch.legal_to_cell_index)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinning needs a CUDA host")
def test_batch_pins_every_tensor():
    pinned = collate([graph_a(), graph_b()], FULL).pin_memory()
    assert pinned.position_count == 2
    assert all(
        value.is_pinned()
        for value in vars(pinned).values()
        if isinstance(value, torch.Tensor)
    )


def test_telemetry_counts_every_budget_exactly():
    stats = telemetry(collate([graph_a(), graph_b()], FULL))
    assert stats == pytest.approx(
        {
            "positions": 2.0,
            "cells_mean": 3.5,
            "cells_max": 4.0,
            "windows_mean": 1.5,
            "windows_max": 2.0,
            "legal_actions_mean": 2.0,
            "legal_actions_max": 2.0,
            "window_incidences_mean": 2.5,
            "window_incidences_max": 4.0,
            "adjacency_edges_mean": 3.0,
            "adjacency_edges_max": 4.0,
            "radius_edges_mean": 2.5,
            "radius_edges_max": 4.0,
            "post_action_rows_mean": float(2 * POST_ACTION_ROWS),
            "post_action_rows_max": float(2 * POST_ACTION_ROWS),
            "windows_empty_mean": 0.0,
            "windows_empty_max": 0.0,
            "windows_own_live_mean": 0.5,
            "windows_own_live_max": 1.0,
            "windows_opp_live_mean": 0.5,
            "windows_opp_live_max": 1.0,
            "windows_mixed_mean": 0.5,
            "windows_mixed_max": 1.0,
        }
    )
