"""Collate-time ACT execution plans: contents, transport, and config binding."""

from __future__ import annotations

from dataclasses import replace
import warnings

import numpy as np
import pytest
import torch

from mantisnet.models.mantis_act.action_encoder import ActionEncoder
from mantisnet.models.mantis_act.actions import TACTICAL_FEATURES
from mantisnet.models.mantis_act.builder import GLOBAL_NUMERIC_FEATURES
from mantisnet.models.mantis_act.config import MantisACTConfig, PRESETS
from mantisnet.models.mantis_act.model import MantisACT
from mantisnet.models.mantis_act.packed import collate
from mantisnet.models.mantis_act.plans import (
    CLASS_REDUCTION_BLOCK_ROWS,
    ClassRowPlan,
    builder_fingerprint,
)
from mantisnet.models.mantis_act.state_trunk import StateTrunk
from mantisnet.models.mantis_act.windows import WINDOW_NUMERIC_FEATURES

from .test_act_packed import graph_a, graph_b

FULL = MantisACTConfig()

_SYNC_DEBUG_PROTOTYPE_NOTICE_PREFIX = (
    "Synchronization debug mode is a prototype feature and does not yet "
    "detect all synchronizing operations"
)


def _synchronisation_detections(
    caught: list[warnings.WarningMessage],
) -> list[str]:
    """Keep CUDA sync detections, excluding only debug mode's own notice."""
    messages = [str(warning.message) for warning in caught]
    return [
        message
        for message in messages
        if "synchron" in message.lower()
        and not message.startswith(_SYNC_DEBUG_PROTOTYPE_NOTICE_PREFIX)
    ]


@pytest.fixture()
def batch():
    return _batch_with_positions(2)


def _batch_with_positions(positions: int):
    # The generic packed-batch fixtures deliberately use small arbitrary
    # feature widths when testing transport alone. This fixture reaches the
    # default model, so make it obey all frozen builder feature schemas.
    templates = (
        graph_a(
            window_numeric=np.zeros(
                (2, WINDOW_NUMERIC_FEATURES), dtype=np.float32
            ),
            action_tactical_numeric=np.zeros(
                (2, TACTICAL_FEATURES), dtype=np.float32
            ),
            global_numeric=np.zeros(GLOBAL_NUMERIC_FEATURES, dtype=np.float32),
        ),
        graph_b(
            window_numeric=np.zeros(
                (1, WINDOW_NUMERIC_FEATURES), dtype=np.float32
            ),
            action_tactical_numeric=np.zeros(
                (2, TACTICAL_FEATURES), dtype=np.float32
            ),
            global_numeric=np.zeros(GLOBAL_NUMERIC_FEATURES, dtype=np.float32),
        ),
    )
    return collate(
        [templates[index % len(templates)] for index in range(positions)],
        FULL,
    )


def _is_dim0_dynamic(value: torch.Tensor) -> bool:
    return 0 in getattr(value, "_dynamo_dynamic_indices", set())


def _ptr(keys: list[int], size: int) -> list[int]:
    counts = [0] * size
    for key in keys:
        counts[key] += 1
    out = [0]
    for count in counts:
        out.append(out[-1] + count)
    return out


def _assert_message_views(edges, plan) -> None:
    if plan.channels == 1:
        src = edges.src.tolist()
        dst = edges.dst.tolist()
        relation = edges.relation.tolist()
        axis = None
    else:
        routed = edges.routed()
        src, dst, relation, axis = (value.tolist() for value in routed)

    rows = list(range(len(src)))
    dst_order = sorted(rows, key=lambda row: (dst[row], row))
    src_order = sorted(rows, key=lambda row: (src[row], row))
    rel_order = sorted(rows, key=lambda row: (relation[row], row))

    assert plan.dst_ptr.tolist() == _ptr([dst[row] for row in dst_order], edges.n_dst)
    assert plan.dst_src.tolist() == [src[row] for row in dst_order]
    assert plan.dst_rel.tolist() == [relation[row] for row in dst_order]
    assert plan.src_ptr.tolist() == _ptr([src[row] for row in src_order], edges.n_src)
    assert plan.src_dst.tolist() == [dst[row] for row in src_order]
    assert plan.src_rel.tolist() == [relation[row] for row in src_order]
    assert plan.rel_ptr.tolist() == _ptr(
        [relation[row] for row in rel_order], edges.num_relations
    )
    assert plan.rel_src.tolist() == [src[row] for row in rel_order]
    assert plan.rel_dst.tolist() == [dst[row] for row in rel_order]
    if axis is None:
        assert plan.dst_axis is plan.src_axis is plan.rel_axis is None
    else:
        assert plan.dst_axis.tolist() == [axis[row] for row in dst_order]
        assert plan.src_axis.tolist() == [axis[row] for row in src_order]
        assert plan.rel_axis.tolist() == [axis[row] for row in rel_order]


def test_full_default_carries_all_eight_stable_message_plans(batch):
    families = (
        batch.plans.state_edges.to_windows,
        batch.plans.state_edges.to_cells,
        batch.plans.state_edges.adjacency,
        batch.plans.state_edges.radius,
    )
    assert all(family is not None for family in families)
    plans = []
    for family in families:
        plans.extend((family.inv_plan, family.axis_plan))
        _assert_message_views(family, family.inv_plan)
        _assert_message_views(family, family.axis_plan)
    assert len(plans) == 8
    assert all(plan is not None for plan in plans)
    assert all(
        column.dtype is torch.int32
        for plan in plans
        for column in vars(plan).values()
        if isinstance(column, torch.Tensor)
    )


def test_radius_routed_subset_is_materialised_in_canonical_order(batch):
    radius = batch.plans.state_edges.radius
    expected_rows = [
        row for row, axis in enumerate(batch.radius_axis_or_neg1.tolist()) if axis >= 0
    ]
    assert radius.axis_rows.tolist() == expected_rows
    expected = tuple(
        values[expected_rows]
        for values in (
            batch.radius_src,
            batch.radius_dst,
            radius.relation,
            batch.radius_axis_or_neg1,
        )
    )
    for got, want in zip(radius.routed(), expected, strict=True):
        assert torch.equal(got, want)


def _assert_class_csr(values: torch.Tensor, plan) -> None:
    flat = values.reshape(-1).tolist()
    want_rows = []
    want_ptr = [0]
    classes = getattr(plan, "n_classes", getattr(plan, "n_sources", None))
    assert classes is not None
    for klass in range(classes):
        want_rows.extend(row for row, value in enumerate(flat) if value == klass)
        want_ptr.append(len(want_rows))
    assert plan.ptr.tolist() == want_ptr
    assert plan.rows.tolist() == want_rows
    if isinstance(plan, ClassRowPlan):
        want_block_ptr = [0]
        want_block_starts = []
        want_block_lengths = []
        for start, end in zip(want_ptr[:-1], want_ptr[1:], strict=True):
            for block_start in range(start, end, CLASS_REDUCTION_BLOCK_ROWS):
                want_block_starts.append(block_start)
                want_block_lengths.append(
                    min(CLASS_REDUCTION_BLOCK_ROWS, end - block_start)
                )
            want_block_ptr.append(len(want_block_starts))
        assert plan.block_ptr.tolist() == want_block_ptr
        assert plan.block_starts.tolist() == want_block_starts
        assert plan.block_lengths.tolist() == want_block_lengths


def test_action_class_and_source_window_csrs_are_stable_and_exact(batch):
    rows = batch.plans.action_rows
    _assert_class_csr(batch.action_post1_class, rows.post1)
    _assert_class_csr(batch.action_pre_status, rows.pre_status)

    flat = batch.action_window_index.reshape(-1).tolist()
    want_rows = []
    want_ptr = [0]
    for window in range(rows.source_window.n_windows):
        want_rows.extend(row for row, value in enumerate(flat) if value == window)
        want_ptr.append(len(want_rows))
    assert rows.source_window.ptr.tolist() == want_ptr
    assert rows.source_window.rows.tolist() == want_rows
    assert rows.source_window.sentinel_rows.tolist() == [
        row for row, value in enumerate(flat) if value == -1
    ]

    base = torch.where(
        batch.legal_to_cell_index >= 0,
        batch.legal_to_cell_index,
        torch.full_like(batch.legal_to_cell_index, batch.cell_occupancy.shape[0]),
    )
    _assert_class_csr(base, rows.base_cell)


def test_initial_embedding_csrs_are_stable_exact_and_match_table_heights(batch):
    rows = batch.plans.embedding_rows
    model = MantisACT(FULL)
    cases = (
        (
            batch.cell_occupancy,
            rows.cell_occupancy,
            model.trunk.cell_embedding.occupancy,
        ),
        (batch.cell_is_legal, rows.cell_legal, model.trunk.cell_embedding.legal),
        (
            batch.cell_nearest_bucket,
            rows.cell_nearest,
            model.trunk.cell_embedding.nearest,
        ),
        (
            batch.window_pattern_class,
            rows.window_pattern,
            model.trunk.window_embedding.pattern,
        ),
        (
            batch.window_status,
            rows.window_status,
            model.trunk.window_embedding.status,
        ),
    )
    for values, plan, table in cases:
        assert plan.n_classes == table.num_embeddings
        _assert_class_csr(values, plan)


def test_collate_and_transport_mark_chunk_rows_but_not_fixed_pointers_dynamic():
    batch = _batch_with_positions(3)

    def assert_annotations(candidate):
        state = candidate.plans.state_edges.to_windows
        action = candidate.plans.action_rows
        for tensor in (
            candidate.global_numeric,
            candidate.cell_offsets,
            candidate.cell_occupancy,
            state.src,
            state.inv_plan.dst_ptr,
            state.inv_plan.dst_src,
            state.inv_plan.rel_src,
            action.post1.rows,
            action.post1.block_starts,
            action.post1.block_lengths,
            action.source_window.ptr,
            action.base_cell.ptr,
            candidate.plans.embedding_rows.cell_occupancy.rows,
            candidate.plans.state_segments.counts,
            candidate.plans.state_segments.row_pos,
        ):
            assert tensor.shape[0] > 1
            assert _is_dim0_dynamic(tensor), tensor.shape

        # Vocabulary cardinalities are architecture constants, even though
        # these pointer arrays are longer than the dynamic-row cutoff.
        assert state.inv_plan.rel_ptr.shape[0] > 2
        assert action.post1.ptr.shape[0] > 2
        assert action.pre_status.ptr.shape[0] > 2
        assert action.post1.block_ptr.shape[0] > 2
        assert candidate.plans.embedding_rows.cell_nearest.ptr.shape[0] > 2
        assert not _is_dim0_dynamic(state.inv_plan.rel_ptr)
        assert not _is_dim0_dynamic(action.post1.ptr)
        assert not _is_dim0_dynamic(action.pre_status.ptr)
        assert not _is_dim0_dynamic(action.post1.block_ptr)
        assert not _is_dim0_dynamic(
            candidate.plans.embedding_rows.cell_nearest.ptr
        )

    assert_annotations(batch)
    assert_annotations(batch.to("cpu"))


def test_nested_plan_rows_share_one_fullgraph_dynamic_compile_across_chunks():
    # Start at two to guard the exact cardinality that the deleted per-stage
    # compilers used to specialize into accidental equality constraints.
    batches = (_batch_with_positions(2), _batch_with_positions(3))
    graphs = []

    def backend(graph, _example_inputs):
        graphs.append(graph)
        return graph.forward

    def plan_probe(candidate):
        state = candidate.plans.state_edges.to_windows.inv_plan
        action = candidate.plans.action_rows
        embeddings = candidate.plans.embedding_rows
        segments = candidate.plans.state_segments
        return (
            candidate.global_numeric.sum()
            + state.dst_ptr.float().sum()
            + state.dst_src.float().sum()
            + state.rel_ptr.float().sum()
            + action.post1.ptr.float().sum()
            + action.post1.rows.float().sum()
            + action.post1.block_ptr.float().sum()
            + action.post1.block_starts.float().sum()
            + action.post1.block_lengths.float().sum()
            + action.base_cell.ptr.float().sum()
            + action.base_cell.rows.float().sum()
            + embeddings.window_pattern.ptr.float().sum()
            + embeddings.window_pattern.rows.float().sum()
            + segments.counts.float().sum()
            + segments.row_pos.float().sum()
        )

    compiled = torch.compile(plan_probe, backend=backend, dynamic=True, fullgraph=True)
    for candidate in batches:
        torch.testing.assert_close(compiled(candidate), plan_probe(candidate))
    assert len(graphs) == 1


def _owners(offsets: torch.Tensor) -> list[int]:
    values = offsets.tolist()
    return [
        position
        for position, (start, end) in enumerate(
            zip(values[:-1], values[1:], strict=True)
        )
        for _ in range(start, end)
    ]


def test_row_positions_phases_and_latent_ranges_are_cpu_oracles(batch):
    plans = batch.plans
    cell_owner = _owners(batch.cell_offsets)
    window_owner = _owners(batch.window_offsets)
    legal_owner = _owners(batch.legal_offsets)
    assert plans.cell_row_pos.tolist() == cell_owner
    assert plans.window_row_pos.tolist() == window_owner
    assert plans.legal_row_pos.tolist() == legal_owner
    phases = batch.phase_id.tolist()
    assert plans.cell_phase.tolist() == [phases[row] for row in cell_owner]
    assert plans.window_phase.tolist() == [phases[row] for row in window_owner]
    assert plans.action_phase.tolist() == [phases[row] for row in legal_owner]

    cells = batch.cell_offsets.tolist()
    windows = batch.window_offsets.tolist()
    total_cells = cells[-1]
    expected = [
        [[cells[p], cells[p + 1]], [total_cells + windows[p], total_cells + windows[p + 1]]]
        for p in range(batch.position_count)
    ]
    assert plans.state_segments.ranges.tolist() == expected
    assert plans.state_segments.row_pos.tolist() == cell_owner + window_owner
    legal = batch.legal_offsets.tolist()
    assert plans.action_segments.ranges.tolist() == [
        [[legal[p], legal[p + 1]]] for p in range(batch.position_count)
    ]
    assert plans.action_segments.row_pos.tolist() == legal_owner


def test_plan_transport_is_recursive_and_fingerprint_is_exact(batch):
    moved = batch.to("cpu")
    assert moved.plans.device.type == "cpu"
    assert moved.plans.state_edges.radius.axis_plan.device.type == "cpu"
    assert moved.plans.action_rows.source_window.ptr.device.type == "cpu"
    assert moved.builder_fingerprint == builder_fingerprint(FULL)


def test_default_planned_fixture_obeys_the_builder_feature_schema(batch):
    assert batch.window_numeric.shape[1] == WINDOW_NUMERIC_FEATURES
    assert batch.action_tactical_numeric.shape[1] == TACTICAL_FEATURES
    assert batch.global_numeric.shape[1] == GLOBAL_NUMERIC_FEATURES
    with torch.no_grad():
        output = MantisACT(FULL).eval()(batch, None)
    assert output.policy_logits.shape == (batch.action_post1_class.shape[0],)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinning needs a CUDA host")
def test_plan_pinning_is_recursive(batch):
    pinned = batch.pin_memory()
    assert pinned.plans.state_edges.radius.inv_plan.dst_ptr.is_pinned()
    assert pinned.plans.state_edges.radius.routed_src.is_pinned()
    assert pinned.plans.action_rows.post1.rows.is_pinned()
    assert pinned.plans.action_rows.post1.block_ptr.is_pinned()
    assert pinned.plans.action_rows.post1.block_starts.is_pinned()
    assert pinned.plans.action_rows.post1.block_lengths.is_pinned()
    assert pinned.plans.action_rows.base_cell.ptr.is_pinned()
    assert pinned.plans.embedding_rows.window_pattern.rows.is_pinned()
    assert pinned.plans.state_segments.ranges.is_pinned()
    assert _is_dim0_dynamic(pinned.plans.state_edges.radius.inv_plan.dst_ptr)
    assert _is_dim0_dynamic(pinned.plans.action_rows.post1.rows)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="sync acceptance needs a CUDA device"
)
def test_planned_default_forward_has_no_device_to_host_synchronisation(batch):
    device_batch = batch.to("cuda")
    model = MantisACT(FULL).eval().cuda()

    # Compilation, allocator growth, and autotuning happen outside the measured
    # region; the repeated signature is the steady-state model forward.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(2):
            model(device_batch, None)
    torch.cuda.synchronize()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            with torch.no_grad(), torch.autocast(
                "cuda", dtype=torch.bfloat16
            ):
                model(device_batch, None)
        finally:
            torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()

    synchronising = _synchronisation_detections(caught)
    assert not synchronising


def test_sync_warning_filter_ignores_only_the_prototype_notice():
    actual_detection = "Synchronizing CUDA operation forced by Tensor.item()"
    unrelated = "this warning has no device event"
    prototype_notice = (
        f"{_SYNC_DEBUG_PROTOTYPE_NOTICE_PREFIX} (Triggered internally at "
        "/pytorch/torch/csrc/cuda/Module.cpp:1003.)"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn(prototype_notice, RuntimeWarning)
        warnings.warn(actual_detection, RuntimeWarning)
        warnings.warn(unrelated, RuntimeWarning)

    assert _synchronisation_detections(caught) == [actual_detection]


def test_class_csr_validation_rejects_a_nonpermutation(batch):
    plan = batch.plans.action_rows.pre_status
    broken_rows = plan.rows.clone()
    broken_rows[0] = broken_rows[1]
    with pytest.raises(ValueError, match="permutation"):
        replace(plan, rows=broken_rows)


def test_class_csr_validation_rejects_a_broken_block_partition(batch):
    plan = batch.plans.action_rows.pre_status
    broken_starts = plan.block_starts.clone()
    broken_starts[0] += 1
    with pytest.raises(ValueError, match="starts at"):
        replace(plan, block_starts=broken_starts)

    broken_lengths = plan.block_lengths.clone()
    broken_lengths[0] = CLASS_REDUCTION_BLOCK_ROWS + 1
    with pytest.raises(ValueError, match="block_lengths"):
        replace(plan, block_lengths=broken_lengths)


def test_collate_requires_the_builder_config():
    with pytest.raises(TypeError, match="cfg"):
        collate([graph_a()])


def test_every_model_entrypoint_refuses_a_plan_from_another_config(batch):
    other = PRESETS["full_no_axis"]
    expected = builder_fingerprint(other)
    assert batch.builder_fingerprint != expected
    with pytest.raises(ValueError, match="planned for builder config"):
        StateTrunk(other)(batch)
    with pytest.raises(ValueError, match="planned for builder config"):
        ActionEncoder(other)(batch, object())
    with pytest.raises(ValueError, match="planned for builder config"):
        MantisACT(other)(batch, None)


def test_every_model_entrypoint_refuses_a_missing_plan(batch):
    bare = replace(batch, plans=None)
    with pytest.raises(ValueError, match="plans is missing"):
        StateTrunk(FULL)(bare)
    with pytest.raises(ValueError, match="plans is missing"):
        ActionEncoder(FULL)(bare, object())
    with pytest.raises(ValueError, match="plans is missing"):
        MantisACT(FULL)(bare, None)
