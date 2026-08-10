"""Relation-gated messages: aggregation, reductions, routing, and guards.

Every input here is hand-built, so what is under test is the message algebra of
§14 and the three edge families of §15 rather than any builder's output. A
naive per-edge Python loop is the oracle for the aggregation: it is written
from the spec's four lines and shares no indexing with the module, so a wrong
gather, a wrong destination, or a dropped edge shows up as a disagreement
rather than as a plausible number.

Several tests read the aggregate back out through the public forward by
setting the destination update MLP to ``x -> relu(agg + s) - s`` — exact in the
region the inputs stay in, and the only way to see the quantity §14 defines
without a private seam. The shift is chosen per test so the arithmetic is
exact in the dtype the test runs in.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from mantisnet.models.mantis_act.config import MantisACTConfig
from mantisnet.models.mantis_act.equivariant import EquivariantState
from mantisnet.models.mantis_act.messages import (
    INCIDENCE_RELATIONS,
    AdjacencyMessage,
    CellWindowIncidence,
    RadiusMessage,
    RelationGatedMessage,
    TypedEdges,
    adjacency_edges,
    adjacency_relation_id,
    attention_by_destination,
    incidence_edges,
    radius_edges,
    radius_relation_count,
    relation_vocabulary_size,
    segment_softmax,
    segment_sum,
)
from mantisnet.models.mantis_act.plans import PlannedEdges
from mantisnet.models.mantis_act.segment_message import message_plan
from mantisnet.models.mantis_act.packed import NUM_AXES, PackedACTBatch
from mantisnet.models.mantis_act.symmetry import (
    D6_ORBITS_DMAX12,
    RELATION_PAD,
    coarse_relation_count,
)

D_INV, D_AXIS, D_REL = 8, 4, 6
N_CELLS, N_WINDOWS, N_LEGAL = 5, 2, 2


def config(**overrides) -> MantisACTConfig:
    """A small model of the default architecture, unless overridden."""
    fields = dict(d_inv=D_INV, d_axis=D_AXIS, d_rel=D_REL, num_heads=2)
    fields.update(overrides)
    return MantisACTConfig(**fields)


def _long(*values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.long)


def batch(**overrides) -> PackedACTBatch:
    """One position: five cells, two windows, and all three edge families.

    Cells 0 and 4 hold own stones, cell 1 an opponent stone, cells 2 and 3 are
    empty and legal. The two windows are represented in their first three and
    first two slots, so the ``-1`` slots the mask excludes are exercised. The
    radius edges deliberately mix on-axis and off-axis routes.
    """
    window_cell_index = torch.tensor(
        [[0, 1, 2, -1, -1, -1], [3, 4, -1, -1, -1, -1]], dtype=torch.long
    )
    fields = dict(
        position_count=1,
        cell_offsets=_long(0, N_CELLS),
        window_offsets=_long(0, N_WINDOWS),
        legal_offsets=_long(0, N_LEGAL),
        adjacency_offsets=_long(0, 4),
        radius_offsets=_long(0, 5),
        cell_occupancy=_long(1, 2, 0, 0, 1),
        cell_is_legal=_long(0, 0, 1, 1, 0),
        cell_nearest_bucket=_long(0, 0, 1, 2, 0),
        legal_to_cell_index=_long(2, 3),
        window_pattern_class=_long(12, 7),
        window_status=_long(3, 1),
        # The identities agree with `window_axis` in their leading column, which
        # is what `ACTGraph._check_consistency` holds a real graph to, and the
        # two lines cross so §16's join has both class families to find.
        window_id=torch.tensor([[0, 0, 0], [2, 2, 1]], dtype=torch.long),
        window_axis=_long(0, 2),
        window_numeric=torch.zeros(N_WINDOWS, 3),
        window_cell_index=window_cell_index,
        window_incidence_class=torch.tensor(
            [[5, 9, 11, -1, -1, -1], [2, 7, -1, -1, -1, -1]], dtype=torch.long
        ),
        window_incidence_mask=window_cell_index >= 0,
        # Both edge families are in §7's (dst, src, relation) order, which is
        # what `_check_ordering` fixes for every graph the packer collates and
        # what `dst_sorted=True` rests on.
        adjacency_src=_long(1, 0, 2, 1),
        adjacency_dst=_long(0, 1, 1, 2),
        adjacency_axis=_long(0, 0, 1, 1),
        radius_src=_long(4, 0, 1, 0, 4),
        radius_dst=_long(1, 2, 2, 3, 3),
        radius_orbit=_long(2, 0, 3, 1, 7),
        radius_axis_or_neg1=_long(1, 0, -1, -1, 2),
        action_window_index=torch.zeros(N_LEGAL, NUM_AXES, 6, dtype=torch.long),
        action_post1_class=torch.zeros(N_LEGAL, NUM_AXES, 6, dtype=torch.long),
        action_pre_status=torch.zeros(N_LEGAL, NUM_AXES, 6, dtype=torch.long),
        action_tactical_numeric=torch.zeros(N_LEGAL, 4),
        phase_id=_long(2),
        moves_remaining=_long(1),
        global_numeric=torch.zeros(1, 8),
        radius_orbit_bound=8,
    )
    fields.update(overrides)
    return PackedACTBatch(**fields)


def edge_set(name: str = "edges", **overrides) -> TypedEdges:
    """A four-edge family over five sources and three destinations.

    The two structural flags are measured off the hand-built columns unless a
    case states them, because here the columns *are* the fixture: production
    families take them from §7 and from the packer's bounds, and
    `test_the_structural_flags_describe_the_families_they_travel_with` is what
    holds those declarations to the data.
    """
    fields = dict(
        src=_long(0, 1, 1, 4),
        dst=_long(0, 0, 2, 2),
        relation=_long(3, 5, 3, 1),
        axis=_long(0, 2, -1, 1),
        n_src=N_CELLS,
        n_dst=3,
        num_relations=8,
        name=name,
    )
    fields.update(overrides)
    dst, axis = fields["dst"], fields["axis"]
    fields.setdefault("dst_sorted", bool((dst[1:] >= dst[:-1]).all()))
    fields.setdefault(
        "fully_routed", axis is None or bool((axis >= 0).all())
    )
    return TypedEdges(**fields)


def planned_edge_set() -> PlannedEdges:
    edges = edge_set()
    axis_rows = (edges.axis >= 0).nonzero(as_tuple=True)[0]
    routed = tuple(
        column.index_select(0, axis_rows)
        for column in (edges.src, edges.dst, edges.relation, edges.axis)
    )
    inv_plan = message_plan(
        edges.src,
        edges.dst,
        edges.relation,
        None,
        edges.n_src,
        edges.n_dst,
        edges.num_relations,
        1,
        dst_sorted=edges.dst_sorted,
    )
    axis_plan = message_plan(
        *routed,
        edges.n_src,
        edges.n_dst,
        edges.num_relations,
        NUM_AXES,
        dst_sorted=edges.dst_sorted,
    )
    return PlannedEdges(
        src=edges.src,
        dst=edges.dst,
        relation=edges.relation,
        axis=edges.axis,
        n_src=edges.n_src,
        n_dst=edges.n_dst,
        num_relations=edges.num_relations,
        dst_sorted=edges.dst_sorted,
        fully_routed=False,
        name=edges.name,
        axis_rows=axis_rows,
        routed_src=routed[0],
        routed_dst=routed[1],
        routed_relation=routed[2],
        routed_axis=routed[3],
        inv_plan=inv_plan,
        axis_plan=axis_plan,
    )


def state(rows: int, generator: torch.Generator, *, d_axis: int = D_AXIS):
    """A small random state for one entity family."""
    inv = torch.randn(rows, D_INV, generator=generator) * 0.5
    axis = (
        torch.randn(rows, NUM_AXES, d_axis, generator=generator) * 0.5
        if d_axis
        else None
    )
    return EquivariantState(inv, axis)


def states(n_src: int, n_dst: int, seed: int, *, d_axis: int = D_AXIS):
    """A source and a destination state, from one seeded generator."""
    generator = torch.Generator().manual_seed(seed)
    return (
        state(n_src, generator, d_axis=d_axis),
        state(n_dst, generator, d_axis=d_axis),
    )


# The shift that makes the destination update the identity on its aggregate.
# Large enough that no aggregate in these tests reaches -SHIFT and the ReLU
# clips, small enough to stay exact in bf16 for the accumulation test.
SHIFT = 4.0


def expose_aggregate(message: RelationGatedMessage) -> None:
    """Make ``forward`` return the §14 aggregate itself.

    ``update([LN(dst), agg]) = out(act(lin_in([LN(dst); agg])))``, and
    ``lin_in``'s matrix splits by its input into a destination half and an
    aggregate half. Zeroing the destination half, setting the bias to
    ``SHIFT``, the aggregate half and ``out``'s matrix to the identity, and
    ``out``'s bias to ``-SHIFT`` leaves ``act(agg + SHIFT) - SHIFT``, which is
    ``agg`` wherever ``agg > -SHIFT`` and the activation is the identity on its
    positive half. The model's own LayerScale is set to one for the same
    reason. Only ReLU is exactly the identity there, so a module built with
    another activation is refused rather than silently measured through a
    curve.
    """
    if not isinstance(message.update_inv.act, nn.ReLU):
        raise TypeError(
            "expose_aggregate needs activation='relu', not "
            f"{type(message.update_inv.act).__name__}"
        )
    with torch.no_grad():
        for update, width, scale in (
            (message.update_inv, message.d_inv, message.scale_inv),
            *(
                [(message.update_axis, message.d_axis, message.scale_axis)]
                if message.route_axis
                else []
            ),
        ):
            update.lin_in.weight[:, : update.d_a].zero_()
            update.lin_in.weight[:, update.d_a :].copy_(torch.eye(width))
            update.lin_in.bias.fill_(SHIFT)
            update.out.weight.copy_(torch.eye(width))
            update.out.bias.fill_(-SHIFT)
            scale.gamma.fill_(1.0)


def aggregates(message: RelationGatedMessage, edges: TypedEdges, source, destination):
    """The two aggregates, read off an ``expose_aggregate``-d module.

    The module returns the updated state, so the aggregate is what the update
    added to the destination it started from.
    """
    updated = message(edges, source, destination)
    axis = (
        None
        if updated.axis is None or destination.axis is None
        else updated.axis - destination.axis
    )
    return updated.inv - destination.inv, axis


def naive_state(message: RelationGatedMessage, edges: TypedEdges, source, destination):
    """The §14 message, aggregation, and update written as a per-edge loop.

    The oracle for :class:`RelationGatedMessage`. It shares no gather, no
    segment reduction, and no flattening with the module, so a wrong source
    row, a wrong destination, a dropped off-axis edge, or an axis message
    landing in the wrong channel all surface as a disagreement.
    """
    src_inv, src_axis = source.inv, source.axis
    dst_inv, dst_axis = destination.inv, destination.axis
    relation = message.relation.weight
    gate_inv = torch.sigmoid(message.wg_inv(relation)) if message.gated else None
    bias_inv = message.wb_inv(relation)
    value_inv = message.wv_inv(message.ln_src_inv(src_inv))

    aggregate_inv = torch.zeros(edges.n_dst, message.d_inv)
    aggregate_axis = torch.zeros(edges.n_dst, NUM_AXES, message.d_axis)
    if message.route_axis:
        gate_axis = torch.sigmoid(message.wg_axis(relation)) if message.gated else None
        bias_axis = message.wb_axis(relation)
        value_axis = message.wv_axis(message.ln_src_axis(src_axis))

    for e in range(len(edges)):
        s, d, r = int(edges.src[e]), int(edges.dst[e]), int(edges.relation[e])
        row = value_inv[s]
        if gate_inv is not None:
            row = row * gate_inv[r]
        aggregate_inv[d] += row + bias_inv[r]
        if not message.route_axis:
            continue
        a = int(edges.axis[e])
        if a < 0:
            continue
        row = value_axis[s, a]
        if gate_axis is not None:
            row = row * gate_axis[r]
        aggregate_axis[d, a] += row + bias_axis[r]

    inv = dst_inv + message.scale_inv(
        message.update_inv(message.ln_dst_inv(dst_inv), aggregate_inv)
    )
    if not message.route_axis:
        return EquivariantState(inv, dst_axis)
    axis = dst_axis + message.scale_axis(
        message.update_axis(message.ln_dst_axis(dst_axis), aggregate_axis)
    )
    return EquivariantState(inv, axis)


# --------------------------------------------------------------------------
# The relation vocabularies come from symmetry, not from a literal


def test_orbit_vocabulary_is_the_orbits_plus_the_reserved_band():
    assert relation_vocabulary_size(config()) == RELATION_PAD + 1
    assert relation_vocabulary_size(config()) == D6_ORBITS_DMAX12 + 4


def test_coarse_vocabulary_is_the_coarse_scheme_s_own():
    cfg = config(d6_relation_mode="coarse_distance_axis")
    assert relation_vocabulary_size(cfg) == coarse_relation_count(cfg.d_max)


def test_radius_vocabulary_joins_geometry_with_source_colour():
    cfg = config()
    assert radius_relation_count(cfg) == 2 * relation_vocabulary_size(cfg)


def test_adjacency_relation_is_the_distance_one_orbit():
    # Orbit ids are ranked by distance, so the one distance-one orbit is first.
    assert adjacency_relation_id(config()) == 0
    assert adjacency_relation_id(config(d6_relation_mode="coarse_distance_axis")) == 0


def test_incidence_vocabulary_is_the_single_joint_table():
    assert INCIDENCE_RELATIONS == 2187


# --------------------------------------------------------------------------
# The aggregation itself


def test_aggregation_matches_a_per_edge_loop():
    torch.manual_seed(0)
    message = RelationGatedMessage(config(), 8).eval()
    edges = edge_set()
    source, destination = states(edges.n_src, edges.n_dst, seed=1)

    got = message(edges, source, destination)
    want = naive_state(message, edges, source, destination)
    torch.testing.assert_close(got.inv, want.inv, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(got.axis, want.axis, atol=1e-5, rtol=1e-5)


def test_aggregation_matches_a_per_edge_loop_without_gates():
    torch.manual_seed(0)
    message = RelationGatedMessage(config(incidence_message="additive"), 8).eval()
    edges = edge_set()
    source, destination = states(edges.n_src, edges.n_dst, seed=2)

    got = message(edges, source, destination)
    want = naive_state(message, edges, source, destination)
    torch.testing.assert_close(got.inv, want.inv, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(got.axis, want.axis, atol=1e-5, rtol=1e-5)


def test_a_destination_with_no_edges_receives_no_message():
    torch.manual_seed(0)
    message = RelationGatedMessage(config(activation="relu"), 8).eval()
    expose_aggregate(message)
    edges = edge_set(dst=_long(0, 0, 0, 0))
    source, destination = states(edges.n_src, edges.n_dst, seed=3)

    aggregate_inv, aggregate_axis = aggregates(message, edges, source, destination)
    assert torch.all(aggregate_inv[1:] == 0.0)
    assert torch.all(aggregate_axis[1:] == 0.0)


def test_an_off_axis_edge_updates_the_invariant_stream_only():
    """§11.3: an edge on no axis has no channel to route through."""
    torch.manual_seed(0)
    message = RelationGatedMessage(config(activation="relu"), 8).eval()
    expose_aggregate(message)
    source, destination = states(N_CELLS, 3, seed=4)

    routed = edge_set(src=_long(0), dst=_long(1), relation=_long(2), axis=_long(1))
    off = edge_set(src=_long(0), dst=_long(1), relation=_long(2), axis=_long(-1))
    routed_inv, routed_axis = aggregates(message, routed, source, destination)
    off_inv, off_axis = aggregates(message, off, source, destination)

    torch.testing.assert_close(routed_inv, off_inv)
    assert torch.all(off_axis == 0.0)
    assert not torch.all(routed_axis[1, 1] == 0.0)


def test_axis_messages_land_in_the_routed_channel_only():
    torch.manual_seed(0)
    message = RelationGatedMessage(config(activation="relu"), 8).eval()
    expose_aggregate(message)
    source, destination = states(N_CELLS, 3, seed=5)
    edges = edge_set(src=_long(3), dst=_long(2), relation=_long(6), axis=_long(2))

    _aggregate_inv, aggregate_axis = aggregates(message, edges, source, destination)
    assert torch.all(aggregate_axis[:, :2] == 0.0)
    assert torch.all(aggregate_axis[:2] == 0.0)
    assert not torch.all(aggregate_axis[2, 2] == 0.0)


def test_axis_channels_permute_with_the_routes():
    """§12.1: relabelling the axes relabels the channels and nothing else.

    A model with per-axis weights, a fixed-order concatenation, or an absolute
    axis embedding fails this; axis-shared parameters pass it exactly.
    """
    torch.manual_seed(0)
    message = RelationGatedMessage(config(), 8).eval()
    edges = edge_set()
    source, destination = states(edges.n_src, edges.n_dst, seed=6)

    permutation = (1, 2, 0)  # axis a is carried to permutation[a]
    routes = _long(*permutation)
    moved_edges = edge_set(
        axis=torch.where(edges.axis >= 0, routes[edges.axis.clamp(min=0)], -1)
    )

    base = message(edges, source, destination)
    moved = message(
        moved_edges,
        source.permute_axes(permutation),
        destination.permute_axes(permutation),
    )
    expected = base.permute_axes(permutation)
    torch.testing.assert_close(moved.inv, expected.inv)
    torch.testing.assert_close(moved.axis, expected.axis)


# --------------------------------------------------------------------------
# §14's attention reduction — the one reduction with an explicit per-edge tensor


def test_an_empty_destination_attends_to_nothing_and_reads_zero():
    messages = torch.arange(6.0).reshape(2, 3)
    index = _long(1, 1)
    out = attention_by_destination(messages, index, 3, torch.tensor([0.5, -0.5]))
    assert torch.all(out[0] == 0.0)
    assert torch.all(out[2] == 0.0)


def test_uniform_attention_is_the_mean():
    """Equal scores give equal weights, so the read is each segment's mean."""
    messages = torch.randn(6, 4, generator=torch.Generator().manual_seed(7))
    index = _long(0, 0, 0, 1, 1, 2)
    attention = attention_by_destination(messages, index, 3, torch.zeros(6))
    torch.testing.assert_close(attention[0], messages[:3].mean(dim=0))
    torch.testing.assert_close(attention[1], messages[3:5].mean(dim=0))
    torch.testing.assert_close(attention[2], messages[5])


def test_attention_weights_normalise_within_a_destination():
    score = torch.tensor([1.0, -2.0, 0.5, 3.0])
    index = _long(0, 0, 2, 2)
    weights = segment_softmax(score, index, 3)
    assert weights.dtype == torch.float32
    torch.testing.assert_close(weights[:2].sum(), torch.tensor(1.0))
    torch.testing.assert_close(weights[2:].sum(), torch.tensor(1.0))


def test_the_three_reductions_disagree_at_the_module():
    """The same weights under each mode, so only the reduction differs."""
    torch.manual_seed(0)
    reference = RelationGatedMessage(config(activation="relu"), 8).eval()
    edges = edge_set(dst=_long(0, 0, 0, 1))
    source, destination = states(edges.n_src, edges.n_dst, seed=8)

    outputs = {}
    for reduce in ("sum", "mean", "attention"):
        message = RelationGatedMessage(
            config(activation="relu", incidence_reduce=reduce), 8
        ).eval()
        missing, unexpected = message.load_state_dict(
            reference.state_dict(), strict=False
        )
        assert not unexpected
        assert all(name.startswith("score_") for name in missing)
        if reduce == "attention":
            with torch.no_grad():
                message.score_inv.copy_(torch.linspace(-1.0, 1.0, D_INV))
                message.score_axis.copy_(torch.linspace(1.0, -1.0, D_AXIS))
        expose_aggregate(message)
        outputs[reduce] = aggregates(message, edges, source, destination)[0]

    # Destination 1 owns a single edge, so every mode agrees there and only
    # the three-edge destination separates them.
    torch.testing.assert_close(outputs["sum"][1], outputs["mean"][1])
    torch.testing.assert_close(outputs["sum"][1], outputs["attention"][1])
    torch.testing.assert_close(
        outputs["mean"][0], outputs["sum"][0] / 3.0, atol=1e-5, rtol=1e-5
    )
    assert not torch.allclose(outputs["attention"][0], outputs["mean"][0])


# --------------------------------------------------------------------------
# The additive control of §29 `full_additive_incidence`


def test_the_additive_control_holds_no_gate_parameters():
    message = RelationGatedMessage(config(incidence_message="additive"), 8)
    assert not message.gated
    assert not hasattr(message, "wg_inv")
    assert not hasattr(message, "wg_axis")


def test_the_additive_control_is_genuinely_additive():
    """``U @ src + E_relation``: the relation term does not depend on the source.

    Under the additive form the difference between two relations is the same
    whatever the source is; under the gated form the gate multiplies the
    source's value, so it is not.
    """
    torch.manual_seed(0)
    source, destination = states(N_CELLS, 4, seed=9)

    def relation_gaps(cfg):
        message = RelationGatedMessage(cfg, 8).eval()
        expose_aggregate(message)
        # Four single-edge destinations: two sources crossed with two relations.
        edges = edge_set(
            src=_long(0, 0, 3, 3),
            dst=_long(0, 1, 2, 3),
            relation=_long(2, 6, 2, 6),
            axis=_long(0, 0, 0, 0),
            n_dst=4,
        )
        aggregate = aggregates(message, edges, source, destination)[0]
        return aggregate[0] - aggregate[1], aggregate[2] - aggregate[3]

    first, second = relation_gaps(
        config(activation="relu", incidence_message="additive")
    )
    torch.testing.assert_close(first, second, atol=1e-5, rtol=1e-5)

    first, second = relation_gaps(config(activation="relu"))
    assert not torch.allclose(first, second, atol=1e-4)


# --------------------------------------------------------------------------
# Numerics (§27)


def test_segment_sum_accumulates_in_fp32_from_bf16_rows():
    values = torch.full((512, 2), 1.0 / 256.0, dtype=torch.bfloat16)
    total = segment_sum(values, torch.zeros(512, dtype=torch.long), 1)
    assert total.dtype == torch.float32
    torch.testing.assert_close(total, torch.full((1, 2), 2.0))


def test_aggregation_stays_fp32_under_bf16_autocast():
    """A bf16 running sum stalls where the fp32 one does not.

    Every message here is exactly ``1/256`` and 512 of them reach one
    destination. In fp32 that is 2.0; accumulated in bf16 it stops near 1.0,
    because from 1.0 upward the spacing is ``1/128`` and each addition rounds
    away. The parameters are set so the message value is exact in bf16 and only
    the accumulation is under test.
    """
    message = RelationGatedMessage(config(activation="relu"), 8).eval()
    with torch.no_grad():
        message.ln_src_inv.weight.fill_(1.0)
        message.ln_src_inv.bias.zero_()
        message.wv_inv.weight.copy_(torch.eye(D_INV) / 128.0)
        message.wg_inv.weight.zero_()  # sigmoid(0) = 1/2, so the message is 1/256
        message.wg_inv.bias.zero_()
        message.wb_inv.weight.zero_()
        message.wb_inv.bias.zero_()
    expose_aggregate(message)

    edges = edge_set(
        src=torch.zeros(512, dtype=torch.long),
        dst=torch.zeros(512, dtype=torch.long),
        relation=torch.zeros(512, dtype=torch.long),
        axis=torch.zeros(512, dtype=torch.long),
        n_dst=1,
    )
    # LayerNorm of an alternating row is exactly +-1, so the value is +-1/128.
    source = EquivariantState(
        torch.tensor([1.0, -1.0] * (D_INV // 2)).expand(N_CELLS, D_INV).contiguous(),
        torch.zeros(N_CELLS, NUM_AXES, D_AXIS),
    )
    destination = EquivariantState(
        torch.zeros(1, D_INV), torch.zeros(1, NUM_AXES, D_AXIS)
    )

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        aggregate = aggregates(message, edges, source, destination)[0]
    assert aggregate.dtype == torch.float32
    expected = torch.tensor([2.0, -2.0] * (D_INV // 2)).unsqueeze(0)
    torch.testing.assert_close(aggregate, expected, atol=0.05, rtol=0.0)


def test_the_forward_runs_under_bf16_autocast_and_stays_finite():
    torch.manual_seed(0)
    message = RelationGatedMessage(config(), 8).eval()
    edges = edge_set()
    source, destination = states(edges.n_src, edges.n_dst, seed=10)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        updated = message(edges, source, destination)
    # The residual stream keeps the state's dtype, not autocast's (§27).
    assert updated.inv.dtype == torch.float32 and updated.axis.dtype == torch.float32
    assert torch.isfinite(updated.inv).all() and torch.isfinite(updated.axis).all()


def test_layer_scale_and_relation_table_take_their_spec_init():
    cfg = config()
    message = RelationGatedMessage(cfg, 16)
    assert torch.all(message.scale_inv.gamma == cfg.layer_scale_init)
    assert torch.all(message.scale_axis.gamma == cfg.layer_scale_init)
    assert message.relation.weight.std().item() == pytest.approx(0.02, abs=0.01)


# --------------------------------------------------------------------------
# What this module leans on instead of re-reading its own inputs
#
# Every column of every family below is bounded by the packer, on the host, in
# numpy, before a tensor exists: `ACTGraph`'s `_VALUE_RANGES` and
# `_INDEX_FIELDS`, run from `__post_init__` so no producer can skip them, and
# `collate`'s `_refuse_crossing`, which checks collation's own offset
# arithmetic. `tests/act/test_act_packed.py` is where each of
# those refusals is exercised. What is left to test here is what this module
# still owns: the host-side structure of an edge family, and the two structural
# flags it now carries instead of measuring its data.


def test_the_structural_flags_describe_the_families_they_travel_with():
    """The declarations are held to the data they claim to describe.

    `dst_sorted` and `fully_routed` are host-side booleans set at the
    construction site from §7 and from the packer's bounds, so nothing in a
    forward reads the columns back. A wrong flag would make the segment
    reduction walk the wrong runs, silently — this is the statement that keeps
    them honest, and it measures each column independently of the builder that
    set the flag.
    """
    to_windows, to_cells = incidence_edges(batch())
    adjacency = adjacency_edges(batch(), config())
    radius = radius_edges(batch(), config())
    expected = {
        "incidence cells->windows": (to_windows, True, True),
        "incidence windows->cells": (to_cells, False, True),
        "hex adjacency": (adjacency, True, True),
        "occupied radius": (radius, True, False),
    }
    for label, (family, dst_sorted, fully_routed) in expected.items():
        assert family.dst_sorted is dst_sorted, label
        assert family.fully_routed is fully_routed, label
        # Only the True direction is a claim about the data. `False` says
        # "sort it" and "take the routed subset", which are correct whatever
        # the column holds; `True` says the work can be skipped, and that is
        # what has to be true.
        if dst_sorted:
            assert bool((family.dst[1:] >= family.dst[:-1]).all()), label
        if fully_routed:
            assert bool((family.axis >= 0).all()), label
        # An unrouted family takes the subset; a routed one has none to take.
        assert (family.axis_rows is None) is fully_routed, label
    # The radius fixture really does mix routed and unrouted rows, so the one
    # `fully_routed=False` above is not vacuous.
    assert not bool((radius.axis >= 0).all())


def test_a_structural_flag_must_be_a_host_side_bool():
    """A device tensor here would be a sync in a forward and a silent truth."""
    with pytest.raises(TypeError, match=r"probe\.dst_sorted must be a host-side bool"):
        edge_set("probe", dst_sorted=torch.tensor(True))
    with pytest.raises(TypeError, match=r"probe\.fully_routed must be a host-side bool"):
        edge_set("probe", fully_routed=1)


def test_a_family_with_no_axis_column_cannot_claim_an_unrouted_subset():
    with pytest.raises(ValueError, match="carries no axis route at all"):
        edge_set("probe", axis=None, fully_routed=False)


def test_a_batch_from_another_relation_space_is_refused_by_its_recorded_ceiling():
    """§11.2's vocabulary is the one index space a configuration resizes.

    `packed._VALUE_RANGES` therefore leaves `radius_orbit` open above and the
    packer records the batch's own ceiling; this is the comparison that refuses
    a batch built for one `d6_relation_mode` or `d_max` to a model built for
    another. Both sides are host-side integers, so the model never reads the
    orbit column back off the device to find out.
    """
    coarse = config(
        d6_relation_mode="coarse_distance_axis", d_max=4, occupied_radius=4
    )
    assert relation_vocabulary_size(coarse) == 8
    # A batch whose orbits reach 47 is an orbit48 batch, not a coarse one.
    with pytest.raises(ValueError, match="different §11.2 relation spaces"):
        radius_edges(batch(radius_orbit_bound=48), coarse)
    # Exactly filling the vocabulary is not a violation.
    assert radius_edges(batch(radius_orbit_bound=8), coarse).num_relations == 16


# --------------------------------------------------------------------------
# The three concrete paths


def test_incidence_edges_follow_the_mask_and_the_window_s_own_axis():
    to_windows, to_cells = incidence_edges(batch())
    assert len(to_windows) == 5  # three slots of window 0, two of window 1
    torch.testing.assert_close(to_windows.src, _long(0, 1, 2, 3, 4))
    torch.testing.assert_close(to_windows.dst, _long(0, 0, 0, 1, 1))
    torch.testing.assert_close(to_windows.relation, _long(5, 9, 11, 2, 7))
    # §12.3: a line message routes into its own line's axis.
    torch.testing.assert_close(to_windows.axis, _long(0, 0, 0, 2, 2))

    # The other direction is the same edges read backwards.
    torch.testing.assert_close(to_cells.src, to_windows.dst)
    torch.testing.assert_close(to_cells.dst, to_windows.src)
    assert (to_windows.n_src, to_windows.n_dst) == (N_CELLS, N_WINDOWS)
    assert (to_cells.n_src, to_cells.n_dst) == (N_WINDOWS, N_CELLS)


def test_radius_edges_join_the_orbit_with_the_source_colour():
    edges = radius_edges(batch(), config())
    # Sources 0 and 4 are own stones, source 1 an opponent's.
    torch.testing.assert_close(edges.relation, _long(4, 0, 7, 2, 14))
    torch.testing.assert_close(edges.axis, _long(1, 0, -1, -1, 2))
    assert edges.num_relations == radius_relation_count(config())
    # Two of the five edges lie on no axis, so the routed subset is taken.
    assert edges.axis_rows is not None
    torch.testing.assert_close(edges.axis_rows, _long(0, 1, 4))


def test_the_same_shape_from_the_two_colours_is_two_relations():
    """§15.2's colour is part of the class, not a separate additive term."""
    own = radius_edges(batch(radius_src=_long(0, 0, 0, 0, 0)), config())
    opponent = radius_edges(batch(radius_src=_long(1, 1, 1, 1, 1)), config())
    assert not torch.equal(own.relation, opponent.relation)
    torch.testing.assert_close(opponent.relation - own.relation, torch.ones(5).long())


def test_adjacency_edges_carry_one_relation_and_always_a_route():
    edges = adjacency_edges(batch(), config())
    assert torch.all(edges.relation == adjacency_relation_id(config()))
    assert edges.axis_rows is None
    torch.testing.assert_close(edges.axis, _long(0, 0, 1, 1))


def test_the_three_paths_update_the_streams_they_point_at():
    torch.manual_seed(0)
    cfg = config()
    packed = batch()
    incidence = CellWindowIncidence(cfg).eval()
    adjacency = AdjacencyMessage(cfg).eval()
    radius = RadiusMessage(cfg).eval()

    cells, windows = states(N_CELLS, N_WINDOWS, seed=11)
    to_windows, to_cells = incidence_edges(packed)

    for updated, rows in (
        (incidence.to_windows(to_windows, cells, windows), N_WINDOWS),
        (incidence.to_cells(to_cells, windows, cells), N_CELLS),
        (adjacency(adjacency_edges(packed, cfg), cells), N_CELLS),
        (radius(radius_edges(packed, cfg), cells), N_CELLS),
    ):
        assert updated.inv.shape == (rows, D_INV)
        assert updated.axis.shape == (rows, NUM_AXES, D_AXIS)
        assert torch.isfinite(updated.inv).all() and torch.isfinite(updated.axis).all()


def test_the_three_paths_run_on_a_position_the_builder_made():
    """The edge builders against real builder output, not a hand-built table.

    A fixture cannot exercise a real index range, the real mix of on-axis and
    off-axis radius routes, or the masked-out slots of a window whose cells the
    scope omits. The counts here are a random playout's and are far from the
    self-play shape ``docs/MANTIS_ACT_DEVIATIONS.md`` measures, which is why
    nothing below asserts a size.
    """
    import random

    import hexo_py

    from mantisnet.models.mantis_act.builder import build
    from mantisnet.models.mantis_act.config import PRESETS
    from mantisnet.models.mantis_act.packed import collate

    cfg = PRESETS["full_act_v4"]
    rng = random.Random(11)
    position = hexo_py.Position()
    for _ in range(40):
        position.advance(*rng.choice(position.legal_moves()))
        assert not position.is_terminal
    packed = collate([build(position, cfg)], cfg)

    to_windows, to_cells = incidence_edges(packed)
    adjacency = adjacency_edges(packed, cfg)
    radius = radius_edges(packed, cfg)
    assert len(to_windows) == int(packed.window_incidence_mask.sum())
    assert len(to_cells) == len(to_windows)
    # §11.3: most radius displacements lie on no axis, so the routed subset is
    # a real subset and the mask path is not dead code.
    assert radius.axis_rows is not None
    assert 0 < len(radius.axis_rows) < len(radius)
    # §15.2's sources are stones.
    assert torch.all(packed.cell_occupancy.index_select(0, radius.src) > 0)

    torch.manual_seed(0)
    incidence = CellWindowIncidence(cfg).eval()
    generator = torch.Generator().manual_seed(16)

    def wide(rows):
        return EquivariantState(
            torch.randn(rows, cfg.d_inv, generator=generator) * 0.1,
            torch.randn(rows, NUM_AXES, cfg.d_axis, generator=generator) * 0.1,
        )

    cells = wide(packed.cell_occupancy.shape[0])
    windows = wide(packed.window_pattern_class.shape[0])
    for updated in (
        incidence.to_windows(to_windows, cells, windows),
        incidence.to_cells(to_cells, windows, cells),
        AdjacencyMessage(cfg).eval()(adjacency, cells),
        RadiusMessage(cfg).eval()(radius, cells),
    ):
        assert torch.isfinite(updated.inv).all() and torch.isfinite(updated.axis).all()


def test_the_two_incidence_directions_share_one_relation_table():
    """§14: relation tables may be shared; projections stay private."""
    incidence = CellWindowIncidence(config())
    assert incidence.to_windows.relation is incidence.to_cells.relation
    assert incidence.to_windows.wv_inv is not incidence.to_cells.wv_inv
    assert incidence.to_windows.update_inv is not incidence.to_cells.update_inv


def test_a_shared_relation_table_is_one_parameter_across_blocks():
    cfg = config()
    shared = CellWindowIncidence(cfg).to_windows.relation
    blocks = [
        AdjacencyMessage(cfg, relation_embedding=None) for _ in range(2)
    ]
    assert blocks[0].message.relation is not blocks[1].message.relation
    table = RelationGatedMessage(cfg, relation_vocabulary_size(cfg)).relation
    shared_blocks = [
        AdjacencyMessage(cfg, relation_embedding=table) for _ in range(2)
    ]
    assert shared_blocks[0].message.relation is shared_blocks[1].message.relation
    assert shared.num_embeddings == INCIDENCE_RELATIONS


def test_a_shared_table_of_the_wrong_size_raises():
    cfg = config()
    table = RelationGatedMessage(cfg, 7).relation
    with pytest.raises(ValueError, match="7 rows for a 8-class family"):
        RelationGatedMessage(cfg, 8, relation_embedding=table)


# --------------------------------------------------------------------------
# The arms that remove a stream (§29)


def test_the_no_axis_arm_holds_no_axis_parameters():
    cfg = config(d_axis=0, use_axis_channels=False, num_axis_latents=0)
    message = RelationGatedMessage(cfg, 8)
    assert not message.route_axis
    assert not any("axis" in name for name, _ in message.named_parameters())

    edges = edge_set()
    source, destination = states(edges.n_src, edges.n_dst, seed=12, d_axis=0)
    updated = message(edges, source, destination)
    assert updated.axis is None
    assert updated.inv.shape == (edges.n_dst, cfg.d_inv)


def test_unrouted_radius_messages_carry_no_axis_at_all():
    cfg = config(route_on_axis_radius_messages=False)
    radius = RadiusMessage(cfg).eval()
    assert not radius.message.route_axis
    edges = radius_edges(batch(), cfg)
    assert edges.axis is None
    with pytest.raises(ValueError, match="routes no axis message"):
        edges.routed()

    cells, _unused = states(N_CELLS, 1, seed=13)
    updated = radius(edges, cells)
    # The axis stream is present and comes back untouched, not zeroed.
    torch.testing.assert_close(updated.axis, cells.axis)
    assert updated.inv.shape == (N_CELLS, cfg.d_inv)
    assert not torch.allclose(updated.inv, cells.inv)


def test_a_message_refuses_an_edge_family_of_another_vocabulary():
    message = RelationGatedMessage(config(), 8)
    edges = edge_set(relation=_long(0, 1, 2, 3), num_relations=9)
    source, destination = states(edges.n_src, edges.n_dst, seed=14)
    with pytest.raises(ValueError, match="9 relation classes against this module's 8"):
        message(edges, source, destination)


def test_a_message_refuses_a_state_of_the_wrong_size():
    message = RelationGatedMessage(config(), 8)
    edges = edge_set()
    source, destination = states(edges.n_src, edges.n_dst, seed=15)
    small = EquivariantState(source.inv[:2], source.axis[:2])
    with pytest.raises(ValueError, match=r"source state covers \(2,\) entities"):
        message(edges, small, destination)


def test_a_routed_message_refuses_a_state_with_no_axis_stream():
    message = RelationGatedMessage(config(), 8)
    edges = edge_set()
    source, destination = states(edges.n_src, edges.n_dst, seed=17)
    with pytest.raises(ValueError, match="needs an axis stream"):
        message(edges, EquivariantState(source.inv), destination)
