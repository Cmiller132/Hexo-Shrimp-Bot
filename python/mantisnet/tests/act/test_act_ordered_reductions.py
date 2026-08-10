"""Deterministic run reductions used inside the outer ACT compile."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd

from mantisnet.models.mantis_act.ordered_reductions import (
    ordered_index_select,
    ordered_row_broadcast,
    ordered_segment_max,
    ordered_two_stage_segment_sum,
)
from mantisnet.models.mantis_act.actions import TACTICAL_FEATURES
from mantisnet.models.mantis_act.builder import GLOBAL_NUMERIC_FEATURES
from mantisnet.models.mantis_act.config import MantisACTConfig
from mantisnet.models.mantis_act.model import MantisACT
from mantisnet.models.mantis_act.packed import collate
from mantisnet.models.mantis_act.windows import WINDOW_NUMERIC_FEATURES

from .test_act_packed import graph_a, graph_b


def _class_plan(index: torch.Tensor, classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows = torch.argsort(index, stable=True)
    counts = torch.bincount(index, minlength=classes)
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    return offsets.to(torch.int32), rows.to(torch.int32)


def test_ordered_index_select_matches_literal_forward_and_exact_table_gradient():
    torch.manual_seed(20260810)
    table = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    index = torch.tensor([3, 0, 3, 4, 0, 3, 1], dtype=torch.long)
    offsets, rows = _class_plan(index, table.shape[0])
    gradient = torch.randn(index.shape[0], table.shape[1], dtype=torch.float64)

    actual = ordered_index_select(table, index, offsets, rows)
    expected = table.index_select(0, index)
    assert torch.equal(actual, expected)
    actual.backward(gradient)

    oracle = torch.zeros_like(table)
    for cls in range(table.shape[0]):
        for row in range(index.shape[0]):
            if int(index[row]) == cls:
                oracle[cls] += gradient[row]
    assert torch.equal(table.grad, oracle)


def test_ordered_row_broadcast_and_segment_max_match_segment_oracles():
    table = torch.tensor(
        [[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    offsets = torch.tensor([0, 2, 3, 6], dtype=torch.int32)
    owners = torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.long)
    gradient = torch.arange(12, dtype=torch.float64).reshape(6, 2)
    actual = ordered_row_broadcast(table, owners, offsets)
    assert torch.equal(actual, table.index_select(0, owners))
    actual.backward(gradient)
    assert torch.equal(
        table.grad,
        torch.stack((gradient[:2].sum(0), gradient[2], gradient[3:].sum(0))),
    )

    values = torch.tensor([1.0, 4.0, -2.0, 3.0, 8.0, 5.0], dtype=torch.float64)
    assert torch.equal(
        ordered_segment_max(values, offsets),
        torch.tensor([4.0, -2.0, 8.0], dtype=torch.float64),
    )


def test_two_stage_segment_sum_matches_fixed_block_oracle_with_an_empty_owner():
    values = (
        torch.arange(14, dtype=torch.float64).reshape(7, 2).remainder(5) - 2
    ) / 8
    # Owner 0 has consecutive blocks of 2 and 1 rows; owner 1 is empty; owner
    # 2 has two consecutive 2-row blocks.  This is the same compact block CSR
    # emitted by plans.py, only with a deliberately tiny block size.
    block_lengths = torch.tensor([2, 1, 2, 2], dtype=torch.int32)
    block_offsets = torch.tensor([0, 2, 2, 4], dtype=torch.int32)

    actual = ordered_two_stage_segment_sum(values, block_offsets, block_lengths)
    expected = torch.stack(
        (
            values[:3].sum(0),
            torch.zeros(2, dtype=torch.float64),
            values[3:].sum(0),
        )
    )
    assert torch.equal(actual, expected)

    all_empty = ordered_two_stage_segment_sum(
        torch.empty(0, 2, dtype=torch.float64),
        torch.zeros(4, dtype=torch.int32),
        torch.empty(0, dtype=torch.int32),
    )
    assert torch.equal(all_empty, torch.zeros(3, 2, dtype=torch.float64))


def test_all_ordered_reductions_pass_float64_gradcheck():
    index = torch.tensor([2, 0, 2, 1, 0], dtype=torch.long)
    class_offsets, rows = _class_plan(index, 3)
    table = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda value: ordered_index_select(value, index, class_offsets, rows),
        (table,),
    )

    segment_offsets = torch.tensor([0, 2, 3, 5], dtype=torch.int32)
    owners = torch.tensor([0, 0, 1, 2, 2], dtype=torch.long)
    context = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda value: ordered_row_broadcast(value, owners, segment_offsets),
        (context,),
    )

    # Unique maxima keep gradcheck away from the max subgradient boundary.
    values = torch.tensor(
        [1.0, 3.0, -2.0, 5.0, 2.0], dtype=torch.float64, requires_grad=True
    )
    assert torch.autograd.gradcheck(
        lambda value: ordered_segment_max(value, segment_offsets),
        (values,),
    )

    blocked = torch.randn(7, 2, dtype=torch.float64, requires_grad=True)
    block_lengths = torch.tensor([2, 1, 2, 2], dtype=torch.int32)
    block_offsets = torch.tensor([0, 2, 2, 4], dtype=torch.int32)
    assert torch.autograd.gradcheck(
        lambda value: ordered_two_stage_segment_sum(
            value, block_offsets, block_lengths
        ),
        (blocked,),
    )


def test_outer_aot_backward_contains_run_reductions_and_no_scatter():
    index = torch.tensor([2, 0, 2, 1, 0], dtype=torch.long)
    class_offsets, rows = _class_plan(index, 3)
    segment_offsets = torch.tensor([0, 2, 3, 5], dtype=torch.int32)
    owners = torch.tensor([0, 0, 1, 2, 2], dtype=torch.long)
    graphs = []

    def compiler(kind):
        def capture(graph, _inputs):
            graphs.append((kind, graph))
            return make_boxed_func(graph.forward)

        return capture

    backend = aot_autograd(
        fw_compiler=compiler("forward"), bw_compiler=compiler("backward")
    )

    def loss(table, context, values):
        selected = ordered_index_select(table, index, class_offsets, rows)
        broadcast = ordered_row_broadcast(context, owners, segment_offsets)
        maxima = ordered_segment_max(values, segment_offsets)
        expanded_max = ordered_row_broadcast(maxima, owners, segment_offsets)
        return (
            selected.square().sum()
            + broadcast.square().sum()
            + (values / expanded_max).square().sum()
        )

    compiled = torch.compile(loss, backend=backend, dynamic=True, fullgraph=True)
    inputs = (
        torch.randn(3, 4, requires_grad=True),
        torch.randn(3, 4, requires_grad=True),
        torch.tensor([1.0, 3.0, 2.0, 5.0, 4.0], requires_grad=True),
    )
    compiled(*inputs).backward()
    assert {kind for kind, _graph in graphs} == {"forward", "backward"}

    targets = [str(node.target) for _kind, graph in graphs for node in graph.graph.nodes]
    forbidden = ("scatter", "index_add", "embedding_dense_backward")
    assert not [target for target in targets if any(name in target for name in forbidden)]
    assert any("segment_reduce" in target for target in targets)


def test_complete_outer_aot_graph_contains_no_scattering_reduction():
    """Every default gather/reduction site is covered, not just the primitives."""
    cfg = replace(
        MantisACTConfig(),
        d_inv=8,
        d_axis=4,
        d_rel=4,
        num_heads=2,
        state_blocks=1,
        action_blocks=1,
        num_inv_latents=2,
        num_axis_latents=1,
        num_action_latents=1,
    )
    graphs = (
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
    batch = collate(graphs, cfg)
    model = MantisACT(cfg).eval()
    captured = []

    def compiler(kind):
        def capture(graph, _inputs):
            captured.append((kind, graph))
            return make_boxed_func(graph.forward)

        return capture

    backend = aot_autograd(
        fw_compiler=compiler("forward"), bw_compiler=compiler("backward")
    )

    def loss(module, packed):
        output = module(packed, 0.2)
        return sum(
            value.float().square().mean()
            for value in (
                output.policy_logits,
                output.critic_logits,
                output.q_value,
                output.q_score,
            )
        )

    compiled = torch.compile(loss, backend=backend, dynamic=True, fullgraph=True)
    compiled(model, batch).backward()
    assert {kind for kind, _graph in captured} == {"forward", "backward"}

    targets = [
        str(node.target) for _kind, graph in captured for node in graph.graph.nodes
    ]
    forbidden = ("scatter", "index_add", "index_put", "embedding_dense_backward")
    offenders = [
        target for target in targets if any(name in target for name in forbidden)
    ]
    assert not offenders
    # One acting-Q maximum plus every ordered table/gather gradient.
    assert sum("segment_reduce" in target for target in targets) >= 10
