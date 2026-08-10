"""Parity and graph-shape gates for traceable horizontal linears."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from mantisnet.models.mantis_act.linear_fusion import horizontal_linears


def _loss(outputs):
    return sum(
        (index + 1.25) * output.square().sum()
        for index, output in enumerate(outputs)
    )


def test_horizontal_linears_match_independent_modules_and_all_gradients_fp64():
    torch.manual_seed(19)
    projections = nn.ModuleList(
        [nn.Linear(7, 5), nn.Linear(7, 3), nn.Linear(7, 11)]
    ).double()
    reference = copy.deepcopy(projections)
    x = torch.randn(2, 4, 7, dtype=torch.float64, requires_grad=True)
    x_reference = x.detach().clone().requires_grad_(True)

    fused = horizontal_linears(x, projections)
    literal = tuple(projection(x_reference) for projection in reference)
    for actual, expected in zip(fused, literal, strict=True):
        # One output-wide GEMM may select a different CPU micro-kernel from the
        # three narrow calls; the only permitted difference is roundoff.
        torch.testing.assert_close(actual, expected, atol=3e-15, rtol=3e-15)

    _loss(fused).backward()
    _loss(literal).backward()
    torch.testing.assert_close(x.grad, x_reference.grad, atol=1e-12, rtol=1e-12)
    for actual, expected in zip(projections.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, atol=1e-12, rtol=1e-12)


def test_horizontal_linears_preserve_parameter_ownership_and_names():
    class Site(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(4, 4, bias=False)
            self.k = nn.Linear(4, 2, bias=False)

        def forward(self, x):
            return horizontal_linears(x, (self.q, self.k))

    site = Site()
    before = tuple(site.state_dict())
    site(torch.randn(3, 4))
    assert tuple(site.state_dict()) == before == ("q.weight", "k.weight")

    graph = torch.fx.symbolic_trace(site).graph
    linear_nodes = [
        node
        for node in graph.nodes
        if node.op == "call_function" and node.target is torch._C._nn.linear
    ]
    assert len(linear_nodes) == 1


def test_horizontal_linears_refuse_nonfusible_projection_sets():
    x = torch.randn(2, 4)
    with pytest.raises(ValueError, match="at least two"):
        horizontal_linears(x, (nn.Linear(4, 3),))
    with pytest.raises(ValueError, match="in_features"):
        horizontal_linears(x, (nn.Linear(4, 3), nn.Linear(5, 3)))
    with pytest.raises(ValueError, match="bias presence"):
        horizontal_linears(x, (nn.Linear(4, 3), nn.Linear(4, 3, bias=False)))
