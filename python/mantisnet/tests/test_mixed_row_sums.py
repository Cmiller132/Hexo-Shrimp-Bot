"""Mixed-scope run-reduced sums: kernel parity and gradients vs the literal.

The oracle is the literal gather/``index_add_`` formulation the mixed paths
used before ``class_row_sum`` and ``incidence_row_sum``; the ops must match
it in value and gradient while never materializing the (E, H) per-edge
gather.  The oracle deliberately does not use the run discovery or any
reordered view beyond what the call site itself hands the op.
"""

from __future__ import annotations

import pytest
import torch

import mantisnet.message_passing as message_impl
from mantisnet.builder import TERN_OCC_CLASSES, collate, from_position
from mantisnet.message_passing import (
    STONE_RUN,
    WINDOW_RUN,
    class_row_sum,
    incidence_plan,
    incidence_row_sum,
)
from mantisnet.model import MantisConfig, MantisNet


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the run-reduction parity cases require CUDA"
)

_DEVICE = torch.device("cuda")
_H = 128


def _literal(
    table: torch.Tensor, gather: torch.Tensor, dest: torch.Tensor, n_dest: int
) -> torch.Tensor:
    rows = table.index_select(0, gather).float()
    out = torch.zeros(
        n_dest, table.shape[1], dtype=torch.float32, device=table.device
    )
    return out.index_add_(0, dest, rows)


def _mixed_batch(positions, case: str):
    graphs = [from_position(pos) for pos in positions]
    if case == "empty":
        selected = [graph for graph in graphs if graph.n_stones == 0]
        assert len(selected) == 1 and selected[0].n_windows == 0
    elif case == "single":
        selected = [graph for graph in graphs if graph.n_stones == 1]
        assert len(selected) == 1 and selected[0].n_windows == 18
    else:
        selected = graphs
    return collate(selected).to(_DEVICE)


def _dec_views(batch) -> tuple[torch.Tensor, torch.Tensor]:
    order = torch.argsort(batch.dec_window.to(torch.int32), stable=True)
    return (
        batch.dec_cell.index_select(0, order),
        batch.dec_window.index_select(0, order),
    )


def _sites(batch) -> dict[str, tuple[torch.Tensor, torch.Tensor, int, int]]:
    """The three real ``class_row_sum`` call shapes of the mixed model."""
    n_windows = int(batch.window_feat.shape[0])
    n_stones = int(batch.stone_own.shape[0])
    n_cells = int(batch.cell_pos.shape[0])
    plan = incidence_plan(batch.inc_stone, batch.inc_window, batch.inc_class)
    return {
        "windows": (batch.inc_class, batch.inc_window, n_windows, WINDOW_RUN),
        "stones": (plan.run_class, plan.run_stone, n_stones, STONE_RUN),
        "cells": (batch.dec_class, batch.dec_cell, n_cells, STONE_RUN),
    }


def _table_for(gather: torch.Tensor, seed: int) -> torch.Tensor:
    n_classes = int(gather.max()) + 1 if gather.numel() else 1
    generator = torch.Generator(device=_DEVICE).manual_seed(seed)
    return torch.randn((n_classes, _H), device=_DEVICE, generator=generator)


@pytest.mark.parametrize("case", ["empty", "single", "ragged"])
@pytest.mark.parametrize("site", ["windows", "stones", "cells"])
def test_class_row_sum_matches_the_literal_scatter(positions, case: str, site: str):
    batch = _mixed_batch(positions, case)
    if case != "empty":
        # Otherwise an absent Triton install would make every parity assertion
        # exercise the torch fallback and prove nothing about the CUDA path.
        assert message_impl.triton is not None
    gather, runs, n_dest, run_len = _sites(batch)[site]
    weight = _table_for(gather, seed=7 + run_len)

    actual = class_row_sum(weight, gather, runs, n_dest, run_len)
    expected = _literal(weight, gather, runs, n_dest)

    assert actual.dtype == torch.float32 and actual.shape == (n_dest, _H)
    torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-5)


@pytest.mark.parametrize("site", ["windows", "stones", "cells"])
def test_class_row_sum_gradient_matches_the_literal_scatter(positions, site: str):
    batch = _mixed_batch(positions, "ragged")
    gather, runs, n_dest, run_len = _sites(batch)[site]
    # Enough distinct classes that the table gradient is a real reduction.
    assert gather.unique().numel() > 3
    base = _table_for(gather, seed=23)
    generator = torch.Generator(device=_DEVICE).manual_seed(29)
    # Small integer-valued gradients remain nonuniform but sum exactly in
    # fp32, so different reduction orders cannot make this comparison flaky.
    upstream = torch.randint(
        -3, 4, (n_dest, _H), device=_DEVICE, generator=generator
    ).float()

    fast = base.detach().clone().requires_grad_()
    (fast_grad,) = torch.autograd.grad(
        class_row_sum(fast, gather, runs, n_dest, run_len), fast, upstream
    )
    ref = base.detach().clone().requires_grad_()
    (ref_grad,) = torch.autograd.grad(_literal(ref, gather, runs, n_dest), ref, upstream)

    torch.testing.assert_close(fast_grad, ref_grad, rtol=2.0e-5, atol=2.0e-5)


@pytest.mark.parametrize("case", ["empty", "single", "ragged"])
@pytest.mark.parametrize("values_dtype", [torch.float32, torch.bfloat16])
def test_incidence_row_sum_matches_the_literal_scatter(
    positions, case: str, values_dtype: torch.dtype
):
    batch = _mixed_batch(positions, case)
    n_windows = int(batch.window_feat.shape[0])
    n_cells = int(batch.cell_pos.shape[0])
    rev_gather, rev_runs = _dec_views(batch)
    generator = torch.Generator(device=_DEVICE).manual_seed(41)
    values = torch.randn(
        (n_windows, _H), device=_DEVICE, dtype=values_dtype, generator=generator
    )

    actual = incidence_row_sum(
        values,
        batch.dec_window,
        batch.dec_cell,
        rev_gather,
        rev_runs,
        n_cells,
        STONE_RUN,
        WINDOW_RUN,
    )
    expected = _literal(values, batch.dec_window, batch.dec_cell, n_cells)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-5)


@pytest.mark.parametrize("values_dtype", [torch.float32, torch.bfloat16])
def test_incidence_row_sum_gradient_matches_the_literal_scatter(
    positions, values_dtype: torch.dtype
):
    batch = _mixed_batch(positions, "ragged")
    n_windows = int(batch.window_feat.shape[0])
    n_cells = int(batch.cell_pos.shape[0])
    rev_gather, rev_runs = _dec_views(batch)
    generator = torch.Generator(device=_DEVICE).manual_seed(43)
    base = torch.randn(
        (n_windows, _H), device=_DEVICE, dtype=values_dtype, generator=generator
    )
    upstream = torch.randint(
        -3, 4, (n_cells, _H), device=_DEVICE, generator=generator
    ).float()

    fast = base.detach().clone().requires_grad_()
    (fast_grad,) = torch.autograd.grad(
        incidence_row_sum(
            fast,
            batch.dec_window,
            batch.dec_cell,
            rev_gather,
            rev_runs,
            n_cells,
            STONE_RUN,
            WINDOW_RUN,
        ),
        fast,
        upstream,
    )
    ref = base.detach().clone().requires_grad_()
    (ref_grad,) = torch.autograd.grad(
        _literal(ref, batch.dec_window, batch.dec_cell, n_cells), ref, upstream
    )

    # Integer upstreams sum exactly in fp32, so both orders round identically
    # at the one output cast and the comparison stays exact-tolerance.
    assert fast_grad.dtype == values_dtype
    torch.testing.assert_close(fast_grad, ref_grad, rtol=2.0e-5, atol=2.0e-5)


@torch.no_grad()
def test_mixed_model_call_sites_match_the_literal_forms(positions, monkeypatch):
    """The integrated mixed trunk and heads equal the literal formulation.

    Primitive parity cannot catch a call site handing the wrong view to the
    right op; replacing the ops with the literal scatter while running the
    same model does.
    """
    torch.manual_seed(3)
    model = MantisNet(MantisConfig()).to(_DEVICE).eval()
    batch = _mixed_batch(positions, "ragged")

    def run():
        with torch.autocast("cuda", torch.bfloat16):
            w, g, cells = model.trunk(batch)
            policy, critic = model.cell_head_logits(w, g, cells, batch)
        return w, g, policy, critic

    fast = run()
    seen: set[str] = set()

    def literal_class(weight, gather, runs, n_dest, run_len):
        assert run_len in (WINDOW_RUN, STONE_RUN)
        seen.add(f"class:{run_len}:{n_dest}")
        return _literal(weight, gather, runs, n_dest)

    def literal_incidence(
        values, gather, runs, rev_gather, rev_runs, n_dest, run_len, rev_run_len
    ):
        assert (run_len, rev_run_len) == (STONE_RUN, WINDOW_RUN)
        assert rev_gather.shape == gather.shape == runs.shape == rev_runs.shape
        seen.add("incidence")
        return _literal(values, gather, runs, n_dest)

    monkeypatch.setattr(message_impl, "class_row_sum", literal_class)
    monkeypatch.setattr(message_impl, "incidence_row_sum", literal_incidence)
    reference = run()

    assert "incidence" in seen and len(seen) >= 3
    for actual, expected in zip(fast, reference):
        torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-5)


@torch.no_grad()
def test_dynamic_fullgraph_compile_matches_the_literal_scatter(positions):
    compiled = torch.compile(class_row_sum, dynamic=True, fullgraph=True)
    for index, case in enumerate(("single", "ragged")):
        batch = _mixed_batch(positions, case)
        gather, runs, n_dest, run_len = _sites(batch)["windows"]
        weight = _table_for(gather, seed=61 + index)

        actual = compiled(weight, gather, runs, n_dest, run_len)
        expected = _literal(weight, gather, runs, n_dest)
        torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-5)
