"""Shared decoder layout, folded head matrix, and kernel contracts.

The oracle transcribes the §6 decoder formula directly: project window rows,
add joint-class embeddings, sum each cell's entries, and overwrite background
cells from the bucket table. The implementation aggregates before projection.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

import mantisnet.decoder as decoder_impl
from mantisnet import collate, from_position
from mantisnet.decoder import (
    _FAILED_SHAPES,
    _aggregate_reference,
    CLASS_SLOTS,
    COEF_WIDTH,
    aggregate,
    head_matrix,
)

from .conftest import JOINT_ORBITS

_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the decoder aggregation kernel requires CUDA"
)


def _spec_decoder_input(w, batch, proj, e_class, e_bg):
    """§6's per-cell decoder input, written the way the spec states it."""
    msg = (w @ proj.t()).index_select(0, batch.dec_window) + e_class[batch.dec_class]
    h = torch.zeros(batch.cell_pos.shape[0], w.shape[1], dtype=w.dtype, device=w.device)
    h.index_add_(0, batch.dec_cell, msg)
    if batch.bg_cell.numel():
        h.index_copy_(0, batch.bg_cell, e_bg[batch.bg_bucket])
    return h


def _batch(positions):
    return collate([from_position(p) for p in positions])


def _windows(batch, h, seed=0):
    generator = torch.Generator().manual_seed(seed)
    n_w = int(batch.dec_window.max()) + 1 if batch.dec_window.numel() else 1
    return torch.randn(n_w, h, generator=generator)


def test_aggregate_row_layout(positions):
    batch = _batch(positions)
    w = _windows(batch, 16)
    rows = aggregate(
        w, batch.dec_window, batch.dec_class, batch.dec_cell,
        batch.bg_cell, batch.bg_bucket, batch.cell_pos.shape[0],
    )
    assert rows.shape == (batch.cell_pos.shape[0], 16 + COEF_WIDTH)

    windows = torch.zeros_like(rows[:, :16])
    classes = torch.zeros_like(rows[:, 16 : 16 + CLASS_SLOTS])
    for cell, window, slot in zip(batch.dec_cell, batch.dec_window, batch.dec_class):
        windows[cell] += w[window]
        classes[cell, slot] += 1
    torch.testing.assert_close(rows[:, :16], windows, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(rows[:, 16 : 16 + CLASS_SLOTS], classes)

    background = torch.zeros(batch.cell_pos.shape[0], COEF_WIDTH - CLASS_SLOTS)
    background[batch.bg_cell, batch.bg_bucket] = 1
    torch.testing.assert_close(rows[:, 16 + CLASS_SLOTS :], background)
    # A background cell is in no live window, so it carries no window or class
    # weight — which is why the aggregation may add where the spec overwrites.
    assert rows[batch.bg_cell, : 16 + CLASS_SLOTS].abs().sum() == 0


def test_head_matrix_reproduces_the_spec_decoder_input(positions, model):
    batch = _batch(positions)
    h = model.cfg.h
    w = _windows(batch, h)
    rows = aggregate(
        w, batch.dec_window, batch.dec_class, batch.dec_cell,
        batch.bg_cell, batch.bg_bucket, batch.cell_pos.shape[0],
    )
    for proj, e_class, e_bg, mlp in (
        (model.p, model.e_pw, model.e_bg, model.mlp_p),
        (model.q, model.e_qw, model.e_qbg, model.mlp_q),
    ):
        spec = mlp.lin_a(
            _spec_decoder_input(w, batch, proj.weight, e_class.weight, e_bg.weight)
        )
        folded = F.linear(
            rows,
            head_matrix(proj.weight, e_class.weight, e_bg.weight, mlp.lin_a.weight),
            mlp.lin_a.bias,
        )
        torch.testing.assert_close(folded, spec, rtol=1e-4, atol=1e-4)


def test_slack_columns_never_contribute(positions, model):
    # The row is padded to COEF_WIDTH for the head GEMM's K. The padding is
    # stored as zero, so whatever a head matrix holds there is inert.
    batch = _batch(positions)
    w = _windows(batch, model.cfg.h)
    rows = aggregate(
        w, batch.dec_window, batch.dec_class, batch.dec_cell,
        batch.bg_cell, batch.bg_bucket, batch.cell_pos.shape[0],
    )
    used = model.cfg.h + CLASS_SLOTS + decoder_impl.BG_SLOTS
    assert rows[:, used:].abs().sum() == 0


@torch.no_grad()
def test_the_joint_class_separates_cells_the_slot_class_could_not(positions, model):
    """Two legal moves that a slot-class decoder cannot rank differently.

    A cell's decoder row is its window rows summed plus its class counts, so two
    cells with the same window multiset and the same counts get one row — and
    therefore one logit and one action value, whatever the weights. Under
    ``(canonical mask, slot class)`` that happens for real move pairs: mirrored
    slots of a non-palindromic window share a class, so extending a stone by
    contact and extending it across a gap are the same input. The joint class
    keys the pair jointly and separates them.

    The pair here is found in the shared position set rather than hand-built, so
    the test asserts the aliasing is reachable in play, not only in principle.
    """
    slot_class = {rank: min(slot, 5 - slot) for rank, (_mask, slot) in enumerate(JOINT_ORBITS)}
    found = []
    for pos in positions:
        batch = _batch([pos])
        old: dict[int, list] = {}
        new: dict[int, list] = {}
        for cell, window, cls in zip(batch.dec_cell, batch.dec_window, batch.dec_class):
            old.setdefault(int(cell), []).append((int(window), slot_class[int(cls)]))
            new.setdefault(int(cell), []).append((int(window), int(cls)))
        by_old: dict[tuple, list[int]] = {}
        for cell, entries in old.items():
            by_old.setdefault(tuple(sorted(entries)), []).append(cell)
        for cells in by_old.values():
            for a, b in zip(cells, cells[1:]):
                if sorted(new[a]) != sorted(new[b]):
                    found.append((pos, batch, a, b))
                    break
            if found:
                break
        if found:
            break
    assert found, "no aliased move pair in the shared positions: nothing to separate"

    pos, batch, a, b = found[0]
    rows = aggregate(
        _windows(batch, model.cfg.h), batch.dec_window, batch.dec_class,
        batch.dec_cell, batch.bg_cell, batch.bg_bucket, batch.cell_pos.shape[0],
    )
    # The window halves agree — the pair shares its live windows — and the class
    # counts are what now differ.
    torch.testing.assert_close(rows[a, : model.cfg.h], rows[b, : model.cfg.h])
    assert not torch.equal(rows[a, model.cfg.h :], rows[b, model.cfg.h :])

    # And so the heads can score them apart, which under the slot class they
    # could not: identical rows give identical outputs whatever the weights. A
    # fresh model has the zero readouts of §10 and scores every cell zero, so
    # the demonstration needs a readout that reads — any trained one does.
    scored = copy.deepcopy(model)
    generator = torch.Generator().manual_seed(11)
    for out in (scored.mlp_p.out, scored.mlp_q.out):
        out.weight.copy_(torch.randn(out.weight.shape, generator=generator) * 0.1)
    _s, w, g = scored.trunk(batch)
    policy, q = scored.cell_heads(w, g, batch)
    assert policy[a] != policy[b]
    assert q[a] != q[b]


@torch.no_grad()
def test_cell_heads_match_the_spec_decode(positions, model):
    batch = _batch(positions)
    _s, w, g = model.trunk(batch)
    policy, q = model.cell_heads(w, g, batch)

    for scores, tail, proj, e_class, e_bg, mlp in (
        (policy, lambda x: x, model.p, model.e_pw, model.e_bg, model.mlp_p),
        (q, torch.tanh, model.q, model.e_qw, model.e_qbg, model.mlp_q),
    ):
        h = _spec_decoder_input(w, batch, proj.weight, e_class.weight, e_bg.weight)
        spec = mlp.out(
            F.relu(mlp.lin_a(h) + mlp.lin_b(g).index_select(0, batch.cell_pos))
        ).squeeze(-1)
        torch.testing.assert_close(scores, tail(spec), rtol=1e-4, atol=1e-4)


@torch.no_grad()
def test_policy_head_matches_the_pair(positions, model):
    batch = _batch(positions)
    _s, w, g = model.trunk(batch)
    policy, _q = model.cell_heads(w, g, batch)
    assert torch.equal(model.policy_head(w, g, batch), policy)


def test_aggregation_gradient_flows_only_to_the_windows(positions):
    batch = _batch(positions)
    w = _windows(batch, 16).requires_grad_(True)
    rows = aggregate(
        w, batch.dec_window, batch.dec_class, batch.dec_cell,
        batch.bg_cell, batch.bg_bucket, batch.cell_pos.shape[0],
    )
    torch.manual_seed(0)
    seed_grad = torch.randn_like(rows)
    rows.backward(seed_grad)

    expected = torch.zeros_like(w)
    for cell, window in zip(batch.dec_cell, batch.dec_window):
        expected[window] += seed_grad[cell, :16]
    torch.testing.assert_close(w.grad, expected, rtol=1e-4, atol=1e-4)


@_CUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_kernel_matches_the_scatter(positions, dtype):
    batch = _batch(positions).to("cuda")
    w = _windows(batch, 128).to("cuda", dtype)
    rows = aggregate(
        w, batch.dec_window, batch.dec_class, batch.dec_cell,
        batch.bg_cell, batch.bg_bucket, batch.cell_pos.shape[0],
    )
    reference = _aggregate_reference(
        w, batch.dec_window, batch.dec_class, batch.dec_cell, batch.cell_pos.shape[0]
    )
    reference[batch.bg_cell, 128 + CLASS_SLOTS + batch.bg_bucket] = 1
    torch.testing.assert_close(rows, reference)
    assert not _FAILED_SHAPES, _FAILED_SHAPES


@_CUDA
def test_kernel_gradient_matches_the_scatter(positions):
    batch = _batch(positions).to("cuda")
    base = _windows(batch, 128).to("cuda")

    def grad_of(fn):
        w = base.clone().requires_grad_(True)
        rows = fn(w)
        torch.manual_seed(1)
        rows.backward(torch.randn_like(rows))
        return w.grad

    kernel = grad_of(
        lambda w: aggregate(
            w, batch.dec_window, batch.dec_class, batch.dec_cell,
            batch.bg_cell, batch.bg_bucket, batch.cell_pos.shape[0],
        )
    )
    scatter = grad_of(
        lambda w: _aggregate_reference(
            w, batch.dec_window, batch.dec_class, batch.dec_cell,
            batch.cell_pos.shape[0],
        )
    )
    torch.testing.assert_close(kernel, scatter, rtol=1e-5, atol=1e-5)
    assert not _FAILED_SHAPES, _FAILED_SHAPES


@_CUDA
@torch.no_grad()
def test_compiled_dynamic_heads_match_eager(positions, model):
    model = model.to("cuda")
    compiled = torch.compile(
        lambda m, b: m.cell_heads(*m.trunk(b)[1:], b), dynamic=True
    )
    try:
        # Several shapes through one dynamic graph: the aggregation stays in it
        # only if its fake kernel tracks the symbolic cell count.
        for count in (2, 5, len(positions)):
            batch = _batch(positions[:count]).to("cuda")
            eager = model.cell_heads(*model.trunk(batch)[1:], batch)
            got = compiled(model, batch)
            for a, b in zip(got, eager):
                torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)
    finally:
        model.to("cpu")
    assert not _FAILED_SHAPES, _FAILED_SHAPES
