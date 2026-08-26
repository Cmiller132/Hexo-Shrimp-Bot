"""The cell pass's relay tables and segment kernels (§5.1b).

The oracle transcribes the §5.1b composition directly over the raw decoder
incidence — gather window messages, sum per cell, ReLU, gather the cell
values back — with order-independent ``index_add_``. The implementation
sorts the incidence once and runs contiguous segment reductions; these tests
hold the two shapes together, forward and backward, on every device.
"""

from __future__ import annotations

import pytest
import torch

import mantisnet.relay as relay_impl
from mantisnet import collate, from_position
from mantisnet.builder import TERN_DEC_CLASSES
from mantisnet.relay import (
    _FAILED_BACKWARD_SHAPES,
    _FAILED_SHAPES,
    _reference,
    cell_pass,
    relay_tables,
)

_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the cell-pass kernels require CUDA"
)


def _spec_cell_pass(x, emb, batch):
    """§5.1b written the way the spec states it, over the raw incidence."""
    msg = x.float().index_select(0, batch.dec_window) + emb.float()[batch.dec_class]
    cells = torch.zeros(
        batch.cell_pos.shape[0], x.shape[1], dtype=torch.float32, device=x.device
    )
    cells.index_add_(0, batch.dec_cell, msg)
    cells = torch.relu(cells)
    agg = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32, device=x.device)
    agg.index_add_(0, batch.dec_window, cells.index_select(0, batch.dec_cell))
    return agg.to(x.dtype)


def _batch(positions):
    return collate([from_position(p) for p in positions])


def _tables(batch, device="cpu"):
    tables = relay_tables(
        batch.dec_cell,
        batch.dec_window,
        batch.dec_class,
        batch.window_feat.shape[0],
        TERN_DEC_CLASSES,
    )
    return tuple(t.to(device) for t in tables)


def _inputs(batch, h=32, seed=0, device="cpu", dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch.window_feat.shape[0], h, generator=generator)
    emb = torch.randn(TERN_DEC_CLASSES, h, generator=generator)
    return x.to(device, dtype), emb.to(device)


def test_relay_tables_are_hand_derived():
    dec_cell = torch.tensor([5, 5, 2, 9, 9, 9])
    dec_window = torch.tensor([0, 3, 1, 0, 2, 3])
    dec_class = torch.tensor([4, 0, 4, 1, 0, 2])
    cell_ptr, edge_window, edge_class, win_ptr, edge_wcell, cls_ptr, edge_ccell = (
        relay_tables(dec_cell, dec_window, dec_class, n_windows=5, n_classes=6)
    )
    # Cell-sorted edge order: cell 2's entry, cell 5's two, cell 9's three.
    assert cell_ptr.tolist() == [0, 1, 3, 6]
    assert edge_window.tolist() == [1, 0, 3, 0, 2, 3]
    assert edge_class.tolist() == [4, 4, 0, 1, 0, 2]
    # Window runs, including window 4's empty one.
    assert win_ptr.tolist() == [0, 2, 3, 4, 6, 6]
    assert edge_wcell.tolist() == [1, 2, 0, 2, 1, 2]
    # Class runs over all six classes; 3 and 5 are empty.
    assert cls_ptr.tolist() == [0, 2, 3, 4, 4, 6, 6]
    assert edge_ccell.tolist() == [1, 2, 2, 2, 0, 1]


def test_relay_tables_refuse_malformed_incidence():
    edges = torch.tensor([0, 1])
    with pytest.raises(ValueError, match="one length"):
        relay_tables(edges, edges, torch.tensor([0]), 2, 4)
    with pytest.raises(ValueError, match="dec_window"):
        relay_tables(edges, torch.tensor([0, 2]), edges, 2, 4)
    with pytest.raises(ValueError, match="dec_class"):
        relay_tables(edges, edges, torch.tensor([0, 4]), 2, 4)


def test_cell_pass_refuses_mismatched_tables(positions):
    batch = _batch(positions)
    tables = _tables(batch)
    x, emb = _inputs(batch)
    with pytest.raises(ValueError, match="win_ptr"):
        cell_pass(x[:-1], emb, *tables)
    with pytest.raises(ValueError, match="cls_ptr"):
        cell_pass(x, emb[:-1], *tables)
    with pytest.raises(ValueError, match="one length"):
        cell_pass(x, emb, tables[0], tables[1][:-1], *tables[2:])


@pytest.mark.parametrize("subset", [slice(None), slice(0, 1), slice(-1, None)])
def test_matches_the_spec_composition(positions, subset):
    # slice(0, 1) is the ply-0 position: no stones, no windows, empty tables.
    batch = _batch(positions[subset])
    x, emb = _inputs(batch)
    got = cell_pass(x, emb, *_tables(batch))
    torch.testing.assert_close(got, _spec_cell_pass(x, emb, batch), rtol=1e-5, atol=1e-5)


def test_gradients_match_the_spec_composition(positions):
    batch = _batch(positions)
    tables = _tables(batch)
    base_x, base_emb = _inputs(batch)
    torch.manual_seed(1)
    seed_grad = torch.randn(base_x.shape)

    def grads_of(fn):
        x = base_x.clone().requires_grad_(True)
        emb = base_emb.clone().requires_grad_(True)
        fn(x, emb).backward(seed_grad)
        return x.grad, emb.grad

    got_x, got_emb = grads_of(lambda x, emb: cell_pass(x, emb, *tables))
    spec_x, spec_emb = grads_of(lambda x, emb: _spec_cell_pass(x, emb, batch))
    # 1e-4 as in the decoder's gradient parity: the class rows sum hundreds of
    # fp32 terms in a different order than autograd's scatter.
    torch.testing.assert_close(got_x, spec_x, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(got_emb, spec_emb, rtol=1e-4, atol=1e-4)


@_CUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_kernel_matches_the_scatter(positions, dtype):
    batch = _batch(positions)
    tables = _tables(batch, "cuda")
    x, emb = _inputs(batch, h=128, device="cuda", dtype=dtype)
    got = cell_pass(x, emb, *tables)
    reference = _reference(x, emb, *tables[:5])
    torch.testing.assert_close(got, reference)
    assert not _FAILED_SHAPES, _FAILED_SHAPES


@_CUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_kernel_gradients_match_the_spec(positions, dtype):
    batch = _batch(positions)
    tables = _tables(batch, "cuda")
    base_x, base_emb = _inputs(batch, h=128, device="cuda", dtype=dtype)
    torch.manual_seed(2)
    seed_grad = torch.randn(base_x.shape, device="cuda").to(dtype)
    spec_batch_dev = batch.to("cuda")

    def grads_of(fn):
        x = base_x.clone().requires_grad_(True)
        emb = base_emb.clone().requires_grad_(True)
        fn(x, emb).backward(seed_grad)
        return x.grad, emb.grad

    got_x, got_emb = grads_of(lambda x, emb: cell_pass(x, emb, *tables))
    spec_x, spec_emb = grads_of(lambda x, emb: _spec_cell_pass(x, emb, spec_batch_dev))
    tol = dict(rtol=1e-4, atol=1e-4) if dtype == torch.float32 else dict(rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(got_x, spec_x, **tol)
    torch.testing.assert_close(got_emb, spec_emb, **tol)
    assert not _FAILED_SHAPES, _FAILED_SHAPES
    assert not _FAILED_BACKWARD_SHAPES, _FAILED_BACKWARD_SHAPES


@_CUDA
def test_kernel_is_deterministic(positions):
    # The segment reductions accumulate in a fixed order, unlike an atomic
    # ``index_add_``: two identical passes must agree bitwise.
    batch = _batch(positions)
    tables = _tables(batch, "cuda")
    base_x, base_emb = _inputs(batch, h=128, device="cuda", dtype=torch.bfloat16)
    torch.manual_seed(3)
    seed_grad = torch.randn(base_x.shape, device="cuda", dtype=torch.bfloat16)

    def run():
        x = base_x.clone().requires_grad_(True)
        emb = base_emb.clone().requires_grad_(True)
        out = cell_pass(x, emb, *tables)
        out.backward(seed_grad)
        return out.detach(), x.grad, emb.grad

    first, second = run(), run()
    for a, b in zip(first, second):
        assert torch.equal(a, b)


@_CUDA
def test_compiled_dynamic_matches_eager(positions):
    compiled = torch.compile(cell_pass, dynamic=True)
    for count in (2, 5, len(positions)):
        batch = _batch(positions[:count])
        tables = _tables(batch, "cuda")
        x, emb = _inputs(batch, h=128, device="cuda", dtype=torch.bfloat16)
        eager = cell_pass(x, emb, *tables)
        got = compiled(x, emb, *tables)
        torch.testing.assert_close(got, eager)
    assert not _FAILED_SHAPES, _FAILED_SHAPES
