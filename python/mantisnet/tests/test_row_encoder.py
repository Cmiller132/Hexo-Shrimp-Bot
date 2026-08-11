"""The Step 4 row encoder: reference semantics and kernel parity."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import hexo_py
import mantisnet.row_encoder as row_impl
from mantisnet.builder import collate, from_position
from mantisnet.row_encoder import encode_rows

_GAMES = [
    [(0, 0), (1, 0), (2, 0)],
    [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (0, 3), (0, 4), (3, 0)],
    [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, -1), (0, 2), (4, -1), (-1, 2)],
]

_DEVICE = torch.device("cuda")
_H = 128
_CUDA_ONLY = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="row-encoder kernel parity requires CUDA"
)


def _batch():
    return collate(
        [
            from_position(hexo_py.Position.replay(moves), action_rows=True)
            for moves in _GAMES
        ]
    )


def _loop_reference(pre_w, table, edge_window, edge_class, edge_cell, n_cells):
    out = torch.zeros(
        n_cells, pre_w.shape[1], dtype=torch.float32, device=pre_w.device
    )
    for w, c, cell in zip(
        edge_window.tolist(), edge_class.tolist(), edge_cell.tolist()
    ):
        out[cell] += F.relu(pre_w[w].float() + table[c].float())
    return out


def _tensor_reference(
    pre_w, table, edge_window, edge_class, edge_cell, n_cells
):
    hidden = F.relu(
        pre_w.float().index_select(0, edge_window)
        + table.float().index_select(0, edge_class)
    )
    return torch.zeros(
        n_cells, pre_w.shape[1], dtype=torch.float32, device=pre_w.device
    ).index_add_(0, edge_cell, hidden)


def _random_views(case: str):
    rng = np.random.default_rng(71)
    n_windows, n_cells, n_classes = 11, 13, 23
    if case == "empty":
        empty = torch.empty(0, dtype=torch.int64, device=_DEVICE)
        return empty, empty, empty, empty, n_windows, n_cells, n_classes

    window_degree = np.zeros(n_windows, dtype=np.int64)
    windows, classes, cells = [], [], []
    # Only the first nine cells and eight windows receive entries. Sampling
    # without replacement per cell and capping window degree pins both run
    # bounds while leaving zero-degree destinations in each direction.
    for cell in range(9):
        candidates = np.flatnonzero(window_degree[:8] < 6)
        rng.shuffle(candidates)
        count = min(int(rng.integers(1, 8)), len(candidates))
        selected = candidates[:count]
        windows.extend(selected.tolist())
        classes.extend(rng.integers(0, n_classes, count).tolist())
        cells.extend([cell] * count)
        window_degree[selected] += 1

    edge_window = torch.tensor(windows, dtype=torch.int64, device=_DEVICE)
    edge_class = torch.tensor(classes, dtype=torch.int64, device=_DEVICE)
    edge_cell = torch.tensor(cells, dtype=torch.int64, device=_DEVICE)
    edge_rev = torch.argsort(edge_window.to(torch.int32), stable=True)
    assert torch.bincount(edge_cell, minlength=n_cells).max() <= 18
    assert torch.bincount(edge_window, minlength=n_windows).max() <= 6
    assert torch.equal(
        torch.bincount(edge_cell, minlength=n_cells)[9:],
        torch.zeros(n_cells - 9, dtype=torch.int64, device=_DEVICE),
    )
    assert torch.equal(
        torch.bincount(edge_window, minlength=n_windows)[8:],
        torch.zeros(n_windows - 8, dtype=torch.int64, device=_DEVICE),
    )
    return (
        edge_window,
        edge_class,
        edge_cell,
        edge_rev,
        n_windows,
        n_cells,
        n_classes,
    )


def _assert_cuda_parity(
    pre_w,
    table,
    edge_window,
    edge_class,
    edge_cell,
    edge_rev,
    n_cells,
    seed,
):
    generator = torch.Generator(device=_DEVICE).manual_seed(seed)
    upstream = torch.randint(
        -3,
        4,
        (n_cells, pre_w.shape[1]),
        device=_DEVICE,
        generator=generator,
    ).float()

    fast_pre = pre_w.detach().clone().requires_grad_()
    fast_table = table.detach().clone().requires_grad_()
    actual = encode_rows(
        fast_pre,
        fast_table,
        edge_window,
        edge_class,
        edge_cell,
        edge_rev,
        n_cells,
    )
    actual_grads = torch.autograd.grad(
        actual, (fast_pre, fast_table), grad_outputs=upstream
    )

    ref_pre = pre_w.detach().clone().requires_grad_()
    ref_table = table.detach().clone().requires_grad_()
    expected = _tensor_reference(
        ref_pre, ref_table, edge_window, edge_class, edge_cell, n_cells
    )
    expected_grads = torch.autograd.grad(
        expected, (ref_pre, ref_table), grad_outputs=upstream
    )

    tolerance = 2.0e-5 if pre_w.dtype == torch.float32 else 2.0e-2
    assert actual.dtype == expected.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert actual_grad.dtype == expected_grad.dtype
        assert torch.isfinite(actual_grad).all()
        torch.testing.assert_close(
            actual_grad, expected_grad, rtol=tolerance, atol=tolerance
        )


def test_reference_matches_a_hand_loop():
    torch.manual_seed(0)
    batch = _batch()
    n_w = int(batch.window_feat.shape[0])
    pre_w = torch.randn(n_w, 16)
    table = torch.randn(729, 16)
    got = encode_rows(
        pre_w,
        table,
        batch.dec_window,
        batch.act_class,
        batch.dec_cell,
        batch.act_rev,
        batch.n_cells,
    )
    want = _loop_reference(
        pre_w, table, batch.dec_window, batch.act_class, batch.dec_cell, batch.n_cells
    )
    assert got.dtype == torch.float32
    assert torch.allclose(got, want, atol=1e-6)


def test_validation_refuses_bad_edges():
    pre_w = torch.randn(4, 8)
    table = torch.randn(729, 8)
    edges = torch.zeros(3, dtype=torch.long)
    with pytest.raises(ValueError, match="one width"):
        encode_rows(pre_w, torch.randn(729, 4), edges, edges, edges, edges, 2)
    with pytest.raises(ValueError, match="int64"):
        encode_rows(pre_w, table, edges.int(), edges, edges, edges, 2)
    with pytest.raises(ValueError, match="one length"):
        encode_rows(pre_w, table, edges, edges[:2], edges, edges, 2)


@_CUDA_ONLY
@pytest.mark.parametrize("case", ["empty", "ragged"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_random_ragged_forward_and_gradients(case: str, dtype: torch.dtype):
    (
        edge_window,
        edge_class,
        edge_cell,
        edge_rev,
        n_windows,
        n_cells,
        n_classes,
    ) = _random_views(case)
    generator = torch.Generator(device=_DEVICE).manual_seed(81)
    pre_w = torch.randn(
        (n_windows, _H), device=_DEVICE, dtype=dtype, generator=generator
    )
    table = torch.randn(
        (n_classes, _H), device=_DEVICE, dtype=dtype, generator=generator
    )
    if case == "ragged":
        assert row_impl.triton is not None

    _assert_cuda_parity(
        pre_w,
        table,
        edge_window,
        edge_class,
        edge_cell,
        edge_rev,
        n_cells,
        seed=82,
    )


@_CUDA_ONLY
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_fixture_batch_forward_and_gradients(positions, dtype: torch.dtype):
    batch = collate(
        [from_position(pos, action_rows=True) for pos in positions]
    ).to(_DEVICE)
    generator = torch.Generator(device=_DEVICE).manual_seed(91)
    pre_w = torch.randn(
        (batch.window_feat.shape[0], _H),
        device=_DEVICE,
        dtype=dtype,
        generator=generator,
    )
    table = torch.randn(
        (729, _H), device=_DEVICE, dtype=dtype, generator=generator
    )
    assert row_impl.triton is not None
    assert batch.dec_window.numel() > 0

    _assert_cuda_parity(
        pre_w,
        table,
        batch.dec_window,
        batch.act_class,
        batch.dec_cell,
        batch.act_rev,
        batch.n_cells,
        seed=92,
    )
