"""Step 15 cell-latent oracles: table derivations against pairwise brute
force, typed attention against a dense per-segment composition, and the
line pass through §5.1c's edge attention.

The attention oracle loops query rows in Python and uses dense softmax
slices, so the vectorized scatter composition and the oracle share nothing
but the definition. CUDA parity of the Triton kernels against the eager
reference rides the same tests through the op's dispatch.
"""

from __future__ import annotations

import math

import pytest
import torch

import mantisnet.cell_latents as cell_latents
from mantisnet import from_position
from mantisnet.builder import AXES, TERN_DEC_CLASSES
from mantisnet.cell_latents import (
    LINE_CLASSES,
    LINE_FAR,
    LINE_SELF,
    CellTables,
    cell_read,
    cell_tables,
    line_tables,
    window_read,
)
from mantisnet.window_pairs import edge_attention


def test_line_class_layout():
    assert LINE_CLASSES == 13  # offsets 1..11, FAR, SELF
    assert LINE_FAR == 11 and LINE_SELF == 12


def _synthetic_incidence(seed: int, n_cells: int = 14, n_windows: int = 12):
    """A gappy incidence: some cells uncovered, some windows entryless."""
    gen = torch.Generator().manual_seed(seed)
    e = 48
    dec_cell = torch.randint(0, n_cells, (e,), generator=gen)
    # Windows 0 and 7 stay entryless: the "full window" case.
    windows = torch.tensor([w for w in range(n_windows) if w not in (0, 7)])
    dec_window = windows[torch.randint(0, windows.numel(), (e,), generator=gen)]
    dec_class = torch.randint(0, TERN_DEC_CLASSES, (e,), generator=gen)
    return dec_cell, dec_window, dec_class, n_windows


def test_cell_tables_match_the_brute_force(positions):
    for pos in positions[:6]:
        g = from_position(pos)
        dec_cell = torch.from_numpy(g.dec_cell)
        dec_window = torch.from_numpy(g.dec_window)
        dec_class = torch.from_numpy(g.dec_class)
        tables = cell_tables(
            dec_cell, dec_window, dec_class, g.n_windows, TERN_DEC_CLASSES
        )
        _check_tables_against(tables, dec_cell, dec_window, dec_class, g.n_windows)


def test_cell_tables_compact_gappy_incidence():
    dec_cell, dec_window, dec_class, n_windows = _synthetic_incidence(3)
    tables = cell_tables(dec_cell, dec_window, dec_class, n_windows, TERN_DEC_CLASSES)
    _check_tables_against(tables, dec_cell, dec_window, dec_class, n_windows)
    # Entryless windows have empty runs.
    for w in (0, 7):
        assert int(tables.win_ptr[w]) == int(tables.win_ptr[w + 1])


def _check_tables_against(
    tables: CellTables, dec_cell, dec_window, dec_class, n_windows
) -> None:
    e = dec_cell.numel()
    covered = sorted(set(dec_cell.tolist()))
    assert tables.covered.tolist() == covered
    compact = {legal: i for i, legal in enumerate(covered)}
    raw = sorted(
        (compact[int(c)], int(w), int(k))
        for c, w, k in zip(dec_cell, dec_window, dec_class)
    )

    # Cell-major view: runs per covered cell, every entry present once.
    got = []
    for cell in range(len(covered)):
        for i in range(int(tables.cell_ptr[cell]), int(tables.cell_ptr[cell + 1])):
            assert int(tables.edge_cell[i]) == cell
            got.append((cell, int(tables.edge_window[i]), int(tables.edge_class[i])))
    assert sorted(got) == raw
    assert all(
        int(tables.cell_ptr[i + 1]) > int(tables.cell_ptr[i])
        for i in range(len(covered))
    ), "covered cells have at least one entry by construction"

    # Window-major view holds the same multiset.
    got_w = []
    for w in range(n_windows):
        for i in range(int(tables.win_ptr[w]), int(tables.win_ptr[w + 1])):
            assert int(tables.edge_wwin[i]) == w
            got_w.append((int(tables.edge_wcell[i]), w, int(tables.edge_wclass[i])))
    assert sorted(got_w) == raw

    # Class views: permutations of their edge orders with nondecreasing
    # classes and cls_ptr at the boundaries.
    for cedge, classes in (
        (tables.cedge_cell, tables.edge_class),
        (tables.cedge_win, tables.edge_wclass),
    ):
        assert torch.equal(cedge.sort().values, torch.arange(e))
        ordered = classes[cedge]
        assert bool((ordered[1:] >= ordered[:-1]).all())
        for cls in range(TERN_DEC_CLASSES):
            lo, hi = int(tables.cls_ptr[cls]), int(tables.cls_ptr[cls + 1])
            assert bool((ordered[lo:hi] == cls).all())


def _oracle_line_class(a, b):
    """The directed line class of edge b -> a, or None. One pair, literal."""
    (ax_a, q_a, r_a), (ax_b, q_b, r_b) = a, b
    if ax_a != ax_b:
        return None
    va = AXES[ax_a]
    dq, dr = q_b - q_a, r_b - r_a
    if dq * int(va[1]) != dr * int(va[0]):
        return None
    offset = dq if int(va[0]) else dr
    if offset == 0:
        return None
    return abs(offset) - 1 if abs(offset) <= 11 else LINE_FAR


def test_line_tables_match_the_brute_force(positions):
    for pos in positions[:6]:
        g = from_position(pos)
        ids = [tuple(map(int, row)) for row in g.window_id]
        n_w = len(ids)
        tables = line_tables(
            torch.from_numpy(g.window_id), torch.zeros(n_w, dtype=torch.long)
        )
        got = set()
        dst = torch.repeat_interleave(
            torch.arange(n_w), tables.ptr[1:] - tables.ptr[:-1]
        )
        for d, s, c in zip(dst.tolist(), tables.src.tolist(), tables.cls.tolist()):
            got.add((d, s, c))

        expected = {(w, w, LINE_SELF) for w in range(n_w)}
        for i in range(n_w):
            for j in range(n_w):
                if i == j:
                    continue
                cls = _oracle_line_class(ids[i], ids[j])
                if cls is not None:
                    expected.add((i, j, cls))
        assert got == expected


def test_line_tables_far_offsets_and_position_isolation():
    # One line per axis with a far member, duplicated in a second position:
    # offsets 5 and 15 give classes 4 and FAR, and nothing crosses positions.
    window_id = torch.tensor(
        [[0, 0, 0], [0, 5, 0], [0, 20, 0], [1, 3, -2], [1, 3, 10]] * 2,
        dtype=torch.long,
    )
    window_pos = torch.tensor([0] * 5 + [1] * 5, dtype=torch.long)
    tables = line_tables(window_id, window_pos)
    n_w = 10
    dst = torch.repeat_interleave(torch.arange(n_w), tables.ptr[1:] - tables.ptr[:-1])
    got = set(zip(dst.tolist(), tables.src.tolist(), tables.cls.tolist()))
    per_position = [
        (0, 1, 4),
        (1, 0, 4),
        (0, 2, LINE_FAR),
        (2, 0, LINE_FAR),
        (1, 2, LINE_FAR),
        (2, 1, LINE_FAR),
        (3, 4, LINE_FAR),
        (4, 3, LINE_FAR),
    ]
    expected = {(w, w, LINE_SELF) for w in range(n_w)}
    for d, s, c in per_position:
        expected.add((d, s, c))
        expected.add((d + 5, s + 5, c))
    assert got == expected


def test_line_views_are_symmetric_and_consistent():
    window_id = torch.tensor(
        [[0, 0, 0], [0, 3, 0], [0, 7, 0], [2, 1, 1], [2, 4, -2]], dtype=torch.long
    )
    tables = line_tables(window_id, torch.zeros(5, dtype=torch.long))
    # Line classes are reversal-symmetric, so the source view is the
    # destination view outright.
    assert tables.sptr is tables.ptr and tables.sdst is tables.src
    assert tables.scls is tables.cls
    e = tables.src.shape[0]
    assert torch.equal(tables.cedge.sort().values, torch.arange(e))
    ordered = tables.cls[tables.cedge]
    assert bool((ordered[1:] >= ordered[:-1]).all())


def _compose(q, k, v, bias, vcls, seg_ptr, src, cls):
    """Dense per-segment softmax oracle, differentiable, fp32."""
    heads, hd = q.shape[1], q.shape[2]
    scale = 1.0 / math.sqrt(hd)
    rows = []
    for i in range(seg_ptr.numel() - 1):
        lo, hi = int(seg_ptr[i]), int(seg_ptr[i + 1])
        if lo == hi:
            rows.append(q.new_zeros((heads, hd), dtype=torch.float32))
            continue
        s, c = src[lo:hi], cls[lo:hi]
        kk = k.float()[s]
        vv = v.float()[s]
        if vcls is not None:
            vv = vv + vcls.float()[c].view(-1, heads, hd)
        scores = (q[i].float()[None, :, :] * kk).sum(-1) * scale
        scores = scores + bias.float().t()[c]
        alpha = scores.softmax(0)
        rows.append((alpha.unsqueeze(-1) * vv).sum(0))
    return torch.stack(rows)


def _random_attention_case(seed: int, device: str = "cpu"):
    dec_cell, dec_window, dec_class, n_windows = _synthetic_incidence(seed)
    tables = CellTables(
        *(
            t.to(device)
            for t in cell_tables(
                dec_cell, dec_window, dec_class, n_windows, TERN_DEC_CLASSES
            )
        )
    )
    gen = torch.Generator().manual_seed(seed + 100)
    heads, hd = 4, 8
    n_cov = tables.covered.numel()

    def leaf(*shape):
        return (
            torch.randn(*shape, generator=gen).to(device).requires_grad_(True)
        )

    q_c = leaf(n_cov, heads, hd)
    k_w = leaf(n_windows, heads, hd)
    v_w = leaf(n_windows, heads, hd)
    q_w = leaf(n_windows, heads, hd)
    k_c = leaf(n_cov, heads, hd)
    v_c = leaf(n_cov, heads, hd)
    bias = leaf(heads, TERN_DEC_CLASSES)
    vcls = leaf(TERN_DEC_CLASSES, heads * hd)
    return tables, q_c, k_w, v_w, q_w, k_c, v_c, bias, vcls


@pytest.mark.parametrize("seed", [0, 1])
def test_cell_read_matches_the_dense_oracle(seed):
    tables, q_c, k_w, v_w, _q_w, _k_c, _v_c, bias, vcls = _random_attention_case(seed)
    out = cell_read(q_c, k_w, v_w, bias, vcls, tables)
    ref = _compose(
        q_c, k_w, v_w, bias, vcls, tables.cell_ptr, tables.edge_window, tables.edge_class
    )
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    upstream = torch.randn_like(ref)
    grads = torch.autograd.grad(out, (q_c, k_w, v_w, bias, vcls), upstream)
    ref_grads = torch.autograd.grad(ref, (q_c, k_w, v_w, bias, vcls), upstream)
    for got, expect in zip(grads, ref_grads):
        torch.testing.assert_close(got, expect, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("seed", [0, 1])
def test_window_read_matches_the_dense_oracle(seed):
    tables, _q_c, _k_w, _v_w, q_w, k_c, v_c, bias, _vcls = _random_attention_case(seed)
    out = window_read(q_w, k_c, v_c, bias, tables)
    ref = _compose(
        q_w, k_c, v_c, bias, None, tables.win_ptr, tables.edge_wcell, tables.edge_wclass
    )
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    upstream = torch.randn_like(ref)
    grads = torch.autograd.grad(out, (q_w, k_c, v_c, bias), upstream)
    ref_grads = torch.autograd.grad(ref, (q_w, k_c, v_c, bias), upstream)
    for got, expect in zip(grads, ref_grads):
        torch.testing.assert_close(got, expect, atol=1e-4, rtol=1e-4)


def test_entryless_windows_read_zero_without_nan():
    tables, _q_c, _k_w, _v_w, q_w, k_c, v_c, bias, _vcls = _random_attention_case(5)
    out = window_read(q_w, k_c, v_c, bias, tables)
    for w in (0, 7):
        assert bool((out[w] == 0).all())
    upstream = torch.randn_like(out)
    grads = torch.autograd.grad(out, (q_w, k_c, v_c, bias), upstream)
    for g in grads:
        assert bool(torch.isfinite(g).all())
    assert bool((grads[0][0] == 0).all()) and bool((grads[0][7] == 0).all())


def test_line_pass_through_edge_attention_matches_dense(positions):
    g = from_position(positions[8])
    n_w = g.n_windows
    tables = line_tables(
        torch.from_numpy(g.window_id), torch.zeros(n_w, dtype=torch.long)
    )
    gen = torch.Generator().manual_seed(11)
    heads, hd = 4, 8
    q = torch.randn(n_w, heads, hd, generator=gen)
    k = torch.randn(n_w, heads, hd, generator=gen)
    v = torch.randn(n_w, heads, hd, generator=gen)
    bias = torch.randn(heads, LINE_CLASSES, generator=gen)
    out = edge_attention(q, k, v, bias, *tables)
    ref = _compose(q, k, v, bias, None, tables.ptr, tables.src, tables.cls)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
class TestCuda:
    def test_kernel_matches_reference_both_directions(self):
        tables, q_c, k_w, v_w, q_w, k_c, v_c, bias, vcls = _random_attention_case(
            7, device="cuda"
        )
        out = cell_read(q_c, k_w, v_w, bias, vcls, tables)
        ref = cell_latents._reference_forward(
            q_c.detach(),
            k_w.detach(),
            v_w.detach(),
            bias.detach(),
            vcls.detach(),
            tables.edge_window,
            tables.edge_class,
            tables.edge_cell,
        )[0]
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

        out_w = window_read(q_w, k_c, v_c, bias, tables)
        ref_w = cell_latents._reference_forward(
            q_w.detach(),
            k_c.detach(),
            v_c.detach(),
            bias.detach(),
            None,
            tables.edge_wcell,
            tables.edge_wclass,
            tables.edge_wwin,
        )[0]
        torch.testing.assert_close(out_w, ref_w, atol=1e-5, rtol=1e-5)

    def test_backward_is_deterministic(self):
        tables, q_c, k_w, v_w, _q_w, _k_c, _v_c, bias, vcls = _random_attention_case(
            9, device="cuda"
        )
        upstream = torch.randn(
            tables.covered.numel(), 4, 8, device="cuda", dtype=torch.float32
        )

        def run():
            out = cell_read(q_c, k_w, v_w, bias, vcls, tables)
            return torch.autograd.grad(out, (q_c, k_w, v_w, bias, vcls), upstream)

        first = run()
        second = run()
        for a, b in zip(first, second):
            assert torch.equal(a, b)
