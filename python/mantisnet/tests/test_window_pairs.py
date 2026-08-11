"""§5.1c pair-table oracle, fold invariance, and the window-attention model.

The oracle re-derives every relation pairwise from the identity triples —
literal line intersection per pair, no sorted joins — so the vectorized
derivation and the brute force share nothing but the definition.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import mantisnet.window_pairs as window_pairs
from mantisnet import MantisConfig, MantisNet, collate, from_position
from mantisnet.builder import AXES
from mantisnet.window_pairs import WA_CLASSES, edge_attention, wa_tables


def _fold(t: int) -> int:
    if 0 <= t <= 5:
        return min(t, 5 - t)
    return 2 + min(-t if t < 0 else t - 5, 3)


def _oracle_relation(a: tuple[int, int, int], b: tuple[int, int, int]):
    """The directed class of edge b -> a, or None. Brute force, one pair."""
    (ax_a, q_a, r_a), (ax_b, q_b, r_b) = a, b
    if ax_a == ax_b:
        # Same axis: related only when on one line, within offset 11.
        va = AXES[ax_a]
        # Colinear iff the start difference is an integer multiple of the axis.
        dq, dr = q_b - q_a, r_b - r_a
        if dq * int(va[1]) != dr * int(va[0]):
            return None
        offset = dq if int(va[0]) else dr
        if not 1 <= abs(offset) <= 11:
            return None
        return abs(offset) - 1
    # Crossing: solve start_a + t * v_a == start_b + u * v_b over the lattice.
    va, vb = AXES[ax_a], AXES[ax_b]
    det = int(va[0]) * int(vb[1]) - int(va[1]) * int(vb[0])
    assert det in (-1, 1), "hex axis pairs are unimodular"
    dq, dr = q_b - q_a, r_b - r_a
    t = (dq * int(vb[1]) - dr * int(vb[0])) // det
    u = -(int(va[0]) * dr - int(va[1]) * dq) // det
    assert (q_a + t * int(va[0]), r_a + t * int(va[1])) == (
        q_b + u * int(vb[0]),
        r_b + u * int(vb[1]),
    )
    if not (-5 <= t <= 10 and -5 <= u <= 10):
        return None
    return 11 + _fold(t) * 6 + _fold(u)


def test_forty_eight_classes():
    assert WA_CLASSES == 48  # 11 colinear + 6 * 6 crossing + SELF


def test_the_fold_is_invariant_under_line_reversal():
    # Reversal sends the span parameter t to 5 - t; each side of a crossing
    # class folds invariantly on its own, which is what makes the product
    # class D6-invariant under independent line reversals.
    for t in range(-5, 11):
        assert _fold(t) == _fold(5 - t)
    assert [_fold(t) for t in range(6)] == [0, 1, 2, 2, 1, 0]
    assert _fold(-1) == _fold(6) == 3
    assert _fold(-2) == _fold(7) == 4
    assert _fold(-5) == _fold(10) == 5


def test_pair_tables_match_the_brute_force(positions):
    for pos in positions:
        g = from_position(pos)
        ids = [tuple(map(int, row)) for row in g.window_id]
        n_w = len(ids)
        window_id = torch.from_numpy(g.window_id)
        tables = wa_tables(window_id, torch.zeros(n_w, dtype=torch.long))
        ptr, src, cls = window_pairs._expanded_edges(tables)

        got = set()
        for dst in range(n_w):
            for e in range(int(ptr[dst]), int(ptr[dst + 1])):
                got.add((dst, int(src[e]), int(cls[e])))

        expected = {(w, w, 47) for w in range(n_w)}
        for i in range(n_w):
            for j in range(n_w):
                if i == j:
                    continue
                relation = _oracle_relation(ids[i], ids[j])
                if relation is not None:
                    expected.add((i, j, relation))
        assert got == expected


def test_the_claim_views_bound_and_cover_the_relations(positions):
    g = from_position(positions[8])
    n_w = g.n_windows
    tables = wa_tables(
        torch.from_numpy(g.window_id), torch.zeros(n_w, dtype=torch.long)
    )
    # Structure: sixteen claims per window, run bounds inside the claimant
    # list, and the hard degree bounds the kernel tiles rely on — at most
    # 48 claimants per cell (sixteen spans on each of three axes), at most
    # 22 colinear partners plus SELF.
    assert tables.claim_lo.shape == tables.claim_hi.shape == (n_w * 16,)
    assert tables.cl_win.shape == tables.cl_fold.shape == (n_w * 16,)
    assert torch.all(tables.claim_lo >= 0)
    assert torch.all(tables.claim_hi <= n_w * 16)
    assert torch.all(tables.claim_hi - tables.claim_lo >= 1)
    assert torch.all(tables.claim_hi - tables.claim_lo <= 48)
    col_degree = tables.col_ptr[1:] - tables.col_ptr[:-1]
    assert torch.all(col_degree >= 1) and torch.all(col_degree <= 23)
    assert torch.all((tables.col_cls <= 10) | (tables.col_cls == 47))
    assert torch.all((tables.cl_fold >= 0) & (tables.cl_fold <= 5))

    # Every claim's own window sits in its run (same axis, so the kernels
    # filter it), pinning each run to the right cell.
    owner = torch.arange(n_w).repeat_interleave(16)
    for c in range(n_w * 16):
        run = tables.cl_win[int(tables.claim_lo[c]) : int(tables.claim_hi[c])]
        assert bool((run == owner[c]).any())


def _batched_tables(graphs):
    window_id = torch.from_numpy(np.concatenate([g.window_id for g in graphs]))
    window_pos = torch.repeat_interleave(
        torch.arange(len(graphs)), torch.tensor([g.n_windows for g in graphs])
    )
    return wa_tables(window_id, window_pos)


def test_pairs_never_cross_positions(positions):
    graphs = [from_position(pos) for pos in positions if from_position(pos).n_windows]
    ptr, src, cls = window_pairs._expanded_edges(_batched_tables(graphs))
    # Stack the same graphs as one batch; every edge must stay inside its
    # position's window range.
    offsets = [0]
    for g in graphs:
        offsets.append(offsets[-1] + g.n_windows)
    dst = torch.repeat_interleave(torch.arange(offsets[-1]), ptr[1:] - ptr[:-1])
    lo = torch.tensor(offsets[:-1])
    hi = torch.tensor(offsets[1:])
    position = torch.searchsorted(hi, dst, right=True)
    assert torch.all(src >= lo[position])
    assert torch.all(src < hi[position])

    # Single-position tables agree with the batched ones for the first graph,
    # per destination as sets: the join visits edges in a different order when
    # other positions interleave the sort.
    sptr, ssrc, scls = window_pairs._expanded_edges(_batched_tables(graphs[:1]))
    n0 = graphs[0].n_windows
    assert torch.equal(sptr, ptr[: n0 + 1])
    for w in range(n0):
        a, b = int(ptr[w]), int(ptr[w + 1])
        got = set(zip(src[a:b].tolist(), cls[a:b].tolist()))
        want = set(zip(ssrc[a:b].tolist(), scls[a:b].tolist()))
        assert got == want


def test_every_window_carries_exactly_one_self_loop():
    g = from_position(__import__("hexo_py").Position.replay([(0, 0), (1, 0), (0, 1)]))
    ptr, src, cls = window_pairs._expanded_edges(_batched_tables([g]))
    counts = ptr[1:] - ptr[:-1]
    assert torch.all(counts >= 1)
    self_edges = (
        src == torch.repeat_interleave(torch.arange(g.n_windows), counts)
    ) & (cls == 47)
    assert int(self_edges.sum()) == g.n_windows


def _naive_attention(q, k, v, bias, ptr, src, cls):
    """Materialized per-edge attention: the oracle the sliced op must match."""
    n_w, _heads, hd = q.shape
    dst = torch.repeat_interleave(torch.arange(n_w), ptr[1:] - ptr[:-1])
    score = (q[dst].float() * k[src].float()).sum(-1) / math.sqrt(hd)
    score = score + bias.t().float()[cls]
    out = q.new_zeros(q.shape, dtype=torch.float32)
    for w in range(n_w):
        edges = slice(int(ptr[w]), int(ptr[w + 1]))
        weights = torch.softmax(score[edges], dim=0)  # (degree, heads)
        out[w] = (weights.unsqueeze(-1) * v[src[edges]].float()).sum(0)
    return out


def _attention_case(positions, seed: int):
    torch.manual_seed(seed)
    g = from_position(positions[8])
    tables = _batched_tables([g])
    heads, hd = 2, 8
    make = lambda *shape: torch.randn(*shape, requires_grad=True)  # noqa: E731
    return (
        make(g.n_windows, heads, hd),
        make(g.n_windows, heads, hd),
        make(g.n_windows, heads, hd),
        make(heads, WA_CLASSES),
        tables,
    )


def test_edge_attention_matches_the_naive_oracle(positions):
    q, k, v, bias, tables = _attention_case(positions, seed=21)
    got = edge_attention(q, k, v, bias, *tables)
    want = _naive_attention(q, k, v, bias, *window_pairs._expanded_edges(tables))
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_edge_attention_slices_and_gradients_match_the_oracle(positions, monkeypatch):
    # A tiny slice forces the multi-slice path the fallback takes on real
    # batches; the backward re-derives alpha from the saved stats, so its
    # parity is the recompute proof.
    monkeypatch.setattr(window_pairs, "_EDGE_SLICE", 3)
    q, k, v, bias, tables = _attention_case(positions, seed=22)
    upstream = torch.randn(q.shape)

    got = edge_attention(q, k, v, bias, *tables)
    got_grads = torch.autograd.grad((got * upstream).sum(), (q, k, v, bias))
    want = _naive_attention(q, k, v, bias, *window_pairs._expanded_edges(tables))
    want_grads = torch.autograd.grad((want * upstream).sum(), (q, k, v, bias))

    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
    for name, a, b in zip(("dq", "dk", "dv", "dbias"), got_grads, want_grads):
        assert torch.isfinite(a).all(), name
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)


_needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the fused window attention requires CUDA"
)


@_needs_cuda
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_fused_kernels_match_the_oracle(positions, dtype):
    q, k, v, bias, tables = _attention_case(positions, seed=31)
    device = tuple(t.to("cuda") for t in tables)
    qd = q.detach().to("cuda").to(dtype).requires_grad_()
    kd = k.detach().to("cuda").to(dtype).requires_grad_()
    vd = v.detach().to("cuda").to(dtype).requires_grad_()
    bd = bias.detach().to("cuda").requires_grad_()
    upstream = torch.randn(q.shape)

    got = edge_attention(qd, kd, vd, bd, *device)
    got_grads = torch.autograd.grad(
        (got * upstream.to("cuda")).sum(), (qd, kd, vd, bd)
    )

    # The oracle starts from the identical (dtype-rounded) values in fp32.
    qr = qd.detach().float().cpu().requires_grad_()
    kr = kd.detach().float().cpu().requires_grad_()
    vr = vd.detach().float().cpu().requires_grad_()
    br = bd.detach().cpu().requires_grad_()
    want = _naive_attention(qr, kr, vr, br, *window_pairs._expanded_edges(tables))
    want_grads = torch.autograd.grad((want * upstream).sum(), (qr, kr, vr, br))

    # dq/dk/dv are stored in the input dtype, so bf16 compares at bf16 grain.
    loose = dtype is torch.bfloat16
    torch.testing.assert_close(
        got.cpu(), want, rtol=1e-2 if loose else 1e-5, atol=1e-2 if loose else 1e-5
    )
    for name, a, b in zip(("dq", "dk", "dv", "dbias"), got_grads, want_grads):
        assert torch.isfinite(a.float()).all(), name
        torch.testing.assert_close(
            a.float().cpu(),
            b.float(),
            rtol=2e-2 if loose else 1e-4,
            atol=2e-2 if loose else 1e-4,
        )


@_needs_cuda
def test_fused_kernels_are_deterministic(positions):
    q, k, v, bias, tables = _attention_case(positions, seed=32)
    device = tuple(t.to("cuda") for t in tables)
    qd = q.detach().to("cuda", torch.bfloat16).requires_grad_()
    kd = k.detach().to("cuda", torch.bfloat16).requires_grad_()
    vd = v.detach().to("cuda", torch.bfloat16).requires_grad_()
    bd = bias.detach().to("cuda").requires_grad_()
    upstream = torch.randn(q.shape, device="cuda")

    runs = []
    for _ in range(2):
        out = edge_attention(qd, kd, vd, bd, *device)
        runs.append(
            (out, *torch.autograd.grad((out * upstream).sum(), (qd, kd, vd, bd)))
        )
    for name, a, b in zip(("out", "dq", "dk", "dv", "dbias"), runs[0], runs[1]):
        assert torch.equal(a, b), name


def _small_wa_config(**extra) -> MantisConfig:
    return MantisConfig(
        h=16,
        blocks=2,
        heads=2,
        value_queries=2,
        value_bins=5,
        policy_hidden=16,
        value_hidden=16,
        **extra,
    )


@torch.no_grad()
def test_window_attention_forward_and_batch_parity(positions):
    torch.manual_seed(11)
    net = MantisNet(_small_wa_config()).eval()
    graphs = [from_position(p) for p in positions]
    batch = collate(graphs)
    together = net(batch, 0.2)
    for tensor in vars(together).values():
        assert torch.isfinite(tensor).all()

    for i, g in enumerate(graphs):
        single = net(collate([g]), 0.2)
        a, b = int(batch.legal_offsets[i]), int(batch.legal_offsets[i + 1])
        assert torch.allclose(
            together.policy_logits[a:b], single.policy_logits, atol=1e-6
        )
        assert torch.allclose(
            together.value[i : i + 1], single.value, atol=1e-6
        )


@torch.no_grad()
def test_window_attention_is_d6_invariant(positions, move_lists):
    import hexo_py

    from mantisnet.klent import telemetry

    torch.manual_seed(12)
    net = MantisNet(_small_wa_config()).eval()
    for moves in (move_lists[6], move_lists[8]):
        pos = hexo_py.Position.replay(moves)
        base = net(collate([from_position(pos)]), 0.2)
        base_policy = dict(zip(pos.legal_moves(), base.policy_logits.tolist()))
        for transform in telemetry.D6_TRANSFORMS[1:]:
            t_pos = hexo_py.Position.replay([transform(m) for m in moves])
            got = net(collate([from_position(t_pos)]), 0.2)
            assert torch.allclose(got.value, base.value, atol=1e-5)
            mapped = dict(zip(t_pos.legal_moves(), got.policy_logits.tolist()))
            for move, logit in base_policy.items():
                assert abs(mapped[transform(move)] - logit) <= 1e-5


def test_window_attention_gradients_reach_every_table(positions):
    torch.manual_seed(13)
    net = MantisNet(_small_wa_config())
    batch = collate([from_position(positions[8])])
    out = net(batch, 0.2)
    (out.value_logits.sum() + out.policy_logits.sum()).backward()
    for name in ("wq_wa", "wk_wa", "wv_wa", "wo_wa"):
        grad = getattr(net.blocks[0], name).weight.grad
        assert grad is not None and torch.isfinite(grad).all()
    bias_grad = net.blocks[0].wa_bias.grad
    assert bias_grad is not None and torch.isfinite(bias_grad).all()
