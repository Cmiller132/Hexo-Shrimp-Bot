"""Shared decoder layout, folded head matrix, and kernel contracts.

The oracle transcribes the §6 decoder formula directly: project window rows,
add slot-class embeddings, sum each cell's entries, and overwrite background
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
def test_cell_heads_match_the_spec_decode(positions, model):
    """§6's logit and appendix B's dueling action value, both transcribed.

    Every readout of the fixture model is zero-initialized, which makes the
    critic's composition hold for any centering weight at all; the perturbed
    copy is what pins it. The scale keeps the advantage off tanh's saturated
    tail, where a difference of large numbers would compare nothing. The
    centering is a per-position Python loop here, not the segment helpers the
    head uses.
    """
    net = copy.deepcopy(model)
    torch.manual_seed(4)
    for out in (net.mlp_p.out, net.mlp_q.out, net.mlp_qbase[-1]):
        torch.nn.init.normal_(out.weight, std=0.1)
        torch.nn.init.normal_(out.bias, std=0.1)

    batch = _batch(positions)
    _s, w, g = net.trunk(batch)
    policy, q = net.cell_heads(w, g, batch)

    def decode(proj, e_class, e_bg, mlp):
        h = _spec_decoder_input(w, batch, proj.weight, e_class.weight, e_bg.weight)
        return mlp.out(
            F.relu(mlp.lin_a(h) + mlp.lin_b(g).index_select(0, batch.cell_pos))
        ).squeeze(-1)

    logits = decode(net.p, net.e_pw, net.e_bg, net.mlp_p)
    torch.testing.assert_close(policy, logits, rtol=1e-4, atol=1e-4)

    advantage = decode(net.q, net.e_qw, net.e_qbg, net.mlp_q)
    baseline = net.mlp_qbase(g).squeeze(-1)
    bounds = batch.legal_offsets.tolist()
    expected = []
    for i in range(batch.n_pos):
        low, high = bounds[i], bounds[i + 1]
        pi = logits[low:high].softmax(0)
        a = advantage[low:high]
        expected.append(torch.tanh(baseline[i] + a - (pi * a).sum()))
    torch.testing.assert_close(q, torch.cat(expected), rtol=1e-4, atol=1e-4)


@torch.no_grad()
def test_policy_head_matches_the_pair(positions, model):
    batch = _batch(positions)
    _s, w, g = model.trunk(batch)
    policy, _q = model.cell_heads(w, g, batch)
    assert torch.equal(model.policy_head(w, g, batch), policy)


def test_the_critic_error_never_reaches_the_policy_decoder(positions, model):
    """Appendix B's ``sg[·]`` on the centering weight, as a gradient contract.

    The critic centers on π_θ from its own forward, so the policy decoder is
    upstream of Q. Detached, it is not upstream of the *gradient*: the taken
    action's ``(Q - G)²`` trains the advantage, the baseline and the trunk, and
    reaches no parameter the policy logit alone owns. That is what keeps π
    meaning the policy its own cross-entropy trained, and it is the only thing
    here that changes if the stop-gradient goes.
    """
    net = copy.deepcopy(model)
    torch.manual_seed(6)
    for out in (net.mlp_p.out, net.mlp_q.out, net.mlp_qbase[-1]):
        torch.nn.init.normal_(out.weight, std=0.1)
        torch.nn.init.normal_(out.bias, std=0.1)

    batch = _batch(positions)
    assert batch.bg_cell.numel(), "the background tables are inert without a background cell"

    def grads_of(loss):
        net.zero_grad(set_to_none=True)
        logits, q = net.cell_heads(*net.trunk(batch)[1:], batch)
        loss(logits, q).backward()
        return {name: p.grad for name, p in net.named_parameters() if p.grad is not None}

    # The objective's critic term: one taken action per position, against targets
    # spread over the range so no position's contribution cancels another's.
    taken = batch.legal_offsets[:-1]
    returns = torch.linspace(-0.9, 0.9, taken.shape[0])
    critic = grads_of(lambda _logits, q: (q.index_select(0, taken) - returns).square().mean())
    policy = grads_of(lambda logits, _q: logits.sum())

    # The §6 decoder, parameter by parameter: nothing else produces the logit,
    # and the logit is the whole of what the critic reads them through.
    policy_only = {
        "p.weight",
        "e_pw.weight",
        "e_bg.weight",
        "mlp_p.lin_a.weight",
        "mlp_p.lin_a.bias",
        "mlp_p.lin_b.weight",
        "mlp_p.out.weight",
        "mlp_p.out.bias",
    }
    assert policy_only.isdisjoint(critic)
    # Every one of them is reachable from the logit, so the line above is a
    # stop-gradient and not a dead path.
    assert policy_only <= set(policy)
    assert all(torch.count_nonzero(policy[name]) > 0 for name in policy_only)
    # What the critic's error does train: both of the advantage's terms, the
    # baseline, and the shared trunk.
    for name in ("q.weight", "mlp_q.lin_a.weight", "mlp_q.out.weight",
                 "mlp_qbase.0.weight", "mlp_qbase.2.weight", "ln_out.weight"):
        assert name in critic and torch.count_nonzero(critic[name]) > 0, name


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
        # only if its fake kernel tracks the symbolic cell count. One position
        # is the case worth naming — the critic's centering reduces every legal
        # cell into a single row there, which is where a scatter lowering that
        # drops its bounds mask would quietly sum padding lanes too.
        for count in (1, 2, 5, len(positions)):
            batch = _batch(positions[:count]).to("cuda")
            eager = model.cell_heads(*model.trunk(batch)[1:], batch)
            got = compiled(model, batch)
            for a, b in zip(got, eager):
                torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)
    finally:
        model.to("cpu")
    assert not _FAILED_SHAPES, _FAILED_SHAPES
