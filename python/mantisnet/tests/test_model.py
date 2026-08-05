"""§12.4 batching equivalence, output contracts, and the CUDA smoke test."""

from __future__ import annotations

import hexo_py
import pytest
import torch
import torch.nn.functional as F

from mantisnet import MantisConfig, MantisNet, collate, from_position
from mantisnet.lab.variants import count_parameters
from mantisnet.model import (
    CRITIC_LOGITS,
    compose_acting_q,
    compose_q,
    return_mass,
)


def _critic_model() -> MantisNet:
    """A small model whose critic readout is not a fresh init's zero rows."""
    torch.manual_seed(2)
    net = MantisNet(
        MantisConfig(
            h=32, blocks=1, heads=2, value_queries=2, value_bins=5,
            policy_hidden=32, value_hidden=32,
        )
    )
    with torch.no_grad():
        net.mlp_q.out.weight.normal_(std=0.5)
        net.mlp_q.out.bias.normal_(std=0.5)
    return net.eval()


@torch.no_grad()
def test_batched_forward_equals_per_position(model, positions):
    graphs = [from_position(p) for p in positions]
    batched = model(collate(graphs), 0.2)
    offset = 0
    for i, g in enumerate(graphs):
        single = model(collate([g]), 0.2)
        n = g.n_legal
        assert torch.allclose(
            batched.policy_logits[offset : offset + n], single.policy_logits, atol=1e-6
        )
        assert torch.allclose(
            batched.q_values[offset : offset + n], single.q_values, atol=1e-6
        )
        assert torch.allclose(
            batched.q_score[offset : offset + n], single.q_score, atol=1e-6
        )
        assert torch.allclose(batched.value[i : i + 1], single.value, atol=1e-6)
        assert torch.allclose(batched.value_dist[i : i + 1], single.value_dist, atol=1e-6)
        offset += n
    assert offset == batched.policy_logits.shape[0]


@torch.no_grad()
def test_output_contracts(model, positions):
    graphs = [from_position(p) for p in positions]
    batch = collate(graphs)
    out = model(batch, 0.2)
    assert out.policy_logits.shape == (sum(g.n_legal for g in graphs),)
    assert out.q_values.shape == out.policy_logits.shape
    assert out.q_score.shape == out.policy_logits.shape
    assert torch.isfinite(out.q_values).all()
    assert batch.legal_offsets.tolist() == [0] + list(
        torch.tensor([g.n_legal for g in graphs]).cumsum(0).tolist()
    )
    assert out.value.shape == (len(graphs),)
    assert torch.all((out.value >= -1) & (out.value <= 1))
    assert torch.allclose(out.value_dist.sum(-1), torch.ones(len(graphs)), atol=1e-6)
    # The scalar is the distribution's decode — the same value every consumer sees.
    assert torch.allclose(out.value, out.value_dist @ model.bin_centers, atol=1e-6)
    assert torch.isfinite(out.policy_logits).all()


@torch.no_grad()
def test_action_values_are_the_categorical_return_masses_composed(positions):
    """Appendix B's composition, written out against the raw readout rows."""
    net = _critic_model()
    batch = collate([from_position(p) for p in positions])
    _s, w, g = net.trunk(batch)
    policy_logits, _score, q = net.cell_heads(w, g, batch, 0.2)
    raw_policy, critic_logits = net.cell_head_logits(w, g, batch)

    assert torch.equal(raw_policy, policy_logits)
    assert critic_logits.shape == (q.shape[0], CRITIC_LOGITS)
    # The reference is in double, so the tolerance is fp32 rounding of the
    # composition, not slack in the formula.
    probs = critic_logits.double().softmax(dim=-1)
    torch.testing.assert_close(q.double(), probs[:, 0] - probs[:, 1], rtol=0, atol=1e-6)
    p_pos, p_neg = return_mass(critic_logits)
    torch.testing.assert_close(
        p_pos + p_neg + critic_logits.float().softmax(dim=-1)[:, 2],
        torch.ones_like(q),
    )
    # The simplex bounds both Q and committed mass whatever the logits.
    assert q.abs().max() < 1.0
    assert q.abs().max() > 0.1
    assert ((p_pos + p_neg) < 1.0).all()
    assert q.dtype == torch.float32


@torch.no_grad()
def test_acting_score_is_q_over_the_position_maximum_committed_mass():
    logits = torch.tensor(
        [
            [3.0, -3.0, -2.0],
            [0.4, -1.2, 0.7],
            [-1.5, 0.2, 0.5],
            [-3.0, -3.4, 2.0],
            [-3.6, -3.0, 2.1],
            [-3.2, -3.2, 2.2],
        ]
    )
    offsets = torch.tensor([0, 3, 6])
    q = compose_q(logits)
    score = compose_acting_q(logits, offsets, 0.0)
    probs = logits.double().softmax(dim=-1)
    mass = probs[:, :2].sum(dim=-1)
    for lo, hi in ((0, 3), (3, 6)):
        torch.testing.assert_close(
            score[lo:hi].double(),
            q[lo:hi].double() / mass[lo:hi].max(),
            rtol=0,
            atol=1e-6,
        )
        assert torch.equal(
            torch.argsort(score[lo:hi], stable=True),
            torch.argsort(q[lo:hi], stable=True),
        )
    assert score.abs().max() < 1.0


def test_categorical_ce_optimum_recovers_expected_return():
    """Two observations of one (s,a): the proper-score optimum composes E[G]."""
    returns = torch.tensor([1.0, -0.5])
    target = torch.stack(
        (returns.clamp(min=0), (-returns).clamp(min=0), 1 - returns.abs()),
        dim=-1,
    )
    logits = torch.nn.Parameter(torch.zeros(3))
    optimizer = torch.optim.LBFGS([logits], lr=1.0, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = -(target * logits.log_softmax(dim=-1)).sum(dim=-1).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    torch.testing.assert_close(
        compose_q(logits), returns.mean(), rtol=0, atol=1e-5
    )


@torch.no_grad()
def test_fresh_init_policy_logits_and_action_values_are_exactly_zero(positions):
    """The appendix-B init override: both decoder readouts start at zero, so
    both mass logits vanish and every action value is exactly zero."""
    torch.manual_seed(0)
    net = MantisNet(MantisConfig()).eval()
    out = net(collate([from_position(p) for p in positions]), 0.2)
    assert torch.count_nonzero(out.policy_logits) == 0
    assert torch.count_nonzero(out.q_values) == 0


def test_dropout_config_runs_and_eval_is_deterministic(positions):
    torch.manual_seed(1)
    net = MantisNet(MantisConfig(dropout=0.1))
    batch = collate([from_position(positions[-1])])
    net.train()
    net(batch, 0.2)  # The CUDA smoke test requires successful execution.
    net.eval()
    with torch.no_grad():
        a, b = net(batch, 0.2), net(batch, 0.2)
    assert torch.equal(a.policy_logits, b.policy_logits)
    assert torch.equal(a.value, b.value)


@torch.no_grad()
def test_cell_pass_matches_literal_incidence_reference():
    # P0's stones occupy two intersecting axes; the remote P1 stones only
    # advance the turn so P0 can place both of them.
    pos = hexo_py.Position.replay(
        [(0, 0), (-8, 8), (-8, 9), (1, 0), (0, 1)]
    )
    batch = collate([from_position(pos)])
    shared = torch.bincount(
        batch.dec_cell, minlength=batch.cell_pos.shape[0]
    )
    assert (shared >= 2).any()

    torch.manual_seed(19)
    cfg = MantisConfig(
        h=16,
        blocks=1,
        heads=2,
        value_queries=2,
        value_bins=5,
        policy_hidden=16,
        value_hidden=16,
    )
    block = MantisNet(cfg).blocks[0].eval()
    w = torch.randn(batch.window_feat.shape[0], cfg.h)

    x = block.u_cp(block.ln_cp_in(w))
    cells = torch.zeros(batch.cell_pos.shape[0], cfg.h, dtype=torch.float32)
    entries = zip(
        batch.dec_window.tolist(),
        batch.dec_class.tolist(),
        batch.dec_cell.tolist(),
    )
    for window, cls, cell in entries:
        cells[cell] += (x[window] + block.e_cp.weight[cls]).float()
    cells = F.relu(cells)
    agg = torch.zeros(w.shape[0], cfg.h, dtype=torch.float32)
    for window, cell in zip(batch.dec_window.tolist(), batch.dec_cell.tolist()):
        agg[window] += cells[cell]
    expected = w + block.drop(block.mlp_cp(block.ln_cp_w(w), agg.to(w.dtype)))

    actual = block._cell_pass(
        w,
        batch.relay_cell_ptr,
        batch.relay_window,
        batch.relay_class,
        batch.relay_win_ptr,
        batch.relay_wcell,
        batch.relay_cls_ptr,
        batch.relay_ccell,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@torch.no_grad()
def test_default_model_runs_and_has_expected_parameter_count(positions):
    torch.manual_seed(29)
    net = MantisNet(MantisConfig()).eval()
    batch = collate([from_position(positions[3])])
    out = net(batch, 0.2)

    assert count_parameters(net) == 1_944_165
    for tensor in vars(out).values():
        assert torch.isfinite(tensor).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
@torch.no_grad()
def test_cuda_bf16_smoke(model, positions):
    device = torch.device("cuda")
    net = model.to(device)
    try:
        batch = collate([from_position(p) for p in positions]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(batch, 0.2)
        assert torch.isfinite(out.policy_logits).all()
        assert torch.isfinite(out.value).all()
        assert out.value_dist.dtype == torch.float32
        assert torch.allclose(
            out.value_dist.sum(-1), torch.ones(batch.n_pos, device=device), atol=1e-3
        )
    finally:
        net.to("cpu")
