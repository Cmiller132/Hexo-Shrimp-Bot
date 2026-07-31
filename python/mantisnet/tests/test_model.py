"""§12.4 batching equivalence, output contracts, and the CUDA smoke test."""

from __future__ import annotations

import pytest
import torch

from mantisnet import MantisConfig, MantisNet, collate, from_position
from mantisnet.model import CRITIC_LOGITS


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
    batched = model(collate(graphs))
    offset = 0
    for i, g in enumerate(graphs):
        single = model(collate([g]))
        n = g.n_legal
        assert torch.allclose(
            batched.policy_logits[offset : offset + n], single.policy_logits, atol=1e-6
        )
        assert torch.allclose(
            batched.q_values[offset : offset + n], single.q_values, atol=1e-6
        )
        assert torch.allclose(batched.value[i : i + 1], single.value, atol=1e-6)
        assert torch.allclose(batched.value_dist[i : i + 1], single.value_dist, atol=1e-6)
        offset += n
    assert offset == batched.policy_logits.shape[0]


@torch.no_grad()
def test_output_contracts(model, positions):
    graphs = [from_position(p) for p in positions]
    batch = collate(graphs)
    out = model(batch)
    assert out.policy_logits.shape == (sum(g.n_legal for g in graphs),)
    assert out.q_values.shape == out.policy_logits.shape
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
def test_action_values_are_the_readout_bounded_by_tanh(positions):
    """Appendix B's scoring, written out against the raw readout row."""
    net = _critic_model()
    batch = collate([from_position(p) for p in positions])
    _s, w, g = net.trunk(batch)
    policy_logits, q = net.cell_heads(w, g, batch)

    assert net.mlp_q.out.weight.shape[0] == CRITIC_LOGITS
    assert q.shape == policy_logits.shape
    # The same readout the head runs, taken raw: the action value is its tanh.
    # The reference is in double, so the tolerance is fp32 rounding of the
    # scoring, not slack in the formula.
    g_q = net.mlp_q.lin_b(g)
    raw = net._cell_scores(
        net._decoder_rows(w, batch, g_q.dtype),
        g_q, batch, net.q, net.e_qw, net.e_qbg, net.mlp_q,
    ).squeeze(-1)
    torch.testing.assert_close(q.double(), torch.tanh(raw.double()), rtol=0, atol=1e-6)
    # The score is bounded whatever the readout holds, which is what keeps
    # exp(Q / (tau + lam)) from sharpening without limit — and this readout is
    # far from the zero one.
    assert q.abs().max() < 1.0
    assert q.abs().max() > 0.1
    assert q.dtype == torch.float32


@torch.no_grad()
def test_fresh_init_policy_logits_and_action_values_are_exactly_zero(positions):
    """The appendix-B init override: both decoder readouts start at zero, so
    both mass logits vanish and every action value is exactly zero."""
    torch.manual_seed(0)
    net = MantisNet(MantisConfig()).eval()
    out = net(collate([from_position(p) for p in positions]))
    assert torch.count_nonzero(out.policy_logits) == 0
    assert torch.count_nonzero(out.q_values) == 0


def test_dropout_config_runs_and_eval_is_deterministic(positions):
    torch.manual_seed(1)
    net = MantisNet(MantisConfig(dropout=0.1))
    batch = collate([from_position(positions[-1])])
    net.train()
    net(batch)  # The CUDA smoke test requires successful execution.
    net.eval()
    with torch.no_grad():
        a, b = net(batch), net(batch)
    assert torch.equal(a.policy_logits, b.policy_logits)
    assert torch.equal(a.value, b.value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
@torch.no_grad()
def test_cuda_bf16_smoke(model, positions):
    device = torch.device("cuda")
    net = model.to(device)
    try:
        batch = collate([from_position(p) for p in positions]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(batch)
        assert torch.isfinite(out.policy_logits).all()
        assert torch.isfinite(out.value).all()
        assert out.value_dist.dtype == torch.float32
        assert torch.allclose(
            out.value_dist.sum(-1), torch.ones(batch.n_pos, device=device), atol=1e-3
        )
    finally:
        net.to("cpu")
