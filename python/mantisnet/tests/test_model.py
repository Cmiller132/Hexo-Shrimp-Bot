"""§12.4 batching equivalence, output contracts, and the CUDA smoke test."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from mantisnet import MantisConfig, MantisNet, collate, from_position


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
def test_zero_init_gives_zero_heads_and_an_identity_critic_tail(model, positions):
    """Appendix B's zero init: the two decoder readouts and the critic tail's
    output linear. The tail therefore starts as the identity, and a fresh
    model's policy logits and action values are exactly zero."""
    batch = collate([from_position(positions[3])])
    _s, w, g = model.trunk(batch)
    w_q, g_q = model.critic_rows(w, g)
    assert torch.equal(w_q, w) and torch.equal(g_q, g)
    policy, q = model.cell_heads(w, g, batch)
    assert torch.count_nonzero(policy) == 0
    assert torch.count_nonzero(q) == 0


@torch.no_grad()
def test_critic_tail_is_private_and_splits_back_exactly(model, positions):
    """A live tail moves the action values and nothing else: §6's policy head
    and §7's value head read the unadapted rows."""
    net = copy.deepcopy(model)
    torch.manual_seed(4)
    for parameter in (net.q_tail[2].weight, net.q_tail[2].bias, net.mlp_q.out.weight):
        nn.init.normal_(parameter, std=0.1)

    batch = collate([from_position(p) for p in positions])
    _s, w, g = net.trunk(batch)
    w_q, g_q = net.critic_rows(w, g)
    rows = torch.cat([w, g], dim=0)
    expected = rows + net.q_tail(net.q_tail_ln(rows))
    assert (w_q.shape, g_q.shape) == (w.shape, g.shape)
    assert torch.equal(torch.cat([w_q, g_q], dim=0), expected)
    assert not torch.equal(w_q, w)

    live, dead = net(batch), model(batch)
    assert torch.equal(live.policy_logits, dead.policy_logits)
    assert torch.equal(live.value_logits, dead.value_logits)
    assert not torch.equal(live.q_values, dead.q_values)
    assert torch.all((live.q_values > -1.0) & (live.q_values < 1.0))


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
