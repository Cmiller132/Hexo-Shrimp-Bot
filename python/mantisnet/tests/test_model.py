"""§12.4 batching equivalence, output contracts, and the CUDA smoke test."""

from __future__ import annotations

import pytest
import torch

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
    assert batch.legal_offsets.tolist() == [0] + list(
        torch.tensor([g.n_legal for g in graphs]).cumsum(0).tolist()
    )
    assert out.value.shape == (len(graphs),)
    assert torch.all((out.value >= -1) & (out.value <= 1))
    assert torch.allclose(out.value_dist.sum(-1), torch.ones(len(graphs)), atol=1e-6)
    # The scalar is the distribution's decode — the same value every consumer sees.
    assert torch.allclose(out.value, out.value_dist @ model.bin_centers, atol=1e-6)
    assert torch.isfinite(out.policy_logits).all()


def test_dropout_config_runs_and_eval_is_deterministic(positions):
    torch.manual_seed(1)
    net = MantisNet(MantisConfig(dropout=0.1))
    batch = collate([from_position(positions[-1])])
    net.train()
    net(batch)  # stochastic, just must run
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
