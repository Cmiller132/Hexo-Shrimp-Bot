"""§12.4 batching equivalence, output contracts, and the CUDA smoke test."""

from __future__ import annotations

import pytest
import torch

from mantisnet import MantisConfig, MantisNet, collate, from_position
from mantisnet.segments import segment_ids


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
    # Appendix B bounds the critic: π′ exponentiates Q/(τ+λ), so an unbounded
    # action value could sharpen the improvement step without limit.
    assert torch.all((out.q_values > -1) & (out.q_values < 1))
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
def test_zero_init_gives_exactly_zero_logits_and_action_values(positions):
    """Appendix B's initialization contract. The dueling critic composes three
    readouts, and all three start at zero: a zero advantage centers to zero and
    a zero baseline adds nothing, so Q is exactly zero — not approximately, and
    not merely equal across a position's cells."""
    torch.manual_seed(2)
    net = MantisNet(MantisConfig()).eval()
    out = net(collate([from_position(p) for p in positions]))
    assert torch.count_nonzero(out.policy_logits) == 0
    assert torch.count_nonzero(out.q_values) == 0


def test_cell_pos_is_the_legal_offsets_segment_index(positions):
    """The critic's centering reduces over ``cell_pos`` because the builder
    already computed it; everything else ragged reduces over
    ``segment_ids(legal_offsets)``. Nothing else asserts they are one index."""
    batch = collate([from_position(p) for p in positions])
    assert torch.equal(batch.cell_pos, segment_ids(batch.legal_offsets))


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
        # The critic composes and bounds in fp32 whatever autocast chose for
        # the head GEMMs, so Q means the same thing under either.
        assert out.q_values.dtype == torch.float32
        assert torch.all((out.q_values > -1) & (out.q_values < 1))
        assert torch.isfinite(out.value).all()
        assert out.value_dist.dtype == torch.float32
        assert torch.allclose(
            out.value_dist.sum(-1), torch.ones(batch.n_pos, device=device), atol=1e-3
        )
    finally:
        net.to("cpu")
