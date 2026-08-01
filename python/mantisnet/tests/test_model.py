"""§12.4 batching equivalence, output contracts, and the CUDA smoke test."""

from __future__ import annotations

import pytest
import torch

from mantisnet import MantisConfig, MantisNet, collate, from_position
from mantisnet.model import CRITIC_LOGITS, compose_acting_q, compose_q


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
def test_action_values_are_the_two_return_masses_composed(positions):
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
    u_pos = torch.sigmoid(critic_logits[:, 0].double())
    u_neg = torch.sigmoid(critic_logits[:, 1].double())
    torch.testing.assert_close(q.double(), u_pos - u_neg, rtol=0, atol=1e-6)
    # Both masses are in (0, 1), so their difference is a bounded action value
    # whatever the logits, and this readout is far from the zero one.
    assert q.abs().max() < 1.0
    assert q.abs().max() > 0.1
    assert q.dtype == torch.float32


@torch.no_grad()
def test_the_acting_score_is_q_over_the_position_s_largest_mass():
    """The score π′ ranks by, against the mass logits it is built from.

    One divisor per position, so it cannot reorder a legal set, and
    |Q| <= u_pos + u_neg <= the largest of them keeps it inside (−1, 1) —
    which is what π′ needs, since it exponentiates the score over τ + λ.
    Written against handmade logits because a small model's decoder gives one
    Q to every cell of a position, which would make the ordering vacuous.
    """
    critic_logits = torch.tensor(
        [
            [3.0, -3.0],  # a decided win: mass near 1, Q near 1
            [0.4, -1.2],
            [-1.5, 0.2],
            # A second position with nothing committed anywhere: both masses
            # near zero needs both logits well below zero, which is the critic
            # saying it expects |G| ~ 0 here, not that it is undecided between
            # the two signs.
            [-3.0, -3.4],
            [-3.6, -3.0],
            [-3.2, -3.2],
        ]
    )
    offsets = torch.tensor([0, 3, 6])
    q = compose_q(critic_logits)
    score = compose_acting_q(critic_logits, offsets, 0.0)
    total = torch.sigmoid(critic_logits.double()).sum(-1)

    for lo, hi in ((0, 3), (3, 6)):
        largest = total[lo:hi].max()
        torch.testing.assert_close(
            score[lo:hi].double(), q[lo:hi].double() / largest, rtol=0, atol=1e-6
        )
        # A single positive divisor: the order over the legal set is untouched.
        assert torch.equal(
            torch.argsort(score[lo:hi], stable=True),
            torch.argsort(q[lo:hi], stable=True),
        )
    assert score.abs().max() <= 1.0
    # The uncommitted position is the one the scaling exists for: its Q spread
    # is tiny and the score's is several times larger.
    assert score[3:6].std() > 4 * q[3:6].std()
    # The decided position keeps its scale, because its largest mass is ~1.
    torch.testing.assert_close(score[0:3], q[0:3], rtol=0, atol=5e-3)


@torch.no_grad()
def test_the_floor_bounds_the_sharpening_of_an_uncommitted_position(positions):
    """Where the whole legal set is uncommitted the divisor would be tiny, so
    the floor is what stops π′ sharpening without limit on a critic that has
    not decided anything."""
    net = _critic_model()
    batch = collate([from_position(p) for p in positions])
    _s, w, g = net.trunk(batch)
    _policy, unfloored, q = net.cell_heads(w, g, batch, 0.0)
    # Two sigmoids sum to under 2, so a floor of 4 binds in every position and
    # the divisor is the floor itself rather than any position's own mass.
    _policy, floored, _q = net.cell_heads(w, g, batch, 4.0)

    torch.testing.assert_close(floored.double(), q.double() / 4.0, rtol=0, atol=1e-7)
    assert float(unfloored.abs().max()) > float(floored.abs().max())


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
