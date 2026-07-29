"""Loss and target contracts: the two-hot projection, the segmented policy
cross-entropy, the §10 decay grouping, and end-to-end gradient wiring."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from mantisnet import (
    collate,
    from_position,
    param_groups,
    policy_loss,
    value_loss,
    value_target,
)


def test_two_hot_is_exact_in_expectation():
    torch.manual_seed(0)
    k = 65
    centers = torch.linspace(-1, 1, k)
    z = torch.cat([torch.rand(101) * 2 - 1, torch.tensor([-1.0, 0.0, 1.0])])
    target = value_target(z, k)
    assert torch.allclose(target @ centers, z, atol=1e-6)
    assert torch.allclose(target.sum(-1), torch.ones_like(z))
    assert torch.all(target >= 0)
    assert (target > 0).sum(-1).max() <= 2
    # Exact hits are one-hot, including both endpoints.
    exact = value_target(torch.tensor([-1.0, 0.0, 1.0]), k)
    assert torch.equal((exact > 0).sum(-1), torch.tensor([1, 1, 1]))


def test_value_target_refuses_out_of_range():
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        value_target(torch.tensor([1.5]), 65)


def test_value_loss_matches_manual_cross_entropy():
    torch.manual_seed(1)
    logits = torch.randn(7, 65)
    z = torch.rand(7) * 2 - 1
    expected = -(value_target(z, 65) * logits.log_softmax(-1)).sum(-1).mean()
    assert torch.allclose(value_loss(logits, z), expected)


def test_policy_loss_matches_a_looped_reference():
    torch.manual_seed(2)
    counts = [1, 4, 17]
    logits = torch.randn(sum(counts))
    offsets = torch.tensor([0, 1, 5, 22])
    targets = torch.cat([torch.rand(c).softmax(0) for c in counts])
    loss = policy_loss(logits, offsets, targets)
    ref = torch.stack(
        [
            -(targets[a:b] * logits[a:b].log_softmax(0)).sum()
            for a, b in zip(offsets[:-1], offsets[1:])
        ]
    ).mean()
    assert torch.allclose(loss, ref, atol=1e-6)


def test_policy_loss_refuses_a_non_distribution():
    logits = torch.zeros(3)
    offsets = torch.tensor([0, 3])
    with pytest.raises(ValueError, match="sum to 1"):
        policy_loss(logits, offsets, torch.tensor([0.5, 0.5, 0.5]))


def test_param_groups_partition_and_membership(model):
    decay, no_decay = param_groups(model, weight_decay=0.1)
    ids_d = {id(p) for p in decay["params"]}
    ids_n = {id(p) for p in no_decay["params"]}
    assert ids_d.isdisjoint(ids_n)
    assert ids_d | ids_n == {id(p) for p in model.parameters()}
    assert no_decay["weight_decay"] == 0.0

    assert id(model.stone_table.weight) in ids_n  # embedding table
    assert id(model.blocks[0].dist_bias) in ids_n  # attention-bias table
    assert id(model.token_base) in ids_n  # ndim 1
    assert id(model.blocks[0].wq.bias) in ids_n  # ndim 1
    assert id(model.blocks[0].wq.weight) in ids_d
    assert id(model.value_queries) in ids_d  # listed nowhere in §10's exclusions


def test_every_parameter_receives_gradient(positions):
    torch.manual_seed(3)
    from mantisnet import MantisConfig, MantisNet

    net = MantisNet(MantisConfig())
    net.train()
    # Includes ply 0, so the background bucket table participates too.
    batch = collate([from_position(p) for p in positions])
    out = net(batch)

    counts = batch.legal_offsets[1:] - batch.legal_offsets[:-1]
    targets = torch.cat([torch.full((int(c),), 1.0 / int(c)) for c in counts])
    z = torch.linspace(-0.9, 0.9, batch.n_pos)
    loss = (
        policy_loss(out.policy_logits, batch.legal_offsets, targets)
        + value_loss(out.value_logits, z)
        + (out.q_values.index_select(0, batch.legal_offsets[:-1]) - z).square().mean()
    )
    loss.backward()
    missing = [n for n, p in net.named_parameters() if p.grad is None]
    assert not missing, f"unreached parameters: {missing}"
