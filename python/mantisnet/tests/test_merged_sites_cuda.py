"""Merged-site trunk on CUDA: the composition launches the fused kernels
(typed site read at 2184 classes, the merged six-edge aggregation, the
site-grid attention / latent cross-reads) and agrees with the CPU
reference composition."""

import pytest
import torch

import hexo_py
from mantisnet.builder import collate, from_position
from mantisnet.model import MantisConfig, MantisNet

from .conftest import random_moves

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

_TINY = dict(h=64, heads=2, blocks=2, policy_hidden=32, value_hidden=32)


@pytest.mark.parametrize(
    "arm",
    ({"merged_sites": True}, {"merged_sites": True, "site_self_attention": False}),
    ids=("full", "latent"),
)
def test_cuda_forward_backward_matches_cpu(arm):
    torch.manual_seed(0)
    batch = collate(
        [
            from_position(hexo_py.Position.replay(random_moves(11, seed=3))),
            from_position(hexo_py.Position.replay(random_moves(7, seed=5))),
        ]
    )
    model = MantisNet(MantisConfig(**_TINY, **arm)).eval()
    with torch.no_grad():
        cpu = model(batch, 0.2)
    gpu_model = MantisNet(MantisConfig(**_TINY, **arm)).cuda()
    gpu_model.load_state_dict(model.state_dict())
    gpu_batch = batch.to("cuda")
    out = gpu_model(gpu_batch, 0.2)
    assert torch.allclose(
        out.policy_logits.cpu(), cpu.policy_logits, atol=5e-4, rtol=1e-3
    )
    assert torch.allclose(out.value.cpu(), cpu.value, atol=5e-4, rtol=1e-3)
    (out.policy_logits.sum() + out.q_values.sum() + out.value.sum()).backward()
    for name, p in gpu_model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), name


@pytest.mark.parametrize(
    "arm",
    ({"merged_sites": True}, {"merged_sites": True, "site_self_attention": False}),
    ids=("full", "latent"),
)
def test_cuda_backward_is_bit_deterministic(arm):
    torch.manual_seed(1)
    batch = collate(
        [from_position(hexo_py.Position.replay(random_moves(9, seed=13)))]
    ).to("cuda")
    model = MantisNet(MantisConfig(**_TINY, **arm)).cuda().train()
    with torch.no_grad():
        torch.nn.init.normal_(model.mlp_p.out.weight, std=0.1)

    def grads():
        model.zero_grad(set_to_none=True)
        out = model(batch, 0.2)
        (out.policy_logits.sum() + out.q_values.sum() + out.value.sum()).backward()
        return {
            name: p.grad.clone()
            for name, p in model.named_parameters()
            if p.grad is not None
        }

    first = grads()
    second = grads()
    assert first.keys() == second.keys()
    # The bar is the split trunk's: gradients that flow through the fused
    # kernels are bit-exact across replays. Embedding-gather gradients
    # (the class-table hoist and the input tables) ride index_add's
    # atomics, exactly as the split trunk's e_ws/e_sw/e_pw/act_table do.
    gather_tables = (
        "e_wsite", "e_pw", "e_qw", "act_table", "window_table",
        "stone_table", "cell_nearest_table", "token_moves",
    )
    for name in first:
        if any(part in name for part in gather_tables):
            continue
        assert torch.equal(first[name], second[name]), name
