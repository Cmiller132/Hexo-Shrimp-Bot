"""The last block's stone rows are dead: no head reads post-trunk stones.

The trunk therefore computes only the four latent-row reads in its last
block — a dense cross-read over the same keys — and returns no stone rows.
These tests pin the two facts that make that an exact cut: the dense read
reproduces the full grid's global rows, and the last block's attention and
FFN parameters still learn through the latent path.
"""

from __future__ import annotations

import math

import torch

from mantisnet import MantisConfig, MantisNet, collate, from_position
from mantisnet.attention import fused_attention


@torch.no_grad()
def test_dense_latent_read_matches_the_full_grid_global_rows(positions):
    torch.manual_seed(11)
    batch = collate([from_position(p) for p in positions[:6]])
    n_pos, max_t = batch.attn_valid.shape
    heads, hd = 4, 8
    seq_lens = batch.attn_valid.sum(dim=1, dtype=torch.int32)
    q = torch.randn(n_pos, heads, max_t, hd)
    k = torch.randn(n_pos, heads, max_t, hd)
    v = torch.randn(n_pos, heads, max_t, hd)
    full = fused_attention(q, k, v, seq_lens)

    scores = (
        torch.einsum("phqd,phkd->phqk", q[:, :, :4].float(), k.float())
        / math.sqrt(hd)
    )
    scores = scores.masked_fill(~batch.attn_valid[:, None, None, :], -torch.inf)
    read = torch.einsum("phqk,phkd->phqd", scores.softmax(dim=-1), v.float())

    torch.testing.assert_close(read, full[:, :, :4], rtol=1e-5, atol=1e-5)


def test_last_block_attention_and_ffn_still_learn(positions):
    torch.manual_seed(5)
    cfg = MantisConfig(
        h=32,
        blocks=2,
        heads=4,
        ffn_factor=2,
        value_queries=2,
        value_bins=9,
        policy_hidden=24,
        value_hidden=20,
        cell_latents=True,
        cell_nodes=True,
        action_tactical=True,
    )
    net = MantisNet(cfg)
    batch = collate([from_position(p) for p in positions[4:8]])

    # The value head is the probe: the cell heads' zero-initialized output
    # layers pass no gradient into the trunk at a fresh model.
    out = net(batch, 0.2)
    out.value_logits.sum().backward()

    last = net.blocks[-1]
    for name in ("wq", "wk", "wv", "wo"):
        grad = getattr(last, name).weight.grad
        assert grad is not None and float(grad.abs().sum()) > 0, name
    for index in (0, 2):
        grad = last.ffn[index].weight.grad
        assert grad is not None and float(grad.abs().sum()) > 0, f"ffn[{index}]"
