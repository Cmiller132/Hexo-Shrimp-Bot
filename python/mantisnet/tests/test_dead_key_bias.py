"""The census and softmax theorem behind bias-free key projections."""

from __future__ import annotations

import torch

from mantisnet import MantisConfig, MantisNet, collate, from_position
from mantisnet.attention import fused_attention
from mantisnet.window_pairs import edge_attention, pair_tables


def test_softmax_key_projection_parameter_census():
    cfg = MantisConfig()
    parameters = dict(MantisNet(cfg).named_parameters())

    for index in range(cfg.blocks):
        for projection in ("wk", "wk_wa"):
            bias_key = f"blocks.{index}.{projection}.bias"
            weight_key = f"blocks.{index}.{projection}.weight"
            assert bias_key not in parameters
            assert weight_key in parameters


@torch.no_grad()
def test_constant_post_projection_key_bias_cancels_in_both_cpu_attention_paths(
    positions,
):
    torch.manual_seed(37)
    cfg = MantisConfig(
        h=32,
        blocks=1,
        heads=4,
        ffn_factor=2,
        value_queries=2,
        value_bins=9,
        policy_hidden=24,
        value_hidden=20,
    )
    net = MantisNet(cfg).eval()
    batch = collate([from_position(position) for position in positions[5:9]])
    block = net.blocks[0]
    heads, head_dim = cfg.heads, cfg.h // cfg.heads

    # Real padded [state latents; stones] rows and geometry from the batch.
    stones = net.stone_table(batch.stone_own)
    latents = net.latent_base[None] + net.token_moves(batch.moves_idx)[:, None]
    n_pos, max_t = latents.shape[0], batch.attn_valid.shape[1]
    rows = stones.new_zeros(n_pos * max_t, cfg.h)
    latent_slot = (
        torch.arange(n_pos)[:, None] * max_t + torch.arange(4)[None, :]
    ).reshape(-1)
    rows.index_copy_(0, latent_slot, latents.reshape(-1, cfg.h))
    rows.index_copy_(0, batch.stone_slot, stones)
    z = block.ln_attn(rows.view(n_pos, max_t, cfg.h))
    q = block.wq(z).view(n_pos, max_t, heads, head_dim).transpose(1, 2)
    k = block.wk(z).view(n_pos, max_t, heads, head_dim).transpose(1, 2)
    v = block.wv(z).view(n_pos, max_t, heads, head_dim).transpose(1, 2)
    seq_lens = batch.attn_valid.sum(dim=1, dtype=torch.int32)
    key_bias = torch.linspace(0.125, 1.0, cfg.h).view(heads, head_dim)
    stone_expected = fused_attention(q, k, v, seq_lens)
    stone_actual = fused_attention(q, k + key_bias[None, :, None, :], v, seq_lens)

    # Real window rows and pair relations from the same collated batch.
    windows = net.window_table(batch.window_feat)
    wz = block.ln_wa(windows)
    n_windows = windows.shape[0]
    wq = block.wq_wa(wz).view(n_windows, heads, head_dim)
    wk = block.wk_wa(wz).view(n_windows, heads, head_dim)
    wv = block.wv_wa(wz).view(n_windows, heads, head_dim)
    pairs = pair_tables(batch.window_id, batch.window_slot // batch.max_w)
    window_bias = torch.linspace(-0.75, -0.125, cfg.h).view(heads, head_dim)
    window_expected = edge_attention(wq, wk, wv, block.wa_bias, *pairs)
    window_actual = edge_attention(
        wq, wk + window_bias[None, :, :], wv, block.wa_bias, *pairs
    )

    torch.testing.assert_close(stone_actual, stone_expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(window_actual, window_expected, rtol=1e-6, atol=1e-6)
