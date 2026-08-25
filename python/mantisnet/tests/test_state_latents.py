"""Baked state-latent layout, literal attention oracle, and accounting."""

from __future__ import annotations

import math

import hexo_py
import pytest
import torch

from mantisnet import MantisConfig, MantisNet, collate, collate_positions, from_position
from mantisnet import window_latents
from mantisnet.klent.selfplay import _chunk_live
from mantisnet.lab.families import infer_config, load_checkpoint
from mantisnet.lab.train import scoped_attention_lengths


def _config(**overrides) -> MantisConfig:
    values = dict(
        h=16,
        blocks=1,
        heads=2,
        value_queries=2,
        value_bins=5,
        policy_hidden=16,
        value_hidden=16,
    )
    values.update(overrides)
    return MantisConfig(**values)


def test_baked_model_has_only_latent_parameters_and_keeps_the_parameter_pin():
    model = MantisNet(MantisConfig())
    assert model.latent_base.shape == (4, model.cfg.h)
    assert "token_base" not in model.state_dict()
    assert sum(parameter.numel() for parameter in model.parameters()) == 4_803_397


def test_python_and_rust_collation_match_the_baked_global_prefix(positions):
    selected = positions[:5]
    graphs = [from_position(position) for position in selected]
    rust = collate_positions(selected)
    python = collate(graphs)

    assert rust.max_t == max(graph.n_stones for graph in graphs) + 4
    assert torch.equal(rust.attn_valid[:, :4], torch.ones_like(rust.attn_valid[:, :4]))
    assert torch.count_nonzero(rust.coords[:, :4]) == 0
    for name, value in vars(python).items():
        got = getattr(rust, name)
        if isinstance(value, torch.Tensor):
            assert got.dtype == value.dtype and torch.equal(got, value), name
        else:
            assert got == value, name

    expected = torch.cat(
        [
            torch.arange(graph.n_stones) + position * rust.max_t + 4
            for position, graph in enumerate(graphs)
        ]
    )
    assert torch.equal(rust.stone_slot, expected)


@torch.no_grad()
def test_window_read_mix_broadcast_matches_literal_fp32_reference(positions):
    torch.manual_seed(41)
    cfg = _config()
    block = MantisNet(cfg).blocks[0].eval()
    batch = collate([from_position(position) for position in positions[:6]])
    w = torch.randn(batch.window_feat.shape[0], cfg.h)
    g = torch.randn(batch.n_pos, 4, cfg.h)

    heads, head_dim = cfg.heads, cfg.h // cfg.heads
    window_pos = batch.window_slot // batch.max_w
    offsets, order = window_latents.window_latent_layout(window_pos, batch.n_pos)
    actual_w, actual_g = block._window_latent_cycle(
        w, g, (window_pos, offsets, order)
    )

    # Read: each latent attends only the real windows of its own position.
    q = block.latent_wq_read(block.latent_ln_read_q(g)).view(
        batch.n_pos, 4, heads, head_dim
    )
    wk = block.latent_wk_read(block.latent_ln_read_w(w)).view(-1, heads, head_dim)
    wv = block.latent_wv_read(block.latent_ln_read_w(w)).view(-1, heads, head_dim)
    read = torch.zeros(batch.n_pos, 4, heads, head_dim, dtype=torch.float32)
    for position in range(batch.n_pos):
        windows = torch.nonzero(window_pos == position).flatten().tolist()
        for latent in range(4):
            for head in range(heads):
                if windows:
                    scores = torch.stack(
                        [q[position, latent, head].dot(wk[index, head]) for index in windows]
                    ).float() / math.sqrt(head_dim)
                    weights = scores.softmax(dim=0)
                    for weight, index in zip(weights, windows):
                        read[position, latent, head] += weight * wv[index, head].float()
    expected_g = g + block.latent_wo_read(read.reshape(batch.n_pos, 4, cfg.h))

    # Mix: literal K-by-K self-attention inside every position and head.
    z = block.latent_ln_mix(expected_g)
    q = block.latent_wq_mix(z).view(batch.n_pos, 4, heads, head_dim)
    k = block.latent_wk_mix(z).view(batch.n_pos, 4, heads, head_dim)
    v = block.latent_wv_mix(z).view(batch.n_pos, 4, heads, head_dim)
    mixed = torch.zeros_like(read)
    for position in range(batch.n_pos):
        for query in range(4):
            for head in range(heads):
                scores = torch.stack(
                    [q[position, query, head].dot(k[position, key, head]) for key in range(4)]
                ).float() / math.sqrt(head_dim)
                weights = scores.softmax(dim=0)
                for weight, key in zip(weights, range(4)):
                    mixed[position, query, head] += weight * v[position, key, head].float()
    expected_g = expected_g + block.latent_wo_mix(
        mixed.reshape(batch.n_pos, 4, cfg.h)
    )

    # Broadcast: every real window attends over its position's four latents.
    q = block.latent_wq_bcast(block.latent_ln_bcast_q(w)).view(
        -1, heads, head_dim
    )
    z = block.latent_ln_bcast_l(expected_g)
    k = block.latent_wk_bcast(z).view(batch.n_pos, 4, heads, head_dim)
    v = block.latent_wv_bcast(z).view(batch.n_pos, 4, heads, head_dim)
    broadcast = torch.zeros(w.shape[0], heads, head_dim, dtype=torch.float32)
    for window in range(w.shape[0]):
        position = int(window_pos[window])
        for head in range(heads):
            scores = torch.stack(
                [q[window, head].dot(k[position, key, head]) for key in range(4)]
            ).float() / math.sqrt(head_dim)
            weights = scores.softmax(dim=0)
            for weight, key in zip(weights, range(4)):
                broadcast[window, head] += weight * v[position, key, head].float()
    expected_w = w + block.latent_wo_bcast(broadcast.reshape(-1, cfg.h))

    torch.testing.assert_close(actual_g, expected_g, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual_w, expected_w, rtol=1e-5, atol=1e-6)


@torch.no_grad()
def test_final_ln_mean_is_the_only_global_reader_context(positions):
    torch.manual_seed(43)
    cfg = _config(blocks=0)
    model = MantisNet(cfg).eval()
    batch = collate([from_position(position) for position in positions[:4]])
    w, pooled, _cells = model.trunk(batch)
    raw = model.latent_base[None] + model.token_moves(batch.moves_idx)[:, None]
    expected = model.ln_out(raw).mean(dim=1)
    torch.testing.assert_close(pooled, expected)

    with torch.no_grad():
        model.mlp_p.out.weight.normal_(std=0.1)
        model.mlp_q.out.weight.normal_(std=0.1)
    got_heads = model.cell_head_logits(w, pooled, None, batch)
    ref_heads = model.cell_head_logits(w, expected, None, batch)
    got_value = model.value_head(w, pooled, batch)
    ref_value = model.value_head(w, expected, batch)
    for got, reference in (*zip(got_heads, ref_heads), *zip(got_value, ref_value)):
        torch.testing.assert_close(got, reference)


@torch.no_grad()
def test_batching_finiteness_and_fresh_zero_contracts(positions):
    torch.manual_seed(47)
    cfg = _config(blocks=2)
    model = MantisNet(cfg).eval()
    graphs = [from_position(position) for position in positions]
    batched = model(collate(graphs), 0.2)
    offset = 0
    for position, graph in enumerate(graphs):
        single = model(collate([graph]), 0.2)
        count = graph.n_legal
        for name in ("policy_logits", "q_score", "q_values"):
            torch.testing.assert_close(
                getattr(batched, name)[offset : offset + count],
                getattr(single, name),
                rtol=1e-5,
                atol=1e-6,
            )
        torch.testing.assert_close(
            batched.value[position : position + 1], single.value, rtol=1e-5, atol=1e-6
        )
        offset += count
    for tensor in vars(batched).values():
        assert torch.isfinite(tensor).all()
    assert torch.count_nonzero(batched.policy_logits) == 0
    assert torch.count_nonzero(batched.q_values) == 0


def test_key_projections_are_bias_free_and_checkpoint_infers_config():
    model = MantisNet(_config())
    parameters = dict(model.named_parameters())
    for name in ("read", "mix", "bcast"):
        assert f"blocks.0.latent_wk_{name}.weight" in parameters
        assert f"blocks.0.latent_wk_{name}.bias" not in parameters
    assert infer_config(model.state_dict()) == model.cfg


def test_pre_bake_single_token_checkpoint_refuses_as_a_baked_stage(tmp_path):
    model = MantisNet(_config())
    state = dict(model.state_dict())
    state["token_base"] = torch.zeros(model.cfg.h)
    del state["latent_base"]
    for key in list(state):
        if ".latent_" in key:
            del state[key]
    path = tmp_path / "single-token.pt"
    torch.save(
        {
            "model": state,
            "versions": {
                "RULES_VERSION": hexo_py.RULES_VERSION,
                "ACTION_ORDER_VERSION": hexo_py.ACTION_ORDER_VERSION,
            },
        },
        path,
    )
    with pytest.raises(ValueError, match="predating a baked stage"):
        load_checkpoint(path)


def test_pair_budget_counts_the_three_extra_rows():
    class Position:
        stone_count = 4
        legal_count = 1

    positions = [Position(), Position()]
    assert _chunk_live(positions, [0, 1], 60, 10, 10) == [[0], [1]]

    # The collect bench's PhaseTimer wrapper preserves the five-argument seam.
    from mantisnet.klent import selfplay
    from mantisnet.lab.bench import PhaseTimer

    with PhaseTimer(lambda batch: batch):
        assert selfplay._chunk_live(positions, [0, 1], 60, 10, 10) == [[0], [1]]


def test_bench_collect_and_packer_use_the_baked_layout_end_to_end(capsys):
    from mantisnet.lab.bench import bench_collect

    report = bench_collect(
        games=1,
        envs=2,
        cap=6,
        seed=3,
        device="cpu",
    )
    capsys.readouterr()
    assert report["samples"] > 0 and report["steps"] > 0

    base = torch.tensor([5, 9]).numpy()
    assert scoped_attention_lengths(base).tolist() == [8, 12]
