"""Step 2 state-latent layout, literal attention oracle, and knob contracts."""

from __future__ import annotations

import math

import hexo_py
import pytest
import torch

from mantisnet import MantisConfig, MantisNet, collate, collate_positions, from_position
from mantisnet.attention import _bucket_index
from mantisnet.klent.selfplay import _chunk_live
from mantisnet.lab.families import infer_config
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
        state_latents=4,
    )
    values.update(overrides)
    return MantisConfig(**values)


def test_state_latent_knob_is_a_path_selector():
    for value in (-1, 1, 2, 3, 5, 8):
        with pytest.raises(ValueError, match=r"\{0, 4\}"):
            MantisConfig(state_latents=value)
    assert MantisConfig().state_latents == 0
    assert MantisConfig(state_latents=4).state_latents == 4


def test_knob_off_has_no_latent_parameters_and_keeps_the_parameter_pin():
    model = MantisNet(MantisConfig())
    assert not any("latent" in name for name, _ in model.named_parameters())
    assert "token_base" in model.state_dict()
    assert sum(parameter.numel() for parameter in model.parameters()) == 4_007_269


def test_knob_on_python_and_rust_collation_grow_only_the_global_prefix(positions):
    selected = positions[:5]
    incumbent = collate_positions(selected)
    rust = collate_positions(selected, state_latents=4)
    python = collate(
        [from_position(position) for position in selected], state_latents=4
    )

    assert rust.state_latents == python.state_latents == 4
    assert rust.max_t == incumbent.max_t + 3
    assert torch.equal(rust.attn_valid[:, :4], torch.ones_like(rust.attn_valid[:, :4]))
    for name, value in vars(python).items():
        got = getattr(rust, name)
        if isinstance(value, torch.Tensor):
            assert got.dtype == value.dtype and torch.equal(got, value), name
        else:
            assert got == value, name

    # The Rust wire batch remains incumbent-shaped; expansion preserves all
    # stone rows while shifting their padded slots by exactly K-1.
    old_width, new_width = incumbent.max_t, rust.max_t
    old_pos = incumbent.stone_slot // old_width
    old_row = incumbent.stone_slot % old_width
    expected = old_pos * new_width + old_row + 3
    assert torch.equal(rust.stone_slot, expected)


def test_every_latent_pair_uses_the_token_bias_bucket(positions):
    batch = collate(
        [from_position(position) for position in positions[:4]], state_latents=4
    )
    seq_lens = batch.attn_valid.sum(dim=1, dtype=torch.int32)
    bucket, _valid = _bucket_index(
        batch.coords, seq_lens, batch.max_t, d_max=3, global_rows=4
    )
    rows = torch.arange(batch.max_t)
    touches_latent = (rows[:, None] < 4) | (rows[None, :] < 4)
    live_pair = (
        rows[None, :, None] < seq_lens[:, None, None]
    ) & (rows[None, None, :] < seq_lens[:, None, None])
    assert torch.all(bucket[live_pair & touches_latent] == 4)


@torch.no_grad()
def test_window_read_mix_broadcast_matches_literal_fp32_reference(positions):
    torch.manual_seed(41)
    cfg = _config()
    block = MantisNet(cfg).blocks[0].eval()
    batch = collate(
        [from_position(position) for position in positions[:6]], state_latents=4
    )
    w = torch.randn(batch.window_feat.shape[0], cfg.h)
    g = torch.randn(batch.n_pos, 4, cfg.h)

    actual_w, actual_g = block._window_latent_cycle(w, g, batch)
    heads, head_dim = cfg.heads, cfg.h // cfg.heads
    window_pos = batch.window_slot // batch.max_w

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
    batch = collate(
        [from_position(position) for position in positions[:4]], state_latents=4
    )
    _s, w, pooled = model.trunk(batch)
    raw = model.latent_base[None] + model.token_moves(batch.moves_idx)[:, None]
    expected = model.ln_out(raw).mean(dim=1)
    torch.testing.assert_close(pooled, expected)

    with torch.no_grad():
        model.mlp_p.out.weight.normal_(std=0.1)
        model.mlp_q.out.weight.normal_(std=0.1)
    got_heads = model.cell_head_logits(w, pooled, batch)
    ref_heads = model.cell_head_logits(w, expected, batch)
    got_value = model.value_head(w, pooled, batch)
    ref_value = model.value_head(w, expected, batch)
    for got, reference in (*zip(got_heads, ref_heads), *zip(got_value, ref_value)):
        torch.testing.assert_close(got, reference)


@torch.no_grad()
def test_knob_on_batching_finiteness_and_fresh_zero_contracts(positions):
    torch.manual_seed(47)
    cfg = _config(blocks=2)
    model = MantisNet(cfg).eval()
    graphs = [from_position(position) for position in positions]
    batched = model(collate(graphs, state_latents=4), 0.2)
    offset = 0
    for position, graph in enumerate(graphs):
        single = model(collate([graph], state_latents=4), 0.2)
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


def test_knob_on_key_projections_are_bias_free_and_checkpoint_infers_config():
    model = MantisNet(_config())
    parameters = dict(model.named_parameters())
    for name in ("read", "mix", "bcast"):
        assert f"blocks.0.latent_wk_{name}.weight" in parameters
        assert f"blocks.0.latent_wk_{name}.bias" not in parameters
    assert infer_config(model.state_dict()).state_latents == 4


def test_pair_budget_counts_the_three_extra_rows():
    class Position:
        stone_count = 4
        legal_count = 1

    positions = [Position(), Position()]
    assert _chunk_live(positions, [0, 1], 60, 10, 10) == [[0, 1]]
    assert _chunk_live(positions, [0, 1], 60, 10, 10, state_latents=4) == [[0], [1]]

    # The collect bench's PhaseTimer wrapper must forward the collector's
    # positional state_latents argument, not pin the old five-arg shape.
    from mantisnet.klent import selfplay
    from mantisnet.lab.bench import PhaseTimer

    with PhaseTimer(lambda batch: batch):
        assert selfplay._chunk_live(positions, [0, 1], 60, 10, 10, 4) == [[0], [1]]


def test_bench_collect_threads_the_knob_end_to_end(capsys):
    from mantisnet.lab.bench import bench_collect

    report = bench_collect(
        games=1,
        envs=2,
        cap=6,
        seed=3,
        device="cpu",
        model_kw={"state_latents": 4},
    )
    capsys.readouterr()
    assert report["samples"] > 0 and report["steps"] > 0

    base = torch.tensor([5, 9]).numpy()
    off = scoped_attention_lengths(base, MantisNet(MantisConfig()))
    on = scoped_attention_lengths(base, MantisNet(_config()))
    assert off is base
    assert on.tolist() == [8, 12]
