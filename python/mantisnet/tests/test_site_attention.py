"""Site attention: the packed joint layout, both backends, and the knob's
model surface (CPU; the flex-vs-reference check is CUDA)."""

from __future__ import annotations

import hexo_py
import math
import pytest
import torch

from mantisnet import MantisConfig, MantisNet, collate, from_position
from mantisnet.site_attention import (
    GLOBAL_ROWS,
    document_mask,
    pack_rows,
    site_attention,
    site_attention_reference,
    site_layout,
)

from .conftest import random_moves

PRODUCTION = dict(
    cell_latents=True,
    cell_nodes=True,
    cell_node_scope="all",
    window_attention=False,
)


def _tiny(**overrides) -> MantisConfig:
    values = dict(
        h=16,
        blocks=2,
        heads=2,
        value_queries=2,
        value_bins=5,
        policy_hidden=16,
        value_hidden=16,
    )
    values.update(overrides)
    return MantisConfig(**values)


def _batch(seeds=(3, 5, 11), plies=(6, 1, 17)):
    positions = [
        hexo_py.Position.replay(random_moves(n, seed))
        for n, seed in zip(plies, seeds, strict=True)
    ]
    return collate([from_position(p) for p in positions])


def _layout(batch):
    seq_lens = batch.attn_valid.sum(dim=1, dtype=torch.int32)
    return site_layout(
        seq_lens,
        batch.legal_offsets,
        batch.cell_pos,
        batch.stone_slot,
        batch.attn_valid.shape[1],
        batch.stone_own.shape[0],
    )


def test_the_knob_requires_cell_state():
    with pytest.raises(ValueError, match="site_attention requires"):
        MantisConfig(site_attention=True)
    with pytest.raises(ValueError, match="site_attention requires"):
        MantisConfig(site_attention=True, cell_latents=True)
    MantisConfig(site_attention=True, **PRODUCTION)


def test_the_knob_excludes_the_subsumed_knobs():
    with pytest.raises(ValueError, match="subsumes cell_structure"):
        MantisConfig(site_attention=True, cell_structure=True, **PRODUCTION)
    with pytest.raises(ValueError, match="subsumes cell_adjacency"):
        MantisConfig(
            site_attention=True, cell_adjacency=True, **PRODUCTION
        )
    with pytest.raises(ValueError, match="replaces the radius read"):
        MantisConfig(
            site_attention=True,
            cell_latents=True,
            cell_nodes=True,
            cell_node_scope="uncovered",
            window_attention=False,
        )


def test_layout_partitions_the_packed_rows():
    batch = _batch()
    layout = _layout(batch)
    rows = torch.cat(
        [
            layout.global_rows.reshape(-1),
            layout.stone_rows,
            layout.cell_rows,
        ]
    )
    assert layout.total == GLOBAL_ROWS * batch.n_pos + batch.stone_own.shape[
        0
    ] + batch.cell_pos.shape[0]
    assert rows.shape[0] == layout.total
    assert torch.equal(rows.sort().values, torch.arange(layout.total))


def test_layout_documents_match_the_entities():
    batch = _batch()
    layout = _layout(batch)
    p = batch.n_pos
    max_t = batch.attn_valid.shape[1]
    assert torch.equal(
        layout.doc.index_select(0, layout.global_rows.reshape(-1)).long(),
        torch.arange(p).repeat_interleave(GLOBAL_ROWS),
    )
    assert torch.equal(
        layout.doc.index_select(0, layout.stone_rows).long(),
        batch.stone_slot // max_t,
    )
    assert torch.equal(
        layout.doc.index_select(0, layout.cell_rows).long(), batch.cell_pos
    )


def test_layout_orders_each_position_as_latents_stones_cells():
    batch = _batch()
    layout = _layout(batch)
    max_t = batch.attn_valid.shape[1]
    seq_lens = batch.attn_valid.sum(dim=1)
    starts = torch.cat(
        [torch.zeros(1, dtype=torch.long), layout.doc.long().bincount().cumsum(0)]
    )
    stone_pos = batch.stone_slot // max_t
    stone_rank = batch.stone_slot - stone_pos * max_t
    assert torch.equal(
        layout.stone_rows, starts.index_select(0, stone_pos) + stone_rank
    )
    cell_rank = torch.arange(
        batch.cell_pos.shape[0]
    ) - batch.legal_offsets.index_select(0, batch.cell_pos)
    assert torch.equal(
        layout.cell_rows,
        starts.index_select(0, batch.cell_pos)
        + seq_lens.index_select(0, batch.cell_pos)
        + cell_rank,
    )


def test_reference_attention_matches_a_per_position_softmax():
    batch = _batch()
    layout = _layout(batch)
    torch.manual_seed(7)
    q = torch.randn(layout.total, 2, 8)
    k = torch.randn(layout.total, 2, 8)
    v = torch.randn(layout.total, 2, 8)
    out = site_attention_reference(q, k, v, layout.doc)
    for pos in range(batch.n_pos):
        rows = (layout.doc == pos).nonzero().squeeze(1)
        qs, ks, vs = q[rows], k[rows], v[rows]
        scores = torch.einsum("qad,kad->qka", qs, ks) / math.sqrt(8)
        expect = torch.einsum("qka,kad->qad", scores.softmax(dim=1), vs)
        torch.testing.assert_close(out[rows], expect, rtol=1e-5, atol=1e-5)


def test_pack_rows_round_trips_every_entity():
    batch = _batch()
    layout = _layout(batch)
    p = batch.n_pos
    h = 16
    torch.manual_seed(11)
    g = torch.randn(p, GLOBAL_ROWS, h)
    s = torch.randn(batch.stone_own.shape[0], h)
    c = torch.randn(batch.cell_pos.shape[0], h)
    rows = pack_rows(g, s, c, layout)
    assert torch.equal(
        rows.index_select(0, layout.global_rows.reshape(-1)),
        g.reshape(-1, h),
    )
    assert torch.equal(rows.index_select(0, layout.stone_rows), s)
    assert torch.equal(rows.index_select(0, layout.cell_rows), c)


def test_the_knob_swaps_bias_and_radius_parameters_for_identity_tables():
    torch.manual_seed(0)
    off = MantisNet(_tiny(**PRODUCTION))
    torch.manual_seed(0)
    on = MantisNet(_tiny(site_attention=True, **PRODUCTION))
    off_names = {name for name, _ in off.named_parameters()}
    on_names = {name for name, _ in on.named_parameters()}
    gone = {name for name in off_names - on_names}
    assert gone, "site attention must remove parameters, not only add"
    for name in gone:
        assert (
            "axis_bias" in name or "orbit_bias" in name or "radius" in name
        ), name
    added = on_names - off_names
    assert added == {"cell_class_table.weight", "cell_coverage_table.weight"}
    assert torch.count_nonzero(on.cell_class_table.weight) == 0
    assert torch.count_nonzero(on.cell_coverage_table.weight) == 0


def test_knob_on_forward_backward_reaches_every_site():
    torch.manual_seed(3)
    model = MantisNet(_tiny(site_attention=True, **PRODUCTION))
    # The zero-init discipline makes every head read the trunk through a
    # zero matrix at initialization, so a fresh model's policy loss reaches
    # no trunk parameter. Perturb the zeros to probe the trained-state path.
    with torch.no_grad():
        for parameter in model.parameters():
            if torch.count_nonzero(parameter) == 0:
                parameter.normal_(std=0.02)
    batch = _batch()
    out = model(batch, mass_floor=0.2)
    assert out.policy_logits.shape == (batch.cell_pos.shape[0],)
    assert torch.isfinite(out.policy_logits).all()
    assert torch.isfinite(out.value).all()
    (out.policy_logits.square().sum() + out.q_score.square().sum()).backward()
    for name in ("cell_class_table.weight", "cell_coverage_table.weight"):
        grad = dict(model.named_parameters())[name].grad
        assert grad is not None and torch.count_nonzero(grad) > 0, name
    wq = dict(model.named_parameters())["blocks.0.wq.weight"].grad
    assert wq is not None and torch.count_nonzero(wq) > 0


def test_knob_on_handles_the_opening_position():
    torch.manual_seed(3)
    model = MantisNet(_tiny(site_attention=True, **PRODUCTION))
    batch = collate([from_position(hexo_py.Position())])
    out = model(batch, mass_floor=0.2)
    assert torch.isfinite(out.policy_logits).all()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the flex backend requires CUDA"
)
def test_flex_survives_torch_compile_with_dynamic_shapes():
    """The fit compiles the trunk; the flex call must survive inductor with
    symbolic row counts (the eager tests cannot see a CantSplit)."""
    device = torch.device("cuda")
    compiled = torch.compile(site_attention, dynamic=True)
    for seeds, plies in (((3, 5, 11), (6, 1, 17)), ((21, 22), (9, 30))):
        batch = _batch(seeds=seeds, plies=plies)
        layout = _layout(batch)
        layout = type(layout)(
            layout.global_rows.to(device),
            layout.stone_rows.to(device),
            layout.cell_rows.to(device),
            layout.doc.to(device),
            layout.total,
        )
        torch.manual_seed(9)
        # The compiled lowering requires head dim >= 16; the model's is 32.
        q = torch.randn(layout.total, 2, 16, device=device, dtype=torch.bfloat16)
        k = torch.randn(layout.total, 2, 16, device=device, dtype=torch.bfloat16)
        v = torch.randn(layout.total, 2, 16, device=device, dtype=torch.bfloat16)
        mask = document_mask(layout)
        fast = compiled(q, k, v, layout, mask)
        slow = site_attention_reference(q, k, v, layout.doc)
        torch.testing.assert_close(fast, slow, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the flex backend requires CUDA"
)
def test_flex_matches_the_reference():
    batch = _batch(seeds=(3, 5, 11, 13), plies=(6, 1, 17, 40))
    layout = _layout(batch)
    device = torch.device("cuda")
    layout = type(layout)(
        layout.global_rows.to(device),
        layout.stone_rows.to(device),
        layout.cell_rows.to(device),
        layout.doc.to(device),
        layout.total,
    )
    torch.manual_seed(5)
    q = torch.randn(layout.total, 2, 8, device=device, dtype=torch.bfloat16)
    k = torch.randn(layout.total, 2, 8, device=device, dtype=torch.bfloat16)
    v = torch.randn(layout.total, 2, 8, device=device, dtype=torch.bfloat16)
    mask = document_mask(layout)
    fast = site_attention(q, k, v, layout, mask)
    slow = site_attention_reference(q, k, v, layout.doc)
    torch.testing.assert_close(fast, slow, rtol=2e-2, atol=2e-2)
