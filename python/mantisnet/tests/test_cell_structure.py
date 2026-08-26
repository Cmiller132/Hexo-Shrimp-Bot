"""`cell_structure`: structured covered-cell init and the nonlinear
two-input cell update (CPU)."""

from __future__ import annotations

import hexo_py
import pytest
import torch

from mantisnet import MantisConfig, MantisNet, collate, from_position
from mantisnet.lab.families import infer_config
from mantisnet.model import COVERAGE_BUCKET_OF, COVERAGE_BUCKETS

from .conftest import d6_transforms


_MOVES = [(0, 0), (3, 0), (-2, 2), (0, 3), (1, -2), (-1, 3)]
# The production trunk, and the same trunk with the tactical knob on top.
PRODUCTION = dict(
    cell_latents=True,
    cell_nodes=True,
    cell_node_scope="all",
)
ARM_B = dict(PRODUCTION, action_tactical=True)


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


def _batch(moves=_MOVES):
    return collate([from_position(hexo_py.Position.replay(moves))])


def _pair(seed: int, **base):
    """A knob-off and a knob-on model sharing every parameter they both own."""
    torch.manual_seed(seed)
    off = MantisNet(_tiny(**base)).eval()
    torch.manual_seed(seed)
    on = MantisNet(_tiny(**base, cell_structure=True)).eval()
    _missing, unexpected = on.load_state_dict(off.state_dict(), strict=False)
    assert not unexpected
    return off, on


def test_the_knob_requires_cell_latents():
    assert MantisConfig().cell_structure is False
    for inert in ({}, {"cell_nodes": True}):
        with pytest.raises(ValueError) as error:
            MantisConfig(cell_structure=True, **inert)
        assert "cell_structure requires cell_latents=True" in str(error.value)
    MantisConfig(cell_latents=True, cell_structure=True)


def test_coverage_buckets_cover_every_reachable_degree():
    # A cell lies in at most eighteen live windows, six per axis.
    assert len(COVERAGE_BUCKET_OF) == 19
    assert set(COVERAGE_BUCKET_OF[1:]) == set(range(COVERAGE_BUCKETS))
    assert list(COVERAGE_BUCKET_OF[1:]) == sorted(COVERAGE_BUCKET_OF[1:])


@pytest.mark.parametrize(
    "base",
    [
        dict(cell_latents=True),
        dict(cell_latents=True, cell_nodes=True),
        dict(cell_latents=True, cell_nodes=True, cell_adjacency=True),
        PRODUCTION,
    ],
    ids=["latents", "nodes", "adjacency", "production"],
)
def test_knob_off_is_bit_identical_to_the_incumbent_path(base):
    batch = _batch()
    torch.manual_seed(3)
    incumbent = MantisNet(_tiny(**base)).eval()
    torch.manual_seed(3)
    explicit_off = MantisNet(_tiny(**base, cell_structure=False)).eval()
    for left, right in zip(
        incumbent.state_dict().values(), explicit_off.state_dict().values()
    ):
        assert torch.equal(left, right)
    with torch.no_grad():
        left, right = incumbent(batch, 0.2), explicit_off(batch, 0.2)
    for name in vars(left):
        assert getattr(left, name).numpy().tobytes() == getattr(
            right, name
        ).numpy().tobytes(), name


@pytest.mark.parametrize(
    "base",
    [dict(cell_latents=True), dict(cell_latents=True, cell_nodes=True)],
    ids=["latents", "nodes"],
)
def test_the_knobs_own_parameters_start_at_zero(base):
    model = MantisNet(_tiny(**base, cell_structure=True))
    assert not model.cell_class_table.weight.any()
    assert not model.cell_coverage_table.weight.any()
    for block in model.blocks:
        assert not block.mlp_c.out.weight.any()
        assert not block.mlp_c.out.bias.any()


@pytest.mark.parametrize(
    "base",
    [dict(cell_latents=True), dict(cell_latents=True, cell_nodes=True)],
    ids=["latents", "nodes"],
)
def test_structured_init_starts_at_the_incumbent_row(base):
    """Two of the three added init terms are zero at init; the third is the
    nearest-stone row, which is a table covered cells newly read rather than
    a new table. Zeroing it makes the whole init exactly the incumbent's
    shared row."""
    batch = _batch()
    off, on = _pair(4, **base)
    ctab = on._cell_tables(batch, len(batch.window_feat))
    with torch.no_grad():
        structured = on._covered_init(batch, ctab)
        nearest = on.cell_nearest_table(
            batch.cell_nearest.index_select(0, ctab.covered)
        )
        assert torch.equal(structured, off._covered_init(batch, ctab) + nearest)
        on.cell_nearest_table.weight.zero_()
        assert torch.equal(on._covered_init(batch, ctab), off._covered_init(batch, ctab))


@pytest.mark.parametrize(
    "base",
    [
        dict(cell_latents=True),
        dict(cell_latents=True, cell_nodes=True),
        dict(cell_latents=True, cell_nodes=True, cell_adjacency=True),
        PRODUCTION,
        ARM_B,
    ],
    ids=["latents", "nodes", "adjacency", "production", "armB"],
)
def test_the_knob_is_a_no_op_at_init(base):
    """The knob adds to the incumbent rather than replacing any of it: the
    additive read path is untouched, the update MLP's output layer is zero,
    and both static tables are zero. The one remaining init-time difference
    is the nearest-stone row covered cells newly read, off a table that is
    not new; zeroing it in both models leaves the arms byte-identical."""
    batch = _batch()
    off, on = _pair(6, **base)
    with torch.no_grad():
        for model in (off, on):
            if hasattr(model, "cell_nearest_table"):
                model.cell_nearest_table.weight.zero_()
        left, right = off(batch, 0.2), on(batch, 0.2)
    for name in vars(left):
        assert getattr(left, name).numpy().tobytes() == getattr(
            right, name
        ).numpy().tobytes(), name
    # And the term is live, not inert: moving the output layer off zero moves
    # the model. The decoder outputs are zero-init, so the value head is what
    # carries a difference at a fresh init.
    with torch.no_grad():
        for block in on.blocks:
            block.mlp_c.out.weight.normal_(std=0.05)
        moved = on(batch, 0.2)
    assert not torch.equal(moved.value_logits, left.value_logits)


def test_gradients_reach_every_new_parameter():
    torch.manual_seed(8)
    model = MantisNet(_tiny(**PRODUCTION, cell_structure=True))
    # The zero-init decoder outputs block upstream flow, and the zero-init
    # cell MLP output likewise gates its own first layer; perturb both first.
    for head in (model.mlp_p, model.mlp_q):
        torch.nn.init.normal_(head.out.weight, std=0.05)
    for block in model.blocks:
        torch.nn.init.normal_(block.mlp_c.out.weight, std=0.05)
    out = model(_batch(), 0.2)
    (out.policy_logits.sum() + out.q_values.sum()).backward()
    grads = dict(model.named_parameters())
    names = [
        "cell_class_table.weight",
        "cell_coverage_table.weight",
        "cell_nearest_table.weight",
        "blocks.0.ln_c.weight",
        "blocks.0.mlp_c.lin_a.weight",
        "blocks.0.mlp_c.lin_b.weight",
        "blocks.0.mlp_c.out.weight",
    ]
    for name in names:
        grad = grads[name].grad
        assert grad is not None and torch.isfinite(grad).all(), name
        assert grad.abs().sum() > 0, name


def test_the_static_encodings_separate_cells_the_incumbent_ties():
    """Covered cells the incumbent initializes identically differ once the
    class and coverage tables move."""
    batch = _batch()
    torch.manual_seed(14)
    model = MantisNet(_tiny(cell_latents=True, cell_structure=True)).eval()
    ctab = model._cell_tables(batch, len(batch.window_feat))
    with torch.no_grad():
        model.cell_nearest_table.weight.zero_()
        assert bool((model._covered_init(batch, ctab) == model.cell_base).all())
        model.cell_class_table.weight.normal_(std=0.1)
        model.cell_coverage_table.weight.normal_(std=0.1)
        rows = model._covered_init(batch, ctab)
    distinct = torch.unique(rows, dim=0)
    assert distinct.shape[0] > 1


@torch.no_grad()
def test_the_structured_model_is_d6_invariant():
    torch.manual_seed(12)
    model = MantisNet(_tiny(**PRODUCTION, cell_structure=True)).eval()
    model.cell_class_table.weight.normal_(std=0.1)
    model.cell_coverage_table.weight.normal_(std=0.1)
    for block in model.blocks:
        block.mlp_c.out.weight.normal_(std=0.05)
    model.mlp_p.out.weight.normal_(std=0.05)
    model.mlp_q.out.weight.normal_(std=0.05)
    base_position = hexo_py.Position.replay(_MOVES)
    base = model(collate([from_position(base_position)]), 0.2)
    policy = dict(zip(base_position.legal_moves(), base.policy_logits.tolist()))
    values = dict(zip(base_position.legal_moves(), base.q_values.tolist()))
    for transform in d6_transforms():
        position = hexo_py.Position.replay([transform(move) for move in _MOVES])
        output = model(collate([from_position(position)]), 0.2)
        moved_policy = dict(zip(position.legal_moves(), output.policy_logits.tolist()))
        moved_values = dict(zip(position.legal_moves(), output.q_values.tolist()))
        for move in policy:
            assert abs(moved_policy[transform(move)] - policy[move]) <= 1e-5
            assert abs(moved_values[transform(move)] - values[move]) <= 1e-5
        assert torch.allclose(output.value, base.value, atol=1e-5)


@torch.no_grad()
def test_forward_shapes_on_a_small_config():
    positions = [
        hexo_py.Position.replay(_MOVES[:index]) for index in (1, 3, len(_MOVES))
    ]
    batch = collate([from_position(position) for position in positions])
    cfg = _tiny(**ARM_B, cell_structure=True)
    torch.manual_seed(13)
    model = MantisNet(cfg).eval()
    windows, glob, cells = model.trunk(batch)
    assert cells is not None and cells.shape == (batch.n_cells, cfg.h)
    assert glob.shape == (len(positions), cfg.h)
    assert windows.shape[1] == cfg.h
    out = model(batch, 0.2)
    assert out.policy_logits.shape == (batch.n_cells,)
    assert out.q_values.shape == (batch.n_cells,)
    assert out.value.shape == (len(positions),)
    assert out.value_logits.shape == (len(positions), cfg.value_bins)
    for name in vars(out):
        assert torch.isfinite(getattr(out, name)).all(), name


def _count(**overrides) -> int:
    return sum(p.numel() for p in MantisNet(MantisConfig(**overrides)).parameters())


def test_parameter_counts_are_pinned_on_and_off_the_knob():
    assert _count() == 4_537_925
    assert _count(**PRODUCTION) == 5_195_909
    assert _count(**ARM_B) == 5_213_957
    assert _count(**ARM_B, cell_structure=True) == 5_506_565


def test_the_knob_is_inferable_from_a_state_dict():
    for cfg in (
        MantisConfig(**ARM_B, cell_structure=True),
        MantisConfig(cell_latents=True, cell_structure=True),
    ):
        assert infer_config(MantisNet(cfg).state_dict()) == cfg
    # An all-cell profile without the knob still names only `cell_nodes`.
    inferred = infer_config(MantisNet(MantisConfig(**ARM_B)).state_dict())
    assert inferred.cell_structure is False and inferred.cell_nodes is True


def test_the_knob_survives_a_state_dict_round_trip():
    cfg = _tiny(**ARM_B, cell_structure=True)
    torch.manual_seed(15)
    source = MantisNet(cfg).eval()
    for block in source.blocks:
        block.mlp_c.out.weight.data.normal_(std=0.05)
    source.cell_class_table.weight.data.normal_(std=0.1)
    target = MantisNet(cfg).eval()
    target.load_state_dict(source.state_dict(), strict=True)
    batch = _batch()
    with torch.no_grad():
        left, right = source(batch, 0.2), target(batch, 0.2)
    for name in vars(left):
        assert torch.equal(getattr(left, name), getattr(right, name)), name
