"""Step M1 merged-site trunk: the unified stone/cell token set (§5M).

Detectors here pin the claims the merge rests on: the six-edge window
incidence is exact, the hoisted per-pattern class sum equals the literal
edge composition, uncovered legal cells are live tokens (they hear the
radius read and the global state), and both mixing arms round-trip the
family registry.
"""

import dataclasses

import pytest
import torch

import hexo_py
from mantisnet.builder import (
    TERN_OCC_CLASSES,
    collate,
    from_position,
)
from mantisnet.lab.families import infer_config, load_checkpoint
from mantisnet.model import MantisConfig, MantisNet
from mantisnet import cell_nodes, message_passing

from .conftest import random_moves

_ARMS = (
    {"merged_sites": True},
    {"merged_sites": True, "site_self_attention": False},
)

_TINY = dict(h=32, heads=2, blocks=2, policy_hidden=32, value_hidden=32)


def _batch(plies=9, seed=3, extra=None):
    games = [random_moves(plies, seed=seed)]
    if extra is not None:
        games.append(extra)
    return collate(
        [from_position(hexo_py.Position.replay(m)) for m in games]
    )


def test_param_pins():
    assert sum(
        p.numel() for p in MantisNet(MantisConfig(merged_sites=True)).parameters()
    ) == 5_048_389
    assert sum(
        p.numel()
        for p in MantisNet(
            MantisConfig(merged_sites=True, site_self_attention=False)
        ).parameters()
    ) == 5_315_141


def test_config_validation():
    with pytest.raises(ValueError, match="cell knobs"):
        MantisConfig(merged_sites=True, cell_latents=True)
    with pytest.raises(ValueError, match="cell knobs"):
        MantisConfig(merged_sites=True, cell_nodes=True)
    with pytest.raises(ValueError, match="inert"):
        MantisConfig(site_self_attention=False)


def test_every_kept_window_has_exactly_six_site_edges():
    batch = _batch(plies=11, seed=5)
    n_stones = batch.stone_own.shape[0]
    msite = torch.cat([batch.inc_stone, batch.dec_cell + n_stones])
    mwin = torch.cat([batch.inc_window, batch.dec_window])
    mcls = torch.cat([batch.inc_class, batch.dec_class + TERN_OCC_CLASSES])
    n_windows = batch.window_feat.shape[0]
    n_sites = n_stones + batch.cell_pos.shape[0]
    tab = cell_nodes.edge_tables(
        mwin, msite, mcls, msite.new_empty(0), n_windows,
        n_sites, TERN_OCC_CLASSES + 726, False,
    )
    runs = tab.win_ptr[1:] - tab.win_ptr[:-1]
    assert runs.shape[0] == n_windows
    assert bool((runs == 6).all()), runs.bincount()
    # Every site sits in at most 18 windows (six per axis).
    site_runs = tab.cell_ptr[1:] - tab.cell_ptr[:-1]
    assert int(site_runs.max()) <= 18


def test_hoisted_class_sum_matches_the_literal_edges():
    """``site_pattern_counts @ e_wsite`` gathered per window equals the
    literal class-row sum over the merged incidence."""
    torch.manual_seed(0)
    batch = _batch(plies=13, seed=9)
    model = MantisNet(MantisConfig(**_TINY, merged_sites=True))
    n_stones = batch.stone_own.shape[0]
    mwin = torch.cat([batch.inc_window, batch.dec_window])
    mcls = torch.cat([batch.inc_class, batch.dec_class + TERN_OCC_CLASSES])
    table = model.blocks[0].e_wsite.weight
    hoisted = (model.site_pattern_counts @ table).index_select(
        0, batch.window_feat
    )
    order = torch.argsort(mwin, stable=True)
    literal = message_passing.class_row_sum(
        table,
        mcls.index_select(0, order),
        mwin.index_select(0, order),
        batch.window_feat.shape[0],
        message_passing.WINDOW_RUN,
    )
    assert torch.allclose(hoisted, literal, atol=1e-5)


@pytest.mark.parametrize("arm", _ARMS, ids=("full", "latent"))
def test_forward_backward_and_decode_layout(arm):
    torch.manual_seed(0)
    batch = _batch(plies=9, seed=3, extra=random_moves(5, seed=11))
    model = MantisNet(MantisConfig(**_TINY, **arm)).train()
    with torch.no_grad():
        for head in (model.mlp_p, model.mlp_q):
            torch.nn.init.normal_(head.out.weight, std=0.1)
    out = model(batch, 0.2)
    assert out.policy_logits.shape[0] == batch.n_cells
    assert out.value.shape[0] == batch.n_pos
    (out.policy_logits.sum() + out.q_values.sum() + out.value.sum()).backward()
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        assert torch.isfinite(p.grad).all(), name
    for probe in (
        model.blocks[0].e_wsite.weight,
        model.blocks[0].sr_vclass.weight,
        model.blocks[0].radius_vclass.weight,
        model.cell_base,
    ):
        assert probe.grad is not None and probe.grad.abs().sum() > 0


@pytest.mark.parametrize("arm", _ARMS, ids=("full", "latent"))
def test_merged_model_is_d6_invariant(arm):
    from mantisnet.klent import telemetry

    transform = telemetry.D6_TRANSFORMS[1]
    cfg = MantisConfig(**_TINY, **arm)
    torch.manual_seed(1)
    model = MantisNet(cfg).eval()
    moves = random_moves(9, seed=21)
    pos = hexo_py.Position.replay(moves)
    turned_pos = hexo_py.Position.replay([transform(m) for m in moves])
    base = collate([from_position(pos)])
    turned = collate([from_position(turned_pos)])
    with torch.no_grad():
        got = model(base, 0.2)
        got_turned = model(turned, 0.2)
    assert torch.allclose(got.value, got_turned.value, atol=1e-5)
    base_map = dict(zip(pos.legal_moves(), got.policy_logits.tolist()))
    turned_map = dict(
        zip(turned_pos.legal_moves(), got_turned.policy_logits.tolist())
    )
    assert set(turned_map) == {transform(m) for m in base_map}
    for move, logit in base_map.items():
        assert turned_map[transform(move)] == pytest.approx(logit, abs=1e-5)


def _uncovered_cells(batch):
    """Legal-cell indices with no decoder (live-window) incidence."""
    covered = torch.zeros(batch.n_cells, dtype=torch.bool)
    covered[batch.dec_cell] = True
    return (~covered).nonzero().squeeze(1)


@pytest.mark.parametrize("arm", _ARMS, ids=("full", "latent"))
def test_uncovered_cells_hear_the_global_state(arm):
    """A legal cell no live window covers still changes its decode row when
    only the global context changes (moves_remaining) — the trunk keeps
    every legal cell in the planning loop."""
    torch.manual_seed(2)
    moves = random_moves(9, seed=7)
    pos = hexo_py.Position.replay(moves)
    graph = from_position(pos)
    batch = collate([graph])
    uncovered = _uncovered_cells(batch)
    assert uncovered.numel(), "playout has no uncovered legal cell; reseed"
    other = dataclasses.replace(graph, moves_remaining=2 - graph.moves_remaining + 1)
    assert other.moves_remaining != graph.moves_remaining
    flipped = collate([other])
    model = MantisNet(MantisConfig(**_TINY, **arm)).eval()
    with torch.no_grad():
        _, _, cells = model.trunk(batch)
        _, _, cells_flipped = model.trunk(flipped)
    delta = (cells - cells_flipped).abs().sum(dim=1)
    assert bool((delta[uncovered] > 0).all())


def test_uncovered_cells_hear_the_radius_read():
    """Two uncovered cells with identical nearest-stone buckets but
    different stone geometry decode differently — the typed radius read,
    not the bucket table, is doing the work."""
    torch.manual_seed(4)
    moves = random_moves(9, seed=7)
    batch = collate([from_position(hexo_py.Position.replay(moves))])
    uncovered = _uncovered_cells(batch)
    buckets = batch.cell_nearest[uncovered]
    same = None
    for bucket in buckets.unique():
        members = uncovered[buckets == bucket]
        if members.numel() >= 2:
            same = members[:2]
            break
    assert same is not None, "no bucket with two uncovered cells; reseed"
    model = MantisNet(MantisConfig(**_TINY, merged_sites=True)).eval()
    with torch.no_grad():
        _, _, cells = model.trunk(batch)
    assert not torch.allclose(cells[same[0]], cells[same[1]])


@pytest.mark.parametrize("arm", _ARMS, ids=("full", "latent"))
def test_checkpoint_round_trips_through_the_family_registry(tmp_path, arm):
    cfg = MantisConfig(**arm)
    torch.manual_seed(3)
    model = MantisNet(cfg)
    inferred = infer_config(model.state_dict())
    assert inferred == cfg
    MantisNet(inferred).load_state_dict(model.state_dict(), strict=True)

    path = tmp_path / "merged.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "versions": {
                "MODEL_REPR_VERSION": hexo_py.MODEL_REPR_VERSION,
                "RULES_VERSION": hexo_py.RULES_VERSION,
                "ACTION_ORDER_VERSION": hexo_py.ACTION_ORDER_VERSION,
                "torch": torch.__version__,
            },
            "iteration": 1,
            "model_config": dataclasses.asdict(cfg),
        },
        path,
    )
    loaded = load_checkpoint(path)
    assert loaded.family.name == "trinomial-joint"
    assert loaded.config == cfg


def test_split_and_merged_dicts_do_not_cross_load():
    """A merged state dict refuses the split profile's identification and
    vice versa, rather than half-loading."""
    merged = MantisNet(MantisConfig(merged_sites=True)).state_dict()
    split = MantisNet(MantisConfig(cell_latents=True)).state_dict()
    assert infer_config(merged).merged_sites
    assert not infer_config(split, heads=4).merged_sites
    with pytest.raises(RuntimeError):
        MantisNet(MantisConfig(cell_latents=True)).load_state_dict(
            merged, strict=True
        )
