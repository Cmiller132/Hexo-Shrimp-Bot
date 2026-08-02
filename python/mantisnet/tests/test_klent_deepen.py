"""The depth graft: more trunk blocks, bitwise the same function."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mantisnet import MantisConfig, MantisNet, collate, from_position
from mantisnet.klent.deepen import (
    BLOCK_OUTPUTS,
    carry_adam,
    deepen,
    deepen_state,
    placement,
)
from mantisnet.klent.run import load_model, model_config, save_checkpoint


def _cfg(blocks: int) -> MantisConfig:
    return MantisConfig(
        h=32, blocks=blocks, heads=2, value_queries=2, value_bins=5,
        policy_hidden=32, value_hidden=32,
    )


def _trained_parent(blocks: int = 4) -> tuple[MantisNet, torch.optim.Adam]:
    """A parent whose readouts are off their zero init and whose Adam has
    stepped, so a carried moment and a preserved function are both testable."""
    torch.manual_seed(3)
    model = MantisNet(_cfg(blocks))
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for p in model.parameters():
        p.grad = torch.randn_like(p) * 0.01
    optimizer.step()
    return model.eval(), optimizer


def test_placement_spreads_inserts_and_never_ends_on_a_new_block():
    assert placement(4, 6) == [0, None, 1, 2, None, 3]
    assert placement(4, 5) == [0, 1, None, 2, 3]
    for parent, blocks in ((4, 6), (4, 5), (4, 8), (2, 3), (6, 9)):
        sites = placement(parent, blocks)
        assert len(sites) == blocks
        assert [s for s in sites if s is not None] == list(range(parent))
        # The heads read the deepest block, and they were fitted against a
        # trained one.
        assert sites[-1] is not None


def test_placement_refuses_a_shallower_or_equal_target():
    for blocks in (3, 4):
        with pytest.raises(ValueError, match="more blocks"):
            placement(4, blocks)


@torch.no_grad()
def test_inserted_blocks_are_exactly_the_identity(positions):
    """The whole claim: the deepened trunk adds 0.0, so every output the
    acting path reads is bitwise the parent's."""
    parent, _opt = _trained_parent()
    sites = placement(4, 6)
    cfg = _cfg(6)
    model = MantisNet(cfg)
    model.load_state_dict(deepen_state(parent.state_dict(), sites, cfg))
    model.eval()

    batch = collate([from_position(p) for p in positions])
    ps, pw, pg = parent.trunk(batch)
    ds, dw, dg = model.trunk(batch)
    assert torch.equal(ps, ds) and torch.equal(pw, dw) and torch.equal(pg, dg)

    a = parent.cell_heads(*parent.trunk(batch)[1:], batch, 0.2)
    b = model.cell_heads(*model.trunk(batch)[1:], batch, 0.2)
    for x, y in zip(a, b):
        assert torch.equal(x, y)


@torch.no_grad()
def test_only_the_four_residual_outputs_are_zeroed(positions):
    """An inserted block is zero on its output, not its input — that is what
    leaves every inserted parameter with a gradient to learn from."""
    sites = placement(4, 6)
    cfg = _cfg(6)
    parent, _opt = _trained_parent()
    state = deepen_state(parent.state_dict(), sites, cfg)
    zeroed = {f"{proj}.{f}" for proj in BLOCK_OUTPUTS for f in ("weight", "bias")}
    for index, source in enumerate(sites):
        if source is None:
            for tail in zeroed:
                assert torch.count_nonzero(state[f"blocks.{index}.{tail}"]) == 0
            others = [
                name for name in state
                if name.startswith(f"blocks.{index}.")
                and name.split(".", 2)[2] not in zeroed
            ]
            assert others, "an inserted block must keep its fresh init elsewhere"
            assert any(torch.count_nonzero(state[name]) for name in others)
        else:
            for tail in zeroed:
                assert torch.equal(
                    state[f"blocks.{index}.{tail}"],
                    parent.state_dict()[f"blocks.{source}.{tail}"],
                )


def test_an_inserted_block_still_receives_gradient():
    """A zeroed output is a starting point, not a dead one."""
    parent, _opt = _trained_parent()
    sites = placement(4, 6)
    cfg = _cfg(6)
    model = MantisNet(cfg)
    model.load_state_dict(deepen_state(parent.state_dict(), sites, cfg))

    import hexo_py
    from mantisnet.builder import collate_positions

    batch = collate_positions([hexo_py.Position.replay([(0, 0), (1, 1), (2, 0)])])
    _s, w, g = model.trunk(batch)
    _policy, _score, q = model.cell_heads(w, g, batch, 0.2)
    q.square().mean().backward()
    inserted = dict(model.blocks[sites.index(None)].named_parameters())
    for proj in BLOCK_OUTPUTS:
        weight = inserted[f"{proj}.weight"]
        assert weight.grad is not None and torch.count_nonzero(weight.grad), proj


def test_carried_moments_follow_the_parent_block_they_came_from():
    parent, optimizer = _trained_parent()
    sites = placement(4, 6)
    cfg = _cfg(6)
    model = MantisNet(cfg)
    model.load_state_dict(deepen_state(parent.state_dict(), sites, cfg))
    parent_names = [n for n, _ in parent.named_parameters()]
    names = [n for n, _ in model.named_parameters()]

    carried = carry_adam(optimizer.state_dict(), parent_names, names, sites)
    torch.optim.Adam(model.parameters()).load_state_dict(carried)

    parent_state = optimizer.state_dict()["state"]
    parent_index = {n: i for i, n in enumerate(parent_names)}
    inserted = {i for i, s in enumerate(sites) if s is None}
    for new_index, name in enumerate(names):
        block = int(name.split(".")[1]) if name.startswith("blocks.") else None
        if block in inserted:
            assert new_index not in carried["state"], name
        else:
            source = (
                name if block is None
                else f"blocks.{sites[block]}." + name.split(".", 2)[2]
            )
            assert torch.equal(
                carried["state"][new_index]["exp_avg"],
                parent_state[parent_index[source]]["exp_avg"],
            )


def test_deepen_writes_a_loadable_checkpoint_and_a_manifest(tmp_path):
    parent, optimizer = _trained_parent()
    old = tmp_path / "parent.pt"
    save_checkpoint(old, parent, optimizer, 75, np.random.default_rng(4))
    new = tmp_path / "deep.pt"

    manifest = deepen(old, new, 6, tau=0.1, lam=0.01, mass_floor=0.2)

    assert manifest["preservation"]["holds"]
    assert manifest["parent_blocks"] == 4 and manifest["blocks"] == 6
    assert manifest["inserted"] == [1, 4]
    assert manifest["parameters"] > manifest["parent_parameters"]
    assert manifest["q_max_abs_delta"] == 0.0
    assert (new.with_suffix(".manifest.json")).exists()

    # The written checkpoint names its own depth, and the loader honours it.
    raw = torch.load(new, map_location="cpu", weights_only=False)
    assert model_config(raw).blocks == 6
    assert load_model(new).cfg.blocks == 6


def test_a_run_refuses_a_checkpoint_of_another_depth(tmp_path):
    """A depth change is a conversion; pointing a 4-block run at a 6-block
    checkpoint must say so rather than dump a shape mismatch."""
    from mantisnet.klent.run import load_checkpoint

    parent, optimizer = _trained_parent()
    old = tmp_path / "parent.pt"
    save_checkpoint(old, parent, optimizer, 75, np.random.default_rng(4))
    new = tmp_path / "deep.pt"
    deepen(old, new, 6, tau=0.1, lam=0.01, mass_floor=0.2)

    shallow = MantisNet(_cfg(4))
    with pytest.raises(ValueError, match="deepen"):
        load_checkpoint(new, shallow, torch.optim.Adam(shallow.parameters()))


def test_a_checkpoint_without_a_recorded_config_reads_as_the_defaults():
    assert model_config({}) == MantisConfig()
    assert model_config({"model_config": {"blocks": 6}}).blocks == 6
