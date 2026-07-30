"""Explicit scalar-critic checkpoint conversion to the dueling critic.

The graft adds the baseline readout and centers the parent's critic decoder on
the raw policy, so it preserves each position's ordering of Q and resets the
level. These tests establish that ordering independently of the manifest's own
claim, pin the Adam remap, and exercise every refusal.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import mantisnet.model as model_module
from mantisnet import MantisConfig, MantisNet, collate, from_position
from mantisnet.klent import graft as graft_module
from mantisnet.klent.graft import (
    _ADDED_KEYS,
    COMPOSITION_TOL,
    INIT_SEED,
    PROBE_SEED,
    graft,
)
from mantisnet.klent.run import _versions, load_checkpoint
from mantisnet.segments import segment_sum


def _parent_checkpoint() -> dict:
    """A trained scalar-critic checkpoint in the parent's own format.

    The critic readout is perturbed away from its zero initialization, because
    a parent whose Q is constant orders nothing and the ordering property would
    have nothing to preserve. One Adam step gives every parameter real moments.
    """
    torch.manual_seed(30)
    model = MantisNet(MantisConfig())
    for out in (model.mlp_p.out, model.mlp_q.out):
        torch.nn.init.normal_(out.weight, std=0.3)
        torch.nn.init.normal_(out.bias, std=0.3)
    optimizer = torch.optim.Adam(model.parameters())
    generator = torch.Generator().manual_seed(31)
    for parameter in model.parameters():
        parameter.grad = torch.randn(parameter.shape, generator=generator)
    optimizer.step()

    added = set(_ADDED_KEYS)
    state = {k: v for k, v in copy.deepcopy(model.state_dict()).items() if k not in added}
    saved = copy.deepcopy(optimizer.state_dict())
    kept = [i for i, (name, _p) in enumerate(model.named_parameters()) if name not in added]
    saved["state"] = {new: saved["state"][old] for new, old in enumerate(kept)}
    saved["param_groups"][0]["params"] = list(range(len(kept)))

    rng = np.random.default_rng(32)
    return {
        "model": state,
        "optimizer": saved,
        "iteration": 151,
        "rng_state": copy.deepcopy(rng.bit_generator.state),
        "versions": _versions(),
    }


def _write(tmp_path, checkpoint) -> tuple:
    old = tmp_path / "old.pt"
    torch.save(checkpoint, old)
    return old, tmp_path / "new.pt", tmp_path / "manifest.json"


@pytest.fixture(scope="module")
def grafted(tmp_path_factory) -> dict:
    """One successful graft, shared by the tests that read its result."""
    directory = tmp_path_factory.mktemp("graft")
    parent = _parent_checkpoint()
    old, new, manifest_path = _write(directory, parent)
    manifest = graft(old, new, tau=0.1, lam=0.01, manifest_path=manifest_path)
    return {
        "parent": parent,
        "converted": torch.load(new, map_location="cpu", weights_only=False),
        "manifest": manifest,
        "paths": (old, new, manifest_path),
    }


def _parent_advantage(model, batch):
    """Appendix B's raw critic score, transcribed from the spec's formula.

    Every parameter it reads is the parent's, carried verbatim by the graft, so
    this is the parent's advantage and ``tanh`` of it is the parent's Q. It goes
    through neither ``cell_heads`` nor ``graft``.
    """
    _s, w, g = model.trunk(batch)
    message = (w @ model.q.weight.t()).index_select(0, batch.dec_window)
    message = message + model.e_qw.weight[batch.dec_class]
    h = torch.zeros(batch.cell_pos.shape[0], w.shape[1])
    h.index_add_(0, batch.dec_cell, message)
    if batch.bg_cell.numel():
        h.index_copy_(0, batch.bg_cell, model.e_qbg.weight[batch.bg_bucket])
    token = model.mlp_q.lin_b(g).index_select(0, batch.cell_pos)
    return model.mlp_q.out(F.relu(model.mlp_q.lin_a(h) + token)).squeeze(-1)


# The transcription above and the head's folded matrix agree to a few times
# fp32 epsilon, so pairs the reference separates by less than this are noise,
# not order. The graft's own manifest needs no such margin: it applies the
# parent's readout to the activation the head itself produced, which is the
# same arithmetic bit for bit.
_REFERENCE_TOL = 1e-4


@torch.no_grad()
def test_graft_preserves_every_position_ordering_and_resets_the_level(grafted, positions):
    """The arm's stated property, on real positions and off an independent
    advantage: no pair of legal cells changes order, some pair still strictly
    orders, and what moved is exactly one level per position — ``v`` at zero
    minus the policy's expectation of the advantage."""
    model = MantisNet(MantisConfig())
    model.load_state_dict(grafted["converted"]["model"])
    model.eval()

    batch = collate([from_position(p) for p in positions])
    advantage = _parent_advantage(model, batch)
    q_parent = torch.tanh(advantage)
    _logits, q_new = model.cell_heads(*model.trunk(batch)[1:], batch)

    assert torch.count_nonzero(model.mlp_qbase[-1].weight) == 0, "v(s) must start at zero"
    bounds = batch.legal_offsets.tolist()
    ordered = 0
    for i in range(batch.n_pos):
        low, high = bounds[i], bounds[i + 1]
        before, after = q_parent[low:high], q_new[low:high]
        strict = before[:, None] > before[None, :] + _REFERENCE_TOL
        assert not bool((strict & (after[:, None] < after[None, :])).any()), f"position {i}"
        ordered += int((strict & (after[:, None] > after[None, :])).sum())
        # Level reset: with ``v`` at zero the only thing subtracted is the raw
        # policy's expectation of the parent's advantage, one level per
        # position, which is why the ordering survives and the values do not.
        pi = _logits[low:high].softmax(0)
        raw = advantage[low:high]
        torch.testing.assert_close(
            after, torch.tanh(raw - (pi * raw).sum()), rtol=1e-4, atol=1e-5
        )
    assert ordered > 0, "no pair of cells is strictly ordered, so nothing was preserved"
    assert float((q_new - q_parent).abs().max()) > 1e-3, "the graft is not function preserving"


@pytest.mark.parametrize("damage", ["reorder", "collapse"])
def test_graft_refuses_a_transform_that_breaks_the_ordering(tmp_path, monkeypatch, damage):
    """The enforcement is a detector: a composition that inverts the parent's
    ordering, or one that throws the trained critic away and leaves Q constant,
    both fail and write nothing."""
    heads = MantisNet.cell_heads

    def damaged(self, w, g, batch):
        logits, q = heads(self, w, g, batch)
        return logits, -q if damage == "reorder" else torch.zeros_like(q)

    monkeypatch.setattr(MantisNet, "cell_heads", damaged)
    old, new, manifest_path = _write(tmp_path, _parent_checkpoint())
    with pytest.raises(ValueError, match="did not preserve the parent's ordering"):
        graft(old, new, tau=0.1, lam=0.01, manifest_path=manifest_path)
    assert not new.exists() and not manifest_path.exists()


def test_adam_state_is_remapped_by_name(grafted):
    """Shared parameters keep their moments even though inserting the baseline
    readout shifts their positions; the added ones start at zero with a zeroed
    step of the parent's own type; the group is contiguous over the new order.
    """
    parent, converted = grafted["parent"], grafted["converted"]
    parent_names = list(parent["model"])
    new_names = list(converted["model"])
    added = set(_ADDED_KEYS)
    assert [n for n in new_names if n not in added] == parent_names

    group = converted["optimizer"]["param_groups"]
    assert len(group) == 1
    assert group[0]["params"] == list(range(len(new_names)))
    assert {k: v for k, v in group[0].items() if k != "params"} == {
        k: v for k, v in parent["optimizer"]["param_groups"][0].items() if k != "params"
    }

    shifted = 0
    for new_id, name in enumerate(new_names):
        entry = converted["optimizer"]["state"][new_id]
        if name in added:
            assert torch.count_nonzero(entry["exp_avg"]) == 0
            assert torch.count_nonzero(entry["exp_avg_sq"]) == 0
            assert entry["exp_avg"].shape == converted["model"][name].shape
            assert entry["exp_avg_sq"].shape == converted["model"][name].shape
            template = parent["optimizer"]["state"][0]["step"]
            assert type(entry["step"]) is type(template) and float(entry["step"]) == 0.0
            continue
        old_id = parent_names.index(name)
        shifted += old_id != new_id
        expected = parent["optimizer"]["state"][old_id]
        assert entry.keys() == expected.keys()
        for field, value in expected.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(entry[field], value)
            else:
                assert entry[field] == value
    assert shifted > 0, "no position moved, so the remap was not exercised"


def test_grafted_checkpoint_loads_and_carries_the_run_forward(grafted, tmp_path):
    """The written file is this build's format: strict loading, and iteration,
    RNG state and versions unchanged."""
    parent, converted = grafted["parent"], grafted["converted"]
    _old, new, _manifest = grafted["paths"]
    assert converted["iteration"] == parent["iteration"] == 151
    assert converted["rng_state"] == parent["rng_state"]
    assert converted["versions"] == parent["versions"]
    for name, value in parent["model"].items():
        torch.testing.assert_close(converted["model"][name], value)

    model = MantisNet(MantisConfig())
    optimizer = torch.optim.Adam(model.parameters())
    rng = np.random.default_rng(99)
    assert load_checkpoint(new, model, optimizer, rng) == 151
    assert rng.bit_generator.state == parent["rng_state"]


def test_manifest_records_the_operating_point_and_what_it_measured(grafted):
    """What a reader needs to reproduce and judge the graft, on disk as strict
    JSON identical to the returned record."""
    manifest, (old, _new, manifest_path) = grafted["manifest"], grafted["paths"]
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest

    assert manifest["arm"] == "dueling-critic"
    assert manifest["source"] == str(old)
    assert manifest["source_iteration"] == 151
    assert manifest["versions"] == _versions()
    assert (manifest["tau"], manifest["lam"]) == (0.1, 0.01)
    assert "tanh(A - E_pi_theta[A])" in manifest["transform"]
    assert manifest["init_seed"] == INIT_SEED
    assert manifest["probe"] == {
        "seed": PROBE_SEED,
        "positions": 64,
        "legal_cells": manifest["probe"]["legal_cells"],
        "prefix_plies": [20, 60],
        "top_k": 16,
    }
    assert manifest["probe"]["legal_cells"] > 64

    # What makes the ordering below a statement about the parent rather than
    # about the grafted head's own arithmetic.
    premise = manifest["premise"]
    assert premise["holds"]
    assert premise["parameters_carried_bitwise"] == len(grafted["parent"]["model"])
    assert premise["baseline_abs_max"] == 0.0
    assert premise["q_abs_diff_from_the_formula_max"] <= COMPOSITION_TOL

    kept = manifest["preservation"]
    assert kept["holds"] and kept["discordant_pairs"] == 0
    assert kept["rank_agreement"] == 1.0
    assert 0 < kept["concordant_pairs"] <= kept["comparable_pairs"]

    measured = manifest["measured"]
    assert set(measured) == {
        "q_abs_diff_max",
        "q_abs_diff_mean",
        "removed_level_median",
        "top_k_sigma_q_parent_median",
        "top_k_sigma_q_new_median",
        "improved_policy_kl_mean",
        "improved_policy_kl_max",
        "order_collapsed_pairs",
    }
    assert measured["q_abs_diff_max"] >= measured["q_abs_diff_mean"] > 0
    assert measured["improved_policy_kl_max"] >= measured["improved_policy_kl_mean"] > 0
    assert measured["top_k_sigma_q_new_median"] != measured["top_k_sigma_q_parent_median"]


def _centre_on_the_uniform_distribution(monkeypatch):
    """A centering weight that is not π_θ. Only the head is corrupted: the
    graft's own reference holds its own binding of the helper."""

    def uniform(values, offsets, seg):
        counts = (offsets[1:] - offsets[:-1]).to(values.dtype)
        return -counts.log().index_select(0, seg)

    monkeypatch.setattr(model_module, "segment_log_softmax", uniform)


def _shift_the_level_to_the_next_position(monkeypatch):
    """Every position centered on its neighbour's level: still one constant per
    position, so still monotone inside each of them."""
    monkeypatch.setattr(
        model_module, "segment_sum", lambda values, offsets: segment_sum(values, offsets).roll(1)
    )


def _start_the_baseline_away_from_zero(monkeypatch):
    """A baseline readout whose output layer is not zero, so v(s) ≠ 0 and the
    level the graft states it removed is not the level it removed."""
    fresh = graft_module._fresh_model

    def perturbed():
        model = fresh()
        torch.manual_seed(41)
        for parameter in model.mlp_qbase[-1].parameters():
            torch.nn.init.normal_(parameter, std=0.3)
        return model

    monkeypatch.setattr(graft_module, "_fresh_model", perturbed)


def _load_a_parameter_the_parent_does_not_hold(monkeypatch):
    """A shared parameter that reaches the measured forward changed — the class
    of edit that would leave the probe measuring something other than the
    parent while every transform in sight stayed correct."""
    load = MantisNet.load_state_dict

    def damaged(self, state, *args, **kwargs):
        zeroed = torch.zeros_like(state["ln_out.weight"])
        return load(self, {**state, "ln_out.weight": zeroed}, *args, **kwargs)

    monkeypatch.setattr(MantisNet, "load_state_dict", damaged)


@pytest.mark.parametrize(
    "corrupt, message",
    [
        (_centre_on_the_uniform_distribution, "not appendix B's composition"),
        (_shift_the_level_to_the_next_position, "not appendix B's composition"),
        (_start_the_baseline_away_from_zero, "not appendix B's composition"),
        (_load_a_parameter_the_parent_does_not_hold, "into a different tensor"),
    ],
)
def test_graft_refuses_a_premise_it_cannot_establish(tmp_path, monkeypatch, corrupt, message):
    """The ordering property is a statement about the *parent*, and these four
    are ways it stops being one without a single pair changing order: ``tanh`` of
    a per-position constant shift is monotone whatever the constant is and
    whatever produced it. Each refusal is the premise's own — matching the
    ordering's message instead would mean the corruption was visible there.
    """
    corrupt(monkeypatch)
    old, new, manifest_path = _write(tmp_path, _parent_checkpoint())
    with pytest.raises(ValueError, match=message):
        graft(old, new, tau=0.1, lam=0.01, manifest_path=manifest_path)
    assert not new.exists() and not manifest_path.exists()


def _drop_optimizer(checkpoint):
    del checkpoint["optimizer"]


def _replace_an_adam_entry_with_a_scalar(checkpoint):
    """An absent entry is a parameter Adam never stepped, which is legitimate;
    an entry that is not a state dict is a malformed file."""
    checkpoint["optimizer"]["state"][len(checkpoint["model"]) - 1] = 0.0


def _widen_readout(checkpoint):
    for key in ("mlp_q.out.weight", "mlp_q.out.bias"):
        checkpoint["model"][key] = checkpoint["model"][key].repeat(
            2, *([1] * (checkpoint["model"][key].ndim - 1))
        )


def _shrink_a_table(checkpoint):
    checkpoint["model"]["stone_table.weight"] = checkpoint["model"]["stone_table.weight"][:1]


def _add_a_baseline_key(checkpoint):
    """A parent that already carries a baseline readout, at another width: this
    graft adds keys and must never overwrite one."""
    checkpoint["model"][_ADDED_KEYS[0]] = torch.zeros(64, 128)


def _reorder(checkpoint):
    names = list(checkpoint["model"])
    checkpoint["model"] = {n: checkpoint["model"][n] for n in reversed(names)}


def _split_param_groups(checkpoint):
    groups = checkpoint["optimizer"]["param_groups"]
    ids = groups[0]["params"]
    checkpoint["optimizer"]["param_groups"] = [
        {**groups[0], "params": ids[: len(ids) // 2]},
        {**groups[0], "params": ids[len(ids) // 2 :]},
    ]


def _renumber_param_ids(checkpoint):
    ids = checkpoint["optimizer"]["param_groups"][0]["params"]
    checkpoint["optimizer"]["param_groups"][0]["params"] = [i + 1 for i in ids]


def _drift_versions(checkpoint):
    checkpoint["versions"] = {**checkpoint["versions"], "MODEL_REPR_VERSION": 999}


def _add_amsgrad_state(checkpoint):
    entry = checkpoint["optimizer"]["state"][0]
    entry["max_exp_avg_sq"] = torch.zeros_like(entry["exp_avg_sq"])


@pytest.mark.parametrize(
    "damage, message",
    [
        (lambda c: c.clear(), "model state dict"),
        (lambda c: c.update(model=[]), "model state dict"),
        (_drop_optimizer, "missing optimizer"),
        (lambda c: c.update(optimizer=[]), "Adam optimizer state dict"),
        (_drift_versions, "versions"),
        (_widen_readout, "scalar critic parent"),
        (_shrink_a_table, "stone_table.weight"),
        (_add_a_baseline_key, "already carries"),
        (_reorder, "registration order"),
        (_split_param_groups, "exactly one Adam param group"),
        (_renumber_param_ids, "packed positions"),
        (_replace_an_adam_entry_with_a_scalar, "is not a state dict"),
        (_add_amsgrad_state, "expected exactly"),
    ],
)
def test_graft_refuses(tmp_path, damage, message):
    checkpoint = _parent_checkpoint()
    damage(checkpoint)
    old, new, manifest_path = _write(tmp_path, checkpoint)
    with pytest.raises(ValueError, match=message):
        graft(old, new, tau=0.1, lam=0.01, manifest_path=manifest_path)
    assert not new.exists() and not manifest_path.exists()


def test_graft_refuses_writing_over_its_source(tmp_path):
    old, _new, manifest_path = _write(tmp_path, _parent_checkpoint())
    with pytest.raises(ValueError, match="must be different paths"):
        graft(old, old, tau=0.1, lam=0.01, manifest_path=manifest_path)
    assert not manifest_path.exists()


def test_cli_requires_the_operating_point(tmp_path, capsys):
    old, new, manifest_path = _write(tmp_path, _parent_checkpoint())
    for argv in (
        [str(old), str(new), "--lam", "0.01", "--manifest", str(manifest_path)],
        [str(old), str(new), "--tau", "0.1", "--manifest", str(manifest_path)],
        [str(old), str(new), "--tau", "0.1", "--lam", "0.01"],
    ):
        with pytest.raises(SystemExit):
            graft_module.main(argv)
        assert "required" in capsys.readouterr().err
    assert not new.exists() and not manifest_path.exists()


def test_graft_carries_an_unstepped_parameter_as_unstepped(tmp_path):
    """Adam's state dict is sparse. KLENT never steps the state-value head, so
    no checkpoint this repo writes holds moments for it, and the conversion
    carries that absence instead of refusing the file or inventing moments."""
    parent = _parent_checkpoint()
    parent_names = list(parent["model"])
    unstepped = [
        name
        for name in parent_names
        if name.startswith(("value_queries", "ln_value", "mlp_v"))
    ]
    assert len(unstepped) == 7, unstepped
    for name in unstepped:
        del parent["optimizer"]["state"][parent_names.index(name)]

    old, new, manifest_path = _write(tmp_path, parent)
    graft(old, new, tau=0.1, lam=0.01, manifest_path=manifest_path)

    converted = torch.load(new, map_location="cpu", weights_only=False)
    names = list(converted["model"])
    state = converted["optimizer"]["state"]
    assert {names[index] for index in state} == set(names) - set(unstepped)
    load_checkpoint(new, MantisNet(MantisConfig()), torch.optim.Adam(
        MantisNet(MantisConfig()).parameters()
    ))
