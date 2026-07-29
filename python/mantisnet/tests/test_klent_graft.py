"""The critic-tail graft: preservation, the Adam remap, the manifest, refusals.

The synthetic parent is this build's state dict without the critic tail, with
one Adam step of moments behind it and a nonzero critic readout — a zero
readout would make every action value zero and preserve nothing.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from mantisnet import MantisConfig, MantisNet, collate, from_position
from mantisnet.klent import graft as graft_module
from mantisnet.klent.graft import _ADDED_KEYS, Q_TOLERANCE, _shared_digest, graft, main
from mantisnet.klent.run import _versions, load_checkpoint

TAU, LAM = 0.1, 0.01
_ITERATION = 151


def _parent_checkpoint() -> dict:
    torch.manual_seed(30)
    model = MantisNet(MantisConfig())
    bound = model.cfg.policy_hidden**-0.5
    with torch.no_grad():
        for out in (model.mlp_p.out, model.mlp_q.out):
            out.weight.uniform_(-bound, bound)
            out.bias.uniform_(-bound, bound)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for index, parameter in enumerate(model.parameters()):
        parameter.grad = torch.full_like(parameter, 0.01 * (index % 5 + 1))
    optimizer.step()

    added = set(_ADDED_KEYS)
    names = [name for name, _p in model.named_parameters()]
    shared = [name for name in names if name not in added]
    state = copy.deepcopy(optimizer.state_dict())
    entries = {name: state["state"][index] for index, name in enumerate(names)}
    rng = np.random.default_rng(31)
    return {
        "model": {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
            if name not in added
        },
        "optimizer": {
            "state": {index: entries[name] for index, name in enumerate(shared)},
            "param_groups": [
                {**state["param_groups"][0], "params": list(range(len(shared)))}
            ],
        },
        "iteration": _ITERATION,
        "rng_state": copy.deepcopy(rng.bit_generator.state),
        "versions": _versions(),
    }


@pytest.fixture(scope="module")
def parent() -> dict:
    return _parent_checkpoint()


@pytest.fixture(scope="module")
def grafted(parent, tmp_path_factory) -> dict:
    """One conversion, shared by the tests that read its output."""
    directory = tmp_path_factory.mktemp("graft")
    old, new, manifest = (
        directory / "old.pt",
        directory / "new.pt",
        directory / "manifest.json",
    )
    torch.save(parent, old)
    returned = graft(old, new, TAU, LAM, manifest)
    return {
        "new": new,
        "manifest": json.loads(manifest.read_text(encoding="utf-8")),
        "returned": returned,
        "checkpoint": torch.load(new, map_location="cpu", weights_only=False),
    }


def _spec_q(state: dict, w, g, batch):
    """The parent's action values by MODEL_SPEC appendix B's own formula.

    Independent of the head's folded matrix and of the graft: it sums the
    projected window rows per cell as the spec writes it, so it agrees with the
    grafted model only if the critic still decodes the trunk's rows.
    """
    n_cells, h = batch.cell_pos.shape[0], w.shape[1]
    message = (w @ state["q.weight"].t()).index_select(0, batch.dec_window) + state[
        "e_qw.weight"
    ][batch.dec_class]
    decoded = torch.zeros(n_cells, h).index_add_(0, batch.dec_cell, message)
    if batch.bg_cell.numel():
        decoded.index_copy_(0, batch.bg_cell, state["e_qbg.weight"][batch.bg_bucket])
    hidden = F.relu(
        F.linear(decoded, state["mlp_q.lin_a.weight"], state["mlp_q.lin_a.bias"])
        + F.linear(g, state["mlp_q.lin_b.weight"]).index_select(0, batch.cell_pos)
    )
    raw = F.linear(hidden, state["mlp_q.out.weight"], state["mlp_q.out.bias"])
    return torch.tanh(raw.squeeze(-1))


@torch.no_grad()
def test_graft_preserves_the_parent_action_values(parent, grafted, positions):
    """The stated property, measured by the graft and again independently."""
    manifest = grafted["manifest"]
    assert manifest["preservation"]["holds"]
    # The tail is the exact identity, so the bound is met bit for bit and not
    # merely within tolerance.
    assert manifest["q_max_abs_delta"] == 0.0
    assert manifest["q_mean_abs_delta"] == 0.0
    assert manifest["readout_input_max_abs_delta"] == 0.0
    assert manifest["kl_max"] == 0.0

    model = MantisNet(MantisConfig())
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    rng = np.random.default_rng(99)
    assert load_checkpoint(grafted["new"], model, optimizer, rng) == _ITERATION
    assert rng.bit_generator.state == parent["rng_state"]
    model.eval()

    batch = collate([from_position(p) for p in positions])
    _s, w, g = model.trunk(batch)
    _policy, q = model.cell_heads(w, g, batch)
    # atol covers only the fp32 reassociation between the spec's per-cell sum
    # and the head's folded matrix; a critic reading anything but the trunk's
    # rows moves Q by orders more.
    torch.testing.assert_close(q, _spec_q(parent["model"], w, g, batch), rtol=0, atol=1e-5)


def test_graft_adds_the_tail_and_copies_every_parent_tensor(parent, grafted):
    state = grafted["checkpoint"]["model"]
    assert set(state) - set(parent["model"]) == set(_ADDED_KEYS)
    assert set(parent["model"]) - set(state) == set()
    for name, tensor in parent["model"].items():
        assert torch.equal(state[name], tensor), name
    assert torch.count_nonzero(state["q_tail.2.weight"]) == 0
    assert torch.count_nonzero(state["q_tail.2.bias"]) == 0
    # The added parameters are a function of the recorded build seed alone.
    torch.manual_seed(grafted["manifest"]["build_seed"])
    fresh = MantisNet(MantisConfig()).state_dict()
    for key in _ADDED_KEYS:
        assert torch.equal(state[key], fresh[key]), key
    assert grafted["checkpoint"]["versions"] == parent["versions"]
    assert grafted["checkpoint"]["rng_state"] == parent["rng_state"]


def test_graft_refuses_a_tensor_the_transform_did_not_take_from_the_file(
    parent, tmp_path, monkeypatch
):
    """The bitwise reference is the source file, read a second time: a transform
    whose copy of a parent tensor is not the one on disk is refused even though
    it agrees with the dict it copied from, which is why the reference cannot be
    that dict."""
    old, new, manifest = tmp_path / "old.pt", tmp_path / "new.pt", tmp_path / "m.json"
    torch.save(parent, old)

    real_load = torch.load
    reads = 0

    def _load(*args, **kwargs):
        nonlocal reads
        checkpoint = real_load(*args, **kwargs)
        reads += 1
        if reads == 1:
            state = checkpoint["model"]
            state["ln_out.weight"] = state["ln_out.weight"] * 1.001
        return checkpoint

    monkeypatch.setattr(torch, "load", _load)
    with pytest.raises(ValueError, match="not the source file's: ln_out.weight"):
        graft(old, new, TAU, LAM, manifest)
    assert not new.exists() and not manifest.exists()


@pytest.mark.parametrize("key", ["ln_out.weight", "ln_value.weight"])
def test_graft_refuses_a_grafted_network_that_is_not_the_parents(
    parent, tmp_path, monkeypatch, key
):
    """The network the probe measures is checked against the source file as
    well, at keys the probe cannot speak for: both of its sides read the grafted
    trunk, so a moved trunk tensor cancels in it, and the state-value head is
    not on its path at all."""

    class _MovesAParentTensor(MantisNet):
        def load_state_dict(self, state, *args, **kwargs):
            result = super().load_state_dict(state, *args, **kwargs)
            with torch.no_grad():
                dict(self.named_parameters())[key].mul_(1.001)
            return result

    monkeypatch.setattr(graft_module, "MantisNet", _MovesAParentTensor)
    old, new, manifest = tmp_path / "old.pt", tmp_path / "new.pt", tmp_path / "m.json"
    torch.save(parent, old)
    with pytest.raises(ValueError, match=f"not the source file's: {key}"):
        graft(old, new, TAU, LAM, manifest)
    assert not new.exists() and not manifest.exists()


def test_adam_state_is_remapped_by_name(parent, grafted):
    """The tail lands in the middle of the parameter order, so a positional
    carry would silently hand the critic decoder the policy's moments."""
    model = MantisNet(MantisConfig())
    names = [name for name, _p in model.named_parameters()]
    shapes = {name: parameter.shape for name, parameter in model.named_parameters()}
    parent_names = [name for name in names if name not in set(_ADDED_KEYS)]
    parent_entries = {
        name: parent["optimizer"]["state"][index]
        for index, name in enumerate(parent_names)
    }

    optimizer = grafted["checkpoint"]["optimizer"]
    assert len(optimizer["param_groups"]) == 1
    group = optimizer["param_groups"][0]
    assert group["params"] == list(range(len(names)))
    assert {k: v for k, v in group.items() if k != "params"} == {
        k: v for k, v in parent["optimizer"]["param_groups"][0].items() if k != "params"
    }
    assert set(optimizer["state"]) == set(range(len(names)))

    for index, name in enumerate(names):
        entry = optimizer["state"][index]
        assert set(entry) == {"step", "exp_avg", "exp_avg_sq"}, name
        if name in parent_entries:
            expected = parent_entries[name]
            assert entry["step"] == expected["step"], name
            for field in ("exp_avg", "exp_avg_sq"):
                assert torch.equal(entry[field], expected[field]), f"{name}.{field}"
        else:
            assert entry["step"] == 0, name
            for field in ("exp_avg", "exp_avg_sq"):
                assert entry[field].shape == shapes[name], f"{name}.{field}"
                assert torch.count_nonzero(entry[field]) == 0, f"{name}.{field}"

    # The remapped state is what a fresh Adam over the new model accepts.
    fresh = torch.optim.Adam(model.parameters(), lr=1.0)
    fresh.load_state_dict(optimizer)
    reloaded = fresh.state_dict()["state"]
    for index, name in enumerate(names):
        if name in parent_entries:
            torch.testing.assert_close(
                reloaded[index]["exp_avg"], parent_entries[name]["exp_avg"]
            )


def test_manifest_reports_the_probe_and_the_operating_point(grafted):
    manifest = grafted["manifest"]
    assert manifest == grafted["returned"]
    assert manifest["arm"] == "critic-tail"
    assert manifest["source"].endswith("old.pt")
    assert manifest["source_iteration"] == _ITERATION
    assert manifest["versions"] == _versions()
    assert manifest["added_keys"] == list(_ADDED_KEYS)
    assert "q_tail" in manifest["transform"]
    assert (manifest["tau"], manifest["lam"]) == (TAU, LAM)
    assert manifest["probe"]["positions"] == 64
    assert manifest["probe"]["legal_cells"] > manifest["probe"]["positions"]
    assert manifest["probe"]["seed"] == graft_module.PROBE_SEED
    assert manifest["probe"]["plies"] == [20, 60]
    # The spread the operator exponentiates is unchanged, and is a real number:
    # a graft that flattened the critic would report a smaller one.
    assert manifest["q_spread_median_new"] == manifest["q_spread_median_parent"] > 0.0
    assert manifest["preservation"]["q_max_abs_delta_tolerance"] == Q_TOLERANCE


def test_manifest_fingerprints_every_parent_tensor(parent, grafted):
    """The recorded digest covers the parent's tensors, all of them and in
    order, and moves when any one of them does."""
    recorded = grafted["manifest"]["preservation"]["shared_tensor_sha256"]
    shared = list(parent["model"])
    assert grafted["manifest"]["preservation"]["shared_tensors_unchanged"] == len(shared)
    assert recorded == _shared_digest(parent["model"], shared)
    moved = {**parent["model"]}
    moved["ln_value.weight"] = moved["ln_value.weight"] * 1.001
    assert _shared_digest(moved, shared) != recorded
    assert _shared_digest(parent["model"], shared[:-1]) != recorded


def test_a_live_tail_fails_the_graft_and_writes_nothing(parent, tmp_path, monkeypatch):
    """The preservation measurement is a detector: give the tail a nonzero
    output linear and the conversion refuses instead of shipping."""

    class _LiveTail(MantisNet):
        def _init_weights(self) -> None:
            super()._init_weights()
            torch.nn.init.normal_(self.q_tail[2].weight, std=0.05)

    monkeypatch.setattr(graft_module, "MantisNet", _LiveTail)
    old, new, manifest = tmp_path / "old.pt", tmp_path / "new.pt", tmp_path / "m.json"
    torch.save(parent, old)
    with pytest.raises(ValueError, match="did not preserve the parent's critic"):
        graft(old, new, TAU, LAM, manifest)
    assert not new.exists() and not manifest.exists()


# Each case mutates a copy of the parent checkpoint in place; the graft must
# refuse it with the given message and leave no output behind.
_REFUSALS = {
    "extra top-level key": (
        lambda c: c.update(extra=1),
        "checkpoint must hold exactly",
    ),
    "model is not a state dict": (
        lambda c: c.update(model=[]),
        "model entry must be a state dict",
    ),
    "stale versions": (
        lambda c: c["versions"].update(torch="0.0.0"),
        "!= this build",
    ),
    "another architecture difference": (
        lambda c: c["model"].__setitem__(
            "stone_table.weight", c["model"]["stone_table.weight"][:1]
        ),
        "stone_table.weight",
    ),
    "unexpected critic readout width": (
        lambda c: c["model"].update(
            {
                "mlp_q.out.weight": c["model"]["mlp_q.out.weight"].expand(2, -1).clone(),
                "mlp_q.out.bias": c["model"]["mlp_q.out.bias"].expand(2).clone(),
            }
        ),
        "mismatched keys: mlp_q.out.bias, mlp_q.out.weight",
    ),
    "the tail is already there": (
        lambda c: c["model"].update({key: torch.zeros(1) for key in _ADDED_KEYS}),
        "already has the critic tail",
    ),
    "parameters out of order": (
        lambda c: c.update(model=dict(reversed(list(c["model"].items())))),
        "not in this build's parameter order",
    ),
    "no optimizer state": (
        lambda c: c.update(optimizer={"param_groups": []}),
        "must contain an Adam optimizer state dict",
    ),
    "two param groups": (
        lambda c: c["optimizer"]["param_groups"].append(
            copy.deepcopy(c["optimizer"]["param_groups"][0])
        ),
        "exactly one param_group",
    ),
    "wrong parameter count": (
        lambda c: c["optimizer"]["param_groups"][0]["params"].pop(),
        "optimizer parameter count does not match",
    ),
    "missing Adam entry": (
        lambda c: c["optimizer"]["state"].pop(3),
        "optimizer has no Adam state for window_table.weight",
    ),
    "missing Adam moment": (
        lambda c: c["optimizer"]["state"][3].pop("exp_avg_sq"),
        "is missing: exp_avg_sq",
    ),
    "moment of the wrong shape": (
        lambda c: c["optimizer"]["state"][3].update(
            exp_avg=c["optimizer"]["state"][3]["exp_avg"][:1]
        ),
        "does not have the parameter's shape",
    ),
}


@pytest.mark.parametrize("case", list(_REFUSALS), ids=list(_REFUSALS))
def test_graft_refusals(parent, tmp_path, case):
    mutate, message = _REFUSALS[case]
    checkpoint = copy.deepcopy(parent)
    mutate(checkpoint)
    old, new, manifest = tmp_path / "old.pt", tmp_path / "new.pt", tmp_path / "m.json"
    torch.save(checkpoint, old)
    with pytest.raises(ValueError, match=message):
        graft(old, new, TAU, LAM, manifest)
    assert not new.exists() and not manifest.exists()


def test_graft_refuses_one_path_for_both_checkpoints(parent, tmp_path):
    old = tmp_path / "old.pt"
    torch.save(parent, old)
    with pytest.raises(ValueError, match="must be different paths"):
        graft(old, old, TAU, LAM, tmp_path / "m.json")


@pytest.mark.parametrize(
    "argv",
    [
        ["old.pt", "new.pt", "--lam", "0.01", "--manifest", "m.json"],
        ["old.pt", "new.pt", "--tau", "0.1", "--manifest", "m.json"],
        ["old.pt", "new.pt", "--tau", "0.1", "--lam", "0.01"],
    ],
)
def test_cli_requires_the_operating_point_and_the_manifest(argv):
    with pytest.raises(SystemExit) as exit_info:
        main(argv)
    assert exit_info.value.code == 2
