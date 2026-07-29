"""The one-way conversion of a scalar-tanh-critic checkpoint (KLENT §7.1).

The parent checkpoints are synthetic but in the parent format: a full-size
model state dict whose action-value readout is one nontrivial row, and a
single-group Adam state whose moments match it.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Callable

import hexo_py
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from mantisnet import MantisConfig, MantisNet, collate, from_position
from mantisnet.klent import graft as graft_mod
from mantisnet.klent.graft import (
    ARM,
    MAX_ABS_DQ,
    MAX_MEAN_KL,
    PROBE_PLIES,
    PROBE_POSITIONS,
    PROBE_TOP_K,
    TRANSFORM,
    _probe_prefixes,
    graft,
    main,
)
from mantisnet.klent.run import _versions, load_checkpoint

_READOUT_KEYS = ("mlp_q.out.weight", "mlp_q.out.bias")
_ITERATION = 151
_TAU, _LAM = 0.1, 0.01


def _parent_checkpoint() -> tuple[dict, list[int], dict[str, int]]:
    """A scalar-critic checkpoint, its optimizer parameter ids, and the name to
    position map of the model that wrote it."""
    torch.manual_seed(30)
    model = MantisNet(MantisConfig())
    optimizer = torch.optim.Adam(model.parameters())
    generator = torch.Generator().manual_seed(41)
    for parameter in model.parameters():
        parameter.grad = torch.randn(parameter.shape, generator=generator) * 0.01
    optimizer.step()

    state = copy.deepcopy(model.state_dict())
    hidden = state[_READOUT_KEYS[0]].shape[1]
    # A trained checkpoint's readouts are not the zero rows a fresh model has:
    # an agreement measured against a zero critic would prove nothing, and a
    # zero policy readout would tie every cell's logit and flatten the spread
    # statistic. Both stay small enough to leave tanh unsaturated.
    for readout in ("mlp_p.out", "mlp_q.out"):
        state[f"{readout}.weight"] = torch.randn(1, hidden, generator=generator) * 0.05
        state[f"{readout}.bias"] = torch.randn(1, generator=generator) * 0.05

    optimizer_state = copy.deepcopy(optimizer.state_dict())
    ids = [pid for group in optimizer_state["param_groups"] for pid in group["params"]]
    positions = {name: i for i, (name, _p) in enumerate(model.named_parameters())}
    for key in _READOUT_KEYS:
        entry = optimizer_state["state"][ids[positions[key]]]
        entry["exp_avg"] = entry["exp_avg"][:1].clone()
        entry["exp_avg_sq"] = entry["exp_avg_sq"][:1].clone()

    checkpoint = {
        "model": state,
        "optimizer": optimizer_state,
        "iteration": _ITERATION,
        "rng_state": copy.deepcopy(np.random.default_rng(31).bit_generator.state),
        "versions": _versions(),
    }
    return checkpoint, ids, positions


@dataclass
class Grafted:
    old: dict
    ids: list[int]
    positions: dict[str, int]
    converted: dict
    manifest: dict
    old_path: object
    new_path: object
    manifest_path: object


@pytest.fixture(scope="module")
def grafted(tmp_path_factory) -> Grafted:
    """One conversion, shared by the tests that read its outputs."""
    old, ids, positions = _parent_checkpoint()
    directory = tmp_path_factory.mktemp("graft")
    old_path = directory / "old.pt"
    new_path = directory / "new.pt"
    manifest_path = directory / "graft.json"
    torch.save(old, old_path)
    manifest = graft(
        old_path, new_path, tau=_TAU, lam=_LAM, manifest_path=manifest_path
    )
    return Grafted(
        old=old,
        ids=ids,
        positions=positions,
        converted=torch.load(new_path, map_location="cpu", weights_only=False),
        manifest=manifest,
        old_path=old_path,
        new_path=new_path,
        manifest_path=manifest_path,
    )


def _parent_action_values(model, batch, weight, bias) -> torch.Tensor:
    """The parent's action values, from appendix B transcribed.

    Independent of both the transform and the graft's own measurement: it sums
    the action-value decoder input per legal cell, runs the head MLP, and then
    applies the parent's one-wide readout and ``tanh``.
    """
    with torch.no_grad():
        _s, w, g = model.trunk(batch)
        msg = (w @ model.q.weight.t()).index_select(0, batch.dec_window)
        msg = msg + model.e_qw.weight[batch.dec_class]
        h = torch.zeros(batch.n_cells, w.shape[1])
        h.index_add_(0, batch.dec_cell, msg)
        if batch.bg_cell.numel():
            h.index_copy_(0, batch.bg_cell, model.e_qbg.weight[batch.bg_bucket])
        hidden = F.relu(
            model.mlp_q.lin_a(h)
            + model.mlp_q.lin_b(g).index_select(0, batch.cell_pos)
        )
        return torch.tanh(F.linear(hidden, weight, bias)).squeeze(-1)


def test_grafted_action_values_are_the_parents(grafted, positions):
    """The stated preservation property, on a batch the graft never saw."""
    model = MantisNet(MantisConfig())
    model.load_state_dict(grafted.converted["model"])
    model.eval()
    batch = collate([from_position(p) for p in positions])

    with torch.no_grad():
        _s, w, g = model.trunk(batch)
        _policy, q_new = model.cell_heads(w, g, batch)
    q_parent = _parent_action_values(
        model, batch, *(grafted.old["model"][key] for key in _READOUT_KEYS)
    )

    assert q_parent.abs().max() > 0.1, "a flat parent critic would prove nothing"
    assert float((q_new - q_parent).abs().max()) <= MAX_ABS_DQ


def test_converted_checkpoint_carries_the_parent_forward(grafted):
    old_state = grafted.old["optimizer"]["state"]
    new = grafted.converted["optimizer"]
    names = list(grafted.positions)

    # Only the readout changes shape; every other tensor is the parent's.
    for key, value in grafted.old["model"].items():
        if key in _READOUT_KEYS:
            assert grafted.converted["model"][key].shape[0] == 2 * value.shape[0]
        else:
            torch.testing.assert_close(grafted.converted["model"][key], value)

    # One contiguous group over this build's named_parameters() order.
    assert len(new["param_groups"]) == 1
    assert new["param_groups"][0]["params"] == list(range(len(names)))
    assert sorted(new["state"]) == list(range(len(names)))
    for key, value in grafted.old["optimizer"]["param_groups"][0].items():
        if key != "params":
            assert new["param_groups"][0][key] == value

    for name, position in grafted.positions.items():
        old_entry = old_state[grafted.ids[position]]
        entry = new["state"][position]
        assert entry.keys() == old_entry.keys()
        # Adam's step is not restarted: the moments carried over stay in the
        # bias correction they were accumulated under.
        assert entry["step"] == old_entry["step"]
        if name in _READOUT_KEYS:
            gain = graft_mod.READOUT_GAIN
            torch.testing.assert_close(
                entry["exp_avg"],
                torch.cat([gain * old_entry["exp_avg"], -gain * old_entry["exp_avg"]]),
            )
            # Second moments are scale-free in Adam's update, so both rows keep
            # the parent's: the first post-graft step is then +-gain times the
            # parent's, which preserves the readout's relation to it.
            torch.testing.assert_close(
                entry["exp_avg_sq"],
                torch.cat([old_entry["exp_avg_sq"], old_entry["exp_avg_sq"]]),
            )
        else:
            for field, value in old_entry.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(entry[field], value)
                else:
                    assert entry[field] == value

    assert grafted.converted["iteration"] == _ITERATION
    assert grafted.converted["rng_state"] == grafted.old["rng_state"]
    assert grafted.converted["versions"] == grafted.old["versions"]

    model = MantisNet(MantisConfig())
    optimizer = torch.optim.Adam(model.parameters())
    rng = np.random.default_rng(99)
    assert load_checkpoint(grafted.new_path, model, optimizer, rng) == _ITERATION
    assert rng.bit_generator.state == grafted.old["rng_state"]


def test_manifest_records_the_operating_point_and_the_measurements(grafted):
    manifest = grafted.manifest
    assert json.loads(grafted.manifest_path.read_text(encoding="utf-8")) == manifest
    assert manifest["arm"] == ARM
    assert manifest["source"] == str(grafted.old_path)
    assert manifest["source_iteration"] == _ITERATION
    assert manifest["versions"] == _versions()
    assert manifest["transform"] == TRANSFORM
    assert (manifest["tau"], manifest["lam"]) == (_TAU, _LAM)

    assert manifest["probe_seed"] == graft_mod.PROBE_SEED
    assert manifest["probe_positions"] == PROBE_POSITIONS
    assert len(manifest["probe_plies"]) == PROBE_POSITIONS
    assert (min(manifest["probe_plies"]), max(manifest["probe_plies"])) == PROBE_PLIES
    assert manifest["probe_legal_cells"] > PROBE_POSITIONS * PROBE_TOP_K

    assert 0 <= manifest["mean_abs_dq"] <= manifest["max_abs_dq"] <= MAX_ABS_DQ
    # π′ is measured in float64, so a correct graft's KL lands orders below the
    # tolerance instead of at a fraction of it and only a real difference can
    # trip the gate. An fp32 π′ rounds to ~1e-7 in the mean and ~1e-5 per
    # position on a graft as correct as this one, which is what this bound
    # refuses to accept.
    kl_bound = MAX_MEAN_KL * 1e-4
    assert abs(manifest["mean_improved_kl"]) <= kl_bound
    assert manifest["mean_improved_kl"] <= manifest["max_improved_kl"] <= kl_bound
    # The spread the arms are compared on is a real number, and this graft is
    # the one conversion that must not move it.
    spread = manifest["top_k_q_std_median_parent"]
    assert spread > 0
    assert manifest["top_k_q_std_median_grafted"] == pytest.approx(
        spread, rel=1e-4, abs=MAX_ABS_DQ
    )
    assert manifest["preservation"] == {
        "property": graft_mod.PRESERVATION,
        "max_abs_dq": manifest["max_abs_dq"],
        "max_abs_dq_tolerance": MAX_ABS_DQ,
        "mean_improved_kl": manifest["mean_improved_kl"],
        "mean_improved_kl_tolerance": MAX_MEAN_KL,
    }


def test_probe_set_is_deterministic_and_nonterminal():
    prefixes = _probe_prefixes()
    assert len(prefixes) == PROBE_POSITIONS
    lengths = [len(moves) for moves in prefixes]
    assert lengths == sorted(lengths)
    assert (lengths[0], lengths[-1]) == PROBE_PLIES
    assert [len(m) for m in _probe_prefixes()] == lengths
    assert _probe_prefixes() == prefixes
    for moves in prefixes[::8]:
        assert not hexo_py.Position.replay(moves).is_terminal


def _widen(checkpoint: dict) -> None:
    for key in _READOUT_KEYS:
        row = checkpoint["model"][key]
        checkpoint["model"][key] = torch.cat([row, row], dim=0)


def _drop_adam_entry(checkpoint: dict) -> None:
    ids = checkpoint["optimizer"]["param_groups"][0]["params"]
    names = [name for name, _p in MantisNet(MantisConfig()).named_parameters()]
    del checkpoint["optimizer"]["state"][ids[names.index(_READOUT_KEYS[0])]]


def _split_groups(checkpoint: dict) -> None:
    group = checkpoint["optimizer"]["param_groups"][0]
    tail = dict(group, params=group["params"][1:])
    group["params"] = group["params"][:1]
    checkpoint["optimizer"]["param_groups"].append(tail)


@pytest.mark.parametrize(
    "mutate, message",
    [
        pytest.param(
            lambda c: c["model"].update(
                stone_table_weight=c["model"]["stone_table.weight"]
            ),
            "stone_table_weight",
            id="unexpected-key",
        ),
        pytest.param(
            lambda c: c["model"].__setitem__(
                "stone_table.weight", c["model"]["stone_table.weight"][:1]
            ),
            "stone_table.weight",
            id="extra-shape-mismatch",
        ),
        pytest.param(_widen, "<nothing>", id="already-grafted"),
        pytest.param(
            lambda c: c["model"].__setitem__(
                _READOUT_KEYS[0], c["model"][_READOUT_KEYS[0]].repeat(3, 1)
            ),
            "must be one row",
            id="readout-shape",
        ),
        pytest.param(
            lambda c: c.__setitem__("model", "not a state dict"),
            "must be a state dict",
            id="malformed-model",
        ),
        pytest.param(
            lambda c: c.__setitem__("rng_state", "not a generator state"),
            "bit-generator state",
            id="malformed-rng-state",
        ),
        pytest.param(
            lambda c: c["versions"].__setitem__("torch", "0.0.0"),
            "!= this build",
            id="foreign-versions",
        ),
        pytest.param(
            lambda c: c.__setitem__("extra", 1), "keys must be exactly", id="extra-key"
        ),
        pytest.param(
            lambda c: c.pop("rng_state"), "keys must be exactly", id="missing-key"
        ),
        pytest.param(
            lambda c: c.__setitem__("optimizer", {"param_groups": []}),
            "Adam state dict",
            id="no-adam-state",
        ),
        pytest.param(_drop_adam_entry, "no Adam state for", id="missing-adam-entry"),
        pytest.param(_split_groups, "single Adam parameter group", id="two-groups"),
    ],
)
def test_graft_refuses(tmp_path, mutate: Callable[[dict], None], message: str):
    checkpoint, _ids, _positions = _parent_checkpoint()
    mutate(checkpoint)
    old, new = tmp_path / "old.pt", tmp_path / "new.pt"
    manifest = tmp_path / "graft.json"
    torch.save(checkpoint, old)

    with pytest.raises(ValueError, match=message):
        graft(old, new, tau=_TAU, lam=_LAM, manifest_path=manifest)
    assert not new.exists() and not manifest.exists()


def test_cli_requires_the_operating_point_and_the_manifest(tmp_path):
    """No defaults for ``--tau``, ``--lam``, or ``--manifest``: the recorded
    measurements are meaningless without the point they were taken at."""
    paths = [str(tmp_path / "old.pt"), str(tmp_path / "new.pt")]
    for argv in (
        paths,
        paths + ["--tau", "0.1"],
        paths + ["--tau", "0.1", "--lam", "0.01"],
        paths + ["--lam", "0.01", "--manifest", str(tmp_path / "graft.json")],
    ):
        with pytest.raises(SystemExit):
            main(argv)


def test_graft_refuses_to_overwrite_its_source(tmp_path):
    checkpoint, _ids, _positions = _parent_checkpoint()
    old = tmp_path / "old.pt"
    torch.save(checkpoint, old)
    with pytest.raises(ValueError, match="different paths"):
        graft(old, old, tau=_TAU, lam=_LAM, manifest_path=tmp_path / "graft.json")


def test_graft_refuses_a_transform_that_moves_q(tmp_path, monkeypatch):
    """The preservation check is a detector: a gain of one composes to
    ``tanh(z/2)``, and the graft must not write that."""
    monkeypatch.setattr(graft_mod, "READOUT_GAIN", 1.0)
    checkpoint, _ids, _positions = _parent_checkpoint()
    old, new = tmp_path / "old.pt", tmp_path / "new.pt"
    manifest = tmp_path / "graft.json"
    torch.save(checkpoint, old)

    with pytest.raises(ValueError, match="not function preserving"):
        graft(old, new, tau=_TAU, lam=_LAM, manifest_path=manifest)
    assert not new.exists() and not manifest.exists()


def test_graft_refuses_a_mangled_shared_tensor(tmp_path, monkeypatch):
    """The detector reaches past the two tensors the transform writes.

    A trunk weight the conversion is supposed to carry over verbatim, moved by
    5%, must not reach a written checkpoint — which it would if the parent's
    action values were read back out of the grafted model's own activations
    instead of out of the parent checkpoint.
    """
    victim = "blocks.0.ffn.0.weight"
    convert = graft_mod._converted_state

    def mangle(old_model: dict) -> dict:
        state = convert(old_model)
        state[victim] = state[victim] * 1.05
        return state

    monkeypatch.setattr(graft_mod, "_converted_state", mangle)
    checkpoint, _ids, _positions = _parent_checkpoint()
    old, new = tmp_path / "old.pt", tmp_path / "new.pt"
    manifest = tmp_path / "graft.json"
    torch.save(checkpoint, old)

    with pytest.raises(ValueError, match="not function preserving"):
        graft(old, new, tau=_TAU, lam=_LAM, manifest_path=manifest)
    assert not new.exists() and not manifest.exists()


def test_the_improved_policy_criterion_sees_what_the_action_values_cannot():
    """Each model's π′ comes from its own policy logits, so the KL criterion is
    not a restatement of the action-value one.

    A policy readout moved by 5% leaves every action value exactly where it was.
    Only KL(π′_new ‖ π′_parent) can see it, and only measured in float64: the
    difference it makes is the size of an fp32 π′'s rounding.
    """
    checkpoint, _ids, _positions = _parent_checkpoint()
    old_model = checkpoint["model"]
    state = graft_mod._converted_state(old_model)
    state["mlp_p.out.weight"] = state["mlp_p.out.weight"] * 1.05
    cfg = MantisConfig()
    model = MantisNet(cfg)
    model.load_state_dict(state)
    model.eval()

    measured = graft_mod._measure(
        model, graft_mod._parent_model(old_model, cfg), _TAU, _LAM
    )
    assert measured["max_abs_dq"] <= MAX_ABS_DQ
    assert abs(measured["mean_improved_kl"]) > MAX_MEAN_KL
