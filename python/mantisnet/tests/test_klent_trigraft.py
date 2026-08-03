"""The v2 scalar-tanh to v2 trinomial checkpoint graft."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch

from mantisnet import MantisConfig, MantisNet
from mantisnet.klent import trigraft as trigraft_module
from mantisnet.klent.run import _versions, load_checkpoint
from mantisnet.klent.trigraft import (
    MAX_ABS_DQ,
    MAX_MEAN_KL,
    READOUT_KEYS,
    trigraft,
)

TAU, LAM, MASS_FLOOR = 0.1, 0.01, 0.2


def _scalar_checkpoint() -> dict:
    """A synthetic joint-decoder scalar parent with dense Adam state."""
    torch.manual_seed(41)
    model = MantisNet(MantisConfig())
    with torch.no_grad():
        model.mlp_q.out.weight.normal_(std=0.05)
        model.mlp_q.out.bias.normal_(std=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for index, parameter in enumerate(model.parameters()):
        parameter.grad = torch.full_like(parameter, 0.01 * (index % 3 + 1))
    optimizer.step()

    names = [name for name, _parameter in model.named_parameters()]
    state = copy.deepcopy(optimizer.state_dict())
    model_state = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }
    for key in READOUT_KEYS:
        model_state[key] = model_state[key][:1].clone()
        index = names.index(key)
        for field in ("exp_avg", "exp_avg_sq"):
            state["state"][index][field] = state["state"][index][field][:1].clone()

    return {
        "model": model_state,
        "optimizer": state,
        "iteration": 300,
        "rng_state": copy.deepcopy(np.random.default_rng(41).bit_generator.state),
        "versions": _versions(),
    }


def _paths(tmp_path):
    return (
        tmp_path / "scalar.pt",
        tmp_path / "trinomial.pt",
        tmp_path / "evidence.json",
    )


def _write(tmp_path, checkpoint=None):
    old, new, manifest = _paths(tmp_path)
    torch.save(checkpoint if checkpoint is not None else _scalar_checkpoint(), old)
    return old, new, manifest


def test_identity_battery_and_manifest_on_a_synthetic_scalar_parent(tmp_path):
    old, new, evidence = _write(tmp_path)
    manifest = trigraft(old, new, TAU, LAM, MASS_FLOOR, evidence)

    assert manifest["preservation"]["holds"]
    assert manifest["max_abs_dq"] <= MAX_ABS_DQ
    assert abs(manifest["mean_improved_kl"]) <= MAX_MEAN_KL
    assert manifest["probe_positions"] == 64
    assert json.loads(evidence.read_text(encoding="utf-8")) == manifest
    assert new.exists()

    model = MantisNet(MantisConfig())
    optimizer = torch.optim.Adam(model.parameters())
    assert load_checkpoint(new, model, optimizer) == 300


def test_readout_and_adam_rows_follow_the_signed_zero_map(tmp_path):
    old, new, evidence = _write(tmp_path)
    parent = torch.load(old, map_location="cpu", weights_only=False)
    trigraft(old, new, TAU, LAM, MASS_FLOOR, evidence)
    child = torch.load(new, map_location="cpu", weights_only=False)
    names = list(parent["model"])

    assert torch.equal(
        child["model"][READOUT_KEYS[0]],
        torch.cat(
            [parent["model"][READOUT_KEYS[0]], -parent["model"][READOUT_KEYS[0]],
             torch.zeros_like(parent["model"][READOUT_KEYS[0]])],
            dim=0,
        ),
    )
    assert torch.equal(
        child["model"][READOUT_KEYS[1]],
        torch.cat(
            [parent["model"][READOUT_KEYS[1]], -parent["model"][READOUT_KEYS[1]],
             torch.full_like(parent["model"][READOUT_KEYS[1]], -20.0)],
            dim=0,
        ),
    )
    for key in READOUT_KEYS:
        index = names.index(key)
        before = parent["optimizer"]["state"][index]
        after = child["optimizer"]["state"][index]
        assert torch.equal(
            after["exp_avg"],
            torch.cat(
                [before["exp_avg"], -before["exp_avg"],
                 torch.zeros_like(before["exp_avg"])],
                dim=0,
            ),
        )
        assert torch.equal(
            after["exp_avg_sq"],
            torch.cat(
                [before["exp_avg_sq"], before["exp_avg_sq"],
                 torch.zeros_like(before["exp_avg_sq"])],
                dim=0,
            ),
        )
        assert after["step"] == before["step"]


def test_two_row_brm_parent_is_refused_by_name_without_outputs(tmp_path):
    checkpoint = _scalar_checkpoint()
    names = list(checkpoint["model"])
    for key in READOUT_KEYS:
        checkpoint["model"][key] = checkpoint["model"][key].repeat(
            (2,) + (1,) * (checkpoint["model"][key].ndim - 1)
        )
        index = names.index(key)
        for field in ("exp_avg", "exp_avg_sq"):
            checkpoint["optimizer"]["state"][index][field] = checkpoint[
                "optimizer"
            ]["state"][index][field].repeat(
                (2,) + (1,) * (checkpoint["model"][key].ndim - 1)
            )
    old, new, evidence = _write(tmp_path, checkpoint)
    with pytest.raises(ValueError, match="2-row BRM parent"):
        trigraft(old, new, TAU, LAM, MASS_FLOOR, evidence)
    assert not new.exists() and not evidence.exists()


def test_wrong_scalar_shape_is_refused_without_outputs(tmp_path):
    checkpoint = _scalar_checkpoint()
    checkpoint["model"][READOUT_KEYS[0]] = torch.zeros(1, 7)
    old, new, evidence = _write(tmp_path, checkpoint)
    with pytest.raises(ValueError, match="mlp_q.out.weight must have shape"):
        trigraft(old, new, TAU, LAM, MASS_FLOOR, evidence)
    assert not new.exists() and not evidence.exists()


def test_missing_readout_adam_state_is_refused_without_outputs(tmp_path):
    checkpoint = _scalar_checkpoint()
    index = list(checkpoint["model"]).index(READOUT_KEYS[0])
    del checkpoint["optimizer"]["state"][index]
    old, new, evidence = _write(tmp_path, checkpoint)
    with pytest.raises(ValueError, match="no Adam state for mlp_q.out.weight"):
        trigraft(old, new, TAU, LAM, MASS_FLOOR, evidence)
    assert not new.exists() and not evidence.exists()


def test_failed_measurement_bound_writes_nothing(tmp_path, monkeypatch):
    old, new, evidence = _write(tmp_path)
    monkeypatch.setattr(trigraft_module, "MAX_ABS_DQ", -1.0)
    with pytest.raises(ValueError, match="preservation failed"):
        trigraft(old, new, TAU, LAM, MASS_FLOOR, evidence)
    assert not new.exists() and not evidence.exists()
