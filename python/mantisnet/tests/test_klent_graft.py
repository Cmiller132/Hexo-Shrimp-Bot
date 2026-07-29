"""Explicit scalar-Q checkpoint conversion to the factored critic."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from mantisnet import MantisConfig, MantisNet
from mantisnet.klent.graft import graft
from mantisnet.klent.run import _versions, load_checkpoint


_READOUT_KEYS = ("mlp_q.out.weight", "mlp_q.out.bias")


def _old_scalar_checkpoint():
    torch.manual_seed(30)
    model = MantisNet(MantisConfig())
    optimizer = torch.optim.Adam(model.parameters())
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    model_state = copy.deepcopy(model.state_dict())
    model_state[_READOUT_KEYS[0]] = model_state[_READOUT_KEYS[0]][:1].clone()
    model_state[_READOUT_KEYS[1]] = model_state[_READOUT_KEYS[1]][:1].clone()

    optimizer_state = copy.deepcopy(optimizer.state_dict())
    param_ids = [
        param_id
        for group in optimizer_state["param_groups"]
        for param_id in group["params"]
    ]
    positions = {name: i for i, (name, _parameter) in enumerate(model.named_parameters())}
    for key in _READOUT_KEYS:
        entry = optimizer_state["state"][param_ids[positions[key]]]
        entry["exp_avg"] = entry["exp_avg"][:1].clone()
        entry["exp_avg_sq"] = entry["exp_avg_sq"][:1].clone()

    rng = np.random.default_rng(31)
    checkpoint = {
        "model": model_state,
        "optimizer": optimizer_state,
        "iteration": 7,
        "rng_state": copy.deepcopy(rng.bit_generator.state),
        "versions": _versions(),
    }
    return checkpoint, param_ids, positions


def test_graft_round_trip_loads_and_preserves_shared_state(tmp_path):
    old, param_ids, positions = _old_scalar_checkpoint()
    old_path, new_path = tmp_path / "old.pt", tmp_path / "new.pt"
    torch.save(old, old_path)

    graft(old_path, new_path)
    converted = torch.load(new_path, map_location="cpu", weights_only=False)

    assert converted["iteration"] == old["iteration"]
    assert converted["rng_state"] == old["rng_state"]
    assert converted["versions"] == old["versions"]
    assert converted["optimizer"]["param_groups"] == old["optimizer"]["param_groups"]
    for key, value in old["model"].items():
        if key not in _READOUT_KEYS:
            torch.testing.assert_close(converted["model"][key], value)
    for key in _READOUT_KEYS:
        assert tuple(converted["model"][key].shape) == (
            2,
            *old["model"][key].shape[1:],
        )
        assert torch.count_nonzero(converted["model"][key]) == 0
        entry = converted["optimizer"]["state"][param_ids[positions[key]]]
        assert torch.count_nonzero(entry["exp_avg"]) == 0
        assert torch.count_nonzero(entry["exp_avg_sq"]) == 0
        assert entry["step"].item() == 0

    changed_ids = {param_ids[positions[key]] for key in _READOUT_KEYS}
    for param_id, old_entry in old["optimizer"]["state"].items():
        if param_id in changed_ids:
            continue
        converted_entry = converted["optimizer"]["state"][param_id]
        assert converted_entry.keys() == old_entry.keys()
        for field, value in old_entry.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(converted_entry[field], value)
            else:
                assert converted_entry[field] == value

    fresh_model = MantisNet(MantisConfig())
    fresh_optimizer = torch.optim.Adam(fresh_model.parameters())
    fresh_rng = np.random.default_rng(99)
    assert load_checkpoint(new_path, fresh_model, fresh_optimizer, fresh_rng) == 7
    for key, value in converted["model"].items():
        torch.testing.assert_close(fresh_model.state_dict()[key], value)
    assert fresh_rng.bit_generator.state == old["rng_state"]


def test_graft_refuses_any_additional_model_mismatch(tmp_path):
    old, _param_ids, _positions = _old_scalar_checkpoint()
    old["model"]["stone_table.weight"] = old["model"]["stone_table.weight"][:-1]
    old_path, new_path = tmp_path / "bad-old.pt", tmp_path / "new.pt"
    torch.save(old, old_path)

    with pytest.raises(ValueError, match="stone_table.weight"):
        graft(old_path, new_path)
    assert not new_path.exists()
