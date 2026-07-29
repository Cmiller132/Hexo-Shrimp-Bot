"""Regenerate the factored critic readout on a trained scalar-Q checkpoint.

This is the deliberate one-time path for preserving a trained trunk while
discarding the obsolete scalar critic output. The two new output rows and
their Adam moments start at zero, preserving MantisNet's zero-init contract.
Normal checkpoint loaders remain strict: conversion must be explicit because
the checkpoint formats are not compatible.

Run from ``python/mantisnet``:

    python -m mantisnet.klent.graft OLD.pt NEW.pt
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import torch

from ..model import MantisConfig, MantisNet


_READOUT_KEYS = ("mlp_q.out.weight", "mlp_q.out.bias")


def _architecture_mismatches(old: dict[str, Any], current: dict[str, Any]) -> list[str]:
    mismatches = sorted(set(old) ^ set(current))
    for key in sorted(set(old) & set(current)):
        old_value, current_value = old[key], current[key]
        if not isinstance(old_value, torch.Tensor) or not isinstance(
            current_value, torch.Tensor
        ):
            if type(old_value) is not type(current_value):
                mismatches.append(key)
        elif old_value.shape != current_value.shape:
            mismatches.append(key)
    return mismatches


def _reset_step(step: Any) -> Any:
    if isinstance(step, torch.Tensor):
        return torch.zeros_like(step)
    return 0


def graft(old_path: Path, new_path: Path) -> None:
    """Convert ``old_path`` and write a strict current-format checkpoint."""
    old_path = Path(old_path)
    new_path = Path(new_path)
    if old_path.resolve() == new_path.resolve():
        raise ValueError("OLD.pt and NEW.pt must be different paths")

    checkpoint = torch.load(old_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("model"), dict
    ):
        raise ValueError("checkpoint must contain a model state dict")

    model = MantisNet(MantisConfig())
    current = model.state_dict()
    old_model = checkpoint["model"]
    mismatches = _architecture_mismatches(old_model, current)
    if set(mismatches) != set(_READOUT_KEYS):
        names = ", ".join(mismatches) if mismatches else "<none>"
        raise ValueError(
            "old model must differ from the current architecture only at "
            f"{', '.join(_READOUT_KEYS)}; mismatched keys: {names}"
        )

    weight, bias = (old_model[key] for key in _READOUT_KEYS)
    expected_old = {
        _READOUT_KEYS[0]: (1, current[_READOUT_KEYS[0]].shape[1]),
        _READOUT_KEYS[1]: (1,),
    }
    wrong_readouts = [
        key
        for key, value in zip(_READOUT_KEYS, (weight, bias))
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_old[key]
    ]
    if wrong_readouts:
        raise ValueError(
            "scalar critic readout has unexpected shape for keys: "
            + ", ".join(wrong_readouts)
        )

    converted = copy.deepcopy(checkpoint)
    for key in _READOUT_KEYS:
        converted["model"][key] = torch.zeros_like(current[key])

    optimizer = converted.get("optimizer")
    if not isinstance(optimizer, dict) or not isinstance(optimizer.get("state"), dict):
        raise ValueError("checkpoint must contain an Adam optimizer state dict")
    groups = optimizer.get("param_groups")
    if not isinstance(groups, list) or any(
        not isinstance(group, dict) or not isinstance(group.get("params"), list)
        for group in groups
    ):
        raise ValueError("optimizer param_groups are malformed")

    saved_param_ids = [param_id for group in groups for param_id in group["params"]]
    named_params = list(model.named_parameters())
    if len(saved_param_ids) != len(named_params):
        raise ValueError(
            "optimizer parameter count does not match model.named_parameters(): "
            f"{len(saved_param_ids)} != {len(named_params)}"
        )
    name_to_position = {name: i for i, (name, _param) in enumerate(named_params)}

    for key in _READOUT_KEYS:
        param_id = saved_param_ids[name_to_position[key]]
        entry = optimizer["state"].get(param_id)
        if not isinstance(entry, dict):
            raise ValueError(f"optimizer has no Adam state for {key}")
        missing = [
            field for field in ("step", "exp_avg", "exp_avg_sq") if field not in entry
        ]
        if missing:
            raise ValueError(f"optimizer state for {key} is missing: {', '.join(missing)}")
        for field in ("exp_avg", "exp_avg_sq"):
            moment = entry[field]
            if not isinstance(moment, torch.Tensor) or tuple(moment.shape) != expected_old[
                key
            ]:
                raise ValueError(f"optimizer {field} has unexpected shape for {key}")
            entry[field] = moment.new_zeros(current[key].shape)
        entry["step"] = _reset_step(entry["step"])

    torch.save(converted, new_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="graft a zero-initialized factored critic onto a scalar-Q checkpoint"
    )
    parser.add_argument("old", type=Path, metavar="OLD.pt")
    parser.add_argument("new", type=Path, metavar="NEW.pt")
    args = parser.parse_args(argv)
    graft(args.old, args.new)


if __name__ == "__main__":
    main()
