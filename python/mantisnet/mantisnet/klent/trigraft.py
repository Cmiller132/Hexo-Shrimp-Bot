"""Graft a v2 scalar-tanh checkpoint into the v2 trinomial critic.

Only ``mlp_q.out`` changes. A scalar row ``(W, b)`` becomes positive
``(W, b)``, negative ``(-W, -b)``, and zero ``(0, -20)``. Therefore

``p_pos - p_neg = tanh(z) / (1 + exp(-20) / (2 cosh(z)))``

and the relative error from the scalar parent's ``tanh(z)`` is at most
``exp(-20) / 2``. The zero row starts with zero Adam first and second moments,
so it inherits neither signed row's optimizer direction or variance; the
scalar entry's step is unchanged. Every other tensor and Adam entry is carried
by name.

Before either artifact is written, a second source read proves the shared
tensors bit-for-bit, strict loaders prove both architectures and the remapped
Adam state, and the fixed 64-position battery bounds max ``|delta Q|`` by
``1e-5`` and mean improved-policy KL by ``1e-6`` at the requested operating
point.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from ..builder import collate_prefixes
from ..model import MantisConfig, MantisNet, compose_q, return_mass
from ..segments import segment_ids, segment_sum
from .graft import _probe_prefixes
from .improve import improved_policy
from .run import _versions

ARM = "trinomial"
READOUT_KEYS = ("mlp_q.out.weight", "mlp_q.out.bias")
MAX_ABS_DQ = 1e-5
MAX_MEAN_KL = 1e-6
ZERO_LOGIT = -20.0
_CHECKPOINT_KEYS = frozenset(
    {"model", "optimizer", "iteration", "rng_state", "versions"}
)
_ADAM_FIELDS = ("step", "exp_avg", "exp_avg_sq")


def _critic_rows(row: Tensor, zero: float = 0.0) -> Tensor:
    """Return the exact ``(+row, -row, constant-zero-row)`` map."""
    return torch.cat([row, -row, torch.full_like(row, zero)], dim=0)


def _check_checkpoint(checkpoint: Any) -> dict[str, Tensor]:
    """Refuse any envelope except the default v2 scalar parent's."""
    if not isinstance(checkpoint, dict) or set(checkpoint) != _CHECKPOINT_KEYS:
        got = sorted(checkpoint) if isinstance(checkpoint, dict) else type(checkpoint).__name__
        raise ValueError(
            f"scalar parent checkpoint must hold exactly {sorted(_CHECKPOINT_KEYS)}, got {got}"
        )
    if checkpoint["versions"] != _versions():
        raise ValueError(
            f"scalar parent versions {checkpoint['versions']} != this v2 build {_versions()}"
        )
    if not isinstance(checkpoint["model"], dict):
        raise ValueError("scalar parent's model entry must be a state dict")
    if not isinstance(checkpoint["iteration"], int):
        raise ValueError("scalar parent's iteration must be an int")
    if not isinstance(checkpoint["rng_state"], dict):
        raise ValueError("scalar parent's rng_state must be a bit-generator state")
    return checkpoint["model"]


def _scalar_parent(state: dict[str, Tensor], cfg: MantisConfig) -> MantisNet:
    """Reconstruct and strict-load the private one-row v2 architecture."""
    parent = MantisNet(cfg)
    parent.mlp_q.out = nn.Linear(parent.mlp_q.out.in_features, 1)
    parent.load_state_dict(state)
    return parent.eval()


def _check_architecture(parent: dict[str, Tensor], child: dict[str, Tensor]) -> None:
    """Require identical v2 models except for a one-row scalar readout."""
    if set(parent) != set(child):
        raise ValueError(
            "scalar parent state-dict keys differ from this build: "
            f"{sorted(set(parent) ^ set(child))}"
        )
    for key in READOUT_KEYS:
        value = parent[key]
        child_value = child[key]
        if not isinstance(value, Tensor):
            raise ValueError(f"scalar parent {key} is {type(value).__name__}, not a tensor")
        if value.ndim and value.shape[0] == 2:
            raise ValueError(
                f"2-row BRM parent at {key} cannot be trigrafted; expected the "
                "one-row v2 scalar-tanh parent"
            )
        want = (1, *child_value.shape[1:])
        if tuple(value.shape) != want:
            raise ValueError(
                f"scalar parent {key} must have shape {want}, got {tuple(value.shape)}"
            )
    wrong = [
        key
        for key in child
        if key not in READOUT_KEYS
        and (
            not isinstance(parent[key], Tensor)
            or tuple(parent[key].shape) != tuple(child[key].shape)
        )
    ]
    if wrong:
        raise ValueError(
            "scalar parent differs outside mlp_q.out at: " + ", ".join(wrong)
        )


def _converted_state(parent: dict[str, Tensor]) -> dict[str, Tensor]:
    """Copy the parent state and replace precisely its two readout tensors."""
    state = {name: tensor.clone() for name, tensor in parent.items()}
    state[READOUT_KEYS[0]] = _critic_rows(parent[READOUT_KEYS[0]])
    state[READOUT_KEYS[1]] = _critic_rows(
        parent[READOUT_KEYS[1]], zero=ZERO_LOGIT
    )
    return state


def _remap_adam(
    saved: Any, parent: dict[str, Tensor], names: list[str]
) -> dict[str, Any]:
    """Carry Adam by name, mapping readout ``m``/``v`` to three rows."""
    if not isinstance(saved, dict) or not isinstance(saved.get("state"), dict):
        raise ValueError("scalar parent must contain an Adam optimizer state dict")
    groups = saved.get("param_groups")
    if (
        not isinstance(groups, list)
        or len(groups) != 1
        or not isinstance(groups[0], dict)
        or not isinstance(groups[0].get("params"), list)
    ):
        raise ValueError("scalar parent optimizer must have exactly one param_group")
    ids = groups[0]["params"]
    if len(ids) != len(names) or len(set(ids)) != len(ids):
        raise ValueError(
            f"scalar parent optimizer has {len(ids)} ids for {len(names)} parameters"
        )

    state: dict[int, dict] = {}
    for index, (name, param_id) in enumerate(zip(names, ids)):
        entry = saved["state"].get(param_id)
        if entry is None:
            if name in READOUT_KEYS:
                raise ValueError(
                    f"scalar parent optimizer has no Adam state for {name}"
                )
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"optimizer state for {name} is not a state dict")
        missing = [field for field in _ADAM_FIELDS if field not in entry]
        if missing:
            raise ValueError(
                f"optimizer state for {name} is missing: {', '.join(missing)}"
            )
        for field in ("exp_avg", "exp_avg_sq"):
            moment = entry[field]
            want = tuple(parent[name].shape)
            if not isinstance(moment, Tensor) or tuple(moment.shape) != want:
                got = tuple(moment.shape) if isinstance(moment, Tensor) else type(moment).__name__
                raise ValueError(
                    f"optimizer {field} for {name} must have shape {want}, got {got}"
                )
        mapped = copy.deepcopy(entry)
        if name in READOUT_KEYS:
            extra = set(mapped) - set(_ADAM_FIELDS)
            if extra:
                raise ValueError(
                    f"optimizer state for {name} carries unmappable fields {sorted(extra)}"
                )
            mapped["exp_avg"] = _critic_rows(mapped["exp_avg"])
            mapped["exp_avg_sq"] = torch.cat(
                [
                    mapped["exp_avg_sq"],
                    mapped["exp_avg_sq"],
                    torch.zeros_like(mapped["exp_avg_sq"]),
                ],
                dim=0,
            )
        state[index] = mapped
    group = {**copy.deepcopy(groups[0]), "params": list(range(len(names)))}
    return {"state": state, "param_groups": [group]}


def _shared_digest(state: dict[str, Tensor], names: list[str]) -> str:
    """SHA-256 over ordered tensor names, metadata, and exact bytes."""
    digest = hashlib.sha256()
    for name in names:
        tensor = state[name].detach().cpu().contiguous()
        digest.update(f"{name} {tensor.dtype} {tuple(tensor.shape)}".encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def _measure(
    parent: MantisNet,
    model: MantisNet,
    tau: float,
    lam: float,
    mass_floor: float,
) -> dict[str, float | int]:
    """Measure Q and improved-policy preservation on the fixed probe."""
    prefixes = _probe_prefixes()
    batch = collate_prefixes(prefixes, [len(moves) for moves in prefixes])

    _s, parent_w, parent_g = parent.trunk(batch)
    policy_parent, scalar = parent.cell_head_logits(parent_w, parent_g, batch)
    q_parent = torch.tanh(scalar.squeeze(-1).float())

    _s, w, g = model.trunk(batch)
    policy_new, score_new, q_new = model.cell_heads(w, g, batch, mass_floor)
    offsets = batch.legal_offsets
    pi_parent = improved_policy(
        policy_parent.double(), q_parent.double(), q_parent.double(), offsets,
        tau, lam,
    )
    pi_new = improved_policy(
        policy_new.double(), score_new.double(), q_new.double(), offsets, tau, lam
    )
    probs = pi_new.probs
    terms = torch.where(
        probs > 0,
        probs * (probs.log() - pi_parent.probs.log()),
        probs.new_zeros(()),
    )
    kl = segment_sum(terms, segment_ids(offsets), batch.n_pos)
    delta = (q_new - q_parent).abs()
    p_pos, p_neg = return_mass(
        model.cell_head_logits(w, g, batch)[1]
    )
    return {
        "probe_positions": batch.n_pos,
        "probe_legal_cells": int(offsets[-1]),
        "max_abs_dq": float(delta.max()),
        "mean_abs_dq": float(delta.mean()),
        "mean_improved_kl": float(kl.mean()),
        "max_improved_kl": float(kl.max()),
        "committed_mass_min": float((p_pos + p_neg).min()),
        "committed_mass_max": float((p_pos + p_neg).max()),
    }


def trigraft(
    old_path: Path | str,
    new_path: Path | str,
    tau: float,
    lam: float,
    mass_floor: float,
    manifest_path: Path | str | None = None,
) -> dict[str, Any]:
    """Convert, prove both measured bounds, then write manifest and checkpoint."""
    if not math.isfinite(mass_floor) or not 0.0 < mass_floor <= 1.0:
        raise ValueError(
            f"--mass-floor must be finite and in (0, 1], got {mass_floor}"
        )
    old_path, new_path = Path(old_path), Path(new_path)
    manifest_path = (
        Path(manifest_path)
        if manifest_path is not None
        else new_path.with_suffix(".manifest.json")
    )
    resolved = [old_path.resolve(), new_path.resolve(), manifest_path.resolve()]
    if len(set(resolved)) != len(resolved):
        raise ValueError("--old, --new, and --manifest must be different paths")

    checkpoint = torch.load(old_path, map_location="cpu", weights_only=False)
    parent_state = _check_checkpoint(checkpoint)
    cfg = MantisConfig()
    model = MantisNet(cfg)
    _check_architecture(parent_state, model.state_dict())
    names = [name for name, _parameter in model.named_parameters()]
    if list(parent_state) != names:
        raise ValueError(
            "scalar parent state dict is not in this build's parameter order, "
            "so optimizer ids cannot be named"
        )

    state = _converted_state(parent_state)
    optimizer = _remap_adam(checkpoint["optimizer"], parent_state, names)
    converted = {
        "model": {name: state[name] for name in names},
        "model_config": dataclasses.asdict(model.cfg),
        "optimizer": optimizer,
        "iteration": checkpoint["iteration"],
        "rng_state": copy.deepcopy(checkpoint["rng_state"]),
        "versions": _versions(),
    }
    model.load_state_dict(converted["model"])
    torch.optim.Adam(model.parameters()).load_state_dict(optimizer)
    model.eval()
    parent = _scalar_parent(parent_state, cfg)

    second = torch.load(old_path, map_location="cpu", weights_only=False)["model"]
    shared = [name for name in names if name not in READOUT_KEYS]
    changed = [
        name
        for name in shared
        if not torch.equal(converted["model"][name], second[name])
        or not torch.equal(model.state_dict()[name], second[name])
    ]
    if changed:
        raise ValueError(
            "trigraft changed shared source tensors: " + ", ".join(changed)
            + "; nothing written"
        )

    expected_readout = {
        READOUT_KEYS[0]: _critic_rows(second[READOUT_KEYS[0]]),
        READOUT_KEYS[1]: _critic_rows(
            second[READOUT_KEYS[1]], zero=ZERO_LOGIT
        ),
    }
    wrong_readout = [
        name
        for name, expected in expected_readout.items()
        if not torch.equal(converted["model"][name], expected)
        or not torch.equal(model.state_dict()[name], expected)
    ]
    if wrong_readout:
        raise ValueError(
            "trigraft readout rows do not match the signed/zero map at: "
            + ", ".join(wrong_readout)
            + "; nothing written"
        )

    measurement = _measure(parent, model, tau, lam, mass_floor)
    holds = (
        measurement["max_abs_dq"] <= MAX_ABS_DQ
        and abs(measurement["mean_improved_kl"]) <= MAX_MEAN_KL
    )
    manifest = {
        "arm": ARM,
        "source": str(old_path),
        "source_iteration": checkpoint["iteration"],
        "versions": converted["versions"],
        "transform": (
            "mlp_q.out scalar row to (+row, -row, zero), with zero bias -20; "
            "all shared tensors copied bit-for-bit"
        ),
        "tau": tau,
        "lam": lam,
        "mass_floor": mass_floor,
        **measurement,
        "preservation": {
            "shared_tensors_unchanged": len(shared),
            "shared_tensor_sha256": _shared_digest(second, shared),
            "max_abs_dq_tolerance": MAX_ABS_DQ,
            "mean_improved_kl_tolerance": MAX_MEAN_KL,
            "holds": holds,
        },
    }
    if not holds:
        raise ValueError(
            "trigraft preservation failed: max |Q_new - Q_parent| = "
            f"{measurement['max_abs_dq']:.3e} (bound {MAX_ABS_DQ:.0e}), mean "
            "KL(pi'_new || pi'_parent) = "
            f"{measurement['mean_improved_kl']:.3e} (bound {MAX_MEAN_KL:.0e}); "
            "nothing written"
        )

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    torch.save(converted, new_path)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--tau", type=float, required=True)
    parser.add_argument("--lam", type=float, required=True)
    parser.add_argument("--mass-floor", type=float, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args(argv)
    manifest = trigraft(
        args.old, args.new, args.tau, args.lam, args.mass_floor, args.manifest
    )
    print(json.dumps(manifest["preservation"], indent=2), flush=True)


if __name__ == "__main__":
    main()
