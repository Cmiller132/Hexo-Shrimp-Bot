"""Deepen a checkpoint's trunk without changing what it computes.

Every trunk sub-block is residual and ends in a linear (``MODEL_SPEC.md`` §5):
windows take ``mlp_w.out``, stones take ``mlp_s.out``, attention takes ``wo``,
and the FFN takes its second layer. A block whose four output projections are
zero therefore adds exactly ``0.0`` to ``(s, w, g)``, so the deepened model's
policy logits, action values, and acting score are *bitwise* the parent's —
this graft has no tolerance, and the gate below asserts equality rather than a
bound.

The zeros sit on each block's output, not its input, so every inserted
parameter still receives gradient on the first step: only the block's
contribution to the residual stream starts at nothing. That is the standard
zero-initialized-residual construction, and it is why an inserted block is a
starting point rather than a dead one.

Inserted blocks are spread evenly through the stack and never land last. Both
cell decoders and the value head read the final block's output, and they were
fitted against a trained one; keeping the deepest block the parent's means the
heads see a familiar representation until the new blocks have learned something.

This is a one-off command-line conversion, not part of the training surface.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import torch

from ..builder import collate_prefixes
from ..model import MantisConfig, MantisNet
from .graft import _probe_prefixes
from .improve import improved_policy
from .run import _versions

# The four residual-branch outputs of a trunk block. Zeroing weight and bias of
# all four is what makes an inserted block the identity.
BLOCK_OUTPUTS = ("mlp_w.out", "mlp_s.out", "wo", "ffn.2")

PRESERVATION = (
    "the deepened model's policy logits, action values, acting score, and "
    "improved policy are bitwise the parent's, because every inserted block "
    "ends each of its four residual branches in a zeroed projection and so "
    "adds exactly 0.0 to the residual stream"
)


def placement(parent_blocks: int, blocks: int) -> list[int | None]:
    """Where each block of the deepened stack comes from; ``None`` is new.

    Insertions go *before* a parent block, so the deepest block is always the
    parent's last. Positions are spread evenly; more than one may land in the
    same gap when the depth ratio demands it.
    """
    if blocks <= parent_blocks:
        raise ValueError(
            f"deepening needs more blocks than the parent's {parent_blocks}, "
            f"got {blocks}"
        )
    added = blocks - parent_blocks
    inserts = [0] * parent_blocks
    for k in range(1, added + 1):
        inserts[round(k * parent_blocks / (added + 1)) % parent_blocks] += 1
    out: list[int | None] = []
    for index in range(parent_blocks):
        out.extend([None] * inserts[index])
        out.append(index)
    if len(out) != blocks:
        raise AssertionError(f"placement produced {len(out)} blocks, not {blocks}")
    return out


def _split(names) -> tuple[set[str], set[str]]:
    """State-dict keys as (non-block keys, per-block suffixes)."""
    plain, suffix = set(), set()
    for name in names:
        if name.startswith("blocks."):
            suffix.add(name.split(".", 2)[2])
        else:
            plain.add(name)
    return plain, suffix


def deepen_state(parent_state: dict, sites: list[int | None], cfg: MantisConfig) -> dict:
    """The deepened state dict: parent blocks copied, inserted blocks zeroed."""
    state = MantisNet(cfg).state_dict()
    parent_plain, parent_suffix = _split(parent_state)
    plain, suffix = _split(state)
    if parent_plain != plain:
        raise ValueError(
            "parent and deepened models disagree outside the trunk: "
            f"{sorted(parent_plain ^ plain)}"
        )
    if parent_suffix != suffix:
        raise ValueError(
            f"parent and deepened blocks disagree: {sorted(parent_suffix ^ suffix)}"
        )

    for name in plain:
        state[name] = parent_state[name].clone()
    zeroed = 0
    for index, source in enumerate(sites):
        if source is None:
            for proj in BLOCK_OUTPUTS:
                for field in ("weight", "bias"):
                    state[f"blocks.{index}.{proj}.{field}"].zero_()
                    zeroed += 1
            continue
        for tail in suffix:
            state[f"blocks.{index}.{tail}"] = parent_state[
                f"blocks.{source}.{tail}"
            ].clone()
    expected = 2 * len(BLOCK_OUTPUTS) * sum(1 for s in sites if s is None)
    if zeroed != expected:
        raise AssertionError(f"zeroed {zeroed} tensors, expected {expected}")
    return state


def carry_adam(
    parent_opt: dict, parent_names: list[str], names: list[str], sites: list[int | None]
) -> dict:
    """The parent's Adam moments, re-keyed onto the deepened parameter order.

    Inserted parameters carry no moment: they are unstepped, exactly as a
    fresh parameter would be, so Adam's bias correction treats them as new
    rather than inheriting a stranger's second moment.
    """
    groups = parent_opt.get("param_groups")
    if not isinstance(groups, list) or len(groups) != 1:
        raise ValueError(
            "the parent must have exactly one Adam param_group; the trainer "
            "builds Adam over model.parameters() in one group"
        )
    if len(groups[0]["params"]) != len(parent_names):
        raise ValueError(
            f"the parent optimizer holds {len(groups[0]['params'])} parameters "
            f"but its model has {len(parent_names)}"
        )

    source = {}
    for new_index, name in enumerate(names):
        if not name.startswith("blocks."):
            source[new_index] = name
            continue
        block, tail = name.split(".", 2)[1:]
        parent_block = sites[int(block)]
        if parent_block is not None:
            source[new_index] = f"blocks.{parent_block}.{tail}"

    parent_index = {name: i for i, name in enumerate(parent_names)}
    state = {}
    for new_index, parent_name in source.items():
        entry = parent_opt["state"].get(parent_index[parent_name])
        if entry is not None:
            state[new_index] = entry
    if not state:
        raise ValueError("the parent's optimizer stepped no parameter at all")
    return {
        "state": state,
        "param_groups": [{**groups[0], "params": list(range(len(names)))}],
    }


def _digest(state: dict) -> str:
    out = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        out.update(f"{name} {tensor.dtype} {tuple(tensor.shape)}".encode())
        out.update(tensor.flatten().to(torch.float64).numpy().tobytes())
    return out.hexdigest()


@torch.no_grad()
def _measure(parent: MantisNet, model: MantisNet, tau, lam, mass_floor) -> dict:
    """Both models over the shared probe, as exact agreement or not."""
    prefixes = _probe_prefixes()
    batch = collate_prefixes(prefixes, [len(moves) for moves in prefixes])
    out = []
    for net in (parent.eval(), model.eval()):
        _s, w, g = net.trunk(batch)
        policy, score, q = net.cell_heads(w, g, batch, mass_floor)
        improved = improved_policy(
            policy, score, q, batch.legal_offsets, tau, lam
        )
        out.append((policy, score, q, improved))
    (p0, s0, q0, i0), (p1, s1, q1, i1) = out
    return {
        "probe_positions": len(prefixes),
        "probe_cells": int(batch.legal_offsets[-1]),
        "policy_bitwise_equal": bool(torch.equal(p0, p1)),
        "q_bitwise_equal": bool(torch.equal(q0, q1)),
        "acting_score_bitwise_equal": bool(torch.equal(s0, s1)),
        "improved_bitwise_equal": bool(torch.equal(i0.probs, i1.probs)),
        "v_hat_bitwise_equal": bool(torch.equal(i0.v_hat, i1.v_hat)),
        "q_max_abs_delta": float((q0 - q1).abs().max()),
        "policy_max_abs_delta": float((p0 - p1).abs().max()),
    }


def deepen(
    old_path: Path | str,
    new_path: Path | str,
    blocks: int,
    tau: float,
    lam: float,
    mass_floor: float,
    manifest_path: Path | str | None = None,
) -> dict:
    """Write the deepened checkpoint and its manifest; return the manifest."""
    old_path, new_path = Path(old_path), Path(new_path)
    manifest_path = (
        Path(manifest_path)
        if manifest_path is not None
        else new_path.with_suffix(".manifest.json")
    )

    checkpoint = torch.load(old_path, map_location="cpu", weights_only=False)
    if checkpoint["versions"] != _versions():
        raise ValueError(
            f"checkpoint versions {checkpoint['versions']} != this build "
            f"{_versions()}"
        )
    parent_cfg = MantisConfig(**checkpoint["model_config"]) if (
        "model_config" in checkpoint
    ) else MantisConfig()
    cfg = dataclasses.replace(parent_cfg, blocks=blocks)
    sites = placement(parent_cfg.blocks, blocks)

    parent = MantisNet(parent_cfg)
    parent.load_state_dict(checkpoint["model"])
    parent_names = [name for name, _ in parent.named_parameters()]

    state = deepen_state(checkpoint["model"], sites, cfg)
    model = MantisNet(cfg)
    model.load_state_dict(state)
    names = [name for name, _ in model.named_parameters()]

    converted = {
        "model": state,
        "model_config": dataclasses.asdict(cfg),
        "optimizer": carry_adam(checkpoint["optimizer"], parent_names, names, sites),
        "iteration": checkpoint["iteration"],
        "rng_state": checkpoint["rng_state"],
        "versions": _versions(),
    }
    # Rejected by Adam's own loader before anything is written, so a checkpoint
    # this build cannot resume never reaches the disk.
    torch.optim.Adam(model.parameters()).load_state_dict(converted["optimizer"])

    manifest = {
        "parent": str(old_path.resolve()),
        "new": str(new_path.resolve()),
        "parent_digest": _digest(checkpoint["model"]),
        "parent_iteration": checkpoint["iteration"],
        "parent_blocks": parent_cfg.blocks,
        "blocks": blocks,
        "placement": sites,
        "inserted": [i for i, s in enumerate(sites) if s is None],
        "parent_parameters": sum(p.numel() for p in parent.parameters()),
        "parameters": sum(p.numel() for p in model.parameters()),
        "carried_moments": len(converted["optimizer"]["state"]),
        "unstepped_parameters": len(names)
        - len(converted["optimizer"]["state"]),
        "tau": tau,
        "lam": lam,
        "mass_floor": mass_floor,
        **_measure(parent, model, tau, lam, mass_floor),
    }
    preserved = all(
        manifest[key]
        for key in (
            "policy_bitwise_equal",
            "q_bitwise_equal",
            "acting_score_bitwise_equal",
            "improved_bitwise_equal",
            "v_hat_bitwise_equal",
        )
    )
    manifest["preservation"] = {"claim": PRESERVATION, "holds": preserved}
    if not preserved:
        raise ValueError(
            "the deepened model does not reproduce its parent bitwise: "
            + json.dumps({k: manifest[k] for k in manifest if k.endswith("equal")})
            + f"; max |dQ| {manifest['q_max_abs_delta']}, max |dpolicy| "
            + f"{manifest['policy_max_abs_delta']}"
        )

    torch.save(converted, new_path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old", type=Path, required=True, help="the parent checkpoint")
    ap.add_argument("--new", type=Path, required=True, help="the deepened checkpoint")
    ap.add_argument("--blocks", type=int, required=True, help="B of the deepened trunk")
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--lam", type=float, default=0.01)
    ap.add_argument("--mass-floor", type=float, default=0.2)
    ap.add_argument("--manifest", type=Path, default=None)
    args = ap.parse_args()
    manifest = deepen(
        args.old, args.new, args.blocks, args.tau, args.lam, args.mass_floor,
        args.manifest,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
