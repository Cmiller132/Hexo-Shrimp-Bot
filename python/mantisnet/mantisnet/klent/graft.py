"""Graft the appendix-B critic tail onto a checkpoint trained without it.

The tail adds keys and rewrites nothing: ``q_tail_ln`` and ``q_tail`` arrive
with the zero output linear of MODEL_SPEC appendix B, so ``rows_q == rows`` and
the grafted model's action values are the parent's. That is the conversion's
stated preservation property, and this module enforces both of its halves —
every parent tensor is compared bit for bit against the source file, and the
grafted network is measured against the parent's own readout on a fixed probe
set. A conversion that fails either writes neither checkpoint nor manifest.

Neither half subsumes the other. The probe reads the grafted trunk on both
sides of its comparison, so a rewritten trunk or embedding tensor cancels in
it, and the state-value head is not on its path at all; the bitwise comparison
covers every parent tensor and says nothing about what the heads do with the
rows they read.

Normal loaders stay strict: the formats are not compatible, so the conversion
is explicit or it does not happen.

Run from ``python/mantisnet``:

    python -m mantisnet.klent.graft OLD.pt NEW.pt --tau T --lam L --manifest OUT.json
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import statistics
from pathlib import Path
from typing import Any

import hexo_py
import torch
import torch.nn.functional as F
from torch import Tensor

from .. import decoder
from ..builder import Batch, collate, from_position
from ..model import MantisConfig, MantisNet
from ..segments import segment_ids, segment_sum
from .improve import improved_policy
from .run import _versions

ARM = "critic-tail"

# What the transform owns: the critic tail's parameters, and nothing else.
_ADDED_KEYS = (
    "q_tail_ln.weight",
    "q_tail_ln.bias",
    "q_tail.0.weight",
    "q_tail.0.bias",
    "q_tail.2.weight",
    "q_tail.2.bias",
)
TRANSFORM = "add q_tail_ln + q_tail with a zero output linear; every parent tensor copied unchanged"

# The added parameters come from a fresh model, so the conversion is a function
# of the parent checkpoint and this constant alone.
BUILD_SEED = 20260729

# The probe set: seeded uniformly-random legal playouts, cut over the ply range
# where a Hexo position has a contested critic and a wide legal set.
PROBE_SEED = 5_112_303
PROBE_POSITIONS = 64
PROBE_PLIES = (20, 60)
# The plausible set per position, over which the critic's spread is reported.
PROBE_TOP_K = 16

# The preservation bound. A zero output linear makes the tail the exact
# identity, so the honest tolerance is the one that only a bit-for-bit
# preserving graft can meet.
Q_TOLERANCE = 1e-6
KL_TOLERANCE = 1e-9

# The parent's cell-head tensors, by state-dict key: window projection,
# slot-class table, background-bucket table, and the MLP's prefix.
_POLICY_HEAD = ("p.weight", "e_pw.weight", "e_bg.weight", "mlp_p")
_CRITIC_HEAD = ("q.weight", "e_qw.weight", "e_qbg.weight", "mlp_q")

_CHECKPOINT_KEYS = frozenset({"model", "optimizer", "iteration", "rng_state", "versions"})


def _architecture_mismatches(old: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """Every key at which ``old`` is not this build's state dict."""
    mismatches = sorted(set(old) ^ set(current))
    for key in sorted(set(old) & set(current)):
        old_value, current_value = old[key], current[key]
        if not isinstance(old_value, Tensor) or not isinstance(current_value, Tensor):
            if type(old_value) is not type(current_value):
                mismatches.append(key)
        elif old_value.shape != current_value.shape:
            mismatches.append(key)
    return sorted(mismatches)


def _shared_digest(state: dict[str, Any], names: list[str]) -> str:
    """SHA-256 over ``names``' tensors in order — name, dtype, shape, and bytes.

    The fingerprint of the conversion's parent half, so the manifest names the
    tensors it certifies instead of only asserting that they were checked.
    """
    digest = hashlib.sha256()
    for name in names:
        tensor = state[name].detach().cpu().contiguous().flatten()
        digest.update(f"{name} {tensor.dtype} {tuple(state[name].shape)}".encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _zero_step(step: Any) -> Any:
    """Adam's step counter at zero, in whatever type the parent stored."""
    if isinstance(step, Tensor):
        return torch.zeros_like(step)
    if isinstance(step, (int, float)):
        return type(step)(0)
    raise ValueError(f"optimizer step has unexpected type {type(step).__name__}")


def _adam_state(
    parent: dict[str, Any], parent_names: list[str], new_names: list[str], shapes: dict[str, tuple]
) -> dict[str, Any]:
    """The parent's Adam state remapped onto the new parameter list, by name.

    The parent's numeric ids index its own ``named_parameters()`` order, which
    is why the caller pins that order first. Shared parameters keep their
    moments and their step; the tail's parameters arrive with zero moments and
    a zero step, which is what Adam would have built for them on the first
    step anyway. Nothing is rescaled: the transform copies every parent tensor
    unchanged, so every carried moment still describes its own parameter.
    """
    if not isinstance(parent, dict) or not isinstance(parent.get("state"), dict):
        raise ValueError("checkpoint must contain an Adam optimizer state dict")
    groups = parent.get("param_groups")
    if (
        not isinstance(groups, list)
        or len(groups) != 1
        or not isinstance(groups[0], dict)
        or not isinstance(groups[0].get("params"), list)
    ):
        raise ValueError(
            "optimizer must have exactly one param_group holding a params list"
        )
    saved_ids = groups[0]["params"]
    if len(saved_ids) != len(parent_names):
        raise ValueError(
            "optimizer parameter count does not match the checkpoint's parameters: "
            f"{len(saved_ids)} != {len(parent_names)}"
        )

    entries: dict[str, dict] = {}
    for name, param_id in zip(parent_names, saved_ids):
        entry = parent["state"].get(param_id)
        if not isinstance(entry, dict):
            raise ValueError(f"optimizer has no Adam state for {name}")
        missing = [f for f in ("step", "exp_avg", "exp_avg_sq") if f not in entry]
        if missing:
            raise ValueError(f"optimizer state for {name} is missing: {', '.join(missing)}")
        for field in ("exp_avg", "exp_avg_sq"):
            moment = entry[field]
            if not isinstance(moment, Tensor) or tuple(moment.shape) != shapes[name]:
                raise ValueError(
                    f"optimizer {field} for {name} does not have the parameter's "
                    f"shape {shapes[name]}"
                )
        entries[name] = entry

    zero_step = _zero_step(entries[parent_names[0]]["step"])
    state: dict[int, dict] = {}
    for index, name in enumerate(new_names):
        if name in entries:
            state[index] = copy.deepcopy(entries[name])
        else:
            state[index] = {
                "step": copy.deepcopy(zero_step),
                "exp_avg": torch.zeros(shapes[name]),
                "exp_avg_sq": torch.zeros(shapes[name]),
            }
    group = {**copy.deepcopy(groups[0]), "params": list(range(len(new_names)))}
    return {"state": state, "param_groups": [group]}


def _probe_positions() -> list:
    """The probe set: ``PROBE_POSITIONS`` non-terminal positions from seeded
    uniformly-random legal playouts, prefix lengths spread evenly over
    ``PROBE_PLIES``. A playout that ends the game is retried under a further
    derived seed, so the set is a function of ``PROBE_SEED`` alone."""
    low, high = PROBE_PLIES
    positions = []
    for index in range(PROBE_POSITIONS):
        plies = low + round(index * (high - low) / (PROBE_POSITIONS - 1))
        for attempt in range(100):
            rng = random.Random(PROBE_SEED + 1_000_003 * index + 1_009 * attempt)
            position = hexo_py.Position()
            for _ in range(plies):
                position.advance(*rng.choice(position.legal_moves()))
            if not position.is_terminal:
                positions.append(position)
                break
        else:
            raise RuntimeError(f"no non-terminal {plies}-ply probe playout in 100 seeds")
    return positions


def _parent_hidden(
    w: Tensor, g: Tensor, batch: Batch, state: dict[str, Any], head: tuple[str, ...]
) -> Tensor:
    """One parent cell head's MLP hidden over the trunk's own rows.

    The reference the manifest measures against, and the reason it is a
    detector: the rows it reads are the trunk's, so a head that decodes
    anything else — the critic tail leaking into the policy, a wrong split back
    into windows and token, a second aggregation over the wrong rows — moves
    the model away from this and not with it.

    It mirrors the head's arithmetic (the folded head matrix over the aggregate
    rows) and differs only in reading the parent checkpoint's raw tensors: a
    differently ordered transcription of the same formula disagrees in fp32 by
    more than the preservation bound, which would leave the bound measuring
    reassociation instead of the graft.
    """
    proj, e_class, e_bg, mlp = head
    rows = decoder.aggregate(
        w,
        batch.dec_window,
        batch.dec_class,
        batch.dec_cell,
        batch.bg_cell,
        batch.bg_bucket,
        batch.cell_pos.shape[0],
    )
    matrix = decoder.head_matrix(
        state[proj], state[e_class], state[e_bg], state[f"{mlp}.lin_a.weight"]
    )
    pre = F.linear(rows, matrix, state[f"{mlp}.lin_a.bias"])
    g_half = F.linear(g, state[f"{mlp}.lin_b.weight"])
    return F.relu(pre + g_half.index_select(0, batch.cell_pos))


def _parent_readout(
    hidden: Tensor, state: dict[str, Any], head: tuple[str, ...]
) -> Tensor:
    """A parent head's raw scalar per legal cell, from its 1-wide readout."""
    mlp = head[3]
    return F.linear(
        hidden, state[f"{mlp}.out.weight"], state[f"{mlp}.out.bias"]
    ).squeeze(-1)


def _segment_kl(new: Tensor, parent: Tensor, offsets: Tensor) -> Tensor:
    """D_KL(π′_new ‖ π′_parent) per position, in fp64."""
    new, parent = new.double(), parent.double()
    term = torch.where(new > 0, new * (new.log() - parent.log()), new.new_zeros(()))
    return segment_sum(term, segment_ids(offsets), offsets.shape[0] - 1)


def _q_spread(policy: Tensor, q: Tensor, offsets: Tensor) -> float:
    """Median over positions of σ(Q) across the policy's top-``PROBE_TOP_K``
    legal cells — the spread the improvement operator actually exponentiates."""
    spreads = []
    for lo, hi in zip(offsets[:-1].tolist(), offsets[1:].tolist()):
        top = policy[lo:hi].topk(min(PROBE_TOP_K, hi - lo)).indices
        spreads.append(float(q[lo:hi].index_select(0, top).std()))
    return statistics.median(spreads)


def _measure(model: MantisNet, parent_model: dict[str, Any], tau: float, lam: float) -> dict:
    """The grafted model against the parent's action values on the probe set."""
    batch = collate([from_position(p) for p in _probe_positions()])

    captured: list[Tensor] = []
    hook = model.mlp_q.out.register_forward_hook(
        lambda _m, inputs, _out: captured.append(inputs[0].detach())
    )
    try:
        with torch.no_grad():
            _s, w, g = model.trunk(batch)
            policy_new, q_new = model.cell_heads(w, g, batch)
    finally:
        hook.remove()
    if len(captured) != 1:
        raise RuntimeError(
            f"the critic readout ran {len(captured)} times in one forward, expected once"
        )

    with torch.no_grad():
        hidden = _parent_hidden(w, g, batch, parent_model, _CRITIC_HEAD)
        q_parent = torch.tanh(_parent_readout(hidden, parent_model, _CRITIC_HEAD))
        policy_parent = _parent_readout(
            _parent_hidden(w, g, batch, parent_model, _POLICY_HEAD),
            parent_model,
            _POLICY_HEAD,
        )

    offsets = batch.legal_offsets
    kl = _segment_kl(
        improved_policy(policy_new, q_new, offsets, tau, lam).probs,
        improved_policy(policy_parent, q_parent, offsets, tau, lam).probs,
        offsets,
    )
    delta = (q_new - q_parent).abs()
    return {
        "probe": {
            "seed": PROBE_SEED,
            "positions": batch.n_pos,
            "plies": list(PROBE_PLIES),
            "legal_cells": batch.n_cells,
            "top_k": PROBE_TOP_K,
        },
        "tau": tau,
        "lam": lam,
        "q_max_abs_delta": float(delta.max()),
        "q_mean_abs_delta": float(delta.mean()),
        "readout_input_max_abs_delta": float((captured[0] - hidden).abs().max()),
        "q_spread_median_parent": _q_spread(policy_parent, q_parent, offsets),
        "q_spread_median_new": _q_spread(policy_new, q_new, offsets),
        "kl_mean": float(kl.mean()),
        "kl_max": float(kl.max()),
    }


def graft(
    old_path: Path, new_path: Path, tau: float, lam: float, manifest_path: Path
) -> dict:
    """Convert ``old_path``, measure the conversion, and write both files.

    Returns the manifest. Nothing is written unless the preservation property
    holds: every parent tensor bit for bit, and the probe's bounds.
    """
    old_path, new_path, manifest_path = Path(old_path), Path(new_path), Path(manifest_path)
    if old_path.resolve() == new_path.resolve():
        raise ValueError("OLD.pt and NEW.pt must be different paths")

    checkpoint = torch.load(old_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or set(checkpoint) != _CHECKPOINT_KEYS:
        raise ValueError(
            f"checkpoint must hold exactly {sorted(_CHECKPOINT_KEYS)}, got "
            f"{sorted(checkpoint) if isinstance(checkpoint, dict) else type(checkpoint).__name__}"
        )
    parent_model = checkpoint["model"]
    if not isinstance(parent_model, dict):
        raise ValueError("checkpoint's model entry must be a state dict")
    if checkpoint["versions"] != _versions():
        raise ValueError(
            f"checkpoint versions {checkpoint['versions']} != this build {_versions()}"
        )

    torch.manual_seed(BUILD_SEED)
    model = MantisNet(MantisConfig())
    current = model.state_dict()
    added = set(_ADDED_KEYS)
    # The transform adds keys and rewrites none, so a parent that already holds
    # the tail is not a parent this conversion can take.
    present = sorted(added & set(parent_model))
    if present:
        raise ValueError(
            f"parent model already has the critic tail: {', '.join(present)}"
        )
    # The architecture comparison is the whole gate on what else the parent may
    # be: it refuses a missing or extra key and any shape this build does not
    # have — including a critic readout that is not the 1-wide scalar one the
    # measurement below applies to the parent's own activations.
    mismatches = _architecture_mismatches(parent_model, current)
    if set(mismatches) != added:
        raise ValueError(
            "parent model must differ from this build only at the critic tail "
            f"({', '.join(_ADDED_KEYS)}); mismatched keys: "
            f"{', '.join(mismatches) if mismatches else '<none>'}"
        )

    new_names = [name for name, _p in model.named_parameters()]
    shared_names = [name for name in new_names if name not in added]
    # The parent's Adam ids are positions in its own parameter order, and this
    # is the only record of that order, so it is pinned rather than assumed.
    if list(parent_model) != shared_names:
        raise ValueError(
            "parent state dict is not in this build's parameter order, so its "
            "optimizer ids cannot be named"
        )

    converted = {
        "model": {
            name: (current[name].clone() if name in added else parent_model[name])
            for name in new_names
        },
        "optimizer": _adam_state(
            checkpoint["optimizer"],
            shared_names,
            new_names,
            {name: tuple(current[name].shape) for name in new_names},
        ),
        "iteration": checkpoint["iteration"],
        "rng_state": copy.deepcopy(checkpoint["rng_state"]),
        "versions": copy.deepcopy(checkpoint["versions"]),
    }

    # Both halves are proven here rather than at the next resume: strict
    # loading for the parameters, this build's Adam for the remapped state.
    model.load_state_dict(converted["model"])
    torch.optim.Adam(model.parameters()).load_state_dict(converted["optimizer"])
    model.eval()

    # The transform's claim that it rewrites nothing, checked on the tensors it
    # is about to write and on the ones the probe below then measures, against
    # the parent's file read a second time. Reading the file again is what makes
    # this a detector rather than a restatement: the reference cannot be the
    # dict the transform copied from, because a rewrite of that dict would
    # agree with itself, and it cannot be the probe, whose two sides share the
    # grafted trunk and never reach the state-value head.
    source_model = torch.load(old_path, map_location="cpu", weights_only=False)["model"]
    grafted_model = model.state_dict()
    rewritten = [
        name
        for name in shared_names
        if not torch.equal(converted["model"][name], source_model[name])
        or not torch.equal(grafted_model[name], source_model[name])
    ]
    if rewritten:
        raise ValueError(
            "the graft must copy every parent tensor unchanged, and these are not "
            f"the source file's: {', '.join(rewritten)}; nothing written"
        )

    manifest = {
        "arm": ARM,
        "source": str(old_path),
        "source_iteration": checkpoint["iteration"],
        "versions": converted["versions"],
        "transform": TRANSFORM,
        "added_keys": list(_ADDED_KEYS),
        "build_seed": BUILD_SEED,
        **_measure(model, parent_model, tau, lam),
    }
    manifest["preservation"] = {
        "property": (
            "every parent tensor is the source file's bit for bit, and the tail's "
            "zero output linear makes it the identity, so the grafted model's "
            "action values and improved policy are the parent's"
        ),
        "shared_tensors_unchanged": len(shared_names),
        "shared_tensor_sha256": _shared_digest(converted["model"], shared_names),
        "q_max_abs_delta_tolerance": Q_TOLERANCE,
        "kl_max_tolerance": KL_TOLERANCE,
        "holds": manifest["q_max_abs_delta"] <= Q_TOLERANCE
        and manifest["kl_max"] <= KL_TOLERANCE,
    }
    if not manifest["preservation"]["holds"]:
        raise ValueError(
            "the graft did not preserve the parent's critic: max |Q_new - Q_parent| = "
            f"{manifest['q_max_abs_delta']:.3e} (bound {Q_TOLERANCE:.0e}), max KL = "
            f"{manifest['kl_max']:.3e} (bound {KL_TOLERANCE:.0e}); nothing written"
        )

    # The evidence lands before the checkpoint it describes, so no grafted
    # checkpoint can exist without its measurements.
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    torch.save(converted, new_path)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="graft the zero-initialized critic tail onto a trained checkpoint"
    )
    parser.add_argument("old", type=Path, metavar="OLD.pt")
    parser.add_argument("new", type=Path, metavar="NEW.pt")
    # The operating point is required: π′ is measured at it, and a manifest that
    # did not name it would not say what it measured.
    parser.add_argument("--tau", type=float, required=True, help="reverse-KL weight")
    parser.add_argument("--lam", type=float, required=True, help="entropy weight")
    parser.add_argument(
        "--manifest", type=Path, required=True, metavar="OUT.json",
        help="where the conversion's measurements are written",
    )
    args = parser.parse_args(argv)
    graft(args.old, args.new, args.tau, args.lam, args.manifest)


if __name__ == "__main__":
    main()
