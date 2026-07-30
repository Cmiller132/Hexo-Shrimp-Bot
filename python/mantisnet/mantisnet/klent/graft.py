"""Graft the joint-class decoder onto a checkpoint trained with slot classes.

MODEL_SPEC §4.3's decoder class keys a window's occupancy and the candidate's own
slot jointly, where it used to key the reversal-invariant slot class alone. The
two cell heads' class tables therefore grow from 3 rows to ``DEC_CLASSES``, and
the conversion fills each new row with the parent row its class replaces: every
joint class has one slot class — the two members of a reversal orbit are mirrored
slots, and mirrored slots share a slot class — so the expansion is a row
replication and nothing else. Each entry of the decoder incidence then contributes
the embedding it contributed before, and the grafted model is the parent as a
function.

Preservation is exact, and is checked without a tolerance three ways: every parent
tensor the transform does not touch is compared bit for bit against the source
file; each expanded row is compared bit for bit against the parent row it claims
to replicate; and MODEL_SPEC §6's decode, transcribed here rather than taken from
the model, is compared bit for bit between the parent's 3-row tables under its
slot classes and the grafted tables under the builder's joint classes. The third
is end-to-end over both heads and the background path, and catches an expansion
written by one index and read by another — a swapped table, a transposed one, an
off-by-one on one side only.

What none of the three can catch is a row map that is consistently wrong, since it
would be applied to both sides: that ``min(slot, 5 - slot)`` is the parent row of
each joint class rests on ``parent_row_of_class`` refusing a class whose two
mirrored slots disagree, and on the test oracle deriving the same map from the
engine's own window walk.

What is not exact is the model's own arithmetic. The class coefficient block widens
from 3 columns to 93, so the head GEMM's K grows and its sum is reassociated; the
folded path is therefore compared against the spec decode under an fp32 bound,
which one unmodified model already needs against itself.

A conversion that fails any of this writes neither checkpoint nor manifest.

This is also where ``MODEL_REPR_VERSION`` moves: the representation changed, so
the parent's own version is refused everywhere else and rewritten here alone.

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
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ..builder import DEC_CLASSES, WINDOW_LEN, Batch, _DEC_CLASS, collate, from_position
from ..model import MantisConfig, MantisNet
from ..segments import segment_ids, segment_sum
from .improve import improved_policy
from .run import _versions

ARM = "joint-slot-decoder"

# What the transform owns: the two decoder class tables, and nothing else.
_EXPANDED_KEYS = ("e_pw.weight", "e_qw.weight")
TRANSFORM = (
    "expand e_pw and e_qw from 3 slot-class rows to 93 joint-class rows, each a "
    "copy of the slot-class row its class replaces; every other parent tensor "
    "copied unchanged"
)

# The representation the parent was built under. The conversion is the one place
# the version moves, so the version it moves from is named rather than inferred.
PARENT_REPR_VERSION = 1

# The parent's class count, from before the joint key.
PARENT_CLASSES = 3

# The probe set: seeded uniformly-random legal playouts, cut over the ply range
# where a Hexo position has a contested critic and a wide legal set.
PROBE_SEED = 5_112_303
PROBE_POSITIONS = 64
PROBE_PLIES = (20, 60)
# The plausible set per position, over which the critic's spread is reported.
PROBE_TOP_K = 16

# The bounds on the model's own folded arithmetic. Preservation itself is exact
# and checked without a tolerance (``spec_decode_bitwise_equal``); these cover the
# separate fact that the head GEMM sums a 93-wide class block where the spec sums
# per-entry embeddings. That reassociation is not the graft's: one unmodified
# model, decoded both ways over its own tables, measures max |ΔQ| 1.5e-06,
# max |Δlogit| 1.6e-06, and max operator KL 1.8e-06 on this probe set. The bounds
# sit just above that, so they fail on a wrong row map and not on fp32.
Q_TOLERANCE = 1e-5
POLICY_TOLERANCE = 1e-4  # raw logits, which are not squashed into [-1, 1]
KL_TOLERANCE = 1e-5

# The parent's cell-head tensors, by state-dict key: window projection, class
# table, background-bucket table, and the MLP's prefix.
_POLICY_HEAD = ("p.weight", "e_pw.weight", "e_bg.weight", "mlp_p")
_CRITIC_HEAD = ("q.weight", "e_qw.weight", "e_qbg.weight", "mlp_q")

_CHECKPOINT_KEYS = frozenset({"model", "optimizer", "iteration", "rng_state", "versions"})


def parent_row_of_class() -> np.ndarray:
    """The parent slot-class row each joint class replicates, one per class.

    Well-defined because a reversal orbit's two members are mirrored slots of
    mirrored masks, and ``min(s, 5 - s)`` is equal on mirrored slots. That is the
    property the whole conversion rests on, so it is derived here from the
    builder's table and checked, not assumed.
    """
    rows = np.full(DEC_CLASSES, -1, dtype=np.int64)
    for mask in range(1, 63):
        for slot in range(WINDOW_LEN):
            cls = int(_DEC_CLASS[mask, slot])
            if cls < 0:
                continue
            slot_class = min(slot, WINDOW_LEN - 1 - slot)
            if rows[cls] >= 0 and rows[cls] != slot_class:
                raise ValueError(
                    f"joint class {cls} spans slot classes {rows[cls]} and "
                    f"{slot_class}, so no parent row replaces it"
                )
            rows[cls] = slot_class
    if (rows < 0).any():
        raise ValueError(f"{int((rows < 0).sum())} joint classes have no parent row")
    if rows.max() >= PARENT_CLASSES:
        raise ValueError(f"slot class {int(rows.max())} is outside the parent's table")
    return rows


_PARENT_ROW = parent_row_of_class()


def _expand(table: Tensor) -> Tensor:
    """A parent class table as its joint-class replication."""
    if tuple(table.shape[1:]) == () or table.shape[0] != PARENT_CLASSES:
        raise ValueError(
            f"parent class table must have {PARENT_CLASSES} rows, got {tuple(table.shape)}"
        )
    return table.index_select(0, torch.from_numpy(_PARENT_ROW))


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


def _adam_state(
    parent: dict[str, Any], names: list[str], shapes: dict[str, tuple]
) -> dict[str, Any]:
    """The parent's Adam state remapped onto the new parameter list, by name.

    The parent's numeric ids index its own ``named_parameters()`` order, which is
    why the caller pins that order first. The transform neither adds nor removes
    a parameter, so every entry is carried; the two expanded tables' moments are
    replicated by the same rows as their weights.

    Replicated rather than zeroed, because Adam's update is a ratio of the two
    moments: a row that inherits both steps as its parent row would have, while a
    zeroed row would take one full-size bias-corrected step first. The conversion
    is meant to be continuous in the optimizer as well as in the function.
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
    if len(saved_ids) != len(names):
        raise ValueError(
            "optimizer parameter count does not match the checkpoint's parameters: "
            f"{len(saved_ids)} != {len(names)}"
        )

    state: dict[int, dict] = {}
    for index, (name, param_id) in enumerate(zip(names, saved_ids)):
        entry = parent["state"].get(param_id)
        if entry is None:
            # Adam's state dict is sparse: a parameter it never stepped has no
            # entry at all, which is how the state-value head KLENT does not
            # train appears in every checkpoint this repo writes. Absence is the
            # parent's own statement that the parameter is unstepped, so it is
            # carried as absence rather than zero-filled.
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"optimizer state for {name} is not a state dict")
        missing = [f for f in ("step", "exp_avg", "exp_avg_sq") if f not in entry]
        if missing:
            raise ValueError(f"optimizer state for {name} is missing: {', '.join(missing)}")
        expanded = name in _EXPANDED_KEYS
        want = (PARENT_CLASSES,) + shapes[name][1:] if expanded else shapes[name]
        for field in ("exp_avg", "exp_avg_sq"):
            moment = entry[field]
            if not isinstance(moment, Tensor) or tuple(moment.shape) != want:
                raise ValueError(
                    f"optimizer {field} for {name} does not have the parameter's "
                    f"shape {want}"
                )
        entry = copy.deepcopy(entry)
        if expanded:
            for field in ("exp_avg", "exp_avg_sq"):
                entry[field] = _expand(entry[field])
        state[index] = entry
    if not state:
        raise ValueError("the parent's optimizer stepped no parameter at all")
    group = {**copy.deepcopy(groups[0]), "params": list(range(len(names)))}
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


def _spec_scores(
    w: Tensor,
    g: Tensor,
    batch: Batch,
    state: dict[str, Any],
    head: tuple[str, ...],
    classes: Tensor,
) -> Tensor:
    """One cell head's raw scalar per legal cell, from MODEL_SPEC §6.

    The formula as the spec writes it — project each window row, add the entry's
    class embedding, sum a cell's entries, and take the background path from the
    bucket table. Given the parent's 3-row table and its slot classes it is the
    parent's decode; given the grafted table and the builder's joint classes it is
    the child's. Neither is the model's own arithmetic: the sum is over per-entry
    embeddings rather than a 93-wide coefficient block, and it never builds a
    folded head matrix.

    Run both ways it is what makes preservation exact rather than approximate.
    The expansion copies rows, so the two runs add the same embedding to the same
    window row for every entry, in the same order — the results agree bit for bit
    or the row map is wrong.
    """
    proj, e_class, e_bg, mlp = head
    msg = F.linear(w, state[proj]).index_select(0, batch.dec_window) + state[
        e_class
    ].index_select(0, classes)
    h = torch.zeros(batch.cell_pos.shape[0], w.shape[1], dtype=w.dtype)
    h.index_add_(0, batch.dec_cell, msg)
    if batch.bg_cell.numel():
        h.index_copy_(0, batch.bg_cell, state[e_bg].index_select(0, batch.bg_bucket))
    hidden = F.relu(
        F.linear(h, state[f"{mlp}.lin_a.weight"], state[f"{mlp}.lin_a.bias"])
        + F.linear(g, state[f"{mlp}.lin_b.weight"]).index_select(0, batch.cell_pos)
    )
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
    """The grafted model against the parent's own decode on the probe set."""
    batch = collate([from_position(p) for p in _probe_positions()])

    child_model = model.state_dict()
    slot_class = torch.from_numpy(_PARENT_ROW).index_select(0, batch.dec_class)
    with torch.no_grad():
        _s, w, g = model.trunk(batch)
        policy_new, q_new = model.cell_heads(w, g, batch)
        # The trunk is shared: the transform touches neither its parameters nor
        # its inputs, so both sides read these rows and the comparison isolates
        # the decoder.
        policy_parent = _spec_scores(w, g, batch, parent_model, _POLICY_HEAD, slot_class)
        q_parent = torch.tanh(
            _spec_scores(w, g, batch, parent_model, _CRITIC_HEAD, slot_class)
        )
        # The same formula over the grafted tables and the joint classes. This is
        # the exact half of preservation: equal bit for bit, or the expansion put
        # a row somewhere the decoder does not read it.
        policy_child = _spec_scores(w, g, batch, child_model, _POLICY_HEAD, batch.dec_class)
        q_child = torch.tanh(
            _spec_scores(w, g, batch, child_model, _CRITIC_HEAD, batch.dec_class)
        )
        exact = torch.equal(policy_child, policy_parent) and torch.equal(q_child, q_parent)

    offsets = batch.legal_offsets
    kl = _segment_kl(
        improved_policy(policy_new, q_new, offsets, tau, lam).probs,
        improved_policy(policy_parent, q_parent, offsets, tau, lam).probs,
        offsets,
    )
    q_delta = (q_new - q_parent).abs()
    policy_delta = (policy_new - policy_parent).abs()
    return {
        "probe": {
            "seed": PROBE_SEED,
            "positions": batch.n_pos,
            "plies": list(PROBE_PLIES),
            "legal_cells": batch.n_cells,
            "decoder_entries": int(batch.dec_class.shape[0]),
            "classes_exercised": int(batch.dec_class.unique().numel()),
            "top_k": PROBE_TOP_K,
        },
        "tau": tau,
        "lam": lam,
        "spec_decode_bitwise_equal": bool(exact),
        "q_max_abs_delta": float(q_delta.max()),
        "q_mean_abs_delta": float(q_delta.mean()),
        "policy_max_abs_delta": float(policy_delta.max()),
        "policy_mean_abs_delta": float(policy_delta.mean()),
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
    holds: every parent tensor bit for bit, every expanded row its parent row bit
    for bit, and the probe's bounds.
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
    # The parent is a build of the representation this conversion converts from,
    # and differs from the running build in exactly that field.
    parent_versions = {**_versions(), "MODEL_REPR_VERSION": PARENT_REPR_VERSION}
    if checkpoint["versions"] != parent_versions:
        raise ValueError(
            f"checkpoint versions {checkpoint['versions']} != the parent build "
            f"{parent_versions}"
        )

    model = MantisNet(MantisConfig())
    current = model.state_dict()
    expanded = set(_EXPANDED_KEYS)
    # The architecture comparison is the whole gate on what else the parent may
    # be: it refuses a missing or extra key and any shape this build does not
    # have, so the only difference it tolerates is the two class tables.
    mismatches = _architecture_mismatches(parent_model, current)
    if set(mismatches) != expanded:
        raise ValueError(
            "parent model must differ from this build only at the decoder class "
            f"tables ({', '.join(_EXPANDED_KEYS)}); mismatched keys: "
            f"{', '.join(mismatches) if mismatches else '<none>'}"
        )
    for key in _EXPANDED_KEYS:
        want = (PARENT_CLASSES, current[key].shape[1])
        if tuple(parent_model[key].shape) != want:
            raise ValueError(
                f"parent {key} must have shape {want}, got {tuple(parent_model[key].shape)}"
            )

    new_names = [name for name, _p in model.named_parameters()]
    shared_names = [name for name in new_names if name not in expanded]
    # The parent's Adam ids are positions in its own parameter order, and this is
    # the only record of that order, so it is pinned rather than assumed.
    if list(parent_model) != new_names:
        raise ValueError(
            "parent state dict is not in this build's parameter order, so its "
            "optimizer ids cannot be named"
        )

    converted = {
        "model": {
            name: (
                _expand(parent_model[name]) if name in expanded else parent_model[name]
            )
            for name in new_names
        },
        "optimizer": _adam_state(
            checkpoint["optimizer"],
            new_names,
            {name: tuple(current[name].shape) for name in new_names},
        ),
        "iteration": checkpoint["iteration"],
        "rng_state": copy.deepcopy(checkpoint["rng_state"]),
        # The representation moved, and this is where it moves.
        "versions": _versions(),
    }

    # Both halves are proven here rather than at the next resume: strict loading
    # for the parameters, this build's Adam for the remapped state.
    model.load_state_dict(converted["model"])
    torch.optim.Adam(model.parameters()).load_state_dict(converted["optimizer"])
    model.eval()

    # The transform's claim about what it rewrites, checked on the tensors it is
    # about to write and on the ones the probe then measures, against the
    # parent's file read a second time. Reading the file again is what makes this
    # a detector rather than a restatement: the reference cannot be the dict the
    # transform copied from, because a rewrite of that dict would agree with
    # itself.
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
            "the graft must copy every untouched parent tensor unchanged, and these "
            f"are not the source file's: {', '.join(rewritten)}; nothing written"
        )
    # Each expanded row is its parent row, bit for bit — the whole content of the
    # transform, and a tolerance-free statement of it.
    misplaced = [
        f"{name}[{cls}]"
        for name in _EXPANDED_KEYS
        for cls in range(DEC_CLASSES)
        if not torch.equal(grafted_model[name][cls], source_model[name][_PARENT_ROW[cls]])
    ]
    if misplaced:
        raise ValueError(
            f"{len(misplaced)} expanded rows are not the parent row they replace: "
            f"{', '.join(misplaced[:8])}; nothing written"
        )

    manifest = {
        "arm": ARM,
        "source": str(old_path),
        "source_iteration": checkpoint["iteration"],
        "parent_versions": parent_versions,
        "versions": converted["versions"],
        "transform": TRANSFORM,
        "expanded_keys": list(_EXPANDED_KEYS),
        "classes": DEC_CLASSES,
        "parent_classes": PARENT_CLASSES,
        "class_to_parent_row": _PARENT_ROW.tolist(),
        **_measure(model, parent_model, tau, lam),
    }
    manifest["preservation"] = {
        "property": (
            "every untouched parent tensor is the source file's bit for bit, each "
            "of the 93 joint-class rows is bit for bit the slot-class row it "
            "replaces, and the spec decode over the grafted tables and joint "
            "classes is bit for bit the parent's over its own tables and slot "
            "classes, so the grafted model is the parent as a function"
        ),
        "shared_tensors_unchanged": len(shared_names),
        "shared_tensor_sha256": _shared_digest(converted["model"], shared_names),
        "expanded_rows_checked": len(_EXPANDED_KEYS) * DEC_CLASSES,
        "arithmetic": (
            "the class coefficient block widens from 3 to 93, so the head GEMM's "
            "sum is reassociated; the deltas below are that fp32 slack, which one "
            "unmodified model already shows between its folded and spec decodes"
        ),
        "q_max_abs_delta_tolerance": Q_TOLERANCE,
        "policy_max_abs_delta_tolerance": POLICY_TOLERANCE,
        "kl_max_tolerance": KL_TOLERANCE,
        "holds": (
            manifest["spec_decode_bitwise_equal"]
            and manifest["q_max_abs_delta"] <= Q_TOLERANCE
            and manifest["policy_max_abs_delta"] <= POLICY_TOLERANCE
            and manifest["kl_max"] <= KL_TOLERANCE
        ),
    }
    if not manifest["preservation"]["holds"]:
        raise ValueError(
            "the graft did not preserve the parent's decode: spec decode bitwise "
            f"equal = {manifest['spec_decode_bitwise_equal']}, max "
            f"|Q_new - Q_parent| = {manifest['q_max_abs_delta']:.3e} (bound "
            f"{Q_TOLERANCE:.0e}), max |logit_new - logit_parent| = "
            f"{manifest['policy_max_abs_delta']:.3e} (bound {POLICY_TOLERANCE:.0e}), "
            f"max KL = {manifest['kl_max']:.3e} (bound {KL_TOLERANCE:.0e}); "
            "nothing written"
        )

    # The evidence lands before the checkpoint it describes, so no grafted
    # checkpoint can exist without its measurements.
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    torch.save(converted, new_path)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="graft the joint-class decoder onto a slot-class checkpoint"
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
