"""Convert a scalar-tanh-critic checkpoint to this build's return-mass critic.

The parent checkpoint's action-value readout is one row ``(W_s, b_s)`` whose
score ``z`` fed ``tanh``. This build's readout is two rows composed as
``sigmoid(z_pos) − sigmoid(z_neg)`` (MODEL_SPEC appendix B), and

    sigmoid(2z) − sigmoid(−2z) = 2·sigmoid(2z) − 1 = tanh(z),

so setting ``W_pos = 2·W_s``, ``b_pos = 2·b_s``, ``W_neg = −2·W_s`` and
``b_neg = −2·b_s`` carries the trained critic over unchanged as a function.
The conversion is one-way and explicit: checkpoint formats are not
interchangeable, and every other loader in this build stays strict.

The graft measures what it claims. It runs a fixed probe set through two
models — the grafted one, and the parent checkpoint in the architecture it was
trained with — and refuses to write anything unless the identity holds to
``MAX_ABS_DQ`` per cell and ``MAX_MEAN_KL`` in the improved policy at the
supplied operating point. The two forwards share no activation and the parent's
shares no code with the transform, so the measurement covers every tensor the
conversion carries over, not only the two it rewrites. ``--tau`` and ``--lam``
are required because the π′ measurements mean nothing without the point they
were taken at.

Run from ``python/mantisnet``:

    python -m mantisnet.klent.graft OLD.pt NEW.pt --tau 0.1 --lam 0.01 \
        --manifest OUT.json
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from ..builder import collate_prefixes
from ..model import MantisConfig, MantisNet
from ..segments import segment_ids, segment_sum
from .improve import improved_policy
from .run import _versions

ARM = "bipolar-return-mass"

# The two readout tensors this transform owns; the parent must differ from this
# build's architecture at exactly these keys and no others.
_READOUT_KEYS = ("mlp_q.out.weight", "mlp_q.out.bias")

# The gain that makes the composition reproduce tanh: sigmoid(2z) − sigmoid(−2z).
READOUT_GAIN = 2.0
TRANSFORM = "W_pos = 2 W_s, b_pos = 2 b_s; W_neg = -2 W_s, b_neg = -2 b_s"

# Every top-level key of the parent checkpoint. Unknown keys are refused rather
# than dropped: what this writes is a complete checkpoint or nothing.
_CHECKPOINT_KEYS = ("model", "optimizer", "iteration", "rng_state", "versions")

_ADAM_FIELDS = ("step", "exp_avg", "exp_avg_sq")

# The probe set: a constant of this module, so two runs of the graft measure
# the same positions. Prefix lengths spread evenly over the ply window.
PROBE_SEED = 314_159
PROBE_POSITIONS = 64
PROBE_PLIES = (20, 60)
PROBE_TOP_K = 16
_PROBE_ATTEMPTS = 100

# The stated preservation property: the grafted model's action values are the
# parent's, per cell and through the improvement operator.
MAX_ABS_DQ = 1e-5
MAX_MEAN_KL = 1e-6
PRESERVATION = (
    "the grafted model and the parent checkpoint, each run in its own "
    "architecture, agree on every probe cell's action value and on the improved "
    "policy over them, because sigmoid(2z) - sigmoid(-2z) = tanh(z)"
)


def _signed_rows(row: Tensor) -> Tensor:
    """The grafted readout's two rows from the parent's one: ``+g``, then ``−g``."""
    return torch.cat([READOUT_GAIN * row, -READOUT_GAIN * row], dim=0)


def _converted_state(old_model: dict) -> dict:
    """This build's state dict from the parent's: the readout rewritten as two
    signed rows, every other tensor the parent's own."""
    converted = dict(old_model)
    for key in _READOUT_KEYS:
        converted[key] = _signed_rows(old_model[key])
    return converted


def _check_checkpoint(checkpoint: Any) -> None:
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a dict")
    keys = set(checkpoint)
    if keys != set(_CHECKPOINT_KEYS):
        raise ValueError(
            f"checkpoint keys must be exactly {sorted(_CHECKPOINT_KEYS)}, got "
            f"{sorted(keys)}"
        )
    if not isinstance(checkpoint["model"], dict):
        raise ValueError("checkpoint['model'] must be a state dict")
    if not isinstance(checkpoint["iteration"], int):
        raise ValueError("checkpoint['iteration'] must be an int")
    if not isinstance(checkpoint["rng_state"], dict):
        raise ValueError("checkpoint['rng_state'] must be a bit-generator state")
    # The converted checkpoint carries the parent's versions forward, so a
    # parent from another build would produce a file this build refuses to
    # load. That belongs here, not at the next resume.
    if checkpoint["versions"] != _versions():
        raise ValueError(
            f"parent versions {checkpoint['versions']} != this build "
            f"{_versions()}"
        )


def _architecture_mismatches(old: dict, current: dict) -> list[str]:
    """Keys where ``old`` does not match this build's state dict."""
    mismatched = set(old) ^ set(current)
    for key in set(old) & set(current):
        value, want = old[key], current[key]
        if not isinstance(value, Tensor) or value.shape != want.shape:
            mismatched.add(key)
    return sorted(mismatched)


def _check_parent_readout(old_model: dict, current: dict) -> None:
    """Refuse a parent whose critic readout is not one scalar-scoring row."""
    hidden = current[_READOUT_KEYS[0]].shape[1]
    expected = {_READOUT_KEYS[0]: (1, hidden), _READOUT_KEYS[1]: (1,)}
    wrong = [
        key
        for key in _READOUT_KEYS
        if not isinstance(old_model[key], Tensor)
        or tuple(old_model[key].shape) != expected[key]
    ]
    if wrong:
        shapes = ", ".join(f"{key}: {tuple(old_model[key].shape)}" for key in wrong)
        raise ValueError(
            f"the parent critic readout must be one row {expected}; got {shapes}"
        )


def _parent_model(old_model: dict, cfg: MantisConfig) -> MantisNet:
    """The parent checkpoint in its own architecture: this build with the
    one-wide critic readout the parent was trained with, strict-loaded.

    The measurement's second path. Reading the parent out of the checkpoint
    rather than out of the grafted model is what lets the preservation numbers
    cover the whole state dict instead of the two tensors this transform writes.
    """
    parent = MantisNet(cfg)
    parent.mlp_q.out = nn.Linear(parent.mlp_q.out.in_features, 1)
    parent.load_state_dict(old_model)
    parent.eval()
    return parent


def _remap_adam(saved: Any, old_model: dict, names: list[str]) -> dict:
    """Rebuild the Adam state onto this build's parameter list, keyed by name.

    The parent's optimizer is one Adam group over ``model.parameters()`` in
    ``named_parameters()`` order, so a saved id's position in that list names
    the parameter it belongs to. Every moment tensor is checked against its
    parameter's shape in the parent state dict, which is what would catch a
    parent whose id order was not that order. The rebuilt group's ``params``
    are contiguous over this build's order.
    """
    if not isinstance(saved, dict) or not isinstance(saved.get("state"), dict):
        raise ValueError("checkpoint['optimizer'] must carry an Adam state dict")
    groups = saved.get("param_groups")
    if (
        not isinstance(groups, list)
        or len(groups) != 1
        or not isinstance(groups[0], dict)
        or not isinstance(groups[0].get("params"), list)
    ):
        raise ValueError(
            "the parent optimizer must be a single Adam parameter group over "
            "model.parameters()"
        )
    ids = groups[0]["params"]
    if len(ids) != len(names) or len(set(ids)) != len(ids):
        raise ValueError(
            f"the optimizer group holds {len(ids)} distinct parameter ids, but "
            f"the model has {len(names)} parameters"
        )

    state: dict[int, dict] = {}
    for position, (name, param_id) in enumerate(zip(names, ids)):
        entry = saved["state"].get(param_id)
        if not isinstance(entry, dict):
            raise ValueError(f"the optimizer has no Adam state for {name}")
        missing = [field for field in _ADAM_FIELDS if field not in entry]
        if missing:
            raise ValueError(
                f"the optimizer state for {name} is missing: {', '.join(missing)}"
            )
        shape = old_model[name].shape
        wrong = [
            field
            for field in ("exp_avg", "exp_avg_sq")
            if not isinstance(entry[field], Tensor) or entry[field].shape != shape
        ]
        if wrong:
            raise ValueError(
                f"the optimizer {', '.join(wrong)} for {name} does not have that "
                f"parameter's shape {tuple(shape)}"
            )
        entry = copy.deepcopy(entry)
        if name in _READOUT_KEYS:
            extra = set(entry) - set(_ADAM_FIELDS)
            if extra:
                raise ValueError(
                    f"the optimizer state for {name} carries {sorted(extra)}, "
                    "which this transform cannot map onto two rows"
                )
            # Each new row is s = ±g times the parent's, so the first moment
            # scales with it and the second does not: Adam's update
            # lr·m̂/(√v̂+ε) is scale-free in v, so carrying (s·m, v, step)
            # makes the first post-graft step exactly s times the parent's —
            # the same step in function space that the readout itself preserves.
            entry["exp_avg"] = _signed_rows(entry["exp_avg"])
            entry["exp_avg_sq"] = torch.cat(
                [entry["exp_avg_sq"], entry["exp_avg_sq"]], dim=0
            )
        state[position] = entry

    group = dict(groups[0])
    group["params"] = list(range(len(names)))
    return {"state": state, "param_groups": [group]}


def _probe_prefixes() -> list[list[tuple[int, int]]]:
    """``PROBE_POSITIONS`` nonterminal move prefixes under ``PROBE_SEED``.

    Uniformly random legal playouts, one generator drawing all of them in
    order, with lengths spread evenly across ``PROBE_PLIES``. A playout that
    ends the game is redrawn, so no probe position is terminal.
    """
    import hexo_py

    rng = np.random.default_rng(PROBE_SEED)
    low, high = PROBE_PLIES
    prefixes: list[list[tuple[int, int]]] = []
    for index in range(PROBE_POSITIONS):
        plies = low + round(index * (high - low) / (PROBE_POSITIONS - 1))
        for _attempt in range(_PROBE_ATTEMPTS):
            position, moves = hexo_py.Position(), []
            for _ in range(plies):
                legal = position.legal_moves()
                move = legal[int(rng.integers(len(legal)))]
                position.advance(*move)
                moves.append(move)
            if not position.is_terminal:
                break
        else:
            raise RuntimeError(
                f"no nonterminal {plies}-ply playout in {_PROBE_ATTEMPTS} draws"
            )
        prefixes.append(moves)
    return prefixes


def _top_k_spread(scores: Tensor, values: Tensor, offsets: Tensor) -> float:
    """Median over positions of the standard deviation of ``values`` over each
    position's ``PROBE_TOP_K`` highest-scoring cells."""
    spreads = []
    bounds = offsets.tolist()
    for start, stop in zip(bounds, bounds[1:]):
        if stop - start < PROBE_TOP_K:
            raise RuntimeError(
                f"a probe position has {stop - start} legal cells, fewer than "
                f"the {PROBE_TOP_K} the spread statistic reads"
            )
        top = scores[start:stop].topk(PROBE_TOP_K).indices
        spreads.append(float(values[start:stop].index_select(0, top).std()))
    return float(np.median(spreads))


def _measure(model: MantisNet, parent: MantisNet, tau: float, lam: float) -> dict:
    """Compare the grafted model against the parent model on the probe set.

    Two forwards over the same positions, one per model. They share no
    activation, and the parent's side reuses neither the transform nor the
    grafted readout, so a difference in any carried-over tensor — not just in
    the two the transform writes — moves the numbers this returns.
    """
    prefixes = _probe_prefixes()
    batch = collate_prefixes(prefixes, [len(moves) for moves in prefixes])

    with torch.no_grad():
        _s, w, g = model.trunk(batch)
        policy_new, q_new = model.cell_heads(w, g, batch)
        _s, w, g = parent.trunk(batch)
        policy_parent, score = parent.cell_head_logits(w, g, batch)
    # The parent's tail, transcribed: its one score per cell through tanh.
    q_parent = torch.tanh(score.float()).squeeze(-1)

    offsets = batch.legal_offsets
    delta = (q_new - q_parent).abs()
    # π′ at the supplied operating point, from the operator acting uses, each
    # model from its own policy logits and action values, in float64. A correct
    # graft's two Q vectors differ only by fp32 rounding, so an fp32 π′ would
    # report a KL the size of MAX_MEAN_KL itself: the tolerance has to bound a
    # difference between the policies, not the operator's own noise.
    pi_new = improved_policy(policy_new.double(), q_new.double(), offsets, tau, lam)
    pi_old = improved_policy(
        policy_parent.double(), q_parent.double(), offsets, tau, lam
    )
    probs = pi_new.probs
    terms = torch.where(probs > 0, probs * (probs.log() - pi_old.probs.log()), 0.0)
    kl = segment_sum(terms, segment_ids(offsets), batch.n_pos)

    return {
        "probe_seed": PROBE_SEED,
        "probe_positions": int(batch.n_pos),
        "probe_legal_cells": int(offsets[-1]),
        "probe_plies": [len(moves) for moves in prefixes],
        "max_abs_dq": float(delta.max()),
        "mean_abs_dq": float(delta.mean()),
        "top_k": PROBE_TOP_K,
        "top_k_q_std_median_parent": _top_k_spread(policy_parent, q_parent, offsets),
        "top_k_q_std_median_grafted": _top_k_spread(policy_new, q_new, offsets),
        "mean_improved_kl": float(kl.mean()),
        "max_improved_kl": float(kl.max()),
    }


def graft(
    old_path: Path | str,
    new_path: Path | str,
    tau: float,
    lam: float,
    manifest_path: Path | str,
) -> dict:
    """Convert ``old_path``, measure the conversion, and write both artifacts.

    Nothing is written unless the preservation property holds: the checkpoint
    and the manifest appear together or not at all.
    """
    old_path, new_path = Path(old_path), Path(new_path)
    manifest_path = Path(manifest_path)
    if old_path.resolve() == new_path.resolve():
        raise ValueError("OLD.pt and NEW.pt must be different paths")

    checkpoint = torch.load(old_path, map_location="cpu", weights_only=False)
    _check_checkpoint(checkpoint)

    cfg = MantisConfig()
    model = MantisNet(cfg)
    current = model.state_dict()
    old_model = checkpoint["model"]
    mismatches = _architecture_mismatches(old_model, current)
    if set(mismatches) != set(_READOUT_KEYS):
        raise ValueError(
            "the parent model must differ from this build's architecture at "
            f"exactly {', '.join(_READOUT_KEYS)}; it differs at "
            f"{', '.join(mismatches) if mismatches else '<nothing>'}"
        )
    _check_parent_readout(old_model, current)

    model.load_state_dict(_converted_state(old_model))
    model.eval()

    names = [name for name, _param in model.named_parameters()]
    optimizer = _remap_adam(checkpoint["optimizer"], old_model, names)

    manifest = {
        "arm": ARM,
        "source": str(old_path),
        "source_iteration": checkpoint["iteration"],
        "versions": checkpoint["versions"],
        "transform": TRANSFORM,
        "tau": tau,
        "lam": lam,
        **_measure(model, _parent_model(old_model, cfg), tau, lam),
    }
    manifest["preservation"] = {
        "property": PRESERVATION,
        "max_abs_dq": manifest["max_abs_dq"],
        "max_abs_dq_tolerance": MAX_ABS_DQ,
        "mean_improved_kl": manifest["mean_improved_kl"],
        "mean_improved_kl_tolerance": MAX_MEAN_KL,
    }
    if (
        manifest["max_abs_dq"] > MAX_ABS_DQ
        or abs(manifest["mean_improved_kl"]) > MAX_MEAN_KL
    ):
        raise ValueError(
            "the graft is not function preserving on the probe set: "
            f"max |Q_new - Q_parent| = {manifest['max_abs_dq']:.3e} (tolerance "
            f"{MAX_ABS_DQ:.0e}), mean KL(pi'_new || pi'_parent) = "
            f"{manifest['mean_improved_kl']:.3e} (tolerance {MAX_MEAN_KL:.0e}); "
            "no checkpoint and no manifest were written"
        )

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer,
            "iteration": checkpoint["iteration"],
            "rng_state": checkpoint["rng_state"],
            "versions": checkpoint["versions"],
        },
        new_path,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="graft this build's return-mass critic onto a scalar-Q checkpoint"
    )
    parser.add_argument("old", type=Path, metavar="OLD.pt")
    parser.add_argument("new", type=Path, metavar="NEW.pt")
    parser.add_argument(
        "--tau", type=float, required=True,
        help="the operating point's reverse-KL weight, for the pi' measurements",
    )
    parser.add_argument(
        "--lam", type=float, required=True,
        help="the operating point's entropy weight, for the pi' measurements",
    )
    parser.add_argument(
        "--manifest", type=Path, required=True, metavar="OUT.json",
        help="where to write the measured conversion record",
    )
    args = parser.parse_args(argv)
    manifest = graft(
        args.old, args.new, tau=args.tau, lam=args.lam, manifest_path=args.manifest
    )
    print(json.dumps(manifest["preservation"], indent=2), flush=True)


if __name__ == "__main__":
    main()
