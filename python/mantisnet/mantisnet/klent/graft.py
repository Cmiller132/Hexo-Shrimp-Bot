"""Apply the joint-class decoder and trinomial critic grafts together.

The parent has three-row slot-class tables in both cell heads and a one-row
scalar critic readout followed by ``tanh``. This build has 93-row joint-class
tables and a three-row categorical critic composed as ``p_pos - p_neg``.
The transforms are disjoint:

* each joint-class row copies the slot-class row it replaces; and
* ``(W, b)`` becomes positive row ``(W, b)``, negative row ``(-W, -b)``,
  and zero row ``(0, -20)``. With a vanishing zero outcome this is the
  softmax gap-``2z`` identity ``p_pos - p_neg = tanh(z)`` up to relative
  error at most ``exp(-20) / 2``.

The conversion applies both in one state-dict pass and remaps each parameter's
Adam moments exactly as its transform requires. There is no single-arm mode.

The joint detector compares every untouched tensor with a second read of the
source file, checks every replicated row bit for bit, and transcribes
MODEL_SPEC §6 independently to compare the parent slot decode with the expanded
joint decode bit for bit. The critic detector runs the complete grafted model and
the parent checkpoint in its own architecture on a separate fixed probe, then
enforces ``MAX_ABS_DQ`` and ``MAX_MEAN_KL`` at the supplied ``--tau``/``--lam``.
A failed check writes neither output.

``MODEL_REPR_VERSION`` moves here because the joint decoder changes the model
representation. Run from ``python/mantisnet``:

    python -m mantisnet.klent.graft OLD.pt NEW.pt --tau T --lam L

The evidence sidecar is derived from ``NEW.pt`` as ``NEW.json``.
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
from torch import Tensor, nn

from ..builder import (
    DEC_CLASSES,
    WINDOW_LEN,
    Batch,
    _DEC_CLASS,
    collate,
    collate_prefixes,
    from_position,
)
from ..model import MantisConfig, MantisNet
from ..segments import segment_ids, segment_sum
from .improve import improved_policy
from .run import _versions

ARM = "joint-trinomial"

_EXPANDED_KEYS = ("e_pw.weight", "e_qw.weight")
_READOUT_KEYS = ("mlp_q.out.weight", "mlp_q.out.bias")
_TRANSFORMED_KEYS = frozenset((*_EXPANDED_KEYS, *_READOUT_KEYS))
ZERO_LOGIT = -20.0
TRANSFORM = (
    "expand e_pw and e_qw from 3 slot-class rows to 93 joint-class rows by "
    "parent-row replication; set the scalar critic readout to W_pos = W_s, "
    "b_pos = b_s, W_neg = -W_s, b_neg = -b_s, W_zero = 0, b_zero = -20; "
    "copy every other parent "
    "tensor unchanged"
)

# The representation the parent was built under. The conversion is the one place
# the version moves, so the version it moves from is named rather than inferred.
PARENT_REPR_VERSION = 1

# The parent's class count, from before the joint key.
PARENT_CLASSES = 3

# The joint decoder's independent spec-decode probe.
JOINT_PROBE_SEED = 5_112_303
JOINT_PROBE_POSITIONS = 64
JOINT_PROBE_PLIES = (20, 60)
JOINT_PROBE_TOP_K = 16

# The critic arm's fixed parent-vs-grafted probe, retained from that arm.
PROBE_SEED = 314_159
PROBE_POSITIONS = 64
PROBE_PLIES = (20, 60)
PROBE_TOP_K = 16
_PROBE_ATTEMPTS = 100

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

# The critic arm's stated preservation bounds. These are fixed evidence gates, not
# tuning parameters.
MAX_ABS_DQ = 1e-5
MAX_MEAN_KL = 1e-6
PRESERVATION = (
    "the composed joint-class and trinomial model agrees with the "
    "slot-class scalar-tanh parent on every probe cell's action value and on "
    "the improved policy, because replicated decoder rows preserve each "
    "incidence contribution and a softmax over (z, -z, -20) agrees with "
    "tanh(z) to the stated bounds"
)

# The graft begins at committed mass approximately one. The floor therefore
# does not bind, but the evidence names the reference recipe's operating point.
_GRAFT_FLOOR = 0.2

# The parent's cell-head tensors, by state-dict key: window projection, class
# table, background-bucket table, and the MLP's prefix.
_POLICY_HEAD = ("p.weight", "e_pw.weight", "e_bg.weight", "mlp_p")
_CRITIC_HEAD = ("q.weight", "e_qw.weight", "e_qbg.weight", "mlp_q")

_CHECKPOINT_KEYS = frozenset(
    {"model", "optimizer", "iteration", "rng_state", "versions"}
)
_ADAM_FIELDS = ("step", "exp_avg", "exp_avg_sq")


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
            f"parent class table must have {PARENT_CLASSES} rows, got "
            f"{tuple(table.shape)}"
        )
    return table.index_select(0, torch.from_numpy(_PARENT_ROW))


def _critic_rows(row: Tensor, zero: float = 0.0) -> Tensor:
    """Map one scalar row to positive, negative, and constant-zero rows."""
    return torch.cat([row, -row, torch.full_like(row, zero)], dim=0)


def _converted_state(old_model: dict[str, Any]) -> dict[str, Any]:
    """Apply both disjoint parameter transforms in one state-dict pass."""
    converted = dict(old_model)
    for key in _EXPANDED_KEYS:
        converted[key] = _expand(old_model[key])
    converted[_READOUT_KEYS[0]] = _critic_rows(old_model[_READOUT_KEYS[0]])
    converted[_READOUT_KEYS[1]] = _critic_rows(
        old_model[_READOUT_KEYS[1]], zero=ZERO_LOGIT
    )
    return converted


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


def _check_checkpoint(checkpoint: Any) -> dict[str, Any]:
    """Validate the complete parent envelope and return its required versions."""
    if not isinstance(checkpoint, dict) or set(checkpoint) != _CHECKPOINT_KEYS:
        got = (
            sorted(checkpoint)
            if isinstance(checkpoint, dict)
            else type(checkpoint).__name__
        )
        raise ValueError(
            f"checkpoint must hold exactly {sorted(_CHECKPOINT_KEYS)}, got {got}"
        )
    if not isinstance(checkpoint["model"], dict):
        raise ValueError("checkpoint's model entry must be a state dict")
    if not isinstance(checkpoint["iteration"], int):
        raise ValueError("checkpoint's iteration must be an int")
    if not isinstance(checkpoint["rng_state"], dict):
        raise ValueError("checkpoint's rng_state must be a bit-generator state")
    parent_versions = {**_versions(), "MODEL_REPR_VERSION": PARENT_REPR_VERSION}
    if checkpoint["versions"] != parent_versions:
        raise ValueError(
            f"checkpoint versions {checkpoint['versions']} != the parent build "
            f"{parent_versions}"
        )
    return parent_versions


def _check_parent_shapes(old_model: dict[str, Any], current: dict[str, Any]) -> None:
    """Refuse anything except the precise slot-table/scalar-readout parent."""
    for key in _EXPANDED_KEYS:
        want = (PARENT_CLASSES, current[key].shape[1])
        value = old_model[key]
        if not isinstance(value, Tensor) or tuple(value.shape) != want:
            got = (
                tuple(value.shape)
                if isinstance(value, Tensor)
                else type(value).__name__
            )
            raise ValueError(f"parent {key} must have shape {want}, got {got}")

    hidden = current[_READOUT_KEYS[0]].shape[1]
    expected = {_READOUT_KEYS[0]: (1, hidden), _READOUT_KEYS[1]: (1,)}
    wrong = [
        key
        for key in _READOUT_KEYS
        if not isinstance(old_model[key], Tensor)
        or tuple(old_model[key].shape) != expected[key]
    ]
    if wrong:
        shapes = []
        for key in wrong:
            value = old_model[key]
            shape = (
                tuple(value.shape)
                if isinstance(value, Tensor)
                else type(value).__name__
            )
            shapes.append(f"{key}: {shape}")
        raise ValueError(
            f"the parent critic readout must be one row {expected}; got "
            f"{', '.join(shapes)}"
        )


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


def _remap_adam(
    saved: Any, old_model: dict[str, Any], names: list[str]
) -> dict[str, Any]:
    """Remap both transforms' Adam moments onto this build's parameter list.

    Joint-table moments replicate by the same parent-row map as their weights.
    A scalar readout's first moment maps to ``(+m, -m, 0)`` and its second to
    ``(v, v, 0)``; the step is unchanged. The zero row therefore begins with
    no inherited optimizer direction or variance.
    """
    if not isinstance(saved, dict) or not isinstance(saved.get("state"), dict):
        raise ValueError("checkpoint must contain an Adam optimizer state dict")
    groups = saved.get("param_groups")
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
    if len(saved_ids) != len(names) or len(set(saved_ids)) != len(saved_ids):
        raise ValueError(
            "optimizer parameter count does not match the checkpoint's parameters: "
            f"{len(saved_ids)} distinct ids for {len(names)} parameters"
        )

    state: dict[int, dict] = {}
    for index, (name, param_id) in enumerate(zip(names, saved_ids)):
        entry = saved["state"].get(param_id)
        if entry is None:
            if name in _READOUT_KEYS:
                raise ValueError(
                    f"the optimizer has no Adam state for {name}, the readout "
                    "this transform rescales"
                )
            # Adam state is sparse; absence is the parent's record that this
            # parameter has never stepped and is carried as absence.
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"optimizer state for {name} is not a state dict")
        missing = [field for field in _ADAM_FIELDS if field not in entry]
        if missing:
            raise ValueError(
                f"optimizer state for {name} is missing: {', '.join(missing)}"
            )
        want = tuple(old_model[name].shape)
        for field in ("exp_avg", "exp_avg_sq"):
            moment = entry[field]
            if not isinstance(moment, Tensor) or tuple(moment.shape) != want:
                raise ValueError(
                    f"optimizer {field} for {name} does not have the parameter's "
                    f"shape {want}"
                )
        entry = copy.deepcopy(entry)
        if name in _EXPANDED_KEYS:
            for field in ("exp_avg", "exp_avg_sq"):
                entry[field] = _expand(entry[field])
        elif name in _READOUT_KEYS:
            extra = set(entry) - set(_ADAM_FIELDS)
            if extra:
                raise ValueError(
                    f"optimizer state for {name} carries {sorted(extra)}, which "
                    "cannot be mapped onto three rows"
                )
            entry["exp_avg"] = _critic_rows(entry["exp_avg"])
            entry["exp_avg_sq"] = torch.cat(
                [
                    entry["exp_avg_sq"],
                    entry["exp_avg_sq"],
                    torch.zeros_like(entry["exp_avg_sq"]),
                ],
                dim=0,
            )
        state[index] = entry
    if not state:
        raise ValueError("the parent's optimizer stepped no parameter at all")
    group = {**copy.deepcopy(groups[0]), "params": list(range(len(names)))}
    return {"state": state, "param_groups": [group]}


def _joint_probe_positions() -> list:
    """The joint probe: non-terminal positions from seeded
    uniformly-random legal playouts, prefix lengths spread evenly over
    ``JOINT_PROBE_PLIES``. A terminal playout is retried under a derived seed."""
    low, high = JOINT_PROBE_PLIES
    positions = []
    for index in range(JOINT_PROBE_POSITIONS):
        plies = low + round(index * (high - low) / (JOINT_PROBE_POSITIONS - 1))
        for attempt in range(100):
            rng = random.Random(
                JOINT_PROBE_SEED + 1_000_003 * index + 1_009 * attempt
            )
            position = hexo_py.Position()
            for _ in range(plies):
                position.advance(*rng.choice(position.legal_moves()))
            if not position.is_terminal:
                positions.append(position)
                break
        else:
            raise RuntimeError(
                f"no non-terminal {plies}-ply probe playout in 100 seeds"
            )
    return positions


def _probe_prefixes() -> list[list[tuple[int, int]]]:
    """The critic graft's deterministic nonterminal move-prefix probe."""
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


def _q_spread(policy: Tensor, q: Tensor, offsets: Tensor, top_k: int) -> float:
    """Median over positions of σ(Q) across the policy's top-``top_k``
    legal cells — the spread the improvement operator actually exponentiates."""
    spreads = []
    for lo, hi in zip(offsets[:-1].tolist(), offsets[1:].tolist()):
        top = policy[lo:hi].topk(min(top_k, hi - lo)).indices
        spreads.append(float(q[lo:hi].index_select(0, top).std()))
    return statistics.median(spreads)


def _measure_joint(
    model: MantisNet, parent_model: dict[str, Any], tau: float, lam: float
) -> dict:
    """Run the joint arm's exact spec-decode and folded-arithmetic checks."""
    batch = collate([from_position(p) for p in _joint_probe_positions()])
    slot_class = torch.from_numpy(_PARENT_ROW).index_select(0, batch.dec_class)
    # This intermediate state exists only for the independent joint detector:
    # expanded tables, but the parent's scalar readout. It isolates the class
    # transform from the disjoint critic transform without creating an artifact.
    joint_state = dict(parent_model)
    for key in _EXPANDED_KEYS:
        joint_state[key] = _expand(parent_model[key])

    with torch.no_grad():
        _s, w, g = model.trunk(batch)
        policy_new, score_new, q_new = model.cell_heads(
            w, g, batch, _GRAFT_FLOOR
        )
        policy_parent = _spec_scores(
            w, g, batch, parent_model, _POLICY_HEAD, slot_class
        )
        q_parent_score = _spec_scores(
            w, g, batch, parent_model, _CRITIC_HEAD, slot_class
        )
        q_parent = torch.tanh(q_parent_score)
        policy_joint = _spec_scores(
            w, g, batch, joint_state, _POLICY_HEAD, batch.dec_class
        )
        q_joint_score = _spec_scores(
            w, g, batch, joint_state, _CRITIC_HEAD, batch.dec_class
        )
        exact = torch.equal(policy_joint, policy_parent) and torch.equal(
            q_joint_score, q_parent_score
        )

    offsets = batch.legal_offsets
    kl = _segment_kl(
        improved_policy(
            policy_new, score_new, q_new, offsets, tau, lam
        ).probs,
        improved_policy(
            policy_parent, q_parent, q_parent, offsets, tau, lam
        ).probs,
        offsets,
    )
    q_delta = (q_new - q_parent).abs()
    policy_delta = (policy_new - policy_parent).abs()
    return {
        "joint_probe": {
            "seed": JOINT_PROBE_SEED,
            "positions": batch.n_pos,
            "plies": list(JOINT_PROBE_PLIES),
            "legal_cells": batch.n_cells,
            "decoder_entries": int(batch.dec_class.shape[0]),
            "classes_exercised": int(batch.dec_class.unique().numel()),
            "top_k": JOINT_PROBE_TOP_K,
        },
        "spec_decode_bitwise_equal": bool(exact),
        "q_max_abs_delta": float(q_delta.max()),
        "q_mean_abs_delta": float(q_delta.mean()),
        "policy_max_abs_delta": float(policy_delta.max()),
        "policy_mean_abs_delta": float(policy_delta.mean()),
        "q_spread_median_parent": _q_spread(
            policy_parent, q_parent, offsets, JOINT_PROBE_TOP_K
        ),
        "q_spread_median_new": _q_spread(
            policy_new, q_new, offsets, JOINT_PROBE_TOP_K
        ),
        "kl_mean": float(kl.mean()),
        "kl_max": float(kl.max()),
    }


def _parent_model(old_model: dict[str, Any], cfg: MantisConfig) -> MantisNet:
    """Strict-load the parent slot-table/scalar-readout architecture."""
    parent = MantisNet(cfg)
    parent.e_pw = nn.Embedding(PARENT_CLASSES, cfg.h)
    parent.e_qw = nn.Embedding(PARENT_CLASSES, cfg.h)
    parent.mlp_q.out = nn.Linear(parent.mlp_q.out.in_features, 1)
    parent.load_state_dict(old_model)
    parent.eval()
    return parent


def _measure(
    model: MantisNet, parent: MantisNet, tau: float, lam: float
) -> dict:
    """Run the categorical arm's complete parent-vs-model probe."""
    prefixes = _probe_prefixes()
    batch = collate_prefixes(prefixes, [len(moves) for moves in prefixes])
    parent_state = parent.state_dict()
    slot_class = torch.from_numpy(_PARENT_ROW).index_select(0, batch.dec_class)

    with torch.no_grad():
        _s, w, g = model.trunk(batch)
        policy_new, score_new, q_new = model.cell_heads(
            w, g, batch, _GRAFT_FLOOR
        )
        # A separate parent trunk forward prevents a mangled shared tensor from
        # disappearing behind shared activations. Its old decoder is evaluated
        # by the independent MODEL_SPEC §6 transcription because this build's
        # folded coefficient layout is necessarily the 93-class child layout.
        _s, parent_w, parent_g = parent.trunk(batch)
        policy_parent = _spec_scores(
            parent_w, parent_g, batch, parent_state, _POLICY_HEAD, slot_class
        )
        q_parent = torch.tanh(
            _spec_scores(
                parent_w, parent_g, batch, parent_state, _CRITIC_HEAD, slot_class
            )
        )

    offsets = batch.legal_offsets
    delta = (q_new - q_parent).abs()
    pi_new = improved_policy(
        policy_new.double(), score_new.double(), q_new.double(), offsets, tau, lam
    )
    pi_parent = improved_policy(
        policy_parent.double(), q_parent.double(), q_parent.double(), offsets,
        tau, lam
    )
    probs = pi_new.probs
    terms = torch.where(
        probs > 0,
        probs * (probs.log() - pi_parent.probs.log()),
        probs.new_zeros(()),
    )
    kl = segment_sum(terms, segment_ids(offsets), batch.n_pos)

    return {
        "probe_seed": PROBE_SEED,
        "probe_positions": int(batch.n_pos),
        "probe_legal_cells": int(offsets[-1]),
        "probe_plies": [len(moves) for moves in prefixes],
        "max_abs_dq": float(delta.max()),
        "mean_abs_dq": float(delta.mean()),
        "top_k": PROBE_TOP_K,
        "top_k_q_std_median_parent": _q_spread(
            policy_parent, q_parent, offsets, PROBE_TOP_K
        ),
        "top_k_q_std_median_grafted": _q_spread(
            policy_new, q_new, offsets, PROBE_TOP_K
        ),
        "mean_improved_kl": float(kl.mean()),
        "max_improved_kl": float(kl.max()),
    }


def graft(
    old_path: Path | str,
    new_path: Path | str,
    tau: float,
    lam: float,
    manifest_path: Path | str | None = None,
) -> dict:
    """Apply both transforms, run both batteries, then write one checkpoint."""
    old_path, new_path = Path(old_path), Path(new_path)
    manifest_path = (
        Path(manifest_path)
        if manifest_path is not None
        else new_path.with_suffix(".json")
    )
    resolved = [old_path.resolve(), new_path.resolve(), manifest_path.resolve()]
    if len(set(resolved)) != len(resolved):
        raise ValueError(
            "OLD.pt, NEW.pt, and the evidence sidecar must be different paths"
        )

    checkpoint = torch.load(old_path, map_location="cpu", weights_only=False)
    parent_versions = _check_checkpoint(checkpoint)
    parent_model = checkpoint["model"]

    cfg = MantisConfig()
    model = MantisNet(cfg)
    current = model.state_dict()
    mismatches = _architecture_mismatches(parent_model, current)
    if set(mismatches) != _TRANSFORMED_KEYS:
        raise ValueError(
            "parent model must differ from this build only at the two decoder "
            "class tables and scalar critic readout; mismatched keys: "
            f"{', '.join(mismatches) if mismatches else '<none>'}"
        )
    _check_parent_shapes(parent_model, current)

    names = [name for name, _parameter in model.named_parameters()]
    shared_names = [name for name in names if name not in _TRANSFORMED_KEYS]
    # The parent's Adam ids are positions in its own parameter order, and this is
    # the only record of that order, so it is pinned rather than assumed.
    if list(parent_model) != names:
        raise ValueError(
            "parent state dict is not in this build's parameter order, so its "
            "optimizer ids cannot be named"
        )

    converted_model = _converted_state(parent_model)
    converted = {
        "model": {name: converted_model[name] for name in names},
        "optimizer": _remap_adam(checkpoint["optimizer"], parent_model, names),
        "iteration": checkpoint["iteration"],
        "rng_state": copy.deepcopy(checkpoint["rng_state"]),
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
        if not torch.equal(
            grafted_model[name][cls], source_model[name][_PARENT_ROW[cls]]
        )
    ]
    if misplaced:
        raise ValueError(
            f"{len(misplaced)} expanded rows are not the parent row they replace: "
            f"{', '.join(misplaced[:8])}; nothing written"
        )

    joint_measurement = _measure_joint(model, parent_model, tau, lam)
    critic_measurement = _measure(
        model, _parent_model(parent_model, cfg), tau, lam
    )
    manifest = {
        "arm": ARM,
        "source": str(old_path),
        "source_iteration": checkpoint["iteration"],
        "parent_versions": parent_versions,
        "versions": converted["versions"],
        "transform": TRANSFORM,
        "expanded_keys": list(_EXPANDED_KEYS),
        "readout_keys": list(_READOUT_KEYS),
        "classes": DEC_CLASSES,
        "parent_classes": PARENT_CLASSES,
        "class_to_parent_row": _PARENT_ROW.tolist(),
        "tau": tau,
        "lam": lam,
        **joint_measurement,
        "mass_floor": _GRAFT_FLOOR,
        **critic_measurement,
    }
    joint_holds = (
        manifest["spec_decode_bitwise_equal"]
        and manifest["q_max_abs_delta"] <= Q_TOLERANCE
        and manifest["policy_max_abs_delta"] <= POLICY_TOLERANCE
        and manifest["kl_max"] <= KL_TOLERANCE
    )
    critic_holds = (
        manifest["max_abs_dq"] <= MAX_ABS_DQ
        and abs(manifest["mean_improved_kl"]) <= MAX_MEAN_KL
    )
    manifest["preservation"] = {
        "property": PRESERVATION,
        "shared_tensors_unchanged": len(shared_names),
        "shared_tensor_sha256": _shared_digest(converted["model"], shared_names),
        "expanded_rows_checked": len(_EXPANDED_KEYS) * DEC_CLASSES,
        "spec_decode_bitwise_equal": manifest["spec_decode_bitwise_equal"],
        "joint_q_max_abs_delta": manifest["q_max_abs_delta"],
        "joint_q_max_abs_delta_tolerance": Q_TOLERANCE,
        "joint_policy_max_abs_delta": manifest["policy_max_abs_delta"],
        "joint_policy_max_abs_delta_tolerance": POLICY_TOLERANCE,
        "joint_kl_max": manifest["kl_max"],
        "joint_kl_max_tolerance": KL_TOLERANCE,
        "max_abs_dq": manifest["max_abs_dq"],
        "max_abs_dq_tolerance": MAX_ABS_DQ,
        "mean_improved_kl": manifest["mean_improved_kl"],
        "mean_improved_kl_tolerance": MAX_MEAN_KL,
        "holds": joint_holds and critic_holds,
    }
    if not joint_holds:
        raise ValueError(
            "the joint decoder checks failed: spec decode bitwise equal = "
            f"{manifest['spec_decode_bitwise_equal']}, max |Q_new - Q_parent| = "
            f"{manifest['q_max_abs_delta']:.3e} (bound {Q_TOLERANCE:.0e}), max "
            f"|logit_new - logit_parent| = {manifest['policy_max_abs_delta']:.3e} "
            f"(bound {POLICY_TOLERANCE:.0e}), max KL = {manifest['kl_max']:.3e} "
            f"(bound {KL_TOLERANCE:.0e}); nothing written"
        )
    if not critic_holds:
        raise ValueError(
            "the trinomial graft is not function preserving on the probe set: max "
            f"|Q_new - Q_parent| = {manifest['max_abs_dq']:.3e} (tolerance "
            f"{MAX_ABS_DQ:.0e}), mean KL(pi'_new || pi'_parent) = "
            f"{manifest['mean_improved_kl']:.3e} (tolerance {MAX_MEAN_KL:.0e}); "
            "nothing written"
        )

    # The evidence lands before the checkpoint it describes, so no grafted
    # checkpoint can exist without its measurements.
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    torch.save(converted, new_path)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="apply the joint-class decoder and trinomial critic graft together"
    )
    parser.add_argument("old", type=Path, metavar="OLD.pt")
    parser.add_argument("new", type=Path, metavar="NEW.pt")
    # The operating point is required: π′ is measured at it, and a manifest that
    # did not name it would not say what it measured.
    parser.add_argument("--tau", type=float, required=True, help="reverse-KL weight")
    parser.add_argument("--lam", type=float, required=True, help="entropy weight")
    args = parser.parse_args(argv)
    manifest = graft(args.old, args.new, args.tau, args.lam)
    print(json.dumps(manifest["preservation"], indent=2), flush=True)


if __name__ == "__main__":
    main()
