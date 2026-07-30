"""Graft this build's dueling critic onto a trained scalar-critic checkpoint.

The parent is a checkpoint whose action-value head was the single tanh-bounded
readout: ``Q_parent = tanh(A)``. This build composes the same decoder score
``A`` with a per-position baseline and centers it on the raw policy, so
``Q_new = tanh(v + A - E_{π_θ}[A])``. The graft therefore *adds keys only* —
``A`` is the parent's critic decoder verbatim — and the baseline arrives at this
build's initialization, where its output layer is zero and ``v(s) = 0``.

That is not function preserving. ``E_{π_θ}[A]`` is subtracted at the graft, so
the level the parent carried inside ``A`` is gone and ``v`` has to learn it
back. What is preserved exactly is the *order*: both ``tanh(A)`` and
``tanh(A - c)`` are monotone in ``A`` within a position, so every position's
ranking of its legal cells survives. The manifest measures that ordering, and
this module writes no checkpoint if one discordant pair exists or if the
ordering came out constant — a transform that discarded the trained critic
would reverse nothing and preserve nothing, and only the second half catches
it. It also reports the size of the level that was removed and how far the
action values and the improved policy moved, which is the point of the arm and
not a pass/fail claim.

The parent's action values are not re-derived through the parent build's code,
which this build does not contain. A forward hook captures what the grafted
model applies its critic readout to, and the *parent's* readout tensors are
applied to that activation. That is the parent's own raw score only if the
forward which produced the activation ran the parent's parameters and the head
composed appendix B on it, so the graft establishes both rather than assuming
them: every shared parameter bitwise, before the probe runs, and the composition
re-derived from the formula with ``v(s) = 0``, refused alongside the ordering.
Those two are the ordering claim's premise and they carry the detection —
``tanh`` of a per-position constant shift is monotone whatever the shift is, so
the ordering by itself would not notice a wrong centering weight, a level
broadcast to the wrong position, or a baseline that is not zero.

Run from ``python/mantisnet``:

    python -m mantisnet.klent.graft OLD.pt NEW.pt --tau T --lam L \
        --manifest OUT.json

``--tau`` and ``--lam`` have no defaults: the manifest's π′ measurements are
meaningless without the operating point they were taken at. On any refusal — a
parent this transform does not own, a premise it cannot establish, a violated
ordering — neither the checkpoint nor the manifest is written.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import statistics
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from ..builder import collate_prefixes
from ..model import MantisConfig, MantisNet
from ..segments import segment_ids, segment_log_softmax, segment_sum
from .improve import improved_policy
from .run import _versions

ARM = "dueling-critic"
TRANSFORM = (
    "add the v(s) baseline readout at fresh init (zero output layer); "
    "carry the parent critic decoder verbatim as A; "
    "Q_new = tanh(A - E_pi_theta[A]) against Q_parent = tanh(A)"
)

# The parameters this graft adds. Every other parameter is carried verbatim.
_ADDED_KEYS = (
    "mlp_qbase.0.weight",
    "mlp_qbase.0.bias",
    "mlp_qbase.2.weight",
    "mlp_qbase.2.bias",
)
# The parent's one-wide critic readout, applied directly by the reference path.
_READOUT_WEIGHT, _READOUT_BIAS = "mlp_q.out.weight", "mlp_q.out.bias"
_ADAM_FIELDS = ("step", "exp_avg", "exp_avg_sq")

# The added parameters' construction. Recorded in the manifest, so the graft is
# reproducible from the parent plus this constant.
INIT_SEED = 20260729
# The probe playouts.
PROBE_SEED = 6172026
PROBE_POSITIONS = 64
PROBE_PLIES = (20, 60)
# π_θ's top cells, over which the action values' spread is reported: the band
# the improvement operator actually chooses between.
PROBE_TOP_K = 16
# How far the head's action values may sit from the composition re-derived here.
# Both sides combine the same terms with the same ragged helpers, so only the
# final combination can reassociate: this is fp32 round-off, not a modelling
# tolerance, and a violation is a different transform rather than a worse one.
COMPOSITION_TOL = 1e-6


def _probe_lines() -> list[list[tuple[int, int]]]:
    """Deterministic nonterminal random playouts, lengths spread over
    ``PROBE_PLIES``.

    One stream from ``PROBE_SEED`` drives every draw, so the probe set is a
    function of that constant alone. A playout that ends the game is redrawn:
    terminal positions are not model inputs.
    """
    import hexo_py

    low, high = PROBE_PLIES
    rng = random.Random(PROBE_SEED)
    lines: list[list[tuple[int, int]]] = []
    for i in range(PROBE_POSITIONS):
        plies = low + round(i * (high - low) / (PROBE_POSITIONS - 1))
        for _attempt in range(100):
            position = hexo_py.Position()
            moves: list[tuple[int, int]] = []
            for _ in range(plies):
                move = rng.choice(position.legal_moves())
                position.advance(*move)
                moves.append(move)
                if position.is_terminal:
                    break
            if not position.is_terminal:
                lines.append(moves)
                break
        else:
            raise RuntimeError(f"no nonterminal {plies}-ply playout in 100 draws")
    return lines


def _architecture_mismatches(parent: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """Every key at which ``parent`` is not this build's architecture."""
    mismatches = sorted(set(parent) ^ set(current))
    for key in sorted(set(parent) & set(current)):
        old_value, new_value = parent[key], current[key]
        if not isinstance(old_value, Tensor) or not isinstance(new_value, Tensor):
            if type(old_value) is not type(new_value):
                mismatches.append(key)
        elif old_value.shape != new_value.shape:
            mismatches.append(key)
    return mismatches


def _zero_step(step: Any) -> Any:
    """Adam's step count, zeroed in whatever type the parent stored it as: a
    0-d tensor for a capturable or fused optimizer, otherwise an int."""
    if isinstance(step, Tensor):
        return torch.zeros_like(step)
    if isinstance(step, (int, float)):
        return type(step)(0)
    raise ValueError(f"cannot zero an Adam step of type {type(step).__name__}")


def _fresh_model() -> MantisNet:
    """This build at its initialization, seeded by ``INIT_SEED``.

    ``fork_rng`` keeps the graft's determinism independent of the caller's RNG
    in both directions.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(INIT_SEED)
        return MantisNet(MantisConfig())


def _check_parent(old_path: Path, checkpoint: Any, current: dict[str, Any], hidden: int) -> dict:
    """Refuse anything that is not a parent this transform owns."""
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError(f"{old_path}: checkpoint must contain a model state dict")
    missing = [f for f in ("optimizer", "iteration", "rng_state", "versions") if f not in checkpoint]
    if missing:
        raise ValueError(f"{old_path}: checkpoint is missing {', '.join(missing)}")
    if checkpoint["versions"] != _versions():
        raise ValueError(
            f"{old_path}: versions {checkpoint['versions']} != this build {_versions()}; "
            "the graft carries versions through verbatim, so the result would not load "
            "and the probe measurements would not describe this build"
        )

    parent = checkpoint["model"]
    # Checked before the architecture, and against a shape written out here
    # rather than read off ``current``, because a wider critic readout is the
    # near miss worth naming: it is a different critic, whose action values the
    # reference path — which applies these two tensors directly — cannot
    # reproduce, and reporting it as one more mismatched key would bury that.
    expected = {_READOUT_WEIGHT: (1, hidden), _READOUT_BIAS: (1,)}
    wrong = [
        f"{key}: {tuple(parent[key].shape) if isinstance(parent.get(key), Tensor) else 'absent'}"
        for key, shape in expected.items()
        if not isinstance(parent.get(key), Tensor) or tuple(parent[key].shape) != shape
    ]
    if wrong:
        raise ValueError(
            f"{old_path}: this graft needs a scalar critic parent, whose readout is one "
            f"row — expected {expected}, got {'; '.join(wrong)}"
        )

    mismatches = _architecture_mismatches(parent, current)
    if set(mismatches) != set(_ADDED_KEYS):
        unexpected = sorted(set(mismatches) - set(_ADDED_KEYS))
        absent = sorted(set(_ADDED_KEYS) - set(mismatches))
        raise ValueError(
            f"{old_path}: the parent must differ from this build's architecture only at "
            f"{', '.join(_ADDED_KEYS)}; also mismatched: "
            f"{', '.join(unexpected) if unexpected else '<none>'}; already matching: "
            f"{', '.join(absent) if absent else '<none>'}"
        )
    present = sorted(key for key in _ADDED_KEYS if key in parent)
    if present:
        raise ValueError(
            f"{old_path}: this graft adds parameters and overwrites none, but the parent "
            f"already carries {', '.join(present)}"
        )
    # The Adam remap resolves the parent's saved parameter ids through this
    # order, and no file content could reveal a different one, so it is a
    # refusal rather than an assumption.
    expected_order = [name for name in current if name not in set(_ADDED_KEYS)]
    if list(parent) != expected_order:
        raise ValueError(
            f"{old_path}: the parent's parameters are not this build's registration order "
            "with the baseline readout removed, so its saved optimizer positions cannot "
            "be resolved by name"
        )
    return parent


def _remap_adam(old_path: Path, saved: Any, parent_names: list[str], new_state: dict[str, Tensor]):
    """The parent's Adam state under this build's parameter order.

    Remapping is by NAME: the parent's group holds the packed positions of its
    own ``named_parameters()``, and inserting the baseline readout shifts every
    position after it. Shared parameters carry their moments and step
    unchanged — this arm rescales no readout row, so there is no first-moment
    factor to apply and the first post-graft step of every carried parameter is
    exactly the step the parent would have taken. The added parameters start at
    zero moments and a zeroed step of the parent's own step type, which is what
    a parameter Adam has never seen must look like.
    """
    if not isinstance(saved, dict) or not isinstance(saved.get("state"), dict):
        raise ValueError(f"{old_path}: checkpoint must contain an Adam optimizer state dict")
    groups = saved.get("param_groups")
    if (
        not isinstance(groups, list)
        or len(groups) != 1
        or not isinstance(groups[0], dict)
        or not isinstance(groups[0].get("params"), list)
    ):
        raise ValueError(
            f"{old_path}: expected exactly one Adam param group over model.parameters(); "
            f"got {groups if not isinstance(groups, list) else f'{len(groups)} groups'}"
        )
    if groups[0]["params"] != list(range(len(parent_names))):
        raise ValueError(
            f"{old_path}: the param group must hold the packed positions of the parent's "
            f"{len(parent_names)} parameters — 0..{len(parent_names) - 1} in order — and "
            f"holds {len(groups[0]['params'])} other ids"
        )

    positions = {name: i for i, name in enumerate(parent_names)}
    added = set(_ADDED_KEYS)
    carried: dict[int, dict] = {}
    for new_id, name in enumerate(new_state):
        if name in added:
            continue
        entry = saved["state"].get(positions[name])
        if entry is None:
            # Adam's state dict is sparse: a parameter it never stepped has no
            # entry at all, which is how the state-value head KLENT does not
            # train appears in every checkpoint this repo writes. Absence is
            # the parent's own statement that the parameter is unstepped, so it
            # is carried as absence rather than filled in.
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"{old_path}: optimizer state for {name} is not a state dict")
        if set(entry) != set(_ADAM_FIELDS):
            raise ValueError(
                f"{old_path}: optimizer state for {name} has fields {sorted(entry)}, "
                f"expected exactly {sorted(_ADAM_FIELDS)}"
            )
        for field in ("exp_avg", "exp_avg_sq"):
            moment = entry[field]
            if not isinstance(moment, Tensor) or moment.shape != new_state[name].shape:
                raise ValueError(
                    f"{old_path}: optimizer {field} for {name} does not have the "
                    f"parameter's shape {tuple(new_state[name].shape)}"
                )
        carried[new_id] = copy.deepcopy(entry)

    step = next(iter(carried.values()))["step"]
    for new_id, name in enumerate(new_state):
        if name not in added:
            continue
        carried[new_id] = {
            "step": _zero_step(step),
            "exp_avg": new_state[name].new_zeros(new_state[name].shape),
            "exp_avg_sq": new_state[name].new_zeros(new_state[name].shape),
        }
    return {
        "state": {new_id: carried[new_id] for new_id in sorted(carried)},
        "param_groups": [{**copy.deepcopy(groups[0]), "params": list(range(len(new_state)))}],
    }


@torch.no_grad()
def _measure(model: MantisNet, parent: dict[str, Any], tau: float, lam: float) -> dict:
    """The manifest's measurement half, on the deterministic probe set.

    The parameter carry is the one thing checked here rather than reported for
    ``graft`` to enforce: a probe that ran some other model's parameters does not
    produce a number that is wrong by a little, it produces a number about a
    different model, so nothing below it would mean anything.
    """
    # The premise's first half: the forward ran the parent's parameters. Checked
    # bitwise rather than assumed — the reference below applies the parent's
    # readout to an activation the *grafted* model produced, which is the
    # parent's activation only if the trunk and decoder that built it are.
    state = model.state_dict()
    moved = [name for name, value in parent.items() if not torch.equal(state[name], value)]
    if moved:
        shown = ", ".join(moved[:8]) + (" ..." if len(moved) > 8 else "")
        raise ValueError(
            f"the graft carried {len(moved)} of the parent's {len(parent)} parameters into "
            f"a different tensor, so the probe would not measure the parent: {shown}; "
            "no checkpoint written"
        )

    lines = _probe_lines()
    batch = collate_prefixes(lines, [len(line) for line in lines])
    offsets = batch.legal_offsets
    positions = offsets.shape[0] - 1
    seg = segment_ids(offsets)

    # The reference: capture the activation the grafted model feeds its critic
    # readout, then apply the *parent's* readout tensors to it. Those tensors and
    # everything upstream of the capture are the parent's, so this is the
    # parent's own raw score.
    captured: list[Tensor] = []
    handle = model.mlp_q.out.register_forward_hook(
        lambda _module, inputs, _output: captured.append(inputs[0])
    )
    try:
        _s, w, g = model.trunk(batch)
        logits, q_new = model.cell_heads(w, g, batch)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"the critic readout ran {len(captured)} times, expected once")
    advantage = F.linear(captured[0], parent[_READOUT_WEIGHT], parent[_READOUT_BIAS]).squeeze(-1)
    q_parent = torch.tanh(advantage)

    pi = segment_log_softmax(logits, offsets, seg).exp()
    level = segment_sum(pi * advantage, offsets)

    # The premise's second half: the head composed appendix B on that score.
    # Re-derived from the formula instead of read back out of the head, because
    # the ordering property holds for any per-position shift at all and so states
    # nothing about which one was applied. A centering weight that is not the raw
    # policy, a level broadcast to the wrong position, and a baseline away from
    # zero all move this residual and none of them moves the ordering.
    baseline_abs_max = float(model.mlp_qbase(g).abs().max())
    residual = float((q_new - torch.tanh(advantage - level.index_select(0, seg))).abs().max())

    improved_new = improved_policy(logits, q_new, offsets, tau, lam).probs.double()
    improved_parent = improved_policy(logits, q_parent, offsets, tau, lam).probs.double()
    terms = improved_new * (improved_new.log() - improved_parent.log())
    kl = segment_sum(torch.where(improved_new > 0, terms, terms.new_zeros(())), offsets)

    bounds = offsets.tolist()
    sigma_parent, sigma_new = [], []
    comparable = discordant = collapsed = 0
    for i in range(positions):
        low, high = bounds[i], bounds[i + 1]
        before, after = q_parent[low:high], q_new[low:high]
        # Only pairs the parent strictly orders can disagree: where the
        # parent's tanh saturated it stated no order at all, and the graft's
        # finer ordering there refines rather than contradicts it. Pairs the
        # graft ties instead are counted separately — that is the direction
        # this arm's own saturation can lose, and it is reported, not enforced.
        strict = before[:, None] > before[None, :]
        comparable += int(strict.sum())
        discordant += int((strict & (after[:, None] < after[None, :])).sum())
        collapsed += int((strict & (after[:, None] == after[None, :])).sum())
        top = logits[low:high].topk(min(PROBE_TOP_K, high - low)).indices
        sigma_parent.append(float(before[top].std()))
        sigma_new.append(float(after[top].std()))
    if comparable == 0:
        raise ValueError(
            "the parent's action values are constant within every probe position: the "
            "ordering property is vacuous, so the graft has nothing to establish"
        )

    difference = (q_new - q_parent).abs()
    return {
        "tau": tau,
        "lam": lam,
        "transform": TRANSFORM,
        "init_seed": INIT_SEED,
        "probe": {
            "seed": PROBE_SEED,
            "positions": positions,
            "legal_cells": int(offsets[-1]),
            "prefix_plies": [min(len(line) for line in lines), max(len(line) for line in lines)],
            "top_k": PROBE_TOP_K,
        },
        "premise": {
            "claim": (
                "the action values compared below are the parent's: every shared parameter "
                "the probe forward ran is bitwise the parent's, and the head is appendix "
                "B's composition of that forward's advantage, with v(s) = 0"
            ),
            "tolerance": (
                f"exact on the parameters and on v(s); {COMPOSITION_TOL:g} absolute on Q, "
                "which is fp32 round-off between the same terms"
            ),
            "parameters_carried_bitwise": len(parent),
            "baseline_abs_max": baseline_abs_max,
            "q_abs_diff_from_the_formula_max": residual,
            "holds": baseline_abs_max == 0.0 and residual <= COMPOSITION_TOL,
        },
        "preservation": {
            "property": (
                "per-position ordering of Q is the parent's: of the pairs of legal cells "
                "the parent strictly orders, none is strictly ordered the other way, and "
                "the ordering is not wholly collapsed"
            ),
            "tolerance": (
                "exact: zero discordant pairs, with at least one pair still strictly ordered"
            ),
            "comparable_pairs": comparable,
            "discordant_pairs": discordant,
            "concordant_pairs": comparable - discordant - collapsed,
            "rank_agreement": 1.0 - discordant / comparable,
            "holds": discordant == 0 and comparable - discordant - collapsed > 0,
        },
        "measured": {
            "q_abs_diff_max": float(difference.max()),
            "q_abs_diff_mean": float(difference.mean()),
            "removed_level_median": statistics.median(level.tolist()),
            "top_k_sigma_q_parent_median": statistics.median(sigma_parent),
            "top_k_sigma_q_new_median": statistics.median(sigma_new),
            "improved_policy_kl_mean": float(kl.mean()),
            "improved_policy_kl_max": float(kl.max()),
            "order_collapsed_pairs": collapsed,
        },
    }


def graft(old_path, new_path, tau: float, lam: float, manifest_path) -> dict:
    """Convert ``old_path``, measure the result, and write both artifacts.

    Nothing is written unless the measurements' premise holds, the ordering
    property holds exactly, and every measurement is finite; the manifest is
    serialized before the checkpoint so a rejected graft leaves no
    half-converted run behind.
    """
    old_path, new_path, manifest_path = Path(old_path), Path(new_path), Path(manifest_path)
    if old_path.resolve() == new_path.resolve():
        raise ValueError(f"OLD.pt and NEW.pt must be different paths, both are {old_path}")

    checkpoint = torch.load(old_path, map_location="cpu", weights_only=False)
    model = _fresh_model()
    fresh = model.state_dict()
    # The whole remap reads parameter order off the state dict, which holds
    # exactly the parameters only while the model registers no persistent
    # buffer.
    if list(fresh) != [name for name, _p in model.named_parameters()]:
        raise RuntimeError(
            "MantisNet's state dict is no longer its parameters in named_parameters() "
            "order; the optimizer remap cannot key on it"
        )
    parent = _check_parent(old_path, checkpoint, fresh, model.cfg.policy_hidden)

    added = set(_ADDED_KEYS)
    converted = {
        "model": {
            name: (fresh[name] if name in added else parent[name]).detach().clone()
            for name in fresh
        },
        "optimizer": _remap_adam(old_path, checkpoint["optimizer"], list(parent), fresh),
        "iteration": checkpoint["iteration"],
        "rng_state": copy.deepcopy(checkpoint["rng_state"]),
        "versions": copy.deepcopy(checkpoint["versions"]),
    }

    model.load_state_dict(converted["model"])
    model.eval()
    manifest = {
        "arm": ARM,
        "source": str(old_path),
        "source_iteration": checkpoint["iteration"],
        "versions": checkpoint["versions"],
        **_measure(model, parent, tau, lam),
    }
    kept = manifest["preservation"]
    if not kept["holds"]:
        raise ValueError(
            "the graft did not preserve the parent's ordering of Q: "
            f"{kept['discordant_pairs']} discordant and {kept['concordant_pairs']} still "
            f"strictly ordered, of {kept['comparable_pairs']} comparable pairs "
            f"(rank agreement {kept['rank_agreement']:.6f}); no checkpoint written"
        )
    premise = manifest["premise"]
    if not premise["holds"]:
        raise ValueError(
            "the grafted head is not appendix B's composition on the parent's advantage, so "
            "the ordering it preserved is a property of the head's own arithmetic and not a "
            f"statement about the parent: max |v(s)| = {premise['baseline_abs_max']:g}, which "
            "this build's initialization makes exactly zero, and max |Q - tanh(A - "
            f"E_pi_theta[A])| = {premise['q_abs_diff_from_the_formula_max']:g} against a "
            f"tolerance of {COMPOSITION_TOL:g}; no checkpoint written"
        )
    payload = json.dumps(manifest, indent=2, allow_nan=False) + "\n"

    torch.save(converted, new_path)
    manifest_path.write_text(payload, encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="graft this build's dueling critic onto a scalar-critic checkpoint"
    )
    parser.add_argument("old", type=Path, metavar="OLD.pt")
    parser.add_argument("new", type=Path, metavar="NEW.pt")
    parser.add_argument(
        "--tau", type=float, required=True, help="the operating point pi' is measured at"
    )
    parser.add_argument(
        "--lam", type=float, required=True, help="the operating point pi' is measured at"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        metavar="OUT.json",
        help="where the graft writes what it measured",
    )
    args = parser.parse_args(argv)
    graft(args.old, args.new, tau=args.tau, lam=args.lam, manifest_path=args.manifest)


if __name__ == "__main__":
    main()
