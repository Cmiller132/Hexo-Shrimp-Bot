"""Packed imitation and outcome-horizon evaluation for lab and KLENT weights."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from ..klent.improve import improved_policy
from ..klent.run import load_model
from ..klent.train import KlentConfig, _gpu_lock
from ..model import compose_acting_q, compose_q
from .corpus import FrozenCorpus
from .train import (
    _as_corpus,
    current_versions,
    pack_inference_indices,
    sample_sizes,
)
from .variants import build_variant, count_parameters, variant_spec


# (label, inclusive lower bound, inclusive upper bound or None).
DISTANCE_BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("1-4", 1, 4),
    ("5-8", 5, 8),
    ("9-12", 9, 12),
    ("13-16", 13, 16),
    ("17-24", 17, 24),
    ("25-32", 25, 32),
    ("33-48", 33, 48),
    ("49-64", 49, 64),
    ("65+", 65, None),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bucket_masks(dist: np.ndarray) -> dict[str, np.ndarray]:
    dist = np.asarray(dist, dtype=np.int64)
    if np.any(dist < 1):
        index = int(np.nonzero(dist < 1)[0][0])
        raise ValueError(f"distance-from-end must be positive; sample {index} has {dist[index]}")
    masks = {
        label: (dist >= lower) & (True if upper is None else dist <= upper)
        for label, lower, upper in DISTANCE_BUCKETS
    }
    covered = np.zeros(len(dist), dtype=np.int8)
    for mask in masks.values():
        covered += mask
    if np.any(covered != 1):
        index = int(np.nonzero(covered != 1)[0][0])
        raise AssertionError(
            f"distance buckets cover sample {index} {int(covered[index])} times"
        )
    return masks


def _imitation_block(
    top1: np.ndarray, top3: np.ndarray, masks: Mapping[str, np.ndarray]
) -> dict[str, object]:
    def metrics(mask: np.ndarray) -> dict[str, object]:
        n = int(mask.sum())
        return {
            "n": n,
            "top1": float(top1[mask].mean()) if n else None,
            "top3": float(top3[mask].mean()) if n else None,
        }

    all_samples = np.ones(len(top1), dtype=bool)
    return {
        "overall": metrics(all_samples),
        "by_distance": {label: metrics(mask) for label, mask in masks.items()},
    }


def _horizon_block(
    prediction: np.ndarray, z: np.ndarray, masks: Mapping[str, np.ndarray]
) -> dict[str, dict[str, object]]:
    if not np.isfinite(prediction).all():
        index = int(np.nonzero(~np.isfinite(prediction))[0][0])
        raise ValueError(f"non-finite horizon prediction at sample {index}: {prediction[index]}")
    block: dict[str, dict[str, object]] = {}
    for label, mask in masks.items():
        n = int(mask.sum())
        selected = prediction[mask]
        targets = z[mask]
        block[label] = {
            "n": n,
            "sign_accuracy": float((np.sign(selected) == targets).mean()) if n else None,
            "mae": float(np.abs(selected - targets).mean()) if n else None,
            "mean_prediction": float(selected.mean()) if n else None,
            "mean_abs_prediction": float(np.abs(selected).mean()) if n else None,
        }
    return block


def _evaluation_heads(model, batch, include_state_value: bool):
    _stones, windows, token = model.trunk(batch)
    policy, critic = model.cell_head_logits(windows, token, batch)
    value = model.value_head(windows, token, batch)[0] if include_state_value else None
    return policy, critic, value


@torch.no_grad()
def evaluate_model(
    model,
    corpus: str | os.PathLike[str] | FrozenCorpus,
    *,
    split: str = "test",
    variant: str = "mantis",
    include_state_value: bool,
    device: str = "cpu",
    autocast: bool | None = None,
    compile: bool = False,
    tau: float = 0.1,
    lam: float = 0.01,
    mass_floor: float = 0.2,
    pair_budget: int = KlentConfig.collect_pair_budget,
    cell_budget: int = KlentConfig.collect_cell_budget,
) -> dict[str, object]:
    """Return metric blocks for an already-loaded model, without writing."""

    if tau < 0 or lam < 0 or tau + lam <= 0:
        raise ValueError(f"need tau, lam >= 0 with positive sum, got ({tau}, {lam})")
    if not 0 < mass_floor <= 1:
        raise ValueError(f"mass_floor must be in (0, 1], got {mass_floor}")
    frozen = _as_corpus(corpus)
    samples = frozen.split_samples(split)
    if not len(samples):
        raise ValueError(f"corpus split {split!r} is empty")
    lengths, cells = sample_sizes(frozen, samples)
    chunks = pack_inference_indices(
        lengths,
        cells,
        pair_budget=pair_budget,
        cell_budget=cell_budget,
    )
    collate = variant_spec(variant).collate
    device_type = torch.device(device).type
    expected_autocast = device_type == "cuda"
    if autocast is not None and autocast is not expected_autocast:
        raise ValueError(
            "packed evaluation uses bf16 autocast exactly on CUDA; "
            f"device={device!r}, autocast={autocast}"
        )
    use_autocast = expected_autocast

    def forward(current_model, batch):
        return _evaluation_heads(current_model, batch, include_state_value)

    if compile:
        forward = torch.compile(forward, dynamic=True)

    n = len(samples)
    top1 = np.zeros(n, dtype=bool)
    top3 = np.zeros(n, dtype=bool)
    v_hat_prediction = np.empty(n, dtype=np.float32)
    state_prediction = np.empty(n, dtype=np.float32) if include_state_value else None
    model.eval()
    for indices in chunks:
        games = [frozen.moves_for(int(samples.game[index])) for index in indices]
        ts = [int(samples.t[index]) for index in indices]
        batch = collate(games, ts).to(device)
        ranks = torch.tensor(
            [int(samples.rank[index]) for index in indices],
            dtype=torch.long,
            device=device,
        )
        counts = batch.legal_offsets[1:] - batch.legal_offsets[:-1]
        bad = (ranks < 0) | (ranks >= counts)
        if torch.any(bad):
            row = int(torch.nonzero(bad, as_tuple=False)[0])
            raise ValueError(
                f"corpus sample {indices[row]} has rank {int(ranks[row])} "
                f"for {int(counts[row])} legal moves"
            )
        with _gpu_lock, torch.autocast(
            device_type, dtype=torch.bfloat16, enabled=use_autocast
        ):
            policy, critic, state_value = forward(model, batch)
        policy = policy.float()
        critic = critic.float()
        q_score = compose_acting_q(critic, batch.legal_offsets, mass_floor)
        q_value = compose_q(critic)
        improved = improved_policy(
            policy, q_score, q_value, batch.legal_offsets, tau, lam
        )
        # Transfer each result block once. Pulling scalar tensors inside the
        # sample loop would serialize CUDA execution once per metric field.
        offsets = batch.legal_offsets.cpu().tolist()
        ranks_cpu = ranks.cpu().tolist()
        policy_cpu = policy.cpu()
        v_hat_cpu = improved.v_hat.float().cpu().numpy()
        sample_indices = np.asarray(indices, dtype=np.intp)
        v_hat_prediction[sample_indices] = v_hat_cpu
        if state_prediction is not None:
            assert state_value is not None
            state_prediction[sample_indices] = state_value.float().cpu().numpy()
        for row, sample_index in enumerate(indices):
            lo, hi = offsets[row], offsets[row + 1]
            logits = policy_cpu[lo:hi]
            rank = ranks_cpu[row]
            top1[sample_index] = int(logits.argmax()) == rank
            top3[sample_index] = bool(
                (logits.topk(min(3, hi - lo)).indices == rank).any()
            )

    z = np.asarray(samples.z, dtype=np.int8)
    masks = _bucket_masks(np.asarray(samples.dist, dtype=np.int64))
    horizon = {"v_hat": _horizon_block(v_hat_prediction, z, masks)}
    if state_prediction is not None:
        horizon["state_value"] = _horizon_block(state_prediction, z, masks)
    return {
        "flags": {
            "device": device,
            "tau": tau,
            "lam": lam,
            "mass_floor": mass_floor,
            "state_value_scored": include_state_value,
            "autocast": use_autocast,
            "compile": compile,
        },
        "imitation": _imitation_block(top1, top3, masks),
        "horizon": horizon,
    }


def _write_scores(path: Path, scores: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(scores, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _checkpoint_metadata(path: Path, kind: str, versions: Mapping[str, object], count: int):
    return {
        "kind": kind,
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "versions": dict(versions),
        "param_count": count,
    }


def evaluate_cell(
    cell_dir: str | os.PathLike[str],
    corpus: str | os.PathLike[str] | FrozenCorpus,
    *,
    split: str = "test",
    device: str = "cpu",
    autocast: bool | None = None,
    compile: bool = False,
    tau: float = 0.1,
    lam: float = 0.01,
    mass_floor: float = 0.2,
) -> dict[str, object]:
    """Load a lab cell, score it, and replace ``cell/scores.json``."""

    cell = Path(cell_dir)
    config_path = cell / "config.json"
    checkpoint_path = cell / "checkpoint_final.pt"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("lab_cell_format") != 1:
        raise ValueError(
            f"unsupported lab cell format {config.get('lab_cell_format')!r} in {config_path}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("lab_cell_format") != 1:
        raise ValueError(
            f"unsupported lab checkpoint format {checkpoint.get('lab_cell_format')!r}"
        )
    versions = current_versions()
    if config.get("versions") != versions or checkpoint.get("versions") != versions:
        raise ValueError(
            f"lab cell versions do not match this build {versions}: "
            f"config={config.get('versions')}, checkpoint={checkpoint.get('versions')}"
        )
    variant = config["variant"]
    model_kw = config["model_kw"]
    if checkpoint.get("variant") != variant or checkpoint.get("model_kw") != model_kw:
        raise ValueError("lab checkpoint variant identity differs from config.json")
    frozen = _as_corpus(corpus)
    expected_sha = config["corpus"]["sha256"]
    if checkpoint.get("corpus_sha256") != expected_sha or frozen.sha256 != expected_sha:
        raise ValueError(
            "lab cell/corpus hash mismatch: "
            f"config={expected_sha}, checkpoint={checkpoint.get('corpus_sha256')}, "
            f"corpus={frozen.sha256}"
        )
    model, _normalized, _spec = build_variant(variant, model_kw)
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device)
    parameter_count = count_parameters(model)
    if config.get("param_count") != parameter_count or checkpoint.get("param_count") != parameter_count:
        raise ValueError(
            f"lab cell parameter count metadata does not match model count {parameter_count}"
        )
    blocks = evaluate_model(
        model,
        frozen,
        split=split,
        variant=variant,
        include_state_value=True,
        device=device,
        autocast=autocast,
        compile=compile,
        tau=tau,
        lam=lam,
        mass_floor=mass_floor,
    )
    scores = {
        "scores_format": 1,
        "variant": variant,
        "model_kw": model_kw,
        "seed": config["seed"],
        "corpus": {"name": frozen.name, "sha256": frozen.sha256},
        "split": split,
        "checkpoint": _checkpoint_metadata(
            checkpoint_path, "lab_cell", versions, parameter_count
        ),
        **blocks,
    }
    _write_scores(cell / "scores.json", scores)
    return scores


def evaluate_checkpoint(
    checkpoint: str | os.PathLike[str],
    corpus: str | os.PathLike[str] | FrozenCorpus,
    *,
    out: str | os.PathLike[str] | None,
    split: str = "test",
    device: str = "cpu",
    autocast: bool | None = None,
    compile: bool = False,
    tau: float = 0.1,
    lam: float = 0.01,
    mass_floor: float = 0.2,
) -> dict[str, object]:
    """Score a production KLENT checkpoint; ``out`` is mandatory."""

    if out is None:
        raise ValueError("--out is required when evaluating a production checkpoint")
    checkpoint_path = Path(checkpoint)
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    versions = raw.get("versions")
    model = load_model(checkpoint_path, device=device)
    parameter_count = count_parameters(model)
    frozen = _as_corpus(corpus)
    blocks = evaluate_model(
        model,
        frozen,
        split=split,
        variant="mantis",
        include_state_value=False,
        device=device,
        autocast=autocast,
        compile=compile,
        tau=tau,
        lam=lam,
        mass_floor=mass_floor,
    )
    scores = {
        "scores_format": 1,
        "variant": "mantis",
        "model_kw": {},
        "seed": None,
        "corpus": {"name": frozen.name, "sha256": frozen.sha256},
        "split": split,
        "checkpoint": _checkpoint_metadata(
            checkpoint_path, "production_klent", versions, parameter_count
        ),
        **blocks,
    }
    _write_scores(Path(out), scores)
    return scores
