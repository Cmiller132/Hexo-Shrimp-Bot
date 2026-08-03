"""Probe the categorical critic's committed return mass on real workloads."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..builder import collate_positions
from ..klent.run import load_model
from ..klent.train import KlentConfig, network_evaluate
from ..model import CRITIC_LOGITS, compose_acting_q, return_mass
from .cohort import corpus_cohort, selfplay_cohort


FLOORS = (0.1, 0.2, 0.3, 0.5)
_QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)


def _distribution(values) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("cannot summarize an empty critic-mass sample")
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "quantiles": {
            f"{q:g}": float(v)
            for q, v in zip(_QUANTILES, np.quantile(arr, _QUANTILES))
        },
    }


def _segment_max(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    return np.asarray(
        [values[a:b].max() for a, b in zip(offsets[:-1], offsets[1:])],
        dtype=np.float64,
    )


def _segment_argmax(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    return np.asarray(
        [int(np.argmax(values[a:b])) for a, b in zip(offsets[:-1], offsets[1:])]
    )


def probe_mass(
    *,
    checkpoint,
    corpus=None,
    split: str = "test",
    envs: int = 32,
    steps: int = 64,
    stride: int = 16,
    seed: int = 0,
    device: str = "cpu",
    compile: bool = False,
) -> dict:
    """Measure ``M = 1 - p_zero`` and acting-score floor sensitivity."""
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if envs <= 0 or steps < 0:
        raise ValueError(f"envs must be positive and steps nonnegative, got {envs}, {steps}")
    model = load_model(Path(checkpoint), device).eval()
    cfg = KlentConfig(
        device=device, autocast=device == "cuda", compile=compile
    )
    if corpus is None:
        positions = selfplay_cohort(
            envs=envs,
            steps=steps,
            evaluate=network_evaluate(model, cfg),
            seed=seed,
            pair_budget=cfg.collect_pair_budget,
            cell_budget=cfg.collect_cell_budget,
        )
    else:
        positions = corpus_cohort(corpus, split=split, count=envs, seed=seed)
    batch = collate_positions(positions).to(device)
    with torch.no_grad(), torch.autocast(
        device, torch.bfloat16, enabled=cfg.autocast
    ):
        _s, w, g = model.trunk(batch)
        _policy, critic = model.cell_head_logits(w, g, batch)
    if critic.shape[-1] != CRITIC_LOGITS or CRITIC_LOGITS != 3:
        raise ValueError(
            f"mass probe requires the trinomial critic, got shape {tuple(critic.shape)}"
        )
    positive, negative = return_mass(critic)
    mass = positive + negative
    q_value = positive - negative
    offsets = batch.legal_offsets.cpu().numpy()
    mass_np = mass.cpu().numpy()
    q_np = q_value.cpu().numpy()
    pooled = mass_np[::stride]
    maxima = _segment_max(mass_np, offsets)
    ratio = np.divide(q_np, mass_np, out=np.zeros_like(q_np), where=mass_np > 0)
    correlation = (
        float(np.corrcoef(np.abs(q_np), mass_np)[0, 1])
        if np.std(np.abs(q_np)) > 0 and np.std(mass_np) > 0
        else None
    )

    sensitivity = {}
    reference_score = None
    reference_top = None
    for floor in FLOORS:
        score = compose_acting_q(critic, batch.legal_offsets, floor).cpu().numpy()
        top = _segment_argmax(score, offsets)
        if floor == 0.2:
            reference_score = score
            reference_top = top
        sensitivity[f"{floor:g}"] = {
            "score": _distribution(score[::stride]),
            "top1_changed_vs_0.2": None,
            "mean_abs_delta_vs_0.2": None,
        }
    assert reference_score is not None and reference_top is not None
    for floor in FLOORS:
        score = compose_acting_q(critic, batch.legal_offsets, floor).cpu().numpy()
        top = _segment_argmax(score, offsets)
        row = sensitivity[f"{floor:g}"]
        row["top1_changed_vs_0.2"] = float(np.mean(top != reference_top))
        row["mean_abs_delta_vs_0.2"] = float(
            np.mean(np.abs(score - reference_score))
        )

    report = {
        "mode": "mass",
        "checkpoint": str(Path(checkpoint)),
        "device": device,
        "positions": len(positions),
        "legal_cells": int(len(mass_np)),
        "pooled_stride": stride,
        "committed_mass": {
            "pooled": _distribution(pooled),
            "per_position_max": _distribution(maxima),
        },
        "q_over_mass": {
            "ratio": _distribution(ratio[::stride]),
            "abs_q": _distribution(np.abs(q_np[::stride])),
            "abs_q_mass_correlation": correlation,
        },
        "acting_score_sensitivity": sensitivity,
    }
    print(json.dumps(report, indent=2))
    return report
