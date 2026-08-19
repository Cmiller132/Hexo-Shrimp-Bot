"""Cross-seed aggregation for lab score artifacts, with no plotting."""

from __future__ import annotations

import json
import math
import os
import statistics
from pathlib import Path
from typing import Iterable, Sequence

from .evaluate import DISTANCE_BUCKETS, SCORES_FORMAT, scores_filename
from .variants import derived_cell_name, normalize_model_kw

LOSS_METRICS = ("policy_nll", "critic_ce", "state_value_ce")


def discover_scores(
    sweep: str | os.PathLike[str], *, split: str = "val", ema: bool = False
) -> list[Path]:
    """Find every cell score file of one split below a sweep, in path order."""

    root = Path(sweep)
    name = scores_filename(split, ema)
    paths = sorted(root.glob(f"*/s*/{name}"))
    if not paths:
        raise ValueError(f"no {name} artifacts found under sweep {root}")
    return paths


def _mean_sd(values: Sequence[float | None], label: str) -> dict[str, object]:
    present = [float(value) for value in values if value is not None]
    if present and len(present) != len(values):
        raise ValueError(f"metric {label} is absent for only some seeds")
    if not present:
        return {"n": 0, "mean": None, "sample_sd": None}
    if not all(math.isfinite(value) for value in present):
        raise ValueError(f"metric {label} contains a non-finite value: {present}")
    return {
        "n": len(present),
        "mean": statistics.fmean(present),
        "sample_sd": statistics.stdev(present) if len(present) >= 2 else None,
    }


def _identity(score: dict) -> tuple[str, str, dict[str, object]]:
    variant = score["variant"]
    model_kw = normalize_model_kw(score.get("model_kw", {}))
    identity = json.dumps([variant, model_kw], sort_keys=True, separators=(",", ":"))
    label = derived_cell_name(variant, model_kw)
    return identity, label, model_kw


def _load_scores(paths: Iterable[str | os.PathLike[str]]) -> list[tuple[Path, dict]]:
    loaded = []
    for raw_path in paths:
        path = Path(raw_path)
        score = json.loads(path.read_text(encoding="utf-8"))
        if score.get("scores_format") != SCORES_FORMAT:
            raise ValueError(
                f"unsupported scores format {score.get('scores_format')!r} in {path}"
            )
        loaded.append((path, score))
    if not loaded:
        raise ValueError("report requires at least one scores path")
    return loaded


def _cell_throughput(scores_path: Path) -> float:
    """Mean epoch throughput from the cell artifact beside ``scores.json``."""

    metrics_path = scores_path.with_name("metrics.jsonl")
    try:
        rows = [
            json.loads(line)
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except FileNotFoundError as exc:
        raise ValueError(f"scores file has no sibling metrics.jsonl: {scores_path}") from exc
    if not rows:
        raise ValueError(f"cell metrics file is empty: {metrics_path}")
    try:
        values = [float(row["samples_per_second"]) for row in rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed samples_per_second in {metrics_path}") from exc
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError(f"non-positive or non-finite samples_per_second in {metrics_path}")
    return statistics.fmean(values)


def aggregate_scores(paths: Iterable[str | os.PathLike[str]]) -> dict[str, object]:
    """Aggregate comparable score files into a JSON-serializable report."""

    loaded = _load_scores(paths)
    hashes = {score["corpus"]["sha256"] for _path, score in loaded}
    if len(hashes) != 1:
        details = ", ".join(
            f"{path}={score['corpus']['sha256']}" for path, score in loaded
        )
        raise ValueError(f"refusing to mix corpus hashes: {details}")
    splits = {score["split"] for _path, score in loaded}
    if len(splits) != 1:
        details = ", ".join(f"{path}={score['split']}" for path, score in loaded)
        raise ValueError(f"refusing to mix corpus splits: {details}")

    grouped: dict[str, list[tuple[Path, dict]]] = {}
    labels: dict[str, str] = {}
    model_kwargs: dict[str, dict[str, object]] = {}
    for path, score in loaded:
        identity, label, model_kw = _identity(score)
        if identity in labels and labels[identity] != label:
            raise ValueError(
                f"one variant identity uses multiple cell names: {labels[identity]!r}, {label!r}"
            )
        grouped.setdefault(identity, []).append((path, score))
        labels[identity] = label
        model_kwargs[identity] = model_kw

    variants: dict[str, object] = {}
    for identity in sorted(grouped, key=lambda key: labels[key]):
        entries = grouped[identity]
        scores = [score for _path, score in entries]
        seeds = [score.get("seed") for score in scores]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"cell {labels[identity]!r} repeats a seed: {seeds}")
        counts = {int(score["checkpoint"]["param_count"]) for score in scores}
        if len(counts) != 1:
            raise ValueError(
                f"cell {labels[identity]!r} has inconsistent parameter counts: {sorted(counts)}"
            )
        throughput = [_cell_throughput(path) for path, _score in entries]
        imitation = _mean_sd(
            [score["imitation"]["overall"]["top1"] for score in scores],
            f"{labels[identity]} imitation top1",
        )
        losses = {
            metric: _mean_sd(
                [
                    score["loss"][metric]["overall"]["mean"]
                    if metric in score["loss"]
                    else None
                    for score in scores
                ],
                f"{labels[identity]} {metric}",
            )
            for metric in LOSS_METRICS
        }

        channels = sorted(
            set().union(*(score["horizon"].keys() for score in scores))
        )
        horizon: dict[str, object] = {}
        for channel in channels:
            if not all(channel in score["horizon"] for score in scores):
                raise ValueError(
                    f"horizon channel {channel!r} is absent for only some seeds "
                    f"of cell {labels[identity]!r}"
                )
            horizon[channel] = {
                label: _mean_sd(
                    [
                        score["horizon"][channel][label]["sign_accuracy"]
                        for score in scores
                    ],
                    f"{labels[identity]} {channel} {label} sign_accuracy",
                )
                for label, _lower, _upper in DISTANCE_BUCKETS
            }

        label = labels[identity]
        if label in variants:
            raise ValueError(f"distinct variant identities share report label {label!r}")
        variants[label] = {
            "variant": scores[0]["variant"],
            "model_kw": model_kwargs[identity],
            "seeds": sorted(seeds, key=lambda value: (value is None, value)),
            "param_count": counts.pop(),
            "samples_per_second": _mean_sd(
                throughput, f"{label} samples_per_second"
            ),
            "imitation_top1": imitation,
            "loss": losses,
            "horizon_sign_accuracy": horizon,
        }

    first = loaded[0][1]
    return {
        "report_format": 2,
        "corpus": dict(first["corpus"]),
        "split": first["split"],
        "ema": bool(first.get("ema", False)),
        "variants": variants,
    }


def _format_stat(stat: dict[str, object]) -> str:
    mean = stat["mean"]
    if mean is None:
        return "n/a"
    sd = stat["sample_sd"]
    return f"{mean:.4f} ± {sd:.4f}" if sd is not None else f"{mean:.4f} ± n/a"


def render_report(report: dict[str, object]) -> str:
    """Render the compact plain-text tables printed by report mode."""

    lines = [
        f"corpus {report['corpus']['name']} ({report['corpus']['sha256']})  "
        f"split {report['split']}{'  ema' if report.get('ema') else ''}",
        "cell  seeds  params  samples/s  imitation top-1  policy NLL  critic CE  state-value CE",
    ]
    variants = report["variants"]
    for label, row in variants.items():
        lines.append(
            f"{label}  {len(row['seeds'])}  {row['param_count']}  "
            f"{_format_stat(row['samples_per_second'])}  "
            f"{_format_stat(row['imitation_top1'])}  "
            + "  ".join(_format_stat(row["loss"][metric]) for metric in LOSS_METRICS)
        )
    lines.extend(["", "horizon sign-accuracy", "cell  channel  " + "  ".join(
        label for label, _lower, _upper in DISTANCE_BUCKETS
    )])
    for label, row in variants.items():
        for channel, buckets in row["horizon_sign_accuracy"].items():
            lines.append(
                f"{label}  {channel}  "
                + "  ".join(
                    _format_stat(buckets[bucket])
                    for bucket, _lower, _upper in DISTANCE_BUCKETS
                )
            )
    return "\n".join(lines)


def build_report(
    scores_paths: Iterable[str | os.PathLike[str]],
    out: str | os.PathLike[str],
    *,
    emit: bool = True,
) -> dict[str, object]:
    """Aggregate, write ``report.json``, and optionally print its text table."""

    report = aggregate_scores(scores_paths)
    destination = Path(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if emit:
        print(render_report(report))
    return report


def report_sweep(
    sweep: str | os.PathLike[str],
    *,
    split: str = "val",
    ema: bool = False,
    emit: bool = True,
) -> dict[str, object]:
    """Convenience entry point for ``report --sweep``."""

    root = Path(sweep)
    return build_report(
        discover_scores(root, split=split, ema=ema),
        root / f"report-{split}{'-ema' if ema else ''}.json",
        emit=emit,
    )
