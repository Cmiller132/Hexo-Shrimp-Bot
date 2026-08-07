"""Cross-seed lab report aggregation and comparability refusals."""

from __future__ import annotations

import json
import math
from copy import deepcopy

import pytest

from mantisnet.lab.evaluate import DISTANCE_BUCKETS
from mantisnet.lab.report import build_report, discover_scores


def _scores(
    seed: int,
    *,
    corpus_sha: str = "a" * 64,
    split: str = "test",
    model_kw: dict | None = None,
    offset: float = 0.0,
    channels: tuple[str, ...] = ("state_value", "v_hat"),
) -> dict:
    top1 = 0.4 + 0.2 * seed + offset
    horizon = {
        channel: {
            label: {
                "n": 2,
                "sign_accuracy": top1 + index * 0.001,
                "mae": 0.5,
                "mean_prediction": 0.0,
                "mean_abs_prediction": 0.25,
            }
            for index, (label, _lower, _upper) in enumerate(DISTANCE_BUCKETS)
        }
        for channel in channels
    }
    return {
        "scores_format": 1,
        "variant": "mantis",
        "model_kw": {"h": 8} if model_kw is None else model_kw,
        "seed": seed,
        "corpus": {"name": "frozen", "sha256": corpus_sha},
        "split": split,
        "checkpoint": {"param_count": 1234},
        "imitation": {
            "overall": {"n": 18, "top1": top1, "top3": 0.9},
            "by_distance": {},
        },
        "horizon": horizon,
    }


def _write(path, value, *, throughput=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    if throughput is not None:
        (path.parent / "metrics.jsonl").write_text(
            json.dumps({"samples_per_second": throughput}) + "\n",
            encoding="utf-8",
        )


def test_report_aggregates_sample_sd_and_writes_plain_table(tmp_path, capsys):
    sweep = tmp_path / "sweep"
    first = sweep / "mantis+h8" / "s0" / "scores.json"
    second = sweep / "mantis+h8" / "s1" / "scores.json"
    _write(first, _scores(0), throughput=100.0)
    _write(second, _scores(1), throughput=120.0)
    assert discover_scores(sweep) == [first, second]

    out = sweep / "report.json"
    report = build_report([first, second], out)
    printed = capsys.readouterr().out
    assert "mantis+h8" in printed and "horizon sign-accuracy" in printed and "±" in printed
    assert out.is_file()
    row = report["variants"]["mantis+h8"]
    assert row["seeds"] == [0, 1]
    assert row["param_count"] == 1234
    assert row["imitation_top1"]["mean"] == pytest.approx(0.5)
    assert row["imitation_top1"]["sample_sd"] == pytest.approx(math.sqrt(0.02))
    assert row["samples_per_second"]["mean"] == pytest.approx(110.0)
    first_bucket = DISTANCE_BUCKETS[0][0]
    assert row["horizon_sign_accuracy"]["v_hat"][first_bucket]["mean"] == pytest.approx(0.5)


def _arm(sweep, cell: str, seeds, **kwargs) -> list:
    """Write one arm's per-seed score files and return their paths."""
    paths = []
    for seed in seeds:
        path = sweep / cell / f"s{seed}" / "scores.json"
        _write(path, _scores(seed, **kwargs), throughput=100.0 + seed)
        paths.append(path)
    return paths


def test_report_pairs_every_arm_against_the_named_baseline(tmp_path, capsys):
    """The paired difference removes the seed's shared draw.

    Both arms move together with the seed here — the control spans 0.4 to 0.8 —
    while the arm's advantage over it is exactly 0.05 at every seed. Raw means
    carry the whole spread of the draw; the paired difference carries none of
    it, which is the entire reason the round reads it.
    """

    sweep = tmp_path / "sweep"
    control = _arm(sweep, "mantis+h8", (0, 1, 2))
    arm = _arm(sweep, "mantis+h16", (0, 1, 2), model_kw={"h": 16}, offset=0.05)

    report = build_report(
        control + arm, sweep / "report.json", baseline="mantis+h8"
    )
    assert report["baseline"] == "mantis+h8"

    raw = report["variants"]["mantis+h16"]["imitation_top1"]
    assert raw["mean"] == pytest.approx(0.65)
    assert raw["sample_sd"] == pytest.approx(0.2)

    paired = report["variants"]["mantis+h16"]["paired_vs_baseline"]["imitation_top1"]
    assert paired["seeds"] == [0, 1, 2]
    assert paired["mean"] == pytest.approx(0.05)
    assert paired["sample_sd"] == pytest.approx(0.0, abs=1e-12)

    # The control against itself is identically zero, which is what says the
    # named arm is the one that was paired against.
    own = report["variants"]["mantis+h8"]["paired_vs_baseline"]["imitation_top1"]
    assert own["mean"] == pytest.approx(0.0) and own["sample_sd"] == pytest.approx(0.0)

    bucket = DISTANCE_BUCKETS[0][0]
    assert report["variants"]["mantis+h16"]["paired_vs_baseline"][
        f"horizon_sign_accuracy/v_hat/{bucket}"
    ]["mean"] == pytest.approx(0.05)

    printed = capsys.readouterr().out
    assert "per-seed paired difference against mantis+h8" in printed


def test_report_pairs_only_the_seeds_and_channels_both_arms_have(tmp_path):
    """An arm holding no state-value head is compared on what it does hold."""

    sweep = tmp_path / "sweep"
    control = _arm(sweep, "mantis+h8", (0, 1))
    arm = _arm(
        sweep,
        "mantis+h16",
        (1, 2),
        model_kw={"h": 16},
        offset=0.05,
        channels=("v_hat",),
    )

    report = build_report(
        control + arm, sweep / "report.json", baseline="mantis+h8", emit=False
    )
    paired = report["variants"]["mantis+h16"]["paired_vs_baseline"]
    assert paired["imitation_top1"]["seeds"] == [1]
    bucket = DISTANCE_BUCKETS[0][0]
    assert f"horizon_sign_accuracy/v_hat/{bucket}" in paired
    assert f"horizon_sign_accuracy/state_value/{bucket}" not in paired


def test_report_refuses_a_baseline_that_names_no_cell(tmp_path):
    sweep = tmp_path / "sweep"
    paths = _arm(sweep, "mantis+h8", (0, 1))
    with pytest.raises(ValueError, match="mantis\\+h8"):
        build_report(paths, sweep / "report.json", baseline="absent", emit=False)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"corpus_sha": "b" * 64}, "corpus hashes"),
        ({"split": "val"}, "corpus splits"),
    ],
)
def test_report_refuses_incomparable_scores(tmp_path, mutation, message):
    first_score = _scores(0)
    second_score = _scores(
        1,
        corpus_sha=mutation.get("corpus_sha", "a" * 64),
        split=mutation.get("split", "test"),
    )
    first = tmp_path / "one" / "scores.json"
    second = tmp_path / "two" / "scores.json"
    _write(first, first_score, throughput=100.0)
    _write(second, second_score, throughput=120.0)
    with pytest.raises(ValueError, match=message):
        build_report([first, second], tmp_path / "report.json", emit=False)


def test_report_refuses_partial_horizon_channels(tmp_path):
    first_score = _scores(0)
    second_score = deepcopy(_scores(1))
    del second_score["horizon"]["state_value"]
    first = tmp_path / "one" / "scores.json"
    second = tmp_path / "two" / "scores.json"
    _write(first, first_score, throughput=100.0)
    _write(second, second_score, throughput=120.0)
    with pytest.raises(ValueError, match="absent for only some seeds"):
        build_report([first, second], tmp_path / "report.json", emit=False)
