"""Cross-seed lab report aggregation and comparability refusals."""

from __future__ import annotations

import json
import math
from copy import deepcopy

import pytest

from mantisnet.lab.evaluate import DISTANCE_BUCKETS
from mantisnet.lab.report import build_report, discover_scores


def _scores(seed: int, *, corpus_sha: str = "a" * 64, split: str = "test") -> dict:
    top1 = 0.4 + 0.2 * seed
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
        for channel in ("state_value", "v_hat")
    }
    return {
        "scores_format": 1,
        "variant": "mantis",
        "model_kw": {"h": 8},
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
