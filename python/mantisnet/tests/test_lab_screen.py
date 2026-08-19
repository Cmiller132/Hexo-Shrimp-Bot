"""Screen protocol v2: fixture noise floor, composite S, and the keep rule."""

from __future__ import annotations

import json
import math
import statistics

import pytest

from mantisnet.lab.evaluate import DISTANCE_BUCKETS, SCORES_FORMAT, scores_filename
from mantisnet.lab.screen import (
    FIXTURE_SEEDS,
    KEEP_S,
    POLICY_FLOOR_Z,
    RECIPE,
    SEEDS,
    WEIGHT_CRITIC,
    WEIGHT_POLICY,
    arm_verdict,
    cell_metrics,
    fixture_stats,
    main,
    render_verdicts,
    screen_verdicts,
)


def _scores(
    seed: int,
    *,
    critic_ce: float,
    policy_nll: float,
    top1: float = 0.42,
    mean_prediction: float = 0.3,
    mean_abs_prediction: float = 0.6,
    split: str = "val",
    ema: bool = False,
) -> dict:
    horizon = {
        channel: {
            label: {
                "n": 10,
                "sign_accuracy": 0.5 + 0.01 * index,
                "mae": 0.9,
                "mean_prediction": mean_prediction,
                "mean_abs_prediction": mean_abs_prediction,
            }
            for index, (label, _lower, _upper) in enumerate(DISTANCE_BUCKETS)
        }
        for channel in ("v_hat", "state_value")
    }
    loss = {
        "policy_nll": {"overall": {"n": 90, "mean": policy_nll}, "by_distance": {}},
        "critic_ce": {"overall": {"n": 90, "mean": critic_ce}, "by_distance": {}},
        "state_value_ce": {"overall": {"n": 90, "mean": 0.7}, "by_distance": {}},
    }
    return {
        "scores_format": SCORES_FORMAT,
        **({"ema": True} if ema else {}),
        "variant": "mantis",
        "model_kw": {},
        "seed": seed,
        "corpus": {"name": "frozen", "sha256": "a" * 64},
        "split": split,
        "checkpoint": {"param_count": 1000},
        "imitation": {"overall": {"n": 90, "top1": top1, "top3": 0.66}, "by_distance": {}},
        "horizon": horizon,
        "loss": loss,
    }


def _write_cell(root, arm: str, seed: int, scores: dict, *, sps: float = 1000.0):
    cell = root / arm / f"s{seed}"
    cell.mkdir(parents=True, exist_ok=True)
    (cell / scores_filename(scores["split"], bool(scores.get("ema")))).write_text(
        json.dumps(scores), encoding="utf-8"
    )
    (cell / "metrics.jsonl").write_text(
        json.dumps({"samples_per_second": sps}) + "\n", encoding="utf-8"
    )
    return cell


# Fixture values: critic 0.650 ± 0.010, policy 2.300 ± 0.020, perfectly
# correlated across seeds (rho = 1) so the composite's variance is exact.
FIXTURE = [(0.640, 2.280), (0.645, 2.290), (0.650, 2.300), (0.655, 2.310), (0.660, 2.320), (0.650, 2.300)]


@pytest.fixture
def sweep(tmp_path):
    for seed, (critic, policy) in enumerate(FIXTURE):
        _write_cell(tmp_path, "fixture", seed, _scores(seed, critic_ce=critic, policy_nll=policy))
    return tmp_path


def test_recipe_is_the_owner_ruling():
    assert RECIPE.epochs == 4
    assert RECIPE.train_subset == 0, "the whole realized train split"
    assert RECIPE.ema_decay == pytest.approx(0.995)
    assert (RECIPE.pair_budget, RECIPE.cell_budget) == (2_000_000, 125_000)
    assert (RECIPE.collect_pair_budget, RECIPE.collect_cell_budget) == (6_000_000, 600_000)
    assert SEEDS == (0, 1, 2) and FIXTURE_SEEDS == (0, 1, 2, 3, 4, 5)
    assert (WEIGHT_CRITIC, WEIGHT_POLICY, KEEP_S, POLICY_FLOOR_Z) == (2.0, 1.0, 2.0, -1.0)


def test_fixture_stats_measure_the_noise_floor(sweep):
    cells = [cell_metrics(sweep / "fixture" / f"s{seed}") for seed in range(6)]
    stats = fixture_stats(cells)
    assert stats.n == 6
    assert stats.mean_critic_ce == pytest.approx(0.650)
    assert stats.sd_critic_ce == pytest.approx(statistics.stdev([c for c, _p in FIXTURE]))
    assert stats.mean_policy_nll == pytest.approx(2.300)
    assert stats.rho == pytest.approx(1.0)
    assert stats.pathological_cells == 0
    assert stats.mean_long_horizon_sign == pytest.approx(statistics.fmean([0.56, 0.57, 0.58]))
    with pytest.raises(ValueError, match="at least 3 seeds"):
        fixture_stats(cells[:2])


def _arm(sweep, name, values, **kw):
    for seed, (critic, policy) in enumerate(values):
        _write_cell(sweep, name, seed, _scores(seed, critic_ce=critic, policy_nll=policy, **kw))
    return sweep / name


def test_composite_and_keep_rule(sweep):
    fixture = fixture_stats([cell_metrics(sweep / "fixture" / f"s{seed}") for seed in range(6)])
    sd_c, sd_p = fixture.sd_critic_ce, fixture.sd_policy_nll
    scale = math.sqrt(1 / 3 + 1 / 6)

    # Critic 3 SD better, policy 1 SD better on every seed -> keep.
    better = _arm(sweep, "better", [(0.650 - 3 * sd_c, 2.300 - sd_p)] * 3)
    verdict = arm_verdict("better", [cell_metrics(p) for p in sorted(better.glob("s*"))], fixture)
    z_c, z_p = 3 / scale, 1 / scale
    expected_S = (2 * z_c + z_p) / math.sqrt(4 + 1 + 4 * fixture.rho)
    assert verdict.z_critic == pytest.approx(z_c)
    assert verdict.z_policy == pytest.approx(z_p)
    assert verdict.S == pytest.approx(expected_S)
    assert verdict.verdict == "keep"
    assert len(verdict.per_seed_S) == 3 and all(s > 0 for s in verdict.per_seed_S)

    # Critic 4 SD better but policy 3 SD worse -> S clears the bar, policy does not.
    regressed = _arm(sweep, "regressed", [(0.650 - 4 * sd_c, 2.300 + 3 * sd_p)] * 3)
    verdict = arm_verdict("regressed", [cell_metrics(p) for p in sorted(regressed.glob("s*"))], fixture)
    assert verdict.S >= KEEP_S and verdict.z_policy < POLICY_FLOOR_Z
    assert verdict.verdict == "policy-regressed"

    # Within noise on both -> neutral, even with a cheerful mean.
    neutral = _arm(sweep, "neutral", [(0.650 - 0.3 * sd_c, 2.300 - 0.3 * sd_p)] * 3)
    verdict = arm_verdict("neutral", [cell_metrics(p) for p in sorted(neutral.glob("s*"))], fixture)
    assert verdict.verdict == "neutral"

    # Critic 3 SD worse -> negative.
    worse = _arm(sweep, "worse", [(0.650 + 3 * sd_c, 2.300)] * 3)
    verdict = arm_verdict("worse", [cell_metrics(p) for p in sorted(worse.glob("s*"))], fixture)
    assert verdict.verdict == "negative"


def test_seeds_may_disagree_when_the_pooled_mean_clears_the_bar(sweep):
    fixture = fixture_stats([cell_metrics(sweep / "fixture" / f"s{seed}") for seed in range(6)])
    sd_c = fixture.sd_critic_ce
    # Two seeds far better, one slightly worse: pooled S >= 2, one negative seed.
    mixed = _arm(sweep, "mixed", [(0.650 - 6 * sd_c, 2.300), (0.650 - 6 * sd_c, 2.300), (0.650 + 0.5 * sd_c, 2.300)])
    verdict = arm_verdict("mixed", [cell_metrics(p) for p in sorted(mixed.glob("s*"))], fixture)
    assert verdict.verdict == "keep"
    assert min(verdict.per_seed_S) < 0 < max(verdict.per_seed_S)


def test_two_pathological_cells_mark_the_arm_whatever_S_says(sweep):
    fixture = fixture_stats([cell_metrics(sweep / "fixture" / f"s{seed}") for seed in range(6)])
    sd_c = fixture.sd_critic_ce
    root = sweep / "patho"
    _write_cell(sweep, "patho", 0, _scores(0, critic_ce=0.650 - 5 * sd_c, policy_nll=2.3, mean_prediction=0.59, mean_abs_prediction=0.6))
    _write_cell(sweep, "patho", 1, _scores(1, critic_ce=0.650 - 5 * sd_c, policy_nll=2.3, mean_prediction=0.0, mean_abs_prediction=0.01))
    _write_cell(sweep, "patho", 2, _scores(2, critic_ce=0.650 - 5 * sd_c, policy_nll=2.3))
    cells = [cell_metrics(p) for p in sorted(root.glob("s*"))]
    assert [c.optimist for c in cells] == [True, False, False]
    assert [c.agnostic for c in cells] == [False, True, False]
    verdict = arm_verdict("patho", cells, fixture)
    assert verdict.S > KEEP_S
    assert verdict.pathological_cells == 2
    assert verdict.verdict == "pathology-prone"


def test_cell_metrics_refuses_the_wrong_file(sweep):
    cell = _write_cell(sweep, "ema-only", 0, _scores(0, critic_ce=0.65, policy_nll=2.3, ema=True))
    with pytest.raises(FileNotFoundError):
        cell_metrics(cell)  # raw scores absent
    metrics = cell_metrics(cell, ema=True)
    assert metrics.critic_ce == pytest.approx(0.65)
    wrong = _scores(0, critic_ce=0.65, policy_nll=2.3, split="test")
    (cell / scores_filename("val")).write_text(json.dumps(wrong), encoding="utf-8")
    with pytest.raises(ValueError, match="does not hold split='val'"):
        cell_metrics(cell)
    with pytest.raises(ValueError, match="repeats a seed"):
        arm_verdict("dup", [metrics, metrics], fixture_stats([cell_metrics(sweep / "fixture" / f"s{s}") for s in range(6)]))


def test_verdict_cli_writes_json_and_prints_table(sweep, capsys):
    fixture = fixture_stats([cell_metrics(sweep / "fixture" / f"s{seed}") for seed in range(6)])
    _arm(sweep, "armX", [(0.650 - 3 * fixture.sd_critic_ce, 2.300 - fixture.sd_policy_nll)] * 3)
    out = sweep / "verdict.json"
    main([
        "verdict",
        "--fixture", str(sweep / "fixture"),
        "--arm", f"armX={sweep / 'armX'}",
        "--out", str(out),
    ])
    printed = capsys.readouterr().out
    assert "armX" in printed and "keep" in printed and "rho" in printed
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["arms"]["armX"]["verdict"] == "keep"
    assert written["fixture"]["n"] == 6
    assert written["weights"] == {"critic": 2.0, "policy": 1.0}
    again = screen_verdicts(sweep / "fixture", {"armX": sweep / "armX"})
    assert again["arms"]["armX"]["S"] == pytest.approx(written["arms"]["armX"]["S"])
    assert render_verdicts(again).splitlines()[0].startswith("fixture ")
