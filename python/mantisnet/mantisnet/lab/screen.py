"""Screen protocol v2: the recipe every arm trains under, the fixture that
measures the protocol's own seed noise, and the composite verdict.

An *arm* is one model configuration trained at ``SEEDS``; the *fixture* is
the baseline configuration trained at ``FIXTURE_SEEDS`` once, and every arm
is judged against it. The verdict combines the two fit losses the screen
cares about — the critic's trinomial cross-entropy and the policy's NLL of
the played move, both on the scored split — as z-scores against the
fixture's seed spread, critic double-weighted::

    z_c = -(arm critic_ce - fixture critic_ce) / SE_c
    z_p = -(arm policy_nll - fixture policy_nll) / SE_p
    S   = (2 z_c + z_p) / sqrt(4 + 1 + 4 rho)

with SE = sd * sqrt(1/n_arm + 1/n_fixture) and rho the fixture's critic /
policy correlation, so S is a unit-normal statistic under "no effect". An
arm **keeps** at S >= 2 with the policy no more than one SE worse; anything
else is neutral or negative, and speed or other benefits are judged outside
this module (the speed gate is the pinned bench instrument, not cell
throughput). A cell whose critic sits in the optimist or agnostic basin is
flagged; two such cells mark the arm pathology-prone whatever S says.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch

from .evaluate import SCORES_FORMAT, scores_filename
from .train import TrainConfig

# The lean screen budgets of the Step 12+ recipe on the whole realized train
# split (train_subset=0 fits the full split), EMA alongside the raw weights.
RECIPE = TrainConfig(
    epochs=4,
    device="cuda",
    autocast=True,
    compile=True,
    cell_budget=125_000,
    pair_budget=2_000_000,
    collect_cell_budget=600_000,
    collect_pair_budget=6_000_000,
    train_subset=0,
    ema_decay=0.995,
)
SEEDS: tuple[int, ...] = (0, 1, 2)
FIXTURE_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
SPLIT = "val"
WEIGHT_CRITIC = 2.0
WEIGHT_POLICY = 1.0
KEEP_S = 2.0
POLICY_FLOOR_Z = -1.0
# Critic basins, read at the 1-4 plies-from-end bucket where the outcome is
# nearly determined: an optimist critic predicts one sign for everything
# (|mean| close to mean |prediction|); an agnostic one predicts ~0.
OPTIMIST_RATIO = 0.97
AGNOSTIC_MEAN_ABS = 0.05
LONG_HORIZON_BUCKETS = ("33-48", "49-64", "65+")


@dataclass(frozen=True)
class CellMetrics:
    """What the verdict reads from one cell's score file."""

    cell: str
    seed: int
    param_count: int
    critic_ce: float
    policy_nll: float
    state_value_ce: float | None
    top1: float
    top3: float
    long_horizon_sign: float
    optimist: bool
    agnostic: bool
    samples_per_second: float

    @property
    def pathological(self) -> bool:
        return self.optimist or self.agnostic


def _finite(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{label} is not a finite number: {value!r}")
    return float(value)


def _basin_flags(horizon: dict) -> tuple[bool, bool]:
    optimist = agnostic = False
    for channel in horizon.values():
        bucket = channel["1-4"]
        if not bucket["n"]:
            continue
        mean_abs = _finite(bucket["mean_abs_prediction"], "mean_abs_prediction")
        mean = _finite(bucket["mean_prediction"], "mean_prediction")
        if mean_abs < AGNOSTIC_MEAN_ABS:
            agnostic = True
        elif abs(mean) / mean_abs > OPTIMIST_RATIO:
            optimist = True
    return optimist, agnostic


def _throughput(cell: Path) -> float:
    rows = [
        json.loads(line)
        for line in (cell / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty metrics.jsonl in {cell}")
    return statistics.fmean(_finite(row["samples_per_second"], "samples_per_second") for row in rows)


def cell_metrics(
    cell_dir: str | os.PathLike[str], *, split: str = SPLIT, ema: bool = False
) -> CellMetrics:
    """Read one cell's verdict inputs from its score file of ``split``."""

    cell = Path(cell_dir)
    path = cell / scores_filename(split, ema)
    scores = json.loads(path.read_text(encoding="utf-8"))
    if scores.get("scores_format") != SCORES_FORMAT:
        raise ValueError(f"unsupported scores format {scores.get('scores_format')!r} in {path}")
    if scores.get("split") != split or bool(scores.get("ema", False)) != ema:
        raise ValueError(f"{path} does not hold split={split!r} ema={ema}")
    loss = scores["loss"]
    v_hat = scores["horizon"]["v_hat"]
    long_n = sum(int(v_hat[label]["n"]) for label in LONG_HORIZON_BUCKETS)
    long_correct = sum(
        int(v_hat[label]["n"]) * _finite(v_hat[label]["sign_accuracy"], f"{label} sign_accuracy")
        for label in LONG_HORIZON_BUCKETS
        if v_hat[label]["n"]
    )
    optimist, agnostic = _basin_flags(scores["horizon"])
    return CellMetrics(
        cell=str(cell),
        seed=int(scores["seed"]),
        param_count=int(scores["checkpoint"]["param_count"]),
        critic_ce=_finite(loss["critic_ce"]["overall"]["mean"], "critic_ce"),
        policy_nll=_finite(loss["policy_nll"]["overall"]["mean"], "policy_nll"),
        state_value_ce=(
            _finite(loss["state_value_ce"]["overall"]["mean"], "state_value_ce")
            if "state_value_ce" in loss
            else None
        ),
        top1=_finite(scores["imitation"]["overall"]["top1"], "top1"),
        top3=_finite(scores["imitation"]["overall"]["top3"], "top3"),
        long_horizon_sign=long_correct / long_n if long_n else float("nan"),
        optimist=optimist,
        agnostic=agnostic,
        samples_per_second=_throughput(cell),
    )


def discover_cells(arm_dir: str | os.PathLike[str]) -> list[Path]:
    """The ``s<seed>`` cell directories below an arm, in seed order."""

    root = Path(arm_dir)
    cells = sorted(
        (path for path in root.glob("s*") if path.is_dir() and path.name[1:].isdigit()),
        key=lambda path: int(path.name[1:]),
    )
    if not cells:
        raise ValueError(f"no s<seed> cells under {root}")
    return cells


def _check_seeds(cells: Sequence[CellMetrics], label: str) -> None:
    seeds = [cell.seed for cell in cells]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{label} repeats a seed: {seeds}")
    counts = {cell.param_count for cell in cells}
    if len(counts) != 1:
        raise ValueError(f"{label} mixes parameter counts: {sorted(counts)}")


@dataclass(frozen=True)
class FixtureStats:
    """The protocol's noise floor: per-metric mean and seed SD, and rho."""

    n: int
    mean_critic_ce: float
    sd_critic_ce: float
    mean_policy_nll: float
    sd_policy_nll: float
    rho: float
    mean_top1: float
    sd_top1: float
    mean_long_horizon_sign: float
    sd_long_horizon_sign: float
    samples_per_second: float
    param_count: int
    pathological_cells: int


def fixture_stats(cells: Sequence[CellMetrics]) -> FixtureStats:
    """Summarize the fixture cells; needs at least three seeds for an SD."""

    if len(cells) < 3:
        raise ValueError(f"a fixture needs at least 3 seeds, got {len(cells)}")
    _check_seeds(cells, "fixture")
    critic = [cell.critic_ce for cell in cells]
    policy = [cell.policy_nll for cell in cells]
    top1 = [cell.top1 for cell in cells]
    long_sign = [cell.long_horizon_sign for cell in cells]
    sd_c = statistics.stdev(critic)
    sd_p = statistics.stdev(policy)
    if sd_c <= 0.0 or sd_p <= 0.0:
        raise ValueError("fixture seeds are identical on a primary metric; no noise floor")
    rho = statistics.correlation(critic, policy)
    return FixtureStats(
        n=len(cells),
        mean_critic_ce=statistics.fmean(critic),
        sd_critic_ce=sd_c,
        mean_policy_nll=statistics.fmean(policy),
        sd_policy_nll=sd_p,
        rho=rho,
        mean_top1=statistics.fmean(top1),
        sd_top1=statistics.stdev(top1),
        mean_long_horizon_sign=statistics.fmean(long_sign),
        sd_long_horizon_sign=statistics.stdev(long_sign),
        samples_per_second=statistics.fmean(cell.samples_per_second for cell in cells),
        param_count=cells[0].param_count,
        pathological_cells=sum(cell.pathological for cell in cells),
    )


@dataclass(frozen=True)
class ArmVerdict:
    """One arm against the fixture."""

    arm: str
    n: int
    seeds: tuple[int, ...]
    param_count: int
    delta_critic_ce: float
    delta_policy_nll: float
    delta_top1: float
    delta_long_horizon_sign: float
    z_critic: float
    z_policy: float
    z_top1: float | None  # None when the fixture's top-1 has no spread
    S: float
    per_seed_S: tuple[float, ...]
    pathological_cells: int
    throughput_ratio: float
    verdict: str  # keep | policy-regressed | neutral | negative | pathology-prone


def _composite(z_c: float, z_p: float, rho: float) -> float:
    variance = WEIGHT_CRITIC**2 + WEIGHT_POLICY**2 + 2.0 * WEIGHT_CRITIC * WEIGHT_POLICY * rho
    if variance <= 0.0:
        raise ValueError(f"composite variance is not positive (rho={rho})")
    return (WEIGHT_CRITIC * z_c + WEIGHT_POLICY * z_p) / math.sqrt(variance)


def arm_verdict(arm: str, cells: Sequence[CellMetrics], fixture: FixtureStats) -> ArmVerdict:
    """Score one arm's cells against the fixture and classify it."""

    if not cells:
        raise ValueError(f"arm {arm!r} has no cells")
    _check_seeds(cells, f"arm {arm!r}")
    n = len(cells)
    scale = math.sqrt(1.0 / n + 1.0 / fixture.n)
    d_c = statistics.fmean(cell.critic_ce for cell in cells) - fixture.mean_critic_ce
    d_p = statistics.fmean(cell.policy_nll for cell in cells) - fixture.mean_policy_nll
    d_top1 = statistics.fmean(cell.top1 for cell in cells) - fixture.mean_top1
    d_long = (
        statistics.fmean(cell.long_horizon_sign for cell in cells)
        - fixture.mean_long_horizon_sign
    )
    z_c = -d_c / (fixture.sd_critic_ce * scale)
    z_p = -d_p / (fixture.sd_policy_nll * scale)
    z_top1 = d_top1 / (fixture.sd_top1 * scale) if fixture.sd_top1 > 0 else None
    S = _composite(z_c, z_p, fixture.rho)
    single = math.sqrt(1.0 + 1.0 / fixture.n)
    per_seed = tuple(
        _composite(
            -(cell.critic_ce - fixture.mean_critic_ce) / (fixture.sd_critic_ce * single),
            -(cell.policy_nll - fixture.mean_policy_nll) / (fixture.sd_policy_nll * single),
            fixture.rho,
        )
        for cell in cells
    )
    pathological = sum(cell.pathological for cell in cells)
    if pathological >= 2:
        verdict = "pathology-prone"
    elif S >= KEEP_S and z_p >= POLICY_FLOOR_Z:
        verdict = "keep"
    elif S >= KEEP_S:
        verdict = "policy-regressed"
    elif S <= -KEEP_S:
        verdict = "negative"
    else:
        verdict = "neutral"
    return ArmVerdict(
        arm=arm,
        n=n,
        seeds=tuple(cell.seed for cell in cells),
        param_count=cells[0].param_count,
        delta_critic_ce=d_c,
        delta_policy_nll=d_p,
        delta_top1=d_top1,
        delta_long_horizon_sign=d_long,
        z_critic=z_c,
        z_policy=z_p,
        z_top1=z_top1,
        S=S,
        per_seed_S=per_seed,
        pathological_cells=pathological,
        throughput_ratio=statistics.fmean(cell.samples_per_second for cell in cells)
        / fixture.samples_per_second,
        verdict=verdict,
    )


def screen_verdicts(
    fixture_dir: str | os.PathLike[str],
    arms: dict[str, str | os.PathLike[str]],
    *,
    split: str = SPLIT,
    ema: bool = False,
) -> dict[str, object]:
    """Read the fixture and every arm from disk; return the JSON verdict."""

    fixture_cells = [cell_metrics(path, split=split, ema=ema) for path in discover_cells(fixture_dir)]
    fixture = fixture_stats(fixture_cells)
    verdicts = {
        name: arm_verdict(
            name,
            [cell_metrics(path, split=split, ema=ema) for path in discover_cells(arm_dir)],
            fixture,
        )
        for name, arm_dir in arms.items()
    }
    return {
        "screen_format": 1,
        "split": split,
        "ema": ema,
        "weights": {"critic": WEIGHT_CRITIC, "policy": WEIGHT_POLICY},
        "keep": {"S": KEEP_S, "policy_floor_z": POLICY_FLOOR_Z},
        "fixture": {"dir": str(fixture_dir), **asdict(fixture)},
        "arms": {name: asdict(verdict) for name, verdict in verdicts.items()},
    }


def render_verdicts(result: dict[str, object]) -> str:
    """Plain-text table of the verdict JSON."""

    fixture = result["fixture"]
    lines = [
        f"fixture {fixture['dir']}  n={fixture['n']}  split {result['split']}"
        f"{'  ema' if result['ema'] else ''}",
        f"  critic CE {fixture['mean_critic_ce']:.4f} ± {fixture['sd_critic_ce']:.4f}   "
        f"policy NLL {fixture['mean_policy_nll']:.4f} ± {fixture['sd_policy_nll']:.4f}   "
        f"rho {fixture['rho']:+.2f}   top-1 {fixture['mean_top1']:.4f} ± {fixture['sd_top1']:.4f}   "
        f"long-horizon sign {fixture['mean_long_horizon_sign']:.4f}   "
        f"pathological {fixture['pathological_cells']}/{fixture['n']}",
        "arm  n  dCriticCE  dPolicyNLL  dTop1(pp)  z_c  z_p  S  per-seed S  patho  sps-ratio  verdict",
    ]
    for name, arm in result["arms"].items():
        per_seed = "/".join(f"{value:+.1f}" for value in arm["per_seed_S"])
        lines.append(
            f"{name}  {arm['n']}  {arm['delta_critic_ce']:+.4f}  {arm['delta_policy_nll']:+.4f}  "
            f"{100 * arm['delta_top1']:+.2f}  {arm['z_critic']:+.2f}  {arm['z_policy']:+.2f}  "
            f"{arm['S']:+.2f}  {per_seed}  {arm['pathological_cells']}/{arm['n']}  "
            f"{arm['throughput_ratio']:.3f}  {arm['verdict']}"
        )
    return "\n".join(lines)


def _parse_arm(value: str) -> tuple[str, str]:
    name, sep, path = value.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError(f"expected name=dir, got {value!r}")
    return name, path


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m mantisnet.lab.screen",
        description="Screen protocol v2: train a cell under the recipe, score it, or judge arms.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="train one cell under RECIPE")
    train.add_argument("--corpus", required=True)
    train.add_argument("--cell", required=True, help="cell directory (…/<arm>/s<seed>)")
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--variant", default="mantis")
    train.add_argument("model_kw", nargs="*", help="key=value overrides for the variant")

    evaluate = commands.add_parser(
        "evaluate", help="score one cell on val and test, raw and EMA weights"
    )
    evaluate.add_argument("--corpus", required=True)
    evaluate.add_argument("--cell", required=True)
    evaluate.add_argument("--device", default="cuda")

    verdict = commands.add_parser("verdict", help="judge arms against the fixture")
    verdict.add_argument("--fixture", required=True, help="fixture arm directory (s0..s5)")
    verdict.add_argument("--arm", action="append", default=[], type=_parse_arm, help="name=dir")
    verdict.add_argument("--split", default=SPLIT, choices=("val", "test"))
    verdict.add_argument("--ema", action="store_true")
    verdict.add_argument("--out", help="verdict JSON destination")

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "train":
        from .train import train_cell
        from .variants import parse_model_kw

        result = train_cell(
            args.corpus,
            args.cell,
            variant=args.variant,
            model_kw=parse_model_kw(args.model_kw),
            seed=args.seed,
            config=RECIPE,
        )
        print(json.dumps({"cell": args.cell, "param_count": result["config"]["param_count"]}))
        return
    if args.command == "evaluate":
        from .evaluate import evaluate_cell

        written = []
        compile_model = torch.device(args.device).type == "cuda"
        for split in ("val", "test"):
            for ema in (False, True):
                evaluate_cell(
                    args.cell,
                    args.corpus,
                    split=split,
                    ema=ema,
                    device=args.device,
                    compile=compile_model,
                    pair_budget=RECIPE.collect_pair_budget,
                    cell_budget=RECIPE.collect_cell_budget,
                )
                written.append(scores_filename(split, ema))
        print(json.dumps({"cell": args.cell, "scores": written}))
        return
    result = screen_verdicts(args.fixture, dict(args.arm), split=args.split, ema=args.ema)
    if args.out:
        Path(args.out).write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(render_verdicts(result))


if __name__ == "__main__":
    main()
