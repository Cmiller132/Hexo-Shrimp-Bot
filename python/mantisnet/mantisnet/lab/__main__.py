"""Command line for the MantisNet laboratory harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _device(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--compile", action="store_true", help="use torch.compile(dynamic=True)"
    )


# The variant list is not an argparse `choices`: resolving it imports torch and
# the whole ACT package, which would be paid by `--help`.  An unknown name is
# refused by `variants.variant_spec`, whose message carries the full list.
_VARIANT_HELP = (
    "lab variant: 'mantis' for MantisNet, or one §29 MantisNet-ACT preset by "
    "name (full_act_v4, full_live_windows, ...)"
)
_ADAM_IMPLEMENTATIONS = ("auto", "fused", "foreach", "scalar")


def _adam_impl(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--adam-impl",
        choices=_ADAM_IMPLEMENTATIONS,
        default="auto",
        help=(
            "Adam execution policy; auto is fused on CUDA and scalar on CPU. "
            "Fused/foreach may differ only in last-bit reduction order"
        ),
    )


def _model_kw(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-kw",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="typed overrides for the chosen variant's configuration dataclass",
    )


def _cohort(parser: argparse.ArgumentParser, *, checkpoint_required=False) -> None:
    parser.add_argument(
        "--checkpoint", required=checkpoint_required, help="production KLENT checkpoint"
    )
    parser.add_argument("--family", help="explicit checkpoint family when structurally ambiguous")
    parser.add_argument("--corpus", help="frozen corpus path or name")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--envs", type=int, default=16, help="cohort position count")
    parser.add_argument("--steps", type=int, default=32, help="self-play lockstep depth")
    parser.add_argument("--seed", type=int, default=0)
    _device(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mantisnet.lab",
        description="Frozen-corpus supervised benchmarks and production model probes.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser("freeze", help="freeze a telemetry corpus")
    freeze.add_argument("--run", required=True, help="one source run directory")
    freeze.add_argument(
        "--iters", type=int, nargs=2, required=True, metavar=("FIRST", "LAST")
    )
    freeze.add_argument("--name", required=True)
    freeze.add_argument("--out", help="destination (default: runs/corpora/NAME)")
    freeze.add_argument("--train-samples", type=int, default=1_000_000)
    freeze.add_argument("--val-samples", type=int, default=100_000)
    freeze.add_argument("--test-samples", type=int, default=100_000)
    freeze.add_argument("--seed", type=int, default=0)
    freeze.add_argument(
        "--fractions",
        type=float,
        nargs=3,
        default=(0.90, 0.05, 0.05),
        metavar=("TRAIN", "VAL", "TEST"),
    )
    freeze.add_argument(
        "--dry-run", action="store_true", help="print selection counts; write nothing"
    )

    train = commands.add_parser("train", help="train supervised variant cells")
    train.add_argument("--corpus", required=True, help="frozen corpus path or name")
    train.add_argument("--sweep", required=True, help="sweep name or directory")
    train.add_argument("--variant", default="mantis", help=_VARIANT_HELP)
    _model_kw(train)
    train.add_argument("--cell", help="override the derived cell name")
    train.add_argument("--seeds", type=int, default=3, help="run seeds 0..N-1")
    train.add_argument("--epochs", type=int, default=8)
    train.add_argument(
        "--lr-schedule", choices=("constant", "cosine"), default="constant"
    )
    train.add_argument("--ema-decay", type=float, default=0.0)
    _adam_impl(train)
    train.add_argument(
        "--batch", dest="batch_size", type=int, help="effective optimizer batch"
    )
    train.add_argument(
        "--cell-budget",
        type=int,
        help="cap accumulation micro-chunks by MantisNet's decoder rows "
        "(memory knob; optimizer batch unchanged)",
    )
    train.add_argument(
        "--graph-cell-budget",
        type=int,
        help="cap accumulation micro-chunks by MantisNet-ACT's graph cells plus "
        "occupied cells (memory knob; optimizer batch unchanged)",
    )
    train.add_argument("--param-budget", type=int)
    train.add_argument("--param-tol", type=float, default=0.02)
    _device(train)

    evaluate = commands.add_parser("evaluate", help="score a checkpoint on a corpus")
    target = evaluate.add_mutually_exclusive_group(required=True)
    target.add_argument("--cell", help="lab cell directory")
    target.add_argument("--checkpoint", help="production KLENT checkpoint")
    evaluate.add_argument("--corpus", required=True, help="frozen corpus path or name")
    evaluate.add_argument("--split", choices=("train", "val", "test"), default="test")
    evaluate.add_argument("--out", help="required for a production checkpoint")
    evaluate.add_argument("--ema", action="store_true")
    evaluate.add_argument("--family", help="explicit checkpoint family when structurally ambiguous")
    evaluate.add_argument("--tau", type=float, default=0.1)
    evaluate.add_argument("--lam", type=float, default=0.01)
    evaluate.add_argument("--mass-floor", type=float, default=0.2)
    _device(evaluate)

    report = commands.add_parser("report", help="aggregate scores across seeds")
    report.add_argument("scores", nargs="*", help="explicit scores.json paths")
    report.add_argument("--sweep", help="sweep directory")
    report.add_argument("--out", help="report.json destination for explicit paths")
    report.add_argument(
        "--baseline",
        help="cell label every arm is differenced against, seed by seed",
    )

    bench = commands.add_parser("bench", help="benchmark production paths")
    bench_modes = bench.add_subparsers(dest="bench_mode", required=True)

    forward = bench_modes.add_parser("forward", help="builder and model forward")
    forward.add_argument("--checkpoint")
    forward.add_argument("--family")
    forward.add_argument("--corpus")
    forward.add_argument("--split", choices=("train", "val", "test"), default="test")
    forward.add_argument("--batch", dest="batch_size", type=int, default=64)
    forward.add_argument("--steps", dest="cohort_steps", type=int, default=32)
    forward.add_argument("--iters", type=int, default=10)
    forward.add_argument("--seed", type=int, default=99)
    _model_kw(forward)
    _device(forward)

    collect = bench_modes.add_parser("collect", help="instrument Collector.collect")
    collect.add_argument("--checkpoint")
    collect.add_argument("--family")
    collect.add_argument("--games", type=int, default=32)
    collect.add_argument("--envs", type=int, default=16)
    collect.add_argument("--cap", type=int, default=512)
    collect.add_argument("--seed", type=int, default=7)
    collect.add_argument("--pair-budget", type=int)
    collect.add_argument("--cell-budget", type=int)
    _model_kw(collect)
    _device(collect)

    fit = bench_modes.add_parser("fit", help="benchmark a production fit epoch")
    fit.add_argument("--checkpoint")
    fit.add_argument("--family")
    fit.add_argument("--corpus")
    fit.add_argument("--split", choices=("train", "val", "test"), default="train")
    fit.add_argument("--games", type=int, default=32)
    fit.add_argument("--envs", type=int, default=16)
    fit.add_argument("--cap", type=int, default=512)
    fit.add_argument("--seed", type=int, default=7)
    fit.add_argument("--pair-budget", type=int)
    fit.add_argument("--cell-budget", type=int)
    _adam_impl(fit)
    _model_kw(fit)
    _device(fit)

    sweep = bench_modes.add_parser("sweep", help="depth by cohort-size stage grid")
    sweep.add_argument("--checkpoint")
    sweep.add_argument("--family")
    sweep.add_argument("--depths", type=int, nargs="+", default=(20, 50, 100))
    sweep.add_argument("--cohorts", type=int, nargs="+", default=(16, 64))
    sweep.add_argument("--iters", type=int, default=3)
    sweep.add_argument("--seed", type=int, default=7)
    sweep.add_argument("--pair-budget", type=int)
    sweep.add_argument("--cell-budget", type=int)
    _model_kw(sweep)
    _device(sweep)

    profile = commands.add_parser("profile", help="attribute trunk/decode/seam stages")
    profile_modes = profile.add_subparsers(dest="profile_mode", required=True)
    for name in ("trunk", "decode", "seam"):
        mode = profile_modes.add_parser(name)
        _cohort(mode, checkpoint_required=True)
        mode.add_argument("--iters", type=int, default=5)

    fit_profile = profile_modes.add_parser(
        "fit", help="profile real optimizer steps in the fit engine"
    )
    fit_profile.add_argument("--checkpoint")
    fit_profile.add_argument("--family")
    fit_profile.add_argument("--corpus", required=True)
    fit_profile.add_argument("--split", choices=("train", "val", "test"), default="val")
    fit_profile.add_argument("--wait", type=int, default=6)
    fit_profile.add_argument("--warmup", type=int, default=2)
    fit_profile.add_argument("--active", type=int, default=8)
    fit_profile.add_argument("--seed", type=int, default=7)
    fit_profile.add_argument("--pair-budget", type=int)
    fit_profile.add_argument("--cell-budget", type=int)
    _adam_impl(fit_profile)
    _model_kw(fit_profile)
    _device(fit_profile)

    mass = commands.add_parser("mass", help="probe checkpoint-family critic mass")
    _cohort(mass, checkpoint_required=True)
    mass.set_defaults(envs=32, steps=64)
    mass.add_argument("--stride", type=int, default=16)

    check = commands.add_parser("check", help="run the model-contract battery")
    check_target = check.add_mutually_exclusive_group(required=True)
    check_target.add_argument("--checkpoint")
    check_target.add_argument("--variant", help=_VARIANT_HELP)
    check.add_argument("--family")
    _model_kw(check)
    check.add_argument("--corpus")
    check.add_argument("--split", choices=("train", "val", "test"), default="test")
    check.add_argument("--envs", type=int, default=2)
    check.add_argument("--steps", type=int, default=12)
    check.add_argument("--seed", type=int, default=0)
    _device(check)

    smoke = commands.add_parser("smoke", help="tiny CPU freeze/train/evaluate/report")
    smoke.add_argument(
        "--work-dir", help="retain artifacts here (default: a temporary directory)"
    )
    return parser


def _named_path(value: str, parent: Path) -> Path:
    path = Path(value)
    if path.exists() or path.is_absolute() or len(path.parts) > 1:
        return path
    return parent / path


def _parse_overrides(parser, values, variant: str = "mantis"):
    from .variants import parse_model_kw

    try:
        return parse_model_kw(values, variant)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))


def _json(value) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "freeze":
        from .corpus import freeze

        out = Path(args.out) if args.out else Path("runs/corpora") / args.name
        result = freeze(
            args.run,
            out,
            tuple(args.iters),
            name=args.name,
            train_samples=args.train_samples,
            val_samples=args.val_samples,
            test_samples=args.test_samples,
            seed=args.seed,
            fractions=tuple(args.fractions),
            dry_run=args.dry_run,
        )
        _json(result)
        return

    if args.command == "train":
        from .train import train_cell
        from .variants import derived_cell_name

        if args.seeds <= 0:
            parser.error("--seeds must be positive")
        model_kw = _parse_overrides(parser, args.model_kw, args.variant)
        corpus = _named_path(args.corpus, Path("runs/corpora"))
        sweep = _named_path(args.sweep, Path("runs/lab"))
        cell = args.cell or derived_cell_name(args.variant, model_kw)
        rows = []
        for seed in range(args.seeds):
            cell_dir = sweep / cell / f"s{seed}"
            trained = train_cell(
                corpus,
                cell_dir,
                variant=args.variant,
                model_kw=model_kw,
                seed=seed,
                epochs=args.epochs,
                lr_schedule=args.lr_schedule,
                ema_decay=args.ema_decay,
                adam_impl=args.adam_impl,
                device=args.device,
                compile=args.compile,
                batch_size=args.batch_size,
                cell_budget=args.cell_budget,
                graph_cell_budget=args.graph_cell_budget,
                param_budget=args.param_budget,
                param_tol=args.param_tol,
            )
            rows.append(
                {
                    "cell_dir": str(cell_dir),
                    "seed": seed,
                    "param_count": trained["config"]["param_count"],
                    "epochs": len(trained["metrics"]),
                }
            )
        _json({"sweep": str(sweep), "cell": cell, "runs": rows})
        return

    if args.command == "evaluate":
        from .evaluate import evaluate_cell, evaluate_checkpoint

        corpus = _named_path(args.corpus, Path("runs/corpora"))
        common = dict(
            split=args.split,
            device=args.device,
            compile=args.compile,
            tau=args.tau,
            lam=args.lam,
            mass_floor=args.mass_floor,
        )
        if args.cell:
            if args.family:
                parser.error("--family applies only with --checkpoint")
            result = evaluate_cell(args.cell, corpus, ema=args.ema, **common)
        else:
            if args.ema:
                parser.error("--ema applies only with --cell")
            result = evaluate_checkpoint(
                args.checkpoint, corpus, out=args.out, family=args.family, **common
            )
        _json(result)
        return

    if args.command == "report":
        from .report import build_report, report_sweep

        if bool(args.sweep) == bool(args.scores):
            parser.error("provide exactly one of --sweep or explicit scores paths")
        if args.sweep:
            report_sweep(args.sweep, baseline=args.baseline)
        else:
            build_report(args.scores, args.out or "report.json", baseline=args.baseline)
        return

    if args.command == "bench":
        from . import bench as bench_module

        values = vars(args).copy()
        values.pop("command")
        mode = values.pop("bench_mode")
        values["model_kw"] = _parse_overrides(parser, values["model_kw"])
        if values.get("corpus"):
            values["corpus"] = _named_path(values["corpus"], Path("runs/corpora"))
        fn = {
            "forward": bench_module.bench_forward,
            "collect": bench_module.bench_collect,
            "fit": bench_module.bench_fit,
            "sweep": bench_module.bench_sweep,
        }[mode]
        fn(**values)
        return

    if args.command == "profile":
        from .profile import run_profile

        values = vars(args).copy()
        values.pop("command")
        mode = values.pop("profile_mode")
        if "model_kw" in values:
            values["model_kw"] = _parse_overrides(parser, values["model_kw"])
        if values.get("corpus"):
            values["corpus"] = _named_path(values["corpus"], Path("runs/corpora"))
        run_profile(mode, **values)
        return

    if args.command == "mass":
        from .mass import probe_mass

        values = vars(args).copy()
        values.pop("command")
        if values.get("corpus"):
            values["corpus"] = _named_path(values["corpus"], Path("runs/corpora"))
        probe_mass(**values)
        return

    if args.command == "check":
        from .check import run_check

        values = vars(args).copy()
        values.pop("command")
        values["model_kw"] = _parse_overrides(
            parser, values["model_kw"], values["variant"]
        )
        if values["checkpoint"] and values["model_kw"]:
            parser.error("--model-kw applies only with --variant")
        if values.get("corpus"):
            values["corpus"] = _named_path(values["corpus"], Path("runs/corpora"))
        run_check(**values)
        return

    if args.command == "smoke":
        from .check import smoke

        smoke(args.work_dir)
        return

    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    main()
