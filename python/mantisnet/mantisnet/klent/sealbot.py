"""SealBot evaluation CLI: single-checkpoint and checkpoint-curve modes.

    python -m mantisnet.klent.sealbot --sealbot D:/SealBot \
        --checkpoint runs/<run>/checkpoint_NNNNNN.pt --games 64 --time 0.1
    python -m mantisnet.klent.sealbot --sealbot D:/SealBot \
        --run runs/<run> --every 250 --sims 32
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from .opponents import (
    SealBotOpponent,
    _mirror,
    load_sealbot,
    opponent_match,
)
from .search import gumbel_choose
from .train import KlentConfig, network_evaluate


def sealbot_match(
    model,
    device: str,
    games: int,
    ply_cap: int,
    rng: np.random.Generator,
    time_limit: float,
    sealbot_root: Path,
    variant: str = "current",
    opening_range: tuple[int, int] = (2, 6),
    max_depth: int | None = None,
    sims: int = 0,
    tau: float = KlentConfig.tau,
    lam: float = KlentConfig.lam,
) -> tuple[dict, list[dict]]:
    """Compose a model chooser and ``SealBotOpponent`` into one match."""
    model.eval()
    cfg = KlentConfig(
        tau=tau,
        lam=lam,
        device=device,
        autocast=device == "cuda",
        compile=False,
    )
    choose = gumbel_choose(
        network_evaluate(model, cfg), tau=tau, lam=lam, sims=sims
    )
    opponent = SealBotOpponent(
        sealbot_root,
        variant=variant,
        time_limit=time_limit,
        max_depth=max_depth,
    )
    return opponent_match(
        choose, opponent, games, ply_cap, rng, opening_range=opening_range
    )


def record_match(
    telemetry,
    result: dict,
    per_game: list[dict],
    *,
    source: str,
    iteration: int | None = None,
    checkpoint: str | None = None,
) -> int:
    """Write a generic opponent match without opponent-specific columns."""
    opponent = telemetry.opponent(
        result["opponent_name"], result["opponent_config"]
    )
    return telemetry.write_eval_match(
        opponent,
        result,
        per_game,
        source=source,
        iteration=iteration,
        checkpoint=checkpoint,
        depth_mean=result["opponent_depth_mean"],
    )


def _checkpoint_iteration(path: Path) -> int | None:
    match = re.fullmatch(r"checkpoint_(\d+)", path.stem)
    return int(match.group(1)) if match else None


def _fmt(result: dict) -> str:
    finite_elo = lambda value: (  # noqa: E731
        f"{value:+.0f}"
        if math.isfinite(value)
        else ("+inf" if value > 0 else "-inf")
    )
    config = result["opponent_config"]
    depth = config["max_depth"]
    strength = f"{config['time_limit']}s, depth {depth or 'uncapped'}"
    return (
        f"vs SealBot({strength}): "
        f"{result['score']:.1f}/{result['games']} "
        f"({100 * result['win_rate']:.1f}%, "
        f"CI {100 * result['ci_lo']:.0f}-{100 * result['ci_hi']:.0f}%) "
        f"elo {finite_elo(result['elo'])} "
        f"({finite_elo(result['elo_lo'])}..{finite_elo(result['elo_hi'])}) "
        f"| P0 {result['score_as_p0']:.1f} P1 {result['score_as_p1']:.1f} "
        f"capped {result['capped']} | depth "
        f"{result['opponent_depth_mean']:.1f} "
        f"plies {result['avg_plies']:.0f} | {result['seconds']:.0f}s"
    )


def _coefficients(run_dir: Path) -> tuple[float, float]:
    path = run_dir / "config.json"
    if not path.exists():
        return KlentConfig.tau, KlentConfig.lam
    config = json.loads(path.read_text(encoding="utf-8"))
    return config["klent"]["tau"], config["klent"]["lam"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sealbot", type=Path, required=True, help="SealBot checkout root"
    )
    parser.add_argument("--variant", choices=("current", "best"), default="current")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--checkpoint", type=Path, help="one checkpoint to measure")
    target.add_argument("--run", type=Path, help="run directory: measure the curve")
    parser.add_argument(
        "--every",
        type=int,
        default=250,
        help="with --run, iteration stride between measured checkpoints",
    )
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument(
        "--time",
        type=float,
        default=0.1,
        help="SealBot seconds per turn (default: 0.1, the in-loop setting)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="optional SealBot depth cap for a weaker ladder rung",
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=0,
        help="model Gumbel line-search simulations (default: 0 = policy argmax)",
    )
    parser.add_argument("--cap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    if args.max_depth is not None and args.max_depth < 1:
        parser.error("--max-depth must be >= 1")
    if args.time <= 0:
        parser.error("--time must be > 0")
    if args.sims < 0:
        parser.error("--sims must be >= 0")

    from .run import load_model
    from .telemetry import open_telemetry

    run_dir = args.checkpoint.parent if args.checkpoint is not None else args.run
    tau, lam = _coefficients(run_dir)

    def play(path):
        model = load_model(path, args.device)
        return sealbot_match(
            model,
            args.device,
            args.games,
            args.cap,
            np.random.default_rng(args.seed),
            args.time,
            args.sealbot,
            args.variant,
            max_depth=args.max_depth,
            sims=args.sims,
            tau=tau,
            lam=lam,
        )

    if args.checkpoint is not None:
        result, per_game = play(args.checkpoint)
        print(f"{args.checkpoint.name}  {_fmt(result)}")
        print(json.dumps({"model_sims": args.sims} | result))
        if (run_dir / "config.json").exists():
            with open_telemetry(run_dir) as telemetry:
                record_match(
                    telemetry,
                    result,
                    per_game,
                    source="cli",
                    iteration=_checkpoint_iteration(args.checkpoint),
                    checkpoint=args.checkpoint.name,
                )
            print(f"recorded in {run_dir / 'telemetry.db'}")
        else:
            print(f"{run_dir} is not a run directory: not recorded")
        return

    checkpoints = sorted(args.run.glob("checkpoint_*.pt"))
    if not checkpoints:
        raise SystemExit(f"no checkpoints in {args.run}")
    picked = [
        path
        for path in checkpoints
        if int(re.search(r"(\d+)", path.stem).group(1)) % args.every == 0
    ]
    if checkpoints[-1] not in picked:
        picked.append(checkpoints[-1])

    out = args.run / "sealbot_curve.jsonl"
    with (
        open(out, "w", encoding="utf-8") as file,
        open_telemetry(args.run) as telemetry,
    ):
        for path in picked:
            # ``play`` recreates the same eval RNG for every checkpoint, so
            # openings and per-root Gumbel streams pair the whole curve.
            result, per_game = play(path)
            row = {"checkpoint": path.name, "model_sims": args.sims} | result
            file.write(json.dumps(row) + "\n")
            file.flush()
            record_match(
                telemetry,
                result,
                per_game,
                source="cli",
                iteration=_checkpoint_iteration(path),
                checkpoint=path.name,
            )
            print(f"{path.name}  {_fmt(result)}", flush=True)
    print(f"wrote {out} and {args.run / 'telemetry.db'}")


if __name__ == "__main__":
    main()
