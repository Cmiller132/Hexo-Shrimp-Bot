"""The run driver: iterations to disk.

What a run leaves behind is the run directory — `config.json` (every knob and
version the run depended on), `metrics.jsonl` (one line per iteration; the
§13 metrics are the experiment, so they persist), and periodic checkpoints
carrying model, optimizer, RNG state, and the iteration counter, so a crash
loses at most one checkpoint interval. `--eval-every N` plays the anchor
match (argmax π_θ vs the line builder at pinned noise, seat balanced) and
merges its score into that iteration's metrics row; the eval RNG is derived
from (run seed, iteration), never drawn from the training stream, so a run's
training trajectory is identical with evaluation on or off.

Run from python/mantisnet:

    uv run python -m mantisnet.klent.run --out runs/shakeout-1 \
        --iterations 100 --games 64

Resume after an interruption with `--resume` (the latest checkpoint in the
run directory is found automatically).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..builder import MODEL_REPR_VERSION
from ..model import MantisConfig, MantisNet
from .evaluate import ANCHOR_NOISE, anchor_match, argmax_choose
from .train import KlentConfig, iterate


def _versions() -> dict:
    import hexo_py

    return {
        "MODEL_REPR_VERSION": MODEL_REPR_VERSION,
        "RULES_VERSION": hexo_py.RULES_VERSION,
        "ACTION_ORDER_VERSION": hexo_py.ACTION_ORDER_VERSION,
        "torch": torch.__version__,
    }


def save_checkpoint(path: Path, model, optimizer, iteration: int, rng) -> None:
    """Everything a resume needs, atomically (write then rename)."""
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,  # iterations completed
            "rng_state": rng.bit_generator.state,
            "versions": _versions(),
        },
        tmp,
    )
    tmp.replace(path)


def load_model(path: Path, device: str = "cpu"):
    """A checkpoint's model half, version-checked — the frozen eval anchor.

    Architecture comes from the default :class:`MantisConfig`, which is what
    every run trains; a checkpoint from other shapes fails the state-dict
    load loudly rather than being adapted to."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt["versions"] != _versions():
        raise ValueError(
            f"checkpoint versions {ckpt['versions']} != this build {_versions()}"
        )
    model = MantisNet(MantisConfig()).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def load_checkpoint(path: Path, model, optimizer, rng) -> int:
    """Restore into the given model/optimizer/rng; returns iterations done.

    Version mismatches are refused: a checkpoint from other rules or another
    representation would train on silently, which is the failure worth a
    hard stop.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt["versions"] != _versions():
        raise ValueError(
            f"checkpoint versions {ckpt['versions']} != this build {_versions()}"
        )
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    rng.bit_generator.state = ckpt["rng_state"]
    return ckpt["iteration"]


def run_training(
    model,
    optimizer,
    cfg: KlentConfig,
    iterations: int,
    out_dir: Path,
    rng,
    checkpoint_every: int = 25,
    start_iteration: int = 0,
    eval_every: int = 0,
    evaluate_fn=None,
    starve_limit: int = 10,
) -> None:
    """Loop `iterate`, appending metrics and checkpointing as it goes.

    ``evaluate_fn(model, done) -> dict`` runs every ``eval_every`` completed
    iterations and its fields join that iteration's metrics row.

    ``starve_limit`` is the unattended-run guard: a policy that has collapsed
    to uniform stops finishing seeded games, and every further iteration is
    ~`ply_cap` plies of collection for an empty buffer. After this many
    consecutive iterations yielding under one sample per game, the run stops
    loudly (checkpointing first) instead of burning the night. 0 disables.
    """
    starved = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.jsonl").open("a", encoding="utf-8") as metrics_file:
        for i in range(start_iteration, iterations):
            t0 = time.perf_counter()
            metrics = iterate(model, optimizer, cfg, rng)
            metrics["iteration"] = i
            metrics["seconds"] = time.perf_counter() - t0  # eval kept out: this
            done = i + 1  # column is the recompile/leak detector and must stay flat
            if eval_every and evaluate_fn is not None and done % eval_every == 0:
                t1 = time.perf_counter()
                metrics.update(evaluate_fn(model, done))
                metrics["eval_seconds"] = time.perf_counter() - t1
            # NaN (an empty stat, e.g. f_unseeded with no unseeded games)
            # becomes null: a `NaN` token would make the file invalid JSON
            # for exactly the tools an operator points at it.
            row = {k: None if v != v else v for k, v in metrics.items()}
            metrics_file.write(json.dumps(row, allow_nan=False) + "\n")
            metrics_file.flush()
            if done % checkpoint_every == 0 or done == iterations:
                save_checkpoint(
                    out_dir / f"checkpoint_{done:06d}.pt", model, optimizer, done, rng
                )
            line = (
                f"iteration {i}: {metrics['seconds']:.1f}s | "
                f"f {metrics['f_seeded']:.2f}/{metrics['f_unseeded']:.2f} | "
                f"buffer {metrics['buffer_samples']} | "
                f"H/log|A| {metrics['acting_norm_entropy']:.3f}"
            )
            if "eval_score" in metrics:
                line += (
                    f" | eval {metrics['eval_score']:.3f}"
                    f" ({metrics['eval_capped']}/{metrics['eval_games']} capped)"
                )
            print(line, flush=True)

            starved = starved + 1 if metrics["buffer_samples"] < cfg.games_per_iteration else 0
            if starve_limit and starved >= starve_limit:
                save_checkpoint(
                    out_dir / f"checkpoint_{done:06d}.pt", model, optimizer, done, rng
                )
                print(
                    f"stopping starved: {starved} consecutive iterations under "
                    f"one sample per game — the policy has collapsed; "
                    f"checkpoint_{done:06d}.pt written",
                    flush=True,
                )
                return


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True, help="run directory")
    ap.add_argument("--iterations", type=int, required=True)
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--tau", type=float, default=KlentConfig.tau)
    ap.add_argument("--lam", type=float, default=KlentConfig.lam)
    ap.add_argument("--lam-ret", type=float, default=KlentConfig.lam_ret)
    ap.add_argument("--seed-fraction", type=float, default=1.0)
    ap.add_argument("--seed-cut", type=int, nargs=2, default=[1, 8], metavar=("LO", "HI"))
    ap.add_argument(
        "--batch", type=int, default=KlentConfig.batch_size,
        help="effective fitting batch, accumulated across packed chunks",
    )
    ap.add_argument(
        "--pair-budget", type=int, default=KlentConfig.pair_budget,
        help="attention pairs per network batch — the VRAM knob",
    )
    ap.add_argument(
        "--cell-budget", type=int, default=KlentConfig.cell_budget,
        help="legal cells (decoder rows) per network batch — the other VRAM knob",
    )
    ap.add_argument("--lr", type=float, default=KlentConfig.lr)
    ap.add_argument("--checkpoint-every", type=int, default=25)
    ap.add_argument(
        "--eval-every", type=int, default=0,
        help="anchor match every N iterations (0 = off)",
    )
    ap.add_argument("--eval-games", type=int, default=64)
    ap.add_argument(
        "--eval-anchor", type=Path, default=None,
        help="a frozen checkpoint as the eval opponent (default: the line builder)",
    )
    ap.add_argument(
        "--starve-limit", type=int, default=10,
        help="stop after N consecutive starved iterations (0 = never)",
    )
    ap.add_argument("--seed", type=int, default=0, help="the run's RNG seed")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--resume", action="store_true", help="continue from the run dir's latest checkpoint")
    args = ap.parse_args(argv)

    cfg = KlentConfig(
        tau=args.tau,
        lam=args.lam,
        lam_ret=args.lam_ret,
        ply_cap=args.cap,
        games_per_iteration=args.games,
        seed_fraction=args.seed_fraction,
        seed_cut=tuple(args.seed_cut),
        batch_size=args.batch,
        pair_budget=args.pair_budget,
        cell_budget=args.cell_budget,
        lr=args.lr,
        device=args.device,
        autocast=args.device == "cuda",
        compile=args.device == "cuda" and not args.no_compile,
    )

    out: Path = args.out
    checkpoints = sorted(out.glob("checkpoint_*.pt")) if out.exists() else []
    if args.resume:
        if not checkpoints:
            raise SystemExit(f"--resume: no checkpoints under {out}")
    elif out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty; use --resume to continue it")

    torch.manual_seed(args.seed)
    model = MantisNet(MantisConfig()).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(args.seed)

    start = 0
    if args.resume:
        start = load_checkpoint(checkpoints[-1], model, optimizer, rng)
        print(f"resumed {checkpoints[-1].name}: {start} iterations done")
    else:
        out.mkdir(parents=True, exist_ok=True)
        config = {
            "klent": dataclasses.asdict(cfg),
            "model": dataclasses.asdict(MantisConfig()),
            "iterations": args.iterations,
            "checkpoint_every": args.checkpoint_every,
            "eval_every": args.eval_every,
            "eval_games": args.eval_games,
            "eval_anchor_noise": ANCHOR_NOISE,
            "seed": args.seed,
            "versions": _versions(),
        }
        (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Every invocation records its resolved knobs: a resume may legitimately
    # change them (the seed curriculum anneals by stop-and-resume), and a
    # run whose later knobs exist only in shell history does not reproduce.
    invocation = {
        "start_iteration": start,
        "iterations": args.iterations,
        "checkpoint_every": args.checkpoint_every,
        "eval_every": args.eval_every,
        "eval_games": args.eval_games,
        "eval_anchor": str(args.eval_anchor) if args.eval_anchor else None,
        "starve_limit": args.starve_limit,
        "seed": args.seed,
        "klent": dataclasses.asdict(cfg),
        "versions": _versions(),
    }
    with (out / "invocations.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(invocation) + "\n")

    opponent = None
    if args.eval_anchor is not None:
        opponent = argmax_choose(load_model(args.eval_anchor, cfg.device), cfg.device)

    def evaluate_fn(m, done):
        # Seeded from (run seed, iteration), never the training stream: the
        # training trajectory is identical with evaluation on or off, and a
        # resumed run replays the same matches it would have played.
        return anchor_match(
            m,
            cfg.device,
            args.eval_games,
            cfg.ply_cap,
            np.random.default_rng([args.seed, done]),
            opponent=opponent,
        )

    run_training(
        model,
        optimizer,
        cfg,
        args.iterations,
        out,
        rng,
        checkpoint_every=args.checkpoint_every,
        start_iteration=start,
        eval_every=args.eval_every,
        evaluate_fn=evaluate_fn,
        starve_limit=args.starve_limit,
    )


if __name__ == "__main__":
    main()
