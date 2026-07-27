"""The run driver: iterations to disk.

What a run leaves behind is the run directory — `config.json` (every knob and
version the run depended on), `metrics.jsonl` (one line per iteration; the
§13 metrics are the experiment, so they persist), and periodic checkpoints
carrying model, optimizer, RNG state, and the iteration counter, so a crash
loses at most one checkpoint interval. Evaluation is deliberately absent for
now; `docs/KLENT_RUN_PLAN.md` plans it.

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
) -> None:
    """Loop `iterate`, appending metrics and checkpointing as it goes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.jsonl").open("a", encoding="utf-8") as metrics_file:
        for i in range(start_iteration, iterations):
            t0 = time.perf_counter()
            metrics = iterate(model, optimizer, cfg, rng)
            metrics["iteration"] = i
            metrics["seconds"] = time.perf_counter() - t0
            metrics_file.write(json.dumps(metrics) + "\n")
            metrics_file.flush()
            done = i + 1
            if done % checkpoint_every == 0 or done == iterations:
                save_checkpoint(
                    out_dir / f"checkpoint_{done:06d}.pt", model, optimizer, done, rng
                )
            print(
                f"iteration {i}: {metrics['seconds']:.1f}s | "
                f"f {metrics['f_seeded']:.2f}/{metrics['f_unseeded']:.2f} | "
                f"buffer {metrics['buffer_samples']} | "
                f"H/log|A| {metrics['acting_norm_entropy']:.3f}",
                flush=True,
            )


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
    ap.add_argument("--batch", type=int, default=1024, help="fitting batch size")
    ap.add_argument("--lr", type=float, default=KlentConfig.lr)
    ap.add_argument("--checkpoint-every", type=int, default=25)
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
            "seed": args.seed,
            "versions": _versions(),
        }
        (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    run_training(
        model,
        optimizer,
        cfg,
        args.iterations,
        out,
        rng,
        checkpoint_every=args.checkpoint_every,
        start_iteration=start,
    )


if __name__ == "__main__":
    main()
