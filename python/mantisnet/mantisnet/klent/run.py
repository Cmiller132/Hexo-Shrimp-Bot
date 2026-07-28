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
import copy
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch

from concurrent.futures import ThreadPoolExecutor

from ..builder import MODEL_REPR_VERSION
from ..model import MantisConfig, MantisNet
from .evaluate import ANCHOR_NOISE, anchor_match, argmax_choose
from .selfplay import episode_samples
from .train import KlentConfig, collect_episodes, fit, generate_prefixes, ground_count


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


def load_checkpoint(path: Path, model, optimizer, rng=None) -> int:
    """Restore into the given model/optimizer/rng; returns iterations done.

    ``rng=None`` restores the learner halves only — how ``--init-from``
    forks a fresh run (own seed, iteration 0) from a trained state.

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
    if rng is not None:
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
    warm_iterations: int = 0,
    anneal: bool = False,
    opponent=None,
) -> None:
    """The pipelined loop: while iteration ``i`` fits on the main thread,
    iteration ``i+1``'s prefixes generate and its episodes collect on a
    worker, acting through a snapshot of the weights as they stood *before*
    fit ``i``. The corpus therefore runs one fit behind the paper's strict
    alternation — the deliberate cost of overlapping the loop's CPU half
    with its GPU half, recorded here because it is an algorithmic property,
    not an implementation detail. Collection draws from its own
    per-iteration stream (seeded off the main one at submission), so a
    resumed run replays the same corpus.

    ``evaluate_fn(model, done) -> dict`` runs every ``eval_every`` completed
    iterations and its fields join that iteration's metrics row.

    ``starve_limit`` is the unattended-run guard: a policy that has collapsed
    to uniform stops finishing seeded games, and every further iteration is
    ~`ply_cap` plies of collection for an empty buffer. After this many
    consecutive iterations yielding under one sample per game, the run stops
    loudly (checkpointing first) instead of burning the night. 0 disables.

    ``anneal`` is the design doc's §5.2 requirement made mechanical: the
    seed-cut ceiling deepens while seeded games keep terminating (f ≥ 0.8)
    and backs off when they stop (f < 0.3), so the corpus walks backward
    from the endgame instead of parking on trivial near-terminal stubs —
    measured to happen with a static cut: self-play stays perfect while
    strength against a real opponent dies. The live ceiling is recorded in
    every metrics row as ``seed_cut_hi``, and it is applied to the
    *collection being submitted*, so a walk step reaches the corpus one
    iteration later.
    """
    starved = 0
    cut_lo, cut_hi = cfg.seed_cut
    out_dir.mkdir(parents=True, exist_ok=True)
    # The collection actor: a persistent clone the worker forwards through
    # while fit mutates the live model. Reloaded (cheap, on-device copies)
    # at each submission, when the worker is idle by construction.
    snapshot = copy.deepcopy(model)

    def submit_collect(pool, for_iteration):
        warm = for_iteration < warm_iterations
        frozen = dataclasses.replace(cfg)  # the anneal mutates cfg between iterations
        prefix_seed = int(rng.integers(2**63))
        play_seed = int(rng.integers(2**63))
        if not warm:
            snapshot.load_state_dict(model.state_dict())

        def job():
            prefixes = generate_prefixes(
                prefix_seed,
                frozen.games_per_iteration - ground_count(frozen, warm),
                frozen.seed_fraction,
                frozen.seed_cut,
                frozen.seed_noise,
            )
            return collect_episodes(
                snapshot, frozen, np.random.default_rng(play_seed),
                warm=warm, prefixes=prefixes, opponent=opponent,
            ), warm

        return pool.submit(job)

    with (
        (out_dir / "metrics.jsonl").open("a", encoding="utf-8") as metrics_file,
        ThreadPoolExecutor(max_workers=1) as pool,
    ):
        pending = submit_collect(pool, start_iteration)
        for i in range(start_iteration, iterations):
            t0 = time.perf_counter()
            (episodes, metrics), was_warm = pending.result()
            metrics["iteration"] = i
            if anneal and i >= warm_iterations:
                f = metrics["f_seeded"]
                if f >= 0.8:
                    cut_hi = min(cut_hi + 2, cfg.ply_cap)
                elif f < 0.3:
                    cut_hi = max(cut_lo, cut_hi - 4)
                cfg.seed_cut = (cut_lo, cut_hi)
            metrics["seed_cut_hi"] = cut_hi
            # The first processed iteration stays sequential: its fit is the
            # train-graph torch.compile, and a concurrent collection hitting
            # the same compiled callable mid-compile was measured to livelock
            # the two threads on the GIL. From the second iteration on, both
            # graphs exist and collection overlaps fit.
            overlap = i > start_iteration
            if overlap and i + 1 < iterations:
                pending = submit_collect(pool, i + 1)  # overlaps the fit below
            samples = [
                s
                for e in episodes
                for s in episode_samples(e, 1.0 if was_warm else cfg.lam_ret)
            ]
            metrics["buffer_samples"] = len(samples)
            if samples:
                metrics.update(fit(model, samples, optimizer, cfg, rng))
            if not overlap and i + 1 < iterations:
                pending = submit_collect(pool, i + 1)
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
            gnd = metrics.get("grounded_score")
            if gnd is not None and gnd == gnd:
                line += f" | gnd {gnd:.2f}"
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
    ap.add_argument(
        "--warm-iterations", type=int, default=0,
        help="bootstrap iterations acting through the line builder's scores",
    )
    ap.add_argument(
        "--anneal", action="store_true",
        help="deepen the seed cut while f_seeded holds (design doc §5.2)",
    )
    ap.add_argument(
        "--ground-fraction", type=float, default=0.0,
        help="fraction of each iteration's games grounded against SealBot",
    )
    ap.add_argument(
        "--sealbot", type=Path, default=None,
        help="SealBot checkout root (required when --ground-fraction > 0)",
    )
    ap.add_argument(
        "--ground-depth", type=int, default=1,
        help="the grounding opponent's search-depth cap",
    )
    ap.add_argument(
        "--ground-time", type=float, default=0.05,
        help="the grounding opponent's per-turn time limit (rarely binds under the depth cap)",
    )
    ap.add_argument(
        "--init-from", type=Path, default=None,
        help="fork a fresh run from a checkpoint's model+optimizer (iteration 0, own seed)",
    )
    ap.add_argument("--seed", type=int, default=0, help="the run's RNG seed")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--resume", action="store_true", help="continue from the run dir's latest checkpoint")
    args = ap.parse_args(argv)

    if args.ground_fraction > 0 and args.sealbot is None:
        raise SystemExit("--ground-fraction > 0 needs --sealbot")
    if args.resume and args.init_from is not None:
        raise SystemExit("--resume and --init-from are exclusive")

    cfg = KlentConfig(
        tau=args.tau,
        lam=args.lam,
        lam_ret=args.lam_ret,
        ply_cap=args.cap,
        games_per_iteration=args.games,
        seed_fraction=args.seed_fraction,
        seed_cut=tuple(args.seed_cut),
        ground_fraction=args.ground_fraction,
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
        if args.init_from is not None:
            forked = load_checkpoint(args.init_from, model, optimizer)
            print(f"initialized from {args.init_from} ({forked} iterations trained)")
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
            "init_from": str(args.init_from) if args.init_from else None,
            "sealbot": str(args.sealbot) if args.sealbot else None,
            "ground_depth": args.ground_depth,
            "ground_time": args.ground_time,
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
        "warm_iterations": args.warm_iterations,
        "anneal": args.anneal,
        "init_from": str(args.init_from) if args.init_from else None,
        "sealbot": str(args.sealbot) if args.sealbot else None,
        "ground_depth": args.ground_depth,
        "ground_time": args.ground_time,
        "seed": args.seed,
        "klent": dataclasses.asdict(cfg),
        "versions": _versions(),
    }
    with (out / "invocations.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(invocation) + "\n")

    opponent = None
    if args.eval_anchor is not None:
        opponent = argmax_choose(load_model(args.eval_anchor, cfg.device), cfg.device)

    ground_opponent = None
    if args.ground_fraction > 0:
        from .sealbot import sealbot_opponent

        ground_opponent = sealbot_opponent(
            args.sealbot, depth=args.ground_depth, time_limit=args.ground_time
        )

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
        warm_iterations=args.warm_iterations,
        anneal=args.anneal,
        opponent=ground_opponent,
    )


if __name__ == "__main__":
    main()
