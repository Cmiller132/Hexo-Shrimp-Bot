"""The run driver: iterations to disk.

What a run leaves behind is the run directory — `config.json` (every knob and
version the run depended on), `metrics.jsonl` (one line per iteration; the
§13 metrics are the experiment, so they persist), `telemetry.db` (the same
metrics plus every game, every self-play ply, every evaluation match and the
machine's counters, queryable — see `telemetry.py`), and periodic checkpoints
carrying model, optimizer, RNG state, and the iteration counter, so a crash
loses at most one checkpoint interval. While it lives it also keeps
`status.json` fresh (the per-lane heartbeat) and honors the `STOP` /
`CHECKPOINT` sentinel files once per iteration. `--eval-every N` plays a seat-balanced
paired match against SealBot — the independent external yardstick, and the
only evaluation — and merges its score into that iteration's metrics row;
the eval RNG is derived from (run seed, iteration), never drawn from the
training stream, so a run's training trajectory is identical with evaluation
on or off.

Run from python/mantisnet:

    uv run python -m mantisnet.klent.run --out runs/pure-1 \
        --iterations 100 --games 64

Resume after an interruption with `--resume` (the latest checkpoint in the
run directory is found automatically).
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import itertools
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from concurrent.futures import ThreadPoolExecutor

from ..builder import MODEL_REPR_VERSION
from ..model import MantisConfig, MantisNet
from .hardware import hardware_sampler
from .selfplay import Collector, episode_samples
from .telemetry import open_telemetry
from .train import KlentConfig, collect_episodes, fit


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
    """A checkpoint's model half, version-checked.

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


class _Status:
    """The run's heartbeat: ``runs/<name>/status.json``, atomically replaced
    (write then rename) at most once a second.

    The format is the deck's contract (docs/DECK_SPEC.md): ``updated`` (UTC
    ISO), ``iteration`` (last committed), and one entry per lane — ``collect``
    ``{iteration, finished, quota, steps, slot_plies}``, ``fit``
    ``{iteration, chunk, chunks}``, ``eval`` ``{iteration}`` — each ``null``
    while that lane is idle. Lanes report from two threads (the collection
    worker and the driver), hence the lock; a write draws nothing from the
    training RNG and adds nothing to the metrics row, exactly like telemetry.
    """

    def __init__(self, out_dir: Path):
        self.path = out_dir / "status.json"
        self._lock = threading.Lock()
        self._state = {"updated": None, "iteration": None,
                       "collect": None, "fit": None, "eval": None}
        self._written = 0.0

    def update(self, force: bool = False, **lanes) -> None:
        with self._lock:
            self._state.update(lanes)
            now = time.monotonic()
            if not force and now - self._written < 1.0:
                return
            self._written = now
            self._state["updated"] = datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            )
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._state), encoding="utf-8")
            tmp.replace(self.path)


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
    """The pipelined loop: while iteration ``i`` fits on the main thread,
    iteration ``i+1``'s episodes collect on a worker, acting through a
    snapshot of the weights as they stood *before* fit ``i``. The corpus
    therefore runs one fit behind the paper's strict alternation — the
    deliberate cost of overlapping the loop's CPU half with its GPU half,
    recorded here because it is an algorithmic property, not an
    implementation detail. Collection draws from the collector's own stream
    (seeded off the main one at construction), so a from-scratch run is
    reproducible; a *resumed* run restarts with empty slots, so it replays
    the same distribution but not the same games — in-flight episodes do
    not survive the process.

    ``evaluate_fn(model, done, telemetry) -> dict`` runs every ``eval_every``
    completed iterations and its fields join that iteration's metrics row; it
    is handed the telemetry writer because a match's per-game detail is
    recorded by whoever played it, through the same call the offline sweep
    uses.

    Telemetry is written on this thread, between the fit and the wait for the
    next collection — it draws nothing from the training RNG and adds nothing
    to the metrics row, so a run's trajectory is identical with it removed
    (which is what `tests/test_telemetry.py` holds it to). It is not optional:
    a run that cannot record what it did is not a run worth having.

    ``starve_limit`` is the unattended-run guard: a policy that stops
    finishing games buys ~`ply_cap` plies of collection per iteration for an
    empty buffer. After this many consecutive iterations yielding under one
    sample per game, the run stops loudly (checkpointing first) instead of
    burning the night. 0 disables.

    Two file seams face outward while the run lives. ``status.json`` is the
    :class:`_Status` heartbeat. And two sentinel files in the run directory
    are honored once per iteration: ``STOP`` (checkpoint, consume, exit 0)
    and ``CHECKPOINT`` (write a checkpoint at the next commit point,
    consume, continue) — how anything outside the process, the deck
    included, asks politely instead of killing.
    """
    starved = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    status = _Status(out_dir)
    # The collection actor: a persistent clone the worker forwards through
    # while fit mutates the live model. Reloaded (cheap, on-device copies)
    # at each submission, when the worker is idle by construction.
    snapshot = copy.deepcopy(model)
    collector = Collector(
        cfg.envs,
        cfg.ply_cap,
        cfg.tau,
        cfg.lam,
        np.random.default_rng(int(rng.integers(2**63))),
        pair_budget=cfg.collect_pair_budget,
        cell_budget=cfg.collect_cell_budget,
    )

    def submit_collect(pool, iteration):
        snapshot.load_state_dict(model.state_dict())
        steps = itertools.count(1)

        def progress(finished, quota, slot_plies):
            status.update(collect={
                "iteration": iteration, "finished": finished, "quota": quota,
                "steps": next(steps), "slot_plies": slot_plies,
            })

        return pool.submit(collect_episodes, snapshot, collector, cfg, progress)

    with (
        (out_dir / "metrics.jsonl").open("a", encoding="utf-8") as metrics_file,
        open_telemetry(out_dir) as telemetry,
        hardware_sampler(cfg.device) as hardware,
        ThreadPoolExecutor(max_workers=1) as pool,
    ):
        telemetry.begin_run(
            config={
                "klent": dataclasses.asdict(cfg),
                "iterations": iterations,
                "checkpoint_every": checkpoint_every,
                "eval_every": eval_every,
                "starve_limit": starve_limit,
            },
            versions=_versions(),
            start_iteration=start_iteration,
        )
        def finish():
            status.update(force=True, collect=None, fit=None, eval=None)

        pending = submit_collect(pool, start_iteration)
        for i in range(start_iteration, iterations):
            t0 = time.perf_counter()
            episodes, metrics = pending.result()
            metrics["iteration"] = i
            # The first processed iteration stays sequential: its fit is the
            # train-graph torch.compile, and a concurrent collection hitting
            # the same compiled callable mid-compile was measured to livelock
            # the two threads on the GIL. From the second iteration on, both
            # graphs exist and collection overlaps fit.
            overlap = i > start_iteration
            if overlap and i + 1 < iterations:
                pending = submit_collect(pool, i + 1)  # overlaps the fit below
            samples = [s for e in episodes for s in episode_samples(e, cfg.lam_ret)]
            metrics["buffer_samples"] = len(samples)
            if samples:

                def fit_progress(chunk, chunks, i=i):
                    status.update(fit={"iteration": i, "chunk": chunk, "chunks": chunks})

                metrics.update(fit(model, samples, optimizer, cfg, rng, fit_progress))
                status.update(fit=None)
            if not overlap and i + 1 < iterations:
                pending = submit_collect(pool, i + 1)
            metrics["seconds"] = time.perf_counter() - t0  # eval kept out: this
            done = i + 1  # column is the recompile/leak detector and must stay flat
            if eval_every and evaluate_fn is not None and done % eval_every == 0:
                status.update(force=True, eval={"iteration": done})
                t1 = time.perf_counter()
                metrics.update(evaluate_fn(model, done, telemetry))
                metrics["eval_seconds"] = time.perf_counter() - t1
                status.update(force=True, eval=None)
            # NaN (an empty stat, e.g. won_length_mean with nothing won)
            # becomes null: a `NaN` token would make the file invalid JSON
            # for exactly the tools an operator points at it.
            row = {k: None if v != v else v for k, v in metrics.items()}
            metrics_file.write(json.dumps(row, allow_nan=False) + "\n")
            metrics_file.flush()
            # The same row the file just took, plus the episodes behind it.
            telemetry.write_iteration(row, episodes, hardware.drain())
            status.update(force=True, iteration=done)
            # CHECKPOINT is the on-demand sentinel: whoever wants a durable
            # artifact now (the deck, an operator) touches the file and the
            # next commit point honors and consumes it.
            requested = (out_dir / "CHECKPOINT").exists()
            if done % checkpoint_every == 0 or done == iterations or requested:
                save_checkpoint(
                    out_dir / f"checkpoint_{done:06d}.pt", model, optimizer, done, rng
                )
                if requested:
                    (out_dir / "CHECKPOINT").unlink()
                    print(
                        f"checkpoint_{done:06d}.pt written (CHECKPOINT sentinel)",
                        flush=True,
                    )
            line = (
                f"iteration {i}: {metrics['seconds']:.1f}s | "
                f"f {metrics['f']:.2f} | "
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
                    f"one sample per game — games are not finishing; "
                    f"checkpoint_{done:06d}.pt written",
                    flush=True,
                )
                finish()
                return
            # STOP is the graceful-stop sentinel: finish the iteration, make
            # the state durable, consume the request, exit clean.
            if (out_dir / "STOP").exists():
                save_checkpoint(
                    out_dir / f"checkpoint_{done:06d}.pt", model, optimizer, done, rng
                )
                (out_dir / "STOP").unlink()
                print(
                    f"stopping: STOP sentinel honored — "
                    f"checkpoint_{done:06d}.pt written",
                    flush=True,
                )
                finish()
                return
        finish()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True, help="run directory")
    ap.add_argument("--iterations", type=int, required=True)
    ap.add_argument(
        "--games", type=int, default=KlentConfig.games_per_iteration,
        help="finished games per iteration (the completion quota)",
    )
    ap.add_argument(
        "--envs", type=int, default=KlentConfig.envs,
        help="persistent self-play slots stepped in lockstep",
    )
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--tau", type=float, default=KlentConfig.tau)
    ap.add_argument("--lam", type=float, default=KlentConfig.lam)
    ap.add_argument("--lam-ret", type=float, default=KlentConfig.lam_ret)
    ap.add_argument(
        "--batch", type=int, default=KlentConfig.batch_size,
        help="effective fitting batch, accumulated across packed chunks",
    )
    ap.add_argument(
        "--pair-budget", type=int, default=KlentConfig.pair_budget,
        help="attention pairs per fit batch — the VRAM knob",
    )
    ap.add_argument(
        "--cell-budget", type=int, default=KlentConfig.cell_budget,
        help="legal cells (decoder rows) per fit batch — the other VRAM knob",
    )
    ap.add_argument(
        "--collect-pair-budget", type=int, default=KlentConfig.collect_pair_budget,
        help="attention pairs per collection batch (no_grad, so larger)",
    )
    ap.add_argument(
        "--collect-cell-budget", type=int, default=KlentConfig.collect_cell_budget,
        help="legal cells per collection batch (no_grad, so larger)",
    )
    ap.add_argument("--lr", type=float, default=KlentConfig.lr)
    ap.add_argument("--checkpoint-every", type=int, default=25)
    ap.add_argument(
        "--eval-every", type=int, default=0,
        help="SealBot match every N iterations (0 = off; needs --sealbot)",
    )
    ap.add_argument("--eval-games", type=int, default=64)
    ap.add_argument(
        "--sealbot", type=Path, default=None,
        help="SealBot checkout root, the external eval opponent",
    )
    ap.add_argument(
        "--eval-depth", type=int, default=1,
        help="SealBot's search-depth cap during eval matches",
    )
    ap.add_argument(
        "--eval-time", type=float, default=0.05,
        help="SealBot's per-turn time limit (rarely binds under the depth cap)",
    )
    ap.add_argument(
        "--starve-limit", type=int, default=10,
        help="stop after N consecutive starved iterations (0 = never)",
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

    if args.eval_every > 0 and args.sealbot is None:
        raise SystemExit("--eval-every > 0 needs --sealbot")
    if args.resume and args.init_from is not None:
        raise SystemExit("--resume and --init-from are exclusive")

    cfg = KlentConfig(
        tau=args.tau,
        lam=args.lam,
        lam_ret=args.lam_ret,
        ply_cap=args.cap,
        games_per_iteration=args.games,
        envs=args.envs,
        batch_size=args.batch,
        pair_budget=args.pair_budget,
        cell_budget=args.cell_budget,
        collect_pair_budget=args.collect_pair_budget,
        collect_cell_budget=args.collect_cell_budget,
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
    # A restored optimizer state carries the *source run's* learning rate in
    # its param groups; the invocation's --lr is the truth, not that relic.
    for group in optimizer.param_groups:
        group["lr"] = cfg.lr

    out.mkdir(parents=True, exist_ok=True)
    config = {
        "klent": dataclasses.asdict(cfg),
        "model": dataclasses.asdict(MantisConfig()),
        "iterations": args.iterations,
        "checkpoint_every": args.checkpoint_every,
        "eval_every": args.eval_every,
        "eval_games": args.eval_games,
        "eval_depth": args.eval_depth,
        "eval_time": args.eval_time,
        "sealbot": str(args.sealbot) if args.sealbot else None,
        "seed": args.seed,
        "init_from": str(args.init_from) if args.init_from else None,
        "versions": _versions(),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Every invocation records its resolved knobs: a resume may legitimately
    # change them, and a run whose later knobs exist only in shell history
    # does not reproduce.
    invocation = {
        "start_iteration": start,
        "iterations": args.iterations,
        "checkpoint_every": args.checkpoint_every,
        "eval_every": args.eval_every,
        "eval_games": args.eval_games,
        "eval_depth": args.eval_depth,
        "eval_time": args.eval_time,
        "sealbot": str(args.sealbot) if args.sealbot else None,
        "starve_limit": args.starve_limit,
        "init_from": str(args.init_from) if args.init_from else None,
        "seed": args.seed,
        "klent": dataclasses.asdict(cfg),
        "versions": _versions(),
    }
    with (out / "invocations.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(invocation) + "\n")

    evaluate_fn = None
    if args.eval_every > 0:
        from .sealbot import record_match, sealbot_match

        def evaluate_fn(m, done, telemetry):
            # Seeded from (run seed, iteration), never the training stream:
            # the training trajectory is identical with evaluation on or off,
            # and a resumed run replays the same matches it would have played.
            result, per_game = sealbot_match(
                m,
                cfg.device,
                args.eval_games,
                cfg.ply_cap,
                np.random.default_rng([args.seed, done]),
                args.eval_time,
                args.sealbot,
                max_depth=args.eval_depth,
            )
            record_match(
                telemetry, result, per_game,
                variant="current", source="driver", iteration=done,
            )
            return {
                "eval_score": result["win_rate"],
                "eval_capped": result["capped"],
                "eval_games": result["games"],
            }

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
