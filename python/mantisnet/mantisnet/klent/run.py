"""Drive KLENT iterations and persist their run artifacts.

The run directory contains the current invocation's ``config.json``, an
append-only ``invocations.jsonl``, append-only iteration events in
``metrics.jsonl``, queryable ``telemetry.db``, periodic checkpoints, and the
``status.json`` heartbeat. A resume restores the latest checkpoint; replayed
iteration numbers can therefore occur in ``metrics.jsonl``, while telemetry
replaces its replayed tail. Checkpoints contain model, optimizer, training RNG,
version identifiers, and the count of completed iterations.

``STOP`` requests a checkpoint and clean exit at the next iteration boundary.
``CHECKPOINT`` requests a checkpoint at that boundary without stopping.
``--eval-every N`` records one seat-balanced match per configured external
opponent every ``N`` completed iterations. Evaluation uses Gumbel line search
when ``--eval-sims`` is positive and raw-policy argmax when it is zero; its RNG
is separate from training.

Run from python/mantisnet:

    uv run python -m mantisnet.klent.run --out runs/pure-1 \
        --iterations 100 --games 64

Resume after an interruption with `--resume` (the latest checkpoint in the
run directory is found automatically).
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import dataclasses
import itertools
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from concurrent.futures import ThreadPoolExecutor

from ..builder import MODEL_REPR_VERSION
from ..model import MantisConfig, MantisNet, strip_legacy_knobs
from .hardware import hardware_sampler
from .selfplay import Collector, episode_samples
from .telemetry import open_telemetry
from .train import KlentConfig, collect_episodes, fit


# Each iteration churns gigabytes of short-lived episode, sample, and chunk
# allocations across freshly spawned prefetch and collate threads. glibc
# retains those freed chunks in its arenas: the trainer's anonymous memory
# grew asymptotically to ~26GB RSS + 16GB swap and earlyoom killed the run
# (2026-08-05, twice). One malloc_trim per committed iteration returns the
# retained pages to the OS for microseconds of work; MALLOC_ARENA_MAX=2 in
# the container environment bounds the arena count itself.
_LIBC = ctypes.CDLL("libc.so.6") if sys.platform == "linux" else None


def _trim_allocator() -> None:
    if _LIBC is not None:
        _LIBC.malloc_trim(0)


def _versions() -> dict:
    import hexo_py

    return {
        "MODEL_REPR_VERSION": MODEL_REPR_VERSION,
        "RULES_VERSION": hexo_py.RULES_VERSION,
        "ACTION_ORDER_VERSION": hexo_py.ACTION_ORDER_VERSION,
        "torch": torch.__version__,
    }


def _participant_config(participant) -> dict | None:
    if participant is None:
        return None
    return {
        "id": participant.id,
        "command": list(participant.command),
        "hello": {
            "checkpoint": participant.checkpoint,
            "variant": participant.variant,
        },
    }


def save_checkpoint(path: Path, model, optimizer, iteration: int, rng) -> None:
    """Everything a resume needs, atomically (write then rename)."""
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": dataclasses.asdict(model.cfg),
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

    The checkpoint records its own :class:`MantisConfig`, so knobbed runs
    load through every consumer of this path without those consumers knowing
    the knobs exist."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt["versions"] != _versions():
        raise ValueError(
            f"checkpoint versions {ckpt['versions']} != this build {_versions()}"
        )
    model = MantisNet(
        MantisConfig(**strip_legacy_knobs(ckpt["model_config"]))
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def load_checkpoint(path: Path, model, optimizer, rng=None) -> int:
    """Restore into the given model/optimizer/rng; returns iterations done.

    ``rng=None`` restores only the model and optimizer for ``--init-from``.
    Version identifiers and the model configuration must equal the running
    build's.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt["versions"] != _versions():
        raise ValueError(
            f"checkpoint versions {ckpt['versions']} != this build {_versions()}"
        )
    recorded = strip_legacy_knobs(ckpt["model_config"])
    running = dataclasses.asdict(model.cfg)
    if recorded != running:
        raise ValueError(
            f"checkpoint model config {recorded} != this run's {running}"
        )
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if rng is not None:
        rng.bit_generator.state = ckpt["rng_state"]
    return ckpt["iteration"]


def load_lab_cell(path: Path, model, model_kw: dict) -> None:
    """Initialize the model from a supervised lab cell's final checkpoint.

    The cell must be the same build and the same normalized model overrides
    as this run — a KLENT run started from supervised pretraining carries the
    trained trunk and critic head, a fresh optimizer, and iteration 0.
    """
    cell = torch.load(path, map_location="cpu", weights_only=False)
    if cell.get("lab_cell_format") != 1:
        raise ValueError(f"{path} is not a lab cell checkpoint")
    if cell["versions"] != _versions():
        raise ValueError(
            f"lab cell versions {cell['versions']} != this build {_versions()}"
        )
    if strip_legacy_knobs(cell["model_kw"]) != strip_legacy_knobs(model_kw):
        raise ValueError(
            f"lab cell model_kw {cell['model_kw']} != this run's {model_kw}"
        )
    model.load_state_dict(cell["model"])


class _Status:
    """The run's heartbeat: ``runs/<name>/status.json``, atomically replaced
    (write then rename) at most once a second.

    The format is the deck's contract (docs/DECK_SPEC.md): ``updated`` (UTC
    ISO), ``iteration`` (last committed), and one entry per lane — ``collect``
    ``{iteration, finished, quota, steps, slot_plies}``, ``fit``
    ``{iteration, chunk, chunks}``, ``eval`` ``{iteration}`` — each ``null``
    while that lane is idle. The lock serializes reports from the collection
    worker and driver. Heartbeat writes do not consume training RNG state.
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
    """Run the pipelined collect, fit, evaluate, and commit loop.

    While iteration ``i`` fits on the main thread, iteration ``i+1`` collects
    through a snapshot taken before fit ``i``. Collection owns a separate RNG
    stream. Resume restores the training RNG but starts with empty collection
    slots; in-flight episodes are not checkpointed.

    ``evaluate_fn(model, done, telemetry) -> dict`` runs every ``eval_every``
    completed iterations. Its fields join that iteration's metric event, and
    it records match detail through the supplied telemetry writer.

    ``starve_limit`` stops after that many consecutive iterations with fewer
    buffer samples than the game quota and writes a checkpoint first. Zero
    disables this guard.

    ``status.json`` and the ``STOP`` and ``CHECKPOINT`` sentinel files form the
    live process interface. Sentinels are checked and consumed once per
    committed iteration.
    """
    starved = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    status = _Status(out_dir)
    # The worker evaluates through this clone while fitting mutates the live model.
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
            # The first processed iteration completes compilation before
            # collection and fitting share the callable.
            overlap = i > start_iteration
            if overlap and i + 1 < iterations:
                pending = submit_collect(pool, i + 1)  # Collection overlaps the fit below.
            samples = [s for e in episodes for s in episode_samples(e, cfg.lam_ret, cfg.gamma)]
            metrics["buffer_samples"] = len(samples)
            if samples:

                def fit_progress(chunk, chunks, i=i):
                    status.update(fit={"iteration": i, "chunk": chunk, "chunks": chunks})

                metrics.update(fit(model, samples, optimizer, cfg, rng, fit_progress))
                status.update(fit=None)
            if not overlap and i + 1 < iterations:
                pending = submit_collect(pool, i + 1)
            # Evaluation time has its own metric and is excluded from iteration time.
            metrics["seconds"] = time.perf_counter() - t0
            done = i + 1
            if eval_every and evaluate_fn is not None and done % eval_every == 0:
                status.update(force=True, eval={"iteration": done})
                t1 = time.perf_counter()
                metrics.update(evaluate_fn(model, done, telemetry))
                metrics["eval_seconds"] = time.perf_counter() - t1
                status.update(force=True, eval=None)
            # Empty statistics become JSON null; NaN is not valid JSON.
            row = {k: None if v != v else v for k, v in metrics.items()}
            metrics_file.write(json.dumps(row, allow_nan=False) + "\n")
            metrics_file.flush()
            # Telemetry receives the same metric event and its source episodes.
            telemetry.write_iteration(row, episodes, hardware.drain())
            # The iteration's transients are dead past this point; drop the
            # references so the trim can hand their pages back to the OS.
            del episodes, samples
            _trim_allocator()
            status.update(force=True, iteration=done)
            # CHECKPOINT requests a durable artifact at this commit point.
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
            if metrics.get("eval_results"):
                rendered = []
                for result in metrics["eval_results"]:
                    entry = (
                        f"{result['opponent_name']} {result['win_rate']:.3f} "
                        f"({result['capped']}/{result['games']} capped, "
                        f"{result['forfeits']} forfeits)"
                    )
                    # Head-to-head entries carry the paired figures the
                    # anchors cannot have; an unbounded Elo stays absent.
                    if result.get("elo") is not None:
                        entry += (
                            f" elo {result['elo']:+.0f}"
                            f" p {result['sign_test_p']:.4f}"
                        )
                    rendered.append(entry)
                line += " | eval " + "; ".join(rendered)
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
            # STOP requests a durable checkpoint and a clean exit.
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
    ap.add_argument(
        "--mass-floor", type=float, default=KlentConfig.mass_floor,
        help="smallest committed mass pi' measures Q against",
    )
    ap.add_argument("--lam-ret", type=float, default=KlentConfig.lam_ret)
    ap.add_argument(
        "--gamma", type=float, default=KlentConfig.gamma,
        help="per-ply return-discount magnitude (1.0 = undiscounted)",
    )
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
        help=(
            "external-opponent matches every N iterations "
            "(0 = off; needs --sealbot and/or --eval-seat)"
        ),
    )
    ap.add_argument("--eval-games", type=int, default=64)
    ap.add_argument(
        "--sealbot", type=Path, default=None,
        help="SealBot checkout root, the external eval opponent",
    )
    ap.add_argument(
        "--eval-seat",
        type=Path,
        default=None,
        help=(
            "one-entry participant JSON naming a native subprocess "
            "evaluation seat"
        ),
    )
    ap.add_argument(
        "--eval-depth", type=int, default=None,
        help="optional SealBot depth cap for a weaker evaluation rung",
    )
    ap.add_argument(
        "--eval-time", type=float, default=0.1,
        help="SealBot's per-turn time limit (default: 0.1, uncapped search)",
    )
    ap.add_argument(
        "--eval-sims", type=int, default=32,
        help="model Gumbel line-search simulations (0 = policy argmax)",
    )
    ap.add_argument(
        "--h2h-ref", type=Path, default=None,
        help=(
            "checkpoint for an in-driver paired head-to-head every "
            "--eval-every iterations: the resolution instrument, since two "
            "independent anchored scores cannot separate what one paired "
            "match can"
        ),
    )
    ap.add_argument(
        "--h2h-pairs", type=int, default=64,
        help="shared openings per head-to-head, each played from both seats",
    )
    ap.add_argument(
        "--starve-limit", type=int, default=10,
        help="stop after N consecutive starved iterations (0 = never)",
    )
    ap.add_argument(
        "--init-from", type=Path, default=None,
        help="fork a fresh run from a checkpoint's model+optimizer (iteration 0, own seed)",
    )
    ap.add_argument(
        "--init-lab-cell", type=Path, default=None,
        help=(
            "initialize the model from a supervised lab cell's "
            "checkpoint_final.pt (fresh optimizer, iteration 0); the cell's "
            "model_kw must equal --model-kw"
        ),
    )
    ap.add_argument(
        "--model-kw", nargs="*", default=None, metavar="KEY=VALUE",
        help=(
            "MantisConfig overrides (for example h=160); recorded "
            "in the run config and every checkpoint"
        ),
    )
    ap.add_argument("--seed", type=int, default=0, help="the run's RNG seed")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--resume", action="store_true", help="continue from the run dir's latest checkpoint")
    args = ap.parse_args(argv)

    if (
        args.eval_every > 0
        and args.sealbot is None
        and args.eval_seat is None
        and args.h2h_ref is None
    ):
        raise SystemExit(
            "--eval-every > 0 needs --sealbot, --eval-seat, and/or --h2h-ref"
        )
    if args.eval_depth is not None and args.eval_depth < 1:
        raise SystemExit("--eval-depth must be >= 1")
    if args.eval_time <= 0:
        raise SystemExit("--eval-time must be > 0")
    if args.eval_sims < 0:
        raise SystemExit("--eval-sims must be >= 0")
    if args.h2h_ref is not None and args.eval_every <= 0:
        raise SystemExit("--h2h-ref plays at eval boundaries; it needs --eval-every > 0")
    if args.h2h_pairs < 2:
        raise SystemExit(
            "--h2h-pairs must be >= 2: one pair has no paired spread to estimate"
        )
    if sum(1 for x in (args.resume, args.init_from, args.init_lab_cell) if x) > 1:
        raise SystemExit("--resume, --init-from, and --init-lab-cell are exclusive")

    seat_participant = None
    if args.eval_seat is not None:
        from .seat import load_participants

        try:
            participants = load_participants(args.eval_seat, minimum=1)
        except (OSError, TypeError, ValueError) as error:
            raise SystemExit(f"--eval-seat: {error}") from error
        if len(participants) != 1:
            raise SystemExit(
                "--eval-seat must contain exactly one participant entry"
            )
        seat_participant = participants[0]

    cfg = KlentConfig(
        tau=args.tau,
        lam=args.lam,
        mass_floor=args.mass_floor,
        lam_ret=args.lam_ret,
        gamma=args.gamma,
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

    # Imported here: the lab package reaches back into klent.graft at import
    # time, so a module-level import would be circular.
    from ..lab.variants import parse_model_kw

    torch.manual_seed(args.seed)
    model_kw = parse_model_kw(args.model_kw)
    model_cfg = MantisConfig(**model_kw)
    model = MantisNet(model_cfg).to(cfg.device)
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
        if args.init_lab_cell is not None:
            load_lab_cell(args.init_lab_cell, model, model_kw)
            print(f"initialized from lab cell {args.init_lab_cell}")
    # The invocation's --lr overrides the value stored in optimizer state.
    for group in optimizer.param_groups:
        group["lr"] = cfg.lr

    out.mkdir(parents=True, exist_ok=True)
    config = {
        "klent": dataclasses.asdict(cfg),
        "model": dataclasses.asdict(model_cfg),
        "model_kw": model_kw,
        "iterations": args.iterations,
        "checkpoint_every": args.checkpoint_every,
        "eval_every": args.eval_every,
        "eval_games": args.eval_games,
        "eval_depth": args.eval_depth,
        "eval_time": args.eval_time,
        "eval_sims": args.eval_sims,
        "sealbot": str(args.sealbot) if args.sealbot else None,
        "eval_seat": {
            "source": str(args.eval_seat),
            "participant": _participant_config(seat_participant),
        }
        if args.eval_seat
        else None,
        "h2h": {"ref": str(args.h2h_ref), "pairs": args.h2h_pairs}
        if args.h2h_ref
        else None,
        "seed": args.seed,
        "init_from": str(args.init_from) if args.init_from else None,
        "init_lab_cell": str(args.init_lab_cell) if args.init_lab_cell else None,
        "versions": _versions(),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # The append-only invocation record preserves resolved knobs across resumes.
    invocation = {
        "start_iteration": start,
        "iterations": args.iterations,
        "checkpoint_every": args.checkpoint_every,
        "eval_every": args.eval_every,
        "eval_games": args.eval_games,
        "eval_depth": args.eval_depth,
        "eval_time": args.eval_time,
        "eval_sims": args.eval_sims,
        "sealbot": str(args.sealbot) if args.sealbot else None,
        "eval_seat": {
            "source": str(args.eval_seat),
            "participant": _participant_config(seat_participant),
        }
        if args.eval_seat
        else None,
        "h2h": {"ref": str(args.h2h_ref), "pairs": args.h2h_pairs}
        if args.h2h_ref
        else None,
        "starve_limit": args.starve_limit,
        "init_from": str(args.init_from) if args.init_from else None,
        "init_lab_cell": str(args.init_lab_cell) if args.init_lab_cell else None,
        "seed": args.seed,
        "klent": dataclasses.asdict(cfg),
        "model_kw": model_kw,
        "versions": _versions(),
    }
    with (out / "invocations.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(invocation) + "\n")

    evaluate_fn = None
    if args.eval_every > 0:
        from .opponents import (
            SealBotOpponent,
            SeatOpponent,
            opponent_match,
        )
        from .search import gumbel_choose
        from .sealbot import record_match
        from .train import network_evaluate

        h2h = None
        if args.h2h_ref is not None:
            from .headtohead import _audit, driver_match

            # The audit pins what the reference *is* — digest and iteration —
            # so the opponents row is a strength-defining record, not a path.
            audit = _audit(args.h2h_ref, "--h2h-ref")
            h2h = {
                "model": load_model(args.h2h_ref, cfg.device),
                "name": f"h2h:{args.h2h_ref.parent.name}/{args.h2h_ref.stem}",
                "config": {
                    "checkpoint": str(args.h2h_ref),
                    "sha256": audit["sha256"],
                    "iteration": audit["iteration"],
                    "pairs": args.h2h_pairs,
                    "sims": args.eval_sims,
                    "tau": cfg.tau,
                    "lam": cfg.lam,
                    "temperature": 1.0,
                    "opening_range": [2, 6],
                    "ply_cap": cfg.ply_cap,
                },
            }

        def evaluate_fn(m, done, telemetry):
            # Evaluation derives its RNG from (run seed, completed iteration).
            m.eval()
            choose = gumbel_choose(
                network_evaluate(m, cfg),
                tau=cfg.tau,
                lam=cfg.lam,
                sims=args.eval_sims,
            )
            opponents = []
            if args.sealbot is not None:
                opponents.append(
                    SealBotOpponent(
                        args.sealbot,
                        time_limit=args.eval_time,
                        max_depth=args.eval_depth,
                    )
                )
            if seat_participant is not None:
                opponents.append(SeatOpponent(seat_participant))

            results = []
            for opponent in opponents:
                result, per_game = opponent_match(
                    choose,
                    opponent,
                    args.eval_games,
                    cfg.ply_cap,
                    # Every anchor sees the same opening schedule and model
                    # RNG stream; adding one cannot perturb another.
                    np.random.default_rng([args.seed, done]),
                )
                record_match(
                    telemetry,
                    result,
                    per_game,
                    source="driver",
                    iteration=done,
                )
                results.append(
                    {
                        "opponent_name": result["opponent_name"],
                        "opponent_config": result["opponent_config"],
                        "score": result["score"],
                        "win_rate": result["win_rate"],
                        "capped": result["capped"],
                        "games": result["games"],
                        "forfeits": result["forfeits"],
                    }
                )
            if h2h is not None:
                result, per_game, stats = driver_match(
                    m,
                    h2h["model"],
                    h2h["name"],
                    h2h["config"],
                    cfg=cfg,
                    pairs=args.h2h_pairs,
                    sims=args.eval_sims,
                    tau=cfg.tau,
                    lam=cfg.lam,
                    ply_cap=cfg.ply_cap,
                    # The same derivation as the anchors: adding or removing
                    # an opponent perturbs no other opponent's schedule.
                    rng=np.random.default_rng([args.seed, done]),
                )
                record_match(
                    telemetry, result, per_game, source="driver", iteration=done
                )
                results.append(
                    {
                        "opponent_name": result["opponent_name"],
                        "opponent_config": result["opponent_config"],
                        "score": result["score"],
                        "win_rate": result["win_rate"],
                        "capped": result["capped"],
                        "games": result["games"],
                        "forfeits": result["forfeits"],
                        # The paired figures are what this opponent exists
                        # for; anchors have no analogue, so only h2h entries
                        # carry them.
                        "elo": result["elo"],
                        "elo_lo": result["elo_lo"],
                        "elo_hi": result["elo_hi"],
                        "sign_test_p": stats["sign_test_p"],
                        "pair_counts": stats["pair_counts"],
                    }
                )
            return {"eval_results": results}

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
