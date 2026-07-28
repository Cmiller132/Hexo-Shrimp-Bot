"""Production-workflow benchmarks for the KLENT loop.

Three modes, each replicating the real path rather than a proxy of it:

``sweep``   Stage costs — Rust collate, network forward, improve+sample —
            over a (stone count × cohort size) grid, chunked under the real
            memory budgets exactly as collection chunks them. Answers
            "what does a position of T stones cost at cohort size B, and
            in which stage".

``collect`` A real ``play_episodes`` run (checkpoint or scripted evaluator),
            instrumented per lockstep step: live-cohort trace, per-phase
            time totals, game-length distribution. Answers "where does an
            iteration's wall clock actually go", including the drain tail
            where a few long survivors run at tiny cohort sizes.

``fit``     A real ``fit`` epoch over the corpus ``collect`` produced.

Run from python/mantisnet (the GPU must be free — stop any training run):

    uv run python bench/bench_loop.py sweep --device cuda --compile
    uv run python bench/bench_loop.py collect --checkpoint runs/<r>/checkpoint_N.pt \
        --games 512 --cap 512
    uv run python bench/bench_loop.py fit --checkpoint runs/<r>/checkpoint_N.pt \
        --games 256 --cap 512

Phase timing in ``collect`` wraps the seams ``play_episodes`` already calls
(`_chunk_live`, `collate_positions`, the evaluator, `improved_policy`) so
the production loop itself stays uninstrumented; the residual after those
phases is engine advance plus Python bookkeeping.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np
import torch

import hexo_py

from mantisnet.builder import collate_positions
from mantisnet.klent import improve as improve_mod
from mantisnet.klent import selfplay as selfplay_mod
from mantisnet.klent.improve import improved_policy
from mantisnet.klent.selfplay import episode_samples, play_episodes
from mantisnet.klent.train import KlentConfig, fit, network_evaluate


def positions_at(stones: int, count: int, seed: int) -> list[hexo_py.Position]:
    """``count`` live positions of exactly ``stones`` stones, by uniform
    random legal playout (which essentially never terminates — the property
    that makes long positions generable at all)."""
    rng = np.random.default_rng(seed)
    out: list[hexo_py.Position] = []
    while len(out) < count:
        pos = hexo_py.Position()
        for _ in range(stones):
            if pos.is_terminal:
                break
            pos.advance(*pos.nth_legal(int(rng.integers(pos.legal_count))))
        if not pos.is_terminal:
            out.append(pos)
    return out


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _vram_reset(device: str) -> None:
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()


def _vram_peak_gib(device: str) -> float:
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 2**30
    return 0.0


def bench_sweep(args) -> None:
    cfg = KlentConfig(
        device=args.device,
        autocast=args.device == "cuda",
        compile=args.compile,
        pair_budget=args.pair_budget,
        cell_budget=args.cell_budget,
    )
    evaluate = network_evaluate(_load_or_fresh(args), cfg)

    print(
        f"device {args.device} compile {args.compile} | budgets: "
        f"{cfg.pair_budget:,} pairs, {cfg.cell_budget:,} cells"
    )
    header = (
        "stones cohort chunks  legal/pos   collate      forward   improve+samp"
        "     total     pos/s  peakGiB"
    )
    print(header)
    for stones in args.stones:
        pool = positions_at(stones, max(args.cohorts), args.seed)
        for cohort in args.cohorts:
            positions = pool[:cohort]
            chunks = selfplay_mod._chunk_live(
                positions, list(range(cohort)), cfg.pair_budget, cfg.cell_budget
            )
            # Warm every shape bucket once (compile/autotune outside the timing).
            for chunk in chunks:
                batch = collate_positions([positions[i] for i in chunk])
                evaluate(batch)
            _sync(args.device)
            _vram_reset(args.device)

            t_collate = t_forward = t_improve = 0.0
            legal = 0
            rng = np.random.default_rng(0)
            for _ in range(args.iters):
                for chunk in chunks:
                    t0 = time.perf_counter()
                    batch = collate_positions([positions[i] for i in chunk])
                    t1 = time.perf_counter()
                    policy, q = evaluate(batch)
                    t2 = time.perf_counter()
                    imp = improved_policy(policy, q, batch.legal_offsets, 0.1, 0.03)
                    offsets = batch.legal_offsets.numpy()
                    flat = imp.probs.numpy().astype(np.float64)
                    widths = np.diff(offsets)
                    flat /= np.repeat(np.add.reduceat(flat, offsets[:-1]), widths)
                    cdf = np.cumsum(flat)
                    base = np.concatenate(([0.0], cdf[offsets[1:-1] - 1]))
                    ranks = np.searchsorted(cdf, base + rng.random(len(chunk))) - offsets[:-1]
                    np.clip(ranks, 0, widths - 1)
                    t3 = time.perf_counter()
                    t_collate += t1 - t0
                    t_forward += t2 - t1
                    t_improve += t3 - t2
                legal = sum(p.legal_count for p in positions)
            n = args.iters
            total = (t_collate + t_forward + t_improve) / n
            print(
                f"{stones:6d} {cohort:6d} {len(chunks):6d} {legal / cohort:10.0f}"
                f" {t_collate / n * 1e3:8.1f}ms {t_forward / n * 1e3:10.1f}ms"
                f" {t_improve / n * 1e3:11.1f}ms {total * 1e3:8.1f}ms"
                f" {cohort / total:9.0f} {_vram_peak_gib(args.device):8.2f}"
            )


class _PhaseTimer:
    """Wraps the seams ``play_episodes`` calls, accumulating per-phase wall
    clock and a per-step live-cohort trace. The evaluator's ``.cpu()`` copy
    already synchronizes the device, so wall clock is honest attribution."""

    def __init__(self, evaluate):
        self._evaluate = evaluate
        self.phases = {"chunk": 0.0, "collate": 0.0, "network": 0.0, "improve": 0.0}
        self.steps: list[tuple[int, float]] = []  # (live count, step-start time)
        self._chunk_live = selfplay_mod._chunk_live
        self._collate = selfplay_mod.collate_positions
        self._improved = selfplay_mod.improved_policy

    def evaluate(self, batch):
        t0 = time.perf_counter()
        out = self._evaluate(batch)
        self.phases["network"] += time.perf_counter() - t0
        return out

    def install(self):
        def chunk_live(positions, live, pair_budget, cell_budget):
            self.steps.append((len(live), time.perf_counter()))
            t0 = time.perf_counter()
            out = self._chunk_live(positions, live, pair_budget, cell_budget)
            self.phases["chunk"] += time.perf_counter() - t0
            return out

        def collate(positions):
            t0 = time.perf_counter()
            out = self._collate(positions)
            self.phases["collate"] += time.perf_counter() - t0
            return out

        def improved(policy, q, offsets, tau, lam):
            t0 = time.perf_counter()
            out = self._improved(policy, q, offsets, tau, lam)
            self.phases["improve"] += time.perf_counter() - t0
            return out

        selfplay_mod._chunk_live = chunk_live
        selfplay_mod.collate_positions = collate
        selfplay_mod.improved_policy = improved

    def uninstall(self):
        selfplay_mod._chunk_live = self._chunk_live
        selfplay_mod.collate_positions = self._collate
        selfplay_mod.improved_policy = self._improved


def bench_collect(args) -> tuple[list, dict]:
    cfg = _cfg(args)
    evaluate = network_evaluate(_load_or_fresh(args), cfg)
    timer = _PhaseTimer(evaluate)

    timer.install()
    _vram_reset(cfg.device)
    try:
        t0 = time.perf_counter()
        episodes, metrics = play_episodes(
            timer.evaluate,
            args.games,
            args.cap,
            0.1,
            0.03,
            np.random.default_rng(args.seed),
            pair_budget=cfg.pair_budget,
            cell_budget=cfg.cell_budget,
        )
        total = time.perf_counter() - t0
    finally:
        timer.uninstall()

    samples = sum(len(e.ranks) for e in episodes)
    lengths = sorted(len(e.moves) for e in episodes)
    won = [len(e.moves) for e in episodes if e.winner is not None]
    accounted = sum(timer.phases.values())
    residual = total - accounted

    # The drain tail: wall clock spent in lockstep steps whose live cohort
    # had shrunk below each threshold. Step k's cost is the gap to step k+1.
    ends = [t for _, t in timer.steps[1:]] + [t0 + total]
    step_cost = [(live, end - start) for (live, start), end in zip(timer.steps, ends)]
    tail = {
        thr: sum(c for live, c in step_cost if live <= thr)
        for thr in (args.games // 16, args.games // 4, args.games // 2)
    }

    report = {
        "games": args.games,
        "cap": args.cap,
        "seconds": round(total, 2),
        "samples": samples,
        "samples_per_s": round(samples / total),
        "f": round(metrics_f(episodes), 3),
        "steps": len(timer.steps),
        "length_p50": lengths[len(lengths) // 2],
        "length_p90": lengths[int(len(lengths) * 0.9)],
        "length_max": lengths[-1],
        "won_length_mean": round(statistics.mean(won), 1) if won else None,
        "phase_seconds": {k: round(v, 2) for k, v in timer.phases.items()},
        "residual_seconds": round(residual, 2),
        "tail_seconds_at_live<=": {str(k): round(v, 2) for k, v in tail.items()},
        "peak_vram_gib": round(_vram_peak_gib(cfg.device), 2),
    }
    print(json.dumps(report, indent=2))
    return episodes, report


def metrics_f(episodes) -> float:
    return sum(e.winner is not None for e in episodes) / len(episodes)


def bench_fit(args) -> None:
    episodes, _ = bench_collect(args)
    samples = [s for e in episodes for s in episode_samples(e, 1.0)]
    if not samples:
        raise SystemExit("no finished games — nothing to fit on")
    model = _load_or_fresh(args)
    cfg = _cfg(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    fit(model, samples, optimizer, cfg, np.random.default_rng(0))  # compile warm
    _sync(args.device)
    t0 = time.perf_counter()
    out = fit(model, samples, optimizer, cfg, np.random.default_rng(1))
    _sync(args.device)
    dt = time.perf_counter() - t0
    print(
        f"fit: {len(samples)} samples in {dt:.2f}s ({len(samples) / dt:,.0f} samples/s), "
        f"{out['fit_steps']} steps"
    )


def _cfg(args) -> KlentConfig:
    return KlentConfig(
        device=args.device,
        autocast=args.device == "cuda",
        compile=args.compile,
        pair_budget=args.pair_budget,
        cell_budget=args.cell_budget,
    )


def _load_or_fresh(args):
    if args.checkpoint:
        from mantisnet.klent.run import load_model

        return load_model(args.checkpoint, args.device)
    from mantisnet import MantisConfig, MantisNet

    torch.manual_seed(0)
    return MantisNet(MantisConfig()).to(args.device).eval()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("sweep", "collect", "fit"))
    ap.add_argument("--checkpoint", default=None, help="a run checkpoint (default: fresh weights)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--games", type=int, default=512)
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--stones", type=int, nargs="+", default=[20, 50, 100, 200, 400])
    ap.add_argument("--cohorts", type=int, nargs="+", default=[16, 64, 256, 1024])
    ap.add_argument("--iters", type=int, default=5, help="sweep repetitions per cell")
    ap.add_argument("--pair-budget", type=int, default=KlentConfig.pair_budget)
    ap.add_argument("--cell-budget", type=int, default=KlentConfig.cell_budget)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    {"sweep": bench_sweep, "collect": bench_collect, "fit": bench_fit}[args.mode](args)


if __name__ == "__main__":
    main()
