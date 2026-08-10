"""Time the complete Rust ACT prefix builder on real self-play prefixes.

Run from ``python/mantisnet`` after rebuilding ``hexo_py`` from this worktree::

    python tests/act/bench_act_rust_builder.py
    python tests/act/bench_act_rust_builder.py --workers 2
    python tests/act/bench_act_rust_builder.py --profile-lifetime
    python tests/act/bench_act_rust_builder.py --require-target

The timed region is the public ``collate_prefixes`` call, including replay,
parallel graph construction and Rust collation, packed validation, and the
zero-copy NumPy-to-PyTorch boundary.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mantisnet.models.mantis_act import PRESETS, collate_prefixes


REAL_GAMES = Path(__file__).resolve().parents[2] / "scratch" / "real_games.json"
DEFAULT_POSITIONS = 64
DEFAULT_MAX_PLY = 200
DEFAULT_TARGET = 3_000.0


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset", choices=sorted(PRESETS), default="full_act_v4"
    )
    parser.add_argument("--positions", type=int, default=DEFAULT_POSITIONS)
    parser.add_argument("--max-ply", type=int, default=DEFAULT_MAX_PLY)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "concurrent full-path calls (use 2 to reproduce fitloop's "
            "prefetch-worker aggregate)"
        ),
    )
    parser.add_argument("--target", type=float, default=DEFAULT_TARGET)
    parser.add_argument(
        "--require-target",
        action="store_true",
        help="exit nonzero when median throughput is below --target",
    )
    parser.add_argument(
        "--profile-lifetime",
        action="store_true",
        help=(
            "also split one-worker calls into Rust/PyO3, packed wrapping, "
            "and output destruction"
        ),
    )
    args = parser.parse_args()
    if args.positions < 2:
        parser.error("--positions must be at least 2 to form a mixed-ply buffer")
    if args.max_ply < 1:
        parser.error("--max-ply must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup must not be negative")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.target <= 0:
        parser.error("--target must be positive")
    if args.profile_lifetime and args.workers != 1:
        parser.error("--profile-lifetime requires --workers 1")
    return args


def _buffer(count: int, max_ply: int, *, stream: int = 0):
    raw_games = json.loads(REAL_GAMES.read_text(encoding="utf-8"))
    if not raw_games:
        raise RuntimeError(f"{REAL_GAMES} contains no games")

    games, plies = [], []
    for index in range(count):
        # Concurrent workers get distinct deterministic game streams instead
        # of reading the same positions from warm caches.  Thirty-one is
        # coprime to the usual corpus shard sizes; modulo still keeps tiny
        # developer fixtures valid.
        game_index = (17 * index + 31 * stream) % len(raw_games)
        game = [tuple(int(value) for value in move) for move in raw_games[game_index]]
        # Evenly cover the inclusive range.  Counts above max_ply deliberately
        # repeat depths: the benchmark size and its ply distribution are
        # independent controls.
        ply = 1 + index * (max_ply - 1) // (count - 1)
        if len(game) <= ply:
            raise RuntimeError(
                f"real game {game_index} has {len(game)} moves, so ply {ply} "
                "is not a nonterminal prefix"
            )
        games.append(game)
        plies.append(ply)
    return games, plies


def _build(games, plies, cfg) -> tuple[int, int]:
    batch = collate_prefixes(games, plies, cfg)
    if batch.position_count != len(games):
        raise RuntimeError(
            f"collator returned {batch.position_count} positions for "
            f"{len(games)} inputs"
        )
    legal = int(batch.legal_offsets[-1])
    windows = int(batch.window_offsets[-1])
    del batch
    return legal, windows


def _one(games, plies, cfg) -> tuple[float, int, int]:
    gc.collect()
    started = time.perf_counter()
    legal, windows = _build(games, plies, cfg)
    return time.perf_counter() - started, legal, windows


def _concurrent_one(
    pool: ThreadPoolExecutor, workloads, cfg
) -> tuple[float, int, int]:
    """Run fitloop-shaped preparation calls and measure aggregate throughput."""
    gc.collect()
    workers = len(workloads)
    gate = threading.Barrier(workers + 1)

    def ready_then_build(games, plies) -> tuple[int, int]:
        gate.wait()
        return _build(games, plies, cfg)

    futures = [
        pool.submit(ready_then_build, games, plies) for games, plies in workloads
    ]
    started = time.perf_counter()
    gate.wait()
    results = [future.result() for future in futures]
    elapsed = time.perf_counter() - started
    return elapsed, sum(result[0] for result in results), sum(
        result[1] for result in results
    )


def _lifetime_one(games, plies, cfg) -> tuple[float, float, float, float]:
    """Attribute the lifetime hidden by a single full-path wall sample.

    The PyO3 boundary transfers Rust vectors into NumPy without a copy, and
    ``torch.from_numpy`` keeps those allocations alive after the temporary
    field mapping is gone.  Deleting the packed container is consequently the
    point at which the dominant radius tables are actually deallocated.  Keep
    that event separate so allocator teardown cannot be mistaken for graph
    construction or validation.
    """
    import hexo_py

    from mantisnet.models.mantis_act.builder import _rust_config
    from mantisnet.models.mantis_act.packed import packed_from_arrays

    normalized_games = [
        [tuple(moves[index]) for index in range(ply)]
        for moves, ply in zip(games, plies)
    ]
    normalized_plies = [int(ply) for ply in plies]
    gc.collect()

    started = time.perf_counter()
    fields = hexo_py.build_act_batch_prefixes(
        normalized_games, normalized_plies, _rust_config(cfg)
    )
    after_binding = time.perf_counter()
    batch = packed_from_arrays(fields, cfg)
    after_wrap = time.perf_counter()
    del fields
    after_fields = time.perf_counter()
    del batch
    after_batch = time.perf_counter()
    return (
        after_binding - started,
        after_wrap - after_binding,
        after_fields - after_wrap,
        after_batch - after_fields,
    )


def main() -> int:
    args = _arguments()
    workloads = [
        _buffer(args.positions, args.max_ply, stream=worker)
        for worker in range(args.workers)
    ]
    games, plies = workloads[0]
    cfg = PRESETS[args.preset]

    if args.workers == 1:
        for _ in range(args.warmup):
            _one(games, plies, cfg)
        samples = [_one(games, plies, cfg) for _ in range(args.repeats)]
    else:
        # Fitloop keeps one pool alive for the epoch.  Keep these worker
        # threads alive across warmups and samples as well, and synchronize
        # each sample only before its timed region so thread startup is not
        # mistaken for builder work.
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for _ in range(args.warmup):
                _concurrent_one(pool, workloads, cfg)
            samples = [
                _concurrent_one(pool, workloads, cfg)
                for _ in range(args.repeats)
            ]
    seconds = [sample[0] for sample in samples]
    aggregate_positions = args.positions * args.workers
    rates = [aggregate_positions / elapsed for elapsed in seconds]
    median_seconds = statistics.median(seconds)
    median_rate = aggregate_positions / median_seconds
    legal, windows = samples[-1][1:]

    print(
        f"preset={args.preset} positions_per_worker={args.positions} "
        f"workers={args.workers} aggregate_positions={aggregate_positions} "
        f"plies={min(plies)}..{max(plies)} warmup={args.warmup} "
        f"repeats={args.repeats}"
    )
    print(
        f"rayon_threads={os.environ.get('RAYON_NUM_THREADS', 'default')} "
        f"mimalloc_purge_delay="
        f"{os.environ.get('MIMALLOC_PURGE_DELAY', 'default')} "
        f"logical_cpus={os.cpu_count()} legal_rows={legal} windows={windows}"
    )
    print("samples_pos_s=" + ",".join(f"{rate:.1f}" for rate in rates))
    print(
        f"median={median_rate:.1f} positions/s "
        f"best={max(rates):.1f} worst={min(rates):.1f} "
        f"median_batch_ms={median_seconds * 1e3:.3f}"
    )

    if args.profile_lifetime:
        for _ in range(args.warmup):
            _lifetime_one(games, plies, cfg)
        lifetime = [
            _lifetime_one(games, plies, cfg) for _ in range(args.repeats)
        ]
        labels = ("rust_pyo3", "packed_wrap", "fields_drop", "batch_drop")
        print(
            "lifetime_median_ms="
            + ",".join(
                f"{label}:"
                f"{statistics.median(sample[index] for sample in lifetime) * 1e3:.3f}"
                for index, label in enumerate(labels)
            )
        )

    if args.require_target and median_rate < args.target:
        print(
            f"FAIL: median {median_rate:.1f} positions/s is below target "
            f"{args.target:.1f}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
