"""Benchmarks for production MantisNet building, collection, and fitting."""

from __future__ import annotations

import json
import statistics
import threading
import time
from contextlib import AbstractContextManager
from pathlib import Path

import numpy as np
import torch

from ..builder import collate, collate_positions, from_position
from ..klent import selfplay as selfplay_mod
from ..klent.improve import improved_policy
from ..klent.selfplay import Collector, episode_samples
from ..klent.train import KlentConfig, fit, network_evaluate
from ..optim import make_adam
from .cohort import corpus_cohort, selfplay_cohort
from .families import family_evaluate, load_checkpoint
from .variants import build_variant


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


def _config(
    device: str,
    compile: bool,
    pair_budget: int | None = None,
    cell_budget: int | None = None,
    adam_impl: str = "auto",
) -> KlentConfig:
    if pair_budget is not None and pair_budget <= 0:
        raise ValueError(f"pair_budget must be positive, got {pair_budget}")
    if cell_budget is not None and cell_budget <= 0:
        raise ValueError(f"cell_budget must be positive, got {cell_budget}")
    return KlentConfig(
        device=device,
        autocast=device == "cuda",
        compile=compile,
        adam_impl=adam_impl,
        pair_budget=(pair_budget if pair_budget is not None else KlentConfig.pair_budget),
        cell_budget=(cell_budget if cell_budget is not None else KlentConfig.cell_budget),
        collect_pair_budget=(
            pair_budget if pair_budget is not None else KlentConfig.collect_pair_budget
        ),
        collect_cell_budget=(
            cell_budget if cell_budget is not None else KlentConfig.collect_cell_budget
        ),
    )


def _load_or_fresh(
    checkpoint: str | Path | None,
    *,
    device: str,
    model_kw: dict | None = None,
    seed: int = 0,
    family: str | None = None,
):
    if checkpoint is not None:
        if model_kw:
            raise ValueError("model_kw applies only to fresh weights, not a checkpoint")
        loaded = load_checkpoint(Path(checkpoint), family=family, device=device)
        return loaded.model, loaded
    if family is not None:
        raise ValueError("--family applies only with --checkpoint")
    torch.manual_seed(seed)
    model, _normalized, _spec = build_variant("mantis", model_kw or {})
    return model.to(device).eval(), None


def _family_fields(loaded) -> dict:
    if loaded is None:
        return {}
    return loaded.metadata


def _positions(
    *,
    model,
    cfg: KlentConfig,
    count: int,
    steps: int,
    seed: int,
    corpus=None,
    split: str = "test",
    loaded=None,
):
    if corpus is not None:
        return corpus_cohort(corpus, split=split, count=count, seed=seed)
    return selfplay_cohort(
        envs=count,
        steps=steps,
        evaluate=(
            family_evaluate(loaded, cfg)
            if loaded is not None
            else network_evaluate(model, cfg)
        ),
        seed=seed,
        device=cfg.device,
        compile=cfg.compile,
        pair_budget=cfg.collect_pair_budget,
        cell_budget=cfg.collect_cell_budget,
    )


def _time_forward(fn, batch, *, iters: int, device: str, autocast: bool) -> float:
    if iters <= 0:
        raise ValueError(f"iters must be positive, got {iters}")

    def once():
        with torch.autocast(device, torch.bfloat16, enabled=autocast):
            fn(batch)

    with torch.no_grad():
        for _ in range(min(3, iters)):
            once()
        _sync(device)
        elapsed = []
        for _ in range(iters):
            start = time.perf_counter()
            once()
            _sync(device)
            elapsed.append(time.perf_counter() - start)
    return statistics.median(elapsed)


def bench_forward(
    *,
    checkpoint=None,
    corpus=None,
    split: str = "test",
    batch_size: int = 64,
    cohort_steps: int = 32,
    iters: int = 10,
    seed: int = 99,
    device: str = "cpu",
    compile: bool = False,
    model_kw: dict | None = None,
    family: str | None = None,
) -> dict:
    """Time Python build/collate, Rust build+collate, and model forward."""
    if batch_size <= 0 or iters <= 0 or cohort_steps < 0:
        raise ValueError(
            "batch_size and iters must be positive and cohort_steps nonnegative"
        )
    cfg = _config(device, compile)
    model, loaded = _load_or_fresh(
        checkpoint, device=device, model_kw=model_kw, seed=seed, family=family
    )
    positions = _positions(
        model=model,
        cfg=cfg,
        count=batch_size,
        steps=cohort_steps,
        seed=seed,
        corpus=corpus,
        split=split,
        loaded=loaded,
    )

    start = time.perf_counter()
    graphs = [from_position(p) for p in positions]
    build_s = time.perf_counter() - start
    start = time.perf_counter()
    collate(graphs)
    collate_s = time.perf_counter() - start
    collate_positions(positions)  # Rayon startup is not part of the measurement.
    start = time.perf_counter()
    batch = collate_positions(positions)
    rust_s = time.perf_counter() - start

    batch_d = batch.to(device)
    eager = lambda b: model(b, cfg.mass_floor)
    modes: list[tuple[str, object]] = [("eager", eager)]
    if compile:
        modes.append(("compiled", torch.compile(eager, dynamic=True)))
    precisions = [("fp32", False)]
    if device == "cuda":
        precisions.append(("bf16", True))

    forward = []
    _vram_reset(device)
    for execution, fn in modes:
        for precision, autocast in precisions:
            seconds = _time_forward(
                fn, batch_d, iters=iters, device=device, autocast=autocast
            )
            forward.append(
                {
                    "execution": execution,
                    "precision": precision,
                    "ms_per_batch": seconds * 1e3,
                    "positions_per_s": len(positions) / seconds,
                }
            )

    report = {
        "mode": "forward",
        **_family_fields(loaded),
        "device": device,
        "positions": len(positions),
        "stones": sum(g.n_stones for g in graphs),
        "live_windows": sum(g.n_windows for g in graphs),
        "legal_cells": sum(g.n_legal for g in graphs),
        "max_t": batch.max_t,
        "max_w": batch.max_w,
        "python_build": {
            "seconds": build_s,
            "positions_per_s": len(positions) / build_s,
        },
        "python_collate": {
            "seconds": collate_s,
            "positions_per_s": len(positions) / collate_s,
        },
        "rust_batch": {
            "seconds": rust_s,
            "positions_per_s": len(positions) / rust_s,
        },
        "forward": forward,
        "peak_vram_gib": _vram_peak_gib(device),
    }
    print(json.dumps(report, indent=2))
    return report


class PhaseTimer(AbstractContextManager):
    """Temporarily time the four collector pipeline phases.

    The monkeypatch is scoped by a context manager and bindings are restored
    even if collection or an evaluator raises.  Separate pipeline lanes update
    the counters under a lock; overlapping busy times may exceed wall time.
    """

    def __init__(self, evaluate):
        self._evaluate = evaluate
        self.phases = {"chunk": 0.0, "collate": 0.0, "network": 0.0, "improve": 0.0}
        self.steps: list[float] = []
        self._lock = threading.Lock()
        self._saved = None

    def _add(self, phase: str, seconds: float) -> None:
        with self._lock:
            self.phases[phase] += seconds

    def evaluate(self, batch):
        start = time.perf_counter()
        out = self._evaluate(batch)
        self._add("network", time.perf_counter() - start)
        return out

    def __enter__(self):
        self._saved = (
            selfplay_mod._chunk_live,
            selfplay_mod.collate_positions,
            selfplay_mod.improved_policy,
        )
        original_chunk, original_collate, original_improved = self._saved

        def chunk_live(positions, live, pair_budget, cell_budget, cap):
            self.steps.append(time.perf_counter())
            start = time.perf_counter()
            out = original_chunk(positions, live, pair_budget, cell_budget, cap)
            self._add("chunk", time.perf_counter() - start)
            return out

        def timed_collate(positions):
            start = time.perf_counter()
            out = original_collate(positions)
            self._add("collate", time.perf_counter() - start)
            return out

        def timed_improved(policy, score, q, offsets, tau, lam):
            start = time.perf_counter()
            out = original_improved(policy, score, q, offsets, tau, lam)
            self._add("improve", time.perf_counter() - start)
            return out

        selfplay_mod._chunk_live = chunk_live
        selfplay_mod.collate_positions = timed_collate
        selfplay_mod.improved_policy = timed_improved
        return self

    def __exit__(self, *exc):
        if self._saved is not None:
            (
                selfplay_mod._chunk_live,
                selfplay_mod.collate_positions,
                selfplay_mod.improved_policy,
            ) = self._saved
            self._saved = None
        return False


def _collect(
    *,
    checkpoint=None,
    games: int = 32,
    envs: int = 16,
    cap: int = 512,
    seed: int = 7,
    device: str = "cpu",
    compile: bool = False,
    pair_budget: int | None = None,
    cell_budget: int | None = None,
    adam_impl: str = "auto",
    model_kw: dict | None = None,
    emit: bool = True,
    family: str | None = None,
):
    if games <= 0 or envs <= 0 or cap <= 0:
        raise ValueError(
            f"games, envs, and cap must be positive, got {games}, {envs}, {cap}"
        )
    cfg = _config(device, compile, pair_budget, cell_budget, adam_impl)
    model, loaded = _load_or_fresh(
        checkpoint, device=device, model_kw=model_kw, seed=seed, family=family
    )
    evaluate = (
        family_evaluate(loaded, cfg)
        if loaded is not None
        else network_evaluate(model, cfg)
    )
    collector = Collector(
        envs,
        cap,
        cfg.tau,
        cfg.lam,
        np.random.default_rng(seed),
        pair_budget=cfg.collect_pair_budget,
        cell_budget=cfg.collect_cell_budget,
    )
    timer = PhaseTimer(evaluate)
    _vram_reset(device)
    start = time.perf_counter()
    with timer:
        episodes, metrics = collector.collect(timer.evaluate, games)
    seconds = time.perf_counter() - start

    lengths = sorted(len(e.moves) for e in episodes)
    won = [len(e.moves) for e in episodes if e.winner is not None]
    samples = sum(len(e.ranks) for e in episodes)
    report = {
        "mode": "collect",
        **_family_fields(loaded),
        "device": device,
        "envs": envs,
        "games_quota": games,
        "games_returned": len(episodes),
        "cap": cap,
        "seconds": seconds,
        "samples": samples,
        "samples_per_s": samples / seconds,
        "carry_plies_in_slots": sum(len(e.ranks) for e in collector.episodes),
        "f": sum(e.winner is not None for e in episodes) / len(episodes),
        "steps": len(timer.steps),
        "ms_per_step": seconds / max(len(timer.steps), 1) * 1e3,
        "length_p50": lengths[len(lengths) // 2],
        "length_p90": lengths[min(int(len(lengths) * 0.9), len(lengths) - 1)],
        "length_max": lengths[-1],
        "won_length_mean": statistics.mean(won) if won else None,
        "phase_busy_seconds": timer.phases,
        "busy_minus_wall": sum(timer.phases.values()) - seconds,
        "peak_vram_gib": _vram_peak_gib(device),
        "acting": metrics,
    }
    if emit:
        print(json.dumps(report, indent=2))
    return model, cfg, episodes, report


def bench_collect(**kwargs) -> dict:
    """Instrument one real :meth:`Collector.collect` call."""
    return _collect(**kwargs)[3]


def bench_fit(
    *,
    checkpoint=None,
    corpus=None,
    split: str = "train",
    games: int = 32,
    envs: int = 16,
    cap: int = 512,
    seed: int = 7,
    device: str = "cpu",
    compile: bool = False,
    pair_budget: int | None = None,
    cell_budget: int | None = None,
    adam_impl: str = "auto",
    steady_warmup: int | None = None,
    steady_measure: int | None = None,
    model_kw: dict | None = None,
    family: str | None = None,
) -> dict:
    """Benchmark a production KLENT fit or one supervised corpus epoch.

    With ``steady_warmup``/``steady_measure`` set, the run is the campaign
    speed harness of ``docs/MANTIS_GRAFT_SPEC.md`` §2.2: one epoch whose first
    ``steady_warmup`` chunks absorb compilation and cache warm-up untimed,
    followed by a CUDA-synchronized window of ``steady_measure`` chunks that
    yields the reported throughput; the epoch then stops early.
    """
    if (steady_warmup is None) != (steady_measure is None):
        raise ValueError(
            "steady_warmup and steady_measure must be given together"
        )
    cfg = _config(device, compile, pair_budget, cell_budget, adam_impl)
    model, loaded = _load_or_fresh(
        checkpoint, device=device, model_kw=model_kw, seed=seed, family=family
    )
    if loaded is not None and loaded.family.name != "trinomial-joint":
        raise ValueError(
            "bench fit has only the current trinomial training objective; "
            f"checkpoint family {loaded.family.name!r} is scoreable but its "
            "historical fitting loss is outside the lab family contract"
        )
    optimizer, adam_resolved = make_adam(
        model.parameters(),
        lr=cfg.lr,
        device=cfg.device,
        implementation=cfg.adam_impl,
    )
    if corpus is None:
        model, cfg, episodes, _ = _collect(
            checkpoint=checkpoint,
            games=games,
            envs=envs,
            cap=cap,
            seed=seed,
            device=device,
            compile=compile,
            pair_budget=pair_budget,
            cell_budget=cell_budget,
            adam_impl=adam_impl,
            model_kw=model_kw,
            emit=False,
            family=family,
        )
        optimizer, adam_resolved = make_adam(
            model.parameters(),
            lr=cfg.lr,
            device=cfg.device,
            implementation=cfg.adam_impl,
        )
        samples = [
            sample
            for episode in episodes
            for sample in episode_samples(episode, cfg.lam_ret, cfg.gamma)
        ]
        if not samples:
            raise ValueError("collection produced no naturally terminated samples")
        sample_count = len(samples)
        source = "collect"

        def run_epoch(epoch_seed, steady=None):
            return fit(
                model,
                samples,
                optimizer,
                cfg,
                np.random.default_rng(epoch_seed),
                steady=steady,
            )
    else:
        from .corpus import load_corpus
        from .train import fit_supervised_epoch, sample_sizes

        frozen = load_corpus(corpus) if isinstance(corpus, (str, Path)) else corpus
        samples = frozen.split_samples(split)
        sample_count = len(samples)
        source = "corpus"
        # Sizing replays the corpus once per split; production fitting reads
        # sizes from its buffer, so the replay stays outside the timed epoch.
        sizes = sample_sizes(frozen, samples)

        def run_epoch(epoch_seed, steady=None):
            return fit_supervised_epoch(
                model,
                optimizer,
                frozen,
                split=split,
                cfg=cfg,
                rng=np.random.default_rng(epoch_seed),
                sizes=sizes,
                steady=steady,
            )

    if steady_warmup is not None:
        # §2.2 window: one epoch, warm-up absorbed by its leading chunks. The
        # reported peak VRAM covers the whole truncated epoch, compilation
        # included — conservative, and identical in kind across arms.
        _vram_reset(device)
        metrics = run_epoch(seed, steady=(steady_warmup, steady_measure))
        window = metrics["steady"]
        sample_count = int(window["samples"])
        seconds = float(window["seconds"])
    else:
        # Compilation, autotuning, and first-use allocator work happen outside
        # the timed epoch.
        run_epoch(seed)
        _sync(device)
        _vram_reset(device)
        start = time.perf_counter()
        metrics = run_epoch(seed + 1)
        _sync(device)
        seconds = time.perf_counter() - start
    report = {
        "mode": "fit",
        **_family_fields(loaded),
        "source": source,
        "device": device,
        "adam_impl": {"requested": adam_impl, "resolved": adam_resolved},
        "samples": sample_count,
        "seconds": seconds,
        "samples_per_s": sample_count / seconds,
        "fit_steps": metrics["fit_steps"],
        "metrics": metrics,
        "peak_vram_gib": _vram_peak_gib(device),
    }
    print(json.dumps(report, indent=2))
    return report


def _improve_and_sample(policy, score, q, offsets, rng, tau, lam):
    imp = improved_policy(policy, score, q, offsets, tau, lam)
    offsets_np = offsets.numpy()
    flat = imp.probs.numpy().astype(np.float64)
    widths = np.diff(offsets_np)
    flat /= np.repeat(np.add.reduceat(flat, offsets_np[:-1]), widths)
    cdf = np.cumsum(flat)
    base = np.concatenate(([0.0], cdf[offsets_np[1:-1] - 1]))
    ranks = np.searchsorted(cdf, base + rng.random(len(widths))) - offsets_np[:-1]
    return np.clip(ranks, 0, widths - 1)


def bench_sweep(
    *,
    checkpoint=None,
    depths=(20, 50, 100),
    cohorts=(16, 64),
    iters: int = 3,
    seed: int = 7,
    device: str = "cpu",
    compile: bool = False,
    pair_budget: int | None = None,
    cell_budget: int | None = None,
    model_kw: dict | None = None,
    family: str | None = None,
) -> dict:
    """Measure collate/forward/improve+sample over depth × cohort size."""
    if iters <= 0:
        raise ValueError(f"iters must be positive, got {iters}")
    if not depths or any(int(depth) < 0 for depth in depths):
        raise ValueError(f"depths must be a nonempty sequence of nonnegative values: {depths}")
    if not cohorts or any(int(cohort) <= 0 for cohort in cohorts):
        raise ValueError(f"cohorts must be a nonempty sequence of positive values: {cohorts}")
    cfg = _config(device, compile, pair_budget, cell_budget)
    model, loaded = _load_or_fresh(
        checkpoint, device=device, model_kw=model_kw, seed=seed, family=family
    )
    evaluate = (
        family_evaluate(loaded, cfg)
        if loaded is not None
        else network_evaluate(model, cfg)
    )
    rows = []
    for depth in depths:
        pool = selfplay_cohort(
            envs=max(cohorts),
            steps=int(depth),
            evaluate=evaluate,
            seed=seed + int(depth),
            device=device,
            compile=compile,
            pair_budget=cfg.collect_pair_budget,
            cell_budget=cfg.collect_cell_budget,
        )
        for cohort_size in cohorts:
            positions = pool[: int(cohort_size)]
            chunks = selfplay_mod._chunk_live(
                positions,
                list(range(len(positions))),
                cfg.collect_pair_budget,
                cfg.collect_cell_budget,
                len(positions),
            )
            for chunk in chunks:
                evaluate(collate_positions([positions[i] for i in chunk]))
            _sync(device)
            _vram_reset(device)
            collate_s = forward_s = improve_s = 0.0
            rng = np.random.default_rng(seed)
            for _ in range(iters):
                for chunk in chunks:
                    start = time.perf_counter()
                    batch = collate_positions([positions[i] for i in chunk])
                    after_collate = time.perf_counter()
                    policy, score, q = evaluate(batch)
                    after_forward = time.perf_counter()
                    _improve_and_sample(
                        policy, score, q, batch.legal_offsets, rng, cfg.tau, cfg.lam
                    )
                    after_improve = time.perf_counter()
                    collate_s += after_collate - start
                    forward_s += after_forward - after_collate
                    improve_s += after_improve - after_forward
            total = (collate_s + forward_s + improve_s) / iters
            rows.append(
                {
                    "depth": int(depth),
                    "cohort": len(positions),
                    "chunks": len(chunks),
                    "legal_per_position": sum(p.legal_count for p in positions)
                    / len(positions),
                    "collate_ms": collate_s / iters * 1e3,
                    "forward_ms": forward_s / iters * 1e3,
                    "improve_sample_ms": improve_s / iters * 1e3,
                    "total_ms": total * 1e3,
                    "positions_per_s": len(positions) / total,
                    "peak_vram_gib": _vram_peak_gib(device),
                }
            )
    report = {
        "mode": "sweep",
        **_family_fields(loaded),
        "device": device,
        "rows": rows,
    }
    print(json.dumps(report, indent=2))
    return report
