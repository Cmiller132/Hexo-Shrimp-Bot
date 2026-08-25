"""Stage-attribution profiles over production-shaped MantisNet cohorts.

Modes: decode (eager trunk/decoder split), seam (the network-evaluation
seam), fit (real optimizer steps in the production fit engine)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from ..builder import collate_positions
from ..optim import make_adam
from ..klent.train import KlentConfig, _policy_q
from .bench import _config, _family_fields, _load_or_fresh
from .cohort import corpus_cohort, selfplay_cohort
from .corpus import load_corpus
from .families import family_evaluate, load_checkpoint
from .train import fit_supervised_epoch, sample_sizes


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _elapsed(device: str, fn):
    _sync(device)
    start = time.perf_counter()
    out = fn()
    _sync(device)
    return out, time.perf_counter() - start


def _positions(model, loaded, *, corpus, split, envs, steps, seed, device, compile):
    if envs <= 0 or steps < 0:
        raise ValueError(f"envs must be positive and steps nonnegative, got {envs}, {steps}")
    if corpus is not None:
        return corpus_cohort(corpus, split=split, count=envs, seed=seed)
    cfg = KlentConfig(
        device=device,
        autocast=device == "cuda",
        compile=compile,
    )
    return selfplay_cohort(
        envs=envs,
        steps=steps,
        evaluate=family_evaluate(loaded, cfg),
        seed=seed,
        pair_budget=cfg.collect_pair_budget,
        cell_budget=cfg.collect_cell_budget,
    )


def _profile_activities(device: str):
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return activities


# Kernel-name needles for self-time attribution; the first matching bucket
# claims a row, so the in-repo Triton kernels come before the generic families
# ("triton_" alone would also claim them).
_FIT_BUCKETS = {
    "relay": (
        "_cell_values_kernel",
        "_cell_grad_kernel",
        "_segment_sum_kernel",
        "_class_partial_kernel",
    ),
    "attention": ("_fused_attention",),
    "gemm": ("gemm", "cutlass", "nvjet", "cublas", "wgrad", "dgrad"),
    "inductor": ("triton_",),
    "eager index/eltwise": ("index", "scatter", "gather", "elementwise"),
    "memcpy": ("memcpy",),
    "optimizer/copy": ("adam", "multi_tensor", "foreach", "copy", "fill", "cat"),
}


def _row_micros(row, device: str) -> float:
    if device != "cuda":
        return row.self_cpu_time_total
    return row.self_device_time_total


def profile_fit(
    *,
    checkpoint=None,
    corpus,
    split: str = "val",
    wait: int = 6,
    warmup: int = 2,
    active: int = 8,
    seed: int = 7,
    device: str = "cpu",
    compile: bool = False,
    pair_budget: int | None = None,
    cell_budget: int | None = None,
    adam_impl: str = "auto",
    model_kw: dict | None = None,
    family: str | None = None,
) -> dict:
    """Profile real optimizer steps inside the production fit engine.

    A warm epoch absorbs compilation, autotuning, and allocator growth; a
    second full epoch runs with a ``torch.profiler`` window over
    ``wait + warmup + active`` steps. Kernel self-time is bucketed by family.
    """
    if wait < 0 or warmup < 0 or active <= 0:
        raise ValueError(
            f"wait and warmup must be nonnegative and active positive, "
            f"got {wait}, {warmup}, {active}"
        )
    cfg = _config(device, compile, pair_budget, cell_budget, adam_impl)
    model, loaded = _load_or_fresh(
        checkpoint, device=device, model_kw=model_kw, seed=seed, family=family
    )
    if loaded is not None and loaded.family.name != "trinomial-joint":
        raise ValueError(
            "profile fit has only the current trinomial training objective; "
            f"checkpoint family {loaded.family.name!r} is scoreable but its "
            "historical fitting loss is outside the lab family contract"
        )
    frozen = load_corpus(corpus) if isinstance(corpus, (str, Path)) else corpus
    samples = frozen.split_samples(split)
    sizes = sample_sizes(frozen, samples)
    optimizer, adam_resolved = make_adam(
        model.parameters(),
        lr=cfg.lr,
        device=cfg.device,
        implementation=cfg.adam_impl,
    )

    def run_epoch(epoch_seed):
        return fit_supervised_epoch(
            model,
            optimizer,
            frozen,
            split=split,
            cfg=cfg,
            rng=np.random.default_rng(epoch_seed),
            sizes=sizes,
        )

    run_epoch(seed)
    _sync(device)

    profiler = torch.profiler.profile(
        activities=_profile_activities(device),
        schedule=torch.profiler.schedule(
            wait=wait, warmup=warmup, active=active, repeat=1
        ),
    )
    steps = 0
    original_step = optimizer.step

    def stepping_step(*args, **kwargs):
        nonlocal steps
        out = original_step(*args, **kwargs)
        steps += 1
        profiler.step()
        return out

    optimizer.step = stepping_step
    try:
        with profiler:
            run_epoch(seed + 1)
    finally:
        optimizer.step = original_step
    needed = wait + warmup + active
    if steps < needed:
        raise ValueError(
            f"split {split!r} ran {steps} optimizer steps; the schedule needs "
            f"wait+warmup+active={needed}"
        )

    averages = profiler.key_averages()
    buckets = dict.fromkeys(_FIT_BUCKETS, 0.0)
    other = 0.0
    for row in averages:
        # Host-side op rows (aten::mm, mantisnet::cell_pass, ProfilerStep*)
        # carry their child kernels' device time again; on CUDA bucket only
        # the device-level rows so the totals add up once. CPU rows nest
        # properly, and there the op rows are the only rows.
        if device == "cuda" and (
            "::" in row.key or row.key.startswith("ProfilerStep")
        ):
            continue
        micros = _row_micros(row, device)
        if not micros:
            continue
        key = row.key.lower()
        for bucket, needles in _FIT_BUCKETS.items():
            if any(needle in key for needle in needles):
                buckets[bucket] += micros
                break
        else:
            other += micros
    total = sum(buckets.values()) + other
    report = {
        "mode": "fit",
        **_family_fields(loaded),
        "device": device,
        "compile": compile,
        "adam_impl": {"requested": adam_impl, "resolved": adam_resolved},
        "split": split,
        "samples": len(samples),
        "schedule": {"wait": wait, "warmup": warmup, "active": active},
        "self_time_ms": {
            **{name: micros / 1e3 for name, micros in buckets.items()},
            "other": other / 1e3,
        },
        "total_ms": total / 1e3,
        "kernel_profiler": averages.table(
            sort_by=(
                "self_cuda_time_total" if device == "cuda" else "self_cpu_time_total"
            ),
            row_limit=45,
        ),
    }
    print(json.dumps(report, indent=2))
    return report


def profile_decode(
    *,
    checkpoint,
    corpus=None,
    split: str = "test",
    envs: int = 16,
    steps: int = 32,
    iters: int = 5,
    seed: int = 0,
    device: str = "cpu",
    compile: bool = False,
    family: str | None = None,
) -> dict:
    """Attribute eager trunk/decoder time, then optional compiled total."""
    if iters <= 0:
        raise ValueError(f"iters must be positive, got {iters}")
    loaded = load_checkpoint(Path(checkpoint), family=family, device=device)
    model = loaded.model
    positions = _positions(
        model, loaded,
        corpus=corpus,
        split=split,
        envs=envs,
        steps=steps,
        seed=seed,
        device=device,
        compile=compile,
    )
    batch = collate_positions(positions).to(device)
    trunk_s = decode_s = 0.0
    with torch.no_grad(), torch.autocast(
        device, torch.bfloat16, enabled=device == "cuda"
    ):
        model(batch, 0.2)
        for _ in range(iters):
            (w, g, cells), dt = _elapsed(device, lambda: model.trunk(batch))
            trunk_s += dt
            _, dt = _elapsed(
                device, lambda: model.cell_heads(w, g, cells, batch, 0.2)
            )
            decode_s += dt

        compiled_ms = None
        if compile:
            compiled_fn = torch.compile(
                lambda m, b: m.cell_heads(*m.trunk(b), b, 0.2), dynamic=True
            )
            compiled_fn(model, batch)
            compiled_total = 0.0
            for _ in range(iters):
                _, dt = _elapsed(device, lambda: compiled_fn(model, batch))
                compiled_total += dt
            compiled_ms = compiled_total / iters * 1e3

        w, g, cells = model.trunk(batch)
        with torch.profiler.profile(activities=_profile_activities(device)) as prof:
            model.cell_heads(w, g, cells, batch, 0.2)
            _sync(device)
    table = prof.key_averages().table(
        sort_by="self_cuda_time_total" if device == "cuda" else "self_cpu_time_total",
        row_limit=25,
    )
    report = {
        "mode": "decode",
        **loaded.metadata,
        "device": device,
        "positions": len(positions),
        "eager": {
            "trunk_ms": trunk_s / iters * 1e3,
            "cell_heads_ms": decode_s / iters * 1e3,
            "total_ms": (trunk_s + decode_s) / iters * 1e3,
        },
        "compiled_total_ms": compiled_ms,
        "decoder_profiler": table,
    }
    print(json.dumps(report, indent=2))
    return report


def _seam_once(model, batch, cfg, policy_q, composition):
    moved, transfer_s = _elapsed(cfg.device, lambda: batch.to(cfg.device))

    def forward():
        with torch.no_grad(), torch.autocast(
            cfg.device, torch.bfloat16, enabled=cfg.autocast
        ):
            return policy_q(model, moved)

    (policy, critic), forward_s = _elapsed(cfg.device, forward)

    def compose_return():
        return (
            policy.float().cpu(),
            composition.q_score(critic, moved.legal_offsets, cfg.mass_floor).cpu(),
            composition.q_value(critic).cpu(),
        )

    _out, compose_s = _elapsed(cfg.device, compose_return)
    return transfer_s, forward_s, compose_s


def profile_seam(
    *,
    checkpoint,
    corpus=None,
    split: str = "test",
    envs: int = 16,
    steps: int = 32,
    iters: int = 5,
    seed: int = 0,
    device: str = "cpu",
    compile: bool = False,
    family: str | None = None,
) -> dict:
    """Split the network-evaluation seam into transfer/forward/composition."""
    if iters <= 0:
        raise ValueError(f"iters must be positive, got {iters}")
    loaded = load_checkpoint(Path(checkpoint), family=family, device=device)
    model = loaded.model
    cfg = KlentConfig(
        device=device, autocast=device == "cuda", compile=compile
    )
    positions = _positions(
        model, loaded,
        corpus=corpus,
        split=split,
        envs=envs,
        steps=steps,
        seed=seed,
        device=device,
        compile=compile,
    )
    batch = collate_positions(positions)

    def measure(fn):
        sums = [0.0, 0.0, 0.0]
        _seam_once(model, batch, cfg, fn, loaded.composition)
        for _ in range(iters):
            row = _seam_once(model, batch, cfg, fn, loaded.composition)
            sums = [a + b for a, b in zip(sums, row)]
        return {
            "transfer_ms": sums[0] / iters * 1e3,
            "forward_ms": sums[1] / iters * 1e3,
            "compose_return_ms": sums[2] / iters * 1e3,
            "total_ms": sum(sums) / iters * 1e3,
        }

    eager = measure(_policy_q)
    compiled = measure(torch.compile(_policy_q, dynamic=True)) if compile else None
    report = {
        "mode": "seam",
        **loaded.metadata,
        "device": device,
        "positions": len(positions),
        "eager": eager,
        "compiled": compiled,
    }
    print(json.dumps(report, indent=2))
    return report


def run_profile(mode: str, **kwargs) -> dict:
    return {
        "decode": profile_decode,
        "seam": profile_seam,
        "fit": profile_fit,
    }[mode](**kwargs)
