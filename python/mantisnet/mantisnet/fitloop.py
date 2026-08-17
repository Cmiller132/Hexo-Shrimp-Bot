"""Shared packed fitting machinery for production and lab training.

The engine owns only epoch mechanics: memory-budget packing, optimizer
grouping, pipelined CPU preparation, sample-weighted accumulation, and the
post-step parameter check. Callers own batch construction and their loss.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

# Chunks prepared concurrently ahead of consumption. The GPU step outpaces a
# single preparation worker, so the pipeline holds a few chunks in flight;
# consumption order is fixed by the packed permutation either way.
_PREFETCH_DEPTH = 4


@dataclass(frozen=True)
class FitBudgets:
    """Limits for one forward chunk and one accumulated optimizer group."""

    batch_size: int
    pair_budget: int
    cell_budget: int


def pack_chunks(
    lengths,
    cells,
    order,
    budgets: FitBudgets,
) -> list[list[int]]:
    """Pack indices under the position, padded-pair, and legal-cell limits.

    ``lengths[i]`` is the padded attention width of sample ``i``. The stable
    descending sort preserves ``order`` within equal lengths, making that
    order the packing tie-breaker. An indivisible sample that exceeds either
    memory budget is retained as a singleton.
    """
    idx = sorted(order, key=lambda i: lengths[i], reverse=True)
    chunks: list[list[int]] = []
    chunk: list[int] = []
    chunk_length, chunk_cells = 0, 0
    for i in idx:
        length = lengths[i]
        cell_count = cells[i]
        if chunk and (
            len(chunk) == budgets.batch_size
            or (len(chunk) + 1) * chunk_length * chunk_length
            > budgets.pair_budget
            or chunk_cells + cell_count > budgets.cell_budget
        ):
            chunks.append(chunk)
            chunk, chunk_cells = [], 0
        if not chunk:
            chunk_length = length
        chunk.append(int(i))
        chunk_cells += cell_count
    if chunk:
        chunks.append(chunk)
    return chunks


def _refuse_nonfinite_parameters(model, step: int) -> None:
    """Refuse a weight update that left the finite range at the named step.

    One non-finite parameter makes every later evaluation non-finite. A fused
    norm keeps the normal path to one synchronization; the parameter walk runs
    only after that check has already failed, so the refusal can name tensors.
    """
    total = torch.stack(torch._foreach_norm(list(model.parameters()))).sum()
    if torch.isfinite(total):
        return
    guilty = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
    raise ValueError(
        f"optimizer step {step} left {len(guilty)} parameter tensors non-finite: "
        + ", ".join(guilty[:8])
        + (" ..." if len(guilty) > 8 else "")
    )


def fit_epoch(
    model,
    optimizer,
    rng: np.random.Generator,
    *,
    lengths,
    cells,
    budgets: FitBudgets,
    prepare: Callable[[list[int]], object],
    step: Callable[[object], tuple[torch.Tensor, dict[str, torch.Tensor]]],
    lock,
    progress=None,
    steady: tuple[int, int] | None = None,
) -> dict[str, float | int | dict[str, float | int]]:
    """Fit one packed epoch and return sample-weighted statistic means.

    ``prepare(indices)`` must be pure; it runs on the prefetch workers, up to
    ``_PREFETCH_DEPTH`` chunks ahead of consumption. ``step(payload)`` runs
    under ``lock`` and returns a per-sample-mean differentiable loss plus
    detached scalar statistics. The engine scales each chunk's loss by its
    fraction of the optimizer group before backpropagating.

    RNG consumption is exactly two permutations: samples before packing, then
    packed chunks before grouping. ``progress(consumed, total)`` is called
    after every consumed chunk.

    ``steady=(warmup, measure)`` is the benchmark window of the graft
    campaign's speed gate (``docs/MANTIS_GRAFT_SPEC.md`` §2.2): consume
    ``warmup`` chunks, synchronize CUDA, time the next ``measure`` chunks,
    synchronize again, then stop the epoch early. The returned mapping gains
    a ``"steady"`` sub-mapping with the windowed throughput and approximate
    per-chunk latency quantiles (chunk boundaries are wall-clock stamps
    without per-chunk synchronization, so quantiles are indicative while the
    windowed rate is exact). Consumption order, packing, and every consumed
    chunk's arithmetic are identical with and without the window; production
    fitting never passes it.
    """
    model.train()
    chunks = pack_chunks(
        lengths,
        cells,
        rng.permutation(len(lengths)),
        budgets,
    )

    groups: list[tuple[list[int], int]] = []
    group: list[int] = []
    count = 0
    for k in rng.permutation(len(chunks)):
        group.append(int(k))
        count += len(chunks[k])
        if count >= budgets.batch_size:
            groups.append((group, count))
            group, count = [], 0
    if group:
        groups.append((group, count))

    order = [k for group, _ in groups for k in group]
    if steady is not None:
        warmup, measure = steady
        if warmup < 1 or measure < 1:
            raise ValueError(
                f"steady window needs warmup >= 1 and measure >= 1, got {steady}"
            )
        if warmup + measure > len(order):
            raise ValueError(
                f"steady window of {warmup}+{measure} chunks exceeds the "
                f"epoch's {len(order)} packed chunks; use a larger sample"
            )
    device = next(model.parameters()).device

    def _sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    stat_sums: dict[str, torch.Tensor] = {}
    total = 0
    fit_step = 0
    steady_result: dict[str, float | int] | None = None
    marks: list[float] = []
    window_t0 = 0.0
    window_s0 = 0
    stop = False
    with ThreadPoolExecutor(max_workers=_PREFETCH_DEPTH) as pool:
        prepped = {
            k: pool.submit(prepare, chunks[k]) for k in order[:_PREFETCH_DEPTH]
        }
        consumed = 0
        for group, group_n in groups:
            optimizer.zero_grad(set_to_none=True)
            for k in group:
                if consumed + _PREFETCH_DEPTH < len(order):
                    nxt = order[consumed + _PREFETCH_DEPTH]
                    prepped[nxt] = pool.submit(prepare, chunks[nxt])
                payload = prepped.pop(k).result()
                consumed += 1
                chunk_n = len(chunks[k])
                with lock:
                    loss, stats = step(payload)
                    (loss * (chunk_n / group_n)).backward()
                    for name, value in stats.items():
                        weighted = value.detach() * chunk_n
                        if name in stat_sums:
                            stat_sums[name] += weighted
                        else:
                            stat_sums[name] = weighted
                total += chunk_n
                if steady is not None:
                    if consumed == warmup:
                        _sync()
                        window_t0 = time.perf_counter()
                        window_s0 = total
                        marks = [window_t0]
                    elif consumed > warmup:
                        if consumed == warmup + measure:
                            # The closing sync drains the queued GPU work so
                            # the window's wall time covers exactly its chunks.
                            _sync()
                        marks.append(time.perf_counter())
                        if consumed == warmup + measure:
                            seconds = marks[-1] - window_t0
                            deltas = np.diff(np.asarray(marks)) * 1e3
                            steady_result = {
                                "warmup_chunks": warmup,
                                "measure_chunks": measure,
                                "seconds": seconds,
                                "samples": total - window_s0,
                                "samples_per_s": (total - window_s0) / seconds,
                                "chunk_ms_median": float(np.median(deltas)),
                                "chunk_ms_p95": float(np.quantile(deltas, 0.95)),
                            }
                            stop = True
                if progress is not None:
                    progress(consumed, len(order))
                if stop:
                    break
            if stop:
                # The truncated group's accumulated gradients are discarded:
                # the window has closed, and a partial optimizer step would
                # train on a group the epoch never finished.
                break
            optimizer.step()
            fit_step += 1
            _refuse_nonfinite_parameters(model, fit_step)

    if total == 0:
        # Preserve the production fit's historical empty-buffer refusal. It
        # previously failed while dividing its named metric totals by zero;
        # returning a partial {"fit_steps": 0} result would violate fit's
        # public four-key metric contract.
        raise ZeroDivisionError("float division by zero")
    result: dict[str, float | int] = {
        name: float(value) / total for name, value in stat_sums.items()
    }
    result["fit_steps"] = fit_step if steady is not None else len(groups)
    if steady_result is not None:
        result["steady"] = steady_result
    return result
