"""Shared packed fitting machinery for production and lab training.

The engine owns only epoch mechanics: the packing loop, optimizer grouping,
pipelined CPU preparation, sample-weighted accumulation, and the post-step
parameter check. Callers own batch construction, their loss, and — through
:class:`ChunkCost` — what a chunk of their architecture costs.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import numpy as np
import torch

# Concurrent preparation workers and queue depth for chunk prefetching.
_PREFETCH_DEPTH = 2


@dataclass(frozen=True)
class FitBudgets:
    """Memory limits the trainer offers a model's ``chunk_cost``.

    Each architecture reads the limits that name its binding quantities.

    ``pair_budget``
        Padded stone-attention pairs (quadratic in longest position).
    ``cell_budget``
        Decoder rows, one per legal cell.
    ``graph_cell_budget``
        Graph cells plus occupied cells (ACT's additive limit, §26).
    """

    pair_budget: int
    cell_budget: int
    graph_cell_budget: int


@runtime_checkable
class ChunkCost(Protocol):
    """What binds one forward chunk, for one architecture.

    ``open`` starts an empty chunk, ``accepts`` asks whether one more sample
    fits beside ``size`` already taken, and ``take`` records it.  A sample
    that fits in no chunk is kept as a singleton.
    """

    def sort_key(self, index: int) -> int:
        """The descending pack order key; equal keys keep the caller's order."""

    def open(self) -> None:
        """Begin a fresh chunk."""

    def accepts(self, index: int, size: int) -> bool:
        """Whether ``index`` fits beside the ``size`` samples already taken."""

    def take(self, index: int) -> None:
        """Record ``index`` as part of the open chunk."""


def pack_chunks(order, batch_size: int, cost: ChunkCost) -> list[list[int]]:
    """Pack indices under the position cap and ``cost``'s memory limits.

    The stable descending sort by ``cost.sort_key`` preserves ``order`` within
    equal keys, making that order the packing tie-breaker. An indivisible
    sample that exceeds a memory limit on its own is retained as a singleton.
    """
    idx = sorted(order, key=cost.sort_key, reverse=True)
    chunks: list[list[int]] = []
    chunk: list[int] = []
    cost.open()
    for i in idx:
        if chunk and (len(chunk) == batch_size or not cost.accepts(i, len(chunk))):
            chunks.append(chunk)
            chunk = []
            cost.open()
        chunk.append(int(i))
        cost.take(i)
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
    sample_count: int,
    batch_size: int,
    cost: ChunkCost,
    prepare: Callable[[list[int]], object],
    step: Callable[[object], tuple[torch.Tensor, dict[str, torch.Tensor]]],
    lock,
    progress=None,
) -> dict[str, float | int]:
    """Fit one packed epoch and return sample-weighted statistic means.

    ``prepare(indices)`` runs on prefetch workers and must be pure.
    ``step(payload)`` runs under ``lock`` and returns ``(loss, stats_dict)``.
    ``progress(consumed, total)`` is called after every consumed chunk.
    """
    model.train()
    chunks = pack_chunks(rng.permutation(sample_count), batch_size, cost)

    groups: list[tuple[list[int], int]] = []
    group: list[int] = []
    count = 0
    for k in rng.permutation(len(chunks)):
        group.append(int(k))
        count += len(chunks[k])
        if count >= batch_size:
            groups.append((group, count))
            group, count = [], 0
    if group:
        groups.append((group, count))

    order = [k for group, _ in groups for k in group]
    stat_sums: dict[str, torch.Tensor] = {}
    total = 0
    fit_step = 0
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
                if progress is not None:
                    progress(consumed, len(order))
            optimizer.step()
            fit_step += 1
            _refuse_nonfinite_parameters(model, fit_step)
            total += group_n

    if total == 0:
        # Empty buffer: refuse rather than divide by zero.
        raise ZeroDivisionError("float division by zero")
    result: dict[str, float | int] = {
        name: float(value) / total for name, value in stat_sums.items()
    }
    result["fit_steps"] = len(groups)
    return result
