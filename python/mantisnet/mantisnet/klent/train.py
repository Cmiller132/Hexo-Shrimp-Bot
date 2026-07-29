"""The KLENT iteration: collect an on-policy buffer, fit once, discard it.

Faithful to the paper's outer loop (design doc §1): a self-play phase fills
the buffer, one fitting epoch consumes it (the reference implementation's
``fitting_epochs=1``), and nothing survives to the next iteration. The loss
is eq. 4 — cross-entropy of π_θ against π′ plus squared error of the taken
action's Q against the λ-return — under plain Adam at the paper's learning
rate. The value head appears nowhere: KLENT has no state-value head, and
v̂ = E_{π′}[Q] does its job.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch

from ..builder import collate_prefixes
from ..losses import policy_loss
from .selfplay import Collector, Sample, collection_stats


@dataclass
class KlentConfig:
    """The paper's values wherever Hexo permits them (design doc §2)."""

    # Verified against the paper's eq. 2 and the reference implementation:
    # reverse KL is the heavier regulariser. Prior exponent tau/(tau+lam) = 0.77.
    tau: float = 0.1  # reverse-KL weight (the paper's beta)
    lam: float = 0.03  # entropy weight (the paper's alpha)
    # e^{-1/16}: the paper's 8-turn horizon at Hexo's two placements per turn
    # (KLENT_PROPOSALS A1). The paper's literal e^{-1/8} would halve it.
    lam_ret: float = 0.939
    # Per-ply return-discount magnitude (the mover-change sign is separate).
    # 1.0 is the reference objective; below 1 it ranks faster wins above
    # slower ones — the conversion pressure an infinite no-draw board lacks.
    gamma: float = 1.0
    ply_cap: int = 512  # §5: capped episodes are dropped whole
    # The completion quota: an iteration's buffer is at least this many
    # *finished* games, from however many slots are in flight.
    games_per_iteration: int = 4096
    envs: int = 1024  # persistent self-play slots (the reference's env count)
    batch_size: int = 4096  # paper's *effective* batch: chunks accumulate to it
    lr: float = 1e-3  # paper's Adam rate
    device: str = "cpu"
    autocast: bool = False  # bf16 autocast for the network passes
    compile: bool = False  # torch.compile the policy/Q pass (one-time cost)
    # VRAM is budgeted, not hoped for: every network batch — fit chunk or
    # collection cohort — is packed under both measured memory axes, so the
    # peak is set here rather than by whatever the corpus happens to contain.
    # Attention memory is quadratic in the batch's longest position (padding),
    # decoder memory linear in its total legal cells. Fit and collection get
    # separate budgets because fit holds the backward graph per cell while
    # collection runs no_grad — and in the pipelined loop both peaks are
    # resident at once, so together they must clear the card.
    pair_budget: int = 8_000_000  # fit: padded (stones + token)^2 pairs per batch
    # 800k cells ≈ 5.8 GiB worst-case fit peak on the iteration-0 corpus —
    # still comfortable on 12 GiB, and half the chunk count (so half the
    # per-chunk Python and launch overhead) of the original 400k.
    cell_budget: int = 800_000  # fit: legal cells (decoder rows) per batch
    collect_pair_budget: int = 24_000_000  # collection (no_grad) equivalents
    collect_cell_budget: int = 2_400_000


def _policy_q(model, batch):
    """The KLENT pass: trunk + the two heads it trains, never the value head."""
    _s, w, g = model.trunk(batch)
    return model.cell_heads(w, g, batch)


# One symbolic-shape graph serves every batch; compiled lazily, shared by
# collection and fitting (the same graph gets the compiled backward).
_policy_q_compiled = None

# Every compiled-callable invocation holds this lock. The GPU work was
# serialized on one stream anyway; what the lock buys is that when dynamo
# (re)compiles on one thread — sporadic new shape buckets keep this possible
# at any iteration — the other thread sleeps at the lock instead of
# thrashing the GIL against the trace, which was measured to stretch a
# seconds-long compile into a minutes-long crawl.
_gpu_lock = threading.Lock()


def _policy_q_fn(cfg: KlentConfig):
    global _policy_q_compiled
    if not cfg.compile:
        return _policy_q
    if _policy_q_compiled is None:
        _policy_q_compiled = torch.compile(_policy_q, dynamic=True)
    return _policy_q_compiled


def network_evaluate(model, cfg: KlentConfig):
    """The self-play evaluator, returning flat CPU tensors so the collection
    loop stays device-ignorant."""
    policy_q = _policy_q_fn(cfg)

    def evaluate(batch):
        with _gpu_lock:
            b = batch.to(cfg.device)
            with torch.no_grad(), torch.autocast(cfg.device, torch.bfloat16, enabled=cfg.autocast):
                policy, q = policy_q(model, b)
            return policy.float().cpu(), q.float().cpu()

    return evaluate


def _rebuild(samples: list[Sample]):
    """Buffer states back into one batch by parallel replay (design doc §12:
    a position is a move prefix). Refuses a sample whose stored π′ no longer
    matches its position's legal count — that misalignment trains against
    scrambled targets and has no downstream symptom."""
    batch = collate_prefixes([s.moves for s in samples], [s.t for s in samples])
    counts = (batch.legal_offsets[1:] - batch.legal_offsets[:-1]).tolist()
    for s, count in zip(samples, counts):
        if count != len(s.improved):
            raise ValueError(
                f"sample at ply {s.t}: stored pi' has {len(s.improved)} entries, "
                f"position has {count} legal moves"
            )
    return batch


def _pack(samples: list[Sample], order, cfg: KlentConfig) -> list[list[int]]:
    """Pack sample indices into fit chunks under ``batch_size`` and both
    memory budgets. Sorted by position size (descending, ties in ``order``'s
    random order), so a chunk's padded attention cost is exact — everyone
    pads to its first element — and one long sample can no longer pad a
    whole mixed chunk up to its own square. A sample too big for the budgets
    alone still gets its own chunk: the buffer is never silently dropped."""
    idx = sorted(order, key=lambda i: samples[i].t, reverse=True)
    chunks: list[list[int]] = []
    chunk: list[int] = []
    chunk_t, cells = 0, 0
    for i in idx:
        t_pad = samples[i].t + 1  # + the global token row
        c = len(samples[i].improved)
        if chunk and (
            len(chunk) == cfg.batch_size
            or (len(chunk) + 1) * chunk_t * chunk_t > cfg.pair_budget
            or cells + c > cfg.cell_budget
        ):
            chunks.append(chunk)
            chunk, cells = [], 0
        if not chunk:
            chunk_t = t_pad
        chunk.append(int(i))
        cells += c
    if chunk:
        chunks.append(chunk)
    return chunks


def fit(
    model,
    samples: list[Sample],
    optimizer,
    cfg: KlentConfig,
    rng: np.random.Generator,
    progress=None,
):
    """One epoch over the buffer at the paper's *effective* batch.

    ``progress(chunk, chunks)`` is called after each consumed chunk — an
    observer for heartbeats, nothing more.

    The memory budgets cap what one forward may hold, so chunks accumulate
    sample-weighted gradients until ~``batch_size`` samples have contributed
    and the optimizer steps once — the paper's batch statistics under the
    packing. A step's gradient equals the mean loss over its whole
    accumulated batch, so the chunking is an implementation detail of memory
    and not of optimization. Returns the sample-weighted mean losses.

    Two throughput facts. Chunk *preparation* (the Rust replay rebuild and
    the target tensors) runs one chunk ahead on a worker thread, so the GPU
    never waits on it. And the loss scalars accumulate on-device — the only
    host sync is the single read at the end; per-chunk ``float()`` reads
    were measured to stall the stream once per chunk."""
    model.train()
    policy_q = _policy_q_fn(cfg)
    chunks = _pack(samples, rng.permutation(len(samples)), cfg)

    groups: list[tuple[list[int], int]] = []
    group: list[int] = []
    count = 0
    for k in rng.permutation(len(chunks)):
        group.append(int(k))
        count += len(chunks[k])
        if count >= cfg.batch_size:
            groups.append((group, count))
            group, count = [], 0
    if group:
        groups.append((group, count))

    def prep(k: int):
        chunk = [samples[i] for i in chunks[k]]
        batch = _rebuild(chunk)
        target = torch.from_numpy(np.concatenate([s.improved for s in chunk]))
        ranks = torch.tensor([s.rank for s in chunk])
        returns = torch.tensor([s.g for s in chunk], dtype=torch.float32)
        return chunk, batch, target, ranks, returns

    order = [k for group, _ in groups for k in group]
    policy_sum = torch.zeros((), device=cfg.device)
    q_sum = torch.zeros((), device=cfg.device)
    total = 0
    with ThreadPoolExecutor(max_workers=1) as pool:
        prepped = {k: pool.submit(prep, k) for k in order[:1]}
        consumed = 0
        for group, group_n in groups:
            optimizer.zero_grad(set_to_none=True)
            for k in group:
                if consumed + 1 < len(order):
                    nxt = order[consumed + 1]
                    prepped[nxt] = pool.submit(prep, nxt)
                chunk, batch, target, ranks, returns = prepped.pop(k).result()
                consumed += 1
                with _gpu_lock:
                    batch = batch.to(cfg.device)
                    target = target.to(cfg.device)
                    ranks = ranks.to(cfg.device)
                    returns = returns.to(cfg.device)

                    with torch.autocast(cfg.device, torch.bfloat16, enabled=cfg.autocast):
                        policy_logits, q_values = policy_q(model, batch)
                    ce = policy_loss(policy_logits.float(), batch.legal_offsets, target)
                    taken = q_values.float().index_select(
                        0, batch.legal_offsets[:-1] + ranks
                    )
                    q_mse = (taken - returns).square().mean()

                    ((ce + q_mse) * (len(chunk) / group_n)).backward()
                    policy_sum += ce.detach() * len(chunk)
                    q_sum += q_mse.detach() * len(chunk)
                if progress is not None:
                    progress(consumed, len(order))
            optimizer.step()
            total += group_n
    return {
        "policy_loss": float(policy_sum) / total,
        "q_loss": float(q_sum) / total,
        "fit_steps": len(groups),
    }


def collect_episodes(
    model, collector: Collector, cfg: KlentConfig, progress=None
) -> tuple[list, dict]:
    """One iteration's corpus: episodes plus the §13 collection metrics.

    Worker-safe by construction — it draws only from the collector's own
    stream and mutates only the collector's slots, which is what lets the
    run driver collect iteration ``i+1`` on a weight snapshot while
    iteration ``i`` fits on the live model. ``progress`` is the collector's
    per-step observer."""
    model.eval()
    episodes, metrics = collector.collect(
        network_evaluate(model, cfg), cfg.games_per_iteration, progress
    )
    metrics.update(collection_stats(episodes))
    return episodes, metrics
