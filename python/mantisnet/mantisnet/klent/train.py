"""Collect and fit one KLENT iteration.

Each iteration consumes one on-policy buffer for one fitting epoch and then
discards it. The objective is policy cross-entropy against π′ plus the taken
action's critic loss: squared error of the composed Q against its λ-return,
plus the weighted binary cross-entropies that calibrate the two return masses.
The state-value head is not part of this path. See ``docs/KLENT_FOR_HEXO.md``
§3 and §5.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ..builder import collate_prefixes
from ..losses import policy_loss
from ..model import compose_acting_q, compose_q
from .selfplay import Collector, Sample, collection_stats


@dataclass
class KlentConfig:
    """KLENT training and batching parameters; see ``KLENT_FOR_HEXO.md`` §10."""

    tau: float = 0.1  # reverse-KL weight (the paper's beta)
    lam: float = 0.03  # entropy weight (the paper's alpha)
    # eta: weight on the critic's return-mass calibration term. The squared
    # error of the composed Q keeps unit weight; this scales the pair of mass
    # cross-entropies that fix where that Q's two halves sit.
    mass_weight: float = 0.25
    # The smallest total return mass π′ will measure Q against, which bounds
    # the sharpening in a position where the whole legal set is uncommitted.
    # Measured on joint-brm-939 at iteration 300 the largest mass in a legal
    # set runs 0.12 unresolved to 0.91 decided, so this binds rarely.
    mass_floor: float = 0.2
    # e^-1/16 corresponds to an eight-turn horizon at two placements per turn.
    lam_ret: float = 0.939
    # Per-ply return-discount magnitude (the mover-change sign is separate).
    # Values below one give earlier outcomes larger magnitude.
    gamma: float = 1.0
    ply_cap: int = 512  # KLENT_FOR_HEXO.md §4.2: capped episodes are dropped whole
    # The completion quota: an iteration's buffer is at least this many
    # *finished* games, from however many slots are in flight.
    games_per_iteration: int = 4096
    envs: int = 1024  # persistent self-play slots (the reference's env count)
    batch_size: int = 4096  # paper's *effective* batch: chunks accumulate to it
    lr: float = 1e-3  # paper's Adam rate
    device: str = "cpu"
    autocast: bool = False  # bf16 autocast for the network passes
    compile: bool = False  # torch.compile the policy/Q pass (one-time cost)
    # Every network batch is packed under attention-pair and legal-cell budgets.
    # Attention memory is quadratic in the batch's longest position (padding),
    # decoder memory linear in its total legal cells. Fit and collection get
    # separate budgets because fit holds the backward graph per cell while
    # collection runs no_grad. Both allocations may be resident concurrently.
    pair_budget: int = 8_000_000  # fit: padded (stones + token)^2 pairs per batch
    cell_budget: int = 800_000  # fit: legal cells (decoder rows) per batch
    collect_pair_budget: int = 24_000_000  # collection (no_grad) equivalents
    collect_cell_budget: int = 2_400_000


def _policy_q(model, batch):
    """The KLENT pass: trunk + the two heads it trains, never the value head.

    It returns the raw pair — policy logits and the critic's mass logits —
    because that is what both callers need from the graph: the fitter takes
    binary losses on the logits, and the acting seam composes Q outside.
    """
    _s, w, g = model.trunk(batch)
    return model.cell_head_logits(w, g, batch)


# One symbolic-shape graph serves every batch; compiled lazily, shared by
# collection and fitting (the same graph gets the compiled backward).
_policy_q_compiled = None

# The lock serializes calls and compilation through the shared callable.
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
    loop stays device-ignorant.

    Three tensors: the policy logits, the score π′ ranks by, and the action
    value v̂ averages. The last two differ by the position's own committed
    return mass."""
    policy_q = _policy_q_fn(cfg)

    def evaluate(batch):
        with _gpu_lock:
            b = batch.to(cfg.device)
            with torch.no_grad(), torch.autocast(cfg.device, torch.bfloat16, enabled=cfg.autocast):
                policy, critic_logits = policy_q(model, b)
            # Composition is fp32 and outside the autocast region, so neither
            # Q the operator consumes depends on autocast.
            return (
                policy.float().cpu(),
                compose_acting_q(
                    critic_logits, b.legal_offsets, cfg.mass_floor
                ).cpu(),
                compose_q(critic_logits).cpu(),
            )

    return evaluate


def _refuse_nonfinite_parameters(model, step: int) -> None:
    """Refuse a weight update that left the finite range, at the step that did it.

    A single non-finite parameter makes every later evaluation non-finite, and
    the first thing that notices is the return recursion refusing a whole
    corpus one collection later — which names neither the step nor the tensor.
    One fused multi-tensor norm and one synchronization per optimizer step buy
    that name; the per-parameter walk runs only when the check has already
    failed.
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


def _rebuild(samples: list[Sample]):
    """Rebuild buffered move prefixes in parallel into one batch.

    Each stored π′ length must equal its replayed position's legal count
    (``KLENT_FOR_HEXO.md`` §4.3).
    """
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
    random order), so each chunk pads to its first element. A sample that
    exceeds a budget alone occupies its own chunk.
    """
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
    """Fit one epoch over the buffer.

    ``progress(chunk, chunks)`` is called after each consumed chunk.

    The memory budgets cap what one forward may hold, so chunks accumulate
    sample-weighted gradients until at least ``batch_size`` samples have
    contributed and the optimizer steps once. Each accumulated gradient is
    the mean over its group. Chunk preparation runs one chunk ahead, and loss
    totals remain on-device until the returned sample-weighted means are read."""
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
    mass_sum = torch.zeros((), device=cfg.device)
    total = 0
    step = 0
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
                        policy_logits, critic_logits = policy_q(model, batch)
                    ce = policy_loss(policy_logits.float(), batch.legal_offsets, target)
                    # The critic loss is the taken action's only: its composed
                    # Q against the λ-return, plus the two masses against the
                    # return's positive and negative parts. Soft targets are
                    # intended — at the cross-entropy optimum each mass is the
                    # conditional expectation of its part, so their difference
                    # is E[G | s, a]. Composition and losses are fp32.
                    taken = critic_logits.index_select(
                        0, batch.legal_offsets[:-1] + ranks
                    ).float()
                    q_mse = (compose_q(taken) - returns).square().mean()
                    z_pos, z_neg = taken.unbind(dim=-1)
                    mass = F.binary_cross_entropy_with_logits(
                        z_pos, returns.clamp(min=0.0)
                    ) + F.binary_cross_entropy_with_logits(
                        z_neg, (-returns).clamp(min=0.0)
                    )

                    loss = ce + q_mse + 0.5 * cfg.mass_weight * mass
                    (loss * (len(chunk) / group_n)).backward()
                    policy_sum += ce.detach() * len(chunk)
                    q_sum += q_mse.detach() * len(chunk)
                    mass_sum += mass.detach() * len(chunk)
                if progress is not None:
                    progress(consumed, len(order))
            optimizer.step()
            step += 1
            _refuse_nonfinite_parameters(model, step)
            total += group_n
    return {
        "policy_loss": float(policy_sum) / total,
        "q_loss": float(q_sum) / total,
        # The unweighted mass bracket, so the calibration reads independently
        # of eta.
        "mass_loss": float(mass_sum) / total,
        "fit_steps": len(groups),
    }


def collect_episodes(
    model, collector: Collector, cfg: KlentConfig, progress=None
) -> tuple[list, dict]:
    """One iteration's corpus: episodes plus the
    ``KLENT_FOR_HEXO.md`` §8 collection metrics.

    It consumes only the collector's RNG and mutates only collector slots, so
    it may run on a worker against a weight snapshot. ``progress`` is called
    once per collector step."""
    model.eval()
    episodes, metrics = collector.collect(
        network_evaluate(model, cfg), cfg.games_per_iteration, progress
    )
    metrics.update(collection_stats(episodes))
    return episodes, metrics
