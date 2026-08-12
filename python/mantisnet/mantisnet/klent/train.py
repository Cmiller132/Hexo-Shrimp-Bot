"""Collect and fit one KLENT iteration.

Each iteration consumes one on-policy buffer for one fitting epoch and then
discards it. The objective is policy cross-entropy against π′ plus one
categorical cross-entropy on the taken action's positive, negative, and zero
return masses. The reported Q squared error is measured, not trained.
The state-value head is not part of this path. See ``docs/KLENT_FOR_HEXO.md``
§3 and §5.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ..builder import collate_prefixes
from ..fitloop import FitBudgets, fit_epoch, pack_chunks
from ..losses import policy_loss
from ..model import compose_acting_q, compose_q
from ..optim import resolve_adam_implementation
from .selfplay import Collector, Sample, collection_stats


@dataclass
class KlentConfig:
    """KLENT training and batching parameters; see ``KLENT_FOR_HEXO.md`` §10."""

    tau: float = 0.1  # reverse-KL weight (the paper's beta)
    lam: float = 0.03  # entropy weight (the paper's alpha)
    # Smallest committed mass π′ measures Q against; this bounds sharpening
    # when a position's legal set assigns almost all probability to zero return.
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
    # Execution policy only: all choices implement the same Adam recipe. Auto
    # resolves to fused on CUDA and scalar on CPU, and the resolved choice is
    # recorded beside the run config.
    adam_impl: str = "auto"
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

    def __post_init__(self) -> None:
        resolve_adam_implementation(self.adam_impl, self.device)


def _policy_q(model, batch):
    """The KLENT pass: trunk + the two heads it trains, never the value head.

    It returns policy logits and the critic's raw categorical logits. The
    fitter scores the taken row; acting composes both Q roles outside.
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
        # Same room as the supervised compile helper: budget-packed chunks
        # split into more size-range graphs than the stock limit of 8.
        torch._dynamo.config.recompile_limit = 64
        _policy_q_compiled = torch.compile(_policy_q, dynamic=True)
    return _policy_q_compiled


def network_evaluate(model, cfg: KlentConfig):
    """The self-play evaluator, returning flat CPU tensors so the collection
    loop stays device-ignorant: policy logits, acting score, and action value."""
    policy_q = _policy_q_fn(cfg)

    def evaluate(batch):
        with _gpu_lock:
            b = batch.to(cfg.device)
            with torch.no_grad(), torch.autocast(cfg.device, torch.bfloat16, enabled=cfg.autocast):
                policy, critic_logits = policy_q(model, b)
            # Composition is fp32 and outside autocast, so neither Q role
            # depends on autocast precision.
            return (
                policy.float().cpu(),
                compose_acting_q(
                    critic_logits, b.legal_offsets, cfg.mass_floor
                ).cpu(),
                compose_q(critic_logits).cpu(),
            )

    evaluate.state_latents = model.cfg.state_latents
    return evaluate


def _rebuild(samples: list[Sample], state_latents: int = 0):
    """Rebuild buffered move prefixes in parallel into one batch.

    Each stored π′ length must equal its replayed position's legal count
    (``KLENT_FOR_HEXO.md`` §4.3).
    """
    batch = collate_prefixes(
        [s.moves for s in samples],
        [s.t for s in samples],
        state_latents=state_latents,
    )
    counts = (batch.legal_offsets[1:] - batch.legal_offsets[:-1]).tolist()
    for s, count in zip(samples, counts):
        if count != len(s.improved):
            raise ValueError(
                f"sample at ply {s.t}: stored pi' has {len(s.improved)} entries, "
                f"position has {count} legal moves"
            )
    return batch


def _pack(
    samples: list[Sample], order, cfg: KlentConfig, state_latents: int = 0
) -> list[list[int]]:
    """Compatibility wrapper for the shared fit-loop packer."""
    return pack_chunks(
        [s.t + max(1, state_latents) for s in samples],
        [len(s.improved) for s in samples],
        order,
        FitBudgets(cfg.batch_size, cfg.pair_budget, cfg.cell_budget),
    )


def fit(
    model,
    samples: list[Sample],
    optimizer,
    cfg: KlentConfig,
    rng: np.random.Generator,
    progress=None,
    steady: tuple[int, int] | None = None,
):
    """Fit one epoch over the buffer.

    ``progress(consumed, chunks)`` is called after each consumed chunk.

    The memory budgets cap what one forward may hold, so chunks accumulate
    sample-weighted gradients until at least ``batch_size`` samples have
    contributed and the optimizer steps once. Each accumulated gradient is
    the mean over its group. Chunk preparation runs pipelined ahead, and loss
    totals remain on-device until the returned sample-weighted means are read."""
    model.train()
    policy_q = _policy_q_fn(cfg)
    state_latents = model.cfg.state_latents

    pin = torch.device(cfg.device).type == "cuda"

    def prep(indices: list[int]):
        chunk = [samples[i] for i in indices]
        batch = _rebuild(chunk, state_latents)
        target = torch.from_numpy(np.concatenate([s.improved for s in chunk]))
        ranks = torch.tensor([s.rank for s in chunk])
        returns = torch.tensor([s.g for s in chunk], dtype=torch.float32)
        if pin:
            # Pinned here, on the prefetch worker, so the step's ``.to`` is a
            # true async DMA; from pageable memory every transfer degrades to
            # a staged synchronous copy on the training thread.
            batch = batch.pin_memory()
            target = target.pin_memory()
            ranks = ranks.pin_memory()
            returns = returns.pin_memory()
        return batch, target, ranks, returns

    def fit_step(payload):
        batch, target, ranks, returns = payload
        batch = batch.to(cfg.device)
        target = target.to(cfg.device, non_blocking=True)
        ranks = ranks.to(cfg.device, non_blocking=True)
        returns = returns.to(cfg.device, non_blocking=True)

        with torch.autocast(cfg.device, torch.bfloat16, enabled=cfg.autocast):
            policy_logits, critic_logits = policy_q(model, batch)
        ce = policy_loss(policy_logits.float(), batch.legal_offsets, target)
        # G in [-1, 1] makes (G⁺, G⁻, 1-|G|) a distribution. Its
        # categorical CE is the critic's sole trained term.
        taken = critic_logits.index_select(
            0, batch.legal_offsets[:-1] + ranks
        ).float()
        critic_target = torch.stack(
            (
                returns.clamp(min=0.0),
                (-returns).clamp(min=0.0),
                1.0 - returns.abs(),
            ),
            dim=-1,
        )
        critic_ce = -(
            critic_target * F.log_softmax(taken, dim=-1)
        ).sum(dim=-1).mean()
        # Cross-arm curve only: measured under no_grad and absent from the
        # objective, so it cannot double-cover Q.
        with torch.no_grad():
            q_mse = (compose_q(taken) - returns).square().mean()

        return ce + critic_ce, {
            "policy_loss": ce.detach(),
            # Measured diagnostic, not a trained term.
            "q_loss": q_mse,
            "critic_ce": critic_ce.detach(),
        }

    return fit_epoch(
        model,
        optimizer,
        rng,
        lengths=[s.t + max(1, state_latents) for s in samples],
        cells=[len(s.improved) for s in samples],
        budgets=FitBudgets(cfg.batch_size, cfg.pair_budget, cfg.cell_budget),
        prepare=prep,
        step=fit_step,
        lock=_gpu_lock,
        progress=progress,
        steady=steady,
    )


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
