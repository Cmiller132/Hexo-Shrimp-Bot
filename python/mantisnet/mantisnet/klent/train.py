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

from ..fitloop import FitBudgets, fit_epoch, pack_chunks
from ..losses import policy_loss
from ..model import compose_acting_q, compose_q
from ..models.mantis_act import ACT_GRAPH_CELL_BUDGET
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
    device: str = "cpu"
    autocast: bool = False  # bf16 autocast for the network passes
    compile: bool = False  # torch.compile the policy/Q pass (one-time cost)
    # Memory budgets for batch packing; fit and collection are separate
    # because fit holds the backward graph.
    pair_budget: int = 8_000_000  # fit: padded (stones + token)^2 pairs
    cell_budget: int = 800_000  # fit: legal cells (decoder rows)
    graph_cell_budget: int = ACT_GRAPH_CELL_BUDGET  # fit: ACT graph cells
    collect_pair_budget: int = 24_000_000  # collection (no_grad)
    collect_cell_budget: int = 2_400_000
    collect_graph_cell_budget: int = 3 * ACT_GRAPH_CELL_BUDGET


def _policy_q(model, batch):
    """The KLENT pass: policy logits and critic categorical logits."""
    return model.policy_q(batch)


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
    """Self-play evaluator returning flat CPU tensors: policy logits, acting
    score, and action value."""
    policy_q = _policy_q_fn(cfg)

    def evaluate(batch):
        with _gpu_lock:
            b = batch.to(cfg.device)
            with torch.no_grad(), torch.autocast(cfg.device, torch.bfloat16, enabled=cfg.autocast):
                policy, critic_logits = policy_q(model, b)
            # Q composition runs in fp32, outside autocast.
            return (
                policy.float().cpu(),
                compose_acting_q(
                    critic_logits, b.legal_offsets, cfg.mass_floor
                ).cpu(),
                compose_q(critic_logits).cpu(),
            )

    return evaluate


def _rebuild(model, samples: list[Sample]):
    """Rebuild buffered move prefixes into one batch of ``model``'s input.

    Each stored pi' length must equal its replayed position's legal count.
    """
    batch = model.collate_prefixes(
        [s.moves for s in samples], [s.t for s in samples]
    )
    counts = (batch.legal_offsets[1:] - batch.legal_offsets[:-1]).tolist()
    for s, count in zip(samples, counts):
        if count != len(s.improved):
            raise ValueError(
                f"sample at ply {s.t}: stored pi' has {len(s.improved)} entries, "
                f"position has {count} legal moves"
            )
    return batch


def _budgets(cfg: KlentConfig) -> FitBudgets:
    """The fitting limits this run offers, whichever architecture reads them."""
    return FitBudgets(
        pair_budget=cfg.pair_budget,
        cell_budget=cfg.cell_budget,
        graph_cell_budget=cfg.graph_cell_budget,
    )


def _chunk_cost(model, samples: list[Sample], cfg: KlentConfig):
    """``model``'s packing law over this buffer's stone and legal counts."""
    return model.chunk_cost(
        [s.t for s in samples],
        [len(s.improved) for s in samples],
        _budgets(cfg),
    )


def _pack(model, samples: list[Sample], order, cfg: KlentConfig) -> list[list[int]]:
    """The chunks one epoch over ``samples`` would fit, in packing order."""
    return pack_chunks(order, cfg.batch_size, _chunk_cost(model, samples, cfg))


def fit(
    model,
    samples: list[Sample],
    optimizer,
    cfg: KlentConfig,
    rng: np.random.Generator,
    progress=None,
):
    """Fit one epoch over the buffer.

    ``progress(consumed, chunks)`` is called after each consumed chunk.
    """
    model.train()
    policy_q = _policy_q_fn(cfg)

    def prep(indices: list[int]):
        chunk = [samples[i] for i in indices]
        batch = _rebuild(model, chunk)
        target = torch.from_numpy(np.concatenate([s.improved for s in chunk]))
        ranks = torch.tensor([s.rank for s in chunk])
        returns = torch.tensor([s.g for s in chunk], dtype=torch.float32)
        return batch, target, ranks, returns

    def fit_step(payload):
        batch, target, ranks, returns = payload
        batch = batch.to(cfg.device)
        target = target.to(cfg.device)
        ranks = ranks.to(cfg.device)
        returns = returns.to(cfg.device)

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
        sample_count=len(samples),
        batch_size=cfg.batch_size,
        cost=_chunk_cost(model, samples, cfg),
        prepare=prep,
        step=fit_step,
        lock=_gpu_lock,
        progress=progress,
    )


def collect_episodes(
    model, collector: Collector, cfg: KlentConfig, progress=None
) -> tuple[list, dict]:
    """Collect one iteration's episodes and section 8 metrics.

    ``progress`` is called once per collector step.
    """
    model.eval()
    episodes, metrics = collector.collect(
        network_evaluate(model, cfg), cfg.games_per_iteration, progress
    )
    metrics.update(collection_stats(episodes))
    return episodes, metrics
