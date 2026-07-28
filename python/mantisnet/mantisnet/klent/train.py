"""The KLENT iteration: collect an on-policy buffer, fit once, discard it.

Faithful to the paper's outer loop (design doc §1): a self-play phase fills
the buffer, one fitting epoch consumes it (O2's assumption), and nothing
survives to the next iteration. The loss is eq. 4 — cross-entropy of π_θ
against π′ plus squared error of the taken action's Q against the λ-return —
under plain Adam at the paper's learning rate. The value head appears
nowhere: KLENT has no state-value head, and v̂ = E_{π′}[Q] does its job.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..builder import collate_prefixes
from ..losses import policy_loss
from .seeds import line_evaluate, seed_prefix
from .selfplay import Sample, collection_stats, episode_samples, play_episodes


@dataclass
class KlentConfig:
    """Design doc §2's starting values, expected to move — not defaults to trust."""

    # Verified against the paper's eq. 2 (2026-07-27): reverse KL is the
    # heavier regulariser. Prior exponent tau/(tau+lam) = 0.77.
    tau: float = 0.1  # reverse-KL weight (the paper's beta)
    lam: float = 0.03  # entropy weight (the paper's alpha)
    # e^{-1/16}: the paper's 8-turn horizon at Hexo's two placements per turn
    # (KLENT_PROPOSALS A1). The paper's literal e^{-1/8} would halve it.
    lam_ret: float = 0.939
    ply_cap: int = 512  # §5: capped episodes are dropped whole
    games_per_iteration: int = 128
    seed_fraction: float = 1.0  # §5.2: anneal toward zero as f allows
    seed_cut: tuple = (1, 8)  # plies cut from a won seed game's end
    seed_noise: float = 0.1
    batch_size: int = 4096  # paper's *effective* batch: chunks accumulate to it
    lr: float = 1e-3  # paper's Adam rate
    device: str = "cpu"
    autocast: bool = False  # bf16 autocast for the network passes
    compile: bool = False  # torch.compile the policy/Q pass (one-time cost)
    # VRAM is budgeted, not hoped for: every network batch — fit chunk or
    # collection cohort — is packed under both measured memory axes, so the
    # peak is set here rather than by whatever the corpus happens to contain.
    # Attention memory is quadratic in the batch's longest position (padding),
    # decoder memory linear in its total legal cells.
    pair_budget: int = 8_000_000  # padded (stones + token)^2 pairs per batch
    cell_budget: int = 400_000  # legal cells (decoder rows) per batch


def _policy_q(model, batch):
    """The KLENT pass: trunk + the two heads it trains, never the value head."""
    _s, w, g = model.trunk(batch)
    return model.policy_head(w, g, batch), model.q_head(w, g, batch)


# One symbolic-shape graph serves every batch; compiled lazily, shared by
# collection and fitting (the same graph gets the compiled backward).
_policy_q_compiled = None


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


def fit(model, samples: list[Sample], optimizer, cfg: KlentConfig, rng: np.random.Generator):
    """One epoch over the buffer at the paper's *effective* batch.

    The memory budgets cap what one forward may hold, so chunks accumulate
    sample-weighted gradients until ~``batch_size`` samples have contributed
    and the optimizer steps once — the paper's batch statistics under the
    packing. A step's gradient equals the mean loss over its whole
    accumulated batch, so the chunking is an implementation detail of memory
    and not of optimization. Returns the sample-weighted mean losses."""
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

    policy_sum, q_sum, total = 0.0, 0.0, 0
    for group, group_n in groups:
        optimizer.zero_grad(set_to_none=True)
        for k in group:
            chunk = [samples[i] for i in chunks[k]]
            batch = _rebuild(chunk).to(cfg.device)
            target = torch.from_numpy(np.concatenate([s.improved for s in chunk])).to(cfg.device)
            ranks = torch.tensor([s.rank for s in chunk], device=cfg.device)
            returns = torch.tensor([s.g for s in chunk], dtype=torch.float32, device=cfg.device)

            with torch.autocast(cfg.device, torch.bfloat16, enabled=cfg.autocast):
                policy_logits, q_values = policy_q(model, batch)
            ce = policy_loss(policy_logits.float(), batch.legal_offsets, target)
            taken = q_values.float().index_select(0, batch.legal_offsets[:-1] + ranks)
            q_mse = (taken - returns).square().mean()

            ((ce + q_mse) * (len(chunk) / group_n)).backward()
            policy_sum += float(ce.detach()) * len(chunk)
            q_sum += float(q_mse.detach()) * len(chunk)
        optimizer.step()
        total += group_n
    return {
        "policy_loss": policy_sum / total,
        "q_loss": q_sum / total,
        "fit_steps": len(groups),
    }


def iterate(
    model, optimizer, cfg: KlentConfig, rng: np.random.Generator, warm: bool = False
) -> dict:
    """One full KLENT iteration. Returns the §13 first-class metrics; an
    empty buffer (f = 0) skips fitting rather than failing — that outcome is
    the signal the seeding knobs exist to move.

    ``warm`` is the bootstrap phase: collection acts through the line
    builder's scores instead of the network, because an honestly-initialized
    π′ is near-uniform and finishes almost no seeded games — measured, not
    supposed. Warm returns are pure Monte-Carlo outcomes (λ_ret = 1): the
    heuristic's v̂ lives on an arbitrary scale and must not bootstrap."""
    prefixes = [
        seed_prefix(rng, cfg.seed_cut, cfg.seed_noise)
        if rng.random() < cfg.seed_fraction
        else []
        for _ in range(cfg.games_per_iteration)
    ]
    model.eval()
    episodes, metrics = play_episodes(
        line_evaluate if warm else network_evaluate(model, cfg),
        prefixes,
        cfg.ply_cap,
        cfg.tau,
        cfg.lam,
        rng,
        pair_budget=cfg.pair_budget,
        cell_budget=cfg.cell_budget,
    )
    metrics.update(collection_stats(episodes))
    metrics["warm"] = int(warm)

    samples = [
        s for e in episodes for s in episode_samples(e, 1.0 if warm else cfg.lam_ret)
    ]
    metrics["buffer_samples"] = len(samples)
    if samples:
        metrics.update(fit(model, samples, optimizer, cfg, rng))
    return metrics
