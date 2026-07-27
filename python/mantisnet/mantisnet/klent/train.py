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

from ..builder import collate, from_position
from ..losses import policy_loss
from .seeds import seed_prefix
from .selfplay import Sample, episode_samples, play_episodes


@dataclass
class KlentConfig:
    """Design doc §2's starting values, expected to move — not defaults to trust."""

    tau: float = 0.03  # reverse-KL weight
    lam: float = 0.1  # entropy weight
    lam_ret: float = 0.883  # λ-return decay, the paper's e^{-1/8}
    ply_cap: int = 512  # §5: capped episodes are dropped whole
    games_per_iteration: int = 128
    seed_fraction: float = 1.0  # §5.2: anneal toward zero as f allows
    seed_cut: tuple = (1, 8)  # plies cut from a won seed game's end
    seed_noise: float = 0.1
    batch_size: int = 4096  # paper's fitting batch
    lr: float = 1e-3  # paper's Adam rate
    device: str = "cpu"
    autocast: bool = False  # bf16 autocast for the network passes


def network_evaluate(model, cfg: KlentConfig):
    """The self-play evaluator: trunk + policy + Q, never the value head.

    Returns flat CPU tensors so the collection loop stays device-ignorant.
    """

    def evaluate(batch):
        b = batch.to(cfg.device)
        with torch.no_grad(), torch.autocast(cfg.device, torch.bfloat16, enabled=cfg.autocast):
            _s, w, g = model.trunk(b)
            policy = model.policy_head(w, g, b)
            q = model.q_head(w, g, b)
        return policy.float().cpu(), q.float().cpu()

    return evaluate


def _rebuild(samples: list[Sample]):
    """Buffer states back into graphs by replay (design doc §12: a position
    is a move prefix). Refuses a sample whose stored π′ no longer matches the
    position's legal count — that misalignment trains against scrambled
    targets and has no downstream symptom."""
    import hexo_py

    graphs = []
    for s in samples:
        pos = hexo_py.Position.replay(list(s.moves[: s.t]))
        graph = from_position(pos)
        if graph.n_legal != len(s.improved):
            raise ValueError(
                f"sample at ply {s.t}: stored pi' has {len(s.improved)} entries, "
                f"position has {graph.n_legal} legal moves"
            )
        graphs.append(graph)
    return graphs


def fit(model, samples: list[Sample], optimizer, cfg: KlentConfig, rng: np.random.Generator):
    """One epoch over the buffer. Returns the mean loss components."""
    model.train()
    order = rng.permutation(len(samples))
    policy_sum, q_sum, steps = 0.0, 0.0, 0
    for start in range(0, len(order), cfg.batch_size):
        chunk = [samples[i] for i in order[start : start + cfg.batch_size]]
        batch = collate(_rebuild(chunk)).to(cfg.device)
        target = torch.from_numpy(np.concatenate([s.improved for s in chunk])).to(cfg.device)
        ranks = torch.tensor([s.rank for s in chunk], device=cfg.device)
        returns = torch.tensor([s.g for s in chunk], dtype=torch.float32, device=cfg.device)

        with torch.autocast(cfg.device, torch.bfloat16, enabled=cfg.autocast):
            _s, w, g = model.trunk(batch)
            policy_logits = model.policy_head(w, g, batch)
            q_values = model.q_head(w, g, batch)
        ce = policy_loss(policy_logits.float(), batch.legal_offsets, target)
        taken = q_values.float().index_select(0, batch.legal_offsets[:-1] + ranks)
        q_mse = (taken - returns).square().mean()

        optimizer.zero_grad(set_to_none=True)
        (ce + q_mse).backward()
        optimizer.step()
        policy_sum += float(ce.detach())
        q_sum += float(q_mse.detach())
        steps += 1
    return {"policy_loss": policy_sum / steps, "q_loss": q_sum / steps, "fit_steps": steps}


def iterate(model, optimizer, cfg: KlentConfig, rng: np.random.Generator) -> dict:
    """One full KLENT iteration. Returns the §13 first-class metrics; an
    empty buffer (f = 0) skips fitting rather than failing — that outcome is
    the signal the seeding knobs exist to move."""
    prefixes = [
        seed_prefix(rng, cfg.seed_cut, cfg.seed_noise)
        if rng.random() < cfg.seed_fraction
        else []
        for _ in range(cfg.games_per_iteration)
    ]
    model.eval()
    episodes, metrics = play_episodes(
        network_evaluate(model, cfg), prefixes, cfg.ply_cap, cfg.tau, cfg.lam, rng
    )

    def fraction(eps):
        return sum(e.winner is not None for e in eps) / len(eps) if eps else float("nan")

    metrics["f_seeded"] = fraction([e for e in episodes if e.seed_len > 0])
    metrics["f_unseeded"] = fraction([e for e in episodes if e.seed_len == 0])
    won = [e for e in episodes if e.winner is not None]
    metrics["won_length_mean"] = (
        sum(len(e.moves) for e in won) / len(won) if won else float("nan")
    )

    samples = [s for e in episodes for s in episode_samples(e, cfg.lam_ret)]
    metrics["buffer_samples"] = len(samples)
    if samples:
        metrics.update(fit(model, samples, optimizer, cfg, rng))
    return metrics
