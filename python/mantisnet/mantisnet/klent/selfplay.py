"""Persistent-slot self-play collector for KLENT (``KLENT_FOR_HEXO.md`` section 4)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from ..builder import collate_positions
from .improve import improved_policy
from .returns import lambda_returns, signs_from_moves_remaining

# Evaluators return flat CPU ``(policy_logits, q_values)`` for one batch.
Evaluate = Callable[[object], tuple[torch.Tensor, torch.Tensor]]


@dataclass
class Episode:
    """One episode from the empty board with per-ply records."""

    moves: list = field(default_factory=list)
    winner: int | None = None  # None exactly when the cap hit
    moves_remaining: list = field(default_factory=list)  # per ply, at acting time
    movers: list = field(default_factory=list)  # per ply, the engine's mover
    ranks: list = field(default_factory=list)  # per ply, the action's legal rank
    improved: list = field(default_factory=list)  # per ply, π′ over the legal set
    v_hats: list = field(default_factory=list)  # per ply, E_{π′}[Q] at acting time
    kls: list = field(default_factory=list)  # per ply, D_KL(π′ ‖ π_θ)
    norm_entropies: list = field(default_factory=list)  # per ply, H(π′)/log|A|
    pi_top1: list = field(default_factory=list)  # per ply, max π′
    pi_chosen: list = field(default_factory=list)  # per ply, π′ of the sampled move


@dataclass
class Sample:
    """One buffer entry: `(S_t, A_t, π′, G_t)` with the state as a move prefix."""

    moves: tuple  # episode move list; S_t = replay(moves[:t])
    t: int
    rank: int  # A_t as its index in engine legal order
    improved: np.ndarray  # π′(·|S_t)
    g: float  # λ-return, in the mover-at-t's frame


def _chunk_live(
    positions,
    live: list[int],
    pair_budget: int,
    cell_budget: int,
    cap: int,
    state_latents: int = 0,
):
    """Split ``live`` into consecutive chunks under the memory budgets and
    the position-count ``cap``."""
    chunks: list[list[int]] = []
    chunk: list[int] = []
    max_t, cells = 0, 0
    for i in live:
        t_pad = positions[i].stone_count + max(1, state_latents)
        c = positions[i].legal_count
        t = max(max_t, t_pad)
        if chunk and (
            len(chunk) == cap
            or (len(chunk) + 1) * t * t > pair_budget
            or cells + c > cell_budget
        ):
            chunks.append(chunk)
            chunk, max_t, cells = [], 0, 0
            t = t_pad
        chunk.append(i)
        max_t, cells = t, cells + c
    if chunk:
        chunks.append(chunk)
    return chunks


class Collector:
    """Persistent self-play cohort of ``envs`` slots.

    ``collect`` returns at least ``episodes`` completed games; in-flight
    games persist to the next call.
    """

    def __init__(
        self,
        envs: int,
        ply_cap: int,
        tau: float,
        lam: float,
        rng: np.random.Generator,
        pair_budget: int = 24_000_000,
        cell_budget: int = 2_400_000,
        state_latents: int = 0,
    ) -> None:
        import hexo_py

        self._position = hexo_py.Position
        self.positions = [self._position() for _ in range(envs)]
        self.episodes = [Episode() for _ in range(envs)]
        self.ply_cap = ply_cap
        self.tau = tau
        self.lam = lam
        self.rng = rng
        self.pair_budget = pair_budget
        self.cell_budget = cell_budget
        if state_latents not in (0, 4):
            raise ValueError(
                f"state_latents must be one of {{0, 4}}, got {state_latents}"
            )
        self.state_latents = state_latents
        # The cap is one quarter of the cohort, clamped to 64–256 slots.
        self.chunk_cap = min(envs, max(64, envs // 4))

    def collect(
        self, evaluate: Evaluate, episodes: int, progress=None
    ) -> tuple[list[Episode], dict]:
        """Step the cohort until ``episodes`` games have ended; return them
        plus the section 8 diagnostics."""
        done: list[Episode] = []
        stats = {"kl": 0.0, "ent": 0.0, "n": 0}

        with (
            ThreadPoolExecutor(max_workers=1) as collate_pool,
            ThreadPoolExecutor(max_workers=1) as sample_pool,
        ):
            while len(done) < episodes:
                order = sorted(
                    range(len(self.positions)),
                    key=lambda i: -self.positions[i].stone_count,
                )
                chunks = _chunk_live(
                    self.positions, order, self.pair_budget, self.cell_budget,
                    self.chunk_cap, self.state_latents,
                )
                batches = [
                    collate_pool.submit(
                        lambda c: collate_positions(
                            [self.positions[i] for i in c],
                            state_latents=self.state_latents,
                        ),
                        chunk,
                    )
                    for chunk in chunks
                ]
                sampled = []
                for chunk, fut in zip(chunks, batches):
                    batch = fut.result()
                    policy_logits, q_score, q_values = evaluate(batch)
                    sampled.append(
                        sample_pool.submit(
                            self._sample, chunk, batch, policy_logits, q_score,
                            q_values, done, stats,
                        )
                    )
                # Advance every slot before the next step reads stone counts.
                for fut in sampled:
                    fut.result()
                if progress is not None:
                    progress(
                        len(done), episodes,
                        [p.stone_count for p in self.positions],
                    )

        metrics = {
            "acting_kl": stats["kl"] / max(stats["n"], 1),
            "acting_norm_entropy": stats["ent"] / max(stats["n"], 1),
        }
        return done, metrics

    def _sample(
        self, chunk, batch, policy_logits, q_score, q_values, done, stats
    ) -> None:
        """Improve, sample, and advance one chunk in sampling-lane order."""
        imp = improved_policy(
            policy_logits,
            q_score,
            q_values,
            batch.legal_offsets,
            self.tau,
            self.lam,
        )
        stats["kl"] += float(imp.kl.sum())
        stats["ent"] += float(imp.norm_entropy.sum())
        stats["n"] += len(chunk)

        # Renormalization in f64 keeps each stored segment within the policy
        # loss normalization tolerance. Sampling and storage use those values.
        offsets = batch.legal_offsets.numpy()
        flat = imp.probs.numpy().astype(np.float64)
        widths = np.diff(offsets)
        flat /= np.repeat(np.add.reduceat(flat, offsets[:-1]), widths)
        cdf = np.cumsum(flat)
        base = np.concatenate(([0.0], cdf[offsets[1:-1] - 1]))
        draws = self.rng.random(len(chunk))  # one uniform per slot, in order
        ranks = np.searchsorted(cdf, base + draws) - offsets[:-1]
        ranks = np.clip(ranks, 0, widths - 1)
        stored = np.split(flat.astype(np.float32), offsets[1:-1])
        v_hats = imp.v_hat.numpy()
        # Compute telemetry reductions from the flat improved-policy array.
        pi_top1 = np.maximum.reduceat(flat, offsets[:-1])
        pi_chosen = flat[offsets[:-1] + ranks]
        kls, norm_entropies = imp.kl.numpy(), imp.norm_entropy.numpy()

        for slot, i in enumerate(chunk):
            ep, pos = self.episodes[i], self.positions[i]
            rank = int(ranks[slot])
            ep.moves_remaining.append(pos.moves_remaining)
            ep.movers.append(pos.current_player)
            ep.ranks.append(rank)
            ep.improved.append(stored[slot])
            ep.v_hats.append(float(v_hats[slot]))
            ep.kls.append(float(kls[slot]))
            ep.norm_entropies.append(float(norm_entropies[slot]))
            ep.pi_top1.append(float(pi_top1[slot]))
            ep.pi_chosen.append(float(pi_chosen[slot]))

            move = pos.nth_legal(rank)
            pos.advance(*move)
            ep.moves.append(move)

            ended = pos.is_terminal or len(ep.moves) >= self.ply_cap
            if ended:
                if pos.is_terminal:
                    ep.winner = pos.winner
                done.append(ep)
                self.positions[i] = self._position()
                self.episodes[i] = Episode()


def collection_stats(episodes: list[Episode]) -> dict:
    """Return the collection metrics defined by ``KLENT_FOR_HEXO.md`` §8."""
    won = [e for e in episodes if e.winner is not None]

    def mean(values):
        values = list(values)
        return sum(values) / len(values) if values else float("nan")

    v_win, v_loss, abs_err = [], [], []
    for e in won:
        for mover, v in zip(e.movers, e.v_hats):
            z = 1.0 if mover == e.winner else -1.0
            (v_win if z > 0 else v_loss).append(v)
            abs_err.append(abs(v - z))

    return {
        "f": len(won) / len(episodes) if episodes else float("nan"),
        "won_length_mean": mean(len(e.moves) for e in won),
        "p0_win_rate": mean(float(e.winner == 0) for e in won),
        # K2 coverage: how often the winning placement was a turn's first stone.
        "first_stone_win_rate": mean(float(e.moves_remaining[-1] == 2) for e in won),
        "v_hat_winner_mean": mean(v_win),
        "v_hat_loser_mean": mean(v_loss),
        "v_hat_mae": mean(abs_err),
    }


def episode_samples(episode: Episode, lam_ret: float, gamma: float) -> list[Sample]:
    """The buffer entries of one episode: empty for a capped one
    (``KLENT_FOR_HEXO.md`` §4.2)."""
    if episode.winner is None:
        return []
    signs = signs_from_moves_remaining(episode.moves_remaining)
    returns = lambda_returns(signs, episode.v_hats, lam_ret, gamma)
    moves = tuple(episode.moves)
    return [
        Sample(
            moves=moves,
            t=j,
            rank=episode.ranks[j],
            improved=episode.improved[j],
            g=float(returns[j]),
        )
        for j in range(len(episode.ranks))
    ]
