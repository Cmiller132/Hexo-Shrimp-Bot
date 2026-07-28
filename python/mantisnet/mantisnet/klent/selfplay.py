"""Self-play collection: batched episodes, acting-time v̂, and the buffer rules.

The paper's self-play phase, at Hexo's placement granularity: every game
starts from the empty board, and every ply the live games' positions are
built into one batch, evaluated once, and each game samples its next
placement from its own segment of π′. What is recorded per ply is exactly
what design doc §4.5 keeps: the action's legal rank, π′ itself, and the
acting-time `v̂` (K6) that the λ-return will consume. What is excluded is
exactly what it excludes: terminal positions never exist as samples, and
every ply of a capped episode (K4) — the reference implementation's
NaN-masked unterminated tail, as a whole-episode drop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from ..builder import collate_positions
from .improve import improved_policy
from .returns import lambda_returns, signs_from_moves_remaining

# The evaluator seam: flat (policy_logits, q_values) on CPU for one Batch.
# Training wraps the network; tests may wrap anything with the same shape.
Evaluate = Callable[[object], tuple[torch.Tensor, torch.Tensor]]


@dataclass
class Episode:
    """One episode from the empty board; per-ply records for every ply."""

    moves: list = field(default_factory=list)
    winner: int | None = None  # None exactly when the cap hit
    moves_remaining: list = field(default_factory=list)  # per ply, at acting time
    movers: list = field(default_factory=list)  # per ply, the engine's mover
    ranks: list = field(default_factory=list)  # per ply, the action's legal rank
    improved: list = field(default_factory=list)  # per ply, π′ over the legal set
    v_hats: list = field(default_factory=list)  # per ply, E_{π′}[Q] at acting time


@dataclass
class Sample:
    """One buffer entry: `(S_t, A_t, π′, G_t)` with the state as a move prefix."""

    moves: tuple  # episode move list; S_t = replay(moves[:t])
    t: int
    rank: int  # A_t as its index in engine legal order
    improved: np.ndarray  # π′(·|S_t)
    g: float  # λ-return, in the mover-at-t's frame


def _chunk_live(positions, live: list[int], pair_budget: int, cell_budget: int):
    """Split the live cohort into consecutive runs under the memory budgets
    (attention pairs are ``count × padded_T²``, decoder rows are legal
    cells). Consecutive, never reordered: the sampling RNG must consume its
    draws in the same game order no matter how the cohort is chunked."""
    chunks: list[list[int]] = []
    chunk: list[int] = []
    max_t, cells = 0, 0
    for i in live:
        t_pad = positions[i].stone_count + 1  # + the global token row
        c = positions[i].legal_count
        t = max(max_t, t_pad)
        if chunk and ((len(chunk) + 1) * t * t > pair_budget or cells + c > cell_budget):
            chunks.append(chunk)
            chunk, max_t, cells = [], 0, 0
            t = t_pad
        chunk.append(i)
        max_t, cells = t, cells + c
    if chunk:
        chunks.append(chunk)
    return chunks


def play_episodes(
    evaluate: Evaluate,
    games: int,
    ply_cap: int,
    tau: float,
    lam: float,
    rng: np.random.Generator,
    pair_budget: int = 8_000_000,
    cell_budget: int = 400_000,
) -> tuple[list[Episode], dict]:
    """Play ``games`` episodes from the empty board, in lockstep.

    Returns the episodes plus the acting-time means of the §13 diagnostics:
    `D_KL(π′ ‖ π_θ)` and `H(π′)/log|A|`.
    """
    import hexo_py

    episodes = [Episode() for _ in range(games)]
    positions = [hexo_py.Position() for _ in range(games)]

    live = list(range(games))
    kl_sum, ent_sum, decisions = 0.0, 0.0, 0
    while live:
        still_live = []
        for chunk in _chunk_live(positions, live, pair_budget, cell_budget):
            batch = collate_positions([positions[i] for i in chunk])
            policy_logits, q_values = evaluate(batch)
            imp = improved_policy(policy_logits, q_values, batch.legal_offsets, tau, lam)

            kl_sum += float(imp.kl.sum())
            ent_sum += float(imp.norm_entropy.sum())
            decisions += len(chunk)

            # All per-cell math runs flat over the chunk. Renormalized in
            # f64 before storage: the fp32 softmax's accumulated denominator
            # leaves |sum−1| ≈ N·1e-8, which at 10^4-cell positions crosses
            # policy_loss's corruption gate. The sampler and the stored
            # target see the same numbers. Sampling is one uniform per game
            # against the segment's slice of one flat CDF.
            offsets = batch.legal_offsets.numpy()
            flat = imp.probs.numpy().astype(np.float64)
            widths = np.diff(offsets)
            flat /= np.repeat(np.add.reduceat(flat, offsets[:-1]), widths)
            cdf = np.cumsum(flat)
            base = np.concatenate(([0.0], cdf[offsets[1:-1] - 1]))
            draws = rng.random(len(chunk))  # one uniform per game, in game order
            ranks = np.searchsorted(cdf, base + draws) - offsets[:-1]
            ranks = np.clip(ranks, 0, widths - 1)
            stored = np.split(flat.astype(np.float32), offsets[1:-1])
            v_hats = imp.v_hat.numpy()

            for slot, i in enumerate(chunk):
                ep, pos = episodes[i], positions[i]
                rank = int(ranks[slot])
                ep.moves_remaining.append(pos.moves_remaining)
                ep.movers.append(pos.current_player)
                ep.ranks.append(rank)
                ep.improved.append(stored[slot])
                ep.v_hats.append(float(v_hats[slot]))

                move = pos.nth_legal(rank)
                pos.advance(*move)
                ep.moves.append(move)

                if pos.is_terminal:
                    ep.winner = pos.winner
                elif len(ep.moves) < ply_cap:
                    still_live.append(i)
        live = still_live

    metrics = {
        "acting_kl": kl_sum / max(decisions, 1),
        "acting_norm_entropy": ent_sum / max(decisions, 1),
    }
    return episodes, metrics


def collection_stats(episodes: list[Episode]) -> dict:
    """The §13 collection-side metrics: the terminating fraction that decides
    whether training data exists, seat and win-half coverage, and the
    v̂-vs-outcome calibration that makes the §9 overestimation bias visible."""
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


def episode_samples(episode: Episode, lam_ret: float) -> list[Sample]:
    """The buffer entries of one episode: empty for a capped one (§5.1)."""
    if episode.winner is None:
        return []
    signs = signs_from_moves_remaining(episode.moves_remaining)
    returns = lambda_returns(signs, episode.v_hats, lam_ret)
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
