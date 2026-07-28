"""Self-play collection: batched episodes, acting-time v̂, and the buffer rules.

Every ply the live games' positions are built into one batch, evaluated once,
and each game samples its next placement from its own segment of π′. What is
recorded per acted ply is exactly what design doc §4.5 keeps: the action's
legal rank, π′ itself, and the acting-time `v̂` (K6) that the λ-return will
consume. What is excluded is exactly what it excludes: terminal positions
never exist as samples, every ply of a capped episode (K4), and every seeded
prefix ply — those are replayed before collection starts, so they are never
recorded at all.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    """One self-play episode; per-ply records cover acted plies only."""

    moves: list  # the whole move list, seed prefix included
    seed_len: int  # plies replayed before π′ took over
    winner: int | None  # None exactly when the cap hit
    moves_remaining: list  # per acted ply, at acting time
    movers: list  # per acted ply, the engine's mover
    ranks: list  # per acted ply, the action's legal rank
    improved: list  # per acted ply, π′ as float32 over the legal set
    v_hats: list  # per acted ply, E_{π′}[Q] at acting time


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
    prefixes: list[list],
    ply_cap: int,
    tau: float,
    lam: float,
    rng: np.random.Generator,
    pair_budget: int = 8_000_000,
    cell_budget: int = 400_000,
) -> tuple[list[Episode], dict]:
    """Play one episode per prefix (empty prefix = unseeded), in lockstep.

    Returns the episodes plus the acting-time means of the §13 diagnostics:
    `D_KL(π′ ‖ π_θ)` and `H(π′)/log|A|`.
    """
    import hexo_py

    episodes = []
    positions = []
    for prefix in prefixes:
        pos = hexo_py.Position.replay(list(prefix))
        if pos.is_terminal:
            raise ValueError("a seed prefix must leave a live game")
        episodes.append(
            Episode(
                moves=list(prefix),
                seed_len=len(prefix),
                winner=None,
                moves_remaining=[],
                movers=[],
                ranks=[],
                improved=[],
                v_hats=[],
            )
        )
        positions.append(pos)

    live = list(range(len(episodes)))
    kl_sum, ent_sum, decisions = 0.0, 0.0, 0
    while live:
        still_live = []
        for chunk in _chunk_live(positions, live, pair_budget, cell_budget):
            batch = collate_positions([positions[i] for i in chunk])
            policy_logits, q_values = evaluate(batch)
            imp = improved_policy(policy_logits, q_values, batch.legal_offsets, tau, lam)

            offsets = batch.legal_offsets.tolist()
            kl_sum += float(imp.kl.sum())
            ent_sum += float(imp.norm_entropy.sum())
            decisions += len(chunk)

            for slot, i in enumerate(chunk):
                ep, pos = episodes[i], positions[i]
                probs = imp.probs[offsets[slot] : offsets[slot + 1]].numpy().astype(np.float64)
                rank = int(rng.choice(len(probs), p=probs / probs.sum()))

                ep.moves_remaining.append(pos.moves_remaining)
                ep.movers.append(pos.current_player)
                ep.ranks.append(rank)
                ep.improved.append(probs.astype(np.float32))
                ep.v_hats.append(float(imp.v_hat[slot]))

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
    """The §13 collection-side metrics: the terminating fractions that decide
    whether training data exists, seat and win-half coverage, seed-curriculum
    state, and the v̂-vs-outcome calibration that makes the §9 overestimation
    bias visible. Calibration reads won episodes only — a capped episode has
    no grounded outcome to calibrate against."""
    won = [e for e in episodes if e.winner is not None]
    seeded = [e for e in episodes if e.seed_len > 0]
    unseeded = [e for e in episodes if e.seed_len == 0]

    def fraction(eps):
        return sum(e.winner is not None for e in eps) / len(eps) if eps else float("nan")

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
        "f_seeded": fraction(seeded),
        "f_unseeded": fraction(unseeded),
        "won_length_mean": mean(len(e.moves) for e in won),
        "seed_len_mean": mean(e.seed_len for e in seeded),
        "seed_len_max": max((e.seed_len for e in seeded), default=0),
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
            t=episode.seed_len + j,
            rank=episode.ranks[j],
            improved=episode.improved[j],
            g=float(returns[j]),
        )
        for j in range(len(episode.ranks))
    ]
