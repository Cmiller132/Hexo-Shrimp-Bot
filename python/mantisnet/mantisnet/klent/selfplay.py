"""Self-play collection: batched episodes, acting-time v̂, and the buffer rules.

Every ply the live games' positions are built into one batch, evaluated once,
and each game samples its next placement from its own segment of π′. What is
recorded per acted ply is exactly what design doc §4.5 keeps: the action's
legal rank, π′ itself, and the acting-time `v̂` (K6) that the λ-return will
consume. What is excluded is exactly what it excludes: terminal positions
never exist as samples, every ply of a capped episode (K4), and every seeded
prefix ply — those are replayed before collection starts, so they are never
recorded at all.

Grounded games (KLENT_PROPOSALS' opponent grounding) put an external
whole-turn opponent in one seat: its plies enter the move list but are never
recorded, so the buffer holds only the model's decisions, judged by a real
outcome an opponent enforced. Self-play conditioning was measured to look
perfect while strength against any real opponent died — grounded games are
the corpus-side answer.
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
    """One episode; per-ply records cover the model's acted plies only."""

    moves: list  # the whole move list: seed prefix and opponent plies included
    seed_len: int  # plies replayed before π′ took over
    winner: int | None  # None exactly when the cap hit
    moves_remaining: list  # per acted ply, at acting time
    movers: list  # per acted ply, the engine's mover
    ranks: list  # per acted ply, the action's legal rank
    improved: list  # per acted ply, π′ as float32 over the legal set
    v_hats: list  # per acted ply, E_{π′}[Q] at acting time
    ts: list  # per acted ply, its absolute index into ``moves``
    opponent_seat: int | None = None  # a grounded game's external seat


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
    opponent=None,
    opponent_seats: list | None = None,
) -> tuple[list[Episode], dict]:
    """Play one episode per prefix (empty prefix = unseeded), in lockstep.

    ``opponent_seats`` marks grounded games: entry ``i`` is the seat an
    external ``opponent(position, moves) -> [(q, r), ...]`` plays in game
    ``i`` (``None`` = self-play). Opponent turns are whole turns — applied
    first each lockstep step, every placement checked legal, never recorded —
    so every position the evaluator sees below has the model to move.

    Returns the episodes plus the acting-time means of the §13 diagnostics:
    `D_KL(π′ ‖ π_θ)` and `H(π′)/log|A|`.
    """
    import hexo_py

    if opponent_seats is None:
        opponent_seats = [None] * len(prefixes)
    if len(opponent_seats) != len(prefixes):
        raise ValueError("opponent_seats must align one-to-one with prefixes")
    if opponent is None and any(s is not None for s in opponent_seats):
        raise ValueError("opponent_seats given without an opponent")

    episodes = []
    positions = []
    for prefix, seat in zip(prefixes, opponent_seats):
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
                ts=[],
                opponent_seat=seat,
            )
        )
        positions.append(pos)

    live = list(range(len(episodes)))
    kl_sum, ent_sum, decisions = 0.0, 0.0, 0
    while live:
        model_to_move = []
        for i in live:
            ep, pos = episodes[i], positions[i]
            seat = ep.opponent_seat
            if seat is not None and pos.current_player == seat:
                turn = opponent(pos, ep.moves)
                if not turn:
                    raise RuntimeError("opponent returned no moves for a live position")
                for move in turn:
                    if pos.is_terminal or pos.current_player != seat:
                        break
                    move = (int(move[0]), int(move[1]))
                    if move not in set(pos.legal_moves()):
                        raise RuntimeError(
                            f"opponent move {move} is illegal at ply {len(ep.moves)}"
                        )
                    pos.advance(*move)
                    ep.moves.append(move)
                if pos.is_terminal:
                    ep.winner = pos.winner
                    continue
                if len(ep.moves) >= ply_cap:
                    continue
            model_to_move.append(i)
        live = model_to_move
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
                ep.ts.append(len(ep.moves))

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
    bias visible. The f/seat/seed stats read self-play episodes only — an
    external opponent terminates its games regardless of what the policy
    knows, so mixing them in would flatter exactly the signals the anneal
    walks on. Grounded games report their own pair: ``f_grounded`` and the
    model's ``grounded_score`` (win 1, cap ½ — the free per-iteration
    strength reading). Calibration pools every episode with an outcome: a
    grounded loss is precisely the outcome self-play v̂ never sees."""
    selfplay = [e for e in episodes if e.opponent_seat is None]
    grounded = [e for e in episodes if e.opponent_seat is not None]
    won = [e for e in selfplay if e.winner is not None]
    seeded = [e for e in selfplay if e.seed_len > 0]
    unseeded = [e for e in selfplay if e.seed_len == 0]

    def fraction(eps):
        return sum(e.winner is not None for e in eps) / len(eps) if eps else float("nan")

    def mean(values):
        values = list(values)
        return sum(values) / len(values) if values else float("nan")

    v_win, v_loss, abs_err = [], [], []
    for e in episodes:
        if e.winner is None:
            continue
        for mover, v in zip(e.movers, e.v_hats):
            z = 1.0 if mover == e.winner else -1.0
            (v_win if z > 0 else v_loss).append(v)
            abs_err.append(abs(v - z))

    def grounded_outcome(e):
        if e.winner is None:
            return 0.5
        return float(e.winner != e.opponent_seat)

    return {
        "f_seeded": fraction(seeded),
        "f_unseeded": fraction(unseeded),
        "f_grounded": fraction(grounded),
        "grounded_score": mean(grounded_outcome(e) for e in grounded),
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
    """The buffer entries of one episode: empty for a capped self-play one
    (§5.1).

    Grounded episodes take pure Monte-Carlo outcomes whatever ``lam_ret``
    says: the λ-return's bootstrap chain runs over consecutive recorded
    plies, and an opponent's unrecorded turns break it. And a *capped*
    grounded episode is a draw (g = 0) rather than dropped — against a real
    opponent, surviving to the cap is an outcome, and it is the only
    gradient toward defense while wins are out of reach."""
    if episode.opponent_seat is None:
        if episode.winner is None:
            return []
        signs = signs_from_moves_remaining(episode.moves_remaining)
        returns = lambda_returns(signs, episode.v_hats, lam_ret)
    elif episode.winner is None:
        returns = np.zeros(len(episode.ranks))
    else:
        returns = np.where(np.asarray(episode.movers) == episode.winner, 1.0, -1.0)
    moves = tuple(episode.moves)
    return [
        Sample(
            moves=moves,
            t=episode.ts[j],
            rank=episode.ranks[j],
            improved=episode.improved[j],
            g=float(returns[j]),
        )
        for j in range(len(episode.ranks))
    ]
