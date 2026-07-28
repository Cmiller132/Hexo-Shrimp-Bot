"""The line-building seeder (design doc §5.2).

Random play essentially never completes six in a row, so unseeded self-play
from an untrained network starts with an empty buffer. Seeded episodes replay
a prefix of a game *that somebody won* and hand control to π′ near the end,
giving backward induction somewhere to start. The seed source here is the
checkpoint-free one the design doc names: a scripted line builder that
extends its own longest clean line and therefore terminates in tens of plies.

The same chooser doubles as a fixed evaluation opponent, and its scoring —
each legal cell's best own live window — is the scripted evaluator the
pipeline tests drive the collection loop with. One implementation, three
callers, all reading the builder's own batch arrays.
"""

from __future__ import annotations

import numpy as np
import torch

from ..builder import NUM_PATTERNS, PATTERN_STONES, collate_positions

_STONES = torch.from_numpy(PATTERN_STONES)


def line_scores(batch) -> torch.Tensor:
    """Per legal cell: the stone count of its best own live window, with a
    six-completing cell scored decisively above every extension."""
    own = batch.window_feat < NUM_PATTERNS
    wscore = torch.where(own, _STONES[batch.window_feat % NUM_PATTERNS], 0)
    per_cell = torch.zeros(batch.n_cells)
    per_cell.index_reduce_(
        0, batch.dec_cell, wscore.index_select(0, batch.dec_window).float(), "amax"
    )
    return torch.where(per_cell >= 5, 8.0, per_cell)


def line_evaluate(batch):
    """The line builder's scoring as an evaluator: cell scores as both
    policy logits and Q. The KLENT warm start (`--warm-iterations`) acts
    through this for a run's first iterations — measured necessity: an
    honestly-initialized π′ is near-uniform and finishes almost no seeded
    games, so the network's first targets must come from play that ends."""
    scores = line_scores(batch)
    return scores, scores.clone()


def line_builder_choose_batch(positions, rng: np.random.Generator, noise: float = 0.1):
    """One placement per position: each game's best-scoring legal cell, with
    ``noise`` chance of a uniformly random legal move. Ties break randomly
    so two line builders produce different games. One collate and one
    scoring pass serve the whole batch."""
    batch = collate_positions(positions)
    scores = line_scores(batch).numpy()
    offsets = batch.legal_offsets.tolist()
    moves = []
    for k, pos in enumerate(positions):
        segment = scores[offsets[k] : offsets[k + 1]]
        if rng.random() < noise:
            moves.append(pos.nth_legal(int(rng.integers(len(segment)))))
        else:
            top = np.flatnonzero(segment == segment.max())
            moves.append(pos.nth_legal(int(top[rng.integers(len(top))])))
    return moves


def line_builder_choose(pos, rng: np.random.Generator, noise: float = 0.1):
    """The batch chooser, for one position."""
    return line_builder_choose_batch([pos], rng, noise)[0]


def line_builder_games(rng: np.random.Generator, n: int, noise: float = 0.1, cap: int = 512):
    """``n`` whole line-builder games from the empty board, in lockstep —
    one collate and one scoring pass per ply serve the entire cohort.
    Playing them one at a time was measured at ~0.9 s per training
    iteration of batch-of-one collates, the loop's single largest cost.

    Returns ``(moves, winners)`` with a winner ``None`` where the cap hit —
    rare at low noise, but a capped seed game must not be used as if won.
    """
    import hexo_py

    positions = [hexo_py.Position() for _ in range(n)]
    moves: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    winners: list[int | None] = [None] * n
    live = list(range(n))
    while live:
        picked = line_builder_choose_batch([positions[i] for i in live], rng, noise)
        still = []
        for i, move in zip(live, picked):
            positions[i].advance(*move)
            moves[i].append(move)
            if positions[i].is_terminal:
                winners[i] = positions[i].winner
            elif len(moves[i]) < cap:
                still.append(i)
        live = still
    return moves, winners


def line_builder_game(rng: np.random.Generator, noise: float = 0.1, cap: int = 512):
    """One whole line-builder game — the cohort player, cohort of one."""
    moves, winners = line_builder_games(rng, 1, noise, cap)
    return moves[0], winners[0]


def seed_prefixes(
    rng: np.random.Generator, count: int, cut_range: tuple[int, int], noise: float = 0.1
):
    """``count`` prefixes of *won* line-builder games, each cut
    ``cut ∈ [lo, hi]`` plies before its end (``cut >= 1`` guarantees a
    non-terminal prefix). Games play in lockstep; the rare capped ones are
    replayed until enough have finished."""
    lo, hi = cut_range
    if lo < 1 or hi < lo:
        raise ValueError(f"cut range must satisfy 1 <= lo <= hi, got {cut_range}")
    won: list[list[tuple[int, int]]] = []
    while len(won) < count:
        games, winners = line_builder_games(rng, count - len(won), noise)
        won += [m for m, w in zip(games, winners) if w is not None]
    cuts = rng.integers(lo, hi + 1, size=count)
    return [m[: max(len(m) - int(c), 0)] for m, c in zip(won, cuts)]


def seed_prefix(rng: np.random.Generator, cut_range: tuple[int, int], noise: float = 0.1):
    """One seed prefix — the cohort generator, cohort of one."""
    return seed_prefixes(rng, 1, cut_range, noise)[0]
