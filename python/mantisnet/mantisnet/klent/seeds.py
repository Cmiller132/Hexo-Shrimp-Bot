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


def line_builder_game(rng: np.random.Generator, noise: float = 0.1, cap: int = 512):
    """A whole line-builder game from the empty board.

    Returns ``(moves, winner)`` with ``winner`` ``None`` if the cap hit —
    rare at low noise, but a capped seed game must not be used as if won.
    """
    import hexo_py

    pos = hexo_py.Position()
    moves: list[tuple[int, int]] = []
    while not pos.is_terminal and len(moves) < cap:
        move = line_builder_choose(pos, rng, noise)
        pos.advance(*move)
        moves.append(move)
    return moves, pos.winner


def seed_prefix(rng: np.random.Generator, cut_range: tuple[int, int], noise: float = 0.1):
    """A prefix of a *won* line-builder game, cut ``cut ∈ [lo, hi]`` plies
    before its end. ``cut >= 1`` guarantees the prefix is non-terminal."""
    lo, hi = cut_range
    if lo < 1 or hi < lo:
        raise ValueError(f"cut range must satisfy 1 <= lo <= hi, got {cut_range}")
    while True:
        moves, winner = line_builder_game(rng, noise)
        if winner is None:
            continue
        cut = int(rng.integers(lo, hi + 1))
        return moves[: max(len(moves) - cut, 0)]
