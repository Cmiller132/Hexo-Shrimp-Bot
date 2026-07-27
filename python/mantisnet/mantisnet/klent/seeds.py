"""The line-building seeder (design doc §5.2).

Random play essentially never completes six in a row, so unseeded self-play
from an untrained network starts with an empty buffer. Seeded episodes replay
a prefix of a game *that somebody won* and hand control to π′ near the end,
giving backward induction somewhere to start. The seed source here is the
checkpoint-free one the design doc names: a scripted line builder that
extends its own longest clean line and therefore terminates in tens of plies.

The same chooser doubles as a fixed evaluation opponent — one implementation,
two callers.
"""

from __future__ import annotations

import numpy as np


def line_builder_choose(pos, rng: np.random.Generator, noise: float = 0.1):
    """One placement: the legal cell with the longest own run through it
    among windows free of opposing stones, with ``noise`` chance of a
    uniformly random legal move. Ties break randomly so two line builders
    produce different games."""
    legal = pos.legal_moves()
    if rng.random() < noise:
        return legal[rng.integers(len(legal))]
    me = pos.current_player
    scores = np.empty(len(legal), dtype=np.int64)
    for i, (q, r) in enumerate(legal):
        best = 0
        for _axis, _sq, _sr, m0, m1 in pos.windows_through(q, r):
            own, theirs = (m0, m1) if me == 0 else (m1, m0)
            if theirs == 0:
                count = bin(own).count("1")
                if count > best:
                    best = count
        scores[i] = best
    top = np.flatnonzero(scores == scores.max())
    return legal[top[rng.integers(len(top))]]


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
