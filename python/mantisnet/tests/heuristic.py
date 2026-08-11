"""A scripted line-extending player for tests.

Scores each legal cell by the stone count of its best own live window, with
a six-completing cell assigned the maximum score. It implements the evaluator
``(batch) -> (logits, q_score, q_value)`` and chooser
``(positions, rng) -> moves`` test
seams and is not part of the training package.
"""

from __future__ import annotations

import numpy as np
import torch

from mantisnet.builder import TERN_PATTERNS, _TERN_DIGITS, _TERN_RANK, collate_positions

_PATTERN_SCORE = np.zeros(TERN_PATTERNS, dtype=np.int64)
for _pattern in range(1, 729):
    _digits = _TERN_DIGITS[_pattern]
    _PATTERN_SCORE[_TERN_RANK[_pattern]] = (
        int((_digits == 1).sum()) if not (_digits == 2).any() else 0
    )
_STONES = torch.from_numpy(_PATTERN_SCORE)


def heuristic_scores(batch) -> torch.Tensor:
    """Per legal cell: the stone count of its best own live window, with a
    six-completing cell assigned the maximum score."""
    wscore = _STONES[batch.window_feat]
    per_cell = torch.zeros(batch.n_cells)
    per_cell.index_reduce_(
        0, batch.dec_cell, wscore.index_select(0, batch.dec_window).float(), "amax"
    )
    return torch.where(per_cell >= 5, 8.0, per_cell)


def heuristic_evaluate(batch):
    """The scores as an evaluator: cell scores as policy logits, and the same
    scores scaled into the critic's [0, 1] output range as Q — acting-time
    v-hat must satisfy `lambda_returns`' [-1, 1] input contract."""
    scores = heuristic_scores(batch)
    q = scores / 8.0
    return scores, q, q


def heuristic_choose(positions, rng: np.random.Generator, noise: float = 0.0):
    """One placement per position: each game's best-scoring legal cell, with
    ``noise`` chance of a uniformly random legal move. Ties break randomly
    so two heuristic players produce different games."""
    batch = collate_positions(positions)
    scores = heuristic_scores(batch).numpy()
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


def heuristic_game(rng: np.random.Generator, noise: float = 0.1, cap: int = 512):
    """One whole heuristic game from the empty board.

    Returns ``(moves, winner)`` with ``winner`` ``None`` where the cap hit.
    """
    import hexo_py

    pos = hexo_py.Position()
    moves: list[tuple[int, int]] = []
    while not pos.is_terminal and len(moves) < cap:
        move = heuristic_choose([pos], rng, noise)[0]
        pos.advance(*move)
        moves.append(move)
    return moves, pos.winner if pos.is_terminal else None
