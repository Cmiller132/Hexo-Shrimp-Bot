"""Evaluation (design doc §11): argmax π_θ, no search, seat balanced.

π′ is a training-time construct; the deployed artefact is the policy head's
argmax. Seats alternate because the game is asymmetric even though the
encoding is not, and a capped game scores a half-win for each side — the
paper's draw convention, visible in the count rather than folded away.
"""

from __future__ import annotations

import numpy as np
import torch

from ..builder import collate_positions


def argmax_choose(model, device: str = "cpu"):
    """A chooser playing the policy head's argmax over the legal set."""

    def choose(pos, _rng):
        batch = collate_positions([pos]).to(device)
        with torch.no_grad():
            _s, w, g = model.trunk(batch)
            logits = model.policy_head(w, g, batch)
        return pos.nth_legal(int(logits.argmax()))

    return choose


def play_match(
    choose_a,
    choose_b,
    games: int,
    ply_cap: int,
    rng: np.random.Generator,
) -> dict:
    """``games`` games with A taking P0 in the even-indexed ones.

    Returns A's score (win 1, cap ½), the capped count, and the game count.
    """
    import hexo_py

    score_a, capped = 0.0, 0
    for k in range(games):
        seats = (choose_a, choose_b) if k % 2 == 0 else (choose_b, choose_a)
        pos = hexo_py.Position()
        plies = 0
        while not pos.is_terminal and plies < ply_cap:
            pos.advance(*seats[pos.current_player](pos, rng))
            plies += 1
        if pos.is_terminal:
            a_seat = 0 if k % 2 == 0 else 1
            score_a += 1.0 if pos.winner == a_seat else 0.0
        else:
            capped += 1
            score_a += 0.5
    return {"score_a": score_a, "games": games, "capped": capped}
