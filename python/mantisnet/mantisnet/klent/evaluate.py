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
from .seeds import line_builder_choose_batch

# The anchor's randomness is part of its identity: an anchor whose noise
# drifts between runs is not an anchor. Pinned here, recorded in config.json.
ANCHOR_NOISE = 0.1


def argmax_choose(model, device: str = "cpu"):
    """A chooser playing the policy head's argmax over each legal set.

    Choosers are batched — ``choose(positions, rng) -> moves`` — so a match
    pays one collate and one forward per lockstep step, not per position."""

    def choose(positions, _rng):
        batch = collate_positions(positions).to(device)
        with torch.no_grad():
            _s, w, g = model.trunk(batch)
            logits = model.policy_head(w, g, batch).cpu()
        offsets = batch.legal_offsets.tolist()
        return [
            pos.nth_legal(int(logits[offsets[k] : offsets[k + 1]].argmax()))
            for k, pos in enumerate(positions)
        ]

    return choose


def play_match(
    choose_a,
    choose_b,
    games: int,
    ply_cap: int,
    rng: np.random.Generator,
) -> dict:
    """``games`` lockstep games with A taking P0 in the even-indexed ones.

    Each outer step advances every live game one ply: the games whose mover
    is A form one batched chooser call, B's the other. Returns A's score
    (win 1, cap ½), the capped count, and the game count.
    """
    import hexo_py

    positions = [hexo_py.Position() for _ in range(games)]
    score_a, capped = 0.0, 0
    plies = [0] * games
    live = list(range(games))
    while live:
        groups: dict = {0: [], 1: []}  # chooser A's games, chooser B's games
        for k in live:
            a_is_mover = (positions[k].current_player == 0) == (k % 2 == 0)
            groups[0 if a_is_mover else 1].append(k)
        for chooser, group in ((choose_a, groups[0]), (choose_b, groups[1])):
            if not group:
                continue
            for k, move in zip(group, chooser([positions[k] for k in group], rng)):
                positions[k].advance(*move)
                plies[k] += 1
        still = []
        for k in live:
            pos = positions[k]
            if pos.is_terminal:
                a_seat = 0 if k % 2 == 0 else 1
                score_a += 1.0 if pos.winner == a_seat else 0.0
            elif plies[k] >= ply_cap:
                capped += 1
                score_a += 0.5
            else:
                still.append(k)
        live = still
    return {"score_a": score_a, "games": games, "capped": capped}


def anchor_match(
    model, device: str, games: int, ply_cap: int, rng: np.random.Generator, opponent=None
) -> dict:
    """Argmax π_θ against the anchor. The default opponent is anchor zero —
    the line builder at pinned noise; a frozen checkpoint's chooser replaces
    it once one is worth freezing (design doc §11). Returns metrics-row
    fields; capped games are visible in ``eval_capped``, never folded away."""
    model.eval()

    if opponent is None:

        def opponent(positions, anchor_rng):
            return line_builder_choose_batch(positions, anchor_rng, ANCHOR_NOISE)

    result = play_match(argmax_choose(model, device), opponent, games, ply_cap, rng)
    return {
        "eval_score": result["score_a"] / result["games"],
        "eval_capped": result["capped"],
        "eval_games": result["games"],
    }
