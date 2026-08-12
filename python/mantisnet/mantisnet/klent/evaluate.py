"""Policy-argmax chooser and two-chooser match loop.

``argmax_choose``: zero-search policy argmax, no randomness.
``play_match``: runs two batched choosers over a seat-paired schedule.
"""

from __future__ import annotations

import time

import torch

from ..builder import collate_positions


def argmax_choose(model, device: str = "cpu"):
    """Batched chooser returning the policy head's argmax over each legal set."""

    def choose(positions, _rng):
        batch = collate_positions(
            positions, state_latents=model.cfg.state_latents
        ).to(device)
        with torch.no_grad():
            _s, w, g = model.trunk(batch)
            logits = model.policy_head(w, g, batch).cpu()
        offsets = batch.legal_offsets.tolist()
        return [
            pos.nth_legal(int(logits[offsets[k] : offsets[k + 1]].argmax()))
            for k, pos in enumerate(positions)
        ]

    return choose


def play_match(choose_a, choose_b, schedule, ply_cap: int, rng):
    """One lockstep game per ``(opening, seat)`` entry of ``schedule``.

    ``schedule`` is ``opponents.shared_openings``' output: A takes ``seat`` and
    B the other, and the two games of a seat pair replay one shared prefix. Each
    outer step advances every live game one ply — the games whose mover is A form
    one batched chooser call, B's games the other — and every chooser call
    receives ``rng``.

    ``ply_cap`` counts the opening's placements, as ``opponent_match``'s cap
    does, so the same cap means the same total game length in both loops.

    Returns A's summary and one row per game, in schedule order.
    """
    import hexo_py

    if not schedule:
        raise ValueError("play_match needs at least one game")
    games = len(schedule)
    positions = [
        hexo_py.Position.replay([tuple(move) for move in opening])
        for opening, _seat in schedule
    ]
    seats = [seat for _opening, seat in schedule]
    moves = [[tuple(move) for move in opening] for opening, _seat in schedule]
    capped = [False] * games
    started = time.monotonic()

    def settle(indices):
        still = []
        for k in indices:
            if positions[k].is_terminal:
                continue
            if len(moves[k]) >= ply_cap:
                capped[k] = True
                continue
            still.append(k)
        return still

    live = settle(list(range(games)))
    while live:
        groups: dict = {0: [], 1: []}  # chooser A's games, chooser B's games
        for k in live:
            groups[0 if positions[k].current_player == seats[k] else 1].append(k)
        for chooser, group in ((choose_a, groups[0]), (choose_b, groups[1])):
            if not group:
                continue
            chosen = chooser([positions[k] for k in group], rng)
            for k, move in zip(group, chosen, strict=True):
                move = (int(move[0]), int(move[1]))
                positions[k].advance(*move)
                moves[k].append(move)
        live = settle(live)

    score_a, per_seat, plies = 0.0, [0.0, 0.0], 0
    rows = []
    for k in range(games):
        game_score = (
            0.5 if capped[k] else float(positions[k].winner == seats[k])
        )
        score_a += game_score
        per_seat[seats[k]] += game_score
        plies += len(moves[k])
        rows.append(
            {
                "seat": seats[k],
                "winner": None if capped[k] else positions[k].winner,
                "capped": capped[k],
                "score_a": game_score,
                "opening": [tuple(move) for move in schedule[k][0]],
                "moves": moves[k],
            }
        )
    summary = {
        "score_a": score_a,
        "games": games,
        "capped": sum(capped),
        "score_a_as_p0": per_seat[0],
        "score_a_as_p1": per_seat[1],
        "avg_plies": plies / games,
        "seconds": time.monotonic() - started,
    }
    return summary, rows
