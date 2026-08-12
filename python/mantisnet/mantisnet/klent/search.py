"""Batched Gumbel sequential-halving search for evaluation.

Independent-line search with sequential halving.  ``gumbel_choose`` follows
the ``choose(positions, rng) -> moves`` contract.  Per-root child seeds
make batched and singleton calls RNG-equivalent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..builder import collate_positions
from .improve import improved_policy

# Gumbel MuZero's root value transform. Keep these together: changing either
# changes the relative scale of root prior/noise and searched line value, as
# ``gumbel_choose``'s ``temperature`` does from the noise side.
C_VISIT = 50
C_SCALE = 1.0


@dataclass
class _Line:
    root_rank: int
    root_logit: float
    gumbel: float
    position: object
    value: float
    policy_rank: int | None = None
    evaluated: bool = False
    terminal: bool = False
    visits: int = 0


@dataclass
class _Search:
    root_player: int
    lines: list[_Line]
    survivors: list[int]
    schedule: list[tuple[int, int]]


def _halving_schedule(sims: int, candidates: int) -> list[tuple[int, int]]:
    """Return ``(survivors, deepenings_each)`` for successive rounds.

    Deepenings double between rounds while the budget permits. The last
    possible halving round receives any remaining equal share. Integer
    remainders smaller than the survivor count stay unused rather than making
    candidates within a round receive unequal resources.
    """
    if sims < 0:
        raise ValueError(f"sims must be >= 0, got {sims}")
    if candidates < 1:
        raise ValueError(f"candidates must be >= 1, got {candidates}")
    rounds = max(1, math.ceil(math.log2(candidates)))
    remaining = sims
    survivors = candidates
    out = []
    for round_idx in range(rounds):
        if remaining < survivors:
            break
        share = remaining // survivors
        deepenings = (
            share
            if round_idx == rounds - 1
            else min(1 << round_idx, share)
        )
        if deepenings < 1:
            break
        out.append((survivors, deepenings))
        remaining -= survivors * deepenings
        survivors = (survivors + 1) // 2
    assert sum(n * depth for n, depth in out) <= sims
    return out


def _terminal_value(position, root_player: int) -> float:
    return 1.0 if position.winner == root_player else -1.0


def gumbel_choose(
    evaluate,
    tau: float,
    lam: float,
    sims: int,
    rng=None,
    temperature: float = 1.0,
):
    """Build a batched Gumbel root-sampling and sequential-halving chooser.

    ``evaluate(batch)`` returns flat CPU ``(policy_logits, q_score, q_value)``.
    ``sims == 0`` is policy-logit argmax.  ``temperature`` scales the root
    Gumbel vector: ``T * Gumbel(0, 1)`` draws the root order from
    ``softmax(logits / T)``; ``T = 0`` is deterministic (top-m by logit).
    """
    if sims < 0:
        raise ValueError(f"sims must be >= 0, got {sims}")
    if not math.isfinite(temperature) or temperature < 0.0:
        raise ValueError(f"temperature must be finite and >= 0, got {temperature}")
    if sims == 0 and temperature != 1.0:
        # Zero-budget argmax draws nothing, so there is no Gumbel to scale.
        # Silently ignoring the request would hand back a deterministic chooser
        # to a caller who asked for a stochastic one.
        raise ValueError(
            f"temperature={temperature} needs sims > 0; sims == 0 is argmax and "
            "draws no Gumbel to scale"
        )

    def choose(positions, call_rng):
        if not positions:
            return []
        parent = call_rng if call_rng is not None else rng
        if parent is None:
            raise ValueError("gumbel chooser needs an RNG")

        state_latents = getattr(evaluate, "state_latents", 0)
        root_batch = collate_positions(positions, state_latents=state_latents)
        root_logits, root_score, root_q = evaluate(root_batch)
        root_logits = root_logits.float().cpu()
        root_score = root_score.float().cpu()
        root_q = root_q.float().cpu()
        offsets = root_batch.legal_offsets.tolist()

        if sims == 0:
            return [
                position.nth_legal(
                    int(root_logits[offsets[k] : offsets[k + 1]].argmax())
                )
                for k, position in enumerate(positions)
            ]

        # One seed per root makes batching an execution detail rather than an
        # RNG decision: sequential singleton calls consume these same seeds.
        seeds = parent.integers(
            0, np.iinfo(np.uint64).max, size=len(positions), dtype=np.uint64
        )
        root_rngs = [np.random.default_rng(seed) for seed in seeds]

        # Candidate sampling uses raw policy logits (Gumbel MuZero convention).
        improved_policy(
            root_logits,
            root_score,
            root_q,
            root_batch.legal_offsets,
            tau,
            lam,
        )

        searches = []
        for k, position in enumerate(positions):
            lo, hi = offsets[k], offsets[k + 1]
            logits = root_logits[lo:hi].numpy()
            legal_count = hi - lo
            candidate_count = min(16, sims // 2, legal_count)
            if candidate_count < 1:
                # Budgets below two use policy argmax because no root set can expand.
                rank = int(root_logits[lo:hi].argmax())
                searches.append((rank, None))
                continue
            # Always drawn, then scaled, so the RNG stream stays a function of
            # the seeds alone and T = 1 is the unscaled draw exactly.
            gumbels = root_rngs[k].gumbel(size=legal_count) * temperature
            sampled = gumbels + logits
            order = np.argsort(-sampled, kind="stable")[:candidate_count]
            lines = []
            for rank in order:
                child = position.copy()
                child.advance(*position.nth_legal(int(rank)))
                terminal = bool(child.is_terminal)
                lines.append(
                    _Line(
                        root_rank=int(rank),
                        root_logit=float(logits[rank]),
                        gumbel=float(gumbels[rank]),
                        position=child,
                        value=(
                            _terminal_value(child, position.current_player)
                            if terminal
                            else 0.0
                        ),
                        terminal=terminal,
                    )
                )
            schedule = _halving_schedule(sims, candidate_count)
            searches.append(
                (
                    None,
                    _Search(
                        root_player=position.current_player,
                        lines=lines,
                        survivors=list(range(candidate_count)),
                        schedule=schedule,
                    ),
                )
            )

        active = [search for _rank, search in searches if search is not None]
        max_rounds = max((len(search.schedule) for search in active), default=0)
        for round_idx in range(max_rounds):
            round_searches = [
                search for search in active if round_idx < len(search.schedule)
            ]
            max_deepenings = max(
                search.schedule[round_idx][1] for search in round_searches
            )
            for wave in range(max_deepenings):
                pending: list[tuple[_Search, _Line]] = []
                for search in round_searches:
                    _survivors, deepenings = search.schedule[round_idx]
                    if wave >= deepenings:
                        continue
                    for line_idx in search.survivors:
                        line = search.lines[line_idx]
                        if line.terminal:
                            continue
                        if line.evaluated:
                            child = line.position.copy()
                            child.advance(
                                *line.position.nth_legal(int(line.policy_rank))
                            )
                            line.position = child
                            if child.is_terminal:
                                line.terminal = True
                                line.value = _terminal_value(
                                    child, search.root_player
                                )
                                line.visits += 1
                                continue
                        pending.append((search, line))
                if not pending:
                    continue
                leaf_batch = collate_positions(
                    [line.position for _search, line in pending],
                    state_latents=state_latents,
                )
                logits, q_score, q_values = evaluate(leaf_batch)
                logits = logits.float().cpu()
                q_score = q_score.float().cpu()
                q_values = q_values.float().cpu()
                leaf = improved_policy(
                    logits,
                    q_score,
                    q_values,
                    leaf_batch.legal_offsets,
                    tau,
                    lam,
                )
                leaf_offsets = leaf_batch.legal_offsets.tolist()
                for j, (search, line) in enumerate(pending):
                    lo, hi = leaf_offsets[j], leaf_offsets[j + 1]
                    line.policy_rank = int(leaf.probs[lo:hi].argmax())
                    sign = (
                        1.0
                        if line.position.current_player == search.root_player
                        else -1.0
                    )
                    line.value = sign * float(leaf.v_hat[j])
                    line.evaluated = True
                    line.visits += 1

            for search in round_searches:
                ranked = sorted(
                    search.survivors,
                    key=lambda line_idx: search.lines[line_idx].value,
                    reverse=True,
                )
                search.survivors = ranked[: (len(ranked) + 1) // 2]

        moves = []
        for position, (fallback_rank, search) in zip(
            positions, searches, strict=True
        ):
            if search is None:
                moves.append(position.nth_legal(fallback_rank))
                continue
            max_visits = max(line.visits for line in search.lines)
            best = max(
                search.survivors,
                key=lambda line_idx: (
                    search.lines[line_idx].gumbel
                    + search.lines[line_idx].root_logit
                    + (C_VISIT + max_visits)
                    * C_SCALE
                    * search.lines[line_idx].value
                ),
            )
            moves.append(position.nth_legal(search.lines[best].root_rank))
        return moves

    return choose
