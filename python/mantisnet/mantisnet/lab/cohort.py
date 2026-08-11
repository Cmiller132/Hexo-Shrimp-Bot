"""Production-shaped position cohorts for lab measurements.

Positions come from stepping the real ``Collector`` or replaying prefixes
from a frozen corpus, not random playouts: random play averages roughly
2,400 legal cells per position at ply 50, versus roughly 540 under real
collection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from ..klent.selfplay import Collector
from ..klent.train import KlentConfig, network_evaluate
from ..model import MantisConfig, MantisNet


class _CohortReady(Exception):
    """Internal control flow used to stop ``Collector.collect`` at a step."""


@dataclass(frozen=True, slots=True)
class CohortCase:
    """One live position paired with the exact engine prefix that produced it."""

    position: object
    moves: tuple[tuple[int, int], ...]


def selfplay_cohort(
    *,
    envs: int = 16,
    steps: int = 32,
    evaluate: Callable | None = None,
    model=None,
    seed: int = 0,
    device: str = "cpu",
    compile: bool = False,
    pair_budget: int | None = None,
    cell_budget: int | None = None,
    with_prefixes: bool = False,
) -> list:
    """Return live positions after ``steps`` real lockstep collector steps.

    Falls back from ``evaluate`` to ``model`` wrapped in
    :func:`network_evaluate` to a fresh default MantisNet. Ended slots are
    reset as in training, so every returned position is live.
    ``with_prefixes`` also returns each position's exact move history.
    """
    if envs <= 0:
        raise ValueError(f"envs must be positive, got {envs}")
    if steps < 0:
        raise ValueError(f"steps must be nonnegative, got {steps}")
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"device must be 'cpu' or 'cuda', got {device!r}")

    cfg = KlentConfig(
        device=device,
        autocast=device == "cuda",
        compile=compile,
        collect_pair_budget=(
            pair_budget if pair_budget is not None else KlentConfig.collect_pair_budget
        ),
        collect_cell_budget=(
            cell_budget if cell_budget is not None else KlentConfig.collect_cell_budget
        ),
    )
    if evaluate is None:
        if model is None:
            import torch

            torch.manual_seed(seed)
            model = MantisNet(MantisConfig()).to(device).eval()
        composition = getattr(model, "family_composition", None)
        if composition is None:
            evaluate = network_evaluate(model, cfg)
        else:
            from .families import composition_evaluate

            evaluate = composition_evaluate(model, composition, cfg)

    collector = Collector(
        envs=envs,
        ply_cap=max(KlentConfig.ply_cap, steps + 1),
        tau=cfg.tau,
        lam=cfg.lam,
        rng=np.random.default_rng(seed),
        pair_budget=cfg.collect_pair_budget,
        cell_budget=cfg.collect_cell_budget,
        # A bare ``evaluate`` carries no config; a knob mismatch behind it is
        # caught loudly by the model's own batch scope check.
        action_rows=bool(
            model is not None and getattr(model, "cfg", None) is not None
            and model.cfg.action_rows
        ),
    )
    if steps == 0:
        positions = list(collector.positions)
        if with_prefixes:
            return [
                CohortCase(position, tuple(episode.moves))
                for position, episode in zip(
                    positions, collector.episodes, strict=True
                )
            ]
        return positions

    seen = 0

    def progress(_finished, _quota, _slot_plies):
        nonlocal seen
        seen += 1
        if seen >= steps:
            raise _CohortReady

    # At most ``envs`` games can end in one lockstep barrier.  This quota is
    # unreachable before the progress callback stops the collector.
    quota = envs * steps + 1
    try:
        collector.collect(evaluate, quota, progress=progress)
    except _CohortReady:
        pass
    else:  # pragma: no cover - protected by the unreachable quota argument.
        raise RuntimeError("collector ended before the requested cohort depth")

    positions = list(collector.positions)
    if len(positions) != envs or any(p.is_terminal for p in positions):
        raise RuntimeError("collector returned a terminal or incomplete cohort")
    if with_prefixes:
        cases = [
            CohortCase(position, tuple(episode.moves))
            for position, episode in zip(positions, collector.episodes, strict=True)
        ]
        if any(len(case.moves) != case.position.stone_count for case in cases):
            raise RuntimeError("collector position and retained prefix differ in length")
        return cases
    return positions


def _sample_arrays(corpus, split: str):
    """Read the two prefix-reference arrays from the corpus contract."""
    samples = corpus.split_samples(split)
    return np.asarray(samples.game), np.asarray(samples.t)


def corpus_cohort(
    corpus,
    *,
    split: str = "test",
    count: int = 16,
    seed: int = 0,
    indices: Sequence[int] | None = None,
    with_prefixes: bool = False,
) -> list:
    """Replay sampled frozen-corpus prefixes into live engine positions.

    ``with_prefixes`` returns :class:`CohortCase` records instead, preserving
    the exact archived histories for representation-aware contract checks.
    """
    if isinstance(corpus, (str, Path)):
        from .corpus import load_corpus

        corpus = load_corpus(corpus)
    games, ts = _sample_arrays(corpus, split)
    if len(games) == 0:
        raise ValueError(f"corpus split {split!r} has no sampled positions")
    if indices is None:
        if count <= 0:
            raise ValueError(f"count must be positive, got {count}")
        take = min(count, len(games))
        indices = np.random.default_rng(seed).choice(len(games), take, replace=False)
    else:
        indices = np.asarray(indices, dtype=np.int64)
        if indices.size == 0:
            raise ValueError("indices must select at least one corpus sample")
        if indices.min() < 0 or indices.max() >= len(games):
            raise IndexError(f"sample index outside split of length {len(games)}")

    import hexo_py

    positions = []
    for i in indices:
        game, t = int(games[int(i)]), int(ts[int(i)])
        moves = corpus.moves_for(game)
        if t < 0 or t >= len(moves):
            raise ValueError(
                f"{split} sample {int(i)} has t={t} outside game {game} "
                f"of length {len(moves)}"
            )
        pos = hexo_py.Position.replay(moves[:t])
        if pos.is_terminal:
            raise ValueError(f"{split} sample {int(i)} replays to a terminal prefix")
        positions.append(
            CohortCase(pos, tuple(moves[:t])) if with_prefixes else pos
        )
    return positions
