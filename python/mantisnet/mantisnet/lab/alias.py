"""Structural alias diagnostic (MANTIS_GRAFT_SPEC §4 Step 4; donor ACT §33).

Two legal actions of one position alias when the builder-side inputs their
decoder rows are computed from are identical — the model cannot score them
apart, whatever its weights. The diagnostic reports alias groups under the
incumbent per-cell inputs (window classes through the cell, or the background
nearest-stone bucket) and under the Step 4 action-row inputs (all 18
post-placement row classes, EMPTY rows as their slot orbits), so a packet can
show what the row path merges and what it splits.

The 64-bit signature hash is only a bucket index; groups are declared on
exact signature equality within a bucket, never on the hash alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..builder import (
    ACTION_EMPTY,
    ACTION_EMPTY_CLASSES,
    WINDOW_LEN,
    PositionGraph,
    from_position,
)

# splitmix64 finalizer constants and an odd polynomial base, fixed so digests
# compare across processes and runs; arithmetic wraps at 64 bits.
_MASK = (1 << 64) - 1
_HASH_BASE = 0x9E3779B97F4A7C15
_MIX_A = 0xBF58476D1CE4E5B9
_MIX_B = 0x94D049BB133111EB

_EMPTY_ORBIT_CLASS = np.array(ACTION_EMPTY_CLASSES, dtype=np.int64)
_SLOT_ORBIT = np.minimum(np.arange(WINDOW_LEN), WINDOW_LEN - 1 - np.arange(WINDOW_LEN))


def _mix64(value: int) -> int:
    value = ((value ^ (value >> 30)) * _MIX_A) & _MASK
    value = ((value ^ (value >> 27)) * _MIX_B) & _MASK
    return value ^ (value >> 31)


def _hash_signature(signature: tuple[int, ...]) -> int:
    digest = len(signature)
    for term in signature:
        digest = _mix64((digest * _HASH_BASE + (term & 0xFFFFFFFF)) & _MASK)
    return digest


def _before_signatures(graph: PositionGraph) -> list[tuple[int, ...]]:
    """Incumbent decoder inputs per legal cell: the sorted window-class
    multiset through it, or a tagged background bucket."""
    signatures: list[tuple[int, ...]] = [() for _ in range(graph.n_legal)]
    order = np.argsort(graph.dec_cell, kind="stable")
    cells = graph.dec_cell[order]
    classes = graph.dec_class[order]
    for cell, lo, hi in zip(*_runs(cells)):
        signatures[cell] = (0, *sorted(int(c) for c in classes[lo:hi]))
    for cell, bucket in zip(graph.bg_cell, graph.bg_bucket):
        signatures[int(cell)] = (1, int(bucket))
    return signatures


def _after_signatures(graph: PositionGraph) -> list[tuple[int, ...]]:
    """Step 4 inputs per legal cell: all 18 row classes, EMPTY rows carried
    as their slot-orbit insert classes."""
    status = graph.action_pre_status
    post1 = graph.action_post1_class
    orbit_class = _EMPTY_ORBIT_CLASS[_SLOT_ORBIT][None, None, :]
    rows = np.where(status == ACTION_EMPTY, orbit_class, post1)
    rows = np.sort(rows.reshape(graph.n_legal, -1), axis=1)
    return [tuple(int(c) for c in row) for row in rows]


def _runs(sorted_values: np.ndarray):
    if len(sorted_values) == 0:
        return (), (), ()
    boundaries = np.flatnonzero(np.diff(sorted_values)) + 1
    starts = np.concatenate([[0], boundaries])
    stops = np.concatenate([boundaries, [len(sorted_values)]])
    return sorted_values[starts].astype(int), starts, stops


@dataclass
class _Tally:
    actions: int = 0
    unique: int = 0
    groups: int = 0
    aliased_actions: int = 0
    max_group: int = 0
    background_merged_groups: int = 0

    def as_dict(self) -> dict:
        return {
            "legal_actions": self.actions,
            "unique_signatures": self.unique,
            "alias_groups": self.groups,
            "aliased_actions": self.aliased_actions,
            "max_alias_group": self.max_group,
            "groups_with_background_cells": self.background_merged_groups,
        }


def _tally_position(
    signatures: list[tuple[int, ...]],
    background: set[int],
    tally: _Tally,
    examples: list,
    legal_qr: np.ndarray,
    example_cap: int,
) -> None:
    buckets: dict[int, dict[tuple[int, ...], list[int]]] = {}
    for cell, signature in enumerate(signatures):
        buckets.setdefault(_hash_signature(signature), {}).setdefault(
            signature, []
        ).append(cell)
    tally.actions += len(signatures)
    for exact in buckets.values():
        for cells in exact.values():
            tally.unique += 1
            if len(cells) > 1:
                tally.groups += 1
                tally.aliased_actions += len(cells)
                tally.max_group = max(tally.max_group, len(cells))
                if any(cell in background for cell in cells):
                    tally.background_merged_groups += 1
                if len(examples) < example_cap:
                    examples.append(
                        [[int(q), int(r)] for q, r in legal_qr[cells]]
                    )


def alias_report(
    positions,
    *,
    example_cap: int = 8,
) -> dict:
    """Alias groups before/after the Step 4 row inputs over ``positions``."""
    before, after = _Tally(), _Tally()
    before_examples: list = []
    after_examples: list = []
    n_positions = 0
    for pos in positions:
        graph = from_position(pos, action_rows=True)
        legal_qr = np.asarray(pos.legal_moves(), dtype=np.int64).reshape(-1, 2)
        background = {int(c) for c in graph.bg_cell}
        _tally_position(
            _before_signatures(graph), background, before, before_examples,
            legal_qr, example_cap,
        )
        _tally_position(
            _after_signatures(graph), background, after, after_examples,
            legal_qr, example_cap,
        )
        n_positions += 1
    if n_positions == 0:
        raise ValueError("the alias diagnostic requires at least one position")
    return {
        "mode": "alias",
        "positions": n_positions,
        "before": before.as_dict(),
        "after": after.as_dict(),
        "sampled_alias_coordinates": {
            "before": before_examples,
            "after": after_examples,
        },
    }


def corpus_alias_report(
    corpus_path: str | Path,
    *,
    split: str = "val",
    sample: int = 2_000,
    seed: int = 0,
    example_cap: int = 8,
) -> dict:
    """Run the diagnostic over a deterministic corpus-position sample."""
    import hexo_py

    from .corpus import load_corpus

    if sample <= 0:
        raise ValueError(f"sample must be positive, got {sample}")
    frozen = load_corpus(Path(corpus_path))
    samples = frozen.split_samples(split)
    if not len(samples):
        raise ValueError(f"corpus split {split!r} is empty")
    count = min(sample, len(samples))
    picks = np.sort(
        np.random.default_rng(seed).choice(len(samples), size=count, replace=False)
    )

    def positions():
        for index in picks:
            moves = frozen.moves_for(int(samples.game[index]))
            t = int(samples.t[index])
            pos = hexo_py.Position.replay(moves[:t])
            if pos.is_terminal:
                raise ValueError(
                    f"corpus sample {index} names a terminal prefix"
                )
            yield pos

    report = alias_report(positions(), example_cap=example_cap)
    report["corpus"] = {"name": frozen.name, "sha256": frozen.sha256}
    report["split"] = split
    report["sample"] = count
    report["seed"] = seed
    return report
