"""Structural alias diagnostic (MANTIS_GRAFT_SPEC §4 Step 4; donor ACT §33).

Two legal actions of one position alias when all 18 post-placement row classes
are identical, with EMPTY rows represented by their slot orbits. The model
cannot score such actions apart through its structural action input, whatever
its weights.

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


def _signatures(graph: PositionGraph) -> list[tuple[int, ...]]:
    """Action inputs per legal cell: all 18 row classes, EMPTY rows carried
    as their slot-orbit insert classes."""
    status = graph.action_pre_status
    post1 = graph.action_post1_class
    orbit_class = _EMPTY_ORBIT_CLASS[_SLOT_ORBIT][None, None, :]
    rows = np.where(status == ACTION_EMPTY, orbit_class, post1)
    rows = np.sort(rows.reshape(graph.n_legal, -1), axis=1)
    return [tuple(int(c) for c in row) for row in rows]


@dataclass
class _Tally:
    actions: int = 0
    unique: int = 0
    groups: int = 0
    aliased_actions: int = 0
    max_group: int = 0

    def as_dict(self) -> dict:
        return {
            "legal_actions": self.actions,
            "unique_signatures": self.unique,
            "alias_groups": self.groups,
            "aliased_actions": self.aliased_actions,
            "max_alias_group": self.max_group,
        }


def _tally_position(
    signatures: list[tuple[int, ...]],
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
                if len(examples) < example_cap:
                    examples.append(
                        [[int(q), int(r)] for q, r in legal_qr[cells]]
                    )


def alias_report(
    positions,
    *,
    example_cap: int = 8,
) -> dict:
    """Alias groups under the baked action-row inputs over ``positions``."""
    tally = _Tally()
    examples: list = []
    n_positions = 0
    for pos in positions:
        graph = from_position(pos)
        legal_qr = np.asarray(pos.legal_moves(), dtype=np.int64).reshape(-1, 2)
        _tally_position(
            _signatures(graph), tally, examples,
            legal_qr, example_cap,
        )
        n_positions += 1
    if n_positions == 0:
        raise ValueError("the alias diagnostic requires at least one position")
    return {
        "mode": "alias",
        "positions": n_positions,
        "baked": tally.as_dict(),
        "sampled_alias_coordinates": examples,
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
