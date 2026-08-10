"""Phase-stratified census of the six MantisNet-ACT action auxiliaries.

Run from ``python/mantisnet``. The frozen corpus is loaded through the lab's
validated public loader; positions are replayed by the authoritative engine,
and labels come from the public Rust-backed auxiliary-label seam.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import hexo_py
from mantisnet.lab.corpus import SPLIT_IDS, SPLIT_NAMES, load_corpus
from mantisnet.models.mantis_act.aux_labels import position_aux_labels
from mantisnet.models.mantis_act.config import PRESETS

PHASES = ("OPENING", "FIRST", "SECOND")


def _phase(position) -> str:
    if int(position.stone_count) == 0:
        return "OPENING"
    if int(position.moves_remaining) == 2:
        return "FIRST"
    return "SECOND"


def _sample_pairs(corpus, split: str, per_phase: int, seed: int):
    samples = corpus.split_samples(split)
    rng = np.random.default_rng(seed)
    pairs: dict[str, list[tuple[int, int]]] = {}

    # There is only one opening position. Repeating it from many games would
    # inflate the action count without adding another board state.
    split_games = np.flatnonzero(corpus.split == SPLIT_IDS[split])
    if not split_games.size:
        raise ValueError(f"corpus split {split!r} contains no games")
    pairs["OPENING"] = [(int(split_games[0]), 0)]

    phase_masks = {
        "FIRST": (samples.t > 0) & ((samples.t % 2) == 1),
        "SECOND": (samples.t > 0) & ((samples.t % 2) == 0),
    }
    for phase, mask in phase_masks.items():
        candidates = np.flatnonzero(mask)
        if candidates.size < per_phase:
            raise ValueError(
                f"split {split!r} has only {candidates.size} sampled {phase} "
                f"positions, fewer than --per-phase={per_phase}"
            )
        chosen = np.sort(rng.choice(candidates, size=per_phase, replace=False))
        pairs[phase] = [
            (int(samples.game[row]), int(samples.t[row])) for row in chosen
        ]
    return pairs


def _census(corpus, pairs):
    cfg = PRESETS["full_act_v4"]
    result = {}
    for phase in PHASES:
        counts: dict[str, np.ndarray] = {}
        positions = 0
        actions = 0
        plies = []
        for game, t in pairs[phase]:
            moves = corpus.moves_for(game)
            position = hexo_py.Position.replay(moves[:t])
            actual_phase = _phase(position)
            if actual_phase != phase:
                raise AssertionError(
                    f"game {game} ply {t} selected as {phase}, engine says "
                    f"{actual_phase}"
                )
            labels = position_aux_labels(position, cfg)
            for name, values in labels.items():
                observed = np.bincount(values)
                if name not in counts:
                    counts[name] = np.zeros(observed.size, dtype=np.int64)
                if observed.size > counts[name].size:
                    counts[name] = np.pad(
                        counts[name], (0, observed.size - counts[name].size)
                    )
                counts[name][: observed.size] += observed
            positions += 1
            actions += int(position.legal_count)
            plies.append(t)

        result[phase] = {
            "positions": positions,
            "legal_actions": actions,
            "ply_min": min(plies),
            "ply_median": float(np.median(plies)),
            "ply_max": max(plies),
            "labels": {
                name: {
                    "counts": values.tolist(),
                    "fractions": (values / actions).tolist(),
                }
                for name, values in sorted(counts.items())
            },
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("runs/corpora/act-abl-v1"),
        help="validated frozen-corpus directory",
    )
    parser.add_argument("--split", choices=SPLIT_NAMES, default="train")
    parser.add_argument("--per-phase", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.per_phase < 1:
        parser.error("--per-phase must be at least 1")

    corpus = load_corpus(args.corpus)
    pairs = _sample_pairs(corpus, args.split, args.per_phase, args.seed)
    payload = {
        "corpus": corpus.name,
        "corpus_sha256": corpus.sha256,
        "split": args.split,
        "seed": args.seed,
        "requested_nonopening_positions_per_phase": args.per_phase,
        "phases": _census(corpus, pairs),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
