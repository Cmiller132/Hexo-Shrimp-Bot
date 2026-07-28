"""Checkpoint cross-play (`KLENT_PROPOSALS.md` A7): the forgetting detector.

Round-robin argmax-vs-argmax between every checkpoint pair of one run
directory, seat balanced, caps scored ½. Monotone improvement — later
checkpoints beating earlier ones — is health; a cycle (A beats B beats C
beats A) is the cyclic-forgetting signature that eval against a fixed
anchor cannot see. Results land in ``crossplay.json`` beside the
checkpoints, keyed ``"a vs b"`` with A's score fraction.

    uv run python -m mantisnet.klent.crossplay --run runs/<name> --games 64
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from .evaluate import argmax_choose, play_match
from .run import load_model


def cross_play(
    run_dir: Path, games: int, ply_cap: int, device: str, seed: int
) -> dict:
    """Every pair once; each pairing's RNG derives from (seed, pair), so a
    matrix reproduces and extends when new checkpoints appear."""
    paths = sorted(run_dir.glob("checkpoint_*.pt"))
    if len(paths) < 2:
        raise SystemExit(f"cross-play needs at least two checkpoints under {run_dir}")
    choosers = {p.name: argmax_choose(load_model(p, device), device) for p in paths}

    names = list(choosers)
    matrix = {}
    for a, b in itertools.combinations(names, 2):
        rng = np.random.default_rng([seed, names.index(a), names.index(b)])
        result = play_match(choosers[a], choosers[b], games, ply_cap, rng)
        matrix[f"{a} vs {b}"] = {
            "score_a": result["score_a"] / games,
            "capped": result["capped"],
            "games": games,
        }
    return matrix


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True, help="a run directory")
    ap.add_argument("--games", type=int, default=64, help="games per pairing")
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    matrix = cross_play(args.run, args.games, args.cap, args.device, args.seed)
    (args.run / "crossplay.json").write_text(
        json.dumps(matrix, indent=2), encoding="utf-8"
    )
    for pair, r in matrix.items():
        print(f"{pair}: {r['score_a']:.3f} ({r['capped']}/{r['games']} capped)")


if __name__ == "__main__":
    main()
