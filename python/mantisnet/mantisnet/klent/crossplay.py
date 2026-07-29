"""Evaluate every checkpoint pair in a run; see ``KLENT_FOR_HEXO.md`` §6.3.

Crossplay is raw-policy argmax against argmax, seat balanced when ``games`` is
even, with capped games scored ½. Each invocation replaces ``crossplay.json``
and the telemetry crossplay table with one result per unordered pair.

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
from .telemetry import open_telemetry


def cross_play(
    run_dir: Path, games: int, ply_cap: int, device: str, seed: int
) -> list[dict]:
    """Evaluate each unordered checkpoint pair once.

    Pair RNG derives from ``(seed, index_a, index_b)``. Rows name checkpoints
    separately; ``crossplay.json`` formats those names as ``"a vs b"`` keys.
    """
    paths = sorted(run_dir.glob("checkpoint_*.pt"))
    if len(paths) < 2:
        raise SystemExit(f"cross-play needs at least two checkpoints under {run_dir}")
    choosers = {p.name: argmax_choose(load_model(p, device), device) for p in paths}

    names = list(choosers)
    rows = []
    for a, b in itertools.combinations(names, 2):
        rng = np.random.default_rng([seed, names.index(a), names.index(b)])
        result = play_match(choosers[a], choosers[b], games, ply_cap, rng)
        rows.append(
            {
                "a": a,
                "b": b,
                "score_a": result["score_a"] / games,
                "capped": result["capped"],
                "games": games,
            }
        )
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True, help="a run directory")
    ap.add_argument("--games", type=int, default=64, help="games per pairing")
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    rows = cross_play(args.run, args.games, args.cap, args.device, args.seed)
    matrix = {
        f"{r['a']} vs {r['b']}": {k: r[k] for k in ("score_a", "capped", "games")}
        for r in rows
    }
    (args.run / "crossplay.json").write_text(
        json.dumps(matrix, indent=2), encoding="utf-8"
    )
    with open_telemetry(args.run) as telemetry:
        telemetry.write_crossplay(rows, ply_cap=args.cap, seed=args.seed)
    for pair, r in matrix.items():
        print(f"{pair}: {r['score_a']:.3f} ({r['capped']}/{r['games']} capped)")


if __name__ == "__main__":
    main()
