"""SealBot as the external yardstick: matches against an independent bot.

SealBot (github.com/Ramora0/SealBot) is a C++ iterative-deepening alpha-beta
bot for this exact game — 6-in-a-row on the infinite hex grid, one stone on
the opening turn and two per turn after. It shares no code, no heuristic,
and no training history with anything in this repo, which is what makes a
score against it a strength measurement rather than a self-referential one:
self-play metrics were measured to look perfect while a checkpoint lost
63/64 to an external opponent (``docs/KLENT_RUN_PLAN.md`` §3). It is the
project's one evaluation, in the paper's anchored-external-opponent role.

Interface facts this module leans on, checked against its source:

- ``MinimaxBot(time_limit).get_move(game)`` reads a ``game.HexGame`` and
  returns a *whole turn* (1–2 placements) from one time-limited search. The
  C++ compares board values against ``game.Player`` members by identity, so
  the mirror must be built from SealBot's own ``game`` module.
- Its candidate generation stays within distance 2 of existing stones
  (``NEIGHBOR_DIST``), strictly inside the engine's distance-8 legality —
  every move is asserted against ``legal_moves()`` anyway.
- Its board is a flat array over coordinates [-70, 69]. A game drifting
  past ±60 is refused loudly here rather than trusted to its padding.

``hexo_py.Position`` stays authoritative throughout: SealBot's ``HexGame``
is rebuilt from the move list at every consultation, its turn state is
asserted against the engine's, and at each finished game the two rule
implementations must name the same winner — a live cross-check between two
independent rule implementations, in the spirit of the §12.1 oracles.

Games run in seat-balanced pairs from shared uniform-random openings: a
deterministic argmax policy against a mostly-deterministic searcher would
otherwise replay near-identical games, and a paired design cancels opening
luck out of the comparison.

CLI::

    python -m mantisnet.klent.sealbot --sealbot D:/SealBot \
        --checkpoint runs/<run>/checkpoint_NNNNNN.pt --games 64 --time 0.1
    python -m mantisnet.klent.sealbot --sealbot D:/SealBot \
        --run runs/<run> --every 250        # strength curve -> sealbot_curve.jsonl
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

from .evaluate import argmax_choose

# SealBot's board covers [-70, 69] with ±5 window padding; refuse well clear.
_COORD_LIMIT = 60

# Concurrent games per lockstep wave. Each MinimaxBot owns a fixed 2^20-entry
# transposition table (~32 MiB), so the wave size caps host memory, not VRAM.
_WAVE = 16

_loaded_variant: str | None = None


def load_sealbot(root: Path, variant: str = "current"):
    """Import ``(game module, MinimaxBot)`` from a SealBot checkout.

    ``minimax_cpp`` is a C extension and can be loaded once per process, so a
    second call must ask for the same variant."""
    global _loaded_variant
    root = Path(root)
    bot_dir = root / variant
    if not (root / "game.py").exists():
        raise FileNotFoundError(f"{root} is not a SealBot checkout: no game.py")
    if not any(bot_dir.glob("minimax_cpp*.pyd")) and not any(
        bot_dir.glob("minimax_cpp*.so")
    ):
        raise FileNotFoundError(
            f"no built minimax_cpp in {bot_dir} — build it there with "
            "`python setup.py build_ext --inplace` (needs pybind11 + setuptools)"
        )
    if _loaded_variant is not None and _loaded_variant != str(bot_dir):
        raise RuntimeError(
            f"SealBot variant {_loaded_variant} is already loaded; "
            "C extensions load once per process"
        )
    for p in (str(bot_dir), str(root)):
        if p not in sys.path:
            sys.path.insert(0, p)
    game_mod = importlib.import_module("game")
    minimax = importlib.import_module("minimax_cpp")
    _loaded_variant = str(bot_dir)
    return game_mod, minimax.MinimaxBot


def _mirror(game_mod, moves):
    """SealBot's ``HexGame`` holding exactly ``moves`` — every placement must
    be accepted, or the two rule implementations have diverged."""
    g = game_mod.HexGame()
    for q, r in moves:
        if g.game_over or not g.make_move(int(q), int(r)):
            raise RuntimeError(
                f"rules mismatch: SealBot's HexGame refused move {(q, r)} "
                f"at placement {g.move_count}"
            )
    return g


def _openings(rng: np.random.Generator, n: int, cut_range: tuple[int, int]):
    """``n`` uniform-random openings of uniform length in ``cut_range`` —
    short shared prefixes that decorrelate otherwise-deterministic games.
    Lengths ≤ 10 cannot be terminal (a win needs six stones of one colour)."""
    import hexo_py

    lo, hi = cut_range
    if not 1 <= lo <= hi <= 10:
        raise ValueError(f"opening range must satisfy 1 <= lo <= hi <= 10: {cut_range}")
    moves: list[list[tuple[int, int]]] = []
    for target in rng.integers(lo, hi + 1, size=n):
        pos = hexo_py.Position()
        opening = []
        for _ in range(int(target)):
            move = pos.nth_legal(int(rng.integers(pos.legal_count)))
            pos.advance(*move)
            opening.append(move)
        moves.append(opening)
    return moves


def _apply(state, move, check_legal: bool):
    q, r = int(move[0]), int(move[1])
    if max(abs(q), abs(r)) > _COORD_LIMIT:
        raise RuntimeError(
            f"game left SealBot's coordinate range: {(q, r)} exceeds ±{_COORD_LIMIT}"
        )
    if check_legal and (q, r) not in set(state["pos"].legal_moves()):
        raise RuntimeError(
            f"SealBot proposed {(q, r)}, illegal at placement {len(state['moves'])}"
        )
    state["pos"].advance(q, r)
    state["moves"].append((q, r))


def _play_wave(states, choose, game_mod, rng, ply_cap):
    """Advance one wave of games to completion, lockstep: the model's games
    batch into one chooser call per step, SealBot's run sequentially (each
    consultation is its own time-limited search)."""
    live = list(range(len(states)))
    while live:
        model_group = [
            k for k in live if states[k]["pos"].current_player == states[k]["seat"]
        ]
        for k, move in zip(
            model_group, choose([states[k]["pos"] for k in model_group], rng)
        ):
            _apply(states[k], move, check_legal=False)
        for k in live:
            s = states[k]
            if k in model_group or s["pos"].is_terminal:
                continue
            mirror = _mirror(game_mod, s["moves"])
            assert mirror.current_player.value - 1 == s["pos"].current_player
            assert mirror.moves_left_in_turn == s["pos"].moves_remaining
            turn = s["bot"].get_move(mirror)
            if not turn:
                raise RuntimeError("SealBot returned no moves for a live position")
            s["depths"].append(s["bot"].last_depth)
            seat = s["pos"].current_player
            for move in turn:
                if s["pos"].is_terminal or s["pos"].current_player != seat:
                    break
                _apply(s, move, check_legal=True)
        still = []
        for k in live:
            s = states[k]
            if s["pos"].is_terminal:
                final = _mirror(game_mod, s["moves"])
                assert final.game_over and final.winner.value - 1 == s["pos"].winner, (
                    "rules mismatch: the two implementations name different winners"
                )
                s["bot"] = None  # release the 32 MiB transposition table
            elif len(s["moves"]) >= ply_cap:
                s["capped"] = True
                s["bot"] = None
            else:
                still.append(k)
        live = still


def _wilson(score: float, n: int, z: float = 1.96):
    p = score / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, centre - spread), min(1.0, centre + spread)


def _elo(score: float) -> float:
    if score <= 0.0:
        return -math.inf
    if score >= 1.0:
        return math.inf
    return -400.0 * math.log10(1.0 / score - 1.0)


def sealbot_match(
    model,
    device: str,
    games: int,
    ply_cap: int,
    rng: np.random.Generator,
    time_limit: float,
    sealbot_root: Path,
    variant: str = "current",
    opening_range: tuple[int, int] = (2, 6),
    max_depth: int | None = None,
) -> dict:
    """``games`` seat-balanced games of argmax π_θ against SealBot, paired
    two per opening. Returns the model's score (win 1, cap ½) with a Wilson
    interval and its Elo transform, plus per-seat scores and SealBot's mean
    search depth — the honest context for the headline number."""
    if games < 2 or games % 2:
        raise ValueError(f"games must be even and >= 2 (paired seats): {games}")
    game_mod, MinimaxBot = load_sealbot(sealbot_root, variant)
    model.eval()
    choose = argmax_choose(model, device)

    import hexo_py

    openings = _openings(rng, games // 2, opening_range)
    states = []
    for g_idx in range(games):
        opening = openings[g_idx // 2]
        bot = MinimaxBot(time_limit)
        if max_depth is not None:
            bot.max_depth = max_depth  # a fixed-depth rung of the ladder
        states.append(
            {
                "pos": hexo_py.Position.replay([tuple(m) for m in opening]),
                "moves": [tuple(m) for m in opening],
                "seat": g_idx % 2,  # even games: the model takes P0
                "bot": bot,
                "depths": [],
                "capped": False,
            }
        )

    t0 = time.monotonic()
    for start in range(0, games, _WAVE):
        _play_wave(states[start : start + _WAVE], choose, game_mod, rng, ply_cap)

    score, per_seat, capped, plies, depths = 0.0, [0.0, 0.0], 0, 0, []
    for s in states:
        if s["capped"]:
            capped += 1
            g_score = 0.5
        else:
            g_score = 1.0 if s["pos"].winner == s["seat"] else 0.0
        score += g_score
        per_seat[s["seat"]] += g_score
        plies += len(s["moves"])
        depths.extend(s["depths"])

    ci_lo, ci_hi = _wilson(score, games)
    return {
        "score": score,
        "games": games,
        "capped": capped,
        "win_rate": score / games,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "elo": _elo(score / games),
        "elo_lo": _elo(ci_lo),
        "elo_hi": _elo(ci_hi),
        "score_as_p0": per_seat[0],
        "score_as_p1": per_seat[1],
        "sealbot_depth_mean": float(np.mean(depths)) if depths else float("nan"),
        "sealbot_time_limit": time_limit,
        "sealbot_max_depth": max_depth,
        "avg_plies": plies / games,
        "seconds": time.monotonic() - t0,
    }


def _fmt(result: dict) -> str:
    fe = lambda e: f"{e:+.0f}" if math.isfinite(e) else ("+inf" if e > 0 else "-inf")  # noqa: E731
    return (
        f"vs SealBot({result['sealbot_time_limit']}s): "
        f"{result['score']:.1f}/{result['games']} "
        f"({100 * result['win_rate']:.1f}%, "
        f"CI {100 * result['ci_lo']:.0f}-{100 * result['ci_hi']:.0f}%) "
        f"elo {fe(result['elo'])} ({fe(result['elo_lo'])}..{fe(result['elo_hi'])}) "
        f"| P0 {result['score_as_p0']:.1f} P1 {result['score_as_p1']:.1f} "
        f"capped {result['capped']} | depth {result['sealbot_depth_mean']:.1f} "
        f"plies {result['avg_plies']:.0f} | {result['seconds']:.0f}s"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sealbot", type=Path, required=True, help="SealBot checkout root")
    parser.add_argument("--variant", choices=("current", "best"), default="current")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--checkpoint", type=Path, help="one checkpoint to measure")
    target.add_argument("--run", type=Path, help="run directory: measure the curve")
    parser.add_argument("--every", type=int, default=250,
                        help="with --run, iteration stride between measured checkpoints")
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--time", type=float, default=0.1, help="SealBot seconds per turn")
    parser.add_argument("--max-depth", type=int, default=None,
                        help="cap SealBot's search depth (a weaker ladder rung)")
    parser.add_argument("--cap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    from .run import load_model

    if args.checkpoint is not None:
        model = load_model(args.checkpoint, args.device)
        result = sealbot_match(
            model, args.device, args.games, args.cap,
            np.random.default_rng(args.seed), args.time, args.sealbot, args.variant,
            max_depth=args.max_depth,
        )
        print(f"{args.checkpoint.name}  {_fmt(result)}")
        print(json.dumps(result))
        return

    checkpoints = sorted(args.run.glob("checkpoint_*.pt"))
    if not checkpoints:
        raise SystemExit(f"no checkpoints in {args.run}")
    picked = [
        p for p in checkpoints
        if int(re.search(r"(\d+)", p.stem).group(1)) % args.every == 0
    ]
    if checkpoints[-1] not in picked:
        picked.append(checkpoints[-1])

    out = args.run / "sealbot_curve.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        for path in picked:
            model = load_model(path, args.device)
            # One seed for every checkpoint: identical openings pair the
            # whole curve, so differences are the model, not the draw.
            result = sealbot_match(
                model, args.device, args.games, args.cap,
                np.random.default_rng(args.seed), args.time, args.sealbot, args.variant,
                max_depth=args.max_depth,
            )
            row = {"checkpoint": path.name} | result
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(f"{path.name}  {_fmt(result)}", flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
