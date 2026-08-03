"""Paired head-to-head between two checkpoints, from any two run directories.

Two independent SealBot scores cannot resolve a small difference: each carries
its own binomial noise plus whatever the opponent's openings and seat draws did
to it, so 64 games a side leave about eight percentage points unresolved. A
direct match removes the anchor, and pairing removes most of what is left. Each
of ``pairs`` uniform-random openings is played twice with the seats swapped, both
models searching at the same ``sims`` through ``search.gumbel_choose``, and the
statistic is the per-pair difference

    d_i = (A's wins in pair i) - 1     in {-1, 0, +1}

whose standard error is reported next to the unpaired one, so the variance the
pairing bought is visible rather than asserted. A pair is one unit: it shares its
opening prefix and its whole RNG stream, derived from ``(seed, pair index)``, and
is therefore reproducible on its own.

A capped game scores ½ as everywhere else in this repo. A cap is not a decision,
so a capped pair is counted apart from the win/split/loss counts, kept out of the
sign test, and named in ``warnings``. Pairs that all carry the same ``d`` — the
shape an all-splits match takes — have no spread to estimate and so get no Elo
interval and no SE ratio, which is also named in ``warnings``.

Run from ``python/mantisnet``:

    python -m mantisnet.klent.headtohead --a A.pt --b B.pt --pairs 64 --sims 32 \
        --tau 0.1 --lam 0.01 --out h2h.json
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .evaluate import play_match
from .opponents import elo, shared_openings, wilson
from .run import load_model
from .search import gumbel_choose
from .train import KlentConfig, network_evaluate

# Both intervals — the marginal Wilson one and the paired Elo one — are two-sided
# 95%, so they are read on the same scale.
Z = 1.96

# The build fields whose disagreement makes the two checkpoints incomparable and
# for which no conversion exists. ``MODEL_REPR_VERSION`` is handled separately
# because it has one.
_INCOMPATIBLE_FIELDS = ("RULES_VERSION", "ACTION_ORDER_VERSION", "torch")

_READ_BLOCK = 1 << 20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_READ_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: float) -> float | None:
    """``None`` for a non-finite Elo, which a 0% or 100% score legitimately gives.

    The output is strict JSON and ``Infinity`` is not part of it."""
    return value if math.isfinite(value) else None


def _audit(path: Path, field: str) -> dict:
    """A checkpoint's audit record: path, digest, iteration, and its own versions."""
    if not path.exists():
        raise FileNotFoundError(f"{field}: no checkpoint at {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not {"versions", "iteration"} <= set(
        checkpoint
    ):
        raise ValueError(f"{field}: {path} has no versions and iteration: not a checkpoint")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "iteration": checkpoint["iteration"],
        "versions": checkpoint["versions"],
    }


def _refuse_incomparable(a: dict, b: dict) -> None:
    """Refuse two checkpoints that do not describe the same game, Torch build, or
    representation. The representation has a bridge; the rest do not."""
    va, vb = a["versions"], b["versions"]
    differing = [f for f in _INCOMPATIBLE_FIELDS if va.get(f) != vb.get(f)]
    if differing:
        raise ValueError(
            "a head-to-head needs one set of rules and one Torch build on both "
            "sides: "
            + "; ".join(
                f"{field} is {va.get(field)!r} in {a['path']} and "
                f"{vb.get(field)!r} in {b['path']}"
                for field in differing
            )
        )
    if va.get("MODEL_REPR_VERSION") != vb.get("MODEL_REPR_VERSION"):
        raise ValueError(
            f"MODEL_REPR_VERSION is {va.get('MODEL_REPR_VERSION')!r} in "
            f"{a['path']} and {vb.get('MODEL_REPR_VERSION')!r} in {b['path']}; "
            "convert the version-1 side with `python -m mantisnet.klent.graft`, "
            "which rewrites it into version 2 while preserving the function "
            "exactly — that exactness is what makes a cross-representation "
            "comparison legitimate at all"
        )


def sign_test(a_wins: int, decisive: int) -> float:
    """The exact two-sided binomial p at p = ½ over ``decisive`` pairs.

    Exact rather than normal-approximated because the interesting matches are the
    close ones, where the tail the approximation gets wrong is the whole answer.
    No decisive pair is no evidence, which is ``p = 1``.
    """
    if decisive < 0 or not 0 <= a_wins <= decisive:
        raise ValueError(f"a_wins must be within 0..decisive, got {a_wins}/{decisive}")
    if decisive == 0:
        return 1.0
    extreme = max(a_wins, decisive - a_wins)
    tail = sum(math.comb(decisive, j) for j in range(extreme, decisive + 1))
    return min(1.0, 2 * tail / 2**decisive)


def paired_statistics(pairs: list[list[dict]]) -> dict:
    """Summarise ``pairs``, each the two ``play_match`` rows of one seat pair.

    A's score and its Wilson interval are the marginal figures, comparable with
    any SealBot evaluation. Everything else is paired: ``d`` is A's wins in the
    pair minus one, ``paired_se`` is the standard error of ``d``'s mean, and
    ``unpaired_se`` is the standard error of the *same* estimand computed as if
    the ``2 * pairs`` games were independent. A's score is ``(1 + mean d) / 2``,
    so its own standard error is half of either.
    """
    if not pairs:
        raise ValueError("paired statistics need at least one pair")
    wrong = [i for i, pair in enumerate(pairs) if len(pair) != 2]
    if wrong:
        raise ValueError(f"pairs {wrong} do not hold exactly two games each")

    k = len(pairs)
    games = [row for pair in pairs for row in pair]
    scores = np.array([row["score_a"] for row in games], dtype=float)
    d = np.array(
        [pair[0]["score_a"] + pair[1]["score_a"] - 1.0 for pair in pairs], dtype=float
    )
    a_wins = float(scores.sum())
    n = 2 * k
    score = a_wins / n

    counts = {"a_both": 0, "split": 0, "b_both": 0, "capped": 0}
    for pair, difference in zip(pairs, d, strict=True):
        if pair[0]["capped"] or pair[1]["capped"]:
            counts["capped"] += 1
        elif difference > 0:
            counts["a_both"] += 1
        elif difference < 0:
            counts["b_both"] += 1
        else:
            counts["split"] += 1
    decisive = counts["a_both"] + counts["b_both"]

    # One pair has no dispersion to estimate; the sample standard deviation is
    # undefined rather than zero, and is reported as absent.
    d_sd = float(d.std(ddof=1)) if k > 1 else None
    paired_se = None if d_sd is None else d_sd / math.sqrt(k)
    unpaired_se = 2.0 * float(scores.std(ddof=1)) / math.sqrt(n)
    # Pairs that all carry the same d have a zero sample spread, which is no
    # estimate of the pairing's variance rather than a variance of zero: there is
    # nothing to divide by and nothing to build an interval from. An all-splits
    # match is exactly that shape, and is the ordinary outcome of two close
    # checkpoints whose seat advantage decides every game.
    se_ratio = None if not paired_se else unpaired_se / paired_se

    score_se = None if not paired_se else paired_se / 2.0
    capped = int(sum(row["capped"] for row in games))
    warnings = []
    if capped:
        warnings.append(
            f"{capped} of {n} games hit the ply cap and scored one half each: a "
            f"cap is not a decision, so {counts['capped']} pair(s) are counted apart "
            "from the win/split/loss counts and excluded from the sign test, and "
            "their d is not in {-1, 0, +1}"
        )
    if k < 2:
        warnings.append(
            "one pair has no paired standard error and no Elo interval; the "
            "paired design needs at least two pairs to measure its own spread"
        )
    if d_sd == 0.0:
        warnings.append(
            f"all {k} pairs have the same d = {float(d[0]):+.2f}, so the sample "
            "has no spread and the paired variance is unestimable: there is no "
            "Elo interval and no SE ratio, and the marginal Wilson interval is "
            "the only bound"
        )

    ci_lo, ci_hi = wilson(a_wins, n, Z)
    return {
        "pairs": k,
        "games": n,
        "a_wins": a_wins,
        "score": score,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "score_as_p0": float(
            sum(row["score_a"] for row in games if row["seat"] == 0)
        ),
        "score_as_p1": float(
            sum(row["score_a"] for row in games if row["seat"] == 1)
        ),
        "capped": capped,
        "pair_counts": counts,
        "decisive_pairs": decisive,
        "d_mean": float(d.mean()),
        "d_sd": d_sd,
        "paired_se": paired_se,
        "unpaired_se": unpaired_se,
        "se_ratio": se_ratio,
        "sign_test_p": sign_test(counts["a_both"], decisive),
        "elo": _finite(elo(score)),
        "elo_lo": None
        if score_se is None
        else _finite(elo(max(0.0, score - Z * score_se))),
        "elo_hi": None
        if score_se is None
        else _finite(elo(min(1.0, score + Z * score_se))),
        "warnings": warnings,
    }


def driver_match(
    model,
    ref_model,
    ref_name: str,
    ref_config: dict,
    *,
    cfg: KlentConfig,
    pairs: int,
    sims: int,
    tau: float,
    lam: float,
    ply_cap: int,
    rng: np.random.Generator,
    temperature: float = 1.0,
) -> tuple[dict, list[dict], dict]:
    """The in-driver paired match: the live model against a fixed reference.

    This is :func:`head_to_head`'s instrument on the training loop's cadence.
    Both models are already in memory, so there is no checkpoint audit here —
    the caller identifies the reference; ``ref_config`` is its strength-defining
    record and travels into the ``opponents`` table unchanged.

    One schedule of ``pairs`` shared openings plays as a single batched match,
    so each lockstep step forwards every live game of a side at once rather
    than two at a time — the offline tool optimizes per-pair reproducibility,
    this one optimizes the GPU it is borrowing from training. Pairing is by
    schedule structure (games ``2i`` and ``2i + 1`` share pair ``i``), which
    one generator for the whole match does not disturb.

    Returns ``(result, per_game, stats)``: the first two shaped exactly for
    ``sealbot.record_match`` — the summary carries ``paired_statistics``' Elo,
    whose interval comes from the paired standard error — and ``stats`` is the
    full paired summary for the caller's own record.
    """
    if pairs < 2:
        raise ValueError(
            f"a driver match needs pairs >= 2 to estimate its paired spread, got {pairs}"
        )
    # Eager on both sides: the compiled callable is shared process-wide and
    # keyed on the training model; evaluating a second architecture through it
    # would recompile under the training loop.
    eval_cfg = dataclasses.replace(cfg, compile=False)
    choose_live, choose_ref = (
        gumbel_choose(
            network_evaluate(net, eval_cfg),
            tau=tau,
            lam=lam,
            sims=sims,
            temperature=temperature,
        )
        for net in (model, ref_model)
    )

    started = time.monotonic()
    schedule = shared_openings(rng, pairs)
    _summary, rows = play_match(choose_live, choose_ref, schedule, ply_cap, rng)
    stats = paired_statistics([rows[k : k + 2] for k in range(0, len(rows), 2)])

    per_game = [
        {
            "seat": row["seat"],
            "winner": row["winner"],
            "capped": row["capped"],
            "forfeit": False,
            "score": row["score_a"],
            "opening_len": len(row["opening"]),
            "depth_mean": None,
            "moves": row["moves"],
        }
        for row in rows
    ]
    result = {
        "score": stats["a_wins"],
        "games": stats["games"],
        "capped": stats["capped"],
        "win_rate": stats["score"],
        "ci_lo": stats["ci_lo"],
        "ci_hi": stats["ci_hi"],
        "elo": stats["elo"],
        "elo_lo": stats["elo_lo"],
        "elo_hi": stats["elo_hi"],
        "score_as_p0": stats["score_as_p0"],
        "score_as_p1": stats["score_as_p1"],
        "forfeits": 0,
        "opponent_name": ref_name,
        "opponent_config": ref_config,
        "opponent_depth_mean": None,
        "avg_plies": sum(len(row["moves"]) for row in rows) / len(rows),
        "seconds": time.monotonic() - started,
    }
    return result, per_game, stats


def head_to_head(
    path_a: Path,
    path_b: Path,
    *,
    pairs: int,
    sims: int,
    tau: float | None,
    lam: float | None,
    ply_cap: int = 512,
    device: str = "cpu",
    seed: int = 0,
    opening_range: tuple[int, int] = (2, 6),
    temperature: float = 1.0,
) -> dict:
    """Play A against B over ``pairs`` seat-swapped pairs and summarise them.

    ``tau`` and ``lam`` are the coefficients both models search at, required when
    ``sims > 0`` and ``None`` when it is zero, where the operator is never
    consulted — the manifest then records that no operating point was used
    instead of naming one that was not.

    ``temperature`` is the root Gumbel scale of :func:`gumbel_choose`, applied
    to both seats because an asymmetric one would measure the difference
    between two search settings as though it were a difference between two
    models. It is recorded with the result: a score at one temperature says
    nothing about a score at another.

    Returns the statistics of :func:`paired_statistics`, an audit record per
    checkpoint, one row per pair, and the match's own parameters — enough for
    someone else to reproduce the match or to say why they cannot.
    """
    if pairs < 1:
        raise ValueError(f"pairs must be >= 1, got {pairs}")
    if sims < 0:
        raise ValueError(f"sims must be >= 0, got {sims}")
    if sims > 0 and (tau is None or lam is None):
        raise ValueError(
            "a searched match acts by the improvement operator on every interior "
            f"step and needs both coefficients: tau={tau}, lam={lam}"
        )
    if ply_cap <= opening_range[1]:
        raise ValueError(
            f"ply_cap must exceed the longest opening ({opening_range[1]}) for "
            f"both models to move at all, got {ply_cap}"
        )
    path_a, path_b = Path(path_a), Path(path_b)
    audit_a, audit_b = _audit(path_a, "--a"), _audit(path_b, "--b")
    _refuse_incomparable(audit_a, audit_b)

    # A pair's generator draws its own opening and then drives both its games, so
    # the pair is a function of ``(seed, pair index)`` alone. Drawing every
    # schedule up front refuses an unusable opening range before any model loads.
    generators = [np.random.default_rng([seed, index]) for index in range(pairs)]
    schedules = [shared_openings(rng, 1, opening_range) for rng in generators]

    started = time.monotonic()
    # The evaluator reads only the device fields; the coefficients act through
    # ``gumbel_choose``, which does not consult them at ``sims == 0``.
    cfg = KlentConfig(device=device, autocast=device == "cuda", compile=False)
    choose_a, choose_b = [
        gumbel_choose(
            network_evaluate(load_model(path, device), cfg),
            tau=tau,
            lam=lam,
            sims=sims,
            temperature=temperature,
        )
        for path in (path_a, path_b)
    ]

    pair_rows, per_pair = [], []
    for index, (schedule, rng) in enumerate(zip(schedules, generators, strict=True)):
        _summary, rows = play_match(choose_a, choose_b, schedule, ply_cap, rng)
        pair_rows.append(rows)
        per_pair.append(
            {
                "pair": index,
                "opening": rows[0]["opening"],
                "seats": [row["seat"] for row in rows],
                "scores": [row["score_a"] for row in rows],
                "winners": [row["winner"] for row in rows],
                "capped": [row["capped"] for row in rows],
                "plies": [len(row["moves"]) for row in rows],
                "d": rows[0]["score_a"] + rows[1]["score_a"] - 1.0,
            }
        )

    return {
        "a": audit_a,
        "b": audit_b,
        **paired_statistics(pair_rows),
        "per_pair": per_pair,
        "match": {
            "pairs": pairs,
            "games": 2 * pairs,
            "sims": sims,
            "tau": tau,
            "lam": lam,
            "temperature": temperature,
            "opening_range": list(opening_range),
            "ply_cap": ply_cap,
            "device": device,
            "seed": seed,
            "seconds": time.monotonic() - started,
        },
    }


def _number(value: float | None, spec: str) -> str:
    """A figure the paired design can legitimately not have, formatted or absent."""
    return "n/a" if value is None else format(value, spec)


def _fmt(result: dict) -> str:
    counts = result["pair_counts"]
    return "\n".join(
        (
            f"A {Path(result['a']['path']).name} @ {result['a']['iteration']}  vs  "
            f"B {Path(result['b']['path']).name} @ {result['b']['iteration']}",
            f"A scores {result['a_wins']:.1f}/{result['games']} = "
            f"{100 * result['score']:.1f}% "
            f"(Wilson {100 * result['ci_lo']:.1f}-{100 * result['ci_hi']:.1f}%) "
            f"| P0 {result['score_as_p0']:.1f} P1 {result['score_as_p1']:.1f} "
            f"| capped {result['capped']}",
            f"{result['pairs']} pairs: A both {counts['a_both']}, split "
            f"{counts['split']}, B both {counts['b_both']}, capped "
            f"{counts['capped']} | sign test p = {result['sign_test_p']:.4f}",
            f"d = {result['d_mean']:+.4f} | SE paired "
            f"{_number(result['paired_se'], '.4f')} vs unpaired "
            f"{_number(result['unpaired_se'], '.4f')} "
            f"(ratio {_number(result['se_ratio'], '.2f')})",
            f"elo {_number(result['elo'], '+.0f')} "
            f"({_number(result['elo_lo'], '+.0f')}.."
            f"{_number(result['elo_hi'], '+.0f')}, from the paired SE) "
            f"| sims {result['match']['sims']} T {result['match']['temperature']:g} "
            f"| {result['match']['seconds']:.0f}s",
        )
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--a", type=Path, required=True, metavar="A.pt")
    parser.add_argument("--b", type=Path, required=True, metavar="B.pt")
    parser.add_argument(
        "--pairs", type=int, required=True, help="openings, each played from both seats"
    )
    parser.add_argument(
        "--sims",
        type=int,
        required=True,
        help="line-search simulations for both models (0 = policy argmax)",
    )
    parser.add_argument(
        "--out", type=Path, required=True, metavar="OUT.json", help="where the result goes"
    )
    parser.add_argument(
        "--cap",
        type=int,
        default=512,
        help="ply cap, opening included; must exceed the longest opening",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--opening-range",
        type=int,
        nargs=2,
        metavar=("LO", "HI"),
        default=(2, 6),
        help="inclusive opening length in placements (default: 2 6)",
    )
    parser.add_argument("--tau", type=float, help="reverse-KL weight, required with --sims")
    parser.add_argument("--lam", type=float, help="entropy weight, required with --sims")
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="root Gumbel scale for both seats; 0 is deterministic, 1 is Gumbel "
        "MuZero's (default: 1.0)",
    )
    args = parser.parse_args(argv)

    # At sims = 0 the operator is never consulted and the coefficients stay
    # absent, in the manifest too: naming one would claim an operating point the
    # match did not use. A searched match uses them on every interior step, and
    # the compared runs' own values are the only correct ones — there is no
    # default that is right for two runs.
    if args.sims > 0 and (args.tau is None or args.lam is None):
        parser.error(
            "--sims > 0 searches through the improvement operator: pass --tau and "
            "--lam, the KLENT coefficients the compared runs were trained at"
        )
    # Checked before the match, because the match is hours long and its only
    # durable output is this one file.
    if not args.out.parent.is_dir():
        parser.error(f"--out: no directory {args.out.parent} to write into")
    if args.out.is_dir():
        parser.error(f"--out: {args.out} is a directory")
    result = head_to_head(
        args.a,
        args.b,
        pairs=args.pairs,
        sims=args.sims,
        tau=args.tau,
        lam=args.lam,
        ply_cap=args.cap,
        device=args.device,
        seed=args.seed,
        opening_range=tuple(args.opening_range),
        temperature=args.temperature,
    )
    # Write then rename, so an interrupt cannot leave a truncated result behind.
    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    tmp.replace(args.out)
    print(_fmt(result))
    for warning in result["warnings"]:
        print(f"WARNING: {warning}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
