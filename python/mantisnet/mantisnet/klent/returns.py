"""The sign function and the λ-return (design doc §4.3, §4.4).

The sign follows mover *change*, not ply parity — K1, the design doc's "most
likely catastrophic bug". It is read off the phase of the acted-on position:
``+1`` exactly at a ``FirstStone`` ply, where the same mover places again.
``moves_remaining`` carries the phase here: 2 is ``FirstStone``; 1 is
``Opening`` or ``SecondStone``, both of which hand the turn over.

The recursion runs backward over a completed, won episode only — capped
episodes are dropped whole before this module ever sees them (§5.1) — so
there is exactly one case:

    G_T = +1
    G_t = s_t · [ (1 − λ)·v̂_{t+1} + λ·G_{t+1} ]        t < T
"""

from __future__ import annotations

import numpy as np


def signs_from_moves_remaining(moves_remaining) -> np.ndarray:
    """``+1`` where the mover keeps the turn (moves_remaining == 2), else ``-1``."""
    mr = np.asarray(moves_remaining, dtype=np.int64)
    if not np.isin(mr, (1, 2)).all():
        raise ValueError("moves_remaining must be 1 or 2 everywhere")
    return np.where(mr == 2, 1, -1).astype(np.float64)


def lambda_returns(signs: np.ndarray, v_hats, lam_ret: float) -> np.ndarray:
    """Per-ply returns ``G_0..G_T`` of a won episode, each in its mover's frame.

    ``signs`` and ``v_hats`` cover plies ``0..T`` — the acting-time values of
    design doc K6; ``v_hats[t]`` is `v̂_t` and only entries ``1..T`` are read.
    ``signs[T]`` is likewise never read: the recursion terminates at the win,
    and a terminal position's frozen phase must answer nothing (§4.3).
    """
    if not 0.0 <= lam_ret <= 1.0:
        raise ValueError(f"lam_ret must lie in [0, 1], got {lam_ret}")
    v = np.asarray(v_hats, dtype=np.float64)
    if v.shape != np.shape(signs) or v.ndim != 1 or len(v) == 0:
        raise ValueError("signs and v_hats must be equal-length, nonempty 1-d arrays")
    g = np.empty_like(v)
    g[-1] = 1.0
    for t in range(len(v) - 2, -1, -1):
        g[t] = signs[t] * ((1.0 - lam_ret) * v[t + 1] + lam_ret * g[t + 1])
    return g
