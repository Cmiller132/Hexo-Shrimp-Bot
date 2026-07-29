"""The mover-change sign and λ-return from ``KLENT_FOR_HEXO.md`` §1.2–§1.3.

The sign follows mover change rather than ply parity. It is read from the
acted-on position's phase:
``+1`` exactly at a ``FirstStone`` ply, where the same mover places again.
``moves_remaining`` carries the phase here: 2 is ``FirstStone``; 1 is
``Opening`` or ``SecondStone``, both of which hand the turn over.

The recursion consumes completed, won episodes; capped episodes are excluded
by the caller:

    G_T = +1
    G_t = s_t · γ · [ (1 − λ)·v̂_{t+1} + λ·G_{t+1} ]        t < T

γ is the per-ply discount magnitude; ``s_t`` carries the mover-change sign.
At γ = 1, outcome timing does not change return magnitude. Values below one
give earlier outcomes larger magnitude.
"""

from __future__ import annotations

import numpy as np


def signs_from_moves_remaining(moves_remaining) -> np.ndarray:
    """``+1`` where the mover keeps the turn (moves_remaining == 2), else ``-1``."""
    mr = np.asarray(moves_remaining, dtype=np.int64)
    if not np.isin(mr, (1, 2)).all():
        raise ValueError("moves_remaining must be 1 or 2 everywhere")
    return np.where(mr == 2, 1, -1).astype(np.float64)


def lambda_returns(signs: np.ndarray, v_hats, lam_ret: float, gamma: float) -> np.ndarray:
    """Per-ply returns ``G_0..G_T`` of a won episode, each in its mover's frame.

    ``signs`` and ``v_hats`` cover plies ``0..T`` — the acting-time values of
    ``KLENT_FOR_HEXO.md`` §1.4 K6; ``v_hats[t]`` is `v̂_t` and only entries ``1..T`` are read.
    ``signs[T]`` is likewise never read: the recursion terminates at the win,
    and a terminal position's frozen phase must answer nothing
    (``KLENT_FOR_HEXO.md`` §1.1).
    """
    if not 0.0 <= lam_ret <= 1.0:
        raise ValueError(f"lam_ret must lie in [0, 1], got {lam_ret}")
    if not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma must lie in (0, 1], got {gamma}")
    v = np.asarray(v_hats, dtype=np.float64)
    if v.shape != np.shape(signs) or v.ndim != 1 or len(v) == 0:
        raise ValueError("signs and v_hats must be equal-length, nonempty 1-d arrays")
    if not (np.isfinite(v).all() and np.abs(v).max() <= 1.0):
        raise ValueError("v_hats must be finite and within [-1, 1]")
    g = np.empty_like(v)
    g[-1] = 1.0
    for t in range(len(v) - 2, -1, -1):
        g[t] = signs[t] * gamma * ((1.0 - lam_ret) * v[t + 1] + lam_ret * g[t + 1])
    # Bounded inputs and gamma <= 1 bound G by construction; leaving the
    # interval means the recursion or its inputs changed incompatibly.
    if np.abs(g).max() > 1.0:
        raise ValueError("lambda-return left [-1, 1]; recursion inputs are corrupt")
    return g
