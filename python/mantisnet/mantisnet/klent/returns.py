"""Mover-change signs and lambda-returns (``KLENT_FOR_HEXO.md`` section 1.2-1.3).

    G_T = +1
    G_t = s_t * gamma * [ (1 - lam) * v_hat_{t+1} + lam * G_{t+1} ]
"""

from __future__ import annotations

import numpy as np

# Tolerance for fp32 segment-softmax summation landing slightly outside [-1, 1].
_RANGE_SLACK = 1e-4


def _refuse_unbounded(x: np.ndarray, name: str) -> None:
    """Refuse a non-finite or out-of-range array, naming the entry that failed."""
    finite = np.isfinite(x)
    excess = np.where(finite, np.abs(x) - 1.0, np.inf)
    if finite.all() and excess.max() <= _RANGE_SLACK:
        return
    worst = int(np.argmax(excess))
    raise ValueError(
        f"{name} must be finite and within [-1, 1]: {name}[{worst}] = {float(x[worst])!r} "
        f"of {x.size} entries, {int((~finite).sum())} non-finite and "
        f"{int((excess[finite] > _RANGE_SLACK).sum())} outside the interval by more "
        f"than the {_RANGE_SLACK} fp32 summation slack"
    )


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
    _refuse_unbounded(v, "v_hats")
    g = np.empty_like(v)
    g[-1] = 1.0
    for t in range(len(v) - 2, -1, -1):
        g[t] = signs[t] * gamma * ((1.0 - lam_ret) * v[t + 1] + lam_ret * g[t + 1])
    # Final range check: G is bounded by the same slack as the inputs.
    _refuse_unbounded(g, "the lambda-return")
    return g
