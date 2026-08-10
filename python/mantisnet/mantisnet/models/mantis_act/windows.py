"""Persistent-window schema and geometry helpers.

Production window enumeration and encoding live in Rust's MantisNet-ACT
builder.  This module retains only the representation constants consumed by
the model and the identity-to-cell expansion used by diagnostics and tests.
"""

from __future__ import annotations

import numpy as np

from .symmetry import AXES

WINDOW_LEN = 6
NUM_AXES = len(AXES)

# §9.3's numeric fields, in packed-column order.
WINDOW_NUMERIC_NAMES = (
    "own_count",
    "opp_count",
    "empty_count",
    "own_max_run",
    "opp_max_run",
)
WINDOW_NUMERIC_FEATURES = len(WINDOW_NUMERIC_NAMES)

# §16's device-side window-pair key packs a line offset into a 17-bit field.
# A sum of two coordinates must stay inside 2**16, so each coordinate stays in
# the engine's signed-i16 range.
WINDOW_COORD_LIMIT = 1 << 15

_SLOTS = np.arange(WINDOW_LEN, dtype=np.int64)


def window_cells(window_id: np.ndarray) -> np.ndarray:
    """Expand ``(axis, start_q, start_r)`` rows to ``(n, 6, 2)`` cells."""
    window_id = np.asarray(window_id, dtype=np.int64).reshape(-1, 3)
    axis = window_id[:, 0]
    if window_id.size and (int(axis.min()) < 0 or int(axis.max()) >= NUM_AXES):
        bad = int(np.argmax((axis < 0) | (axis >= NUM_AXES)))
        raise ValueError(
            f"window {bad} names native axis {int(axis[bad])}, not one of "
            f"0..{NUM_AXES - 1}"
        )
    return window_id[:, None, 1:] + AXES[axis][:, None, :] * _SLOTS[None, :, None]


__all__ = [
    "NUM_AXES",
    "WINDOW_COORD_LIMIT",
    "WINDOW_LEN",
    "WINDOW_NUMERIC_FEATURES",
    "WINDOW_NUMERIC_NAMES",
    "window_cells",
]
