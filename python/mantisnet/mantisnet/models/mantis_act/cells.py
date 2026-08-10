"""Cell occupancy vocabulary for the Rust-built MantisNet-ACT graph."""

from __future__ import annotations

# §8.2 occupancy values relative to the side to move.
OCCUPANCY_EMPTY, OCCUPANCY_OWN, OCCUPANCY_OPP = 0, 1, 2

__all__ = ["OCCUPANCY_EMPTY", "OCCUPANCY_OWN", "OCCUPANCY_OPP"]
