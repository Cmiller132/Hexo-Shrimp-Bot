"""Action-feature schema for the Rust-built MantisNet-ACT graph."""

from __future__ import annotations

# §19.3's deterministic tactical fields, in packed-column order.
TACTICAL_FEATURE_NAMES: tuple[str, ...] = (
    "immediate_win",
    "max_own_count_after",
    "max_opponent_count_before",
    "own_five_windows_after",
    "own_four_windows_after",
    "opponent_five_windows_hit",
    "opponent_four_windows_hit",
    "opponent_five_windows_remaining",
    "opponent_four_windows_remaining",
    "blocks_all_immediate_threats",
    "mixed_windows_created",
    "nonempty_pre_windows",
)
TACTICAL_FEATURES = len(TACTICAL_FEATURE_NAMES)

__all__ = ["TACTICAL_FEATURE_NAMES", "TACTICAL_FEATURES"]
