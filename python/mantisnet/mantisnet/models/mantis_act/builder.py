"""MantisNet-ACT builder: engine position to Rust-built graph or packed batch."""

from __future__ import annotations

from collections.abc import Sequence

from .config import MantisACTConfig
from .packed import ACTGraph, PackedACTBatch, packed_from_arrays

# §13.3's state-derived scalars, in packed-column order.
GLOBAL_NUMERIC_NAMES: tuple[str, ...] = (
    "log1p_stones",
    "own_stone_fraction",
    "opponent_stone_fraction",
    "log1p_legal_count",
    "log1p_window_count",
    "own_live_window_fraction",
    "opponent_live_window_fraction",
    "mixed_window_fraction",
)
GLOBAL_NUMERIC_FEATURES = len(GLOBAL_NUMERIC_NAMES)

# The Rust boundary accepts exactly the fields that change builder output.
_RUST_CONFIG_FIELDS: tuple[str, ...] = (
    "window_scope",
    "cell_scope",
    "d6_relation_mode",
    "d_max",
    "occupied_radius",
    "use_cell_adjacency",
    "use_occupied_radius_edges",
    "use_global_numeric_features",
    "use_window_numeric_features",
    "use_action_tactical_features",
)


def _rust_config(cfg: MantisACTConfig) -> dict[str, object]:
    """The closed builder-only config consumed by ``hexo_py``."""
    return {name: getattr(cfg, name) for name in _RUST_CONFIG_FIELDS}


def _graph_from_rust(arrays: dict[str, object]) -> ACTGraph:
    """Run the public per-position validation over one Rust result."""
    return ACTGraph(**arrays)


def build(position, cfg: MantisACTConfig) -> ACTGraph:
    """Build and validate one graph from an authoritative engine position."""
    import hexo_py

    return _graph_from_rust(hexo_py.build_act_graph(position, _rust_config(cfg)))


def collate_positions(positions: Sequence, cfg: MantisACTConfig) -> PackedACTBatch:
    """Build and collate positions in parallel in Rust."""
    import hexo_py

    fields = hexo_py.build_act_batch(list(positions), _rust_config(cfg))
    return packed_from_arrays(fields, cfg)


def collate_prefixes(
    games: Sequence[Sequence[tuple[int, int]]],
    ts: Sequence[int],
    cfg: MantisACTConfig,
) -> PackedACTBatch:
    """Replay and Rust-collate ``games[i][:ts[i]]`` in input order."""
    import hexo_py

    games, ts = list(games), list(ts)
    if len(games) != len(ts):
        raise ValueError(f"{len(games)} games against {len(ts)} prefix lengths")
    normalized_games = []
    normalized_ts = []
    for index, (moves, t) in enumerate(zip(games, ts)):
        t = int(t)
        move_count = len(moves)
        if not 0 <= t <= move_count:
            raise ValueError(
                f"prefix {index} asks for {t} moves of a {move_count}-move game"
            )
        # Suffix moves are outside the prefix and therefore are not parsed.
        normalized_games.append([tuple(moves[i]) for i in range(t)])
        normalized_ts.append(t)
    fields = hexo_py.build_act_batch_prefixes(
        normalized_games, normalized_ts, _rust_config(cfg)
    )
    return packed_from_arrays(fields, cfg)


__all__ = [
    "GLOBAL_NUMERIC_FEATURES",
    "GLOBAL_NUMERIC_NAMES",
    "build",
    "collate_positions",
    "collate_prefixes",
]
