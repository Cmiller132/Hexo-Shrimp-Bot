"""MantisNet-ACT builder: position to graph, batch to packed batch.

Orchestrates ``windows``, ``cells``, and ``actions``, adding phase (§13.1),
global scalars (§13.3), and ``moves_remaining``. Stone ownership is converted
here: ``stone_own`` is ``0`` for the mover, ``1`` for the opponent.

Config flags decide whether each edge family and feature block exists.
Terminal positions are refused (§2).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .actions import action_tables, tactical_features
from .cells import adjacency_edges, incidence, radius_edges, relevant_cells
from .config import MantisACTConfig
from .packed import (
    PHASE_FIRST,
    PHASE_OPENING,
    PHASE_SECOND,
    ACTGraph,
    PackedACTBatch,
    collate,
)
from .pattern_classes import MIXED, OPP_LIVE, OWN_LIVE
from .windows import enumerate_windows

# §13.3's state-derived scalars, in the order the spec lists them. Every entry
# is a function of the current position alone: no history, no recency, no
# absolute move number, and no board origin (§38).
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


def _empty_edges(columns: int) -> tuple[np.ndarray, ...]:
    """An edge family the configuration disables, as its empty columns."""
    return tuple(np.empty(0, dtype=np.int64) for _ in range(columns))


def _fraction(count: int, total: int) -> float:
    """A share of a total, and zero when there is nothing to share.

    The stoneless opening and the windowless early board both make the
    denominator zero; the count itself is carried separately.
    """
    return count / total if total else 0.0


def _phase_id(n_stones: int, moves_remaining: int) -> int:
    """The three-way placement phase of §13.1."""
    if moves_remaining == 2:
        return PHASE_FIRST
    return PHASE_OPENING if n_stones == 0 else PHASE_SECOND


def _global_numeric(
    window_status: np.ndarray,
    n_stones: int,
    n_own: int,
    n_legal: int,
    cfg: MantisACTConfig,
) -> np.ndarray:
    """The §13.3 scalars of one position, or a zero-width vector when disabled."""
    if not cfg.use_global_numeric_features:
        return np.zeros(0, dtype=np.float32)
    n_windows = len(window_status)
    return np.array(
        (
            np.log1p(n_stones),
            _fraction(n_own, n_stones),
            _fraction(n_stones - n_own, n_stones),
            np.log1p(n_legal),
            np.log1p(n_windows),
            _fraction(int((window_status == OWN_LIVE).sum()), n_windows),
            _fraction(int((window_status == OPP_LIVE).sum()), n_windows),
            _fraction(int((window_status == MIXED).sum()), n_windows),
        ),
        dtype=np.float32,
    )


def build_from_arrays(
    stone_qr,
    stone_owner,
    mover: int,
    legal_qr,
    moves_remaining: int,
    cfg: MantisACTConfig,
) -> ACTGraph:
    """Build one position's graph from the stone, seat, and legal-move lists.

    ``stone_qr`` is ``(n_stones, 2)`` and ``stone_owner`` ``(n_stones,)`` in the
    engine's absolute seats; ``mover`` is the side to move, against which the
    seats become the mover-relative colours every table downstream carries.
    ``legal_qr`` is ``(n_legal, 2)`` in engine legal order, preserved and never
    sorted: output row ``j`` is ``legal_moves[j]`` (§8.3).

    Raises ``ValueError`` for a terminal position, for a ``mover`` or seat that
    is not a player, for a ``moves_remaining`` outside ``1..2``, and for every
    malformed input the stages below name themselves. The graph is validated
    before it is returned.
    """
    stone_qr = np.asarray(stone_qr, dtype=np.int64).reshape(-1, 2)
    stone_owner = np.asarray(stone_owner, dtype=np.int64).reshape(-1)
    legal_qr = np.asarray(legal_qr, dtype=np.int64).reshape(-1, 2)
    if len(stone_owner) != len(stone_qr):
        raise ValueError(
            f"{len(stone_qr)} stone coordinates against {len(stone_owner)} owners"
        )
    if int(mover) not in (0, 1):
        raise ValueError(f"mover must be player 0 or 1, got {mover!r}")
    foreign = np.flatnonzero((stone_owner != 0) & (stone_owner != 1))
    if foreign.size:
        bad = int(foreign[0])
        raise ValueError(
            f"stone_owner[{bad}] = {int(stone_owner[bad])} is neither player 0 nor 1"
        )
    if len(legal_qr) == 0:
        raise ValueError("terminal position: the builder refuses it")
    if int(moves_remaining) not in (1, 2):
        raise ValueError(f"moves_remaining must be 1 or 2, got {moves_remaining!r}")

    moves_remaining = int(moves_remaining)
    stone_own = (stone_owner != int(mover)).astype(np.int64)
    phase_id = _phase_id(len(stone_qr), moves_remaining)

    window_set = enumerate_windows(stone_qr, stone_own, legal_qr, cfg)
    cell_set = relevant_cells(
        stone_qr, stone_own, legal_qr, window_set.cell_coords(), cfg
    )
    window_cell_index, window_incidence_class, window_incidence_mask = incidence(
        window_set, cell_set
    )
    adjacency_src, adjacency_dst, adjacency_axis = (
        adjacency_edges(cell_set) if cfg.use_cell_adjacency else _empty_edges(3)
    )
    radius_src, radius_dst, radius_orbit, radius_axis = (
        radius_edges(cell_set, stone_qr, stone_own, cfg)
        if cfg.use_occupied_radius_edges
        else _empty_edges(4)
    )
    tables = action_tables(window_set, stone_qr, stone_own, legal_qr, cfg)

    window_numeric = (
        window_set.numeric
        if cfg.use_window_numeric_features
        else np.zeros((window_set.n_windows, 0), dtype=np.float32)
    )
    return ACTGraph(
        cell_qr=cell_set.qr,
        cell_occupancy=cell_set.occupancy,
        cell_is_legal=cell_set.is_legal,
        cell_is_occupied=cell_set.is_occupied,
        cell_nearest_bucket=cell_set.nearest_bucket,
        legal_to_cell_index=cell_set.legal_to_cell_index,
        window_id=window_set.window_id,
        window_pattern_class=window_set.pattern_class,
        window_status=window_set.status,
        window_axis=window_set.axis,
        window_numeric=window_numeric,
        window_cell_index=window_cell_index,
        window_incidence_class=window_incidence_class,
        window_incidence_mask=window_incidence_mask,
        adjacency_src=adjacency_src,
        adjacency_dst=adjacency_dst,
        adjacency_axis=adjacency_axis,
        radius_src=radius_src,
        radius_dst=radius_dst,
        radius_orbit=radius_orbit,
        radius_axis_or_neg1=radius_axis,
        action_window_index=tables.action_window_index,
        action_post1_class=tables.action_post1_class,
        action_pre_status=tables.action_pre_status,
        action_tactical_numeric=tactical_features(tables, cfg),
        global_numeric=_global_numeric(
            window_set.status,
            len(stone_qr),
            int((stone_own == 0).sum()),
            len(legal_qr),
            cfg,
        ),
        moves_remaining=moves_remaining,
        phase_id=phase_id,
    )


def build(position, cfg: MantisACTConfig) -> ACTGraph:
    """Build one graph from a ``hexo_py.Position``. Terminal positions raise.

    Reads only the position's stone list, legal-move list, side to move, and
    placements left in the turn — nothing else, so the engine remains the
    independent oracle the stages below are tested against (§30).
    """
    if position.is_terminal:
        raise ValueError("terminal position: the builder refuses it")
    stones = np.asarray(position.stones(), dtype=np.int64).reshape(-1, 3)
    legal = np.asarray(position.legal_moves(), dtype=np.int64).reshape(-1, 2)
    return build_from_arrays(
        stones[:, :2],
        stones[:, 2],
        position.current_player,
        legal,
        position.moves_remaining,
        cfg,
    )


def collate_positions(positions: Sequence, cfg: MantisACTConfig) -> PackedACTBatch:
    """Build a sequence of ``hexo_py.Position`` objects into one packed batch."""
    return collate([build(position, cfg) for position in positions])


def collate_prefixes(
    games: Sequence[Sequence[tuple[int, int]]],
    ts: Sequence[int],
    cfg: MantisACTConfig,
) -> PackedACTBatch:
    """Replay ``games[i][:ts[i]]`` and collate the resulting positions.

    Stored fitting positions are move prefixes (``docs/KLENT_FOR_HEXO.md``
    §4.3). A length past the end of its game raises; a prefix whose position
    is terminal is refused by :func:`build`.
    """
    import hexo_py

    games, ts = list(games), list(ts)
    if len(games) != len(ts):
        raise ValueError(f"{len(games)} games against {len(ts)} prefix lengths")
    graphs = []
    for i, (moves, t) in enumerate(zip(games, ts)):
        moves, t = [tuple(move) for move in moves], int(t)
        if not 0 <= t <= len(moves):
            raise ValueError(
                f"prefix {i} asks for {t} moves of a {len(moves)}-move game"
            )
        graphs.append(build(hexo_py.Position.replay(moves[:t]), cfg))
    return collate(graphs)
