"""The MantisNet-ACT builder: one position to a graph, many to a packed batch.

This module is orchestration and nothing else. Every table it returns comes
from one of the four representation modules — ``windows``, ``cells``,
``actions``, ``pairs`` — and its own work is the order they run in, the three
position-level quantities none of them owns (§13.1's phase, §13.3's global
scalars, and ``moves_remaining``), and the refusals that belong to the stage as
a whole rather than to any one part of it.

Index conventions this module fixes:

- Stone ownership: callers hand in the engine's absolute player ids together
  with the side to move, and every module downstream takes ``stone_own``, ``0``
  for the mover's stones and ``1`` for the opponent's. The conversion happens
  here, once, so no representation module ever sees an absolute seat and the
  colours cannot be relative to two different sides in two tables of one graph.
- Phase (§13.1): ``FIRST`` is ``moves_remaining == 2``; with one placement left
  the board decides, empty being ``OPENING`` and nonempty ``SECOND``.
  ``moves_remaining`` stays authoritative for the KLENT return sign, and the
  phase id is a model feature derived from it (§2).
- Global scalars: ``GLOBAL_NUMERIC_NAMES`` in that order (§13.3). Counts enter
  as ``log1p`` and shares as fractions of their own total, so no entry depends
  on the board's origin, on the move number, or on any history.

Which config fields this module reads, and which it leaves to the model.
``use_cell_adjacency`` and ``use_occupied_radius_edges`` decide whether an edge
family exists at all, so a disabled family is absent rather than emitted and
ignored. ``use_global_numeric_features`` and ``use_window_numeric_features``
decide whether their feature block has any width, which is how ``actions.py``
already treats the tactical vector: a disabled block contributes exactly
nothing rather than a learned constant on a column of zeros. Everything else
that names how a stage behaves rather than whether its rows exist —
``route_on_axis_radius_messages``, ``use_three_way_phase``,
``use_counterfactual_action_windows`` — gates a computation the model performs
over a table whose shape §25 fixes at ``[num_legal, 3, 6]`` or a per-edge
column, so the builder emits that table under every configuration and the
model decides what to read.

Terminal positions are refused, matching ``mantisnet/builder.py``'s contract:
a state with no legal move is never evaluated or bootstrapped (§2), so a
builder that encoded one would be feeding the trainer a state it must not see.
The refusal is stated here as well as inside the modules that can detect it
from their own inputs, because an empty legal list reaches this function before
any of them run, and a completed six-in-a-row reaches ``windows.py`` before
``actions.py`` can see it.
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
from .pairs import pair_rows
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

    The stoneless opening and the windowless early board both make a
    denominator zero. Zero is the honest reading — no stone is the mover's when
    there are no stones — and the count it divides is carried separately, so
    the pair cannot be mistaken for a real ratio of one.
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
    ``legal_qr`` is ``(n_legal, 2)`` in engine legal order, which is preserved
    and never sorted: output row ``j`` is ``legal_moves[j]`` (§8.3).

    Raises ``ValueError`` for a terminal position — an empty legal list here,
    or a completed six-in-a-row inside ``windows.py`` — for a ``mover`` or a
    seat that is not a player, for a ``moves_remaining`` outside ``1..2``, and
    for every malformed input the stages below name themselves. The graph is
    validated before it is returned, so a table that violates §7's ordering or
    an index bound fails here rather than at an embedding lookup.
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
    rows = pair_rows(
        window_set,
        cell_set,
        stone_qr,
        stone_own,
        legal_qr,
        moves_remaining,
        phase_id,
        cfg,
    )

    window_numeric = (
        window_set.numeric
        if cfg.use_window_numeric_features
        else np.zeros((window_set.n_windows, 0), dtype=np.float32)
    )
    graph = ACTGraph(
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
        **vars(rows),
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
    graph.validate()
    return graph


def build(position, cfg: MantisACTConfig) -> ACTGraph:
    """Build one graph from a ``hexo_py.Position``. Terminal positions raise.

    The engine is read for the position's contents and for nothing else: its
    stone list, its legal-move list in engine order, the side to move, and the
    placements left in the turn. No stage below asks it what a window holds,
    which cells are legal after a hypothetical placement, or whether a
    placement wins, so the engine remains the independent oracle those stages
    are tested against (§30).
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
    §4.3), so a batch is named by a game and a length rather than by a board.
    A length past the end of its game is a caller error and raises; a prefix
    whose position is terminal is refused by :func:`build`, which is the same
    refusal §2 makes of the trainer.
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
