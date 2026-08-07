"""MantisNet-ACT graph containers (§7, §25, §26).

``ACTGraph`` is one position's node families and indices between them.
``collate`` batches them, shifting each index by its family's offset into the
batch frame. ``-1`` is the sole sentinel (survives collation unchanged).

``__post_init__`` runs ``_validate``: an instance that exists has passed bounds
checks on every index field. CSR offsets are ``(position_count + 1,)`` with a
leading zero.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from .pattern_classes import (
    ALL_CELL_WINDOW_REL_CLASSES,
    ALL_WINDOW_PATTERN_CLASSES,
    EMPTY,
    MIXED,
    OPP_LIVE,
    OWN_LIVE,
    POST1_REL_CLASSES,
)
from .windows import WINDOW_COORD_LIMIT

WINDOW_LEN = 6
NUM_AXES = 3
# 3 axes x 6 candidate slots through every legal cell (§19.2).
POST_ACTION_ROWS = NUM_AXES * WINDOW_LEN

# The rules' legality radius, and the §8.2 nearest-stone bucket vocabulary it
# fixes: one bucket per hex distance 0..LEGAL_RADIUS, plus one for a cell no
# stone reaches. A cell scope holds nothing outside a stone's disk, so this
# vocabulary is the rules', not a configuration's, and `_VALUE_RANGES` closes
# the field against it here rather than leaving the ceiling to whichever module
# emits the buckets.
LEGAL_RADIUS = 8
NEAREST_UNREACHED = LEGAL_RADIUS + 1
NEAREST_BUCKETS = LEGAL_RADIUS + 2

# Three-way placement phase (§13.1). ``moves_remaining`` stays authoritative for
# the KLENT return sign; this id is a model feature.
PHASE_OPENING, PHASE_FIRST, PHASE_SECOND = 0, 1, 2

# Telemetry labels for the §9.3 window statuses, keyed by the status value, so
# a status count cannot be mislabelled by an assumption about their order.
WINDOW_STATUS_NAMES = {
    EMPTY: "empty",
    OWN_LIVE: "own_live",
    OPP_LIVE: "opp_live",
    MIXED: "mixed",
}

SENTINEL = -1

# Node families, and the field whose length defines each one.
_CELLS, _WINDOWS, _LEGAL, _ADJACENCY, _RADIUS = (
    "cells",
    "windows",
    "legal",
    "adjacency",
    "radius",
)
_FAMILY_SIZED_BY = {
    _CELLS: "cell_occupancy",
    _WINDOWS: "window_pattern_class",
    _LEGAL: "legal_to_cell_index",
    _ADJACENCY: "adjacency_src",
    _RADIUS: "radius_src",
}

# Every array field of ``ACTGraph``: dtype and shape, where a string extent
# names a family and ``None`` is a free feature width.
_GRAPH_ARRAYS: tuple[tuple[str, type, tuple[object, ...]], ...] = (
    ("cell_qr", np.int64, (_CELLS, 2)),
    ("cell_occupancy", np.int64, (_CELLS,)),
    ("cell_is_legal", np.int64, (_CELLS,)),
    ("cell_is_occupied", np.int64, (_CELLS,)),
    ("cell_nearest_bucket", np.int64, (_CELLS,)),
    ("legal_to_cell_index", np.int64, (_LEGAL,)),
    ("window_id", np.int64, (_WINDOWS, 3)),
    ("window_pattern_class", np.int64, (_WINDOWS,)),
    ("window_status", np.int64, (_WINDOWS,)),
    ("window_axis", np.int64, (_WINDOWS,)),
    ("window_numeric", np.float32, (_WINDOWS, None)),
    ("window_cell_index", np.int64, (_WINDOWS, WINDOW_LEN)),
    ("window_incidence_class", np.int64, (_WINDOWS, WINDOW_LEN)),
    ("window_incidence_mask", np.bool_, (_WINDOWS, WINDOW_LEN)),
    ("adjacency_src", np.int64, (_ADJACENCY,)),
    ("adjacency_dst", np.int64, (_ADJACENCY,)),
    ("adjacency_axis", np.int64, (_ADJACENCY,)),
    ("radius_src", np.int64, (_RADIUS,)),
    ("radius_dst", np.int64, (_RADIUS,)),
    ("radius_orbit", np.int64, (_RADIUS,)),
    ("radius_axis_or_neg1", np.int64, (_RADIUS,)),
    ("action_window_index", np.int64, (_LEGAL, NUM_AXES, WINDOW_LEN)),
    ("action_post1_class", np.int64, (_LEGAL, NUM_AXES, WINDOW_LEN)),
    ("action_pre_status", np.int64, (_LEGAL, NUM_AXES, WINDOW_LEN)),
    ("action_tactical_numeric", np.float32, (_LEGAL, None)),
    ("global_numeric", np.float32, (None,)),
)

# Closed value ranges: ``(field, low, high)``, ``high`` of ``None`` meaning
# unbounded above. A bound is either fixed by the spec or read from the table
# the code indexes; a vocabulary a config can resize is left open above.
_VALUE_RANGES: tuple[tuple[str, int, int | None], ...] = (
    ("cell_occupancy", 0, 2),  # EMPTY / OWN / OPP relative to the mover (§8.2)
    ("cell_is_legal", 0, 1),
    ("cell_is_occupied", 0, 1),
    ("cell_nearest_bucket", 0, NEAREST_BUCKETS - 1),
    ("window_pattern_class", 0, ALL_WINDOW_PATTERN_CLASSES - 1),
    ("window_status", EMPTY, MIXED),
    ("window_axis", 0, NUM_AXES - 1),
    # A window identity is three columns, and the axis one is already covered
    # above through the field that must equal it (`_check_consistency`). What
    # this entry adds is the coordinate range `windows.WINDOW_COORD_LIMIT`
    # fixes: past it a start coordinate wraps in one of the two packings a
    # window identity passes through and silently merges two distinct windows.
    # `windows.py` refuses it on its own input; this is the same bound stated
    # where every graph passes, however it was built.
    ("window_id", 1 - WINDOW_COORD_LIMIT, WINDOW_COORD_LIMIT - 1),
    ("adjacency_axis", 0, NUM_AXES - 1),
    # ``d6_relation_mode`` chooses the relation vocabulary (§11.2), so the
    # ceiling is the model's to state; ``collate`` records this batch's own as
    # ``radius_orbit_bound`` and `messages.radius_edges` compares the two.
    ("radius_orbit", 0, None),
    ("radius_axis_or_neg1", SENTINEL, NUM_AXES - 1),
    ("action_post1_class", 0, POST1_REL_CLASSES - 1),
    ("action_pre_status", EMPTY, MIXED),
)

# Fields whose values index another family: ``(field, row family, target
# family, sentinel allowed)``. The row family is the family the field's rows
# belong to, which is what makes the §26 cross-position check possible.
_INDEX_FIELDS: tuple[tuple[str, str, str, bool], ...] = (
    # The sentinel is the `occupied_only` cell scope of §29, whose node set
    # holds no empty cell at all, so a legal action has no cell to point at and
    # the model creates its action state after the trunk.
    ("legal_to_cell_index", _LEGAL, _CELLS, True),
    ("window_cell_index", _WINDOWS, _CELLS, True),
    ("adjacency_src", _ADJACENCY, _CELLS, False),
    ("adjacency_dst", _ADJACENCY, _CELLS, False),
    ("radius_src", _RADIUS, _CELLS, False),
    ("radius_dst", _RADIUS, _CELLS, False),
    ("action_window_index", _LEGAL, _WINDOWS, True),
)

# The packed batch's array fields, in spec §25 order. ``cell_qr`` is builder
# metadata and stays out of the model's input (§7).
#
# ``window_id`` is the one identity table that crosses into the batch, and it
# is not a §25 field: §16's typed window↔window edges are a join of the window
# identities, cheaper to ship than the edges they generate. It is never
# renumbered — a window identity is an absolute coordinate triple, not an
# index into a family — and no parameter is ever selected by it.
# `messages.window_window_edges` is its only reader, deriving D6-invariant
# relation classes from it.
_PACKED_ARRAYS: tuple[str, ...] = (
    "cell_occupancy",
    "cell_is_legal",
    "cell_nearest_bucket",
    "legal_to_cell_index",
    "window_id",
    "window_pattern_class",
    "window_status",
    "window_axis",
    "window_numeric",
    "window_cell_index",
    "window_incidence_class",
    "window_incidence_mask",
    "adjacency_src",
    "adjacency_dst",
    "adjacency_axis",
    "radius_src",
    "radius_dst",
    "radius_orbit",
    "radius_axis_or_neg1",
    "action_window_index",
    "action_post1_class",
    "action_pre_status",
    "action_tactical_numeric",
)

_INDEX_TARGETS = {name: (target, sentinel) for name, _, target, sentinel in _INDEX_FIELDS}


def _lex_steps(columns: tuple[np.ndarray, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Compare consecutive rows of a lexicographic key, major column first.

    Returns the ``(greater, equal)`` masks of each adjacent pair, so a caller
    asks for strict ordering with ``greater`` and sorted order with
    ``greater | equal``.
    """
    n = max(len(columns[0]) - 1, 0)
    greater = np.zeros(n, dtype=bool)
    equal = np.ones(n, dtype=bool)
    for column in columns:
        greater |= equal & (column[1:] > column[:-1])
        equal &= column[1:] == column[:-1]
    return greater, equal


def _check_sorted(
    what: str, key: str, columns: tuple[np.ndarray, ...], *, strict: bool
) -> None:
    """Refuse a family that is not in its §7 order, naming the first break."""
    greater, equal = _lex_steps(columns)
    ordered = greater if strict else greater | equal
    if bool(ordered.all()):
        return
    row = int(np.flatnonzero(~ordered)[0]) + 1
    before = tuple(int(column[row - 1]) for column in columns)
    after = tuple(int(column[row]) for column in columns)
    raise ValueError(
        f"{what} must be sorted by {key}: row {row} is {after} after {before}"
    )


def _check_range(name: str, values: np.ndarray, low: int, high: int | None) -> None:
    """Refuse a field carrying a value outside its fixed range."""
    if values.size == 0:
        return
    if int(values.min()) < low:
        flat = int(values.argmin())
        raise ValueError(
            f"{name} must be >= {low}: found {int(values.min())} at flat index {flat}"
        )
    if high is not None and int(values.max()) > high:
        flat = int(values.argmax())
        raise ValueError(
            f"{name} must be <= {high}: found {int(values.max())} at flat index {flat}"
        )


def _to_global(index: np.ndarray, offset: int, *, sentinel: bool) -> np.ndarray:
    """Shift one position's indices into the batch frame, keeping ``-1`` as ``-1``.

    An offset sentinel reads as a genuine index into the preceding position's
    slice — in range, wrongly typed, and invisible to every downstream shape and
    dtype check. The two cases are therefore separate branches here rather than
    one arithmetic expression, and this is the only place in the module that
    adds an offset to an index.

    The values need no floor check of their own: `_INDEX_FIELDS` gives every
    index field its floor — the sentinel where one is allowed and zero where
    none is — and `ACTGraph.__post_init__` has run it before any graph exists.
    """
    if sentinel:
        return np.where(index == SENTINEL, SENTINEL, index + offset)
    return index + offset


@dataclass(frozen=True)
class ACTGraph:
    """One position's node families and index tables, in numpy (§7, §25).

    Every array is in this position's own frame. The family sizes are read off
    the arrays rather than stored, so a graph cannot claim a count its tables
    disagree with. ``cell_qr`` is builder metadata for dedup, ordering, tests,
    and diagnostics and never reaches the model (§7); ``window_id`` is that too,
    and is additionally the join key §16's typed window↔window edges are
    enumerated from, which is why `collate` carries it and `cell_qr` stays
    behind.

    Construction validates (`_validate`), so an ``ACTGraph`` that exists is a
    graph that passed and nothing downstream re-derives a field's dtype,
    shape, ordering, or value range as a defence.
    """

    # Cells (§8), sorted lexicographically by (q, r).
    cell_qr: np.ndarray  # (n_cells, 2) metadata only
    cell_occupancy: np.ndarray  # (n_cells,) 0 EMPTY / 1 OWN / 2 OPP
    cell_is_legal: np.ndarray  # (n_cells,) 0/1
    cell_is_occupied: np.ndarray  # (n_cells,) 0/1
    cell_nearest_bucket: np.ndarray  # (n_cells,) clamped nearest-stone bucket
    # Legal actions (§8.3), in engine order, never sorted. The index is -1
    # throughout when the cell scope represents no empty cell (§29).
    legal_to_cell_index: np.ndarray  # (n_legal,)
    # Persistent windows (§9), sorted by (native_axis, start_q, start_r).
    window_id: np.ndarray  # (n_windows, 3) (native_axis, start_q, start_r)
    window_pattern_class: np.ndarray  # (n_windows,)
    window_status: np.ndarray  # (n_windows,) EMPTY/OWN_LIVE/OPP_LIVE/MIXED
    window_axis: np.ndarray  # (n_windows,) 0..2
    window_numeric: np.ndarray  # (n_windows, F) float32 normalised counts/runs
    # Cell<->window incidence (§10). The mask marks the slots whose cell the
    # scope represents, so it is exactly ``window_cell_index >= 0``.
    window_cell_index: np.ndarray  # (n_windows, 6), -1 outside the cell scope
    window_incidence_class: np.ndarray  # (n_windows, 6), -1 where masked out
    window_incidence_mask: np.ndarray  # (n_windows, 6) bool
    # Local cell geometry (§15), sorted by (dst, src, relation).
    adjacency_src: np.ndarray  # (e_adj,)
    adjacency_dst: np.ndarray  # (e_adj,)
    adjacency_axis: np.ndarray  # (e_adj,) structural undirected axis
    radius_src: np.ndarray  # (e_rad,) occupied source
    radius_dst: np.ndarray  # (e_rad,)
    radius_orbit: np.ndarray  # (e_rad,) D6 displacement orbit
    radius_axis_or_neg1: np.ndarray  # (e_rad,) axis route, -1 off-axis
    # Counterfactual action rows (§19.2): 18 per legal action, dense.
    action_window_index: np.ndarray  # (n_legal, 3, 6), -1 with no pre-action window
    action_post1_class: np.ndarray  # (n_legal, 3, 6)
    action_pre_status: np.ndarray  # (n_legal, 3, 6)
    action_tactical_numeric: np.ndarray  # (n_legal, T) float32
    # Position scalars (§13).
    global_numeric: np.ndarray  # (G,) float32
    moves_remaining: int  # 1 or 2
    phase_id: int  # OPENING / FIRST / SECOND

    def __post_init__(self) -> None:
        self._validate()

    @property
    def n_cells(self) -> int:
        return len(self.cell_occupancy)

    @property
    def n_windows(self) -> int:
        return len(self.window_pattern_class)

    @property
    def n_legal(self) -> int:
        return len(self.legal_to_cell_index)

    @property
    def n_adjacency(self) -> int:
        return len(self.adjacency_src)

    @property
    def n_radius(self) -> int:
        return len(self.radius_src)

    def family_sizes(self) -> dict[str, int]:
        """Each node family's size, as the index tables must respect it."""
        return {family: len(getattr(self, field)) for family, field in _FAMILY_SIZED_BY.items()}

    def _validate(self) -> None:
        """Refuse a graph that violates §7 ordering, a shape, or an index bound.

        Runs from ``__post_init__``, so it is the one gate every graph passes
        however it was built. It raises ``TypeError`` for a wrong container or
        dtype and ``ValueError`` for everything else, always naming the field
        and the offending value.
        """
        sizes = self.family_sizes()

        for name, dtype, shape in _GRAPH_ARRAYS:
            array = getattr(self, name)
            if not isinstance(array, np.ndarray):
                raise TypeError(f"{name} must be a numpy array, got {type(array).__name__}")
            if array.dtype != dtype:
                raise TypeError(f"{name} must be {np.dtype(dtype)}, got {array.dtype}")
            expected = tuple(sizes[e] if isinstance(e, str) else e for e in shape)
            if len(array.shape) != len(expected) or any(
                e is not None and a != e for a, e in zip(array.shape, expected)
            ):
                shown = tuple("*" if e is None else e for e in expected)
                raise ValueError(f"{name} must have shape {shown}, got {array.shape}")

        for name, low, high in _VALUE_RANGES:
            _check_range(name, getattr(self, name), low, high)

        # A sentinel-bearing field's floor is the sentinel itself, so the same
        # range check refuses both an out-of-range target and a stray negative
        # that is not the sentinel.
        for name, _row_family, target, sentinel in _INDEX_FIELDS:
            _check_range(
                name, getattr(self, name), SENTINEL if sentinel else 0, sizes[target] - 1
            )

        self._check_ordering()
        self._check_consistency()

    def _check_ordering(self) -> None:
        """The §7 orders: node families strictly, edges non-strictly."""
        _check_sorted("cell nodes", "(q, r)", tuple(self.cell_qr.T), strict=True)
        _check_sorted(
            "persistent windows",
            "(native_axis, start_q, start_r)",
            tuple(self.window_id.T),
            strict=True,
        )
        _check_sorted(
            "cell adjacency edges",
            "(dst, src, axis)",
            (self.adjacency_dst, self.adjacency_src, self.adjacency_axis),
            strict=False,
        )
        _check_sorted(
            "occupied radius edges",
            "(dst, src, orbit)",
            (self.radius_dst, self.radius_src, self.radius_orbit),
            strict=False,
        )

    def _check_consistency(self) -> None:
        """Agreements between fields that describe the same fact twice."""
        occupied = self.cell_occupancy != 0
        if not np.array_equal(self.cell_is_occupied.astype(bool), occupied):
            bad = int(np.flatnonzero(self.cell_is_occupied.astype(bool) != occupied)[0])
            raise ValueError(
                f"cell_is_occupied disagrees with cell_occupancy at cell {bad}: "
                f"{int(self.cell_is_occupied[bad])} against occupancy "
                f"{int(self.cell_occupancy[bad])}"
            )
        both = np.flatnonzero((self.cell_is_legal != 0) & occupied)
        if both.size:
            raise ValueError(f"cell {int(both[0])} is both legal and occupied")
        # A cell scope either represents every legal cell or none of them: a
        # legal cell is empty, so `occupied_only` omits all of them and the
        # other two hold all of them. A mixture is a half-built node set, which
        # would leave some actions with a state and some without.
        named = self.legal_to_cell_index >= 0
        if named.any() and not named.all():
            bad = int(np.flatnonzero(~named)[0])
            raise ValueError(
                f"legal action {bad} has no cell node while others do: a cell "
                "scope holds either every legal cell or none of them"
            )
        if int(self.cell_is_legal.sum()) != int(named.sum()):
            raise ValueError(
                f"{int(self.cell_is_legal.sum())} cells are flagged legal but "
                f"legal_to_cell_index names {int(named.sum())}"
            )
        if named.all() and self.n_legal:
            if len(np.unique(self.legal_to_cell_index)) != self.n_legal:
                raise ValueError("legal_to_cell_index must name each legal cell once")
            if not bool(np.all(self.cell_is_legal[self.legal_to_cell_index])):
                bad = int(np.flatnonzero(self.cell_is_legal[self.legal_to_cell_index] == 0)[0])
                raise ValueError(
                    f"legal action {bad} maps to cell "
                    f"{int(self.legal_to_cell_index[bad])}, which is not flagged legal"
                )

        # §15.2 emits a radius edge from a stone. An empty source produces a
        # relation of ``2 * orbit + 0`` that is perfectly in range and means
        # something the position does not contain, so no shape, dtype, index
        # bound, or round trip downstream can see it — it is checked here, on
        # the graph, where the occupancy that contradicts it is still beside it.
        if self.radius_src.size:
            source_occupancy = self.cell_occupancy[self.radius_src]
            empty = np.flatnonzero(source_occupancy == 0)
            if empty.size:
                bad = int(empty[0])
                raise ValueError(
                    f"radius edge {bad} has source cell "
                    f"{int(self.radius_src[bad])}, which is empty: §15.2's edges "
                    "run from occupied cells"
                )

        # A window's native axis is stated twice: as the model's `window_axis`
        # column and as the leading component of the identity §16's edge join
        # reads. The two are read by different stages — the embedding routes a
        # line message by the first, the pair join decides collinear against
        # crossing by the second — so a disagreement is not a shape error
        # anywhere and would simply make the two stages describe different
        # boards.
        if not np.array_equal(self.window_axis, self.window_id[:, 0]):
            bad = int(np.flatnonzero(self.window_axis != self.window_id[:, 0])[0])
            raise ValueError(
                f"window {bad} names native axis {int(self.window_axis[bad])} in "
                f"window_axis and {int(self.window_id[bad, 0])} in its identity"
            )

        represented = self.window_cell_index >= 0
        if not np.array_equal(self.window_incidence_mask, represented):
            row, slot = (self.window_incidence_mask != represented).nonzero()
            raise ValueError(
                f"window_incidence_mask disagrees with window_cell_index at "
                f"window {int(row[0])} slot {int(slot[0])}: mask "
                f"{bool(self.window_incidence_mask[row[0], slot[0]])} against cell "
                f"index {int(self.window_cell_index[row[0], slot[0]])}"
            )
        unclassed = represented & (self.window_incidence_class < 0)
        if unclassed.any():
            rows, slots = unclassed.nonzero()
            row, slot = int(rows[0]), int(slots[0])
            raise ValueError(
                f"window_incidence_class is {int(self.window_incidence_class[row, slot])} "
                f"at represented window {row} slot {slot}"
            )
        _check_range(
            "window_incidence_class",
            self.window_incidence_class,
            SENTINEL,
            ALL_CELL_WINDOW_REL_CLASSES - 1,
        )

        if self.moves_remaining not in (1, 2):
            raise ValueError(f"moves_remaining must be 1 or 2, got {self.moves_remaining}")
        if self.phase_id not in (PHASE_OPENING, PHASE_FIRST, PHASE_SECOND):
            raise ValueError(f"phase_id must be 0, 1, or 2, got {self.phase_id}")
        if (self.phase_id == PHASE_FIRST) != (self.moves_remaining == 2):
            raise ValueError(
                f"phase_id {self.phase_id} disagrees with moves_remaining "
                f"{self.moves_remaining} (§13.1: FIRST means two placements remain)"
            )
        stones = int(self.cell_is_occupied.sum())
        if self.phase_id == PHASE_OPENING and stones:
            raise ValueError(f"OPENING phase with {stones} occupied cells")
        if self.phase_id == PHASE_SECOND and not stones:
            raise ValueError("SECOND phase with an empty board")


@dataclass
class PackedACTBatch:
    """Many positions' graphs concatenated into one model input (§25, §26).

    Node families are concatenated in graph order and every index is shifted
    into the batch frame, so no edge crosses a position. The ``(P + 1,)`` CSR
    offsets give each family's per-position slice; nothing in the forward
    rediscovers a segment boundary.
    """

    position_count: int
    cell_offsets: torch.Tensor  # (P + 1,) long
    window_offsets: torch.Tensor
    legal_offsets: torch.Tensor
    adjacency_offsets: torch.Tensor
    radius_offsets: torch.Tensor

    cell_occupancy: torch.Tensor  # (N_cells,) long
    cell_is_legal: torch.Tensor
    cell_nearest_bucket: torch.Tensor
    legal_to_cell_index: torch.Tensor  # (N_legal,) global cell index

    window_id: torch.Tensor  # (N_win, 3) (native_axis, start_q, start_r), §16 only
    window_pattern_class: torch.Tensor  # (N_win,) long
    window_status: torch.Tensor
    window_axis: torch.Tensor
    window_numeric: torch.Tensor  # (N_win, F) float32
    window_cell_index: torch.Tensor  # (N_win, 6) global cell index, -1 allowed
    window_incidence_class: torch.Tensor  # (N_win, 6)
    window_incidence_mask: torch.Tensor  # (N_win, 6) bool

    adjacency_src: torch.Tensor  # (E_adj,) global cell index
    adjacency_dst: torch.Tensor
    adjacency_axis: torch.Tensor

    radius_src: torch.Tensor  # (E_rad,) global cell index
    radius_dst: torch.Tensor
    radius_orbit: torch.Tensor
    radius_axis_or_neg1: torch.Tensor

    action_window_index: torch.Tensor  # (N_legal, 3, 6) global window index, -1 allowed
    action_post1_class: torch.Tensor
    action_pre_status: torch.Tensor
    action_tactical_numeric: torch.Tensor  # (N_legal, T) float32

    phase_id: torch.Tensor  # (P,) long
    moves_remaining: torch.Tensor  # (P,) long
    global_numeric: torch.Tensor  # (P, G) float32

    # One past the largest orbit id in ``radius_orbit``, or 0 for a batch with
    # no radius edge. Host-side. §11.2's orbit vocabulary is the one index
    # space a configuration resizes, so ``_VALUE_RANGES`` leaves
    # ``radius_orbit`` open above and the packer records the batch's own
    # ceiling instead; `messages.radius_edges` compares it against the model's
    # ``relation_vocabulary_size`` to refuse a batch and a model built under
    # different ``d6_relation_mode``/``d_max`` settings.
    radius_orbit_bound: int

    def to(self, device) -> "PackedACTBatch":
        """The same batch with every tensor on ``device``."""
        moved = {
            name: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for name, v in vars(self).items()
        }
        return PackedACTBatch(**moved)

    def pin_memory(self) -> "PackedACTBatch":
        """The same batch in pinned host memory, so ``to`` is a true async DMA.

        ``non_blocking`` silently degrades to a synchronous staged copy from
        pageable memory; a prefetch worker pins ahead of the transfer instead.
        """
        pinned = {
            name: (v.pin_memory() if isinstance(v, torch.Tensor) else v)
            for name, v in vars(self).items()
        }
        return PackedACTBatch(**pinned)


def _offsets(counts: np.ndarray) -> np.ndarray:
    """A family's ``(P + 1,)`` CSR offsets from its per-position counts."""
    return np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)


def _row_positions(offsets: np.ndarray) -> np.ndarray:
    """The position owning each row of a family."""
    return np.repeat(np.arange(len(offsets) - 1, dtype=np.int64), np.diff(offsets))


def _refuse_crossing(
    name: str, index: np.ndarray, row_positions: np.ndarray, target_offsets: np.ndarray
) -> None:
    """Refuse an index whose target lies in another position (§26, §30.18).

    The check re-derives each endpoint's position from the collated offsets
    rather than trusting the shift that produced it, so a per-position index
    that was already out of its own family's range — the way an edge crosses a
    position in practice — is caught here instead of aliasing silently onto a
    neighbour's node.
    """
    if index.size == 0:
        # An empty family has no endpoint to cross a position with. It is
        # reachable: the opening board holds no window at all, so a batch of
        # only that position gives `window_cell_index` shape (0, 6), whose
        # `-1` extent numpy cannot infer against a zero row count.
        return
    flat = index.reshape(len(row_positions), -1) if index.ndim > 1 else index[:, None]
    live = flat >= 0
    target_positions = np.searchsorted(target_offsets, flat, side="right") - 1
    crossing = live & (target_positions != row_positions[:, None])
    if not crossing.any():
        return
    rows, columns = crossing.nonzero()
    row, column = int(rows[0]), int(columns[0])
    raise ValueError(
        f"{name} crosses a batch position: row {row} of position "
        f"{int(row_positions[row])} points at {int(flat[row, column])}, which is "
        f"in position {int(target_positions[row, column])}"
    )


def collate(graphs: Sequence[ACTGraph]) -> PackedACTBatch:
    """Concatenate position graphs into one packed batch (§25, §26).

    Every index is shifted by its target family's offset, ``-1`` sentinels are
    preserved, and every shifted index is checked to land inside its own
    position. Nothing a graph already states about itself is re-checked here —
    an ``ACTGraph`` cannot exist without having passed ``_validate``. This
    function's own arithmetic is what `_refuse_crossing` guards: a family
    shifted by the wrong offset is a fault no graph can carry on its own.
    """
    if not graphs:
        raise ValueError("empty batch: collate needs at least one position")

    counts = {
        family: np.array([len(getattr(g, field)) for g in graphs], dtype=np.int64)
        for family, field in _FAMILY_SIZED_BY.items()
    }
    offsets = {family: _offsets(count) for family, count in counts.items()}

    packed: dict[str, torch.Tensor] = {}
    for name in _PACKED_ARRAYS:
        target = _INDEX_TARGETS.get(name)
        parts = []
        for i, graph in enumerate(graphs):
            array = getattr(graph, name)
            if target is not None:
                family, sentinel = target
                array = _to_global(array, int(offsets[family][i]), sentinel=sentinel)
            parts.append(array)
        widths = {part.shape[1:] for part in parts}
        if len(widths) != 1:
            raise ValueError(
                f"{name} has inconsistent feature widths across positions: {widths}"
            )
        packed[name] = torch.from_numpy(np.ascontiguousarray(np.concatenate(parts)))

    for name, row_family, target_family, _sentinel in _INDEX_FIELDS:
        _refuse_crossing(
            name,
            packed[name].numpy(),
            _row_positions(offsets[row_family]),
            offsets[target_family],
        )

    global_numeric = np.stack([g.global_numeric for g in graphs])
    orbits = packed["radius_orbit"]
    return PackedACTBatch(
        position_count=len(graphs),
        radius_orbit_bound=0 if orbits.numel() == 0 else int(orbits.max()) + 1,
        cell_offsets=torch.from_numpy(offsets[_CELLS]),
        window_offsets=torch.from_numpy(offsets[_WINDOWS]),
        legal_offsets=torch.from_numpy(offsets[_LEGAL]),
        adjacency_offsets=torch.from_numpy(offsets[_ADJACENCY]),
        radius_offsets=torch.from_numpy(offsets[_RADIUS]),
        phase_id=torch.tensor([g.phase_id for g in graphs], dtype=torch.long),
        moves_remaining=torch.tensor([g.moves_remaining for g in graphs], dtype=torch.long),
        global_numeric=torch.from_numpy(np.ascontiguousarray(global_numeric)),
        **packed,
    )


# The graph-cell budget a fitting chunk is packed under (§26), in graph cells
# plus occupied cells. See :class:`ACTChunkCost` for what that quantity is and
# why it is the one that binds; the number is where the measured throughput of
# ``policy_q`` stops rising and before the measured peak leaves the card.
ACT_GRAPH_CELL_BUDGET = 48_000


class ACTChunkCost:
    """What binds one MantisNet-ACT fitting chunk (``fitloop.ChunkCost``, §26).

    Every family §26 lists is ragged and concatenated; this architecture pads
    nothing, so unlike MantisNet it has no term quadratic in a chunk's longest
    position, and one additive limit covers it. The unit is *graph cells plus
    occupied cells*:

    ```text
    cost(sample) = cells + stones = 2 * stones + legal actions
    ```

    using the identity ``cells == stones + legal`` that holds exactly on this
    game (§8.1 of ``docs/MANTIS_ACT_DEVIATIONS.md``). Both terms are known from
    a stored sample without building its graph, which is what a packer needs.

    Measured against real self-play, this unit tracks memory flat to under 1%
    across the ply range, while radius edges — the largest family by row count
    — drive almost none of the memory, since the fused segment message
    recomputes them in backward rather than holding them; a budget in edge
    count would be wrong by several times end to end. ``ACT_GRAPH_CELL_BUDGET``
    is set where measured throughput plateaus and before the card runs out.
    """

    def __init__(self, stones, legal, graph_cell_budget: int) -> None:
        if len(stones) != len(legal):
            raise ValueError(
                f"{len(stones)} stone counts against {len(legal)} legal-move counts"
            )
        self._stones = stones
        self._legal = legal
        self._budget = int(graph_cell_budget)
        self._total = 0

    def units(self, index: int) -> int:
        """Sample ``index``'s graph cells plus its occupied cells."""
        return 2 * int(self._stones[index]) + int(self._legal[index])

    def sort_key(self, index: int) -> int:
        return self.units(index)

    def open(self) -> None:
        self._total = 0

    def accepts(self, index: int, size: int) -> bool:
        return self._total + self.units(index) <= self._budget

    def take(self, index: int) -> None:
        self._total += self.units(index)


def _segment_counts(offsets: torch.Tensor) -> torch.Tensor:
    """Per-position row counts of a CSR family."""
    return offsets[1:] - offsets[:-1]


def _segment_sums(offsets: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Per-position sums of a per-row integer quantity."""
    positions = torch.repeat_interleave(
        torch.arange(len(offsets) - 1, device=offsets.device), _segment_counts(offsets)
    )
    return torch.zeros(len(offsets) - 1, dtype=torch.long, device=offsets.device).index_add_(
        0, positions, values.long()
    )


def telemetry(batch: PackedACTBatch) -> dict[str, float]:
    """Per-position mean and max of every §26/§34 packer quantity.

    One pass of segment reductions over the batch's own tensors: exact integer
    counts, no sampling and no host-side loop over rows. Downstream stages set
    packer limits from these, so every number is the quantity itself rather than
    a proxy for it.
    """
    windows = _segment_counts(batch.window_offsets)
    legal = _segment_counts(batch.legal_offsets)
    quantities = {
        "cells": _segment_counts(batch.cell_offsets),
        "windows": windows,
        "legal_actions": legal,
        "window_incidences": _segment_sums(
            batch.window_offsets, batch.window_incidence_mask.sum(dim=1)
        ),
        "adjacency_edges": _segment_counts(batch.adjacency_offsets),
        "radius_edges": _segment_counts(batch.radius_offsets),
        # Dense by construction: 18 rows per legal action (§19.2).
        "post_action_rows": legal * POST_ACTION_ROWS,
    }
    for status, label in WINDOW_STATUS_NAMES.items():
        quantities[f"windows_{label}"] = _segment_sums(
            batch.window_offsets, batch.window_status == status
        )

    stats: dict[str, float] = {"positions": float(batch.position_count)}
    for name, counts in quantities.items():
        stats[f"{name}_mean"] = float(counts.double().mean().item())
        stats[f"{name}_max"] = float(counts.max().item())
    return stats
