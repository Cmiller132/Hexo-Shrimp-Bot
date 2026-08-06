"""Relevant cell nodes, cell-window incidence, and local cell geometry.

This module implements §8, §10 and §15: the finite cell node set of a position,
the six geometric slots of every persistent window, and the two sparse edge
families between cells. It owns no window enumeration — the window coordinates
arrive as an argument — and no engine call: a cell is legal iff it is empty and
within ``LEGAL_RADIUS`` steps of a stone, and the engine stays an independent
oracle for that rule in the tests.

Index conventions this module fixes (each is part of the representation):

- Cell index: the rank of a cell's ``(q, r)`` in ascending lexicographic order
  (§7). Coordinates pack into one int64 key whose order *is* lexicographic
  order, so the sorted key array serves both as the node order and as the
  lookup structure. ``CellSet.index_of`` answers ``-1`` for a coordinate the
  scope omits, the package's one sentinel.
- Nearest-stone bucket: the hex distance to the closest stone, ``0`` on an
  occupied cell, and ``NEAREST_UNREACHED`` when no stone lies within
  ``LEGAL_RADIUS`` — which covers both a cell of an empty window persisted by
  the ``action_relevant`` scope and the stoneless opening. The clamp bucket and
  the unreached bucket are distinct so that a legal cell landing on the latter
  is a violation of the legality rule rather than an ordinary far cell.
- Adjacency edge: a directed edge between cells one hex step apart, labelled
  with the *undirected* axis it runs along, so the two directions of one
  neighbour pair carry the same label (§15.1).
- Radius edge: a directed edge from an occupied cell to a represented cell at
  hex distance ``1..occupied_radius``, labelled with the D6 relation class of
  the displacement ``dst - src`` and with the axis that displacement lies on,
  or ``-1`` off-axis (§15.2, §11.3). Distance ``0`` is not a radius edge:
  §11.2 gives the self relation its own reserved id, and no orbit table classes
  the zero displacement.

Both edge families and the nearest-stone pass walk a precomputed hex disk out
from each stone or cell rather than a cell-by-stone outer product. The disk is
a table of ``3 d (d + 1) + 1`` displacements built once per radius, so radius
work is ``O(stones * disk)`` and independent of how many cells the halo holds
— the difference that decides whether radius-12 edges are affordable in dense
late positions (§15.2, §26).

Whether the two edge families are built at all is the orchestrator's decision:
``use_cell_adjacency`` and ``use_occupied_radius_edges`` say *whether* and are
read by ``builder.py``; the config fields read here — ``cell_scope``,
``occupied_radius``, ``d_max``, ``d6_relation_mode`` — say *how*.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np

from .config import MantisACTConfig
from .packed import NUM_AXES, SENTINEL
from .pattern_classes import CELL_WINDOW_CLASS, TERNARY_CODES
from .symmetry import AXES, coarse_relation, hex_distance, on_axis, orbit_table
from .windows import window_cells

# A placement is legal iff its cell is empty and within this many hex steps of
# an occupied cell. The opening placement is the one exception: it has no stone
# to measure from.
LEGAL_RADIUS = 8

# Nearest-stone buckets: the distances 0..LEGAL_RADIUS, then the bucket for a
# cell no stone reaches.
NEAREST_UNREACHED = LEGAL_RADIUS + 1
NEAREST_BUCKETS = LEGAL_RADIUS + 2

# Cell occupancy relative to the side to move (§8.2).
OCCUPANCY_EMPTY, OCCUPANCY_OWN, OCCUPANCY_OPP = 0, 1, 2

# Coordinate packing for the sort key. Board coordinates are i16-bounded, so a
# stride of 2**21 against a component bound of 2**20 is collision-free and
# leaves the key monotone in (q, r): one step in q outruns the whole range of r.
_COORD_LIMIT = 1 << 20
_QSHIFT = 1 << 21

# The six unit steps, and the undirected axis each runs along. A symmetry may
# send a step to its negation, so the label is the axis and never the sign.
_STEPS = np.concatenate([AXES, -AXES])
_STEP_AXIS = np.concatenate([np.arange(NUM_AXES), np.arange(NUM_AXES)])


def _coords(name: str, qr) -> np.ndarray:
    """Read an ``(..., 2)`` coordinate argument as a flat ``(n, 2)`` int64 array."""
    qr = np.asarray(qr, dtype=np.int64)
    if qr.ndim == 0 or qr.shape[-1] != 2:
        raise ValueError(f"{name} must be (..., 2) coordinates, got shape {qr.shape}")
    return qr.reshape(-1, 2)


def _pack(qr: np.ndarray) -> np.ndarray:
    """Pack ``(n, 2)`` coordinates into int64 keys ordered as ``(q, r)`` is."""
    outside = np.abs(qr) >= _COORD_LIMIT
    if outside.any():
        bad = qr[outside.any(axis=1)][0]
        raise ValueError(
            f"coordinate ({int(bad[0])}, {int(bad[1])}) exceeds the "
            f"+-{_COORD_LIMIT} the cell sort key packs into one int64"
        )
    return qr[:, 0] * _QSHIFT + qr[:, 1]


def _lookup(key: np.ndarray, qr) -> np.ndarray:
    """Indices of ``(..., 2)`` coordinates in a sorted key array, ``-1`` if absent."""
    qr = np.asarray(qr, dtype=np.int64)
    if qr.ndim == 0 or qr.shape[-1] != 2:
        raise ValueError(f"coordinates must be (..., 2), got shape {qr.shape}")
    if len(key) == 0:
        return np.full(qr.shape[:-1], SENTINEL, dtype=np.int64)
    wanted = _pack(qr.reshape(-1, 2))
    position = np.searchsorted(key, wanted)
    np.clip(position, 0, len(key) - 1, out=position)
    return np.where(key[position] == wanted, position, SENTINEL).reshape(qr.shape[:-1])


def _first_duplicate(qr: np.ndarray) -> np.ndarray | None:
    """The first repeated coordinate of an ``(n, 2)`` array, or ``None``."""
    _unique, first, counts = np.unique(_pack(qr), return_index=True, return_counts=True)
    repeated = first[counts > 1]
    return qr[int(repeated.min())] if repeated.size else None


@functools.lru_cache(maxsize=None)
def _disk(radius: int) -> tuple[np.ndarray, np.ndarray]:
    """Every displacement out to ``radius``, and its distance, distance-sorted.

    The zero displacement is first and each shell is contiguous, so a caller
    slices a shell with ``searchsorted`` and drops the centre with ``[1:]``. The
    table is cached and read-only: one per radius per process, shared by every
    position built in it.
    """
    if radius < 0:
        raise ValueError(f"disk radius must not be negative, got {radius}")
    span = np.arange(-radius, radius + 1, dtype=np.int64)
    mesh_q, mesh_r = np.meshgrid(span, span, indexing="ij")
    offsets = np.stack([mesh_q.ravel(), mesh_r.ravel()], axis=1)
    distance = hex_distance(offsets[:, 0], offsets[:, 1])
    inside = distance <= radius
    offsets, distance = offsets[inside], distance[inside]
    order = np.argsort(distance, kind="stable")
    offsets, distance = offsets[order], distance[order]
    offsets.setflags(write=False)
    distance.setflags(write=False)
    return offsets, distance


@dataclass(frozen=True)
class CellSet:
    """One position's cell nodes in the §7 order, with their §8.2 fields.

    Every array is indexed by cell index. ``key`` is the packed sort key of
    ``qr``, kept because it is both the node order and the lookup structure and
    rebuilding it per query would dominate the build. ``scope`` is the
    ``cell_scope`` the set was built under, which is what lets :func:`incidence`
    hold the default scope to its all-slots-present promise (§10).
    """

    scope: str
    qr: np.ndarray  # (n, 2) sorted lexicographically, builder/debug only
    key: np.ndarray  # (n,) packed sort key of qr
    occupancy: np.ndarray  # (n,) EMPTY / OWN / OPP relative to the mover
    is_legal: np.ndarray  # (n,) 0/1
    is_occupied: np.ndarray  # (n,) 0/1
    nearest_bucket: np.ndarray  # (n,) 0..NEAREST_UNREACHED
    legal_to_cell_index: np.ndarray  # (n_legal,) in engine legal order (§8.3)

    def __len__(self) -> int:
        return len(self.key)

    def index_of(self, qr) -> np.ndarray:
        """Cell indices of an ``(..., 2)`` coordinate array, ``-1`` where absent.

        Vectorised over any leading shape; the result drops the trailing pair
        axis. Absence is a legitimate answer — a window slot outside the scope,
        a neighbour past the halo — so it is the sentinel rather than an error,
        and the callers that cannot tolerate one check for it themselves.
        """
        return _lookup(self.key, qr)


def _nearest_buckets(key: np.ndarray, stone_qr: np.ndarray) -> np.ndarray:
    """The §8.2 nearest-stone bucket of every cell.

    Each stone claims the cells of its radius-``LEGAL_RADIUS`` disk, and the
    shells are written back to front so the nearest claim is the one that
    survives. Writing whole shells makes this a loop over the nine distances
    rather than over stones or cells, and it needs no per-cell minimum over a
    ragged list of claims. A cell no disk reaches keeps ``NEAREST_UNREACHED``,
    which is the clamp §8.2 asks for and, on a legal cell, a rule violation.
    """
    nearest = np.full(len(key), NEAREST_UNREACHED, dtype=np.int64)
    if len(stone_qr) == 0:
        return nearest
    offsets, distance = _disk(LEGAL_RADIUS)
    shell = np.searchsorted(distance, np.arange(LEGAL_RADIUS + 2))
    claimed = _lookup(key, stone_qr[:, None, :] + offsets[None, :, :])
    for d in range(LEGAL_RADIUS, -1, -1):
        hit = claimed[:, shell[d] : shell[d + 1]]
        nearest[hit[hit >= 0]] = d
    return nearest


def relevant_cells(
    stone_qr,
    stone_own,
    legal_qr,
    window_cells,
    cfg: MantisACTConfig,
) -> CellSet:
    """The §8.1 cell node set of one position, in the §7 order.

    ``stone_qr`` is ``(n_stones, 2)`` and ``stone_own`` ``(n_stones,)`` with
    ``0`` for the side to move and ``1`` for the opponent; ``legal_qr`` is
    ``(n_legal, 2)`` in engine legal order; ``window_cells`` holds the
    coordinates of the persistent windows' slots, any shape ending in ``2``,
    which ``windows.py`` owns and this module never re-derives.

    The scope selects which of the three sources contribute:

    - ``occupied_only`` — the stones alone. §29 makes this the control whose
      legal actions are created after the trunk, so its ``legal_to_cell_index``
      is all ``-1``: a legal cell has no node to point at.
    - ``occupied_and_legal`` — the stones and the legal cells.
    - ``window_and_legal`` — those plus every window slot, so an empty and
      currently illegal cell inside a nonempty window is still a node (§8.1).

    ``window_cells`` is ignored by the two narrower scopes: they are defined as
    node sets that exclude it, not as scopes that were handed nothing.
    """
    stone_qr = _coords("stone_qr", stone_qr)
    legal_qr = _coords("legal_qr", legal_qr)
    window_cells = _coords("window_cells", window_cells)
    stone_own = np.asarray(stone_own, dtype=np.int64).reshape(-1)
    if len(stone_own) != len(stone_qr):
        raise ValueError(
            f"stone_own has {len(stone_own)} entries for {len(stone_qr)} stones"
        )
    outside = (stone_own < 0) | (stone_own > 1)
    if outside.any():
        raise ValueError(
            f"stone_own must be 0 (mover) or 1 (opponent), got "
            f"{int(stone_own[outside][0])}"
        )
    for name, source in (("stone_qr", stone_qr), ("legal_qr", legal_qr)):
        repeated = _first_duplicate(source)
        if repeated is not None:
            raise ValueError(
                f"{name} lists ({int(repeated[0])}, {int(repeated[1])}) twice"
            )

    if cfg.cell_scope == "occupied_only":
        sources = (stone_qr,)
    elif cfg.cell_scope == "occupied_and_legal":
        sources = (stone_qr, legal_qr)
    elif cfg.cell_scope == "window_and_legal":
        sources = (stone_qr, legal_qr, window_cells)
    else:
        raise ValueError(f"unknown cell_scope {cfg.cell_scope!r}")

    stacked = np.concatenate(sources)
    # The packed key is monotone in (q, r), so the unique keys come back in the
    # §7 cell order and the rows that produced them are the node coordinates.
    key, first = np.unique(_pack(stacked), return_index=True)
    qr = stacked[first]

    occupancy = np.zeros(len(key), dtype=np.int64)
    is_occupied = np.zeros(len(key), dtype=np.int64)
    is_legal = np.zeros(len(key), dtype=np.int64)
    stone_index = _lookup(key, stone_qr)
    occupancy[stone_index] = OCCUPANCY_OWN + stone_own
    is_occupied[stone_index] = 1

    legal_index = _lookup(key, legal_qr)
    present = legal_index[legal_index >= 0]
    occupied_legal = present[is_occupied[present] == 1]
    if occupied_legal.size:
        bad = qr[occupied_legal[0]]
        raise ValueError(
            f"legal cell ({int(bad[0])}, {int(bad[1])}) holds a stone: a placement "
            "is legal only on an empty cell"
        )
    is_legal[present] = 1

    nearest_bucket = _nearest_buckets(key, stone_qr)
    # The legality rule this module computes against: a legal cell lies within
    # LEGAL_RADIUS of a stone, and only the opening placement has no stone to
    # measure from. A legal cell no stone reaches means the caller's legal list
    # and the rule disagree, which would silently poison the halo.
    if len(stone_qr):
        unreached = present[nearest_bucket[present] == NEAREST_UNREACHED]
        if unreached.size:
            bad = qr[unreached[0]]
            raise ValueError(
                f"legal cell ({int(bad[0])}, {int(bad[1])}) is more than "
                f"{LEGAL_RADIUS} hex steps from every stone"
            )

    return CellSet(
        scope=cfg.cell_scope,
        qr=qr,
        key=key,
        occupancy=occupancy,
        is_legal=is_legal,
        is_occupied=is_occupied,
        nearest_bucket=nearest_bucket,
        legal_to_cell_index=legal_index,
    )


def incidence(
    window_set, cell_set: CellSet
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The §10 cell-window incidence tables of the persistent windows.

    ``window_set`` is the ``windows.WindowSet`` of the position, read for two
    fields: its ``window_id`` of ``(native_axis, start_q, start_r)`` triples in
    the §7 order, and its ``code``, the raw ternary code of each window's six
    slots. The raw code is what §10.1 classes against — a pattern class has
    already quotiented by reversal, and a slot's class must be taken in the same
    joint orbit as the pattern it sits in. The slot geometry is decoded by
    ``windows.window_cells``, which is where the slot-order convention lives.

    Returns ``(window_cell_index, window_incidence_class, window_incidence_mask)``,
    each ``[num_windows, 6]``. A slot whose cell the scope omits carries ``-1``
    in both tables and ``False`` in the mask. Under the default
    ``window_and_legal`` scope every slot of every persistent window is a node
    by construction, so a gap there is refused rather than masked away: it means
    the coordinates the cell set was built from are not this window set's, and
    every incidence message through the missing slot would silently vanish.
    """
    window_id = np.asarray(window_set.window_id, dtype=np.int64).reshape(-1, 3)
    code = np.asarray(window_set.code, dtype=np.int64).reshape(-1)
    if len(code) != len(window_id):
        raise ValueError(
            f"window code has {len(code)} entries for {len(window_id)} windows"
        )
    outside = (code < 0) | (code >= TERNARY_CODES)
    if outside.any():
        raise ValueError(
            f"window code must be 0..{TERNARY_CODES - 1}, got {int(code[outside][0])}"
        )

    coords = window_cells(window_id)
    window_cell_index = cell_set.index_of(coords)
    window_incidence_mask = window_cell_index >= 0
    window_incidence_class = np.where(
        window_incidence_mask, CELL_WINDOW_CLASS[code], SENTINEL
    )

    if cell_set.scope == "window_and_legal" and not window_incidence_mask.all():
        rows, slots = (~window_incidence_mask).nonzero()
        row, missing = int(rows[0]), int(slots[0])
        bad = coords[row, missing]
        raise ValueError(
            f"window {tuple(int(v) for v in window_id[row])} slot {missing} is cell "
            f"({int(bad[0])}, {int(bad[1])}), which the 'window_and_legal' cell set "
            "does not hold: it was built from other window coordinates"
        )
    return window_cell_index, window_incidence_class, window_incidence_mask


def _sorted_edges(
    dst: np.ndarray, src: np.ndarray, relation: np.ndarray, *rest: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Put an edge family into the §7 ``(dst, src, relation)`` order."""
    order = np.lexsort((relation, src, dst))
    return tuple(column[order] for column in (src, dst, relation, *rest))


def adjacency_edges(cell_set: CellSet) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The §15.1 directed edges between cells one hex step apart.

    Returns ``(src, dst, axis)``, the axis being the undirected one the step
    runs along, so an edge and its reverse are labelled alike. Every cell tries
    all six steps and keeps those landing on another node, which emits each
    directed edge exactly once and makes the family symmetric: a step's reverse
    is also one of the six.
    """
    index = cell_set.index_of(cell_set.qr[:, None, :] + _STEPS[None, :, :])
    present = index >= 0
    src = np.broadcast_to(
        np.arange(len(cell_set), dtype=np.int64)[:, None], index.shape
    )[present]
    axis = np.broadcast_to(_STEP_AXIS[None, :], index.shape)[present]
    return _sorted_edges(index[present], src, axis)


def radius_edges(
    cell_set: CellSet,
    stone_qr,
    stone_own,
    cfg: MantisACTConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The §15.2 edges from every stone to every cell within ``occupied_radius``.

    Returns ``(src, dst, orbit, axis_or_neg1)``. The relation is the D6 class of
    the displacement ``dst - src`` under the configured ``d6_relation_mode``,
    and the axis is the one that displacement lies on, ``-1`` off-axis (§11.3).
    The source's colour is not carried on the edge: it is
    ``cell_set.occupancy[src]``, and stating it twice would let the two copies
    disagree.

    The threshold is a radius and nothing else. §15.3 forbids a fixed top-K
    cutoff with coordinate-order tie breaking, because ties at the cutoff would
    be broken by an order that is not D6-invariant: a position and its
    reflection would then keep different neighbours, and every claim of exact
    equivariance would be false while every shape and count stayed plausible.

    The enumeration walks the radius disk out from each stone, so it costs
    ``O(stones * disk)`` however many cells the halo holds, and each candidate's
    displacement — hence its relation class and its axis route — is the disk
    offset itself, classed once per call over the disk rather than once per edge.
    """
    stone_qr = _coords("stone_qr", stone_qr)
    stone_own = np.asarray(stone_own, dtype=np.int64).reshape(-1)
    if len(stone_own) != len(stone_qr):
        raise ValueError(
            f"stone_own has {len(stone_own)} entries for {len(stone_qr)} stones"
        )
    source = cell_set.index_of(stone_qr)
    absent = source < 0
    if absent.any():
        bad = stone_qr[absent][0]
        raise ValueError(
            f"stone ({int(bad[0])}, {int(bad[1])}) is not a cell node: every cell "
            "scope holds the occupied cells"
        )
    stated = np.where(cell_set.occupancy[source] == OCCUPANCY_OWN, 0, 1)
    disagree = stated != stone_own
    if disagree.any():
        bad = int(np.flatnonzero(disagree)[0])
        raise ValueError(
            f"stone {bad} at ({int(stone_qr[bad, 0])}, {int(stone_qr[bad, 1])}) is "
            f"stone_own={int(stone_own[bad])} against occupancy "
            f"{int(cell_set.occupancy[source[bad]])} in the cell set"
        )

    # Distance 0 is not a radius edge: §11.2 reserves an id for the self
    # relation, and no orbit table classes the zero displacement.
    offsets = _disk(cfg.occupied_radius)[0][1:]
    if cfg.d6_relation_mode == "orbit48":
        relation = orbit_table(cfg.d_max).lookup(offsets[:, 0], offsets[:, 1])
    elif cfg.d6_relation_mode == "coarse_distance_axis":
        relation = coarse_relation(offsets[:, 0], offsets[:, 1], cfg.d_max)
    else:
        raise ValueError(f"unknown d6_relation_mode {cfg.d6_relation_mode!r}")
    route = on_axis(offsets[:, 0], offsets[:, 1])

    index = cell_set.index_of(stone_qr[:, None, :] + offsets[None, :, :])
    present = index >= 0
    return _sorted_edges(
        index[present],
        np.broadcast_to(source[:, None], index.shape)[present],
        np.broadcast_to(relation[None, :], index.shape)[present],
        np.broadcast_to(route[None, :], index.shape)[present],
    )
