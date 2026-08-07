"""Cell nodes, cell-window incidence, and local cell geometry (§8, §10, §15).

Builds the finite cell set, its six geometric window slots, and two sparse
edge families (adjacency §15.1 and occupied-radius §15.2). Cell index is
lexicographic ``(q, r)`` order (§7); a cell is legal iff empty and within
``LEGAL_RADIUS`` steps of a stone.

Adjacency edges are directed, labelled with the undirected axis. Radius edges
run from occupied to represented cells at distance ``1..occupied_radius``,
labelled with the D6 orbit class and axis (or ``-1`` off-axis). Both families
walk a precomputed hex disk, so work is ``O(stones * disk)`` (§26).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np

from .config import MantisACTConfig
from .packed import LEGAL_RADIUS, NEAREST_UNREACHED, NUM_AXES, SENTINEL
from .pattern_classes import CELL_WINDOW_CLASS, TERNARY_CODES
from .symmetry import AXES, coarse_relation, hex_distance, on_axis, orbit_table
from .windows import window_cells

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
    slices a shell with ``searchsorted`` and drops the centre with ``[1:]``.
    Cached and read-only: one table per radius per process.
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
    ``qr``, kept as both the node order and the lookup structure. ``scope`` is
    the ``cell_scope`` the set was built under, which lets :func:`incidence`
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
        axis. Absence (a window slot outside the scope, a neighbour past the
        halo) is a legitimate answer, so it returns the sentinel rather than
        raising; callers that cannot tolerate one check for it themselves.
        """
        return _lookup(self.key, qr)


def _nearest_buckets(key: np.ndarray, stone_qr: np.ndarray) -> np.ndarray:
    """The §8.2 nearest-stone bucket of every cell.

    Each stone claims the cells of its radius-``LEGAL_RADIUS`` disk, with
    shells written back to front so the nearest claim survives — a loop over
    the nine distances rather than over stones or cells. A cell no disk
    reaches keeps ``NEAREST_UNREACHED``, which on a legal cell is a rule
    violation.
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

    - ``occupied_only`` — the stones alone (§29's control). Its
      ``legal_to_cell_index`` is all ``-1``: a legal cell has no node to point
      at.
    - ``occupied_and_legal`` — the stones and the legal cells.
    - ``window_and_legal`` — those plus every window slot, so an empty and
      currently illegal cell inside a nonempty window is still a node (§8.1).

    ``window_cells`` is ignored by the two narrower scopes.
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
    # A legal cell lies within LEGAL_RADIUS of a stone; only the opening
    # placement has no stone to measure from.
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

    ``window_set`` is read for its ``window_id`` of ``(native_axis, start_q,
    start_r)`` triples in the §7 order, and its ``code``, the raw ternary code
    of each window's six slots — §10.1 classes against the raw code since a
    pattern class has already quotiented by reversal. Slot geometry is decoded
    by ``windows.window_cells``.

    Returns ``(window_cell_index, window_incidence_class, window_incidence_mask)``,
    each ``[num_windows, 6]``. A slot whose cell the scope omits carries ``-1``
    in both tables and ``False`` in the mask. Under the default
    ``window_and_legal`` scope every slot of every persistent window is a node
    by construction, so a gap there is refused rather than masked away.
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
    runs along, so an edge and its reverse are labelled alike. Every cell
    tries all six steps and keeps those landing on another node.
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

    Returns ``(src, dst, orbit, axis_or_neg1)``. The relation is the D6 class
    of the displacement ``dst - src`` under the configured
    ``d6_relation_mode``, and the axis is the one that displacement lies on,
    ``-1`` off-axis (§11.3). The source's colour is not carried on the edge:
    it is ``cell_set.occupancy[src]``.

    The threshold is a radius and nothing else: §15.3 forbids a fixed top-K
    cutoff with coordinate-order tie breaking, since such ties are not
    D6-invariant.

    The enumeration walks the radius disk out from each stone, costing
    ``O(stones * disk)`` independent of halo size; each candidate's
    displacement is the disk offset itself, classed once per call over the
    disk rather than once per edge.
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
