"""Structural alias diagnostic (§33) and forward-cost telemetry (§34).

Reports structural aliases (actions whose builder-side bundle hashes
identically under canonical D6 folding), forward cost (node/edge counts,
timing, VRAM), and per-phase model behaviour (policy entropy, Q spread,
auxiliary accuracy).

Alias signature lines: cell fields, per-axis window triples, per-axis
post-placement rows, and radius orbit multiset.  Each is folded under the
group and hashed as a 64-bit polynomial.  ``geometry`` describes sampled
actions independently of the representation.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import torch

from .builder import build
from .config import ARCHITECTURE_ID, PRESETS, MantisACTConfig
from .heads import AUX_SPECS, MASK_SUFFIX
from .packed import (
    NUM_AXES,
    PHASE_FIRST,
    PHASE_OPENING,
    PHASE_SECOND,
    WINDOW_LEN,
    ACTGraph,
    collate,
    telemetry,
)
from .pattern_classes import ALL_CELL_WINDOW_REL_CLASSES, MIXED
from .summary import ParameterSummary, parameter_summary
from .symmetry import AXES, hex_distance, orbit_table

# §33's signature lines, in the order they are combined.
SIGNATURE_LINES: tuple[str, ...] = ("cell", "windows", "post1", "orbit")

# What a representation described an action by. `no_window` is an action that
# lies in no persistent window.
KIND_IN_WINDOW = "in_window"
KIND_NO_WINDOW = "no_window"

# §13.1's three placement phases, for the §34 split.
PHASE_NAMES: dict[int, str] = {
    PHASE_OPENING: "OPENING",
    PHASE_FIRST: "FIRST",
    PHASE_SECOND: "SECOND",
}

# How many of a position's highest-policy actions §34's Q spread is taken
# over — the top of the ranking rather than the whole legal set, whose spread
# is dominated by the halo.
TOP_POLICY_ACTIONS = 8

# The six unit steps, for the neighbour occupancies of the geometry descriptor.
_STEPS = np.concatenate([AXES, -AXES])

# Occupancy codes of the geometry descriptor, matching §8.2's cell field.
_EMPTY, _OWN, _OPP = 0, 1, 2

# §9.3's four window statuses, as a radix for packing a window's description
# into one integer code.
_STATUS_CLASSES = MIXED + 1


# --------------------------------------------------------------------------
# §33: the 64-bit line hash

# splitmix64's finalizer constants and an odd base for the polynomial, fixed
# so a digest compares across processes and across builds of the same position.
_HASH_BASE = np.uint64(0x9E3779B97F4A7C15)
_MIX_A = np.uint64(0xBF58476D1CE4E5B9)
_MIX_B = np.uint64(0x94D049BB133111EB)
_LENGTH_MUL = np.uint64(0xD6E8FEB86659FD93)
_TAG_MUL = np.uint64(0xA24BAED4963EE407)
_SHIFT_A, _SHIFT_B, _SHIFT_C = np.uint64(30), np.uint64(27), np.uint64(31)

# One tag per hashed quantity, so an empty windows line and an empty orbit
# line hash to different digests.
_TAG_CELL, _TAG_WINDOWS, _TAG_POST1, _TAG_ORBIT = 1, 2, 3, 4
_TAG_AXIS_GROUP, _TAG_VALUE = 5, 6

_POWERS: dict[int, np.ndarray] = {}


def _mix(value: np.ndarray) -> np.ndarray:
    """splitmix64's finalizer, elementwise over uint64."""
    with np.errstate(over="ignore"):
        value = value ^ (value >> _SHIFT_A)
        value = value * _MIX_A
        value = value ^ (value >> _SHIFT_B)
        value = value * _MIX_B
        return value ^ (value >> _SHIFT_C)


def _powers_of_base(width: int) -> np.ndarray:
    """``B**1 .. B**width`` mod 2^64, cached per width."""
    cached = _POWERS.get(width)
    if cached is None:
        powers = np.empty(width, dtype=np.uint64)
        value = _HASH_BASE
        with np.errstate(over="ignore"):
            for k in range(width):
                powers[k] = value
                value = value * _HASH_BASE
        powers.setflags(write=False)
        _POWERS[width] = powers
        cached = powers
    return cached


def _fold(values: np.ndarray, lengths, tag: int) -> np.ndarray:
    """Hash each row of a ``(n, W)`` uint64 table into one digest.

    ``lengths`` is each row's real entry count (a scalar for a fixed-width
    table); a padded slot must hold ``0`` so the digest stays independent of
    ``W``.
    """
    with np.errstate(over="ignore"):
        total = (values * _powers_of_base(values.shape[-1])).sum(
            axis=-1, dtype=np.uint64
        )
        salt = _LENGTH_MUL * np.asarray(lengths).astype(np.uint64)
        return _mix(total + salt + _TAG_MUL * np.uint64(tag))


# The digest of a line that describes nothing: every line but `post1` for a
# legal action the cell scope represents by no node (§8.3).
ABSENT_LINE: np.ndarray = _fold(np.zeros((1, 0), dtype=np.uint64), 0, 0)[0]


def combine(lines: Mapping[str, np.ndarray]) -> np.ndarray:
    """Fold one action's four line digests into its signature digest.

    Public because deleting a line is how §33's negative control is expressed:
    replacing a line's column with a constant and recombining is exactly the
    representation that does not read it.
    """
    missing = [name for name in SIGNATURE_LINES if name not in lines]
    unexpected = [name for name in lines if name not in SIGNATURE_LINES]
    if missing or unexpected:
        raise ValueError(
            f"signature lines {missing} are missing and {unexpected} are not "
            f"lines; §33's are {list(SIGNATURE_LINES)}"
        )
    stacked = np.stack(
        [np.asarray(lines[name], dtype=np.uint64) for name in SIGNATURE_LINES], axis=1
    )
    return _fold(stacked, len(SIGNATURE_LINES), _TAG_VALUE)


@dataclass(frozen=True)
class ActionSignatures:
    """One representation's per-legal-action structural description (§33).

    ``lines[name][j]`` is action ``j``'s digest of that signature line and
    ``kind[j]`` says what the representation described it by. ``value`` is
    derived from ``lines`` rather than passed, so it cannot disagree with
    them. Actions are in engine legal order throughout.
    """

    label: str
    kind: tuple[str, ...]
    lines: dict[str, np.ndarray]
    value: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        counts = {name: len(column) for name, column in self.lines.items()}
        if len(set(counts.values()) | {len(self.kind)}) > 1:
            raise ValueError(
                f"{self.label}: {len(self.kind)} kinds against line lengths "
                f"{counts}, so they describe different action counts"
            )
        object.__setattr__(self, "value", combine(self.lines))

    def __len__(self) -> int:
        return len(self.kind)


def _group_starts(counts: np.ndarray) -> np.ndarray:
    """The first row of each group in a table sorted by group id."""
    return np.cumsum(counts) - counts


def _within_group(owner: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Each row's index inside its own group, over a table sorted by owner."""
    return np.arange(len(owner), dtype=np.int64) - _group_starts(counts)[owner]


def _scatter(
    owner: np.ndarray, within: np.ndarray, code: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    """A ragged family as a zero-padded ``(groups, width)`` uint64 table."""
    table = np.zeros(shape, dtype=np.uint64)
    table[owner, within] = code
    return table


def _canonical_axis_groups(axis_digest: np.ndarray, tag: int) -> np.ndarray:
    """Three per-axis digests, ordered by value and folded into one.

    Sorting the digests is what removes the absolute axis id: a board symmetry
    permutes the three axes, so a description that kept which one is which
    would not be a function of the position (§12.2).
    """
    return _fold(np.sort(axis_digest, axis=1), NUM_AXES, tag)


def act_signatures(graph: ACTGraph) -> ActionSignatures:
    """§33's signature of every legal action of one ACT graph.

    Every line is built from the tables the model itself reads — cell fields,
    incidence, counterfactual rows, and radius edges — and nothing else, so
    the signature describes the representation rather than the board.
    Coordinates never enter it (§7).
    """
    n_cells, n_legal = graph.n_cells, graph.n_legal
    cell_of = graph.legal_to_cell_index
    has_cell = cell_of >= 0
    # A sentinel index would read row 0; every line it reaches is overwritten
    # with `ABSENT_LINE` below.
    cell = np.where(has_cell, cell_of, 0)

    # --- the cell line -----------------------------------------------------
    cell_codes = (
        np.stack(
            (graph.cell_occupancy, graph.cell_is_legal, graph.cell_nearest_bucket),
            axis=1,
        ).astype(np.uint64)
        + np.uint64(1)
    )
    cell_line = np.where(
        has_cell, _fold(cell_codes, cell_codes.shape[1], _TAG_CELL)[cell], ABSENT_LINE
    )

    # --- the incident-window line ------------------------------------------
    rows, slots = np.nonzero(graph.window_cell_index >= 0)
    incident_cell = graph.window_cell_index[rows, slots]
    incident_code = (
        (graph.window_pattern_class[rows] * _STATUS_CLASSES + graph.window_status[rows])
        * ALL_CELL_WINDOW_REL_CLASSES
        + graph.window_incidence_class[rows, slots]
        + 1
    ).astype(np.uint64)
    group = incident_cell * NUM_AXES + graph.window_axis[rows]
    order = np.lexsort((incident_code, group))
    group_counts = np.bincount(group, minlength=n_cells * NUM_AXES)
    within = _within_group(group[order], group_counts)
    if within.size and int(within.max()) >= WINDOW_LEN:
        crowded = int(group[order][int(np.argmax(within))])
        raise ValueError(
            f"cell {crowded // NUM_AXES} lies in more than {WINDOW_LEN} persistent "
            f"windows on axis {crowded % NUM_AXES}; a cell has {WINDOW_LEN} windows "
            "per axis at most, so this graph's incidence is not a window incidence"
        )
    axis_digest = _fold(
        _scatter(
            group[order], within, incident_code[order], (n_cells * NUM_AXES, WINDOW_LEN)
        ),
        group_counts,
        _TAG_AXIS_GROUP,
    ).reshape(n_cells, NUM_AXES)
    window_line = np.where(
        has_cell,
        _canonical_axis_groups(axis_digest, _TAG_WINDOWS)[cell],
        ABSENT_LINE,
    )

    # --- the eighteen counterfactual rows ----------------------------------
    # A transform that reverses an axis reverses its six candidate slots, so
    # each axis is canonicalised against its own reversal before the three are
    # ordered (§19.2).
    post_codes = (
        (
            (graph.action_post1_class * _STATUS_CLASSES + graph.action_pre_status) * 2
            + (graph.action_window_index >= 0)
            + 1
        )
        .astype(np.uint64)
        .reshape(-1, WINDOW_LEN)
    )
    post_axis = np.minimum(
        _fold(post_codes, WINDOW_LEN, _TAG_AXIS_GROUP),
        _fold(post_codes[:, ::-1], WINDOW_LEN, _TAG_AXIS_GROUP),
    ).reshape(n_legal, NUM_AXES)
    post_line = _canonical_axis_groups(post_axis, _TAG_POST1)

    # --- the radius-edge orbit line ----------------------------------------
    radius_code = (
        graph.radius_orbit * 3 + graph.cell_occupancy[graph.radius_src] + 1
    ).astype(np.uint64)
    radius_counts = np.bincount(graph.radius_dst, minlength=n_cells)
    order = np.lexsort((radius_code, graph.radius_dst))
    owner = graph.radius_dst[order]
    orbit_by_cell = _fold(
        _scatter(
            owner,
            _within_group(owner, radius_counts),
            radius_code[order],
            (n_cells, int(radius_counts.max()) if n_cells else 0),
        ),
        radius_counts,
        _TAG_ORBIT,
    )
    orbit_line = np.where(has_cell, orbit_by_cell[cell], ABSENT_LINE)

    in_window = has_cell & (np.bincount(incident_cell, minlength=n_cells)[cell] > 0)
    return ActionSignatures(
        label=ARCHITECTURE_ID,
        kind=tuple(np.where(in_window, KIND_IN_WINDOW, KIND_NO_WINDOW).tolist()),
        lines={
            "cell": cell_line,
            "windows": window_line,
            "post1": post_line,
            "orbit": orbit_line,
        },
    )


# --------------------------------------------------------------------------
# §33: the geometry a sampled alias group is read against


def geometry(
    stone_qr,
    stone_own,
    legal_qr,
    *,
    d_max: int = 12,
) -> tuple[dict[str, object], ...]:
    """Each given legal action's own geometry, described by neither representation.

    ``stone_own`` is ``0`` for the mover's stones and ``1`` for the opponent's.
    Sampled aliases are read against this descriptor: an entry that differs
    inside a group is something the board distinguishes and the signature
    does not. ``omitted_stones`` is the one field reaching past ``d_max``,
    described by distance and colour since no relation vocabulary reaches
    past the orbit table's radius-12 cap (§11.1).
    """
    stone_qr = np.asarray(stone_qr, dtype=np.int64).reshape(-1, 2)
    stone_own = np.asarray(stone_own, dtype=np.int64).reshape(-1)
    legal_qr = np.asarray(legal_qr, dtype=np.int64).reshape(-1, 2)
    if len(stone_own) != len(stone_qr):
        raise ValueError(
            f"stone_own has {len(stone_own)} entries for {len(stone_qr)} stones"
        )
    if d_max < 1:
        raise ValueError(f"d_max must be at least 1, got {d_max}")

    occupancy = {
        (int(q), int(r)): _OWN if own == 0 else _OPP
        for (q, r), own in zip(stone_qr, stone_own)
    }
    table = orbit_table(d_max)

    out: list[dict[str, object]] = []
    for q, r in legal_qr:
        dq, dr = stone_qr[:, 0] - int(q), stone_qr[:, 1] - int(r)
        distance = hex_distance(dq, dr)
        near = distance <= d_max
        colour = np.where(stone_own[near] == 0, _OWN, _OPP)
        far_colour = np.where(stone_own[~near] == 0, _OWN, _OPP)
        orbit = table.lookup(dq[near], dr[near])
        out.append(
            {
                "coordinate": (int(q), int(r)),
                "nearest_stone": int(distance.min()) if distance.size else -1,
                "stone_displacements": tuple(
                    sorted(
                        (int(a), int(b), int(c))
                        for a, b, c in zip(dq[near], dr[near], colour)
                    )
                ),
                "stone_orbits": tuple(
                    sorted((int(a), int(b)) for a, b in zip(orbit, colour))
                ),
                "omitted_stones": tuple(
                    sorted(
                        (int(a), int(b)) for a, b in zip(distance[~near], far_colour)
                    )
                ),
                "neighbour_occupancy": tuple(
                    sorted(
                        occupancy.get((int(q) + int(sq), int(r) + int(sr)), _EMPTY)
                        for sq, sr in _STEPS
                    )
                ),
            }
        )
    return tuple(out)


def graph_geometry(
    graph: ACTGraph, rows: Sequence[int] | None = None, *, d_max: int = 12
) -> tuple[dict[str, object], ...]:
    """:func:`geometry` for ``rows`` of a built graph's legal actions.

    ``rows`` defaults to every legal action. Stones and legal coordinates are
    recovered from the graph's own ``cell_qr`` metadata, which §25 keeps out
    of the model and out of every signature line.
    """
    occupied = np.flatnonzero(graph.cell_is_occupied)
    legal_cells = graph.legal_to_cell_index
    if rows is not None:
        legal_cells = legal_cells[np.asarray(rows, dtype=np.int64)]
    if int((legal_cells < 0).sum()):
        raise ValueError(
            "this cell scope represents no legal cell, so the graph carries no "
            "coordinate for a legal action; build the position under a scope "
            "that does to describe its geometry"
        )
    return geometry(
        graph.cell_qr[occupied],
        np.where(graph.cell_occupancy[occupied] == 1, 0, 1),
        graph.cell_qr[legal_cells],
        d_max=d_max,
    )


# --------------------------------------------------------------------------
# §33: grouping and the report


@dataclass(frozen=True)
class AliasGroup:
    """Legal actions one representation describes identically."""

    value: int
    kind: str
    rows: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.rows)


@dataclass(frozen=True)
class AliasSample:
    """One reported group: its coordinates and the geometry that differs."""

    kind: str
    rows: tuple[int, ...]
    coordinates: tuple[tuple[int, int], ...]
    differing: tuple[tuple[str, tuple[str, ...]], ...]
    identical: tuple[str, ...]

    def text(self) -> str:
        head = "  ".join(f"({q}, {r})" for q, r in self.coordinates)
        lines = [f"    {self.kind} x{len(self.rows)}: {head}"]
        for field_name, values in self.differing:
            lines.append(f"      differs  {field_name}")
            for coordinate, value in zip(self.coordinates, values):
                lines.append(f"        ({coordinate[0]}, {coordinate[1]}) {value}")
        if self.identical:
            lines.append(f"      identical  {', '.join(self.identical)}")
        return "\n".join(lines)


@dataclass(frozen=True)
class AliasReport:
    """§33's report for one position under one representation."""

    label: str
    legal_actions: int
    unique_signatures: int
    groups: tuple[AliasGroup, ...]
    samples: tuple[AliasSample, ...]

    @property
    def alias_groups(self) -> int:
        return len(self.groups)

    @property
    def max_group_size(self) -> int:
        return max((len(group) for group in self.groups), default=0)

    @property
    def aliased_actions(self) -> int:
        """Legal actions sharing a signature with at least one other."""
        return sum(len(group) for group in self.groups)

    def groups_of_kind(self, kind: str) -> tuple[AliasGroup, ...]:
        return tuple(group for group in self.groups if group.kind == kind)

    def text(self) -> str:
        lines = [
            f"  {self.label}",
            f"    legal actions       {self.legal_actions}",
            f"    unique signatures   {self.unique_signatures}",
            f"    alias groups        {self.alias_groups}",
            f"    max group size      {self.max_group_size}",
            f"    aliased actions     {self.aliased_actions}",
        ]
        for kind in (KIND_IN_WINDOW, KIND_NO_WINDOW):
            groups = self.groups_of_kind(kind)
            lines.append(
                f"    {kind:<18}  {len(groups)} group(s), "
                f"{sum(len(group) for group in groups)} action(s)"
            )
        lines.extend(sample.text() for sample in self.samples)
        return "\n".join(lines)


def _render(value: object) -> str:
    """One geometry entry as a bounded line of text."""
    text = repr(value)
    return text if len(text) <= 110 else f"{text[:107]}..."


def signature_groups(signatures: ActionSignatures) -> tuple[AliasGroup, ...]:
    """Every group of two or more actions sharing a signature, largest first.

    Ordered largest first and then by first action, so the grouping is a
    function of the position rather than of a dictionary's iteration order.
    """
    unique, inverse, counts = np.unique(
        signatures.value, return_inverse=True, return_counts=True
    )
    order = np.argsort(inverse, kind="stable")
    starts = np.cumsum(counts) - counts
    groups = []
    for index in np.flatnonzero(counts > 1):
        rows = tuple(int(row) for row in order[starts[index] : starts[index] + counts[index]])
        kinds = {signatures.kind[row] for row in rows}
        if len(kinds) != 1:
            raise ValueError(
                f"{signatures.label}: signature {int(unique[index]):016x} groups "
                f"actions of kinds {sorted(kinds)}, so the kind is not a function "
                "of the signature it is derived from"
            )
        groups.append(AliasGroup(value=int(unique[index]), kind=kinds.pop(), rows=rows))
    groups.sort(key=lambda group: (-len(group), group.rows[0]))
    return tuple(groups)


def alias_report(
    signatures: ActionSignatures,
    describe: Callable[[tuple[int, ...]], Sequence[Mapping[str, object]]] | None = None,
    *,
    samples: int = 3,
) -> AliasReport:
    """Group ``signatures`` and describe a sample of the groups (§33).

    ``describe`` maps a group's rows to one geometry entry each, called once
    per sampled group rather than over the whole action set.
    """
    if samples < 0:
        raise ValueError(f"samples={samples} must not be negative")

    groups = signature_groups(signatures)
    described: list[AliasSample] = []
    for group in groups[:samples]:
        coordinates: tuple[tuple[int, int], ...] = ()
        differing: list[tuple[str, tuple[str, ...]]] = []
        identical: list[str] = []
        if describe is not None:
            entries = list(describe(group.rows))
            if len(entries) != len(group.rows):
                raise ValueError(
                    f"describe returned {len(entries)} geometry entries for a "
                    f"group of {len(group.rows)} actions"
                )
            coordinates = tuple(tuple(entry["coordinate"]) for entry in entries)
            for name in entries[0]:
                if name == "coordinate":
                    continue
                values = [entry[name] for entry in entries]
                if all(value == values[0] for value in values):
                    identical.append(name)
                else:
                    differing.append((name, tuple(_render(v) for v in values)))
        described.append(
            AliasSample(
                kind=group.kind,
                rows=group.rows,
                coordinates=coordinates,
                differing=tuple(differing),
                identical=tuple(identical),
            )
        )

    return AliasReport(
        label=signatures.label,
        legal_actions=len(signatures),
        unique_signatures=len(signatures) - sum(len(g) - 1 for g in groups),
        groups=groups,
        samples=tuple(described),
    )


def act_alias_report(
    position, cfg: MantisACTConfig | None = None, *, samples: int = 3
) -> AliasReport:
    """Build ``position`` and report its structural aliases under ``cfg``."""
    cfg = cfg or PRESETS["full_act_v4"]
    graph = build(position, cfg)
    return alias_report(
        act_signatures(graph),
        lambda rows: graph_geometry(graph, rows, d_max=cfg.d_max),
        samples=samples,
    )


# --------------------------------------------------------------------------
# §34: what the model is doing, split by placement phase


@dataclass(frozen=True)
class PhaseReport:
    """§34's per-phase model diagnostics over one batch.

    ``policy_entropy`` and ``top_q_std`` are means over the phase's positions;
    ``aux_accuracy`` holds one entry per §24.1 head the model carries, over the
    phase's labelled rows alone. A phase the batch does not hold is not
    reported: a mean over nothing is not zero.
    """

    phase: str
    positions: int
    legal_actions: int
    policy_entropy: float
    top_q_std: float
    aux_accuracy: dict[str, float]

    def text(self) -> str:
        lines = [
            f"    {self.phase:<8}{self.positions:>5} position(s), "
            f"{self.legal_actions:>7} legal action(s)",
            f"      policy entropy          {self.policy_entropy:12.4f}",
            f"      top-{TOP_POLICY_ACTIONS} Q std           {self.top_q_std:12.4f}",
        ]
        for name in sorted(self.aux_accuracy):
            lines.append(f"      {name:<24}{self.aux_accuracy[name]:12.4f}")
        return "\n".join(lines)


def _auxiliary_predictions(aux: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Each §24.1 head's predicted class, from the logits it emitted.

    §24.2's window-fate head is not among them: its rows are windows rather
    than actions, and its label is how the game ends, which a position does
    not carry.
    """
    predictions: dict[str, torch.Tensor] = {}
    for name, logits in aux.items():
        if name.endswith(MASK_SUFFIX) or name not in AUX_SPECS:
            continue
        predictions[name] = (
            (logits > 0).long() if AUX_SPECS[name].logits == 1 else logits.argmax(dim=-1)
        )
    return predictions


def phase_diagnostics(
    output,
    phase_id: torch.Tensor,
    labels: Mapping[str, np.ndarray] | None = None,
    *,
    top: int = TOP_POLICY_ACTIONS,
) -> tuple[PhaseReport, ...]:
    """§34's OPENING/FIRST/SECOND split of one forward's outputs.

    ``output`` is an `model.ACTOutput`, ``phase_id`` the batch's own
    per-position phase, and ``labels`` the §24.1 labels of every legal action
    of the batch in the same flat order (`aux_labels.position_aux_labels`
    computes them per position). A model holding auxiliary heads without
    labels to score them against is refused.
    """
    if top < 1:
        raise ValueError(f"top={top} must be at least 1")
    offsets = output.legal_offsets.detach().cpu().numpy()
    phases = phase_id.detach().cpu().numpy()
    if len(phases) != len(offsets) - 1:
        raise ValueError(
            f"{len(phases)} phases against {len(offsets) - 1} positions of offsets"
        )

    predictions = _auxiliary_predictions(output.aux)
    labels = dict(labels or {})
    missing = sorted(set(predictions) - set(labels))
    if missing:
        raise ValueError(
            f"the model holds §24.1 head(s) {missing} and no label was given for "
            "them, so §34's accuracy column would silently omit a head that is "
            "training; `aux_labels.position_aux_labels` computes them"
        )
    for name, value in labels.items():
        if len(value) != int(offsets[-1]):
            raise ValueError(
                f"label {name!r} has {len(value)} rows against the batch's "
                f"{int(offsets[-1])} legal actions"
            )

    policy = output.policy_logits.detach().float()
    q_value = output.q_value.detach().float()

    reports = []
    for phase, name in PHASE_NAMES.items():
        selected = np.flatnonzero(phases == phase)
        if not selected.size:
            continue
        entropy, spread, rows = [], [], 0
        for index in selected:
            lo, hi = int(offsets[index]), int(offsets[index + 1])
            rows += hi - lo
            logits = policy[lo:hi]
            log_p = torch.log_softmax(logits, dim=0)
            entropy.append(float(-(log_p.exp() * log_p).sum()))
            best = logits.topk(min(top, hi - lo)).indices
            chosen = q_value[lo:hi][best]
            spread.append(float(chosen.std(correction=0)) if chosen.numel() > 1 else 0.0)

        row_phase = np.repeat(phases, np.diff(offsets)) == phase
        accuracy = {}
        for aux_name, predicted in predictions.items():
            mask = row_phase & output.aux[aux_name + MASK_SUFFIX].detach().cpu().numpy()
            if not mask.any():
                continue
            correct = predicted.detach().cpu().numpy()[mask] == labels[aux_name][mask]
            accuracy[aux_name] = float(correct.mean())

        reports.append(
            PhaseReport(
                phase=name,
                positions=len(selected),
                legal_actions=rows,
                policy_entropy=float(np.mean(entropy)),
                top_q_std=float(np.mean(spread)),
                aux_accuracy=accuracy,
            )
        )
    return tuple(reports)


# --------------------------------------------------------------------------
# §34: what one forward costs


@dataclass(frozen=True)
class ProfileReport:
    """§34's per-run figures for one model over one set of positions."""

    label: str
    positions: int
    legal_actions: int
    device: str
    graph: dict[str, float]
    seconds: dict[str, float]
    rate: dict[str, float]
    memory_mib: dict[str, float]
    parameters: ParameterSummary
    phases: tuple[PhaseReport, ...]

    def text(self) -> str:
        lines = [f"  {self.label} on {self.device}, {self.positions} position(s)"]
        for name, block in (
            ("seconds", self.seconds),
            ("rate", self.rate),
            ("memory MiB", self.memory_mib),
        ):
            lines.append(f"    {name}")
            for key, value in block.items():
                lines.append(f"      {key:<24}{value:12.4f}")
        lines.append("    graph")
        for key in sorted(self.graph):
            lines.append(f"      {key:<24}{self.graph[key]:12.2f}")
        lines.append("    by phase")
        lines.extend(phase.text() for phase in self.phases)
        lines.append("    parameters")
        lines.append(self.parameters.text())
        return "\n".join(lines)


def _synchronise(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _elapsed(device: torch.device, start: float) -> float:
    _synchronise(device)
    return time.perf_counter() - start


def profile(
    positions: Sequence,
    cfg: MantisACTConfig | None = None,
    *,
    device: str = "cpu",
    repeats: int = 3,
    mass_floor: float | None = 0.2,
    model=None,
) -> ProfileReport:
    """§34's build, collate, forward, backward, throughput, memory, and phases.

    Each stage is timed ``repeats`` times and the fastest run is kept, to avoid
    reporting scheduler noise alongside the model. The backward is over the sum
    of the policy logits, which touches every parameter the forward used.

    ``model`` lets a caller profile a model it already holds; without one a
    fresh model is built from ``cfg``. Passing both is refused. A model
    carrying §24.1 heads has its labels built and timed here as its own stage,
    since it repeats the window enumeration.
    """
    from .aux_labels import position_aux_labels  # deferred: it imports the heads
    from .model import MantisACT  # deferred: `model` imports every stage

    if not positions:
        raise ValueError("profile needs at least one position")
    if repeats < 1:
        raise ValueError(f"repeats={repeats} must be at least 1")
    if model is not None and cfg is not None:
        raise ValueError(
            "pass either a model or a config, not both: they would describe two "
            "architectures and the report names one"
        )
    cfg = cfg or (model.cfg if model is not None else PRESETS["full_act_v4"])
    target = torch.device(device)
    module = (model if model is not None else MantisACT(cfg)).to(target)
    scores_auxiliaries = module.heads.auxiliaries is not None

    stages = ("build", "collate", "labels", "forward", "backward")
    best = dict.fromkeys(stages, float("inf"))
    if target.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target)
    batch = None
    labels: dict[str, np.ndarray] = {}
    for _ in range(repeats):
        start = time.perf_counter()
        graphs = [build(position, cfg) for position in positions]
        best["build"] = min(best["build"], time.perf_counter() - start)

        start = time.perf_counter()
        batch = collate(graphs).to(target)
        best["collate"] = min(best["collate"], _elapsed(target, start))

        start = time.perf_counter()
        if scores_auxiliaries:
            per_position = [position_aux_labels(p, cfg) for p in positions]
            labels = {
                name: np.concatenate([entry[name] for entry in per_position])
                for name in per_position[0]
            }
        best["labels"] = min(best["labels"], time.perf_counter() - start)

        module.zero_grad(set_to_none=True)
        start = time.perf_counter()
        out = module(batch, mass_floor=mass_floor)
        best["forward"] = min(best["forward"], _elapsed(target, start))

        loss = out.policy_logits.float().sum()
        start = time.perf_counter()
        loss.backward()
        best["backward"] = min(best["backward"], _elapsed(target, start))
    module.zero_grad(set_to_none=True)

    counts = telemetry(batch)
    legal = int(batch.legal_offsets[-1])
    step = best["forward"] + best["backward"]
    return ProfileReport(
        label=cfg.architecture_id,
        positions=len(positions),
        legal_actions=legal,
        device=str(target),
        graph=counts,
        seconds=dict(best),
        rate={
            "positions_per_second": len(positions) / step,
            "legal_actions_per_second": legal / step,
            "positions_built_per_second": len(positions) / best["build"],
        },
        memory_mib={
            "peak_allocated": (
                torch.cuda.max_memory_allocated(target) / 2**20
                if target.type == "cuda"
                else 0.0
            ),
            "peak_reserved": (
                torch.cuda.max_memory_reserved(target) / 2**20
                if target.type == "cuda"
                else 0.0
            ),
        },
        parameters=parameter_summary(module),
        phases=phase_diagnostics(out, batch.phase_id, labels),
    )


# --------------------------------------------------------------------------
# The command


def _corpus_positions(corpus_path: str, games: int, plies: Sequence[int]) -> list:
    """Real self-play positions from a frozen corpus, one per (game, ply).

    Every requested ply is built from every selected game, so the sweep is a
    rectangle comparable across games. Only games at least ``max(plies)`` long
    are selected; a corpus holding fewer than ``games`` of them is refused.

    The lab package is imported here rather than at module scope, since it
    depends on this one.
    """
    import hexo_py

    from ...lab.corpus import load_corpus

    if games < 1:
        raise ValueError(f"games={games} must be at least 1")
    if not plies or min(plies) < 1:
        raise ValueError(f"every ply must be at least 1, got {tuple(plies)}")

    corpus = load_corpus(corpus_path)
    deepest = max(plies)
    long_enough = [
        game
        for game in range(corpus.n_games)
        if len(corpus.moves_for(game)) >= deepest
    ]
    if len(long_enough) < games:
        raise ValueError(
            f"{corpus.name} holds {len(long_enough)} games of at least {deepest} "
            f"plies, and {games} were asked for"
        )

    positions = []
    for game in long_enough[:games]:
        moves = corpus.moves_for(game)
        for ply in plies:
            positions.append((game, ply, hexo_py.Position.replay(moves[:ply])))
    return positions


def _profiled_model(cfg: MantisACTConfig, auxiliaries: Sequence[str]):
    """A model carrying the named §24.1 heads, or the plain preset model."""
    from .model import MantisACT

    if not auxiliaries:
        return None, cfg
    unknown = [name for name in auxiliaries if name not in AUX_SPECS]
    if unknown:
        raise ValueError(f"unknown auxiliaries {unknown}; §24.1 names {list(AUX_SPECS)}")
    with_heads = replace(cfg, enable_action_aux_heads=True)
    return (
        MantisACT(with_heads, aux_weights={name: 1.0 for name in auxiliaries}),
        with_heads,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m mantisnet.models.mantis_act.diagnostics`` (§33, §34)."""
    parser = argparse.ArgumentParser(
        prog="mantisnet.models.mantis_act.diagnostics",
        description="§33's structural alias diagnostic and §34's telemetry.",
    )
    parser.add_argument("command", choices=("alias", "profile"))
    parser.add_argument("--corpus", required=True, help="a frozen corpus directory")
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument(
        "--plies", type=int, nargs="+", default=(21, 61, 121, 161),
        help="the ply of each game to build",
    )
    parser.add_argument("--preset", default="full_act_v4", choices=sorted(PRESETS))
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--aux", nargs="*", default=(), choices=sorted(AUX_SPECS),
        help="§24.1 auxiliary heads to build and score in the profile",
    )
    args = parser.parse_args(argv)

    cfg = PRESETS[args.preset]
    selected = _corpus_positions(args.corpus, args.games, args.plies)

    if args.command == "profile":
        model, cfg = _profiled_model(cfg, args.aux)
        report = profile(
            [position for _game, _ply, position in selected],
            None if model is not None else cfg,
            model=model,
            device=args.device,
            repeats=args.repeats,
        )
        print(report.text())
        return 0

    totals = {"legal": 0, "unique": 0, "groups": 0, "aliased": 0, "max": 0}
    for game, ply, position in selected:
        report = act_alias_report(position, cfg, samples=args.samples)
        print(f"game {game} ply {ply}")
        print(report.text())
        totals["legal"] += report.legal_actions
        totals["unique"] += report.unique_signatures
        totals["groups"] += report.alias_groups
        totals["aliased"] += report.aliased_actions
        totals["max"] = max(totals["max"], report.max_group_size)
    print(
        f"{args.preset} over {len(selected)} position(s): "
        f"{totals['legal']} legal actions, {totals['unique']} unique signatures, "
        f"{totals['groups']} alias groups, {totals['aliased']} aliased actions, "
        f"largest group {totals['max']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - the command's entry point
    raise SystemExit(main())


__all__ = [
    "ABSENT_LINE",
    "KIND_IN_WINDOW",
    "KIND_NO_WINDOW",
    "PHASE_NAMES",
    "SIGNATURE_LINES",
    "TOP_POLICY_ACTIONS",
    "ActionSignatures",
    "AliasGroup",
    "AliasReport",
    "AliasSample",
    "PhaseReport",
    "ProfileReport",
    "act_alias_report",
    "act_signatures",
    "signature_groups",
    "alias_report",
    "combine",
    "geometry",
    "graph_geometry",
    "main",
    "phase_diagnostics",
    "profile",
]
