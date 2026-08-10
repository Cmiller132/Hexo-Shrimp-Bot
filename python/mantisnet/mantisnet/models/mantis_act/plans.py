"""CPU-built execution plans for the ACT state and action stacks.

The packed graph is a one-use training input.  Sorting its message edges or
discovering routed subsets after the graph reaches CUDA therefore puts pure
preparation work on every step.  This module derives those views once, beside
``packed.collate``, while the concatenated arrays are still on the CPU.

All kernel-facing indices and CSR pointers are contiguous ``int32`` tensors.
The row-to-position and phase vectors remain ``int64`` because they are also
ordinary torch gather indices.  Every sort is stable: rows tied on a source,
destination, relation, embedding class, or source window retain the canonical
packed row order, which fixes the order of deterministic reductions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib

import numpy as np
import torch
from torch import Tensor

from .config import MantisACTConfig
from .pattern_classes import (
    ALL_CELL_WINDOW_REL_CLASSES,
    MIXED,
    POST1_REL_CLASSES,
)
from .segment_message import MessagePlan
from .symmetry import RELATION_PAD, coarse_relation, coarse_relation_count, orbit_table


AXIS_CHANNELS = 3
INCIDENCE_RELATIONS = ALL_CELL_WINDOW_REL_CLASSES
WINDOW_STATUSES = MIXED + 1
_INT32_MAX = np.iinfo(np.int32).max

# Exactly the switches consumed while enumerating an ACTGraph or deriving the
# plans in this file.  Model-only widths, depths, heads, and auxiliary outputs
# can share one packed batch; representation or edge changes cannot.
_BUILDER_FIELDS = (
    "architecture_id",
    "window_scope",
    "cell_scope",
    "use_axis_channels",
    "use_global_numeric_features",
    "use_window_numeric_features",
    "use_action_tactical_features",
    "d6_relation_mode",
    "d_max",
    "use_cell_adjacency",
    "use_occupied_radius_edges",
    "occupied_radius",
    "route_on_axis_radius_messages",
)


def builder_fingerprint(cfg: MantisACTConfig) -> str:
    """The exact semantic configuration under which a packed plan was built.

    This is intentionally narrower than the checkpoint architecture hash: an
    auxiliary head or trunk depth does not alter a packed graph or any CSR and
    must not make an otherwise identical batch unusable.  The explicit list is
    held against builder/plan tests whenever a new representation switch is
    introduced.
    """
    payload = "\n".join(f"{name}={getattr(cfg, name)!r}" for name in _BUILDER_FIELDS)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def relation_vocabulary_size(cfg: MantisACTConfig) -> int:
    """The geometry relation vocabulary selected by ``cfg`` (spec section 11)."""

    if cfg.d6_relation_mode == "orbit48":
        return RELATION_PAD + 1
    if cfg.d6_relation_mode == "coarse_distance_axis":
        return coarse_relation_count(cfg.d_max)
    raise ValueError(f"unknown d6_relation_mode {cfg.d6_relation_mode!r}")


def radius_relation_count(cfg: MantisACTConfig) -> int:
    """The joint ``(geometry class, source colour)`` vocabulary."""

    return 2 * relation_vocabulary_size(cfg)


def adjacency_relation_id(cfg: MantisACTConfig) -> int:
    """The one D6 relation class occupied by a distance-one displacement."""

    if cfg.d6_relation_mode == "orbit48":
        return int(orbit_table(cfg.d_max).lookup(1, 0))
    if cfg.d6_relation_mode == "coarse_distance_axis":
        return int(coarse_relation(1, 0, cfg.d_max))
    raise ValueError(f"unknown d6_relation_mode {cfg.d6_relation_mode!r}")


def _tensor(
    name: str,
    value: Tensor,
    *,
    dtype: torch.dtype,
    rows: int | None = None,
    device: torch.device | None = None,
) -> torch.device:
    """Structural tensor validation that never reads a device value."""

    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if value.dtype != dtype:
        raise TypeError(f"{name} must be {dtype}, got {value.dtype}")
    if value.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got {tuple(value.shape)}")
    if rows is not None and int(value.shape[0]) != rows:
        raise ValueError(f"{name} must have {rows} rows, got {int(value.shape[0])}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if device is not None and value.device != device:
        raise ValueError(f"{name} is on {value.device}, other plan tensors on {device}")
    return value.device


def _move(value, device, non_blocking: bool):
    if isinstance(value, Tensor):
        return value.to(device, non_blocking=non_blocking)
    if hasattr(value, "to"):
        return value.to(device, non_blocking=non_blocking)
    return value


def _pin(value):
    if isinstance(value, Tensor):
        return value.pin_memory()
    if hasattr(value, "pin_memory"):
        return value.pin_memory()
    return value


@dataclass(frozen=True, eq=False)
class LatentSegments:
    """The precomputed multi-range view of ragged latent rows."""

    ranges: Tensor
    range_base: Tensor
    counts: Tensor
    row_pos: Tensor
    n_rows: int
    positions: int
    families: int

    def __post_init__(self) -> None:
        for name in ("n_rows", "positions", "families"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"LatentSegments.{name} must be a nonnegative int, got {value!r}"
                )
        if self.families < 1:
            raise ValueError("LatentSegments.families must be at least 1")
        expected = {
            "ranges": (self.positions, self.families, 2),
            "range_base": (self.positions, self.families),
            "counts": (self.positions,),
            "row_pos": (self.n_rows,),
        }
        dtypes = {
            "ranges": torch.int32,
            "range_base": torch.int32,
            "counts": torch.int32,
            "row_pos": torch.int64,
        }
        device = None
        for name, shape in expected.items():
            value = getattr(self, name)
            if not isinstance(value, Tensor):
                raise TypeError(
                    f"LatentSegments.{name} must be a tensor, got {type(value).__name__}"
                )
            if value.dtype != dtypes[name]:
                raise TypeError(
                    f"LatentSegments.{name} must be {dtypes[name]}, got {value.dtype}"
                )
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"LatentSegments.{name} must be {shape}, got {tuple(value.shape)}"
                )
            if not value.is_contiguous():
                raise ValueError(f"LatentSegments.{name} must be contiguous")
            if device is None:
                device = value.device
            elif value.device != device:
                raise ValueError(
                    f"LatentSegments.{name} is on {value.device}, other fields on {device}"
                )
        if device is not None and device.type == "cpu":
            starts, ends = self.ranges[..., 0], self.ranges[..., 1]
            if bool(((starts < 0) | (ends < starts) | (ends > self.n_rows)).any()):
                raise ValueError("LatentSegments.ranges contains an invalid row span")
            lengths = ends - starts
            if not torch.equal(lengths.sum(dim=1), self.counts):
                raise ValueError("LatentSegments.counts disagrees with its ranges")
            expected_base = lengths.cumsum(dim=1) - lengths
            if not torch.equal(expected_base, self.range_base):
                raise ValueError("LatentSegments.range_base disagrees with its ranges")
            if self.row_pos.numel() and bool(
                ((self.row_pos < 0) | (self.row_pos >= self.positions)).any()
            ):
                raise ValueError("LatentSegments.row_pos contains an invalid position")
            owner_counts = torch.bincount(
                self.row_pos, minlength=self.positions
            ).to(torch.int32)
            if not torch.equal(owner_counts, self.counts):
                raise ValueError("LatentSegments.row_pos disagrees with its counts")

    @property
    def device(self) -> torch.device:
        return self.ranges.device

    def to(self, device, *, non_blocking: bool = True) -> "LatentSegments":
        return LatentSegments(
            **{
                name: _move(value, device, non_blocking)
                for name, value in vars(self).items()
            }
        )

    def pin_memory(self) -> "LatentSegments":
        return LatentSegments(**{name: _pin(value) for name, value in vars(self).items()})


@dataclass(frozen=True, eq=False)
class PlannedEdges:
    """One explicit edge family and its two precomputed message views."""

    src: Tensor
    dst: Tensor
    relation: Tensor
    axis: Tensor | None
    n_src: int
    n_dst: int
    num_relations: int
    dst_sorted: bool
    fully_routed: bool
    name: str
    axis_rows: Tensor | None
    routed_src: Tensor | None
    routed_dst: Tensor | None
    routed_relation: Tensor | None
    routed_axis: Tensor | None
    inv_plan: MessagePlan
    axis_plan: MessagePlan | None

    def __post_init__(self) -> None:
        for name in ("n_src", "n_dst", "num_relations"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"{self.name}.{name} must be a nonnegative int, got {value!r}"
                )
        for name in ("dst_sorted", "fully_routed"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{self.name}.{name} must be a host-side bool")

        rows = int(self.src.shape[0]) if isinstance(self.src, Tensor) else None
        device = _tensor(f"{self.name}.src", self.src, dtype=torch.int64)
        _tensor(f"{self.name}.dst", self.dst, dtype=torch.int64, rows=rows, device=device)
        _tensor(
            f"{self.name}.relation",
            self.relation,
            dtype=torch.int64,
            rows=rows,
            device=device,
        )
        if self.axis is not None:
            _tensor(
                f"{self.name}.axis",
                self.axis,
                dtype=torch.int64,
                rows=rows,
                device=device,
            )
        if self.axis_rows is not None:
            _tensor(
                f"{self.name}.axis_rows",
                self.axis_rows,
                dtype=torch.int64,
                device=device,
            )
        routed_columns = (
            self.routed_src,
            self.routed_dst,
            self.routed_relation,
            self.routed_axis,
        )
        if any(value is not None for value in routed_columns) and not all(
            value is not None for value in routed_columns
        ):
            raise ValueError(f"{self.name} must carry all four routed columns or none")
        if self.axis_rows is None and any(value is not None for value in routed_columns):
            raise ValueError(f"{self.name} carries routed columns without a subset index")
        if self.axis_rows is not None:
            routed_rows = int(self.axis_rows.shape[0])
            for column_name, column in zip(
                ("routed_src", "routed_dst", "routed_relation", "routed_axis"),
                routed_columns,
                strict=True,
            ):
                _tensor(
                    f"{self.name}.{column_name}",
                    column,
                    dtype=torch.int64,
                    rows=routed_rows,
                    device=device,
                )
        if self.axis is None and not self.fully_routed:
            raise ValueError(f"{self.name} has no axis column but claims an unrouted subset")
        if self.fully_routed and self.axis_rows is not None:
            raise ValueError(f"{self.name} is fully routed but carries axis_rows")

        if self.inv_plan.channels != 1:
            raise ValueError(f"{self.name}.inv_plan must have one channel")
        if self.inv_plan.device != device:
            raise ValueError(
                f"{self.name}.inv_plan is on {self.inv_plan.device}, edges on {device}"
            )
        if self.inv_plan.n_edges != rows:
            raise ValueError(
                f"{self.name}.inv_plan has {self.inv_plan.n_edges} rows, edges have {rows}"
            )
        if (self.inv_plan.n_src, self.inv_plan.n_dst, self.inv_plan.n_relations) != (
            self.n_src,
            self.n_dst,
            self.num_relations,
        ):
            raise ValueError(f"{self.name}.inv_plan metadata disagrees with its edge family")
        if self.axis_plan is not None:
            if self.axis is None:
                raise ValueError(f"{self.name} has an axis plan but no axis column")
            if self.axis_plan.channels != AXIS_CHANNELS:
                raise ValueError(f"{self.name}.axis_plan must have three channels")
            if self.axis_plan.device != device:
                raise ValueError(
                    f"{self.name}.axis_plan is on {self.axis_plan.device}, edges on {device}"
                )
            expected_axis_rows = rows if self.axis_rows is None else int(self.axis_rows.shape[0])
            if self.axis_plan.n_edges != expected_axis_rows:
                raise ValueError(
                    f"{self.name}.axis_plan has {self.axis_plan.n_edges} rows against "
                    f"the routed subset's {expected_axis_rows}"
                )
            if (
                self.axis_plan.n_src,
                self.axis_plan.n_dst,
                self.axis_plan.n_relations,
            ) != (self.n_src, self.n_dst, self.num_relations):
                raise ValueError(f"{self.name}.axis_plan metadata disagrees with its family")

        if device.type == "cpu":
            for column, bound in (
                (self.src, self.n_src),
                (self.dst, self.n_dst),
                (self.relation, self.num_relations),
            ):
                if column.numel() and bool(((column < 0) | (column >= bound)).any()):
                    raise ValueError(f"{self.name} contains an out-of-range edge index")
            if self.axis is not None and self.axis.numel():
                if bool(((self.axis < -1) | (self.axis >= AXIS_CHANNELS)).any()):
                    raise ValueError(f"{self.name}.axis contains a route outside -1..2")
                routed = (self.axis >= 0).nonzero(as_tuple=True)[0]
                if self.fully_routed and int(routed.numel()) != rows:
                    raise ValueError(f"{self.name} claims fully_routed with an unrouted row")
                if self.axis_rows is not None and not torch.equal(self.axis_rows, routed):
                    raise ValueError(
                        f"{self.name}.axis_rows must be the stable routed subset"
                    )
                if self.axis_rows is not None:
                    expected = tuple(
                        column.index_select(0, routed)
                        for column in (self.src, self.dst, self.relation, self.axis)
                    )
                    if any(
                        not torch.equal(got, want)
                        for got, want in zip(routed_columns, expected, strict=True)
                    ):
                        raise ValueError(
                            f"{self.name}'s routed columns do not match axis_rows"
                        )

    @property
    def device(self) -> torch.device:
        return self.src.device

    def __len__(self) -> int:
        return int(self.src.shape[0])

    def routed(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if self.axis is None:
            raise ValueError(f"{self.name} routes no axis message")
        if self.axis_rows is None:
            return self.src, self.dst, self.relation, self.axis
        if any(
            value is None
            for value in (
                self.routed_src,
                self.routed_dst,
                self.routed_relation,
                self.routed_axis,
            )
        ):
            raise RuntimeError(f"{self.name} lost its materialised routed columns")
        return (
            self.routed_src,
            self.routed_dst,
            self.routed_relation,
            self.routed_axis,
        )

    def plan(self, channels: int) -> MessagePlan:
        if channels == 1:
            return self.inv_plan
        if channels == AXIS_CHANNELS and self.axis_plan is not None:
            return self.axis_plan
        if channels not in (1, AXIS_CHANNELS):
            raise ValueError(f"channels must be 1 or 3, got {channels}")
        raise ValueError(f"{self.name} has no precomputed axis plan")

    def to(self, device, *, non_blocking: bool = True) -> "PlannedEdges":
        return PlannedEdges(
            **{
                name: _move(value, device, non_blocking)
                for name, value in vars(self).items()
            }
        )

    def pin_memory(self) -> "PlannedEdges":
        return PlannedEdges(**{name: _pin(value) for name, value in vars(self).items()})


@dataclass(frozen=True, eq=False)
class StatePlans:
    """The four sparse message families reused by every state block."""

    to_windows: PlannedEdges
    to_cells: PlannedEdges
    adjacency: PlannedEdges | None
    radius: PlannedEdges | None

    @property
    def device(self) -> torch.device:
        return self.to_windows.device

    @property
    def window_window(self):
        """Typed window attention is an ablation and keeps its lazy reference join."""
        return None

    def __post_init__(self) -> None:
        for name in ("to_cells", "adjacency", "radius"):
            value = getattr(self, name)
            if value is not None and value.device != self.device:
                raise ValueError(
                    f"state plan {name} is on {value.device}, incidence is on {self.device}"
                )

    def to(self, device, *, non_blocking: bool = True) -> "StatePlans":
        return StatePlans(
            **{
                name: _move(value, device, non_blocking)
                for name, value in vars(self).items()
            }
        )

    def pin_memory(self) -> "StatePlans":
        return StatePlans(**{name: _pin(value) for name, value in vars(self).items()})


@dataclass(frozen=True, eq=False)
class ClassRowPlan:
    """A flat row grid stably grouped by one embedding class."""

    n_rows: int
    n_classes: int
    ptr: Tensor
    rows: Tensor
    name: str

    def __post_init__(self) -> None:
        if self.n_rows < 0 or self.n_classes < 1:
            raise ValueError(
                f"{self.name} needs nonnegative rows and positive classes, got "
                f"{self.n_rows}, {self.n_classes}"
            )
        device = _tensor(
            f"{self.name}.ptr", self.ptr, dtype=torch.int32, rows=self.n_classes + 1
        )
        _tensor(
            f"{self.name}.rows",
            self.rows,
            dtype=torch.int32,
            rows=self.n_rows,
            device=device,
        )
        if device.type == "cpu":
            if int(self.ptr[0]) != 0 or int(self.ptr[-1]) != self.n_rows:
                raise ValueError(
                    f"{self.name}.ptr must span 0..{self.n_rows}, got "
                    f"{int(self.ptr[0])}..{int(self.ptr[-1])}"
                )
            if bool((self.ptr[1:] < self.ptr[:-1]).any()):
                raise ValueError(f"{self.name}.ptr must be monotone")
            if self.n_rows:
                if bool(((self.rows < 0) | (self.rows >= self.n_rows)).any()):
                    raise ValueError(f"{self.name}.rows contains an out-of-range row")
                expected = torch.arange(self.n_rows, dtype=torch.int32)
                if not torch.equal(self.rows.sort().values, expected):
                    raise ValueError(f"{self.name}.rows must be a permutation of the row grid")

    @property
    def device(self) -> torch.device:
        return self.ptr.device

    def to(self, device, *, non_blocking: bool = True) -> "ClassRowPlan":
        return ClassRowPlan(
            self.n_rows,
            self.n_classes,
            self.ptr.to(device, non_blocking=non_blocking),
            self.rows.to(device, non_blocking=non_blocking),
            self.name,
        )

    def pin_memory(self) -> "ClassRowPlan":
        return ClassRowPlan(
            self.n_rows,
            self.n_classes,
            self.ptr.pin_memory(),
            self.rows.pin_memory(),
            self.name,
        )


@dataclass(frozen=True, eq=False)
class SourceWindowPlan:
    """Action rows grouped by persistent source window, plus sentinel rows."""

    n_rows: int
    n_windows: int
    ptr: Tensor
    rows: Tensor
    sentinel_rows: Tensor

    def __post_init__(self) -> None:
        if self.n_rows < 0 or self.n_windows < 0:
            raise ValueError("source-window row and window counts must be nonnegative")
        device = _tensor(
            "action source-window ptr",
            self.ptr,
            dtype=torch.int32,
            rows=self.n_windows + 1,
        )
        _tensor("action source-window rows", self.rows, dtype=torch.int32, device=device)
        _tensor(
            "action source-window sentinel_rows",
            self.sentinel_rows,
            dtype=torch.int32,
            device=device,
        )
        if int(self.rows.shape[0]) + int(self.sentinel_rows.shape[0]) != self.n_rows:
            raise ValueError("source-window live and sentinel rows do not cover the row grid")
        if device.type == "cpu":
            live = int(self.rows.shape[0])
            if int(self.ptr[0]) != 0 or int(self.ptr[-1]) != live:
                raise ValueError(
                    f"source-window ptr must span 0..{live}, got "
                    f"{int(self.ptr[0])}..{int(self.ptr[-1])}"
                )
            if bool((self.ptr[1:] < self.ptr[:-1]).any()):
                raise ValueError("source-window ptr must be monotone")
            combined = torch.cat((self.rows, self.sentinel_rows))
            if combined.numel():
                if bool(((combined < 0) | (combined >= self.n_rows)).any()):
                    raise ValueError("source-window plan contains an out-of-range row")
                expected = torch.arange(self.n_rows, dtype=torch.int32)
                if not torch.equal(combined.sort().values, expected):
                    raise ValueError(
                        "source-window live and sentinel rows must partition the row grid"
                    )

    @property
    def device(self) -> torch.device:
        return self.ptr.device

    def to(self, device, *, non_blocking: bool = True) -> "SourceWindowPlan":
        return SourceWindowPlan(
            self.n_rows,
            self.n_windows,
            self.ptr.to(device, non_blocking=non_blocking),
            self.rows.to(device, non_blocking=non_blocking),
            self.sentinel_rows.to(device, non_blocking=non_blocking),
        )

    def pin_memory(self) -> "SourceWindowPlan":
        return SourceWindowPlan(
            self.n_rows,
            self.n_windows,
            self.ptr.pin_memory(),
            self.rows.pin_memory(),
            self.sentinel_rows.pin_memory(),
        )


@dataclass(frozen=True, eq=False)
class ActionRowPlans:
    post1: ClassRowPlan
    pre_status: ClassRowPlan
    source_window: SourceWindowPlan

    def __post_init__(self) -> None:
        if len({self.post1.n_rows, self.pre_status.n_rows, self.source_window.n_rows}) != 1:
            raise ValueError("the three action row plans describe different row counts")
        if len({self.post1.device, self.pre_status.device, self.source_window.device}) != 1:
            raise ValueError("the three action row plans are on different devices")

    @property
    def device(self) -> torch.device:
        return self.post1.device

    def to(self, device, *, non_blocking: bool = True) -> "ActionRowPlans":
        return ActionRowPlans(
            self.post1.to(device, non_blocking=non_blocking),
            self.pre_status.to(device, non_blocking=non_blocking),
            self.source_window.to(device, non_blocking=non_blocking),
        )

    def pin_memory(self) -> "ActionRowPlans":
        return ActionRowPlans(
            self.post1.pin_memory(),
            self.pre_status.pin_memory(),
            self.source_window.pin_memory(),
        )


@dataclass(frozen=True, eq=False)
class ACTPlans:
    """Every batch-only execution view moved with ``PackedACTBatch``."""

    state_edges: StatePlans
    action_rows: ActionRowPlans
    cell_row_pos: Tensor
    window_row_pos: Tensor
    legal_row_pos: Tensor
    cell_phase: Tensor
    window_phase: Tensor
    action_phase: Tensor
    state_segments: LatentSegments
    action_segments: LatentSegments

    def __post_init__(self) -> None:
        device = self.state_edges.device
        for name in (
            "cell_row_pos",
            "window_row_pos",
            "legal_row_pos",
            "cell_phase",
            "window_phase",
            "action_phase",
        ):
            _tensor(name, getattr(self, name), dtype=torch.int64, device=device)
        for name in ("action_rows", "state_segments", "action_segments"):
            value = getattr(self, name)
            if value.device != device:
                raise ValueError(f"{name} is on {value.device}, state plans on {device}")
        if int(self.cell_row_pos.shape[0]) != int(self.cell_phase.shape[0]):
            raise ValueError("cell row positions and phases have different lengths")
        if int(self.window_row_pos.shape[0]) != int(self.window_phase.shape[0]):
            raise ValueError("window row positions and phases have different lengths")
        if int(self.legal_row_pos.shape[0]) != int(self.action_phase.shape[0]):
            raise ValueError("legal row positions and action phases have different lengths")
        n_cells = int(self.cell_row_pos.shape[0])
        n_windows = int(self.window_row_pos.shape[0])
        n_legal = int(self.legal_row_pos.shape[0])
        state = self.state_edges
        if (state.to_windows.n_src, state.to_windows.n_dst) != (n_cells, n_windows):
            raise ValueError("cells->windows plans disagree with row-position lengths")
        if (state.to_cells.n_src, state.to_cells.n_dst) != (n_windows, n_cells):
            raise ValueError("windows->cells plans disagree with row-position lengths")
        for name in ("adjacency", "radius"):
            edges = getattr(state, name)
            if edges is not None and (edges.n_src, edges.n_dst) != (n_cells, n_cells):
                raise ValueError(f"{name} plans disagree with the cell row count")
        if self.action_rows.post1.n_rows != n_legal * AXIS_CHANNELS * 6:
            raise ValueError("action class plans do not cover the 3x6 row grid")
        if (
            self.state_segments.n_rows != n_cells + n_windows
            or self.state_segments.families != 2
        ):
            raise ValueError("state latent segments disagree with the cell/window rows")
        if self.action_segments.n_rows != n_legal or self.action_segments.families != 1:
            raise ValueError("action latent segments disagree with the legal rows")
        if self.action_segments.positions != self.state_segments.positions:
            raise ValueError("state and action plans describe different position counts")
        if device.type == "cpu":
            positions = self.state_segments.positions
            for name in ("cell_row_pos", "window_row_pos", "legal_row_pos"):
                rows = getattr(self, name)
                if rows.numel() and bool(((rows < 0) | (rows >= positions)).any()):
                    raise ValueError(f"{name} contains an invalid position")
                if rows.numel() > 1 and bool((rows[1:] < rows[:-1]).any()):
                    raise ValueError(f"{name} must preserve position-major row order")
            for name in ("cell_phase", "window_phase", "action_phase"):
                phase = getattr(self, name)
                if phase.numel() and bool(((phase < 0) | (phase > 2)).any()):
                    raise ValueError(f"{name} contains a phase outside 0..2")
            combined = torch.cat((self.cell_row_pos, self.window_row_pos))
            if not torch.equal(combined, self.state_segments.row_pos):
                raise ValueError("state latent ownership disagrees with row positions")
            if not torch.equal(self.legal_row_pos, self.action_segments.row_pos):
                raise ValueError("action latent ownership disagrees with legal rows")

    @property
    def device(self) -> torch.device:
        return self.cell_row_pos.device

    def to(self, device, *, non_blocking: bool = True) -> "ACTPlans":
        return ACTPlans(
            **{
                name: _move(value, device, non_blocking)
                for name, value in vars(self).items()
            }
        )

    def pin_memory(self) -> "ACTPlans":
        return ACTPlans(**{name: _pin(value) for name, value in vars(self).items()})


def _as_int32(name: str, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.size:
        low, high = int(values.min()), int(values.max())
        if low < np.iinfo(np.int32).min or high > _INT32_MAX:
            raise ValueError(f"{name} has range {low}..{high}, outside signed int32")
    return np.ascontiguousarray(values, dtype=np.int32)


def _torch_int64(values: np.ndarray) -> Tensor:
    return torch.from_numpy(np.ascontiguousarray(values, dtype=np.int64))


def _torch_int32(name: str, values: np.ndarray) -> Tensor:
    return torch.from_numpy(_as_int32(name, values))


def _csr(sorted_key: np.ndarray, n_keys: int, name: str) -> np.ndarray:
    if n_keys > _INT32_MAX:
        raise ValueError(f"{name} has {n_keys} keys, outside signed int32")
    probes = np.arange(n_keys + 1, dtype=np.int64)
    return _as_int32(name, np.searchsorted(sorted_key, probes, side="left"))


def _message_plan(
    src: np.ndarray,
    dst: np.ndarray,
    relation: np.ndarray,
    axis: np.ndarray | None,
    *,
    n_src: int,
    n_dst: int,
    n_relations: int,
    channels: int,
    dst_sorted: bool,
    name: str,
) -> MessagePlan:
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    relation = np.asarray(relation, dtype=np.int64)
    if not (src.shape == dst.shape == relation.shape and src.ndim == 1):
        raise ValueError(f"{name} edge columns must be equal-length vectors")
    if channels == AXIS_CHANNELS:
        if axis is None:
            raise ValueError(f"{name} axis plan needs routes")
        axis = np.asarray(axis, dtype=np.int64)
        if axis.shape != src.shape or (axis.size and (axis.min() < 0 or axis.max() >= 3)):
            raise ValueError(f"{name} axis routes must be a 0..2 vector")
    elif channels == 1:
        axis = None
    else:
        raise ValueError(f"channels must be 1 or 3, got {channels}")
    for column_name, values, bound in (
        ("src", src, n_src),
        ("dst", dst, n_dst),
        ("relation", relation, n_relations),
    ):
        if values.size and (values.min() < 0 or values.max() >= bound):
            raise ValueError(f"{name}.{column_name} lies outside 0..{bound - 1}")
    if src.size > _INT32_MAX:
        raise ValueError(f"{name} has {src.size} edges, outside signed int32")

    def order_for(key: np.ndarray) -> np.ndarray:
        return np.argsort(key, kind="stable")

    dst_order = np.arange(src.size, dtype=np.int64) if dst_sorted else order_for(dst)
    if dst_sorted and dst.size and np.any(dst[1:] < dst[:-1]):
        raise ValueError(f"{name} declares destination-sorted rows but is not sorted")
    src_order = order_for(src)
    rel_order = order_for(relation)

    def take(values: np.ndarray | None, order: np.ndarray) -> Tensor | None:
        return None if values is None else _torch_int32(name, values[order])

    return MessagePlan(
        n_src=int(n_src),
        n_dst=int(n_dst),
        n_relations=int(n_relations),
        n_edges=int(src.size),
        channels=int(channels),
        dst_ptr=_torch_int32(name, _csr(dst[dst_order], n_dst, f"{name}.dst_ptr")),
        dst_src=take(src, dst_order),
        dst_rel=take(relation, dst_order),
        dst_axis=take(axis, dst_order),
        src_ptr=_torch_int32(name, _csr(src[src_order], n_src, f"{name}.src_ptr")),
        src_dst=take(dst, src_order),
        src_rel=take(relation, src_order),
        src_axis=take(axis, src_order),
        rel_ptr=_torch_int32(
            name, _csr(relation[rel_order], n_relations, f"{name}.rel_ptr")
        ),
        rel_src=take(src, rel_order),
        rel_dst=take(dst, rel_order),
        rel_axis=take(axis, rel_order),
    )


def _planned_edges(
    src: np.ndarray,
    dst: np.ndarray,
    relation: np.ndarray,
    axis: np.ndarray | None,
    *,
    n_src: int,
    n_dst: int,
    n_relations: int,
    dst_sorted: bool,
    fully_routed: bool,
    build_axis: bool,
    name: str,
) -> PlannedEdges:
    src = np.ascontiguousarray(src, dtype=np.int64)
    dst = np.ascontiguousarray(dst, dtype=np.int64)
    relation = np.ascontiguousarray(relation, dtype=np.int64)
    axis = None if axis is None else np.ascontiguousarray(axis, dtype=np.int64)
    axis_rows = None
    routed_columns: tuple[np.ndarray | None, ...] = (None, None, None, None)
    axis_plan = None
    if build_axis:
        if axis is None:
            raise ValueError(f"{name} needs an axis plan but has no routes")
        rows = np.arange(src.size, dtype=np.int64) if fully_routed else np.flatnonzero(axis >= 0)
        if not fully_routed:
            axis_rows = np.ascontiguousarray(rows, dtype=np.int64)
            routed_columns = tuple(
                np.ascontiguousarray(values[rows], dtype=np.int64)
                for values in (src, dst, relation, axis)
            )
        axis_plan = _message_plan(
            src[rows],
            dst[rows],
            relation[rows],
            axis[rows],
            n_src=n_src,
            n_dst=n_dst,
            n_relations=n_relations,
            channels=AXIS_CHANNELS,
            dst_sorted=dst_sorted,
            name=f"{name} axis",
        )
    return PlannedEdges(
        src=_torch_int64(src),
        dst=_torch_int64(dst),
        relation=_torch_int64(relation),
        axis=None if axis is None else _torch_int64(axis),
        n_src=int(n_src),
        n_dst=int(n_dst),
        num_relations=int(n_relations),
        dst_sorted=bool(dst_sorted),
        fully_routed=bool(fully_routed),
        name=name,
        axis_rows=None if axis_rows is None else _torch_int64(axis_rows),
        routed_src=(
            None if routed_columns[0] is None else _torch_int64(routed_columns[0])
        ),
        routed_dst=(
            None if routed_columns[1] is None else _torch_int64(routed_columns[1])
        ),
        routed_relation=(
            None if routed_columns[2] is None else _torch_int64(routed_columns[2])
        ),
        routed_axis=(
            None if routed_columns[3] is None else _torch_int64(routed_columns[3])
        ),
        inv_plan=_message_plan(
            src,
            dst,
            relation,
            None,
            n_src=n_src,
            n_dst=n_dst,
            n_relations=n_relations,
            channels=1,
            dst_sorted=dst_sorted,
            name=f"{name} invariant",
        ),
        axis_plan=axis_plan,
    )


def _class_rows(values: np.ndarray, n_classes: int, name: str) -> ClassRowPlan:
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    if values.size and (values.min() < 0 or values.max() >= n_classes):
        raise ValueError(f"{name} classes lie outside 0..{n_classes - 1}")
    order = np.argsort(values, kind="stable")
    ptr = _csr(values[order], n_classes, f"{name}.ptr")
    # The stable order is a permutation and each run is the class its pointer
    # names; assert both independently of the kernel that will consume it.
    if not np.array_equal(np.sort(order), np.arange(values.size)):
        raise ValueError(f"{name} stable class order is not a row permutation")
    for cls in range(n_classes):
        rows = order[int(ptr[cls]) : int(ptr[cls + 1])]
        if rows.size and not np.all(values[rows] == cls):
            raise ValueError(f"{name} CSR run {cls} contains another class")
    return ClassRowPlan(
        n_rows=int(values.size),
        n_classes=int(n_classes),
        ptr=_torch_int32(name, ptr),
        rows=_torch_int32(name, order),
        name=name,
    )


def _source_windows(values: np.ndarray, n_windows: int) -> SourceWindowPlan:
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    if values.size and (values.min() < -1 or values.max() >= n_windows):
        raise ValueError(f"action_window_index lies outside -1..{n_windows - 1}")
    live_rows = np.flatnonzero(values >= 0)
    order = np.argsort(values[live_rows], kind="stable")
    rows = live_rows[order]
    ptr = _csr(values[rows], n_windows, "action source-window ptr")
    sentinel = np.flatnonzero(values < 0)
    for window in range(n_windows):
        run = rows[int(ptr[window]) : int(ptr[window + 1])]
        if run.size and not np.all(values[run] == window):
            raise ValueError(
                f"action source-window CSR run {window} contains another window"
            )
    if sentinel.size and not np.all(values[sentinel] == -1):
        raise ValueError("action source-window sentinel rows contain a live window")
    return SourceWindowPlan(
        n_rows=int(values.size),
        n_windows=int(n_windows),
        ptr=_torch_int32("action source-window ptr", ptr),
        rows=_torch_int32("action source-window rows", rows),
        sentinel_rows=_torch_int32("action sentinel rows", sentinel),
    )


def _row_positions(offsets: np.ndarray, name: str) -> np.ndarray:
    offsets = np.asarray(offsets, dtype=np.int64)
    if offsets.ndim != 1 or offsets.size < 2:
        raise ValueError(f"{name} offsets must be a (P + 1,) vector")
    counts = np.diff(offsets)
    if np.any(counts < 0):
        raise ValueError(f"{name} offsets must be ascending")
    return np.repeat(np.arange(counts.size, dtype=np.int64), counts)


def _latent_segments(
    offsets: tuple[np.ndarray, ...], row_pos: tuple[np.ndarray, ...]
) -> LatentSegments:
    if not offsets or len(offsets) != len(row_pos):
        raise ValueError("latent segments need matching offset and row-position families")
    positions = len(offsets[0]) - 1
    if any(len(value) - 1 != positions for value in offsets):
        raise ValueError("latent segment families disagree on position count")
    totals = np.asarray([value[-1] for value in offsets], dtype=np.int64)
    bases = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(totals)[:-1]))
    starts = np.stack([value[:-1] for value in offsets], axis=1) + bases
    ends = np.stack([value[1:] for value in offsets], axis=1) + bases
    lengths = ends - starts
    ranges = np.stack((starts, ends), axis=-1)
    range_base = np.cumsum(lengths, axis=1) - lengths
    combined = np.concatenate(row_pos)
    return LatentSegments(
        ranges=_torch_int32("latent ranges", ranges),
        range_base=_torch_int32("latent range_base", range_base),
        counts=_torch_int32("latent counts", lengths.sum(axis=1)),
        row_pos=_torch_int64(combined),
        n_rows=int(combined.size),
        positions=int(positions),
        families=int(len(offsets)),
    )


def build_plans(
    cfg: MantisACTConfig,
    *,
    arrays: Mapping[str, np.ndarray],
    cell_offsets: np.ndarray,
    window_offsets: np.ndarray,
    legal_offsets: np.ndarray,
    phase_id: np.ndarray,
) -> ACTPlans:
    """Build every execution plan from one already-collated CPU graph."""

    n_cells = int(cell_offsets[-1])
    n_windows = int(window_offsets[-1])
    n_legal = int(legal_offsets[-1])

    mask = np.asarray(arrays["window_incidence_mask"], dtype=bool)
    windows, slots = np.nonzero(mask)
    cells = arrays["window_cell_index"][windows, slots]
    incidence_relation = arrays["window_incidence_class"][windows, slots]
    incidence_axis = arrays["window_axis"][windows]
    to_windows = _planned_edges(
        cells,
        windows,
        incidence_relation,
        incidence_axis,
        n_src=n_cells,
        n_dst=n_windows,
        n_relations=INCIDENCE_RELATIONS,
        dst_sorted=True,
        fully_routed=True,
        build_axis=cfg.use_axis_channels,
        name="incidence cells->windows",
    )
    to_cells = _planned_edges(
        windows,
        cells,
        incidence_relation,
        incidence_axis,
        n_src=n_windows,
        n_dst=n_cells,
        n_relations=INCIDENCE_RELATIONS,
        dst_sorted=False,
        fully_routed=True,
        build_axis=cfg.use_axis_channels,
        name="incidence windows->cells",
    )

    adjacency = None
    if cfg.use_cell_adjacency:
        relation_id = adjacency_relation_id(cfg)
        relation_count = relation_vocabulary_size(cfg)
        adjacency = _planned_edges(
            arrays["adjacency_src"],
            arrays["adjacency_dst"],
            np.full(arrays["adjacency_src"].shape, relation_id, dtype=np.int64),
            arrays["adjacency_axis"],
            n_src=n_cells,
            n_dst=n_cells,
            n_relations=relation_count,
            dst_sorted=True,
            fully_routed=True,
            build_axis=cfg.use_axis_channels,
            name="hex adjacency",
        )

    radius = None
    if cfg.use_occupied_radius_edges:
        radius_src = arrays["radius_src"]
        occupancy = arrays["cell_occupancy"][radius_src]
        if occupancy.size and np.any((occupancy != 1) & (occupancy != 2)):
            raise ValueError("a radius plan has a source cell that is not occupied")
        relation = 2 * arrays["radius_orbit"] + (occupancy == 2).astype(np.int64)
        radius = _planned_edges(
            radius_src,
            arrays["radius_dst"],
            relation,
            (
                arrays["radius_axis_or_neg1"]
                if cfg.route_on_axis_radius_messages
                else None
            ),
            n_src=n_cells,
            n_dst=n_cells,
            n_relations=radius_relation_count(cfg),
            dst_sorted=True,
            fully_routed=not cfg.route_on_axis_radius_messages,
            build_axis=cfg.use_axis_channels and cfg.route_on_axis_radius_messages,
            name="occupied radius",
        )

    action_rows = ActionRowPlans(
        post1=_class_rows(
            arrays["action_post1_class"], POST1_REL_CLASSES, "action post1"
        ),
        pre_status=_class_rows(
            arrays["action_pre_status"], WINDOW_STATUSES, "action pre_status"
        ),
        source_window=_source_windows(arrays["action_window_index"], n_windows),
    )

    cell_row_pos = _row_positions(cell_offsets, "cell")
    window_row_pos = _row_positions(window_offsets, "window")
    legal_row_pos = _row_positions(legal_offsets, "legal")
    phase_id = np.asarray(phase_id, dtype=np.int64)
    positions = len(cell_offsets) - 1
    if phase_id.shape != (positions,):
        raise ValueError(f"phase_id must be ({positions},), got {phase_id.shape}")

    return ACTPlans(
        state_edges=StatePlans(to_windows, to_cells, adjacency, radius),
        action_rows=action_rows,
        cell_row_pos=_torch_int64(cell_row_pos),
        window_row_pos=_torch_int64(window_row_pos),
        legal_row_pos=_torch_int64(legal_row_pos),
        cell_phase=_torch_int64(phase_id[cell_row_pos]),
        window_phase=_torch_int64(phase_id[window_row_pos]),
        action_phase=_torch_int64(phase_id[legal_row_pos]),
        state_segments=_latent_segments(
            (cell_offsets, window_offsets), (cell_row_pos, window_row_pos)
        ),
        action_segments=_latent_segments((legal_offsets,), (legal_row_pos,)),
    )


_BATCH_PLAN_ARRAYS = (
    "window_incidence_mask",
    "window_cell_index",
    "window_incidence_class",
    "window_axis",
    "adjacency_src",
    "adjacency_dst",
    "adjacency_axis",
    "radius_src",
    "radius_dst",
    "radius_orbit",
    "radius_axis_or_neg1",
    "cell_occupancy",
    "action_post1_class",
    "action_pre_status",
    "action_window_index",
)


def build_plans_from_cpu_batch(cfg: MantisACTConfig, batch) -> ACTPlans:
    """Re-plan a deliberately transformed CPU batch.

    Normal batches are planned inside :func:`packed.collate`; this narrow seam
    exists for law/oracle tests that intentionally relabel or reorder a packed
    batch after collation.  Refusing devices here preserves the rule that plan
    construction never reads device data.
    """

    tensors = {
        name: getattr(batch, name)
        for name in _BATCH_PLAN_ARRAYS
    }
    tensors.update(
        cell_offsets=batch.cell_offsets,
        window_offsets=batch.window_offsets,
        legal_offsets=batch.legal_offsets,
        phase_id=batch.phase_id,
    )
    for name, value in tensors.items():
        if not isinstance(value, Tensor) or value.device.type != "cpu":
            raise ValueError(f"{name} must be a CPU tensor before plan construction")
    return build_plans(
        cfg,
        arrays={
            name: value.numpy()
            for name, value in tensors.items()
            if name in _BATCH_PLAN_ARRAYS
        },
        cell_offsets=batch.cell_offsets.numpy(),
        window_offsets=batch.window_offsets.numpy(),
        legal_offsets=batch.legal_offsets.numpy(),
        phase_id=batch.phase_id.numpy(),
    )


__all__ = [
    "ACTPlans",
    "ActionRowPlans",
    "ClassRowPlan",
    "INCIDENCE_RELATIONS",
    "LatentSegments",
    "PlannedEdges",
    "SourceWindowPlan",
    "StatePlans",
    "adjacency_relation_id",
    "build_plans",
    "build_plans_from_cpu_batch",
    "builder_fingerprint",
    "radius_relation_count",
    "relation_vocabulary_size",
]
