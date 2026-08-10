"""Global state trunk: §18's eleven-step blocks over cells, windows, and latents.

Turns a ``PackedACTBatch`` into per-cell, per-window, and per-position latent
states by running ``state_blocks`` copies of:

1. windows <- messages from window-slot cells
2. cells <- messages from incident windows
3. cells <- hex-adjacency messages
4. cells <- occupied-radius orbit48 messages
5. (optional) windows <- typed window attention (§16)
6-8. latent read / self-attention / broadcast (one ``LatentPass``)
9. cell AxisMix + FFN
10. window AxisMix + FFN
11. phase FiLM per entity type (§13.2)

Owns initial embeddings (§8.2, §9.3) and six final norms (§18).
``debug_forward`` additionally returns intermediate states for D6 tests (§31).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .cells import OCCUPANCY_OPP
from .config import MantisACTConfig
from .equivariant import (
    AXIS_CHANNELS,
    AxisMix,
    EquivariantFFN,
    EquivariantNorm,
    EquivariantState,
    PhaseFiLM,
    activation_module,
    run_equivariant_stage,
)
from .latent_attention import LatentSegments
from .latents import LatentPass, LatentState, RaggedStream, StateLatents
from .messages import (
    INCIDENCE_RELATIONS,
    AdjacencyMessage,
    CellWindowIncidence,
    RadiusMessage,
    TypedEdges,
    TypedWindowAttention,
    WindowWindowEdges,
    adjacency_edges,
    incidence_edges,
    make_relation_embedding,
    radius_edges,
    radius_relation_count,
    relation_vocabulary_size,
    window_window_edges,
)
from .packed import NEAREST_BUCKETS, PackedACTBatch
from .plans import StatePlans, builder_fingerprint
from .pattern_classes import ALL_WINDOW_PATTERN_CLASSES, EMPTY, MIXED
from .windows import WINDOW_NUMERIC_FEATURES

# §8.2's two derived cell vocabularies. Each is the size of the table this
# module indexes, read from the code that emits the column rather than
# restated, so a widened vocabulary is a shape error here instead of a silently
# aliased embedding row. The third, the nearest-stone bucket, needs no name of
# its own: `packed.NEAREST_BUCKETS` is both the table's height and the closed
# ceiling `ACTGraph` refuses the column against.
CELL_OCCUPANCY_CLASSES = OCCUPANCY_OPP + 1
CELL_LEGAL_CLASSES = 2

# §9.3's window status vocabulary, EMPTY through MIXED.
WINDOW_STATUSES = MIXED - EMPTY + 1

# §27: embeddings, relation tables, and learned bases.
EMBEDDING_INIT_STD = 0.02


def _require_width(name: str, block: Tensor, width: int) -> None:
    """Refuse a feature block whose width the configuration did not ask for."""
    if block.ndim != 2:
        raise ValueError(f"{name} must be (N, F), got shape {tuple(block.shape)}")
    if block.shape[1] != width:
        raise ValueError(
            f"{name} carries {block.shape[1]} columns against this "
            f"configuration's {width}"
        )


def refuse_unimplemented_paths(cfg: MantisACTConfig) -> None:
    """Refuse the configurations this trunk cannot honour, naming what is missing.

    Called from every construction path, so no arrangement of this module
    produces a model that silently drops one of them.
    """
    if cfg.use_full_cell_attention:
        raise NotImplementedError(
            "use_full_cell_attention=True is dense attention over every cell, "
            "which is quadratic in the largest node family (§3.14, §26); the "
            "global latents of §17 are the path that replaces it"
        )
    if cfg.phase_conditioning != "film":
        raise NotImplementedError(
            f"phase_conditioning={cfg.phase_conditioning!r} is not implemented: "
            "the state trunk has no token stream to carry a phase token, and "
            "§29 names no preset that asks for one"
        )


# --------------------------------------------------------------------------
# Initial embeddings (§8.2, §9.3)


class CellEmbedding(nn.Module):
    """§8.2: a cell's initial invariant and axis state.

    ```text
    h_inv  = E_occupancy[3] + E_legal[2] + E_nearest_distance[bucket]
    h_axis = one shared learned base, replicated across all three channels
    ```

    Every invariant feature is a D6 invariant of the cell, and the axis half
    starts identical in all three channels: a bare cell has no direction of its
    own, and direction reaches it from the structural messages of §15 and the
    incidence routes of §10 (§8.2). A per-channel base would be a parameter
    attached to an absolute axis (§12.2).

    The three indices are bounded by ``packed._VALUE_RANGES`` and are not
    re-read here; since they are class codes rather than indices into another
    family, ``collate`` does not shift them.
    """

    def __init__(self, cfg: MantisACTConfig) -> None:
        super().__init__()
        self.occupancy = nn.Embedding(CELL_OCCUPANCY_CLASSES, cfg.d_inv)
        self.legal = nn.Embedding(CELL_LEGAL_CLASSES, cfg.d_inv)
        self.nearest = nn.Embedding(NEAREST_BUCKETS, cfg.d_inv)
        for table in (self.occupancy, self.legal, self.nearest):
            nn.init.normal_(table.weight, std=EMBEDDING_INIT_STD)
        self.axis_base = (
            nn.Parameter(torch.randn(cfg.d_axis) * EMBEDDING_INIT_STD)
            if cfg.d_axis
            else None
        )

    def forward(self, batch: PackedACTBatch) -> EquivariantState:
        inv = (
            self.occupancy(batch.cell_occupancy)
            + self.legal(batch.cell_is_legal)
            + self.nearest(batch.cell_nearest_bucket)
        )
        if self.axis_base is None:
            return EquivariantState(inv)
        axis = self.axis_base.to(inv.dtype).expand(
            inv.shape[0], AXIS_CHANNELS, self.axis_base.shape[0]
        )
        return EquivariantState(inv, axis)


class WindowEmbedding(nn.Module):
    """§9.3: a window's initial invariant and axis state.

    ```text
    h_inv     = E_window_pattern[378] + E_window_status[4] + MLP(counts and runs)
    h_axis[a] = shared neutral base + [a == native_axis] * W_native(h_inv)
    ```

    The pattern class is canonical under slot reversal and the status and
    counts are functions of it, so the whole invariant half is a D6 invariant.
    The pattern is projected into the native-axis channel with the other
    channels at a shared neutral value: one shared projection produces one
    vector, and only the channel it lands in depends on the window's axis, so
    no parameter is ever selected by an axis id (§12.1).

    ``use_window_numeric_features=False`` makes the numeric block absent rather
    than a column of zeros a projection would still read; ``window_numeric`` is
    then checked to be zero-width.

    The three indices are bounded by ``packed._VALUE_RANGES`` and are not
    re-read here; none is an index into another family, so ``collate`` does
    not shift them.
    """

    def __init__(self, cfg: MantisACTConfig) -> None:
        super().__init__()
        self.pattern = nn.Embedding(ALL_WINDOW_PATTERN_CLASSES, cfg.d_inv)
        self.status = nn.Embedding(WINDOW_STATUSES, cfg.d_inv)
        for table in (self.pattern, self.status):
            nn.init.normal_(table.weight, std=EMBEDDING_INIT_STD)

        self.numeric_width = (
            WINDOW_NUMERIC_FEATURES if cfg.use_window_numeric_features else 0
        )
        self.numeric = (
            nn.Sequential(
                nn.Linear(self.numeric_width, cfg.d_inv),
                activation_module(cfg.activation),
                nn.Linear(cfg.d_inv, cfg.d_inv),
            )
            if self.numeric_width
            else None
        )
        if cfg.d_axis:
            self.axis_base = nn.Parameter(torch.randn(cfg.d_axis) * EMBEDDING_INIT_STD)
            # One shared projection of the invariant state into whichever channel
            # the window's own axis is. Bias-free: a bias would be a second copy
            # of the neutral base, on the native channel only.
            self.to_native = nn.Linear(cfg.d_inv, cfg.d_axis, bias=False)
        else:
            self.axis_base = None
            self.to_native = None

    def forward(self, batch: PackedACTBatch) -> EquivariantState:
        _require_width("window_numeric", batch.window_numeric, self.numeric_width)

        inv = self.pattern(batch.window_pattern_class) + self.status(batch.window_status)
        if self.numeric is not None:
            inv = inv + self.numeric(batch.window_numeric.to(inv.dtype))
        if self.axis_base is None:
            return EquivariantState(inv)

        native = F.one_hot(batch.window_axis, AXIS_CHANNELS).to(inv.dtype)
        axis = self.axis_base.to(inv.dtype) + native.unsqueeze(-1) * self.to_native(
            inv
        ).unsqueeze(-2)
        return EquivariantState(inv, axis)


# --------------------------------------------------------------------------
# The edge families a block runs over


@dataclass(frozen=True, eq=False)
class StateEdges:
    """The typed edge families of one batch, built once and reused per block.

    Building them once per forward rather than once per block keeps the edge
    derivations — the incidence traversal, the radius family's relation join,
    the window-pair join, and each family's routed subset — off the per-block
    path, along with the CSR views `TypedEdges.plan` caches on them.
    ``adjacency``, ``radius`` and ``window_window`` are ``None`` exactly when
    the configuration disables their path.
    """

    to_windows: TypedEdges
    to_cells: TypedEdges
    adjacency: TypedEdges | None
    radius: TypedEdges | None
    window_window: WindowWindowEdges | None


def state_edges(batch: PackedACTBatch, cfg: MantisACTConfig) -> StateEdges | StatePlans:
    """The §18 edge families of ``batch`` under ``cfg`` (§10, §15, §16)."""
    expected = builder_fingerprint(cfg)
    if batch.plans is not None:
        if batch.builder_fingerprint != expected:
            raise ValueError(
                f"the batch was planned for builder config "
                f"{batch.builder_fingerprint!r}, but this state trunk expects "
                f"{expected!r}; rebuild it with collate(graphs, cfg)"
            )
        if cfg.window_window_mode != "typed_collinear_crossing":
            return batch.plans.state_edges
    to_windows, to_cells = incidence_edges(batch)
    typed_windows = cfg.window_window_mode == "typed_collinear_crossing"
    return StateEdges(
        to_windows=to_windows,
        to_cells=to_cells,
        adjacency=adjacency_edges(batch, cfg) if cfg.use_cell_adjacency else None,
        radius=radius_edges(batch, cfg) if cfg.use_occupied_radius_edges else None,
        window_window=window_window_edges(batch) if typed_windows else None,
    )


class _RelationTables(nn.Module):
    """The three relation vocabularies, shared across blocks (§14).

    `share_relation_embeddings_across_blocks` shares these embeddings while
    projections and update MLPs stay block-private. The three families have
    different vocabularies — §10.1's joint incidence classes, the geometry
    classes a hex step belongs to, and §15.2's joint (geometry, source colour)
    product — so they are always three separate tables.
    """

    def __init__(self, cfg: MantisACTConfig) -> None:
        super().__init__()
        self.incidence = make_relation_embedding(INCIDENCE_RELATIONS, cfg.d_rel)
        self.adjacency = make_relation_embedding(relation_vocabulary_size(cfg), cfg.d_rel)
        self.radius = make_relation_embedding(radius_relation_count(cfg), cfg.d_rel)


# --------------------------------------------------------------------------
# The block (§18)


class StateTrunkBlock(nn.Module):
    """One of §18's blocks: the eleven steps, in their given order.

    Steps 6-8 are `latents.LatentPass`, supplied by the caller since the latent
    bases live on the stack rather than on a block. Everything else is held
    here; every stage is a pre-norm residual branch that owns its norms and
    its LayerScale (§18, §27), with separate norms per entity type and per
    stream. Step 11's FiLM is one modulation per entity type, applied to the
    block's accumulated state (§13.2's "in every state block") rather than to
    each stage's own residual branch.
    """

    def __init__(
        self, cfg: MantisACTConfig, *, relations: _RelationTables | None = None
    ) -> None:
        super().__init__()
        refuse_unimplemented_paths(cfg)
        self.cfg = cfg

        # 1, 2: both directions of the cell<->window incidence.
        self.incidence = CellWindowIncidence(
            cfg, relation_embedding=None if relations is None else relations.incidence
        )
        # 3: hex adjacency between cells.
        self.adjacency = (
            AdjacencyMessage(
                cfg,
                relation_embedding=None if relations is None else relations.adjacency,
            )
            if cfg.use_cell_adjacency
            else None
        )
        # 4: every stone within the radius, typed by its exact D6 orbit.
        self.radius = (
            RadiusMessage(
                cfg, relation_embedding=None if relations is None else relations.radius
            )
            if cfg.use_occupied_radius_edges
            else None
        )
        # 5: §16's optional direct window-to-window path.
        self.window_attention = (
            TypedWindowAttention(cfg)
            if cfg.window_window_mode == "typed_collinear_crossing"
            else None
        )
        # 9, 10: entity-specific cross-channel mixing and feed-forward.
        self.cell_mix = AxisMix(cfg)
        self.cell_ffn = EquivariantFFN(cfg)
        self.window_mix = AxisMix(cfg)
        self.window_ffn = EquivariantFFN(cfg)
        # 11: phase conditioning, identity at initialisation (§13.2, §27).
        self.cell_film = PhaseFiLM(cfg)
        self.window_film = PhaseFiLM(cfg)

    def forward(
        self,
        cells: EquivariantState,
        windows: EquivariantState,
        latents: LatentState,
        *,
        edges: StateEdges | StatePlans,
        latent_pass: LatentPass,
        latent_segments: LatentSegments | None = None,
        cell_offsets: Tensor,
        cell_row_pos: Tensor,
        window_offsets: Tensor,
        window_row_pos: Tensor,
        cell_phase: Tensor,
        window_phase: Tensor,
    ) -> tuple[EquivariantState, EquivariantState, LatentState]:
        """The three families after this block's eleven steps."""
        windows = self.incidence.to_windows(edges.to_windows, cells, windows)
        cells = self.incidence.to_cells(edges.to_cells, windows, cells)

        if self.adjacency is not None:
            if edges.adjacency is None:
                raise ValueError(
                    "this block runs hex adjacency but the edge set carries no "
                    "adjacency family: the config that built the block and the "
                    "one that built the edges disagree on use_cell_adjacency"
                )
            cells = self.adjacency(edges.adjacency, cells)
        if self.radius is not None:
            if edges.radius is None:
                raise ValueError(
                    "this block runs occupied-radius messages but the edge set "
                    "carries no radius family: the config that built the block "
                    "and the one that built the edges disagree on "
                    "use_occupied_radius_edges"
                )
            cells = self.radius(edges.radius, cells)

        if self.window_attention is not None:
            if edges.window_window is None:
                raise ValueError(
                    "this block runs typed window attention but the edge set "
                    "carries no window-pair family: the config that built the "
                    "block and the one that built the edges disagree on "
                    "window_window_mode"
                )
            windows = self.window_attention(edges.window_window, windows)

        latents, entities = latent_pass(
            latents,
            {
                "cell": RaggedStream(cells, cell_offsets, cell_row_pos),
                "window": RaggedStream(windows, window_offsets, window_row_pos),
            },
            segments=latent_segments,
        )
        cells, windows = entities["cell"].state, entities["window"].state

        cells = run_equivariant_stage(
            cells,
            self.cell_mix,
            self.cell_ffn,
            film=self.cell_film,
            phase_id=cell_phase,
        )
        windows = run_equivariant_stage(
            windows,
            self.window_mix,
            self.window_ffn,
            film=self.window_film,
            phase_id=window_phase,
        )
        return cells, windows, latents


# --------------------------------------------------------------------------
# The trunk


@dataclass(frozen=True, eq=False)
class TrunkOutput:
    """What the state trunk answers: §18's three final families.

    ``cells`` and ``windows`` are `EquivariantState` over the batch's flat node
    rows, in the batch frame the packed offsets describe; ``latents`` is the
    per-position `LatentState`. Equality is identity: the fields are tensors.
    """

    cells: EquivariantState
    windows: EquivariantState
    latents: LatentState
    position_count: int


class _Trace:
    """The debug forward's collector (§31): selected sites, nothing else.

    A site name records up to two tensors, ``<site>.inv`` and ``<site>.axis``,
    and a stream the configuration removes records neither. Membership is
    checked before the store, so asking for one block's cells does not
    materialise a reference to every other tensor in the forward.
    """

    def __init__(self, sites: frozenset[str]) -> None:
        self.sites = sites
        self.tensors: dict[str, Tensor] = {}

    def state(self, site: str, state: EquivariantState) -> None:
        if site not in self.sites:
            return
        self.tensors[f"{site}.inv"] = state.inv
        if state.axis is not None:
            self.tensors[f"{site}.axis"] = state.axis

    def latents(self, site: str, latents: LatentState) -> None:
        if site not in self.sites:
            return
        if latents.inv is not None:
            self.tensors[f"{site}.inv"] = latents.inv
        if latents.axis is not None:
            self.tensors[f"{site}.axis"] = latents.axis


class StateTrunk(nn.Module):
    """§18: the initial embeddings, the state blocks, and the six final norms.

    ```python
    trunk = StateTrunk(cfg)
    out = trunk(batch)                       # production
    out, tensors = trunk.debug_forward(batch)  # §31's intermediate states
    ```

    The forward is linear in every node and edge family (§3.14, §26): one gather
    and one segment reduction per edge family per block, and one pass over the
    nodes per latent read and broadcast. Nothing is quadratic in cells, windows,
    or actions, and no path here loops over nodes in Python.
    """

    def __init__(self, cfg: MantisACTConfig) -> None:
        super().__init__()
        refuse_unimplemented_paths(cfg)
        self.cfg = cfg
        self.builder_fingerprint = builder_fingerprint(cfg)

        self.cell_embedding = CellEmbedding(cfg)
        self.window_embedding = WindowEmbedding(cfg)

        # Held on the trunk as well as on the blocks that read them, so the
        # sharing is visible in the module tree; `parameters()` deduplicates by
        # identity, so a shared table is counted once.
        self.relations = (
            _RelationTables(cfg) if cfg.share_relation_embeddings_across_blocks else None
        )
        self.blocks = nn.ModuleList(
            StateTrunkBlock(cfg, relations=self.relations)
            for _ in range(cfg.state_blocks)
        )
        self.latents = StateLatents(cfg)

        # §18: separate final norms per entity type and stream, never one reused.
        self.final_cell = EquivariantNorm(cfg)
        self.final_window = EquivariantNorm(cfg)
        self.final_latent_inv = (
            nn.LayerNorm(cfg.d_inv) if cfg.num_inv_latents else None
        )
        self.final_latent_axis = (
            nn.LayerNorm(cfg.d_axis) if cfg.num_axis_latents else None
        )

    # --- the debug surface (§31) -------------------------------------------

    def debug_sites(self) -> tuple[str, ...]:
        """Every site `debug_forward` can expose, in forward order."""
        names = ["input"]
        names += [f"block{index}" for index in range(len(self.blocks))]
        names.append("final")
        return tuple(
            f"{site}.{entity}" for site in names for entity in ("cell", "window", "latent")
        )

    def _sites(self, capture: Sequence[str] | None) -> frozenset[str]:
        available = self.debug_sites()
        if capture is None:
            return frozenset(available)
        requested = tuple(capture)
        unknown = [name for name in requested if name not in available]
        if unknown:
            raise ValueError(
                f"unknown debug site(s) {unknown}; this trunk exposes "
                f"{list(available)}"
            )
        return frozenset(requested)

    # --- the forward -------------------------------------------------------

    def forward(self, batch: PackedACTBatch) -> TrunkOutput:
        """The trunk's three final families. Intermediates are not computed."""
        return self._run(batch, None)

    def debug_forward(
        self, batch: PackedACTBatch, capture: Sequence[str] | None = None
    ) -> tuple[TrunkOutput, dict[str, Tensor]]:
        """The same forward, plus the selected intermediate tensors (§31).

        ``capture`` names sites from :meth:`debug_sites` and defaults to all of
        them. The returned dict is keyed ``<site>.inv`` and ``<site>.axis``; a
        stream the configuration removes contributes no key. The result is
        numerically identical to :meth:`forward` — the collector reads the same
        tensors rather than recomputing anything.
        """
        trace = _Trace(self._sites(capture))
        return self._run(batch, trace), trace.tensors

    def _run(self, batch: PackedACTBatch, trace: _Trace | None) -> TrunkOutput:
        if batch.plans is None:
            raise ValueError(
                "PackedACTBatch.plans is missing; build batches with "
                "collate(graphs, cfg) so execution plans are made on the CPU"
            )
        if batch.builder_fingerprint != self.builder_fingerprint:
            raise ValueError(
                f"the batch was planned for builder config "
                f"{batch.builder_fingerprint!r}, but this state trunk expects "
                f"{self.builder_fingerprint!r}; rebuild it with "
                "collate(graphs, model.cfg)"
            )
        plans = batch.plans
        cells = self.cell_embedding(batch)
        windows = self.window_embedding(batch)
        latents = self.latents.initial(batch.global_numeric)
        edges = state_edges(batch, self.cfg)

        # Which position owns each cell row and each window row, built once for
        # the whole forward: the latent read, the latent broadcast, and the
        # phase gather below all want the same two vectors, and they depend on
        # the batch alone. The row counts come off the families' own tables,
        # which is what makes `row_positions` sync-free (see its docstring).
        cell_row_pos = plans.cell_row_pos
        window_row_pos = plans.window_row_pos

        # §13.1's phase is a per-position scalar; every node of a position
        # carries its own position's, gathered once for the whole forward.
        cell_phase = plans.cell_phase
        window_phase = plans.window_phase

        if trace is not None:
            trace.state("input.cell", cells)
            trace.state("input.window", windows)
            trace.latents("input.latent", latents)

        for index, block in enumerate(self.blocks):
            cells, windows, latents = block(
                cells,
                windows,
                latents,
                edges=edges,
                latent_pass=self.latents[index],
                latent_segments=plans.state_segments,
                cell_offsets=batch.cell_offsets,
                cell_row_pos=cell_row_pos,
                window_offsets=batch.window_offsets,
                window_row_pos=window_row_pos,
                cell_phase=cell_phase,
                window_phase=window_phase,
            )
            if trace is not None:
                trace.state(f"block{index}.cell", cells)
                trace.state(f"block{index}.window", windows)
                trace.latents(f"block{index}.latent", latents)

        cells = self.final_cell(cells)
        windows = self.final_window(windows)
        latents = LatentState(
            inv=None if self.final_latent_inv is None else self.final_latent_inv(latents.inv),
            axis=(
                None
                if self.final_latent_axis is None
                else self.final_latent_axis(latents.axis)
            ),
        )
        if trace is not None:
            trace.state("final.cell", cells)
            trace.state("final.window", windows)
            trace.latents("final.latent", latents)

        return TrunkOutput(
            cells=cells,
            windows=windows,
            latents=latents,
            position_count=int(batch.position_count),
        )


__all__ = [
    "CELL_LEGAL_CLASSES",
    "CELL_OCCUPANCY_CLASSES",
    "WINDOW_STATUSES",
    "CellEmbedding",
    "StateEdges",
    "StateTrunk",
    "StateTrunkBlock",
    "TrunkOutput",
    "WindowEmbedding",
    "refuse_unimplemented_paths",
    "state_edges",
]
