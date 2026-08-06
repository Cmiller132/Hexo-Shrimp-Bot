"""The global state trunk: §18's blocks over cells, windows, and latents.

This module turns a :class:`~mantis_act.packed.PackedACTBatch` into the three
families every downstream stage reads — a state per cell, a state per window,
and the per-position latents — by running ``state_blocks`` copies of §18's
eleven-step block. It owns the initial embeddings of §8.2 and §9.3, the block
itself, and the six separate final norms §18 requires; everything a step does
is one of the modules `equivariant`, `messages`, and `latents` already export,
and nothing here reimplements one.

The eleven steps, in the order §18 fixes them:

```text
 1. windows <- relation-gated messages from represented window-slot cells
 2. cells   <- relation-gated messages from incident windows
 3. cells   <- local hex-adjacency messages
 4. cells   <- occupied-radius orbit48 messages
 5. optional windows <- typed window attention          (refused; see below)
 6. state latents read cells and windows
 7. latent self-attention and invariant/axis mixing
 8. latents broadcast to cells and windows
 9. cell-specific AxisMix + FFN
10. window-specific AxisMix + FFN
11. phase FiLM
```

Steps 6 to 8 are one `latents.LatentPass`, which is where they live; the pass
belongs to the stack that owns the latent bases, so a block is handed the pass
for its own depth rather than holding it.

Every stage is a pre-norm residual branch that owns its norms and its
LayerScale, so the block is a chain of state-to-state maps and adds no delta
itself. After the last block six final norms run — cell invariant, cell axis,
window invariant, window axis, latent invariant, latent axis — each its own
`nn.LayerNorm`, because §18 says in as many words not to reuse one final norm
across entity types.

Why the trunk is equivariant (§12.1). The two embeddings are the only new
construction here, and each is checked against §12.2 explicitly:

- A cell's invariant features are occupancy, legality, and a nearest-stone
  bucket, all D6 invariants, and its three axis channels start from one shared
  learned base replicated across them (§8.2). No absolute axis reaches a
  parameter.
- A window's invariant features are its reversal-canonical pattern class, its
  status, and its normalised counts and runs — again all invariants. Its axis
  tensor is a shared neutral base in every channel plus **the window's own
  invariant state, projected by one shared matrix into the channel of its
  native axis** (§9.3). That is §12.3's "route line messages into the
  structural native axis": the projected vector does not depend on which axis
  the window lies on, only its destination channel does, and the native axis
  permutes with the board. An `nn.Embedding(3, ...)` indexed by the native axis
  would be the forbidden construction; a one-hot scatter of one shared vector
  is not, and the tests separate the two by permuting the channels and the
  structural axis labels together.

Everything downstream of the embeddings inherits equivariance from the modules
it composes, each of which carries its own argument.

Two readings of §18 this module fixes, since the text admits more than one:

- **Step 11 is one FiLM per entity type per block**, applied to the state the
  block's ten preceding stages have written. §18 words it "phase FiLM on every
  residual branch"; every stage here owns its residual and its LayerScale
  internally, so there is no exposed branch delta to modulate, and §13.2's own
  wording — "in every state block" — is what is implemented. Cells and windows
  each get their own modulation; the latents take their phase through the nodes
  they read, which are already conditioned.
- **The initial embeddings of §8.2 and §9.3 live here**, because they are the
  trunk's input and no other module produces them.

What this trunk refuses, loudly rather than quietly (§16, §13.2, §3.14):

- ``window_window_mode="typed_collinear_crossing"`` (§16, preset
  ``full_with_typed_window_attention``). The typed collinear/crossing path
  needs a window-to-window edge family, and deriving one needs each window's
  identity ``(native_axis, start_q, start_r)`` and its position — neither of
  which `PackedACTBatch` carries, because §7 keeps coordinates and window
  identities out of the model's input. Enumerating the pairs inside the model
  instead would be quadratic in windows, which §3.14 forbids outright. The
  missing input is named in the refusal.
- ``phase_conditioning="token_only"``. §13.2 offers it as a toggle and §29 asks
  for no preset that uses it; the state trunk has no token stream to put a
  phase token in, and inventing one would be building for a requirement that
  does not exist.
- ``use_full_cell_attention=True``. Dense attention over all cells is quadratic
  in the largest node family; §16 and §26 make the latents the global path
  precisely so that it is not needed.

The debug forward (§31). `StateTrunk.forward` returns the three final families
and nothing else. `StateTrunk.debug_forward` returns them together with a dict
of selected intermediate tensors — the cell, window, and latent invariant and
axis states at the input, after each block, and after the final norms — which
is what the D6 intermediate-state tests of §31.4 to §31.7 read. The production
path carries a ``None`` in place of the collector, so it pays one identity
comparison per site and stores nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .cells import NEAREST_BUCKETS, OCCUPANCY_OPP
from .config import MantisACTConfig
from .equivariant import (
    AXIS_CHANNELS,
    AxisMix,
    EquivariantFFN,
    EquivariantNorm,
    EquivariantState,
    PhaseFiLM,
    activation_module,
)
from .latents import LatentPass, LatentState, RaggedStream, StateLatents, row_positions
from .messages import (
    INCIDENCE_RELATIONS,
    AdjacencyMessage,
    CellWindowIncidence,
    RadiusMessage,
    TypedEdges,
    adjacency_edges,
    incidence_edges,
    make_relation_embedding,
    radius_edges,
    radius_relation_count,
    relation_vocabulary_size,
)
from .packed import PackedACTBatch
from .pattern_classes import ALL_WINDOW_PATTERN_CLASSES, EMPTY, MIXED
from .windows import WINDOW_NUMERIC_FEATURES

# §8.2's three cell vocabularies. Each is the size of the table the builder
# indexes, read from that builder rather than restated, so a widened bucket
# range is a shape error here instead of a silently aliased embedding row.
CELL_OCCUPANCY_CLASSES = OCCUPANCY_OPP + 1
CELL_LEGAL_CLASSES = 2
CELL_NEAREST_BUCKETS = NEAREST_BUCKETS

# §9.3's window status vocabulary, EMPTY through MIXED.
WINDOW_STATUSES = MIXED - EMPTY + 1

# §27: embeddings, relation tables, and learned bases.
EMBEDDING_INIT_STD = 0.02


def _require_range(name: str, values: Tensor, high: int) -> None:
    """Refuse a class index outside ``0..high - 1``, naming the row (§27, house).

    An out-of-range index into an embedding either raises deep inside ATen with
    no field name or, for a negative one, silently reads the far end of the
    table and returns a plausible wrong row. Both are checked here instead,
    once per family per forward.
    """
    if values.numel() == 0:
        return
    low, top = int(values.min()), int(values.max())
    if low < 0:
        raise ValueError(f"{name} must be >= 0: found {low} at row {int(values.argmin())}")
    if top >= high:
        raise ValueError(
            f"{name} must be < {high}: found {top} at row {int(values.argmax())}"
        )


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
    if cfg.window_window_mode == "typed_collinear_crossing":
        raise NotImplementedError(
            "window_window_mode='typed_collinear_crossing' (§16) needs a typed "
            "window-to-window edge family, and PackedACTBatch carries neither "
            "window_id nor a per-window position to derive one from: §7 keeps "
            "window identities and coordinates out of the model's input. "
            "Enumerating the pairs in the model instead would be quadratic in "
            "windows, which §3.14 forbids. The builder must emit "
            "window_window_src/dst/relation edges before this path can exist"
        )
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
    starts identical in all three channels because a bare cell has no direction
    of its own: direction reaches it from the structural messages of §15 and the
    incidence routes of §10, which is exactly what §8.2 says. A per-channel base
    would be a parameter attached to an absolute axis (§12.2).
    """

    def __init__(self, cfg: MantisACTConfig) -> None:
        super().__init__()
        self.occupancy = nn.Embedding(CELL_OCCUPANCY_CLASSES, cfg.d_inv)
        self.legal = nn.Embedding(CELL_LEGAL_CLASSES, cfg.d_inv)
        self.nearest = nn.Embedding(CELL_NEAREST_BUCKETS, cfg.d_inv)
        for table in (self.occupancy, self.legal, self.nearest):
            nn.init.normal_(table.weight, std=EMBEDDING_INIT_STD)
        self.axis_base = (
            nn.Parameter(torch.randn(cfg.d_axis) * EMBEDDING_INIT_STD)
            if cfg.d_axis
            else None
        )

    def forward(self, batch: PackedACTBatch) -> EquivariantState:
        _require_range("cell_occupancy", batch.cell_occupancy, CELL_OCCUPANCY_CLASSES)
        _require_range("cell_is_legal", batch.cell_is_legal, CELL_LEGAL_CLASSES)
        _require_range(
            "cell_nearest_bucket", batch.cell_nearest_bucket, CELL_NEAREST_BUCKETS
        )
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

    The pattern class is canonical under slot reversal and the status and the
    counts are functions of it, so the whole invariant half is a D6 invariant.
    §9.3 then asks for the pattern to be projected into the native-axis channel
    with the other channels at a shared neutral value, which is what the mask
    does: one shared projection produces one vector, and only the channel it
    lands in depends on the window's axis. Under a board transform the window's
    native axis moves by the induced permutation and the vector moves with it,
    which is the §12.1 law; no parameter is ever selected by an axis id.

    ``use_window_numeric_features=False`` makes the numeric block absent rather
    than a column of zeros a projection would still read, so the batch's
    ``window_numeric`` is checked to be zero-width in that arm.
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
        _require_range(
            "window_pattern_class", batch.window_pattern_class, ALL_WINDOW_PATTERN_CLASSES
        )
        _require_range("window_status", batch.window_status, WINDOW_STATUSES)
        _require_range("window_axis", batch.window_axis, AXIS_CHANNELS)
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

    Every family validates its own indices at construction (`TypedEdges`), so
    building them once per forward rather than once per block also means the
    bounds are checked once. ``adjacency`` and ``radius`` are ``None`` exactly
    when the configuration disables their path.
    """

    to_windows: TypedEdges
    to_cells: TypedEdges
    adjacency: TypedEdges | None
    radius: TypedEdges | None


def state_edges(batch: PackedACTBatch, cfg: MantisACTConfig) -> StateEdges:
    """The §18 edge families of ``batch`` under ``cfg`` (§10, §15)."""
    to_windows, to_cells = incidence_edges(batch)
    return StateEdges(
        to_windows=to_windows,
        to_cells=to_cells,
        adjacency=adjacency_edges(batch, cfg) if cfg.use_cell_adjacency else None,
        radius=radius_edges(batch, cfg) if cfg.use_occupied_radius_edges else None,
    )


class _RelationTables(nn.Module):
    """The three relation vocabularies, shared across blocks (§14).

    §14 lets relation embeddings be shared while projections and update MLPs
    stay block-private, and `share_relation_embeddings_across_blocks` selects
    it. The three families have three different vocabularies — §10.1's 2187
    joint incidence classes, the geometry classes a hex step belongs to, and
    §15.2's joint (geometry, source colour) product — so they are three tables
    and never one.
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

    Steps 6 to 8 are `latents.LatentPass`, which the caller supplies because the
    latent bases live on the stack rather than on a block. Everything else is
    held here, and every stage is a pre-norm residual branch that owns its
    norms and its LayerScale (§18, §27): cells and windows have separate norms
    at every site because they are separate entity types, and every module's
    invariant and axis halves have separate norms because they are separate
    streams.

    Step 11's FiLM is one modulation per entity type, applied to the block's
    accumulated state at the position §18 gives it — which is also what §13.2
    asks for, "in every state block". Every stage above owns its own residual
    and its own LayerScale, so there is no branch delta for a FiLM to sit on
    without reaching inside those modules; modulating the state the branches
    have written to is the same conditioning applied once instead of ten times.
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
        edges: StateEdges,
        latent_pass: LatentPass,
        cell_offsets: Tensor,
        window_offsets: Tensor,
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

        # 5 is absent by construction: `refuse_unimplemented_paths` rejects the
        # only mode that asks for it.

        latents, entities = latent_pass(
            latents,
            {
                "cell": RaggedStream(cells, cell_offsets),
                "window": RaggedStream(windows, window_offsets),
            },
        )
        cells, windows = entities["cell"].state, entities["window"].state

        cells = self.cell_ffn(self.cell_mix(cells))
        windows = self.window_ffn(self.window_mix(windows))

        cells = self.cell_film(cells, cell_phase)
        windows = self.window_film(windows, window_phase)
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
        cells = self.cell_embedding(batch)
        windows = self.window_embedding(batch)
        latents = self.latents.initial(batch.global_numeric)
        edges = state_edges(batch, self.cfg)

        # §13.1's phase is a per-position scalar; every node of a position
        # carries its own position's, gathered once for the whole forward.
        cell_phase = batch.phase_id.index_select(0, row_positions(batch.cell_offsets))
        window_phase = batch.phase_id.index_select(
            0, row_positions(batch.window_offsets)
        )

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
                cell_offsets=batch.cell_offsets,
                window_offsets=batch.window_offsets,
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
    "CELL_NEAREST_BUCKETS",
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
