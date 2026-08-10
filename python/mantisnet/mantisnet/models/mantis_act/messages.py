"""Relation-gated messages over typed sparse edges (§14-§16).

One generic module covers cell<->window incidence, hex adjacency, and
occupied-radius orbit edges; ``TypedWindowAttention`` (§16) is separate
because its segment softmax cannot share the fused single-walk reduction.

Message: ``sigmoid(Wg(E_rel[r])) * Wv(LN(src)) + Wb(E_rel[r])``, summed by
destination in fp32.  Reductions: ``sum`` (default), ``mean``, ``attention``
(§14 ablations); ``incidence_message="additive"`` drops the gate (§29).

Axis-stream parameters are shared across channels; a relation id is a D6
invariant; edges on no axis update the invariant stream only (§12.1).
Each module is a pre-norm residual branch over ``EquivariantState`` with
its own LayerScale (§27).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from ...window_pairs import WA_CLASSES, derive_pair_tables, edge_attention
from .cells import OCCUPANCY_OPP
from .config import MantisACTConfig
from .equivariant import (
    AXIS_CHANNELS,
    EquivariantNorm,
    EquivariantResidual,
    EquivariantState,
    LayerScale,
    activation_module,
    at_least_fp32,
)
from .latent_attention import row_positions
from .packed import PackedACTBatch
from .plans import (
    INCIDENCE_RELATIONS,
    PlannedEdges,
    adjacency_relation_id,
    radius_relation_count,
    relation_vocabulary_size,
)
from .segment_message import MessagePlan, message_plan, relation_gated_message

# §16's typed collinear/crossing vocabulary: 11 collinear offsets, 36 crossing
# fold products, and the self loop. Read off `window_pairs`, which generates the
# classes, so this is the size of the table indexed rather than a second
# statement of it.
WINDOW_WINDOW_RELATIONS = WA_CLASSES

# §27: embeddings, relation tables, and latent bases.
EMBEDDING_INIT_STD = 0.02

_REDUCTIONS = ("sum", "mean", "attention")


def make_relation_embedding(num_relations: int, d_rel: int) -> nn.Embedding:
    """A relation table initialised ``N(0, 0.02)`` (§27).

    The trunk builds one per vocabulary and hands it to every block when
    ``share_relation_embeddings_across_blocks`` is set; the projections and
    update MLPs that read it stay block-private either way (§14).
    """
    if num_relations < 1:
        raise ValueError(f"num_relations must be at least 1, got {num_relations}")
    if d_rel < 1:
        raise ValueError(f"d_rel must be at least 1, got {d_rel}")
    table = nn.Embedding(num_relations, d_rel)
    nn.init.normal_(table.weight, std=EMBEDDING_INIT_STD)
    return table


# --------------------------------------------------------------------------
# Segment reductions (§27: every one of them in fp32)


def segment_sum(values: Tensor, index: Tensor, n_segments: int) -> Tensor:
    """Sum ``(E, D)`` rows into ``(n_segments, D)`` by ``index``, in fp32."""
    if values.ndim != 2:
        raise ValueError(f"values must be (E, D), got shape {tuple(values.shape)}")
    if index.shape != values.shape[:1]:
        raise ValueError(
            f"index must be ({values.shape[0]},), got shape {tuple(index.shape)}"
        )
    out = values.new_zeros((n_segments, values.shape[1]), dtype=torch.float32)
    return out.index_add_(0, index, values.float())


def segment_softmax(scores: Tensor, index: Tensor, n_segments: int) -> Tensor:
    """Softmax ``(E,)`` scores within each segment, in fp32 (§27).

    A segment owning no row is left alone: its shift is ``-inf`` and its total
    is zero, and no edge indexes it, so neither reaches an output.
    """
    scores = scores.float()
    shift = torch.full(
        (n_segments,), float("-inf"), dtype=torch.float32, device=scores.device
    ).scatter_reduce_(0, index, scores, reduce="amax", include_self=True)
    weights = (scores - shift.index_select(0, index)).exp()
    total = torch.zeros(
        n_segments, dtype=torch.float32, device=scores.device
    ).index_add_(0, index, weights)
    return weights / total.index_select(0, index)


def attention_by_destination(
    messages: Tensor, index: Tensor, n_segments: int, score: Tensor
) -> Tensor:
    """§14's attention reduction of per-edge messages, in fp32.

    Unlike ``sum`` and ``mean``, a segment softmax needs every edge's score
    before any destination's weights are known, so this is the one reduction
    with an explicit ``(E, d)`` tensor rather than the fused walk.
    """
    weights = segment_softmax(score, index, n_segments)
    return segment_sum(messages.float() * weights.unsqueeze(1), index, n_segments)


# --------------------------------------------------------------------------
# Typed edge sets


@dataclass(frozen=True, eq=False)
class TypedEdges:
    """One typed sparse edge family, with the structure its kernels reduce over.

    ``src`` and ``dst`` index the source and destination node families in the
    batch frame; ``relation`` indexes the family's relation vocabulary; ``axis``
    is the structural axis an edge routes its line message through, ``-1`` for
    an edge on no axis, or ``None`` for a family that routes no axis message at
    all.

    ``dst_sorted`` and ``fully_routed`` are structural properties of the family
    rather than measurements of its data, stated host-side by the builder
    rather than probed from the device:

    - ``dst_sorted``: the rows arrive destination-ascending, so
      `segment_message.message_plan` adopts that order instead of sorting.
    - ``fully_routed``: every row carries a real axis, so ``routed()`` may hand
      its columns on untouched; a family that may carry ``-1`` pays one subset
      gather.

    Index bounds are the packer's, checked in numpy before a tensor exists
    (``_VALUE_RANGES``/``_INDEX_FIELDS``) and re-derived after concatenation by
    ``collate``'s ``_refuse_crossing``. What this dataclass validates is
    everything a host can see for free: container, dtype, rank, row agreement,
    device, and sizes. ``name`` is the family's name in an error message.
    """

    src: Tensor
    dst: Tensor
    relation: Tensor
    axis: Tensor | None
    n_src: int
    n_dst: int
    num_relations: int
    dst_sorted: bool
    fully_routed: bool
    name: str = "edges"
    # Rows whose axis route is a real axis, or ``None`` when every row is.
    # Taken once here since the trunk reuses one edge set across every block.
    axis_rows: Tensor | None = field(init=False, default=None)
    # The fused message's CSR views, keyed by channel count and built on first
    # use; reused across every block of the trunk.
    _plans: dict[int, MessagePlan] = field(
        init=False, default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:
        named = {"src": self.src, "dst": self.dst, "relation": self.relation}
        if self.axis is not None:
            named["axis"] = self.axis
        for field_name, values in named.items():
            label = f"{self.name}.{field_name}"
            if not isinstance(values, Tensor):
                raise TypeError(f"{label} must be a tensor, got {type(values).__name__}")
            if values.dtype != torch.int64:
                raise TypeError(f"{label} must be int64, got {values.dtype}")
            if values.ndim != 1:
                raise ValueError(f"{label} must be 1-D, got shape {tuple(values.shape)}")
            if values.shape != self.src.shape:
                raise ValueError(
                    f"{label} has {values.shape[0]} rows against src's "
                    f"{self.src.shape[0]}"
                )
            if values.device != self.src.device:
                raise ValueError(
                    f"{label} is on {values.device}, src on {self.src.device}"
                )
        for field_name in ("n_src", "n_dst", "num_relations"):
            size = getattr(self, field_name)
            if not isinstance(size, int) or size < 0:
                raise ValueError(
                    f"{self.name}.{field_name} must be a nonnegative int, got {size!r}"
                )
        for field_name in ("dst_sorted", "fully_routed"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(
                    f"{self.name}.{field_name} must be a host-side bool, got "
                    f"{getattr(self, field_name)!r}"
                )
        if self.axis is None and not self.fully_routed:
            raise ValueError(
                f"{self.name} carries no axis route at all, so fully_routed=False "
                "describes nothing: a family without an axis column routes no "
                "axis message and has no unrouted subset"
            )

        if self.axis is not None and not self.fully_routed:
            object.__setattr__(
                self, "axis_rows", (self.axis >= 0).nonzero(as_tuple=True)[0]
            )

    def __len__(self) -> int:
        return int(self.src.shape[0])

    def routed(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """``(src, dst, relation, axis)`` of the rows that route an axis message."""
        if self.axis is None:
            raise ValueError("this edge family routes no axis message")
        if self.axis_rows is None:
            return self.src, self.dst, self.relation, self.axis
        rows = self.axis_rows
        return (
            self.src.index_select(0, rows),
            self.dst.index_select(0, rows),
            self.relation.index_select(0, rows),
            self.axis.index_select(0, rows),
        )

    def plan(self, channels: int) -> MessagePlan:
        """This family's CSR views for the fused message (§14), cached.

        ``channels`` is 1 for the invariant stream and :data:`AXIS_CHANNELS`
        for the axis stream, whose plan covers the routed rows alone — an
        off-axis row routes nothing, so it is excluded rather than walked.
        """
        cached = self._plans.get(channels)
        if cached is not None:
            return cached
        if channels == 1:
            edges = (self.src, self.dst, self.relation, None)
        else:
            src, dst, relation, axis = self.routed()
            edges = (src, dst, relation, axis)
        # An axis plan covers a subset of the rows taken in order, and a
        # subsequence of an ascending sequence is ascending, so the family's own
        # ordering carries to both plans.
        self._plans[channels] = message_plan(
            *edges,
            self.n_src,
            self.n_dst,
            self.num_relations,
            channels,
            dst_sorted=self.dst_sorted,
        )
        return self._plans[channels]


EdgeSet = TypedEdges | PlannedEdges


def _edge_cardinalities(edges: EdgeSet) -> tuple[int, int]:
    """Source/destination counts without specializing planned chunk metadata."""
    if isinstance(edges, PlannedEdges):
        plan = edges.inv_plan
        return plan.src_ptr.shape[0] - 1, plan.dst_ptr.shape[0] - 1
    return edges.n_src, edges.n_dst


def incidence_edges(batch: PackedACTBatch) -> tuple[TypedEdges, TypedEdges]:
    """The cell↔window incidence of §10, both directions (§18.1, §18.2).

    One traversal of the ``(N_windows, 6)`` slot tables produces both: the mask
    selects the slots whose cell the scope represents, and the surviving slots
    give the cell, the joint ``(pattern, slot)`` relation class, and — as the
    route — the window's own native axis (§12.3). Returned in trunk order:
    cells into windows first, then windows into cells, which reads the windows
    the first pass just updated.

    Every column is bounded by the packer (`packed.py`'s ``_VALUE_RANGES`` and
    ``_INDEX_FIELDS``, re-derived after collation by ``_refuse_crossing``) and
    is not re-read here. ``mask.nonzero()`` is the one device read on this
    path: it discovers how many slots the scope represents, once per batch.
    """
    mask = batch.window_incidence_mask
    if mask.dtype != torch.bool:
        raise TypeError(f"window_incidence_mask must be bool, got {mask.dtype}")
    windows, slots = mask.nonzero(as_tuple=True)
    cells = batch.window_cell_index[windows, slots]
    relation = batch.window_incidence_class[windows, slots]
    axis = batch.window_axis.index_select(0, windows)
    n_cells = int(batch.cell_occupancy.shape[0])
    n_windows = int(batch.window_pattern_class.shape[0])

    to_windows = TypedEdges(
        src=cells,
        dst=windows,
        relation=relation,
        axis=axis,
        n_src=n_cells,
        n_dst=n_windows,
        num_relations=INCIDENCE_RELATIONS,
        # `nonzero` walks the (N_windows, 6) mask row-major, so the window index
        # it returns is nondecreasing by construction.
        dst_sorted=True,
        fully_routed=True,
        name="incidence cells->windows",
    )
    to_cells = TypedEdges(
        src=windows,
        dst=cells,
        relation=relation,
        axis=axis,
        n_src=n_windows,
        n_dst=n_cells,
        num_relations=INCIDENCE_RELATIONS,
        # The same rows read backwards: window-major, and a window's cells are
        # in slot order rather than cell order. This is the one family whose
        # destination view the plan has to sort.
        dst_sorted=False,
        fully_routed=True,
        name="incidence windows->cells",
    )
    return to_windows, to_cells


def adjacency_edges(batch: PackedACTBatch, cfg: MantisACTConfig) -> TypedEdges:
    """The §15.1 hex-distance-one edges between cells (§18.3).

    Every such displacement lies on an axis and belongs to one orbit, so the
    relation is constant across the family (one host-side integer from
    ``adjacency_relation_id(cfg)``, checked against ``cfg``'s vocabulary here)
    and the axis route is always real, making the family ``fully_routed``.
    """
    axis = batch.adjacency_axis
    n_cells = int(batch.cell_occupancy.shape[0])
    relation_id = adjacency_relation_id(cfg)
    num_relations = relation_vocabulary_size(cfg)
    if not 0 <= relation_id < num_relations:
        raise ValueError(
            f"the hex-step relation id {relation_id} of d6_relation_mode "
            f"{cfg.d6_relation_mode!r} lies outside its own "
            f"{num_relations}-class vocabulary"
        )
    return TypedEdges(
        src=batch.adjacency_src,
        dst=batch.adjacency_dst,
        relation=torch.full_like(batch.adjacency_src, relation_id),
        axis=axis,
        n_src=n_cells,
        n_dst=n_cells,
        num_relations=num_relations,
        # §7 sorts this family by (dst, src, axis) and `_check_ordering`
        # (`packed.py:388-408`) refuses a graph that is not; `collate` shifts
        # each position's rows by its own offset and concatenates them in
        # position order, so the concatenation is still destination-ascending.
        dst_sorted=True,
        fully_routed=True,
        name="hex adjacency",
    )


def radius_edges(batch: PackedACTBatch, cfg: MantisACTConfig) -> TypedEdges:
    """The §15.2 occupied-source to represented-destination edges (§18.4).

    The relation joins the displacement's D6 class with the source stone's
    OWN/OPP colour. The route is the axis the displacement lies on and ``-1``
    off it, so an off-axis edge updates the invariant stream only;
    ``route_on_axis_radius_messages=False`` drops the route from the whole
    family, and then no axis parameters exist on this path at all — the family
    is not ``fully_routed`` otherwise.

    ``radius_orbit``'s upper bound moves with ``d6_relation_mode``/``d_max``, so
    the packer records the batch's own ceiling as ``radius_orbit_bound`` and it
    is compared here against this ``cfg``'s vocabulary — the only check that
    catches a batch and a model built under different relation spaces. That a
    radius edge's source is occupied is a semantic claim rather than an index
    bound, checked separately at `packed.py`'s ``_check_consistency``.
    """
    base = relation_vocabulary_size(cfg)
    if batch.radius_orbit_bound > base:
        raise ValueError(
            f"the batch carries radius orbits up to "
            f"{batch.radius_orbit_bound - 1}, outside the {base}-class "
            f"vocabulary of d6_relation_mode {cfg.d6_relation_mode!r} at "
            f"d_max={cfg.d_max}: the graph and this model were built for "
            "different §11.2 relation spaces"
        )
    occupancy = batch.cell_occupancy.index_select(0, batch.radius_src)
    relation = 2 * batch.radius_orbit + (occupancy == OCCUPANCY_OPP).long()
    n_cells = int(batch.cell_occupancy.shape[0])
    return TypedEdges(
        src=batch.radius_src,
        dst=batch.radius_dst,
        relation=relation,
        axis=batch.radius_axis_or_neg1 if cfg.route_on_axis_radius_messages else None,
        n_src=n_cells,
        n_dst=n_cells,
        num_relations=radius_relation_count(cfg),
        # §7 sorts this family by (dst, src, orbit) and `_check_ordering`
        # (`packed.py:388-408`) refuses a graph that is not; `collate`
        # concatenates the positions in order with per-position shifts, so the
        # concatenation is still destination-ascending.
        dst_sorted=True,
        # §11.3 gives an off-axis displacement the route -1 by design
        # (`packed.py:153`), so this family always takes the routed subset.
        fully_routed=not cfg.route_on_axis_radius_messages,
        name="occupied radius",
    )


@dataclass(frozen=True, eq=False)
class WindowWindowEdges:
    """§16's typed collinear/crossing window↔window edges, in three CSR views.

    ``ptr``/``src``/``cls`` is the destination-major view a forward reduces
    over, and ``cptr``/``cedge`` the class-major view the class-bias gradient
    reduces over. The source-major view the backward's ``dk``/``dv`` sweep
    walks is not stored separately: the edge set is closed under reversal, so
    it is ``ptr``, ``src`` and ``scls`` — the same destination-major view with
    the class mirrored — which is why :meth:`views` passes the first two twice.

    ``n_windows`` is the family both endpoints index, host-side. Built once per
    batch and reused by every block.
    """

    ptr: Tensor
    src: Tensor
    cls: Tensor
    scls: Tensor
    cptr: Tensor
    cedge: Tensor
    n_windows: int

    def views(self) -> tuple[Tensor, ...]:
        """The eight arguments `window_pairs.edge_attention` takes, in order."""
        return (
            self.ptr,
            self.src,
            self.cls,
            self.ptr,
            self.src,
            self.scls,
            self.cptr,
            self.cedge,
        )

    def __len__(self) -> int:
        return int(self.src.shape[0])


def window_window_edges(batch: PackedACTBatch) -> WindowWindowEdges:
    """The §16 typed window↔window edges of ``batch`` (§18.5).

    Two windows relate in exactly one of two ways, both functions of the
    identity triples ``(native_axis, start_q, start_r)`` alone: collinear at a
    signed start offset of at most eleven, or crossing at the one lattice cell
    two non-parallel hex lines meet in. `window_pairs` enumerates both by
    sorted join and folds each into a D6-invariant class; the join and the
    flash kernels that reduce over it are imported rather than reimplemented.

    Derived on the device, once per batch, rather than built by the builder and
    shipped: the edge views are two orders of magnitude larger than the window
    identities they are a join of, so the identities cross the bus and the
    edges never do. The join is data-dependent and runs through `window_pairs`'
    custom op to stay inside a compiled graph.
    """
    n_windows = int(batch.window_pattern_class.shape[0])
    window_pos = row_positions(batch.window_offsets, n_windows)
    ptr, src, cls, scls, cptr, cedge = derive_pair_tables(batch.window_id, window_pos)
    return WindowWindowEdges(
        ptr=ptr,
        src=src,
        cls=cls,
        scls=scls,
        cptr=cptr,
        cedge=cedge,
        n_windows=n_windows,
    )


class TypedWindowAttention(nn.Module):
    """§16's typed window↔window attention, as §18's optional step 5.

    A pre-norm residual branch over the window `EquivariantState`. Each stream
    runs multi-head attention over the edge views of :func:`window_window_edges`,
    with one learned additive score bias per (head, relation class) and the
    softmax taken per destination segment in fp32. Every window carries a self
    loop, so no segment is empty.

    ```text
    score(dst, src) = q_dst . k_src / sqrt(head_dim) + bias[head, class(dst, src)]
    out_dst         = sum_src softmax(score) * v_src
    delta           = W_out(out)
    ```

    Equivariant (§12.1): a relation class is a D6 invariant (`window_pairs`
    folds a collinear edge by ``|offset|`` and a crossing edge by the pair of
    per-side folds ``min(t, 5 - t)`` and ``max(-t, t - 5)``), so the invariant
    stream is a function of invariants alone. The axis stream attends within a
    channel — channel ``a`` of a destination reads channel ``a`` of its
    sources, with one shared projection set and one bias table broadcast across
    all three channels, never a per-channel table (§12.2).

    The three channels ride in the head dimension — the axis query is laid out
    as ``(window, 3 * num_heads, head_dim)``, giving one softmax per
    ``(window, channel, head)`` over the same edge tables the invariant stream
    uses, rather than a second, three-times-larger edge set.

    The query, key and value projections are bias-free: a value bias is spanned
    by the output projection's own bias, and the score's constant term is
    already the relation bias table.
    """

    def __init__(self, cfg: MantisACTConfig) -> None:
        super().__init__()
        if cfg.window_window_mode != "typed_collinear_crossing":
            raise ValueError(
                f"window_window_mode={cfg.window_window_mode!r} does not ask for "
                "typed window attention"
            )
        self.heads = cfg.num_heads
        self.d_inv = cfg.d_inv
        self.d_axis = cfg.d_axis

        self.norm = EquivariantNorm(cfg)
        self.q_inv = nn.Linear(cfg.d_inv, cfg.d_inv, bias=False)
        self.k_inv = nn.Linear(cfg.d_inv, cfg.d_inv, bias=False)
        self.v_inv = nn.Linear(cfg.d_inv, cfg.d_inv, bias=False)
        self.out_inv = nn.Linear(cfg.d_inv, cfg.d_inv)
        # Zero at init, so every destination starts with the uniform weights
        # §27 asks of a relation attention bias.
        self.bias_inv = nn.Parameter(torch.zeros(cfg.num_heads, WINDOW_WINDOW_RELATIONS))

        if cfg.d_axis:
            if cfg.d_axis % cfg.num_heads:
                raise ValueError(
                    f"d_axis={cfg.d_axis} must divide into num_heads="
                    f"{cfg.num_heads} heads for typed window attention"
                )
            self.q_axis = nn.Linear(cfg.d_axis, cfg.d_axis, bias=False)
            self.k_axis = nn.Linear(cfg.d_axis, cfg.d_axis, bias=False)
            self.v_axis = nn.Linear(cfg.d_axis, cfg.d_axis, bias=False)
            self.out_axis = nn.Linear(cfg.d_axis, cfg.d_axis)
            self.bias_axis = nn.Parameter(
                torch.zeros(cfg.num_heads, WINDOW_WINDOW_RELATIONS)
            )
        else:
            self.q_axis = self.k_axis = self.v_axis = self.out_axis = None
            self.bias_axis = None

        self.residual = EquivariantResidual(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def _check(self, edges: WindowWindowEdges, windows: EquivariantState) -> None:
        """Refuse an edge set or a state that does not match this module."""
        if windows.leading_shape != (edges.n_windows,):
            raise ValueError(
                f"window state covers {windows.leading_shape} windows against the "
                f"edge family's ({edges.n_windows},)"
            )
        if windows.d_inv != self.d_inv:
            raise ValueError(
                f"window state is d_inv={windows.d_inv} against this attention's "
                f"{self.d_inv}"
            )
        if windows.has_axis != (self.q_axis is not None):
            built = "with" if self.q_axis is not None else "without"
            given = "one" if windows.has_axis else "none"
            raise ValueError(
                f"typed window attention was built {built} an axis stream, but "
                f"the state has {given}"
            )
        if windows.has_axis and windows.d_axis != self.d_axis:
            raise ValueError(
                f"window state is d_axis={windows.d_axis} against this "
                f"attention's {self.d_axis}"
            )

    def forward(
        self, edges: WindowWindowEdges, windows: EquivariantState
    ) -> EquivariantState:
        """The windows after they have attended to their typed partners."""
        self._check(edges, windows)
        views = edges.views()
        n_windows = edges.n_windows
        z = self.norm(windows)

        heads = self.heads
        head_dim = self.d_inv // heads
        out = edge_attention(
            self.q_inv(z.inv).view(n_windows, heads, head_dim),
            self.k_inv(z.inv).view(n_windows, heads, head_dim),
            self.v_inv(z.inv).view(n_windows, heads, head_dim),
            self.bias_inv,
            *views,
        )
        # The kernel answers in fp32 whatever the stream is (§27); the delta is
        # cast back to the state's own precision before the residual, so the
        # stream's dtype stays the state's and not the reduction's.
        delta_inv = self.drop(self.out_inv(out.reshape(n_windows, self.d_inv))).to(
            windows.inv.dtype
        )

        if self.q_axis is None:
            return self.residual(windows, delta_inv)

        # The three channels ride in the head dimension: head `c * heads + h` is
        # channel `c`'s head `h`, so the kernel's per-(window, head) softmax is
        # the per-(window, channel, head) softmax this stream wants, over the
        # same edge tables. The bias table is one set of `heads` rows repeated
        # across the channels rather than three sets, which is what keeps the
        # score free of a per-absolute-axis parameter (§12.2).
        shape = (n_windows, AXIS_CHANNELS * heads, self.d_axis // heads)
        out = edge_attention(
            self.q_axis(z.axis).reshape(shape),
            self.k_axis(z.axis).reshape(shape),
            self.v_axis(z.axis).reshape(shape),
            self.bias_axis.repeat(AXIS_CHANNELS, 1),
            *views,
        )
        delta_axis = self.drop(
            self.out_axis(out.reshape(n_windows, AXIS_CHANNELS, self.d_axis))
        ).to(windows.axis.dtype)
        return self.residual(windows, delta_inv, delta_axis)


# --------------------------------------------------------------------------
# The message module


class _PairMLP(nn.Module):
    """``MLP([a; b])`` — §14's update MLP over the destination and its aggregate.

    The concatenation is materialised and one linear runs over it, rather than
    two linears running over the halves with their results added — the two
    forms compute the same function, but the wide input costs fewer launches
    under autocast. ``b`` is the fp32 segment-reduction accumulator (§27) and
    ``a`` the normed destination in the autocast dtype.
    """

    def __init__(
        self, d_a: int, d_b: int, d_hidden: int, d_out: int, activation: nn.Module
    ) -> None:
        super().__init__()
        self.d_a = int(d_a)
        self.d_b = int(d_b)
        self.lin_in = nn.Linear(d_a + d_b, d_hidden)
        self.act = activation
        self.out = nn.Linear(d_hidden, d_out)

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        if a.shape[-1] != self.d_a or b.shape[-1] != self.d_b:
            raise ValueError(
                f"this update MLP pairs a {self.d_a}-wide left operand with a "
                f"{self.d_b}-wide right one, got {a.shape[-1]} and {b.shape[-1]}"
            )
        return self.out(self.act(self.lin_in(torch.cat((a, b.to(a.dtype)), dim=-1))))


class RelationGatedMessage(nn.Module):
    """§14's typed sparse edge module, for one edge family and direction.

    Holds one norm and value projection per stream on the source side, the
    relation's gate and bias projections, and the destination's norm, update
    MLP and LayerScale. The axis half exists exactly when the model has axis
    channels and this family routes them (``route_axis``); otherwise it holds
    no axis parameters. The value projections are bias-free — the relation's
    bias is already the additive term.
    """

    def __init__(
        self,
        cfg: MantisACTConfig,
        num_relations: int,
        *,
        route_axis: bool = True,
        relation_embedding: nn.Embedding | None = None,
    ) -> None:
        super().__init__()
        if cfg.incidence_message not in ("relation_gated", "additive"):
            raise ValueError(f"unknown incidence_message {cfg.incidence_message!r}")
        if cfg.incidence_reduce not in _REDUCTIONS:
            raise ValueError(f"unknown incidence_reduce {cfg.incidence_reduce!r}")
        if num_relations < 1:
            raise ValueError(f"num_relations must be at least 1, got {num_relations}")

        self.num_relations = int(num_relations)
        self.d_inv = cfg.d_inv
        self.d_axis = cfg.d_axis
        self.activation = cfg.activation
        self.reduce = cfg.incidence_reduce
        self.gated = cfg.incidence_message == "relation_gated"
        self.route_axis = bool(route_axis and cfg.use_axis_channels)

        if relation_embedding is None:
            relation_embedding = make_relation_embedding(num_relations, cfg.d_rel)
        elif relation_embedding.num_embeddings != num_relations:
            raise ValueError(
                f"shared relation table has {relation_embedding.num_embeddings} rows "
                f"for a {num_relations}-class family"
            )
        elif relation_embedding.embedding_dim != cfg.d_rel:
            raise ValueError(
                f"shared relation table is {relation_embedding.embedding_dim} wide "
                f"against d_rel={cfg.d_rel}"
            )
        self.relation = relation_embedding

        d_inv = cfg.d_inv
        self.ln_src_inv = nn.LayerNorm(d_inv)
        self.wv_inv = nn.Linear(d_inv, d_inv, bias=False)
        self.wb_inv = nn.Linear(cfg.d_rel, d_inv)
        self.ln_dst_inv = nn.LayerNorm(d_inv)
        self.update_inv = _PairMLP(
            d_inv, d_inv, d_inv, d_inv, activation_module(cfg.activation)
        )
        self.scale_inv = LayerScale(d_inv, cfg.layer_scale_init)
        if self.gated:
            self.wg_inv = nn.Linear(cfg.d_rel, d_inv)
        if self.reduce == "attention":
            # Zero at init, so every destination starts with the uniform
            # weights §27 asks of a relation attention bias.
            self.score_inv = nn.Parameter(torch.zeros(d_inv))

        if self.route_axis:
            d_axis = cfg.d_axis
            self.ln_src_axis = nn.LayerNorm(d_axis)
            self.wv_axis = nn.Linear(d_axis, d_axis, bias=False)
            self.wb_axis = nn.Linear(cfg.d_rel, d_axis)
            self.ln_dst_axis = nn.LayerNorm(d_axis)
            self.update_axis = _PairMLP(
                d_axis, d_axis, d_axis, d_axis, activation_module(cfg.activation)
            )
            self.scale_axis = LayerScale(d_axis, cfg.layer_scale_init)
            if self.gated:
                self.wg_axis = nn.Linear(cfg.d_rel, d_axis)
            if self.reduce == "attention":
                self.score_axis = nn.Parameter(torch.zeros(d_axis))

        self.drop = nn.Dropout(cfg.dropout)

    def _check(
        self,
        edges: EdgeSet,
        source: EquivariantState,
        destination: EquivariantState,
    ) -> None:
        """Refuse an edge set or a state that does not match this module."""
        if edges.num_relations != self.num_relations:
            raise ValueError(
                f"edge family has {edges.num_relations} relation classes against "
                f"this module's {self.num_relations}"
            )
        n_src, n_dst = _edge_cardinalities(edges)
        for name, state, count in (
            ("source", source, n_src),
            ("destination", destination, n_dst),
        ):
            if state.leading_shape != (count,):
                raise ValueError(
                    f"{name} state covers {state.leading_shape} entities against "
                    f"the edge family's ({count},)"
                )
            if state.d_inv != self.d_inv:
                raise ValueError(
                    f"{name} state is d_inv={state.d_inv} against this message's "
                    f"{self.d_inv}"
                )
        if not self.route_axis:
            return
        if edges.axis is None:
            raise ValueError(
                "this message routes an axis stream, but the edge family carries "
                "no axis route"
            )
        for name, state in (("source", source), ("destination", destination)):
            state.require_axis(f"the {name} of a routed relation-gated message")
            if state.d_axis != self.d_axis:
                raise ValueError(
                    f"{name} state is d_axis={state.d_axis} against this message's "
                    f"{self.d_axis}"
                )

    def _aggregate(
        self,
        values: Tensor,
        edges: EdgeSet,
        channels: int,
        *,
        gate_projection: nn.Linear | None,
        bias_projection: nn.Linear,
        score_vector: Tensor | None,
    ) -> Tensor:
        """One stream's aggregate of §14's messages, in fp32.

        Both streams run this: the invariant one with ``channels=1`` over node
        rows, the axis one with ``channels=AXIS_CHANNELS`` over the
        ``(node, axis)`` slots of a flattened view, so a routed message
        addresses a row rather than a channel of a ``(E, 3, d_axis)``
        intermediate. The gate and bias are functions of the relation alone, so
        they are projected once over the whole vocabulary and read per edge
        inside the kernel.

        Every table the kernel reads is fp32 (§27): the accumulator is fp32 in
        registers and the value, gate and bias gradients come out of
        contiguous segment reductions rather than atomic scatters — CUDA has no
        native bf16 ``atomicAdd`` and emulates one with compare-and-swap.
        """
        rel = self.relation.weight
        values = at_least_fp32(values)
        gate = (
            at_least_fp32(torch.sigmoid(gate_projection(rel)))
            if gate_projection is not None
            else None
        )
        bias = at_least_fp32(bias_projection(rel))
        if self.reduce == "attention":
            return self._attend(values, edges, channels, gate, bias, score_vector)
        plan = edges.plan(channels)
        total = relation_gated_message(values, gate, bias, plan)
        if self.reduce == "mean":
            counts = plan.destination_counts().clamp(min=1.0)
            total = total / counts.unsqueeze(1)
        return total

    def _attend(
        self,
        values: Tensor,
        edges: EdgeSet,
        channels: int,
        gate: Tensor | None,
        bias: Tensor,
        score_vector: Tensor | None,
    ) -> Tensor:
        """§14's attention ablation, over the explicit per-edge messages.

        A segment softmax needs every edge's score before any destination's
        weights are known, so this reduction keeps the ``(E, d)`` formulation
        the fused sum/mean path avoids.
        """
        _n_src, n_dst = _edge_cardinalities(edges)
        if channels == 1:
            src_slots, dst_slots, relation = edges.src, edges.dst, edges.relation
            n_segments = n_dst
        else:
            edge_src, edge_dst, relation, edge_axis = edges.routed()
            src_slots = edge_src * channels + edge_axis
            dst_slots = edge_dst * channels + edge_axis
            n_segments = n_dst * channels
        messages = values.index_select(0, src_slots)
        if gate is not None:
            messages = messages * gate.index_select(0, relation)
        messages = messages + bias.index_select(0, relation)
        score = (messages.float() * score_vector).sum(dim=1)
        return attention_by_destination(messages, dst_slots, n_segments, score)

    def forward(
        self,
        edges: EdgeSet,
        source: EquivariantState,
        destination: EquivariantState,
    ) -> EquivariantState:
        """The destination after this edge family's messages reach it (§14).

        A pre-norm residual branch: the norms, messages, aggregation and update
        MLPs run on the state and the LayerScaled result is added back. A
        family that routes no axis message leaves the destination's axis
        stream unchanged rather than returning a zero for it.
        """
        self._check(edges, source, destination)
        attending = self.reduce == "attention"

        aggregate = self._aggregate(
            self.wv_inv(self.ln_src_inv(source.inv)),
            edges,
            1,
            gate_projection=self.wg_inv if self.gated else None,
            bias_projection=self.wb_inv,
            score_vector=self.score_inv if attending else None,
        )
        delta_inv = self.drop(self.update_inv(self.ln_dst_inv(destination.inv), aggregate))
        inv = destination.inv + self.scale_inv(delta_inv).to(destination.inv.dtype)

        if not self.route_axis:
            return EquivariantState(inv, destination.axis)

        # The axis stream runs over the (node, axis) slots of a flat view of the
        # three channels, so an edge's route selects a row rather than a channel
        # of a wider intermediate. Edges on no axis are dropped by the plan:
        # they have no channel to land in (§11.3).
        aggregate = self._aggregate(
            self.wv_axis(self.ln_src_axis(source.axis)).reshape(-1, self.d_axis),
            edges,
            AXIS_CHANNELS,
            gate_projection=self.wg_axis if self.gated else None,
            bias_projection=self.wb_axis,
            score_vector=self.score_axis if attending else None,
        ).reshape(destination.inv.shape[0], AXIS_CHANNELS, self.d_axis)
        delta_axis = self.drop(
            self.update_axis(self.ln_dst_axis(destination.axis), aggregate)
        )
        axis = destination.axis + self.scale_axis(delta_axis).to(destination.axis.dtype)
        return EquivariantState(inv, axis)


# --------------------------------------------------------------------------
# The three paths the state trunk runs


class CellWindowIncidence(nn.Module):
    """Both directions of the §10 cell↔window incidence (§18.1, §18.2).

    The two directions share one relation table — they are the same edges read
    the other way — but keep private projections and update MLPs (§14). The
    trunk runs ``to_windows`` first, then ``to_cells``, which reads the windows
    the first pass just updated.
    """

    def __init__(
        self,
        cfg: MantisACTConfig,
        *,
        relation_embedding: nn.Embedding | None = None,
    ) -> None:
        super().__init__()
        if relation_embedding is None:
            relation_embedding = make_relation_embedding(INCIDENCE_RELATIONS, cfg.d_rel)
        self.to_windows = RelationGatedMessage(
            cfg, INCIDENCE_RELATIONS, relation_embedding=relation_embedding
        )
        self.to_cells = RelationGatedMessage(
            cfg, INCIDENCE_RELATIONS, relation_embedding=relation_embedding
        )


class AdjacencyMessage(nn.Module):
    """§15.1: messages between cells one hex step apart (§18.3).

    One relation class and always an axis route: a cell learns the immediate
    shape of the line it sits on, in the channel of that line's own axis.
    """

    def __init__(
        self,
        cfg: MantisACTConfig,
        *,
        relation_embedding: nn.Embedding | None = None,
    ) -> None:
        super().__init__()
        self.message = RelationGatedMessage(
            cfg, relation_vocabulary_size(cfg), relation_embedding=relation_embedding
        )

    def forward(self, edges: TypedEdges, cells: EquivariantState) -> EquivariantState:
        """The cells after their one-step neighbours have messaged them."""
        return self.message(edges, cells, cells)


class RadiusMessage(nn.Module):
    """§15.2: occupied-source to represented-destination messages (§18.4).

    Carries the exact D6 orbit of the displacement jointly with the source
    stone's colour, so a far legal cell with no current window is still
    described by the shape of the stones around it. ``route_on_axis_radius_messages``
    turns the axis route off for the whole family, leaving no axis parameters.
    """

    def __init__(
        self,
        cfg: MantisACTConfig,
        *,
        relation_embedding: nn.Embedding | None = None,
    ) -> None:
        super().__init__()
        self.message = RelationGatedMessage(
            cfg,
            radius_relation_count(cfg),
            route_axis=cfg.route_on_axis_radius_messages,
            relation_embedding=relation_embedding,
        )

    def forward(self, edges: TypedEdges, cells: EquivariantState) -> EquivariantState:
        """The cells after every stone within the radius has messaged them."""
        return self.message(edges, cells, cells)


__all__ = [
    "INCIDENCE_RELATIONS",
    "WINDOW_WINDOW_RELATIONS",
    "AdjacencyMessage",
    "CellWindowIncidence",
    "RadiusMessage",
    "RelationGatedMessage",
    "TypedEdges",
    "TypedWindowAttention",
    "WindowWindowEdges",
    "adjacency_edges",
    "adjacency_relation_id",
    "attention_by_destination",
    "incidence_edges",
    "make_relation_embedding",
    "radius_edges",
    "radius_relation_count",
    "relation_vocabulary_size",
    "segment_softmax",
    "segment_sum",
    "window_window_edges",
]
