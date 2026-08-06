"""Relation-gated messages over typed sparse edges, and the trunk's three paths.

This module implements §14 and §15 of ``docs/MANTIS_ACT_SPEC.md``: one generic
module for a typed sparse edge family, and the three concrete families the
state trunk runs — cell↔window incidence in both directions (§18.1, §18.2),
hex adjacency between cells (§15.1, §18.3), and the occupied-to-cell radius
edges typed by their D6 orbit (§15.2, §18.4).

What a message is (§14). An edge carries a relation id ``r`` and, when its
displacement lies on an axis, the route ``a`` of that axis. The relation
supplies a multiplicative gate and an additive bias; the source supplies the
value::

    msg_inv  = sigmoid(Wg_inv(E_rel[r])) * Wv_inv(LN(src.h_inv)) + Wb_inv(E_rel[r])
    msg_axis = sigmoid(Wg_axis(E_rel[r])) * Wv_axis(LN(src.h_axis[a])) + Wb_axis(E_rel[r])

Messages are summed by destination in fp32 and the destination's two streams
are updated through their own MLPs. Sum is the default because incidence count
is signal: a cell in four windows differs from a cell in one, and a mean erases
exactly that. ``mean`` and ``attention`` stay reachable as the §14 ablations,
and ``incidence_message="additive"`` drops the gate for the legacy
``U @ src + E_relation`` control of §29's ``full_additive_incidence``.

Why this is equivariant (§12.1). Every axis-stream parameter — the norm, the
value projection, the gate and bias projections, the update MLP, the attention
score vector — is one shared set applied independently to whichever channel an
edge routes through, and the relation id is a D6 invariant. Under a transform
``g`` an edge routed through axis ``a`` becomes an edge routed through
``pi_g(a)`` between the images of its endpoints, and the same shared parameters
produce the same vector in channel ``pi_g(a)`` of the image destination. No
absolute axis id is ever an embedding index, no axis has its own weights, and
the three channels are never concatenated in a fixed order (§12.2). An edge
that lies on no axis updates the invariant stream only, which is invariant
because "lies on no axis" is itself preserved by the group.

Index conventions this module fixes:

- A relation id is an index into one embedding table per edge family; the
  vocabularies are :data:`INCIDENCE_RELATIONS` for cell↔window incidence
  (§10.1's single 2187-row joint table), :func:`relation_vocabulary_size` for
  hex adjacency, and :func:`radius_relation_count` for the radius edges.
- The radius relation is the *joint* id ``2 * orbit + is_opponent``. §15.2 puts
  the orbit class and the source's OWN/OPP colour on the same edge, and adding
  two embeddings instead would make the relation's gate the sum of a colour
  term and a geometry term — unable to say that a shape means one thing from
  a stone of one colour and another from the other. §10.1 refuses that
  factorisation for the incidence relation for the same reason; the product
  space here is 104 rows, so exactness costs nothing.
- An axis route is ``0..2``, or ``-1`` for an edge on no axis. ``-1`` routes no
  axis message; it is never an index.

Where indices are checked. :class:`TypedEdges` validates every index against
the family size and the relation vocabulary at construction, so an out-of-range
or ``-1`` index raises naming the field, the value, and the edge — once per
batch, at the point the builder's output is turned into an edge set, rather
than once per block inside the forward. A ``-1`` is a builder fault: an
unrepresented window slot must be masked out of the incidence table, not
carried into it, and both ``index_select`` and an embedding lookup would
otherwise read the far end of the table and return a plausible wrong row.

Numerics (§27). Parameters are fp32 and the forward runs under bf16 autocast
unchanged. Every segment reduction and every softmax is taken in fp32, so a
long destination's aggregate does not stall the way a bf16 running sum does;
the update MLP runs in whatever dtype autocast chose and its delta is cast back
to the residual stream's dtype before the add, so the stream's precision is the
state's and not autocast's.

Composition. A message module is a pre-norm residual branch over
``EquivariantState``, the same shape as ``equivariant.AxisMix`` and
``equivariant.EquivariantFFN``: a state goes in and the updated state comes
out, with the branch's own LayerScale on the delta (§27). The trunk therefore
chains stages rather than adding deltas itself, and §18.11's phase FiLM is a
stage of the block like the others.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from .cells import OCCUPANCY_OPP, OCCUPANCY_OWN
from .config import MantisACTConfig
from .equivariant import (
    AXIS_CHANNELS,
    EquivariantState,
    LayerScale,
    activation_module,
    at_least_fp32,
)
from .packed import PackedACTBatch
from .pattern_classes import ALL_CELL_WINDOW_REL_CLASSES
from .symmetry import (
    RELATION_PAD,
    coarse_relation,
    coarse_relation_count,
    orbit_table,
)

# §10.1's one joint (pattern, slot) relation table. The `nonempty` window scope
# never emits the three all-empty classes; the table is still the 2187-row one,
# so a scope change does not renumber a relation.
INCIDENCE_RELATIONS = ALL_CELL_WINDOW_REL_CLASSES

# §27: embeddings, relation tables, and latent bases.
EMBEDDING_INIT_STD = 0.02

_REDUCTIONS = ("sum", "mean", "attention")


def relation_vocabulary_size(cfg: MantisACTConfig) -> int:
    """The geometry relation vocabulary of ``cfg``'s ``d6_relation_mode``.

    Under ``orbit48`` that is the 48 exact orbits plus the four reserved ids of
    §11.2, whatever ``d_max`` is: a smaller radius leaves the unused orbit ids
    empty rather than shifting the reserved band down, so one embedding shape
    serves every radius. Under ``coarse_distance_axis`` it is that scheme's own
    space, which reserves nothing.
    """
    if cfg.d6_relation_mode == "orbit48":
        return RELATION_PAD + 1
    if cfg.d6_relation_mode == "coarse_distance_axis":
        return coarse_relation_count(cfg.d_max)
    raise ValueError(f"unknown d6_relation_mode {cfg.d6_relation_mode!r}")


def radius_relation_count(cfg: MantisACTConfig) -> int:
    """The joint ``(geometry class, source colour)`` vocabulary of §15.2."""
    return 2 * relation_vocabulary_size(cfg)


def adjacency_relation_id(cfg: MantisACTConfig) -> int:
    """The one relation class a hex step belongs to, from the orbit table.

    Every distance-one displacement is a single D6 orbit, so hex adjacency is
    one relation and the id is read from the same table the radius edges use
    rather than written down. Sharing the space means an adjacency edge and a
    distance-one radius edge name the same class.
    """
    if cfg.d6_relation_mode == "orbit48":
        return int(orbit_table(cfg.d_max).lookup(1, 0))
    if cfg.d6_relation_mode == "coarse_distance_axis":
        return int(coarse_relation(1, 0, cfg.d_max))
    raise ValueError(f"unknown d6_relation_mode {cfg.d6_relation_mode!r}")


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


def _check_range(name: str, values: Tensor, low: int, high: int) -> None:
    """Refuse an index outside ``[low, high)``, naming the value and its row."""
    if values.numel() == 0:
        return
    smallest = int(values.min())
    if smallest < low:
        raise ValueError(
            f"{name} must be >= {low}: found {smallest} at row "
            f"{int(values.argmin())}"
        )
    largest = int(values.max())
    if largest >= high:
        raise ValueError(
            f"{name} must be < {high}: found {largest} at row "
            f"{int(values.argmax())}"
        )


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


def segment_counts(index: Tensor, n_segments: int) -> Tensor:
    """How many rows each segment owns, as fp32."""
    ones = torch.ones(index.shape[0], dtype=torch.float32, device=index.device)
    return torch.zeros(n_segments, dtype=torch.float32, device=index.device).index_add_(
        0, index, ones
    )


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


def aggregate_by_destination(
    messages: Tensor,
    index: Tensor,
    n_segments: int,
    *,
    reduce: str,
    score: Tensor | None = None,
) -> Tensor:
    """Reduce per-edge messages into per-destination rows in fp32 (§14).

    ``sum`` is the default of §14: the number of edges into a destination is
    signal, and both ``mean`` and ``attention`` normalise it away, which is why
    they are ablations rather than alternatives. ``attention`` needs a per-edge
    score; the other two refuse one, because a score that reached neither the
    weights nor an error would be a silently dead branch.
    """
    if reduce not in _REDUCTIONS:
        raise ValueError(f"unknown reduce {reduce!r}; expected one of {list(_REDUCTIONS)}")
    if (score is None) != (reduce != "attention"):
        raise ValueError(
            f"reduce={reduce!r} with score={'a tensor' if score is not None else None}: "
            'a score is required by "attention" and read by nothing else'
        )
    if reduce == "sum":
        return segment_sum(messages, index, n_segments)
    if reduce == "mean":
        total = segment_sum(messages, index, n_segments)
        return total / segment_counts(index, n_segments).clamp(min=1.0).unsqueeze(1)
    weights = segment_softmax(score, index, n_segments)
    return segment_sum(messages.float() * weights.unsqueeze(1), index, n_segments)


# --------------------------------------------------------------------------
# Typed edge sets


@dataclass(frozen=True, eq=False)
class TypedEdges:
    """One typed sparse edge family, validated at construction (§14, §26).

    ``src`` and ``dst`` index the source and destination node families in the
    batch frame; ``relation`` indexes the family's relation vocabulary; ``axis``
    is the structural axis an edge routes its line message through, ``-1`` for
    an edge on no axis, or ``None`` for a family that routes no axis message at
    all. The sizes and the vocabulary travel with the edges so that every bound
    is checked exactly once, here, rather than per block in a forward.

    ``name`` is the family's name in an error message, so a ``-1`` that came
    out of a masked-in incidence slot says which table it came from.
    """

    src: Tensor
    dst: Tensor
    relation: Tensor
    axis: Tensor | None
    n_src: int
    n_dst: int
    num_relations: int
    name: str = "edges"
    # Rows whose axis route is a real axis, or ``None`` when every row is. The
    # subset is taken once here because the trunk reuses one edge set across
    # every block.
    axis_rows: Tensor | None = field(init=False, default=None)

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

        _check_range(f"{self.name}.src", self.src, 0, self.n_src)
        _check_range(f"{self.name}.dst", self.dst, 0, self.n_dst)
        _check_range(f"{self.name}.relation", self.relation, 0, self.num_relations)
        if self.axis is not None:
            _check_range(f"{self.name}.axis", self.axis, -1, AXIS_CHANNELS)
            routed = self.axis >= 0
            if not bool(routed.all()):
                object.__setattr__(self, "axis_rows", routed.nonzero(as_tuple=True)[0])

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


def incidence_edges(batch: PackedACTBatch) -> tuple[TypedEdges, TypedEdges]:
    """The cell↔window incidence of §10, both directions (§18.1, §18.2).

    One traversal of the ``(N_windows, 6)`` slot tables produces both: the mask
    selects the slots whose cell the scope represents, and the surviving slots
    give the cell, the joint ``(pattern, slot)`` relation class, and — as the
    route — the window's own native axis, which is the structural axis of the
    line the message travels along (§12.3). Returned in trunk order: cells into
    windows first, then windows into cells, which must read the windows the
    first pass just updated.
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
        name="incidence windows->cells",
    )
    return to_windows, to_cells


def adjacency_edges(batch: PackedACTBatch, cfg: MantisACTConfig) -> TypedEdges:
    """The §15.1 hex-distance-one edges between cells (§18.3).

    Every such displacement lies on an axis and belongs to one orbit, so the
    relation is constant across the family and the axis route is always real.
    """
    axis = batch.adjacency_axis
    _check_range("adjacency_axis", axis, 0, AXIS_CHANNELS)
    n_cells = int(batch.cell_occupancy.shape[0])
    return TypedEdges(
        src=batch.adjacency_src,
        dst=batch.adjacency_dst,
        relation=torch.full_like(batch.adjacency_src, adjacency_relation_id(cfg)),
        axis=axis,
        n_src=n_cells,
        n_dst=n_cells,
        num_relations=relation_vocabulary_size(cfg),
        name="hex adjacency",
    )


def radius_edges(batch: PackedACTBatch, cfg: MantisACTConfig) -> TypedEdges:
    """The §15.2 occupied-source to represented-destination edges (§18.4).

    The relation joins the displacement's D6 class with the source stone's
    OWN/OPP colour, which the builder deliberately leaves on the cell rather
    than restating on the edge. The route is the axis the displacement lies on
    and ``-1`` off it, so an off-axis edge updates the invariant stream only;
    ``route_on_axis_radius_messages=False`` drops the route from the whole
    family, and then no axis parameters exist on this path at all.

    This is the largest edge family in a real position — about 71,700 rows at
    ply 161 of stack-939 self-play against 7,200 incidences — so it is one
    gather per stream and one segment reduction, with no per-edge Python and
    nothing quadratic in cells.
    """
    orbits = batch.radius_orbit
    base = relation_vocabulary_size(cfg)
    _check_range("radius_orbit", orbits, 0, base)
    occupancy = batch.cell_occupancy.index_select(0, batch.radius_src)
    _check_range(
        "occupancy of radius_src", occupancy, OCCUPANCY_OWN, OCCUPANCY_OPP + 1
    )
    relation = 2 * orbits + (occupancy == OCCUPANCY_OPP).long()
    n_cells = int(batch.cell_occupancy.shape[0])
    return TypedEdges(
        src=batch.radius_src,
        dst=batch.radius_dst,
        relation=relation,
        axis=batch.radius_axis_or_neg1 if cfg.route_on_axis_radius_messages else None,
        n_src=n_cells,
        n_dst=n_cells,
        num_relations=radius_relation_count(cfg),
        name="occupied radius",
    )


# --------------------------------------------------------------------------
# The message module


class _PairMLP(nn.Module):
    """``MLP([a; b])`` with the concatenation folded into two input linears.

    A linear over a concatenation is the sum of two linears, so this keeps the
    spec's parameters and arithmetic without materialising the wide input.
    """

    def __init__(
        self, d_a: int, d_b: int, d_hidden: int, d_out: int, activation: nn.Module
    ) -> None:
        super().__init__()
        self.lin_a = nn.Linear(d_a, d_hidden)
        self.lin_b = nn.Linear(d_b, d_hidden, bias=False)
        self.act = activation
        self.out = nn.Linear(d_hidden, d_out)

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return self.out(self.act(self.lin_a(a) + self.lin_b(b)))


class RelationGatedMessage(nn.Module):
    """§14's typed sparse edge module, for one edge family and direction.

    Holds one norm and value projection per stream on the source side, the
    relation's gate and bias projections, and the destination's norm, update
    MLP and LayerScale. Both streams are optional in the sense that the axis
    half exists exactly when the model has axis channels *and* this family
    routes them: a family that carries no route (``route_axis=False``) carries
    no axis parameters either, so nothing is left orphaned.

    The value projections are bias-free. §14 writes them as bare matrices and
    the relation's bias is the additive term; a second constant would be the
    same parameter twice.
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
        edges: TypedEdges,
        source: EquivariantState,
        destination: EquivariantState,
    ) -> None:
        """Refuse an edge set or a state that does not match this module."""
        if edges.num_relations != self.num_relations:
            raise ValueError(
                f"edge family has {edges.num_relations} relation classes against "
                f"this module's {self.num_relations}"
            )
        for name, state, count in (
            ("source", source, edges.n_src),
            ("destination", destination, edges.n_dst),
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
        src_slots: Tensor,
        dst_slots: Tensor,
        relation: Tensor,
        n_segments: int,
        *,
        gate_projection: nn.Linear | None,
        bias_projection: nn.Linear,
        score_vector: Tensor | None,
    ) -> Tensor:
        """One stream's per-edge messages and their fp32 aggregate (§14).

        Both streams run this: the invariant one over node rows, the axis one
        over ``(node, axis)`` slots of a flattened view, which is what lets a
        routed message be a single gather rather than a ``(E, 3, d_axis)``
        intermediate. The gate and bias are functions of the relation alone, so
        they are projected once over the whole vocabulary — tens of rows — and
        gathered per edge, rather than projected per edge.
        """
        rel = self.relation.weight
        # Every gather is taken from an fp32 source, which is §27's fp32
        # segment reduction read in the direction that actually costs
        # something. `index_select` backward is `index_add_` into a zero
        # tensor of the *source's* dtype, so a bf16 source makes the gradient
        # of a gather a bf16 atomic scatter — and CUDA has no native bf16
        # atomicAdd, so it becomes a compare-and-swap loop whose contention is
        # worst exactly here: the ply-161 radius family scatters 573k edge
        # gradients back over 104 relation rows. Measured on a 4070 Ti that
        # one scatter is 12.5 ms in bf16 against 0.45 ms in fp32.
        # The aggregate is fp32 either way, so promoting at the gather rather
        # than after it materialises no tensor that did not already exist.
        messages = at_least_fp32(values).index_select(0, src_slots)
        if gate_projection is not None:
            gate = at_least_fp32(torch.sigmoid(gate_projection(rel)))
            messages = messages * gate.index_select(0, relation)
        messages = messages + at_least_fp32(bias_projection(rel)).index_select(0, relation)
        score = (
            (messages.float() * score_vector).sum(dim=1)
            if score_vector is not None
            else None
        )
        return aggregate_by_destination(
            messages, dst_slots, n_segments, reduce=self.reduce, score=score
        )

    def forward(
        self,
        edges: TypedEdges,
        source: EquivariantState,
        destination: EquivariantState,
    ) -> EquivariantState:
        """The destination after this edge family's messages reach it (§14).

        A pre-norm residual branch, in the shape the rest of the package's
        branches take: the state goes in, the norms, messages, aggregation and
        update MLPs run on it, and the LayerScaled result is added back. A
        family that routes no axis message leaves the destination's axis stream
        exactly as it found it rather than returning a zero for it.
        """
        self._check(edges, source, destination)
        attending = self.reduce == "attention"

        aggregate = self._aggregate(
            self.wv_inv(self.ln_src_inv(source.inv)),
            edges.src,
            edges.dst,
            edges.relation,
            edges.n_dst,
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
        # of a wider intermediate. Edges on no axis are dropped first: they have
        # no channel to land in (§11.3).
        edge_src, edge_dst, edge_relation, edge_axis = edges.routed()
        aggregate = self._aggregate(
            self.wv_axis(self.ln_src_axis(source.axis)).reshape(-1, self.d_axis),
            edge_src * AXIS_CHANNELS + edge_axis,
            edge_dst * AXIS_CHANNELS + edge_axis,
            edge_relation,
            edges.n_dst * AXIS_CHANNELS,
            gate_projection=self.wg_axis if self.gated else None,
            bias_projection=self.wb_axis,
            score_vector=self.score_axis if attending else None,
        ).reshape(edges.n_dst, AXIS_CHANNELS, self.d_axis)
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
    the other way, so the class of a slot means the same thing in both — and
    keep private projections and update MLPs, which is what §14 fixes as shared
    and what it fixes as block-private. The trunk runs ``to_windows`` first and
    then ``to_cells``, which reads the windows that pass just updated.
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

    One relation class and always an axis route, so this path is where a cell
    learns the immediate shape of the line it sits on, in the channel of that
    line's own axis.
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
    stone's colour, so a far legal cell that lies in no current window is still
    described by the shape of the stones around it. ``route_on_axis_radius_messages``
    turns the axis route off for the whole family, and then the module holds no
    axis parameters.
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
    "AdjacencyMessage",
    "CellWindowIncidence",
    "RadiusMessage",
    "RelationGatedMessage",
    "TypedEdges",
    "adjacency_edges",
    "adjacency_relation_id",
    "aggregate_by_destination",
    "incidence_edges",
    "make_relation_embedding",
    "radius_edges",
    "radius_relation_count",
    "relation_vocabulary_size",
    "segment_counts",
    "segment_softmax",
    "segment_sum",
]
