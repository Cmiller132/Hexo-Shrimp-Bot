"""Global state latents and action-set latents: §17 and §21.

This is the module that makes the architecture affordable. Cells, windows, and
actions never see each other directly — §3.14 forbids any quadratic attention
over those families, and §26 budgets the global path at
``O(K * (N_cells + N_windows))``. A fixed handful of per-position latent tokens
reads every node, mixes among themselves, and writes back, so board-wide
context costs one pass over the nodes per block instead of one pass over the
node pairs. Every operation here is linear in the node count; the only dense
attention is over the latents themselves, whose count is a configured constant.

The representation law (§12.1) governs both streams, and it forces three
things about the parameters:

- **Axis latents have one learned base, replicated across all three channels**
  (§17.1). A per-channel base is a parameter attached to an absolute axis,
  which §12.2 forbids outright: after a 60° rotation the channel that carried
  base 0 would carry base 1's identity and the model's answer would move.
- **Every axis-side projection is shared over the three channels**, and every
  axis-side read and broadcast pairs channel ``a`` of the latent with channel
  ``a`` of the node. Both sides permute together, so their contraction does
  not.
- **Invariant and axis streams talk only through symmetric pooling** — an
  `AxisPool` over the channel set going up, one shared broadcast coming down
  (§17.3). Anything else would have to name a channel.

The entity-type embedding of the read is not an axis identity: it separates
cells from windows, which no board symmetry exchanges, and §17.2 asks for it by
name.

Why the latent pair is not an `EquivariantState`. That container holds one
entity's two streams over a shared leading shape; §17.1 gives the latents
*different counts* — ``K_inv`` invariant tokens and ``K_axis`` axis tokens —
so they are two entity families, not one. `LatentState` is that pair, and the
pairing §12.4 needs is manufactured where `AxisMix` is applied: each axis
latent is mixed against the mean of the invariant latents, and the invariant
delta `AxisMix` computes for that pooled partner is averaged back over the
axis latents. Both directions are symmetric pools over a whole set, which is
exactly the invariant↔axis channel §17.3 permits, and §12.4 itself stays in
one place.

Axis latents therefore require invariant latents: an axis latent has no
invariant half of its own, so with ``K_inv == 0`` there is nothing for
`AxisMix` or the broadcast pool to pair with. That combination is refused by
name rather than silently substituting a zero partner. No preset asks for it.

Zero is a supported count otherwise. ``full_no_latents`` (``global_mode
= "none"``), ``full_one_latent``, ``full_no_action_latents``, and
``full_no_axis`` each remove a stream; a removed stream constructs no
parameters and its pass returns its inputs unchanged, which is what §32
requires of a disabled optional module.

Working set. The read materialises one ``(N, K, C, heads, head_dim)`` fp32
tensor per stream, which is the linear cost §26 budgets times a constant, and
at batch sizes past a few dozen positions it is the largest allocation in the
block. §36 puts correctness before kernels and §17.5 permits exactly this
shape of implementation; the fused segment-attention kernel that removes the
materialisation is Stage E's, and its parity target is this code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from .builder import GLOBAL_NUMERIC_FEATURES
from .config import MantisACTConfig
from .equivariant import (
    AXIS_CHANNELS,
    AxisMix,
    AxisPool,
    EquivariantNorm,
    EquivariantState,
    LayerScale,
    activation_module,
    at_least_fp32,
)


def row_positions(offsets: Tensor) -> Tensor:
    """The position owning each row of a ragged family, from its CSR offsets."""
    if offsets.ndim != 1 or offsets.shape[0] < 1:
        raise ValueError(
            f"offsets must be a 1-D (P + 1,) tensor, got {tuple(offsets.shape)}"
        )
    counts = offsets[1:] - offsets[:-1]
    return torch.repeat_interleave(
        torch.arange(offsets.shape[0] - 1, device=offsets.device), counts
    )


def segment_cross_attention(
    q: Tensor, k: Tensor, v: Tensor, row_pos: Tensor, position_count: int
) -> Tensor:
    """Latent queries attend over their own position's ragged rows (§17.5).

    ``q`` is ``(P, K, C, heads, head_dim)``: ``K`` latent slots per position,
    each holding ``C`` channels that the keys share. ``C`` is 1 for the
    invariant stream and 3 for the axis stream, where channel ``a`` of the
    query only ever meets channel ``a`` of the key — that pairing is what makes
    the axis read equivariant rather than merely parameter-shared.

    ``k`` and ``v`` are ``(N, C, heads, head_dim)`` flat rows and ``row_pos``
    is their ``(N,)`` position index; rows need not be grouped by position, so
    several node families can be concatenated into one softmax. Scores,
    softmax, and the weighted sum run at no less than fp32 (§27).

    A position with no rows scores nothing and reads zero. It is reachable —
    ``full_occupied_cells_only`` has no cell nodes at all on an empty board —
    and zero is the only finite answer a softmax over an empty set can give.

    Cost is ``O(N * K * C * heads * head_dim)``: every row is scored against
    its own position's ``K`` latents and against nothing else, which is the
    whole reason the global path is linear in nodes (§3.14, §26).
    """
    if q.ndim != 5:
        raise ValueError(f"q must be (P, K, C, heads, head_dim), got {tuple(q.shape)}")
    if k.ndim != 4 or k.shape != v.shape:
        raise ValueError(
            f"k and v must share shape (N, C, heads, head_dim), got "
            f"{tuple(k.shape)} and {tuple(v.shape)}"
        )
    if q.shape[0] != position_count:
        raise ValueError(
            f"q holds {q.shape[0]} positions but position_count={position_count}"
        )
    if q.shape[2:] != k.shape[1:]:
        raise ValueError(
            f"q channels/heads/head_dim {tuple(q.shape[2:])} disagree with k's "
            f"{tuple(k.shape[1:])}"
        )
    if row_pos.ndim != 1 or row_pos.shape[0] != k.shape[0]:
        raise ValueError(
            f"row_pos must be ({k.shape[0]},) to match the key rows, got "
            f"{tuple(row_pos.shape)}"
        )

    channels, heads, head_dim = k.shape[1:]
    slots = q.shape[1]
    scale = 1.0 / math.sqrt(head_dim)
    stats = (position_count, slots, channels, heads)

    # (N, K, C, heads): each row against its own position's latents, and no
    # other row's. The queries are promoted *before* the gather, not after:
    # `index_select` backward is `index_add_` into a zero tensor of the
    # source's dtype, and a bf16 scatter of N row gradients over P latent rows
    # runs CUDA's compare-and-swap emulation of an atomic it has no instruction
    # for (§27).
    query = at_least_fp32(q).index_select(0, row_pos)
    score = (query * at_least_fp32(k).unsqueeze(1)).sum(-1)
    score = score * scale

    # Segment softmax per (position, slot, channel, head), shifted by the
    # segment maximum so a long ragged segment cannot overflow the exponential.
    maxima = score.new_full(stats, torch.finfo(score.dtype).min)
    maxima.index_reduce_(0, row_pos, score, "amax", include_self=True)
    weight = (score - maxima.index_select(0, row_pos)).exp_()
    total = score.new_zeros(stats).index_add_(0, row_pos, weight)

    out = score.new_zeros((*stats, head_dim)).index_add_(
        0, row_pos, weight.unsqueeze(-1) * at_least_fp32(v).unsqueeze(1)
    )
    return out / torch.where(total > 0, total, torch.ones_like(total)).unsqueeze(-1)


def _dense_softmax_attention(score: Tensor, value: Tensor, dim: int) -> Tensor:
    """Softmax ``score`` over ``dim`` at fp32 and contract it against ``value``.

    Used for every attention whose key set is a configured constant — latent
    self-mixing across ``K``, and the broadcast where each node reads a fixed
    handful of context rows. ``value`` carries one trailing ``head_dim``
    dimension past ``score``'s shape.
    """
    weight = at_least_fp32(score).softmax(dim=dim)
    return (weight.unsqueeze(-1) * at_least_fp32(value)).sum(dim=dim)


@dataclass(frozen=True)
class LatentState:
    """One batch's latent tokens (§17.1).

    ``inv`` is ``(P, K_inv, d_inv)`` and ``axis`` is ``(P, K_axis, 3, d_axis)``.
    A stream the configuration removes is ``None`` rather than a zero-width
    tensor, so a path that forgot to check raises instead of silently
    contributing nothing.
    """

    inv: Tensor | None
    axis: Tensor | None


@dataclass(frozen=True)
class RaggedStream:
    """One node family the latents read from and broadcast to.

    ``state`` is the family's `EquivariantState` over ``(N,)`` rows and
    ``offsets`` is its ``(P + 1,)`` CSR offsets from the packed batch, so the
    rows of one position are a contiguous slice and no read crosses a position
    (§26).
    """

    state: EquivariantState
    offsets: Tensor

    @property
    def rows(self) -> int:
        return int(self.state.inv.shape[0])


def _init_table(rows: int, width: int) -> nn.Parameter:
    """An embedding or latent base at §27's ``N(0, 0.02)``."""
    return nn.Parameter(torch.randn(rows, width) * 0.02)


def _check_counts(cfg: MantisACTConfig, num_inv: int, num_axis: int) -> None:
    """The latent counts a stack or a pass can actually implement."""
    if num_inv < 0 or num_axis < 0:
        raise ValueError(
            f"latent counts must not be negative: num_inv={num_inv}, "
            f"num_axis={num_axis}"
        )
    if num_axis > 0 and cfg.d_axis == 0:
        raise ValueError(f"num_axis={num_axis} needs axis channels, but d_axis=0")
    if num_axis > 0 and num_inv == 0:
        raise ValueError(
            f"num_axis={num_axis} axis latents with num_inv=0: an axis latent "
            "carries no invariant half, so §17.3's AxisMix and its symmetric "
            "pooling have no invariant partner to mix with"
        )


class LatentPass(nn.Module):
    """One block's latent read, self-mix, and broadcast (§17.2–§17.4, §21).

    Parameters are block-private, matching §14's rule for the message-passing
    projections; only the latent bases, which live on `StateLatents` and
    `ActionLatents`, are shared across a stack.

    ``entity_names`` fixes the node families this pass serves, in the order
    their entity-type embedding rows are laid out: cells and windows for the
    state stack, actions alone for the action stack. Every family gets its own
    norms per §18 and §27; the projections are shared across families and the
    families are told apart by the type embedding §17.2 asks for.
    """

    def __init__(
        self,
        cfg: MantisACTConfig,
        *,
        num_inv: int,
        num_axis: int,
        entity_names: Sequence[str],
    ) -> None:
        super().__init__()
        _check_counts(cfg, num_inv, num_axis)
        names = tuple(entity_names)
        if not names or len(set(names)) != len(names):
            raise ValueError(f"entity_names must be distinct and nonempty, got {names}")

        self.cfg = cfg
        self.num_inv = num_inv
        self.num_axis = num_axis
        self.entity_names = names
        self.has_inv = num_inv > 0
        self.has_axis = num_axis > 0
        # The invariant read pools the *nodes'* axis states even when there is
        # no axis latent, because §17.2 makes that pool part of the invariant
        # key. It needs axis channels to exist, not axis latents.
        self.pools_node_axis = self.has_inv and cfg.d_axis > 0

        d_inv, d_axis, heads = cfg.d_inv, cfg.d_axis, cfg.num_heads
        if d_axis and d_axis % heads:
            raise ValueError(f"d_axis={d_axis} must divide into num_heads={heads} heads")
        self.head_dim_inv = d_inv // heads
        self.head_dim_axis = d_axis // heads if d_axis else 0
        self.drop = nn.Dropout(cfg.dropout)
        if not self.enabled:
            return

        families = len(names)

        # --- read (§17.2) -------------------------------------------------
        # One norm pair per family: its invariant half is the invariant key,
        # its axis half feeds both the symmetric pool and the axis keys.
        self.norm_src = nn.ModuleList(EquivariantNorm(cfg) for _ in names)
        self.type_read_inv = _init_table(families, d_inv)
        self.norm_read_q_inv = nn.LayerNorm(d_inv)
        self.q_read_inv = nn.Linear(d_inv, d_inv)
        self.k_read_inv = nn.Linear(d_inv, d_inv)
        self.v_read_inv = nn.Linear(d_inv, d_inv)
        self.o_read_inv = nn.Linear(d_inv, d_inv)
        self.scale_read_inv = LayerScale(d_inv, cfg.layer_scale_init)
        if self.pools_node_axis:
            self.pool_src_axis = AxisPool(cfg)
            self.pool_to_inv = nn.Linear(d_axis, d_inv)
        if self.has_axis:
            self.type_read_axis = _init_table(families, d_axis)
            self.norm_read_q_axis = nn.LayerNorm(d_axis)
            self.q_read_axis = nn.Linear(d_axis, d_axis)
            self.k_read_axis = nn.Linear(d_axis, d_axis)
            self.v_read_axis = nn.Linear(d_axis, d_axis)
            self.o_read_axis = nn.Linear(d_axis, d_axis)
            self.scale_read_axis = LayerScale(d_axis, cfg.layer_scale_init)

        # --- mix (§17.3) --------------------------------------------------
        self.norm_mix_inv = nn.LayerNorm(d_inv)
        self.q_mix_inv = nn.Linear(d_inv, d_inv)
        self.k_mix_inv = nn.Linear(d_inv, d_inv)
        self.v_mix_inv = nn.Linear(d_inv, d_inv)
        self.o_mix_inv = nn.Linear(d_inv, d_inv)
        self.scale_mix_inv = LayerScale(d_inv, cfg.layer_scale_init)
        if self.has_axis:
            self.norm_mix_axis = nn.LayerNorm(d_axis)
            self.q_mix_axis = nn.Linear(d_axis, d_axis)
            self.k_mix_axis = nn.Linear(d_axis, d_axis)
            self.v_mix_axis = nn.Linear(d_axis, d_axis)
            self.o_mix_axis = nn.Linear(d_axis, d_axis)
            self.scale_mix_axis = LayerScale(d_axis, cfg.layer_scale_init)
            self.axis_mix = AxisMix(cfg)

        # --- broadcast (§17.4) --------------------------------------------
        # The invariant context rows are the invariant latents and the
        # channel-pooled axis latents, told apart by a source embedding with
        # one row per group present. Pooling is what keeps the axis
        # contribution invariant.
        self.norm_bcast_src_inv = nn.LayerNorm(d_inv)
        self.type_bcast_src = _init_table(1 + int(self.has_axis), d_inv)
        if self.has_axis:
            self.norm_bcast_src_axis = nn.LayerNorm(d_axis)
            self.pool_latent_axis = AxisPool(cfg)
            self.axis_to_inv = nn.Linear(d_axis, d_inv)
        self.k_bcast_inv = nn.Linear(d_inv, d_inv)
        self.v_bcast_inv = nn.Linear(d_inv, d_inv)
        self.norm_bcast_q_inv = nn.ModuleList(nn.LayerNorm(d_inv) for _ in names)
        self.q_bcast_inv = nn.Linear(d_inv, d_inv)
        self.o_bcast_inv = nn.Linear(d_inv, d_inv)
        self.scale_bcast_inv = LayerScale(d_inv, cfg.layer_scale_init)
        if self.has_axis:
            self.k_bcast_axis = nn.Linear(d_axis, d_axis)
            self.v_bcast_axis = nn.Linear(d_axis, d_axis)
            self.norm_bcast_q_axis = nn.ModuleList(nn.LayerNorm(d_axis) for _ in names)
            self.q_bcast_axis = nn.Linear(d_axis, d_axis)
            self.o_bcast_axis = nn.Linear(d_axis, d_axis)
            self.scale_bcast_axis = LayerScale(d_axis, cfg.layer_scale_init)

    @property
    def enabled(self) -> bool:
        """Whether this pass holds parameters and contributes anything (§32)."""
        return self.has_inv or self.has_axis

    def _ordered(self, entities: Mapping[str, RaggedStream]) -> list[RaggedStream]:
        """The named families in embedding-row order, refusing a mismatch."""
        if set(entities) != set(self.entity_names):
            raise ValueError(
                f"entities {sorted(entities)} do not match this pass's families "
                f"{list(self.entity_names)}"
            )
        streams = [entities[name] for name in self.entity_names]
        counts = {int(s.offsets.shape[0]) - 1 for s in streams}
        if len(counts) != 1:
            raise ValueError(f"families disagree on position count: {sorted(counts)}")
        for name, stream in zip(self.entity_names, streams):
            if int(stream.offsets[-1]) != stream.rows:
                raise ValueError(
                    f"{name} offsets end at {int(stream.offsets[-1])} but the "
                    f"family carries {stream.rows} rows"
                )
            if stream.state.d_axis != self.cfg.d_axis:
                raise ValueError(
                    f"{name} carries an axis width of {stream.state.d_axis} but "
                    f"the configuration has d_axis={self.cfg.d_axis}"
                )
        return streams

    def _require(self, latents: LatentState) -> tuple[Tensor, Tensor | None]:
        """The latent tensors this pass was built to update."""
        if self.has_inv and latents.inv is None:
            raise ValueError("this pass has invariant latents but got LatentState.inv=None")
        if self.has_axis and latents.axis is None:
            raise ValueError("this pass has axis latents but got LatentState.axis=None")
        return latents.inv, latents.axis

    def read(
        self, latents: LatentState, entities: Mapping[str, RaggedStream]
    ) -> LatentState:
        """Latents attend over every node of their own position (§17.2)."""
        if not self.enabled:
            return latents
        streams = self._ordered(entities)
        inv, axis = self._require(latents)
        positions = int(streams[0].offsets.shape[0]) - 1
        heads = self.cfg.num_heads
        row_pos = torch.cat([row_positions(s.offsets) for s in streams])
        normed = [self.norm_src[i](s.state) for i, s in enumerate(streams)]

        keys = []
        for index, source in enumerate(normed):
            feature = source.inv + self.type_read_inv[index]
            if self.pools_node_axis:
                # A symmetric pool over the whole channel set: invariant, so
                # it may enter the invariant key (§17.2, §12.3).
                feature = feature + self.pool_to_inv(self.pool_src_axis(source))
            keys.append(feature)
        rows = torch.cat(keys)
        n_rows = rows.shape[0]
        out = segment_cross_attention(
            self.q_read_inv(self.norm_read_q_inv(inv)).view(
                positions, self.num_inv, 1, heads, self.head_dim_inv
            ),
            self.k_read_inv(rows).view(n_rows, 1, heads, self.head_dim_inv),
            self.v_read_inv(rows).view(n_rows, 1, heads, self.head_dim_inv),
            row_pos,
            positions,
        )
        delta = self.o_read_inv(
            out.reshape(positions, self.num_inv, self.cfg.d_inv).to(inv.dtype)
        )
        inv = inv + self.scale_read_inv(self.drop(delta))

        if self.has_axis:
            rows = torch.cat(
                [
                    source.axis + self.type_read_axis[index]
                    for index, source in enumerate(normed)
                ]
            )
            n_rows = rows.shape[0]
            out = segment_cross_attention(
                self.q_read_axis(self.norm_read_q_axis(axis)).view(
                    positions, self.num_axis, AXIS_CHANNELS, heads, self.head_dim_axis
                ),
                self.k_read_axis(rows).view(
                    n_rows, AXIS_CHANNELS, heads, self.head_dim_axis
                ),
                self.v_read_axis(rows).view(
                    n_rows, AXIS_CHANNELS, heads, self.head_dim_axis
                ),
                row_pos,
                positions,
            )
            delta = self.o_read_axis(
                out.reshape(
                    positions, self.num_axis, AXIS_CHANNELS, self.cfg.d_axis
                ).to(axis.dtype)
            )
            axis = axis + self.scale_read_axis(self.drop(delta))

        return LatentState(inv=inv, axis=axis)

    def mix(self, latents: LatentState) -> LatentState:
        """Latents self-attend within their stream, then `AxisMix` (§17.3)."""
        if not self.enabled:
            return latents
        inv, axis = self._require(latents)
        heads = self.cfg.num_heads

        positions, slots, _ = inv.shape
        z = self.norm_mix_inv(inv)
        shape = (positions, slots, heads, self.head_dim_inv)
        score = torch.einsum(
            "pqhd,pkhd->pqkh", self.q_mix_inv(z).view(shape), self.k_mix_inv(z).view(shape)
        ) / math.sqrt(self.head_dim_inv)
        out = _dense_softmax_attention(score, self.v_mix_inv(z).view(shape).unsqueeze(1), 2)
        delta = self.o_mix_inv(out.reshape(positions, slots, self.cfg.d_inv).to(inv.dtype))
        inv = inv + self.scale_mix_inv(self.drop(delta))

        if not self.has_axis:
            return LatentState(inv=inv, axis=axis)

        slots = axis.shape[1]
        z = self.norm_mix_axis(axis)
        shape = (positions, slots, AXIS_CHANNELS, heads, self.head_dim_axis)
        # Attention runs inside each channel under one shared parameter set, so
        # the three channels permute with the board (§12.3).
        score = torch.einsum(
            "pqahd,pkahd->pqkah",
            self.q_mix_axis(z).view(shape),
            self.k_mix_axis(z).view(shape),
        ) / math.sqrt(self.head_dim_axis)
        out = _dense_softmax_attention(score, self.v_mix_axis(z).view(shape).unsqueeze(1), 2)
        delta = self.o_mix_axis(
            out.reshape(positions, slots, AXIS_CHANNELS, self.cfg.d_axis).to(axis.dtype)
        )
        axis = axis + self.scale_mix_axis(self.drop(delta))

        # §12.4 over the manufactured pairing: each axis latent against the
        # mean of the invariant latents. `AxisMix` returns both gated
        # residuals; the invariant one belongs to the pooled partner, so it is
        # averaged back over the axis latents and applied to every invariant
        # latent. Both directions are pools over a whole set, which is the
        # invariant<->axis channel §17.3 permits.
        paired = EquivariantState(
            inv.mean(dim=1, keepdim=True).expand(-1, self.num_axis, -1), axis
        )
        mixed = self.axis_mix(paired)
        inv = inv + (mixed.inv - paired.inv).mean(dim=1, keepdim=True)
        return LatentState(inv=inv, axis=mixed.axis)

    def broadcast(
        self, latents: LatentState, entities: Mapping[str, RaggedStream]
    ) -> dict[str, RaggedStream]:
        """Nodes read the latents back (§17.4).

        Each node attends over the small fixed context set of its own position,
        so the cost is one gather and one softmax per node — linear again, and
        the softmax key count is a configured constant rather than a node
        count.
        """
        if not self.enabled:
            return dict(entities)
        streams = self._ordered(entities)
        inv, axis = self._require(latents)
        heads = self.cfg.num_heads
        d_inv, d_axis = self.cfg.d_inv, self.cfg.d_axis

        context = [self.norm_bcast_src_inv(inv) + self.type_bcast_src[0]]
        if self.has_axis:
            normed_axis = self.norm_bcast_src_axis(axis)
            pooled = EquivariantState(
                inv.mean(dim=1, keepdim=True).expand(-1, self.num_axis, -1), normed_axis
            )
            context.append(
                self.axis_to_inv(self.pool_latent_axis(pooled)) + self.type_bcast_src[1]
            )
        context = torch.cat(context, dim=1)  # (P, R, d_inv)
        positions, context_rows = context.shape[0], context.shape[1]
        # The context is promoted before every node gathers from it. The
        # softmax and its weighted sum are fp32 either way (§27), but the
        # gather's *backward* is an `index_add_` in the source's dtype, and
        # here that scatters one gradient row per node onto a handful of
        # per-position context rows: in bf16 that is CUDA's compare-and-swap
        # emulation under maximal contention, which measured 63% of the whole
        # trunk's backward before the promotion moved above the gather.
        inv_shape = (positions, context_rows, heads, self.head_dim_inv)
        key_inv = at_least_fp32(self.k_bcast_inv(context)).view(inv_shape)
        value_inv = at_least_fp32(self.v_bcast_inv(context)).view(inv_shape)
        if self.has_axis:
            axis_shape = (
                positions,
                self.num_axis,
                AXIS_CHANNELS,
                heads,
                self.head_dim_axis,
            )
            key_axis = at_least_fp32(self.k_bcast_axis(normed_axis)).view(axis_shape)
            value_axis = at_least_fp32(self.v_bcast_axis(normed_axis)).view(axis_shape)

        updated: dict[str, RaggedStream] = {}
        for index, (name, stream) in enumerate(zip(self.entity_names, streams)):
            pos = row_positions(stream.offsets)
            n_rows = stream.rows
            node = stream.state

            query = self.q_bcast_inv(self.norm_bcast_q_inv[index](node.inv)).view(
                n_rows, heads, self.head_dim_inv
            )
            score = (query.unsqueeze(1) * key_inv.index_select(0, pos)).sum(-1)
            out = _dense_softmax_attention(
                score / math.sqrt(self.head_dim_inv), value_inv.index_select(0, pos), 1
            )
            delta = self.o_bcast_inv(out.reshape(n_rows, d_inv).to(node.inv.dtype))
            node_inv = node.inv + self.scale_bcast_inv(self.drop(delta))

            node_axis = node.axis
            if self.has_axis:
                query = self.q_bcast_axis(self.norm_bcast_q_axis[index](node.axis)).view(
                    n_rows, AXIS_CHANNELS, heads, self.head_dim_axis
                )
                # Channel `a` of the node meets channel `a` of the latent and
                # nothing else, which is what carries §12.1 through the write.
                score = (query.unsqueeze(1) * key_axis.index_select(0, pos)).sum(-1)
                out = _dense_softmax_attention(
                    score / math.sqrt(self.head_dim_axis),
                    value_axis.index_select(0, pos),
                    1,
                )
                delta = self.o_bcast_axis(
                    out.reshape(n_rows, AXIS_CHANNELS, d_axis).to(node.axis.dtype)
                )
                node_axis = node.axis + self.scale_bcast_axis(self.drop(delta))

            updated[name] = RaggedStream(
                EquivariantState(node_inv, node_axis), stream.offsets
            )
        return updated

    def forward(
        self, latents: LatentState, entities: Mapping[str, RaggedStream]
    ) -> tuple[LatentState, dict[str, RaggedStream]]:
        """Read, mix, then broadcast — steps 6 to 8 of the §18 block."""
        if not self.enabled:
            return latents, dict(entities)
        latents = self.mix(self.read(latents, entities))
        return latents, self.broadcast(latents, entities)


class _LatentStack(nn.Module):
    """Latent bases plus one `LatentPass` per block.

    The bases are stack-level because they are the tokens' identity: one set
    initialised once and carried through every block, not re-created per block.
    Invariant latents have distinct learned identities; each axis latent has a
    single learned base replicated across the three channels, because a
    per-channel base would attach a parameter to an absolute axis, which §12.2
    forbids and §17.1 and §27 restate.
    """

    def __init__(
        self,
        cfg: MantisACTConfig,
        *,
        num_inv: int,
        num_axis: int,
        blocks: int,
        entity_names: Sequence[str],
    ) -> None:
        super().__init__()
        if blocks < 0:
            raise ValueError(f"blocks={blocks} must not be negative")
        # Checked here as well as in every pass: with `blocks == 0` no pass is
        # built, and the stack would otherwise hold bases it never validates.
        _check_counts(cfg, num_inv, num_axis)
        if blocks == 0 and (num_inv or num_axis):
            raise ValueError(
                f"blocks=0 with {num_inv} invariant and {num_axis} axis latents: "
                "no pass would read the bases, so they would be parameters no "
                "forward touches"
            )
        self.cfg = cfg
        self.num_inv = num_inv
        self.num_axis = num_axis
        self.passes = nn.ModuleList(
            LatentPass(cfg, num_inv=num_inv, num_axis=num_axis, entity_names=entity_names)
            for _ in range(blocks)
        )
        if num_inv > 0:
            self.base_inv = _init_table(num_inv, cfg.d_inv)
        if num_axis > 0:
            self.base_axis = _init_table(num_axis, cfg.d_axis)

    @property
    def enabled(self) -> bool:
        return self.num_inv > 0 or self.num_axis > 0

    def __len__(self) -> int:
        return len(self.passes)

    def __getitem__(self, index: int) -> LatentPass:
        return self.passes[index]

    def _bases(self, positions: int, device, dtype) -> LatentState:
        inv = None
        if self.num_inv > 0:
            inv = self.base_inv.to(device=device, dtype=dtype).expand(
                positions, self.num_inv, self.cfg.d_inv
            )
        axis = None
        if self.num_axis > 0:
            axis = (
                self.base_axis.to(device=device, dtype=dtype)[:, None, :]
                .expand(self.num_axis, AXIS_CHANNELS, self.cfg.d_axis)
                .expand(positions, self.num_axis, AXIS_CHANNELS, self.cfg.d_axis)
            )
        return LatentState(inv=inv, axis=axis)


class StateLatents(_LatentStack):
    """The global state latents of §17, one pass per state trunk block.

    ``initial`` seeds them from the learned bases and, when
    ``use_global_numeric_features`` is on, from §13.3's state-derived scalars.
    Only the invariant latents take that seed: the scalars are invariant, and
    adding an invariant quantity to the axis channels would say nothing about
    direction while making the three channels' inputs identical.

    Use it from a trunk as::

        latents = state_latents.initial(batch.global_numeric)
        for index, block in enumerate(blocks):
            ...
            latents, entities = state_latents[index](latents, entities)
    """

    def __init__(self, cfg: MantisACTConfig) -> None:
        super().__init__(
            cfg,
            num_inv=cfg.num_inv_latents,
            num_axis=cfg.num_axis_latents,
            blocks=cfg.state_blocks,
            entity_names=("cell", "window"),
        )
        self.global_width = (
            GLOBAL_NUMERIC_FEATURES if cfg.use_global_numeric_features else 0
        )
        if self.global_width and cfg.num_inv_latents:
            self.global_mlp = nn.Sequential(
                nn.Linear(self.global_width, cfg.d_inv),
                activation_module(cfg.activation),
                nn.Linear(cfg.d_inv, cfg.num_inv_latents * cfg.d_inv),
            )

    def initial(self, global_numeric: Tensor) -> LatentState:
        """The starting latents of a batch, from its ``(P, G)`` global scalars."""
        if global_numeric.ndim != 2:
            raise ValueError(
                f"global_numeric must be (P, G), got {tuple(global_numeric.shape)}"
            )
        if global_numeric.shape[1] != self.global_width:
            raise ValueError(
                f"global_numeric has {global_numeric.shape[1]} columns but this "
                f"configuration expects {self.global_width}"
            )
        positions = global_numeric.shape[0]
        state = self._bases(positions, global_numeric.device, global_numeric.dtype)
        if self.global_width and self.num_inv > 0:
            seed = self.global_mlp(global_numeric).view(
                positions, self.num_inv, self.cfg.d_inv
            )
            state = LatentState(inv=state.inv + seed, axis=state.axis)
        return state


class ActionLatents(_LatentStack):
    """The action-set latents of §21, one pass per action block.

    Two invariant queries over the legal action set: read, self-mix, broadcast.
    They are a separate stack from `StateLatents` on purpose — §21 keeps them
    apart so the state latents never carry post-placement effects — and they
    have no axis stream, because §21 asks for invariant queries only. Direction
    reaches the action axis channels through the block's own `AxisMix` (§22.4),
    not through a second latent family.
    """

    def __init__(self, cfg: MantisACTConfig) -> None:
        super().__init__(
            cfg,
            num_inv=cfg.num_action_latents,
            num_axis=0,
            blocks=cfg.action_blocks,
            entity_names=("action",),
        )

    def initial(self, positions: int, *, device=None, dtype=None) -> LatentState:
        """The starting action latents of a batch of ``positions`` positions."""
        if positions < 0:
            raise ValueError(f"positions={positions} must not be negative")
        return self._bases(positions, device, dtype or torch.get_default_dtype())


__all__ = [
    "ActionLatents",
    "LatentPass",
    "LatentState",
    "RaggedStream",
    "StateLatents",
    "row_positions",
    "segment_cross_attention",
]
