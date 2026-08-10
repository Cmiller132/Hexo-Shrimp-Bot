"""Global state latents and action-set latents (§17, §21).

A fixed set of per-position latent tokens reads every node once per block and
writes back, keeping the global path linear in node count (§3.14, §26)
instead of attending over node pairs. Axis latents share one learned base
across all three channels and every axis-side projection is shared across
channels (§12.2); the invariant and axis streams mix only through symmetric
pooling (§17.3). `LatentState` holds the invariant and axis latent tensors;
`StateLatents` and `ActionLatents` own the learned bases for the state trunk
and the action stack.
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
from .latent_attention import (
    latent_broadcast,
    latent_read,
    latent_segments,
    row_positions,
)


def _dense_softmax_attention(score: Tensor, value: Tensor, dim: int) -> Tensor:
    """Softmax ``score`` over ``dim`` at fp32 and contract it against ``value``.

    ``value`` carries one trailing ``head_dim`` dimension past ``score``'s shape.
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

    ``state`` is the family's `EquivariantState` over ``(N,)`` rows, ``offsets``
    is its ``(P + 1,)`` CSR offsets from the packed batch, so the rows of one
    position are a contiguous slice and no read crosses a position (§26), and
    ``row_pos`` is the same information the other way round — the position
    owning each row. It travels with the stream rather than being derived per
    use because the trunk builds it once per family per forward and every
    block's read and broadcast reuses it.
    """

    state: EquivariantState
    offsets: Tensor
    row_pos: Tensor

    def __post_init__(self) -> None:
        if self.row_pos.ndim != 1 or int(self.row_pos.shape[0]) != self.rows:
            raise ValueError(
                f"row_pos is {tuple(self.row_pos.shape)} against the family's "
                f"({self.rows},) rows"
            )

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
    """One block's latent read, self-mix, and broadcast (§17.2-§17.4, §21).

    Parameters are block-private; only the latent bases, held on `StateLatents`
    and `ActionLatents`, are shared across a stack. ``entity_names`` fixes the
    node families this pass serves — cells and windows for the state stack,
    actions alone for the action stack — each with its own norms, sharing the
    read/mix/broadcast projections and told apart by a type embedding (§17.2).
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
        # Pools the nodes' axis states into the invariant key (§17.2); needs
        # axis channels to exist, not axis latents.
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
        # A mismatched offset/row count is caught by `row_positions`'s own ATen
        # check and by `RaggedStream.__post_init__`, so it is not re-checked here.
        for name, stream in zip(self.entity_names, streams):
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

    def _segments(self, streams, inv: Tensor, segments):
        """Build or validate the shared ragged view used by a latent read."""
        positions = int(streams[0].offsets.shape[0]) - 1
        if segments is None:
            return latent_segments(
                [stream.offsets for stream in streams],
                [stream.row_pos for stream in streams],
            )
        expected_rows = sum(stream.rows for stream in streams)
        if (
            segments.positions != positions
            or segments.families != len(streams)
            or segments.n_rows != expected_rows
        ):
            raise ValueError(
                "the precomputed latent segments describe "
                f"P={segments.positions}, F={segments.families}, "
                f"N={segments.n_rows} against P={positions}, "
                f"F={len(streams)}, N={expected_rows}"
            )
        if segments.device != inv.device:
            raise ValueError(
                f"latent segments are on {segments.device}, states on {inv.device}"
            )
        return segments

    def read(
        self,
        latents: LatentState,
        entities: Mapping[str, RaggedStream],
        *,
        segments=None,
    ) -> LatentState:
        """Latents attend over every node of their own position (§17.2)."""
        if not self.enabled:
            return latents
        streams = self._ordered(entities)
        inv, axis = self._require(latents)
        positions = int(streams[0].offsets.shape[0]) - 1
        heads = self.cfg.num_heads
        # One multi-range view shared by both streams; the ranges depend only
        # on the offsets.
        segments = self._segments(streams, inv, segments)
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
        out = latent_read(
            self.q_read_inv(self.norm_read_q_inv(inv)).view(
                positions, self.num_inv, 1, heads, self.head_dim_inv
            ),
            self.k_read_inv(rows).view(n_rows, 1, heads, self.head_dim_inv),
            self.v_read_inv(rows).view(n_rows, 1, heads, self.head_dim_inv),
            segments,
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
            out = latent_read(
                self.q_read_axis(self.norm_read_q_axis(axis)).view(
                    positions, self.num_axis, AXIS_CHANNELS, heads, self.head_dim_axis
                ),
                self.k_read_axis(rows).view(
                    n_rows, AXIS_CHANNELS, heads, self.head_dim_axis
                ),
                self.v_read_axis(rows).view(
                    n_rows, AXIS_CHANNELS, heads, self.head_dim_axis
                ),
                segments,
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

        # §12.4's pairing: each axis latent against the mean of the invariant
        # latents. AxisMix's invariant residual is averaged back over the axis
        # latents and applied to every invariant latent (§17.3).
        paired = EquivariantState(
            inv.mean(dim=1, keepdim=True).expand(-1, self.num_axis, -1), axis
        )
        mixed = self.axis_mix(paired)
        inv = inv + (mixed.inv - paired.inv).mean(dim=1, keepdim=True)
        return LatentState(inv=inv, axis=mixed.axis)

    def broadcast(
        self, latents: LatentState, entities: Mapping[str, RaggedStream]
    ) -> dict[str, RaggedStream]:
        """Nodes read the latents back (§17.4): one gather and one softmax per
        node, over its position's fixed-size context set."""
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
        # Context layout is (P, R, C, heads, head_dim); a node's channel `a`
        # meets latent channel `a` only, which carries §12.1 through the write.
        inv_shape = (positions, context_rows, 1, heads, self.head_dim_inv)
        key_inv = self.k_bcast_inv(context).view(inv_shape)
        value_inv = self.v_bcast_inv(context).view(inv_shape)
        if self.has_axis:
            axis_shape = (
                positions,
                self.num_axis,
                AXIS_CHANNELS,
                heads,
                self.head_dim_axis,
            )
            key_axis = self.k_bcast_axis(normed_axis).view(axis_shape)
            value_axis = self.v_bcast_axis(normed_axis).view(axis_shape)

        updated: dict[str, RaggedStream] = {}
        for index, (name, stream) in enumerate(zip(self.entity_names, streams)):
            pos = stream.row_pos
            n_rows = stream.rows
            node = stream.state

            query = self.q_bcast_inv(self.norm_bcast_q_inv[index](node.inv)).view(
                n_rows, 1, heads, self.head_dim_inv
            )
            out = latent_broadcast(query, key_inv, value_inv, pos, stream.offsets)
            delta = self.o_bcast_inv(out.reshape(n_rows, d_inv).to(node.inv.dtype))
            node_inv = node.inv + self.scale_bcast_inv(self.drop(delta))

            node_axis = node.axis
            if self.has_axis:
                query = self.q_bcast_axis(self.norm_bcast_q_axis[index](node.axis)).view(
                    n_rows, AXIS_CHANNELS, heads, self.head_dim_axis
                )
                out = latent_broadcast(query, key_axis, value_axis, pos, stream.offsets)
                delta = self.o_bcast_axis(
                    out.reshape(n_rows, AXIS_CHANNELS, d_axis).to(node.axis.dtype)
                )
                node_axis = node.axis + self.scale_bcast_axis(self.drop(delta))

            updated[name] = RaggedStream(
                EquivariantState(node_inv, node_axis), stream.offsets, stream.row_pos
            )
        return updated

    def forward(
        self,
        latents: LatentState,
        entities: Mapping[str, RaggedStream],
        *,
        segments=None,
    ) -> tuple[LatentState, dict[str, RaggedStream]]:
        """Read, mix, then broadcast — steps 6 to 8 of the §18 block."""
        if not self.enabled:
            return latents, dict(entities)

        # Whole-pass fusion is deliberately coupled to collate-built plans.
        # Ablations and direct module callers without plans retain the literal
        # formulation below, including its public validation behaviour.
        if segments is not None:
            from .fused_latent import (
                action_eps,
                action_latent_pass,
                action_parameters,
                state_eps,
                state_latent_pass,
                state_parameters,
                supports_action_pass,
                supports_state_pass,
            )

            if supports_state_pass(self):
                streams = self._ordered(entities)
                inv, axis = self._require(latents)
                segments = self._segments(streams, inv, segments)
                assert axis is not None
                cell, window = streams
                result = state_latent_pass(
                    inv,
                    axis,
                    cell.state.inv,
                    cell.state.axis,
                    window.state.inv,
                    window.state.axis,
                    segments=segments,
                    cell_offsets=cell.offsets,
                    cell_row_pos=cell.row_pos,
                    window_offsets=window.offsets,
                    window_row_pos=window.row_pos,
                    params=state_parameters(self),
                    heads=self.cfg.num_heads,
                    activation=self.cfg.activation,
                    eps=state_eps(self),
                    dropout=self.cfg.dropout,
                    axis_pool_mode=self.cfg.axis_pool_mode,
                )
                return LatentState(result[0], result[1]), {
                    "cell": RaggedStream(
                        EquivariantState(result[2], result[3]),
                        cell.offsets,
                        cell.row_pos,
                    ),
                    "window": RaggedStream(
                        EquivariantState(result[4], result[5]),
                        window.offsets,
                        window.row_pos,
                    ),
                }

            if supports_action_pass(self):
                streams = self._ordered(entities)
                inv, axis = self._require(latents)
                segments = self._segments(streams, inv, segments)
                action = streams[0]
                result = action_latent_pass(
                    inv,
                    action.state.inv,
                    action.state.axis,
                    segments=segments,
                    action_offsets=action.offsets,
                    action_row_pos=action.row_pos,
                    params=action_parameters(self),
                    heads=self.cfg.num_heads,
                    activation=self.cfg.activation,
                    eps=action_eps(self),
                    dropout=self.cfg.dropout,
                    axis_pool_mode=self.cfg.axis_pool_mode,
                )
                return LatentState(result[0], axis), {
                    "action": RaggedStream(
                        EquivariantState(result[1], action.state.axis),
                        action.offsets,
                        action.row_pos,
                    )
                }

        latents = self.mix(self.read(latents, entities, segments=segments))
        return latents, self.broadcast(latents, entities)


class _LatentStack(nn.Module):
    """Latent bases plus one `LatentPass` per block.

    The bases are stack-level: one set of tokens carried through every block.
    Each axis latent shares a single learned base across its three channels;
    a per-channel base would attach a parameter to an absolute axis (§12.2,
    §17.1, §27).
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
        # Checked here too: with `blocks == 0` no pass is built, so the stack
        # would otherwise hold bases it never validates.
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
    ``use_global_numeric_features`` is on, from §13.3's state-derived scalars;
    only the invariant latents take that seed.

    ```python
    latents = state_latents.initial(batch.global_numeric)
    for index, block in enumerate(blocks):
        latents, entities = state_latents[index](latents, entities)
    ```
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

    A separate stack from `StateLatents` so the state latents never carry
    post-placement effects. Invariant-only (§21); direction reaches the action
    axis channels through the block's own `AxisMix` (§22.4) instead.
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
]
