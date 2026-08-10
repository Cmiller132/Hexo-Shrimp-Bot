"""Action encoder: §19's counterfactual windows, §21's latents, §22's blocks.

Encodes what each legal placement would do to the board. Each action is built
from eighteen windows (§19), reads state latent context and action-set latents
(§21, §22), and never writes back to the trunk.

Build order: base cell state -> post-placement rows -> tactical vector ->
per-block latent context, action-set latent read/mix/broadcast, AxisMix, FFN,
phase FiLM. Missing-window rows read a learned sentinel state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .actions import TACTICAL_FEATURES
from .class_embedding import class_pair_embedding
from .config import MantisACTConfig
from .equivariant import (
    AXIS_CHANNELS,
    AxisMix,
    AxisPool,
    EquivariantFFN,
    EquivariantNorm,
    EquivariantResidual,
    EquivariantState,
    LayerScale,
    PhaseFiLM,
    activation_module,
    at_least_fp32,
    run_equivariant_stage,
)
from .latent_attention import latent_broadcast
from .linear_fusion import horizontal_linears
from .latents import ActionLatents, LatentPass, LatentState, RaggedStream
from .messages import make_relation_embedding
from .ordered_reductions import ordered_index_select
from .packed import POST_ACTION_ROWS, WINDOW_LEN, PackedACTBatch
from .pattern_classes import POST1_REL_CLASSES
from .post_rows import row_gate, sentinel_gather
from .plans import builder_fingerprint
from .state_trunk import (
    EMBEDDING_INIT_STD,
    WINDOW_STATUSES,
    TrunkOutput,
    refuse_unimplemented_paths,
)

# The row shape §19.2 and §25 fix: three axes by six candidate slots, dense.
ACTION_ROW_SHAPE = (AXIS_CHANNELS, WINDOW_LEN)


def _require_family_size(name: str, produced: int, batch_rows: int) -> None:
    """Refuse a trunk output whose row count does not match the batch's family.

    The packer bounds its index tables against the batch's node families
    (`packed.py:367-372`, re-derived after concatenation at
    `packed.py:674-680`); this checks that the trunk preserved those row counts.
    """
    if produced != batch_rows:
        raise ValueError(
            f"the trunk produced {produced} {name} states against the batch's "
            f"{batch_rows}: the packer's index bounds describe the batch's "
            "families, so a trunk that did not preserve rows would leave every "
            "gather here bounded against the wrong table"
        )


def _require_rows(name: str, values: Tensor, rows: int) -> None:
    """Refuse an action table that is not ``(rows, 3, 6)`` (§19.2, §25)."""
    expected = (rows, *ACTION_ROW_SHAPE)
    if tuple(values.shape) != expected:
        raise ValueError(
            f"{name} must be {expected} — all {POST_ACTION_ROWS} post-placement "
            f"rows of every legal action — got {tuple(values.shape)}"
        )


def _base(width: int) -> nn.Parameter:
    """A shared learned base vector at §27's ``N(0, 0.02)``."""
    return nn.Parameter(torch.randn(width) * EMBEDDING_INIT_STD)


def _mlp(width_in: int, hidden: int, width_out: int, cfg: MantisACTConfig) -> nn.Module:
    """The package's two-layer feed-forward shape, with ``cfg``'s activation."""
    return nn.Sequential(
        nn.Linear(width_in, hidden),
        activation_module(cfg.activation),
        nn.Linear(hidden, width_out),
    )


# --------------------------------------------------------------------------
# §19.1 Base action state


class ActionBaseState(nn.Module):
    """§19.1: an action starts as the trunk's final state of its own cell.

    ```text
    A_inv  = C_inv[legal_to_cell_index]
    A_axis = C_axis[legal_to_cell_index]
    ```

    Row ``j`` is ``legal_moves()[j]``; the gather preserves engine order (§8.3,
    §37.5) and never sorts.

    ``legal_to_cell_index`` is ``-1`` throughout under
    ``cell_scope="occupied_only"`` (§29), where the node set holds no legal
    cell. Those actions start from a shared learned base instead, reached by
    padding the cell table with it. The index is bounded to ``-1..n_cells - 1``
    by `packed.py:165`, re-derived post-concatenation by `packed.py:674-680`.
    """

    def __init__(self, cfg: MantisACTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.no_cell_inv = _base(cfg.d_inv)
        self.no_cell_axis = _base(cfg.d_axis) if cfg.d_axis else None

    def forward(self, batch: PackedACTBatch, cells: EquivariantState) -> EquivariantState:
        index = batch.legal_to_cell_index
        n_cells = cells.inv.shape[0]
        _require_family_size("cell", n_cells, batch.cell_occupancy.shape[0])
        if index.ndim != 1:
            raise ValueError(
                f"legal_to_cell_index must be (N_legal,), got {tuple(index.shape)}"
            )

        rows = torch.where(index >= 0, index, torch.full_like(index, n_cells))
        if batch.plans is None:
            raise ValueError("action base state requires collate-built execution plans")
        plan = batch.plans.action_rows.base_cell
        table = torch.cat([cells.inv, self.no_cell_inv.to(cells.inv.dtype)[None]])
        inv = ordered_index_select(table, rows, plan.ptr, plan.rows)
        if cells.axis is None:
            return EquivariantState(inv)
        pad = self.no_cell_axis.to(cells.axis.dtype).expand(
            1, AXIS_CHANNELS, self.cfg.d_axis
        )
        axis = ordered_index_select(
            torch.cat([cells.axis, pad]), rows, plan.ptr, plan.rows
        )
        return EquivariantState(inv, axis)


# --------------------------------------------------------------------------
# §19.2 The eighteen post-placement windows


@dataclass(frozen=True, eq=False)
class WindowRows:
    """The eighteen rows' window states, ``(N_legal, 3, 6, width)`` per stream.

    Not an `EquivariantState`: a row's axis half is one channel, the one its
    own axis names, so there is no channel dimension beside the invariant
    half. Dimension 1 indexes the row grid, not an entity's axis stream, and
    reaches an action's axis channels only when the summary is written back.
    ``axis`` is ``None`` under ``full_no_axis``. Equality is identity.
    """

    inv: Tensor
    axis: Tensor | None


class PostPlacementEncoder(nn.Module):
    """§19.2: all eighteen post-placement window rows of every legal action.

    For each of the six candidate slots on each of the three axes:

    ```text
    w      = W_final[action_window_index]  or  the shared pre-empty-window state
    e      = E_post1[action_post1_class] + E_status[action_pre_status]
    row    = MLP(sigmoid(W_g e) * (W_v LN(w) + W_b e))
    ```

    One shared row encoder over all eighteen rows of every action. The
    relation-gated combination is §14's, applied here as a dense contraction
    over a fixed ``[N, 3, 6]`` block rather than a segment reduction over an
    edge list. The gather-through-gate path is one registered op
    (`post_rows.row_gate`) to avoid keeping several ``(M, d)`` intermediates
    alive for the backward.

    The six rows of an axis are summed, not attended over, in fp32 (§27) — a
    softmax would normalise away how many of an action's six windows on that
    axis are live. Each axis summary is added to the matching action axis
    channel, and the three are pooled into the invariant stream by
    ``sum_a phi(s_a) / 3`` (§12.4's symmetric pool).

    ``action_post1_class`` and ``action_pre_status`` carry no sentinel and are
    bounded by the packer before a tensor exists (`packed.py:154-155`).
    """

    def __init__(self, cfg: MantisACTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d_inv, d_axis, d_rel = cfg.d_inv, cfg.d_axis, cfg.d_rel

        # Shared learned state of a row whose window the scope does not
        # persist, one base per stream, replicated over the three channels.
        self.pre_empty_inv = _base(d_inv)
        self.pre_empty_axis = _base(d_axis) if d_axis else None

        self.post1 = make_relation_embedding(POST1_REL_CLASSES, d_rel)
        self.pre_status = make_relation_embedding(WINDOW_STATUSES, d_rel)

        self.ln_inv = nn.LayerNorm(d_inv)
        # Bias-free per §14: the relation's own bias is the additive term.
        self.wv_inv = nn.Linear(d_inv, d_inv, bias=False)
        self.wb_inv = nn.Linear(d_rel, d_inv)
        self.wg_inv = nn.Linear(d_rel, d_inv)
        self.row_inv = _mlp(d_inv, cfg.ffn_mult * d_inv, d_inv, cfg)
        if d_axis:
            self.ln_axis = nn.LayerNorm(d_axis)
            self.wv_axis = nn.Linear(d_axis, d_axis, bias=False)
            self.wb_axis = nn.Linear(d_rel, d_axis)
            self.wg_axis = nn.Linear(d_rel, d_axis)
            self.row_axis = _mlp(d_axis, cfg.ffn_mult * d_axis, d_axis, cfg)
        else:
            self.ln_axis = None

        self.phi = nn.Sequential(nn.Linear(d_inv, d_inv), activation_module(cfg.activation))
        self.pool_inv = _mlp(d_inv, cfg.ffn_mult * d_inv, d_inv, cfg)
        self.residual = EquivariantResidual(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def window_rows(
        self, batch: PackedACTBatch, windows: EquivariantState
    ) -> WindowRows:
        """The eighteen rows' window states, ``(N_legal, 3, 6, ...)``.

        A row with a persistent window reads that window's final trunk state;
        a row without one reads the shared pre-empty-window base via
        `post_rows.sentinel_gather`, which resolves the ``-1`` index inside the
        gather rather than through a padded table. The axis half of a row is
        the window's channel ``a``, ``a`` being the row's window's own axis,
        derived from the row's flattened ``(window, channel)`` grid position.
        """
        index = batch.action_window_index
        n_windows = windows.inv.shape[0]
        n_legal = batch.legal_to_cell_index.shape[0]
        _require_rows("action_window_index", index, n_legal)
        _require_family_size(
            "window", n_windows, batch.window_pattern_class.shape[0]
        )

        if batch.plans is None:
            raise ValueError("post-placement rows require collate-built ACT plans")
        source_plan = batch.plans.action_rows.source_window
        inv = sentinel_gather(
            windows.inv,
            self.pre_empty_inv,
            index,
            1,
            source_plan.ptr,
            source_plan.rows,
            source_plan.sentinel_rows,
        )
        if windows.axis is None:
            return WindowRows(inv, None)
        axis = sentinel_gather(
            windows.axis.reshape(-1, self.cfg.d_axis),
            self.pre_empty_axis,
            index,
            AXIS_CHANNELS,
            source_plan.ptr,
            source_plan.rows,
            source_plan.sentinel_rows,
        )
        return WindowRows(inv, axis)

    def forward(
        self,
        batch: PackedACTBatch,
        actions: EquivariantState,
        windows: EquivariantState,
    ) -> EquivariantState:
        """``actions`` plus the eighteen rows' contribution (§19.2)."""
        n_legal = actions.inv.shape[0]
        _require_rows("action_post1_class", batch.action_post1_class, n_legal)
        _require_rows("action_pre_status", batch.action_pre_status, n_legal)

        source = self.window_rows(batch, windows)
        grid = (n_legal, AXIS_CHANNELS, WINDOW_LEN)
        if batch.plans is None:
            relation = self.post1(batch.action_post1_class) + self.pre_status(
                batch.action_pre_status
            )
            flat_relation = relation.reshape(-1, self.cfg.d_rel)
        else:
            post_plan = batch.plans.action_rows.post1
            status_plan = batch.plans.action_rows.pre_status
            flat_relation = class_pair_embedding(
                self.post1.weight,
                self.pre_status.weight,
                batch.action_post1_class.reshape(-1),
                batch.action_pre_status.reshape(-1),
                post_plan.ptr,
                post_plan.rows,
                status_plan.ptr,
                status_plan.rows,
                post_plan.block_ptr,
                post_plan.block_starts,
                post_plan.block_lengths,
                status_plan.block_ptr,
                status_plan.block_starts,
                status_plan.block_lengths,
            )

        gated = row_gate(
            source.inv.reshape(-1, self.cfg.d_inv),
            self.ln_inv.weight,
            self.ln_inv.bias,
            self.wv_inv.weight,
            flat_relation,
            self.wb_inv.weight,
            self.wb_inv.bias,
            self.wg_inv.weight,
            self.wg_inv.bias,
            self.ln_inv.eps,
        )
        # This MLP's hidden tensor is wider than all eighteen action rows.
        # The enclosing post checkpoint removes it from the original forward;
        # nesting here also keeps it out of the enclosing replay tape, so the
        # two row streams do not coexist at the backward peak.
        row = checkpoint(
            self.row_inv,
            gated,
            use_reentrant=False,
            preserve_rng_state=True,
        ).view(*grid, self.cfg.d_inv)
        # §27: the reduction over the six candidate slots is fp32.
        summary_inv = at_least_fp32(row).sum(dim=2).to(row.dtype)

        delta_axis = None
        if source.axis is not None:
            if self.ln_axis is None:
                raise ValueError(
                    f"this encoder was built for d_axis={self.cfg.d_axis} and "
                    "holds no axis half, but the windows carry an axis stream"
                )
            gated = row_gate(
                source.axis.reshape(-1, self.cfg.d_axis),
                self.ln_axis.weight,
                self.ln_axis.bias,
                self.wv_axis.weight,
                flat_relation,
                self.wb_axis.weight,
                self.wb_axis.bias,
                self.wg_axis.weight,
                self.wg_axis.bias,
                self.ln_axis.eps,
            )
            row = checkpoint(
                self.row_axis,
                gated,
                use_reentrant=False,
                preserve_rng_state=True,
            ).view(*grid, self.cfg.d_axis)
            delta_axis = self.drop(at_least_fp32(row).sum(dim=2).to(row.dtype))

        pooled = at_least_fp32(self.phi(summary_inv)).mean(dim=1).to(summary_inv.dtype)
        return self.residual(actions, self.drop(self.pool_inv(pooled)), delta_axis)


# --------------------------------------------------------------------------
# §19.3 The deterministic tactical vector


class TacticalEncoder(nn.Module):
    """§19.3: the deterministic tactical scalars, through a small invariant MLP.

    Every field, from `actions.tactical_features`, is a D6 invariant, so it
    enters the invariant stream only. Not built at all when
    ``use_action_tactical_features=False`` (§32): a disabled module holds no
    parameters rather than projecting a column of zeros.
    """

    def __init__(self, cfg: MantisACTConfig, width: int) -> None:
        super().__init__()
        if width < 1:
            raise ValueError(
                f"a {width}-column tactical block has nothing to encode; the "
                "caller must not build a TacticalEncoder under "
                "use_action_tactical_features=False (§32)"
            )
        self.width = width
        self.mlp = _mlp(width, cfg.d_inv, cfg.d_inv, cfg)
        self.scale = LayerScale(cfg.d_inv, cfg.layer_scale_init)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, actions: EquivariantState, numeric: Tensor) -> EquivariantState:
        if numeric.ndim != 2 or numeric.shape[0] != actions.inv.shape[0]:
            raise ValueError(
                f"action_tactical_numeric must be ({actions.inv.shape[0]}, "
                f"{self.width}), got {tuple(numeric.shape)}"
            )
        if numeric.shape[1] != self.width:
            raise ValueError(
                f"action_tactical_numeric carries {numeric.shape[1]} columns "
                f"against this configuration's {self.width}"
            )
        delta = self.drop(self.mlp(numeric.to(actions.inv.dtype)))
        return EquivariantState(actions.inv + self.scale(delta), actions.axis)


# --------------------------------------------------------------------------
# §22.1 State latent context


class StateContextBroadcast(nn.Module):
    """§22.1: actions read the state latents of their own position.

    The context rows are the invariant state latents and the channel-pooled
    axis latents, told apart by a source embedding; each action attends over
    that fixed set, so the key count is a configured constant rather than an
    action count (§3.14). The axis half pairs channel ``a`` of the action with
    channel ``a`` of the latent only (§12.1).

    Read-only: the latents are not updated or returned here (§21, §22). This is
    the action stack's counterpart to `latents.LatentPass`, which both reads
    and writes for the state trunk's own families.
    """

    def __init__(self, cfg: MantisACTConfig) -> None:
        super().__init__()
        if cfg.num_inv_latents < 1:
            raise ValueError(
                "StateContextBroadcast needs invariant state latents to read; "
                "the caller must not build one when num_inv_latents=0 (§32)"
            )
        heads = cfg.num_heads
        d_inv, d_axis = cfg.d_inv, cfg.d_axis
        if d_axis and d_axis % heads:
            raise ValueError(f"d_axis={d_axis} must divide into num_heads={heads} heads")

        self.cfg = cfg
        self.num_inv = cfg.num_inv_latents
        self.num_axis = cfg.num_axis_latents
        self.has_axis = cfg.num_axis_latents > 0 and d_axis > 0
        self.head_dim_inv = d_inv // heads
        self.head_dim_axis = d_axis // heads if d_axis else 0

        self.norm_src_inv = nn.LayerNorm(d_inv)
        self.type_src = nn.Parameter(
            torch.randn(1 + int(self.has_axis), d_inv) * EMBEDDING_INIT_STD
        )
        if self.has_axis:
            self.norm_src_axis = nn.LayerNorm(d_axis)
            self.pool_src_axis = AxisPool(cfg)
            self.axis_to_inv = nn.Linear(d_axis, d_inv)

        self.k_inv = nn.Linear(d_inv, d_inv)
        self.v_inv = nn.Linear(d_inv, d_inv)
        self.norm_q_inv = nn.LayerNorm(d_inv)
        self.q_inv = nn.Linear(d_inv, d_inv)
        self.o_inv = nn.Linear(d_inv, d_inv)
        self.scale_inv = LayerScale(d_inv, cfg.layer_scale_init)
        if self.has_axis:
            self.k_axis = nn.Linear(d_axis, d_axis)
            self.v_axis = nn.Linear(d_axis, d_axis)
            self.norm_q_axis = nn.LayerNorm(d_axis)
            self.q_axis = nn.Linear(d_axis, d_axis)
            self.o_axis = nn.Linear(d_axis, d_axis)
            self.scale_axis = LayerScale(d_axis, cfg.layer_scale_init)
        self.drop = nn.Dropout(cfg.dropout)

    def _require(self, latents: LatentState) -> tuple[Tensor, Tensor | None]:
        if latents.inv is None:
            raise ValueError(
                "this broadcast reads invariant state latents but the trunk "
                "returned LatentState.inv=None"
            )
        if self.has_axis and latents.axis is None:
            raise ValueError(
                "this broadcast reads axis state latents but the trunk returned "
                "LatentState.axis=None"
            )
        return latents.inv, latents.axis

    def forward(
        self,
        actions: EquivariantState,
        latents: LatentState,
        positions: Tensor,
        legal_offsets: Tensor,
    ) -> EquivariantState:
        inv, axis = self._require(latents)
        heads = self.cfg.num_heads
        n_legal = actions.inv.shape[0]

        context = [self.norm_src_inv(inv) + self.type_src[0]]
        if self.has_axis:
            normed_axis = self.norm_src_axis(axis)
            # A symmetric pool over the channel set keeps this contribution
            # invariant (§17.3).
            pooled = EquivariantState(
                inv.mean(dim=1, keepdim=True).expand(-1, self.num_axis, -1), normed_axis
            )
            context.append(self.axis_to_inv(self.pool_src_axis(pooled)) + self.type_src[1])
        rows = torch.cat(context, dim=1)
        n_positions, n_rows = rows.shape[0], rows.shape[1]

        # Promoted before the attention (§27). `latent_attention.latent_broadcast`
        # reduces each position's contiguous action run, so no per-node context
        # tensor is materialised — §17.4's broadcast with a one-slot context.
        shape = (n_positions, n_rows, 1, heads, self.head_dim_inv)
        key, value = horizontal_linears(rows, (self.k_inv, self.v_inv))
        key = at_least_fp32(key).view(shape)
        value = at_least_fp32(value).view(shape)
        query = at_least_fp32(self.q_inv(self.norm_q_inv(actions.inv))).view(
            n_legal, 1, heads, self.head_dim_inv
        )
        out = latent_broadcast(query, key, value, positions, legal_offsets)
        delta = self.o_inv(out.reshape(n_legal, self.cfg.d_inv).to(actions.inv.dtype))
        action_inv = actions.inv + self.scale_inv(self.drop(delta))

        action_axis = actions.axis
        if self.has_axis:
            if actions.axis is None:
                raise ValueError(
                    "this broadcast carries an axis half but the action state "
                    "has no axis stream"
                )
            shape = (
                n_positions,
                self.num_axis,
                AXIS_CHANNELS,
                heads,
                self.head_dim_axis,
            )
            key, value = horizontal_linears(
                normed_axis, (self.k_axis, self.v_axis)
            )
            key = at_least_fp32(key).view(shape)
            value = at_least_fp32(value).view(shape)
            query = at_least_fp32(self.q_axis(self.norm_q_axis(actions.axis))).view(
                n_legal, AXIS_CHANNELS, heads, self.head_dim_axis
            )
            out = latent_broadcast(query, key, value, positions, legal_offsets)
            delta = self.o_axis(
                out.reshape(n_legal, AXIS_CHANNELS, self.cfg.d_axis).to(actions.axis.dtype)
            )
            action_axis = actions.axis + self.scale_axis(self.drop(delta))
        return EquivariantState(action_inv, action_axis)


# --------------------------------------------------------------------------
# §22 The shared action block


class ActionBlock(nn.Module):
    """One of §22's shared action blocks, in the order §22 gives its steps.

    ```text
    1. broadcast state latent context
    3. optional action-set latent read/mix/broadcast
    4. AxisMix
    5. invariant and shared-axis FFNs
    6. phase FiLM
    ```

    Every stage is a pre-norm residual branch owning its norms and its
    LayerScale. The action-set latents belong to the stack that owns their
    bases, so the block is handed the pass for its own depth rather than
    holding it, as the state trunk does.
    """

    def __init__(self, cfg: MantisACTConfig) -> None:
        super().__init__()
        refuse_unimplemented_paths(cfg)
        self.cfg = cfg
        self.state_context = (
            StateContextBroadcast(cfg) if cfg.num_inv_latents else None
        )
        self.mix = AxisMix(cfg)
        self.ffn = EquivariantFFN(cfg)
        self.film = PhaseFiLM(cfg)

    def forward(
        self,
        actions: EquivariantState,
        action_latents: LatentState,
        *,
        state_latents: LatentState,
        latent_pass: LatentPass,
        latent_segments=None,
        legal_offsets: Tensor,
        positions: Tensor,
        action_phase: Tensor,
    ) -> tuple[EquivariantState, LatentState]:
        """The action state and the action-set latents after this block."""
        if self.state_context is not None:
            actions = self.state_context(
                actions, state_latents, positions, legal_offsets
            )

        action_latents, entities = latent_pass(
            action_latents,
            {"action": RaggedStream(actions, legal_offsets, positions)},
            segments=latent_segments,
        )
        actions = entities["action"].state

        actions = run_equivariant_stage(
            actions,
            self.mix,
            self.ffn,
            film=self.film,
            phase_id=action_phase,
        )
        return actions, action_latents


# --------------------------------------------------------------------------
# The stage


@dataclass(frozen=True, eq=False)
class ActionOutput:
    """What the action encoder answers: one state per legal action.

    ``actions`` is an `EquivariantState` over the batch's flat legal rows, in
    engine order within each position, and ``legal_offsets`` gives each
    position's slice. ``latents`` is the action-set `LatentState` after the
    last block, read by §24's auxiliaries and the diagnostics. Equality is
    identity.
    """

    actions: EquivariantState
    latents: LatentState
    legal_offsets: Tensor
    position_count: int


class ActionEncoder(nn.Module):
    """§19, §21, §22: the trunk's board state to one embedding per legal action.

    ```python
    trunk = StateTrunk(cfg)
    encoder = ActionEncoder(cfg)
    out = encoder(batch, trunk(batch))
    ```

    The forward is linear in the legal action count (§26): the eighteen rows
    are a constant per action, and both latent attentions read over a
    configured constant rather than the action count.

    ``TrunkOutput`` is read and never written. The action-set latents are a
    separate stack from the state latents (§21).
    """

    def __init__(self, cfg: MantisACTConfig) -> None:
        super().__init__()
        refuse_unimplemented_paths(cfg)
        self.cfg = cfg
        self.builder_fingerprint = builder_fingerprint(cfg)

        self.base = ActionBaseState(cfg)
        self.post = (
            PostPlacementEncoder(cfg) if cfg.use_counterfactual_action_windows else None
        )
        self.tactical_width = (
            TACTICAL_FEATURES if cfg.use_action_tactical_features else 0
        )
        self.tactical = (
            TacticalEncoder(cfg, self.tactical_width) if self.tactical_width else None
        )
        self.latents = ActionLatents(cfg)
        self.blocks = nn.ModuleList(ActionBlock(cfg) for _ in range(cfg.action_blocks))
        # A final norm per entity type and stream (§18), matching the trunk;
        # §23's private adapters are the next pre-norm stack.
        self.final = EquivariantNorm(cfg)

    def forward(self, batch: PackedACTBatch, trunk: TrunkOutput) -> ActionOutput:
        """One `EquivariantState` per legal action, in engine order."""
        if batch.plans is None:
            raise ValueError(
                "PackedACTBatch.plans is missing; build batches with "
                "collate(graphs, cfg) so execution plans are made on the CPU"
            )
        if batch.builder_fingerprint != self.builder_fingerprint:
            raise ValueError(
                f"the batch was planned for builder config "
                f"{batch.builder_fingerprint!r}, but this action encoder expects "
                f"{self.builder_fingerprint!r}; rebuild it with "
                "collate(graphs, model.cfg)"
            )
        plans = batch.plans
        position_count = batch.global_numeric.shape[0]
        if trunk.position_count != position_count:
            raise ValueError(
                f"the trunk output describes {trunk.position_count} positions "
                f"against this batch's {position_count}"
            )
        n_legal = batch.legal_to_cell_index.shape[0]
        # ATen refuses an `output_size` that disagrees with the offsets' own
        # total, so this enforces `legal_offsets[-1] == n_legal` on the device.
        positions = plans.legal_row_pos

        actions = self.base(batch, trunk.cells)
        if self.post is not None:
            # The eighteen-row counterfactual block is the fit path's dominant
            # saved-activation site. Non-reentrant checkpointing retains only
            # its inputs and replays the same ordered custom kernels during
            # backward; RNG preservation keeps configured dropout identical.
            actions = checkpoint(
                self.post,
                batch,
                actions,
                trunk.windows,
                use_reentrant=False,
                preserve_rng_state=True,
            )
        if self.tactical is not None:
            actions = self.tactical(actions, batch.action_tactical_numeric)
        elif batch.action_tactical_numeric.shape[1]:
            raise ValueError(
                f"action_tactical_numeric carries "
                f"{batch.action_tactical_numeric.shape[1]} columns against this "
                "configuration's 0: use_action_tactical_features is off, so the "
                "block must be absent rather than ignored"
            )

        action_phase = plans.action_phase
        latents = self.latents.initial(
            position_count,
            device=actions.inv.device,
            dtype=actions.inv.dtype,
        )
        for index, block in enumerate(self.blocks):
            actions, latents = block(
                actions,
                latents,
                state_latents=trunk.latents,
                latent_pass=self.latents[index],
                latent_segments=plans.action_segments,
                legal_offsets=batch.legal_offsets,
                positions=positions,
                action_phase=action_phase,
            )

        return ActionOutput(
            actions=self.final(actions),
            latents=latents,
            legal_offsets=batch.legal_offsets,
            position_count=position_count,
        )


__all__ = [
    "ACTION_ROW_SHAPE",
    "ActionBaseState",
    "ActionBlock",
    "ActionEncoder",
    "ActionOutput",
    "PostPlacementEncoder",
    "StateContextBroadcast",
    "TacticalEncoder",
    "WindowRows",
]
