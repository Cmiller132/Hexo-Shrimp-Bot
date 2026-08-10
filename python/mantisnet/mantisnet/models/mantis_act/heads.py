"""Policy/critic fork (§23) and optional auxiliary heads (§24).

``A_shared`` is forked into policy (one logit) and critic (``[z_pos, z_neg,
z_zero]``).  Three separation modes (§29, §35.10): ``private_adapters``
(default), ``separate_output_mlps``, ``single_shared_head``.

Critic composition is fp32 outside autocast (§27).  ``mass_floor`` is a
forward argument, not in the architecture hash (§23.2).  Auxiliaries are
absent by default; zero weight means absent (§24).  §24.1's overlap rule
refuses auxiliaries that duplicate §19.3 tactical inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn

from .actions import TACTICAL_FEATURE_NAMES
from .config import MantisACTConfig
from .equivariant import (
    AxisMix,
    AxisPool,
    EquivariantFFN,
    EquivariantNorm,
    EquivariantState,
    LayerScale,
    activation_module,
    at_least_fp32,
    run_equivariant_stage,
)
from .latents import LatentState, row_positions
from .ordered_reductions import ordered_row_broadcast, ordered_segment_max
from .packed import PHASE_FIRST
from .pattern_classes import OPP_LIVE, OWN_LIVE

# §23.2's three outcome logits, and the paper-faithful scalar's one. Every
# reader takes the checkpoint's critic width from here.
CATEGORICAL_CRITIC_LOGITS = 3
SCALAR_CRITIC_LOGITS = 1

# §24.1.2: a window holds six cells, so an own post-action maximum occupancy is
# one of 0..6. The action always adds a stone, so 0 never occurs; the class is
# kept so the label is the count itself rather than the count minus one.
OWN_OCCUPANCY_CLASSES = 7

# The cap of §24.1's "capped categorical" counts: 0..3 exactly, and 4 meaning
# four or more. An action lying in four live opponent threats at once, or
# creating four own five-windows at once, has settled the game whatever the
# exact number is, so the tail is one class rather than a long sparse one.
AUX_COUNT_CAP = 4
AUX_COUNT_CLASSES = AUX_COUNT_CAP + 1

# §24.2's fates of a live window, by the end of the game: completed by its own
# colour, killed by the other colour placing in it, or still standing when the
# game ends elsewhere.
WINDOW_FATE_NAMES: tuple[str, ...] = ("completed", "goes_mixed", "unresolved")
WINDOW_FATE_CLASSES = len(WINDOW_FATE_NAMES)


@dataclass(frozen=True)
class AuxSpec:
    """One §24.1 action auxiliary: its width, its mask, and its overlap.

    ``logits`` is 1 for a binary head and the class count for a categorical
    one. ``first_placement_only`` restricts the label to ``phase == FIRST``
    rows, which is §24.1's condition on auxiliaries 5 and 6. ``tactical``
    names the §19.3 input fields the label duplicates, empty when there is no
    overlap.
    """

    name: str
    logits: int
    first_placement_only: bool
    tactical: tuple[str, ...]


# §24.1's six action auxiliaries, in its own order.
ACTION_AUXILIARIES: tuple[AuxSpec, ...] = (
    AuxSpec("win_now", 1, False, ("immediate_win",)),
    AuxSpec("own_max_occupancy", OWN_OCCUPANCY_CLASSES, False, ("max_own_count_after",)),
    AuxSpec(
        "opponent_threats_hit",
        AUX_COUNT_CLASSES,
        False,
        ("opponent_five_windows_hit", "opponent_four_windows_hit"),
    ),
    AuxSpec(
        "own_five_windows_after",
        AUX_COUNT_CLASSES,
        False,
        ("own_five_windows_after",),
    ),
    # Both deterministic functions of the board and a hypothetical placement.
    AuxSpec("winning_partner_exists", 1, True, ()),
    AuxSpec("winning_partner_count", AUX_COUNT_CLASSES, True, ()),
)

AUX_SPECS: dict[str, AuxSpec] = {spec.name: spec for spec in ACTION_AUXILIARIES}

# The overlap table is checked against §19.3's own field list at import: a
# renamed tactical field would otherwise leave an auxiliary claiming an overlap
# with an input that no longer exists, and the mask would quietly stop firing.
_UNKNOWN_TACTICAL = {
    field
    for spec in ACTION_AUXILIARIES
    for field in spec.tactical
    if field not in TACTICAL_FEATURE_NAMES
}
if _UNKNOWN_TACTICAL:
    raise RuntimeError(
        f"auxiliary overlap names {sorted(_UNKNOWN_TACTICAL)}, which are not "
        f"§19.3 tactical fields: {list(TACTICAL_FEATURE_NAMES)}"
    )

WINDOW_FATE_HEAD = "window_fate"

# The suffix under which a head's per-row label mask is returned beside its
# logits, so `aux` stays the flat `dict[str, Tensor]` §25 specifies.
MASK_SUFFIX = ".mask"


def critic_logit_width(cfg: MantisACTConfig) -> int:
    """The trailing width of ``critic_logits`` under ``cfg.critic_type``."""
    if cfg.critic_type == "categorical3":
        return CATEGORICAL_CRITIC_LOGITS
    if cfg.critic_type == "scalar_tanh":
        return SCALAR_CRITIC_LOGITS
    raise ValueError(f"critic_type={cfg.critic_type!r} is not implemented")


# --------------------------------------------------------------------------
# The §23.2 composition. fp32, outside autocast, MantisNet's operator exactly.


def _composed(critic_logits: Tensor) -> Tensor:
    """``(..., 3)`` categorical logits as an fp32 probability simplex.

    Promoted before the softmax, with autocast disabled around it (by the
    tensor's own device type), so the result does not depend on the caller's
    autocast precision (§27).
    """
    if critic_logits.shape[-1] != CATEGORICAL_CRITIC_LOGITS:
        raise ValueError(
            f"the categorical composition needs {CATEGORICAL_CRITIC_LOGITS} "
            f"logits, got a trailing width of {critic_logits.shape[-1]}"
        )
    with torch.autocast(device_type=critic_logits.device.type, enabled=False):
        return at_least_fp32(critic_logits).softmax(dim=-1)


def return_mass(critic_logits: Tensor) -> tuple[Tensor, Tensor]:
    """Decode ``(..., 3)`` categorical logits as positive/negative mass, fp32.

    The softmax rows are ``(p_pos, p_neg, p_zero)``. At the categorical
    cross-entropy optimum the returned pair is ``(E[G⁺], E[G⁻])``; their sum is
    ``E|G|`` and is at most one because the omitted zero mass is the remainder
    of the same simplex.
    """
    p_pos, p_neg, _p_zero = _composed(critic_logits).unbind(dim=-1)
    return p_pos, p_neg


def compose_q(critic_logits: Tensor) -> Tensor:
    """Compose ``(..., 3)`` categorical logits into action values, fp32.

    ``Q = p_pos - p_neg`` is the quantity the λ-return targets and v̂ averages.
    The categorical simplex makes ``Q`` lie in ``(-1, 1)`` and committed mass
    ``p_pos + p_neg`` lie in ``(0, 1)`` by construction.
    """
    p_pos, p_neg = return_mass(critic_logits)
    return p_pos - p_neg


def committed_mass(critic_logits: Tensor) -> Tensor:
    """§23.2's ``M = p_pos + p_neg``, the mass the critic commits, fp32."""
    p_pos, p_neg = return_mass(critic_logits)
    return p_pos + p_neg


def _segment_max(values: Tensor, offsets: Tensor) -> Tensor:
    """Per-position maximum over position-major legal-action runs."""
    return ordered_segment_max(values, offsets)


def compose_acting_q(
    critic_logits: Tensor,
    legal_offsets: Tensor,
    mass_floor: float,
    *,
    row_position: Tensor | None = None,
) -> Tensor:
    """Return Q divided by the position's floored maximum committed mass, fp32.

    ``M = p_pos + p_neg = 1 - p_zero`` is structurally in ``(0, 1)``. One
    positive divisor is shared by every legal cell in a position, so the score
    preserves Q's order while expressing it in units of the most committed
    action. Since ``|Q| <= M <= max M``, an unfloored score lies in ``(-1, 1)``;
    ``mass_floor`` additionally bounds sharpening when all actions put most
    probability on zero return.
    """
    if not 0.0 < mass_floor <= 1.0:
        raise ValueError(
            f"mass_floor={mass_floor} must lie in (0, 1]: it floors a committed "
            "mass, which is structurally in (0, 1)"
        )
    if critic_logits.ndim != 2:
        raise ValueError(
            f"the acting score needs flat (N, 3) logits, got shape "
            f"{tuple(critic_logits.shape)}"
        )
    p_pos, p_neg = return_mass(critic_logits)
    # ATen refuses an `output_size` that disagrees with the offsets' own
    # total, enforcing on the device that the offsets end where the critic's
    # rows do.
    segment = (
        row_positions(legal_offsets, critic_logits.shape[0])
        if row_position is None
        else row_position
    )
    scale = _segment_max(p_pos + p_neg, legal_offsets).clamp(min=mass_floor)
    return (p_pos - p_neg) / ordered_row_broadcast(scale, segment, legal_offsets)


# --------------------------------------------------------------------------
# The readout body (§23.1) and the private adapters (§23)


class InvariantReadout(nn.Module):
    """§23.1's invariant readout: the action state plus a symmetric axis pool.

    A head's output is invariant, so the axis stream may only enter through a
    symmetric pool over the three channels (§12.3, §12.5); `AxisPool` is that
    pool.

    The body's hidden width defaults to ``ffn_mult * d_inv``. Auxiliaries pass
    a narrower one.
    """

    def __init__(self, cfg: MantisACTConfig, *, hidden: int | None = None) -> None:
        super().__init__()
        self.hidden = cfg.ffn_mult * cfg.d_inv if hidden is None else int(hidden)
        if self.hidden < 1:
            raise ValueError(f"readout hidden width must be >= 1, got {self.hidden}")
        self.norm = EquivariantNorm(cfg)
        self.pool = AxisPool(cfg) if cfg.d_axis else None
        width = cfg.d_inv + cfg.d_axis
        self.body = nn.Sequential(
            nn.Linear(width, self.hidden), activation_module(cfg.activation)
        )

    def forward(self, state: EquivariantState) -> Tensor:
        """The ``(..., hidden)`` invariant readout of ``state``."""
        z = self.norm(state)
        if self.pool is None:
            return self.body(z.inv)
        return self.body(torch.cat((z.inv, self.pool(z)), dim=-1))


def _zero_output(in_features: int, out_features: int) -> nn.Linear:
    """A head's final projection, zero-initialised (§23.1, §23.2, §27)."""
    layer = nn.Linear(in_features, out_features)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class LatentContext(nn.Module):
    """The per-position latent summary a private adapter row reads (§23).

    The invariant latents are averaged over their slots and the axis latents
    over theirs, then gathered to every action row of that position. The axis
    average is taken per channel, so channel ``a`` of the summary reaches
    channel ``a`` of the action and nothing else (§12.1).

    A stream the configuration has no latents for contributes no branch and no
    parameters: with ``num_axis_latents == 0`` the action's axis channels take
    no latent context at all.
    """

    def __init__(
        self, cfg: MantisACTConfig, *, has_inv: bool, has_axis: bool
    ) -> None:
        super().__init__()
        if not has_inv and not has_axis:
            raise ValueError(
                "LatentContext with neither stream holds no parameters; the "
                "caller must not build one under global_mode='none'"
            )
        if has_axis and not cfg.d_axis:
            raise ValueError(
                f"axis latent context needs axis channels, but d_axis={cfg.d_axis}"
            )
        self.has_inv = has_inv
        self.has_axis = has_axis
        if has_inv:
            self.norm_inv = nn.LayerNorm(cfg.d_inv)
            self.to_inv = nn.Linear(cfg.d_inv, cfg.d_inv)
            self.scale_inv = LayerScale(cfg.d_inv, cfg.layer_scale_init)
        if has_axis:
            self.norm_axis = nn.LayerNorm(cfg.d_axis)
            self.to_axis = nn.Linear(cfg.d_axis, cfg.d_axis)
            self.scale_axis = LayerScale(cfg.d_axis, cfg.layer_scale_init)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        state: EquivariantState,
        latents: LatentState,
        row_position: Tensor,
        row_offsets: Tensor,
    ) -> EquivariantState:
        inv, axis = state.inv, state.axis
        # A LayerScale gain is fp32, so a residual under bf16 autocast promotes
        # its stream. Only one of the two branches below may exist, so the
        # promotion is carried to the other stream too below, since
        # `EquivariantState` requires the pair to share a dtype.
        if self.has_inv:
            if latents.inv is None:
                raise ValueError(
                    "this adapter reads invariant latents but got LatentState.inv=None"
                )
            context = self.to_inv(self.norm_inv(latents.inv).mean(dim=1))
            inv = inv + self.scale_inv(
                self.drop(
                    ordered_row_broadcast(context, row_position, row_offsets).to(
                        inv.dtype
                    )
                )
            )
        if self.has_axis:
            if latents.axis is None:
                raise ValueError(
                    "this adapter reads axis latents but got LatentState.axis=None"
                )
            if axis is None:
                raise ValueError(
                    "this adapter writes an axis context, but the action state "
                    "has no axis stream"
                )
            context = self.to_axis(self.norm_axis(latents.axis).mean(dim=1))
            axis = axis + self.scale_axis(
                self.drop(
                    ordered_row_broadcast(context, row_position, row_offsets).to(
                        axis.dtype
                    )
                )
            )
        if axis is not None and axis.dtype != inv.dtype:
            common = torch.promote_types(inv.dtype, axis.dtype)
            inv, axis = inv.to(common), axis.to(common)
        return EquivariantState(inv, axis)


class PrivateAdapterBlock(nn.Module):
    """One private equivariant residual block of a §23 adapter.

    Latent context, then §12.4's `AxisMix`, then the two-stream FFN — each a
    pre-norm residual branch owning its own norms and LayerScale, as a §18
    state block's stages do. No message passing here: a private adapter reads
    only what its own head is about to decide from.
    """

    def __init__(
        self, cfg: MantisACTConfig, *, latent_inv: bool, latent_axis: bool
    ) -> None:
        super().__init__()
        self.context = (
            LatentContext(cfg, has_inv=latent_inv, has_axis=latent_axis)
            if latent_inv or latent_axis
            else None
        )
        self.mix = AxisMix(cfg)
        self.ffn = EquivariantFFN(cfg)

    def forward(
        self,
        state: EquivariantState,
        latents: LatentState | None,
        row_position: Tensor,
        row_offsets: Tensor,
    ) -> EquivariantState:
        if self.context is not None:
            if latents is None:
                raise ValueError(
                    "this adapter block reads latents but none were given"
                )
            state = self.context(state, latents, row_position, row_offsets)
        return run_equivariant_stage(state, self.mix, self.ffn)


class PrivateAdapter(nn.Module):
    """§23's ``PolicyPrivateAdapter`` / ``CriticPrivateAdapter``.

    ``blocks`` copies of :class:`PrivateAdapterBlock`, private to one head. The
    two adapters of a model share no parameter, and each is built from the
    configuration's own block count for its head.
    """

    def __init__(
        self,
        cfg: MantisACTConfig,
        blocks: int,
        *,
        latent_inv: bool,
        latent_axis: bool,
    ) -> None:
        super().__init__()
        if blocks < 1:
            raise ValueError(
                f"a private adapter needs at least one block, got {blocks}"
            )
        self.blocks = nn.ModuleList(
            PrivateAdapterBlock(cfg, latent_inv=latent_inv, latent_axis=latent_axis)
            for _ in range(blocks)
        )

    def forward(
        self,
        state: EquivariantState,
        latents: LatentState | None,
        row_position: Tensor,
        row_offsets: Tensor,
    ) -> EquivariantState:
        for block in self.blocks:
            state = block(state, latents, row_position, row_offsets)
        return state


# --------------------------------------------------------------------------
# The two outputs (§23.1, §23.2)


class PolicyHead(nn.Module):
    """§23.1: one raw logit per legal action, in engine legal order.

    The output layer is zero-initialised, so a fresh model's logits are one
    constant over the whole action set and the initial policy is exactly
    uniform within every position.
    """

    def __init__(
        self, cfg: MantisACTConfig, *, readout: InvariantReadout | None = None
    ) -> None:
        super().__init__()
        self.readout = InvariantReadout(cfg) if readout is None else readout
        self.out = _zero_output(self.readout.hidden, 1)

    def forward(self, state: EquivariantState) -> Tensor:
        return self.out(self.readout(state)).squeeze(-1)


class CriticHead(nn.Module):
    """§23.2: the categorical three-class critic, or the scalar alternative.

    ``categorical3`` emits ``[z_pos, z_neg, z_zero]`` and composes them through
    the fp32 operator at the top of this module. ``scalar_tanh`` is the
    paper-faithful head kept for the ablation: one raw scalar whose ``tanh`` is
    the action value. It has no committed-mass decomposition, so mass-based
    acting-score scaling is refused there by name.
    """

    def __init__(
        self, cfg: MantisACTConfig, *, readout: InvariantReadout | None = None
    ) -> None:
        super().__init__()
        self.critic_type = cfg.critic_type
        self.width = critic_logit_width(cfg)
        self.readout = InvariantReadout(cfg) if readout is None else readout
        self.out = _zero_output(self.readout.hidden, self.width)

    def forward(self, state: EquivariantState) -> Tensor:
        return self.out(self.readout(state))

    def compose(
        self,
        critic_logits: Tensor,
        *,
        legal_offsets: Tensor,
        mass_floor: float | None,
        row_position: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """``(q_value, q_score, committed_mass)`` in fp32 (§23.2, §25, §27).

        ``mass_floor=None`` means no acting-score scaling is configured, so
        the score is the value itself.
        """
        if critic_logits.shape[-1] != self.width:
            raise ValueError(
                f"this {self.critic_type} critic emits {self.width} logits, got "
                f"a trailing width of {critic_logits.shape[-1]}"
            )
        if self.critic_type == "scalar_tanh":
            if mass_floor is not None:
                raise ValueError(
                    "mass_floor scales an acting score by the position's "
                    "maximum committed mass, which only the categorical3 critic "
                    "decomposes; a scalar_tanh critic has no p_pos/p_neg to sum. "
                    "Pass mass_floor=None, or set critic_type='categorical3'"
                )
            with torch.autocast(
                device_type=critic_logits.device.type, enabled=False
            ):
                q_value = torch.tanh(at_least_fp32(critic_logits)).squeeze(-1)
            return q_value, q_value, None

        q_value = compose_q(critic_logits)
        mass = committed_mass(critic_logits)
        if mass_floor is None:
            return q_value, q_value, mass
        q_score = compose_acting_q(
            critic_logits,
            legal_offsets,
            mass_floor,
            row_position=row_position,
        )
        return q_value, q_score, mass


class StateValueHead(nn.Module):
    """§23.3: an explicit auxiliary state value over the state latents.

    Not instantiated by default. Its parameters are reported separately since
    it is its own submodule.

    The axis latents reach the value through `AxisPool`, paired with the mean
    of the invariant latents the way `latents.LatentPass.mix` pairs them,
    since an axis latent has no invariant half of its own and the pool's
    score needs one. The output layer is zero-initialised.
    """

    def __init__(self, cfg: MantisACTConfig) -> None:
        super().__init__()
        if cfg.num_inv_latents < 1:
            raise ValueError(
                "enable_state_value_head=True needs invariant state latents to "
                f"read, but num_inv_latents={cfg.num_inv_latents}"
            )
        self.num_axis = cfg.num_axis_latents
        self.norm_inv = nn.LayerNorm(cfg.d_inv)
        self.pool = AxisPool(cfg) if self.num_axis else None
        width = cfg.d_inv + (cfg.d_axis if self.pool is not None else 0)
        hidden = cfg.ffn_mult * cfg.d_inv
        self.body = nn.Sequential(
            nn.Linear(width, hidden), activation_module(cfg.activation)
        )
        self.out = _zero_output(hidden, 1)

    def forward(self, latents: LatentState) -> Tensor:
        """The ``(P,)`` state value in ``[-1, 1]``, fp32."""
        if latents.inv is None:
            raise ValueError("the state-value head got LatentState.inv=None")
        pooled = self.norm_inv(latents.inv).mean(dim=1)
        if self.pool is not None:
            if latents.axis is None:
                raise ValueError(
                    "this state-value head pools axis latents but got "
                    "LatentState.axis=None"
                )
            paired = EquivariantState(
                latents.inv.mean(dim=1, keepdim=True).expand(-1, self.num_axis, -1),
                latents.axis,
            )
            pooled = torch.cat((pooled, self.pool(paired).mean(dim=1)), dim=-1)
        raw = self.out(self.body(pooled))
        with torch.autocast(device_type=raw.device.type, enabled=False):
            return torch.tanh(at_least_fp32(raw)).squeeze(-1)


# --------------------------------------------------------------------------
# The optional auxiliaries (§24)


def masked_auxiliaries(cfg: MantisACTConfig) -> dict[str, tuple[str, ...]]:
    """The §24.1 auxiliaries this configuration refuses, and why.

    Empty when ``use_action_tactical_features`` is off (§29's
    ``full_no_tactical_inputs``), under which every auxiliary is available.
    """
    if not cfg.use_action_tactical_features:
        return {}
    return {
        spec.name: spec.tactical for spec in ACTION_AUXILIARIES if spec.tactical
    }


class ActionAuxiliaryHeads(nn.Module):
    """§24.1's training-only heads over the shared action state.

    One head per strictly positive weight, each with its own readout body, so a
    head that is not asked for holds no parameters and a head that is removed
    takes all of its own with it. The weights are kept beside the heads so the
    loss reads the number the head was built from.

    The auxiliaries read ``A_shared``: they are neither the policy's question
    nor the critic's, and reading a private adapter would make an ablation of
    one head move the other's auxiliary loss.
    """

    def __init__(
        self,
        cfg: MantisACTConfig,
        weights: Mapping[str, float],
        *,
        instantiate_zero_weight: bool = False,
    ) -> None:
        super().__init__()
        unknown = [name for name in weights if name not in AUX_SPECS]
        if unknown:
            raise ValueError(
                f"unknown action auxiliaries {sorted(unknown)}; §24.1 names "
                f"{list(AUX_SPECS)}"
            )
        negative = {name: w for name, w in weights.items() if w < 0.0}
        if negative:
            raise ValueError(f"auxiliary weights must not be negative: {negative}")

        masked = masked_auxiliaries(cfg)
        for name, weight in weights.items():
            if name not in masked:
                continue
            if weight == 0.0 and not instantiate_zero_weight:
                continue
            fields = ", ".join(masked[name])
            raise ValueError(
                f"auxiliary {name!r} predicts a label the model is already given "
                f"as the §19.3 input(s) {fields}, so its accuracy is a decode of "
                "its own input rather than learned representation (§24.1). "
                "Either drop this auxiliary, or run the learned-only input "
                "ablation: use_action_tactical_features=False, which is the "
                "full_no_tactical_inputs preset"
            )

        self.weights = {
            name: float(weight)
            for name, weight in weights.items()
            if weight > 0.0 or instantiate_zero_weight
        }
        hidden = cfg.d_inv
        self.readouts = nn.ModuleDict(
            {name: InvariantReadout(cfg, hidden=hidden) for name in self.weights}
        )
        self.outputs = nn.ModuleDict(
            {
                name: _zero_output(hidden, AUX_SPECS[name].logits)
                for name in self.weights
            }
        )
        self.names = tuple(name for name in AUX_SPECS if name in self.weights)
        self.needs_phase = any(
            AUX_SPECS[name].first_placement_only for name in self.names
        )

    def forward(
        self,
        state: EquivariantState,
        row_position: Tensor,
        phase_id: Tensor | None,
    ) -> dict[str, Tensor]:
        """Each head's logits, and the mask of the rows carrying a label."""
        if self.needs_phase and phase_id is None:
            first = [
                name for name in self.names if AUX_SPECS[name].first_placement_only
            ]
            raise ValueError(
                f"auxiliaries {first} are labelled on first-placement states "
                "only, so they need phase_id"
            )
        out: dict[str, Tensor] = {}
        first_mask = None
        if self.needs_phase:
            first_mask = (phase_id == PHASE_FIRST).index_select(0, row_position)
        for name in self.names:
            spec = AUX_SPECS[name]
            logits = self.outputs[name](self.readouts[name](state))
            if spec.logits == 1:
                logits = logits.squeeze(-1)
            out[name] = logits
            out[name + MASK_SUFFIX] = (
                first_mask
                if spec.first_placement_only
                else torch.ones_like(row_position, dtype=torch.bool)
            )
        return out


class WindowFateHead(nn.Module):
    """§24.2: the future-fate experiment, over live windows only.

    Mixed and empty windows are masked: a mixed window is already dead for
    both players, and an empty window has no colour whose fate the classes
    describe. The mask is derived from the batch's own ``window_status``.
    """

    def __init__(self, cfg: MantisACTConfig, weight: float) -> None:
        super().__init__()
        if weight <= 0.0:
            raise ValueError(
                f"window-fate weight {weight} builds no head; a zero weight "
                "means the head is absent (§24)"
            )
        self.weight = float(weight)
        hidden = cfg.d_inv
        self.readout = InvariantReadout(cfg, hidden=hidden)
        self.out = _zero_output(hidden, WINDOW_FATE_CLASSES)

    def forward(
        self, windows: EquivariantState, window_status: Tensor
    ) -> dict[str, Tensor]:
        if window_status.shape[0] != windows.inv.shape[0]:
            raise ValueError(
                f"window_status carries {window_status.shape[0]} rows against "
                f"{windows.inv.shape[0]} window states"
            )
        live = (window_status == OWN_LIVE) | (window_status == OPP_LIVE)
        return {
            WINDOW_FATE_HEAD: self.out(self.readout(windows)),
            WINDOW_FATE_HEAD + MASK_SUFFIX: live,
        }


# --------------------------------------------------------------------------
# The fork (§23)


@dataclass(frozen=True, eq=False)
class HeadOutput:
    """What §23 and §24 answer for one batch.

    ``policy_logits`` and ``critic_logits`` are flat over every legal action of
    every position, in engine legal order; ``q_value`` and ``q_score`` are
    their fp32 composition (§25). ``committed_mass`` is §23.2's ``M``, absent
    under ``scalar_tanh``. ``state_value`` is §23.3's, absent unless enabled.
    ``aux`` holds each optional head's logits and its ``.mask`` of labelled
    rows. Equality is identity.
    """

    policy_logits: Tensor
    critic_logits: Tensor
    q_value: Tensor
    q_score: Tensor
    committed_mass: Tensor | None
    state_value: Tensor | None
    aux: dict[str, Tensor]


class ActionHeads(nn.Module):
    """§23's fork, §23.3's state value, and §24's auxiliaries.

    ```python
    heads = ActionHeads(cfg)
    out = heads(
        actions,                       # A_shared over legal rows
        legal_offsets=batch.legal_offsets,
        latents=trunk_out.latents,     # the state latents (§23, §23.3)
        mass_floor=klent_cfg.mass_floor,
    )
    ```

    ``mass_floor`` is required rather than defaulted: it decides the quantity
    π′ ranks by, and a caller that has not said which it wants has not decided
    (§23.2). ``None`` is the explicit "no scaling".
    """

    def __init__(
        self,
        cfg: MantisACTConfig,
        *,
        aux_weights: Mapping[str, float] | None = None,
        window_fate_weight: float = 0.0,
        instantiate_zero_weight: bool = False,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        latent_inv = cfg.num_inv_latents > 0
        latent_axis = cfg.num_axis_latents > 0

        mode = cfg.head_separation
        if mode == "single_shared_head":
            # One body, two output layers. The projections cannot be shared too:
            # they have different widths, and §35.10 measures the shared body.
            shared = InvariantReadout(cfg)
            self.policy = PolicyHead(cfg, readout=shared)
            self.critic = CriticHead(cfg, readout=shared)
            self.policy_adapter = None
            self.critic_adapter = None
        elif mode == "separate_output_mlps":
            self.policy = PolicyHead(cfg)
            self.critic = CriticHead(cfg)
            self.policy_adapter = None
            self.critic_adapter = None
        elif mode == "private_adapters":
            if cfg.policy_private_blocks < 1 or cfg.critic_private_blocks < 1:
                raise ValueError(
                    f"head_separation='private_adapters' with "
                    f"policy_private_blocks={cfg.policy_private_blocks} and "
                    f"critic_private_blocks={cfg.critic_private_blocks}: a head "
                    "with no private block is head_separation="
                    "'separate_output_mlps', which names that model already"
                )
            self.policy_adapter = PrivateAdapter(
                cfg,
                cfg.policy_private_blocks,
                latent_inv=latent_inv,
                latent_axis=latent_axis,
            )
            self.critic_adapter = PrivateAdapter(
                cfg,
                cfg.critic_private_blocks,
                latent_inv=latent_inv,
                latent_axis=latent_axis,
            )
            self.policy = PolicyHead(cfg)
            self.critic = CriticHead(cfg)
        else:
            raise ValueError(f"head_separation={mode!r} is not implemented")

        self.reads_latents = self.policy_adapter is not None and (
            latent_inv or latent_axis
        )

        self.state_value = StateValueHead(cfg) if cfg.enable_state_value_head else None

        weights = dict(aux_weights or {})
        if not cfg.enable_action_aux_heads:
            asked = sorted(name for name, w in weights.items() if w != 0.0)
            if asked:
                raise ValueError(
                    f"enable_action_aux_heads=False but weights name {asked}; "
                    "§24 heads are absent unless the configuration enables them"
                )
            weights = {}
        self.auxiliaries = (
            ActionAuxiliaryHeads(
                cfg, weights, instantiate_zero_weight=instantiate_zero_weight
            )
            if cfg.enable_action_aux_heads
            else None
        )
        if self.auxiliaries is not None and not self.auxiliaries.names:
            raise ValueError(
                "enable_action_aux_heads=True selects no head: every weight in "
                f"{dict(weights)} is zero, and a zero weight means the head is "
                "absent (§24)"
            )

        if cfg.enable_window_fate_head:
            if window_fate_weight <= 0.0:
                raise ValueError(
                    "enable_window_fate_head=True with window_fate_weight="
                    f"{window_fate_weight}: a zero weight means the head is "
                    "absent (§24)"
                )
            self.window_fate = WindowFateHead(cfg, window_fate_weight)
        else:
            if window_fate_weight != 0.0:
                raise ValueError(
                    f"window_fate_weight={window_fate_weight} but "
                    "enable_window_fate_head=False"
                )
            self.window_fate = None

    def _check_state(self, name: str, state: EquivariantState) -> None:
        cfg = self.cfg
        if state.d_inv != cfg.d_inv:
            raise ValueError(
                f"{name} carries d_inv={state.d_inv} against this "
                f"configuration's {cfg.d_inv}"
            )
        if state.d_axis != cfg.d_axis:
            raise ValueError(
                f"{name} carries d_axis={state.d_axis} against this "
                f"configuration's {cfg.d_axis}"
            )

    def logits(
        self,
        actions: EquivariantState,
        *,
        legal_offsets: Tensor,
        latents: LatentState | None = None,
        row_position: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """The raw ``(policy_logits, critic_logits)`` of §23, uncomposed.

        This is the pair KLENT's fitter scores and its evaluator composes
        outside autocast; :meth:`forward` is the same computation plus §25's
        composed outputs and the optional heads.
        """
        self._check_state("the action state", actions)
        if self.reads_latents and latents is None:
            raise ValueError(
                "the private adapters read the state latents, but none were given"
            )
        if row_position is None:
            row_position = row_positions(legal_offsets, actions.inv.shape[0])
        return self._fork(actions, row_position, legal_offsets, latents)

    def _fork(
        self,
        actions: EquivariantState,
        row_position: Tensor,
        row_offsets: Tensor,
        latents: LatentState | None,
    ) -> tuple[Tensor, Tensor]:
        """The two logit families, off one already-validated action state."""
        policy_state = actions
        critic_state = actions
        if self.policy_adapter is not None:
            policy_state = self.policy_adapter(
                actions, latents, row_position, row_offsets
            )
            critic_state = self.critic_adapter(
                actions, latents, row_position, row_offsets
            )
        return self.policy(policy_state), self.critic(critic_state)

    def forward(
        self,
        actions: EquivariantState,
        *,
        legal_offsets: Tensor,
        mass_floor: float | None,
        latents: LatentState | None = None,
        phase_id: Tensor | None = None,
        windows: EquivariantState | None = None,
        window_status: Tensor | None = None,
        row_position: Tensor | None = None,
    ) -> HeadOutput:
        """Every head this configuration holds, for one packed batch."""
        self._check_state("the action state", actions)
        if self.reads_latents and latents is None:
            raise ValueError(
                "the private adapters read the state latents, but none were given"
            )
        if row_position is None:
            row_position = row_positions(legal_offsets, actions.inv.shape[0])
        policy_logits, critic_logits = self._fork(
            actions, row_position, legal_offsets, latents
        )
        q_value, q_score, mass = self.critic.compose(
            critic_logits,
            legal_offsets=legal_offsets,
            mass_floor=mass_floor,
            row_position=row_position,
        )

        aux: dict[str, Tensor] = {}
        if self.auxiliaries is not None:
            aux.update(self.auxiliaries(actions, row_position, phase_id))
        if self.window_fate is not None:
            if windows is None or window_status is None:
                raise ValueError(
                    "the window-fate head needs the trunk's window states and "
                    "the batch's window_status"
                )
            self._check_state("the window state", windows)
            aux.update(self.window_fate(windows, window_status))

        state_value = None
        if self.state_value is not None:
            if latents is None:
                raise ValueError(
                    "the state-value head reads the state latents, but none "
                    "were given"
                )
            state_value = self.state_value(latents)

        return HeadOutput(
            policy_logits=policy_logits,
            critic_logits=critic_logits,
            q_value=q_value,
            q_score=q_score,
            committed_mass=mass,
            state_value=state_value,
            aux=aux,
        )


__all__ = [
    "ACTION_AUXILIARIES",
    "AUX_COUNT_CAP",
    "AUX_COUNT_CLASSES",
    "AUX_SPECS",
    "CATEGORICAL_CRITIC_LOGITS",
    "MASK_SUFFIX",
    "OWN_OCCUPANCY_CLASSES",
    "SCALAR_CRITIC_LOGITS",
    "WINDOW_FATE_CLASSES",
    "WINDOW_FATE_HEAD",
    "WINDOW_FATE_NAMES",
    "ActionAuxiliaryHeads",
    "ActionHeads",
    "InvariantReadout",
    "AuxSpec",
    "CriticHead",
    "HeadOutput",
    "LatentContext",
    "PolicyHead",
    "PrivateAdapter",
    "PrivateAdapterBlock",
    "StateValueHead",
    "WindowFateHead",
    "committed_mass",
    "compose_acting_q",
    "compose_q",
    "critic_logit_width",
    "masked_auxiliaries",
    "return_mass",
]
