"""Equivariant state container and primitives (§12, §13.2).

Every entity carries ``h_inv [..., d_inv]`` and ``h_axis [..., 3, d_axis]``.
Under board transform ``g`` with axis permutation ``pi_g``:
``h_inv'(T_g(i)) = h_inv(i)``, ``h_axis'(T_g(i), pi_g(a)) = h_axis(i, a)``.

Axis-stream parameters act on the trailing width only; cross-channel
quantities (``sum_b u_b``, ``mean_a phi(u_a)``) are symmetric and invariant.
Numerics follow §27 (fp32 params, bf16 autocast, fp32 softmax).

``d_axis == 0`` is the ``full_no_axis`` arm of §29: ``axis=None`` throughout,
no axis parameters or norms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import MantisACTConfig
from .packed import PHASE_FIRST, PHASE_OPENING, PHASE_SECOND

# The board has three undirected axes and every axis stream carries one channel
# per axis. Nothing here indexes a channel by an absolute axis id; this is a
# shape, not a vocabulary.
AXIS_CHANNELS = 3

# The three-way placement phase of §13.1, in its builder-assigned id order.
# The embedding table is indexed by that id directly, so a renumbering that
# left a gap or an offset would train the wrong row rather than fail.
PHASE_IDS = (PHASE_OPENING, PHASE_FIRST, PHASE_SECOND)
if PHASE_IDS != tuple(range(len(PHASE_IDS))):
    raise RuntimeError(f"phase ids must be a dense 0-based range, got {PHASE_IDS}")

# `use_three_way_phase=False` keeps the same authoritative input id and folds
# OPENING into SECOND inside the model: both are a turn's last placement, and
# the distinction between them is the board's emptiness rather than anything
# `moves_remaining` — the quantity the KLENT return sign reads — can see.
_TWO_WAY_ROWS = (0, 1, 0)

_ACTIVATIONS = {"silu": nn.SiLU, "gelu": nn.GELU, "relu": nn.ReLU}

# §27: embeddings and learned bases.
_EMBEDDING_STD = 0.02


def activation_module(name: str) -> nn.Module:
    """The activation named by ``cfg.activation`` (§6), or a named refusal."""
    factory = _ACTIVATIONS.get(name)
    if factory is None:
        raise ValueError(
            f"activation={name!r} is not one of {sorted(_ACTIVATIONS)}"
        )
    return factory()


def at_least_fp32(tensor: Tensor) -> Tensor:
    """Promote a reduced-precision tensor to fp32, leaving fp32/fp64 alone.

    §27 requires every softmax and every segment reduction in fp32. A literal
    ``.float()`` would demote an fp64 reference tensor; this promotes only
    where needed. The package's one promotion helper, shared with `messages`
    and `latents`.
    """
    return tensor.to(torch.promote_types(tensor.dtype, torch.float32))


def _checked_permutation(permutation: Sequence[int]) -> tuple[int, ...]:
    """Validate an axis-channel permutation in the §12.1 direction."""
    values = tuple(int(a) for a in permutation)
    if sorted(values) != list(range(AXIS_CHANNELS)):
        raise ValueError(
            f"axis permutation must be a permutation of "
            f"{list(range(AXIS_CHANNELS))}, got {values}"
        )
    return values


def permute_axis_channels(axis: Tensor, permutation: Sequence[int]) -> Tensor:
    """Carry a ``(..., 3, d_axis)`` axis stream along ``permutation``.

    ``permutation[a]`` is the channel that source channel ``a`` is carried
    *to*, which is the convention of :func:`mantis_act.symmetry.axis_permutation`
    and of §12.1: the result ``y`` satisfies
    ``y[..., permutation[a], :] == axis[..., a, :]``.
    """
    values = _checked_permutation(permutation)
    if axis.ndim < 2 or axis.shape[-2] != AXIS_CHANNELS:
        raise ValueError(
            f"axis stream must be (..., {AXIS_CHANNELS}, d_axis), got shape "
            f"{tuple(axis.shape)}"
        )
    # `values` maps source to destination; index_select needs the destination's
    # source, so invert it.
    source = [0] * AXIS_CHANNELS
    for a, image in enumerate(values):
        source[image] = a
    index = torch.tensor(source, dtype=torch.long, device=axis.device)
    return axis.index_select(-2, index)


@dataclass(frozen=True, eq=False)
class EquivariantState:
    """One entity family's ``(h_inv, h_axis)`` pair (§12.1).

    ``inv`` is ``(..., d_inv)`` and ``axis`` is ``(..., 3, d_axis)`` over the
    same leading shape, dtype, and device; every agreement is checked here.

    ``axis=None`` is the ``full_no_axis`` model of §29: the axis half is
    genuinely absent. A zero-width axis tensor is refused.

    Instances are immutable; ``dataclasses.replace`` re-runs every check.
    Equality is identity.
    """

    inv: Tensor
    axis: Tensor | None = None

    def __post_init__(self) -> None:
        inv = self.inv
        if not isinstance(inv, Tensor):
            raise TypeError(f"inv must be a Tensor, got {type(inv).__name__}")
        if inv.ndim < 1 or inv.shape[-1] < 1:
            raise ValueError(
                f"inv must be (..., d_inv) with d_inv >= 1, got shape "
                f"{tuple(inv.shape)}"
            )
        if not inv.is_floating_point():
            raise ValueError(f"inv must be floating point, got dtype {inv.dtype}")

        axis = self.axis
        if axis is None:
            return
        if not isinstance(axis, Tensor):
            raise TypeError(f"axis must be a Tensor or None, got {type(axis).__name__}")
        if axis.ndim != inv.ndim + 1:
            raise ValueError(
                f"axis must be (..., {AXIS_CHANNELS}, d_axis) beside an "
                f"(..., d_inv) inv: inv has shape {tuple(inv.shape)}, axis has "
                f"shape {tuple(axis.shape)}"
            )
        if axis.shape[-2] != AXIS_CHANNELS:
            raise ValueError(
                f"axis must carry {AXIS_CHANNELS} channels, got "
                f"{axis.shape[-2]} in shape {tuple(axis.shape)}"
            )
        if axis.shape[-1] < 1:
            raise ValueError(
                "a zero-width axis half must be absent (axis=None), not a "
                f"zero-width tensor of shape {tuple(axis.shape)}: §29 "
                "full_no_axis retains no unused axis parameters"
            )
        if tuple(axis.shape[:-2]) != tuple(inv.shape[:-1]):
            raise ValueError(
                f"inv and axis must share their leading shape: inv is "
                f"{tuple(inv.shape[:-1])}, axis is {tuple(axis.shape[:-2])}"
            )
        if axis.dtype != inv.dtype:
            raise ValueError(
                f"inv and axis must share a dtype, got {inv.dtype} and {axis.dtype}"
            )
        if axis.device != inv.device:
            raise ValueError(
                f"inv and axis must share a device, got {inv.device} and {axis.device}"
            )

    @property
    def d_inv(self) -> int:
        return int(self.inv.shape[-1])

    @property
    def d_axis(self) -> int:
        """The axis width, ``0`` when the axis half is absent."""
        return 0 if self.axis is None else int(self.axis.shape[-1])

    @property
    def has_axis(self) -> bool:
        return self.axis is not None

    @property
    def leading_shape(self) -> tuple[int, ...]:
        """The entity shape both streams share, without either width."""
        return tuple(self.inv.shape[:-1])

    @property
    def dtype(self) -> torch.dtype:
        return self.inv.dtype

    @property
    def device(self) -> torch.device:
        return self.inv.device

    def require_axis(self, who: str) -> Tensor:
        """The axis stream, or a refusal naming the caller that needed it."""
        if self.axis is None:
            raise ValueError(f"{who} needs an axis stream, but this state has none")
        return self.axis

    def permute_axes(self, permutation: Sequence[int]) -> EquivariantState:
        """The state with its axis channels carried along ``permutation`` (§12.1).

        The invariant stream is untouched: that is the law's first line.
        """
        values = _checked_permutation(permutation)
        if self.axis is None:
            return self
        return EquivariantState(self.inv, permute_axis_channels(self.axis, values))

    def to(self, *args, **kwargs) -> EquivariantState:
        """Both streams moved or cast together."""
        return EquivariantState(
            self.inv.to(*args, **kwargs),
            None if self.axis is None else self.axis.to(*args, **kwargs),
        )


class LayerScale(nn.Module):
    """A learned per-width residual gain, initialised ``layer_scale_init`` (§27).

    On an axis stream the gain has width ``d_axis`` and broadcasts over the
    channel dimension, so all three channels are scaled by the same vector: a
    per-channel gain would be a per-absolute-axis parameter (§12.2).
    """

    def __init__(self, width: int, init: float) -> None:
        super().__init__()
        if width < 1:
            raise ValueError(f"LayerScale width must be at least 1, got {width}")
        self.gamma = nn.Parameter(torch.full((width,), float(init)))

    def forward(self, delta: Tensor) -> Tensor:
        if delta.shape[-1] != self.gamma.shape[0]:
            raise ValueError(
                f"LayerScale of width {self.gamma.shape[0]} received a delta of "
                f"width {delta.shape[-1]}"
            )
        return self.gamma * delta


class EquivariantNorm(nn.Module):
    """The LayerNorm pair for one entity type's two streams (§18, §27).

    One instance belongs to one stream pair of one entity family; the trunk
    holds as many of these as it has (entity, site) combinations.

    The axis norm holds a single set of ``d_axis`` parameters applied
    independently to all three channels — three sets, one per channel, would
    be §12.2's forbidden per-absolute-axis norm.
    """

    def __init__(self, cfg: MantisACTConfig) -> None:
        super().__init__()
        if cfg.norm != "layernorm":
            raise ValueError(f"norm={cfg.norm!r} is not implemented")
        self.inv = nn.LayerNorm(cfg.d_inv)
        self.axis = nn.LayerNorm(cfg.d_axis) if cfg.d_axis else None

    def forward(self, state: EquivariantState) -> EquivariantState:
        expected_inv = int(self.inv.normalized_shape[0])
        if state.d_inv != expected_inv:
            raise ValueError(
                f"norm expects d_inv={expected_inv}, got a state of width "
                f"{state.d_inv}"
            )
        if state.has_axis != (self.axis is not None):
            built = "with" if self.axis is not None else "without"
            given = "one" if state.has_axis else "none"
            raise ValueError(
                f"norm was built {built} an axis stream, but the state has {given}"
            )
        if self.axis is None:
            return EquivariantState(self.inv(state.inv))
        expected_axis = int(self.axis.normalized_shape[0])
        if state.d_axis != expected_axis:
            raise ValueError(
                f"norm expects d_axis={expected_axis}, got a state of width "
                f"{state.d_axis}"
            )
        return EquivariantState(self.inv(state.inv), self.axis(state.axis))


class EquivariantResidual(nn.Module):
    """``state + layer_scale * delta`` with one gain per stream (§27).

    The pre-norm half of a block is :class:`EquivariantNorm`; this is the other
    half, so a block reads ``residual(state, *branch(norm(state)))``.
    """

    def __init__(self, cfg: MantisACTConfig) -> None:
        super().__init__()
        self.inv = LayerScale(cfg.d_inv, cfg.layer_scale_init)
        self.axis = (
            LayerScale(cfg.d_axis, cfg.layer_scale_init) if cfg.d_axis else None
        )

    def forward(
        self,
        state: EquivariantState,
        delta_inv: Tensor,
        delta_axis: Tensor | None = None,
    ) -> EquivariantState:
        if delta_inv.shape != state.inv.shape:
            raise ValueError(
                f"invariant delta of shape {tuple(delta_inv.shape)} does not "
                f"match the state's {tuple(state.inv.shape)}"
            )
        if (delta_axis is None) != (state.axis is None):
            raise ValueError(
                "an axis delta is required exactly when the state has an axis "
                f"stream: state has_axis={state.has_axis}, delta_axis "
                f"{'given' if delta_axis is not None else 'missing'}"
            )
        inv = state.inv + self.inv(delta_inv)
        if delta_axis is None:
            return EquivariantState(inv)
        if self.axis is None:
            raise ValueError(
                "this residual was built without an axis stream but received "
                "an axis delta"
            )
        if delta_axis.shape != state.axis.shape:
            raise ValueError(
                f"axis delta of shape {tuple(delta_axis.shape)} does not match "
                f"the state's {tuple(state.axis.shape)}"
            )
        return EquivariantState(inv, state.axis + self.axis(delta_axis))


class EquivariantFFN(nn.Module):
    """The pre-norm residual FFN of a block, over both streams (§18).

    Two independent feed-forward networks, one per stream, each with its own
    norm and its own LayerScale. The axis network is applied to the trailing
    width of a ``(..., 3, d_axis)`` tensor, which is one shared MLP evaluated
    independently on all three channels (§12.3) — the channels never meet here,
    so this stage is equivariant for the same reason a pointwise function is.
    Cross-channel communication is :class:`AxisMix`'s job alone.
    """

    def __init__(self, cfg: MantisACTConfig) -> None:
        super().__init__()
        self.activation = cfg.activation
        self.norm = EquivariantNorm(cfg)
        self.inv = nn.Sequential(
            nn.Linear(cfg.d_inv, cfg.ffn_mult * cfg.d_inv),
            activation_module(cfg.activation),
            nn.Linear(cfg.ffn_mult * cfg.d_inv, cfg.d_inv),
        )
        self.axis = (
            nn.Sequential(
                nn.Linear(cfg.d_axis, cfg.ffn_mult * cfg.d_axis),
                activation_module(cfg.activation),
                nn.Linear(cfg.ffn_mult * cfg.d_axis, cfg.d_axis),
            )
            if cfg.d_axis
            else None
        )
        self.residual = EquivariantResidual(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, state: EquivariantState) -> EquivariantState:
        z = self.norm(state)
        delta_axis = None if self.axis is None else self.drop(self.axis(z.axis))
        return self.residual(state, self.drop(self.inv(z.inv)), delta_axis)


class AxisMix(nn.Module):
    """§12.4: the one place the three axis channels talk to each other.

    ```text
    u_a          = LN_axis(x_a)
    total        = sum_b u_b
    other_a      = (total - u_a) / 2
    delta_a      = MLP_axis([u_a, other_a, W_inv_to_axis(LN_inv(h_inv))])
    x_a         += layer_scale_axis * delta_a
    axis_summary = sum_a phi_axis(u_a) / 3
    h_inv       += layer_scale_inv * MLP_inv([LN_inv(h_inv), axis_summary])
    ```

    Both residual branches read the pre-update state, so the two are computed
    from one set of norms.

    Equivariant because: ``LN_axis`` is one shared map, so ``u`` permutes with
    a channel permutation; ``total`` is a sum over the whole channel set and
    is unchanged; ``other_a`` is built from ``total`` and ``u_a`` and so
    permutes; ``MLP_axis`` is one shared map, so ``delta`` and the axis update
    permute; ``axis_summary`` is a mean over the whole channel set and is
    unchanged, so the invariant update is unchanged (§12.1).

    The concatenation is of ``[self, others, invariant]``, never of channels
    ``0/1/2`` in a fixed order (§12.2's forbidden per-absolute-axis parameter).

    Under ``d_axis == 0`` (§29 ``full_no_axis``) the module holds no
    parameters; §18's "AxisMix + FFN" stage degenerates to the FFN alone.
    """

    def __init__(self, cfg: MantisACTConfig) -> None:
        super().__init__()
        self.activation = cfg.activation
        self.d_axis = cfg.d_axis
        if not cfg.d_axis:
            return
        self.norm = EquivariantNorm(cfg)
        # The invariant stream's one route into the axis stream, broadcast
        # identically to all three channels (§12.3).
        self.inv_to_axis = nn.Linear(cfg.d_inv, cfg.d_axis)
        hidden_axis = cfg.ffn_mult * cfg.d_axis
        self.mlp_axis = nn.Sequential(
            # Three blocks of d_axis — [u_a, other_a, invariant context] — and
            # not one block per channel. This MLP sees one channel at a time.
            nn.Linear(3 * cfg.d_axis, hidden_axis),
            activation_module(cfg.activation),
            nn.Linear(hidden_axis, cfg.d_axis),
        )
        # phi_axis is nonlinear on purpose: a linear map would commute with the
        # mean and make the summary a function of `sum_a u_a` alone, which the
        # axis branch already has as `total`.
        self.phi_axis = nn.Sequential(
            nn.Linear(cfg.d_axis, cfg.d_axis),
            activation_module(cfg.activation),
        )
        hidden_inv = cfg.ffn_mult * cfg.d_inv
        self.mlp_inv = nn.Sequential(
            nn.Linear(cfg.d_inv + cfg.d_axis, hidden_inv),
            activation_module(cfg.activation),
            nn.Linear(hidden_inv, cfg.d_inv),
        )
        self.residual = EquivariantResidual(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, state: EquivariantState) -> EquivariantState:
        if state.d_axis != self.d_axis:
            raise ValueError(
                f"AxisMix was built for d_axis={self.d_axis}, got a state of "
                f"width {state.d_axis}"
            )
        if not self.d_axis:
            return state

        z = self.norm(state)
        u = z.axis
        total = u.sum(dim=-2, keepdim=True)
        other = (total - u) / (AXIS_CHANNELS - 1)
        context = self.inv_to_axis(z.inv).unsqueeze(-2).expand(u.shape)
        delta_axis = self.drop(self.mlp_axis(torch.cat((u, other, context), dim=-1)))

        axis_summary = self.phi_axis(u).mean(dim=-2)
        delta_inv = self.drop(self.mlp_inv(torch.cat((z.inv, axis_summary), dim=-1)))
        return self.residual(state, delta_inv, delta_axis)


class AxisPool(nn.Module):
    """§12.5: the axis stream collapsed to one invariant ``d_axis`` vector.

    ``"mean"`` is the arithmetic mean over channels. ``"learned_attention"`` is

    ```text
    score_a  = w.T @ tanh(Wa @ x_a + Wi @ h_inv)
    weight_a = softmax(score over axes)
    pool     = sum_a weight_a * x_a
    ```

    The learned pool is invariant because the scores and the channels permute
    together: ``Wa``, ``w``, and the broadcast invariant term are shared over
    ``a``, so a channel permutation carries ``score_a`` to ``score_pi(a)``
    exactly as it carries ``x_a`` to ``x_pi(a)``, and ``sum_a weight_a * x_a``
    pairs each weight with the channel it was computed from regardless of
    order.

    The softmax runs at no less than fp32 (§27). This is the pooling every
    invariant consumer of an axis stream uses — the head pool of §12.5, and
    the symmetric pool §17.2/§17.3 need for invariant/axis communication.

    Refuses to be built without an axis stream (§29 ``full_no_axis``).
    """

    def __init__(self, cfg: MantisACTConfig, hidden: int | None = None) -> None:
        super().__init__()
        if not cfg.d_axis:
            raise ValueError(
                "AxisPool has nothing to pool with d_axis=0; the caller must "
                "not build one under the full_no_axis arm (§29)"
            )
        if cfg.axis_pool_mode not in ("mean", "learned_attention"):
            raise ValueError(
                f"axis_pool_mode={cfg.axis_pool_mode!r} is not one of "
                "['learned_attention', 'mean']"
            )
        self.mode = cfg.axis_pool_mode
        self.d_axis = cfg.d_axis
        self.out_features = cfg.d_axis
        if self.mode == "learned_attention":
            width = cfg.d_axis if hidden is None else int(hidden)
            if width < 1:
                raise ValueError(f"AxisPool hidden width must be >= 1, got {width}")
            self.from_axis = nn.Linear(cfg.d_axis, width)
            self.from_inv = nn.Linear(cfg.d_inv, width, bias=False)
            self.score = nn.Linear(width, 1, bias=False)

    def forward(self, state: EquivariantState) -> Tensor:
        """The pooled ``(..., d_axis)`` vector, invariant under §12.1."""
        axis = state.require_axis("AxisPool")
        if state.d_axis != self.d_axis:
            raise ValueError(
                f"AxisPool was built for d_axis={self.d_axis}, got a state of "
                f"width {state.d_axis}"
            )
        if self.mode == "mean":
            return axis.mean(dim=-2)
        scores = self.score(
            torch.tanh(self.from_axis(axis) + self.from_inv(state.inv).unsqueeze(-2))
        )
        weight = at_least_fp32(scores).softmax(dim=-2)
        return (weight * at_least_fp32(axis)).sum(dim=-2).to(axis.dtype)


class PhaseFiLM(nn.Module):
    """§13.2: phase-conditioned feature modulation, identity at initialisation.

    ```text
    scale, bias = PhaseMLP(E_phase[phase_id])
    h = scale * h + bias
    ```

    Separate invariant and axis projections, with one axis ``(scale, bias)``
    pair shared across the three channels, so the modulation commutes with any
    channel permutation.

    The final projections are zero-initialised and the scale is read as
    ``1 + delta``, so a fresh model's FiLM is exactly the identity (§27).

    The phase id is the authoritative three-way id of §13.1 and enters only
    here. Nothing downstream sees it; the KLENT return sign still reads
    ``moves_remaining``. ``use_three_way_phase=False`` folds OPENING and
    SECOND onto one embedding row while keeping the same input vocabulary.
    """

    def __init__(self, cfg: MantisACTConfig, d_phase: int | None = None) -> None:
        super().__init__()
        if cfg.phase_conditioning != "film":
            raise ValueError(
                f"phase_conditioning={cfg.phase_conditioning!r} does not use "
                "FiLM; the caller must not build one"
            )
        # §6 names no phase-embedding width. d_rel is the configuration's width
        # for a small learned table, which is what this is.
        width = cfg.d_rel if d_phase is None else int(d_phase)
        if width < 1:
            raise ValueError(f"phase embedding width must be >= 1, got {width}")

        rows = tuple(range(len(PHASE_IDS))) if cfg.use_three_way_phase else _TWO_WAY_ROWS
        self.register_buffer(
            "phase_row", torch.tensor(rows, dtype=torch.long), persistent=False
        )
        self.embed = nn.Embedding(len(set(rows)), width)
        nn.init.normal_(self.embed.weight, std=_EMBEDDING_STD)
        self.phase_mlp = nn.Sequential(
            nn.Linear(width, width), activation_module(cfg.activation)
        )
        self.to_inv = nn.Linear(width, 2 * cfg.d_inv)
        self.to_axis = nn.Linear(width, 2 * cfg.d_axis) if cfg.d_axis else None
        for projection in (self.to_inv, self.to_axis):
            if projection is not None:
                nn.init.zeros_(projection.weight)
                nn.init.zeros_(projection.bias)

    def _rows(self, phase_id: Tensor, leading: tuple[int, ...]) -> Tensor:
        """This FiLM's embedding row for every entity, from §13.1's phase id.

        The values are validated upstream: ``batch.phase_id`` is built at
        `packed.py:692` from per-graph ``ACTGraph.phase_id``, which
        `packed.py:491-502` refuses unless it is 0, 1, or 2 and agrees with
        ``moves_remaining`` and the occupied-cell count. That gate runs from
        ``__post_init__`` (`packed.py:324-325`), so it applies regardless of
        which builder produced the graph.

        What is enforced here is that the selection cannot alias: advanced
        indexing wraps a negative subscript, so ``phase_row[phase_id]`` would
        silently answer ``-1`` with the last phase's row. ``index_select``
        bounds-checks in both directions instead, with no synchronisation cost.
        """
        if phase_id.dtype != torch.long:
            raise ValueError(
                f"phase_id must be int64, got dtype {phase_id.dtype}"
            )
        try:
            broadcast = tuple(torch.broadcast_shapes(phase_id.shape, leading))
        except RuntimeError:
            broadcast = None
        if broadcast != leading:
            raise ValueError(
                f"phase_id of shape {tuple(phase_id.shape)} does not broadcast "
                f"onto the state's entity shape {leading}"
            )
        index = phase_id.reshape(-1)
        try:
            flat = self.phase_row.index_select(0, index)
        except IndexError as exc:
            raise ValueError(
                f"phase_id values must lie in 0..{len(PHASE_IDS) - 1}, got a "
                f"range of {int(index.min())}..{int(index.max())}"
            ) from exc
        return flat.view(phase_id.shape)

    def _selector(self, rows: Tensor, dtype: torch.dtype) -> Tensor:
        """A one-hot row picker over this FiLM's phase vocabulary.

        Selecting a class table's row per entity by matmul rather than by
        gather avoids `embedding_dense_backward`'s index sort and contended
        scatter: ``one_hot @ table`` backward is ``one_hot.T @ grad``, a plain
        matmul over a two- or three-column operand.

        The one-hot is exact in every floating dtype: each output element is
        one ``1.0`` times a table entry plus zeros.

        ``rows`` is in range by construction: it is a row of ``phase_row``,
        which `_rows` selects with ``index_select`` (bounds-checked).
        """
        return F.one_hot(rows, self.embed.num_embeddings).to(dtype)

    def forward(self, state: EquivariantState, phase_id: Tensor) -> EquivariantState:
        rows = self._rows(phase_id, state.leading_shape)

        # The modulation is a function of the phase class alone (at most three
        # classes, §13.1), so `embed -> phase_mlp -> to_inv/to_axis` is
        # evaluated on the vocabulary once rather than per entity; only the
        # row selection and the affine run per entity.
        code = self.phase_mlp(self.embed.weight)
        inv_table = self.to_inv(code)
        selector = self._selector(rows, inv_table.dtype)

        scale, bias = (selector @ inv_table).chunk(2, dim=-1)
        inv = (1 + scale) * state.inv + bias
        if state.axis is None:
            return EquivariantState(inv)
        if self.to_axis is None:
            raise ValueError(
                "this FiLM was built without an axis projection but received a "
                "state with an axis stream"
            )
        axis_scale, axis_bias = (selector @ self.to_axis(code)).chunk(2, dim=-1)
        axis = (1 + axis_scale).unsqueeze(-2) * state.axis + axis_bias.unsqueeze(-2)
        return EquivariantState(inv, axis)


def run_equivariant_stage(
    state: EquivariantState,
    mix: AxisMix,
    ffn: EquivariantFFN,
    *,
    film: PhaseFiLM | None = None,
    phase_id: Tensor | None = None,
) -> EquivariantState:
    """Run one whole AxisMix/FFN/FiLM stage through its registered op.

    This is the single parameter-packing seam for every symmetric entity
    family.  Keeping the order here avoids separate cell, window, action, and
    head call sites acquiring subtly different operator ABIs.  Unsupported
    ablations retain the literal module formulation.
    """
    if (film is None) != (phase_id is None):
        raise ValueError("film and phase_id must be supplied together")

    # Device-law mutation tests deliberately replace one stage with an
    # arbitrary wrapper module.  Such a wrapper has no registered-op parameter
    # ABI; executing the literal composition preserves both the mutation and
    # the detector that is meant to catch it.
    if not isinstance(mix, AxisMix) or not isinstance(ffn, EquivariantFFN):
        result = ffn(mix(state))
        return result if film is None else film(result, phase_id)

    if mix.activation != ffn.activation:
        raise ValueError(
            "AxisMix and EquivariantFFN activation disagree: "
            f"{mix.activation!r} versus {ffn.activation!r}"
        )

    if state.axis is None or not mix.d_axis:
        result = ffn(mix(state))
        return result if film is None else film(result, phase_id)

    stochastic = (
        (mix.drop.training and mix.drop.p != 0.0)
        or (ffn.drop.training and ffn.drop.p != 0.0)
    )
    if stochastic:
        result = ffn(mix(state))
        return result if film is None else film(result, phase_id)

    if mix.norm.axis is None or mix.residual.axis is None:
        raise ValueError("AxisMix has an axis width but lacks axis modules")
    if ffn.norm.axis is None or ffn.axis is None or ffn.residual.axis is None:
        raise ValueError("EquivariantFFN has an axis width but lacks axis modules")

    mix_parameters = (
        mix.norm.inv.weight,
        mix.norm.inv.bias,
        mix.norm.axis.weight,
        mix.norm.axis.bias,
        mix.inv_to_axis.weight,
        mix.inv_to_axis.bias,
        mix.mlp_axis[0].weight,
        mix.mlp_axis[0].bias,
        mix.mlp_axis[2].weight,
        mix.mlp_axis[2].bias,
        mix.phi_axis[0].weight,
        mix.phi_axis[0].bias,
        mix.mlp_inv[0].weight,
        mix.mlp_inv[0].bias,
        mix.mlp_inv[2].weight,
        mix.mlp_inv[2].bias,
        mix.residual.inv.gamma,
        mix.residual.axis.gamma,
    )
    ffn_parameters = (
        ffn.norm.inv.weight,
        ffn.norm.inv.bias,
        ffn.norm.axis.weight,
        ffn.norm.axis.bias,
        ffn.inv[0].weight,
        ffn.inv[0].bias,
        ffn.inv[2].weight,
        ffn.inv[2].bias,
        ffn.axis[0].weight,
        ffn.axis[0].bias,
        ffn.axis[2].weight,
        ffn.axis[2].bias,
        ffn.residual.inv.gamma,
        ffn.residual.axis.gamma,
    )

    film_parameters = None
    phase_row = None
    if film is not None:
        if film.to_axis is None:
            raise ValueError("phase FiLM lacks the required axis projection")
        film_parameters = (
            film.embed.weight,
            film.phase_mlp[0].weight,
            film.phase_mlp[0].bias,
            film.to_inv.weight,
            film.to_inv.bias,
            film.to_axis.weight,
            film.to_axis.bias,
        )
        phase_row = film.phase_row

    from .fused_equivariant import equivariant_stage

    out_inv, out_axis = equivariant_stage(
        state.inv,
        state.axis,
        mix_parameters,
        ffn_parameters,
        phase_id=phase_id,
        phase_row=phase_row,
        film=film_parameters,
        activation=mix.activation,
        mix_eps=(mix.norm.inv.eps, mix.norm.axis.eps),
        ffn_eps=(ffn.norm.inv.eps, ffn.norm.axis.eps),
    )
    return EquivariantState(out_inv, out_axis)


__all__ = [
    "AXIS_CHANNELS",
    "PHASE_IDS",
    "AxisMix",
    "AxisPool",
    "EquivariantFFN",
    "EquivariantNorm",
    "EquivariantResidual",
    "EquivariantState",
    "LayerScale",
    "PhaseFiLM",
    "activation_module",
    "at_least_fp32",
    "permute_axis_channels",
    "run_equivariant_stage",
]
