"""§12 and §13.2 of `mantis_act.equivariant`: the representation law itself.

Every equivariance assertion here is paired with a control that makes it able
to fail. The three axis channels are permuted with all six permutations of
``{0, 1, 2}`` and with the twelve permutations the board group actually induces
(from `mantis_act.symmetry`, which knows nothing of these modules), and each
module is then re-run through a deliberately broken variant carrying a
per-absolute-axis parameter — the exact construction §12.2 forbids. A check
that passes on the honest module and also passes on the broken one is not a
check, so both directions are asserted.

Two further vacuity guards: a module whose channels never interact would be
trivially equivariant, and a learned pool whose weights were uniform would be
the mean pool wearing parameters. Both are ruled out directly.

Arithmetic runs in fp64 so the tolerance can be tight enough that the controls
have nowhere to hide: a real violation here is O(0.1), and the residual float
noise from summing three channels in a different order is O(1e-15).
"""

from __future__ import annotations

import dataclasses
import itertools

import pytest
import torch
from torch import nn

from mantisnet.models.mantis_act.config import MantisACTConfig
from mantisnet.models.mantis_act.equivariant import (
    AXIS_CHANNELS,
    PHASE_IDS,
    AxisMix,
    AxisPool,
    EquivariantFFN,
    EquivariantNorm,
    EquivariantResidual,
    EquivariantState,
    LayerScale,
    PhaseFiLM,
    activation_module,
    permute_axis_channels,
)
from mantisnet.models.mantis_act.packed import (
    PHASE_FIRST,
    PHASE_OPENING,
    PHASE_SECOND,
)
from mantisnet.models.mantis_act.symmetry import D6_TRANSFORMS, axis_permutation

# Small widths keep the fp64 forwards cheap; d_inv stays divisible by the
# default head count, which the config validates.
D_INV, D_AXIS = 8, 4

# fp64 headroom. A channel-order-dependent sum differs in the last ulp; a
# broken module differs in the first digit.
TOL = 1e-11
BROKEN = 1e-3

ALL_PERMUTATIONS = list(itertools.permutations(range(AXIS_CHANNELS)))
D6_PERMUTATIONS = [axis_permutation(t) for t in range(len(D6_TRANSFORMS))]


def cfg(**overrides) -> MantisACTConfig:
    base = {"d_inv": D_INV, "d_axis": D_AXIS}
    return dataclasses.replace(MantisACTConfig(), **(base | overrides))


def randomize(module: nn.Module, seed: int = 0) -> nn.Module:
    """Give every parameter a value large enough that a bug cannot cancel.

    Fresh FiLM and LayerScale parameters are deliberately zero or tiny (§27),
    which would let a broken branch contribute nothing to the output; an
    equivariance test on an initialised model is therefore mostly a test of
    zero. Every test that asks whether a module is equivariant asks it of a
    module whose parameters are all live.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(
                torch.randn(parameter.shape, generator=generator, dtype=torch.float64)
                .to(parameter.dtype)
                * 0.5
            )
    return module


def state(
    n: int = 5,
    d_inv: int = D_INV,
    d_axis: int = D_AXIS,
    seed: int = 1,
    leading: tuple[int, ...] | None = None,
) -> EquivariantState:
    generator = torch.Generator().manual_seed(seed)
    shape = (n,) if leading is None else leading

    def draw(*trailing: int) -> torch.Tensor:
        return torch.randn(
            (*shape, *trailing), generator=generator, dtype=torch.float64
        )

    return EquivariantState(
        draw(d_inv), draw(AXIS_CHANNELS, d_axis) if d_axis else None
    )


def difference(left, right) -> float:
    """Max absolute elementwise difference of two states or two tensors."""
    if isinstance(left, EquivariantState):
        parts = [(left.inv, right.inv)]
        if left.has_axis:
            parts.append((left.axis, right.axis))
        return max(difference(a, b) for a, b in parts)
    return float((left - right).detach().abs().max())


def equivariance_error(module, source: EquivariantState, permutation) -> float:
    """``max |f(P x) - P f(x)|``, with ``P`` the identity on an invariant output.

    Both the state-valued modules and the tensor-valued pool go through this:
    an :class:`EquivariantState` output has its axis half permuted before the
    comparison, and a bare tensor output is invariant and is compared directly.
    """
    with torch.no_grad():
        moved = module(source.permute_axes(permutation))
        expected = module(source)
        if isinstance(expected, EquivariantState):
            expected = expected.permute_axes(permutation)
    return difference(moved, expected)


class PerAxisBiasMix(AxisMix):
    """`AxisMix` plus one bias per absolute axis — §12.2's forbidden form.

    This is the control. It differs from the honest module by a single term
    whose leading dimension is the axis channel, which is exactly the mistake
    the equivariance check exists to catch. The term is a buffer rather than a
    parameter so :func:`randomize` leaves its magnitude alone: a control whose
    size a helper could quietly shrink is a control that could stop failing.
    """

    def __init__(self, config: MantisACTConfig) -> None:
        super().__init__(config)
        self.register_buffer(
            "per_axis_bias",
            torch.tensor([1.0, 0.0, -1.0]).unsqueeze(-1).expand(
                AXIS_CHANNELS, config.d_axis
            ).contiguous(),
        )

    def forward(self, source: EquivariantState) -> EquivariantState:
        out = super().forward(source)
        return dataclasses.replace(out, axis=out.axis + self.per_axis_bias)


class PerAxisScorePool(AxisPool):
    """`AxisPool` whose attention scores carry a per-absolute-axis bias.

    The pool's invariance rests on the scores permuting with the channels; this
    control breaks precisely that pairing and nothing else.
    """

    def __init__(self, config: MantisACTConfig) -> None:
        super().__init__(config)
        self.register_buffer("per_axis_score", torch.tensor([2.0, 0.0, -2.0]))

    def forward(self, source: EquivariantState) -> torch.Tensor:
        axis = source.require_axis("PerAxisScorePool")
        scores = self.score(
            torch.tanh(self.from_axis(axis) + self.from_inv(source.inv).unsqueeze(-2))
        ) + self.per_axis_score.unsqueeze(-1)
        weight = scores.softmax(dim=-2)
        return (weight * axis).sum(dim=-2)


# --------------------------------------------------------------------------
# The state container


def test_state_reports_its_widths_and_shape():
    s = state(n=7)
    assert s.d_inv == D_INV
    assert s.d_axis == D_AXIS
    assert s.has_axis
    assert s.leading_shape == (7,)
    assert s.dtype == torch.float64


def test_state_without_axis_is_genuinely_absent():
    s = state(d_axis=0)
    assert s.axis is None
    assert s.d_axis == 0
    assert not s.has_axis
    with pytest.raises(ValueError, match="AxisPool needs an axis stream"):
        s.require_axis("AxisPool")
    # A no-axis state is unchanged by a permutation: there is nothing to carry.
    assert s.permute_axes((1, 2, 0)) is s


def test_state_refuses_a_zero_width_axis_tensor():
    # §29 full_no_axis: the axis half is absent, not a zero-width tensor kept
    # alive so axis-shaped code can keep running.
    with pytest.raises(ValueError, match="zero-width axis half must be absent"):
        EquivariantState(torch.zeros(4, D_INV), torch.zeros(4, AXIS_CHANNELS, 0))


@pytest.mark.parametrize(
    "axis, message",
    [
        (torch.zeros(4, 2, D_AXIS), "must carry 3 channels"),
        (torch.zeros(4, 4, D_AXIS), "must carry 3 channels"),
        (torch.zeros(5, AXIS_CHANNELS, D_AXIS), "share their leading shape"),
        (torch.zeros(4, D_AXIS), r"beside an \(\.\.\., d_inv\) inv"),
        (torch.zeros(4, AXIS_CHANNELS, D_AXIS, dtype=torch.float64), "share a dtype"),
    ],
)
def test_state_refuses_a_mismatched_axis_half(axis, message):
    with pytest.raises(ValueError, match=message):
        EquivariantState(torch.zeros(4, D_INV), axis)


def test_state_refuses_a_degenerate_invariant_half():
    with pytest.raises(ValueError, match="d_inv >= 1"):
        EquivariantState(torch.zeros(4, 0))
    with pytest.raises(ValueError, match="floating point"):
        EquivariantState(torch.zeros(4, D_INV, dtype=torch.long))
    with pytest.raises(TypeError, match="inv must be a Tensor"):
        EquivariantState([1.0, 2.0])


def test_replacing_a_stream_revalidates():
    s = state()
    assert dataclasses.replace(s, inv=s.inv * 2).d_inv == D_INV
    with pytest.raises(ValueError, match="share their leading shape"):
        dataclasses.replace(s, axis=torch.zeros(9, AXIS_CHANNELS, D_AXIS))


# --------------------------------------------------------------------------
# The permutation itself


@pytest.mark.parametrize("permutation", ALL_PERMUTATIONS)
def test_permutation_follows_the_representation_law(permutation):
    axis = torch.randn(6, AXIS_CHANNELS, D_AXIS, dtype=torch.float64)
    moved = permute_axis_channels(axis, permutation)
    # §12.1: h_axis'(pi(a)) = h_axis(a).
    for a, image in enumerate(permutation):
        assert torch.equal(moved[:, image], axis[:, a])


def test_permutation_composes_and_has_an_identity():
    s = state()
    assert torch.equal(s.permute_axes((0, 1, 2)).axis, s.axis)
    # Applying `first` then `then` carries a to then[first[a]].
    first, then = (1, 2, 0), (2, 0, 1)
    twice = s.permute_axes(first).permute_axes(then)
    once = s.permute_axes(tuple(then[first[a]] for a in range(AXIS_CHANNELS)))
    assert torch.equal(twice.axis, once.axis)


@pytest.mark.parametrize("bad", [(0, 1, 1), (0, 1), (0, 1, 3), (0, 1, 2, 0)])
def test_permutation_refuses_a_non_permutation(bad):
    with pytest.raises(ValueError, match="must be a permutation"):
        permute_axis_channels(torch.zeros(2, AXIS_CHANNELS, D_AXIS), bad)


def test_permutation_refuses_a_wrong_channel_count():
    with pytest.raises(ValueError, match=r"must be \(\.\.\., 3, d_axis\)"):
        permute_axis_channels(torch.zeros(2, 4, D_AXIS), (0, 1, 2))


def test_the_group_induces_every_axis_permutation():
    # The six permutations the tests below sweep are not an arbitrary set: D6
    # realises all of them, so equivariance under the sweep is equivariance
    # under the group's axis action.
    assert set(D6_PERMUTATIONS) == set(ALL_PERMUTATIONS)


# --------------------------------------------------------------------------
# AxisMix (§12.4)


@pytest.mark.parametrize("permutation", ALL_PERMUTATIONS)
def test_axis_mix_commutes_with_an_axis_permutation(permutation):
    mix = randomize(AxisMix(cfg()).double().eval())
    assert equivariance_error(mix, state(), permutation) < TOL


@pytest.mark.parametrize("permutation", D6_PERMUTATIONS)
def test_axis_mix_commutes_with_every_group_induced_permutation(permutation):
    mix = randomize(AxisMix(cfg()).double().eval(), seed=3)
    assert equivariance_error(mix, state(seed=4), permutation) < TOL


@pytest.mark.parametrize("permutation", ALL_PERMUTATIONS)
def test_a_per_axis_bias_is_detected(permutation):
    """The control: the same check, on a module carrying a forbidden parameter."""
    broken = randomize(PerAxisBiasMix(cfg()).double().eval())
    error = equivariance_error(broken, state(), permutation)
    if permutation == (0, 1, 2):
        assert error < TOL  # the identity cannot detect anything
    else:
        assert error > BROKEN


def test_axis_mix_actually_mixes_the_channels():
    """Otherwise equivariance would be the vacuous truth about a pointwise map.

    The probe has to change the *direction* of channel 0's vector. §12.4 reads
    the other channels through ``LN_axis``, and a LayerNorm is invariant to an
    affine rescaling of its input, so shifting or scaling a whole channel is a
    perturbation this module is entitled not to see.
    """
    mix = randomize(AxisMix(cfg()).double().eval())
    source = state()
    perturbed = source.axis.clone()
    perturbed[:, 0] += torch.randn(
        (perturbed.shape[0], D_AXIS),
        generator=torch.Generator().manual_seed(11),
        dtype=torch.float64,
    )
    with torch.no_grad():
        base = mix(source)
        moved = mix(dataclasses.replace(source, axis=perturbed))
    # Channel 0's change must reach channels 1 and 2 and the invariant stream.
    assert difference(base.axis[:, 1], moved.axis[:, 1]) > 1e-3
    assert difference(base.axis[:, 2], moved.axis[:, 2]) > 1e-3
    assert difference(base.inv, moved.inv) > 1e-3


def test_axis_mix_matches_the_spec_formulation_term_by_term():
    """§12.4 restated line for line, including `other_a = (total - u_a) / 2`.

    The restatement borrows the module's own submodules — this is about the
    wiring, not the weights — but the arithmetic between them is written out
    afresh, so a wrong divisor, a dropped LayerScale, or a residual taken from
    the post-update state shows up as a mismatch. Rebuilding with `/ 3` in
    place of `/ 2` is the negative half: it must not also match.
    """
    mix = randomize(AxisMix(cfg()).double().eval())
    source = state()
    with torch.no_grad():
        out = mix(source)
        u = mix.norm.axis(source.axis)
        z = mix.norm.inv(source.inv)
        total = u.sum(dim=-2, keepdim=True)
        context = mix.inv_to_axis(z).unsqueeze(-2).expand(u.shape)

        def rebuild(divisor: float) -> EquivariantState:
            other = (total - u) / divisor
            delta_axis = mix.mlp_axis(torch.cat((u, other, context), dim=-1))
            summary = mix.phi_axis(u).sum(dim=-2) / AXIS_CHANNELS
            delta_inv = mix.mlp_inv(torch.cat((z, summary), dim=-1))
            return EquivariantState(
                source.inv + mix.residual.inv.gamma * delta_inv,
                source.axis + mix.residual.axis.gamma * delta_axis,
            )

        assert difference(out, rebuild(2.0)) < TOL
        assert difference(out, rebuild(3.0)) > 1e-3


def test_axis_mix_residual_is_scaled_and_near_identity_at_init():
    mix = AxisMix(cfg()).double().eval()
    source = state()
    with torch.no_grad():
        out = mix(source)
    assert torch.allclose(mix.residual.inv.gamma, torch.full((D_INV,), 1e-2).double())
    assert torch.allclose(mix.residual.axis.gamma, torch.full((D_AXIS,), 1e-2).double())
    # LayerScale at 1e-2 keeps a fresh block close to the identity but not
    # equal to it: the branch is live, only quiet (§27).
    assert 0.0 < difference(out, source) < 0.5


def test_axis_mix_holds_no_axis_parameters_without_an_axis_stream():
    mix = AxisMix(cfg(d_axis=0, use_axis_channels=False, num_axis_latents=0))
    assert list(mix.parameters()) == []
    source = state(d_axis=0)
    assert mix(source) is source


def test_axis_mix_refuses_a_state_of_the_wrong_axis_width():
    mix = AxisMix(cfg())
    with pytest.raises(ValueError, match="built for d_axis=4"):
        mix(state(d_axis=6).to(torch.float32))


# --------------------------------------------------------------------------
# AxisPool (§12.5)


@pytest.mark.parametrize("mode", ["mean", "learned_attention"])
@pytest.mark.parametrize("permutation", ALL_PERMUTATIONS)
def test_axis_pool_is_invariant_under_an_axis_permutation(mode, permutation):
    pool = randomize(AxisPool(cfg(axis_pool_mode=mode)).double().eval())
    assert equivariance_error(pool, state(), permutation) < TOL


@pytest.mark.parametrize("permutation", ALL_PERMUTATIONS)
def test_a_per_axis_score_bias_is_detected(permutation):
    """The control for §12.5: scores that no longer permute with the channels."""
    broken = randomize(PerAxisScorePool(cfg()).double().eval())
    error = equivariance_error(broken, state(), permutation)
    if permutation == (0, 1, 2):
        assert error < TOL
    else:
        assert error > BROKEN


def test_learned_pool_is_not_the_mean_pool():
    """Otherwise its invariance would be the mean's, and its parameters dead."""
    pool = randomize(AxisPool(cfg()).double().eval())
    source = state()
    with torch.no_grad():
        learned = pool(source)
    assert difference(learned, source.axis.mean(dim=-2)) > 1e-3


def test_mean_pool_holds_no_parameters():
    assert list(AxisPool(cfg(axis_pool_mode="mean")).parameters()) == []


def test_learned_pool_softmax_runs_at_no_less_than_fp32():
    # §27. Under bf16 autocast the scores are promoted rather than exponentiated
    # in bf16, so a large score spread still produces finite, normalised weights.
    pool = randomize(AxisPool(cfg()).eval())
    source = state(seed=9).to(torch.float32)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = pool(source)
    assert torch.isfinite(out).all()
    # The pool is a convex combination of the channels, so it lies inside their
    # elementwise hull whatever the compute dtype.
    lower = source.axis.min(dim=-2).values
    upper = source.axis.max(dim=-2).values
    assert (out.double() >= lower - 1e-2).all()
    assert (out.double() <= upper + 1e-2).all()


def test_axis_pool_refuses_to_exist_without_an_axis_stream():
    # §29 full_no_axis keeps no unused axis parameters, so the caller must not
    # build a pool at all rather than build a dead one.
    with pytest.raises(ValueError, match="nothing to pool with d_axis=0"):
        AxisPool(cfg(d_axis=0, use_axis_channels=False, num_axis_latents=0))


# --------------------------------------------------------------------------
# PhaseFiLM (§13.2)


def phases(n: int = 6) -> torch.Tensor:
    return torch.tensor([PHASE_OPENING, PHASE_FIRST, PHASE_SECOND] * 2)[:n]


def test_fresh_film_is_exactly_the_identity():
    """§13.2/§27: scale 1, bias 0, bitwise — not merely close."""
    film = PhaseFiLM(cfg()).double().eval()
    source = state(n=6)
    with torch.no_grad():
        out = film(source, phases())
    assert torch.equal(out.inv, source.inv)
    assert torch.equal(out.axis, source.axis)


def test_fresh_film_is_the_identity_without_an_axis_stream():
    film = PhaseFiLM(cfg(d_axis=0, use_axis_channels=False, num_axis_latents=0))
    film = film.double().eval()
    source = state(n=6, d_axis=0)
    with torch.no_grad():
        out = film(source, phases())
    assert torch.equal(out.inv, source.inv)
    assert out.axis is None
    assert film.to_axis is None


def test_film_projections_start_at_zero_and_the_embedding_does_not():
    film = PhaseFiLM(cfg())
    assert torch.count_nonzero(film.to_inv.weight) == 0
    assert torch.count_nonzero(film.to_inv.bias) == 0
    assert torch.count_nonzero(film.to_axis.weight) == 0
    assert torch.count_nonzero(film.to_axis.bias) == 0
    # §27: N(0, 0.02). A dead table would make FiLM permanently the identity.
    assert 0.0 < float(film.embed.weight.detach().std()) < 0.05


def test_trained_film_modulates_and_distinguishes_the_phases():
    film = randomize(PhaseFiLM(cfg()).double().eval())
    source = state(n=3)
    with torch.no_grad():
        outs = [
            film(source, torch.full((3,), phase, dtype=torch.long))
            for phase in PHASE_IDS
        ]
    assert difference(outs[0], source) > 1e-3
    for left, right in itertools.combinations(outs, 2):
        assert difference(left, right) > 1e-3


@pytest.mark.parametrize("permutation", ALL_PERMUTATIONS)
def test_film_commutes_with_an_axis_permutation(permutation):
    """The axis scale and bias are one pair shared across the three channels."""
    film = randomize(PhaseFiLM(cfg()).double().eval())
    source = state(n=6)
    phase = phases()
    with torch.no_grad():
        moved = film(source.permute_axes(permutation), phase)
        expected = film(source, phase).permute_axes(permutation)
    assert difference(moved, expected) < TOL


def test_two_way_phase_folds_opening_onto_second():
    film = randomize(
        PhaseFiLM(cfg(use_three_way_phase=False)).double().eval()
    )
    assert film.embed.num_embeddings == 2
    source = state(n=3)
    with torch.no_grad():
        outs = {
            phase: film(source, torch.full((3,), phase, dtype=torch.long))
            for phase in PHASE_IDS
        }
    # Both are a turn's last placement; only FIRST is a different phase.
    assert difference(outs[PHASE_OPENING], outs[PHASE_SECOND]) < TOL
    assert difference(outs[PHASE_OPENING], outs[PHASE_FIRST]) > 1e-3


def test_film_broadcasts_a_per_position_phase_over_latents():
    # (P, K) latents against a (P, 1) phase: §17.1's shape, one phase per
    # position rather than one per latent.
    film = randomize(PhaseFiLM(cfg()).double().eval())
    source = state(leading=(4, 2))
    with torch.no_grad():
        out = film(source, torch.tensor([[0], [1], [2], [1]]))
    assert out.inv.shape == source.inv.shape
    assert out.axis.shape == source.axis.shape


def test_film_refuses_a_phase_id_it_cannot_index_by():
    """Both directions, because only one of them is the dangerous one.

    Above the vocabulary any gather refuses. *Below* it, torch's advanced
    indexing wraps: ``phase_row[-1]`` is the last phase's row, in range and
    wrongly typed, and ``-1`` is this representation's sentinel — it is what
    ``window_cell_index``, ``legal_to_cell_index`` and ``radius_axis_or_neg1``
    all carry — so a site that gathered a phase through a sentinel-bearing
    column would receive another phase's modulation with nothing raised at any
    stage. That is the symmetric fault ``CLAUDE.md`` names, and the reason the
    selection is an ``index_select`` rather than a subscript.
    """
    film = PhaseFiLM(cfg())
    source = state(n=3).to(torch.float32)
    with pytest.raises(ValueError, match="must be int64"):
        film(source, torch.tensor([0.0, 1.0, 2.0]))
    with pytest.raises(ValueError, match="does not broadcast"):
        film(source, torch.zeros(5, dtype=torch.long))

    for three_way in (True, False):
        film = PhaseFiLM(cfg(use_three_way_phase=three_way))
        # The three ids the vocabulary holds all index a row, in both arms.
        rows = film._rows(torch.tensor([0, 1, 2]), (3,))
        assert int(rows.min()) >= 0 and int(rows.max()) < len(PHASE_IDS)
        # And neither end of the range is answered with a row.
        with pytest.raises(ValueError, match=r"must lie in 0\.\.2"):
            film._rows(torch.tensor([0, 1, len(PHASE_IDS)]), (3,))
        with pytest.raises(ValueError, match=r"must lie in 0\.\.2"):
            film._rows(torch.tensor([0, 1, -1]), (3,))
        # Through the module's own entry point, not only the helper's.
        with pytest.raises(ValueError, match=r"must lie in 0\.\.2"):
            film.double()(state(n=3), torch.tensor([0, 1, -1]))


def test_film_refuses_to_exist_under_token_only_conditioning():
    with pytest.raises(ValueError, match="does not use FiLM"):
        PhaseFiLM(cfg(phase_conditioning="token_only"))


def test_film_phase_ids_agree_with_the_builder():
    assert PHASE_IDS == (PHASE_OPENING, PHASE_FIRST, PHASE_SECOND) == (0, 1, 2)


# --------------------------------------------------------------------------
# Norms, residuals, and the FFN (§18, §27)


def test_the_axis_norm_is_one_shared_set_of_parameters():
    # Three per-channel norms would be §12.2's per-absolute-axis norm; the
    # parameter count is the direct evidence.
    norm = EquivariantNorm(cfg())
    assert tuple(norm.axis.weight.shape) == (D_AXIS,)
    assert tuple(norm.inv.weight.shape) == (D_INV,)


def test_norm_instances_do_not_share_parameters():
    # §18 requires a norm per entity type and stream; each construction must
    # therefore be its own parameters, not a lookup into a shared one.
    left, right = EquivariantNorm(cfg()), EquivariantNorm(cfg())
    assert left.inv.weight is not right.inv.weight
    assert left.axis.weight is not right.axis.weight


def test_norm_refuses_a_state_of_the_wrong_shape():
    norm = EquivariantNorm(cfg())
    with pytest.raises(ValueError, match="expects d_inv=8"):
        norm(state(d_inv=16).to(torch.float32))
    with pytest.raises(ValueError, match="expects d_axis=4"):
        norm(state(d_axis=6).to(torch.float32))
    with pytest.raises(ValueError, match="built with an axis stream"):
        norm(state(d_axis=0).to(torch.float32))


def test_norm_without_an_axis_stream_builds_no_axis_parameters():
    norm = EquivariantNorm(cfg(d_axis=0, use_axis_channels=False, num_axis_latents=0))
    assert norm.axis is None
    with pytest.raises(ValueError, match="built without an axis stream"):
        norm(state().to(torch.float32))


def test_layer_scale_gains_start_at_the_configured_value():
    scale = LayerScale(D_AXIS, 1e-2)
    assert torch.equal(scale.gamma, torch.full((D_AXIS,), 1e-2))
    with pytest.raises(ValueError, match="width 3"):
        scale(torch.zeros(2, 3))
    with pytest.raises(ValueError, match="at least 1"):
        LayerScale(0, 1e-2)


def test_residual_adds_the_scaled_delta_and_checks_both_streams():
    residual = EquivariantResidual(cfg()).double()
    source = state()
    delta_inv = torch.ones_like(source.inv)
    delta_axis = torch.ones_like(source.axis)
    with torch.no_grad():
        out = residual(source, delta_inv, delta_axis)
    # The gains are built in fp32 (§27), so `1e-2` is the fp32 neighbour of a
    # tenth of a tenth; the exact addition is checked against the gain itself.
    assert torch.allclose(residual.inv.gamma, torch.full((D_INV,), 1e-2).double())
    assert torch.allclose(residual.axis.gamma, torch.full((D_AXIS,), 1e-2).double())
    assert difference(out.inv, source.inv + residual.inv.gamma) < TOL
    assert difference(out.axis, source.axis + residual.axis.gamma) < TOL
    with pytest.raises(ValueError, match="required exactly when"):
        residual(source, delta_inv)
    with pytest.raises(ValueError, match="invariant delta of shape"):
        residual(source, delta_inv[:2], delta_axis)


@pytest.mark.parametrize("permutation", ALL_PERMUTATIONS)
def test_ffn_commutes_with_an_axis_permutation(permutation):
    ffn = randomize(EquivariantFFN(cfg()).double().eval())
    assert equivariance_error(ffn, state(), permutation) < TOL


def test_ffn_keeps_its_two_streams_apart():
    """The FFN is pointwise per stream; only AxisMix crosses between them."""
    ffn = randomize(EquivariantFFN(cfg()).double().eval())
    source = state()
    with torch.no_grad():
        base = ffn(source)
        moved = ffn(dataclasses.replace(source, inv=source.inv + 1.0))
    assert difference(base.axis, moved.axis) < TOL
    assert difference(base.inv, moved.inv) > 1e-3


def test_ffn_without_an_axis_stream_has_no_axis_half():
    no_axis = cfg(d_axis=0, use_axis_channels=False, num_axis_latents=0)
    ffn = EquivariantFFN(no_axis).double().eval()
    assert ffn.axis is None
    assert ffn.residual.axis is None
    source = state(d_axis=0)
    with torch.no_grad():
        out = ffn(source)
    assert out.axis is None
    assert difference(out.inv, source.inv) > 0.0


def test_every_module_runs_under_bf16_autocast_and_stays_finite():
    # §27: parameters stay fp32 and the forward tolerates bf16 autocast.
    configuration = cfg()
    modules = [
        AxisMix(configuration),
        EquivariantFFN(configuration),
        PhaseFiLM(configuration),
        AxisPool(configuration),
    ]
    for module in modules:
        randomize(module.eval())
        for parameter in module.parameters():
            assert parameter.dtype == torch.float32
    source = state(n=6).to(torch.float32)
    with torch.autocast("cpu", dtype=torch.bfloat16), torch.no_grad():
        chained = modules[1](modules[0](source))
        chained = modules[2](chained, phases())
        pooled = modules[3](chained)
    assert torch.isfinite(chained.inv).all()
    assert torch.isfinite(chained.axis).all()
    assert torch.isfinite(pooled).all()


def test_gradients_reach_every_parameter():
    """A branch nothing differentiates is a branch that does not train."""
    configuration = cfg()
    mix, ffn, film = (
        AxisMix(configuration),
        EquivariantFFN(configuration),
        PhaseFiLM(configuration),
    )
    for module in (mix, ffn, film):
        randomize(module.double())
    out = film(ffn(mix(state(n=6))), phases())
    (out.inv.square().sum() + out.axis.square().sum()).backward()
    for module in (mix, ffn, film):
        for name, parameter in module.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name


def test_activation_factory_refuses_an_unknown_name():
    assert isinstance(activation_module("silu"), torch.nn.SiLU)
    with pytest.raises(ValueError, match="activation='tanh' is not one of"):
        activation_module("tanh")
