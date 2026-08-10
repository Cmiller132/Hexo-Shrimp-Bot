"""The §23 fork, its fp32 composition, and §24's optional auxiliaries.

What is being detected here, and why each detector is independent of the code
it watches:

- **KLENT's operator did not move (§23.2).** The composition is checked
  *against MantisNet's own* `return_mass`, `compose_q`, and `compose_acting_q`
  on shared inputs, bitwise. That is the only check that can see a new
  architecture quietly redefining `q_score` or `q_value`, because both
  implementations are self-consistent and every round trip inside either one
  would pass. The one deliberate difference — an fp64 input is promoted rather
  than demoted — is asserted as a difference, so it cannot be lost.
- **fp32 and outside autocast (§27).** The composition of bf16 logits is
  required to equal the composition of the same values in fp32, and to come
  back fp32. A composition that inherited autocast's precision would still be
  finite, still be ordered, and still pass every shape test.
- **Zero initialisation (§23.1, §23.2, §27).** Exact equality, not closeness:
  the initial policy is one constant over the whole action set and the initial
  Q is exactly zero. A near-zero output layer would train, and only an exact
  check notices it is not the specified initialisation.
- **The §12.1 law on the heads.** Permuting the three axis channels of the
  action state and of the latents must leave every output unchanged, since
  every output here is an invariant. §12.2's forbidden construction — reading
  channel 0 by its absolute id — is run through the same check, so the check
  is known to be able to fail.
- **§26 batching.** A batch of P positions equals P single forwards. That is
  the detector for the acting score's segment maximum leaking across a
  position boundary, which no single-position test can see.
- **§24's "a zero weight means absent".** Absence is checked as *no
  parameters*, not as a zero multiplier, and the overlap refusal is checked to
  name the §19.3 input it collides with.

Positions come from seeded random playouts through the engine, and are used
only where a real builder output is the point — the phase of a first placement,
the status of a window. As ``docs/MANTIS_ACT_DEVIATIONS.md`` records, random
play is nothing like self-play density, so nothing here asserts a family size.
The heads see exactly one row per legal action, so density reaches them only as
a row count, and the synthetic action states below carry the shapes a real one
would.
"""

from __future__ import annotations

import random
from dataclasses import replace

import hexo_py
import pytest
import torch
from torch import nn

from mantisnet.model import compose_acting_q as mantis_compose_acting_q
from mantisnet.model import compose_q as mantis_compose_q
from mantisnet.model import return_mass as mantis_return_mass
from mantisnet.models.mantis_act.actions import TACTICAL_FEATURE_NAMES
from mantisnet.models.mantis_act.builder import build
from mantisnet.models.mantis_act.config import PRESETS, MantisACTConfig
from mantisnet.models.mantis_act.equivariant import (
    AXIS_CHANNELS,
    EquivariantState,
    permute_axis_channels,
)
from mantisnet.models.mantis_act.heads import (
    ACTION_AUXILIARIES,
    AUX_SPECS,
    CATEGORICAL_CRITIC_LOGITS,
    MASK_SUFFIX,
    SCALAR_CRITIC_LOGITS,
    WINDOW_FATE_CLASSES,
    WINDOW_FATE_HEAD,
    ActionHeads,
    HeadOutput,
    InvariantReadout,
    compose_acting_q,
    compose_q,
    committed_mass,
    critic_logit_width,
    masked_auxiliaries,
    return_mass,
)
from mantisnet.models.mantis_act.latents import LatentState
from mantisnet.models.mantis_act.packed import (
    PHASE_FIRST,
    collate,
)
from mantisnet.models.mantis_act.pattern_classes import OPP_LIVE, OWN_LIVE
from mantisnet.models.mantis_act.summary import parameter_summary

SEED = 20260806

FULL = PRESETS["full_act_v4"]

# Every permutation of the three axis channels except the identity.
PERMUTATIONS = ((1, 2, 0), (2, 0, 1), (0, 2, 1), (2, 1, 0), (1, 0, 2))

# Both movers, both stones of a turn, all three phases, and boards deep enough
# to carry live and mixed windows.
PLIES = (0, 1, 2, 5, 21, 60)

# A representative ragged action set: positions of different legal counts,
# including one with a single action, where a segment maximum over the position
# is its own row.
COUNTS = (1, 4, 7, 3)

MASS_FLOOR = 0.2


# --------------------------------------------------------------------------
# Fixtures


def make_offsets(counts=COUNTS) -> torch.Tensor:
    """CSR offsets for a ragged family with the given per-position counts."""
    return torch.tensor([0, *torch.tensor(counts).cumsum(0).tolist()])


def make_state(cfg: MantisACTConfig, rows: int, *, seed: int) -> EquivariantState:
    """A synthetic entity family of ``rows`` rows under ``cfg``'s widths."""
    generator = torch.Generator().manual_seed(seed)
    inv = torch.randn(rows, cfg.d_inv, generator=generator)
    if not cfg.d_axis:
        return EquivariantState(inv)
    axis = torch.randn(rows, AXIS_CHANNELS, cfg.d_axis, generator=generator)
    return EquivariantState(inv, axis)


def make_latents(
    cfg: MantisACTConfig, positions: int, *, seed: int
) -> LatentState | None:
    """The state latents ``cfg`` describes, or ``None`` when it has none."""
    generator = torch.Generator().manual_seed(seed)
    inv = (
        torch.randn(positions, cfg.num_inv_latents, cfg.d_inv, generator=generator)
        if cfg.num_inv_latents
        else None
    )
    axis = (
        torch.randn(
            positions,
            cfg.num_axis_latents,
            AXIS_CHANNELS,
            cfg.d_axis,
            generator=generator,
        )
        if cfg.num_axis_latents
        else None
    )
    if inv is None and axis is None:
        return None
    return LatentState(inv=inv, axis=axis)


def enliven(module: nn.Module, *, seed: int = SEED) -> nn.Module:
    """Give a fresh module a parameterisation whose outputs actually vary.

    Two things about the specified initialisation make an untrained module a
    poor probe for anything but the initialisation itself: LayerScale is 1e-2,
    so every residual branch is a whisper, and both final projections are
    exactly zero, so every output is the same constant. Tests that are about
    the initialisation use a fresh module; every other test uses this one.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if name.endswith("gamma") and parameter.dim() == 1:
                parameter.copy_(
                    torch.empty_like(parameter).normal_(0.5, 0.2, generator=generator)
                )
            elif name.endswith("out.weight") or name.endswith("out.bias"):
                parameter.copy_(
                    torch.empty_like(parameter).normal_(0.0, 0.3, generator=generator)
                )
    return module.eval()


@pytest.fixture(scope="module")
def heads() -> ActionHeads:
    torch.manual_seed(SEED)
    return enliven(ActionHeads(FULL))


@pytest.fixture(scope="module")
def offsets() -> torch.Tensor:
    return make_offsets()


@pytest.fixture(scope="module")
def actions(offsets) -> EquivariantState:
    return make_state(FULL, int(offsets[-1]), seed=SEED + 1)


@pytest.fixture(scope="module")
def latents(offsets) -> LatentState:
    return make_latents(FULL, int(offsets.shape[0]) - 1, seed=SEED + 2)


def run(heads, actions, offsets, latents, **kwargs) -> HeadOutput:
    """The default forward under test, with KLENT's floor configured."""
    kwargs.setdefault("mass_floor", MASS_FLOOR)
    return heads(actions, legal_offsets=offsets, latents=latents, **kwargs)


def playout(plies: int, seed: int) -> list[tuple[int, int]]:
    """A seeded nonterminal random playout of exactly ``plies`` placements."""
    for attempt in range(100):
        rng = random.Random(seed * 7919 + attempt * 31 + plies)
        position = hexo_py.Position()
        moves: list[tuple[int, int]] = []
        for _ in range(plies):
            move = rng.choice(position.legal_moves())
            position.advance(*move)
            moves.append(move)
        if not position.is_terminal:
            return moves
    raise AssertionError(f"no nonterminal {plies}-ply playout in 100 seeds")


@pytest.fixture(scope="module")
def real_batch():
    """Real builder output: every phase, and windows of every status."""
    graphs = [
        build(hexo_py.Position.replay(playout(plies, SEED)), FULL) for plies in PLIES
    ]
    return collate(graphs, FULL)


# --------------------------------------------------------------------------
# Construction (§23, §29)


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_every_preset_forks_and_runs(preset):
    """§37.1: every named arm constructs and answers both heads."""
    cfg = PRESETS[preset]
    module = ActionHeads(cfg)
    offsets = make_offsets()
    out = run(
        module,
        make_state(cfg, int(offsets[-1]), seed=SEED),
        offsets,
        make_latents(cfg, len(COUNTS), seed=SEED),
    )
    assert out.policy_logits.shape == (int(offsets[-1]),)
    assert out.critic_logits.shape == (int(offsets[-1]), critic_logit_width(cfg))
    assert out.q_value.dtype is torch.float32
    assert torch.isfinite(out.q_value).all()
    assert out.state_value is None
    assert out.aux == {}


def test_the_three_separations_construct_and_differ(actions, offsets, latents):
    """§23, §35.10: private adapters, output-only separation, one shared body."""
    outputs = {}
    modules = {}
    for mode in ("private_adapters", "separate_output_mlps", "single_shared_head"):
        torch.manual_seed(SEED)
        module = enliven(ActionHeads(replace(FULL, head_separation=mode)))
        modules[mode] = module
        outputs[mode] = run(module, actions, offsets, latents)

    private = modules["private_adapters"]
    assert private.policy_adapter is not None and private.critic_adapter is not None
    # Private means private: the two adapters share no parameter object.
    policy_ids = {id(p) for p in private.policy_adapter.parameters()}
    critic_ids = {id(p) for p in private.critic_adapter.parameters()}
    assert policy_ids and not (policy_ids & critic_ids)
    assert private.policy.readout is not private.critic.readout

    separate = modules["separate_output_mlps"]
    assert separate.policy_adapter is None
    assert separate.policy.readout is not separate.critic.readout

    shared = modules["single_shared_head"]
    assert shared.policy_adapter is None
    assert shared.policy.readout is shared.critic.readout
    # The output layers still cannot be shared: their widths differ.
    assert shared.policy.out.out_features == 1
    assert shared.critic.out.out_features == CATEGORICAL_CRITIC_LOGITS

    # Same seed for all three, so a mode that changed nothing about the
    # computation would answer identically. The comparison is over the pair:
    # `separate_output_mlps` and `single_shared_head` draw the policy body from
    # the same point in the same generator, so it is the critic — reading a body
    # the shared arm does not have — that separates those two.
    for left in outputs:
        for right in outputs:
            if left < right:
                assert not (
                    torch.allclose(
                        outputs[left].policy_logits, outputs[right].policy_logits
                    )
                    and torch.allclose(
                        outputs[left].critic_logits, outputs[right].critic_logits
                    )
                )
    private = outputs["private_adapters"]
    for other in ("separate_output_mlps", "single_shared_head"):
        assert not torch.allclose(private.policy_logits, outputs[other].policy_logits)


def test_a_private_adapter_with_no_block_is_refused():
    """A model both modes name is refused rather than built twice over."""
    with pytest.raises(ValueError, match="separate_output_mlps"):
        ActionHeads(replace(FULL, policy_private_blocks=0))


def test_the_shared_head_preset_keeps_its_unread_block_counts():
    """§29's full_shared_head is full_act_v4 with only head_separation moved."""
    cfg = PRESETS["full_shared_head"]
    assert cfg.policy_private_blocks == FULL.policy_private_blocks
    ActionHeads(cfg)


def test_the_no_axis_arm_holds_no_axis_parameter():
    """§29 full_no_axis, §32: a removed stream keeps no parameters."""
    cfg = PRESETS["full_no_axis"]
    module = ActionHeads(cfg)
    assert all("axis" not in name for name, _ in module.named_parameters())
    offsets = make_offsets()
    out = run(
        module,
        make_state(cfg, int(offsets[-1]), seed=SEED),
        offsets,
        make_latents(cfg, len(COUNTS), seed=SEED),
    )
    assert torch.isfinite(out.policy_logits).all()


def test_the_no_latents_arm_reads_no_latents():
    """§29 full_no_latents: the adapters hold no latent-context parameters."""
    cfg = PRESETS["full_no_latents"]
    module = ActionHeads(cfg)
    assert module.reads_latents is False
    assert module.policy_adapter.blocks[0].context is None
    offsets = make_offsets()
    out = run(module, make_state(cfg, int(offsets[-1]), seed=SEED), offsets, None)
    assert torch.isfinite(out.policy_logits).all()


def test_a_mismatched_action_width_is_refused(heads, offsets, latents):
    narrow = make_state(replace(FULL, d_inv=32), int(offsets[-1]), seed=SEED)
    with pytest.raises(ValueError, match="d_inv"):
        run(heads, narrow, offsets, latents)


def test_a_row_count_disagreeing_with_the_offsets_is_refused(heads, latents):
    """The offsets must end where the action state's rows do.

    The heads gather their per-position context by ``row_positions``, which is
    given the action row count as ATen's ``output_size`` — refused, on the
    device, when it disagrees with the offsets' own total
    (``result_size == cumsum_ptr[size - 1]``). The predicate is the one the
    heads used to read back for themselves, now enforced by the call that
    consumes it rather than beside it.
    """
    offsets = make_offsets()
    short = make_state(FULL, int(offsets[-1]) - 1, seed=SEED)
    with pytest.raises(RuntimeError, match="size does not match"):
        run(heads, short, offsets, latents)


# --------------------------------------------------------------------------
# Zero initialisation (§23.1, §23.2, §27)


def test_a_fresh_policy_is_one_constant_over_every_legal_action(
    actions, offsets, latents
):
    """§23.1: zero-initialised output, so the initial policy is uniform."""
    torch.manual_seed(SEED)
    out = run(ActionHeads(FULL), actions, offsets, latents)
    first = float(out.policy_logits[0].detach())
    assert torch.equal(out.policy_logits, torch.full_like(out.policy_logits, first))
    assert first == 0.0


def test_a_fresh_critic_is_exactly_zero(actions, offsets, latents):
    """§23.2: zero-initialised output, so Q is exactly zero — not nearly."""
    torch.manual_seed(SEED)
    out = run(ActionHeads(FULL), actions, offsets, latents)
    assert torch.equal(out.critic_logits, torch.zeros_like(out.critic_logits))
    assert torch.equal(out.q_value, torch.zeros_like(out.q_value))
    assert torch.equal(out.q_score, torch.zeros_like(out.q_score))
    # Uniform over three classes, so the committed mass is exactly two thirds.
    assert torch.allclose(
        out.committed_mass, torch.full_like(out.committed_mass, 2.0 / 3.0), atol=0
    )


def test_a_fresh_scalar_critic_is_exactly_zero(actions, offsets, latents):
    torch.manual_seed(SEED)
    cfg = replace(FULL, critic_type="scalar_tanh")
    out = ActionHeads(cfg)(
        actions, legal_offsets=offsets, latents=latents, mass_floor=None
    )
    assert out.critic_logits.shape == (int(offsets[-1]), SCALAR_CRITIC_LOGITS)
    assert torch.equal(out.q_value, torch.zeros_like(out.q_value))
    assert out.committed_mass is None


def test_a_fresh_state_value_is_exactly_zero(actions, offsets, latents):
    torch.manual_seed(SEED)
    module = ActionHeads(replace(FULL, enable_state_value_head=True))
    out = run(module, actions, offsets, latents)
    assert out.state_value.shape == (len(COUNTS),)
    assert torch.equal(out.state_value, torch.zeros_like(out.state_value))


# --------------------------------------------------------------------------
# The composition (§23.2, §25, §27)


def random_logits(rows: int, *, seed: int = SEED) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, CATEGORICAL_CRITIC_LOGITS, generator=generator) * 2.0


def test_the_composition_is_mantisnets_operator_bitwise(offsets):
    """§23.2: KLENT's q_score/q_value are not silently redefined."""
    logits = random_logits(int(offsets[-1]))
    ours = return_mass(logits)
    theirs = mantis_return_mass(logits)
    assert torch.equal(ours[0], theirs[0]) and torch.equal(ours[1], theirs[1])
    assert torch.equal(compose_q(logits), mantis_compose_q(logits))
    assert torch.equal(
        compose_acting_q(logits, offsets, MASS_FLOOR),
        mantis_compose_acting_q(logits, offsets, MASS_FLOOR),
    )
    assert torch.equal(committed_mass(logits), theirs[0] + theirs[1])


def test_the_composition_is_fp32_whatever_the_logits_are(offsets):
    """§27: the composition does not inherit autocast's precision."""
    logits = random_logits(int(offsets[-1]))
    reduced = logits.bfloat16()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        inside = compose_q(reduced)
        inside_score = compose_acting_q(reduced, offsets, MASS_FLOOR)
    assert inside.dtype is torch.float32
    assert inside_score.dtype is torch.float32
    # Composing the same values outside autocast gives the same numbers.
    assert torch.equal(inside, compose_q(reduced))
    assert torch.equal(inside, compose_q(reduced.float()))
    assert torch.equal(inside_score, compose_acting_q(reduced, offsets, MASS_FLOOR))


def test_an_fp64_reference_is_promoted_rather_than_demoted(offsets):
    """The one deviation from MantisNet's `.float()`, asserted as one.

    §27 asks for "no less than fp32"; the package's one promotion helper keeps
    an fp64 input at fp64, where MantisNet's ``.float()`` would demote it. No
    dtype KLENT runs at is affected, and this is the check that says so.
    """
    logits = random_logits(int(offsets[-1])).double()
    assert compose_q(logits).dtype is torch.float64
    assert mantis_compose_q(logits).dtype is torch.float32
    assert torch.allclose(compose_q(logits).float(), mantis_compose_q(logits))


def test_equal_logits_give_exactly_zero_q(offsets):
    rows = int(offsets[-1])
    for value in (-3.0, 0.0, 7.5):
        logits = torch.full((rows, CATEGORICAL_CRITIC_LOGITS), value)
        assert torch.equal(compose_q(logits), torch.zeros(rows))


def test_the_acting_score_scales_by_the_positions_own_maximum_mass(offsets):
    """§23.2: one positive divisor per position, and Q's order preserved."""
    logits = random_logits(int(offsets[-1]))
    mass = committed_mass(logits)
    value = compose_q(logits)
    score = compose_acting_q(logits, offsets, MASS_FLOOR)
    for position in range(len(COUNTS)):
        lo, hi = int(offsets[position]), int(offsets[position + 1])
        divisor = max(float(mass[lo:hi].max()), MASS_FLOOR)
        assert torch.allclose(score[lo:hi], value[lo:hi] / divisor, atol=1e-6)
        order = torch.argsort(value[lo:hi])
        assert torch.equal(torch.argsort(score[lo:hi]), order)
    assert float(score.abs().max()) < 1.0


def test_the_floor_binds_when_every_action_commits_little(offsets):
    """A position that puts almost all mass on zero return is not sharpened."""
    rows = int(offsets[-1])
    logits = torch.zeros(rows, CATEGORICAL_CRITIC_LOGITS)
    logits[:, 2] = 12.0  # p_zero ~ 1, so the committed mass is far below 0.2
    logits[:, 0] = 1.0
    mass = committed_mass(logits)
    assert float(mass.max()) < MASS_FLOOR
    assert torch.allclose(
        compose_acting_q(logits, offsets, MASS_FLOOR), compose_q(logits) / MASS_FLOOR
    )


def test_a_nonsense_floor_is_refused(offsets):
    logits = random_logits(int(offsets[-1]))
    for floor in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="mass_floor"):
            compose_acting_q(logits, offsets, floor)


def test_the_scalar_critic_refuses_mass_scaling(actions, offsets, latents):
    """§23.2: a point estimate has no p_pos/p_neg to sum, and says so."""
    module = enliven(ActionHeads(replace(FULL, critic_type="scalar_tanh")))
    with pytest.raises(ValueError, match="categorical3"):
        run(module, actions, offsets, latents)
    out = module(actions, legal_offsets=offsets, latents=latents, mass_floor=None)
    assert torch.equal(out.q_value, out.q_score)
    assert torch.allclose(out.q_value, torch.tanh(out.critic_logits.squeeze(-1)))
    assert float(out.q_value.detach().abs().max()) < 1.0


def test_no_floor_means_no_scaling(actions, offsets, latents, heads):
    """``mass_floor=None`` is the explicit absence, not a default floor."""
    out = heads(actions, legal_offsets=offsets, latents=latents, mass_floor=None)
    assert torch.equal(out.q_score, out.q_value)
    floored = run(heads, actions, offsets, latents)
    assert not torch.equal(floored.q_score, floored.q_value)


def test_the_acting_score_does_not_cross_a_position(heads, actions, offsets, latents):
    """§26: the segment maximum is the position's own."""
    whole = run(heads, actions, offsets, latents)
    start = 0
    for position, count in enumerate(COUNTS):
        rows = slice(start, start + count)
        alone = run(
            heads,
            EquivariantState(
                actions.inv[rows],
                None if actions.axis is None else actions.axis[rows],
            ),
            torch.tensor([0, count]),
            LatentState(
                inv=latents.inv[position : position + 1],
                axis=latents.axis[position : position + 1],
            ),
        )
        assert torch.allclose(alone.policy_logits, whole.policy_logits[rows], atol=1e-6)
        assert torch.allclose(alone.q_value, whole.q_value[rows], atol=1e-6)
        assert torch.allclose(alone.q_score, whole.q_score[rows], atol=1e-6)
        start += count


# --------------------------------------------------------------------------
# The representation law (§12.1)


class _AbsoluteAxisReadout(InvariantReadout):
    """§12.2's forbidden construction, as the negative control.

    Reads axis channel 0 by its absolute id instead of pooling symmetrically
    over the three. Everything else about the head is unchanged, so a check
    that cannot see this cannot see a real equivariance break either.
    """

    def forward(self, state: EquivariantState) -> torch.Tensor:
        z = self.norm(state)
        return self.body(torch.cat((z.inv, z.axis[..., 0, :]), dim=-1))


@pytest.mark.parametrize("permutation", PERMUTATIONS)
def test_the_outputs_are_invariant_under_an_axis_permutation(
    heads, actions, offsets, latents, permutation
):
    """§12.1: every head output is an invariant, so it does not move."""
    base = run(heads, actions, offsets, latents)
    turned = run(
        heads,
        actions.permute_axes(permutation),
        offsets,
        LatentState(
            inv=latents.inv, axis=permute_axis_channels(latents.axis, permutation)
        ),
    )
    assert torch.allclose(turned.policy_logits, base.policy_logits, atol=1e-6)
    assert torch.allclose(turned.critic_logits, base.critic_logits, atol=1e-6)
    assert torch.allclose(turned.q_value, base.q_value, atol=1e-6)


def test_the_invariance_check_can_fail(actions, offsets, latents):
    torch.manual_seed(SEED)
    module = enliven(ActionHeads(FULL))
    module.policy.readout = enliven(_AbsoluteAxisReadout(FULL), seed=SEED + 5)
    base = run(module, actions, offsets, latents)
    turned = run(
        module,
        actions.permute_axes(PERMUTATIONS[0]),
        offsets,
        LatentState(
            inv=latents.inv,
            axis=permute_axis_channels(latents.axis, PERMUTATIONS[0]),
        ),
    )
    assert not torch.allclose(turned.policy_logits, base.policy_logits, atol=1e-6)


def test_a_latent_context_that_names_a_channel_would_be_visible(
    heads, actions, offsets, latents
):
    """Permuting only the latents, and not the actions, must move the output.

    The adapter's latent context pairs channel ``a`` of the latent with channel
    ``a`` of the action. If it pooled the latent axis into an invariant instead,
    the outputs above would be invariant for a reason that has nothing to do
    with the pairing, and this check would fail.
    """
    base = run(heads, actions, offsets, latents)
    half = run(
        heads,
        actions,
        offsets,
        LatentState(
            inv=latents.inv,
            axis=permute_axis_channels(latents.axis, PERMUTATIONS[0]),
        ),
    )
    assert not torch.allclose(half.policy_logits, base.policy_logits, atol=1e-6)


# --------------------------------------------------------------------------
# Auxiliaries (§24)


def test_no_auxiliary_exists_by_default(heads, actions, offsets, latents):
    """§24: every auxiliary is off, and off means absent."""
    assert not FULL.enable_action_aux_heads
    assert not FULL.enable_window_fate_head
    assert not FULL.enable_state_value_head
    assert heads.auxiliaries is None
    assert heads.window_fate is None
    assert heads.state_value is None
    names = [name for name, _ in heads.named_parameters()]
    for spec in ACTION_AUXILIARIES:
        assert all(spec.name not in name for name in names)
    assert run(heads, actions, offsets, latents).aux == {}


def aux_config(**overrides) -> MantisACTConfig:
    """A configuration in which the unmasked auxiliaries may be asked for."""
    return replace(FULL, enable_action_aux_heads=True, **overrides)


def test_an_auxiliary_is_named_when_present(actions, offsets, latents):
    """§24: independently named weights, and one head's own parameters."""
    weights = {"winning_partner_exists": 0.5, "winning_partner_count": 0.25}
    module = enliven(ActionHeads(aux_config(), aux_weights=weights))
    assert module.auxiliaries.weights == weights
    names = [name for name, _ in module.named_parameters()]
    for head in weights:
        assert any(head in name for name in names)
    assert all("win_now" not in name for name in names)

    out = run(module, actions, offsets, latents, phase_id=torch.zeros(len(COUNTS), dtype=torch.long))
    assert set(out.aux) == {
        head + suffix for head in weights for suffix in ("", MASK_SUFFIX)
    }
    rows = int(offsets[-1])
    assert out.aux["winning_partner_exists"].shape == (rows,)
    assert out.aux["winning_partner_count"].shape == (
        rows,
        AUX_SPECS["winning_partner_count"].logits,
    )


def test_a_zero_weight_means_the_head_is_absent(actions, offsets, latents):
    """§24: absent, not instantiated and multiplied by zero."""
    module = ActionHeads(
        aux_config(),
        aux_weights={"winning_partner_exists": 0.5, "winning_partner_count": 0.0},
    )
    assert module.auxiliaries.names == ("winning_partner_exists",)
    names = [name for name, _ in module.named_parameters()]
    assert all("winning_partner_count" not in name for name in names)

    # The explicit debug option §24 allows, and nothing else, instantiates it.
    debug = ActionHeads(
        aux_config(),
        aux_weights={"winning_partner_count": 0.0},
        instantiate_zero_weight=True,
    )
    assert debug.auxiliaries.names == ("winning_partner_count",)


def test_enabling_the_auxiliaries_without_selecting_one_is_refused():
    with pytest.raises(ValueError, match="zero weight"):
        ActionHeads(aux_config(), aux_weights={"winning_partner_exists": 0.0})


def test_an_auxiliary_asked_for_with_the_flag_off_is_refused():
    with pytest.raises(ValueError, match="enable_action_aux_heads"):
        ActionHeads(FULL, aux_weights={"winning_partner_exists": 0.5})


def test_an_unknown_or_negative_weight_is_refused():
    with pytest.raises(ValueError, match="unknown"):
        ActionHeads(aux_config(), aux_weights={"not_a_head": 1.0})
    with pytest.raises(ValueError, match="negative"):
        ActionHeads(aux_config(), aux_weights={"win_now": -1.0})


# --- §24.1's overlap mask ---------------------------------------------------


def test_the_overlap_table_names_real_inputs():
    """Every claimed overlap is a §19.3 field that still exists."""
    for spec in ACTION_AUXILIARIES:
        for field in spec.tactical:
            assert field in TACTICAL_FEATURE_NAMES


def test_the_masked_auxiliaries_are_exactly_the_overlapping_ones():
    masked = masked_auxiliaries(FULL)
    assert set(masked) == {
        "win_now",
        "own_max_occupancy",
        "opponent_threats_hit",
        "own_five_windows_after",
    }
    # The learned-only input ablation removes the mask entirely (§24.1, §29).
    assert masked_auxiliaries(PRESETS["full_no_tactical_inputs"]) == {}


@pytest.mark.parametrize("name", sorted(masked_auxiliaries(FULL)))
def test_an_overlapping_auxiliary_is_refused_by_the_input_it_duplicates(name):
    with pytest.raises(ValueError) as raised:
        ActionHeads(aux_config(), aux_weights={name: 0.5})
    message = str(raised.value)
    assert name in message
    for field in AUX_SPECS[name].tactical:
        assert field in message
    # The refusal states §24.1's alternative, by the name of the preset.
    assert "full_no_tactical_inputs" in message


def test_every_auxiliary_is_available_under_the_learned_only_ablation(
    actions, offsets, latents
):
    """§24.1's other remedy: no exact tactical input, so no overlap."""
    cfg = replace(PRESETS["full_no_tactical_inputs"], enable_action_aux_heads=True)
    module = enliven(
        ActionHeads(cfg, aux_weights={spec.name: 0.1 for spec in ACTION_AUXILIARIES})
    )
    assert module.auxiliaries.names == tuple(AUX_SPECS)
    out = run(
        module,
        actions,
        offsets,
        latents,
        phase_id=torch.zeros(len(COUNTS), dtype=torch.long),
    )
    for spec in ACTION_AUXILIARIES:
        logits = out.aux[spec.name]
        expected = (int(offsets[-1]),) if spec.logits == 1 else (
            int(offsets[-1]),
            spec.logits,
        )
        assert logits.shape == expected


# --- the masks --------------------------------------------------------------


def test_the_first_placement_auxiliaries_are_masked_to_first_placements(real_batch):
    """§24.1: auxiliaries 5 and 6 are labelled on FIRST states only."""
    module = enliven(
        ActionHeads(aux_config(), aux_weights={"winning_partner_exists": 0.5})
    )
    rows = int(real_batch.legal_offsets[-1])
    out = run(
        module,
        make_state(FULL, rows, seed=SEED),
        real_batch.legal_offsets,
        make_latents(FULL, real_batch.position_count, seed=SEED),
        phase_id=real_batch.phase_id,
    )
    mask = out.aux["winning_partner_exists" + MASK_SUFFIX]
    assert mask.dtype is torch.bool
    counts = real_batch.legal_offsets[1:] - real_batch.legal_offsets[:-1]
    expected = torch.repeat_interleave(real_batch.phase_id == PHASE_FIRST, counts)
    assert torch.equal(mask, expected)
    # A real batch of these plies contains both phases, so the mask is not
    # vacuously true or vacuously false.
    assert bool(mask.any()) and not bool(mask.all())


def test_a_first_placement_auxiliary_without_a_phase_is_refused(
    actions, offsets, latents
):
    module = ActionHeads(aux_config(), aux_weights={"winning_partner_exists": 0.5})
    with pytest.raises(ValueError, match="phase_id"):
        run(module, actions, offsets, latents)


def test_the_window_fate_head_sees_live_windows_only(real_batch):
    """§24.2: mixed windows are already dead for both players, so they mask."""
    cfg = replace(FULL, enable_window_fate_head=True)
    module = enliven(ActionHeads(cfg, window_fate_weight=0.1))
    windows = make_state(FULL, int(real_batch.window_offsets[-1]), seed=SEED + 3)
    out = run(
        module,
        make_state(FULL, int(real_batch.legal_offsets[-1]), seed=SEED),
        real_batch.legal_offsets,
        make_latents(FULL, real_batch.position_count, seed=SEED),
        windows=windows,
        window_status=real_batch.window_status,
    )
    logits = out.aux[WINDOW_FATE_HEAD]
    mask = out.aux[WINDOW_FATE_HEAD + MASK_SUFFIX]
    assert logits.shape == (windows.inv.shape[0], WINDOW_FATE_CLASSES)
    live = (real_batch.window_status == OWN_LIVE) | (
        real_batch.window_status == OPP_LIVE
    )
    assert torch.equal(mask, live)
    # These plies carry mixed windows, so the mask actually removes rows.
    assert bool(mask.any()) and not bool(mask.all())


def test_the_window_fate_head_needs_the_windows_it_reads(actions, offsets, latents):
    module = ActionHeads(
        replace(FULL, enable_window_fate_head=True), window_fate_weight=0.1
    )
    with pytest.raises(ValueError, match="window"):
        run(module, actions, offsets, latents)


def test_a_window_fate_weight_without_the_flag_is_refused():
    with pytest.raises(ValueError, match="enable_window_fate_head"):
        ActionHeads(FULL, window_fate_weight=0.1)
    with pytest.raises(ValueError, match="absent"):
        ActionHeads(replace(FULL, enable_window_fate_head=True), window_fate_weight=0.0)


# --------------------------------------------------------------------------
# The state value head (§23.3)


def test_the_state_value_head_is_absent_by_default_and_reported_separately(
    actions, offsets, latents
):
    """§23.3: not instantiated by default; its parameters on their own line."""
    torch.manual_seed(SEED)
    plain = ActionHeads(FULL)
    assert plain.state_value is None
    assert all(
        "state_value" not in group
        for group, _ in parameter_summary(plain, depth=1).groups
    )

    module = enliven(ActionHeads(replace(FULL, enable_state_value_head=True)))
    summary = parameter_summary(module, depth=1)
    reported = dict(summary.groups)["state_value"]
    assert reported == sum(p.numel() for p in module.state_value.parameters())
    assert summary.total == parameter_summary(plain, depth=1).total + reported

    out = run(module, actions, offsets, latents)
    assert out.state_value.shape == (len(COUNTS),)
    assert out.state_value.dtype is torch.float32
    assert float(out.state_value.detach().abs().max()) <= 1.0


def test_the_state_value_head_is_invariant_under_an_axis_permutation(
    actions, offsets, latents
):
    module = enliven(ActionHeads(replace(FULL, enable_state_value_head=True)))
    base = run(module, actions, offsets, latents)
    turned = run(
        module,
        actions.permute_axes(PERMUTATIONS[1]),
        offsets,
        LatentState(
            inv=latents.inv,
            axis=permute_axis_channels(latents.axis, PERMUTATIONS[1]),
        ),
    )
    assert torch.allclose(turned.state_value, base.state_value, atol=1e-6)


# --------------------------------------------------------------------------
# The KLENT seam (§2, §25)


def test_the_uncomposed_pair_is_the_forwards_own(heads, actions, offsets, latents):
    """§25: `logits` is what KLENT's fitter scores, and it is not a second path."""
    policy, critic = heads.logits(actions, legal_offsets=offsets, latents=latents)
    out = run(heads, actions, offsets, latents)
    assert torch.equal(policy, out.policy_logits)
    assert torch.equal(critic, out.critic_logits)


def test_a_bf16_autocast_forward_is_finite(actions, offsets, latents):
    """§37.11: the model runs under autocast and answers finite values."""
    torch.manual_seed(SEED)
    module = enliven(ActionHeads(FULL))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = run(module, actions, offsets, latents)
    assert torch.isfinite(out.policy_logits).all()
    assert out.q_value.dtype is torch.float32
    assert torch.isfinite(out.q_value).all()


def test_autocast_survives_a_half_present_latent_context():
    """§29 full_one_latent: invariant latents, no axis latents, axis channels.

    Only the invariant half of the latent context exists there, and a LayerScale
    gain is fp32, so under autocast one stream promotes and the other does not.
    `EquivariantState` requires the pair to share a dtype, so a module that let
    the halves diverge would raise rather than answer.
    """
    cfg = PRESETS["full_one_latent"]
    assert cfg.num_inv_latents and not cfg.num_axis_latents and cfg.d_axis
    torch.manual_seed(SEED)
    module = enliven(ActionHeads(cfg))
    offsets = make_offsets()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = run(
            module,
            make_state(cfg, int(offsets[-1]), seed=SEED),
            offsets,
            make_latents(cfg, len(COUNTS), seed=SEED),
        )
    assert torch.isfinite(out.q_value).all()


def test_the_heads_take_a_gradient(actions, offsets, latents):
    torch.manual_seed(SEED)
    module = enliven(ActionHeads(FULL))
    out = run(module, actions, offsets, latents)
    (out.policy_logits.sum() + out.critic_logits.sum()).backward()
    missing = [
        name
        for name, parameter in module.named_parameters()
        if parameter.grad is None or not torch.isfinite(parameter.grad).all()
    ]
    assert missing == []


def test_every_optional_head_is_read_by_the_forward_when_it_is_on(real_batch):
    """§32's rule, on the configuration §29 does not name.

    `test_act_numerics.py` holds every preset to "no parameter the forward does
    not read", and no preset turns an optional head on — so the arm where they
    exist is checked here instead. The loss sums every output the module emits,
    which is what the §24 training stage would do with its own weights.
    """
    cfg = replace(
        FULL,
        enable_action_aux_heads=True,
        enable_window_fate_head=True,
        enable_state_value_head=True,
    )
    torch.manual_seed(SEED)
    module = enliven(
        ActionHeads(
            cfg,
            aux_weights={"winning_partner_exists": 1.0, "winning_partner_count": 1.0},
            window_fate_weight=1.0,
        )
    )
    rows = int(real_batch.legal_offsets[-1])
    out = run(
        module,
        make_state(cfg, rows, seed=SEED),
        real_batch.legal_offsets,
        make_latents(cfg, real_batch.position_count, seed=SEED),
        phase_id=real_batch.phase_id,
        windows=make_state(cfg, int(real_batch.window_offsets[-1]), seed=SEED + 3),
        window_status=real_batch.window_status,
    )
    loss = out.policy_logits.sum() + out.critic_logits.sum() + out.state_value.sum()
    for name, logits in out.aux.items():
        if not name.endswith(MASK_SUFFIX):
            loss = loss + logits.sum()
    loss.backward()
    missing = [
        name
        for name, parameter in module.named_parameters()
        if parameter.grad is None or not torch.isfinite(parameter.grad).all()
    ]
    assert missing == []


def test_a_head_output_compares_by_identity(heads, actions, offsets, latents):
    left = run(heads, actions, offsets, latents)
    right = run(heads, actions, offsets, latents)
    assert left != right
    assert left == left
