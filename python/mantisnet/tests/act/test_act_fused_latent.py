"""Whole LatentPass fusion against independent and module references.

The independent attention oracle walks positions, latent slots, channels, and
rows in Python using only ``row_pos``.  It shares neither the implementation's
multi-range indexing nor its Triton kernels.  Running the original module with
those oracle attentions then holds the complete registered pass—including all
norms, projections, pools, residuals, self-mix, and AxisMix—against a spec-line
reference independent of the fusion's parameter indexing.

Float64 module parity and gradcheck exercise the recompute autograd callback.
CUDA tests additionally require successful compiled forward/backward launches,
all-gradient parity, and bitwise repeatability, so a cached eager fallback
cannot pass as a device run.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from types import SimpleNamespace

import pytest
import torch

from mantisnet.models.mantis_act import fused_latent as fused
from mantisnet.models.mantis_act import latent_attention as attention
from mantisnet.models.mantis_act import latents as eager
from mantisnet.models.mantis_act.config import MantisACTConfig
from mantisnet.models.mantis_act.equivariant import EquivariantState
from mantisnet.models.mantis_act.latent_attention import (
    latent_segments,
    row_positions,
)
from mantisnet.models.mantis_act.latents import LatentPass, LatentState, RaggedStream


SEED = 20260809
TOL = 2e-4
_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the compiled latent pass needs CUDA"
)


def cfg(d_inv=4, d_axis=2, heads=1, ffn_mult=1):
    return dataclasses.replace(
        MantisACTConfig(),
        d_inv=d_inv,
        d_axis=d_axis,
        d_rel=max(2, d_axis),
        num_heads=heads,
        ffn_mult=ffn_mult,
        state_blocks=1,
        action_blocks=1,
        policy_private_blocks=1,
        critic_private_blocks=1,
    )


def randomise(module, seed=SEED):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            value = torch.randn(
                parameter.shape, generator=generator, dtype=torch.float64
            )
            parameter.copy_((0.2 * value).to(parameter.dtype))
    return module


def state_case(
    configuration=None,
    *,
    device="cpu",
    dtype=torch.float64,
    cell_counts=(2, 3),
    window_counts=(1, 3),
):
    configuration = cfg() if configuration is None else configuration
    if len(cell_counts) != len(window_counts) or not cell_counts:
        raise ValueError("state families need matching nonempty position counts")
    module = randomise(
        LatentPass(
            configuration,
            num_inv=2,
            num_axis=1,
            entity_names=("cell", "window"),
        )
    ).to(device=device, dtype=dtype)
    cell_offsets = torch.tensor(
        (0, *itertools.accumulate(cell_counts)), device=device
    )
    window_offsets = torch.tensor(
        (0, *itertools.accumulate(window_counts)), device=device
    )
    n_cell = sum(cell_counts)
    n_window = sum(window_counts)
    positions = len(cell_counts)
    cell_pos = row_positions(cell_offsets, n_cell)
    window_pos = row_positions(window_offsets, n_window)
    segments = latent_segments(
        (cell_offsets, window_offsets), (cell_pos, window_pos)
    )
    generator = torch.Generator().manual_seed(SEED + 1)

    def draw(*shape):
        value = torch.randn(*shape, generator=generator, dtype=torch.float64)
        return value.to(device=device, dtype=dtype).requires_grad_(True)

    states = (
        draw(positions, 2, configuration.d_inv),
        draw(positions, 1, 3, configuration.d_axis),
        draw(n_cell, configuration.d_inv),
        draw(n_cell, 3, configuration.d_axis),
        draw(n_window, configuration.d_inv),
        draw(n_window, 3, configuration.d_axis),
    )
    plans = (cell_offsets, cell_pos, window_offsets, window_pos)
    return module, segments, plans, states


def action_case(
    configuration=None,
    *,
    device="cpu",
    dtype=torch.float64,
    action_counts=(3, 4),
):
    configuration = cfg() if configuration is None else configuration
    if not action_counts:
        raise ValueError("the action case needs at least one position")
    module = randomise(
        LatentPass(
            configuration,
            num_inv=2,
            num_axis=0,
            entity_names=("action",),
        ),
        SEED + 2,
    ).to(device=device, dtype=dtype)
    offsets = torch.tensor(
        (0, *itertools.accumulate(action_counts)), device=device
    )
    n_action = sum(action_counts)
    positions = len(action_counts)
    row_pos = row_positions(offsets, n_action)
    segments = latent_segments((offsets,), (row_pos,))
    generator = torch.Generator().manual_seed(SEED + 3)

    def draw(*shape):
        value = torch.randn(*shape, generator=generator, dtype=torch.float64)
        return value.to(device=device, dtype=dtype).requires_grad_(True)

    states = (
        draw(positions, 2, configuration.d_inv),
        draw(n_action, configuration.d_inv),
        draw(n_action, 3, configuration.d_axis),
    )
    return module, segments, (offsets, row_pos), states


def run_state(module, segments, plans, states):
    cell_offsets, cell_pos, window_offsets, window_pos = plans
    return fused.state_latent_pass(
        *states,
        segments=segments,
        cell_offsets=cell_offsets,
        cell_row_pos=cell_pos,
        window_offsets=window_offsets,
        window_row_pos=window_pos,
        params=fused.state_parameters(module),
        heads=module.cfg.num_heads,
        activation=module.cfg.activation,
        eps=fused.state_eps(module),
        dropout=module.cfg.dropout,
        axis_pool_mode=module.cfg.axis_pool_mode,
    )


def run_action(module, segments, plans, states):
    offsets, row_pos = plans
    return fused.action_latent_pass(
        *states,
        segments=segments,
        action_offsets=offsets,
        action_row_pos=row_pos,
        params=fused.action_parameters(module),
        heads=module.cfg.num_heads,
        activation=module.cfg.activation,
        eps=fused.action_eps(module),
        dropout=module.cfg.dropout,
        axis_pool_mode=module.cfg.axis_pool_mode,
    )


def forward_state(module, segments, plans, states):
    cell_offsets, cell_pos, window_offsets, window_pos = plans
    latent, entities = module(
        LatentState(states[0], states[1]),
        {
            "cell": RaggedStream(
                EquivariantState(states[2], states[3]), cell_offsets, cell_pos
            ),
            "window": RaggedStream(
                EquivariantState(states[4], states[5]), window_offsets, window_pos
            ),
        },
        segments=segments,
    )
    return (
        latent.inv,
        latent.axis,
        entities["cell"].state.inv,
        entities["cell"].state.axis,
        entities["window"].state.inv,
        entities["window"].state.axis,
    )


def module_state(module, segments, plans, states):
    """The literal module composition retained as the independent oracle."""
    cell_offsets, cell_pos, window_offsets, window_pos = plans
    entities = {
        "cell": RaggedStream(
            EquivariantState(states[2], states[3]), cell_offsets, cell_pos
        ),
        "window": RaggedStream(
            EquivariantState(states[4], states[5]), window_offsets, window_pos
        ),
    }
    latent = module.mix(
        module.read(LatentState(states[0], states[1]), entities, segments=segments)
    )
    entities = module.broadcast(latent, entities)
    return (
        latent.inv,
        latent.axis,
        entities["cell"].state.inv,
        entities["cell"].state.axis,
        entities["window"].state.inv,
        entities["window"].state.axis,
    )


def forward_action(module, segments, plans, states):
    offsets, row_pos = plans
    latent, entities = module(
        LatentState(states[0], None),
        {
            "action": RaggedStream(
                EquivariantState(states[1], states[2]), offsets, row_pos
            )
        },
        segments=segments,
    )
    return latent.inv, entities["action"].state.inv


def module_action(module, segments, plans, states):
    """The literal invariant-only action composition used as the oracle."""
    offsets, row_pos = plans
    entities = {
        "action": RaggedStream(
            EquivariantState(states[1], states[2]), offsets, row_pos
        )
    }
    latent = module.mix(
        module.read(LatentState(states[0], None), entities, segments=segments)
    )
    entities = module.broadcast(latent, entities)
    return latent.inv, entities["action"].state.inv


def oracle_read(q, k, v, segments):
    """Spec §17.2, walking ownership rather than the kernel's range plan."""
    positions, slots, channels, heads, head_dim = q.shape
    scale = 1.0 / math.sqrt(head_dim)
    output = torch.zeros(
        q.shape, dtype=torch.promote_types(q.dtype, torch.float32), device=q.device
    )
    owners = segments.row_pos.tolist()
    for position in range(positions):
        rows = [row for row, owner in enumerate(owners) if owner == position]
        if not rows:
            continue
        for slot in range(slots):
            for channel in range(channels):
                for head in range(heads):
                    scores = torch.stack(
                        [
                            (q[position, slot, channel, head].double()
                             * k[row, channel, head].double()).sum()
                            * scale
                            for row in rows
                        ]
                    )
                    weights = scores.softmax(0)
                    output[position, slot, channel, head] = sum(
                        weights[index] * v[row, channel, head].double()
                        for index, row in enumerate(rows)
                    )
    return output


def oracle_broadcast(q, k, v, node_pos, offsets):
    """Spec §17.4, one node and context row at a time."""
    del offsets
    rows, channels, heads, head_dim = q.shape
    context = int(k.shape[1])
    scale = 1.0 / math.sqrt(head_dim)
    output = torch.zeros(
        q.shape, dtype=torch.promote_types(q.dtype, torch.float32), device=q.device
    )
    for row in range(rows):
        position = int(node_pos[row])
        for channel in range(channels):
            for head in range(heads):
                scores = torch.stack(
                    [
                        (q[row, channel, head].double()
                         * k[position, source, channel, head].double()).sum()
                        * scale
                        for source in range(context)
                    ]
                )
                weights = scores.softmax(0)
                output[row, channel, head] = sum(
                    weights[source]
                    * v[position, source, channel, head].double()
                    for source in range(context)
                )
    return output


def assert_close_tuple(got, want, tolerance=1e-11):
    assert len(got) == len(want)
    for actual, expected in zip(got, want):
        torch.testing.assert_close(
            actual, expected, rtol=tolerance, atol=tolerance
        )


def relative(got, want):
    return float(
        (got - want).abs().max() / want.abs().max().clamp(min=1e-30)
    )


def gradient_relatives(got, want):
    """Compare gradients without dividing numerical zero by numerical zero.

    The latent key-projection biases are mathematically dead: adding one
    constant to every key in a softmax changes no output.  Horizontal q/k/v
    projection fusion can nevertheless reassociate round-off in those zero
    gradients.  Use the whole-model oracle's convention and normalise such
    tensors by the global gradient scale rather than their own noise floor.
    """
    global_scale = max(float(value.detach().abs().max()) for value in want)
    numerical_zero = 32 * torch.finfo(torch.float32).eps * global_scale
    result = []
    for actual, expected in zip(got, want):
        own_scale = float(expected.detach().abs().max())
        scale = global_scale if own_scale <= numerical_zero else own_scale
        difference = float((actual.detach() - expected.detach()).abs().max())
        result.append(difference / scale)
    return result


def test_dense_attention_names_its_fp32_softmax_and_reduction_dtype():
    generator = torch.Generator().manual_seed(SEED + 5)
    score = torch.randn(2, 3, 4, 2, generator=generator).bfloat16()
    value = torch.randn(2, 1, 4, 2, 3, generator=generator).bfloat16()
    got = fused._dense_attention(score, value, 2)
    weight = score.float().softmax(dim=2)
    want = (weight.unsqueeze(-1) * value.float()).sum(dim=2)
    assert got.dtype == torch.float32
    torch.testing.assert_close(got, want, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("variant", ("state", "action"))
def test_float64_forward_matches_the_independent_attention_oracle(
    monkeypatch, variant
):
    monkeypatch.setattr(eager, "latent_read", oracle_read)
    monkeypatch.setattr(eager, "latent_broadcast", oracle_broadcast)
    if variant == "state":
        module, segments, plans, states = state_case()
        got = run_state(module, segments, plans, states)
        want = module_state(module, segments, plans, states)
    else:
        module, segments, plans, states = action_case()
        got = run_action(module, segments, plans, states)
        want = module_action(module, segments, plans, states)
    assert_close_tuple(got, want)


@pytest.mark.parametrize("variant", ("state", "action"))
def test_float64_forward_and_every_gradient_match_original_modules(variant):
    if variant == "state":
        module, segments, plans, states = state_case()
        want = module_state(module, segments, plans, states)
        got = run_state(module, segments, plans, states)
    else:
        module, segments, plans, states = action_case()
        want = module_action(module, segments, plans, states)
        got = run_action(module, segments, plans, states)
    assert_close_tuple(got, want)
    generator = torch.Generator().manual_seed(SEED + 4)
    seeds = tuple(
        torch.randn(value.shape, generator=generator, dtype=torch.float64)
        for value in got
    )
    inputs = (*states, *module.parameters())
    want_loss = sum((value * seed).sum() for value, seed in zip(want, seeds))
    got_loss = sum((value * seed).sum() for value, seed in zip(got, seeds))
    want_grad = torch.autograd.grad(want_loss, inputs, retain_graph=True)
    got_grad = torch.autograd.grad(got_loss, inputs)
    assert_close_tuple(got_grad, want_grad, tolerance=2e-10)


@pytest.mark.parametrize("variant", ("state", "action"))
def test_registered_recompute_backward_passes_full_float64_gradcheck(variant):
    tiny = cfg(d_inv=2, d_axis=1, heads=1, ffn_mult=1)
    if variant == "state":
        module, segments, plans, states = state_case(tiny)
        parameters = fused.state_parameters(module)

        def function(*values):
            return run_state(module, segments, plans, values[:6])

    else:
        module, segments, plans, states = action_case(tiny)
        parameters = fused.action_parameters(module)

        def function(*values):
            return run_action(module, segments, plans, values[:3])

    # Replace the module parameters read by the wrapper with the gradcheck
    # values while retaining the public registered-op ABI.
    def checked(*values):
        state_count = 6 if variant == "state" else 3
        if variant == "state":
            cell_offsets, cell_pos, window_offsets, window_pos = plans
            return fused.state_latent_pass(
                *values[:state_count],
                segments=segments,
                cell_offsets=cell_offsets,
                cell_row_pos=cell_pos,
                window_offsets=window_offsets,
                window_row_pos=window_pos,
                params=values[state_count:],
                heads=tiny.num_heads,
                activation=tiny.activation,
                eps=fused.state_eps(module),
            )
        offsets, row_pos = plans
        return fused.action_latent_pass(
            *values[:state_count],
            segments=segments,
            action_offsets=offsets,
            action_row_pos=row_pos,
            params=values[state_count:],
            heads=tiny.num_heads,
            activation=tiny.activation,
            eps=fused.action_eps(module),
        )

    assert torch.autograd.gradcheck(
        checked,
        (*states, *parameters),
        eps=1e-6,
        atol=2e-6,
        rtol=2e-5,
        fast_mode=True,
    )


@pytest.mark.parametrize("variant", ("state", "action"))
def test_registered_backward_reenables_grad_for_recompute(monkeypatch, variant):
    """Autograd invokes custom-op callbacks with grad recording disabled.

    The latent pass must create fresh differentiable leaves and execute both its
    recomputed forward and ``autograd.grad`` inside ``torch.enable_grad()``.
    Exercise the registered callback (rather than calling its helper directly)
    so a future narrowing or removal of that context cannot be hidden by the
    ordinary eager forward's grad mode.
    """
    if variant == "state":
        module, segments, plans, states = state_case()
        output = run_state(module, segments, plans, states)
        state_count = 6
    else:
        module, segments, plans, states = action_case()
        output = run_action(module, segments, plans, states)
        state_count = 3

    original_factory = fused._forward_function
    observed = []

    def checked_factory(
        selected_variant,
        heads,
        activation,
        autocast_code,
        eps,
        architecture,
        use_compiled,
    ):
        assert use_compiled
        literal = original_factory(
            selected_variant,
            heads,
            activation,
            autocast_code,
            eps,
            architecture,
            False,
        )

        def checked_recompute(*values):
            assert torch.is_grad_enabled()
            plan_count = (
                fused._STATE_PLAN_COUNT
                if selected_variant == "state"
                else fused._ACTION_PLAN_COUNT
            )
            leaves = (
                *values[:state_count],
                *values[state_count + plan_count :],
            )
            assert all(value.is_leaf and value.requires_grad for value in leaves)
            observed.append(True)
            return literal(*values)

        return checked_recompute

    # The forward above took the ordinary CPU/reference route.  Select the
    # eligible recompute route only after its autograd context has been saved,
    # and replace compilation with an observing wrapper around the same literal
    # math so this boundary regression remains CPU-only and deterministic.
    monkeypatch.setattr(fused, "_supported", lambda *args, **kwargs: True)
    monkeypatch.setattr(fused, "_forward_function", checked_factory)
    loss = sum(value.square().sum() for value in output)
    with torch.no_grad():
        gradients = torch.autograd.grad(
            loss,
            (*states, *module.parameters()),
        )

    assert observed == [True]
    assert all(torch.isfinite(value).all() for value in gradients)


@pytest.mark.parametrize("permutation", tuple(itertools.permutations(range(3))))
def test_state_pass_obeys_the_axis_permutation_law(permutation):
    module, segments, plans, states = state_case()
    base = run_state(module, segments, plans, states)
    moved_states = (
        states[0],
        states[1][:, :, permutation, :],
        states[2],
        states[3][:, permutation, :],
        states[4],
        states[5][:, permutation, :],
    )
    moved = run_state(module, segments, plans, moved_states)
    for index in (0, 2, 4):
        torch.testing.assert_close(moved[index], base[index], rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(
        moved[1], base[1][:, :, permutation, :], rtol=1e-11, atol=1e-11
    )
    for index in (3, 5):
        torch.testing.assert_close(
            moved[index], base[index][:, permutation, :], rtol=1e-11, atol=1e-11
        )


def test_action_invariant_outputs_ignore_axis_channel_order():
    module, segments, plans, states = action_case()
    base = run_action(module, segments, plans, states)
    moved = run_action(
        module,
        segments,
        plans,
        (states[0], states[1], states[2][:, (2, 0, 1), :]),
    )
    assert_close_tuple(moved, base)


def test_support_predicates_keep_ablations_on_the_eager_path():
    state, *_ = state_case()
    action, *_ = action_case()
    assert fused.supports_state_pass(state)
    assert fused.supports_action_pass(action)

    mean = dataclasses.replace(cfg(), axis_pool_mode="mean")
    mean_state = LatentPass(
        mean, num_inv=2, num_axis=1, entity_names=("cell", "window")
    )
    mean_action = LatentPass(
        mean, num_inv=2, num_axis=0, entity_names=("action",)
    )
    assert not fused.supports_state_pass(mean_state)
    assert not fused.supports_action_pass(mean_action)


def test_support_predicates_reject_mutated_dropout_and_pool_modules():
    state, *_ = state_case()
    action, *_ = action_case()

    state.drop.p = 0.5
    action.drop.p = 0.5
    assert not fused.supports_state_pass(state)
    assert not fused.supports_action_pass(action)

    state, *_ = state_case()
    state.axis_mix.drop.p = 0.5
    assert not fused.supports_state_pass(state)

    state, *_ = state_case()
    state.pool_src_axis.mode = "mean"
    assert not fused.supports_state_pass(state)

    action, *_ = action_case()
    action.pool_src_axis.mode = "mean"
    assert not fused.supports_action_pass(action)


def test_support_predicates_reject_a_parameter_preserving_forward_wrapper():
    state, *_ = state_case()
    state.q_read_inv = torch.nn.Sequential(state.q_read_inv)
    assert len(tuple(state.parameters())) == fused._STATE_COUNT
    assert not fused.supports_state_pass(state)


@pytest.mark.parametrize("variant", ("state", "action"))
def test_latent_pass_forward_dispatches_the_default_precomputed_plan(
    monkeypatch, variant
):
    """The model-facing module seam must actually enter the registered op."""
    if variant == "state":
        module, segments, plans, states = state_case()
        target = "state_latent_pass"
        forward = forward_state
        reference = module_state
    else:
        module, segments, plans, states = action_case()
        target = "action_latent_pass"
        forward = forward_action
        reference = module_action

    calls = []
    original = getattr(fused, target)

    def observed(*args, **kwargs):
        calls.append(kwargs["segments"])
        return original(*args, **kwargs)

    monkeypatch.setattr(fused, target, observed)
    got = forward(module, segments, plans, states)
    want = reference(module, segments, plans, states)
    assert calls == [segments]
    assert_close_tuple(got, want)


@pytest.mark.parametrize("variant", ("state", "action"))
@pytest.mark.parametrize("reason", ("no_plan", "mean_pool_ablation"))
def test_latent_pass_forward_keeps_unsupported_calls_eager(
    monkeypatch, variant, reason
):
    """Lazy callers and non-default pools never enter the fused ABI."""
    configuration = cfg()
    if reason == "mean_pool_ablation":
        configuration = dataclasses.replace(configuration, axis_pool_mode="mean")

    if variant == "state":
        module, segments, plans, states = state_case(configuration)
        forward = forward_state
        reference = module_state
    else:
        module, segments, plans, states = action_case(configuration)
        forward = forward_action
        reference = module_action

    def fused_was_not_expected(*args, **kwargs):
        raise AssertionError("unsupported LatentPass entered the fused path")

    monkeypatch.setattr(fused, "state_latent_pass", fused_was_not_expected)
    monkeypatch.setattr(fused, "action_latent_pass", fused_was_not_expected)
    supplied = None if reason == "no_plan" else segments
    got = forward(module, supplied, plans, states)
    want = reference(module, segments, plans, states)
    assert_close_tuple(got, want)


@pytest.mark.parametrize("variant", ("state", "action"))
def test_registered_schema_fake_autograd_and_dynamic_aot(variant):
    if variant == "state":
        module, segments, plans, states = state_case()
        cell_offsets, cell_pos, window_offsets, window_pos = plans
        arguments = (
            *states,
            fused._plan_tensors(
                segments,
                ((cell_offsets, cell_pos), (window_offsets, window_pos)),
            ),
            list(fused.state_parameters(module)),
            module.cfg.num_heads,
            module.cfg.activation,
            0,
            list(fused.state_eps(module)),
        )
        operator = fused._state_op
    else:
        module, segments, plans, states = action_case()
        offsets, row_pos = plans
        arguments = (
            *states,
            fused._plan_tensors(segments, ((offsets, row_pos),)),
            list(fused.action_parameters(module)),
            module.cfg.num_heads,
            module.cfg.activation,
            0,
            list(fused.action_eps(module)),
        )
        operator = fused._action_op
    assert torch.library.opcheck(operator, arguments) == {
        "test_schema": "SUCCESS",
        "test_autograd_registration": "SUCCESS",
        "test_faketensor": "SUCCESS",
        "test_aot_dispatch_dynamic": "SUCCESS",
    }


@pytest.mark.parametrize("variant", ("state", "action"))
def test_dynamic_signature_erases_regular_ragged_sizes_but_separates_singletons(
    variant,
):
    configuration = cfg()
    if variant == "state":
        cases = (
            state_case(configuration, cell_counts=(2, 3), window_counts=(1, 3)),
            state_case(
                configuration,
                cell_counts=(4, 2, 3),
                window_counts=(2, 4, 1),
            ),
        )
    else:
        cases = (
            action_case(configuration, action_counts=(3, 4)),
            action_case(configuration, action_counts=(2, 5, 3)),
        )

    signatures = []
    for module, segments, plans, states in cases:
        families = tuple(zip(plans[::2], plans[1::2]))
        plan_tensors = fused._plan_tensors(segments, families)
        parameters = (
            fused.state_parameters(module)
            if variant == "state"
            else fused.action_parameters(module)
        )
        signatures.append(
            fused._architecture_signature(states, tuple(plan_tensors), parameters)
        )
    assert signatures[0] == signatures[1]

    # A leading singleton is irreducibly static in Dynamo and must occupy a
    # bounded, explicitly separate cache bucket rather than surprise-recompile
    # a regular-shape entry.
    if variant == "state":
        module, segments, plans, states = state_case(
            configuration, cell_counts=(1,), window_counts=(1,)
        )
        parameters = fused.state_parameters(module)
    else:
        module, segments, plans, states = action_case(
            configuration, action_counts=(1,)
        )
        parameters = fused.action_parameters(module)
    plan_tensors = fused._plan_tensors(
        segments, tuple(zip(plans[::2], plans[1::2]))
    )
    singleton = fused._architecture_signature(
        states, tuple(plan_tensors), parameters
    )
    assert singleton != signatures[0]


def test_static_signatures_get_independent_dynamo_code_objects(monkeypatch):
    captured = []

    def capture(function, **kwargs):
        captured.append((function, kwargs))
        return function

    fused.clear_compile_caches()
    monkeypatch.setattr(torch, "compile", capture)
    settings = ("state", 2, "silu", 0, (1e-5,), ("architecture-a",))
    try:
        first = fused._forward_function(*settings, True)
        again = fused._forward_function(*settings, True)
        second = fused._forward_function(
            "state", 2, "silu", 0, (1e-5,), ("architecture-b",), True
        )
        assert first is again
        assert first.__code__ is not second.__code__
        assert first.__name__ != second.__name__
        assert len(captured) == 2
        assert all(options["dynamic"] for _function, options in captured)
        assert all(options["fullgraph"] for _function, options in captured)
    finally:
        # Do not leave the identity compiler's functions in the process cache
        # for later CUDA tests after monkeypatch restores the real compiler.
        fused.clear_compile_caches()


def test_eligible_compile_failures_are_named_and_never_cache_an_eager_fallback(
    monkeypatch,
):
    module, segments, plans, states = action_case()
    parameters = fused.action_parameters(module)
    plan_tensors = tuple(fused._plan_tensors(segments, (plans,)))
    fused.clear_failure_caches()
    monkeypatch.setattr(fused, "_supported", lambda *args, **kwargs: True)

    def failed_forward(*args, **kwargs):
        raise RuntimeError("synthetic compiler failure")

    monkeypatch.setattr(fused, "_launch_forward", failed_forward)
    with pytest.raises(
        fused.FusedLatentCompileError,
        match="refusing to silently de-fuse the ACT trunk",
    ) as caught:
        fused._dispatch_forward(
            "action",
            states,
            plan_tensors,
            parameters,
            module.cfg.num_heads,
            module.cfg.activation,
            0,
            fused.action_eps(module),
        )
    assert "synthetic compiler failure" in str(caught.value.__cause__)
    assert len(fused._FAILED_FORWARD_SHAPES) == 1
    assert "synthetic compiler failure" in next(
        iter(fused._FAILED_FORWARD_SHAPES.values())
    )

    context = SimpleNamespace(
        saved_tensors=(*states, *plan_tensors, *parameters),
        state_count=3,
        plan_count=len(plan_tensors),
        settings=(
            module.cfg.num_heads,
            module.cfg.activation,
            0,
            list(fused.action_eps(module)),
        ),
        variant="action",
    )

    def failed_factory(*args, **kwargs):
        def fail(*tensors):
            raise RuntimeError("synthetic backward compiler failure")

        return fail

    monkeypatch.setattr(fused, "_forward_function", failed_factory)
    with pytest.raises(
        fused.FusedLatentCompileError,
        match="refusing to silently de-fuse the ACT trunk",
    ) as caught:
        fused._recompute_gradients(
            context,
            tuple(torch.ones_like(value) for value in states[:2]),
        )
    assert "synthetic backward compiler failure" in str(caught.value.__cause__)
    assert len(fused._FAILED_BACKWARD_SHAPES) == 1
    assert "synthetic backward compiler failure" in next(
        iter(fused._FAILED_BACKWARD_SHAPES.values())
    )


def cuda_cfg():
    return cfg(d_inv=8, d_axis=4, heads=2, ffn_mult=2)


@pytest.mark.parametrize("variant", ("state", "action"))
@_CUDA
def test_cuda_one_dynamic_graph_serves_many_regular_ragged_shapes(variant):
    """No position/entity/plan length may trigger a second Dynamo graph."""
    configuration = cuda_cfg()
    if variant == "state":
        shapes = (
            ((2, 3), (1, 3)),
            ((3, 2, 4), (2, 3, 1)),
            ((2, 2, 2, 2), (1, 2, 3, 2)),
            ((4, 3, 2, 5, 2), (2, 1, 4, 2, 3)),
        )

        def build(shape):
            return state_case(
                configuration,
                device="cuda",
                dtype=torch.float32,
                cell_counts=shape[0],
                window_counts=shape[1],
            )

        run = run_state
    else:
        shapes = ((3, 4), (2, 5, 3), (4, 2, 3, 5), (2, 3, 4, 2, 5))

        def build(shape):
            return action_case(
                configuration,
                device="cuda",
                dtype=torch.float32,
                action_counts=shape,
            )

        run = run_action

    fused.clear_compile_caches()
    fused.clear_failure_caches()
    fused.reset_launch_stats()

    def exercise(shape):
        module, segments, plans, states = build(shape)
        output = run(module, segments, plans, states)
        torch.autograd.grad(
            sum(value.square().sum() for value in output),
            (*states, *module.parameters()),
        )

    # Establish the graph before turning every subsequent guard miss into an
    # immediate failure.  This catches shape recompiles without changing either
    # global recompile limit.
    exercise(shapes[0])
    with torch._dynamo.config.patch(error_on_recompile=True):
        for shape in shapes[1:]:
            exercise(shape)

    stats = fused.launch_stats()
    assert stats[f"{variant}_forward_eligible"] == len(shapes)
    assert stats[f"{variant}_forward_launched"] == len(shapes)
    assert stats[f"{variant}_backward_eligible"] == len(shapes)
    assert stats[f"{variant}_backward_launched"] == len(shapes)
    assert not fused._FAILED_FORWARD_SHAPES
    assert not fused._FAILED_BACKWARD_SHAPES


@pytest.mark.parametrize("variant", ("state", "action"))
@_CUDA
def test_cuda_compiled_forward_backward_match_and_really_launch(variant):
    if variant == "state":
        module, segments, plans, states = state_case(
            cuda_cfg(), device="cuda", dtype=torch.float32
        )
        reference = module_state(module, segments, plans, states)
        run = run_state
    else:
        module, segments, plans, states = action_case(
            cuda_cfg(), device="cuda", dtype=torch.float32
        )
        reference = module_action(module, segments, plans, states)
        run = run_action

    fused.clear_failure_caches()
    fused.reset_launch_stats()
    attention._FAILED_SHAPES.clear()
    attention._FAILED_BACKWARD_SHAPES.clear()
    got = run(module, segments, plans, states)
    seeds = tuple(torch.randn_like(value) for value in got)
    got_loss = sum((value * seed).sum() for value, seed in zip(got, seeds))
    ref_loss = sum(
        (value * seed).sum() for value, seed in zip(reference, seeds)
    )
    inputs = (*states, *module.parameters())
    reference_grad = torch.autograd.grad(ref_loss, inputs, retain_graph=True)
    got_grad = torch.autograd.grad(got_loss, inputs)
    for index, (actual, expected) in enumerate(zip(got, reference)):
        assert relative(actual, expected) <= TOL, f"output[{index}]"
    gradient_names = (
        *(f"state[{index}]" for index in range(len(states))),
        *(name for name, _parameter in module.named_parameters()),
    )
    for name, error in zip(
        gradient_names, gradient_relatives(got_grad, reference_grad)
    ):
        assert error <= TOL, f"{name}: {error:.6e}"

    stats = fused.launch_stats()
    assert stats[f"{variant}_forward_eligible"] == 1
    assert stats[f"{variant}_forward_launched"] == 1
    assert stats[f"{variant}_backward_eligible"] == 1
    assert stats[f"{variant}_backward_launched"] == 1
    assert not fused._FAILED_FORWARD_SHAPES
    assert not fused._FAILED_BACKWARD_SHAPES
    assert not attention._FAILED_SHAPES
    assert not attention._FAILED_BACKWARD_SHAPES


@pytest.mark.parametrize("variant", ("state", "action"))
@_CUDA
def test_cuda_bf16_is_reported_against_fp32_anchor_and_really_launches(
    variant, record_property
):
    """Document AMP drift without using bf16 as its own parity oracle.

    The §36 gate is fp32 fused versus the literal module.  Under bf16 autocast,
    horizontally concatenating q/k/v projections deliberately reassociates
    their backward, and mathematically-zero key-bias gradients make a per-
    tensor relative metric especially meaningless.  The AMP contract is a
    fused run reported against its fused fp32 anchor, finiteness, and proof
    that both forward and backward really launched.
    """
    if variant == "state":
        module, segments, plans, states = state_case(
            cuda_cfg(), device="cuda", dtype=torch.float32
        )
        run = run_state
    else:
        module, segments, plans, states = action_case(
            cuda_cfg(), device="cuda", dtype=torch.float32
        )
        run = run_action

    fused.clear_failure_caches()
    fused.reset_launch_stats()
    attention._FAILED_SHAPES.clear()
    attention._FAILED_BACKWARD_SHAPES.clear()
    anchor = run(module, segments, plans, states)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        amp = run(module, segments, plans, states)
    seeds = tuple(torch.randn_like(value) for value in anchor)
    inputs = (*states, *module.parameters())
    anchor_grad = torch.autograd.grad(
        sum((value * seed).sum() for value, seed in zip(anchor, seeds)),
        inputs,
    )
    amp_grad = torch.autograd.grad(
        sum((value * seed).sum() for value, seed in zip(amp, seeds)),
        inputs,
    )

    assert all(torch.isfinite(value).all() for value in (*anchor, *anchor_grad))
    assert all(torch.isfinite(value).all() for value in (*amp, *amp_grad))
    output_errors = [
        relative(actual, expected) for actual, expected in zip(amp, anchor)
    ]
    gradient_errors = gradient_relatives(amp_grad, anchor_grad)
    gradient_names = (
        *(f"state[{index}]" for index in range(len(states))),
        *(name for name, _parameter in module.named_parameters()),
    )
    worst_output = max(enumerate(output_errors), key=lambda item: item[1])
    worst_gradient = max(
        zip(gradient_names, gradient_errors), key=lambda item: item[1]
    )
    record_property("bf16_fp32_worst_output", worst_output[1])
    record_property("bf16_fp32_worst_output_index", worst_output[0])
    record_property("bf16_fp32_worst_gradient", worst_gradient[1])
    record_property("bf16_fp32_worst_gradient_name", worst_gradient[0])

    stats = fused.launch_stats()
    assert stats[f"{variant}_forward_eligible"] == 2
    assert stats[f"{variant}_forward_launched"] == 2
    assert stats[f"{variant}_backward_eligible"] == 2
    assert stats[f"{variant}_backward_launched"] == 2
    assert not fused._FAILED_FORWARD_SHAPES
    assert not fused._FAILED_BACKWARD_SHAPES
    assert not attention._FAILED_SHAPES
    assert not attention._FAILED_BACKWARD_SHAPES


@pytest.mark.parametrize("variant", ("state", "action"))
@_CUDA
def test_compiled_fp32_outputs_and_all_gradients_are_bitwise_deterministic(variant):
    if variant == "state":
        module, segments, plans, states = state_case(
            cuda_cfg(), device="cuda", dtype=torch.float32
        )
        run = run_state
    else:
        module, segments, plans, states = action_case(
            cuda_cfg(), device="cuda", dtype=torch.float32
        )
        run = run_action
    originals = tuple(value.detach().clone() for value in states)
    fused.clear_failure_caches()
    fused.reset_launch_stats()
    attention._FAILED_SHAPES.clear()
    attention._FAILED_BACKWARD_SHAPES.clear()

    def once():
        leaves = tuple(
            value.detach().clone().requires_grad_(True) for value in originals
        )
        output = run(module, segments, plans, leaves)
        gradients = torch.autograd.grad(
            sum(value.sum() for value in output), (*leaves, *module.parameters())
        )
        return (*output, *gradients)

    first = once()
    for other in (once(), once()):
        for left, right in zip(first, other):
            assert torch.equal(left, right)

    stats = fused.launch_stats()
    assert stats[f"{variant}_forward_eligible"] == 3
    assert stats[f"{variant}_forward_launched"] == 3
    assert stats[f"{variant}_backward_eligible"] == 3
    assert stats[f"{variant}_backward_launched"] == 3
    assert not fused._FAILED_FORWARD_SHAPES
    assert not fused._FAILED_BACKWARD_SHAPES
    assert not attention._FAILED_SHAPES
    assert not attention._FAILED_BACKWARD_SHAPES
