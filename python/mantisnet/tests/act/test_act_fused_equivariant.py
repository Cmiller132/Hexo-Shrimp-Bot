"""Whole-stage AxisMix/FFN/FiLM parity, gradients, symmetry, and dispatch.

The float64 oracle below is written directly from §§12.4, 13.2, and 18.  It
uses explicit matrix products and an explicit LayerNorm rather than either the
registered operator or its pure torch implementation.  CPU gradcheck validates
the recompute VJP; CUDA then holds the separately compiled forward and backward
against the original modules and requires successful launch accounting, so a
cached eager fallback cannot masquerade as a device run.
"""

from __future__ import annotations

import dataclasses
import itertools

import pytest
import torch

from mantisnet.models.mantis_act.config import MantisACTConfig
from mantisnet.models.mantis_act.equivariant import (
    AxisMix,
    EquivariantFFN,
    EquivariantState,
    PhaseFiLM,
    run_equivariant_stage,
)
from mantisnet.models.mantis_act import fused_equivariant as fused


SEED = 20260809
TOL = 2e-4
_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the compiled stage needs CUDA"
)


def cfg(
    d_inv: int = 8,
    d_axis: int = 4,
    d_rel: int = 4,
    num_heads: int = 2,
    ffn_mult: int = 2,
) -> MantisACTConfig:
    return dataclasses.replace(
        MantisACTConfig(),
        d_inv=d_inv,
        d_axis=d_axis,
        d_rel=d_rel,
        num_heads=num_heads,
        ffn_mult=ffn_mult,
    )


def randomise(module: torch.nn.Module, seed: int) -> torch.nn.Module:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            value = torch.randn(
                parameter.shape, generator=generator, dtype=torch.float64
            )
            parameter.copy_((0.25 * value).to(parameter.dtype))
    return module


def mix_parameters(module: AxisMix) -> tuple[torch.Tensor, ...]:
    return (
        module.norm.inv.weight,
        module.norm.inv.bias,
        module.norm.axis.weight,
        module.norm.axis.bias,
        module.inv_to_axis.weight,
        module.inv_to_axis.bias,
        module.mlp_axis[0].weight,
        module.mlp_axis[0].bias,
        module.mlp_axis[2].weight,
        module.mlp_axis[2].bias,
        module.phi_axis[0].weight,
        module.phi_axis[0].bias,
        module.mlp_inv[0].weight,
        module.mlp_inv[0].bias,
        module.mlp_inv[2].weight,
        module.mlp_inv[2].bias,
        module.residual.inv.gamma,
        module.residual.axis.gamma,
    )


def ffn_parameters(module: EquivariantFFN) -> tuple[torch.Tensor, ...]:
    return (
        module.norm.inv.weight,
        module.norm.inv.bias,
        module.norm.axis.weight,
        module.norm.axis.bias,
        module.inv[0].weight,
        module.inv[0].bias,
        module.inv[2].weight,
        module.inv[2].bias,
        module.axis[0].weight,
        module.axis[0].bias,
        module.axis[2].weight,
        module.axis[2].bias,
        module.residual.inv.gamma,
        module.residual.axis.gamma,
    )


def film_parameters(module: PhaseFiLM) -> tuple[torch.Tensor, ...]:
    return (
        module.embed.weight,
        module.phase_mlp[0].weight,
        module.phase_mlp[0].bias,
        module.to_inv.weight,
        module.to_inv.bias,
        module.to_axis.weight,
        module.to_axis.bias,
    )


def modules(dtype=torch.float64, device="cpu", configuration=None):
    configuration = cfg() if configuration is None else configuration
    mix = randomise(AxisMix(configuration), SEED).to(device=device, dtype=dtype)
    ffn = randomise(EquivariantFFN(configuration), SEED + 1).to(
        device=device, dtype=dtype
    )
    film = randomise(PhaseFiLM(configuration), SEED + 2).to(
        device=device, dtype=dtype
    )
    return mix, ffn, film


def state(dtype=torch.float64, device="cpu", rows=3, d_inv=8, d_axis=4):
    generator = torch.Generator().manual_seed(SEED + 3)
    inv = torch.randn(rows, d_inv, generator=generator, dtype=torch.float64)
    axis = torch.randn(
        rows, 3, d_axis, generator=generator, dtype=torch.float64
    )
    return (
        inv.to(device=device, dtype=dtype).requires_grad_(True),
        axis.to(device=device, dtype=dtype).requires_grad_(True),
    )


def run_fused(inv, axis, mix, ffn, film, phase):
    return fused.equivariant_stage(
        inv,
        axis,
        mix_parameters(mix),
        ffn_parameters(ffn),
        phase_id=phase,
        phase_row=film.phase_row,
        film=film_parameters(film),
        activation="silu",
        mix_eps=(mix.norm.inv.eps, mix.norm.axis.eps),
        ffn_eps=(ffn.norm.inv.eps, ffn.norm.axis.eps),
    )


def stage_tensors(inv, axis, mix, ffn, film, phase):
    return (
        inv,
        axis,
        *mix_parameters(mix),
        *ffn_parameters(ffn),
        phase,
        film.phase_row,
        *film_parameters(film),
    )


def silu(value):
    return value * torch.sigmoid(value)


def norm(value, weight, bias, eps):
    mean = value.sum(dim=-1, keepdim=True) / value.shape[-1]
    centred = value - mean
    variance = centred.square().sum(dim=-1, keepdim=True) / value.shape[-1]
    return centred * torch.rsqrt(variance + eps) * weight + bias


def oracle(inv, axis, mix, ffn, film, phase):
    """Independent float64 transcription, using no functional layer helpers."""
    m = mix_parameters(mix)
    q = ffn_parameters(ffn)
    z_inv = norm(inv, m[0], m[1], mix.norm.inv.eps)
    u = norm(axis, m[2], m[3], mix.norm.axis.eps)
    other = (u.sum(dim=1, keepdim=True) - u) / 2
    context = (z_inv @ m[4].T + m[5])[:, None, :].expand_as(u)
    axis_in = torch.cat((u, other, context), dim=-1)
    delta_axis = silu(axis_in @ m[6].T + m[7]) @ m[8].T + m[9]
    summary = silu(u @ m[10].T + m[11]).sum(dim=1) / 3
    inv_in = torch.cat((z_inv, summary), dim=-1)
    delta_inv = silu(inv_in @ m[12].T + m[13]) @ m[14].T + m[15]
    inv = inv + m[16] * delta_inv
    axis = axis + m[17] * delta_axis

    z_inv = norm(inv, q[0], q[1], ffn.norm.inv.eps)
    z_axis = norm(axis, q[2], q[3], ffn.norm.axis.eps)
    inv = inv + q[12] * (silu(z_inv @ q[4].T + q[5]) @ q[6].T + q[7])
    axis = axis + q[13] * (
        silu(z_axis @ q[8].T + q[9]) @ q[10].T + q[11]
    )

    p = film_parameters(film)
    rows = torch.empty_like(phase)
    for entity in range(phase.numel()):
        rows[entity] = film.phase_row[phase[entity]]
    code = silu(p[0] @ p[1].T + p[2])
    inv_table = code @ p[3].T + p[4]
    axis_table = code @ p[5].T + p[6]
    inv_scale, inv_bias = inv_table[rows].chunk(2, dim=-1)
    axis_scale, axis_bias = axis_table[rows].chunk(2, dim=-1)
    return (
        (1 + inv_scale) * inv + inv_bias,
        (1 + axis_scale[:, None, :]) * axis + axis_bias[:, None, :],
    )


def relative(got, want) -> float:
    return float((got - want).abs().max() / want.abs().max().clamp(min=1e-30))


def test_float64_forward_matches_the_independent_spec_oracle():
    mix, ffn, film = modules()
    inv, axis = state()
    phase = torch.tensor([0, 1, 2])
    got = run_fused(inv, axis, mix, ffn, film, phase)
    want = oracle(inv, axis, mix, ffn, film, phase)
    torch.testing.assert_close(got[0], want[0], rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(got[1], want[1], rtol=1e-11, atol=1e-11)


def test_float64_forward_matches_the_original_module_composition():
    mix, ffn, film = modules()
    inv, axis = state()
    phase = torch.tensor([0, 1, 2])
    got = run_fused(inv, axis, mix, ffn, film, phase)
    want = film(ffn(mix(EquivariantState(inv, axis))), phase)
    torch.testing.assert_close(got[0], want.inv, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(got[1], want.axis, rtol=1e-11, atol=1e-11)


def test_registered_schema_fake_tensor_and_autograd_contract():
    mix, ffn, film = modules()
    inv, axis = state()
    phase = torch.tensor([0, 1, 2])
    torch.library.opcheck(
        fused._stage_op,
        (
            inv,
            axis,
            *mix_parameters(mix),
            *ffn_parameters(ffn),
            phase,
            film.phase_row,
            *film_parameters(film),
            "silu",
            True,
            0,
            float(mix.norm.inv.eps),
            float(mix.norm.axis.eps),
            float(ffn.norm.inv.eps),
            float(ffn.norm.axis.eps),
        ),
    )


def test_registered_no_film_schema_fake_tensor_and_autograd_contract():
    mix, ffn, _film = modules()
    inv, axis = state()
    empty_long = torch.empty(0, dtype=torch.long)
    empty_float = inv.new_empty(0)
    torch.library.opcheck(
        fused._stage_op,
        (
            inv,
            axis,
            *mix_parameters(mix),
            *ffn_parameters(ffn),
            empty_long,
            empty_long,
            *(empty_float for _ in range(7)),
            "silu",
            False,
            0,
            float(mix.norm.inv.eps),
            float(mix.norm.axis.eps),
            float(ffn.norm.inv.eps),
            float(ffn.norm.axis.eps),
        ),
    )


def test_dispatch_follows_the_dropout_submodule_mode(monkeypatch):
    mix, ffn, _film = modules()
    inv, axis = state()
    mix.eval()
    ffn.eval()
    mix.drop.p = 0.5
    mix.drop.train()

    def forbidden(*args, **kwargs):
        raise AssertionError("active dropout must keep the literal path")

    monkeypatch.setattr(fused, "equivariant_stage", forbidden)
    result = run_equivariant_stage(EquivariantState(inv, axis), mix, ffn)
    assert result.inv.shape == inv.shape
    assert result.axis.shape == axis.shape


def test_inactive_dropout_can_fuse_even_when_its_parent_is_training(monkeypatch):
    mix, ffn, _film = modules()
    inv, axis = state()
    mix.train()
    ffn.train()
    mix.drop.p = 0.5
    ffn.drop.p = 0.5
    mix.drop.eval()
    ffn.drop.eval()
    called = False

    def identity(got_inv, got_axis, *args, **kwargs):
        nonlocal called
        called = True
        return got_inv, got_axis

    monkeypatch.setattr(fused, "equivariant_stage", identity)
    result = run_equivariant_stage(EquivariantState(inv, axis), mix, ffn)
    assert called
    assert result.inv is inv
    assert result.axis is axis


def test_the_registered_recompute_backward_passes_float64_gradcheck():
    # One row and minimal legal widths keep finite differences cheap enough for
    # the complete forty-one-input gradient; the production widths run below.
    tiny = cfg(d_inv=2, d_axis=1, d_rel=2, num_heads=1, ffn_mult=1)
    mix, ffn, film = modules(configuration=tiny)
    inv, axis = state(rows=1, d_inv=2, d_axis=1)
    phase = torch.tensor([2])
    differentiable = (
        inv,
        axis,
        *mix_parameters(mix),
        *ffn_parameters(ffn),
        *film_parameters(film),
    )

    def function(*values):
        split_mix = values[2:20]
        split_ffn = values[20:34]
        split_film = values[34:]
        return fused.equivariant_stage(
            values[0],
            values[1],
            split_mix,
            split_ffn,
            phase_id=phase,
            phase_row=film.phase_row,
            film=split_film,
        )

    assert torch.autograd.gradcheck(
        function, differentiable, eps=1e-6, atol=2e-7, rtol=2e-5
    )


@pytest.mark.parametrize("use_film", [False, True])
def test_registered_compiled_functional_vjp_survives_no_grad_callback(
    monkeypatch,
    use_film,
):
    """Compile the exact registered backward while its callback is no-grad.

    The original regression replaced ``torch.compile`` with identity and
    therefore could not reproduce Inductor lowering a recomputed forward as
    an inference graph.  ``aot_eager`` exercises the same AOTAutograd
    boundary on this CPU-only host.  The functional VJP is compiled as one
    unit and does not depend on recomputed outputs carrying ``grad_fn``.
    """
    mix, ffn, film = modules(dtype=torch.float32)
    inv, axis = state(dtype=torch.float32, rows=3)
    phase = torch.tensor([0, 1, 2])
    parameters = (*mix_parameters(mix), *ffn_parameters(ffn))
    if use_film:
        parameters = (*parameters, *film_parameters(film))

    def launch(state_inv, state_axis, state_phase):
        if use_film:
            return run_fused(
                state_inv,
                state_axis,
                mix,
                ffn,
                film,
                state_phase,
            )
        return fused.equivariant_stage(
            state_inv,
            state_axis,
            mix_parameters(mix),
            ffn_parameters(ffn),
            activation="silu",
            mix_eps=(mix.norm.inv.eps, mix.norm.axis.eps),
            ffn_eps=(ffn.norm.inv.eps, ffn.norm.axis.eps),
        )

    backward_grad_modes = []
    launch_backward = fused._launch_backward

    def record_backward_context(*args, **kwargs):
        backward_grad_modes.append(torch.is_grad_enabled())
        return launch_backward(*args, **kwargs)

    compile_function = torch.compile

    def compile_with_aot_eager(function, **kwargs):
        # emulate_precision_casts is an Inductor option, not an aot_eager
        # option.  Everything else (fullgraph and dynamic shapes) is retained.
        kwargs.pop("options", None)
        return compile_function(function, backend="aot_eager", **kwargs)

    monkeypatch.setattr(fused, "_supported", lambda *args: True)
    monkeypatch.setattr(torch, "compile", compile_with_aot_eager)
    monkeypatch.setattr(fused, "_launch_backward", record_backward_context)
    fused.clear_compile_caches()
    fused.clear_failure_caches()
    fused.reset_launch_stats()
    try:
        outputs = launch(inv, axis, phase)
        gradients = torch.autograd.grad(
            outputs[0].sum() + outputs[1].sum(),
            (inv, axis, *parameters),
        )
        # The first call warms one general-row forward and functional-VJP
        # graph.  A different entity count must reuse both immediately.
        other_inv, other_axis = state(dtype=torch.float32, rows=5)
        other_phase = torch.tensor([0, 1, 2, 1, 0])
        with torch._dynamo.config.patch(error_on_recompile=True):
            other_outputs = launch(other_inv, other_axis, other_phase)
            other_gradients = torch.autograd.grad(
                other_outputs[0].sum() + other_outputs[1].sum(),
                (other_inv, other_axis, *parameters),
            )
        assert backward_grad_modes == [False, False]
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert all(
            torch.isfinite(gradient).all() for gradient in other_gradients
        )
        assert fused.launch_stats() == {
            "forward_eligible": 2,
            "forward_launched": 2,
            "backward_eligible": 2,
            "backward_launched": 2,
        }
        assert not fused._FAILED_FORWARD_SHAPES
        assert not fused._FAILED_BACKWARD_SHAPES
    finally:
        fused.clear_compile_caches()
        fused.clear_failure_caches()
        fused.reset_launch_stats()


def test_outer_aot_graph_executes_the_opaque_registered_backward(monkeypatch):
    """AOTAutograd's FunctionalTensor branch must remain executable.

    While tracing an outer graph, the registered formula emits
    ``act_equivariant_stage_backward`` as an opaque node.  Its real kernel
    must execute the same functional VJP; retaining the old tape-based CUDA
    refusal here made an otherwise valid compiled model fail only at runtime.
    """
    mix, ffn, _film = modules(dtype=torch.float32)
    parameters = (*mix_parameters(mix), *ffn_parameters(ffn))
    eps = (
        float(mix.norm.inv.eps),
        float(mix.norm.axis.eps),
        float(ffn.norm.inv.eps),
        float(ffn.norm.axis.eps),
    )
    compile_function = torch.compile

    def compile_with_aot_eager(function, **kwargs):
        kwargs.pop("options", None)
        return compile_function(function, backend="aot_eager", **kwargs)

    def arguments(rows):
        inv, axis = state(dtype=torch.float32, rows=rows)
        empty_long = torch.empty(0, dtype=torch.long)
        empty_float = inv.new_empty(0)
        tensors = (
            inv,
            axis,
            *parameters,
            empty_long,
            empty_long,
            *(empty_float for _ in range(7)),
        )
        fused._mark_dynamic_rows(tensors, False)
        return tensors

    def outer(*tensors):
        return fused._stage_op(
            *tensors,
            "silu",
            False,
            0,
            *eps,
        )

    monkeypatch.setattr(fused, "_supported", lambda *args: True)
    monkeypatch.setattr(torch, "compile", compile_with_aot_eager)
    fused.clear_compile_caches()
    fused.clear_failure_caches()
    fused.reset_launch_stats()
    try:
        compiled = torch.compile(outer, fullgraph=True, dynamic=True)

        def once(rows):
            tensors = arguments(rows)
            outputs = compiled(*tensors)
            gradients = torch.autograd.grad(
                outputs[0].sum() + outputs[1].sum(),
                (tensors[0], tensors[1], *parameters),
            )
            assert all(torch.isfinite(value).all() for value in gradients)
            return outputs, gradients

        # One execution is sufficient to force AOT's opaque backward node to
        # run its real implementation.  Dynamic N=3 -> N=5 reuse of the inner
        # VJP is covered above; an outer aot_eager wrapper changes the
        # ADInplaceOrView TLS regime between trace and runtime and therefore
        # owns a separate finite inner-forward guard on CPU.
        once(3)

        assert fused.launch_stats() == {
            "forward_eligible": 1,
            "forward_launched": 1,
            "backward_eligible": 1,
            "backward_launched": 1,
        }
        assert not fused._FAILED_FORWARD_SHAPES
        assert not fused._FAILED_BACKWARD_SHAPES
    finally:
        fused.clear_compile_caches()
        fused.clear_failure_caches()
        fused.reset_launch_stats()


@pytest.mark.parametrize("permutation", tuple(itertools.permutations(range(3))))
def test_whole_stage_obeys_the_axis_permutation_law(permutation):
    mix, ffn, film = modules()
    inv, axis = state()
    phase = torch.tensor([0, 1, 2])
    base_inv, base_axis = run_fused(inv, axis, mix, ffn, film, phase)
    moved_inv, moved_axis = run_fused(
        inv, axis[:, permutation, :], mix, ffn, film, phase
    )
    torch.testing.assert_close(moved_inv, base_inv, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(
        moved_axis, base_axis[:, permutation, :], rtol=1e-11, atol=1e-11
    )


def test_optional_film_path_is_exactly_the_original_mix_then_ffn():
    mix, ffn, _film = modules()
    inv, axis = state()
    got = fused.equivariant_stage(
        inv, axis, mix_parameters(mix), ffn_parameters(ffn)
    )
    want = ffn(mix(EquivariantState(inv, axis)))
    torch.testing.assert_close(got[0], want.inv, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(got[1], want.axis, rtol=1e-11, atol=1e-11)


def test_eligible_compile_failure_is_named_and_never_silently_falls_back(
    monkeypatch,
):
    mix, ffn, film = modules(dtype=torch.float32)
    inv, axis = state(dtype=torch.float32)
    phase = torch.tensor([0, 1, 2])

    def failed_compile(*args, **kwargs):
        raise RuntimeError("synthetic compiler refusal")

    monkeypatch.setattr(fused, "_supported", lambda *args: True)
    monkeypatch.setattr(fused, "_launch_forward", failed_compile)
    fused.clear_failure_caches()
    fused.reset_launch_stats()
    try:
        with pytest.raises(
            fused.EquivariantStageFusionError,
            match="refusing to silently de-fuse the trunk",
        ):
            run_fused(inv, axis, mix, ffn, film, phase)
        assert fused.launch_stats()["forward_eligible"] == 1
        assert fused.launch_stats()["forward_launched"] == 0
        assert fused._FAILED_FORWARD_SHAPES
    finally:
        fused.clear_failure_caches()
        fused.reset_launch_stats()


def test_static_architectures_own_distinct_forward_and_backward_frames(
    monkeypatch,
):
    monkeypatch.setattr(
        torch,
        "compile",
        lambda function, **_kwargs: function,
    )
    fused.clear_compile_caches()
    eps = (1e-5, 1e-5, 1e-5, 1e-5)
    try:
        forward_a = fused._forward_function(
            "silu", True, 0, eps, True, ("architecture-a",)
        )
        forward_b = fused._forward_function(
            "silu", True, 0, eps, True, ("architecture-b",)
        )
        backward_a = fused._backward_function(
            "silu", True, 0, eps, True, ("architecture-a",)
        )
        backward_b = fused._backward_function(
            "silu", True, 0, eps, True, ("architecture-b",)
        )
        assert forward_a.__code__ is not forward_b.__code__
        assert backward_a.__code__ is not backward_b.__code__
        assert forward_a.__name__ != forward_b.__name__
        assert backward_a.__name__ != backward_b.__name__
    finally:
        fused.clear_compile_caches()


def test_compiled_forward_has_only_finite_static_buckets_across_row_shapes(
    monkeypatch,
):
    """The general-row callable must not recompile for each packed chunk."""
    mix, ffn, film = modules(dtype=torch.float32)
    original_compile = torch.compile
    graph_count = 0

    def backend(graph, _example_inputs):
        nonlocal graph_count
        graph_count += 1
        return graph.forward

    def counted_compile(function, **kwargs):
        # Inductor-specific options are irrelevant to this Dynamo guard test.
        kwargs.pop("options", None)
        return original_compile(function, backend=backend, **kwargs)

    monkeypatch.setattr(torch, "compile", counted_compile)
    fused.clear_compile_caches()
    eps = (
        float(mix.norm.inv.eps),
        float(mix.norm.axis.eps),
        float(ffn.norm.inv.eps),
        float(ffn.norm.axis.eps),
    )
    try:
        with torch._dynamo.config.patch(error_on_recompile=True):
            for rows in (1, 2, 3, 5, 8, 13, 21, 34):
                inv, axis = state(
                    dtype=torch.float32,
                    rows=rows,
                )
                phase = torch.arange(rows, dtype=torch.long).remainder(3)
                tensors = stage_tensors(inv, axis, mix, ffn, film, phase)
                cotangents = (torch.ones_like(inv), torch.ones_like(axis))
                fused._mark_dynamic_rows(tensors, True, cotangents)
                function = fused._forward_function(
                    "silu",
                    True,
                    0,
                    eps,
                    True,
                    fused._compile_key(tensors, "silu", True, 0, eps),
                )
                got_inv, got_axis = function(*tensors)
                assert got_inv.shape == inv.shape
                assert got_axis.shape == axis.shape
                gradients = fused._launch_backward(
                    tensors,
                    "silu",
                    True,
                    0,
                    eps,
                    *cotangents,
                )
                assert len(gradients) == len(tensors) - 2

        # Singleton strides are an unavoidable Dynamo specialization.  Every
        # N>=2 size above shares one general forward and one backward graph.
        assert graph_count == 4
    finally:
        fused.clear_compile_caches()


@_CUDA
def test_cuda_compiled_fp32_forward_backward_match_modules_and_really_launch():
    mix, ffn, film = modules(dtype=torch.float32, device="cuda")
    inv, axis = state(dtype=torch.float32, device="cuda", rows=7)
    phase = torch.tensor([0, 1, 2, 0, 2, 1, 0], device="cuda")
    parameters = (*mix_parameters(mix), *ffn_parameters(ffn), *film_parameters(film))
    seed_inv = torch.randn_like(inv)
    seed_axis = torch.randn_like(axis)

    fused.clear_failure_caches()
    fused.reset_launch_stats()
    reference = film(ffn(mix(EquivariantState(inv, axis))), phase)
    got = run_fused(inv, axis, mix, ffn, film, phase)
    reference_loss = (reference.inv * seed_inv).sum() + (
        reference.axis * seed_axis
    ).sum()
    got_loss = (got[0] * seed_inv).sum() + (got[1] * seed_axis).sum()
    wanted_grad = torch.autograd.grad(
        reference_loss, (inv, axis, *parameters), retain_graph=True
    )
    got_grad = torch.autograd.grad(got_loss, (inv, axis, *parameters))

    assert relative(got[0], reference.inv) <= TOL
    assert relative(got[1], reference.axis) <= TOL
    for actual, expected in zip(got_grad, wanted_grad):
        assert relative(actual, expected) <= TOL
    assert fused.launch_stats() == {
        "forward_eligible": 1,
        "forward_launched": 1,
        "backward_eligible": 1,
        "backward_launched": 1,
    }
    assert not fused._FAILED_FORWARD_SHAPES
    assert not fused._FAILED_BACKWARD_SHAPES


@_CUDA
def test_cuda_bf16_error_is_reported_against_fp32_anchor_and_really_launches(
    record_property,
):
    """bf16 is documented against fp32, not gated against eager bf16.

    The old device test accidentally applied the fp32 reassociation bar to a
    pair of bf16-autocast computations.  That made a legal one-ULP difference
    between Inductor's recompute VJP and eager CUDA fail at ``2e-4`` even
    though the whole-model fp32 parity and bf16-anchor gates were green.  Keep
    the strict bar in the fp32 test above; this cell applies the repository's
    whole-stage policy and instead records output and gradient drift against
    fp32 while gating finiteness and real fused launches.
    """
    mix, ffn, film = modules(dtype=torch.float32, device="cuda")
    inv, axis = state(dtype=torch.float32, device="cuda", rows=7)
    phase = torch.tensor([0, 1, 2, 0, 2, 1, 0], device="cuda")
    parameters = (*mix_parameters(mix), *ffn_parameters(ffn), *film_parameters(film))
    seed_inv = torch.randn_like(inv)
    seed_axis = torch.randn_like(axis)

    anchor = film(ffn(mix(EquivariantState(inv, axis))), phase)
    anchor_loss = (anchor.inv * seed_inv).sum() + (
        anchor.axis * seed_axis
    ).sum()
    anchor_grad = torch.autograd.grad(
        anchor_loss, (inv, axis, *parameters), retain_graph=True
    )

    fused.clear_failure_caches()
    fused.reset_launch_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        got = run_fused(inv, axis, mix, ffn, film, phase)
        got_loss = (got[0] * seed_inv).sum() + (got[1] * seed_axis).sum()
    got_grad = torch.autograd.grad(got_loss, (inv, axis, *parameters))

    output_errors = (
        relative(got[0].float(), anchor.inv.float()),
        relative(got[1].float(), anchor.axis.float()),
    )
    gradient_errors = tuple(
        relative(actual.float(), expected.float())
        for actual, expected in zip(got_grad, anchor_grad)
    )
    worst_gradient_index = max(
        range(len(gradient_errors)), key=gradient_errors.__getitem__
    )
    record_property("bf16_fp32_inv_output", output_errors[0])
    record_property("bf16_fp32_axis_output", output_errors[1])
    record_property("bf16_fp32_worst_gradient", gradient_errors[worst_gradient_index])
    record_property("bf16_fp32_worst_gradient_index", worst_gradient_index)
    assert all(torch.isfinite(value).all() for value in (*got, *got_grad))
    assert fused.launch_stats() == {
        "forward_eligible": 1,
        "forward_launched": 1,
        "backward_eligible": 1,
        "backward_launched": 1,
    }
    assert not fused._FAILED_FORWARD_SHAPES
    assert not fused._FAILED_BACKWARD_SHAPES


@_CUDA
def test_cuda_forward_and_recompute_backward_do_not_recompile_across_rows():
    mix, ffn, film = modules(dtype=torch.float32, device="cuda")
    parameters = (*mix_parameters(mix), *ffn_parameters(ffn), *film_parameters(film))

    def once(rows):
        inv, axis = state(dtype=torch.float32, device="cuda", rows=rows)
        phase = torch.arange(rows, device="cuda", dtype=torch.long).remainder(3)
        outputs = run_fused(inv, axis, mix, ffn, film, phase)
        torch.autograd.grad(
            outputs[0].sum() + outputs[1].sum(),
            (inv, axis, *parameters),
        )

    fused.clear_failure_caches()
    fused.reset_launch_stats()
    once(5)  # Warm the general N>=2 forward and recompute-backward graphs.
    with torch._dynamo.config.patch(error_on_recompile=True):
        # Every other general row count must reuse the warmed symbolic graph,
        # including the compiled forward's AOT backward.
        for rows in (7, 11, 19):
            once(rows)
    torch.cuda.synchronize()

    stats = fused.launch_stats()
    assert stats["forward_eligible"] == stats["forward_launched"] == 4
    assert stats["backward_eligible"] == stats["backward_launched"] == 4
    assert not fused._FAILED_FORWARD_SHAPES
    assert not fused._FAILED_BACKWARD_SHAPES


@_CUDA
def test_compiled_fp32_forward_and_all_gradients_are_bitwise_deterministic():
    mix, ffn, film = modules(dtype=torch.float32, device="cuda")
    phase = torch.tensor([0, 1, 2, 1, 0], device="cuda")
    parameters = (*mix_parameters(mix), *ffn_parameters(ffn), *film_parameters(film))

    def once():
        inv, axis = state(dtype=torch.float32, device="cuda", rows=5)
        out = run_fused(inv, axis, mix, ffn, film, phase)
        loss = out[0].sum() + out[1].sum()
        gradients = torch.autograd.grad(loss, (inv, axis, *parameters))
        return (*out, *gradients)

    first = once()
    for other in (once(), once()):
        for left, right in zip(first, other):
            assert torch.equal(left, right)
