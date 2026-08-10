"""Whole-stage §14 message fusion against an independent edge-loop oracle.

The oracle owns the original edge rows and never reads a CSR column.  That is
the detector for a symmetric plan bug: the operator and its literal fallback
may share the same wrong CSR view, while the edge loop cannot.  Float64
gradcheck covers every state and parameter input.  CUDA tests separately
require successful compiled forward/backward launches, compare all gradients
to the literal formulation, and retain the fp32 bitwise determinism control.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

import mantisnet.models.mantis_act.fused_message as fused
from mantisnet.models.mantis_act.fused_message import relation_gated_message_stage
from mantisnet.models.mantis_act.segment_message import MessagePlan, message_plan


_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the compiled message stage needs CUDA"
)

N_SOURCE = 3
N_DESTINATION = 2
N_RELATIONS = 3
D_INV = 3
D_AXIS = 2
D_REL = 2

EDGE_SOURCE = torch.tensor([0, 1, 2, 0, 2], dtype=torch.long)
EDGE_DESTINATION = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
EDGE_RELATION = torch.tensor([2, 0, 1, 2, 0], dtype=torch.long)
EDGE_AXIS = torch.tensor([0, 2, 1, 2, 0], dtype=torch.long)


@dataclass(frozen=True)
class Case:
    floats: tuple[torch.Tensor, ...]
    inv_plan: MessagePlan
    axis_plan: MessagePlan


def _draw(
    generator: torch.Generator,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    *,
    norm_weight: bool = False,
    layer_scale: bool = False,
) -> torch.Tensor:
    value = torch.randn(shape, generator=generator, dtype=torch.float64)
    if norm_weight:
        value = 1.0 + 0.1 * value
    elif layer_scale:
        value = 0.1 * value
    return value.to(dtype)


def _stream_parameters(
    generator: torch.Generator, width: int, dtype: torch.dtype
) -> tuple[torch.Tensor, ...]:
    return (
        _draw(generator, (width,), dtype, norm_weight=True),
        _draw(generator, (width,), dtype),
        _draw(generator, (width, width), dtype),
        _draw(generator, (width, D_REL), dtype),
        _draw(generator, (width,), dtype),
        _draw(generator, (width, D_REL), dtype),
        _draw(generator, (width,), dtype),
        _draw(generator, (width,), dtype, norm_weight=True),
        _draw(generator, (width,), dtype),
        _draw(generator, (width, 2 * width), dtype),
        _draw(generator, (width,), dtype),
        _draw(generator, (width, width), dtype),
        _draw(generator, (width,), dtype),
        _draw(generator, (width,), dtype, layer_scale=True),
    )


def _case(dtype: torch.dtype = torch.float64) -> Case:
    generator = torch.Generator().manual_seed(20260809)
    floats = (
        _draw(generator, (N_SOURCE, D_INV), dtype),
        _draw(generator, (N_SOURCE, 3, D_AXIS), dtype),
        _draw(generator, (N_DESTINATION, D_INV), dtype),
        _draw(generator, (N_DESTINATION, 3, D_AXIS), dtype),
        _draw(generator, (N_RELATIONS, D_REL), dtype),
        *_stream_parameters(generator, D_INV, dtype),
        *_stream_parameters(generator, D_AXIS, dtype),
    )
    inv_plan = message_plan(
        EDGE_SOURCE,
        EDGE_DESTINATION,
        EDGE_RELATION,
        None,
        N_SOURCE,
        N_DESTINATION,
        N_RELATIONS,
        1,
        dst_sorted=True,
    )
    axis_plan = message_plan(
        EDGE_SOURCE,
        EDGE_DESTINATION,
        EDGE_RELATION,
        EDGE_AXIS,
        N_SOURCE,
        N_DESTINATION,
        N_RELATIONS,
        3,
        dst_sorted=True,
    )
    return Case(floats, inv_plan, axis_plan)


def _shape_case(
    n_source: int,
    n_destination: int,
    n_edges: int,
    dtype: torch.dtype = torch.float32,
    *,
    n_axis_edges: int | None = None,
) -> Case:
    """A fixed-architecture case whose packed row counts can vary."""
    base = _case(dtype)
    generator = torch.Generator().manual_seed(
        20260810 + n_source + 3 * n_destination + 5 * n_edges
    )
    edge_source = torch.arange(n_edges, dtype=torch.long) % n_source
    edge_destination = (
        2 * torch.arange(n_edges, dtype=torch.long) + 1
    ) % n_destination
    edge_relation = torch.arange(n_edges, dtype=torch.long) % N_RELATIONS
    if n_axis_edges is None:
        n_axis_edges = n_edges
    axis_source = torch.arange(n_axis_edges, dtype=torch.long) % n_source
    axis_destination = (
        2 * torch.arange(n_axis_edges, dtype=torch.long) + 1
    ) % n_destination
    axis_relation = torch.arange(n_axis_edges, dtype=torch.long) % N_RELATIONS
    edge_axis = torch.arange(n_axis_edges, dtype=torch.long) % 3
    floats = (
        torch.randn(n_source, D_INV, generator=generator, dtype=dtype),
        torch.randn(n_source, 3, D_AXIS, generator=generator, dtype=dtype),
        torch.randn(n_destination, D_INV, generator=generator, dtype=dtype),
        torch.randn(n_destination, 3, D_AXIS, generator=generator, dtype=dtype),
        torch.randn(N_RELATIONS, D_REL, generator=generator, dtype=dtype),
        *base.floats[5:],
    )
    inv_plan = message_plan(
        edge_source,
        edge_destination,
        edge_relation,
        None,
        n_source,
        n_destination,
        N_RELATIONS,
        1,
        dst_sorted=False,
    )
    axis_plan = message_plan(
        axis_source,
        axis_destination,
        axis_relation,
        edge_axis,
        n_source,
        n_destination,
        N_RELATIONS,
        3,
        dst_sorted=False,
    )
    return Case(floats, inv_plan, axis_plan)


def _move(
    case: Case,
    device: str,
    *,
    requires_grad: bool,
) -> Case:
    floats = tuple(
        value.detach().to(device).requires_grad_(requires_grad) for value in case.floats
    )
    return Case(floats, case.inv_plan.to(device), case.axis_plan.to(device))


def _call(case: Case) -> tuple[torch.Tensor, torch.Tensor]:
    return relation_gated_message_stage(
        *case.floats[:5],
        case.floats[5:19],
        case.floats[19:],
        case.inv_plan,
        case.axis_plan,
    )


def _literal(case: Case, autocast_code: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    return fused._reference(
        case.floats,
        tuple(fused._plan_tensors(case.inv_plan, case.axis_plan)),
        activation="silu",
        autocast_code=autocast_code,
        eps=(1e-5, 1e-5, 1e-5, 1e-5),
        fused_segment=False,
    )


def _oracle_stream(
    source: torch.Tensor,
    destination: torch.Tensor,
    relation: torch.Tensor,
    parameters: tuple[torch.Tensor, ...],
    *,
    axis_stream: bool,
) -> torch.Tensor:
    (
        source_norm_weight,
        source_norm_bias,
        value_weight,
        gate_weight,
        gate_bias,
        bias_weight,
        bias_bias,
        destination_norm_weight,
        destination_norm_bias,
        update_in_weight,
        update_in_bias,
        update_out_weight,
        update_out_bias,
        gamma,
    ) = parameters
    width = source.shape[-1]
    values = F.linear(
        F.layer_norm(source, (width,), source_norm_weight, source_norm_bias),
        value_weight,
    )
    gate = torch.sigmoid(F.linear(relation, gate_weight, gate_bias))
    bias = F.linear(relation, bias_weight, bias_bias)
    if axis_stream:
        aggregate = torch.zeros(
            N_DESTINATION, 3, width, dtype=source.dtype, device=source.device
        )
    else:
        aggregate = torch.zeros(
            N_DESTINATION, width, dtype=source.dtype, device=source.device
        )
    # Independent indexing: original edge rows, not destination/source/relation
    # CSR columns or their run lengths.
    for source_row, destination_row, relation_row, axis in zip(
        EDGE_SOURCE.tolist(),
        EDGE_DESTINATION.tolist(),
        EDGE_RELATION.tolist(),
        EDGE_AXIS.tolist(),
    ):
        value = values[source_row, axis] if axis_stream else values[source_row]
        message = gate[relation_row] * value + bias[relation_row]
        if axis_stream:
            aggregate[destination_row, axis] += message
        else:
            aggregate[destination_row] += message

    z = F.layer_norm(
        destination,
        (width,),
        destination_norm_weight,
        destination_norm_bias,
    )
    hidden = F.silu(F.linear(torch.cat((z, aggregate), dim=-1), update_in_weight, update_in_bias))
    delta = F.linear(hidden, update_out_weight, update_out_bias)
    return destination + gamma * delta


def _oracle(case: Case) -> tuple[torch.Tensor, torch.Tensor]:
    inv = _oracle_stream(
        case.floats[0],
        case.floats[2],
        case.floats[4],
        case.floats[5:19],
        axis_stream=False,
    )
    axis = _oracle_stream(
        case.floats[1],
        case.floats[3],
        case.floats[4],
        case.floats[19:],
        axis_stream=True,
    )
    return inv, axis


def _relative(got: torch.Tensor, want: torch.Tensor) -> float:
    gap = float((got.detach().cpu() - want.detach().cpu()).abs().max())
    scale = float(want.detach().cpu().abs().max())
    return gap if scale == 0.0 else gap / scale


def test_float64_reference_matches_independent_edge_loop() -> None:
    case = _case()
    with torch.no_grad():
        got = _call(case)
        want = _oracle(case)
    for actual, expected in zip(got, want):
        assert torch.allclose(actual, expected, rtol=1e-11, atol=1e-11)


def test_float64_gradcheck_covers_states_and_every_parameter() -> None:
    base = _case()
    floats = tuple(value.detach().requires_grad_(True) for value in base.floats)

    def function(*values):
        return _call(Case(tuple(values), base.inv_plan, base.axis_plan))

    assert torch.autograd.gradcheck(
        function,
        floats,
        eps=1e-6,
        atol=2e-5,
        rtol=2e-4,
        fast_mode=True,
    )


def test_registered_schema_fake_tensor_and_autograd_contract() -> None:
    base = _case()
    floats = tuple(value.detach().requires_grad_(True) for value in base.floats)
    torch.library.opcheck(
        fused._message_stage_op,
        (
            *floats,
            fused._plan_tensors(base.inv_plan, base.axis_plan),
            "silu",
            0,
            1e-5,
            1e-5,
            1e-5,
            1e-5,
        ),
    )


def test_recompute_vjp_shell_uses_registered_ordered_segment_backward() -> None:
    base = _case()
    plans = tuple(fused._plan_tensors(base.inv_plan, base.axis_plan))
    cotangents = (
        torch.randn(N_DESTINATION, D_INV, dtype=torch.float64),
        torch.randn(N_DESTINATION, 3, D_AXIS, dtype=torch.float64),
    )

    def transformed(*values):
        return fused._reference(
            tuple(values),
            plans,
            activation="silu",
            autocast_code=0,
            eps=(1e-5, 1e-5, 1e-5, 1e-5),
            fused_segment=True,
            transformable_segment=True,
        )

    _output, vjp = torch.func.vjp(transformed, *base.floats)
    got = vjp(cotangents)

    literal_floats = tuple(
        value.detach().requires_grad_(True) for value in base.floats
    )
    literal_case = Case(literal_floats, base.inv_plan, base.axis_plan)
    want = torch.autograd.grad(_literal(literal_case), literal_floats, cotangents)
    for actual, expected in zip(got, want):
        assert torch.allclose(actual, expected, rtol=1e-11, atol=1e-11)


def test_compiled_registered_recompute_backward_survives_autograd_no_grad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The functorch VJP must work in a custom op's no-grad backward context."""

    def compile_stage(function):
        return torch.compile(
            function,
            backend="eager",
            fullgraph=True,
            dynamic=True,
        )

    backward_grad_modes: list[bool] = []
    registered_backward = fused._message_stage_backward_op

    def observe_backward_context(*args, **kwargs):
        backward_grad_modes.append(torch.is_grad_enabled())
        return registered_backward(*args, **kwargs)

    monkeypatch.setattr(fused, "_compile_stage", compile_stage)
    monkeypatch.setattr(fused, "_supported", lambda *args: True)
    monkeypatch.setattr(fused, "_message_stage_backward_op", observe_backward_context)
    fused.clear_compile_caches()
    fused.clear_failure_caches()
    fused.reset_launch_stats()

    base = _case(torch.float32)
    got_case = Case(
        tuple(value.detach().requires_grad_(True) for value in base.floats),
        base.inv_plan,
        base.axis_plan,
    )
    want_case = Case(
        tuple(value.detach().requires_grad_(True) for value in base.floats),
        base.inv_plan,
        base.axis_plan,
    )
    cotangents = (
        torch.randn(N_DESTINATION, D_INV),
        torch.randn(N_DESTINATION, 3, D_AXIS),
    )

    try:
        got = _call(got_case)
        want = _literal(want_case)
        got_gradients = torch.autograd.grad(got, got_case.floats, cotangents)
        want_gradients = torch.autograd.grad(want, want_case.floats, cotangents)
    finally:
        # Do not leave the eager test backend cached for a later CUDA test.
        fused.clear_compile_caches()
        fused.clear_failure_caches()

    assert backward_grad_modes == [False]
    assert fused.launch_stats() == {
        "forward_eligible": 1,
        "forward_launched": 1,
        "backward_eligible": 1,
        "backward_launched": 1,
    }
    for actual, expected in zip(got_gradients, want_gradients):
        assert torch.allclose(actual, expected, rtol=2e-4, atol=2e-5)


def test_unsupported_activation_uses_the_literal_equations() -> None:
    case = _case(torch.float32)
    got = relation_gated_message_stage(
        *case.floats[:5],
        case.floats[5:19],
        case.floats[19:],
        case.inv_plan,
        case.axis_plan,
        activation="relu",
    )
    want = fused._reference(
        case.floats,
        tuple(fused._plan_tensors(case.inv_plan, case.axis_plan)),
        activation="relu",
        autocast_code=0,
        eps=(1e-5, 1e-5, 1e-5, 1e-5),
        fused_segment=False,
    )
    for actual, expected in zip(got, want):
        assert torch.equal(actual, expected)


def test_dynamic_compile_boundary_uses_one_graph_for_all_chunk_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CountingBackend:
        def __init__(self) -> None:
            self.graphs = 0

        def __call__(self, graph, example_inputs):
            del example_inputs
            self.graphs += 1
            return graph.forward

    backend = CountingBackend()

    def compile_stage(function):
        return torch.compile(
            function,
            backend=backend,
            fullgraph=True,
            dynamic=True,
        )

    monkeypatch.setattr(fused, "_compile_stage", compile_stage)
    fused.clear_compile_caches()
    cases = [
        _shape_case(5, 4, 9),
        _shape_case(8, 6, 13, n_axis_edges=7),
        _shape_case(11, 7, 21, n_axis_edges=11),
    ]
    cases = [
        Case(
            tuple(value.detach().requires_grad_(True) for value in case.floats),
            case.inv_plan,
            case.axis_plan,
        )
        for case in cases
    ]
    try:
        for case in cases:
            fused._launch_forward(
                case.floats,
                tuple(fused._plan_tensors(case.inv_plan, case.axis_plan)),
                "silu",
                0,
                (1e-5, 1e-5, 1e-5, 1e-5),
            )
        assert backend.graphs == 1
        assert fused._forward_function.cache_info().currsize == 1

        for case in cases:
            plans = tuple(fused._plan_tensors(case.inv_plan, case.axis_plan))
            fused._launch_backward(
                case.floats,
                plans,
                "silu",
                0,
                (1e-5, 1e-5, 1e-5, 1e-5),
                torch.randn(case.floats[2].shape),
                torch.randn(case.floats[3].shape),
                compiled=True,
            )
        assert backend.graphs == 2
        assert fused._backward_function.cache_info().currsize == 1
    finally:
        # Do not leave an eager counting backend cached for a later CUDA test.
        fused.clear_compile_caches()


def test_static_signature_partitions_singletons_and_requires_grad() -> None:
    singleton = _shape_case(1, 2, 1)
    ordinary_a = _shape_case(5, 4, 9)
    ordinary_b = _shape_case(8, 6, 13)

    def signature(case: Case):
        return fused._static_signature(
            case.floats,
            tuple(fused._plan_tensors(case.inv_plan, case.axis_plan)),
        )

    assert signature(singleton) != signature(ordinary_a)
    assert signature(ordinary_a) == signature(ordinary_b)
    requiring_grad = Case(
        tuple(value.detach().requires_grad_(True) for value in ordinary_a.floats),
        ordinary_a.inv_plan,
        ordinary_a.axis_plan,
    )
    assert signature(requiring_grad) != signature(ordinary_a)


def test_static_families_own_independent_dynamo_code_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = []

    def capture(function):
        captured.append(function)
        return function

    monkeypatch.setattr(fused, "_compile_stage", capture)
    fused.clear_compile_caches()
    settings = ("silu", 0, (1e-5, 1e-5, 1e-5, 1e-5), True)
    try:
        first_forward = fused._forward_function(
            *settings, ("architecture-a",)
        )
        again_forward = fused._forward_function(
            *settings, ("architecture-a",)
        )
        second_forward = fused._forward_function(
            *settings, ("architecture-b",)
        )
        first_backward = fused._backward_function(
            *settings, ("architecture-a",)
        )
        again_backward = fused._backward_function(
            *settings, ("architecture-a",)
        )
        second_backward = fused._backward_function(
            *settings, ("architecture-b",)
        )

        assert first_forward is again_forward
        assert first_backward is again_backward
        assert first_forward.__code__ is not second_forward.__code__
        assert first_backward.__code__ is not second_backward.__code__
        assert first_forward.__name__ != second_forward.__name__
        assert first_backward.__name__ != second_backward.__name__
        assert len(captured) == 4
        assert len({function.__code__ for function in captured}) == 4
    finally:
        fused.clear_compile_caches()


def test_eligible_compile_failure_is_named_and_never_uses_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(torch.float32)
    plans = fused._plan_tensors(case.inv_plan, case.axis_plan)
    sentinel = RuntimeError("synthetic compiler failure")

    monkeypatch.setattr(fused, "_supported", lambda *args: True)
    monkeypatch.setattr(
        fused,
        "_reference",
        lambda *args, **kwargs: pytest.fail("eligible path silently de-fused"),
    )
    monkeypatch.setattr(
        fused, "_launch_forward", lambda *args, **kwargs: (_ for _ in ()).throw(sentinel)
    )
    fused.clear_failure_caches()
    with pytest.raises(
        fused.MessageStageCompilationError,
        match="refusing to silently de-fuse",
    ):
        _call(case)

    monkeypatch.setattr(
        fused, "_launch_backward", lambda *args, **kwargs: (_ for _ in ()).throw(sentinel)
    )
    with pytest.raises(
        fused.MessageStageCompilationError,
        match="refusing to silently de-fuse",
    ):
        fused._message_stage_backward_op(
            *case.floats,
            plans,
            "silu",
            0,
            1e-5,
            1e-5,
            1e-5,
            1e-5,
            torch.randn_like(case.floats[2]),
            torch.randn_like(case.floats[3]),
        )
    fused.clear_failure_caches()


@_CUDA
def test_cuda_compiled_forward_and_backward_never_recompile_for_new_chunk_shape() -> None:
    fused.clear_failure_caches()
    fused.clear_compile_caches()
    fused.reset_launch_stats()

    def run(base: Case) -> None:
        case = _move(base, "cuda", requires_grad=True)
        output = _call(case)
        torch.autograd.grad(
            output,
            case.floats,
            (
                torch.randn_like(output[0]),
                torch.randn_like(output[1]),
            ),
        )

    # Warm the one fixed R=3 architecture before turning every attempted
    # recompile into a hard regression failure.
    run(_shape_case(5, 4, 9))
    with torch._dynamo.config.patch(error_on_recompile=True):
        run(_shape_case(8, 6, 13, n_axis_edges=7))
        run(_shape_case(11, 7, 21, n_axis_edges=11))

    stats = fused.launch_stats()
    assert stats == {
        "forward_eligible": 3,
        "forward_launched": 3,
        "backward_eligible": 3,
        "backward_launched": 3,
    }
    assert fused._forward_function.cache_info().currsize == 1
    assert fused._backward_function.cache_info().currsize == 1
    assert not fused._FAILED_FORWARD_SHAPES
    assert not fused._FAILED_BACKWARD_SHAPES


@_CUDA
def test_cuda_compiled_forward_and_all_gradients_match_literal() -> None:
    fused.clear_failure_caches()
    fused.reset_launch_stats()
    base = _case(torch.float32)
    got_case = _move(base, "cuda", requires_grad=True)
    want_case = _move(base, "cuda", requires_grad=True)
    generator = torch.Generator(device="cuda").manual_seed(911)
    grad_inv = torch.randn(
        N_DESTINATION, D_INV, device="cuda", generator=generator
    )
    grad_axis = torch.randn(
        N_DESTINATION, 3, D_AXIS, device="cuda", generator=generator
    )

    got = _call(got_case)
    want = _literal(want_case)
    got_gradients = torch.autograd.grad(got, got_case.floats, (grad_inv, grad_axis))
    want_gradients = torch.autograd.grad(
        want, want_case.floats, (grad_inv, grad_axis)
    )

    for actual, expected in zip(got, want):
        assert _relative(actual, expected) < 2e-5
    for actual, expected in zip(got_gradients, want_gradients):
        assert _relative(actual, expected) < 2e-4

    stats = fused.launch_stats()
    assert stats == {
        "forward_eligible": 1,
        "forward_launched": 1,
        "backward_eligible": 1,
        "backward_launched": 1,
    }
    assert not fused._FAILED_FORWARD_SHAPES
    assert not fused._FAILED_BACKWARD_SHAPES


@_CUDA
def test_cuda_fp32_is_bitwise_deterministic_forward_and_backward() -> None:
    fused.clear_failure_caches()
    base = _case(torch.float32)
    grad_inv = torch.randn(N_DESTINATION, D_INV, device="cuda")
    grad_axis = torch.randn(N_DESTINATION, 3, D_AXIS, device="cuda")

    def once():
        case = _move(base, "cuda", requires_grad=True)
        output = _call(case)
        gradients = torch.autograd.grad(
            output, case.floats, (grad_inv, grad_axis)
        )
        return tuple(value.detach().clone() for value in (*output, *gradients))

    first = once()
    second = once()
    assert all(torch.equal(a, b) for a, b in zip(first, second))


@_CUDA
def test_bf16_autocast_is_reported_against_fp32_anchor_and_launches(
    record_property,
) -> None:
    """Report bf16 drift against fp32 while gating finiteness and dispatch.

    This is the same policy as the whole-stage bf16 test: fp32 owns the strict
    parity bar, while bf16 is documented against that anchor rather than gated
    against itself.  The eager-autocast baseline is recorded separately so a
    compiler-specific drift remains visible in the test report.
    """
    fused.clear_failure_caches()
    fused.reset_launch_stats()
    base = _case(torch.float32)
    fp32_case = _move(base, "cuda", requires_grad=False)
    amp_case = _move(base, "cuda", requires_grad=False)
    fp32 = _literal(fp32_case)
    eager_amp = _literal(amp_case, autocast_code=2)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        amp = _call(amp_case)
    for stream, actual, eager, anchor in zip(
        ("invariant", "axis"), amp, eager_amp, fp32
    ):
        eager_error = _relative(eager.float(), anchor.float())
        compiled_error = _relative(actual.float(), anchor.float())
        compiled_eager_delta = _relative(actual.float(), eager.float())
        record_property(f"bf16_fp32_{stream}", compiled_error)
        record_property(f"eager_bf16_fp32_{stream}", eager_error)
        record_property(f"compiled_eager_bf16_{stream}", compiled_eager_delta)
        assert torch.isfinite(actual).all()
    stats = fused.launch_stats()
    assert stats["forward_eligible"] == stats["forward_launched"] == 1
    assert not fused._FAILED_FORWARD_SHAPES
