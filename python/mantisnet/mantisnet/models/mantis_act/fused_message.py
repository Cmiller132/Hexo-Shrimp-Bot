"""Compiled whole-stage relation-gated message update for the ACT default.

The ordinary :class:`messages.RelationGatedMessage` formulation exposes every
LayerNorm, projection, activation, cast, concatenation, and residual as a
separate eager operation.  This registered operator keeps the already fused,
ordered segment reduction as its irregular core and presents the dense work
around both invariant and axis streams to Inductor as one stage.

Backward saves only inputs, parameters, and the immutable CSR plans.  A
separately compiled :func:`torch.func.vjp` recomputes the forward; the nested
segment operator's registered backward supplies deterministic source- and
relation-major reductions without atomics.  The supported path is exactly the
``full_act_v4`` message choice: relation-gated, sum reduction, routed axis
stream, SiLU, and zero dropout.  CPU, fp64, and unsupported signatures execute
the literal tensor formulation used for parity and gradcheck.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
from types import FunctionType
from typing import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .equivariant import AXIS_CHANNELS, at_least_fp32
from .segment_message import (
    MessagePlan,
    _reference as _segment_reference,
    _segment_message_backward_op,
    _segment_message_op,
)


_STREAM_PARAMETER_COUNT = 14
_FLOAT_COUNT = 5 + 2 * _STREAM_PARAMETER_COUNT

# One invariant plan without axis columns, followed by one routed-axis plan.
_INV_PLAN_COUNT = 9
_AXIS_PLAN_COUNT = 12
_PLAN_COUNT = _INV_PLAN_COUNT + _AXIS_PLAN_COUNT

_CUDA_DTYPES = frozenset({torch.float16, torch.bfloat16, torch.float32})
_ACTIVATIONS = frozenset({"silu", "gelu", "relu"})

# The relation table and every feature/parameter width are architectural.  Row
# counts and CSR column lengths instead depend on the packed chunk.  Keeping
# that distinction explicit is essential: PT2 intentionally specializes a
# size-one relation table (adjacency), but no new graph may be compiled merely
# because the next chunk contains a different number of entities or edges.
_StaticSignature = tuple[object, ...]


class MessageStageCompilationError(RuntimeError):
    """An eligible CUDA message stage could not retain its fused contract."""

_FAILED_FORWARD_SHAPES: dict[tuple[object, ...], str] = {}
_FAILED_BACKWARD_SHAPES: dict[tuple[object, ...], str] = {}

_LAUNCH_STATS = {
    "forward_eligible": 0,
    "forward_launched": 0,
    "backward_eligible": 0,
    "backward_launched": 0,
}


def reset_launch_stats() -> None:
    """Reset supported-versus-successful counts used by the device law."""
    for name in _LAUNCH_STATS:
        _LAUNCH_STATS[name] = 0


def launch_stats() -> dict[str, int]:
    """Return a copy of the stage's launch accounting."""
    return dict(_LAUNCH_STATS)


def clear_failure_caches() -> None:
    """Clear sticky compile failures (intended for isolated tests only)."""
    _FAILED_FORWARD_SHAPES.clear()
    _FAILED_BACKWARD_SHAPES.clear()


def clear_compile_caches() -> None:
    """Discard compiled callable handles without changing global Dynamo state."""
    _forward_function.cache_clear()
    _backward_function.cache_clear()


def _activation(value: Tensor, name: str) -> Tensor:
    if name == "silu":
        return F.silu(value)
    if name == "gelu":
        return F.gelu(value)
    if name == "relu":
        return F.relu(value)
    raise ValueError(f"activation={name!r} is not one of {sorted(_ACTIVATIONS)}")


def _autocast_dtype(code: int) -> torch.dtype | None:
    if code == 0:
        return None
    if code == 1:
        return torch.float16
    if code == 2:
        return torch.bfloat16
    raise ValueError(f"unknown CUDA autocast code {code}")


def _autocast_code(value: Tensor) -> int:
    if not value.is_cuda or not torch.is_autocast_enabled("cuda"):
        return 0
    dtype = torch.get_autocast_dtype("cuda")
    if dtype == torch.float16:
        return 1
    if dtype == torch.bfloat16:
        return 2
    return 0


def _split_floats(
    tensors: tuple[Tensor, ...],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, tuple[Tensor, ...], tuple[Tensor, ...]]:
    if len(tensors) != _FLOAT_COUNT:
        raise ValueError(
            f"message stage expects {_FLOAT_COUNT} floating tensors, got {len(tensors)}"
        )
    source_inv, source_axis, destination_inv, destination_axis, relation = tensors[:5]
    inv_start = 5
    axis_start = inv_start + _STREAM_PARAMETER_COUNT
    return (
        source_inv,
        source_axis,
        destination_inv,
        destination_axis,
        relation,
        tensors[inv_start:axis_start],
        tensors[axis_start:],
    )


def _split_plans(
    plans: tuple[Tensor, ...],
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    if len(plans) != _PLAN_COUNT:
        raise ValueError(
            f"message stage expects {_PLAN_COUNT} plan tensors, got {len(plans)}"
        )
    return plans[:_INV_PLAN_COUNT], plans[_INV_PLAN_COUNT:]


class _TransformableSegment(torch.autograd.Function):
    """The existing registered segment ops with a torch.func-compatible shell.

    PyTorch 2.11's generated ``torch.library`` autograd wrapper supports eager
    autograd and AOTAutograd but does not itself expose the ``setup_context``
    method required by ``torch.func.vjp``.  This shell supplies only that
    transform contract; both computations still delegate to the one registered
    forward/backward implementation in :mod:`segment_message`.
    """

    generate_vmap_rule = True

    @staticmethod
    def forward(
        values: Tensor,
        gate: Tensor,
        bias: Tensor,
        dst_ptr: Tensor,
        dst_src: Tensor,
        dst_rel: Tensor,
        dst_axis: Tensor | None,
        src_ptr: Tensor,
        src_dst: Tensor,
        src_rel: Tensor,
        src_axis: Tensor | None,
        rel_ptr: Tensor,
        rel_src: Tensor,
        rel_dst: Tensor,
        rel_axis: Tensor | None,
        channels: int,
    ) -> Tensor:
        return _segment_message_op(
            values,
            gate,
            bias,
            dst_ptr,
            dst_src,
            dst_rel,
            dst_axis,
            src_ptr,
            src_dst,
            src_rel,
            src_axis,
            rel_ptr,
            rel_src,
            rel_dst,
            rel_axis,
            channels,
        )

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        del output
        ctx.channels = inputs[-1]
        ctx.save_for_backward(*inputs[:-1])

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        d_values, d_gate, d_bias = _segment_message_backward_op(
            *ctx.saved_tensors, ctx.channels, grad_out.contiguous()
        )
        return (d_values, d_gate, d_bias) + (None,) * 13


def _aggregate(
    values: Tensor,
    gate: Tensor,
    bias: Tensor,
    plan: tuple[Tensor, ...],
    channels: int,
    *,
    fused_segment: bool,
    transformable_segment: bool,
) -> Tensor:
    if channels == 1:
        (
            dst_ptr,
            dst_src,
            dst_rel,
            src_ptr,
            src_dst,
            src_rel,
            rel_ptr,
            rel_src,
            rel_dst,
        ) = plan
        dst_axis = src_axis = rel_axis = None
    else:
        (
            dst_ptr,
            dst_src,
            dst_rel,
            dst_axis,
            src_ptr,
            src_dst,
            src_rel,
            src_axis,
            rel_ptr,
            rel_src,
            rel_dst,
            rel_axis,
        ) = plan

    if fused_segment:
        operator = (
            _TransformableSegment.apply
            if transformable_segment
            else _segment_message_op
        )
        return operator(
            values.contiguous(),
            gate.contiguous(),
            bias.contiguous(),
            dst_ptr,
            dst_src,
            dst_rel,
            dst_axis,
            src_ptr,
            src_dst,
            src_rel,
            src_axis,
            rel_ptr,
            rel_src,
            rel_dst,
            rel_axis,
            channels,
        )
    return _segment_reference(
        values,
        gate,
        bias,
        dst_ptr,
        dst_src,
        dst_rel,
        dst_axis,
        channels,
    )


def _stream(
    source: Tensor,
    destination: Tensor,
    relation: Tensor,
    parameters: tuple[Tensor, ...],
    plan: tuple[Tensor, ...],
    channels: int,
    activation: str,
    source_eps: float,
    destination_eps: float,
    *,
    fused_segment: bool,
    transformable_segment: bool,
) -> Tensor:
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
        layer_scale,
    ) = parameters

    source_norm = F.layer_norm(
        source,
        (source.shape[-1],),
        source_norm_weight,
        source_norm_bias,
        source_eps,
    )
    values = F.linear(source_norm, value_weight)
    if channels == AXIS_CHANNELS:
        values = values.reshape(-1, values.shape[-1])
    values = at_least_fp32(values)

    gate = at_least_fp32(torch.sigmoid(F.linear(relation, gate_weight, gate_bias)))
    bias = at_least_fp32(F.linear(relation, bias_weight, bias_bias))
    aggregate = _aggregate(
        values,
        gate,
        bias,
        plan,
        channels,
        fused_segment=fused_segment,
        transformable_segment=transformable_segment,
    )
    if channels == AXIS_CHANNELS:
        # Infer the row count from the segment output.  Feeding the equivalent
        # ``destination.shape[0]`` into reshape makes functorch's recompute VJP
        # specialize that otherwise-dynamic dimension during guard creation.
        aggregate = aggregate.reshape(-1, AXIS_CHANNELS, aggregate.shape[-1])

    destination_norm = F.layer_norm(
        destination,
        (destination.shape[-1],),
        destination_norm_weight,
        destination_norm_bias,
        destination_eps,
    )
    pair = torch.cat((destination_norm, aggregate.to(destination_norm.dtype)), dim=-1)
    if channels == AXIS_CHANNELS:
        # Linear's functorch backward flattens leading batch dimensions using
        # their concrete product.  Flatten explicitly with an inferred row
        # count so the entity dimension remains symbolic in the recompute graph.
        pair = pair.reshape(-1, pair.shape[-1])
    hidden = _activation(F.linear(pair, update_in_weight, update_in_bias), activation)
    delta = F.linear(hidden, update_out_weight, update_out_bias)
    if channels == AXIS_CHANNELS:
        delta = delta.reshape(-1, AXIS_CHANNELS, delta.shape[-1])
    return destination + (layer_scale * delta).to(destination.dtype)


def _reference(
    tensors: tuple[Tensor, ...],
    plans: tuple[Tensor, ...],
    *,
    activation: str,
    autocast_code: int,
    eps: tuple[float, float, float, float],
    fused_segment: bool,
    transformable_segment: bool = False,
) -> tuple[Tensor, Tensor]:
    (
        source_inv,
        source_axis,
        destination_inv,
        destination_axis,
        relation,
        inv_parameters,
        axis_parameters,
    ) = _split_floats(tensors)
    inv_plan, axis_plan = _split_plans(plans)

    def run() -> tuple[Tensor, Tensor]:
        inv = _stream(
            source_inv,
            destination_inv,
            relation,
            inv_parameters,
            inv_plan,
            1,
            activation,
            eps[0],
            eps[1],
            fused_segment=fused_segment,
            transformable_segment=transformable_segment,
        )
        axis = _stream(
            source_axis,
            destination_axis,
            relation,
            axis_parameters,
            axis_plan,
            AXIS_CHANNELS,
            activation,
            eps[2],
            eps[3],
            fused_segment=fused_segment,
            transformable_segment=transformable_segment,
        )
        return inv, axis

    dtype = _autocast_dtype(autocast_code)
    if dtype is None:
        return run()
    # A custom-op implementation executes below its own dispatch key.  Re-enter
    # autocast explicitly so fp32 parameters follow the eager module's policy.
    with torch.autocast("cuda", dtype=dtype):
        return run()


def _size_bucket(value: Tensor) -> int:
    """Return PT2's unavoidable tiny-size bucket for a leading dimension."""
    size = int(value.shape[0])
    return size if size <= 2 else 3


def _static_signature(
    tensors: tuple[Tensor, ...],
    plans: tuple[Tensor, ...],
    gradients: tuple[Tensor, Tensor] = (),
) -> _StaticSignature:
    """Describe only architecture plus unavoidable zero/one specializations."""
    source_inv, source_axis, destination_inv, destination_axis, relation = tensors[:5]
    return (
        source_inv.device.type,
        source_inv.device.index,
        *(value.dtype for value in tensors),
        *(value.requires_grad for value in tensors),
        *(value.dtype for value in gradients),
        *(value.requires_grad for value in gradients),
        tuple(source_inv.shape[1:]),
        tuple(source_axis.shape[1:]),
        tuple(destination_inv.shape[1:]),
        tuple(destination_axis.shape[1:]),
        # Relation cardinality is fixed by the message family.  In particular,
        # adjacency R=1 cannot share the incidence R=2187 graph.
        tuple(relation.shape),
        *(tuple(parameter.shape) for parameter in tensors[5:]),
        # PT2 always specializes leading dimensions of size zero or one.  Give
        # those cases their own callable rather than allowing them to poison a
        # later ordinary-size chunk with a recompile.
        *(_size_bucket(value) for value in tensors[:4]),
        *(_size_bucket(value) for value in plans),
    )


_PLAN_POINTER_INDICES = frozenset(
    (
        0,  # invariant destination pointer: destination rows + 1
        3,  # invariant source pointer: source rows + 1
        6,  # invariant relation pointer: relation rows + 1
        _INV_PLAN_COUNT,  # axis destination pointer
        _INV_PLAN_COUNT + 4,  # axis source pointer
        _INV_PLAN_COUNT + 8,  # axis relation pointer
    )
)

_DYNAMIC_PLAN_INDICES = tuple(
    index
    for index in range(_PLAN_COUNT)
    # A pointer's length is an affine function of an already-marked entity-row
    # dimension, or of the architecture-fixed relation cardinality.  Marking
    # both sides as independent dynamic symbols makes ShapeEnv reject that
    # relation for small signatures (for example N_dst=2 and len(ptr)=3).
    # The compiled graph infers pointer shapes from those owner dimensions;
    # edge/routing columns remain independently dynamic here.
    if index not in _PLAN_POINTER_INDICES
)


def _mark_dynamic_dim0(value: Tensor) -> None:
    # Zero/one have distinct tensor semantics, while PT2 specializes a two-row
    # destination when it is equated to a custom op's pointer-derived output
    # shape.  _static_signature gives those finite cases their own family;
    # every production-sized row dimension remains polymorphic.
    if value.ndim and value.shape[0] > 2:
        torch._dynamo.mark_dynamic(value, 0)


def _mark_dynamic_inputs(
    tensors: tuple[Tensor, ...],
    plans: tuple[Tensor, ...],
    gradients: tuple[Tensor, Tensor] = (),
) -> None:
    """Annotate every chunk-varying row dimension before entering Dynamo."""
    for value in tensors[:4]:
        _mark_dynamic_dim0(value)
    for index in _DYNAMIC_PLAN_INDICES:
        _mark_dynamic_dim0(plans[index])
    for value in gradients:
        _mark_dynamic_dim0(value)


def _compile_stage(function: Callable) -> Callable:
    return torch.compile(
        function,
        fullgraph=True,
        dynamic=True,
        options={"emulate_precision_casts": True},
    )


def _isolate_code_object(
    function: Callable,
    direction: str,
    activation: str,
    autocast_code: int,
    eps: tuple[float, float, float, float],
    static_signature: _StaticSignature,
) -> Callable:
    """Give one static family its own Dynamo recompile-budget owner."""
    identity = repr(
        (direction, activation, autocast_code, eps, static_signature)
    ).encode("utf-8")
    suffix = hashlib.sha256(identity).hexdigest()[:12]
    name = f"_message_{direction}_{suffix}"
    return FunctionType(
        function.__code__.replace(co_name=name, co_qualname=name),
        function.__globals__,
        name,
        function.__defaults__,
        function.__closure__,
    )


@lru_cache(maxsize=None)
def _forward_function(
    activation: str,
    autocast_code: int,
    eps: tuple[float, float, float, float],
    compiled: bool,
    static_signature: _StaticSignature = (),
) -> Callable[..., tuple[Tensor, Tensor]]:
    def forward(*args: Tensor) -> tuple[Tensor, Tensor]:
        tensors = tuple(args[:_FLOAT_COUNT])
        plans = tuple(args[_FLOAT_COUNT:])
        return _reference(
            tensors,
            plans,
            activation=activation,
            autocast_code=autocast_code,
            eps=eps,
            fused_segment=compiled,
            transformable_segment=False,
        )

    if not compiled:
        return forward
    # Dynamo's recompile budget is owned by the Python code object rather than
    # the torch.compile wrapper.  Deterministically clone it once per fixed
    # family so tiny tests, adjacency/incidence/radius, and autocast cannot
    # exhaust one another's budgets.
    forward = _isolate_code_object(
        forward,
        "forward",
        activation,
        autocast_code,
        eps,
        static_signature,
    )
    return _compile_stage(forward)


@lru_cache(maxsize=None)
def _backward_function(
    activation: str,
    autocast_code: int,
    eps: tuple[float, float, float, float],
    compiled: bool,
    static_signature: _StaticSignature = (),
) -> Callable[..., tuple[Tensor, ...]]:
    def backward(*args: Tensor) -> tuple[Tensor, ...]:
        primals = tuple(args[:_FLOAT_COUNT])
        plans = tuple(args[_FLOAT_COUNT : _FLOAT_COUNT + _PLAN_COUNT])
        grad_inv, grad_axis = args[-2:]

        def forward(*floating: Tensor) -> tuple[Tensor, Tensor]:
            return _reference(
                tuple(floating),
                plans,
                activation=activation,
                autocast_code=autocast_code,
                eps=eps,
                fused_segment=compiled,
                transformable_segment=compiled,
            )

        _outputs, vjp = torch.func.vjp(forward, *primals)
        return vjp((grad_inv, grad_axis))

    if not compiled:
        return backward
    backward = _isolate_code_object(
        backward,
        "backward",
        activation,
        autocast_code,
        eps,
        static_signature,
    )
    return _compile_stage(backward)


def _shape_key(
    tensors: tuple[Tensor, ...],
    plans: tuple[Tensor, ...],
    activation: str,
    autocast_code: int,
    eps: tuple[float, float, float, float],
) -> tuple[object, ...]:
    source_inv, source_axis, destination_inv, destination_axis, relation = tensors[:5]
    inv_plan, axis_plan = _split_plans(plans)
    return (
        source_inv.device.type,
        source_inv.device.index,
        source_inv.dtype,
        source_axis.dtype,
        destination_inv.dtype,
        destination_axis.dtype,
        int(source_inv.shape[-1]),
        int(source_axis.shape[-1]),
        int(relation.shape[0]),
        int(relation.shape[1]),
        int(inv_plan[1].numel()),
        int(axis_plan[1].numel()),
        activation,
        autocast_code,
        *eps,
    )


def _supported(
    tensors: tuple[Tensor, ...],
    plans: tuple[Tensor, ...],
    activation: str,
    autocast_code: int,
) -> bool:
    source_inv, source_axis, destination_inv, destination_axis, _relation = tensors[:5]
    states = (source_inv, source_axis, destination_inv, destination_axis)
    return (
        source_inv.is_cuda
        and all(value.is_cuda for value in states)
        and all(value.dtype in _CUDA_DTYPES for value in states)
        and all(value.device == source_inv.device for value in tensors)
        and all(value.device == source_inv.device for value in plans)
        and all(value.dtype == torch.float32 for value in tensors[4:])
        and all(value.dtype == torch.int32 for value in plans)
        and source_inv.shape[-1] <= 512
        and source_axis.shape[-1] <= 512
        and activation == "silu"
        and autocast_code in (0, 1, 2)
        and destination_inv.shape[0] > 0
    )


def _expect_shape(name: str, value: Tensor, expected: tuple[int, ...]) -> None:
    if tuple(value.shape) != expected:
        raise ValueError(f"{name} must be {expected}, got {tuple(value.shape)}")


def _validate_stream(
    name: str,
    parameters: tuple[Tensor, ...],
    width: int,
    relation_width: int,
) -> None:
    if len(parameters) != _STREAM_PARAMETER_COUNT:
        raise ValueError(
            f"{name} parameters must carry {_STREAM_PARAMETER_COUNT} tensors"
        )
    shapes = (
        ("source norm weight", (width,)),
        ("source norm bias", (width,)),
        ("value weight", (width, width)),
        ("gate weight", (width, relation_width)),
        ("gate bias", (width,)),
        ("bias weight", (width, relation_width)),
        ("bias bias", (width,)),
        ("destination norm weight", (width,)),
        ("destination norm bias", (width,)),
        ("update input weight", (width, 2 * width)),
        ("update input bias", (width,)),
        ("update output weight", (width, width)),
        ("update output bias", (width,)),
        ("LayerScale gamma", (width,)),
    )
    for parameter, (part, shape) in zip(parameters, shapes):
        _expect_shape(f"{name} {part}", parameter, shape)
        if not parameter.is_floating_point():
            raise ValueError(f"{name} {part} must be floating point")


def _validate_plan(
    name: str,
    plan: tuple[Tensor, ...],
    *,
    n_source: int,
    n_destination: int,
    n_relations: int,
    axis: bool,
) -> None:
    expected_count = _AXIS_PLAN_COUNT if axis else _INV_PLAN_COUNT
    if len(plan) != expected_count:
        raise ValueError(f"{name} plan expects {expected_count} tensors")
    if any(value.dtype != torch.int32 for value in plan):
        raise TypeError(f"every {name} plan tensor must be int32")
    if any(value.ndim != 1 or not value.is_contiguous() for value in plan):
        raise ValueError(f"every {name} plan tensor must be contiguous and one-dimensional")
    if axis:
        dst_ptr, dst_src, dst_rel, dst_axis = plan[:4]
        src_ptr, src_dst, src_rel, src_axis = plan[4:8]
        rel_ptr, rel_src, rel_dst, rel_axis = plan[8:]
        edge_columns = (
            dst_src,
            dst_rel,
            dst_axis,
            src_dst,
            src_rel,
            src_axis,
            rel_src,
            rel_dst,
            rel_axis,
        )
    else:
        dst_ptr, dst_src, dst_rel = plan[:3]
        src_ptr, src_dst, src_rel = plan[3:6]
        rel_ptr, rel_src, rel_dst = plan[6:]
        edge_columns = (dst_src, dst_rel, src_dst, src_rel, rel_src, rel_dst)
    _expect_shape(f"{name} destination ptr", dst_ptr, (n_destination + 1,))
    _expect_shape(f"{name} source ptr", src_ptr, (n_source + 1,))
    _expect_shape(f"{name} relation ptr", rel_ptr, (n_relations + 1,))
    n_edges = int(dst_src.numel())
    if any(int(value.numel()) != n_edges for value in edge_columns):
        raise ValueError(f"every {name} plan edge column must have {n_edges} rows")


def _validate(
    tensors: tuple[Tensor, ...],
    plans: tuple[Tensor, ...],
    activation: str,
    autocast_code: int,
    eps: tuple[float, float, float, float],
) -> None:
    if activation not in _ACTIVATIONS:
        raise ValueError(
            f"activation={activation!r} is not one of {sorted(_ACTIVATIONS)}"
        )
    if autocast_code not in (0, 1, 2):
        raise ValueError(f"unknown CUDA autocast code {autocast_code}")
    if any(value <= 0.0 for value in eps):
        raise ValueError(f"LayerNorm eps values must be positive, got {eps}")
    (
        source_inv,
        source_axis,
        destination_inv,
        destination_axis,
        relation,
        inv_parameters,
        axis_parameters,
    ) = _split_floats(tensors)
    inv_plan, axis_plan = _split_plans(plans)
    if source_inv.ndim != 2 or destination_inv.ndim != 2:
        raise ValueError("invariant streams must be (N, D)")
    if (
        source_axis.ndim != 3
        or destination_axis.ndim != 3
        or source_axis.shape[1] != AXIS_CHANNELS
        or destination_axis.shape[1] != AXIS_CHANNELS
    ):
        raise ValueError(
            f"axis streams must be (N, {AXIS_CHANNELS}, D_axis)"
        )
    if source_inv.shape[0] != source_axis.shape[0]:
        raise ValueError("source invariant and axis streams disagree on row count")
    if destination_inv.shape[0] != destination_axis.shape[0]:
        raise ValueError("destination invariant and axis streams disagree on row count")
    if source_inv.shape[1] != destination_inv.shape[1]:
        raise ValueError("source and destination invariant widths must agree")
    if source_axis.shape[2] != destination_axis.shape[2]:
        raise ValueError("source and destination axis widths must agree")
    if relation.ndim != 2 or relation.shape[0] < 1 or relation.shape[1] < 1:
        raise ValueError("relation table must be (R, D_rel) with positive dimensions")
    if any(not value.is_floating_point() for value in tensors):
        raise ValueError("every message-stage state and parameter must be floating point")
    device = source_inv.device
    if any(value.device != device for value in tensors) or any(
        value.device != device for value in plans
    ):
        raise ValueError("every message-stage tensor must be on one device")
    _validate_stream(
        "invariant", inv_parameters, int(source_inv.shape[1]), int(relation.shape[1])
    )
    _validate_stream(
        "axis", axis_parameters, int(source_axis.shape[2]), int(relation.shape[1])
    )
    _validate_plan(
        "invariant",
        inv_plan,
        n_source=int(source_inv.shape[0]),
        n_destination=int(destination_inv.shape[0]),
        n_relations=int(relation.shape[0]),
        axis=False,
    )
    _validate_plan(
        "axis",
        axis_plan,
        n_source=int(source_inv.shape[0]),
        n_destination=int(destination_inv.shape[0]),
        n_relations=int(relation.shape[0]),
        axis=True,
    )


def _launch_forward(
    tensors: tuple[Tensor, ...],
    plans: tuple[Tensor, ...],
    activation: str,
    autocast_code: int,
    eps: tuple[float, float, float, float],
) -> tuple[Tensor, Tensor]:
    # The registered autograd formula owns differentiation.  Internal forward
    # history is neither used nor observable, so normalize requires_grad here
    # and avoid doubling the compiled family when inference precedes training.
    tensors = tuple(value.detach() for value in tensors)
    _mark_dynamic_inputs(tensors, plans)
    signature = _static_signature(tensors, plans)
    function = _forward_function(activation, autocast_code, eps, True, signature)
    return function(*tensors, *plans)


def _launch_backward(
    tensors: tuple[Tensor, ...],
    plans: tuple[Tensor, ...],
    activation: str,
    autocast_code: int,
    eps: tuple[float, float, float, float],
    grad_inv: Tensor,
    grad_axis: Tensor,
    *,
    compiled: bool,
) -> tuple[Tensor, ...]:
    signature: _StaticSignature = ()
    if compiled:
        # torch.func.vjp establishes its own differentiable wrapper around the
        # primals.  Incoming requires_grad flags therefore carry no semantics
        # and must not become Dynamo guards.
        tensors = tuple(value.detach() for value in tensors)
        grad_inv = grad_inv.detach()
        grad_axis = grad_axis.detach()
        gradients = (grad_inv, grad_axis)
        _mark_dynamic_inputs(tensors, plans, gradients)
        signature = _static_signature(tensors, plans, gradients)
    function = _backward_function(
        activation, autocast_code, eps, compiled, signature
    )
    return function(*tensors, *plans, grad_inv, grad_axis)


def _arguments(
    source_inv: Tensor,
    source_axis: Tensor,
    destination_inv: Tensor,
    destination_axis: Tensor,
    relation: Tensor,
    inv_source_norm_weight: Tensor,
    inv_source_norm_bias: Tensor,
    inv_value_weight: Tensor,
    inv_gate_weight: Tensor,
    inv_gate_bias: Tensor,
    inv_bias_weight: Tensor,
    inv_bias_bias: Tensor,
    inv_destination_norm_weight: Tensor,
    inv_destination_norm_bias: Tensor,
    inv_update_in_weight: Tensor,
    inv_update_in_bias: Tensor,
    inv_update_out_weight: Tensor,
    inv_update_out_bias: Tensor,
    inv_layer_scale: Tensor,
    axis_source_norm_weight: Tensor,
    axis_source_norm_bias: Tensor,
    axis_value_weight: Tensor,
    axis_gate_weight: Tensor,
    axis_gate_bias: Tensor,
    axis_bias_weight: Tensor,
    axis_bias_bias: Tensor,
    axis_destination_norm_weight: Tensor,
    axis_destination_norm_bias: Tensor,
    axis_update_in_weight: Tensor,
    axis_update_in_bias: Tensor,
    axis_update_out_weight: Tensor,
    axis_update_out_bias: Tensor,
    axis_layer_scale: Tensor,
) -> tuple[Tensor, ...]:
    return (
        source_inv,
        source_axis,
        destination_inv,
        destination_axis,
        relation,
        inv_source_norm_weight,
        inv_source_norm_bias,
        inv_value_weight,
        inv_gate_weight,
        inv_gate_bias,
        inv_bias_weight,
        inv_bias_bias,
        inv_destination_norm_weight,
        inv_destination_norm_bias,
        inv_update_in_weight,
        inv_update_in_bias,
        inv_update_out_weight,
        inv_update_out_bias,
        inv_layer_scale,
        axis_source_norm_weight,
        axis_source_norm_bias,
        axis_value_weight,
        axis_gate_weight,
        axis_gate_bias,
        axis_bias_weight,
        axis_bias_bias,
        axis_destination_norm_weight,
        axis_destination_norm_bias,
        axis_update_in_weight,
        axis_update_in_bias,
        axis_update_out_weight,
        axis_update_out_bias,
        axis_layer_scale,
    )


@torch.library.custom_op("mantisnet::act_message_stage", mutates_args=())
def _message_stage_op(
    source_inv: Tensor,
    source_axis: Tensor,
    destination_inv: Tensor,
    destination_axis: Tensor,
    relation: Tensor,
    inv_source_norm_weight: Tensor,
    inv_source_norm_bias: Tensor,
    inv_value_weight: Tensor,
    inv_gate_weight: Tensor,
    inv_gate_bias: Tensor,
    inv_bias_weight: Tensor,
    inv_bias_bias: Tensor,
    inv_destination_norm_weight: Tensor,
    inv_destination_norm_bias: Tensor,
    inv_update_in_weight: Tensor,
    inv_update_in_bias: Tensor,
    inv_update_out_weight: Tensor,
    inv_update_out_bias: Tensor,
    inv_layer_scale: Tensor,
    axis_source_norm_weight: Tensor,
    axis_source_norm_bias: Tensor,
    axis_value_weight: Tensor,
    axis_gate_weight: Tensor,
    axis_gate_bias: Tensor,
    axis_bias_weight: Tensor,
    axis_bias_bias: Tensor,
    axis_destination_norm_weight: Tensor,
    axis_destination_norm_bias: Tensor,
    axis_update_in_weight: Tensor,
    axis_update_in_bias: Tensor,
    axis_update_out_weight: Tensor,
    axis_update_out_bias: Tensor,
    axis_layer_scale: Tensor,
    plans: list[Tensor],
    activation: str,
    autocast_code: int,
    inv_source_eps: float,
    inv_destination_eps: float,
    axis_source_eps: float,
    axis_destination_eps: float,
) -> tuple[Tensor, Tensor]:
    tensors = _arguments(
        source_inv,
        source_axis,
        destination_inv,
        destination_axis,
        relation,
        inv_source_norm_weight,
        inv_source_norm_bias,
        inv_value_weight,
        inv_gate_weight,
        inv_gate_bias,
        inv_bias_weight,
        inv_bias_bias,
        inv_destination_norm_weight,
        inv_destination_norm_bias,
        inv_update_in_weight,
        inv_update_in_bias,
        inv_update_out_weight,
        inv_update_out_bias,
        inv_layer_scale,
        axis_source_norm_weight,
        axis_source_norm_bias,
        axis_value_weight,
        axis_gate_weight,
        axis_gate_bias,
        axis_bias_weight,
        axis_bias_bias,
        axis_destination_norm_weight,
        axis_destination_norm_bias,
        axis_update_in_weight,
        axis_update_in_bias,
        axis_update_out_weight,
        axis_update_out_bias,
        axis_layer_scale,
    )
    tensors = tuple(value.contiguous() for value in tensors)
    plan_tuple = tuple(value.contiguous() for value in plans)
    eps = (
        inv_source_eps,
        inv_destination_eps,
        axis_source_eps,
        axis_destination_eps,
    )
    _validate(tensors, plan_tuple, activation, autocast_code, eps)
    reference = lambda: _reference(  # noqa: E731
        tensors,
        plan_tuple,
        activation=activation,
        autocast_code=autocast_code,
        eps=eps,
        fused_segment=False,
    )
    if not _supported(tensors, plan_tuple, activation, autocast_code):
        return reference()
    _LAUNCH_STATS["forward_eligible"] += 1
    key = _shape_key(tensors, plan_tuple, activation, autocast_code, eps)
    if key in _FAILED_FORWARD_SHAPES:
        raise MessageStageCompilationError(
            "eligible CUDA relation-gated message forward previously failed "
            "to compile; refusing to silently de-fuse the trunk: "
            f"{_FAILED_FORWARD_SHAPES[key]}"
        )
    try:
        result = _launch_forward(tensors, plan_tuple, activation, autocast_code, eps)
        _LAUNCH_STATS["forward_launched"] += 1
        return result
    except Exception as exc:
        _FAILED_FORWARD_SHAPES[key] = f"{type(exc).__name__}: {exc}"
        raise MessageStageCompilationError(
            "eligible CUDA relation-gated message forward failed to compile; "
            "refusing to silently de-fuse the trunk: "
            f"{_FAILED_FORWARD_SHAPES[key]}"
        ) from exc


@_message_stage_op.register_fake
def _(
    source_inv: Tensor,
    source_axis: Tensor,
    destination_inv: Tensor,
    destination_axis: Tensor,
    relation: Tensor,
    inv_source_norm_weight: Tensor,
    inv_source_norm_bias: Tensor,
    inv_value_weight: Tensor,
    inv_gate_weight: Tensor,
    inv_gate_bias: Tensor,
    inv_bias_weight: Tensor,
    inv_bias_bias: Tensor,
    inv_destination_norm_weight: Tensor,
    inv_destination_norm_bias: Tensor,
    inv_update_in_weight: Tensor,
    inv_update_in_bias: Tensor,
    inv_update_out_weight: Tensor,
    inv_update_out_bias: Tensor,
    inv_layer_scale: Tensor,
    axis_source_norm_weight: Tensor,
    axis_source_norm_bias: Tensor,
    axis_value_weight: Tensor,
    axis_gate_weight: Tensor,
    axis_gate_bias: Tensor,
    axis_bias_weight: Tensor,
    axis_bias_bias: Tensor,
    axis_destination_norm_weight: Tensor,
    axis_destination_norm_bias: Tensor,
    axis_update_in_weight: Tensor,
    axis_update_in_bias: Tensor,
    axis_update_out_weight: Tensor,
    axis_update_out_bias: Tensor,
    axis_layer_scale: Tensor,
    plans: list[Tensor],
    activation: str,
    autocast_code: int,
    inv_source_eps: float,
    inv_destination_eps: float,
    axis_source_eps: float,
    axis_destination_eps: float,
) -> tuple[Tensor, Tensor]:
    del (
        source_inv,
        source_axis,
        relation,
        inv_source_norm_weight,
        inv_source_norm_bias,
        inv_value_weight,
        inv_gate_weight,
        inv_gate_bias,
        inv_bias_weight,
        inv_bias_bias,
        inv_destination_norm_weight,
        inv_destination_norm_bias,
        inv_update_in_weight,
        inv_update_in_bias,
        inv_update_out_weight,
        inv_update_out_bias,
        inv_layer_scale,
        axis_source_norm_weight,
        axis_source_norm_bias,
        axis_value_weight,
        axis_gate_weight,
        axis_gate_bias,
        axis_bias_weight,
        axis_bias_bias,
        axis_destination_norm_weight,
        axis_destination_norm_bias,
        axis_update_in_weight,
        axis_update_in_bias,
        axis_update_out_weight,
        axis_update_out_bias,
        axis_layer_scale,
        plans,
        activation,
        autocast_code,
        inv_source_eps,
        inv_destination_eps,
        axis_source_eps,
        axis_destination_eps,
    )
    return (
        torch.empty_like(destination_inv),
        torch.empty_like(destination_axis),
    )


_MessageGradientTuple = tuple[
    Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
    Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
    Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
    Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
]


@torch.library.custom_op("mantisnet::act_message_stage_backward", mutates_args=())
def _message_stage_backward_op(
    source_inv: Tensor,
    source_axis: Tensor,
    destination_inv: Tensor,
    destination_axis: Tensor,
    relation: Tensor,
    inv_source_norm_weight: Tensor,
    inv_source_norm_bias: Tensor,
    inv_value_weight: Tensor,
    inv_gate_weight: Tensor,
    inv_gate_bias: Tensor,
    inv_bias_weight: Tensor,
    inv_bias_bias: Tensor,
    inv_destination_norm_weight: Tensor,
    inv_destination_norm_bias: Tensor,
    inv_update_in_weight: Tensor,
    inv_update_in_bias: Tensor,
    inv_update_out_weight: Tensor,
    inv_update_out_bias: Tensor,
    inv_layer_scale: Tensor,
    axis_source_norm_weight: Tensor,
    axis_source_norm_bias: Tensor,
    axis_value_weight: Tensor,
    axis_gate_weight: Tensor,
    axis_gate_bias: Tensor,
    axis_bias_weight: Tensor,
    axis_bias_bias: Tensor,
    axis_destination_norm_weight: Tensor,
    axis_destination_norm_bias: Tensor,
    axis_update_in_weight: Tensor,
    axis_update_in_bias: Tensor,
    axis_update_out_weight: Tensor,
    axis_update_out_bias: Tensor,
    axis_layer_scale: Tensor,
    plans: list[Tensor],
    activation: str,
    autocast_code: int,
    inv_source_eps: float,
    inv_destination_eps: float,
    axis_source_eps: float,
    axis_destination_eps: float,
    grad_inv: Tensor,
    grad_axis: Tensor,
) -> _MessageGradientTuple:
    tensors = _arguments(
        source_inv,
        source_axis,
        destination_inv,
        destination_axis,
        relation,
        inv_source_norm_weight,
        inv_source_norm_bias,
        inv_value_weight,
        inv_gate_weight,
        inv_gate_bias,
        inv_bias_weight,
        inv_bias_bias,
        inv_destination_norm_weight,
        inv_destination_norm_bias,
        inv_update_in_weight,
        inv_update_in_bias,
        inv_update_out_weight,
        inv_update_out_bias,
        inv_layer_scale,
        axis_source_norm_weight,
        axis_source_norm_bias,
        axis_value_weight,
        axis_gate_weight,
        axis_gate_bias,
        axis_bias_weight,
        axis_bias_bias,
        axis_destination_norm_weight,
        axis_destination_norm_bias,
        axis_update_in_weight,
        axis_update_in_bias,
        axis_update_out_weight,
        axis_update_out_bias,
        axis_layer_scale,
    )
    tensors = tuple(value.contiguous() for value in tensors)
    plan_tuple = tuple(value.contiguous() for value in plans)
    eps = (
        inv_source_eps,
        inv_destination_eps,
        axis_source_eps,
        axis_destination_eps,
    )
    reference = lambda: _launch_backward(  # noqa: E731
        tensors,
        plan_tuple,
        activation,
        autocast_code,
        eps,
        grad_inv.contiguous(),
        grad_axis.contiguous(),
        compiled=False,
    )
    if not _supported(tensors, plan_tuple, activation, autocast_code):
        return reference()
    _LAUNCH_STATS["backward_eligible"] += 1
    key = (*_shape_key(tensors, plan_tuple, activation, autocast_code, eps), "backward")
    if key in _FAILED_BACKWARD_SHAPES:
        raise MessageStageCompilationError(
            "eligible CUDA relation-gated message backward previously failed "
            "to compile; refusing to silently de-fuse the trunk: "
            f"{_FAILED_BACKWARD_SHAPES[key]}"
        )
    try:
        result = _launch_backward(
            tensors,
            plan_tuple,
            activation,
            autocast_code,
            eps,
            grad_inv.contiguous(),
            grad_axis.contiguous(),
            compiled=True,
        )
        _LAUNCH_STATS["backward_launched"] += 1
        return result
    except Exception as exc:
        _FAILED_BACKWARD_SHAPES[key] = f"{type(exc).__name__}: {exc}"
        raise MessageStageCompilationError(
            "eligible CUDA relation-gated message backward failed to compile; "
            "refusing to silently de-fuse the trunk: "
            f"{_FAILED_BACKWARD_SHAPES[key]}"
        ) from exc


@_message_stage_backward_op.register_fake
def _(
    source_inv: Tensor,
    source_axis: Tensor,
    destination_inv: Tensor,
    *args,
) -> _MessageGradientTuple:
    # The remaining thirty tensors precede the plan list and scalar settings.
    tensors = (source_inv, source_axis, destination_inv, *args[:30])
    return tuple(torch.empty_like(value) for value in tensors)


def _setup_context(ctx, inputs, output) -> None:
    floats = inputs[:_FLOAT_COUNT]
    plans = inputs[_FLOAT_COUNT]
    ctx.settings = inputs[_FLOAT_COUNT + 1 :]
    ctx.plan_count = len(plans)
    ctx.save_for_backward(*floats, *plans)


def _dispatch_backward(ctx, grad_inv: Tensor, grad_axis: Tensor):
    saved = ctx.saved_tensors
    floats = saved[:_FLOAT_COUNT]
    plans = list(saved[_FLOAT_COUNT : _FLOAT_COUNT + ctx.plan_count])
    gradients = _message_stage_backward_op(
        *floats, plans, *ctx.settings, grad_inv, grad_axis
    )
    # Tensor-list inputs retain their pytree structure in a custom operator's
    # autograd contract.  The CSR tensors are integers, so every leaf is
    # nondifferentiable, but the list itself must still be returned.
    return (
        *gradients,
        [None for _ in range(ctx.plan_count)],
        *(None for _ in ctx.settings),
    )


_message_stage_op.register_autograd(_dispatch_backward, setup_context=_setup_context)


def _plan_tensors(inv_plan: MessagePlan, axis_plan: MessagePlan) -> list[Tensor]:
    if inv_plan.channels != 1:
        raise ValueError(f"invariant plan must have one channel, got {inv_plan.channels}")
    if axis_plan.channels != AXIS_CHANNELS:
        raise ValueError(
            f"axis plan must have {AXIS_CHANNELS} channels, got {axis_plan.channels}"
        )
    if any(
        value is not None
        for value in (inv_plan.dst_axis, inv_plan.src_axis, inv_plan.rel_axis)
    ):
        raise ValueError("an invariant message plan must not carry axis columns")
    if any(value is None for value in (axis_plan.dst_axis, axis_plan.src_axis, axis_plan.rel_axis)):
        raise ValueError("a routed-axis message plan must carry all axis columns")
    return [
        inv_plan.dst_ptr,
        inv_plan.dst_src,
        inv_plan.dst_rel,
        inv_plan.src_ptr,
        inv_plan.src_dst,
        inv_plan.src_rel,
        inv_plan.rel_ptr,
        inv_plan.rel_src,
        inv_plan.rel_dst,
        axis_plan.dst_ptr,
        axis_plan.dst_src,
        axis_plan.dst_rel,
        axis_plan.dst_axis,
        axis_plan.src_ptr,
        axis_plan.src_dst,
        axis_plan.src_rel,
        axis_plan.src_axis,
        axis_plan.rel_ptr,
        axis_plan.rel_src,
        axis_plan.rel_dst,
        axis_plan.rel_axis,
    ]


def relation_gated_message_stage(
    source_inv: Tensor,
    source_axis: Tensor,
    destination_inv: Tensor,
    destination_axis: Tensor,
    relation: Tensor,
    inv_parameters: Sequence[Tensor],
    axis_parameters: Sequence[Tensor],
    inv_plan: MessagePlan,
    axis_plan: MessagePlan,
    *,
    activation: str = "silu",
    source_eps: tuple[float, float] = (1e-5, 1e-5),
    destination_eps: tuple[float, float] = (1e-5, 1e-5),
) -> tuple[Tensor, Tensor]:
    """Run both streams of the default §14 pre-norm residual message stage.

    Each parameter sequence follows its owning ``RelationGatedMessage`` stream:
    source norm weight/bias, value weight, gate weight/bias, bias weight/bias,
    destination norm weight/bias, update input weight/bias, update output
    weight/bias, and LayerScale gamma.  Passing existing parameters directly
    preserves state-dict names and optimizer identity.
    """
    inv_parameters = tuple(inv_parameters)
    axis_parameters = tuple(axis_parameters)
    if len(inv_parameters) != _STREAM_PARAMETER_COUNT:
        raise ValueError(
            f"inv_parameters must carry {_STREAM_PARAMETER_COUNT} tensors, "
            f"got {len(inv_parameters)}"
        )
    if len(axis_parameters) != _STREAM_PARAMETER_COUNT:
        raise ValueError(
            f"axis_parameters must carry {_STREAM_PARAMETER_COUNT} tensors, "
            f"got {len(axis_parameters)}"
        )
    if len(source_eps) != 2 or len(destination_eps) != 2:
        raise ValueError("source_eps and destination_eps must each contain inv and axis")
    plans = _plan_tensors(inv_plan, axis_plan)
    return _message_stage_op(
        source_inv.contiguous(),
        source_axis.contiguous(),
        destination_inv.contiguous(),
        destination_axis.contiguous(),
        relation.contiguous(),
        *(value.contiguous() for value in inv_parameters),
        *(value.contiguous() for value in axis_parameters),
        [value.contiguous() for value in plans],
        activation,
        _autocast_code(source_inv),
        float(source_eps[0]),
        float(destination_eps[0]),
        float(source_eps[1]),
        float(destination_eps[1]),
    )


__all__ = [
    "MessageStageCompilationError",
    "clear_compile_caches",
    "clear_failure_caches",
    "launch_stats",
    "relation_gated_message_stage",
    "reset_launch_stats",
]
