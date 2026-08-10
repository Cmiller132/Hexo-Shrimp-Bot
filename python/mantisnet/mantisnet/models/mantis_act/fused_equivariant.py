"""Compiled whole-stage AxisMix, FFN, and optional phase FiLM.

The default ACT block evaluates three densely connected, pre-norm stages in
sequence.  Expressing those stages as ordinary modules asks eager dispatch to
launch every norm, pointwise expression, cast, and activation separately.  The
operator in this file presents the sequence as one registered operation while
letting Inductor choose the Triton and GEMM decomposition inside that boundary.

Backward deliberately saves only the inputs and parameters.  Its registered
autograd callback invokes a shape-polymorphic compiled ``torch.func.grad`` of
the literal stage, so the original forward's intermediates are recomputed
rather than saved.  This is the same memory-for-arithmetic trade used by the
hand-written ACT kernels, without maintaining a forty-one-gradient derivation
by hand.  The functional transform is important: a registered autograd
formula is entered with GradMode disabled, and a compiled forward invoked from
that context can be lowered as an inference graph even inside
``torch.enable_grad``.  Functorch establishes its own differentiation
dispatch and lets Dynamo/Inductor compile the complete VJP directly.

The public ABI takes the parameters of the existing ``AxisMix``,
``EquivariantFFN``, and ``PhaseFiLM`` modules.  It owns no parameters, so using
it cannot alter state-dict names.  ``dropout == 0`` is required by the caller;
recomputation cannot reproduce an eager dropout mask without making that mask
an operator input.  The default ``full_act_v4`` configuration has zero dropout.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
from types import FunctionType
from typing import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


_AXES = 3
_MIX_COUNT = 18
_FFN_COUNT = 14
_FILM_COUNT = 7
_BASE_TENSOR_COUNT = 2 + _MIX_COUNT + _FFN_COUNT
_PHASE_ID_INDEX = _BASE_TENSOR_COUNT
_PHASE_ROW_INDEX = _PHASE_ID_INDEX + 1
_FILM_START = _PHASE_ROW_INDEX + 1
_TENSOR_COUNT = _FILM_START + _FILM_COUNT
_NONDIFFERENTIABLE = frozenset({_PHASE_ID_INDEX, _PHASE_ROW_INDEX})
_FULL_GRAD_INDICES = tuple(
    index for index in range(_TENSOR_COUNT) if index not in _NONDIFFERENTIABLE
)
_BASE_GRAD_INDICES = tuple(range(_BASE_TENSOR_COUNT))

_ACTIVATIONS = frozenset({"silu", "gelu", "relu"})
_CUDA_DTYPES = frozenset({torch.float16, torch.bfloat16, torch.float32})
_COMPILE_OPTIONS = {
    # Inductor normally removes a bf16-downcast/fp32-upcast pair when adjacent
    # pointwise operations fuse.  Eager autocast materialises that boundary;
    # preserving it keeps this optimisation function-preserving rather than
    # merely closer to an fp64 answer.
    "emulate_precision_casts": True,
}

# Compile failures are retained only as diagnostics for the device-law tests.
# They are never used as a fallback cache: an eligible CUDA stage is part of
# the model's performance contract and must fail loudly rather than silently
# replacing the fused trunk with its literal formulation.
_FAILED_FORWARD_SHAPES: dict[tuple[object, ...], str] = {}
_FAILED_BACKWARD_SHAPES: dict[tuple[object, ...], str] = {}

_LAUNCH_STATS = {
    "forward_eligible": 0,
    "forward_launched": 0,
    "backward_eligible": 0,
    "backward_launched": 0,
}


class EquivariantStageFusionError(RuntimeError):
    """An eligible CUDA equivariant stage could not stay on its fused path."""


def reset_launch_stats() -> None:
    """Reset successful-launch accounting used by the CUDA device law."""
    for name in _LAUNCH_STATS:
        _LAUNCH_STATS[name] = 0


def launch_stats() -> dict[str, int]:
    """A copy of the supported-versus-successful launch counters."""
    return dict(_LAUNCH_STATS)


def clear_failure_caches() -> None:
    """Clear compile diagnostics without perturbing warm compiled callables."""
    _FAILED_FORWARD_SHAPES.clear()
    _FAILED_BACKWARD_SHAPES.clear()


def clear_compile_caches() -> None:
    """Discard compiled architecture variants so a test can warm from zero."""
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


def _axis_linear(value: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """Apply a linear map to ``(N, 3, D)`` without concretising ``N``.

    Functorch's generated Linear backward flattens all leading batch
    dimensions using their hinted concrete product.  Doing that reshape in
    the primal with an inferred first dimension keeps the packed entity count
    symbolic in the compiled functional VJP.
    """
    flat = value.reshape(-1, value.shape[-1])
    output = F.linear(flat, weight, bias)
    return output.reshape(-1, _AXES, output.shape[-1])


def _fresh_dynamo_frame(
    function: Callable,
    *,
    kind: str,
    identity: tuple[object, ...],
) -> Callable:
    """Clone a function frame so finite static variants cannot share guards.

    Dynamo's recompile accounting is attached to the Python code object, not
    merely to the callable returned by :func:`torch.compile`.  The closures
    created by one cached factory otherwise all share one code object, so a
    tiny-width oracle test can consume the production architecture's budget.
    Each factory cache entry is a genuine static regime and gets one frame;
    dynamic row shapes continue to reuse that frame's first graph.
    """
    if not isinstance(function, FunctionType):
        raise TypeError(f"expected a Python function, got {type(function)!r}")
    suffix = hashlib.sha256(repr(identity).encode("utf-8")).hexdigest()[:12]
    name = f"_equivariant_{kind}_{suffix}"
    cloned = FunctionType(
        function.__code__.replace(co_name=name, co_qualname=name),
        function.__globals__,
        name,
        function.__defaults__,
        function.__closure__,
    )
    cloned.__kwdefaults__ = function.__kwdefaults__
    return cloned


def _unpack(
    tensors: tuple[Tensor, ...],
) -> tuple[
    Tensor,
    Tensor,
    tuple[Tensor, ...],
    tuple[Tensor, ...],
    Tensor,
    Tensor,
    tuple[Tensor, ...],
]:
    inv, axis = tensors[:2]
    mix_start = 2
    ffn_start = mix_start + _MIX_COUNT
    mix = tensors[mix_start:ffn_start]
    ffn = tensors[ffn_start:_BASE_TENSOR_COUNT]
    phase_id = tensors[_PHASE_ID_INDEX]
    phase_row = tensors[_PHASE_ROW_INDEX]
    film = tensors[_FILM_START:]
    return inv, axis, mix, ffn, phase_id, phase_row, film


def _reference(
    tensors: tuple[Tensor, ...],
    *,
    activation: str,
    use_film: bool,
    eps: tuple[float, float, float, float],
) -> tuple[Tensor, Tensor]:
    """Literal eager equations from §§12.4, 13.2, and 18."""
    inv, axis, mix, ffn, phase_id, phase_row, film = _unpack(tensors)
    (
        mix_inv_norm_weight,
        mix_inv_norm_bias,
        mix_axis_norm_weight,
        mix_axis_norm_bias,
        mix_inv_to_axis_weight,
        mix_inv_to_axis_bias,
        mix_axis_in_weight,
        mix_axis_in_bias,
        mix_axis_out_weight,
        mix_axis_out_bias,
        mix_phi_weight,
        mix_phi_bias,
        mix_inv_in_weight,
        mix_inv_in_bias,
        mix_inv_out_weight,
        mix_inv_out_bias,
        mix_inv_gamma,
        mix_axis_gamma,
    ) = mix
    (
        ffn_inv_norm_weight,
        ffn_inv_norm_bias,
        ffn_axis_norm_weight,
        ffn_axis_norm_bias,
        ffn_inv_in_weight,
        ffn_inv_in_bias,
        ffn_inv_out_weight,
        ffn_inv_out_bias,
        ffn_axis_in_weight,
        ffn_axis_in_bias,
        ffn_axis_out_weight,
        ffn_axis_out_bias,
        ffn_inv_gamma,
        ffn_axis_gamma,
    ) = ffn

    # AxisMix: both branches read one common pre-update normalised state.
    z_inv = F.layer_norm(
        inv,
        (inv.shape[-1],),
        mix_inv_norm_weight,
        mix_inv_norm_bias,
        eps[0],
    )
    u = F.layer_norm(
        axis,
        (axis.shape[-1],),
        mix_axis_norm_weight,
        mix_axis_norm_bias,
        eps[1],
    )
    total = u.sum(dim=-2, keepdim=True)
    other = (total - u) / (_AXES - 1)
    context = F.linear(
        z_inv, mix_inv_to_axis_weight, mix_inv_to_axis_bias
    ).unsqueeze(-2)
    context = context.expand_as(u)
    axis_hidden = _activation(
        _axis_linear(
            torch.cat((u, other, context), dim=-1),
            mix_axis_in_weight,
            mix_axis_in_bias,
        ),
        activation,
    )
    delta_axis = _axis_linear(
        axis_hidden, mix_axis_out_weight, mix_axis_out_bias
    )

    axis_summary = _activation(
        _axis_linear(u, mix_phi_weight, mix_phi_bias), activation
    ).mean(dim=-2)
    inv_hidden = _activation(
        F.linear(
            torch.cat((z_inv, axis_summary), dim=-1),
            mix_inv_in_weight,
            mix_inv_in_bias,
        ),
        activation,
    )
    delta_inv = F.linear(inv_hidden, mix_inv_out_weight, mix_inv_out_bias)
    mixed_inv = inv + mix_inv_gamma * delta_inv
    mixed_axis = axis + mix_axis_gamma * delta_axis

    # The entity FFN owns a second pair of norms and LayerScale parameters.
    ffn_z_inv = F.layer_norm(
        mixed_inv,
        (mixed_inv.shape[-1],),
        ffn_inv_norm_weight,
        ffn_inv_norm_bias,
        eps[2],
    )
    ffn_z_axis = F.layer_norm(
        mixed_axis,
        (mixed_axis.shape[-1],),
        ffn_axis_norm_weight,
        ffn_axis_norm_bias,
        eps[3],
    )
    ffn_delta_inv = F.linear(
        _activation(
            F.linear(ffn_z_inv, ffn_inv_in_weight, ffn_inv_in_bias), activation
        ),
        ffn_inv_out_weight,
        ffn_inv_out_bias,
    )
    ffn_delta_axis = _axis_linear(
        _activation(
            _axis_linear(
                ffn_z_axis, ffn_axis_in_weight, ffn_axis_in_bias
            ),
            activation,
        ),
        ffn_axis_out_weight,
        ffn_axis_out_bias,
    )
    out_inv = mixed_inv + ffn_inv_gamma * ffn_delta_inv
    out_axis = mixed_axis + ffn_axis_gamma * ffn_delta_axis

    if not use_film:
        return out_inv, out_axis

    (
        film_embed_weight,
        film_mlp_weight,
        film_mlp_bias,
        film_inv_weight,
        film_inv_bias,
        film_axis_weight,
        film_axis_bias,
    ) = film
    # ``phase_row`` is PhaseFiLM's fixed 3-way-to-table mapping.  index_select
    # retains its bounds-checking semantics, including rejection of negatives.
    rows = phase_row.index_select(0, phase_id)
    code = _activation(
        F.linear(film_embed_weight, film_mlp_weight, film_mlp_bias), activation
    )
    inv_table = F.linear(code, film_inv_weight, film_inv_bias)
    # Build the selector without ``one_hot``: its fake-tensor backward asks for
    # a data-dependent class-count specialization when the entity row count is
    # symbolic.  ``index_select`` keeps one_hot's negative/out-of-range error
    # semantics, while equality produces the same deterministic dense selector
    # and therefore the same ordered table-gradient matmul.
    phase_columns = torch.arange(
        film_embed_weight.shape[0], device=rows.device, dtype=rows.dtype
    )
    checked_rows = phase_columns.index_select(0, rows)
    selector = checked_rows.unsqueeze(-1).eq(phase_columns).to(inv_table.dtype)
    inv_scale, inv_bias = (selector @ inv_table).chunk(2, dim=-1)
    out_inv = (1 + inv_scale) * out_inv + inv_bias

    axis_table = F.linear(code, film_axis_weight, film_axis_bias)
    axis_scale, axis_bias = (selector @ axis_table).chunk(2, dim=-1)
    out_axis = (
        (1 + axis_scale).unsqueeze(-2) * out_axis
        + axis_bias.unsqueeze(-2)
    )
    return out_inv, out_axis


def _autocast_dtype(code: int) -> torch.dtype | None:
    if code == 0:
        return None
    if code == 1:
        return torch.float16
    if code == 2:
        return torch.bfloat16
    raise ValueError(f"unknown CUDA autocast code {code}")


def _run_reference(
    tensors: tuple[Tensor, ...],
    *,
    activation: str,
    use_film: bool,
    autocast_code: int,
    eps: tuple[float, float, float, float],
) -> tuple[Tensor, Tensor]:
    dtype = _autocast_dtype(autocast_code)
    if dtype is None:
        return _reference(
            tensors, activation=activation, use_film=use_film, eps=eps
        )
    # custom-op implementations execute below their dispatch key.  Re-enter
    # autocast explicitly so fp32 module parameters have the same linear and
    # LayerNorm policy as the eager modules they replace.
    with torch.autocast("cuda", dtype=dtype):
        return _reference(
            tensors, activation=activation, use_film=use_film, eps=eps
        )


def _loss_function(
    activation: str,
    use_film: bool,
    autocast_code: int,
    eps: tuple[float, float, float, float],
) -> Callable[..., Tensor]:
    def loss(*args: Tensor) -> Tensor:
        tensors = tuple(args[:_TENSOR_COUNT])
        grad_inv, grad_axis = args[_TENSOR_COUNT:]
        out_inv, out_axis = _run_reference(
            tensors,
            activation=activation,
            use_film=use_film,
            autocast_code=autocast_code,
            eps=eps,
        )
        # A scalar dot product is the VJP seed torch.func.grad expects.
        return (out_inv * grad_inv).sum() + (out_axis * grad_axis).sum()

    return loss


@lru_cache(maxsize=None)
def _forward_function(
    activation: str,
    use_film: bool,
    autocast_code: int,
    eps: tuple[float, float, float, float],
    compiled: bool,
    architecture: tuple[object, ...] = (),
) -> Callable[..., tuple[Tensor, Tensor]]:
    # ``architecture`` selects both this cache entry and its isolated Dynamo
    # frame.  LayerNorm shapes and GEMM contraction widths are truly static;
    # asking one frame to learn every test/ablation width consumes its
    # recompile budget.  Row counts are the dynamic dimensions.
    def forward(*tensors: Tensor) -> tuple[Tensor, Tensor]:
        return _run_reference(
            tuple(tensors),
            activation=activation,
            use_film=use_film,
            autocast_code=autocast_code,
            eps=eps,
        )

    if not compiled:
        return forward
    return torch.compile(
        _fresh_dynamo_frame(
            forward,
            kind="forward",
            identity=(
                activation,
                use_film,
                autocast_code,
                eps,
                architecture,
            ),
        ),
        fullgraph=True,
        dynamic=True,
        options=_COMPILE_OPTIONS,
    )


@lru_cache(maxsize=None)
def _backward_function(
    activation: str,
    use_film: bool,
    autocast_code: int,
    eps: tuple[float, float, float, float],
    compiled: bool,
    architecture: tuple[object, ...] = (),
) -> Callable[..., tuple[Tensor, ...]]:
    indices = _FULL_GRAD_INDICES if use_film else _BASE_GRAD_INDICES
    transform = torch.func.grad(
        _loss_function(activation, use_film, autocast_code, eps),
        argnums=indices,
    )
    if not compiled:
        return transform
    return torch.compile(
        _fresh_dynamo_frame(
            transform,
            kind="backward",
            identity=(
                activation,
                use_film,
                autocast_code,
                eps,
                architecture,
            ),
        ),
        fullgraph=True,
        dynamic=True,
        options=_COMPILE_OPTIONS,
    )


def _shape_key(
    tensors: tuple[Tensor, ...],
    activation: str,
    use_film: bool,
    autocast_code: int,
    eps: tuple[float, float, float, float],
) -> tuple[object, ...]:
    inv, axis = tensors[:2]
    film_embed = tensors[_FILM_START]
    return (
        inv.device.type,
        inv.device.index,
        inv.dtype,
        axis.dtype,
        int(inv.shape[-1]),
        int(axis.shape[-1]),
        int(tensors[8].shape[0]),
        int(tensors[14].shape[0]),
        int(tensors[24].shape[0]),
        int(tensors[28].shape[0]),
        int(film_embed.shape[0]) if use_film else 0,
        int(film_embed.shape[-1]) if use_film else 0,
        # Symbolic-shape rules intentionally specialize singleton dimensions
        # because their stride relations differ.  Give that finite regime its
        # own callable instead of allowing it to trigger a later recompile of
        # the general N>=2 graph.  Empty activations are not CUDA-eligible.
        "singleton" if int(inv.shape[0]) == 1 else "general-rows",
        activation,
        use_film,
        autocast_code,
        *eps,
    )


def _compile_key(
    tensors: tuple[Tensor, ...],
    activation: str,
    use_film: bool,
    autocast_code: int,
    eps: tuple[float, float, float, float],
    gradients: tuple[Tensor, Tensor] | None = None,
) -> tuple[object, ...]:
    """Static Dynamo regimes; activation row counts are intentionally absent."""
    key: tuple[object, ...] = (
        *_shape_key(tensors, activation, use_film, autocast_code, eps),
        "grad-enabled",
        torch.is_grad_enabled(),
        "input-autograd-regime",
        *((type(tensor), tensor.requires_grad) for tensor in tensors),
    )
    if gradients is not None:
        key = (
            *key,
            "cotangent-regime",
            *((tensor.dtype, tensor.requires_grad) for tensor in gradients),
        )
    return key


def _mark_dynamic_rows(
    tensors: tuple[Tensor, ...],
    use_film: bool,
    gradients: tuple[Tensor, Tensor] | None = None,
) -> None:
    """Declare every activation-row dimension before entering Dynamo.

    ``dynamic=True`` permits symbolic shapes, but it does not communicate that
    the independently supplied invariant, axis, phase, and cotangent tensors
    all vary together across packed chunks.  In particular, Dynamo may retain
    the first concrete hint (and always treats 0/1 specially) until a later
    call forces another graph.  Explicit declarations make the first graph
    polymorphic across the row counts seen by cells, windows, and actions.

    The stage supports only non-empty CUDA activations, hence the lower bound.
    FiLM's phase vector and backward cotangents share that same row count.  The
    fixed-width module parameters deliberately remain static.
    """
    def mark_row_tensor(tensor: Tensor) -> None:
        # Dynamo intentionally specializes a size-one dimension.  Its caller
        # is already isolated by the singleton compile-key bucket, so declaring
        # it dynamic would add an unsatisfiable constraint on some releases.
        if int(tensor.shape[0]) == 1:
            torch._dynamo.mark_static(tensor, 0)
        else:
            torch._dynamo.mark_dynamic(tensor, 0)
        if tensor.ndim > 1:
            torch._dynamo.mark_static(tensor, tuple(range(1, tensor.ndim)))

    row_indices = {0, 1}
    if use_film:
        row_indices.add(_PHASE_ID_INDEX)
    for index, tensor in enumerate(tensors):
        if index in row_indices:
            mark_row_tensor(tensor)
        else:
            # dynamic=True must not turn architecture widths (notably FiLM's
            # one_hot class count and LayerNorm widths) into symbols.
            torch._dynamo.mark_static(tensor)
    if gradients is not None:
        mark_row_tensor(gradients[0])
        mark_row_tensor(gradients[1])


def _supported(
    tensors: tuple[Tensor, ...], activation: str, use_film: bool
) -> bool:
    inv, axis = tensors[:2]
    parameter_indices = tuple(range(2, _BASE_TENSOR_COUNT))
    if use_film:
        parameter_indices = (
            *parameter_indices,
            *range(_FILM_START, _TENSOR_COUNT),
        )
    return (
        inv.is_cuda
        and axis.is_cuda
        and inv.dtype in _CUDA_DTYPES
        and axis.dtype in _CUDA_DTYPES
        and all(tensors[index].dtype == torch.float32 for index in parameter_indices)
        and activation in _ACTIVATIONS
        and inv.shape[0] > 0
    )


def _expect_shape(name: str, tensor: Tensor, expected: tuple[int, ...]) -> None:
    if tuple(tensor.shape) != expected:
        raise ValueError(f"{name} must be {expected}, got {tuple(tensor.shape)}")


def _validate(
    tensors: tuple[Tensor, ...], activation: str, use_film: bool
) -> None:
    if len(tensors) != _TENSOR_COUNT:
        raise ValueError(
            f"equivariant stage expects {_TENSOR_COUNT} tensor inputs, "
            f"got {len(tensors)}"
        )
    if activation not in _ACTIVATIONS:
        raise ValueError(
            f"activation={activation!r} is not one of {sorted(_ACTIVATIONS)}"
        )
    inv, axis, mix, ffn, phase_id, phase_row, film = _unpack(tensors)
    if inv.ndim != 2:
        raise ValueError(f"inv must be (N, D), got {tuple(inv.shape)}")
    if axis.ndim != 3 or axis.shape[0] != inv.shape[0] or axis.shape[1] != _AXES:
        raise ValueError(
            f"axis must be (N, {_AXES}, A) beside inv, got {tuple(axis.shape)}"
        )
    if not inv.is_floating_point() or not axis.is_floating_point():
        raise ValueError("inv and axis must be floating point")
    if inv.dtype != axis.dtype:
        raise ValueError("inv and axis must have one dtype")
    if inv.device != axis.device:
        raise ValueError("inv and axis must be on one device")
    if any(tensor.device != inv.device for tensor in tensors[2:]):
        raise ValueError("every equivariant-stage tensor must be on one device")

    n_rows, d_inv = int(inv.shape[0]), int(inv.shape[1])
    d_axis = int(axis.shape[2])
    if d_inv < 1 or d_axis < 1:
        raise ValueError("the compiled default stage requires positive stream widths")
    for name, tensor, shape in (
        ("mix inv norm weight", mix[0], (d_inv,)),
        ("mix inv norm bias", mix[1], (d_inv,)),
        ("mix axis norm weight", mix[2], (d_axis,)),
        ("mix axis norm bias", mix[3], (d_axis,)),
        ("mix inv-to-axis weight", mix[4], (d_axis, d_inv)),
        ("mix inv-to-axis bias", mix[5], (d_axis,)),
        ("mix axis input weight", mix[6], (mix[6].shape[0], 3 * d_axis)),
        ("mix axis input bias", mix[7], (mix[6].shape[0],)),
        ("mix axis output weight", mix[8], (d_axis, mix[6].shape[0])),
        ("mix axis output bias", mix[9], (d_axis,)),
        ("mix phi weight", mix[10], (d_axis, d_axis)),
        ("mix phi bias", mix[11], (d_axis,)),
        ("mix inv input weight", mix[12], (mix[12].shape[0], d_inv + d_axis)),
        ("mix inv input bias", mix[13], (mix[12].shape[0],)),
        ("mix inv output weight", mix[14], (d_inv, mix[12].shape[0])),
        ("mix inv output bias", mix[15], (d_inv,)),
        ("mix inv gamma", mix[16], (d_inv,)),
        ("mix axis gamma", mix[17], (d_axis,)),
        ("ffn inv norm weight", ffn[0], (d_inv,)),
        ("ffn inv norm bias", ffn[1], (d_inv,)),
        ("ffn axis norm weight", ffn[2], (d_axis,)),
        ("ffn axis norm bias", ffn[3], (d_axis,)),
        ("ffn inv input weight", ffn[4], (ffn[4].shape[0], d_inv)),
        ("ffn inv input bias", ffn[5], (ffn[4].shape[0],)),
        ("ffn inv output weight", ffn[6], (d_inv, ffn[4].shape[0])),
        ("ffn inv output bias", ffn[7], (d_inv,)),
        ("ffn axis input weight", ffn[8], (ffn[8].shape[0], d_axis)),
        ("ffn axis input bias", ffn[9], (ffn[8].shape[0],)),
        ("ffn axis output weight", ffn[10], (d_axis, ffn[8].shape[0])),
        ("ffn axis output bias", ffn[11], (d_axis,)),
        ("ffn inv gamma", ffn[12], (d_inv,)),
        ("ffn axis gamma", ffn[13], (d_axis,)),
    ):
        _expect_shape(name, tensor, tuple(int(value) for value in shape))
        if not tensor.is_floating_point():
            raise ValueError(f"{name} must be floating point")

    if phase_id.dtype != torch.long or phase_row.dtype != torch.long:
        raise ValueError("phase_id and phase_row must be int64")
    if use_film:
        _expect_shape("phase_id", phase_id, (n_rows,))
        _expect_shape("phase_row", phase_row, (3,))
        if film[0].ndim != 2:
            raise ValueError(
                f"film embedding must be (K, P), got {tuple(film[0].shape)}"
            )
        phases, d_phase = int(film[0].shape[0]), int(film[0].shape[1])
        if phases < 1 or d_phase < 1:
            raise ValueError("film embedding dimensions must be positive")
        for name, tensor, shape in (
            ("film MLP weight", film[1], (d_phase, d_phase)),
            ("film MLP bias", film[2], (d_phase,)),
            ("film inv weight", film[3], (2 * d_inv, d_phase)),
            ("film inv bias", film[4], (2 * d_inv,)),
            ("film axis weight", film[5], (2 * d_axis, d_phase)),
            ("film axis bias", film[6], (2 * d_axis,)),
        ):
            _expect_shape(name, tensor, shape)
        if any(not tensor.is_floating_point() for tensor in film):
            raise ValueError("every FiLM parameter must be floating point")


def _launch_forward(
    tensors: tuple[Tensor, ...],
    activation: str,
    use_film: bool,
    autocast_code: int,
    eps: tuple[float, float, float, float],
) -> tuple[Tensor, Tensor]:
    _mark_dynamic_rows(tensors, use_film)
    function = _forward_function(
        activation,
        use_film,
        autocast_code,
        eps,
        True,
        _compile_key(tensors, activation, use_film, autocast_code, eps),
    )
    return function(*tensors)


def _ordered_gradients(
    compact: tuple[Tensor, ...], tensors: tuple[Tensor, ...], use_film: bool
) -> tuple[Tensor, ...]:
    if use_film:
        return compact
    # The no-FiLM closure differentiates only the base streams and parameters.
    # The backward custom op has one fixed schema, so append the seven unused
    # placeholder gradients in the same order as the full path.
    return (*compact, *(torch.zeros_like(tensor) for tensor in tensors[_FILM_START:]))


def _launch_backward(
    tensors: tuple[Tensor, ...],
    activation: str,
    use_film: bool,
    autocast_code: int,
    eps: tuple[float, float, float, float],
    grad_inv: Tensor,
    grad_axis: Tensor,
) -> tuple[Tensor, ...]:
    # The autograd engine invokes registered formulas with GradMode disabled.
    # Do not try to rebuild a tape by toggling GradMode around a separately
    # compiled forward: real Inductor may already have lowered that callable
    # as an inference graph, yielding outputs without grad_fn.  Compile the
    # complete functional VJP instead.  torch.func supplies differentiation
    # dispatch independently of requires_grad and of the surrounding mode.
    recompute = tuple(tensor.detach() for tensor in tensors)
    grad_inv = grad_inv.detach()
    grad_axis = grad_axis.detach()
    gradients = (grad_inv, grad_axis)
    _mark_dynamic_rows(recompute, use_film, gradients)
    function = _backward_function(
        activation,
        use_film,
        autocast_code,
        eps,
        True,
        _compile_key(
            recompute,
            activation,
            use_film,
            autocast_code,
            eps,
            gradients,
        ),
    )
    compact = function(*recompute, grad_inv, grad_axis)
    return _ordered_gradients(compact, tensors, use_film)


def _eager_backward(
    tensors: tuple[Tensor, ...],
    activation: str,
    use_film: bool,
    autocast_code: int,
    eps: tuple[float, float, float, float],
    grad_inv: Tensor,
    grad_axis: Tensor,
) -> tuple[Tensor, ...]:
    function = _backward_function(
        activation, use_film, autocast_code, eps, False
    )
    compact = function(*tensors, grad_inv, grad_axis)
    return _ordered_gradients(compact, tensors, use_film)


def _dispatch_backward(
    tensors: tuple[Tensor, ...],
    activation: str,
    use_film: bool,
    autocast_code: int,
    eps: tuple[float, float, float, float],
    grad_inv: Tensor,
    grad_axis: Tensor,
) -> tuple[Tensor, ...]:
    """Select the compiled functional VJP from the autograd callback.

    The complete ``torch.func.grad`` transform is the compilation unit.  It
    therefore remains differentiable when this helper is called from the
    engine's no-grad turn and does not rely on a nested compiled forward
    returning tensors with ``grad_fn`` metadata.
    """
    reference = lambda: _eager_backward(  # noqa: E731
        tensors,
        activation,
        use_film,
        autocast_code,
        eps,
        grad_inv,
        grad_axis,
    )
    if not _supported(tensors, activation, use_film):
        return reference()
    _LAUNCH_STATS["backward_eligible"] += 1
    key = (
        *_shape_key(tensors, activation, use_film, autocast_code, eps),
        "backward",
    )
    try:
        result = _launch_backward(
            tensors,
            activation,
            use_film,
            autocast_code,
            eps,
            grad_inv.contiguous(),
            grad_axis.contiguous(),
        )
        _LAUNCH_STATS["backward_launched"] += 1
        return result
    except Exception as exc:
        _FAILED_BACKWARD_SHAPES[key] = f"{type(exc).__name__}: {exc}"
        raise EquivariantStageFusionError(
            "eligible CUDA equivariant stage backward failed; refusing to "
            "silently de-fuse the trunk. Static signature: "
            f"{key}. Cause: {_FAILED_BACKWARD_SHAPES[key]}"
        ) from exc


@torch.library.custom_op("mantisnet::act_equivariant_stage", mutates_args=())
def _stage_op(
    inv: Tensor,
    axis: Tensor,
    mix_inv_norm_weight: Tensor,
    mix_inv_norm_bias: Tensor,
    mix_axis_norm_weight: Tensor,
    mix_axis_norm_bias: Tensor,
    mix_inv_to_axis_weight: Tensor,
    mix_inv_to_axis_bias: Tensor,
    mix_axis_in_weight: Tensor,
    mix_axis_in_bias: Tensor,
    mix_axis_out_weight: Tensor,
    mix_axis_out_bias: Tensor,
    mix_phi_weight: Tensor,
    mix_phi_bias: Tensor,
    mix_inv_in_weight: Tensor,
    mix_inv_in_bias: Tensor,
    mix_inv_out_weight: Tensor,
    mix_inv_out_bias: Tensor,
    mix_inv_gamma: Tensor,
    mix_axis_gamma: Tensor,
    ffn_inv_norm_weight: Tensor,
    ffn_inv_norm_bias: Tensor,
    ffn_axis_norm_weight: Tensor,
    ffn_axis_norm_bias: Tensor,
    ffn_inv_in_weight: Tensor,
    ffn_inv_in_bias: Tensor,
    ffn_inv_out_weight: Tensor,
    ffn_inv_out_bias: Tensor,
    ffn_axis_in_weight: Tensor,
    ffn_axis_in_bias: Tensor,
    ffn_axis_out_weight: Tensor,
    ffn_axis_out_bias: Tensor,
    ffn_inv_gamma: Tensor,
    ffn_axis_gamma: Tensor,
    phase_id: Tensor,
    phase_row: Tensor,
    film_embed_weight: Tensor,
    film_mlp_weight: Tensor,
    film_mlp_bias: Tensor,
    film_inv_weight: Tensor,
    film_inv_bias: Tensor,
    film_axis_weight: Tensor,
    film_axis_bias: Tensor,
    activation: str,
    use_film: bool,
    autocast_code: int,
    mix_inv_eps: float,
    mix_axis_eps: float,
    ffn_inv_eps: float,
    ffn_axis_eps: float,
) -> tuple[Tensor, Tensor]:
    tensors = (
        inv,
        axis,
        mix_inv_norm_weight,
        mix_inv_norm_bias,
        mix_axis_norm_weight,
        mix_axis_norm_bias,
        mix_inv_to_axis_weight,
        mix_inv_to_axis_bias,
        mix_axis_in_weight,
        mix_axis_in_bias,
        mix_axis_out_weight,
        mix_axis_out_bias,
        mix_phi_weight,
        mix_phi_bias,
        mix_inv_in_weight,
        mix_inv_in_bias,
        mix_inv_out_weight,
        mix_inv_out_bias,
        mix_inv_gamma,
        mix_axis_gamma,
        ffn_inv_norm_weight,
        ffn_inv_norm_bias,
        ffn_axis_norm_weight,
        ffn_axis_norm_bias,
        ffn_inv_in_weight,
        ffn_inv_in_bias,
        ffn_inv_out_weight,
        ffn_inv_out_bias,
        ffn_axis_in_weight,
        ffn_axis_in_bias,
        ffn_axis_out_weight,
        ffn_axis_out_bias,
        ffn_inv_gamma,
        ffn_axis_gamma,
        phase_id,
        phase_row,
        film_embed_weight,
        film_mlp_weight,
        film_mlp_bias,
        film_inv_weight,
        film_inv_bias,
        film_axis_weight,
        film_axis_bias,
    )
    eps = (mix_inv_eps, mix_axis_eps, ffn_inv_eps, ffn_axis_eps)
    _validate(tensors, activation, use_film)
    reference = lambda: _forward_function(  # noqa: E731
        activation, use_film, autocast_code, eps, False
    )(*tensors)
    if not _supported(tensors, activation, use_film):
        return reference()
    _LAUNCH_STATS["forward_eligible"] += 1
    key = _shape_key(tensors, activation, use_film, autocast_code, eps)
    try:
        result = _launch_forward(
            tensors, activation, use_film, autocast_code, eps
        )
        _LAUNCH_STATS["forward_launched"] += 1
        return result
    except Exception as exc:
        _FAILED_FORWARD_SHAPES[key] = f"{type(exc).__name__}: {exc}"
        raise EquivariantStageFusionError(
            "eligible CUDA equivariant stage forward failed; refusing to "
            "silently de-fuse the trunk. Static signature: "
            f"{key}. Cause: {_FAILED_FORWARD_SHAPES[key]}"
        ) from exc


@_stage_op.register_fake
def _(
    inv: Tensor,
    axis: Tensor,
    *args,
) -> tuple[Tensor, Tensor]:
    # LayerScale is fp32 in the CUDA path, so residual promotion makes both
    # outputs fp32 even when the incoming activation is reduced precision.
    mix_inv_gamma = args[16]
    mix_axis_gamma = args[17]
    ffn_inv_gamma = args[30]
    ffn_axis_gamma = args[31]
    inv_dtype = torch.promote_types(
        torch.promote_types(inv.dtype, mix_inv_gamma.dtype), ffn_inv_gamma.dtype
    )
    axis_dtype = torch.promote_types(
        torch.promote_types(axis.dtype, mix_axis_gamma.dtype),
        ffn_axis_gamma.dtype,
    )
    return (
        torch.empty_like(inv, dtype=inv_dtype),
        torch.empty_like(axis, dtype=axis_dtype),
    )


# The backward operator returns one tensor for every differentiable tensor
# input: 34 stream/stage tensors plus seven FiLM parameters.  phase_id and
# phase_row are integer routing data and deliberately absent.
_StageGradientTuple = tuple[
    Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
    Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
    Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
    Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
    Tensor, Tensor, Tensor, Tensor, Tensor,
]


@torch.library.custom_op(
    "mantisnet::act_equivariant_stage_backward", mutates_args=()
)
def _stage_backward_op(
    inv: Tensor,
    axis: Tensor,
    mix_inv_norm_weight: Tensor,
    mix_inv_norm_bias: Tensor,
    mix_axis_norm_weight: Tensor,
    mix_axis_norm_bias: Tensor,
    mix_inv_to_axis_weight: Tensor,
    mix_inv_to_axis_bias: Tensor,
    mix_axis_in_weight: Tensor,
    mix_axis_in_bias: Tensor,
    mix_axis_out_weight: Tensor,
    mix_axis_out_bias: Tensor,
    mix_phi_weight: Tensor,
    mix_phi_bias: Tensor,
    mix_inv_in_weight: Tensor,
    mix_inv_in_bias: Tensor,
    mix_inv_out_weight: Tensor,
    mix_inv_out_bias: Tensor,
    mix_inv_gamma: Tensor,
    mix_axis_gamma: Tensor,
    ffn_inv_norm_weight: Tensor,
    ffn_inv_norm_bias: Tensor,
    ffn_axis_norm_weight: Tensor,
    ffn_axis_norm_bias: Tensor,
    ffn_inv_in_weight: Tensor,
    ffn_inv_in_bias: Tensor,
    ffn_inv_out_weight: Tensor,
    ffn_inv_out_bias: Tensor,
    ffn_axis_in_weight: Tensor,
    ffn_axis_in_bias: Tensor,
    ffn_axis_out_weight: Tensor,
    ffn_axis_out_bias: Tensor,
    ffn_inv_gamma: Tensor,
    ffn_axis_gamma: Tensor,
    phase_id: Tensor,
    phase_row: Tensor,
    film_embed_weight: Tensor,
    film_mlp_weight: Tensor,
    film_mlp_bias: Tensor,
    film_inv_weight: Tensor,
    film_inv_bias: Tensor,
    film_axis_weight: Tensor,
    film_axis_bias: Tensor,
    activation: str,
    use_film: bool,
    autocast_code: int,
    mix_inv_eps: float,
    mix_axis_eps: float,
    ffn_inv_eps: float,
    ffn_axis_eps: float,
    grad_inv: Tensor,
    grad_axis: Tensor,
) -> _StageGradientTuple:
    tensors = (
        inv,
        axis,
        mix_inv_norm_weight,
        mix_inv_norm_bias,
        mix_axis_norm_weight,
        mix_axis_norm_bias,
        mix_inv_to_axis_weight,
        mix_inv_to_axis_bias,
        mix_axis_in_weight,
        mix_axis_in_bias,
        mix_axis_out_weight,
        mix_axis_out_bias,
        mix_phi_weight,
        mix_phi_bias,
        mix_inv_in_weight,
        mix_inv_in_bias,
        mix_inv_out_weight,
        mix_inv_out_bias,
        mix_inv_gamma,
        mix_axis_gamma,
        ffn_inv_norm_weight,
        ffn_inv_norm_bias,
        ffn_axis_norm_weight,
        ffn_axis_norm_bias,
        ffn_inv_in_weight,
        ffn_inv_in_bias,
        ffn_inv_out_weight,
        ffn_inv_out_bias,
        ffn_axis_in_weight,
        ffn_axis_in_bias,
        ffn_axis_out_weight,
        ffn_axis_out_bias,
        ffn_inv_gamma,
        ffn_axis_gamma,
        phase_id,
        phase_row,
        film_embed_weight,
        film_mlp_weight,
        film_mlp_bias,
        film_inv_weight,
        film_inv_bias,
        film_axis_weight,
        film_axis_bias,
    )
    eps = (mix_inv_eps, mix_axis_eps, ffn_inv_eps, ffn_axis_eps)
    # AOTAutograd represents the registered formula with this opaque node.
    # Its implementation runs below Autograd dispatch, so rebuilding a tape
    # here would be invalid.  `_dispatch_backward` uses the compiled
    # torch.func VJP, whose differentiation dispatch is independent of that
    # TLS state, and is therefore the same safe path used by the eager
    # registered callback.
    return _dispatch_backward(
        tensors,
        activation,
        use_film,
        autocast_code,
        eps,
        grad_inv,
        grad_axis,
    )


@_stage_backward_op.register_fake
def _(
    inv: Tensor,
    axis: Tensor,
    *args,
) -> _StageGradientTuple:
    # Forward tensor arguments occupy args[:41]: the first two were named,
    # followed by 41 more.  Scalar settings and output cotangents follow.
    tensors = (inv, axis, *args[: _TENSOR_COUNT - 2])
    return tuple(
        torch.empty_like(tensor)
        for index, tensor in enumerate(tensors)
        if index not in _NONDIFFERENTIABLE
    )


def _stage_setup(ctx, inputs, output) -> None:
    ctx.settings = inputs[_TENSOR_COUNT:]
    ctx.save_for_backward(*inputs[:_TENSOR_COUNT])


def _is_functional_wrapper(tensor: Tensor) -> bool:
    """Whether AOTAutograd wrapped this value for functionalization."""
    if torch._is_functional_tensor(tensor):
        return True
    # AOT's Python FunctionalTensor shell owns the actual functional tensor in
    # ``elem``; torch._is_functional_tensor intentionally reports only on the
    # inner TensorImpl in PyTorch 2.11.
    inner = getattr(tensor, "elem", None)
    return isinstance(inner, Tensor) and torch._is_functional_tensor(inner)


def _stage_dispatch(ctx, grad_inv: Tensor, grad_axis: Tensor):
    tensors = ctx.saved_tensors
    (
        activation,
        use_film,
        autocast_code,
        mix_inv_eps,
        mix_axis_eps,
        ffn_inv_eps,
        ffn_axis_eps,
    ) = ctx.settings
    if _is_functional_wrapper(tensors[0]):
        # AOTAutograd traces registered formulas with FunctionalTensor
        # wrappers.  Keep that contract representable as the registered
        # backward schema; executing Python recompute while the wrapper mode
        # is suspended would strand FunctionalTensors outside their mode.
        gradients = _stage_backward_op(
            *tensors, *ctx.settings, grad_inv, grad_axis
        )
    else:
        gradients = _dispatch_backward(
            tensors,
            activation,
            use_film,
            autocast_code,
            (mix_inv_eps, mix_axis_eps, ffn_inv_eps, ffn_axis_eps),
            grad_inv,
            grad_axis,
        )
    gradient_iter = iter(gradients)
    tensor_gradients = tuple(
        None if index in _NONDIFFERENTIABLE else next(gradient_iter)
        for index in range(_TENSOR_COUNT)
    )
    return (*tensor_gradients, *(None for _ in ctx.settings))


_stage_op.register_autograd(_stage_dispatch, setup_context=_stage_setup)


def _autocast_code(inv: Tensor) -> int:
    if not inv.is_cuda or not torch.is_autocast_enabled("cuda"):
        return 0
    dtype = torch.get_autocast_dtype("cuda")
    if dtype == torch.float16:
        return 1
    if dtype == torch.bfloat16:
        return 2
    return 0


def equivariant_stage(
    inv: Tensor,
    axis: Tensor,
    mix: Sequence[Tensor],
    ffn: Sequence[Tensor],
    *,
    phase_id: Tensor | None = None,
    phase_row: Tensor | None = None,
    film: Sequence[Tensor] | None = None,
    activation: str = "silu",
    dropout: float = 0.0,
    mix_eps: tuple[float, float] = (1e-5, 1e-5),
    ffn_eps: tuple[float, float] = (1e-5, 1e-5),
) -> tuple[Tensor, Tensor]:
    """Run a default entity's AxisMix + FFN + optional phase FiLM.

    ``mix`` follows the owning ``AxisMix`` module in this order: norm inv
    weight/bias, norm axis weight/bias, inv-to-axis weight/bias, axis MLP input
    weight/bias and output weight/bias, phi weight/bias, invariant MLP input
    weight/bias and output weight/bias, then invariant/axis LayerScale gamma.

    ``ffn`` similarly carries its two norm pairs, invariant MLP input/output,
    axis MLP input/output, and the two LayerScale gammas.  If ``film`` is given,
    it is embedding weight, phase-MLP weight/bias, invariant projection
    weight/bias, and axis projection weight/bias; ``phase_id`` and
    ``phase_row`` are then required.  Existing module parameters can therefore
    be passed directly without wrapping or renaming them.
    """
    if float(dropout) != 0.0:
        raise ValueError(
            "the recompute equivariant stage requires dropout=0; a nonzero "
            "mask would have to be an explicit operator input"
        )
    mix = tuple(mix)
    ffn = tuple(ffn)
    if len(mix) != _MIX_COUNT:
        raise ValueError(f"mix must carry {_MIX_COUNT} tensors, got {len(mix)}")
    if len(ffn) != _FFN_COUNT:
        raise ValueError(f"ffn must carry {_FFN_COUNT} tensors, got {len(ffn)}")
    use_film = film is not None
    if use_film:
        film = tuple(film)
        if len(film) != _FILM_COUNT:
            raise ValueError(
                f"film must carry {_FILM_COUNT} tensors, got {len(film)}"
            )
        if phase_id is None or phase_row is None:
            raise ValueError("phase_id and phase_row are required with FiLM")
    else:
        if phase_id is not None or phase_row is not None:
            raise ValueError("phase routing was supplied without FiLM parameters")
        phase_id = torch.empty(0, dtype=torch.long, device=inv.device)
        phase_row = torch.empty(0, dtype=torch.long, device=inv.device)
        film = tuple(inv.new_empty(0) for _ in range(_FILM_COUNT))

    if len(mix_eps) != 2 or len(ffn_eps) != 2:
        raise ValueError("mix_eps and ffn_eps must each contain inv and axis eps")
    return _stage_op(
        inv.contiguous(),
        axis.contiguous(),
        *(tensor.contiguous() for tensor in mix),
        *(tensor.contiguous() for tensor in ffn),
        phase_id.contiguous(),
        phase_row.contiguous(),
        *(tensor.contiguous() for tensor in film),
        activation,
        use_film,
        _autocast_code(inv),
        float(mix_eps[0]),
        float(mix_eps[1]),
        float(ffn_eps[0]),
        float(ffn_eps[1]),
    )


__all__ = [
    "EquivariantStageFusionError",
    "clear_compile_caches",
    "clear_failure_caches",
    "equivariant_stage",
    "launch_stats",
    "reset_launch_stats",
]
