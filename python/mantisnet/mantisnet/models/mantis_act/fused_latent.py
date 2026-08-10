"""Registered whole-pass latent read, mix, and broadcast operators.

The flash-style kernels in :mod:`latent_attention` already make the ragged
softmaxes ordered and memory-efficient.  The remaining eager ``LatentPass``
still dispatches every norm, q/k/v/o projection, cast, residual, and symmetric
pool separately.  This module wraps the complete default state or action pass
in one registered operator and asks Inductor to fuse that surrounding work,
while retaining the existing ordered attention cores as opaque inner ops.

The outer forward owns no parameters.  It receives the original
``LatentPass.parameters()`` sequence, so state-dict names and checkpoint
structure remain unchanged.  Backward saves only inputs, plans, and parameter
references, then recomputes the compiled forward and differentiates it.  A
``torch.func.grad`` transform cannot cross a ``torch.library.custom_op`` in
PyTorch 2.11 without a registered functorch rule; the established latent
attention ops have autograd rules but no functorch rule.  Consequently the
registered autograd callback performs the recompute under ``torch.enable_grad``
and invokes the AOT-compiled forward's backward.  This remains a recompute
backward, but is intentionally kept out of a second custom-op implementation.

Only the default layouts are accepted: two state families with invariant and
axis latents, or one action family with invariant latents; learned axis pools,
positive axis width, and zero dropout.  Ablations continue through the eager
``LatentPass`` formulation.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import math
from types import FunctionType
from typing import Callable, Sequence
import weakref

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from .equivariant import AxisMix, AxisPool
from .latent_attention import _latent_broadcast_op, _latent_read_op
from .plans import LatentSegments


_AXES = 3
_STATE_COUNT = 115
_ACTION_COUNT = 47
_STATE_PLAN_COUNT = 8
_ACTION_PLAN_COUNT = 6
_CUDA_DTYPES = frozenset({torch.float16, torch.bfloat16, torch.float32})
_ACTIVATIONS = frozenset({"silu", "gelu", "relu"})
_COMPILE_OPTIONS = {"emulate_precision_casts": True}

# Fail closed if a parameter-preserving wrapper or child replacement changes
# a forward that this op deliberately bypasses.  These are digests of ordered
# ``name:type`` module lines followed by ordered parameter names, not model or
# checkpoint identities. Widths and latent counts do not enter, so every
# naturally shape-polymorphic default layout remains eligible.
_STATE_LAYOUTS = {
    "silu": "950eb4ff76eabd8462ddb0d867d232184d7242d4b16083f797ab100a37897c91",
    "gelu": "2f44362516ce525d809f237de8c9db3499d6da569497365ae3773ea147ebc1ed",
    "relu": "47e86d98cc3deb5c236dd6bcd254a02d9e472e2044133e5a0eb2bb06bdd246e4",
}
_ACTION_LAYOUT = "fa484dcf809813d4df4a9050a276c8a4ef55b54be136fb45ce9cb5690a7d01c0"
_LAYOUT_DIGEST_CACHE: weakref.WeakKeyDictionary[object, str] = (
    weakref.WeakKeyDictionary()
)

_FAILED_FORWARD_SHAPES: dict[tuple[object, ...], str] = {}
_FAILED_BACKWARD_SHAPES: dict[tuple[object, ...], str] = {}
_LAUNCH_STATS = {
    "state_forward_eligible": 0,
    "state_forward_launched": 0,
    "state_backward_eligible": 0,
    "state_backward_launched": 0,
    "action_forward_eligible": 0,
    "action_forward_launched": 0,
    "action_backward_eligible": 0,
    "action_backward_launched": 0,
}


class FusedLatentCompileError(RuntimeError):
    """The default CUDA latent stage could not stay on its fused path."""


def reset_launch_stats() -> None:
    for name in _LAUNCH_STATS:
        _LAUNCH_STATS[name] = 0


def launch_stats() -> dict[str, int]:
    return dict(_LAUNCH_STATS)


def clear_failure_caches() -> None:
    _FAILED_FORWARD_SHAPES.clear()
    _FAILED_BACKWARD_SHAPES.clear()


def clear_compile_caches() -> None:
    """Drop Python references to compiled latent-stage callables.

    This is a test/profiling hook, not part of normal dispatch.  Dynamo's own
    process cache is deliberately left intact: clearing it here would hide the
    very shared-budget regressions this module must prevent.
    """
    _forward_function.cache_clear()


def _activation(value: Tensor, name: str) -> Tensor:
    if name == "silu":
        return F.silu(value)
    if name == "gelu":
        return F.gelu(value)
    if name == "relu":
        return F.relu(value)
    raise ValueError(f"activation={name!r} is not one of {sorted(_ACTIVATIONS)}")


def _norm(value: Tensor, weight: Tensor, bias: Tensor, eps: float) -> Tensor:
    return F.layer_norm(value, (value.shape[-1],), weight, bias, eps)


def _multi_linear(
    value: Tensor, pairs: Sequence[tuple[Tensor, Tensor]]
) -> tuple[Tensor, ...]:
    """Horizontally concatenate projections sharing one input."""
    widths = tuple(int(weight.shape[0]) for weight, _bias in pairs)
    weight = torch.cat(tuple(item[0] for item in pairs), dim=0)
    bias = torch.cat(tuple(item[1] for item in pairs), dim=0)
    return F.linear(value, weight, bias).split(widths, dim=-1)


def _axis_pool(
    inv: Tensor,
    axis: Tensor,
    from_axis_weight: Tensor,
    from_axis_bias: Tensor,
    from_inv_weight: Tensor,
    score_weight: Tensor,
) -> Tensor:
    scores = F.linear(
        torch.tanh(
            F.linear(axis, from_axis_weight, from_axis_bias)
            + F.linear(inv, from_inv_weight).unsqueeze(-2)
        ),
        score_weight,
    )
    accumulation_dtype = torch.promote_types(scores.dtype, torch.float32)
    weight = F.softmax(scores, dim=-2, dtype=accumulation_dtype)
    return torch.sum(
        weight * axis.to(accumulation_dtype),
        dim=-2,
        dtype=accumulation_dtype,
    ).to(axis.dtype)


def _dense_attention(score: Tensor, value: Tensor, dim: int) -> Tensor:
    accumulation_dtype = torch.promote_types(score.dtype, torch.float32)
    weight = F.softmax(score, dim=dim, dtype=accumulation_dtype)
    return torch.sum(
        weight.unsqueeze(-1) * value.to(accumulation_dtype),
        dim=dim,
        dtype=accumulation_dtype,
    )


def _read(
    q: Tensor, k: Tensor, v: Tensor, plans: tuple[Tensor, ...]
) -> Tensor:
    out, _maximum, _denominator = _latent_read_op(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        plans[0],
        plans[1],
        plans[2],
        plans[3],
    )
    return out


def _broadcast(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    row_pos: Tensor,
    offsets: Tensor,
) -> Tensor:
    return _latent_broadcast_op(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        row_pos,
        offsets,
    )


def _axis_mix(
    inv: Tensor,
    axis: Tensor,
    params: tuple[Tensor, ...],
    activation: str,
    eps_inv: float,
    eps_axis: float,
) -> tuple[Tensor, Tensor]:
    z_inv = _norm(inv, params[0], params[1], eps_inv)
    u = _norm(axis, params[2], params[3], eps_axis)
    other = (u.sum(dim=-2, keepdim=True) - u) / (_AXES - 1)
    context = F.linear(z_inv, params[4], params[5]).unsqueeze(-2).expand_as(u)
    delta_axis = F.linear(
        _activation(
            F.linear(
                torch.cat((u, other, context), dim=-1), params[6], params[7]
            ),
            activation,
        ),
        params[8],
        params[9],
    )
    summary = _activation(F.linear(u, params[10], params[11]), activation).mean(
        dim=-2
    )
    delta_inv = F.linear(
        _activation(
            F.linear(
                torch.cat((z_inv, summary), dim=-1), params[12], params[13]
            ),
            activation,
        ),
        params[14],
        params[15],
    )
    return inv + params[16] * delta_inv, axis + params[17] * delta_axis


def _state_reference(
    states: tuple[Tensor, ...],
    plans: tuple[Tensor, ...],
    p: tuple[Tensor, ...],
    *,
    heads: int,
    activation: str,
    eps: tuple[float, ...],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    latent_inv, latent_axis, cell_inv, cell_axis, window_inv, window_axis = states
    positions, num_inv, d_inv = latent_inv.shape
    num_axis, d_axis = int(latent_axis.shape[1]), int(latent_axis.shape[-1])
    head_inv, head_axis = d_inv // heads, d_axis // heads

    # Read: one family-private norm pair, then shared horizontal k/v maps.
    norm_cell_inv = _norm(cell_inv, p[3], p[4], eps[0])
    norm_cell_axis = _norm(cell_axis, p[5], p[6], eps[1])
    norm_window_inv = _norm(window_inv, p[7], p[8], eps[2])
    norm_window_axis = _norm(window_axis, p[9], p[10], eps[3])

    cell_key = norm_cell_inv + p[0][0]
    cell_key = cell_key + F.linear(
        _axis_pool(norm_cell_inv, norm_cell_axis, *p[22:26]), p[26], p[27]
    )
    window_key = norm_window_inv + p[0][1]
    window_key = window_key + F.linear(
        _axis_pool(norm_window_inv, norm_window_axis, *p[22:26]), p[26], p[27]
    )
    rows_inv = torch.cat((cell_key, window_key), dim=0)
    q_inv = F.linear(
        _norm(latent_inv, p[11], p[12], eps[4]), p[13], p[14]
    ).view(positions, num_inv, 1, heads, head_inv)
    k_inv, v_inv = _multi_linear(rows_inv, ((p[15], p[16]), (p[17], p[18])))
    out_inv = _read(
        q_inv,
        k_inv.view(-1, 1, heads, head_inv),
        v_inv.view(-1, 1, heads, head_inv),
        plans,
    )
    latent_inv = latent_inv + p[21] * F.linear(
        out_inv.reshape(positions, num_inv, d_inv).to(latent_inv.dtype),
        p[19],
        p[20],
    )

    rows_axis = torch.cat(
        (norm_cell_axis + p[1][0], norm_window_axis + p[1][1]), dim=0
    )
    q_axis = F.linear(
        _norm(latent_axis, p[28], p[29], eps[5]), p[30], p[31]
    ).view(positions, num_axis, _AXES, heads, head_axis)
    k_axis, v_axis = _multi_linear(
        rows_axis, ((p[32], p[33]), (p[34], p[35]))
    )
    out_axis = _read(
        q_axis,
        k_axis.view(-1, _AXES, heads, head_axis),
        v_axis.view(-1, _AXES, heads, head_axis),
        plans,
    )
    latent_axis = latent_axis + p[38] * F.linear(
        out_axis.reshape(positions, num_axis, _AXES, d_axis).to(
            latent_axis.dtype
        ),
        p[36],
        p[37],
    )

    # Mix: q/k/v are one horizontal projection of each normalised stream.
    z_inv = _norm(latent_inv, p[39], p[40], eps[6])
    q_mix, k_mix, v_mix = _multi_linear(
        z_inv,
        ((p[41], p[42]), (p[43], p[44]), (p[45], p[46])),
    )
    shape_inv = (positions, num_inv, heads, head_inv)
    q_mix, k_mix, v_mix = (
        value.view(shape_inv) for value in (q_mix, k_mix, v_mix)
    )
    score = torch.einsum("pqhd,pkhd->pqkh", q_mix, k_mix) / math.sqrt(head_inv)
    mixed = _dense_attention(score, v_mix.unsqueeze(1), 2)
    latent_inv = latent_inv + p[49] * F.linear(
        mixed.reshape(positions, num_inv, d_inv).to(latent_inv.dtype),
        p[47],
        p[48],
    )

    z_axis = _norm(latent_axis, p[50], p[51], eps[7])
    q_mix, k_mix, v_mix = _multi_linear(
        z_axis,
        ((p[52], p[53]), (p[54], p[55]), (p[56], p[57])),
    )
    shape_axis = (positions, num_axis, _AXES, heads, head_axis)
    q_mix, k_mix, v_mix = (
        value.view(shape_axis) for value in (q_mix, k_mix, v_mix)
    )
    score = torch.einsum(
        "pqahd,pkahd->pqkah", q_mix, k_mix
    ) / math.sqrt(head_axis)
    mixed = _dense_attention(score, v_mix.unsqueeze(1), 2)
    latent_axis = latent_axis + p[60] * F.linear(
        mixed.reshape(positions, num_axis, _AXES, d_axis).to(latent_axis.dtype),
        p[58],
        p[59],
    )

    paired_inv = latent_inv.mean(dim=1, keepdim=True).expand(-1, num_axis, -1)
    axis_mix_inv, latent_axis = _axis_mix(
        paired_inv, latent_axis, p[61:79], activation, eps[8], eps[9]
    )
    latent_inv = latent_inv + (axis_mix_inv - paired_inv).mean(
        dim=1, keepdim=True
    )

    # Broadcast: invariant context includes a symmetric pool of axis latents.
    context_inv = [_norm(latent_inv, p[79], p[80], eps[10]) + p[2][0]]
    normed_axis = _norm(latent_axis, p[81], p[82], eps[11])
    pooled_inv = latent_inv.mean(dim=1, keepdim=True).expand(-1, num_axis, -1)
    pooled_axis = _axis_pool(pooled_inv, normed_axis, *p[83:87])
    context_inv.append(F.linear(pooled_axis, p[87], p[88]) + p[2][1])
    context_inv = torch.cat(context_inv, dim=1)
    context_rows = int(context_inv.shape[1])
    key_inv, value_inv = _multi_linear(
        context_inv, ((p[89], p[90]), (p[91], p[92]))
    )
    inv_shape = (positions, context_rows, 1, heads, head_inv)
    key_inv, value_inv = key_inv.view(inv_shape), value_inv.view(inv_shape)

    axis_shape = (positions, num_axis, _AXES, heads, head_axis)
    key_axis, value_axis = _multi_linear(
        normed_axis, ((p[102], p[103]), (p[104], p[105]))
    )
    key_axis, value_axis = key_axis.view(axis_shape), value_axis.view(axis_shape)

    def update_family(
        inv: Tensor,
        axis: Tensor,
        family: int,
        offsets: Tensor,
        row_pos: Tensor,
    ) -> tuple[Tensor, Tensor]:
        n_rows = int(inv.shape[0])
        q_inv = F.linear(
            _norm(inv, p[93 + 2 * family], p[94 + 2 * family], eps[12 + family]),
            p[97],
            p[98],
        ).view(n_rows, 1, heads, head_inv)
        out = _broadcast(q_inv, key_inv, value_inv, row_pos, offsets)
        inv = inv + p[101] * F.linear(
            out.reshape(n_rows, d_inv).to(inv.dtype), p[99], p[100]
        )

        q_axis = F.linear(
            _norm(
                axis,
                p[106 + 2 * family],
                p[107 + 2 * family],
                eps[14 + family],
            ),
            p[110],
            p[111],
        ).view(n_rows, _AXES, heads, head_axis)
        out = _broadcast(q_axis, key_axis, value_axis, row_pos, offsets)
        axis = axis + p[114] * F.linear(
            out.reshape(n_rows, _AXES, d_axis).to(axis.dtype), p[112], p[113]
        )
        return inv, axis

    cell_inv, cell_axis = update_family(
        cell_inv, cell_axis, 0, plans[4], plans[5]
    )
    window_inv, window_axis = update_family(
        window_inv, window_axis, 1, plans[6], plans[7]
    )
    return latent_inv, latent_axis, cell_inv, cell_axis, window_inv, window_axis


def _action_reference(
    states: tuple[Tensor, ...],
    plans: tuple[Tensor, ...],
    p: tuple[Tensor, ...],
    *,
    heads: int,
    activation: str,
    eps: tuple[float, ...],
) -> tuple[Tensor, Tensor]:
    del activation  # the action pass's only activation lives in its axis pool
    latent_inv, action_inv, action_axis = states
    positions, slots, d_inv = latent_inv.shape
    d_axis = int(action_axis.shape[-1])
    head_inv = d_inv // heads

    norm_inv = _norm(action_inv, p[2], p[3], eps[0])
    norm_axis = _norm(action_axis, p[4], p[5], eps[1])
    rows = norm_inv + p[0][0]
    rows = rows + F.linear(
        _axis_pool(norm_inv, norm_axis, *p[17:21]), p[21], p[22]
    )
    q = F.linear(_norm(latent_inv, p[6], p[7], eps[2]), p[8], p[9]).view(
        positions, slots, 1, heads, head_inv
    )
    key, value = _multi_linear(rows, ((p[10], p[11]), (p[12], p[13])))
    out = _read(
        q,
        key.view(-1, 1, heads, head_inv),
        value.view(-1, 1, heads, head_inv),
        plans,
    )
    latent_inv = latent_inv + p[16] * F.linear(
        out.reshape(positions, slots, d_inv).to(latent_inv.dtype), p[14], p[15]
    )

    z = _norm(latent_inv, p[23], p[24], eps[3])
    query, key, value = _multi_linear(
        z, ((p[25], p[26]), (p[27], p[28]), (p[29], p[30]))
    )
    shape = (positions, slots, heads, head_inv)
    query, key, value = (item.view(shape) for item in (query, key, value))
    score = torch.einsum("pqhd,pkhd->pqkh", query, key) / math.sqrt(head_inv)
    out = _dense_attention(score, value.unsqueeze(1), 2)
    latent_inv = latent_inv + p[33] * F.linear(
        out.reshape(positions, slots, d_inv).to(latent_inv.dtype), p[31], p[32]
    )

    context = _norm(latent_inv, p[34], p[35], eps[4]) + p[1][0]
    key, value = _multi_linear(context, ((p[36], p[37]), (p[38], p[39])))
    shape = (positions, slots, 1, heads, head_inv)
    key, value = key.view(shape), value.view(shape)
    n_rows = int(action_inv.shape[0])
    query = F.linear(
        _norm(action_inv, p[40], p[41], eps[5]), p[42], p[43]
    ).view(n_rows, 1, heads, head_inv)
    out = _broadcast(query, key, value, plans[5], plans[4])
    action_inv = action_inv + p[46] * F.linear(
        out.reshape(n_rows, d_inv).to(action_inv.dtype), p[44], p[45]
    )
    # action_axis contributes to the read pool but is deliberately not updated.
    if d_axis < 1:
        raise ValueError("the default action pass requires an action axis stream")
    return latent_inv, action_inv


def _autocast_dtype(code: int) -> torch.dtype | None:
    if code == 0:
        return None
    if code == 1:
        return torch.float16
    if code == 2:
        return torch.bfloat16
    raise ValueError(f"unknown CUDA autocast code {code}")


def _run(
    variant: str,
    states: tuple[Tensor, ...],
    plans: tuple[Tensor, ...],
    params: tuple[Tensor, ...],
    heads: int,
    activation: str,
    autocast_code: int,
    eps: tuple[float, ...],
) -> tuple[Tensor, ...]:
    reference = _state_reference if variant == "state" else _action_reference
    dtype = _autocast_dtype(autocast_code)
    if dtype is None:
        return reference(
            states, plans, params, heads=heads, activation=activation, eps=eps
        )
    with torch.autocast("cuda", dtype=dtype):
        return reference(
            states, plans, params, heads=heads, activation=activation, eps=eps
        )


@lru_cache(maxsize=None)
def _forward_function(
    variant: str,
    heads: int,
    activation: str,
    autocast_code: int,
    eps: tuple[float, ...],
    architecture: tuple[object, ...],
    compiled: bool,
) -> Callable[..., tuple[Tensor, ...]]:
    state_count = 6 if variant == "state" else 3
    plan_count = _STATE_PLAN_COUNT if variant == "state" else _ACTION_PLAN_COUNT

    def forward(*tensors: Tensor) -> tuple[Tensor, ...]:
        states = tuple(tensors[:state_count])
        plans = tuple(tensors[state_count : state_count + plan_count])
        params = tuple(tensors[state_count + plan_count :])
        return _run(
            variant,
            states,
            plans,
            params,
            heads,
            activation,
            autocast_code,
            eps,
        )

    if not compiled:
        return forward
    # Dynamo's recompile budget belongs to a Python code object, not to the
    # wrapper returned by ``torch.compile``.  Merely caching one wrapper per
    # architecture would therefore make a tiny test model, the real model,
    # autocast, and grad/no-grad variants consume one shared budget.  Clone the
    # frame once per fixed signature; all ragged sizes inside that frame remain
    # symbolic and must reuse its single graph.
    identity = repr(
        (variant, heads, activation, autocast_code, eps, architecture)
    ).encode("utf-8")
    suffix = hashlib.sha256(identity).hexdigest()[:12]
    name = f"_latent_{variant}_{suffix}"
    forward = FunctionType(
        forward.__code__.replace(co_name=name, co_qualname=name),
        forward.__globals__,
        name,
        forward.__defaults__,
        forward.__closure__,
    )
    return torch.compile(
        forward,
        fullgraph=True,
        dynamic=True,
        options=_COMPILE_OPTIONS,
    )


def _architecture_signature(
    states: tuple[Tensor, ...],
    plans: tuple[Tensor, ...],
    params: tuple[Tensor, ...],
) -> tuple[object, ...]:
    """Static layout plus the irreducible zero/singleton shape buckets.

    Dynamo necessarily specialises dimensions of size zero and one even after
    ``mark_dynamic``.  They therefore select their own bounded callable entry;
    every leading size of two or greater shares the polymorphic bucket.
    Autograd mode and ``requires_grad`` are guards too, so they are explicit
    instead of becoming surprise recompiles inside one entry.
    """

    def layout(value: Tensor, *, erase_leading: bool) -> tuple[object, ...]:
        shape = tuple(int(size) for size in value.shape)
        leading_bucket = min(shape[0], 2) if erase_leading else None
        return (
            value.device.type,
            value.device.index,
            value.dtype,
            value.requires_grad,
            value.ndim,
            leading_bucket,
            shape[1:] if erase_leading else shape,
        )

    return (
        torch.is_grad_enabled(),
        tuple(layout(value, erase_leading=True) for value in states),
        tuple(layout(value, erase_leading=True) for value in plans),
        tuple(layout(value, erase_leading=False) for value in params),
    )


def _mark_dynamic_rows(
    variant: str,
    states: tuple[Tensor, ...],
    plans: tuple[Tensor, ...],
) -> None:
    """Declare every batch-dependent latent-stage dimension to Dynamo.

    ``dynamic=True`` permits symbolic shapes, but it does not prevent Dynamo
    from specialising dimensions whose first trace happens to satisfy a
    stronger equality.  Every ACT chunk changes its position count, entity
    row counts, and matching CSR/row-position lengths.  Marking those leading
    dimensions before *each* compiled boundary gives one graph per static
    stage layout instead of consuming the process-wide recompile budget.

    Backward creates detached recompute leaves, which do not inherit tensor
    attributes, so its call site deliberately invokes this helper again.
    """
    expected_states = 6 if variant == "state" else 3
    expected_plans = _STATE_PLAN_COUNT if variant == "state" else _ACTION_PLAN_COUNT
    if len(states) != expected_states or len(plans) != expected_plans:
        raise ValueError(f"invalid {variant} latent tensors for dynamic marking")
    for value in (*states, *plans):
        # All state tensors are row-major, and all plan tensors have a
        # batch-dependent leading position/row dimension.  Remaining state
        # widths, family counts, axes, heads, and latent slots are static model
        # layout and therefore intentionally stay specialised.
        # Zero and singleton dimensions cannot be symbolic in Dynamo; their
        # explicit architecture buckets above keep them bounded instead.
        if int(value.shape[0]) >= 2:
            torch._dynamo.mark_dynamic(value, 0)


def _supported(
    variant: str,
    states: tuple[Tensor, ...],
    params: tuple[Tensor, ...],
    activation: str,
) -> bool:
    expected = _STATE_COUNT if variant == "state" else _ACTION_COUNT
    return (
        len(params) == expected
        and states[0].is_cuda
        and all(state.is_cuda for state in states)
        and states[0].dtype in _CUDA_DTYPES
        and all(state.dtype == states[0].dtype for state in states)
        and all(parameter.dtype == torch.float32 for parameter in params)
        and activation in _ACTIVATIONS
        and int(states[0].shape[0]) > 0
        and int(states[0].shape[1]) > 0
    )


def _validate_common(
    variant: str,
    states: tuple[Tensor, ...],
    plans: tuple[Tensor, ...],
    params: tuple[Tensor, ...],
    heads: int,
    activation: str,
    eps: tuple[float, ...],
) -> None:
    expected_states = 6 if variant == "state" else 3
    expected_plans = _STATE_PLAN_COUNT if variant == "state" else _ACTION_PLAN_COUNT
    expected_params = _STATE_COUNT if variant == "state" else _ACTION_COUNT
    expected_eps = 16 if variant == "state" else 6
    if len(states) != expected_states:
        raise ValueError(f"{variant} pass expects {expected_states} state tensors")
    if len(plans) != expected_plans:
        raise ValueError(f"{variant} pass expects {expected_plans} plan tensors")
    if len(params) != expected_params:
        raise ValueError(
            f"default {variant} LatentPass has {expected_params} parameters, "
            f"got {len(params)}; use its eager path for this ablation"
        )
    if len(eps) != expected_eps:
        raise ValueError(f"{variant} pass expects {expected_eps} norm eps values")
    if heads < 1:
        raise ValueError(f"heads must be positive, got {heads}")
    if activation not in _ACTIVATIONS:
        raise ValueError(
            f"activation={activation!r} is not one of {sorted(_ACTIVATIONS)}"
        )
    device = states[0].device
    if any(tensor.device != device for tensor in (*states, *plans, *params)):
        raise ValueError("every fused latent tensor must be on one device")
    if any(not state.is_floating_point() for state in states):
        raise ValueError("every latent/entity state must be floating point")
    if any(not parameter.is_floating_point() for parameter in params):
        raise ValueError("every latent-pass parameter must be floating point")


def _launch_forward(
    variant: str,
    states: tuple[Tensor, ...],
    plans: tuple[Tensor, ...],
    params: tuple[Tensor, ...],
    heads: int,
    activation: str,
    autocast_code: int,
    eps: tuple[float, ...],
) -> tuple[Tensor, ...]:
    function = _forward_function(
        variant,
        heads,
        activation,
        autocast_code,
        eps,
        _architecture_signature(states, plans, params),
        True,
    )
    return function(*states, *plans, *params)


def _dispatch_forward(
    variant: str,
    states: tuple[Tensor, ...],
    plans: tuple[Tensor, ...],
    params: tuple[Tensor, ...],
    heads: int,
    activation: str,
    autocast_code: int,
    eps: tuple[float, ...],
) -> tuple[Tensor, ...]:
    _validate_common(variant, states, plans, params, heads, activation, eps)
    architecture = _architecture_signature(states, plans, params)
    reference = lambda: _forward_function(  # noqa: E731
        variant, heads, activation, autocast_code, eps, architecture, False
    )(*states, *plans, *params)
    if not _supported(variant, states, params, activation):
        return reference()
    _LAUNCH_STATS[f"{variant}_forward_eligible"] += 1
    _mark_dynamic_rows(variant, states, plans)
    failure_key = (
        variant,
        heads,
        activation,
        autocast_code,
        eps,
        architecture,
    )
    try:
        result = _launch_forward(
            variant,
            states,
            plans,
            params,
            heads,
            activation,
            autocast_code,
            eps,
        )
        _LAUNCH_STATS[f"{variant}_forward_launched"] += 1
        return result
    except Exception as exc:
        _FAILED_FORWARD_SHAPES[failure_key] = f"{type(exc).__name__}: {exc}"
        raise FusedLatentCompileError(
            f"eligible CUDA {variant} latent forward failed; refusing to "
            "silently de-fuse the ACT trunk"
        ) from exc


@torch.library.custom_op("mantisnet::act_state_latent_pass", mutates_args=())
def _state_op(
    latent_inv: Tensor,
    latent_axis: Tensor,
    cell_inv: Tensor,
    cell_axis: Tensor,
    window_inv: Tensor,
    window_axis: Tensor,
    plans: list[Tensor],
    params: list[Tensor],
    heads: int,
    activation: str,
    autocast_code: int,
    eps: list[float],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return _dispatch_forward(
        "state",
        (latent_inv, latent_axis, cell_inv, cell_axis, window_inv, window_axis),
        tuple(plans),
        tuple(params),
        heads,
        activation,
        autocast_code,
        tuple(eps),
    )


@_state_op.register_fake
def _(
    latent_inv,
    latent_axis,
    cell_inv,
    cell_axis,
    window_inv,
    window_axis,
    plans,
    params,
    heads,
    activation,
    autocast_code,
    eps,
):
    output_dtype = torch.promote_types(latent_inv.dtype, params[21].dtype)
    return tuple(
        torch.empty_like(value, dtype=output_dtype)
        for value in (
            latent_inv,
            latent_axis,
            cell_inv,
            cell_axis,
            window_inv,
            window_axis,
        )
    )


@torch.library.custom_op("mantisnet::act_action_latent_pass", mutates_args=())
def _action_op(
    latent_inv: Tensor,
    action_inv: Tensor,
    action_axis: Tensor,
    plans: list[Tensor],
    params: list[Tensor],
    heads: int,
    activation: str,
    autocast_code: int,
    eps: list[float],
) -> tuple[Tensor, Tensor]:
    return _dispatch_forward(
        "action",
        (latent_inv, action_inv, action_axis),
        tuple(plans),
        tuple(params),
        heads,
        activation,
        autocast_code,
        tuple(eps),
    )


@_action_op.register_fake
def _(
    latent_inv,
    action_inv,
    action_axis,
    plans,
    params,
    heads,
    activation,
    autocast_code,
    eps,
):
    latent_dtype = torch.promote_types(latent_inv.dtype, params[16].dtype)
    action_dtype = torch.promote_types(action_inv.dtype, params[46].dtype)
    return (
        torch.empty_like(latent_inv, dtype=latent_dtype),
        torch.empty_like(action_inv, dtype=action_dtype),
    )


def _setup(variant: str, ctx, inputs) -> None:
    state_count = 6 if variant == "state" else 3
    states = tuple(inputs[:state_count])
    plans = tuple(inputs[state_count])
    params = tuple(inputs[state_count + 1])
    ctx.variant = variant
    ctx.state_count = state_count
    ctx.plan_count = len(plans)
    ctx.param_count = len(params)
    ctx.settings = inputs[state_count + 2 :]
    ctx.save_for_backward(*states, *plans, *params)


def _state_setup(ctx, inputs, output) -> None:
    _setup("state", ctx, inputs)


def _action_setup(ctx, inputs, output) -> None:
    _setup("action", ctx, inputs)


def _recompute_gradients(ctx, grad_outputs: tuple[Tensor, ...]):
    saved = ctx.saved_tensors
    states = tuple(saved[: ctx.state_count])
    plans = tuple(saved[ctx.state_count : ctx.state_count + ctx.plan_count])
    params = tuple(saved[ctx.state_count + ctx.plan_count :])
    heads, activation, autocast_code, eps = ctx.settings
    eps = tuple(eps)
    supported = _supported(ctx.variant, states, params, activation)
    if supported:
        _LAUNCH_STATS[f"{ctx.variant}_backward_eligible"] += 1

    def run(use_compiled: bool):
        with torch.enable_grad():
            state_leaves = tuple(
                value.detach().requires_grad_(True) for value in states
            )
            param_leaves = tuple(
                value.detach().requires_grad_(True) for value in params
            )
            if use_compiled:
                _mark_dynamic_rows(ctx.variant, state_leaves, plans)
            function = _forward_function(
                ctx.variant,
                heads,
                activation,
                autocast_code,
                eps,
                _architecture_signature(state_leaves, plans, param_leaves),
                use_compiled,
            )
            output = function(*state_leaves, *plans, *param_leaves)
            seeds = tuple(
                torch.zeros_like(value) if gradient is None else gradient
                for value, gradient in zip(output, grad_outputs)
            )
            gradients = torch.autograd.grad(
                output,
                (*state_leaves, *param_leaves),
                seeds,
                allow_unused=False,
            )
        return gradients

    if supported:
        architecture = _architecture_signature(states, plans, params)
        failure_key = (
            ctx.variant,
            heads,
            activation,
            autocast_code,
            eps,
            architecture,
            "backward",
        )
        try:
            gradients = run(True)
            _LAUNCH_STATS[f"{ctx.variant}_backward_launched"] += 1
        except Exception as exc:
            _FAILED_BACKWARD_SHAPES[failure_key] = (
                f"{type(exc).__name__}: {exc}"
            )
            raise FusedLatentCompileError(
                f"eligible CUDA {ctx.variant} latent recompute backward "
                "failed; refusing to silently de-fuse the ACT trunk"
            ) from exc
    else:
        gradients = run(False)
    state_grads = gradients[: ctx.state_count]
    param_grads = list(gradients[ctx.state_count :])
    plan_grads = [None] * ctx.plan_count
    return (*state_grads, plan_grads, param_grads, None, None, None, None)


def _state_backward(ctx, *grad_outputs):
    return _recompute_gradients(ctx, tuple(grad_outputs))


def _action_backward(ctx, *grad_outputs):
    return _recompute_gradients(ctx, tuple(grad_outputs))


_state_op.register_autograd(_state_backward, setup_context=_state_setup)
_action_op.register_autograd(_action_backward, setup_context=_action_setup)


def _autocast_code(sample: Tensor) -> int:
    if not sample.is_cuda or not torch.is_autocast_enabled("cuda"):
        return 0
    dtype = torch.get_autocast_dtype("cuda")
    if dtype == torch.float16:
        return 1
    if dtype == torch.bfloat16:
        return 2
    return 0


def state_parameters(latent_pass) -> tuple[Tensor, ...]:
    """The default two-family pass's parameters in registered-module order."""
    if tuple(latent_pass.entity_names) != ("cell", "window"):
        raise ValueError(
            "state latent fusion requires entity_names=('cell', 'window')"
        )
    params = tuple(latent_pass.parameters())
    if len(params) != _STATE_COUNT:
        raise ValueError(
            f"default state LatentPass has {_STATE_COUNT} parameters, got "
            f"{len(params)}; use the eager path for this ablation"
        )
    return params


def _zero_dropout(module) -> bool:
    return type(module) is nn.Dropout and float(module.p) == 0.0


def _learned_axis_pool(module) -> bool:
    return type(module) is AxisPool and module.mode == "learned_attention"


def _layout_digest(module) -> str:
    cached = _LAYOUT_DIGEST_CACHE.get(module)
    if cached is not None:
        return cached
    modules = "\n".join(
        f"{name}:{type(child).__module__}.{type(child).__qualname__}"
        for name, child in module.named_modules()
    )
    parameters = "\n".join(dict(module.named_parameters()))
    payload = f"{modules}\n--parameters--\n{parameters}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    _LAYOUT_DIGEST_CACHE[module] = digest
    return digest


def supports_state_pass(latent_pass) -> bool:
    """Whether ``latent_pass`` has the default two-family fused layout."""
    cfg = latent_pass.cfg
    return (
        tuple(latent_pass.entity_names) == ("cell", "window")
        and latent_pass.has_inv
        and latent_pass.has_axis
        and cfg.d_axis > 0
        and cfg.axis_pool_mode == "learned_attention"
        and cfg.dropout == 0.0
        and cfg.activation in _ACTIVATIONS
        and _zero_dropout(latent_pass.drop)
        and type(latent_pass.axis_mix) is AxisMix
        and latent_pass.axis_mix.d_axis == cfg.d_axis
        and latent_pass.axis_mix.activation == cfg.activation
        and _zero_dropout(latent_pass.axis_mix.drop)
        and _learned_axis_pool(latent_pass.pool_src_axis)
        and _learned_axis_pool(latent_pass.pool_latent_axis)
        and _layout_digest(latent_pass) == _STATE_LAYOUTS[cfg.activation]
    )


def action_parameters(latent_pass) -> tuple[Tensor, ...]:
    """The default invariant-only action pass's registered parameters."""
    if tuple(latent_pass.entity_names) != ("action",):
        raise ValueError("action latent fusion requires entity_names=('action',)")
    params = tuple(latent_pass.parameters())
    if len(params) != _ACTION_COUNT:
        raise ValueError(
            f"default action LatentPass has {_ACTION_COUNT} parameters, got "
            f"{len(params)}; use the eager path for this ablation"
        )
    return params


def supports_action_pass(latent_pass) -> bool:
    """Whether ``latent_pass`` has the default one-family fused layout."""
    cfg = latent_pass.cfg
    return (
        tuple(latent_pass.entity_names) == ("action",)
        and latent_pass.has_inv
        and not latent_pass.has_axis
        and cfg.d_axis > 0
        and cfg.axis_pool_mode == "learned_attention"
        and cfg.dropout == 0.0
        and cfg.activation in _ACTIVATIONS
        and _zero_dropout(latent_pass.drop)
        and _learned_axis_pool(latent_pass.pool_src_axis)
        and _layout_digest(latent_pass) == _ACTION_LAYOUT
    )


def state_eps(latent_pass) -> tuple[float, ...]:
    return (
        latent_pass.norm_src[0].inv.eps,
        latent_pass.norm_src[0].axis.eps,
        latent_pass.norm_src[1].inv.eps,
        latent_pass.norm_src[1].axis.eps,
        latent_pass.norm_read_q_inv.eps,
        latent_pass.norm_read_q_axis.eps,
        latent_pass.norm_mix_inv.eps,
        latent_pass.norm_mix_axis.eps,
        latent_pass.axis_mix.norm.inv.eps,
        latent_pass.axis_mix.norm.axis.eps,
        latent_pass.norm_bcast_src_inv.eps,
        latent_pass.norm_bcast_src_axis.eps,
        latent_pass.norm_bcast_q_inv[0].eps,
        latent_pass.norm_bcast_q_inv[1].eps,
        latent_pass.norm_bcast_q_axis[0].eps,
        latent_pass.norm_bcast_q_axis[1].eps,
    )


def action_eps(latent_pass) -> tuple[float, ...]:
    return (
        latent_pass.norm_src[0].inv.eps,
        latent_pass.norm_src[0].axis.eps,
        latent_pass.norm_read_q_inv.eps,
        latent_pass.norm_mix_inv.eps,
        latent_pass.norm_bcast_src_inv.eps,
        latent_pass.norm_bcast_q_inv[0].eps,
    )


def _plan_tensors(
    segments: LatentSegments,
    families: Sequence[tuple[Tensor, Tensor]],
) -> list[Tensor]:
    return [
        segments.ranges,
        segments.range_base,
        segments.counts,
        segments.row_pos,
        *(value for pair in families for value in pair),
    ]


def state_latent_pass(
    latent_inv: Tensor,
    latent_axis: Tensor,
    cell_inv: Tensor,
    cell_axis: Tensor,
    window_inv: Tensor,
    window_axis: Tensor,
    *,
    segments: LatentSegments,
    cell_offsets: Tensor,
    cell_row_pos: Tensor,
    window_offsets: Tensor,
    window_row_pos: Tensor,
    params: Sequence[Tensor],
    heads: int,
    activation: str,
    eps: Sequence[float],
    dropout: float = 0.0,
    axis_pool_mode: str = "learned_attention",
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Default state pass; returns latent, cell, and window stream pairs."""
    if float(dropout) != 0.0:
        raise ValueError("fused latent recomputation requires dropout=0")
    if axis_pool_mode != "learned_attention":
        raise ValueError("the fused default path requires learned axis pools")
    plans = _plan_tensors(
        segments,
        ((cell_offsets, cell_row_pos), (window_offsets, window_row_pos)),
    )
    return _state_op(
        latent_inv.contiguous(),
        latent_axis.contiguous(),
        cell_inv.contiguous(),
        cell_axis.contiguous(),
        window_inv.contiguous(),
        window_axis.contiguous(),
        [value.contiguous() for value in plans],
        [value.contiguous() for value in params],
        int(heads),
        activation,
        _autocast_code(latent_inv),
        [float(value) for value in eps],
    )


def action_latent_pass(
    latent_inv: Tensor,
    action_inv: Tensor,
    action_axis: Tensor,
    *,
    segments: LatentSegments,
    action_offsets: Tensor,
    action_row_pos: Tensor,
    params: Sequence[Tensor],
    heads: int,
    activation: str,
    eps: Sequence[float],
    dropout: float = 0.0,
    axis_pool_mode: str = "learned_attention",
) -> tuple[Tensor, Tensor]:
    """Default action pass; caller carries ``action_axis`` through unchanged."""
    if float(dropout) != 0.0:
        raise ValueError("fused latent recomputation requires dropout=0")
    if axis_pool_mode != "learned_attention":
        raise ValueError("the fused default path requires learned axis pools")
    plans = _plan_tensors(segments, ((action_offsets, action_row_pos),))
    return _action_op(
        latent_inv.contiguous(),
        action_inv.contiguous(),
        action_axis.contiguous(),
        [value.contiguous() for value in plans],
        [value.contiguous() for value in params],
        int(heads),
        activation,
        _autocast_code(latent_inv),
        [float(value) for value in eps],
    )


__all__ = [
    "FusedLatentCompileError",
    "action_eps",
    "action_latent_pass",
    "action_parameters",
    "clear_compile_caches",
    "clear_failure_caches",
    "launch_stats",
    "reset_launch_stats",
    "supports_action_pass",
    "supports_state_pass",
    "state_eps",
    "state_latent_pass",
    "state_parameters",
]
