"""Fused window-latent read/broadcast parity and determinism."""

from __future__ import annotations

import pytest
import torch

import mantisnet.window_latents as impl
from mantisnet.window_latents import (
    _reference_broadcast_forward,
    _reference_read_forward,
    broadcast_attention,
    read_attention,
    window_latent_layout,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the window-latent kernels require CUDA"
)

_DEVICE = torch.device("cuda")
_P = 4
_SLOTS = 4
_HEADS = 2
_HD = 16


def _position_rows(*, unsorted: bool) -> torch.Tensor:
    # Position 1 is deliberately empty.
    position = torch.tensor(
        [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 3, 3],
        dtype=torch.long,
        device=_DEVICE,
    )
    if unsorted:
        permutation = torch.tensor(
            [4, 0, 10, 6, 1, 8, 11, 2, 3, 9, 5, 7],
            dtype=torch.long,
            device=_DEVICE,
        )
        position = position.index_select(0, permutation)
    return position


def _inputs(dtype: torch.dtype, seed: int, *, unsorted: bool):
    generator = torch.Generator(device=_DEVICE).manual_seed(seed)
    window_pos = _position_rows(unsorted=unsorted)
    windows = window_pos.shape[0]
    q_read = torch.randn(
        (_P, _SLOTS, _HEADS, _HD),
        device=_DEVICE,
        dtype=dtype,
        generator=generator,
    )
    k_read = torch.randn(
        (windows, _HEADS, _HD), device=_DEVICE, dtype=dtype, generator=generator
    )
    v_read = torch.randn(
        (windows, _HEADS, _HD), device=_DEVICE, dtype=dtype, generator=generator
    )
    q_broadcast = torch.randn(
        (windows, _HEADS, _HD), device=_DEVICE, dtype=dtype, generator=generator
    )
    k_broadcast = torch.randn(
        (_P, _SLOTS, _HEADS, _HD),
        device=_DEVICE,
        dtype=dtype,
        generator=generator,
    )
    v_broadcast = torch.randn(
        (_P, _SLOTS, _HEADS, _HD),
        device=_DEVICE,
        dtype=dtype,
        generator=generator,
    )
    offsets, order = window_latent_layout(window_pos, _P)
    return (
        q_read,
        k_read,
        v_read,
        q_broadcast,
        k_broadcast,
        v_broadcast,
        window_pos,
        offsets,
        order,
    )


def _assert_kernels_healthy() -> None:
    torch.cuda.synchronize()
    assert not impl._FAILED_READ_SHAPES, impl._FAILED_READ_SHAPES
    assert not impl._FAILED_READ_BACKWARD_SHAPES, impl._FAILED_READ_BACKWARD_SHAPES
    assert not impl._FAILED_BROADCAST_SHAPES, impl._FAILED_BROADCAST_SHAPES
    assert not impl._FAILED_BROADCAST_BACKWARD_SHAPES, (
        impl._FAILED_BROADCAST_BACKWARD_SHAPES
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("unsorted", [False, True])
def test_fused_forward_and_gradients_match_literal_references(
    dtype: torch.dtype, unsorted: bool
):
    values = _inputs(dtype, 10_000 + int(unsorted), unsorted=unsorted)
    q_read, k_read, v_read, q_bcast, k_bcast, v_bcast = values[:6]
    window_pos, offsets, order = values[6:]
    generator = torch.Generator(device=_DEVICE).manual_seed(10_100)
    go_read = torch.randn(q_read.shape, device=_DEVICE, generator=generator)
    go_bcast = torch.randn(q_bcast.shape, device=_DEVICE, generator=generator)

    fast_read_inputs = [x.detach().clone().requires_grad_() for x in values[:3]]
    ref_read_inputs = [x.detach().clone().requires_grad_() for x in values[:3]]
    fast_read = read_attention(
        *fast_read_inputs, window_pos, offsets, order
    )
    ref_read = _reference_read_forward(
        *ref_read_inputs, offsets, order
    )[0]
    fast_read_grads = torch.autograd.grad(
        (fast_read * go_read).sum(), tuple(fast_read_inputs)
    )
    ref_read_grads = torch.autograd.grad(
        (ref_read * go_read).sum(), tuple(ref_read_inputs)
    )

    fast_bcast_inputs = [x.detach().clone().requires_grad_() for x in values[3:6]]
    ref_bcast_inputs = [x.detach().clone().requires_grad_() for x in values[3:6]]
    fast_bcast = broadcast_attention(
        *fast_bcast_inputs, window_pos, offsets, order
    )
    ref_bcast = _reference_broadcast_forward(
        *ref_bcast_inputs, window_pos
    )[0]
    fast_bcast_grads = torch.autograd.grad(
        (fast_bcast * go_bcast).sum(), tuple(fast_bcast_inputs)
    )
    ref_bcast_grads = torch.autograd.grad(
        (ref_bcast * go_bcast).sum(), tuple(ref_bcast_inputs)
    )

    _assert_kernels_healthy()
    atol = 2.0e-5 if dtype == torch.float32 else 2.0e-2
    rtol = 2.0e-5 if dtype == torch.float32 else 2.0e-2
    torch.testing.assert_close(fast_read, ref_read, atol=atol, rtol=rtol)
    torch.testing.assert_close(fast_bcast, ref_bcast, atol=atol, rtol=rtol)
    for actual, expected in zip(
        (*fast_read_grads, *fast_bcast_grads),
        (*ref_read_grads, *ref_bcast_grads),
    ):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def test_read_zero_window_position_is_exactly_zero():
    values = _inputs(torch.float32, 20_000, unsorted=True)
    q, k, v = [x.requires_grad_() for x in values[:3]]
    window_pos, offsets, order = values[6:]
    out = read_attention(q, k, v, window_pos, offsets, order)
    grads = torch.autograd.grad(out.square().sum(), (q, k, v))
    _assert_kernels_healthy()
    assert torch.equal(out[1], torch.zeros_like(out[1]))
    assert torch.equal(grads[0][1], torch.zeros_like(grads[0][1]))


@pytest.mark.parametrize("autocast", [False, True])
def test_autocast_on_and_off_preserve_fp32_edge_math(autocast: bool):
    source = _inputs(torch.float32, 30_000 + autocast, unsorted=False)
    window_pos, offsets, order = source[6:]
    projections = [
        torch.nn.Linear(_HD, _HD, bias=False, device=_DEVICE) for _ in range(6)
    ]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        projected = [layer(x) for layer, x in zip(projections, source[:6])]
        read = read_attention(*projected[:3], window_pos, offsets, order)
        broadcast = broadcast_attention(
            *projected[3:], window_pos, offsets, order
        )
    expected_read = _reference_read_forward(
        *projected[:3], offsets, order
    )[0]
    expected_broadcast = _reference_broadcast_forward(
        *projected[3:], window_pos
    )[0]
    _assert_kernels_healthy()
    assert read.dtype == broadcast.dtype == torch.float32
    atol = 2.0e-5 if not autocast else 2.0e-2
    torch.testing.assert_close(read, expected_read, atol=atol, rtol=atol)
    torch.testing.assert_close(
        broadcast, expected_broadcast, atol=atol, rtol=atol
    )


def test_backward_is_bitwise_deterministic_without_atomics():
    values = _inputs(torch.bfloat16, 40_000, unsorted=True)
    window_pos, offsets, order = values[6:]
    generator = torch.Generator(device=_DEVICE).manual_seed(40_001)
    go_read = torch.randn(values[0].shape, device=_DEVICE, generator=generator)
    go_bcast = torch.randn(values[3].shape, device=_DEVICE, generator=generator)

    def run():
        inputs = [x.detach().clone().requires_grad_() for x in values[:6]]
        read = read_attention(*inputs[:3], window_pos, offsets, order)
        broadcast = broadcast_attention(
            *inputs[3:], window_pos, offsets, order
        )
        return torch.autograd.grad(
            (read * go_read).sum() + (broadcast * go_bcast).sum(), tuple(inputs)
        )

    first = run()
    second = run()
    _assert_kernels_healthy()
    for left, right in zip(first, second):
        assert torch.equal(left, right)


def test_dynamic_fullgraph_compile_keeps_both_ops_opaque():
    def attention(qr, kr, vr, qb, kb, vb, position, offsets, order):
        return (
            read_attention(qr, kr, vr, position, offsets, order),
            broadcast_attention(qb, kb, vb, position, offsets, order),
        )

    compiled = torch.compile(attention, dynamic=True, fullgraph=True)
    values = _inputs(torch.bfloat16, 50_000, unsorted=False)
    actual_read, actual_broadcast = compiled(*values)
    expected_read = _reference_read_forward(
        *values[:3], values[7], values[8]
    )[0]
    expected_broadcast = _reference_broadcast_forward(
        *values[3:6], values[6]
    )[0]
    _assert_kernels_healthy()
    torch.testing.assert_close(actual_read, expected_read, atol=2.0e-2, rtol=2.0e-2)
    torch.testing.assert_close(
        actual_broadcast, expected_broadcast, atol=2.0e-2, rtol=2.0e-2
    )
