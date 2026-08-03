"""Fused stone-attention parity, masking, gradients, and compilation."""

from __future__ import annotations

import math

import pytest
import torch

import mantisnet.attention as attention_impl
from mantisnet.attention import (
    _FAILED_BACKWARD_SHAPES,
    _FAILED_SHAPES,
    _attention_reference,
    fused_attention,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the fused attention kernel requires CUDA"
)

_DEVICE = torch.device("cuda")
_HEADS = 4
_D_MAX = 12


def _seq_lens(p: int, t: int, ragged: bool) -> torch.Tensor:
    if not ragged:
        values = [t] * p
    else:
        pattern = [t, 1, max(1, t - 1), max(1, t // 2), min(t, 33), min(t, 65)]
        values = [pattern[i % len(pattern)] for i in range(p)]
    return torch.tensor(values, dtype=torch.int32, device=_DEVICE)


def _coords(p: int, t: int, seq_lens: torch.Tensor) -> torch.Tensor:
    # A compact axial grid exercises every unclamped distance and the clamp.
    slot = torch.arange(t, dtype=torch.int32, device=_DEVICE)
    stone = (slot - 1).clamp_min(0)
    one = torch.stack(
        (
            stone.remainder(31) - 15,
            stone.div(31, rounding_mode="floor") - 8,
        ),
        -1,
    )
    coords = one.unsqueeze(0).expand(p, -1, -1).clone()
    coords[:, 0] = 0
    valid = slot.unsqueeze(0) < seq_lens.unsqueeze(1)
    return coords.masked_fill(~valid.unsqueeze(-1), 0)


def _qkv(p: int, t: int, d: int, dtype: torch.dtype, seed: int):
    generator = torch.Generator(device=_DEVICE).manual_seed(seed)

    def make() -> torch.Tensor:
        # This is the transposed layout produced by the model's Q/K/V projections.
        x = torch.randn(
            (p, t, _HEADS, d), device=_DEVICE, dtype=dtype, generator=generator
        )
        return x.transpose(1, 2)

    return make(), make(), make()


def _dist_bias(dtype: torch.dtype, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=_DEVICE).manual_seed(seed)
    return (
        torch.randn(
            (_HEADS, _D_MAX + 2),
            device=_DEVICE,
            dtype=dtype,
            generator=generator,
        )
        * 0.25
    )


def _dense_bias(
    coords: torch.Tensor, seq_lens: torch.Tensor, dist_bias: torch.Tensor
) -> torch.Tensor:
    """Independently form the bias tensor for the hand-written softmax check."""
    t = coords.shape[1]
    dq = coords[:, :, None, 0] - coords[:, None, :, 0]
    dr = coords[:, :, None, 1] - coords[:, None, :, 1]
    distance = torch.maximum(dq.abs(), torch.maximum(dr.abs(), (dq + dr).abs()))
    bucket = distance.clamp(1, _D_MAX) - 1

    index = torch.arange(t, device=coords.device)
    bucket = bucket.masked_fill(index[:, None] == index[None, :], _D_MAX)
    token = (index[:, None] == 0) | (index[None, :] == 0)
    bucket = bucket.masked_fill(token, _D_MAX + 1)
    key_valid = index.unsqueeze(0) < seq_lens.unsqueeze(1)
    bucket = torch.where(
        key_valid[:, None, :], bucket, bucket.new_full((), _D_MAX + 2)
    )

    table = torch.cat(
        [dist_bias, dist_bias.new_full((_HEADS, 1), -3.0e4)], dim=1
    )
    return table[:, bucket.long()].permute(1, 0, 2, 3)


@pytest.mark.parametrize("p", [1, 3, 64])
@pytest.mark.parametrize("t", [1, 2, 31, 64, 65, 200, 513])
def test_forward_matches_sdpa_across_shape_grid(p: int, t: int):
    q, k, v = _qkv(p, t, 32, torch.bfloat16, seed=10_000 + p * 1000 + t)
    dist_bias = _dist_bias(torch.float32, seed=20_000 + p * 1000 + t)

    cases = [("dense", _seq_lens(p, t, ragged=False))]
    ragged = _seq_lens(p, t, ragged=True)
    if not torch.equal(ragged, cases[0][1]):
        cases.append(("ragged", ragged))

    for name, seq_lens in cases:
        coords = _coords(p, t, seq_lens)
        with torch.no_grad():
            expected = _attention_reference(q, k, v, coords, seq_lens, dist_bias)
            actual = fused_attention(q, k, v, coords, seq_lens, dist_bias)

        torch.cuda.synchronize()
        assert not _FAILED_SHAPES, _FAILED_SHAPES
        assert actual.shape == q.shape
        assert actual.dtype == q.dtype
        assert actual.stride() == q.stride()
        assert torch.isfinite(actual).all(), name
        max_abs = (actual.float() - expected.float()).abs().max().item()
        assert max_abs <= 2.0e-2, f"{name}: max abs diff {max_abs:.6g}"

        invalid = (
            torch.arange(t, device=_DEVICE).unsqueeze(0) >= seq_lens.unsqueeze(1)
        )
        invalid_rows = actual.permute(0, 2, 1, 3)[invalid]
        assert torch.equal(invalid_rows, torch.zeros_like(invalid_rows)), name
        del actual, expected


@pytest.mark.parametrize(
    ("d", "dtype"),
    [(16, torch.bfloat16), (32, torch.float16), (64, torch.bfloat16)],
)
def test_supported_specializations_use_triton(d: int, dtype: torch.dtype):
    assert attention_impl.triton is not None
    p, t = 2, 65
    seq_lens = torch.tensor([t, 17], dtype=torch.int32, device=_DEVICE)
    coords = _coords(p, t, seq_lens)
    q, k, v = _qkv(p, t, d, dtype, seed=30_000 + d)
    dist_bias = _dist_bias(torch.float32, seed=31_000 + d)

    with torch.no_grad():
        expected = _attention_reference(q, k, v, coords, seq_lens, dist_bias)
        actual = fused_attention(q, k, v, coords, seq_lens, dist_bias)

    torch.cuda.synchronize()
    assert not _FAILED_SHAPES, _FAILED_SHAPES
    max_abs = (actual.float() - expected.float()).abs().max().item()
    assert max_abs <= 2.0e-2, f"D={d}, dtype={dtype}: max abs diff {max_abs:.6g}"


def test_fp32_pad_keys_have_exactly_zero_weight():
    p, t, d = 2, 8, 16
    seq_lens = torch.tensor([t, 3], dtype=torch.int32, device=_DEVICE)
    coords = _coords(p, t, seq_lens)
    q, k, _ = _qkv(p, t, d, torch.float32, seed=31)
    q = q * 0.2
    k = k * 0.2
    dist_bias = _dist_bias(torch.float32, seed=32)

    # With identity values, the first T output channels are the attention weights.
    v = torch.zeros((p, _HEADS, t, d), dtype=torch.float32, device=_DEVICE)
    v[..., :t] = torch.eye(t, dtype=torch.float32, device=_DEVICE)

    actual = fused_attention(q, k, v, coords, seq_lens, dist_bias)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    scores = scores + _dense_bias(coords, seq_lens, dist_bias)
    key_valid = (
        torch.arange(t, device=_DEVICE)[None, None, None, :]
        < seq_lens[:, None, None, None]
    )
    weights = scores.masked_fill(~key_valid, -torch.inf).softmax(-1)
    expected = torch.matmul(weights, v)
    query_valid = (
        torch.arange(t, device=_DEVICE)[None, None, :, None]
        < seq_lens[:, None, None, None]
    )
    expected = expected.masked_fill(~query_valid, 0)

    torch.testing.assert_close(actual, expected, atol=5.0e-4, rtol=5.0e-4)
    pad_weights = actual[1, :, :3, 3:t]
    assert torch.equal(pad_weights, torch.zeros_like(pad_weights))


def test_fp32_gradients_match_sdpa():
    p, t, d = 2, 7, 32
    seq_lens = torch.tensor([t, 4], dtype=torch.int32, device=_DEVICE)
    coords = _coords(p, t, seq_lens)
    q0, k0, v0 = _qkv(p, t, d, torch.float32, seed=41)
    bias0 = _dist_bias(torch.float32, seed=42)
    generator = torch.Generator(device=_DEVICE).manual_seed(43)
    upstream = torch.randn(
        (p, _HEADS, t, d),
        dtype=torch.float32,
        device=_DEVICE,
        generator=generator,
    )

    fast_inputs = [
        x.detach().clone().requires_grad_() for x in (q0, k0, v0, bias0)
    ]
    ref_inputs = [
        x.detach().clone().requires_grad_() for x in (q0, k0, v0, bias0)
    ]

    fast_out = fused_attention(
        fast_inputs[0],
        fast_inputs[1],
        fast_inputs[2],
        coords,
        seq_lens,
        fast_inputs[3],
    )
    fast_grads = torch.autograd.grad((fast_out * upstream).sum(), tuple(fast_inputs))

    ref_out = _attention_reference(
        ref_inputs[0],
        ref_inputs[1],
        ref_inputs[2],
        coords,
        seq_lens,
        ref_inputs[3],
    )
    ref_grads = torch.autograd.grad((ref_out * upstream).sum(), tuple(ref_inputs))

    for name, actual, expected in zip(
        ("q", "k", "v", "dist_bias"), fast_grads, ref_grads
    ):
        assert torch.isfinite(actual).all(), name
        denominator = expected.norm().clamp_min(1.0e-7)
        relative = ((actual - expected).norm() / denominator).item()
        assert relative <= 3.0e-2, f"{name}: relative gradient error {relative:.6g}"


_GRAD_SHAPES = tuple(
    (p, t, 32)
    for p in (1, 3, 64)
    for t in (2, 31, 64, 65, 200, 513)
) + (
    (3, 65, 16),
    (3, 65, 64),
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(("p", "t", "d"), _GRAD_SHAPES)
def test_triton_gradients_match_reference_across_shape_grid(
    p: int, t: int, d: int, dtype: torch.dtype
):
    q0, k0, v0 = _qkv(
        p,
        t,
        d,
        dtype,
        seed=40_000 + p * 1000 + t * 10 + d,
    )
    bias0 = _dist_bias(torch.float32, seed=50_000 + p * 1000 + t * 10 + d)
    generator = torch.Generator(device=_DEVICE).manual_seed(
        60_000 + p * 1000 + t * 10 + d
    )
    upstream = torch.randn(
        (p, _HEADS, t, d),
        dtype=dtype,
        device=_DEVICE,
        generator=generator,
    )

    cases = [("dense", _seq_lens(p, t, ragged=False))]
    ragged = _seq_lens(p, t, ragged=True)
    if not torch.equal(ragged, cases[0][1]):
        cases.append(("ragged", ragged))

    for case_name, seq_lens in cases:
        coords = _coords(p, t, seq_lens)
        fast_inputs = [
            x.detach().clone().requires_grad_() for x in (q0, k0, v0, bias0)
        ]
        ref_inputs = [
            x.detach().clone().requires_grad_() for x in (q0, k0, v0, bias0)
        ]

        fast_out = fused_attention(
            fast_inputs[0],
            fast_inputs[1],
            fast_inputs[2],
            coords,
            seq_lens,
            fast_inputs[3],
        )
        fast_grads = torch.autograd.grad(
            (fast_out * upstream).sum(), tuple(fast_inputs)
        )

        ref_out = _attention_reference(
            ref_inputs[0],
            ref_inputs[1],
            ref_inputs[2],
            coords,
            seq_lens,
            ref_inputs[3],
        )
        ref_grads = torch.autograd.grad(
            (ref_out * upstream).sum(), tuple(ref_inputs)
        )

        torch.cuda.synchronize()
        assert not _FAILED_SHAPES, _FAILED_SHAPES
        assert not _FAILED_BACKWARD_SHAPES, _FAILED_BACKWARD_SHAPES
        for grad_name, actual, expected in zip(
            ("q", "k", "v", "dist_bias"), fast_grads, ref_grads
        ):
            assert torch.isfinite(actual).all(), f"{case_name}: {grad_name}"
            expected_float = expected.float()
            denominator = expected_float.norm().clamp_min(1.0e-7)
            relative = (
                (actual.float() - expected_float).norm() / denominator
            ).item()
            assert relative <= 3.0e-2, (
                f"{case_name}, P={p}, T={t}, D={d}, dtype={dtype}, {grad_name}: "
                f"relative gradient error {relative:.6g}"
            )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_triton_backward_has_exactly_zero_gradients_for_padded_rows(
    dtype: torch.dtype,
):
    p, t, d = 3, 65, 32
    seq_lens = torch.tensor([t, 17, 1], dtype=torch.int32, device=_DEVICE)
    coords = _coords(p, t, seq_lens)
    q, k, v = _qkv(p, t, d, dtype, seed=70_000)
    inputs = [x.requires_grad_() for x in (q, k, v)]
    dist_bias = _dist_bias(torch.float32, seed=70_001).requires_grad_()
    generator = torch.Generator(device=_DEVICE).manual_seed(70_002)
    upstream = torch.randn(
        (p, _HEADS, t, d),
        dtype=dtype,
        device=_DEVICE,
        generator=generator,
    )

    out = fused_attention(
        inputs[0], inputs[1], inputs[2], coords, seq_lens, dist_bias
    )
    grads = torch.autograd.grad((out * upstream).sum(), tuple(inputs))
    torch.cuda.synchronize()

    assert not _FAILED_SHAPES, _FAILED_SHAPES
    assert not _FAILED_BACKWARD_SHAPES, _FAILED_BACKWARD_SHAPES
    invalid = torch.arange(t, device=_DEVICE).unsqueeze(0) >= seq_lens.unsqueeze(1)
    assert invalid.any()
    for name, grad in zip(("q", "k", "v"), grads):
        padded_rows = grad.permute(0, 2, 1, 3)[invalid]
        assert torch.equal(padded_rows, torch.zeros_like(padded_rows)), name


def test_dynamic_fullgraph_compile_keeps_attention_opaque():
    def attention(q, k, v, coords, seq_lens, dist_bias):
        return fused_attention(q, k, v, coords, seq_lens, dist_bias)

    compiled = torch.compile(attention, dynamic=True, fullgraph=True)
    dist_bias = _dist_bias(torch.float32, seed=51)
    for t in (31, 65):
        seq_lens = _seq_lens(3, t, ragged=True)
        coords = _coords(3, t, seq_lens)
        q, k, v = _qkv(3, t, 32, torch.bfloat16, seed=52 + t)
        with torch.no_grad():
            actual = compiled(q, k, v, coords, seq_lens, dist_bias)
            expected = _attention_reference(q, k, v, coords, seq_lens, dist_bias)
        assert not _FAILED_SHAPES, _FAILED_SHAPES
        assert actual.stride() == q.stride()
        max_abs = (actual.float() - expected.float()).abs().max().item()
        assert max_abs <= 2.0e-2, f"T={t}: max abs diff {max_abs:.6g}"


def test_dynamic_fullgraph_compile_backward_reaches_dist_bias():
    def loss(q, k, v, coords, seq_lens, dist_bias):
        out = fused_attention(q, k, v, coords, seq_lens, dist_bias)
        return out.float().square().mean()

    compiled = torch.compile(loss, dynamic=True, fullgraph=True)
    p, t, d = 2, 17, 32
    seq_lens = torch.tensor([t, 6], dtype=torch.int32, device=_DEVICE)
    coords = _coords(p, t, seq_lens)
    q, k, v = _qkv(p, t, d, torch.bfloat16, seed=61)
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    dist_bias = _dist_bias(torch.float32, seed=62).requires_grad_()

    compiled(q, k, v, coords, seq_lens, dist_bias).backward()
    torch.cuda.synchronize()

    assert not _FAILED_SHAPES, _FAILED_SHAPES
    assert not _FAILED_BACKWARD_SHAPES, _FAILED_BACKWARD_SHAPES
    for name, tensor in (
        ("q", q),
        ("k", k),
        ("v", v),
        ("dist_bias", dist_bias),
    ):
        assert tensor.grad is not None, name
        assert torch.isfinite(tensor.grad).all(), name
        assert tensor.grad.abs().sum().item() > 0, name
