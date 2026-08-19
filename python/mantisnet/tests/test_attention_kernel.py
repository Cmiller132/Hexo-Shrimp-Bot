"""Fused stone-attention parity, masking, gradients, and compilation."""

from __future__ import annotations

import math

import pytest
import torch

import mantisnet.attention as attention_impl
from mantisnet.attention import (
    _FAILED_BACKWARD_SHAPES,
    _FAILED_SHAPES,
    AXIS_CLASSES,
    AXIS_ROWS,
    BIAS_ROWS,
    FAR_BUCKET,
    ORBIT_CLASSES,
    ORBIT_RADIUS,
    PAD_BUCKET,
    SELF_BUCKET,
    TOKEN_BUCKET,
    _attention_reference,
    _bucket_index,
    axis_index,
    compose_bias_table,
    fused_attention,
    orbit_lut,
)
from mantisnet.builder import orbit48_id


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the fused attention kernel requires CUDA"
)

_DEVICE = torch.device("cuda")
_HEADS = 4


def _lut() -> torch.Tensor:
    return orbit_lut(_DEVICE)


def _seq_lens(p: int, t: int, ragged: bool) -> torch.Tensor:
    if not ragged:
        values = [t] * p
    else:
        pattern = [t, 1, max(1, t - 1), max(1, t // 2), min(t, 33), min(t, 65)]
        values = [pattern[i % len(pattern)] for i in range(p)]
    return torch.tensor(values, dtype=torch.int32, device=_DEVICE)


def _coords(p: int, t: int, seq_lens: torch.Tensor) -> torch.Tensor:
    # A compact axial grid exercises every orbit within the radius and FAR.
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


def _orbit_bias(dtype: torch.dtype, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=_DEVICE).manual_seed(seed)
    return (
        torch.randn(
            (_HEADS, BIAS_ROWS),
            device=_DEVICE,
            dtype=dtype,
            generator=generator,
        )
        * 0.25
    )


def _literal_bucket(
    qi: int, ki: int, coords: torch.Tensor, live: int, global_rows: int = 1
) -> int:
    """The §4.1 bucket law written out by hand, off the builder's orbit
    function — independent of the attention module's table and kernels."""
    if ki >= live:
        return PAD_BUCKET
    if qi < global_rows or ki < global_rows:
        return TOKEN_BUCKET
    if qi == ki:
        return SELF_BUCKET
    dq = int(coords[qi, 0]) - int(coords[ki, 0])
    dr = int(coords[qi, 1]) - int(coords[ki, 1])
    distance = max(abs(dq), abs(dr), abs(dq + dr))
    if distance == 0:
        # Only a padded query row (coordinates zeroed) can coincide with a
        # live key; its output is zeroed anyway, and the table maps the zero
        # displacement to SELF.
        return SELF_BUCKET
    if distance > ORBIT_RADIUS:
        return FAR_BUCKET
    return orbit48_id(dq, dr)


def _dense_bias(
    coords: torch.Tensor, seq_lens: torch.Tensor, orbit_bias: torch.Tensor
) -> torch.Tensor:
    """Independently form the bias tensor for the hand-written softmax check."""
    p, t = coords.shape[:2]
    bucket = torch.empty((p, t, t), dtype=torch.long)
    for pi in range(p):
        live = int(seq_lens[pi])
        for qi in range(t):
            for ki in range(t):
                bucket[pi, qi, ki] = _literal_bucket(qi, ki, coords[pi].cpu(), live)
    table = torch.cat(
        [orbit_bias, orbit_bias.new_full((_HEADS, 1), -3.0e4)], dim=1
    )
    return table[:, bucket.to(_DEVICE)].permute(1, 0, 2, 3)


def test_orbit_bucket_semantics_are_hand_derived():
    coords = torch.tensor(
        [
            [
                [0, 0],
                [0, 0],
                [1, 0],
                [0, 2],
                [2, -2],
                [0, -2],
                [13, 0],
                [-1, 13],
                [3, 1],
                [1, 1],
                [1, 0],
            ]
        ],
        dtype=torch.int32,
        device=_DEVICE,
    )
    seq_lens = torch.tensor([10], dtype=torch.int32, device=_DEVICE)

    bucket, valid = _bucket_index(coords, seq_lens, 11, _lut())

    # Every pair against the literal law, then the named precedences.
    for qi in range(11):
        for ki in range(11):
            assert bucket[0, qi, ki].item() == _literal_bucket(
                qi, ki, coords[0].cpu(), 10
            ), (qi, ki)
    named = (
        ("distance one is the first orbit", 1, 2, 0),
        ("on-axis distance two", 1, 3, orbit48_id(0, -2)),
        ("off-axis distance two", 1, 9, orbit48_id(-1, -1)),
        ("same orbit under rotation", 1, 4, bucket[0, 1, 5].item()),
        ("far beyond the radius", 1, 6, FAR_BUCKET),
        ("far beyond the radius, other axis", 1, 7, FAR_BUCKET),
        ("self overrides orbit", 2, 2, SELF_BUCKET),
        ("query token overrides orbit", 0, 2, TOKEN_BUCKET),
        ("key token overrides orbit", 2, 0, TOKEN_BUCKET),
        ("pad overrides orbit", 1, 10, PAD_BUCKET),
        ("pad overrides token", 0, 10, PAD_BUCKET),
        ("pad overrides self", 10, 10, PAD_BUCKET),
    )
    for name, query, key, literal_bucket in named:
        assert bucket[0, query, key].item() == literal_bucket, name
    assert valid.tolist() == [
        [True, True, True, True, True, True, True, True, True, True, False]
    ]
    # The displacement orbit is D6-invariant, so the bias is symmetric in
    # the pair: (dq, dr) and (-dq, -dr) share a row.
    stones = bucket[0, 1:10, 1:10]
    assert torch.equal(stones, stones.t())


def test_orbit_lut_covers_every_orbit_and_only_them():
    lut = _lut().cpu()
    side = 2 * ORBIT_RADIUS + 1
    orbits = set()
    for dq in range(-ORBIT_RADIUS, ORBIT_RADIUS + 1):
        for dr in range(-ORBIT_RADIUS, ORBIT_RADIUS + 1):
            entry = int(lut[(dq + ORBIT_RADIUS) * side + (dr + ORBIT_RADIUS)])
            distance = max(abs(dq), abs(dr), abs(dq + dr))
            if distance == 0:
                assert entry == SELF_BUCKET
            elif distance > ORBIT_RADIUS:
                assert entry == FAR_BUCKET
            else:
                assert entry == orbit48_id(dq, dr)
                orbits.add(entry)
    assert orbits == set(range(ORBIT_CLASSES))


def test_axis_index_groups_orbits_by_distance_and_axis():
    # Independent of the implementation: group the displacements of each
    # orbit and check they share one (distance, on-axis) class, that the
    # classes are numbered in (distance, on-axis first) order, and that the
    # three non-orbit rows keep their own coarse rows.
    index = axis_index("cpu")
    assert index.shape == (BIAS_ROWS,)
    seen: dict[int, tuple[int, bool]] = {}
    for dq in range(-ORBIT_RADIUS, ORBIT_RADIUS + 1):
        for dr in range(-ORBIT_RADIUS, ORBIT_RADIUS + 1):
            distance = max(abs(dq), abs(dr), abs(dq + dr))
            if distance == 0 or distance > ORBIT_RADIUS:
                continue
            key = (distance, dq == 0 or dr == 0 or dq + dr == 0)
            orbit = orbit48_id(dq, dr)
            assert seen.setdefault(orbit, key) == key, (orbit, key)
    classes = sorted(set(seen.values()), key=lambda k: (k[0], not k[1]))
    assert len(classes) == AXIS_CLASSES
    for orbit, key in seen.items():
        assert int(index[orbit]) == classes.index(key), (orbit, key)
    assert int(index[FAR_BUCKET]) == AXIS_CLASSES
    assert int(index[SELF_BUCKET]) == AXIS_CLASSES + 1
    assert int(index[TOKEN_BUCKET]) == AXIS_CLASSES + 2
    assert AXIS_ROWS == AXIS_CLASSES + 3


def test_compose_bias_table_is_coarse_row_plus_residual():
    index = axis_index("cpu")
    generator = torch.Generator().manual_seed(7)
    axis_bias = torch.randn((_HEADS, AXIS_ROWS), generator=generator)
    orbit_bias = torch.randn((_HEADS, ORBIT_CLASSES), generator=generator)
    table = compose_bias_table(axis_bias, orbit_bias, index)
    assert table.shape == (_HEADS, BIAS_ROWS)
    for orbit in range(ORBIT_CLASSES):
        expected = axis_bias[:, int(index[orbit])] + orbit_bias[:, orbit]
        assert torch.equal(table[:, orbit], expected)
    for bucket in (FAR_BUCKET, SELF_BUCKET, TOKEN_BUCKET):
        assert torch.equal(table[:, bucket], axis_bias[:, int(index[bucket])])
    # A zero residual is exactly the coarse table: the residual form starts
    # where the (distance, on-axis) vocabulary starts.
    zero = compose_bias_table(axis_bias, torch.zeros_like(orbit_bias), index)
    assert torch.equal(zero, axis_bias.index_select(1, index))
    with pytest.raises(ValueError, match="axis_bias"):
        compose_bias_table(axis_bias[:, :-1], orbit_bias, index)
    with pytest.raises(ValueError, match="orbit_bias"):
        compose_bias_table(axis_bias, orbit_bias[:, :-1], index)


@pytest.mark.parametrize(
    "shape",
    [(_HEADS, 0), (_HEADS, BIAS_ROWS - 1), (_HEADS, BIAS_ROWS + 1)],
)
def test_public_attention_rejects_wrong_bias_shape(shape: tuple[int, int]):
    p, t, d = 1, 2, 16
    seq_lens = _seq_lens(p, t, ragged=False)
    coords = _coords(p, t, seq_lens)
    q, k, v = _qkv(p, t, d, torch.float16, seed=9_000)
    orbit_bias = torch.zeros(shape, dtype=torch.float32, device=_DEVICE)

    with pytest.raises(ValueError, match="bias_table"):
        fused_attention(q, k, v, coords, seq_lens, orbit_bias, _lut())


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
    orbit_bias = _orbit_bias(torch.float32, seed=31_000 + d)

    with torch.no_grad():
        expected = _attention_reference(q, k, v, coords, seq_lens, orbit_bias, _lut())
        actual = fused_attention(q, k, v, coords, seq_lens, orbit_bias, _lut())

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
    orbit_bias = _orbit_bias(torch.float32, seed=32)

    # With identity values, the first T output channels are the attention weights.
    v = torch.zeros((p, _HEADS, t, d), dtype=torch.float32, device=_DEVICE)
    v[..., :t] = torch.eye(t, dtype=torch.float32, device=_DEVICE)

    actual = fused_attention(q, k, v, coords, seq_lens, orbit_bias, _lut())
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    scores = scores + _dense_bias(coords, seq_lens, orbit_bias)
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
    bias0 = _orbit_bias(torch.float32, seed=42)
    generator = torch.Generator(device=_DEVICE).manual_seed(43)
    upstream = torch.randn(
        (p, _HEADS, t, d),
        dtype=torch.float32,
        device=_DEVICE,
        generator=generator,
    )

    fast_inputs = [x.detach().clone().requires_grad_() for x in (q0, k0, v0, bias0)]
    ref_inputs = [x.detach().clone().requires_grad_() for x in (q0, k0, v0, bias0)]

    fast_out = fused_attention(*fast_inputs[:3], coords, seq_lens, fast_inputs[3], _lut())
    fast_grads = torch.autograd.grad((fast_out * upstream).sum(), tuple(fast_inputs))

    ref_out = _attention_reference(
        *ref_inputs[:3], coords, seq_lens, ref_inputs[3], _lut()
    )
    ref_grads = torch.autograd.grad((ref_out * upstream).sum(), tuple(ref_inputs))

    for name, actual, expected in zip(("q", "k", "v", "orbit_bias"), fast_grads, ref_grads):
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
def test_triton_forward_and_gradients_match_reference_across_shape_grid(
    p: int, t: int, d: int, dtype: torch.dtype
):
    q0, k0, v0 = _qkv(p, t, d, dtype, seed=80_000 + p * 1000 + t * 10 + d)
    bias0 = _orbit_bias(torch.float32, seed=90_000 + p * 1000 + t * 10 + d)
    generator = torch.Generator(device=_DEVICE).manual_seed(
        110_000 + p * 1000 + t * 10 + d
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
            *fast_inputs[:3], coords, seq_lens, fast_inputs[3], _lut()
        )
        fast_grads = torch.autograd.grad(
            (fast_out * upstream).sum(), tuple(fast_inputs)
        )

        ref_out = _attention_reference(
            *ref_inputs[:3], coords, seq_lens, ref_inputs[3], _lut()
        )
        ref_grads = torch.autograd.grad(
            (ref_out * upstream).sum(), tuple(ref_inputs)
        )

        torch.cuda.synchronize()
        assert not _FAILED_SHAPES, _FAILED_SHAPES
        assert not _FAILED_BACKWARD_SHAPES, _FAILED_BACKWARD_SHAPES
        assert fast_out.shape == q0.shape
        assert fast_out.dtype == q0.dtype
        assert fast_out.stride() == q0.stride()
        assert torch.isfinite(fast_out).all(), case_name
        invalid = (
            torch.arange(t, device=_DEVICE).unsqueeze(0) >= seq_lens.unsqueeze(1)
        )
        invalid_rows = fast_out.detach().permute(0, 2, 1, 3)[invalid]
        assert torch.equal(invalid_rows, torch.zeros_like(invalid_rows)), case_name
        max_abs = (fast_out.float() - ref_out.float()).abs().max().item()
        assert max_abs <= 2.0e-2, (
            f"{case_name}, P={p}, T={t}, D={d}, dtype={dtype}: "
            f"max abs diff {max_abs:.6g}"
        )
        for grad_name, actual, expected in zip(
            ("q", "k", "v", "orbit_bias"), fast_grads, ref_grads
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


def test_public_attention_reads_orbit_rows():
    p, t, d = 1, 4, 16
    seq_lens = torch.tensor([t], dtype=torch.int32, device=_DEVICE)
    coords = torch.tensor(
        [[[0, 0], [0, 0], [1, 0], [2, 1]]],
        dtype=torch.int32,
        device=_DEVICE,
    )
    q = torch.zeros((p, _HEADS, t, d), dtype=torch.float16, device=_DEVICE)
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    v[..., :t] = torch.eye(t, dtype=torch.float16, device=_DEVICE)
    stock_bias = torch.zeros((_HEADS, BIAS_ROWS), dtype=torch.float32, device=_DEVICE)
    orbit_bias = stock_bias.clone()
    # Boost the distance-one orbit: rows 1 and 2 are neighbours.
    orbit_bias[:, orbit48_id(1, 0)] = 20.0

    with torch.no_grad():
        expected = _attention_reference(q, k, v, coords, seq_lens, orbit_bias, _lut())
        actual = fused_attention(q, k, v, coords, seq_lens, orbit_bias, _lut())
        stock = fused_attention(q, k, v, coords, seq_lens, stock_bias, _lut())

    torch.cuda.synchronize()
    assert not _FAILED_SHAPES, _FAILED_SHAPES
    max_abs = (actual.float() - expected.float()).abs().max().item()
    assert max_abs <= 2.0e-2, f"max abs diff {max_abs:.6g}"
    neighbour_query_diff = (actual[:, :, 1].float() - stock[:, :, 1].float()).abs()
    assert neighbour_query_diff.max().item() > 0.5


def test_orbit_bias_gradient_histogram_covers_present_rows():
    p, t, d = 1, 4, 32
    seq_lens = torch.tensor([t], dtype=torch.int32, device=_DEVICE)
    coords = torch.tensor(
        [[[0, 0], [0, 0], [1, 0], [20, 0]]],
        dtype=torch.int32,
        device=_DEVICE,
    )
    q = torch.zeros((p, _HEADS, t, d), dtype=torch.float16, device=_DEVICE)
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    v[..., :t] = torch.eye(t, dtype=torch.float16, device=_DEVICE)
    orbit_bias = torch.zeros(
        (_HEADS, BIAS_ROWS), dtype=torch.float32, device=_DEVICE, requires_grad=True
    )
    upstream = torch.zeros_like(q)
    upstream[:, :, 1, 2:4] = 1

    out = fused_attention(q, k, v, coords, seq_lens, orbit_bias, _lut())
    (grad,) = torch.autograd.grad((out * upstream).sum(), (orbit_bias,))
    torch.cuda.synchronize()

    assert not _FAILED_SHAPES, _FAILED_SHAPES
    assert not _FAILED_BACKWARD_SHAPES, _FAILED_BACKWARD_SHAPES
    assert torch.isfinite(grad).all()
    # Query row 1 sees the token (row 0), itself, a distance-one neighbour
    # (row 2), and a FAR stone (row 3); the softmax couples all four rows.
    present_rows = [TOKEN_BUCKET, SELF_BUCKET, orbit48_id(-1, 0), FAR_BUCKET]
    present = grad[:, present_rows]
    assert torch.count_nonzero(present).item() == present.numel()
    absent_rows = [b for b in range(BIAS_ROWS) if b not in present_rows]
    absent = grad[:, absent_rows]
    assert torch.equal(absent, torch.zeros_like(absent))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_triton_backward_has_exactly_zero_gradients_for_padded_rows(
    dtype: torch.dtype,
):
    p, t, d = 3, 65, 32
    seq_lens = torch.tensor([t, 17, 1], dtype=torch.int32, device=_DEVICE)
    coords = _coords(p, t, seq_lens)
    q, k, v = _qkv(p, t, d, dtype, seed=70_000)
    inputs = [x.requires_grad_() for x in (q, k, v)]
    orbit_bias = _orbit_bias(torch.float32, seed=70_001).requires_grad_()
    generator = torch.Generator(device=_DEVICE).manual_seed(70_002)
    upstream = torch.randn(
        (p, _HEADS, t, d),
        dtype=dtype,
        device=_DEVICE,
        generator=generator,
    )

    out = fused_attention(*inputs, coords, seq_lens, orbit_bias, _lut())
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
    def attention(q, k, v, coords, seq_lens, orbit_bias, lut):
        return fused_attention(q, k, v, coords, seq_lens, orbit_bias, lut)

    compiled = torch.compile(attention, dynamic=True, fullgraph=True)
    orbit_bias = _orbit_bias(torch.float32, seed=51)
    for t in (31, 65):
        seq_lens = _seq_lens(3, t, ragged=True)
        coords = _coords(3, t, seq_lens)
        q, k, v = _qkv(3, t, 32, torch.bfloat16, seed=52 + t)
        with torch.no_grad():
            actual = compiled(q, k, v, coords, seq_lens, orbit_bias, _lut())
            expected = _attention_reference(
                q, k, v, coords, seq_lens, orbit_bias, _lut()
            )
        assert not _FAILED_SHAPES, _FAILED_SHAPES
        assert actual.stride() == q.stride()
        max_abs = (actual.float() - expected.float()).abs().max().item()
        assert max_abs <= 2.0e-2, f"T={t}: max abs diff {max_abs:.6g}"


def test_dynamic_fullgraph_compile_backward_reaches_orbit_bias():
    def loss(q, k, v, coords, seq_lens, orbit_bias, lut):
        out = fused_attention(q, k, v, coords, seq_lens, orbit_bias, lut)
        return out.float().square().mean()

    compiled = torch.compile(loss, dynamic=True, fullgraph=True)
    p, t, d = 2, 17, 32
    seq_lens = torch.tensor([t, 6], dtype=torch.int32, device=_DEVICE)
    coords = _coords(p, t, seq_lens)
    q, k, v = _qkv(p, t, d, torch.bfloat16, seed=61)
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    orbit_bias = _orbit_bias(torch.float32, seed=62).requires_grad_()

    compiled(q, k, v, coords, seq_lens, orbit_bias, _lut()).backward()
    torch.cuda.synchronize()

    assert not _FAILED_SHAPES, _FAILED_SHAPES
    assert not _FAILED_BACKWARD_SHAPES, _FAILED_BACKWARD_SHAPES
    for name, tensor in (("q", q), ("k", k), ("v", v), ("orbit_bias", orbit_bias)):
        assert tensor.grad is not None, name
        assert torch.isfinite(tensor.grad).all(), name
        assert tensor.grad.abs().sum().item() > 0, name
