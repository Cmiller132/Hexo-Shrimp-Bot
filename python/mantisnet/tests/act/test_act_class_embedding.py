"""The fused §19.2 class lookups against an independent CSR oracle.

The two class grids contain many duplicate ids.  Forward is only two table
lookups, but backward is the important part: every class must receive an
ordered reduction over the flattened action rows, with no atomic scatter.
The oracle below builds its class buckets in Python and accumulates rows one
at a time; it does not use either implementation helper from the custom op.

Float64 CPU data is dyadic, so its forward and gradients are required to be
bit exact.  CUDA fp32 is held to the same fp32-accumulation oracle.  Bfloat16
is checked both against the oracle after quantising the inputs and against the
unquantised fp32 anchor; the latter bound includes input and output rounding.
"""

from __future__ import annotations

import pytest
import torch

from mantisnet.models.mantis_act import class_embedding as kernel
from mantisnet.models.mantis_act.class_embedding import class_pair_embedding


SEED = 20260809

_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the fused class lookup needs CUDA"
)


def stable_csr(index: torch.Tensor, classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build class-major rows with a plain stable Python bucket pass."""
    buckets: list[list[int]] = [[] for _ in range(classes)]
    for row, value in enumerate(index.tolist()):
        value = int(value)
        if not 0 <= value < classes:
            raise ValueError(f"class {value} is outside [0, {classes})")
        buckets[value].append(row)

    ptr = [0]
    rows: list[int] = []
    for bucket in buckets:
        rows.extend(bucket)
        ptr.append(len(rows))
    return torch.tensor(ptr, dtype=torch.int32), torch.tensor(rows, dtype=torch.int32)


def oracle_forward(
    post_weight: torch.Tensor,
    status_weight: torch.Tensor,
    post_index: torch.Tensor,
    status_index: torch.Tensor,
) -> torch.Tensor:
    """§19.2's relation row, evaluated independently one row at a time."""
    accumulate = torch.float64 if post_weight.dtype == torch.float64 else torch.float32
    rows = []
    for post_class, status_class in zip(post_index.tolist(), status_index.tolist()):
        value = post_weight[int(post_class)].to(accumulate)
        value = value + status_weight[int(status_class)].to(accumulate)
        rows.append(value.to(post_weight.dtype))
    if not rows:
        return post_weight.new_empty((0, post_weight.shape[1]))
    return torch.stack(rows)


def oracle_gradients(
    grad_out: torch.Tensor,
    post_index: torch.Tensor,
    status_index: torch.Tensor,
    post_classes: int,
    status_classes: int,
    output_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ordered table gradients, accumulated in original flattened-row order."""
    accumulate = torch.float64 if grad_out.dtype == torch.float64 else torch.float32
    width = int(grad_out.shape[1])
    post = torch.zeros(post_classes, width, dtype=accumulate)
    status = torch.zeros(status_classes, width, dtype=accumulate)
    for row, (post_class, status_class) in enumerate(
        zip(post_index.tolist(), status_index.tolist())
    ):
        gradient = grad_out[row].to(accumulate)
        post[int(post_class)] += gradient
        status[int(status_class)] += gradient
    return post.to(output_dtype), status.to(output_dtype)


def fixed_case(width: int = 5):
    post_index = torch.tensor(
        [6, 1, 6, 0, 3, 1, 2, 6, 3, 3, 0, 1, 6, 2, 3], dtype=torch.int32
    )
    status_index = torch.tensor(
        [2, 0, 2, 3, 1, 0, 2, 1, 3, 3, 0, 2, 2, 1, 0], dtype=torch.int32
    )
    post_ptr, post_rows = stable_csr(post_index, 7)
    status_ptr, status_rows = stable_csr(status_index, 4)
    post_weight = (
        torch.arange(7 * width, dtype=torch.float64).reshape(7, width) - 11
    ) / 8
    status_weight = (
        torch.arange(4 * width, dtype=torch.float64).reshape(4, width) + 3
    ) / 16
    return (
        post_weight,
        status_weight,
        post_index,
        status_index,
        post_ptr,
        post_rows,
        status_ptr,
        status_rows,
    )


def random_case(*, rows: int, width: int, post_classes: int = 29):
    generator = torch.Generator().manual_seed(SEED + rows + width)
    post_index = torch.randint(
        post_classes, (rows,), generator=generator, dtype=torch.int32
    )
    status_index = torch.randint(4, (rows,), generator=generator, dtype=torch.int32)
    # Make every class nonempty while retaining substantial repeated-class runs.
    post_index[:post_classes] = torch.arange(post_classes, dtype=torch.int32)
    status_index[:4] = torch.arange(4, dtype=torch.int32)
    post_ptr, post_rows = stable_csr(post_index, post_classes)
    status_ptr, status_rows = stable_csr(status_index, 4)
    return (
        torch.randn(post_classes, width, generator=generator),
        torch.randn(4, width, generator=generator),
        post_index,
        status_index,
        post_ptr,
        post_rows,
        status_ptr,
        status_rows,
    )


def on_device(case, device: str, dtype: torch.dtype):
    post, status, *plans = case
    return (
        post.to(device=device, dtype=dtype),
        status.to(device=device, dtype=dtype),
        *(plan.to(device) for plan in plans),
    )


def relative(got: torch.Tensor, want: torch.Tensor) -> float:
    gap = float((got.detach().float().cpu() - want.detach().float().cpu()).abs().max())
    scale = float(want.detach().float().cpu().abs().max())
    return gap if scale == 0.0 else gap / scale


# ---------------------------------------------------------------------------
# The CPU formula and its plan


def test_python_csr_is_stable_and_covers_each_row_once():
    case = fixed_case()
    for index, ptr, rows, classes in (
        (case[2], case[4], case[5], 7),
        (case[3], case[6], case[7], 4),
    ):
        assert sorted(rows.tolist()) == list(range(index.numel()))
        assert ptr.tolist()[0] == 0
        assert ptr.tolist()[-1] == index.numel()
        for class_id in range(classes):
            got = rows[ptr[class_id] : ptr[class_id + 1]].tolist()
            want = [
                row for row, value in enumerate(index.tolist()) if value == class_id
            ]
            assert got == want


def test_cpu_forward_and_table_gradients_are_exact_against_the_oracle():
    case = fixed_case()
    post = case[0].requires_grad_(True)
    status = case[1].requires_grad_(True)
    got = class_pair_embedding(post, status, *case[2:])
    want = oracle_forward(post.detach(), status.detach(), case[2], case[3])
    assert torch.equal(got, want)

    grad_out = (
        torch.arange(got.numel(), dtype=torch.float64).reshape_as(got).remainder(17) - 8
    ) / 16
    got_grad = torch.autograd.grad(got, (post, status), grad_out)
    want_grad = oracle_gradients(
        grad_out, case[2], case[3], post.shape[0], status.shape[0], torch.float64
    )
    assert torch.equal(got_grad[0], want_grad[0])
    assert torch.equal(got_grad[1], want_grad[1])


def test_gradcheck_of_the_registered_recompute_backward():
    case = fixed_case(width=3)
    post = (case[0] / 7).requires_grad_(True)
    status = (case[1] / 11).requires_grad_(True)

    def run(post_weight, status_weight):
        return class_pair_embedding(post_weight, status_weight, *case[2:])

    assert torch.autograd.gradcheck(run, (post, status), eps=1e-6, atol=1e-8)


def test_noncontiguous_tables_and_indices_are_normalised_by_the_public_wrapper():
    case = fixed_case()
    post = case[0].t().contiguous().t()
    status = case[1].t().contiguous().t()
    index_storage = torch.stack((case[2], case[2]), dim=1)
    post_index = index_storage[:, 0]
    assert not post.is_contiguous()
    assert not status.is_contiguous()
    assert not post_index.is_contiguous()
    got = class_pair_embedding(post, status, post_index, *case[3:])
    want = oracle_forward(post, status, post_index, case[3])
    assert torch.equal(got, want)


def test_structurally_invalid_inputs_fail_loudly():
    case = fixed_case()
    with pytest.raises(ValueError, match="relation weights must both"):
        class_pair_embedding(case[0].flatten(), *case[1:])
    with pytest.raises(ValueError, match="wide against"):
        class_pair_embedding(case[0], case[1][:, :-1], *case[2:])
    with pytest.raises(ValueError, match="share a dtype"):
        class_pair_embedding(case[0], case[1].float(), *case[2:])
    with pytest.raises(TypeError, match="post_index must be int32 or int64"):
        class_pair_embedding(case[0], case[1], case[2].float(), *case[3:])
    with pytest.raises(ValueError, match="against status_index"):
        class_pair_embedding(case[0], case[1], case[2], case[3][:-1], *case[4:])
    with pytest.raises(ValueError, match="cover every flattened action row"):
        class_pair_embedding(*case[:5], case[5][:-1], *case[6:])
    with pytest.raises(ValueError, match="post_ptr must have"):
        class_pair_embedding(*case[:4], case[4][:-1], *case[5:])


# ---------------------------------------------------------------------------
# The Triton forward and ordered reduction against the independent oracle


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@_CUDA
def test_cuda_forward_and_backward_match_the_fp32_anchor(dtype):
    host = random_case(rows=521, width=64)
    post, status, *plans = on_device(host, "cuda", dtype)
    post.requires_grad_(True)
    status.requires_grad_(True)
    generator = torch.Generator().manual_seed(SEED + 1)
    grad_fp32 = torch.randn(521, 64, generator=generator)
    grad_out = grad_fp32.to(device="cuda", dtype=dtype)

    got = class_pair_embedding(post, status, *plans)
    got_grad = torch.autograd.grad(got, (post, status), grad_out)

    quantised_post = post.detach().cpu()
    quantised_status = status.detach().cpu()
    want = oracle_forward(quantised_post, quantised_status, host[2], host[3])
    want_grad = oracle_gradients(
        grad_out.cpu(), host[2], host[3], post.shape[0], status.shape[0], dtype
    )
    kernel_atol = 3e-6 if dtype == torch.float32 else 1.6e-2
    kernel_rtol = 3e-6 if dtype == torch.float32 else 5e-3
    torch.testing.assert_close(got.cpu(), want, atol=kernel_atol, rtol=kernel_rtol)
    for actual, expected in zip(got_grad, want_grad):
        torch.testing.assert_close(
            actual.cpu(), expected, atol=kernel_atol, rtol=kernel_rtol
        )

    # §27's bf16 report is meaningful only relative to the unquantised fp32
    # computation, so retain that anchor in addition to same-dtype parity.
    anchor = oracle_forward(host[0], host[1], host[2], host[3])
    anchor_grad = oracle_gradients(
        grad_fp32, host[2], host[3], post.shape[0], status.shape[0], torch.float32
    )
    anchor_atol = 3e-6 if dtype == torch.float32 else 0.15
    anchor_rtol = 3e-6 if dtype == torch.float32 else 3e-2
    torch.testing.assert_close(
        got.float().cpu(), anchor, atol=anchor_atol, rtol=anchor_rtol
    )
    for actual, expected in zip(got_grad, anchor_grad):
        torch.testing.assert_close(
            actual.float().cpu(), expected, atol=anchor_atol, rtol=anchor_rtol
        )


@_CUDA
def test_repeated_class_gradients_are_bitwise_deterministic_in_fp32():
    host = random_case(rows=1025, width=64, post_classes=17)
    post, status, *plans = on_device(host, "cuda", torch.float32)
    post.requires_grad_(True)
    status.requires_grad_(True)
    generator = torch.Generator(device="cuda").manual_seed(SEED + 2)
    grad_out = torch.randn(1025, 64, generator=generator, device="cuda")

    runs = []
    for _ in range(3):
        out = class_pair_embedding(post, status, *plans)
        gradients = torch.autograd.grad(out, (post, status), grad_out)
        runs.append((out, *gradients))
    for later in runs[1:]:
        for first, other in zip(runs[0], later):
            assert torch.equal(first, other)


@_CUDA
def test_supported_cuda_signature_really_launches_forward_and_backward(monkeypatch):
    """A cached/reference fallback cannot satisfy the device acceptance test."""
    assert kernel._FAILED_SHAPES == {}
    assert kernel._FAILED_BACKWARD_SHAPES == {}
    host = random_case(rows=263, width=64)
    post, status, *plans = on_device(host, "cuda", torch.float32)
    post.requires_grad_(True)
    status.requires_grad_(True)
    assert kernel._supported(post, 64, 263)

    calls = {"forward": 0, "backward": 0}
    original_forward = kernel._launch_forward
    original_backward = kernel._launch_backward

    def counted_forward(*args, **kwargs):
        result = original_forward(*args, **kwargs)
        calls["forward"] += 1
        return result

    def counted_backward(*args, **kwargs):
        result = original_backward(*args, **kwargs)
        calls["backward"] += 1
        return result

    monkeypatch.setattr(kernel, "_launch_forward", counted_forward)
    monkeypatch.setattr(kernel, "_launch_backward", counted_backward)
    class_pair_embedding(post, status, *plans).square().sum().backward()
    torch.cuda.synchronize()

    assert calls == {"forward": 1, "backward": 1}
    assert kernel._FAILED_SHAPES == {}
    assert kernel._FAILED_BACKWARD_SHAPES == {}
