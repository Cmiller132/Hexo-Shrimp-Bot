"""The fused §19.2 row ops against their references (§36).

Three layers, because they catch different things.

The oracle is a per-row Python loop written from §19.2's own four lines. It
shares no indexing with either implementation, so it catches a wrong sentinel,
a row that read the wrong window, an axis half taken from the wrong channel, or
a gate applied to the wrong term. It runs against the torch reference on a
board small enough to loop over.

The parity tests hold the kernel against that reference on the builder's real
row grid, at both stream widths and in both channel modes, with the sentinel
fraction the builder actually produces — 89% of the rows at ply 21 and 78% at
ply 161, which is the regime the sentinel branch exists for.

The gradient tests are `torch.autograd.gradcheck` in float64 against the
analytic reference backward, plus a fused-versus-reference comparison of the
same gradients in fp32. Float64 falls back to the reference by signature, so
gradcheck validates the *formula* — that recomputing the forward really does
give the backward every term it needs — and the parity test validates the
kernel that implements it. Neither alone would.

Tolerances. Every disagreement here is fp32 reassociation: the kernel sums a
row's width in one order and cuBLAS in another, and the table gradients are
summed over a grid of programs rather than by one GEMM. Measured across the
cases below the worst is 6.9e-7 relative for a table gradient and under 5e-7
for everything else, and the bound is set two orders above the measurement.
"""

from __future__ import annotations

import math
import random

import hexo_py
import pytest
import torch

from mantisnet.models.mantis_act import post_rows as kernel
from mantisnet.models.mantis_act.builder import build
from mantisnet.models.mantis_act.config import PRESETS
from mantisnet.models.mantis_act.equivariant import AXIS_CHANNELS
from mantisnet.models.mantis_act.packed import collate
from mantisnet.models.mantis_act.post_rows import row_gate, sentinel_gather

SEED = 20260806
FULL = PRESETS["full_act_v4"]

# Boards where the sentinel is everything (an empty board has no window at all),
# where it is most of the grid, and where the window family is dense.
PLIES = (0, 2, 21, 60)
SMALL_PLIES = (0, 2)

# Justified in the module docstring.
TOL = 5e-5

_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the fused row ops need CUDA"
)


# --------------------------------------------------------------------------
# Fixtures


def playout(plies: int, seed: int) -> list[tuple[int, int]]:
    for attempt in range(100):
        rng = random.Random(seed * 7919 + attempt * 31 + plies)
        position = hexo_py.Position()
        moves: list[tuple[int, int]] = []
        for _ in range(plies):
            move = rng.choice(position.legal_moves())
            position.advance(*move)
            moves.append(move)
        if not position.is_terminal:
            return moves
    raise AssertionError(f"no nonterminal {plies}-ply playout in 100 seeds")


@pytest.fixture(scope="module")
def batches() -> dict[int, object]:
    return {
        plies: collate([build(hexo_py.Position.replay(playout(plies, SEED)), FULL)])
        for plies in PLIES
    }


def gather_inputs(batch, channels: int, device, dtype):
    """A source table, a base, and the batch's own ``(N, 3, 6)`` row grid."""
    rows = int(batch.window_pattern_class.shape[0])
    width = FULL.d_inv if channels == 1 else FULL.d_axis
    generator = torch.Generator().manual_seed(SEED + channels)
    source = torch.randn(
        max(rows, 1) * channels, width, generator=generator, dtype=torch.float64
    )
    base = torch.randn(width, generator=generator, dtype=torch.float64)
    index = batch.action_window_index.to(device)
    if rows == 0:
        # An empty board persists no window at all, and every row is sentinel.
        source = source[:0]
    return (
        source[: rows * channels].to(device=device, dtype=dtype),
        base.to(device=device, dtype=dtype),
        index,
    )


def gate_inputs(total: int, width: int, rel_width: int, device, dtype):
    generator = torch.Generator().manual_seed(SEED + width)

    def make(*shape, scale=0.3):
        return (
            torch.randn(*shape, generator=generator, dtype=torch.float64) * scale
        ).to(device=device, dtype=dtype)

    return (
        make(total, width),
        make(width, scale=1.0) + 1.0,
        make(width),
        make(width, width),
        make(total, rel_width),
        make(width, rel_width),
        make(width),
        make(width, rel_width),
        make(width),
    )


def leaves(tensors):
    return [t.detach().clone().requires_grad_(True) for t in tensors]


def worst(a: torch.Tensor, b: torch.Tensor) -> float:
    return (
        (a - b).abs().max() / b.abs().max().clamp(min=1e-30)
    ).item()


# --------------------------------------------------------------------------
# The independent oracle (§36)


def oracle_gather(source, base, index, channels) -> torch.Tensor:
    """§19.2's row states, one row at a time, in Python and float64."""
    n_legal, axes, slots = index.shape
    width = int(base.shape[0])
    out = torch.zeros(n_legal, axes, slots, width, dtype=torch.float64)
    for action in range(n_legal):
        for axis in range(axes):
            for slot in range(slots):
                window = int(index[action, axis, slot])
                if window < 0:
                    out[action, axis, slot] = base.double()
                elif channels == 1:
                    out[action, axis, slot] = source[window].double()
                else:
                    out[action, axis, slot] = source[window * channels + axis].double()
    return out


def oracle_gate(source, ln_weight, ln_bias, wv, relation, wb, bb, wg, bg, eps):
    """§14's combination, one row at a time, in Python and float64."""
    total, width = source.shape
    out = torch.zeros(total, width, dtype=torch.float64)
    for row in range(total):
        w = source[row].double()
        mean = w.mean()
        var = ((w - mean) ** 2).mean()
        u = (w - mean) / math.sqrt(float(var) + eps) * ln_weight.double() + (
            ln_bias.double()
        )
        e = relation[row].double()
        value = wv.double() @ u
        bias = wb.double() @ e + bb.double()
        gate = torch.sigmoid(wg.double() @ e + bg.double())
        out[row] = gate * (value + bias)
    return out


@pytest.mark.parametrize("plies", SMALL_PLIES)
@pytest.mark.parametrize("channels", (1, AXIS_CHANNELS))
def test_the_reference_gather_is_the_per_row_oracle(batches, plies, channels):
    source, base, index = gather_inputs(
        batches[plies], channels, "cpu", torch.float64
    )
    torch.testing.assert_close(
        kernel._gather_reference(source, base, index, channels),
        oracle_gather(source, base, index, channels),
        atol=0,
        rtol=0,
    )


def test_the_reference_gate_is_the_per_row_oracle():
    supplied = gate_inputs(97, FULL.d_inv, FULL.d_rel, "cpu", torch.float64)
    torch.testing.assert_close(
        kernel._row_gate_reference(*supplied, 1e-5),
        oracle_gate(*supplied, 1e-5),
        atol=1e-12,
        rtol=1e-12,
    )


def test_the_row_grid_really_does_carry_the_sentinel(batches):
    """The branch these ops exist for is present, and dominant."""
    empty = batches[0].action_window_index
    assert empty.numel() and bool((empty < 0).all())
    dense = batches[60].action_window_index
    assert bool((dense >= 0).any()) and bool((dense < 0).any())


# --------------------------------------------------------------------------
# Parity: kernel against reference, on the builder's own row grids


@_CUDA
@pytest.mark.parametrize("plies", PLIES)
@pytest.mark.parametrize("channels", (1, AXIS_CHANNELS))
def test_the_fused_gather_matches_the_padded_table(batches, plies, channels):
    supplied = gather_inputs(batches[plies], channels, "cuda", torch.float32)
    fused, reference = leaves(supplied[:2]), leaves(supplied[:2])
    index = supplied[2]
    got = sentinel_gather(*fused, index, channels)
    want = kernel._gather_reference(*reference, index, channels)
    torch.testing.assert_close(got, want, atol=0, rtol=0)

    upstream = torch.randn_like(want)
    got.backward(upstream)
    want.backward(upstream)
    torch.testing.assert_close(fused[0].grad, reference[0].grad, atol=TOL, rtol=TOL)

    # The base's gradient is the one quantity the two paths cannot agree on to
    # fp32 reassociation: it is a sum of every sentinel row — 89% of the grid
    # at ply 21 — and the summands are signed, so the total is small against
    # them and the relative disagreement is large whatever the order. The two
    # are therefore both held against the same sum taken in float64, which is
    # the only comparison that says which one is right; the fused answer is a
    # tree over a fixed grid of programs, the reference's is `index_add_`,
    # whose atomic order is not even the same twice.
    exact = (
        upstream.double()
        .reshape(-1, upstream.shape[-1])[index.reshape(-1) < 0]
        .sum(dim=0)
    )
    scale = exact.abs().max().clamp(min=1e-30)
    fused_error = (fused[1].grad.double() - exact).abs().max() / scale
    eager_error = (reference[1].grad.double() - exact).abs().max() / scale
    assert fused_error < 1e-3 and fused_error <= eager_error * 4


@_CUDA
@pytest.mark.parametrize(
    "width,rel_width", ((FULL.d_inv, FULL.d_rel), (FULL.d_axis, FULL.d_rel))
)
def test_the_fused_gate_matches_the_eager_chain(width, rel_width):
    supplied = gate_inputs(4001, width, rel_width, "cuda", torch.float32)
    fused, reference = leaves(supplied), leaves(supplied)
    got = row_gate(*fused, 1e-5)
    want = kernel._row_gate_reference(*reference, 1e-5)
    assert worst(got, want) < TOL

    upstream = torch.randn_like(want)
    got.backward(upstream)
    want.backward(upstream)
    for name, a, b in zip(
        (
            "source",
            "ln_weight",
            "ln_bias",
            "wv",
            "relation",
            "wb",
            "bb",
            "wg",
            "bg",
        ),
        fused,
        reference,
        strict=True,
    ):
        assert worst(a.grad, b.grad) < TOL, name


@_CUDA
def test_an_unsupported_width_answers_from_the_reference():
    """The guard is a signature test, so a wide row is correct, not refused."""
    wide = 128
    assert not kernel._supported(torch.empty(1, device="cuda"), wide, 8)
    supplied = gate_inputs(65, wide, 32, "cuda", torch.float32)
    torch.testing.assert_close(
        row_gate(*supplied, 1e-5),
        kernel._row_gate_reference(*supplied, 1e-5),
        atol=0,
        rtol=0,
    )


# --------------------------------------------------------------------------
# The formula itself, by finite differences


def test_gradcheck_of_the_analytic_gather_backward(batches):
    """float64 falls back to the reference, so this checks the formula itself."""
    source, base, index = gather_inputs(batches[2], AXIS_CHANNELS, "cpu", torch.float64)
    trimmed = index[:6]
    supplied = tuple(leaves([source, base]))
    run = lambda s, b: sentinel_gather(s, b, trimmed, AXIS_CHANNELS)  # noqa: E731
    assert torch.autograd.gradcheck(run, supplied, eps=1e-6, atol=1e-8)


def test_gradcheck_of_the_analytic_gate_backward():
    supplied = tuple(leaves(gate_inputs(23, 16, 8, "cpu", torch.float64)))
    run = lambda *args: row_gate(*args, 1e-5)  # noqa: E731
    assert torch.autograd.gradcheck(run, supplied, eps=1e-6, atol=1e-8)
