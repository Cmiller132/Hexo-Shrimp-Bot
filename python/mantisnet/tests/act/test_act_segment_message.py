"""The fused relation-gated segment message (§14) against its reference (§36).

Three layers, because they catch different things.

The oracle is a per-edge Python loop written from §14's four lines. It shares
no indexing with either implementation, so it catches a wrong gather, a wrong
destination, a dropped edge, or a channel that leaked between axis streams. It
runs against the torch reference on a board small enough to loop over.

The parity tests hold the kernel against that reference on the builder's real
edge families — both directions of the incidence, hex adjacency, and the
radius edges, in both streams, gated and additive. A fused kernel compared only
against itself has no detector at all: §36 requires the reference to exist and
requires this comparison, and the same requirement is why the reference stays
the literal ``(E, d)`` formulation the kernel exists to avoid.

The gradient tests are ``torch.autograd.gradcheck`` in float64 against the
analytic backward, plus a fused-versus-reference comparison of the same
gradients in fp32. Float64 falls back to the reference by signature, so
gradcheck validates the *formula* — that the message's bilinearity really does
let the backward recompute the per-edge message rather than store it — and the
parity test validates the kernel that implements it. Neither alone would.

Tolerances. Every disagreement here is fp32 reassociation: the two paths sum
one destination's or one relation's contributions in different orders. The
tightest case is the forward, whose longest run is a few hundred edges; the
loosest is the relation-table gradient of hex adjacency, where one class owns
every edge of the family and a fifty-thousand-term fp32 sum genuinely carries
about ``sqrt(n) * eps`` of accumulated rounding. Measured across the families
below the worst is 1.0e-5 relative for a relation-table gradient and 8.3e-7 for
everything else, and each bound is set an order of magnitude above its
measurement.
"""

from __future__ import annotations

import random

import hexo_py
import pytest
import torch

import mantisnet.models.mantis_act.segment_message as kernel
from mantisnet.models.mantis_act.builder import build
from mantisnet.models.mantis_act.config import PRESETS
from mantisnet.models.mantis_act.equivariant import AXIS_CHANNELS
from mantisnet.models.mantis_act.messages import (
    adjacency_edges,
    incidence_edges,
    radius_edges,
)
from mantisnet.models.mantis_act.packed import collate
from mantisnet.models.mantis_act.segment_message import (
    _reference,
    _reference_backward,
    message_plan,
    relation_gated_message,
)

SEED = 20260806
FULL = PRESETS["full_act_v4"]

# Boards dense enough that every family is populated and ragged: a late
# position's radius edges dominate, and its destination runs reach the
# hundreds, which is the regime the kernel is written for.
PLIES = (2, 5, 21, 60)
SMALL_PLIES = (2, 5)

_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the fused message needs CUDA"
)

# Justified in the module docstring.
TOL = 1e-5
TABLE_TOL = 1e-4

D_INV, D_AXIS = 16, 6


# --------------------------------------------------------------------------
# Fixtures


def playout(plies: int, seed: int) -> list[tuple[int, int]]:
    """A seeded nonterminal random playout of exactly ``plies`` placements."""
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


def _families(plies):
    batch = collate(
        [build(hexo_py.Position.replay(playout(p, SEED)), FULL) for p in plies]
    )
    to_windows, to_cells = incidence_edges(batch)
    return {
        "incidence cells->windows": to_windows,
        "incidence windows->cells": to_cells,
        "hex adjacency": adjacency_edges(batch, FULL),
        "occupied radius": radius_edges(batch, FULL),
    }


@pytest.fixture(scope="module")
def families():
    return _families(PLIES)


@pytest.fixture(scope="module")
def small():
    return _families(SMALL_PLIES)


def cases(families):
    """Every (family, stream, gating) combination the trunk can reach."""
    for name, family in families.items():
        for channels, width in ((1, D_INV), (AXIS_CHANNELS, D_AXIS)):
            if channels > 1 and family.axis is None:
                continue
            for gated in (True, False):
                label = f"{name} channels={channels} gated={gated}"
                yield label, family, channels, width, gated


def plan_on(family, channels, device="cpu"):
    """This family's plan, with every table on ``device``."""
    if device == "cpu":
        return family.plan(channels)
    columns = (
        (family.src, family.dst, family.relation, None)
        if channels == 1
        else family.routed()
    )
    return message_plan(
        *(None if column is None else column.to(device) for column in columns),
        family.n_src,
        family.n_dst,
        family.num_relations,
        channels,
        dst_sorted=family.dst_sorted,
    )


def inputs_for(plan, width, *, gated, device="cpu", dtype=torch.float32, seed=0):
    generator = torch.Generator().manual_seed(seed)

    def draw(*shape, positive=False):
        raw = (
            torch.rand(*shape, generator=generator)
            if positive
            else torch.randn(*shape, generator=generator)
        )
        return raw.to(device=device, dtype=dtype).requires_grad_(True)

    return (
        draw(plan.n_src * plan.channels, width),
        draw(plan.n_relations, width, positive=True) if gated else None,
        draw(plan.n_relations, width),
    )


def reference_forward(values, gate, bias, plan):
    return _reference(
        values,
        gate,
        bias,
        plan.dst_ptr,
        plan.dst_src,
        plan.dst_rel,
        plan.dst_axis,
        plan.channels,
    )


def reference_backward(values, gate, bias, plan, grad_out):
    return _reference_backward(
        values,
        gate,
        bias,
        plan.dst_ptr,
        plan.dst_src,
        plan.dst_rel,
        plan.dst_axis,
        plan.channels,
        grad_out,
    )


def relative(got: torch.Tensor, want: torch.Tensor) -> float:
    got, want = got.detach().cpu(), want.detach().cpu()
    scale = float(want.abs().max())
    gap = float((got - want).abs().max())
    return gap if scale == 0.0 else gap / scale


# --------------------------------------------------------------------------
# The oracle


def oracle(values, gate, bias, plan):
    """§14's message, one edge at a time, in Python and in float64.

    The gate and bias arrive already projected over the vocabulary, which is
    what the module does — they are functions of the relation alone — so what
    this checks is the per-edge algebra, the destination, and the axis route.
    """
    channels = plan.channels
    out = torch.zeros(
        plan.n_dst * channels, values.shape[1], dtype=torch.float64
    )
    destinations = plan.edge_destinations().tolist()
    sources = plan.dst_src.tolist()
    relations = plan.dst_rel.tolist()
    routes = [0] * len(sources) if channels == 1 else plan.dst_axis.tolist()
    for source, dst, relation, route in zip(
        sources, destinations, relations, routes
    ):
        message = values[source * channels + route].double()
        if gate is not None:
            message = message * gate[relation].double()
        out[dst * channels + route] += message + bias[relation].double()
    return out


def test_the_reference_is_the_per_edge_message_of_section_14(small):
    for label, family, channels, width, gated in cases(small):
        plan = plan_on(family, channels)
        values, gate, bias = inputs_for(plan, width, gated=gated)
        with torch.no_grad():
            got = reference_forward(values, gate, bias, plan)
            want = oracle(values, gate, bias, plan)
        assert relative(got.double(), want) < TOL, label


# --------------------------------------------------------------------------
# The plan: what the three views promise


def _triples(plan):
    """Each view's ``(dst, src, relation)`` rows, as one sortable int64 key."""
    span = max(plan.n_src, plan.n_dst, plan.n_relations) + 1

    def key(dst, src, relation):
        return (dst.long() * span + src.long()) * span + relation.long()

    def run_keys(ptr, size):
        counts = (ptr[1:] - ptr[:-1]).long()
        return torch.repeat_interleave(torch.arange(size), counts)

    return {
        "destination": key(plan.edge_destinations(), plan.dst_src, plan.dst_rel),
        "source": key(
            plan.src_dst, run_keys(plan.src_ptr, plan.n_src), plan.src_rel
        ),
        "relation": key(
            plan.rel_dst, plan.rel_src, run_keys(plan.rel_ptr, plan.n_relations)
        ),
    }


def test_every_view_holds_the_same_edges_in_a_different_order(families):
    for name, family in families.items():
        for channels in (1, AXIS_CHANNELS):
            if channels > 1 and family.axis is None:
                continue
            views = _triples(plan_on(family, channels))
            canonical = views["destination"].sort().values
            for view, keys in views.items():
                assert torch.equal(keys.sort().values, canonical), f"{name} {view}"


def test_the_csr_offsets_are_ascending_and_span_the_family(families):
    for name, family in families.items():
        for channels in (1, AXIS_CHANNELS):
            if channels > 1 and family.axis is None:
                continue
            plan = plan_on(family, channels)
            for view, ptr, size in (
                ("destination", plan.dst_ptr, plan.n_dst),
                ("source", plan.src_ptr, plan.n_src),
                ("relation", plan.rel_ptr, plan.n_relations),
            ):
                assert ptr.numel() == size + 1, f"{name} {view}"
                assert int(ptr[0]) == 0, f"{name} {view}"
                assert int(ptr[-1]) == plan.n_edges, f"{name} {view}"
                assert bool((ptr[1:] >= ptr[:-1]).all()), f"{name} {view}"


def test_section_7_orders_every_family_but_the_reverse_incidence(families):
    """The destination view is adopted, not rebuilt — except where it cannot be.

    §7 sorts ordinary graph edges by ``(dst, src, relation)`` and the packer
    concatenates positions in order, so hex adjacency, the radius edges and the
    forward direction of the incidence arrive destination-ascending and the
    plan reuses the arrays it was given. The reverse direction of the incidence
    is the exception: it is a window-major slot table read the other way, so
    its destinations are in cell order only by accident. The plan sorts that
    one, once per batch, and this test is what keeps the claim honest if a
    builder's ordering ever changes.
    """
    adopted = {
        "incidence cells->windows": True,
        "incidence windows->cells": False,
        "hex adjacency": True,
        "occupied radius": True,
    }
    for name, family in families.items():
        assert family.dst_sorted is adopted[name], name
        # The declaration against the data it claims to describe, measured
        # here rather than in the forward. This is the whole detector for a
        # `dst_sorted` a builder set wrongly: nothing downstream would fail
        # loudly, the segment reduction would simply walk the wrong runs.
        ascending = bool((family.dst[1:] >= family.dst[:-1]).all())
        assert ascending is adopted[name], name
        plan = plan_on(family, 1)
        if adopted[name]:
            assert torch.equal(plan.dst_src, family.src.to(torch.int32)), name
        assert bool((plan.edge_destinations().diff() >= 0).all()), name


def test_the_plan_is_built_once_and_reused(families):
    family = families["hex adjacency"]
    assert family.plan(1) is family.plan(1)
    assert family.plan(AXIS_CHANNELS) is not family.plan(1)


def test_an_axis_plan_is_built_over_the_routed_subset_alone(families):
    """§11.3's ``-1`` never reaches a plan, and this is why it cannot.

    `TypedEdges.routed` is the only thing in the package that builds an axis
    plan, and it filters on ``axis >= 0`` whenever the family did not declare
    itself ``fully_routed`` off `packed.py:147` or `packed.py:148`. The radius
    family is the one that carries the sentinel, so it is the one that has to
    show a strictly smaller, sentinel-free axis plan.
    """
    family = families["occupied radius"]
    assert family.axis is not None
    assert not family.fully_routed
    _src, _dst, _relation, axis = family.routed()
    assert int(axis.min()) >= 0
    assert axis.numel() == int((family.axis >= 0).sum()) < family.axis.numel()
    assert family.plan(AXIS_CHANNELS).n_edges == axis.numel()


def test_a_plan_refuses_a_channel_count_that_is_not_a_stream(families):
    family = families["hex adjacency"]
    with pytest.raises(ValueError, match="channels must be 1 or 3"):
        message_plan(
            family.src,
            family.dst,
            family.relation,
            family.axis,
            family.n_src,
            family.n_dst,
            family.num_relations,
            2,
            dst_sorted=family.dst_sorted,
        )


def test_a_message_refuses_a_table_the_plan_does_not_describe(families):
    plan = plan_on(families["hex adjacency"], 1)
    values, gate, bias = inputs_for(plan, D_INV, gated=True)
    with pytest.raises(ValueError, match="relation classes"):
        relation_gated_message(values, gate, bias[:-1], plan)
    with pytest.raises(ValueError, match="rows against the plan"):
        relation_gated_message(values[:-1], gate, bias, plan)


# --------------------------------------------------------------------------
# Gradients of the reference: the formula the kernel implements


def test_gradcheck_of_the_analytic_backward(small):
    """float64 falls back to the reference, so this checks the formula itself.

    The message is bilinear in the value and the gate and affine in the bias,
    which is the whole reason the backward may recompute it instead of storing
    a per-edge tensor. A numerical derivative is what says the claim holds.
    """
    family = small["hex adjacency"]
    for channels, width in ((1, 3), (AXIS_CHANNELS, 2)):
        plan = plan_on(family, channels)
        for gated in (True, False):
            values, gate, bias = inputs_for(
                plan, width, gated=gated, dtype=torch.float64
            )
            supplied = tuple(t for t in (values, gate, bias) if t is not None)

            def run(*args, gated=gated):
                rest = list(args)
                value = rest.pop(0)
                gate_arg = rest.pop(0) if gated else None
                return relation_gated_message(value, gate_arg, rest.pop(0), plan)

            assert torch.autograd.gradcheck(run, supplied, eps=1e-6, atol=1e-8)


def test_the_additive_control_receives_no_gate_gradient(families):
    """§14's ``incidence_message="additive"`` has no gate to differentiate.

    Without the gate the aggregate is the plain sum of a destination's sources
    and its relations' biases, so the bias gradient of an all-ones output
    gradient is exactly the family's class histogram — a closed form, not a
    second implementation.
    """
    plan = plan_on(families["occupied radius"], 1)
    values, _, bias = inputs_for(plan, D_INV, gated=False)
    relation_gated_message(values, None, bias, plan).sum().backward()
    counts = torch.bincount(plan.dst_rel.long(), minlength=plan.n_relations)
    assert torch.allclose(
        bias.grad, counts.float().unsqueeze(1).expand_as(bias.grad)
    )
    assert values.grad is not None


# --------------------------------------------------------------------------
# The kernel against the reference


@_CUDA
def test_the_fused_forward_matches_the_reference(families):
    for label, family, channels, width, gated in cases(families):
        host, device = plan_on(family, channels), plan_on(family, channels, "cuda")
        values, gate, bias = inputs_for(host, width, gated=gated)
        on_device = tuple(
            None if t is None else t.detach().cuda() for t in (values, gate, bias)
        )
        with torch.no_grad():
            want = reference_forward(values, gate, bias, host)
            got = relation_gated_message(*on_device, device)
        assert relative(got, want) < TOL, label


@_CUDA
def test_the_fused_backward_matches_the_reference(families):
    for label, family, channels, width, gated in cases(families):
        host, device = plan_on(family, channels), plan_on(family, channels, "cuda")
        values, gate, bias = inputs_for(host, width, gated=gated)
        generator = torch.Generator().manual_seed(11)
        grad_out = torch.randn(host.n_dst * channels, width, generator=generator)
        answers = reference_backward(values, gate, bias, host, grad_out)
        want = [t for t in answers if t.numel()]

        on_device = tuple(
            None if t is None else t.detach().cuda().requires_grad_(True)
            for t in (values, gate, bias)
        )
        out = relation_gated_message(*on_device, device)
        got = torch.autograd.grad(
            out, [t for t in on_device if t is not None], grad_out.cuda()
        )
        # The value gradient is a node-length run; the two table gradients sum
        # a whole relation class, which is where the reassociation lives.
        for index, (a, b, bound) in enumerate(
            zip(got, want, [TOL] + [TABLE_TOL] * (len(want) - 1))
        ):
            assert relative(a, b) < bound, f"{label} gradient {index}"


@_CUDA
def test_repeated_runs_agree_bit_for_bit(families):
    """Determinism, not merely accuracy.

    The D6 tolerance analysis reads a residual as reassociation noise. That
    reading is only available if the residual is the same every time, which is
    what the sliced-partial relation gradient buys over an atomic scatter.
    """
    for label, family, channels, width, gated in cases(families):
        plan = plan_on(family, channels, "cuda")
        values, gate, bias = inputs_for(
            plan, width, gated=gated, device="cuda", seed=3
        )
        generator = torch.Generator(device="cuda").manual_seed(5)
        grad_out = torch.randn(
            plan.n_dst * channels, width, generator=generator, device="cuda"
        )
        supplied = [t for t in (values, gate, bias) if t is not None]
        runs = []
        for _ in range(3):
            out = relation_gated_message(values, gate, bias, plan)
            runs.append((out, *torch.autograd.grad(out, supplied, grad_out)))
        for later in runs[1:]:
            for first, other in zip(runs[0], later):
                assert torch.equal(first, other), label


@_CUDA
def test_the_fallback_agrees_with_the_kernel_it_replaces(families):
    """An unsupported signature must scatter rather than fail.

    The failure caches are how a launch that raised once stops being retried,
    so poisoning them is the honest way to reach the fallback: the op takes the
    same inputs and must answer the same numbers.
    """
    plan = plan_on(families["occupied radius"], 1, "cuda")
    values, gate, bias = inputs_for(
        plan, D_INV, gated=True, device="cuda", seed=7
    )
    supplied = (values, gate, bias)
    fused = relation_gated_message(*supplied, plan)
    fused_grads = torch.autograd.grad(fused, supplied, torch.ones_like(fused))

    key = kernel._shape_key(values, 1, True)
    backward_key = key + (torch.float32,)
    kernel._FAILED_SHAPES[key] = "poisoned by the test"
    kernel._FAILED_BACKWARD_SHAPES[backward_key] = "poisoned by the test"
    try:
        fallen = relation_gated_message(*supplied, plan)
        fallen_grads = torch.autograd.grad(fallen, supplied, torch.ones_like(fallen))
    finally:
        kernel._FAILED_SHAPES.pop(key, None)
        kernel._FAILED_BACKWARD_SHAPES.pop(backward_key, None)

    assert relative(fallen, fused) < TOL
    for index, (fell, fast) in enumerate(zip(fallen_grads, fused_grads)):
        assert relative(fell, fast) < TABLE_TOL, f"gradient {index}"


@_CUDA
def test_the_whole_trunk_agrees_with_the_reference_path():
    """§36's random-weight parity, taken at the level the model runs at.

    A per-family comparison can miss a stream that is wired to the wrong plan,
    because both sides would then read the same wrong thing. This runs the four
    state blocks over a real batch twice — once on the kernel and once with the
    kernel refused, so every message falls back to the gather — and compares
    the trunk's outputs and every parameter gradient.

    Gradients are compared against the *model's* largest gradient rather than
    each tensor's own scale. The latent passes' attention-score biases are
    initialised to zero for uniform weights (§27) and their gradients are a
    near-total cancellation — they land around 1e-6 where the model's largest
    gradient is 1e5 — so their own-relative error is meaningless while their
    absolute error is nothing. Measured worst case: 5.9e-7 relative on the
    outputs and 4.9e-8 of the model scale on the gradients.
    """
    from mantisnet.models.mantis_act.builder import collate_positions
    from mantisnet.models.mantis_act.state_trunk import StateTrunk

    torch.manual_seed(SEED)
    trunk = StateTrunk(FULL).cuda()
    batch = collate_positions(
        [hexo_py.Position.replay(playout(p, SEED)) for p in PLIES], FULL
    ).to("cuda")

    def run():
        trunk.zero_grad(set_to_none=True)
        out = trunk(batch)
        streams = (
            out.cells.inv,
            out.cells.axis,
            out.windows.inv,
            out.windows.axis,
            out.latents.inv,
            out.latents.axis,
        )
        sum(stream.square().sum() for stream in streams).backward()
        return (
            tuple(stream.detach() for stream in streams),
            {
                name: parameter.grad.detach().clone()
                for name, parameter in trunk.named_parameters()
                if parameter.grad is not None
            },
        )

    fused = run()
    supported = kernel._supported
    kernel._supported = lambda *args: False
    try:
        fallen = run()
    finally:
        kernel._supported = supported

    for index, (fell, fast) in enumerate(zip(fallen[0], fused[0])):
        assert relative(fell, fast) < TOL, f"output {index}"
    assert set(fallen[1]) == set(fused[1])
    scale = max(float(grad.abs().max()) for grad in fused[1].values())
    for name, fast in fused[1].items():
        gap = float((fallen[1][name] - fast).abs().max())
        assert gap < 1e-6 * scale, name


@_CUDA
def test_a_host_tensor_is_an_unsupported_signature(families):
    family = families["hex adjacency"]
    host, device = plan_on(family, 1), plan_on(family, 1, "cuda")
    values, _, _ = inputs_for(host, D_INV, gated=True)
    assert not kernel._supported(values, host.dst_ptr)
    assert kernel._supported(values.detach().cuda(), device.dst_ptr)
    assert not kernel._supported(values.detach().cuda().bfloat16(), device.dst_ptr)
