"""Numerics and interface: §27's precision and initialisation rules, §32's tests.

Everything here is checked against the tensors and parameters a model actually
holds, never against what a docstring says it holds. The five detectors, and why
each is independent of the code it watches:

- **Precision where §27 requires it.** A `torch.overrides.TorchFunctionMode`
  records the dtype of every operand of every softmax, every segment scatter,
  and every gather the trunk performs under bf16 autocast, and requires fp32.
  This is a tap on the torch API rather than a reading of the source, so a new
  reduction added anywhere in the package is covered the day it is written, and
  a promotion that a comment claims but the code omits is caught. The negative
  control runs a deliberately bf16 reduction through the same mode and requires
  it to be flagged, so the detector is known to be able to fail.

  The rule covers gathers as well as reductions because `index_select` backward
  *is* a segment sum: it is `index_add_` into a zero tensor of the **source's**
  dtype. A bf16 source therefore makes the gradient of every message an atomic
  bf16 scatter, which CUDA has no instruction for and emulates with a
  compare-and-swap loop — measured at 12.5 ms against fp32's 0.45 ms for the
  ply-161 radius family, and 90% of the whole trunk's backward before the
  promotions moved above their gathers. The precision rule and the performance
  rule are the same rule here.

- **Disabled means absent.** Two general checks stand behind the per-arm ones:
  no parameter has a zero-size dimension (a retained axis tensor of width 0
  would), and **every** parameter of every arm receives a gradient from a real
  forward (a parameter no forward reads would not). Together they catch an
  orphaned subsystem without naming it, which a per-module assertion cannot:
  the per-arm assertions below say what should be gone, and these two say that
  nothing else was left behind.

- **FiLM is bitwise identity at initialisation**, not approximately so, in fp32
  and under autocast, for every phase id and both streams; and perturbing one
  weight breaks it, so the equality is a real constraint rather than a
  comparison of two paths that could both be dead.

- **Initialisation is audited over pooled seeds.** A single draw cannot tell
  `N(0, 0.02)` from `N(0, 0.03)`, so every embedding and learned base is
  measured over sixteen freshly seeded models pooled together, which separates
  0.02 from torch's `N(0, 1)` embedding default and from a Linear's fan-in
  default by hundreds of standard errors.

- **The model summary partitions the parameters.** Its total is required to
  equal `sum(p.numel())`, a shared relation table is required to be counted
  once, and the default trunk's count is measured against §6's 2.5-4M whole-
  model target.

Positions are **real stack-939 self-play positions**, embedded below. As
`docs/MANTIS_ACT_DEVIATIONS.md` records, uniformly random legal play scatters
stones 2.7 cells apart where trained self-play is at 1.03, which gives random
positions five times the legal cells and a fifteenth of the mixed windows of a
real position of the same depth. Every node and edge family scales off that
density, so a numerics test on random playouts exercises accumulations an order
of magnitude shorter than the ones training will run.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import replace

import pytest
import torch
from torch import Tensor, nn
from torch.overrides import TorchFunctionMode

import hexo_py

from mantisnet.models.mantis_act import messages as messages_module
from mantisnet.models.mantis_act import segment_message as segment_message_module
from mantisnet.models.mantis_act.builder import build
from mantisnet.models.mantis_act.config import PRESETS, MantisACTConfig
from mantisnet.models.mantis_act.equivariant import (
    AXIS_CHANNELS,
    AxisMix,
    AxisPool,
    EquivariantState,
    LayerScale,
    PhaseFiLM,
)
from mantisnet.models.mantis_act.latents import ActionLatents, LatentPass
from mantisnet.models.mantis_act.messages import RelationGatedMessage
from mantisnet.models.mantis_act.packed import collate
from mantisnet.models.mantis_act.plans import build_plans_from_cpu_batch
from mantisnet.models.mantis_act.state_trunk import StateTrunk
from mantisnet.models.mantis_act.summary import parameter_summary, subsystem_of

SEED = 20260806

FULL = PRESETS["full_act_v4"]

# §6's target for the *whole* model. The state trunk is one part of it, so the
# ceiling is the constraint this file can check and the headroom is what it
# reports for the stages that are not written yet.
PARAMETER_TARGET = (2_500_000, 4_000_000)

TRUNK_PRESETS = tuple(PRESETS)

# The arms §29 and §32 require to remove a subsystem outright. `full_no_pair` is
# absent from the list because §20 is not implemented at all — see
# `test_full_no_pair_names_nothing_because_partner_modeling_is_gone`.
DISABLED_ARMS = (
    "full_no_axis",
    "full_no_latents",
    "full_one_latent",
    "full_no_action_latents",
    "full_additive_incidence",
)

# Two real stack-939 self-play games, iteration 350, truncated to 161 plies:
# game_id 1436945 (383 plies) and 1436956 (299 plies), the first two games of
# that iteration by game index with a length past 161. Flat (q, r) in play
# order, exactly as `telemetry.unpack_moves` returns them.
REAL_GAMES: tuple[tuple[int, ...], ...] = (
    (
        0,0,-1,-7,-1,0,-1,-5,4,-3,4,-2,-2,-5,-3,-4,0,-8,0,-6,5,-5,11,-13,7,-5,
        10,-12,8,-13,8,-12,-2,-6,-4,0,15,-13,15,-14,-4,2,12,-11,-3,1,13,-11,-2,
        1,13,-14,7,-6,5,-6,10,-9,6,-3,-1,-4,-1,-2,5,-2,0,-1,1,-7,4,-7,12,-13,1,
        -4,-3,-1,-2,-2,1,-6,-3,0,-3,2,-3,3,-3,-2,-4,-2,-2,2,-6,0,-2,-1,9,-10,4,
        -10,10,-8,11,-8,11,-9,10,-7,5,-11,9,-7,9,-8,7,-11,5,-10,13,-8,13,-9,5,
        -8,-3,-7,5,-13,5,-14,-3,-6,-2,-8,14,-9,12,-7,-1,-9,-6,-4,17,-12,18,-13,
        -7,-3,-2,-4,7,-14,14,-8,-2,-3,7,-15,8,-15,9,-16,6,-15,7,-16,7,-17,7,
        -18,7,-13,0,-13,-2,-11,-1,-12,-2,-10,-1,-11,12,-8,0,-11,-4,-4,-4,-3,-1,
        -10,18,-12,9,-11,18,-11,9,-12,19,-13,4,-12,17,-13,4,-13,6,-8,17,-11,7,
        -8,16,-10,6,-6,1,-2,6,-7,4,-4,4,-5,5,-7,7,-9,3,-4,1,-3,0,-3,0,-2,3,-5,
        2,-3,3,-3,-1,-3,11,-11,23,-13,23,-12,23,-14,22,-13,21,-12,21,-10,21,
        -11,22,-12,23,-10,22,-11,20,-9,24,-10,2,2,2,0,2,1,1,1,3,1,3,2,3,0,1,2,
        0,3,1,3,0,4,2,3,6,2,7,1,8,0,7,2,3,5,6,5,5,5,5,3,5,6,6,-4,6,-2,4,7,
    ),
    (
        0,0,3,3,2,-2,-8,0,3,4,-5,0,-7,-1,-7,2,2,-1,-5,-1,-4,-2,-5,-2,-3,-3,-4,
        -1,0,1,-6,-1,2,4,-8,1,1,4,-4,-3,2,5,-3,-4,4,3,-6,-3,2,3,2,2,-5,-3,-2,9,
        -4,1,-1,8,-1,-4,-2,-3,1,2,-1,-3,3,1,-1,-2,4,0,0,-4,-2,-1,-3,-1,-2,-4,2,
        6,-6,-5,-3,2,-6,2,-5,2,-3,1,4,-1,-11,2,3,0,-6,0,3,-1,4,-2,5,-1,4,-3,
        -11,0,-13,2,-10,2,-12,1,-13,0,-15,2,-14,2,-13,1,-11,1,-11,-1,-11,-2,-11,
        4,-17,4,-5,4,-16,3,0,-1,-7,3,-4,3,-6,5,-4,0,-4,4,-3,3,-7,1,-2,2,-10,0,
        -9,3,-12,0,-2,0,-2,3,-3,0,-5,3,-4,2,-6,4,0,-2,-15,3,-3,4,-14,3,-16,4,
        -18,4,-8,4,-12,2,-9,-1,-7,-3,-13,3,-7,-2,-17,6,-18,7,-9,0,-7,7,-4,7,-5,
        7,-1,3,-7,10,-8,10,-9,10,-6,9,-8,8,-8,9,-8,11,-6,6,-7,8,-7,9,-7,6,-7,
        11,-6,8,-9,11,-4,8,-10,12,-2,6,-9,9,-9,8,-2,8,0,8,-10,9,-11,9,5,-3,6,-3,
        -2,-7,-8,6,-17,3,-19,5,-9,6,-3,5,-8,12,-4,6,-6,10,-2,-6,2,-6,1,-5,-19,3,
        -3,-6,-3,-5,-1,-6,-4,-4,-5,-6,-8,5,-3,-8,-8,3,-5,-5,-4,-5,-1,-5,-7,-5,
        -10,3,-7,5,-10,5,-4,5,
    ),
)

# The plies §34 and the deviation register's cost tables use: an opening-ish
# board, the middle game, and two late positions where every edge family is at
# its largest.
PLIES = (21, 61, 121, 161)

# A cheap pair for the per-preset loops, which pay a build and a backward per
# arm.
SHORT_PLIES = (21, 61)


def moves(game: int, plies: int) -> list[tuple[int, int]]:
    flat = REAL_GAMES[game]
    return [(flat[2 * i], flat[2 * i + 1]) for i in range(plies)]


def position(game: int, plies: int):
    return hexo_py.Position.replay(moves(game, plies))


def real_batch(cfg: MantisACTConfig, plies=PLIES, games=(0, 1)):
    """A packed batch of real self-play positions at each of ``plies``."""
    graphs = [build(position(game, ply), cfg) for game in games for ply in plies]
    return collate(graphs, cfg)


@pytest.fixture(scope="module")
def batch():
    return real_batch(FULL)


@pytest.fixture(scope="module")
def small_batch():
    return real_batch(FULL, plies=SHORT_PLIES, games=(0,))


@pytest.fixture(scope="module")
def trunk() -> StateTrunk:
    torch.manual_seed(SEED)
    return StateTrunk(FULL)


def trunk_loss(out) -> Tensor:
    """A scalar every stream of the output reaches, so a backward covers all."""
    terms = [out.cells.inv, out.windows.inv]
    for stream in (out.cells.axis, out.windows.axis, out.latents.inv, out.latents.axis):
        if stream is not None:
            terms.append(stream)
    return sum(term.float().square().mean() for term in terms)


# --------------------------------------------------------------------------
# bf16 autocast, forward and backward (§27, §32)


def test_bf16_autocast_forward_and_backward_are_finite_on_real_positions(batch):
    """§32's smoke test, on the density the model will actually be trained at."""
    torch.manual_seed(SEED)
    module = StateTrunk(FULL)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = module(batch)

    for name, stream in (
        ("cells.inv", out.cells.inv),
        ("cells.axis", out.cells.axis),
        ("windows.inv", out.windows.inv),
        ("windows.axis", out.windows.axis),
        ("latents.inv", out.latents.inv),
        ("latents.axis", out.latents.axis),
    ):
        assert stream is not None, name
        assert torch.isfinite(stream).all(), f"{name} is not finite"
        # §27: the residual streams are the state's precision, not autocast's.
        assert stream.dtype is torch.float32, name

    loss = trunk_loss(out)
    assert torch.isfinite(loss)
    loss.backward()

    for name, parameter in module.named_parameters():
        assert parameter.dtype is torch.float32, f"{name} is not stored in fp32"
        assert parameter.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(parameter.grad).all(), f"{name} has a nonfinite gradient"
        assert parameter.grad.dtype is torch.float32, f"{name}'s gradient is not fp32"

    # A step, because "finite parameters" is a statement about what training
    # leaves behind rather than about one backward.
    optimiser = torch.optim.AdamW(module.parameters(), lr=1e-3)
    optimiser.step()
    for name, parameter in module.named_parameters():
        assert torch.isfinite(parameter).all(), f"{name} is not finite after a step"
        assert parameter.dtype is torch.float32, name


def test_the_forward_is_finite_at_every_ply_separately(batch):
    """A batch hides a position whose own outputs are not finite."""
    torch.manual_seed(SEED)
    module = StateTrunk(FULL).eval()
    for game in (0, 1):
        for ply in PLIES:
            single = collate([build(position(game, ply), FULL)], FULL)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                out = module(single)
            assert torch.isfinite(out.cells.inv).all(), (game, ply)
            assert torch.isfinite(out.cells.axis).all(), (game, ply)
            assert torch.isfinite(out.windows.inv).all(), (game, ply)
            assert torch.isfinite(out.latents.inv).all(), (game, ply)


def test_no_batch_normalisation_anywhere(trunk):
    """§27: no BatchNorm. Its running statistics would also cross positions."""
    assert not any(
        isinstance(module, nn.modules.batchnorm._BatchNorm)
        for module in trunk.modules()
    )


# --------------------------------------------------------------------------
# fp32 where §27 requires it, measured at the operation (§27, §32)

# Every torch call whose precision §27 fixes. The reductions are named because
# they are the segment sums and softmaxes; `index_select` is named because its
# *backward* is one — `index_add_` into a zero tensor of the source's dtype.
_REDUCTIONS = frozenset(
    {
        "softmax",
        "log_softmax",
        "index_add",
        "index_add_",
        "index_reduce_",
        "index_put_",
        "scatter_add_",
        "scatter_reduce_",
    }
)
_GATHERS = frozenset({"index_select", "gather"})
_WATCHED = _REDUCTIONS | _GATHERS

_FP32_OR_BETTER = (torch.float32, torch.float64)


class NumericsWatch(TorchFunctionMode):
    """Records the dtypes of every §27-governed operation in its scope.

    A mode rather than a monkeypatch: it sees the call whatever module made it
    and whatever name that module imported it under, so a reduction added to
    the package later is covered without this file being edited. It runs above
    autocast, so what it records is the dtype the *code* chose, not the one
    autocast would have imposed on the kernel afterwards — which is the thing
    §27 is a rule about.
    """

    def __init__(self) -> None:
        super().__init__()
        self.violations: list[tuple[str, torch.dtype]] = []
        self.seen: dict[str, int] = {}

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = getattr(func, "__name__", "")
        if name in _WATCHED:
            self.seen[name] = self.seen.get(name, 0) + 1
            for value in list(args) + list(kwargs.values()):
                if isinstance(value, Tensor) and value.is_floating_point():
                    if value.dtype not in _FP32_OR_BETTER:
                        self.violations.append((name, value.dtype))
        return func(*args, **(kwargs or {}))


def test_every_softmax_gather_and_segment_reduction_runs_in_fp32(
    batch, monkeypatch
):
    """§27, checked at the operation rather than read off a docstring."""
    torch.manual_seed(SEED)
    module = StateTrunk(FULL)
    watch = NumericsWatch()
    fused_segments: list[tuple[torch.dtype, torch.dtype, torch.dtype]] = []
    original_segment = segment_message_module._reference

    def checked_segment(values, gate, bias, *args, **kwargs):
        fused_segments.append((values.dtype, gate.dtype, bias.dtype))
        return original_segment(values, gate, bias, *args, **kwargs)

    monkeypatch.setattr(segment_message_module, "_reference", checked_segment)
    with watch, torch.autocast("cpu", dtype=torch.bfloat16):
        out = module(batch)
    trunk_loss(out).backward()

    assert not watch.violations, (
        "operations §27 requires in fp32 ran in reduced precision: "
        f"{sorted(set(watch.violations))}"
    )
    # A pass with nothing recorded would be vacuous: the trunk must have run
    # both kinds for the check to mean anything.
    assert fused_segments
    assert all(
        dtype in _FP32_OR_BETTER
        for segment in fused_segments
        for dtype in segment
    ), fused_segments
    # The hand-written segment op is intentionally opaque to TorchFunctionMode.
    # Each checked segment performs both its ordered gathers and its reduction;
    # count that boundary alongside ordinary torch operations so this gate stays
    # non-vacuous under the outer whole-model compile.
    reductions = sum(count for name, count in watch.seen.items() if name in _REDUCTIONS)
    reductions += len(fused_segments)
    gathers = sum(count for name, count in watch.seen.items() if name in _GATHERS)
    gathers += len(fused_segments)
    assert reductions > 0 and gathers > 0, watch.seen


def test_the_fp32_detector_flags_a_reduced_precision_reduction():
    """The negative control: the watch above is known to be able to fail."""
    watch = NumericsWatch()
    index = torch.zeros(8, dtype=torch.long)
    with watch:
        # A segment sum and a softmax, both in bf16, both of the exact shape
        # the message path runs.
        torch.zeros(4, 3, dtype=torch.bfloat16).index_add_(
            0, index, torch.ones(8, 3, dtype=torch.bfloat16)
        )
        torch.ones(4, 3, dtype=torch.bfloat16).softmax(dim=-1)
        torch.ones(4, 3, dtype=torch.bfloat16).index_select(0, index[:2])
    flagged = {name for name, _dtype in watch.violations}
    assert flagged == {"index_add_", "softmax", "index_select"}
    assert all(dtype is torch.bfloat16 for _name, dtype in watch.violations)


def test_a_segment_sum_is_fp32_even_from_a_bf16_source():
    """`messages.segment_sum` promotes; it does not inherit its input's dtype."""
    values = torch.ones(6, 4, dtype=torch.bfloat16)
    index = torch.tensor([0, 0, 1, 1, 1, 2])
    out = messages_module.segment_sum(values, index, 3)
    assert out.dtype is torch.float32
    assert torch.equal(out, torch.tensor([[2.0] * 4, [3.0] * 4, [1.0] * 4]))

    weights = messages_module.segment_softmax(
        torch.zeros(6, dtype=torch.bfloat16), index, 3
    )
    assert weights.dtype is torch.float32


def test_a_bf16_accumulation_of_a_real_segment_loses_what_fp32_keeps(batch):
    """Why §27 names segment sums, measured on a real destination's depth.

    bf16 carries eight mantissa bits, so a running sum stops resolving a term
    once the partial sum is 256 times larger than it — and the busiest cell of
    a real ply-161 position takes messages from over a hundred stones. This
    measures the error of accumulating that many unit-scale terms in each
    precision, so the rule is shown to be load-bearing rather than assumed.
    """
    counts = torch.bincount(batch.radius_dst, minlength=int(batch.cell_offsets[-1]))
    deepest = int(counts.max())
    assert deepest > 100, deepest

    torch.manual_seed(SEED)
    values = torch.randn(deepest, dtype=torch.float64).abs()
    exact = float(values.sum())

    def sequential(dtype: torch.dtype) -> float:
        total = torch.zeros((), dtype=dtype)
        for value in values.to(dtype):
            total = total + value
        return float(total)

    bf16_error = abs(sequential(torch.bfloat16) - exact) / exact
    fp32_error = abs(sequential(torch.float32) - exact) / exact
    assert bf16_error > 1e-3, bf16_error
    assert fp32_error < 1e-5, fp32_error


# --------------------------------------------------------------------------
# Disabled optional modules contribute exactly nothing (§29, §32)


@pytest.mark.parametrize("preset", TRUNK_PRESETS)
def test_every_preset_holds_no_zero_width_parameter(preset):
    """A retained-but-empty tensor is the shape a removed stream leaves behind."""
    module = StateTrunk(PRESETS[preset])
    empty = [
        name for name, parameter in module.named_parameters() if parameter.numel() == 0
    ]
    assert not empty, empty


@pytest.mark.parametrize("preset", TRUNK_PRESETS)
def test_every_parameter_of_every_preset_is_read_by_the_forward(preset):
    """§32's "disabled modules contribute exactly nothing", from the other side.

    A parameter no forward reaches gets no gradient. That is the general
    detector for an orphaned subsystem: the per-arm assertions below say what
    should be absent, and this says nothing else was left behind — including in
    the arms that remove nothing.
    """
    cfg = PRESETS[preset]
    torch.manual_seed(SEED)
    module = StateTrunk(cfg)
    out = module(real_batch(cfg, plies=SHORT_PLIES, games=(0,)))
    trunk_loss(out).backward()
    orphans = [
        name for name, parameter in module.named_parameters() if parameter.grad is None
    ]
    assert not orphans, f"{preset} holds parameters no forward reads: {orphans}"


@pytest.mark.parametrize("preset", DISABLED_ARMS)
def test_each_disabled_arm_constructs_and_costs_no_more_than_the_full_model(
    preset, trunk
):
    module = StateTrunk(PRESETS[preset])
    assert len(module.blocks) == PRESETS[preset].state_blocks
    full_total = parameter_summary(trunk).total
    assert parameter_summary(module).total <= full_total
    # The parameter names of an arm that removes a path are a subset of the
    # full model's: an ablation does not introduce parameters of its own.
    full_names = {name for name, _ in trunk.named_parameters()}
    arm_names = {name for name, _ in module.named_parameters()}
    assert arm_names <= full_names, sorted(arm_names - full_names)


def test_full_no_axis_retains_no_axis_parameter_anywhere():
    """§29: "do not retain unused axis parameters", checked structurally.

    A name-based check would miss the axis-path parameters that are not called
    "axis" — `WindowEmbedding.to_native`, `LatentPass.pool_to_inv`,
    `LatentPass.axis_to_inv`, an `AxisPool`'s three projections. This walks the
    module tree for the axis-carrying halves themselves.
    """
    cfg = PRESETS["full_no_axis"]
    assert cfg.d_axis == 0 and not cfg.use_axis_channels
    module = StateTrunk(cfg)

    assert not any(isinstance(m, AxisPool) for m in module.modules())
    for name, child in module.named_modules():
        if isinstance(child, AxisMix):
            assert not list(child.parameters()), name
        if isinstance(child, RelationGatedMessage):
            assert child.route_axis is False, name
            for attribute in ("wv_axis", "wb_axis", "wg_axis", "ln_src_axis"):
                assert not hasattr(child, attribute), f"{name}.{attribute}"
        if isinstance(child, PhaseFiLM):
            assert child.to_axis is None, name
        if isinstance(child, LatentPass):
            assert child.has_axis is False and child.pools_node_axis is False, name

    assert module.cell_embedding.axis_base is None
    assert module.window_embedding.axis_base is None
    assert module.window_embedding.to_native is None
    assert module.final_cell.axis is None and module.final_window.axis is None
    assert module.final_latent_axis is None
    assert not hasattr(module.latents, "base_axis")

    # And no parameter of the arm carries the full model's axis width in a
    # position only an axis parameter would: the full model's axis stream is
    # gone, so its per-channel tensors are gone with it.
    out = module(real_batch(cfg, plies=SHORT_PLIES, games=(0,)))
    assert out.cells.axis is None and out.windows.axis is None
    assert out.latents.axis is None


def test_full_no_latents_and_full_one_latent_hold_only_what_they_keep():
    no_latents = StateTrunk(PRESETS["full_no_latents"])
    assert not list(no_latents.latents.parameters())
    assert not any(p.enabled for p in no_latents.latents.passes)
    assert no_latents.final_latent_inv is None
    assert no_latents.final_latent_axis is None

    one = StateTrunk(PRESETS["full_one_latent"])
    assert one.latents.base_inv.shape[0] == 1
    assert not hasattr(one.latents, "base_axis")
    assert one.final_latent_axis is None
    assert one.final_latent_inv is not None
    for child in one.modules():
        if isinstance(child, LatentPass):
            assert child.has_axis is False
            for attribute in ("q_read_axis", "k_bcast_axis", "axis_mix", "type_read_axis"):
                assert not hasattr(child, attribute), attribute
            # The *nodes'* axis states are still pooled into the invariant key
            # (§17.2): that needs axis channels, not axis latents.
            assert child.pools_node_axis is True


def test_full_no_action_latents_removes_the_action_stack_and_nothing_else(trunk):
    """§21's stack is a separate family: turning it off must not touch the trunk."""
    cfg = PRESETS["full_no_action_latents"]
    assert cfg.num_action_latents == 0 and cfg.use_action_set_latents is False

    stack = ActionLatents(cfg)
    assert stack.enabled is False
    assert not list(stack.parameters())
    assert all(not p.enabled for p in stack.passes)

    full_stack = ActionLatents(FULL)
    assert full_stack.enabled is True
    assert list(full_stack.parameters())

    # The state trunk is the same model in both arms, parameter for parameter.
    torch.manual_seed(SEED)
    without = StateTrunk(cfg)
    assert parameter_summary(without).total == parameter_summary(trunk).total


def test_full_additive_incidence_removes_exactly_the_gate_projections(trunk):
    """§29's additive control: `U h + E_r`, so every gate projection is gone."""
    cfg = PRESETS["full_additive_incidence"]
    module = StateTrunk(cfg)

    gates = 0
    for name, child in module.named_modules():
        if isinstance(child, RelationGatedMessage):
            assert child.gated is False, name
            assert not hasattr(child, "wg_inv"), name
            assert not hasattr(child, "wg_axis"), name
    for name, child in trunk.named_modules():
        if isinstance(child, RelationGatedMessage):
            assert child.gated is True, name
            gates += child.wg_inv.weight.numel() + child.wg_inv.bias.numel()
            if child.route_axis:
                gates += child.wg_axis.weight.numel() + child.wg_axis.bias.numel()
    assert gates > 0

    # Exactly the gates, and nothing else: the difference is accounted for to
    # the scalar, so a removal that also dropped something else would fail.
    assert parameter_summary(trunk).total - parameter_summary(module).total == gates


def test_full_no_pair_names_nothing_because_partner_modeling_is_gone():
    """§29's `full_no_pair` is absent, and the register is what says why.

    §20 is dropped whole (`docs/MANTIS_ACT_DEVIATIONS.md`): there is no pair
    module, no pair tensor, and no `pair_scope` field, so `full_no_pair` would
    name the same architecture as `full_act_v4`. The check is that the absence
    is complete rather than partial — a surviving `pair_*` config field would
    be a configuration nothing reads.
    """
    assert "full_no_pair" not in PRESETS
    fields = {f for f in MantisACTConfig.__dataclass_fields__}
    assert not [name for name in fields if "pair" in name], sorted(fields)
    assert "use_action_pair_messages" not in fields


# --------------------------------------------------------------------------
# FiLM is the identity at initialisation (§13.2, §27)


def film_state(cfg: MantisACTConfig, rows: int = 7) -> EquivariantState:
    torch.manual_seed(SEED)
    inv = torch.randn(rows, cfg.d_inv)
    axis = torch.randn(rows, AXIS_CHANNELS, cfg.d_axis) if cfg.d_axis else None
    return EquivariantState(inv, axis)


@pytest.mark.parametrize("phase", (0, 1, 2))
def test_film_is_bitwise_identity_at_initialisation(phase):
    """§27: "FiLM initialized to identity" -- exactly, not to a tolerance."""
    torch.manual_seed(SEED)
    film = PhaseFiLM(FULL)
    state = film_state(FULL)
    phase_id = torch.full((state.leading_shape[0],), phase, dtype=torch.long)

    out = film(state, phase_id)
    assert torch.equal(out.inv, state.inv)
    assert torch.equal(out.axis, state.axis)

    # And under autocast, where the modulation itself is computed in bf16: a
    # zero projection is zero in every dtype, so the identity survives.
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = film(state, phase_id)
    assert torch.equal(out.inv, state.inv)
    assert torch.equal(out.axis, state.axis)


def test_a_perturbed_film_is_no_longer_the_identity():
    """The negative control for the equality above."""
    torch.manual_seed(SEED)
    film = PhaseFiLM(FULL)
    state = film_state(FULL)
    phase_id = torch.zeros(state.leading_shape[0], dtype=torch.long)
    with torch.no_grad():
        film.to_inv.bias[0] = 1e-3
    out = film(state, phase_id)
    assert not torch.equal(out.inv, state.inv)


def test_every_film_in_the_trunk_starts_at_identity(trunk, small_batch):
    """Not one module in isolation: every FiLM the trunk actually runs."""
    films = [m for m in trunk.modules() if isinstance(m, PhaseFiLM)]
    assert len(films) == 2 * FULL.state_blocks
    for film in films:
        assert torch.equal(film.to_inv.weight, torch.zeros_like(film.to_inv.weight))
        assert torch.equal(film.to_inv.bias, torch.zeros_like(film.to_inv.bias))
        assert torch.equal(film.to_axis.weight, torch.zeros_like(film.to_axis.weight))
        assert torch.equal(film.to_axis.bias, torch.zeros_like(film.to_axis.bias))

    # A fresh trunk's phase conditioning is a no-op end to end: relabelling
    # every position's phase cannot move the output.
    torch.manual_seed(SEED)
    fresh = StateTrunk(FULL).eval()
    with torch.no_grad():
        before = fresh(small_batch)
        rotated = replace(small_batch, phase_id=(small_batch.phase_id + 1) % 3)
        rotated = replace(
            rotated, plans=build_plans_from_cpu_batch(FULL, rotated)
        )
        after = fresh(rotated)
    assert torch.equal(before.cells.inv, after.cells.inv)
    assert torch.equal(before.windows.axis, after.windows.axis)


# --------------------------------------------------------------------------
# Initialisation (§27)

# §27's `N(0, 0.02)` for embeddings, relation tables, and latent bases, and the
# window the pooled estimate must land in. Sixteen seeds pooled put the standard
# error of the estimate near 1% for the smallest table here, so this window is
# many standard errors wide while still excluding torch's `N(0, 1)` embedding
# default and a Linear's fan-in default by orders of magnitude.
EMBEDDING_STD = 0.02
STD_WINDOW = (0.017, 0.023)
INIT_SEEDS = 16

# The learned bases §27 names beside the embedding tables. Every one is a
# `nn.Parameter` rather than an `nn.Embedding`, so they are listed by the
# attribute name their owner gives them.
BASE_PARAMETERS = (
    "axis_base",
    "base_inv",
    "base_axis",
    "type_read_inv",
    "type_read_axis",
    "type_bcast_src",
)


def pooled_initialisations(cfg: MantisACTConfig) -> dict[str, list[Tensor]]:
    """Every embedding and learned base of ``cfg``, over ``INIT_SEEDS`` models."""
    pooled: dict[str, list[Tensor]] = {}
    for seed in range(INIT_SEEDS):
        torch.manual_seed(SEED + seed)
        module = StateTrunk(cfg)
        for name, child in module.named_modules():
            if isinstance(child, nn.Embedding):
                pooled.setdefault(f"{name}.weight", []).append(child.weight.detach())
        for name, parameter in module.named_parameters():
            if name.split(".")[-1] in BASE_PARAMETERS:
                pooled.setdefault(name, []).append(parameter.detach())
    return pooled


def test_every_embedding_and_learned_base_is_normal_0_02():
    """§27, pooled over seeds so the estimate can actually reject 0.02."""
    pooled = pooled_initialisations(FULL)
    assert pooled, "no embedding or learned base was found to check"
    # Every table §27 names must be present, or the audit is checking a subset.
    assert any("relation" in name for name in pooled)
    assert any(name.endswith("base_inv") for name in pooled)
    assert any(name.endswith("base_axis") for name in pooled)
    assert any("pattern" in name for name in pooled)

    for name, draws in sorted(pooled.items()):
        flat = torch.cat([draw.reshape(-1) for draw in draws])
        std = float(flat.std())
        mean = float(flat.mean())
        assert STD_WINDOW[0] < std < STD_WINDOW[1], f"{name}: std {std:.4f}"
        # The mean is zero to within four standard errors of the pooled sample.
        assert abs(mean) < 4 * EMBEDDING_STD / math.sqrt(flat.numel()), (
            f"{name}: mean {mean:.5f}"
        )


def test_a_default_initialised_table_fails_the_same_window():
    """The negative control: torch's own embedding default is `N(0, 1)`."""
    default = nn.Embedding(378, 64)
    assert not STD_WINDOW[0] < float(default.weight.detach().std()) < STD_WINDOW[1]


@pytest.mark.parametrize("init", (1e-2, 0.5))
def test_every_layer_scale_starts_at_the_configured_value(init):
    """§27: "default LayerScale initialized 1e-2 for fresh training"."""
    cfg = replace(FULL, layer_scale_init=init)
    torch.manual_seed(SEED)
    module = StateTrunk(cfg)
    scales = [m for m in module.modules() if isinstance(m, LayerScale)]
    assert scales
    for scale in scales:
        assert torch.equal(scale.gamma, torch.full_like(scale.gamma, init))


def test_the_default_layer_scale_is_the_specs_value():
    assert MantisACTConfig().layer_scale_init == 1e-2


def test_relation_attention_biases_start_at_zero():
    """§27: "relation attention biases initialized zero"."""
    cfg = replace(FULL, incidence_reduce="attention")
    torch.manual_seed(SEED)
    module = StateTrunk(cfg)
    scores = [
        (name, parameter)
        for name, parameter in module.named_parameters()
        if name.rsplit(".", 1)[-1].startswith("score_")
    ]
    assert scores
    for name, parameter in scores:
        assert torch.equal(parameter, torch.zeros_like(parameter)), name


def test_the_axis_bases_are_learned_once_and_replicated(trunk, small_batch):
    """§27: "Axis bases are learned once and replicated over the channels"."""
    assert trunk.cell_embedding.axis_base.shape == (FULL.d_axis,)
    assert trunk.window_embedding.axis_base.shape == (FULL.d_axis,)
    assert trunk.latents.base_axis.shape == (FULL.num_axis_latents, FULL.d_axis)
    # And the replication is real: a cell's three channels start identical.
    cells = trunk.cell_embedding(small_batch)
    assert torch.equal(cells.axis[:, 0], cells.axis[:, 1])
    assert torch.equal(cells.axis[:, 0], cells.axis[:, 2])


# --------------------------------------------------------------------------
# The model summary (§6, §32, §34)


def test_the_summary_total_equals_sum_of_numel(trunk):
    """§32: "model summary parameter totals match `sum(p.numel())`"."""
    summary = parameter_summary(trunk)
    assert summary.total == sum(p.numel() for p in trunk.parameters())
    assert summary.trainable_tensors == len(list(trunk.parameters()))
    assert summary.total == sum(count for _name, count in summary.groups)


def test_the_summary_is_a_partition_of_the_named_parameters(trunk):
    """Every parameter lands in exactly one subsystem, and none is dropped."""
    counted: Counter[str] = Counter()
    for name, parameter in trunk.named_parameters():
        counted[subsystem_of(name)] += parameter.numel()
    trunk_groups = dict(parameter_summary(trunk).groups)
    assert dict(counted) == trunk_groups
    assert len(trunk_groups) > 4, trunk_groups
    # The subsystems are the trunk's own parts, named from the module tree.
    assert "cell_embedding" in trunk_groups
    assert "window_embedding" in trunk_groups
    assert any(name.startswith("blocks.") for name in trunk_groups)
    assert any(name.startswith("latents") for name in trunk_groups)


def test_the_summary_counts_a_shared_relation_table_once(trunk):
    """§14's sharing is a parameter saving, and the summary must show it as one."""
    private = StateTrunk(replace(FULL, share_relation_embeddings_across_blocks=False))
    shared_summary = parameter_summary(trunk)
    private_summary = parameter_summary(private)

    tables = sum(
        table.weight.numel()
        for table in (
            trunk.relations.incidence,
            trunk.relations.adjacency,
            trunk.relations.radius,
        )
    )
    # Sharing costs one copy; the private arm pays one per block.
    assert private_summary.total - shared_summary.total == tables * (
        FULL.state_blocks - 1
    )
    assert shared_summary.groups[0][1] <= shared_summary.total


def test_the_summary_renders_as_a_table(trunk):
    text = parameter_summary(trunk).text()
    assert "total" in text
    assert f"{parameter_summary(trunk).total:,}" in text
    for name, _count in parameter_summary(trunk).groups:
        assert name in text


def test_the_default_trunk_leaves_room_for_the_stages_that_follow(trunk):
    """§6 targets 2.5-4M for the *whole* model; the trunk is one part of it.

    The ceiling is what this file can assert. The number itself is the useful
    output: the action encoder, the action blocks, and the private policy and
    critic adapters have to fit in what is left.
    """
    total = parameter_summary(trunk).total
    assert total < PARAMETER_TARGET[1], (
        f"the state trunk alone is {total:,} parameters, past §6's whole-model "
        f"ceiling of {PARAMETER_TARGET[1]:,}"
    )
    headroom = PARAMETER_TARGET[1] - total
    assert headroom > 1_000_000, (
        f"{total:,} parameters in the trunk leaves only {headroom:,} for the "
        "action encoder, the action blocks, and both private adapters"
    )
