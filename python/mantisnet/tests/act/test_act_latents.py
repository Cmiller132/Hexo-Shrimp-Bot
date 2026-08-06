"""The global state and action-set latents: §17, §21, and the §12.1 law.

Five things are being detected here, and each has its own independent oracle
rather than a round trip against the implementation:

- the ragged packed attention agrees with a per-position dense loop written
  from the other direction (an explicit ``softmax`` over a slice, not a
  segment max/exp/sum), so a mispacked offset or a segment reduction that
  leaks across a position boundary shows up;
- a batch of P positions equals P single-position forwards, which is the same
  detector applied to the whole module;
- permuting the three axis channels of every node permutes the output's axis
  channels and leaves every invariant quantity alone. This is §12.1 stated as
  a test, and it is the one that catches a per-axis parameter: a module that
  learned a different weight, bias, norm, or base for channel 0 fails it and
  nothing else here would;
- permuting the *order* of nodes within a position changes nothing, which is
  what a set-valued read has to satisfy;
- §32's requirement that a disabled optional module holds no parameters and
  contributes nothing.

Plus §3.14 and §26's requirement that the global path is linear in the node
count — measured by counting the elements every torch operation produces,
which is deterministic where a wall-clock ratio is not.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import replace

import pytest
import torch

from mantisnet.models.mantis_act.config import PRESETS, MantisACTConfig
from mantisnet.models.mantis_act.equivariant import (
    AXIS_CHANNELS,
    EquivariantState,
    permute_axis_channels,
)
from mantisnet.models.mantis_act.latents import (
    ActionLatents,
    LatentPass,
    LatentState,
    RaggedStream,
    StateLatents,
    row_positions,
    segment_cross_attention,
)

# Deterministic, and small enough that a shape error stays readable.
SEED = 20260806

# The four nonidentity axis permutations a D6 element can induce, plus the two
# that fix a channel: every element of S3 except the identity.
PERMUTATIONS = [(1, 2, 0), (2, 0, 1), (0, 2, 1), (2, 1, 0), (1, 0, 2)]


def _offsets(counts: list[int]) -> torch.Tensor:
    return torch.tensor(list(itertools.accumulate(counts, initial=0)), dtype=torch.long)


def _stream(cfg: MantisACTConfig, counts: list[int]) -> RaggedStream:
    """A random node family with ``counts[p]`` rows in position ``p``."""
    offsets = _offsets(counts)
    total = int(offsets[-1])
    axis = None if cfg.d_axis == 0 else torch.randn(total, AXIS_CHANNELS, cfg.d_axis)
    return RaggedStream(EquivariantState(torch.randn(total, cfg.d_inv), axis), offsets)


def _state_module(cfg: MantisACTConfig) -> StateLatents:
    torch.manual_seed(SEED)
    module = StateLatents(cfg)
    module.eval()
    return module


def _slice(stream: RaggedStream, position: int) -> RaggedStream:
    lo, hi = int(stream.offsets[position]), int(stream.offsets[position + 1])
    axis = None if stream.state.axis is None else stream.state.axis[lo:hi]
    return RaggedStream(
        EquivariantState(stream.state.inv[lo:hi], axis),
        torch.tensor([0, hi - lo], dtype=torch.long),
    )


def _rows(stream: RaggedStream, index: torch.Tensor) -> RaggedStream:
    """The same family with its rows reordered inside their positions."""
    axis = None if stream.state.axis is None else stream.state.axis[index]
    return RaggedStream(
        EquivariantState(stream.state.inv[index], axis), stream.offsets
    )


def _turn(stream: RaggedStream, permutation) -> RaggedStream:
    return RaggedStream(stream.state.permute_axes(permutation), stream.offsets)


def _reference_cross_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    """Per-position dense attention, written independently of the packed form.

    One explicit slice and one explicit ``softmax`` per position, which is the
    slow reference §17.5 permits for exactly this comparison.
    """
    head_dim = q.shape[-1]
    out = []
    for position in range(offsets.shape[0] - 1):
        lo, hi = int(offsets[position]), int(offsets[position + 1])
        if hi == lo:
            out.append(torch.zeros_like(q[position], dtype=torch.float32))
            continue
        score = torch.einsum(
            "kchd,nchd->knch", q[position].float(), k[lo:hi].float()
        ) / math.sqrt(head_dim)
        out.append(torch.einsum("knch,nchd->kchd", score.softmax(dim=1), v[lo:hi].float()))
    return torch.stack(out)


class _WorkCounter(torch.overrides.TorchFunctionMode):
    """Total elements produced by every torch operation inside the block.

    A deterministic stand-in for cost: doubling the node count doubles it when
    the work is linear and quadruples it when it is not. Wall-clock timing
    would answer the same question and would flake.
    """

    def __init__(self) -> None:
        super().__init__()
        self.elements = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        for item in result if isinstance(result, (tuple, list)) else (result,):
            if isinstance(item, torch.Tensor):
                self.elements += item.numel()
        return result


# --- the ragged primitive (§17.5) -----------------------------------------


def test_segment_attention_matches_the_per_position_reference() -> None:
    torch.manual_seed(SEED)
    offsets = _offsets([5, 1, 9, 4])
    positions, slots, channels, heads, head_dim = 4, 3, AXIS_CHANNELS, 2, 6
    total = int(offsets[-1])
    q = torch.randn(positions, slots, channels, heads, head_dim)
    k = torch.randn(total, channels, heads, head_dim)
    v = torch.randn(total, channels, heads, head_dim)

    packed = segment_cross_attention(q, k, v, row_positions(offsets), positions)
    torch.testing.assert_close(
        packed, _reference_cross_attention(q, k, v, offsets), rtol=1e-5, atol=1e-6
    )


def test_segment_attention_ignores_row_order_within_a_position() -> None:
    """Rows may arrive in any order: several families share one softmax."""
    torch.manual_seed(SEED)
    offsets = _offsets([6, 6])
    q = torch.randn(2, 2, 1, 2, 4)
    k = torch.randn(12, 1, 2, 4)
    v = torch.randn(12, 1, 2, 4)
    row_pos = row_positions(offsets)

    order = torch.tensor([3, 0, 5, 1, 4, 2, 9, 11, 6, 10, 7, 8])
    torch.testing.assert_close(
        segment_cross_attention(q, k[order], v[order], row_pos[order], 2),
        segment_cross_attention(q, k, v, row_pos, 2),
        rtol=1e-5,
        atol=1e-6,
    )


def test_segment_attention_reads_zero_from_an_empty_position() -> None:
    """``full_occupied_cells_only`` on an empty board has no rows at all."""
    torch.manual_seed(SEED)
    offsets = _offsets([0, 3])
    q = torch.randn(2, 2, 1, 2, 4)
    k = torch.randn(3, 1, 2, 4)
    out = segment_cross_attention(q, k, k, row_positions(offsets), 2)
    assert torch.isfinite(out).all()
    assert (out[0] == 0).all()
    assert (out[1] != 0).any()


def test_segment_attention_refuses_mismatched_shapes() -> None:
    q = torch.randn(2, 2, 1, 2, 4)
    k = torch.randn(5, 1, 2, 4)
    with pytest.raises(ValueError, match="row_pos must be"):
        segment_cross_attention(q, k, k, torch.zeros(4, dtype=torch.long), 2)
    wide = torch.randn(5, 3, 2, 4)
    with pytest.raises(ValueError, match="disagree"):
        segment_cross_attention(q, wide, wide, torch.zeros(5, dtype=torch.long), 2)
    with pytest.raises(ValueError, match="position_count"):
        segment_cross_attention(q, k, k, torch.zeros(5, dtype=torch.long), 3)


# --- the module (§17.2-§17.4, §21) ----------------------------------------


def test_a_pass_moves_every_stream() -> None:
    """A sanity floor: the default pass is not accidentally the identity."""
    cfg = MantisACTConfig()
    module = _state_module(cfg)
    torch.manual_seed(SEED + 1)
    entities = {"cell": _stream(cfg, [7, 4]), "window": _stream(cfg, [3, 5])}
    latents = module.initial(torch.randn(2, 8))

    moved, updated = module[0](latents, entities)
    assert not torch.allclose(moved.inv, latents.inv)
    assert not torch.allclose(moved.axis, latents.axis)
    for name, stream in entities.items():
        assert not torch.allclose(updated[name].state.inv, stream.state.inv)
        assert not torch.allclose(updated[name].state.axis, stream.state.axis)


def test_batch_equals_single_position_forwards() -> None:
    cfg = MantisACTConfig()
    module = _state_module(cfg)
    torch.manual_seed(SEED + 2)
    entities = {"cell": _stream(cfg, [7, 2, 11]), "window": _stream(cfg, [4, 6, 1])}
    global_numeric = torch.randn(3, 8)

    batched, updated = module[0](module.initial(global_numeric), entities)

    for position in range(3):
        one = {name: _slice(s, position) for name, s in entities.items()}
        alone, alone_updated = module[0](
            module.initial(global_numeric[position : position + 1]), one
        )
        torch.testing.assert_close(
            alone.inv[0], batched.inv[position], rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            alone.axis[0], batched.axis[position], rtol=1e-5, atol=1e-6
        )
        for name in entities:
            lo = int(entities[name].offsets[position])
            hi = int(entities[name].offsets[position + 1])
            torch.testing.assert_close(
                alone_updated[name].state.inv,
                updated[name].state.inv[lo:hi],
                rtol=1e-5,
                atol=1e-6,
            )
            torch.testing.assert_close(
                alone_updated[name].state.axis,
                updated[name].state.axis[lo:hi],
                rtol=1e-5,
                atol=1e-6,
            )


def test_latents_do_not_depend_on_node_order() -> None:
    """The read is over a set: reordering a position's cells changes nothing."""
    cfg = MantisACTConfig()
    module = _state_module(cfg)
    torch.manual_seed(SEED + 3)
    entities = {"cell": _stream(cfg, [8, 5]), "window": _stream(cfg, [3, 4])}
    latents = module.initial(torch.randn(2, 8))
    plain, updated = module[0](latents, entities)

    # A permutation confined to each position, so nothing crosses one.
    order = torch.cat(
        [torch.tensor([5, 0, 7, 2, 1, 6, 3, 4]), 8 + torch.tensor([3, 1, 4, 0, 2])]
    )
    shuffled = dict(entities)
    shuffled["cell"] = _rows(entities["cell"], order)

    moved, moved_updated = module[0](latents, shuffled)
    torch.testing.assert_close(moved.inv, plain.inv, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(moved.axis, plain.axis, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        moved_updated["cell"].state.inv,
        updated["cell"].state.inv[order],
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        moved_updated["window"].state.inv,
        updated["window"].state.inv,
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize("permutation", PERMUTATIONS)
def test_axis_channels_permute_with_the_board(permutation) -> None:
    """§12.1 at the module boundary, and the detector for per-axis parameters.

    A board symmetry permutes the three undirected axes, so every node's and
    every latent's axis channels permute together. A module holding a
    different weight, bias, norm, or base per absolute axis produces a
    different answer under that relabelling; one sharing its parameters over
    the channels produces the relabelled same answer.
    """
    cfg = MantisACTConfig()
    module = _state_module(cfg)
    torch.manual_seed(SEED + 4)
    entities = {"cell": _stream(cfg, [6, 3]), "window": _stream(cfg, [2, 5])}
    latents = module.initial(torch.randn(2, 8))
    plain, updated = module[0](latents, entities)

    turned = {name: _turn(s, permutation) for name, s in entities.items()}
    # The axis latents' bases are replicated across channels, so permuting the
    # incoming latents is the identity; permuting them explicitly states the
    # claim instead of relying on that.
    turned_latents = LatentState(
        inv=latents.inv, axis=permute_axis_channels(latents.axis, permutation)
    )

    moved, moved_updated = module[0](turned_latents, turned)
    torch.testing.assert_close(moved.inv, plain.inv, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        moved.axis, permute_axis_channels(plain.axis, permutation), rtol=1e-5, atol=1e-6
    )
    for name in entities:
        torch.testing.assert_close(
            moved_updated[name].state.inv, updated[name].state.inv, rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            moved_updated[name].state.axis,
            permute_axis_channels(updated[name].state.axis, permutation),
            rtol=1e-5,
            atol=1e-6,
        )


def test_axis_bases_are_one_base_replicated_across_channels() -> None:
    """§17.1 and §27: a per-channel base would be a per-absolute-axis parameter."""
    cfg = MantisACTConfig()
    module = _state_module(cfg)
    latents = module.initial(torch.randn(3, 8))
    assert latents.axis.shape == (3, cfg.num_axis_latents, AXIS_CHANNELS, cfg.d_axis)
    for channel in range(1, AXIS_CHANNELS):
        torch.testing.assert_close(latents.axis[:, :, 0], latents.axis[:, :, channel])
    # The invariant latents, by contrast, keep distinct identities.
    assert not torch.allclose(latents.inv[:, 0], latents.inv[:, 1])


def test_global_scalars_seed_only_the_invariant_latents() -> None:
    """§13.3 initialises invariant state latents; the axis bases are untouched."""
    cfg = MantisACTConfig()
    module = _state_module(cfg)
    first = module.initial(torch.zeros(2, 8))
    second = module.initial(torch.randn(2, 8))
    assert not torch.allclose(first.inv, second.inv)
    torch.testing.assert_close(first.axis, second.axis)

    with pytest.raises(ValueError, match="columns"):
        module.initial(torch.zeros(2, 7))


def test_action_latents_are_invariant_only_and_separate() -> None:
    """§21: two invariant queries over the legal action set, with own bases."""
    cfg = MantisACTConfig()
    torch.manual_seed(SEED)
    module = ActionLatents(cfg)
    module.eval()
    assert len(module) == cfg.action_blocks

    latents = module.initial(2)
    assert latents.axis is None
    assert latents.inv.shape == (2, cfg.num_action_latents, cfg.d_inv)

    torch.manual_seed(SEED + 5)
    actions = {"action": _stream(cfg, [5, 3])}
    moved, updated = module[0](latents, actions)
    assert not torch.allclose(moved.inv, latents.inv)
    assert not torch.allclose(updated["action"].state.inv, actions["action"].state.inv)
    # No action axis latents, so the action axis channels are not written here;
    # §22.4's AxisMix is what moves direction into them.
    torch.testing.assert_close(
        updated["action"].state.axis, actions["action"].state.axis
    )

    with pytest.raises(ValueError, match="do not match"):
        module[0](latents, {"cell": actions["action"]})


def test_axis_latents_without_invariant_latents_are_refused() -> None:
    """An axis latent has no invariant half for §17.3's AxisMix to pair with."""
    cfg = replace(MantisACTConfig(), num_inv_latents=0)
    with pytest.raises(ValueError, match="no invariant half"):
        StateLatents(cfg)
    # And with no blocks at all, where no pass would run the check.
    with pytest.raises(ValueError, match="no invariant half"):
        StateLatents(replace(cfg, state_blocks=0))


def test_latents_with_no_blocks_to_read_them_are_refused() -> None:
    """Bases no pass consumes are parameters no forward touches."""
    with pytest.raises(ValueError, match="no pass would read the bases"):
        StateLatents(replace(MantisACTConfig(), state_blocks=0))
    with pytest.raises(ValueError, match="no pass would read the bases"):
        ActionLatents(replace(MantisACTConfig(), action_blocks=0))
    # Zero blocks is fine when there are no latents to strand.
    assert list(StateLatents(replace(PRESETS["full_no_latents"], state_blocks=0)).parameters()) == []


# --- disabled configurations (§32) ----------------------------------------


def test_no_latents_at_all_holds_no_parameters_and_changes_nothing() -> None:
    cfg = PRESETS["full_no_latents"]
    module = _state_module(cfg)
    assert list(module.parameters()) == []
    torch.manual_seed(SEED + 6)
    entities = {"cell": _stream(cfg, [6, 4]), "window": _stream(cfg, [3, 2])}
    latents = module.initial(torch.randn(2, 8))
    assert latents.inv is None and latents.axis is None

    moved, updated = module[0](latents, entities)
    assert moved is latents
    for name, stream in entities.items():
        assert updated[name] is stream


def test_one_latent_keeps_no_axis_latent_parameters() -> None:
    """``full_one_latent``: one invariant latent, no axis latents anywhere.

    The invariant read's symmetric pool over the *nodes'* axis states survives,
    because §17.2 makes that pool part of the invariant key rather than part of
    the axis latent stream.
    """
    cfg = PRESETS["full_one_latent"]
    module = _state_module(cfg)
    torch.manual_seed(SEED + 7)
    entities = {"cell": _stream(cfg, [6, 4]), "window": _stream(cfg, [3, 2])}
    latents = module.initial(torch.randn(2, 8))
    assert latents.axis is None
    assert latents.inv.shape == (2, 1, cfg.d_inv)

    one = module[0]
    assert not one.has_axis and one.pools_node_axis
    for attribute in (
        "type_read_axis",
        "q_read_axis",
        "o_read_axis",
        "scale_read_axis",
        "q_mix_axis",
        "scale_mix_axis",
        "axis_mix",
        "pool_latent_axis",
        "axis_to_inv",
        "k_bcast_axis",
        "q_bcast_axis",
        "scale_bcast_axis",
    ):
        assert not hasattr(one, attribute), attribute

    moved, updated = one(latents, entities)
    assert moved.axis is None
    assert not torch.allclose(moved.inv, latents.inv)
    for name, stream in entities.items():
        # The invariant broadcast still lands; the axis channels are untouched.
        assert not torch.allclose(updated[name].state.inv, stream.state.inv)
        torch.testing.assert_close(updated[name].state.axis, stream.state.axis)


def test_action_latents_disabled_hold_no_parameters() -> None:
    cfg = PRESETS["full_no_action_latents"]
    torch.manual_seed(SEED)
    module = ActionLatents(cfg)
    assert list(module.parameters()) == []
    latents = module.initial(2)
    torch.manual_seed(SEED + 8)
    actions = {"action": _stream(cfg, [4, 2])}
    moved, updated = module[0](latents, actions)
    assert moved is latents
    assert updated["action"] is actions["action"]


def test_no_axis_channels_at_all() -> None:
    """``full_no_axis``: the streams carry no axis half and nothing reads one."""
    cfg = PRESETS["full_no_axis"]
    module = _state_module(cfg)
    torch.manual_seed(SEED + 9)
    entities = {"cell": _stream(cfg, [5, 3]), "window": _stream(cfg, [2, 4])}
    latents = module.initial(torch.randn(2, 8))
    assert latents.axis is None

    moved, updated = module[0](latents, entities)
    assert moved.axis is None
    for name in entities:
        assert updated[name].state.axis is None
        assert not torch.allclose(
            updated[name].state.inv, entities[name].state.inv
        )

    # A stream that disagrees with the configuration is refused, not ignored.
    wrong = dict(entities)
    wrong["cell"] = RaggedStream(
        EquivariantState(
            entities["cell"].state.inv,
            torch.randn(entities["cell"].rows, AXIS_CHANNELS, 4),
        ),
        entities["cell"].offsets,
    )
    with pytest.raises(ValueError, match="axis width"):
        module[0](latents, wrong)


def test_ragged_offsets_that_disagree_with_the_rows_are_refused() -> None:
    cfg = MantisACTConfig()
    module = _state_module(cfg)
    torch.manual_seed(SEED + 10)
    entities = {"cell": _stream(cfg, [5, 3]), "window": _stream(cfg, [2, 4])}
    latents = module.initial(torch.randn(2, 8))
    broken = dict(entities)
    broken["cell"] = RaggedStream(entities["cell"].state, _offsets([5, 2]))
    with pytest.raises(ValueError, match="offsets end at"):
        module[0](latents, broken)


# --- cost (§3.14, §26) -----------------------------------------------------


def test_cost_is_linear_in_the_node_count() -> None:
    """The whole argument for latents: work grows with N, not with N squared.

    The measured quantity is the growth exponent ``p`` in ``work ~ N**p``. It
    is exactly reproducible — the counter sums tensor sizes, which depend on
    the node counts alone and not on any value — so the bound can be tight.
    The base size matters: a quadratic term only outgrows this module's large
    linear constant past a few hundred nodes, and at 64 cells a dense
    all-pairs attention still hides inside a loose bound. At 256 it does not.
    """
    cfg = MantisACTConfig()
    module = _state_module(cfg)
    latents = module.initial(torch.randn(1, 8))

    measured = {}
    for scale in (1, 2, 4):
        torch.manual_seed(SEED + 11)
        entities = {
            "cell": _stream(cfg, [256 * scale]),
            "window": _stream(cfg, [192 * scale]),
        }
        with _WorkCounter() as counter:
            module[0](latents, entities)
        measured[scale] = counter.elements

    for scale in (2, 4):
        exponent = math.log(measured[scale] / measured[1]) / math.log(scale)
        # Slightly under 1: the latent self-mix is a constant the node count
        # does not touch. Anything at or above 1.05 is a path that grows with
        # the node count more than once, which §3.14 forbids outright.
        assert 0.95 < exponent < 1.05, (scale, exponent, measured)


# --- numerics (§27, §32) ---------------------------------------------------


def test_bf16_autocast_forward_and_backward_are_finite() -> None:
    cfg = MantisACTConfig()
    module = _state_module(cfg)
    torch.manual_seed(SEED + 12)
    entities = {"cell": _stream(cfg, [9, 6]), "window": _stream(cfg, [4, 7])}
    latents = module.initial(torch.randn(2, 8))

    with torch.autocast("cpu", dtype=torch.bfloat16):
        moved, updated = module[0](latents, entities)
    total = (
        moved.inv.float().square().sum()
        + moved.axis.float().square().sum()
        + updated["cell"].state.axis.float().square().sum()
    )
    assert torch.isfinite(total)
    total.backward()
    for name, parameter in module.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name


def test_layer_scale_and_bases_use_the_specified_initialisation() -> None:
    cfg = MantisACTConfig()
    module = _state_module(cfg)
    gains = [
        (name, parameter)
        for name, parameter in module.named_parameters()
        if name.endswith("gamma")
    ]
    assert gains
    for name, parameter in gains:
        assert torch.allclose(
            parameter, torch.full_like(parameter, cfg.layer_scale_init)
        ), name
    # N(0, 0.02) bases: nowhere near a default-initialised linear's spread.
    assert module.base_inv.std().item() < 0.05
    assert module.base_axis.std().item() < 0.05


def test_axis_head_division_is_refused_loudly() -> None:
    """§6 checks only ``d_inv``; an axis width that does not divide is caught here."""
    cfg = replace(MantisACTConfig(), d_axis=26)
    with pytest.raises(ValueError, match="d_axis=26"):
        StateLatents(cfg)


def test_a_pass_serves_only_the_families_it_was_built_for() -> None:
    cfg = MantisACTConfig()
    torch.manual_seed(SEED)
    with pytest.raises(ValueError, match="distinct and nonempty"):
        LatentPass(cfg, num_inv=2, num_axis=0, entity_names=("cell", "cell"))
    with pytest.raises(ValueError, match="distinct and nonempty"):
        LatentPass(cfg, num_inv=2, num_axis=0, entity_names=())
