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

The fused kernels of `latent_attention` are held to that same reference, which
is what §36 requires of an optimized path. Four more things are being detected
in the second half of this module:

- the kernel's forward and backward agree with the gather formulation on real
  ragged shapes, including a position with no rows and a two-family
  concatenation whose rows are *not* sorted by position;
- the reference's written-out backward really is the derivative — a float64
  ``gradcheck``, which falls back by signature and so validates the formula the
  kernels implement rather than the kernels themselves;
- repeated runs agree bit for bit, which the segment reductions buy over an
  atomic scatter and which the D6 tolerance analysis needs;
- the fused path never allocates a per-node score matrix. That is the whole
  point of the change and the one property a parity test cannot see, so it is
  measured against the allocator.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import replace

import pytest
import torch

import mantisnet.models.mantis_act.latent_attention as kernel
from mantisnet.models.mantis_act.config import PRESETS, MantisACTConfig
from mantisnet.models.mantis_act.equivariant import (
    AXIS_CHANNELS,
    EquivariantState,
    permute_axis_channels,
)
from mantisnet.models.mantis_act.latent_attention import (
    broadcast_reference,
    latent_broadcast,
    latent_read,
    latent_segments,
    read_reference,
    validate_read,
)
from mantisnet.models.mantis_act.latents import (
    ActionLatents,
    LatentPass,
    LatentState,
    RaggedStream,
    StateLatents,
    row_positions,
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
    total = sum(counts)
    axis = None if cfg.d_axis == 0 else torch.randn(total, AXIS_CHANNELS, cfg.d_axis)
    return RaggedStream(
        EquivariantState(torch.randn(total, cfg.d_inv), axis),
        offsets,
        row_positions(offsets, total),
    )


def _state_module(cfg: MantisACTConfig) -> StateLatents:
    torch.manual_seed(SEED)
    module = StateLatents(cfg)
    module.eval()
    return module


def _slice(stream: RaggedStream, position: int) -> RaggedStream:
    lo, hi = int(stream.offsets[position]), int(stream.offsets[position + 1])
    axis = None if stream.state.axis is None else stream.state.axis[lo:hi]
    offsets = torch.tensor([0, hi - lo], dtype=torch.long)
    return RaggedStream(
        EquivariantState(stream.state.inv[lo:hi], axis),
        offsets,
        row_positions(offsets, hi - lo),
    )


def _rows(stream: RaggedStream, index: torch.Tensor) -> RaggedStream:
    """The same family with its rows reordered inside their positions."""
    axis = None if stream.state.axis is None else stream.state.axis[index]
    return RaggedStream(
        EquivariantState(stream.state.inv[index], axis),
        stream.offsets,
        stream.row_pos,
    )


def _turn(stream: RaggedStream, permutation) -> RaggedStream:
    return RaggedStream(
        stream.state.permute_axes(permutation), stream.offsets, stream.row_pos
    )


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

    packed = read_reference(q, k, v, row_positions(offsets, total), positions)[0]
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
    row_pos = row_positions(offsets, 12)

    order = torch.tensor([3, 0, 5, 1, 4, 2, 9, 11, 6, 10, 7, 8])
    torch.testing.assert_close(
        read_reference(q, k[order], v[order], row_pos[order], 2)[0],
        read_reference(q, k, v, row_pos, 2)[0],
        rtol=1e-5,
        atol=1e-6,
    )


def test_segment_attention_reads_zero_from_an_empty_position() -> None:
    """``full_occupied_cells_only`` on an empty board has no rows at all."""
    torch.manual_seed(SEED)
    offsets = _offsets([0, 3])
    q = torch.randn(2, 2, 1, 2, 4)
    k = torch.randn(3, 1, 2, 4)
    out = read_reference(q, k, k, row_positions(offsets, 3), 2)[0]
    assert torch.isfinite(out).all()
    assert (out[0] == 0).all()
    assert (out[1] != 0).any()


def test_the_read_refuses_mismatched_shapes() -> None:
    """The op's one front door, ahead of the fused/reference dispatch."""
    q = torch.randn(2, 2, 1, 2, 4)
    k = torch.randn(5, 1, 2, 4)
    with pytest.raises(ValueError, match="row_pos must be"):
        validate_read(q, k, k, torch.zeros(4, dtype=torch.long))
    wide = torch.randn(5, 3, 2, 4)
    with pytest.raises(ValueError, match="disagree"):
        validate_read(q, wide, wide, torch.zeros(5, dtype=torch.long))
    three = _offsets([2, 2, 1])
    with pytest.raises(ValueError, match="the segments describe 3"):
        latent_read(
            q, k, k, latent_segments([three], [row_positions(three, 5)])
        )


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
        entities["cell"].row_pos,
    )
    with pytest.raises(ValueError, match="axis width"):
        module[0](latents, wrong)


def test_row_positions_refuses_a_row_count_the_offsets_do_not_reach() -> None:
    """``output_size`` is a check, not a hint, and it runs on every call.

    This is what replaced the host reads of ``offsets[-1]`` scattered through
    the trunk, the action encoder and the heads. ATen compares the requested
    size against the offsets' own cumulative total
    (``result_size == cumsum_ptr[size - 1]``) inside the kernel that builds the
    vector, so the predicate is enforced by the code that consumes it, on both
    devices, rather than read back beside it.
    """
    offsets = _offsets([5, 0, 3])
    torch.testing.assert_close(
        row_positions(offsets, 8), torch.tensor([0] * 5 + [2] * 3)
    )
    for wrong in (7, 9):
        with pytest.raises(RuntimeError, match="size does not match"):
            row_positions(offsets, wrong)
    with pytest.raises(ValueError, match="n_rows must not be negative"):
        row_positions(offsets, -1)


def test_latent_segments_refuses_a_row_vector_per_family_it_does_not_have() -> None:
    """The two views of the key rows are given, so they cannot be given apart."""
    families = [_offsets(c) for c in COUNTS]
    rows = [row_positions(o, sum(c)) for o, c in zip(families, COUNTS)]
    assert latent_segments(families, rows).families == len(families)
    with pytest.raises(ValueError, match="row-position vectors"):
        latent_segments(families, rows[:-1])
    with pytest.raises(ValueError, match="not a 1-D row vector"):
        latent_segments(families, [*rows[:-1], rows[-1].unsqueeze(0)])


def test_ragged_offsets_that_disagree_with_the_rows_are_refused() -> None:
    """A family's offsets must end where its rows do, on both of its two views.

    The predicate is enforced twice and neither reads a value back from the
    device. ``row_positions`` is given the family's row count as ATen's
    ``output_size``, which refuses a value disagreeing with the offsets' own
    total (``result_size == cumsum_ptr[size - 1]``) — on every call rather than
    once per pass. `RaggedStream` then holds the vector that came out to the
    state it travels with, so a row-position vector built for one family cannot
    be carried beside another.
    """
    cfg = MantisACTConfig()
    torch.manual_seed(SEED + 10)
    entities = {"cell": _stream(cfg, [5, 3]), "window": _stream(cfg, [2, 4])}

    short = _offsets([5, 2])  # ends at 7 where the cell family carries 8 rows
    with pytest.raises(RuntimeError, match="size does not match"):
        row_positions(short, entities["cell"].rows)

    with pytest.raises(ValueError, match=r"row_pos is \(7,\)"):
        RaggedStream(entities["cell"].state, short, row_positions(short, 7))


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


# --- the fused kernels against the reference (§36) --------------------------

_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the fused latent attention needs CUDA"
)

# Every disagreement below is fp32 reassociation: the kernel sums a segment in
# registers where the reference sums the same rows through an `index_add_`, in
# a different order. Measured worst case across these shapes is 9.0e-7, and the
# bound is set an order of magnitude above it.
KERNEL_TOL = 1e-5

# The trunk's own two read signatures at the default configuration: four
# invariant latents over one channel of four 16-wide heads, and two axis
# latents over three channels of four 6-wide heads.
SIGNATURES = ((4, 1, 4, 16), (2, AXIS_CHANNELS, 4, 6))

# Two node families whose concatenation is *not* sorted by position, which is
# what the trunk hands the read; position 4 owns no row at all, which
# ``full_occupied_cells_only`` reaches on an empty board.
COUNTS = ([5, 1, 9, 4, 0], [3, 7, 0, 6, 0])


def relative(got: torch.Tensor, want: torch.Tensor) -> float:
    got, want = got.detach().float().cpu(), want.detach().float().cpu()
    scale = float(want.abs().max())
    gap = float((got - want).abs().max())
    return gap if scale == 0.0 else gap / scale


def _draw(shape, device, dtype, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=generator).to(device=device, dtype=dtype)


def _read_inputs(signature, device, *, counts=COUNTS, dtype=torch.float32, seed=0):
    slots, channels, heads, head_dim = signature
    family_offsets = [_offsets(c).to(device) for c in counts]
    segments = latent_segments(
        family_offsets, [row_positions(o, sum(c)) for o, c in zip(family_offsets, counts)]
    )
    rows, tail = segments.n_rows, (channels, heads, head_dim)
    return (
        segments,
        _draw((segments.positions, slots, *tail), device, dtype, seed),
        _draw((rows, *tail), device, dtype, seed + 1),
        _draw((rows, *tail), device, dtype, seed + 2),
    )


def _broadcast_inputs(signature, device, *, dtype=torch.float32, seed=0):
    """One node family reading a per-position context of ``R`` rows.

    ``R`` is the invariant broadcast's ``K_inv + 1`` — the invariant latents
    plus the pooled axis one — or the axis broadcast's ``K_axis``.
    """
    context, channels, heads, head_dim = signature
    offsets = _offsets([9, 0, 14, 3, 6]).to(device)
    node_pos = row_positions(offsets, 32)
    rows, tail = int(node_pos.shape[0]), (channels, heads, head_dim)
    positions = int(offsets.shape[0]) - 1
    return (
        offsets,
        node_pos,
        _draw((rows, *tail), device, dtype, seed),
        _draw((positions, context, *tail), device, dtype, seed + 1),
        _draw((positions, context, *tail), device, dtype, seed + 2),
    )


def _peak(call):
    """Bytes the allocator's high-water mark rose by over one call."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    held = call()
    torch.cuda.synchronize()
    high = torch.cuda.max_memory_allocated() - base
    del held
    return high


def test_the_segments_describe_the_same_rows_as_the_position_index() -> None:
    """The two views of a read's key rows cannot be allowed to disagree.

    ``row_pos`` is what the reference and the per-node gradient index by;
    ``ranges`` is what the ragged walk steps through. They are built from the
    same offsets, and this is the statement that the derivation is right: every
    range of a position holds exactly that position's rows, and the ranges of
    all positions partition the concatenation.
    """
    segments = latent_segments(
        [_offsets(c) for c in COUNTS], [row_positions(_offsets(c), sum(c)) for c in COUNTS]
    )
    assert segments.n_rows == sum(sum(c) for c in COUNTS)
    seen = torch.zeros(segments.n_rows, dtype=torch.long)
    for position in range(segments.positions):
        owned = 0
        for family in range(segments.families):
            lo, hi = (int(x) for x in segments.ranges[position, family])
            assert int(segments.range_base[position, family]) == owned
            owned += hi - lo
            assert (segments.row_pos[lo:hi] == position).all()
            seen[lo:hi] += 1
        assert owned == int(segments.counts[position])
    assert (seen == 1).all()


@pytest.mark.parametrize("signature", SIGNATURES)
def test_the_reference_backward_is_the_derivative(signature) -> None:
    """float64 falls back by signature, so this checks the formula itself.

    The kernels recompute every attention weight in the backward from the
    queries, the keys and the saved segment statistics rather than keeping the
    per-node score matrix the forward would otherwise have to store. A
    numerical derivative is what says that recomputation is the gradient.
    """
    slots, channels, heads, _ = signature
    small = (slots, channels, heads, 3)
    segments, q, k, v = _read_inputs(
        small, "cpu", counts=([3, 0, 2], [1, 2, 0]), dtype=torch.float64, seed=5
    )
    leaves = tuple(t.requires_grad_(True) for t in (q, k, v))
    assert torch.autograd.gradcheck(
        lambda *args: latent_read(*args, segments), leaves, eps=1e-6, atol=1e-8
    )

    offsets, node_pos, bq, bk, bv = _broadcast_inputs(
        small, "cpu", dtype=torch.float64, seed=9
    )
    leaves = tuple(t.requires_grad_(True) for t in (bq, bk, bv))
    assert torch.autograd.gradcheck(
        lambda *args: latent_broadcast(*args, node_pos, offsets),
        leaves,
        eps=1e-6,
        atol=1e-8,
    )


@_CUDA
@pytest.mark.parametrize("signature", SIGNATURES)
def test_the_fused_read_matches_the_reference(signature) -> None:
    segments, q, k, v = _read_inputs(signature, "cuda")
    with torch.no_grad():
        want, _m, _l = read_reference(q, k, v, segments.row_pos, segments.positions)
        got = latent_read(q, k, v, segments)
    assert relative(got, want) < KERNEL_TOL
    # A position with no rows reads zero rather than a nan.
    assert torch.isfinite(got).all()
    assert (got[4] == 0).all()


@_CUDA
@pytest.mark.parametrize("signature", SIGNATURES)
def test_the_fused_read_backward_matches_the_reference(signature) -> None:
    segments, q, k, v = _read_inputs(signature, "cuda", seed=3)
    grad_out = _draw(q.shape, "cuda", torch.float32, 21)

    def gradients(call):
        leaves = [t.clone().requires_grad_(True) for t in (q, k, v)]
        return torch.autograd.grad(call(*leaves), leaves, grad_out)

    want = gradients(
        lambda a, b, c: read_reference(a, b, c, segments.row_pos, segments.positions)[0]
    )
    got = gradients(lambda a, b, c: latent_read(a, b, c, segments))
    for name, fast, slow in zip("qkv", got, want):
        assert relative(fast, slow) < KERNEL_TOL, name


@_CUDA
@pytest.mark.parametrize("signature", SIGNATURES)
def test_the_fused_broadcast_matches_the_reference(signature) -> None:
    slots, channels, heads, head_dim = signature
    context = (slots + 1, channels, heads, head_dim)
    offsets, node_pos, q, k, v = _broadcast_inputs(context, "cuda")
    with torch.no_grad():
        want = broadcast_reference(q, k, v, node_pos)
        got = latent_broadcast(q, k, v, node_pos, offsets)
    assert relative(got, want) < KERNEL_TOL

    grad_out = _draw(q.shape, "cuda", torch.float32, 31)

    def gradients(call):
        leaves = [t.clone().requires_grad_(True) for t in (q, k, v)]
        return torch.autograd.grad(call(*leaves), leaves, grad_out)

    want = gradients(lambda a, b, c: broadcast_reference(a, b, c, node_pos))
    got = gradients(lambda a, b, c: latent_broadcast(a, b, c, node_pos, offsets))
    for name, fast, slow in zip("qkv", got, want):
        assert relative(fast, slow) < KERNEL_TOL, name


@_CUDA
@pytest.mark.parametrize("signature", SIGNATURES)
def test_repeated_fused_runs_agree_bit_for_bit(signature) -> None:
    """Determinism, not merely accuracy.

    The D6 tolerance analysis reads a residual as reassociation noise. That
    reading is only available if the residual is the same every time, which is
    what the sliced segment reductions buy over an atomic scatter.
    """
    segments, q, k, v = _read_inputs(signature, "cuda", seed=13)
    grad_out = _draw(q.shape, "cuda", torch.float32, 41)
    runs = []
    for _ in range(3):
        leaves = [t.clone().requires_grad_(True) for t in (q, k, v)]
        out = latent_read(*leaves, segments)
        runs.append((out.detach(), *torch.autograd.grad(out, leaves, grad_out)))
    for later in runs[1:]:
        for first, other in zip(runs[0], later):
            assert torch.equal(first, other)


@_CUDA
@pytest.mark.parametrize("signature", SIGNATURES)
def test_the_fused_path_allocates_no_score_matrix(signature) -> None:
    """The whole point of the change, and the one thing parity cannot see.

    The gather formulation keeps ``(N, K, C, heads)`` scores and several
    ``(N, K, C, heads, head_dim)`` tensors alive. The fused one keeps the
    ``(P, K, C, heads, head_dim)`` output, its statistics, and the split
    partials — so its working set is a function of the *configuration* and not
    of the node count, and doubling the nodes must not move it at all. That is
    the tight statement; the absolute bound below a single score matrix is the
    loose one, and neither can be met by trimming a constant.
    """
    slots, channels, heads, head_dim = signature
    base = ([4000, 6000, 3000, 5000], [2500, 1500, 4000, 2000])
    measured = {}
    for scale in (1, 2):
        counts = [[rows * scale for rows in family] for family in base]
        segments, q, k, v = _read_inputs(signature, "cuda", counts=counts, seed=17)
        with torch.no_grad():
            fused = _peak(lambda: latent_read(q, k, v, segments))
            gathered = _peak(
                lambda: read_reference(
                    q, k, v, segments.row_pos, segments.positions
                )[0]
            )
        measured[scale] = (
            fused,
            gathered,
            segments.n_rows * slots * channels * heads * 4,
        )
        del segments, q, k, v

    assert measured[2][0] <= measured[1][0], measured
    fused, gathered, score_matrix = measured[1]
    assert fused < score_matrix / 4, (fused, score_matrix)
    assert fused * 20 < gathered, (fused, gathered)

    offsets, node_pos, bq, bk, bv = _broadcast_inputs(
        (slots + 1, channels, heads, head_dim), "cuda", seed=19
    )
    with torch.no_grad():
        fused = _peak(lambda: latent_broadcast(bq, bk, bv, node_pos, offsets))
        gathered = _peak(lambda: broadcast_reference(bq, bk, bv, node_pos))
    assert fused * 4 < gathered, (fused, gathered)


@_CUDA
def test_the_fallback_agrees_with_the_kernel_it_replaces() -> None:
    """An unsupported signature must gather rather than fail.

    The failure caches are how a launch that raised once stops being retried,
    so poisoning them is the honest way to reach the fallback: the ops take the
    same inputs and must answer the same numbers.
    """
    segments, q, k, v = _read_inputs(SIGNATURES[0], "cuda", seed=23)
    grad_out = _draw(q.shape, "cuda", torch.float32, 51)

    def run():
        leaves = [t.clone().requires_grad_(True) for t in (q, k, v)]
        out = latent_read(*leaves, segments)
        return (out.detach(), *torch.autograd.grad(out, leaves, grad_out))

    fused = run()
    key = kernel._shape_key("read", q, segments.families)
    kernel._FAILED_SHAPES[key] = "poisoned by the test"
    kernel._FAILED_BACKWARD_SHAPES[key] = "poisoned by the test"
    try:
        fallen = run()
    finally:
        kernel._FAILED_SHAPES.pop(key, None)
        kernel._FAILED_BACKWARD_SHAPES.pop(key, None)
    for index, (fell, fast) in enumerate(zip(fallen, fused)):
        assert relative(fell, fast) < KERNEL_TOL, index


@_CUDA
def test_a_whole_pass_agrees_with_the_gather_path() -> None:
    """§36's random-weight parity, taken at the level the model runs at.

    A per-kernel comparison can miss a stream wired to the wrong context or the
    wrong offsets, because both sides would then read the same wrong thing.
    This runs the default pass over ragged families twice — once on the kernels
    and once with them refused, so every attention falls back to the gather —
    and compares the outputs and every parameter gradient.

    Gradients are compared against the *pass's* largest gradient rather than
    each tensor's own scale: the LayerScale gains start at 1e-2 and several
    branch gradients are near-total cancellations, so their own-relative error
    is meaningless while their absolute error is nothing.
    """
    cfg = MantisACTConfig()
    module = _state_module(cfg).cuda()
    torch.manual_seed(SEED + 20)
    entities = {}
    for name, counts in (("cell", [40, 17, 0, 25]), ("window", [23, 8, 0, 12])):
        stream = _stream(cfg, counts)
        entities[name] = RaggedStream(
            EquivariantState(stream.state.inv.cuda(), stream.state.axis.cuda()),
            stream.offsets.cuda(),
            stream.row_pos.cuda(),
        )
    global_numeric = torch.randn(4, 8, device="cuda")

    def run():
        module.zero_grad(set_to_none=True)
        moved, updated = module[0](module.initial(global_numeric), entities)
        streams = (
            moved.inv,
            moved.axis,
            updated["cell"].state.inv,
            updated["cell"].state.axis,
            updated["window"].state.inv,
            updated["window"].state.axis,
        )
        sum(stream.square().sum() for stream in streams).backward()
        return (
            tuple(stream.detach() for stream in streams),
            {
                name: parameter.grad.detach().clone()
                for name, parameter in module.named_parameters()
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
        assert relative(fell, fast) < KERNEL_TOL, f"output {index}"
    assert set(fallen[1]) == set(fused[1])
    scale = max(float(grad.abs().max()) for grad in fused[1].values())
    for name, fast in fused[1].items():
        assert float((fallen[1][name] - fast).abs().max()) < 1e-6 * scale, name


@_CUDA
def test_a_host_tensor_is_an_unsupported_signature() -> None:
    segments, q, k, v = _read_inputs(SIGNATURES[0], "cpu")
    assert not kernel._supported(q, segments.n_rows)
    assert kernel._supported(q.cuda(), segments.n_rows)
    assert not kernel._supported(q.cuda().double(), segments.n_rows)
