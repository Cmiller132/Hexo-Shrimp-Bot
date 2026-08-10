"""The action model: §19's counterfactual encoder, §21's latents, §22's blocks.

What is being detected here, and why each detector is independent of the code
it watches:

- **Engine order (§8.3, §37.5).** Output row ``j`` must be ``legal_moves()[j]``.
  The check gives every cell a state equal to its own row index and then reads
  the action rows back against the engine's own move list, so a sort anywhere
  in the gather shows up as a permuted column rather than as a plausible
  reordering nothing compares.
- **All eighteen rows, for every action (§19.2, §37.8).** Two checks. The first
  is structural: the rows a cell in *no persistent window* reads are the shared
  pre-empty-window state, exactly where ``action_window_index == -1`` and
  nowhere else. The second is causal: perturbing one of those eighteen class
  codes moves that action's output and no other action's, which is what proves
  the row is actually read rather than merely present in the batch.
- **The ``-1`` class (CLAUDE.md's hazard).** `nn.Embedding` and `index_select`
  accept a negative index and read the far end of the table, so a ``-1`` in a
  class field is a wrong row with no error and no round-trip test that can see
  it. It must raise, and so must a window index below the one sentinel §25
  allows.
- **Actions read, never write (§21, §22).** The trunk's cell, window, and
  latent tensors are compared value for value across the action forward. A
  write-back would make the state depend on post-placement information and no
  output shape would change.
- **Action-set permutation invariance (§21).** The action stack is
  permutation-*equivariant* over the legal set: permuting the actions of a
  position permutes the outputs and changes nothing else. That is the statement
  §21 needs — an action's own output does not depend on the order of its
  alternatives — in the form that also catches a positional leak.
- **The representation law (§12.1) through the new absolute axis.** Two checks,
  and the second is the real one. Dimension 1 of the ``[N, 3, 6]`` action
  tables is the only absolute axis this stage sees; relabelling it together
  with the batch's other structural axis fields must permute the action axis
  stream and leave the invariant stream alone. Then the same law is checked by
  replaying a real game through each of the twelve D6 transforms and rebuilding
  both positions, which puts the builder's own axis assignment in the chain. A
  deliberately broken encoder — one that reads a fixed absolute channel, which
  is §12.2's forbidden construction — is run through the same check, so the
  check is known to be able to fail.
- **Batching (§26).** A batch of P positions equals P single-position forwards,
  which is the detector for a softmax or a gather leaking across a position.
- **§32's disabled modules**: an arm that removes a path holds no parameter for
  it and contributes nothing.

Positions come from seeded random playouts through the engine. As
``docs/MANTIS_ACT_DEVIATIONS.md`` records, random play is nothing like the
self-play density this model will see, so nothing here asserts a family size —
the counts are only asked to be nonzero where a path would otherwise be dead
code.
"""

from __future__ import annotations

import random
from dataclasses import replace

import hexo_py
import numpy as np
import pytest
import torch

from mantisnet.models.mantis_act import action_encoder as action_encoder_module
from mantisnet.models.mantis_act.action_encoder import (
    ActionBaseState,
    ActionEncoder,
    ActionOutput,
    PostPlacementEncoder,
    StateContextBroadcast,
    WindowRows,
)
from mantisnet.models.mantis_act.builder import build
from mantisnet.models.mantis_act.config import PRESETS
from mantisnet.models.mantis_act.equivariant import (
    AXIS_CHANNELS,
    EquivariantState,
    permute_axis_channels,
)
from mantisnet.models.mantis_act.packed import (
    POST_ACTION_ROWS,
    WINDOW_LEN,
    PackedACTBatch,
    collate,
)
from mantisnet.models.mantis_act.pattern_classes import POST1_REL_CLASSES
from mantisnet.models.mantis_act.plans import (
    build_plans_from_cpu_batch,
    builder_fingerprint,
)
from mantisnet.models.mantis_act.state_trunk import WINDOW_STATUSES, StateTrunk
from mantisnet.models.mantis_act.model import MantisACT
from mantisnet.models.mantis_act.summary import parameter_summary
from mantisnet.models.mantis_act.symmetry import (
    D6_TRANSFORMS,
    axis_permutation,
    transform_coords,
)

SEED = 20260806

FULL = PRESETS["full_act_v4"]

# Both movers, both stones of a turn, all three phases, an empty board with no
# window at all, and boards dense enough that every path is populated.
PLIES = (0, 1, 2, 5, 21, 60, 120)

PERMUTATIONS = ((1, 2, 0), (2, 0, 1), (0, 2, 1), (2, 1, 0), (1, 0, 2))

ENCODER_PRESETS = tuple(PRESETS)


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


@pytest.fixture(scope="module")
def move_lists() -> dict[int, list[tuple[int, int]]]:
    return {plies: playout(plies, SEED) for plies in PLIES}


@pytest.fixture(scope="module")
def graphs(move_lists):
    return {
        plies: build(hexo_py.Position.replay(moves), FULL)
        for plies, moves in move_lists.items()
    }


@pytest.fixture(scope="module")
def batch(graphs):
    return collate([graphs[plies] for plies in PLIES], FULL)


def excited(module):
    """A module whose branches move the state, rather than whisper at 1e-2.

    Untrained LayerScales are ``layer_scale_init``, which makes every residual
    branch a rounding error; the equivariance, batching, and permutation checks
    want the branches to actually change the answer, so they would pass on a
    model that had stopped computing.
    """
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if name.endswith("gamma") and parameter.dim() == 1:
                parameter.normal_(0.5, 0.2)
    return module.eval()


@pytest.fixture(scope="module")
def trunk() -> StateTrunk:
    torch.manual_seed(SEED)
    return excited(StateTrunk(FULL))


@pytest.fixture(scope="module")
def encoder() -> ActionEncoder:
    torch.manual_seed(SEED + 1)
    return excited(ActionEncoder(FULL))


def one(graph, cfg=FULL) -> PackedACTBatch:
    """One position as a batch of one."""
    return collate([graph], cfg)


def replan(packed: PackedACTBatch, cfg) -> PackedACTBatch:
    """Bind an intentionally malformed/reordered CPU batch to ``cfg``."""
    return replace(
        packed,
        plans=build_plans_from_cpu_batch(cfg, packed),
        builder_fingerprint=builder_fingerprint(cfg),
    )


def run(trunk, encoder, packed) -> ActionOutput:
    return encoder(packed, trunk(packed))


def _model_outputs_and_gradients(model, packed):
    policy, critic = model.policy_q(packed)
    (policy.float().square().sum() + critic.float().square().sum()).backward()
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }
    return (policy.detach(), critic.detach()), gradients


def test_action_post_recompute_is_exact_against_the_literal_backward(
    graphs, monkeypatch
):
    """Selective recompute changes storage, not the §19.2 function or gradients."""
    packed = one(graphs[21])
    torch.manual_seed(SEED + 31)
    literal = MantisACT(FULL).train()
    recomputed = MantisACT(FULL).train()
    recomputed.load_state_dict(literal.state_dict())

    real_checkpoint = action_encoder_module.checkpoint
    monkeypatch.setattr(
        action_encoder_module,
        "checkpoint",
        lambda function, *args, **kwargs: function(*args),
    )
    expected_outputs, expected_gradients = _model_outputs_and_gradients(
        literal, packed
    )
    monkeypatch.setattr(action_encoder_module, "checkpoint", real_checkpoint)
    actual_outputs, actual_gradients = _model_outputs_and_gradients(
        recomputed, packed
    )

    for actual, expected in zip(actual_outputs, expected_outputs, strict=True):
        assert torch.equal(actual, expected)
    assert actual_gradients.keys() == expected_gradients.keys()
    for name in actual_gradients:
        assert torch.equal(actual_gradients[name], expected_gradients[name]), name


def test_action_post_is_replayed_once_by_backward(graphs, monkeypatch):
    packed = one(graphs[21])
    torch.manual_seed(SEED + 32)
    model = MantisACT(FULL).train()
    calls = 0
    row_calls = {"inv": 0, "axis": 0}
    post = model.actions.post
    assert post is not None
    forward = post.forward
    row_inv_forward = post.row_inv.forward
    row_axis_forward = post.row_axis.forward

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return forward(*args, **kwargs)

    def counted_row_inv(*args, **kwargs):
        row_calls["inv"] += 1
        return row_inv_forward(*args, **kwargs)

    def counted_row_axis(*args, **kwargs):
        row_calls["axis"] += 1
        return row_axis_forward(*args, **kwargs)

    monkeypatch.setattr(post, "forward", counted)
    monkeypatch.setattr(post.row_inv, "forward", counted_row_inv)
    monkeypatch.setattr(post.row_axis, "forward", counted_row_axis)
    _model_outputs_and_gradients(model, packed)
    assert calls == 2
    assert row_calls == {"inv": 3, "axis": 3}


def windowless(graph) -> np.ndarray:
    """The legal actions whose eighteen rows name no persistent window at all.

    These are §19.2's whole point: a cell in no nonempty window used to be
    described by its distance to the nearest stone, and is now described by
    what placing a stone there would produce.
    """
    return np.flatnonzero((graph.action_window_index < 0).all(axis=(1, 2)))


# --------------------------------------------------------------------------
# Construction (§22, §29)


@pytest.mark.parametrize("preset", ENCODER_PRESETS)
def test_every_preset_that_reaches_the_action_model_constructs(preset):
    cfg = PRESETS[preset]
    module = ActionEncoder(cfg)
    assert len(module.blocks) == cfg.action_blocks
    assert len(module.latents) == cfg.action_blocks


def test_each_block_holds_its_own_parameters(encoder):
    first, second = encoder.blocks[0], encoder.blocks[1]
    assert first.ffn.norm.inv is not second.ffn.norm.inv
    assert first.state_context.q_inv is not second.state_context.q_inv
    assert first.mix is not second.mix
    assert first.film is not second.film


def test_the_stage_runs_on_every_ply_and_both_phases(encoder, trunk, batch, graphs):
    out = run(trunk, encoder, batch)
    assert isinstance(out, ActionOutput)
    n_legal = int(batch.legal_offsets[-1])
    assert out.actions.inv.shape == (n_legal, FULL.d_inv)
    assert out.actions.axis.shape == (n_legal, AXIS_CHANNELS, FULL.d_axis)
    assert out.latents.inv.shape == (len(PLIES), FULL.num_action_latents, FULL.d_inv)
    assert out.latents.axis is None  # §21: invariant queries only
    assert out.position_count == len(PLIES)
    assert torch.isfinite(out.actions.inv).all()
    assert torch.isfinite(out.actions.axis).all()
    # The empty board really has no window, so the pre-empty-window path is
    # exercised on a position where it is the only path there is.
    assert graphs[0].n_windows == 0


# --------------------------------------------------------------------------
# §19.1 Engine order


def test_row_j_is_engine_legal_move_j(move_lists, graphs):
    """§8.3, §37.5: the legal ordering is a contract, never sorted."""
    moves = move_lists[21]
    position = hexo_py.Position.replay(moves)
    graph = graphs[21]

    # Every cell's state is its own row index, so an action's gathered state
    # names the cell it came from.
    n_cells = graph.n_cells
    marker = torch.arange(n_cells, dtype=torch.float32)
    cells = EquivariantState(
        marker[:, None].expand(n_cells, FULL.d_inv).contiguous(),
        marker[:, None, None].expand(n_cells, AXIS_CHANNELS, FULL.d_axis).contiguous(),
    )
    torch.manual_seed(SEED)
    gathered = ActionBaseState(FULL)(one(graph), cells)

    legal = [tuple(move) for move in position.legal_moves()]
    assert len(legal) == graph.n_legal
    read_back = [
        tuple(int(c) for c in graph.cell_qr[int(row)])
        for row in gathered.inv[:, 0].tolist()
    ]
    assert read_back == legal
    # The axis half is gathered from the same rows, not from a second mapping.
    torch.testing.assert_close(gathered.axis[:, 0, 0], gathered.inv[:, 0])


def test_the_occupied_only_arm_starts_every_action_from_the_shared_base(move_lists):
    """§8.3's all-sentinel mapping: no clamped index reads cell zero."""
    cfg = PRESETS["full_occupied_cells_only"]
    graph = build(hexo_py.Position.replay(move_lists[21]), cfg)
    assert (graph.legal_to_cell_index == -1).all()

    torch.manual_seed(SEED)
    module = ActionBaseState(cfg)
    packed = one(graph, cfg)
    state = module(packed, StateTrunk(cfg)(packed).cells)
    first = state.inv[:1].expand_as(state.inv)
    torch.testing.assert_close(state.inv, first)
    torch.testing.assert_close(state.inv[0], module.no_cell_inv)


# --------------------------------------------------------------------------
# §19.2 The eighteen post-placement rows


def test_every_action_carries_all_eighteen_rows(batch, graphs):
    n_legal = int(batch.legal_offsets[-1])
    for name in ("action_window_index", "action_post1_class", "action_pre_status"):
        assert getattr(batch, name).shape == (n_legal, AXIS_CHANNELS, WINDOW_LEN)
    assert AXIS_CHANNELS * WINDOW_LEN == POST_ACTION_ROWS
    # And the case §19.2 exists for is present: cells in no persistent window.
    assert sum(len(windowless(graphs[plies])) for plies in PLIES) > 0


def test_the_pre_empty_state_is_used_exactly_where_the_index_is_minus_one(graphs):
    """§19.2's shared learned pre-empty-window state, row by row."""
    graph = graphs[21]
    packed = one(graph)
    assert graph.n_windows > 0 and (graph.action_window_index < 0).any()

    n_windows = graph.n_windows
    marker = torch.arange(n_windows, dtype=torch.float32)
    windows = EquivariantState(
        marker[:, None].expand(n_windows, FULL.d_inv).contiguous(),
        marker[:, None, None]
        .expand(n_windows, AXIS_CHANNELS, FULL.d_axis)
        .contiguous(),
    )
    torch.manual_seed(SEED)
    module = PostPlacementEncoder(FULL)
    rows = module.window_rows(packed, windows)

    index = torch.from_numpy(graph.action_window_index)
    present = index >= 0
    assert present.any() and (~present).any()
    # A present row reads its own window, and no other.
    torch.testing.assert_close(rows.inv[present][:, 0], index[present].float())
    torch.testing.assert_close(rows.axis[present][:, 0], index[present].float())
    # An absent row reads the shared base, in both streams.
    absent_inv = rows.inv[~present]
    torch.testing.assert_close(absent_inv, module.pre_empty_inv.expand_as(absent_inv))
    absent_axis = rows.axis[~present]
    torch.testing.assert_close(absent_axis, module.pre_empty_axis.expand_as(absent_axis))


def test_the_axis_half_of_a_row_is_the_windows_own_axis_channel(graphs):
    """§12.3: a line message is routed into the structural native axis."""
    graph = graphs[21]
    n_windows = graph.n_windows
    # Channel c of every window carries the value c, so a gathered row reports
    # which channel it was read from.
    axis = torch.arange(AXIS_CHANNELS, dtype=torch.float32)[None, :, None].expand(
        n_windows, AXIS_CHANNELS, FULL.d_axis
    )
    windows = EquivariantState(torch.zeros(n_windows, FULL.d_inv), axis.contiguous())
    torch.manual_seed(SEED)
    rows = PostPlacementEncoder(FULL).window_rows(one(graph), windows)

    index = torch.from_numpy(graph.action_window_index)
    present = index >= 0
    want = torch.arange(AXIS_CHANNELS, dtype=torch.float32)[None, :, None].expand(
        index.shape
    )
    torch.testing.assert_close(rows.axis[..., 0][present], want[present])


def test_perturbing_one_of_the_eighteen_rows_moves_that_action_and_that_axis(graphs, trunk):
    """The rows are read, not merely carried: §19.2 on a cell in no window.

    The check runs on the counterfactual encoder alone rather than the whole
    stage, because §21's action-set latents are *supposed* to carry one action's
    change to the others' context. Before that mixing, one row belongs to one
    action and to one axis channel, and this is where that can be asserted.
    """
    graph = graphs[21]
    candidates = windowless(graph)
    if not len(candidates):
        pytest.skip("this playout has no legal cell outside every window")
    action, axis, slot = int(candidates[0]), 1, 3

    packed = one(graph)
    state = trunk(packed)
    torch.manual_seed(SEED)
    base = excited(ActionBaseState(FULL))(packed, state.cells)
    post = excited(PostPlacementEncoder(FULL))

    before = post(packed, base, state.windows)
    broken = replace(packed, action_post1_class=packed.action_post1_class.clone())
    original = int(broken.action_post1_class[action, axis, slot])
    broken.action_post1_class[action, axis, slot] = (original + 17) % POST1_REL_CLASSES
    broken = replan(broken, FULL)
    after = post(broken, base, state.windows)

    moved = ~torch.isclose(before.inv, after.inv).all(dim=1)
    assert bool(moved[action]), "the perturbed row was not read at all"
    assert int(moved.sum()) == 1, "one action's row reached another action"
    # And it reached that action's own axis channel, not the other two: the row
    # lies on axis 1, so channels 0 and 2 must be untouched (§12.3, §19.2).
    channels = ~torch.isclose(before.axis[action], after.axis[action]).all(dim=-1)
    assert channels.tolist() == [False, True, False]


def test_a_negative_class_is_refused_before_the_encoder_ever_gathers(graphs):
    """CLAUDE.md's hazard: a ``-1`` class is a wrong row, not an error.

    `POST1_CLASS` marks a ``(code, slot)`` pair whose slot is not own with
    ``-1``, and both numpy fancy-indexing and `nn.Embedding` read that as the
    far end of the table without complaint; on CUDA an out-of-range index is an
    asynchronous device-side assert rather than a catchable exception. What
    makes the failure legible and device-independent is that the bound runs on
    the host, in numpy, before a tensor exists — `packed.py:154` and
    `packed.py:155` — and because neither field indexes another family,
    ``collate`` does not shift them and the per-graph check is equal-strength
    on the packed batch. This is the statement that the two bounds the encoder
    leans on are the two vocabularies its tables are sized by.
    """
    assert int(np.arange(POST1_REL_CLASSES)[np.array([-1])][0]) == POST1_REL_CLASSES - 1

    graph = graphs[21]
    for field, high, label in (
        ("action_post1_class", POST1_REL_CLASSES, "action_post1_class"),
        ("action_pre_status", WINDOW_STATUSES, "action_pre_status"),
    ):
        table = getattr(graph, field).copy()
        table[0, 0, 0] = -1
        with pytest.raises(ValueError, match=rf"{label} must be >= 0"):
            replace(graph, **{field: table})

        table = getattr(graph, field).copy()
        table[0, 0, 0] = high
        with pytest.raises(ValueError, match=rf"{label} must be <= {high - 1}"):
            replace(graph, **{field: table})


def test_a_window_index_below_the_one_sentinel_is_refused_by_the_packer(graphs):
    """§25 gives these two fields one sentinel, and `-2` is not it.

    `packed.py:165` and `packed.py:171` bound both against the family they
    point into with ``-1`` as the floor, and `ACTGraph.__post_init__` runs that
    table before a graph exists — so the refusal is the graph's own, against its
    own family sizes, rather than something collation is left to notice.
    """
    graph = graphs[21]
    index = graph.action_window_index.copy()
    index[0, 0, 0] = -2
    with pytest.raises(ValueError, match=r"action_window_index must be >= -1"):
        replace(graph, action_window_index=index)

    legal = graph.legal_to_cell_index.copy()
    legal[0] = -2
    with pytest.raises(ValueError, match=r"legal_to_cell_index must be >= -1"):
        replace(graph, legal_to_cell_index=legal)


def test_a_trunk_that_did_not_preserve_its_rows_is_refused(encoder, trunk, graphs):
    """The packer bounds these indices against the *batch's* families.

    That is the same bound as the trunk output's only while the trunk is
    row-preserving, and nothing else in the package states that it is — so the
    encoder states it, on two host-side shapes, where the packer cannot see it.
    """
    packed = one(graphs[21])
    state = trunk(packed)
    short_cells = EquivariantState(
        state.cells.inv[:-1],
        None if state.cells.axis is None else state.cells.axis[:-1],
    )
    with pytest.raises(ValueError, match="cell states against the batch's"):
        ActionBaseState(FULL)(packed, short_cells)

    short_windows = EquivariantState(
        state.windows.inv[:-1],
        None if state.windows.axis is None else state.windows.axis[:-1],
    )
    with pytest.raises(ValueError, match="window states against the batch's"):
        PostPlacementEncoder(FULL).window_rows(packed, short_windows)


# --------------------------------------------------------------------------
# §21, §22 Actions read the state and never write into it


def test_the_action_stage_does_not_mutate_the_trunks_state(encoder, trunk, batch):
    """§22: an action embedding may read cells, windows, and latents only."""
    out = trunk(batch)
    kept = {
        "cells.inv": out.cells.inv.clone(),
        "cells.axis": out.cells.axis.clone(),
        "windows.inv": out.windows.inv.clone(),
        "windows.axis": out.windows.axis.clone(),
        "latents.inv": out.latents.inv.clone(),
        "latents.axis": out.latents.axis.clone(),
    }
    encoder(batch, out)
    for name, want in kept.items():
        family, stream = name.split(".")
        got = getattr(getattr(out, family), stream)
        assert torch.equal(got, want), f"the action stage wrote into {name}"


def test_the_action_latents_are_a_separate_stack_from_the_states(encoder, trunk):
    """§21: state latents must not carry post-placement effects."""
    assert encoder.latents is not trunk.latents
    assert encoder.latents.num_axis == 0
    assert encoder.latents.num_inv == FULL.num_action_latents


# --------------------------------------------------------------------------
# §21 Permutation invariance over the action set


def permute_actions(packed: PackedACTBatch, order) -> PackedACTBatch:
    """The same position with its legal actions listed in another order.

    Every per-action row of §25 moves together. The engine's order is the one
    this model must emit (§8.3), so this is a test manipulation and not a
    supported input; what it asks is whether an action's answer depends on
    where its alternatives sit in the list.
    """
    permuted = replace(
        packed,
        legal_to_cell_index=packed.legal_to_cell_index[order],
        action_window_index=packed.action_window_index[order],
        action_post1_class=packed.action_post1_class[order],
        action_pre_status=packed.action_pre_status[order],
        action_tactical_numeric=packed.action_tactical_numeric[order],
    )
    return replace(permuted, plans=build_plans_from_cpu_batch(FULL, permuted))


def test_permuting_the_action_set_permutes_the_outputs_and_nothing_else(
    encoder, trunk, graphs
):
    """§21: permutation-invariant context, and §3.14's forbidden alternative.

    The stage is permutation-*equivariant* over the legal set, which is the
    statement §21 needs in the form that also catches a positional leak: an
    action's own output is unchanged by the order of the others, and the
    action-set latents — a symmetric read over the whole set — are unchanged
    outright.
    """
    packed = one(graphs[21])
    n_legal = int(packed.legal_offsets[-1])
    assert n_legal > 1
    order = torch.randperm(n_legal, generator=torch.Generator().manual_seed(SEED))

    before = run(trunk, encoder, packed)
    after = run(trunk, encoder, permute_actions(packed, order))
    torch.testing.assert_close(
        after.actions.inv, before.actions.inv[order], atol=2e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        after.actions.axis, before.actions.axis[order], atol=2e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        after.latents.inv, before.latents.inv, atol=2e-5, rtol=1e-5
    )


# --------------------------------------------------------------------------
# §12.1 The representation law through the action tables


def relabel(packed: PackedACTBatch, permutation) -> PackedACTBatch:
    """The batch with every absolute axis identity relabelled by ``permutation``.

    Three structural fields of the state input, plus dimension 1 of the three
    action tables — the axis a post-placement row's window lies on, which is the
    only absolute axis the action stage sees at all. Everything else it reads
    is already a D6 invariant, so this is exactly the group's action on the
    input and §12.1 must carry it to the output.
    """
    table = torch.tensor(permutation, dtype=torch.long)
    route = packed.radius_axis_or_neg1
    relabelled = replace(
        packed,
        window_axis=table[packed.window_axis],
        adjacency_axis=table[packed.adjacency_axis],
        radius_axis_or_neg1=torch.where(route >= 0, table[route.clamp(min=0)], route),
        action_window_index=permute_axis_channels(packed.action_window_index, permutation),
        action_post1_class=permute_axis_channels(packed.action_post1_class, permutation),
        action_pre_status=permute_axis_channels(packed.action_pre_status, permutation),
    )
    return replace(relabelled, plans=build_plans_from_cpu_batch(FULL, relabelled))


@pytest.mark.parametrize("permutation", PERMUTATIONS)
def test_relabelling_the_axes_permutes_only_the_action_axis_stream(
    encoder, trunk, graphs, permutation
):
    packed = one(graphs[21])
    before = run(trunk, encoder, packed)
    after = run(trunk, encoder, relabel(packed, permutation))
    torch.testing.assert_close(
        after.actions.inv, before.actions.inv, atol=2e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        after.actions.axis,
        permute_axis_channels(before.actions.axis, permutation),
        atol=2e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        after.latents.inv, before.latents.inv, atol=2e-5, rtol=1e-5
    )


def test_reading_a_fixed_axis_channel_fails_the_same_check(trunk, graphs):
    """The detector is known to be able to fail (CLAUDE.md, §12.2).

    The broken encoder reads channel 0 of every row's window instead of the
    channel the row's own axis names — a per-absolute-axis read, which is what
    §12.2 forbids and what a wrong index into the flattened window/channel pair
    would produce.
    """

    class _FixedChannel(PostPlacementEncoder):
        def window_rows(self, batch, windows):
            rows = super().window_rows(batch, windows)
            if rows.axis is None:
                return rows
            fixed = rows.axis[:, :1].expand_as(rows.axis).contiguous()
            return WindowRows(rows.inv, fixed)

    torch.manual_seed(SEED + 1)
    broken = excited(ActionEncoder(FULL))
    torch.manual_seed(SEED + 1)
    broken.post = excited(_FixedChannel(FULL))

    packed = one(graphs[21])
    before = broken(packed, trunk(packed))
    after = broken(relabel(packed, (1, 2, 0)), trunk(relabel(packed, (1, 2, 0))))
    with pytest.raises(AssertionError):
        torch.testing.assert_close(
            after.actions.axis,
            permute_axis_channels(before.actions.axis, (1, 2, 0)),
            atol=2e-5,
            rtol=1e-5,
        )


def action_correspondence(base, image, transform_index) -> np.ndarray:
    """Row of ``image``'s action set holding the transform of each base action.

    Engine legal order is not preserved by a board transform — it is the
    engine's, not a sort — so the two action sets are matched by the coordinate
    each action places on, which is what the transform does map.
    """
    rows = {
        tuple(qr): index
        for index, qr in enumerate(image.cell_qr[image.legal_to_cell_index].tolist())
    }
    moved = transform_coords(
        transform_index, base.cell_qr[base.legal_to_cell_index]
    )
    return np.array([rows[tuple(qr)] for qr in moved.tolist()], dtype=np.int64)


@pytest.mark.parametrize("transform_index", range(len(D6_TRANSFORMS)))
def test_every_d6_transform_maps_the_action_states(
    encoder, trunk, move_lists, transform_index
):
    """§37.3 and §37.5 on real engine positions, through the builder.

    The relabel above permutes the action tables that a correct builder would
    have emitted. This replays the game itself through each of the twelve
    transforms and rebuilds both positions, so the axis the Rust action-row
    builder assigns to each of the eighteen rows, the reversal-canonical
    post-placement class, and the model's routing are all in the chain being
    checked.
    """
    moves = move_lists[21]
    base = build(hexo_py.Position.replay(moves), FULL)
    turned = build(
        hexo_py.Position.replay(
            [D6_TRANSFORMS[transform_index](move) for move in moves]
        ),
        FULL,
    )
    assert base.n_legal == turned.n_legal

    before = run(trunk, encoder, one(base))
    after = run(trunk, encoder, one(turned))
    order = torch.from_numpy(action_correspondence(base, turned, transform_index))
    permutation = axis_permutation(transform_index)

    torch.testing.assert_close(
        after.actions.inv.index_select(0, order),
        before.actions.inv,
        atol=2e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        after.actions.axis.index_select(0, order),
        permute_axis_channels(before.actions.axis, permutation),
        atol=2e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        after.latents.inv, before.latents.inv, atol=2e-5, rtol=1e-5
    )


# --------------------------------------------------------------------------
# §26 Batching


def test_a_batch_agrees_with_single_position_forwards(encoder, trunk, graphs, batch):
    out = run(trunk, encoder, batch)
    compared = 0
    for index, plies in enumerate(PLIES):
        graph = graphs[plies]
        if graph.n_windows == 0:
            # `packed.collate` cannot pack a lone position whose window family
            # is empty; the empty board is covered inside the batch above.
            continue
        single = run(trunk, encoder, one(graph))
        rows = slice(
            int(batch.legal_offsets[index]), int(batch.legal_offsets[index + 1])
        )
        torch.testing.assert_close(
            single.actions.inv, out.actions.inv[rows], atol=2e-5, rtol=1e-4
        )
        torch.testing.assert_close(
            single.actions.axis, out.actions.axis[rows], atol=2e-5, rtol=1e-4
        )
        torch.testing.assert_close(
            single.latents.inv, out.latents.inv[index : index + 1], atol=2e-5, rtol=1e-4
        )
        compared += 1
    assert compared >= len(PLIES) - 1


# --------------------------------------------------------------------------
# §27 Numerics


def test_bf16_autocast_forward_and_backward_are_finite(batch):
    torch.manual_seed(SEED)
    state = StateTrunk(FULL)
    module = ActionEncoder(FULL)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = module(batch, state(batch))
    loss = out.actions.inv.float().square().mean() + out.actions.axis.float().square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads)
    assert all(p.dtype is torch.float32 for p in module.parameters())


def test_the_residual_streams_stay_fp32_under_autocast(trunk, batch):
    torch.manual_seed(SEED)
    module = ActionEncoder(FULL).eval()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = module(batch, trunk(batch))
    assert out.actions.inv.dtype is torch.float32
    assert out.actions.axis.dtype is torch.float32


# --------------------------------------------------------------------------
# §32 The arms that remove a path


def test_the_disabled_tactical_block_contributes_nothing(move_lists, batch, trunk):
    cfg = PRESETS["full_no_tactical_inputs"]
    module = ActionEncoder(cfg).eval()
    assert module.tactical is None and module.tactical_width == 0
    assert not any("tactical" in name for name, _ in module.named_parameters())

    graph = build(hexo_py.Position.replay(move_lists[21]), cfg)
    packed = one(graph, cfg)
    assert packed.action_tactical_numeric.shape[1] == 0
    out = module(packed, StateTrunk(cfg)(packed))
    assert torch.isfinite(out.actions.inv).all()

    # And the full model refuses that zero-width block rather than reading a
    # projection of nothing; the disabled model refuses a populated one.
    with pytest.raises(ValueError, match="action_tactical_numeric carries 0 columns"):
        full_bound = replan(packed, FULL)
        ActionEncoder(FULL)(full_bound, StateTrunk(FULL)(full_bound))
    with pytest.raises(ValueError, match="action_tactical_numeric carries 12 columns"):
        disabled_bound = replan(batch, cfg)
        module(disabled_bound, StateTrunk(cfg)(disabled_bound))


def test_the_no_action_latents_arm_holds_no_action_latent_parameters(graphs):
    cfg = PRESETS["full_no_action_latents"]
    batch = collate([graphs[plies] for plies in PLIES], cfg)
    module = ActionEncoder(cfg).eval()
    assert not any(pass_.enabled for pass_ in module.latents.passes)
    assert not list(module.latents.parameters())
    out = module(batch, StateTrunk(cfg)(batch))
    assert out.latents.inv is None
    assert torch.isfinite(out.actions.inv).all()


def test_the_no_latents_arm_reads_no_state_latent_context(graphs):
    cfg = PRESETS["full_no_latents"]
    batch = collate([graphs[plies] for plies in PLIES], cfg)
    module = ActionEncoder(cfg).eval()
    assert all(block.state_context is None for block in module.blocks)
    out = module(batch, StateTrunk(cfg)(batch))
    assert out.latents.inv is None
    assert torch.isfinite(out.actions.inv).all()
    with pytest.raises(ValueError, match="num_inv_latents"):
        StateContextBroadcast(cfg)


def test_the_state_context_read_is_the_gathered_attention_it_replaced(
    trunk, graphs
):
    """§22.1 through `latent_attention.latent_broadcast`, against the gather.

    The broadcast reads a configured-constant context per position, which is
    §17.4's shape exactly, so the read is that op rather than a second one
    shaped like it. What a shared op cannot check for itself is the mapping:
    which axis of the context is the softmax's, which dimension carries the
    three channels, and whether the scale is the head's. Those are restated
    here as the literal gathered chain the module used to run — promote the
    context, gather it onto every action, score, softmax over the context rows,
    and contract — so a transposed key or a softmax over the channel dimension
    is a wrong answer rather than a differently-shaped one that still runs.
    """
    packed = one(graphs[21])
    state = trunk(packed)
    torch.manual_seed(SEED + 3)
    module = excited(StateContextBroadcast(FULL))
    n_legal = int(packed.legal_to_cell_index.shape[0])
    positions = torch.zeros(n_legal, dtype=torch.long)

    torch.manual_seed(SEED + 4)
    actions = EquivariantState(
        torch.randn(n_legal, FULL.d_inv),
        torch.randn(n_legal, AXIS_CHANNELS, FULL.d_axis),
    )
    got = module(actions, state.latents, positions, packed.legal_offsets)

    inv, axis = state.latents.inv, state.latents.axis
    heads = FULL.num_heads
    context = [module.norm_src_inv(inv) + module.type_src[0]]
    normed_axis = module.norm_src_axis(axis)
    pooled = EquivariantState(
        inv.mean(dim=1, keepdim=True).expand(-1, module.num_axis, -1), normed_axis
    )
    context.append(module.axis_to_inv(module.pool_src_axis(pooled)) + module.type_src[1])
    rows = torch.cat(context, dim=1)

    def attend(query, key, value, dim):
        score = (query.unsqueeze(1) * key.index_select(0, positions)).sum(-1)
        score = score / (key.shape[-1] ** 0.5)
        weight = score.softmax(dim=1)
        return (weight.unsqueeze(-1) * value.index_select(0, positions)).sum(dim=1)

    shape = (rows.shape[0], rows.shape[1], heads, module.head_dim_inv)
    out = attend(
        module.q_inv(module.norm_q_inv(actions.inv)).view(
            n_legal, heads, module.head_dim_inv
        ),
        module.k_inv(rows).view(shape),
        module.v_inv(rows).view(shape),
        1,
    )
    want_inv = actions.inv + module.scale_inv(
        module.o_inv(out.reshape(n_legal, FULL.d_inv))
    )

    shape = (
        inv.shape[0],
        module.num_axis,
        AXIS_CHANNELS,
        heads,
        module.head_dim_axis,
    )
    out = attend(
        module.q_axis(module.norm_q_axis(actions.axis)).view(
            n_legal, AXIS_CHANNELS, heads, module.head_dim_axis
        ),
        module.k_axis(normed_axis).view(shape),
        module.v_axis(normed_axis).view(shape),
        1,
    )
    want_axis = actions.axis + module.scale_axis(
        module.o_axis(out.reshape(n_legal, AXIS_CHANNELS, FULL.d_axis))
    )

    torch.testing.assert_close(got.inv, want_inv, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(got.axis, want_axis, atol=2e-6, rtol=2e-6)


def test_the_no_axis_arm_holds_no_axis_parameters_anywhere(graphs):
    cfg = PRESETS["full_no_axis"]
    batch = collate([graphs[plies] for plies in PLIES], cfg)
    module = ActionEncoder(cfg).eval()
    assert module.base.no_cell_axis is None
    assert module.post.pre_empty_axis is None
    assert not any("axis" in name for name, _ in module.named_parameters())
    out = module(batch, StateTrunk(cfg)(batch))
    assert out.actions.axis is None
    assert torch.isfinite(out.actions.inv).all()


def test_the_counterfactual_encoder_can_be_removed_whole(graphs):
    cfg = replace(FULL, use_counterfactual_action_windows=False)
    batch = collate([graphs[plies] for plies in PLIES], cfg)
    module = ActionEncoder(cfg).eval()
    assert module.post is None
    assert not any("post" in name for name, _ in module.named_parameters())
    out = module(batch, StateTrunk(cfg)(batch))
    assert torch.isfinite(out.actions.inv).all()


# --------------------------------------------------------------------------
# §6, §32 Parameters


def test_the_parameter_summary_partitions_every_trainable_scalar(encoder):
    summary = parameter_summary(encoder)
    assert summary.total == sum(p.numel() for p in encoder.parameters())
    assert sum(count for _name, count in summary.groups) == summary.total
    # The eighteen-row encoder is the stage's own work and must not be a
    # rounding error beside the block stack.
    by_name = dict(summary.groups)
    assert by_name["post.post1"] == POST1_REL_CLASSES * FULL.d_rel
