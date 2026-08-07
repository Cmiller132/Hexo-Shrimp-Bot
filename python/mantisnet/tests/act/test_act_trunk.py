"""The state trunk: §18's blocks, §31's debug forward, and the §12.1 law.

What is being detected here, and why each detector is independent of the code
it watches:

- **The representation law (§12.1) on the whole trunk.** Two tests, and the
  second is the real one. The first relabels the batch's three structural axis
  fields by a channel permutation — the only place an absolute axis appears in
  the model's input at all — and requires every invariant output unchanged and
  every axis output permuted with it. The second replays a *real game through
  each of the twelve D6 transforms*, builds both positions from the engine, and
  matches the cell and window nodes by their transformed geometry. That one
  goes through the builder, the pattern classes, the orbit table, and the
  trunk together, so a wrong axis route anywhere in the chain fails it. A
  deliberately broken embedding — one learned base per absolute axis, which is
  §12.2's forbidden construction — is run through the same check, so the check
  is known to be able to fail.
- **Batching (§26, §31.10).** A batch of P positions equals P single-position
  forwards. That is the detector for a segment reduction or a softmax leaking
  across a position boundary, which no single-position test can see.
- **The debug forward (§31).** Its tensors are the production forward's, not a
  recomputation, and the production forward returns none of them.
- **§18's structure**: one block per configured depth, and six distinct final
  norms — a trunk that reused one would still train, and only a test that
  compares the module identities would notice.
- **§32's disabled modules**: an arm that removes a stream holds no parameter
  for it and contributes nothing.

Positions come from seeded random playouts through the engine. As
``docs/MANTIS_ACT_DEVIATIONS.md`` records, random play is nothing like the
self-play density this model will see, so nothing here asserts a family size —
the counts are only asked to be nonzero where a path would otherwise be dead
code.
"""

from __future__ import annotations

import random
import warnings
from dataclasses import replace

import hexo_py
import numpy as np
import pytest
import torch
from torch import nn

from mantisnet.models.mantis_act.builder import build
from mantisnet.models.mantis_act.cells import relevant_cells
from mantisnet.models.mantis_act.config import PRESETS, MantisACTConfig
from mantisnet.models.mantis_act.equivariant import (
    AXIS_CHANNELS,
    EquivariantState,
    permute_axis_channels,
)
from mantisnet.models.mantis_act.latents import row_positions
from mantisnet.models.mantis_act.packed import (
    _VALUE_RANGES,
    NEAREST_BUCKETS,
    PHASE_FIRST,
    PHASE_OPENING,
    PHASE_SECOND,
    PackedACTBatch,
    collate,
)
from mantisnet.models.mantis_act.pattern_classes import ALL_WINDOW_PATTERN_CLASSES
from mantisnet.models.mantis_act.state_trunk import (
    CELL_LEGAL_CLASSES,
    CELL_OCCUPANCY_CLASSES,
    WINDOW_STATUSES,
    CellEmbedding,
    StateTrunk,
    StateTrunkBlock,
    TrunkOutput,
    WindowEmbedding,
    state_edges,
)
from mantisnet.models.mantis_act.symmetry import (
    D6_TRANSFORMS,
    axis_permutation,
    transform_coords,
)
from mantisnet.models.mantis_act.windows import window_cells

SEED = 20260806

FULL = PRESETS["full_act_v4"]

# Both movers, both stones of a turn, all three phases, an empty board with no
# window at all, and boards dense enough that every edge family is populated.
PLIES = (0, 1, 2, 5, 21, 60, 120)

# Every permutation of the three axis channels except the identity: the four a
# D6 element induces, plus the two transpositions that fix a channel.
PERMUTATIONS = ((1, 2, 0), (2, 0, 1), (0, 2, 1), (2, 1, 0), (1, 0, 2))

TRUNK_PRESETS = tuple(PRESETS)

# §16's arm, whose step 5 the other presets do not hold.
TYPED_PRESET = "full_with_typed_window_attention"


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
    """Every ply in one packed batch, in ply order."""
    return collate([graphs[plies] for plies in PLIES])


@pytest.fixture(scope="module")
def trunk() -> StateTrunk:
    torch.manual_seed(SEED)
    module = StateTrunk(FULL)
    # Untrained LayerScales are 1e-2, which makes every branch a whisper; the
    # equivariance and batching checks want the branches to actually move the
    # state, so the module is given a nontrivial random parameterisation.
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if name.endswith("gamma") and parameter.dim() == 1:
                parameter.normal_(0.5, 0.2)
    return module.eval()


def one(graph) -> PackedACTBatch:
    """One position as a batch of one."""
    return collate([graph])


# --------------------------------------------------------------------------
# Construction (§18, §29)


@pytest.mark.parametrize("preset", TRUNK_PRESETS)
def test_every_preset_that_reaches_the_trunk_constructs(preset):
    cfg = PRESETS[preset]
    module = StateTrunk(cfg)
    assert len(module.blocks) == cfg.state_blocks
    assert len(module.latents) == cfg.state_blocks


def test_typed_window_attention_exists_only_in_the_arm_that_asks_for_it():
    """§18 step 5 is a stage of the block, not a disabled branch inside one."""
    assert all(block.window_attention is None for block in StateTrunk(FULL).blocks)
    typed = StateTrunk(PRESETS[TYPED_PRESET])
    assert all(block.window_attention is not None for block in typed.blocks)
    # Every block holds its own attention parameters (§14's block-private rule).
    first, second = typed.blocks[0].window_attention, typed.blocks[1].window_attention
    assert first.q_inv.weight is not second.q_inv.weight
    assert first.bias_inv is not second.bias_inv




def test_the_quadratic_and_tokenised_paths_are_refused():
    with pytest.raises(NotImplementedError, match="use_full_cell_attention"):
        StateTrunk(replace(FULL, use_full_cell_attention=True))
    with pytest.raises(NotImplementedError, match="phase_conditioning"):
        StateTrunk(replace(FULL, phase_conditioning="token_only"))


def test_the_six_final_norms_are_six_distinct_modules(trunk):
    """§18: separate final norms per entity type and stream, never one reused."""
    finals = [
        trunk.final_cell.inv,
        trunk.final_cell.axis,
        trunk.final_window.inv,
        trunk.final_window.axis,
        trunk.final_latent_inv,
        trunk.final_latent_axis,
    ]
    assert all(isinstance(norm, nn.LayerNorm) for norm in finals)
    assert len({id(norm) for norm in finals}) == len(finals)
    # None of them is a block's norm either: a final norm shared with a block
    # would normalise twice and would still run.
    inside = {id(m) for block in trunk.blocks for m in block.modules()}
    assert not inside & {id(norm) for norm in finals}


def test_each_block_holds_its_own_norms_and_projections(trunk):
    first, second = trunk.blocks[0], trunk.blocks[1]
    assert first.cell_ffn.norm.inv is not second.cell_ffn.norm.inv
    assert first.cell_ffn.norm.inv is not first.window_ffn.norm.inv
    assert first.cell_mix is not first.window_mix
    assert first.incidence.to_windows.wv_inv is not second.incidence.to_windows.wv_inv


def test_relation_tables_are_shared_across_blocks_but_projections_are_not(trunk):
    """§14: relation embeddings may be shared, everything else is block-private."""
    tables = [block.incidence.to_windows.relation for block in trunk.blocks]
    assert all(table is tables[0] for table in tables)
    assert tables[0] is trunk.relations.incidence
    assert trunk.blocks[0].radius.message.relation is trunk.relations.radius
    assert trunk.relations.adjacency is not trunk.relations.radius

    private = StateTrunk(replace(FULL, share_relation_embeddings_across_blocks=False))
    assert (
        private.blocks[0].incidence.to_windows.relation
        is not private.blocks[1].incidence.to_windows.relation
    )
    assert private.relations is None
    assert sum(p.numel() for p in private.parameters()) > sum(
        p.numel() for p in trunk.parameters()
    )


# --------------------------------------------------------------------------
# Real positions (§18, §26)


def test_the_ply_set_covers_all_three_phases(batch):
    assert set(batch.phase_id.tolist()) == {PHASE_OPENING, PHASE_FIRST, PHASE_SECOND}


def test_the_trunk_runs_on_every_ply_and_both_phases(trunk, graphs, batch):
    out = trunk(batch)
    assert isinstance(out, TrunkOutput)
    n_cells = int(batch.cell_offsets[-1])
    n_windows = int(batch.window_offsets[-1])
    assert out.cells.inv.shape == (n_cells, FULL.d_inv)
    assert out.cells.axis.shape == (n_cells, AXIS_CHANNELS, FULL.d_axis)
    assert out.windows.inv.shape == (n_windows, FULL.d_inv)
    assert out.windows.axis.shape == (n_windows, AXIS_CHANNELS, FULL.d_axis)
    assert out.latents.inv.shape == (len(PLIES), FULL.num_inv_latents, FULL.d_inv)
    assert out.latents.axis.shape == (
        len(PLIES),
        FULL.num_axis_latents,
        AXIS_CHANNELS,
        FULL.d_axis,
    )
    assert out.position_count == len(PLIES)
    for tensor in (out.cells.inv, out.cells.axis, out.windows.inv, out.latents.inv):
        assert torch.isfinite(tensor).all()

    # The empty board is in the batch and really has no window, so the window
    # family's zero-row case is exercised rather than assumed.
    assert graphs[0].n_windows == 0
    # And the dense plies populate every edge family, so no path is dead code.
    edges = state_edges(batch, FULL)
    assert len(edges.to_windows) > 0 and len(edges.adjacency) > 0
    assert len(edges.radius) > 0


def test_a_batch_agrees_with_single_position_forwards(trunk, graphs, batch):
    """§31.10: ragged packing must not leak across a position boundary."""
    out = trunk(batch)
    compared = 0
    for index, plies in enumerate(PLIES):
        graph = graphs[plies]
        if graph.n_windows == 0:
            # `packed.collate` cannot pack a lone position whose window family
            # is empty; the empty board is covered inside the batch above.
            continue
        single = trunk(one(graph))
        cells = slice(int(batch.cell_offsets[index]), int(batch.cell_offsets[index + 1]))
        windows = slice(
            int(batch.window_offsets[index]), int(batch.window_offsets[index + 1])
        )
        torch.testing.assert_close(single.cells.inv, out.cells.inv[cells], atol=2e-5, rtol=1e-4)
        torch.testing.assert_close(
            single.cells.axis, out.cells.axis[cells], atol=2e-5, rtol=1e-4
        )
        torch.testing.assert_close(
            single.windows.inv, out.windows.inv[windows], atol=2e-5, rtol=1e-4
        )
        torch.testing.assert_close(
            single.latents.inv, out.latents.inv[index : index + 1], atol=2e-5, rtol=1e-4
        )
        torch.testing.assert_close(
            single.latents.axis, out.latents.axis[index : index + 1], atol=2e-5, rtol=1e-4
        )
        compared += 1
    assert compared >= len(PLIES) - 1


# --------------------------------------------------------------------------
# The representation law (§12.1, §12.2, §31.4-§31.7)


def relabel(packed, permutation):
    """The same batch with its three structural axis fields relabelled.

    Those fields — a window's native axis, an adjacency edge's axis, and a
    radius edge's on-axis route — are the only absolute axis identities in the
    model's input. Everything else it reads (orbit ids, pattern classes, joint
    incidence classes) is already a D6 invariant, so relabelling them is exactly
    the group's action on the input, and §12.1 must carry it to the output.
    """
    table = torch.tensor(permutation, dtype=torch.long)
    route = packed.radius_axis_or_neg1
    return replace(
        packed,
        window_axis=table[packed.window_axis],
        adjacency_axis=table[packed.adjacency_axis],
        radius_axis_or_neg1=torch.where(route >= 0, table[route.clamp(min=0)], route),
    )


def assert_law(before: TrunkOutput, after: TrunkOutput, permutation, *, atol=1e-5):
    """§12.1: invariants unchanged, axis channels carried by ``permutation``."""
    torch.testing.assert_close(after.cells.inv, before.cells.inv, atol=atol, rtol=1e-5)
    torch.testing.assert_close(after.windows.inv, before.windows.inv, atol=atol, rtol=1e-5)
    torch.testing.assert_close(after.latents.inv, before.latents.inv, atol=atol, rtol=1e-5)
    for got, want in (
        (after.cells.axis, before.cells.axis),
        (after.windows.axis, before.windows.axis),
        (after.latents.axis, before.latents.axis),
    ):
        torch.testing.assert_close(
            got, permute_axis_channels(want, permutation), atol=atol, rtol=1e-5
        )


@pytest.mark.parametrize("permutation", PERMUTATIONS)
def test_relabelling_the_axes_permutes_only_the_axis_streams(trunk, graphs, permutation):
    packed = one(graphs[21])
    assert_law(trunk(packed), trunk(relabel(packed, permutation)), permutation)


class _PerAxisWindowEmbedding(WindowEmbedding):
    """§12.2's forbidden construction, as the negative control.

    One learned base **per absolute axis** instead of one shared base placed in
    the structural native channel. Everything else is the real embedding, so
    what the test below detects is exactly that difference.
    """

    def __init__(self, cfg: MantisACTConfig) -> None:
        super().__init__(cfg)
        self.per_axis = nn.Embedding(AXIS_CHANNELS, cfg.d_axis)
        nn.init.normal_(self.per_axis.weight, std=1.0)

    def forward(self, packed) -> EquivariantState:
        state = super().forward(packed)
        channels = torch.arange(AXIS_CHANNELS, device=state.inv.device)
        return EquivariantState(state.inv, state.axis + self.per_axis(channels))


def test_a_per_absolute_axis_base_fails_the_same_check(graphs):
    """The detector is known to be able to fail (CLAUDE.md, §12.2)."""
    torch.manual_seed(SEED)
    broken = StateTrunk(FULL).eval()
    broken.window_embedding = _PerAxisWindowEmbedding(FULL).eval()
    packed = one(graphs[21])
    with pytest.raises(AssertionError):
        assert_law(broken(packed), broken(relabel(packed, (1, 2, 0))), (1, 2, 0))


def cell_correspondence(base, image, transform_index) -> np.ndarray:
    """Row of ``image`` holding the transform of each row of ``base``."""
    rows = {tuple(qr): index for index, qr in enumerate(image.cell_qr.tolist())}
    moved = transform_coords(transform_index, base.cell_qr)
    return np.array([rows[tuple(qr)] for qr in moved.tolist()], dtype=np.int64)


def window_correspondence(base, image, transform_index) -> np.ndarray:
    """Row of ``image`` holding the transform of each window of ``base``.

    A window is matched by its cell *set*: a transform may reverse a window's
    slot order, so its stored identity is not simply the transform of the
    original's, but the six coordinates it covers are.
    """

    def key(window_id):
        cells = window_cells(window_id)
        return [tuple(sorted(tuple(qr) for qr in row.tolist())) for row in cells]

    rows = {shape: index for index, shape in enumerate(key(image.window_id))}
    turned = transform_coords(
        transform_index, window_cells(base.window_id).reshape(-1, 2)
    ).reshape(-1, 6, 2)
    shapes = [tuple(sorted(tuple(qr) for qr in row.tolist())) for row in turned]
    return np.array([rows[shape] for shape in shapes], dtype=np.int64)


@pytest.mark.parametrize("transform_index", range(len(D6_TRANSFORMS)))
def test_every_d6_transform_maps_the_trunks_states(trunk, move_lists, transform_index):
    """§31.4-§31.7 on real engine positions, through the builder and the trunk."""
    moves = move_lists[21]
    base = build(hexo_py.Position.replay(moves), FULL)
    turned = build(
        hexo_py.Position.replay(
            [D6_TRANSFORMS[transform_index](move) for move in moves]
        ),
        FULL,
    )
    assert base.n_cells == turned.n_cells and base.n_windows == turned.n_windows

    before, after = trunk(one(base)), trunk(one(turned))
    permutation = axis_permutation(transform_index)
    cells = torch.from_numpy(cell_correspondence(base, turned, transform_index))
    windows = torch.from_numpy(window_correspondence(base, turned, transform_index))

    mapped = TrunkOutput(
        cells=EquivariantState(
            after.cells.inv.index_select(0, cells), after.cells.axis.index_select(0, cells)
        ),
        windows=EquivariantState(
            after.windows.inv.index_select(0, windows),
            after.windows.axis.index_select(0, windows),
        ),
        latents=after.latents,
        position_count=after.position_count,
    )
    assert_law(before, mapped, permutation, atol=2e-5)


def test_the_cell_axis_channels_start_identical_and_the_window_channels_do_not(graphs):
    """§8.2 and §9.3's two different initial axis tensors."""
    torch.manual_seed(SEED)
    packed = one(graphs[21])
    cells = CellEmbedding(FULL)(packed)
    torch.testing.assert_close(cells.axis[:, 0], cells.axis[:, 1])
    torch.testing.assert_close(cells.axis[:, 0], cells.axis[:, 2])

    windows = WindowEmbedding(FULL)(packed)
    native = packed.window_axis
    channels = torch.arange(AXIS_CHANNELS)
    is_native = channels[None, :] == native[:, None]
    # Off-native channels are one shared neutral base, so they are all equal;
    # the native channel is not, because §9.3 projects the pattern into it.
    off = windows.axis[~is_native].reshape(-1, FULL.d_axis)
    torch.testing.assert_close(off, off[:1].expand_as(off))
    on = windows.axis[is_native]
    assert not torch.allclose(on, off[:1].expand_as(on))


# --------------------------------------------------------------------------
# The debug forward (§31)


def test_the_debug_forward_returns_the_production_forwards_own_tensors(trunk, batch):
    plain = trunk(batch)
    out, tensors = trunk.debug_forward(batch)
    for got, want in (
        (out.cells.inv, plain.cells.inv),
        (out.cells.axis, plain.cells.axis),
        (out.windows.inv, plain.windows.inv),
        (out.latents.axis, plain.latents.axis),
    ):
        assert torch.equal(got, want)
    # The recorded final states are the returned ones, not a recomputation.
    assert tensors["final.cell.inv"] is out.cells.inv
    assert tensors["final.window.axis"] is out.windows.axis
    assert tensors["final.latent.inv"] is out.latents.inv


def test_every_block_exposes_both_streams_of_all_three_families(trunk, batch):
    _out, tensors = trunk.debug_forward(batch)
    sites = trunk.debug_sites()
    assert sites[:3] == ("input.cell", "input.window", "input.latent")
    assert len(sites) == 3 * (FULL.state_blocks + 2)
    assert set(tensors) == {f"{site}.{stream}" for site in sites for stream in ("inv", "axis")}
    for index in range(FULL.state_blocks):
        assert tensors[f"block{index}.cell.axis"].shape[1:] == (AXIS_CHANNELS, FULL.d_axis)
        assert tensors[f"block{index}.latent.inv"].shape == (
            batch.position_count,
            FULL.num_inv_latents,
            FULL.d_inv,
        )
    # Successive blocks really do move the state, so a trace of a no-op trunk
    # could not pass the equivariance tests by being constant.
    assert not torch.allclose(tensors["block0.cell.inv"], tensors["block1.cell.inv"])


def test_a_selected_capture_records_only_what_was_asked_for(trunk, batch):
    _out, tensors = trunk.debug_forward(batch, capture=["block2.window"])
    assert set(tensors) == {"block2.window.inv", "block2.window.axis"}
    _out, tensors = trunk.debug_forward(batch, capture=[])
    assert tensors == {}


def test_an_unknown_debug_site_is_refused(trunk, batch):
    with pytest.raises(ValueError, match="block9.cell"):
        trunk.debug_forward(batch, capture=["block9.cell"])


def test_the_production_forward_exposes_no_intermediates(trunk, batch):
    out = trunk(batch)
    assert not hasattr(out, "debug")
    assert set(vars(out)) == {"cells", "windows", "latents", "position_count"}


# --------------------------------------------------------------------------
# Numerics (§27, §32)


def test_bf16_autocast_forward_and_backward_are_finite(batch):
    torch.manual_seed(SEED)
    module = StateTrunk(FULL)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = module(batch)
    loss = (
        out.cells.inv.float().square().mean()
        + out.windows.axis.float().square().mean()
        + out.latents.inv.float().square().mean()
    )
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads)
    assert all(p.dtype is torch.float32 for p in module.parameters())


def test_the_residual_streams_stay_fp32_under_autocast(batch):
    """§27: parameters and the state itself are fp32; only the branches cast."""
    torch.manual_seed(SEED)
    module = StateTrunk(FULL).eval()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = module(batch)
    assert out.cells.inv.dtype is torch.float32
    assert out.cells.axis.dtype is torch.float32
    assert out.latents.axis.dtype is torch.float32


# --------------------------------------------------------------------------
# The arms that remove a path (§29, §32)


def test_the_no_axis_arm_holds_no_axis_parameters_anywhere(batch):
    cfg = PRESETS["full_no_axis"]
    module = StateTrunk(cfg).eval()
    assert module.final_cell.axis is None and module.final_latent_axis is None
    assert module.cell_embedding.axis_base is None
    assert module.window_embedding.to_native is None
    assert not any("axis" in name for name, _ in module.named_parameters())
    out = module(batch)
    assert out.cells.axis is None and out.windows.axis is None
    assert out.latents.axis is None
    assert torch.isfinite(out.cells.inv).all()


def test_the_no_latents_arm_holds_no_latent_parameters(batch):
    cfg = PRESETS["full_no_latents"]
    module = StateTrunk(cfg).eval()
    assert module.final_latent_inv is None and module.final_latent_axis is None
    assert not any(pass_.enabled for pass_ in module.latents.passes)
    assert not list(module.latents.parameters())
    out = module(batch)
    assert out.latents.inv is None and out.latents.axis is None
    assert torch.isfinite(out.cells.inv).all()


def test_a_disabled_geometry_path_is_absent_rather_than_zeroed(batch):
    cfg = replace(FULL, use_cell_adjacency=False, use_occupied_radius_edges=False)
    module = StateTrunk(cfg).eval()
    assert module.blocks[0].adjacency is None and module.blocks[0].radius is None
    edges = state_edges(batch, cfg)
    assert edges.adjacency is None and edges.radius is None
    assert torch.isfinite(module(batch).cells.inv).all()


def zeros(rows: int, cfg: MantisACTConfig) -> EquivariantState:
    return EquivariantState(
        torch.zeros(rows, cfg.d_inv),
        torch.zeros(rows, AXIS_CHANNELS, cfg.d_axis),
    )


def test_a_block_refuses_an_edge_set_its_config_disagrees_with(trunk, batch):
    """A path the block runs and the edge set omits is a mismatch, not a skip."""
    edges = state_edges(batch, replace(FULL, use_cell_adjacency=False))
    n_cells, n_windows = int(batch.cell_offsets[-1]), int(batch.window_offsets[-1])
    with pytest.raises(ValueError, match="use_cell_adjacency"):
        trunk.blocks[0](
            zeros(n_cells, FULL),
            zeros(n_windows, FULL),
            trunk.latents.initial(batch.global_numeric),
            edges=edges,
            latent_pass=trunk.latents[0],
            cell_offsets=batch.cell_offsets,
            cell_row_pos=row_positions(batch.cell_offsets, n_cells),
            window_offsets=batch.window_offsets,
            window_row_pos=row_positions(batch.window_offsets, n_windows),
            cell_phase=torch.zeros(n_cells, dtype=torch.long),
            window_phase=torch.zeros(n_windows, dtype=torch.long),
        )


def test_a_typed_attention_block_refuses_an_edge_set_without_window_pairs(batch):
    """§18 step 5 skipped silently would make the arm a costly copy of §29's."""
    block = StateTrunkBlock(PRESETS[TYPED_PRESET])
    edges = state_edges(batch, FULL)
    assert edges.window_window is None
    latents = StateTrunk(FULL)
    n_cells, n_windows = int(batch.cell_offsets[-1]), int(batch.window_offsets[-1])
    with pytest.raises(ValueError, match="window_window_mode"):
        block(
            zeros(n_cells, FULL),
            zeros(n_windows, FULL),
            latents.latents.initial(batch.global_numeric),
            edges=edges,
            latent_pass=latents.latents[0],
            cell_offsets=batch.cell_offsets,
            cell_row_pos=row_positions(batch.cell_offsets, n_cells),
            window_offsets=batch.window_offsets,
            window_row_pos=row_positions(batch.window_offsets, n_windows),
            cell_phase=torch.zeros(n_cells, dtype=torch.long),
            window_phase=torch.zeros(n_windows, dtype=torch.long),
        )


def test_the_window_numeric_block_is_absent_when_it_is_disabled(graphs):
    cfg = replace(FULL, use_window_numeric_features=False)
    embedding = WindowEmbedding(cfg)
    assert embedding.numeric is None
    packed = collate([build(hexo_py.Position.replay(playout(21, SEED)), cfg)])
    assert packed.window_numeric.shape[1] == 0
    assert torch.isfinite(embedding(packed).inv).all()
    # And the full model's embedding refuses that zero-width block rather than
    # reading a projection of nothing.
    with pytest.raises(ValueError, match="window_numeric carries 0 columns"):
        WindowEmbedding(FULL)(packed)


# --------------------------------------------------------------------------
# The forward does not stall the host


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="a host stall needs a real device"
)
def test_the_forward_stalls_the_host_at_most_twice(batch):
    """§26's forward reads nothing back off the device except two row counts.

    Every bound this trunk indexes with is fixed on the host before a tensor
    exists, every ragged row count it needs is a host-side tensor shape, and
    every structural property of an edge family travels with the family. What
    is left is two ``nonzero`` calls, and neither is a check: each discovers a
    count no host tensor carries — how many window slots the cell scope
    represents, and how many radius displacements lie on an axis — and each
    runs once per batch rather than once per block.

    This is the property the whole shape of `messages.TypedEdges`,
    `latents.RaggedStream` and `latent_attention.row_positions` exists to hold,
    and nothing else in this suite can see it: a re-introduced ``int(t.min())``
    or a ``repeat_interleave`` without ``output_size`` is numerically perfect
    and costs a third of the step.
    """
    device_batch = batch.to("cuda")
    torch.manual_seed(SEED)
    module = StateTrunk(FULL).cuda().eval()

    def run():
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            module(device_batch)

    def stalls():
        torch.cuda.synchronize()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.cuda.set_sync_debug_mode("warn")
            try:
                run()
            finally:
                torch.cuda.set_sync_debug_mode("default")
        return [w for w in caught if "synchron" in str(w.message).lower()]

    # Allocator growth, lazy plan construction and any kernel autotune are not
    # the forward; they are burned before the count is taken. The first entry
    # into the warning mode itself reports one extra stall whatever runs under
    # it, so that pass is burned too and the count is taken twice after it.
    for _ in range(3):
        run()
    stalls()

    first, second = stalls(), stalls()
    assert len(first) == len(second) == 2, [str(w.message) for w in second]


# --------------------------------------------------------------------------
# Loud refusals of malformed input (house rule)


def test_every_class_index_the_embeddings_read_is_bounded_before_they_read_it():
    """The embeddings index without re-reading, so this is where the bounds are.

    Each of the six class columns the two embeddings gather with is refused by
    `ACTGraph`'s own validation against the same vocabulary the table is sized
    by, on the host, in numpy, once per position — and none of them is an index
    into another family, so `collate` does not shift them and the per-graph
    check is equal-strength on the packed batch. That includes the nearest-stone
    bucket: its ceiling is closed in `_VALUE_RANGES` beside the other five, so
    the check is the packer's and reaches every producer rather than living in
    one builder helper.
    """
    stones = np.array([[0, 0], [1, 0]], dtype=np.int64)
    own = np.array([0, 1], dtype=np.int64)
    legal = np.array([[0, 1], [1, 1]], dtype=np.int64)
    good = relevant_cells(stones, own, legal, stones, FULL)
    assert int(good.nearest_bucket.max()) < NEAREST_BUCKETS

    # The vocabularies each table is sized by, against the packer's own ranges.
    assert (CELL_OCCUPANCY_CLASSES, CELL_LEGAL_CLASSES) == (3, 2)
    bounds = dict((name, (low, high)) for name, low, high in _VALUE_RANGES)
    assert bounds["cell_occupancy"] == (0, CELL_OCCUPANCY_CLASSES - 1)
    assert bounds["cell_is_legal"] == (0, CELL_LEGAL_CLASSES - 1)
    assert bounds["cell_nearest_bucket"] == (0, NEAREST_BUCKETS - 1)
    assert bounds["window_pattern_class"] == (0, ALL_WINDOW_PATTERN_CLASSES - 1)
    assert bounds["window_status"] == (0, WINDOW_STATUSES - 1)
    assert bounds["window_axis"] == (0, AXIS_CHANNELS - 1)
