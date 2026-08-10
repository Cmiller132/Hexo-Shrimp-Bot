"""The assembled graph end to end, and §30.18 over a mixed batch.

The stage modules each have their own tests against their own oracles; this one
is about what only the assembled graph can be wrong about. Three classes of
fault live here and nowhere else: a stage handed another stage's arrays in the
wrong frame or the wrong colour convention, a configuration that no preset
combination has ever been built under, and an index that is valid inside one
position and meaningless once the batch concatenates it.

So the engine is the oracle for the position-level facts — the legal list and
its order, the placements left in the turn, the stones and their seats — and
the graph is read for the rest. Counts are pinned twice over: as bands around
values measured on the seeded playouts below, which say a family is still the
size a Stage E budget was written against, and as exact relations between
presets, which say a scope still changes exactly what it claims to. The bands
alone would not catch a scope regression that moves a family by a few percent
— dropping mixed windows moves the window count by four — and the relations
alone would not catch a family that shrank uniformly.
"""

from __future__ import annotations

from dataclasses import fields, replace

import hexo_py
import numpy as np
import pytest

from mantisnet.models.mantis_act import (
    GLOBAL_NUMERIC_NAMES,
    POST_ACTION_ROWS,
    PRESETS,
    MantisACTConfig,
    build,
    build_from_arrays,
    collate,
    collate_positions,
    collate_prefixes,
)
from mantisnet.models.mantis_act.packed import (
    PHASE_FIRST,
    PHASE_OPENING,
    PHASE_SECOND,
)
from mantisnet.models.mantis_act.pattern_classes import (
    EMPTY,
    MIXED,
    OPP_LIVE,
    OWN_LIVE,
)

from ..conftest import random_moves
# A game that really ended, by engine replay. `test_act_windows` states that
# construction; restating it here would be a second claim about what a won
# position looks like.
from .test_act_windows import won_position

# Both movers, both stones of a turn, all three phases, and boards from empty
# to past the depth a real game reaches.
PLIES = (0, 1, 2, 5, 20, 60, 120, 151)
SEED = 7

FULL = PRESETS["full_act_v4"]

# The config fields the builder itself reads. Two presets agreeing on all of
# them produce the same graph from the same position, so one representative of
# each distinct combination covers every graph the preset set can ask for.
BUILDER_FIELDS = (
    "window_scope",
    "cell_scope",
    "d6_relation_mode",
    "d_max",
    "occupied_radius",
    "use_cell_adjacency",
    "use_occupied_radius_edges",
    "use_window_numeric_features",
    "use_global_numeric_features",
    "use_action_tactical_features",
)

# The presets that reach a distinct combination of those fields, in preset
# order. Pinned as a literal as well as derived, so a preset added with a new
# builder-visible setting fails here rather than going unbuilt.
BUILDER_PRESETS = (
    "full_act_v4",
    "full_live_windows",
    "full_action_relevant_windows",
    "full_coarse_geometry",
    "full_radius6",
    "full_occupied_cells_only",
    "full_no_tactical_inputs",
)

# Family sizes measured on the seeded playouts below. The playout is
# deterministic, so these are exact today; the band is what makes the check a
# statement about how dense a board of that depth is rather than about one
# sequence of dice, which is the quantity a packer limit is set from.
MEASURED = {
    20: {"cells": 1334, "windows": 345, "adjacency": 7722, "radius": 8493, "legal": 1314},
    60: {"cells": 3027, "windows": 1012, "adjacency": 17650, "radius": 26238, "legal": 2967},
    120: {"cells": 4365, "windows": 1967, "adjacency": 25592, "radius": 53994, "legal": 4245},
}
BAND = 0.25


@pytest.fixture(scope="module")
def act_moves() -> dict[int, list[tuple[int, int]]]:
    """One seeded nonterminal playout per depth, as its move list."""
    return {plies: random_moves(plies, seed=SEED) for plies in PLIES}


@pytest.fixture(scope="module")
def act_positions(act_moves) -> dict[int, hexo_py.Position]:
    return {plies: hexo_py.Position.replay(moves) for plies, moves in act_moves.items()}


def builder_signature(cfg: MantisACTConfig) -> tuple:
    return tuple(getattr(cfg, name) for name in BUILDER_FIELDS)


def owning_position(offsets, index) -> np.ndarray:
    """The batch position each index of a family falls in."""
    return np.searchsorted(np.asarray(offsets), np.asarray(index), side="right") - 1


def engine_arrays(pos: hexo_py.Position):
    """A position as the ``(stone_qr, stone_owner, mover, legal_qr)`` build takes."""
    stones = np.asarray(pos.stones(), dtype=np.int64).reshape(-1, 3)
    legal = np.asarray(pos.legal_moves(), dtype=np.int64).reshape(-1, 2)
    return stones[:, :2], stones[:, 2], pos.current_player, legal


# --------------------------------------------------------------------------
# Every preset, every depth


def test_the_builder_affecting_presets_are_the_ones_exercised():
    seen: dict[tuple, str] = {}
    for name, cfg in PRESETS.items():
        seen.setdefault(builder_signature(cfg), name)
    assert tuple(seen.values()) == BUILDER_PRESETS
    # Every field named above is a real field, so a renamed one is not silently
    # dropped from the signature.
    assert set(BUILDER_FIELDS) <= {f.name for f in fields(MantisACTConfig)}


@pytest.mark.parametrize("preset", BUILDER_PRESETS)
def test_every_preset_builds_and_validates_at_every_depth(preset, act_positions):
    cfg = PRESETS[preset]
    for plies, pos in act_positions.items():
        graph = build(pos, cfg)
        assert graph.n_legal == pos.legal_count, f"{preset} at ply {plies}"
        assert graph.moves_remaining == pos.moves_remaining
        assert int(graph.cell_is_occupied.sum()) == (
            pos.stone_count if cfg.cell_scope != "occupied_only" else graph.n_cells
        )
        assert graph.action_window_index.shape == (graph.n_legal, 3, 6)
        assert graph.action_post1_class.shape == (graph.n_legal, 3, 6)
        assert graph.action_pre_status.shape == (graph.n_legal, 3, 6)
        # §30.13: the counterfactual rows are dense, so every action has all 18
        # and every one of them carries a class.
        assert graph.action_post1_class.size == graph.n_legal * POST_ACTION_ROWS
        assert (graph.action_post1_class >= 0).all()


def test_legal_rows_follow_the_engines_own_order(act_positions):
    """§30.12: output row ``j`` is ``legal_moves[j]``, cell node and all."""
    for pos in act_positions.values():
        graph = build(pos, FULL)
        legal = np.asarray(pos.legal_moves(), dtype=np.int64).reshape(-1, 2)
        assert np.array_equal(graph.cell_qr[graph.legal_to_cell_index], legal)
        # The cell nodes are in (q, r) order and the actions are not, so the
        # mapping is a real permutation rather than the identity.
        assert graph.n_legal == len(legal)


def test_occupied_only_gives_every_action_the_sentinel(act_positions):
    """§29's control: legal cells are not nodes, so no action has one."""
    cfg = PRESETS["full_occupied_cells_only"]
    for pos in act_positions.values():
        graph = build(pos, cfg)
        assert (graph.legal_to_cell_index == -1).all()
        assert graph.n_cells == pos.stone_count
        assert int(graph.cell_is_legal.sum()) == 0
        # Its incidence is the occupied slots alone: each stone sits in 18
        # windows, all of which the nonempty scope persists.
        assert int(graph.window_incidence_mask.sum()) == 18 * pos.stone_count


# --------------------------------------------------------------------------
# Position-level fields the stage modules do not own


def test_the_phase_follows_moves_remaining_and_the_board(act_positions):
    seen = set()
    for pos in act_positions.values():
        graph = build(pos, FULL)
        if pos.moves_remaining == 2:
            expected = PHASE_FIRST
        elif pos.stone_count == 0:
            expected = PHASE_OPENING
        else:
            expected = PHASE_SECOND
        assert graph.phase_id == expected
        seen.add(expected)
    assert seen == {PHASE_OPENING, PHASE_FIRST, PHASE_SECOND}


def test_the_global_vector_is_the_spec_list_read_off_the_position(act_positions):
    for pos in act_positions.values():
        graph = build(pos, FULL)
        assert graph.global_numeric.shape == (len(GLOBAL_NUMERIC_NAMES),)
        stones = pos.stones()
        own = sum(1 for _q, _r, player in stones if player == pos.current_player)
        status = graph.window_status

        def share(count: int, total: int) -> float:
            return count / total if total else 0.0

        expected = [
            np.log1p(len(stones)),
            share(own, len(stones)),
            share(len(stones) - own, len(stones)),
            np.log1p(pos.legal_count),
            np.log1p(graph.n_windows),
            share(int((status == OWN_LIVE).sum()), graph.n_windows),
            share(int((status == OPP_LIVE).sum()), graph.n_windows),
            share(int((status == MIXED).sum()), graph.n_windows),
        ]
        assert graph.global_numeric.tolist() == pytest.approx(expected, rel=1e-6)


def test_a_disabled_numeric_block_has_no_width(act_positions):
    pos = act_positions[20]
    off = replace(
        FULL,
        use_global_numeric_features=False,
        use_window_numeric_features=False,
        use_action_tactical_features=False,
    )
    bare, full = build(pos, off), build(pos, FULL)
    assert bare.global_numeric.shape == (0,)
    assert bare.window_numeric.shape == (bare.n_windows, 0)
    assert bare.action_tactical_numeric.shape == (bare.n_legal, 0)
    # Only the widths change: a disabled block removes an input, not a node.
    assert (bare.n_cells, bare.n_windows, bare.n_legal) == (
        full.n_cells,
        full.n_windows,
        full.n_legal,
    )
    assert full.window_numeric.shape[1] > 0
    assert full.action_tactical_numeric.shape[1] > 0


# --------------------------------------------------------------------------
# Family sizes: measured bands, and the relations between the scopes


@pytest.mark.parametrize("plies", sorted(MEASURED))
def test_family_sizes_land_in_their_measured_bands(plies, act_positions):
    graph = build(act_positions[plies], FULL)
    actual = {
        "cells": graph.n_cells,
        "windows": graph.n_windows,
        "adjacency": graph.n_adjacency,
        "radius": graph.n_radius,
        "legal": graph.n_legal,
    }
    for name, measured in MEASURED[plies].items():
        assert actual[name] == pytest.approx(measured, rel=BAND), (
            f"{name} at ply {plies} is {actual[name]}, measured {measured}"
        )


def test_the_scopes_change_exactly_what_they_claim(act_positions):
    for plies in (20, 60, 120):
        pos = act_positions[plies]
        full = build(pos, FULL)
        live = build(pos, PRESETS["full_live_windows"])
        relevant = build(pos, PRESETS["full_action_relevant_windows"])

        # Windows: live drops exactly the mixed ones, action_relevant adds the
        # empty ones through legal cells and nothing else.
        mixed = int((full.window_status == MIXED).sum())
        assert mixed > 0
        assert live.n_windows == full.n_windows - mixed
        assert relevant.n_windows > 4 * full.n_windows
        assert int((relevant.window_status == EMPTY).sum()) == (
            relevant.n_windows - full.n_windows
        )

        # Cells: `window_and_legal` and `occupied_and_legal` coincide on this
        # game. A nonempty window's cells lie within five steps of its stone,
        # and the legal radius is eight, so every empty cell of a persistent
        # window is already a legal cell. `occupied_only` is therefore the one
        # cell scope that changes the node set.
        wide = build(pos, replace(FULL, cell_scope="occupied_and_legal"))
        assert full.n_cells == wide.n_cells == pos.stone_count + full.n_legal
        assert build(pos, PRESETS["full_occupied_cells_only"]).n_cells == pos.stone_count
        # Under that scope every slot of every persistent window is a node.
        assert int(full.window_incidence_mask.sum()) == 6 * full.n_windows

        # Geometry: radius six is the same relation vocabulary over a smaller
        # disk, so the edge count falls by roughly the disks' area ratio.
        six = build(pos, PRESETS["full_radius6"])
        assert 2.0 < full.n_radius / six.n_radius < 5.0
        assert six.n_adjacency == full.n_adjacency
        coarse = build(pos, PRESETS["full_coarse_geometry"])
        assert coarse.n_radius == full.n_radius
        assert int(coarse.radius_orbit.max()) < int(full.radius_orbit.max())


# --------------------------------------------------------------------------
# §30.18 — batching


@pytest.fixture(scope="module")
def mixed_batch(act_positions):
    """Positions of four sizes and all three phases, and their packed batch."""
    chosen = [act_positions[plies] for plies in (0, 1, 2, 5, 20, 60)]
    graphs = [build(pos, FULL) for pos in chosen]
    return chosen, graphs, collate(graphs, FULL)


def test_the_offsets_slice_each_position_back_out(mixed_batch):
    _positions, graphs, batch = mixed_batch
    assert batch.position_count == len(graphs)
    families = {
        "cell_offsets": [g.n_cells for g in graphs],
        "window_offsets": [g.n_windows for g in graphs],
        "legal_offsets": [g.n_legal for g in graphs],
        "adjacency_offsets": [g.n_adjacency for g in graphs],
        "radius_offsets": [g.n_radius for g in graphs],
    }
    for name, counts in families.items():
        offsets = getattr(batch, name).numpy()
        assert offsets[0] == 0
        assert np.array_equal(np.diff(offsets), np.array(counts))

    # A slice of a packed family is its position's own array again.
    cell_offsets = batch.cell_offsets.numpy()
    for i, graph in enumerate(graphs):
        lo, hi = cell_offsets[i], cell_offsets[i + 1]
        assert np.array_equal(
            batch.cell_occupancy.numpy()[lo:hi], graph.cell_occupancy
        )


def test_no_edge_crosses_a_batch_position(mixed_batch):
    """§30.18, over every field that indexes another family."""
    _positions, graphs, batch = mixed_batch
    cells = batch.cell_offsets.numpy()
    windows = batch.window_offsets.numpy()
    legal = batch.legal_offsets.numpy()

    def rows_of(offsets):
        return np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))

    checks = (
        ("legal_to_cell_index", legal, cells),
        ("window_cell_index", windows, cells),
        ("adjacency_src", batch.adjacency_offsets.numpy(), cells),
        ("adjacency_dst", batch.adjacency_offsets.numpy(), cells),
        ("radius_src", batch.radius_offsets.numpy(), cells),
        ("radius_dst", batch.radius_offsets.numpy(), cells),
        ("action_window_index", legal, windows),
    )
    for name, row_offsets, target_offsets in checks:
        index = getattr(batch, name).numpy()
        flat = index.reshape(len(rows_of(row_offsets)), -1)
        present = flat >= 0
        owner = owning_position(target_offsets, np.where(present, flat, 0))
        rows = np.broadcast_to(rows_of(row_offsets)[:, None], flat.shape)
        assert np.array_equal(owner[present], rows[present]), name
        # The sentinel is not shifted into the previous position's slice.
        assert flat[~present].tolist() == [-1] * int((~present).sum())

    # The batch is only a detector if its positions have different sizes.
    assert len({g.n_cells for g in graphs}) == len(graphs)


def test_an_index_that_would_cross_a_position_never_reaches_collation(mixed_batch):
    """Where the crossing fault is actually stopped, on a real built graph.

    One past its own family's last cell is the *next* position's first cell
    once the offsets are applied — plausible after concatenation and invisible
    to every shape and dtype check downstream. It never gets that far: the
    graph refuses it at construction against its own family's size, which is
    the stronger statement of the two because it does not depend on what else
    happens to be in the batch.
    """
    _positions, graphs, _batch = mixed_batch
    leading = graphs[-2]
    crossing = leading.adjacency_dst.copy()
    crossing[0] = leading.n_cells
    with pytest.raises(
        ValueError, match=rf"adjacency_dst must be <= {leading.n_cells - 1}"
    ):
        replace(leading, adjacency_dst=crossing)


def test_collate_positions_and_collate_prefixes_agree(act_moves, act_positions):
    plies = [0, 2, 5, 20]
    direct = collate_positions([act_positions[p] for p in plies], FULL)
    # The same games, named by a prefix length rather than by a board.
    prefixes = collate_prefixes(
        [list(act_moves[p]) for p in plies], [len(act_moves[p]) for p in plies], FULL
    )
    for name in ("cell_occupancy", "legal_to_cell_index", "radius_orbit", "phase_id"):
        assert np.array_equal(
            getattr(direct, name).numpy(), getattr(prefixes, name).numpy()
        )
    # A shorter prefix of the same game is a different, earlier position.
    earlier = collate_prefixes([act_moves[20]], [5], FULL)
    assert int(earlier.cell_offsets[-1]) < int(direct.cell_offsets[-1])


# --------------------------------------------------------------------------
# Refusals


def test_a_terminal_position_is_refused(act_positions):
    pos = won_position(axis=0)
    assert pos.is_terminal
    with pytest.raises(ValueError, match="terminal position"):
        build(pos, FULL)
    # And through the array path, where an empty legal list is the only
    # evidence available before any stage runs.
    stone_qr, stone_owner, mover, _legal = engine_arrays(act_positions[20])
    with pytest.raises(ValueError, match="terminal position"):
        build_from_arrays(
            stone_qr, stone_owner, mover, np.empty((0, 2), dtype=np.int64), 1, FULL
        )


def test_malformed_input_is_refused_by_name():
    stone_qr = np.array([[0, 0], [1, 0]], dtype=np.int64)
    owners = np.array([0, 1], dtype=np.int64)
    legal = np.array([[2, 0]], dtype=np.int64)

    with pytest.raises(ValueError, match="2 stone coordinates against 1 owners"):
        build_from_arrays(stone_qr, owners[:1], 0, legal, 1, FULL)
    with pytest.raises(ValueError, match="mover must be player 0 or 1, got 2"):
        build_from_arrays(stone_qr, owners, 2, legal, 1, FULL)
    with pytest.raises(ValueError, match=r"stone_owner\[1\] = 5 is neither player"):
        build_from_arrays(
            stone_qr, np.array([0, 5], dtype=np.int64), 0, legal, 1, FULL
        )
    with pytest.raises(ValueError, match="moves_remaining must be 1 or 2, got 3"):
        build_from_arrays(stone_qr, owners, 0, legal, 3, FULL)


def test_collate_prefixes_refuses_a_length_that_is_not_a_prefix(act_moves):
    moves = act_moves[20]
    with pytest.raises(ValueError, match="1 games against 2 prefix lengths"):
        collate_prefixes([moves], [1, 2], FULL)
    with pytest.raises(ValueError, match="prefix 0 asks for 21 moves of a 20-move"):
        collate_prefixes([moves], [21], FULL)
