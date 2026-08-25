"""Step 13 cell-node representation, geometry, and wiring laws (CPU)."""

from __future__ import annotations

import itertools

import hexo_py
import pytest
import torch

from mantisnet import MantisConfig, MantisNet, collate, from_position
from mantisnet.builder import ORBIT48_CLASSES, orbit48_id

from .conftest import d6_transforms


def _cube_oracle_images(dq: int, dr: int) -> set[tuple[int, int]]:
    cube = (dq, dr, -dq - dr)
    return {
        (sign * permutation[0], sign * permutation[1])
        for permutation in itertools.permutations(cube)
        for sign in (-1, 1)
    }


def test_orbit48_matches_independent_cube_permutation_oracle():
    representatives = sorted(
        {
            (max(abs(q), abs(r), abs(q + r)), *min(_cube_oracle_images(q, r)))
            for q in range(-12, 13)
            for r in range(-12, 13)
            if 1 <= max(abs(q), abs(r), abs(q + r)) <= 12
        }
    )
    assert len(representatives) == ORBIT48_CLASSES
    ranks = {(q, r): rank for rank, (_distance, q, r) in enumerate(representatives)}
    for dq in range(-12, 13):
        for dr in range(-12, 13):
            distance = max(abs(dq), abs(dr), abs(dq + dr))
            if 1 <= distance <= 12:
                assert orbit48_id(dq, dr) == ranks[min(_cube_oracle_images(dq, dr))]


def _edge_rows(graph):
    stones = [tuple(row) for row in graph.stone_qr.tolist()]
    cells = [tuple(row) for row in graph.cell_qr.tolist()]
    return {
        (
            stones[int(graph.radius_src[i])],
            cells[int(graph.radius_dst[i])],
            int(graph.radius_orbit[i]),
            int(graph.radius_own[i]),
            int(graph.radius_on_axis[i]),
        )
        for i in range(len(graph.radius_src))
    }


def test_new_builder_inputs_are_d6_invariant_and_only_permuted():
    moves = [(0, 0), (3, 0), (-2, 2), (0, 3), (1, -2), (-1, 3)]
    base = from_position(hexo_py.Position.replay(moves))
    base_cells = {
        tuple(coord): int(base.cell_nearest[i])
        for i, coord in enumerate(base.cell_qr.tolist())
    }
    base_edges = _edge_rows(base)

    for transform in d6_transforms():
        graph = from_position(hexo_py.Position.replay([transform(move) for move in moves]))
        cells = {
            tuple(coord): int(graph.cell_nearest[i])
            for i, coord in enumerate(graph.cell_qr.tolist())
        }
        assert cells == {transform(coord): value for coord, value in base_cells.items()}
        assert _edge_rows(graph) == {
            (transform(source), transform(destination), orbit, own, on_axis)
            for source, destination, orbit, own, on_axis in base_edges
        }


def _tiny(**overrides) -> MantisConfig:
    values = dict(
        h=16,
        blocks=1,
        heads=2,
        value_queries=2,
        value_bins=5,
        policy_hidden=16,
        value_hidden=16,
    )
    values.update(overrides)
    return MantisConfig(**values)


def test_knob_off_is_byte_identical_to_the_incumbent_path():
    position = hexo_py.Position.replay([(0, 0), (3, 0), (-2, 2), (0, 3)])
    batch = collate([from_position(position)])
    torch.manual_seed(2026)
    incumbent = MantisNet(_tiny())
    torch.manual_seed(2026)
    explicit_off = MantisNet(_tiny(cell_nodes=False, cell_adjacency=False))
    for left, right in zip(incumbent.state_dict().values(), explicit_off.state_dict().values()):
        assert torch.equal(left, right)
    incumbent.eval()
    explicit_off.eval()
    with torch.no_grad():
        left = incumbent(batch, 0.2)
        right = explicit_off(batch, 0.2)
    for name in vars(left):
        assert getattr(left, name).cpu().numpy().tobytes() == getattr(
            right, name
        ).cpu().numpy().tobytes()


@pytest.mark.parametrize("scope", ["all", "uncovered"])
def test_far_cell_alias_is_removed_by_radius_edges(scope):
    batch = collate([from_position(hexo_py.Position.replay([(0, 0)]))])
    covered = set(batch.dec_cell.tolist())
    uncovered = [cell for cell in range(batch.n_cells) if cell not in covered]
    orbit_by_cell = {
        int(dst): int(orbit)
        for dst, orbit in zip(batch.radius_dst.tolist(), batch.radius_orbit.tolist())
    }
    representatives = {}
    for cell in uncovered:
        representatives.setdefault(orbit_by_cell[cell], cell)
    cells = list(representatives.values())
    assert len(cells) > 2

    torch.manual_seed(7)
    off = MantisNet(_tiny(cell_latents=True)).eval()
    torch.manual_seed(7)
    on = MantisNet(_tiny(cell_nodes=True, cell_node_scope=scope)).eval()
    with torch.no_grad():
        off_cells = off.trunk(batch)[2]
        on_cells = on.trunk(batch)[2]
    for left, right in itertools.combinations(cells, 2):
        assert torch.equal(off_cells[left], off_cells[right])
        assert not torch.equal(on_cells[left], on_cells[right])


def _table_edges(tables):
    return set(
        zip(
            tables.edge_window.tolist(),
            tables.edge_cell.tolist(),
            tables.edge_class.tolist(),
        )
    )


def test_uncovered_scope_filters_radius_and_adjacency_destinations():
    position = hexo_py.Position.replay([(0, 0), (3, 0), (-2, 2), (0, 3)])
    batch = collate([from_position(position)])
    all_model = MantisNet(_tiny(cell_nodes=True, cell_adjacency=True))
    uncovered_model = MantisNet(
        _tiny(
            cell_nodes=True,
            cell_node_scope="uncovered",
            cell_adjacency=True,
        )
    )
    covered = all_model._cell_tables(batch, len(batch.window_feat)).covered
    covered_set = set(covered.tolist())

    all_radius = _table_edges(
        all_model._radius_tables(batch, covered, len(batch.stone_own))
    )
    uncovered_radius = _table_edges(
        uncovered_model._radius_tables(batch, covered, len(batch.stone_own))
    )
    assert uncovered_radius < all_radius
    assert not ({destination for _, destination, _ in uncovered_radius} & covered_set)
    assert uncovered_radius == {
        edge for edge in all_radius if edge[1] not in covered_set
    }

    all_adjacency = _table_edges(all_model._adjacency_tables(batch, covered))
    uncovered_adjacency = _table_edges(
        uncovered_model._adjacency_tables(batch, covered)
    )
    assert uncovered_adjacency < all_adjacency
    assert not (
        {destination for _, destination, _ in uncovered_adjacency} & covered_set
    )
    assert uncovered_adjacency == {
        edge for edge in all_adjacency if edge[1] not in covered_set
    }


@torch.no_grad()
@pytest.mark.parametrize("scope", ["all", "uncovered"])
def test_cell_node_model_is_d6_invariant(scope):
    moves = [(0, 0), (3, 0), (-2, 2), (0, 3), (1, -2), (-1, 3)]
    torch.manual_seed(11)
    model = MantisNet(_tiny(cell_nodes=True, cell_node_scope=scope)).eval()
    model.mlp_p.out.weight.normal_(std=0.05)
    model.mlp_q.out.weight.normal_(std=0.05)
    base_position = hexo_py.Position.replay(moves)
    base = model(collate([from_position(base_position)]), 0.2)
    policy = dict(zip(base_position.legal_moves(), base.policy_logits.tolist()))
    values = dict(zip(base_position.legal_moves(), base.q_values.tolist()))
    for transform in d6_transforms():
        position = hexo_py.Position.replay([transform(move) for move in moves])
        output = model(collate([from_position(position)]), 0.2)
        transformed_policy = dict(zip(position.legal_moves(), output.policy_logits.tolist()))
        transformed_values = dict(zip(position.legal_moves(), output.q_values.tolist()))
        for move in policy:
            assert abs(transformed_policy[transform(move)] - policy[move]) <= 1e-5
            assert abs(transformed_values[transform(move)] - values[move]) <= 1e-5
        assert torch.allclose(output.value, base.value, atol=1e-5)


def test_parameter_counts_are_pinned_for_cell_node_states_and_scopes():
    assert sum(p.numel() for p in MantisNet(MantisConfig()).parameters()) == 4_537_925
    for scope in ("all", "uncovered"):
        assert sum(
            p.numel()
            for p in MantisNet(
                MantisConfig(cell_nodes=True, cell_node_scope=scope)
            ).parameters()
        ) == 5_195_909
        assert sum(
            p.numel()
            for p in MantisNet(
                MantisConfig(
                    cell_nodes=True,
                    cell_node_scope=scope,
                    cell_adjacency=True,
                )
            ).parameters()
        ) == 5_462_165


def test_cell_adjacency_is_a_separate_validated_subknob():
    try:
        MantisConfig(cell_adjacency=True)
    except ValueError as error:
        assert "requires cell_nodes=True" in str(error)
    else:
        raise AssertionError("an inert adjacency knob was accepted")
    model = MantisNet(MantisConfig(cell_nodes=True, cell_adjacency=True))
    assert sum(parameter.numel() for parameter in model.parameters()) == 5_462_165


def test_cell_node_scope_is_validated():
    assert MantisConfig().cell_node_scope == "all"
    with pytest.raises(ValueError) as error:
        MantisConfig(cell_node_scope="uncovered")
    assert "cell_node_scope='uncovered' requires cell_nodes=True" in str(error.value)

    with pytest.raises(ValueError) as error:
        MantisConfig(cell_nodes=True, cell_node_scope="nearby")
    message = str(error.value)
    assert "cell_node_scope must be one of" in message
    assert "'all'" in message
    assert "'uncovered'" in message
