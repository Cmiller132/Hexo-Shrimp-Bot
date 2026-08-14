"""Step 13 cell-node representation, geometry, and wiring laws (CPU)."""

from __future__ import annotations

import itertools

import hexo_py
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
        tuple(coord): (
            int(base.cell_occupancy[i]),
            int(base.cell_is_legal[i]),
            int(base.cell_nearest[i]),
        )
        for i, coord in enumerate(base.cell_qr.tolist())
    }
    base_edges = _edge_rows(base)

    for transform in d6_transforms():
        graph = from_position(hexo_py.Position.replay([transform(move) for move in moves]))
        cells = {
            tuple(coord): (
                int(graph.cell_occupancy[i]),
                int(graph.cell_is_legal[i]),
                int(graph.cell_nearest[i]),
            )
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


def test_far_cell_alias_is_removed_by_radius_edges():
    batch = collate([from_position(hexo_py.Position.replay([(0, 0)]))])
    covered = set(batch.dec_cell.tolist())
    uncovered = [cell for cell in range(batch.n_cells) if cell not in covered]
    orbit_by_cell = {
        int(dst): int(orbit)
        for dst, orbit in zip(batch.radius_dst.tolist(), batch.radius_orbit.tolist())
    }
    pair = next(
        (a, b)
        for a, b in itertools.combinations(uncovered, 2)
        if orbit_by_cell[a] != orbit_by_cell[b]
    )

    torch.manual_seed(7)
    off = MantisNet(_tiny(cell_latents=True)).eval()
    torch.manual_seed(7)
    on = MantisNet(_tiny(cell_nodes=True)).eval()
    with torch.no_grad():
        off_cells = off.trunk(batch)[3]
        on_cells = on.trunk(batch)[3]
    assert torch.equal(off_cells[pair[0]], off_cells[pair[1]])
    assert not torch.equal(on_cells[pair[0]], on_cells[pair[1]])


@torch.no_grad()
def test_cell_node_model_is_d6_invariant():
    moves = [(0, 0), (3, 0), (-2, 2), (0, 3), (1, -2), (-1, 3)]
    torch.manual_seed(11)
    model = MantisNet(_tiny(cell_nodes=True)).eval()
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


def test_parameter_counts_are_pinned_for_both_cell_node_states():
    assert sum(p.numel() for p in MantisNet(MantisConfig()).parameters()) == 4_803_813
    assert sum(
        p.numel() for p in MantisNet(MantisConfig(cell_nodes=True)).parameters()
    ) == 5_462_437


def test_cell_adjacency_is_a_separate_validated_subknob():
    try:
        MantisConfig(cell_adjacency=True)
    except ValueError as error:
        assert "requires cell_nodes=True" in str(error)
    else:
        raise AssertionError("an inert adjacency knob was accepted")
    model = MantisNet(MantisConfig(cell_nodes=True, cell_adjacency=True))
    assert sum(parameter.numel() for parameter in model.parameters()) == 5_728_693
