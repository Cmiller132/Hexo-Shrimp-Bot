"""Exhaustive parity across the singular and packed Rust ACT boundaries.

The public singular builder must preserve the validated ``ACTGraph`` contract,
and the production batch builder must match Python collation of those graphs
field for field.  Its Rust-built execution plans must also match the retained
Python ``build_plans`` oracle recursively, including every CSR ordering and
class block partition. Real self-play prefixes matter here: random playouts
have a very different cell and edge density from the fitting path.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path

import hexo_py
import numpy as np
import pytest
import torch

from mantisnet.models.mantis_act import (
    PRESETS,
    ACTGraph,
    MantisACTConfig,
    PackedACTBatch,
    build,
    collate,
    collate_prefixes,
)


REAL_GAMES = Path(__file__).resolve().parents[2] / "scratch" / "real_games.json"
REAL_POSITION_COUNT = 64
MAX_PLY = 200
FLOAT_RTOL = 1e-7
FLOAT_ATOL = 1e-8

# Pinned rather than discovered from the dataclass, so adding an ACTGraph field
# makes this harness fail until the new field has a stated parity comparison.
GRAPH_ARRAY_FIELDS = (
    "cell_qr",
    "cell_occupancy",
    "cell_is_legal",
    "cell_is_occupied",
    "cell_nearest_bucket",
    "legal_to_cell_index",
    "window_id",
    "window_pattern_class",
    "window_status",
    "window_axis",
    "window_numeric",
    "window_cell_index",
    "window_incidence_class",
    "window_incidence_mask",
    "adjacency_src",
    "adjacency_dst",
    "adjacency_axis",
    "radius_src",
    "radius_dst",
    "radius_orbit",
    "radius_axis_or_neg1",
    "action_window_index",
    "action_post1_class",
    "action_pre_status",
    "action_tactical_numeric",
    "global_numeric",
)
GRAPH_SCALAR_FIELDS = ("moves_remaining", "phase_id")

RUST_CONFIG_FIELDS = (
    "window_scope",
    "cell_scope",
    "d6_relation_mode",
    "d_max",
    "occupied_radius",
    "use_cell_adjacency",
    "use_occupied_radius_edges",
    "use_global_numeric_features",
    "use_window_numeric_features",
    "use_action_tactical_features",
)

FULL = PRESETS["full_act_v4"]
BUILDER_CONFIGS: tuple[tuple[str, MantisACTConfig], ...] = (
    ("full_act_v4", FULL),
    ("full_live_windows", PRESETS["full_live_windows"]),
    (
        "full_action_relevant_windows",
        PRESETS["full_action_relevant_windows"],
    ),
    ("full_coarse_geometry", PRESETS["full_coarse_geometry"]),
    ("full_radius6", PRESETS["full_radius6"]),
    ("full_occupied_cells_only", PRESETS["full_occupied_cells_only"]),
    ("full_no_tactical_inputs", PRESETS["full_no_tactical_inputs"]),
    (
        "occupied_and_legal",
        replace(FULL, cell_scope="occupied_and_legal"),
    ),
    (
        "all_builder_toggles_off",
        replace(
            FULL,
            use_cell_adjacency=False,
            use_occupied_radius_edges=False,
            use_global_numeric_features=False,
            use_window_numeric_features=False,
            use_action_tactical_features=False,
        ),
    ),
)

# These switches alter only the execution plans, not the ACT graph arrays the
# original builder matrix covers.  Keep them separate so all nineteen
# pre-fusion detector cases remain intact while the new Rust plan boundary is
# forced through both optional-plan branches.
PLAN_ONLY_CONFIGS: tuple[tuple[str, MantisACTConfig], ...] = (
    ("full_no_axis", PRESETS["full_no_axis"]),
    (
        "radius_axis_routing_off",
        replace(FULL, route_on_axis_radius_messages=False),
    ),
)


@dataclass(frozen=True)
class RealCase:
    """One deterministic prefix of one cached stack-939 self-play game."""

    game_index: int
    game: tuple[tuple[int, int], ...]
    ply: int
    position: object


@pytest.fixture(scope="module")
def real_cases() -> tuple[RealCase, ...]:
    raw_games = json.loads(REAL_GAMES.read_text(encoding="utf-8"))
    if len(raw_games) < REAL_POSITION_COUNT:
        raise RuntimeError(
            f"{REAL_GAMES} has {len(raw_games)} games, fewer than the "
            f"{REAL_POSITION_COUNT} parity cases"
        )

    cases = []
    for index in range(REAL_POSITION_COUNT):
        # Coprime with the 512-game cache size, so the first 64 cases use 64
        # distinct games.  Integer interpolation includes both ply 1 and 200.
        game_index = (17 * index) % len(raw_games)
        game = tuple(tuple(int(value) for value in move) for move in raw_games[game_index])
        ply = 1 + index * (MAX_PLY - 1) // (REAL_POSITION_COUNT - 1)
        if len(game) <= ply:
            raise RuntimeError(
                f"real game {game_index} has {len(game)} moves, so ply {ply} "
                "is not a nonterminal prefix"
            )
        position = hexo_py.Position.replay(game[:ply])
        if position.is_terminal:
            raise RuntimeError(
                f"real game {game_index} is already terminal at prefix {ply}"
            )
        cases.append(RealCase(game_index, game, ply, position))
    return tuple(cases)


def _direct_rust_graph(position, cfg: MantisACTConfig) -> ACTGraph:
    """Construct the public container directly from the singular PyO3 result."""
    config = {name: getattr(cfg, name) for name in RUST_CONFIG_FIELDS}
    return ACTGraph(**hexo_py.build_act_graph(position, config))


def _max_differences(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    max_abs = float(difference.max())
    nonzero = expected != 0
    max_rel = (
        float((difference[nonzero] / np.abs(expected[nonzero])).max())
        if bool(nonzero.any())
        else 0.0
    )
    return max_abs, max_rel


def _compare_array(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    context: str,
    field: str,
) -> tuple[float, float] | None:
    if not isinstance(actual, np.ndarray):
        raise AssertionError(f"{context}.{field}: Rust value is not an ndarray")
    if actual.dtype != expected.dtype:
        raise AssertionError(
            f"{context}.{field}: actual dtype {actual.dtype} != expected {expected.dtype}"
        )
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{context}.{field}: actual shape {actual.shape} != expected {expected.shape}"
        )
    if np.array_equal(actual, expected):
        return None

    if not np.issubdtype(expected.dtype, np.floating):
        np.testing.assert_array_equal(
            actual,
            expected,
            err_msg=f"{context}.{field} differs between actual and expected",
        )
        raise AssertionError("unreachable")

    max_abs, max_rel = _max_differences(actual, expected)
    try:
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=FLOAT_RTOL,
            atol=FLOAT_ATOL,
            equal_nan=False,
            err_msg=(
                f"{context}.{field}: max_abs={max_abs:.9g}, "
                f"max_rel={max_rel:.9g}"
            ),
        )
    except AssertionError as error:
        raise AssertionError(
            f"{context}.{field} exceeds float parity tolerance: "
            f"max_abs={max_abs:.9g}, max_rel={max_rel:.9g}"
        ) from error
    return max_abs, max_rel


def _record_difference(
    differences: dict[str, tuple[float, float]],
    field: str,
    measured: tuple[float, float] | None,
) -> None:
    if measured is None:
        return
    before = differences.get(field, (0.0, 0.0))
    differences[field] = (max(before[0], measured[0]), max(before[1], measured[1]))


def _report_float_differences(
    label: str, differences: dict[str, tuple[float, float]]
) -> None:
    if not differences:
        return
    detail = ", ".join(
        f"{field}(max_abs={absolute:.9g}, max_rel={relative:.9g})"
        for field, (absolute, relative) in sorted(differences.items())
    )
    warnings.warn(
        f"{label} has tolerated non-bit-exact float fields: {detail}",
        RuntimeWarning,
        stacklevel=2,
    )


def _compare_graphs(
    actual: ACTGraph, expected: ACTGraph, *, context: str
) -> dict[str, tuple[float, float]]:
    if not isinstance(actual, ACTGraph):
        raise AssertionError(
            f"{context}: public Rust build returned {type(actual).__name__}, not ACTGraph"
        )
    differences: dict[str, tuple[float, float]] = {}
    for field in GRAPH_ARRAY_FIELDS:
        measured = _compare_array(
            getattr(actual, field),
            getattr(expected, field),
            context=context,
            field=field,
        )
        _record_difference(differences, field, measured)
    for field in GRAPH_SCALAR_FIELDS:
        actual_value, expected_value = getattr(actual, field), getattr(expected, field)
        if actual_value != expected_value:
            raise AssertionError(
                f"{context}.{field}: actual {actual_value!r} != "
                f"expected {expected_value!r}"
            )
    return differences


def _compare_batches(
    actual: PackedACTBatch, expected: PackedACTBatch, *, context: str
) -> dict[str, tuple[float, float]]:
    if not isinstance(actual, PackedACTBatch):
        raise AssertionError(
            f"{context}: public prefix collator returned {type(actual).__name__}, "
            "not PackedACTBatch"
        )
    differences: dict[str, tuple[float, float]] = {}
    for description in fields(PackedACTBatch):
        field = description.name
        if field == "plans":
            _compare_plan_tree(
                getattr(actual, field),
                getattr(expected, field),
                context=f"{context}.plans",
            )
            continue
        rust_value, python_value = getattr(actual, field), getattr(expected, field)
        if isinstance(python_value, torch.Tensor):
            if not isinstance(rust_value, torch.Tensor):
                raise AssertionError(f"{context}.{field}: Rust value is not a tensor")
            if rust_value.device != python_value.device:
                raise AssertionError(
                    f"{context}.{field}: Rust device {rust_value.device} != "
                    f"Python {python_value.device}"
                )
            measured = _compare_array(
                rust_value.detach().cpu().numpy(),
                python_value.detach().cpu().numpy(),
                context=context,
                field=field,
            )
            _record_difference(differences, field, measured)
        elif rust_value != python_value:
            raise AssertionError(
                f"{context}.{field}: Rust {rust_value!r} != Python {python_value!r}"
            )
    return differences


def _compare_plan_tree(actual: object, expected: object, *, context: str) -> None:
    """Compare every nested execution-plan field without a hand-maintained list.

    The dataclasses in ``plans.py`` and ``MessagePlan`` are the public schema.
    Walking their declared fields makes a newly added plan artifact fail here
    automatically until Rust emits it, while tensor comparison remains exact:
    all plan arrays are integer index, pointer, ownership, or partition data.
    """

    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            raise AssertionError(
                f"{context}: Rust value is {type(actual).__name__}, not a tensor"
            )
        if actual.dtype != expected.dtype:
            raise AssertionError(
                f"{context}: Rust dtype {actual.dtype} != Python {expected.dtype}"
            )
        if actual.device != expected.device:
            raise AssertionError(
                f"{context}: Rust device {actual.device} != Python {expected.device}"
            )
        if actual.shape != expected.shape:
            raise AssertionError(
                f"{context}: Rust shape {tuple(actual.shape)} != "
                f"Python {tuple(expected.shape)}"
            )
        if not actual.is_contiguous():
            raise AssertionError(f"{context}: Rust tensor is not contiguous")
        np.testing.assert_array_equal(
            actual.detach().cpu().numpy(),
            expected.detach().cpu().numpy(),
            err_msg=f"{context} differs between Rust and Python plan builders",
        )
        return

    if is_dataclass(expected) and not isinstance(expected, type):
        if type(actual) is not type(expected):
            raise AssertionError(
                f"{context}: Rust type {type(actual).__name__} != "
                f"Python {type(expected).__name__}"
            )
        for description in fields(expected):
            _compare_plan_tree(
                getattr(actual, description.name),
                getattr(expected, description.name),
                context=f"{context}.{description.name}",
            )
        return

    if actual != expected:
        raise AssertionError(
            f"{context}: Rust {actual!r} != Python {expected!r}"
        )


def test_the_harness_names_every_act_graph_field() -> None:
    assert tuple(field.name for field in fields(ACTGraph)) == (
        *GRAPH_ARRAY_FIELDS,
        *GRAPH_SCALAR_FIELDS,
    )


@pytest.mark.parametrize(
    ("config_name", "cfg"), BUILDER_CONFIGS, ids=[name for name, _cfg in BUILDER_CONFIGS]
)
def test_public_builder_matches_singular_rust_boundary_on_real_positions(
    config_name: str,
    cfg: MantisACTConfig,
    real_cases: tuple[RealCase, ...],
) -> None:
    assert len(real_cases) >= 64
    differences: dict[str, tuple[float, float]] = {}
    for case in real_cases:
        context = f"{config_name}[game={case.game_index},ply={case.ply}]"
        public = build(case.position, cfg)
        direct = _direct_rust_graph(case.position, cfg)
        for field, measured in _compare_graphs(public, direct, context=context).items():
            _record_difference(differences, field, measured)
    _report_float_differences(config_name, differences)


@pytest.mark.parametrize(
    ("config_name", "cfg"), BUILDER_CONFIGS, ids=[name for name, _cfg in BUILDER_CONFIGS]
)
def test_rust_prefix_batch_matches_collated_python_reference(
    config_name: str,
    cfg: MantisACTConfig,
    real_cases: tuple[RealCase, ...],
) -> None:
    # `collate_prefixes` receives one already-concatenated, globally-offset
    # dictionary from Rust.  Exercise the full 64-position real-game matrix:
    # this comparison is the independent detector for Rust's offset shifting
    # and concatenation, not merely a public-wrapper smoke test.
    selected = real_cases
    rust = collate_prefixes(
        [case.game for case in selected],
        [case.ply for case in selected],
        cfg,
    )
    singular = collate([build(case.position, cfg) for case in selected], cfg)
    differences = _compare_batches(rust, singular, context=f"{config_name}[batch]")
    _report_float_differences(f"{config_name} batch", differences)


@pytest.mark.parametrize(
    ("config_name", "cfg"),
    PLAN_ONLY_CONFIGS,
    ids=[name for name, _cfg in PLAN_ONLY_CONFIGS],
)
def test_rust_plan_only_configurations_match_python_reference(
    config_name: str,
    cfg: MantisACTConfig,
    real_cases: tuple[RealCase, ...],
) -> None:
    """Cover switches that change plans while leaving graph arrays unchanged."""

    selected = real_cases
    rust = collate_prefixes(
        [case.game for case in selected],
        [case.ply for case in selected],
        cfg,
    )
    python = collate([build(case.position, cfg) for case in selected], cfg)
    differences = _compare_batches(
        rust,
        python,
        context=f"{config_name}[batch]",
    )
    _report_float_differences(f"{config_name} batch", differences)
