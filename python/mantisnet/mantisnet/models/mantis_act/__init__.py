"""MantisNet-ACT v4: an exactly D6-equivariant board and action representation.

A position becomes a graph of cells, windows, and legal actions. Cells are
every stone, every legal placement, and every cell of a persistent window.
Windows are the six-cell line segments holding a stone, encoded as a ternary
pattern quotiented by slot reversal. Actions are the legal placements, each
encoded counterfactually from the eighteen windows a stone placed there would
join. ``docs/MANTIS_ACT_SPEC.md`` is normative; the section marks throughout
the modules point into it.
"""

from __future__ import annotations

from .builder import (
    GLOBAL_NUMERIC_FEATURES,
    GLOBAL_NUMERIC_NAMES,
    build,
    build_from_arrays,
    collate_positions,
    collate_prefixes,
)
from .config import (
    ARCHITECTURE_ID,
    ENUM_VOCABULARIES,
    MANTIS_ACT_REPR_VERSION,
    PRESETS,
    UNHASHED_FIELDS,
    MantisACTConfig,
    architecture_hash,
    summarise,
)
from .model import ACT_CHECKPOINT_FORMAT, ACTOutput, MantisACT, config_from_record
from .packed import (
    ACT_GRAPH_CELL_BUDGET,
    PHASE_FIRST,
    PHASE_OPENING,
    PHASE_SECOND,
    POST_ACTION_ROWS,
    WINDOW_LEN,
    ACTChunkCost,
    ACTGraph,
    PackedACTBatch,
    collate,
    telemetry,
)
from .pattern_classes import (
    ALL_CELL_WINDOW_REL_CLASSES,
    ALL_WINDOW_PATTERN_CLASSES,
    EMPTY,
    MIXED,
    NONEMPTY_CELL_WINDOW_REL_CLASSES,
    NONEMPTY_WINDOW_PATTERN_CLASSES,
    OPP_LIVE,
    OWN_LIVE,
    POST1_REL_CLASSES,
    TERNARY_CODES,
)
from .summary import ParameterSummary, parameter_summary
from .symmetry import D6_ORBITS_DMAX12

__all__ = [
    # Configuration and its named arms (§6, §29).
    "ARCHITECTURE_ID",
    "MANTIS_ACT_REPR_VERSION",
    "MantisACTConfig",
    "PRESETS",
    "ENUM_VOCABULARIES",
    "UNHASHED_FIELDS",
    "architecture_hash",
    "summarise",
    # The assembled model, its output, and its checkpoint identity (§25, §28).
    # The three stages it composes stay in their own modules: `state_trunk`,
    # `action_encoder`, and `heads`.
    "MantisACT",
    "ACTOutput",
    "ACT_CHECKPOINT_FORMAT",
    "config_from_record",
    # Parameters by subsystem (§6, §32, §34).
    "ParameterSummary",
    "parameter_summary",
    # Builder entry points (§7, §26).
    "build",
    "build_from_arrays",
    "collate_positions",
    "collate_prefixes",
    "GLOBAL_NUMERIC_NAMES",
    "GLOBAL_NUMERIC_FEATURES",
    # Containers and their plumbing (§25, §26, §34).
    "ACTGraph",
    "PackedACTBatch",
    "collate",
    "telemetry",
    # The §26 packer limit and the law that reads it.
    "ACTChunkCost",
    "ACT_GRAPH_CELL_BUDGET",
    "WINDOW_LEN",
    "POST_ACTION_ROWS",
    "PHASE_OPENING",
    "PHASE_FIRST",
    "PHASE_SECOND",
    # Asserted class counts (§9.2, §10.1, §11.1, §19.2) and the statuses they
    # go with. Each is checked against its construction at import.
    "TERNARY_CODES",
    "ALL_WINDOW_PATTERN_CLASSES",
    "NONEMPTY_WINDOW_PATTERN_CLASSES",
    "ALL_CELL_WINDOW_REL_CLASSES",
    "NONEMPTY_CELL_WINDOW_REL_CLASSES",
    "POST1_REL_CLASSES",
    "D6_ORBITS_DMAX12",
    "EMPTY",
    "OWN_LIVE",
    "OPP_LIVE",
    "MIXED",
]
