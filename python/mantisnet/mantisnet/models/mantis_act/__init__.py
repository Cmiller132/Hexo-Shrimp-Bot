"""MantisNet-ACT v4: an exactly D6-equivariant board and action representation.

``docs/MANTIS_ACT_SPEC.md`` is normative for this package; the section marks
throughout the modules point into it. This is a second architecture alongside
MantisNet, not a revision of it: the two share the game, the engine, and the
KLENT seam, and no module at all.

What the representation is. A position becomes a graph of two persistent node
families and one derived one. *Cells* are the finite set of coordinates the
position makes relevant — every stone, every legal placement, and every cell
of a persistent window, so an empty and currently illegal intersection inside a
live shape is still a node. *Windows* are the six-cell line segments the board
actually contains: all of them that hold a stone, encoded as a ternary pattern
over their six slots quotiented by slot reversal. *Actions* are the legal
placements, each encoded from the eighteen windows a stone placed there would
join — three axes by six candidate slots — evaluated counterfactually rather
than looked up. Cells and windows are joined by an exact joint relation class
of ``(pattern, slot)``; cells are joined to each other by hex adjacency and by
displacement edges from every stone within radius twelve, typed by the 48
exact D6 orbits of that displacement.

How that differs from MantisNet at the representation level:

- MantisNet's nodes are stones and *live* windows — one colour only. A window
  holding both colours is dead for scoring and is dropped, so a blocked line
  and an empty line look alike. Here mixed windows are ordinary nodes with
  their own pattern class and status, and empty cells inside windows are nodes
  too, which is what lets an intersection or a fork be one shared entity rather
  than an inference from two window states.
- MantisNet types its geometry by hex distance plus an on-axis flag. Here a
  displacement carries the exact class of its D6 orbit, of which there are 48
  through radius twelve, so two shapes that a distance bucket cannot tell apart
  are not aliased into one relation.
- MantisNet describes a legal cell by the live windows through it and, failing
  that, by its distance to the nearest stone. Here every legal cell is
  described by what placing a stone there would produce, in all eighteen
  windows, so a cell in no current window is encoded by its own future rather
  than by a background alias.
- Line direction is carried by three axis-equivariant channels that permute
  under the group, never by per-axis parameters, and nothing in the model sees
  an absolute coordinate, an absolute axis id, or a move number.

The builder is exact and vectorised, calls the engine only to read a position,
and refuses terminal states. Class counts are asserted at import: a table with
the wrong number of orbits would train a silently aliased embedding, which no
round-trip test can see.
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
    MantisACTConfig,
    architecture_hash,
    summarise,
)
from .packed import (
    PHASE_FIRST,
    PHASE_OPENING,
    PHASE_SECOND,
    POST_ACTION_ROWS,
    WINDOW_LEN,
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
from .symmetry import D6_ORBITS_DMAX12

__all__ = [
    # Configuration and its named arms (§6, §29).
    "ARCHITECTURE_ID",
    "MANTIS_ACT_REPR_VERSION",
    "MantisACTConfig",
    "PRESETS",
    "ENUM_VOCABULARIES",
    "architecture_hash",
    "summarise",
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
