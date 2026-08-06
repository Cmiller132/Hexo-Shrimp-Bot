"""Ternary window-pattern reversal classes (§9.2, §10.1, §19.2).

A six-cell line window is read in slot order ``k = 0..5`` along its native
axis, each slot holding ``0`` empty, ``1`` own, ``2`` opponent relative to the
side to move; its raw code is ``sum_k v_k * 3**k``, one of ``TERNARY_CODES``. A
board reflection reverses a window's slot order and nothing else, so the finest
description of a window the model may see is the quotient of the raw codes by
``k -> 5 - k``. Every table here is that quotient or a statistic constant on
it, built once at import from first principles, with each count the spec
asserts checked against the construction rather than assumed.

Index conventions this module fixes (each is part of the representation):

- Pattern class: the rank of the reversal orbit's representative — the orbit's
  minimum raw code — in ascending raw order, one of
  ``ALL_WINDOW_PATTERN_CLASSES``. Class ``0`` is the all-empty pattern, the one
  class the ``nonempty`` window scope never uses.
- Cell-window class: the rank of the ``(pattern, slot)`` orbit under the joint
  involution ``(p, s) -> (reverse(p), 5 - s)``, in ascending ``(code, slot)``
  order, one of ``ALL_CELL_WINDOW_REL_CLASSES``. §10.1 requires the joint
  orbit: canonicalizing the pattern separately from a coarse slot descriptor
  merges relations that differ, which is an exact alias rather than a
  compression.
- Post-placement class: the same joint orbit restricted to the pairs a
  hypothetical own placement can produce — slot ``s`` of the post-placement
  pattern holds an own stone — ranked among those pairs alone, one of
  ``POST1_REL_CLASSES``.

``POST1_CLASS[code, slot]`` is ``-1`` wherever ``slot`` does not hold an own
stone in ``code``: such a pair is not a post-placement state and has no class.
The action builder indexes the table with the pattern it obtained by writing an
own stone into ``slot``, so every index it forms is defined by construction,
and a ``-1`` reaching a relation embedding is a builder fault — the builder
must refuse it there rather than embed a wrapped-around row.

Reversal permutes a window's slots without changing any slot's value, so the
status, the three occupancy counts, and both maximum contiguous runs take one
value across an orbit. They are therefore well defined per class even though
they are tabulated per raw code; the tests check that equality exhaustively.
"""

from __future__ import annotations

import numpy as np

# Slots per window, and the slot values relative to the side to move. Value 0
# is empty and needs no name here: it is only ever the absence of the two.
_SLOTS = 6
_OWN = 1
_OPP = 2

TERNARY_CODES = 3**_SLOTS

# Spec §9.2, §10.1 and §19.2 each derive one of these counts independently, so
# a disagreement with the constructions below is a wrong construction here.
ALL_WINDOW_PATTERN_CLASSES = 378
NONEMPTY_WINDOW_PATTERN_CLASSES = 377
ALL_CELL_WINDOW_REL_CLASSES = 2187
NONEMPTY_CELL_WINDOW_REL_CLASSES = 2184
POST1_REL_CLASSES = 729

# Window status (§9.3). The ids are the two color-presence flags in binary, so
# the flags add to the status.
EMPTY, OWN_LIVE, OPP_LIVE, MIXED = 0, 1, 2, 3


def _check_count(name: str, expected: int, actual: int) -> None:
    """Refuse a table whose construction disagrees with the spec's count.

    Raising rather than asserting keeps the check alive under ``python -O``: a
    table that reaches a builder with the wrong number of classes trains a
    silently aliased embedding, which no downstream test can see.
    """
    if actual != expected:
        raise AssertionError(
            f"{name} came out {actual}, not the {expected} the spec derives "
            "independently; the construction above is wrong"
        )


def _orbit_ranks(partner: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rank the orbits of an involution by their representatives' raw order.

    ``partner`` sends each item's index to the index reversal maps it to, and
    ``valid`` selects the subset being classed. An orbit's representative is
    its minimum index, and its class is that representative's rank among the
    representatives — the numbering ``mantisnet/builder.py::_orbit_classes``
    gives the binary case, since ascending order reaches every orbit at its
    representative first. Items outside ``valid`` get ``-1``.

    Both preconditions are checked rather than documented: an index array that
    is not an involution, or a subset the involution leaves, would silently
    produce a table that maps two distinct relations onto one class.
    """
    items = np.arange(len(partner), dtype=np.int64)
    if not np.array_equal(partner[partner], items):
        bad = int(np.argmax(partner[partner] != items))
        raise ValueError(f"partner is not an involution: item {bad} does not return")
    if not np.array_equal(valid[partner], valid):
        bad = int(np.argmax(valid[partner] != valid))
        raise ValueError(f"the classed subset is not closed under the involution at item {bad}")
    canon = np.minimum(items, partner)
    representative = np.unique(canon[valid])
    ranks = np.full(len(partner), -1, dtype=np.int64)
    ranks[valid] = np.searchsorted(representative, canon[valid])
    return ranks, representative


def _max_contiguous_run(present: np.ndarray) -> np.ndarray:
    """Longest run of consecutive true slots in each row of an (n, 6) mask."""
    run = np.zeros(len(present), dtype=np.int64)
    best = np.zeros(len(present), dtype=np.int64)
    for k in range(_SLOTS):
        run = np.where(present[:, k], run + 1, 0)
        best = np.maximum(best, run)
    return best


_CODES = np.arange(TERNARY_CODES, dtype=np.int64)
_POWERS = 3 ** np.arange(_SLOTS, dtype=np.int64)
# (729, 6) slot values: slot k of a code sits in its 3**k place.
_DIGITS = (_CODES[:, None] // _POWERS[None, :]) % 3

# Reversal writes the value of slot 5 - k into the 3**k place.
REVERSE_CODE = (_DIGITS[:, ::-1] * _POWERS[None, :]).sum(axis=1, dtype=np.int64)

PATTERN_CLASS, CLASS_REPRESENTATIVE = _orbit_ranks(
    REVERSE_CODE, np.ones(TERNARY_CODES, dtype=bool)
)
# Code 0 is the only stoneless code, and it is its own reverse, so exactly one
# class is empty (§9.2) — the class the `nonempty` window scope never emits.
CLASS_IS_EMPTY = CLASS_REPRESENTATIVE == 0

# The joint (pattern, slot) involution of §10.1, over pairs indexed
# `code * 6 + slot` so that index order is ascending `(code, slot)` order.
_PAIRS = TERNARY_CODES * _SLOTS
_PAIR_CODE = np.repeat(_CODES, _SLOTS)
_PAIR_SLOT = np.tile(np.arange(_SLOTS, dtype=np.int64), TERNARY_CODES)
_PAIR_PARTNER = REVERSE_CODE[_PAIR_CODE] * _SLOTS + (_SLOTS - 1 - _PAIR_SLOT)

CELL_WINDOW_CLASS = _orbit_ranks(_PAIR_PARTNER, np.ones(_PAIRS, dtype=bool))[0].reshape(
    TERNARY_CODES, _SLOTS
)

# Post-placement pairs (§19.2): the candidate slot holds the own stone just
# written there, so only those pairs are classed. The subset is closed under
# the involution — reversal carries an own stone at slot k to slot 5 - k — and
# the involution has no fixed point, since no slot is its own mirror.
_POST1_VALID = (_DIGITS == _OWN).reshape(-1)
POST1_CLASS = _orbit_ranks(_PAIR_PARTNER, _POST1_VALID)[0].reshape(TERNARY_CODES, _SLOTS)

OWN_COUNT = (_DIGITS == _OWN).sum(axis=1, dtype=np.int64)
OPP_COUNT = (_DIGITS == _OPP).sum(axis=1, dtype=np.int64)
EMPTY_COUNT = _SLOTS - OWN_COUNT - OPP_COUNT
STATUS = (OWN_COUNT > 0).astype(np.int64) * OWN_LIVE + (OPP_COUNT > 0).astype(
    np.int64
) * OPP_LIVE
OWN_MAX_RUN = _max_contiguous_run(_DIGITS == _OWN)
OPP_MAX_RUN = _max_contiguous_run(_DIGITS == _OPP)

# The tables are module state that embedding rows are indexed by, so an
# in-place write anywhere would silently repartition a trained embedding with
# nothing able to observe it. They are frozen at import; a consumer that needs
# to modify one takes a copy.
for _table in (
    REVERSE_CODE,
    PATTERN_CLASS,
    CLASS_REPRESENTATIVE,
    CLASS_IS_EMPTY,
    CELL_WINDOW_CLASS,
    POST1_CLASS,
    STATUS,
    OWN_COUNT,
    OPP_COUNT,
    EMPTY_COUNT,
    OWN_MAX_RUN,
    OPP_MAX_RUN,
):
    _table.setflags(write=False)
del _table

# The nonempty counts are taken over codes 1..728: code 0 is the all-empty
# pattern, and its orbits are disjoint from every other code's because
# reversal preserves occupancy.
_check_count(
    "ALL_WINDOW_PATTERN_CLASSES", ALL_WINDOW_PATTERN_CLASSES, len(CLASS_REPRESENTATIVE)
)
_check_count(
    "NONEMPTY_WINDOW_PATTERN_CLASSES",
    NONEMPTY_WINDOW_PATTERN_CLASSES,
    len(np.unique(PATTERN_CLASS[1:])),
)
_check_count(
    "ALL_CELL_WINDOW_REL_CLASSES",
    ALL_CELL_WINDOW_REL_CLASSES,
    int(CELL_WINDOW_CLASS.max()) + 1,
)
_check_count(
    "NONEMPTY_CELL_WINDOW_REL_CLASSES",
    NONEMPTY_CELL_WINDOW_REL_CLASSES,
    len(np.unique(CELL_WINDOW_CLASS[1:])),
)
_check_count("POST1_REL_CLASSES", POST1_REL_CLASSES, int(POST1_CLASS.max()) + 1)
