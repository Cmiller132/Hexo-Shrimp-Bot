"""The ternary class tables against pure-Python orbit oracles (§30.4–§30.7).

Every oracle here decodes a raw code into a slot tuple and reverses the tuple,
so it shares no construction with the module's numpy tables: a symmetric error
in the digit packing or the involution shows up as a count or a grouping
disagreement rather than cancelling.
"""

from __future__ import annotations

import numpy as np
import pytest

from mantisnet.models.mantis_act import pattern_classes as pc

SLOTS = 6


def slots(code: int) -> tuple[int, ...]:
    """A raw code as its six slot values, slot 0 first."""
    return tuple((code // 3**k) % 3 for k in range(SLOTS))


def code_of(values: tuple[int, ...]) -> int:
    return sum(v * 3**k for k, v in enumerate(values))


def reverse(code: int) -> int:
    return code_of(slots(code)[::-1])


ALL_CODES = range(pc.TERNARY_CODES)
# The two orbit sets, as the oracle builds them: an orbit is a frozen set of
# raw items, and the classes must be exactly these groups.
PATTERN_ORBITS = {frozenset({c, reverse(c)}) for c in ALL_CODES}
JOINT_ORBITS = {
    frozenset({(c, s), (reverse(c), SLOTS - 1 - s)}) for c in ALL_CODES for s in range(SLOTS)
}
POST1_ORBITS = {
    frozenset({(c, s), (reverse(c), SLOTS - 1 - s)})
    for c in ALL_CODES
    for s in range(SLOTS)
    if slots(c)[s] == 1
}


def test_ternary_code_space() -> None:
    assert pc.TERNARY_CODES == 729
    assert [reverse(c) for c in ALL_CODES] == list(pc.REVERSE_CODE)
    assert list(pc.REVERSE_CODE[pc.REVERSE_CODE]) == list(ALL_CODES)


def test_pattern_class_counts() -> None:
    """§30.4: 378 reversal classes, 377 of them nonempty."""
    assert len(PATTERN_ORBITS) == pc.ALL_WINDOW_PATTERN_CLASSES == 378
    # The count decomposes as the spec derives it: palindromes are fixed, the
    # rest pair up.
    palindromes = [c for c in ALL_CODES if reverse(c) == c]
    assert len(palindromes) == 27
    assert (729 - 27) // 2 + 27 == 378

    classes = set(pc.PATTERN_CLASS.tolist())
    assert classes == set(range(378))
    nonempty = set(pc.PATTERN_CLASS[1:].tolist())
    assert len(nonempty) == pc.NONEMPTY_WINDOW_PATTERN_CLASSES == 377
    assert classes - nonempty == {int(pc.PATTERN_CLASS[0])}


def test_cell_window_class_counts() -> None:
    """§30.5: 2187 joint (pattern, slot) classes, 2184 over nonempty patterns."""
    assert len(JOINT_ORBITS) == pc.ALL_CELL_WINDOW_REL_CLASSES == 2187
    # No fixed point: s == 5 - s has no integer solution, so every orbit has
    # two members and 729 * 6 halves exactly.
    assert all(len(orbit) == 2 for orbit in JOINT_ORBITS)
    assert 729 * SLOTS // 2 == 2187

    assert set(pc.CELL_WINDOW_CLASS.reshape(-1).tolist()) == set(range(2187))
    nonempty = set(pc.CELL_WINDOW_CLASS[1:].reshape(-1).tolist())
    assert len(nonempty) == pc.NONEMPTY_CELL_WINDOW_REL_CLASSES == 2184
    # The three classes the nonempty scope never uses are the all-empty
    # pattern's own orbits, and no nonempty pattern shares them.
    assert set(pc.CELL_WINDOW_CLASS[0].tolist()).isdisjoint(nonempty)
    assert len(set(pc.CELL_WINDOW_CLASS[0].tolist())) == 3


def test_post1_class_counts() -> None:
    """§30.6: 729 post-one-placement joint classes over 1458 raw pairs."""
    raw = [(c, s) for c in ALL_CODES for s in range(SLOTS) if slots(c)[s] == 1]
    assert len(raw) == SLOTS * 3**5 == 1458
    assert len(POST1_ORBITS) == pc.POST1_REL_CLASSES == 729

    classed = pc.POST1_CLASS[pc.POST1_CLASS >= 0]
    assert len(classed) == 1458
    assert set(classed.tolist()) == set(range(729))


def test_post1_is_defined_exactly_on_own_candidate_slots() -> None:
    """The -1 entries are exactly the pairs no placement can produce.

    A negative index does not raise in numpy or in an embedding lookup, it
    reads a row from the far end of the table — so the sentinel is worth
    pinning: the builder's contract is that it never forms one, and every entry
    it does form is a class.
    """
    own_here = np.array([[slots(c)[s] == 1 for s in range(SLOTS)] for c in ALL_CODES])
    assert np.array_equal(pc.POST1_CLASS >= 0, own_here)
    assert int(pc.POST1_CLASS.min()) == -1
    assert set(pc.POST1_CLASS[~own_here].tolist()) == {-1}


@pytest.mark.parametrize(
    "table, orbits, indices",
    [
        ("PATTERN_CLASS", PATTERN_ORBITS, [(c,) for c in ALL_CODES]),
        (
            "CELL_WINDOW_CLASS",
            JOINT_ORBITS,
            [(c, s) for c in ALL_CODES for s in range(SLOTS)],
        ),
        (
            "POST1_CLASS",
            POST1_ORBITS,
            [(c, s) for c in ALL_CODES for s in range(SLOTS) if slots(c)[s] == 1],
        ),
    ],
)
def test_classes_are_a_valid_quotient(table, orbits, indices) -> None:
    """§30.7: constant on each reversal orbit, distinct across orbits.

    Constancy alone would be satisfied by a table that collapses everything;
    distinctness alone by the raw code. Together they say the table *is* the
    quotient, which is what lets one embedding row stand for one relation.
    """
    lookup = getattr(pc, table)
    grouped: dict[int, set] = {}
    for index in indices:
        item = index[0] if len(index) == 1 else index
        grouped.setdefault(int(lookup[index]), set()).add(item)
    assert {frozenset(members) for members in grouped.values()} == orbits


def test_class_representative_is_the_orbit_minimum() -> None:
    for code in ALL_CODES:
        cls = int(pc.PATTERN_CLASS[code])
        assert int(pc.CLASS_REPRESENTATIVE[cls]) == min(code, reverse(code))
    # Ascending representative order is the numbering, so it is sorted and the
    # empty class comes first.
    assert list(pc.CLASS_REPRESENTATIVE) == sorted(pc.CLASS_REPRESENTATIVE)
    assert list(np.flatnonzero(pc.CLASS_IS_EMPTY)) == [int(pc.PATTERN_CLASS[0])] == [0]


def test_joint_class_is_not_a_pattern_class_and_coarse_slot_alias() -> None:
    """§10.1: the joint orbit must not factor through separate canonicalization.

    Canonicalizing the pattern on its own and folding the slot to its own
    mirror-invariant descriptor `min(s, 5 - s)` is the factored table §10.1
    forbids. It is coarser here, so it aliases distinct relations — which is
    what the joint construction exists to avoid.
    """
    factored: dict[tuple[int, int], set[int]] = {}
    for code in ALL_CODES:
        for slot in range(SLOTS):
            key = (int(pc.PATTERN_CLASS[code]), min(slot, SLOTS - 1 - slot))
            factored.setdefault(key, set()).add(int(pc.CELL_WINDOW_CLASS[code, slot]))
    assert len(factored) < pc.ALL_CELL_WINDOW_REL_CLASSES
    assert any(len(joint) > 1 for joint in factored.values())


def test_derived_statistics_match_a_slot_oracle() -> None:
    for code in ALL_CODES:
        values = slots(code)
        own = "".join("1" if v == 1 else "." for v in values).split(".")
        opp = "".join("1" if v == 2 else "." for v in values).split(".")
        assert int(pc.OWN_COUNT[code]) == values.count(1)
        assert int(pc.OPP_COUNT[code]) == values.count(2)
        assert int(pc.EMPTY_COUNT[code]) == values.count(0)
        assert int(pc.OWN_MAX_RUN[code]) == max(len(run) for run in own)
        assert int(pc.OPP_MAX_RUN[code]) == max(len(run) for run in opp)
        expected = {
            (False, False): pc.EMPTY,
            (True, False): pc.OWN_LIVE,
            (False, True): pc.OPP_LIVE,
            (True, True): pc.MIXED,
        }[(1 in values, 2 in values)]
        assert int(pc.STATUS[code]) == expected


def test_derived_statistics_are_constant_on_the_reversal_orbit() -> None:
    """What makes the per-code statistics well defined per class (§9.3)."""
    for name in ("STATUS", "OWN_COUNT", "OPP_COUNT", "EMPTY_COUNT", "OWN_MAX_RUN", "OPP_MAX_RUN"):
        table = getattr(pc, name)
        assert np.array_equal(table, table[pc.REVERSE_CODE]), name


def test_tables_have_the_declared_shapes_and_dtypes() -> None:
    for name, shape in [
        ("REVERSE_CODE", (729,)),
        ("PATTERN_CLASS", (729,)),
        ("CLASS_REPRESENTATIVE", (378,)),
        ("CELL_WINDOW_CLASS", (729, 6)),
        ("POST1_CLASS", (729, 6)),
        ("STATUS", (729,)),
        ("OWN_COUNT", (729,)),
        ("OPP_COUNT", (729,)),
        ("EMPTY_COUNT", (729,)),
        ("OWN_MAX_RUN", (729,)),
        ("OPP_MAX_RUN", (729,)),
    ]:
        table = getattr(pc, name)
        assert table.shape == shape, name
        assert table.dtype == np.int64, name
    assert pc.CLASS_IS_EMPTY.shape == (378,)
    assert pc.CLASS_IS_EMPTY.dtype == np.bool_
