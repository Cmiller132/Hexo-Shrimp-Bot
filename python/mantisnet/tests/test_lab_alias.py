"""The Step 4 structural alias diagnostic."""

from __future__ import annotations

import pytest

import hexo_py
from mantisnet.lab.alias import alias_report

_GAMES = [
    [(0, 0), (1, 0), (2, 0)],
    [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (0, 3), (0, 4), (3, 0)],
    [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, -1), (0, 2), (4, -1), (-1, 2)],
]


def test_single_stone_position_shows_the_background_split_and_merge():
    """One stone: 216 legal cells, of which the 30 colinear-within-5 cells are
    window-covered and the other 186 are background. The incumbent inputs
    split the background by nearest-stone bucket; the action-row inputs carry
    all-EMPTY rows for every background cell, merging them into one group."""
    pos = hexo_py.Position.replay([(0, 0)])
    report = alias_report([pos])
    assert report["positions"] == 1
    assert report["before"]["legal_actions"] == 216
    assert report["after"]["legal_actions"] == 216

    assert report["before"]["groups_with_background_cells"] >= 3
    assert report["before"]["max_alias_group"] < 186
    assert report["after"]["max_alias_group"] == 186
    assert report["after"]["groups_with_background_cells"] == 1


def test_alias_counts_are_d6_invariant():
    """The report is a pure function of builder structure, so a transformed
    board reports identical tallies."""
    from mantisnet.klent import telemetry

    transform = telemetry.D6_TRANSFORMS[1]
    for moves in _GAMES:
        base = alias_report([hexo_py.Position.replay(moves)])
        turned = alias_report(
            [hexo_py.Position.replay([transform(m) for m in moves])]
        )
        assert base["before"] == turned["before"]
        assert base["after"] == turned["after"]


def test_alias_report_refuses_no_positions():
    with pytest.raises(ValueError, match="at least one position"):
        alias_report([])
