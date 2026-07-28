"""The generic opponent seam owns match protocol, not any one adapter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mantisnet.klent.opponents import FORFEIT, opponent_match


def choose_first(positions, _rng):
    return [position.nth_legal(0) for position in positions]


class ForfeitingChooser:
    """Plays first-legal until its second consultation, then forfeits."""

    def __init__(self):
        self.calls = 0

    def __call__(self, positions, _rng):
        self.calls += 1
        if self.calls >= 2:
            return [FORFEIT for _ in positions]
        return [position.nth_legal(0) for position in positions]


@dataclass(frozen=True)
class ForfeitOpponent:
    name: str = "forfeiter"

    @property
    def config(self):
        return {}

    def make_chooser(self, _ply_cap):
        return ForfeitingChooser()


@dataclass(frozen=True)
class FixedOpponent:
    name: str = "scripted"

    @property
    def config(self):
        return {"rank": 0}

    def make_chooser(self, _ply_cap):
        return choose_first


def test_generic_match_uses_only_the_opponent_chooser_seam():
    result, games = opponent_match(
        choose_first,
        FixedOpponent(),
        games=4,
        ply_cap=12,
        rng=np.random.default_rng(12),
    )
    assert result["opponent_name"] == "scripted"
    assert result["opponent_config"] == {"rank": 0}
    assert result["score_as_p0"] + result["score_as_p1"] == result["score"]
    assert len(games) == 4
    assert [game["seat"] for game in games] == [0, 1, 0, 1]
    assert result["forfeits"] == 0 and all(not g["forfeit"] for g in games)


def test_forfeit_scores_as_a_model_win_and_ends_the_game():
    result, games = opponent_match(
        choose_first,
        ForfeitOpponent(),
        games=2,
        ply_cap=40,
        rng=np.random.default_rng(3),
    )
    forfeited = [g for g in games if g["forfeit"]]
    assert forfeited and result["forfeits"] == len(forfeited)
    for game in forfeited:
        assert game["score"] == 1.0 and game["winner"] == game["seat"]
        assert not game["capped"]
    assert result["score"] >= len(forfeited)
