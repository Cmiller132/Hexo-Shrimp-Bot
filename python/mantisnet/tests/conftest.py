"""Shared fixtures: real engine positions and the D6 transform group.

Every position here comes from the engine by replay — never from a
board-shaped constructor — so what the builder is tested against is what the
rules actually produce.
"""

from __future__ import annotations

import random

import hexo_py
import pytest
import torch

from mantisnet import MantisConfig, MantisNet
from mantisnet.klent import telemetry

# Ply depths for the shared position set: both movers, all three turn phases,
# both stones of a turn, and boards from empty to crowded.
PLIES = [0, 1, 2, 3, 5, 9, 12, 21, 34, 60]


def random_moves(plies: int, seed: int) -> list[tuple[int, int]]:
    """A uniformly random legal playout of exactly `plies` placements that
    does not end the game, retrying seeds where random play won by accident."""
    for attempt in range(100):
        rng = random.Random(seed * 1_000_003 + attempt * 1_009 + plies)
        pos = hexo_py.Position()
        moves: list[tuple[int, int]] = []
        for _ in range(plies):
            pos.advance(*(m := rng.choice(pos.legal_moves())))
            moves.append(m)
        if not pos.is_terminal:
            return moves
    raise AssertionError(f"no non-terminal {plies}-ply playout in 100 seeds")


@pytest.fixture(scope="session")
def move_lists() -> list[list[tuple[int, int]]]:
    return [random_moves(p, seed=7) for p in PLIES] + [random_moves(45, seed=1234)]


@pytest.fixture(scope="session")
def positions(move_lists) -> list[hexo_py.Position]:
    return [hexo_py.Position.replay(m) for m in move_lists]


@pytest.fixture(scope="session")
def model() -> MantisNet:
    torch.manual_seed(0)
    net = MantisNet(MantisConfig())
    net.eval()
    return net


def d6_transforms():
    """The 12 board symmetries as maps on (q, r). Index 0 is the identity.

    The same group the telemetry read layer canonicalizes openings with —
    one definition, used by the model's §12.3 invariance tests and by the
    opening atlas. The model never sees these transforms, so sharing them
    deletes no detector; the group itself is held to the rules by
    `test_telemetry.py`, which replays transformed games through the engine.
    """
    return telemetry.D6_TRANSFORMS


def oracle_live_windows(pos: hexo_py.Position) -> dict[tuple[int, int, int], tuple[int, int]]:
    """Live windows by the engine's own walk: identity (axis, start_q, start_r)
    to (colour relative to the mover, occupancy mask).

    This is the independent oracle of MODEL_SPEC §12.1 — the builder never
    calls `windows_through`.
    """
    mover = pos.current_player
    live: dict[tuple[int, int, int], tuple[int, int]] = {}
    for q, r, _p in pos.stones():
        for axis, sq, sr, m0, m1 in pos.windows_through(q, r):
            if (m0 > 0) == (m1 > 0):
                continue  # dead (mixed); through a stone, so never empty
            mover_mask = m0 if mover == 0 else m1
            colour = 0 if mover_mask else 1
            live[(axis, sq, sr)] = (colour, m0 | m1)
    return live
