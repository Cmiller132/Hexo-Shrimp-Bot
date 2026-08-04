"""Shared engine-position and D6-transform fixtures.

Every position is constructed by engine replay rather than a board-shaped
constructor.
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
    does not end the game, retrying playouts that terminate early."""
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

    The model's §12.3 invariance tests and telemetry opening atlas use this
    group. ``test_telemetry.py`` validates each transform by engine replay.
    """
    return telemetry.D6_TRANSFORMS


def reverse6(mask: int) -> int:
    """Bit-reverse a 6-bit mask, by string reversal rather than by shifts."""
    return int(f"{mask:06b}"[::-1], 2)


# The joint classes (§4.3), derived independently of the builder's tables: the
# orbits of ``(mask, slot)`` under a joint reversal, ranked by sorting the
# representatives rather than by the builder's ascending scan. Same contract,
# different derivation — including of the ranking convention the Rust encoder
# shares, which is why these are oracles and not restatements. ``occupied``
# selects the decoder's empty-slot orbits or the incidence's occupied-slot ones.
def _reversal_orbits(occupied: bool) -> list[tuple[int, int]]:
    return sorted(
        {
            min((mask, slot), (reverse6(mask), 5 - slot))
            for mask in range(1, 63)
            for slot in range(6)
            if bool(mask >> slot & 1) == occupied
        }
    )


JOINT_ORBITS = _reversal_orbits(occupied=False)
OCC_ORBITS = _reversal_orbits(occupied=True)
_JOINT_RANK = {pair: rank for rank, pair in enumerate(JOINT_ORBITS)}
_OCC_RANK = {pair: rank for rank, pair in enumerate(OCC_ORBITS)}


def joint_class(occupancy: int, slot: int) -> int:
    """The decoder class of a window occupancy and a candidate's slot in it."""
    return _JOINT_RANK[min((occupancy, slot), (reverse6(occupancy), 5 - slot))]


def occ_class(occupancy: int, slot: int) -> int:
    """The incidence class of a window occupancy and a stone's slot in it."""
    return _OCC_RANK[min((occupancy, slot), (reverse6(occupancy), 5 - slot))]


def oracle_live_windows(pos: hexo_py.Position) -> dict[tuple[int, int, int], tuple[int, int]]:
    """Live windows by the engine's own walk: identity (axis, start_q, start_r)
    to (colour relative to the mover, occupancy mask).

    The builder does not call ``windows_through``, so this is the independent
    oracle required by MODEL_SPEC §12.1.
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
