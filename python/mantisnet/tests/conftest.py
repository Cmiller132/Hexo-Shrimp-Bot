"""Shared engine-position and D6-transform fixtures.

Every position is constructed by engine replay rather than a board-shaped
constructor.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import hexo_py
import pytest
import torch

from mantisnet import MantisConfig, MantisNet
from mantisnet.klent import telemetry


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "cuda_lane: tests whose file touches the GPU; run serially"
    )
    # Under xdist every worker would otherwise open a full-width OMP pool —
    # n workers × n cores threads thrash the box into being slower than the
    # serial run. Serial invocations keep torch's full width.
    if os.environ.get("PYTEST_XDIST_WORKER"):
        torch.set_num_threads(2)

# Ply depths for the shared position set: both movers, all three turn phases,
# both stones of a turn, and boards from empty to crowded.
PLIES = [0, 1, 2, 3, 5, 9, 12, 21, 34, 60]

# Files whose tests touch the GPU. The fast suite runs in two lanes,
# because even `torch.manual_seed` initializes a CUDA context and one
# context per xdist worker exhausts the card:
#   CUDA_VISIBLE_DEVICES= pytest tests/ -n auto -m "not cuda_lane"
#   pytest tests/ -m cuda_lane
# The first fans every CPU test across workers with the GPU invisible (the
# CUDA tests skip); the second runs the GPU files serially — the
# one-GPU-consumer rule. The xdist_group is defense for a plain `-n auto`
# run: all GPU files share one worker instead of thirty-two.
_CUDA_FILES = {
    "test_attention_kernel.py",
    "test_cell_latents.py",
    "test_message_passing.py",
    "test_mixed_row_sums.py",
    "test_model.py",
    "test_optim.py",
    "test_relay.py",
    "test_row_encoder.py",
    "test_window_latent_kernel.py",
    "test_window_pairs.py",
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        if Path(item.fspath).name in _CUDA_FILES:
            item.add_marker(pytest.mark.xdist_group("cuda"))
            item.add_marker(pytest.mark.cuda_lane)


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
