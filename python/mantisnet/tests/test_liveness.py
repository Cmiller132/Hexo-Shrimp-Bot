"""§12.5 liveness: blocking is absence, and dead windows have no contents.

The dense mixed-ball fixture enters through the builder's raw-array API,
whose caller owns position legality.
"""

from __future__ import annotations

import numpy as np

from mantisnet import build, from_position
from mantisnet.builder import AXES

from .conftest import oracle_live_windows


def test_blocking_removes_the_window(positions):
    killed = 0
    for pos in positions:
        g = from_position(pos)
        live_ids = {tuple(w) for w in g.window_id}
        legal = set(pos.legal_moves())
        # A live window of the opponent's colour with a legal empty slot: the
        # mover's stone there makes it mixed, and mixed windows are not
        # entities.
        for (axis, sq, sr), (colour, occ) in oracle_live_windows(pos).items():
            if colour != 1:
                continue
            for k in range(6):
                if occ >> k & 1:
                    continue
                cell = (sq + k * int(AXES[axis, 0]), sr + k * int(AXES[axis, 1]))
                if cell not in legal:
                    continue
                after = pos.copy()
                after.advance(*cell)
                if after.is_terminal:
                    continue
                g2 = from_position(after)
                assert (axis, sq, sr) in live_ids
                assert (axis, sq, sr) not in {tuple(w) for w in g2.window_id}
                killed += 1
                break
            else:
                continue
            break
    assert killed >= 3, "the position set never exercised a block"


def _mixed_ball(radius: int) -> tuple[np.ndarray, np.ndarray]:
    """A filled hex ball whose every fully-interior window is mixed: colour by
    (q - r) mod 3, which cycles through all residues along all three axes, so
    any six consecutive cells — even with one removed — hold both colours."""
    cells = [
        (q, r)
        for q in range(-radius, radius + 1)
        for r in range(-radius, radius + 1)
        if max(abs(q), abs(r), abs(q + r)) <= radius
    ]
    qr = np.array(cells, dtype=np.int64)
    owner = ((qr[:, 0] - qr[:, 1]) % 3 != 0).astype(np.int64)
    return qr, owner


def test_dead_window_contents_do_not_reach_the_graph():
    # The pocket (0,0) sits 6 away from the ball edge, so all 18 windows
    # through it lie fully inside the ball: every one is dead whether the
    # pocket is empty or filled, and the window entity set must not move.
    qr, owner = _mixed_ball(6)
    pocket = (qr[:, 0] == 0) & (qr[:, 1] == 0)
    without = build(qr[~pocket], owner[~pocket], mover=0, legal_qr=[(0, 0)], moves_remaining=1)
    filled = owner.copy()
    filled[pocket] = 0
    with_stone = build(qr, filled, mover=0, legal_qr=[(7, 0)], moves_remaining=1)

    ids_a = {tuple(w): int(f) for w, f in zip(without.window_id, without.window_feat)}
    ids_b = {tuple(w): int(f) for w, f in zip(with_stone.window_id, with_stone.window_feat)}
    assert ids_a == ids_b
    assert ids_a, "the ball's rim should still produce live windows"
    # The pocket's own 18 windows are all dead in both graphs — without this
    # the equality above could hold vacuously on the wrong geometry.
    for axis in range(3):
        for k in range(6):
            wid = (axis, -k * int(AXES[axis, 0]), -k * int(AXES[axis, 1]))
            assert wid not in ids_a
