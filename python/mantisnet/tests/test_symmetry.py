"""§12.3 / §8: exact D6 invariance on engine positions.

Value equal and per-move policy equal through the coordinate transform, for
all 11 non-identity symmetries, to atol 1e-5 — exact in exact arithmetic;
node reordering changes summation order, so bit-equality is not required.
"""

from __future__ import annotations

import hexo_py
import torch

from mantisnet import collate, from_position

from .conftest import d6_transforms


@torch.no_grad()
def _forward(model, pos):
    return model(collate([from_position(pos)]), 0.2)


def test_d6_invariance(model, move_lists):
    transforms = d6_transforms()
    for moves in move_lists:
        base = hexo_py.Position.replay(moves)
        out = _forward(model, base)
        legal = base.legal_moves()
        by_move = dict(zip(legal, out.policy_logits.tolist()))
        q_by_move = dict(zip(legal, out.q_values.tolist()))

        for t in transforms[1:]:
            tpos = hexo_py.Position.replay([t(m) for m in moves])
            tout = _forward(model, tpos)

            assert torch.allclose(tout.value, out.value, atol=1e-5)
            assert torch.allclose(tout.value_dist, out.value_dist, atol=1e-5)

            tlegal = tpos.legal_moves()
            t_by_move = dict(zip(tlegal, tout.policy_logits.tolist()))
            t_q_by_move = dict(zip(tlegal, tout.q_values.tolist()))
            assert set(t_by_move) == {t(m) for m in by_move}
            for m, logit in by_move.items():
                assert abs(t_by_move[t(m)] - logit) <= 1e-5
                assert abs(t_q_by_move[t(m)] - q_by_move[m]) <= 1e-5


def test_transform_set_is_a_group_of_twelve(move_lists):
    # 12 distinct maps, each a bijection fixing the origin and preserving the
    # legal structure (replay would raise otherwise, and counts must agree).
    transforms = d6_transforms()
    probe = [(1, 0), (0, 1), (1, -1), (3, -2)]
    images = {tuple(t(m) for m in probe) for t in transforms}
    assert len(images) == 12
    moves = move_lists[-1]
    n = hexo_py.Position.replay(moves).legal_count
    for t in transforms:
        assert hexo_py.Position.replay([t(m) for m in moves]).legal_count == n
