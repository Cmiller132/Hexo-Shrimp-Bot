"""The §5.1 window class term is one gather from a per-pattern table.

``_TERN_OCC_CLASS`` is constant on joint ``(pattern, slot)`` reversal
orbits, so a window's occupied-slot class multiset is a function of its
canonical pattern. The trunk therefore computes ``counts @ e_ws`` once and
gathers per window; this pins that path to the literal per-edge sum, on real
batch geometry including the empty board.
"""

from __future__ import annotations

import torch

from mantisnet.builder import collate, from_position
from mantisnet.model import MantisConfig, MantisNet


@torch.no_grad()
def test_pattern_rows_match_the_per_edge_class_sum(positions):
    torch.manual_seed(9)
    model = MantisNet(MantisConfig()).eval()
    batch = collate([from_position(p) for p in positions])
    for block in model.blocks:
        fast = (model.pattern_counts @ block.e_ws.weight).index_select(
            0, batch.window_feat
        )
        rows = block.e_ws.weight.index_select(0, batch.inc_class).float()
        literal = torch.zeros(batch.window_feat.shape[0], model.cfg.h)
        literal.index_add_(0, batch.inc_window, rows)
        torch.testing.assert_close(fast, literal, rtol=2.0e-5, atol=2.0e-5)
