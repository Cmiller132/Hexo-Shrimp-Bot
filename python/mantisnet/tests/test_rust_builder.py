"""The Rust batch builder against the Python builder, field for field.

The Python builder is the reference implementation. Every integral tensor
returned by ``hexo_py.build_batch`` must equal its Python-builder counterpart
(MODEL_SPEC §12.7).
"""

from __future__ import annotations

import hexo_py
import pytest
import torch

from mantisnet import collate, collate_positions, collate_prefixes, from_position
from mantisnet.builder import batch_from_arrays
from mantisnet.lab.check import _assert_pairs_equal

from .test_klent_returns import FIRST_STONE_WIN

_TENSOR_FIELDS = [
    "stone_own",
    "window_feat",
    "window_id",
    "moves_idx",
    "inc_stone",
    "inc_window",
    "inc_class",
    "stone_slot",
    "coords",
    "attn_valid",
    "window_slot",
    "value_valid",
    "legal_offsets",
    "cell_pos",
    "dec_cell",
    "dec_window",
    "dec_class",
    "bg_cell",
    "bg_bucket",
]


def _assert_equal(rust, python):
    assert (rust.n_pos, rust.max_t, rust.max_w, rust.n_cells) == (
        python.n_pos,
        python.max_t,
        python.max_w,
        python.n_cells,
    )
    for name in _TENSOR_FIELDS:
        a, b = getattr(rust, name), getattr(python, name)
        assert a.dtype == b.dtype, f"{name}: {a.dtype} != {b.dtype}"
        assert torch.equal(a, b), f"{name} differs"


def test_position_batch_parity(positions):
    _assert_equal(
        collate_positions(positions), collate([from_position(p) for p in positions])
    )


def test_single_position_batches_including_ply_zero(positions):
    for pos in [hexo_py.Position(), positions[-1]]:
        _assert_equal(collate_positions([pos]), collate([from_position(pos)]))


def test_prefix_batch_parity(move_lists):
    games = [m for m in move_lists if m]
    ts = [len(m) for m in games] + [1, len(games[-1]) // 2]
    games = games + [games[-1], games[-1]]
    rust = collate_prefixes(games, ts)
    python = collate(
        [from_position(hexo_py.Position.replay(list(g[:t]))) for g, t in zip(games, ts)]
    )
    _assert_equal(rust, python)


def test_pair_table_parity(positions, move_lists):
    """§5.1c: the same run boundaries, and the same edge set inside each run.

    The Rust builder joins each position on its own and offsets the result; the
    Python oracle runs one join over the whole batch. Edge order inside a run is
    therefore each derivation's own, and the sets are the contract.
    """
    _assert_pairs_equal(
        collate_positions(positions, pairs=True),
        collate([from_position(p) for p in positions], pairs=True),
    )
    # Ply zero has no windows at all, and a single position exercises the
    # collation with one window base rather than several.
    for pos in [hexo_py.Position(), positions[-1]]:
        _assert_pairs_equal(
            collate_positions([pos], pairs=True),
            collate([from_position(pos)], pairs=True),
        )
    games = [m for m in move_lists if m]
    ts = [len(m) for m in games]
    _assert_pairs_equal(
        collate_prefixes(games, ts, pairs=True),
        collate(
            [from_position(hexo_py.Position.replay(list(g))) for g in games], pairs=True
        ),
    )


def test_pair_tables_stay_opt_in(positions):
    plain = collate_positions(positions[:3])
    assert plain.wa_ptr is None and plain.wa_cedge is None
    arrays = hexo_py.build_batch(positions[:3])
    assert [name for name in arrays if name.startswith("wa_")] == []
    # The flag describes the arrays; neither direction is quietly repaired.
    with pytest.raises(ValueError, match="are missing"):
        batch_from_arrays(pairs=True, **arrays)
    with pytest.raises(ValueError, match="without pairs=True"):
        batch_from_arrays(**hexo_py.build_batch(positions[:3], pairs=True))


def test_terminal_position_refused():
    pos = hexo_py.Position.replay(FIRST_STONE_WIN)
    assert pos.is_terminal
    with pytest.raises(ValueError, match="terminal"):
        collate_positions([pos])
    with pytest.raises(ValueError, match="terminal"):
        collate_prefixes([FIRST_STONE_WIN], [len(FIRST_STONE_WIN)])


def test_bad_prefix_refused():
    with pytest.raises(ValueError, match="exceeds"):
        collate_prefixes([[(0, 0)]], [5])
