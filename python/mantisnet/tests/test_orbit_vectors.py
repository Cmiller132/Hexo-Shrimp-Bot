"""The `orbit_vectors` knob: the §4.1 bias's content term.

Everything here runs on CPU, through the dense reference path. The Triton
specializations that read the same per-row table live in
``test_attention_kernel.py`` behind the CUDA skip.
"""

from __future__ import annotations

import math
import random

import hexo_py
import pytest
import torch

from mantisnet import MantisConfig, MantisNet, collate, from_position
from mantisnet.attention import (
    BIAS_ROWS,
    TABLE_WIDTH,
    _attention_reference,
    _bucket_index,
    compose_row_bias_table,
    orbit_lut,
)
from mantisnet.lab.families import infer_config

_HEADS = 4

_PRODUCTION = dict(
    cell_latents=True,
    cell_nodes=True,
    cell_node_scope="all",
    window_attention=False,
)
_ARM_B = dict(_PRODUCTION, action_tactical=True)

# The shape the parity pins are measured at: small enough to write down, wide
# enough that the §5.3 attention actually mixes rows.
_PIN_CFG = dict(
    h=32, blocks=2, heads=4, value_queries=2, value_bins=5,
    policy_hidden=32, value_hidden=32,
)


def _playout(plies: int, seed: int) -> list[tuple[int, int]]:
    """A random non-terminal playout of exactly ``plies`` placements — the
    same construction ``conftest.random_moves`` uses, restated so the pinned
    inputs below do not move if the fixture set does."""
    for attempt in range(100):
        rng = random.Random(seed * 1_000_003 + attempt * 1_009 + plies)
        pos = hexo_py.Position()
        moves: list[tuple[int, int]] = []
        for _ in range(plies):
            pos.advance(*(move := rng.choice(pos.legal_moves())))
            moves.append(move)
        if not pos.is_terminal:
            return moves
    raise AssertionError(f"no non-terminal {plies}-ply playout in 100 seeds")


def _pin_batch():
    return collate(
        [
            from_position(hexo_py.Position.replay(_playout(plies, seed=7)))
            for plies in (5, 21)
        ]
    )


def _pin_model(**knobs) -> MantisNet:
    """A fixed, fully non-zero model at the pin shape. A fresh init has zero
    heads and zero bias tables, which would hide most of the attention path."""
    torch.manual_seed(11)
    net = MantisNet(MantisConfig(**_PIN_CFG, **knobs))
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.normal_(std=0.05)
    return net.eval()


def test_the_knob_is_off_by_default_and_owns_one_parameter_per_block():
    assert MantisConfig().orbit_vectors is False
    off = MantisNet(MantisConfig(h=32, blocks=2, heads=4))
    assert not any("orbit_vec" in name for name in off.state_dict())
    assert not hasattr(off.blocks[0], "orbit_vec")

    on = MantisNet(MantisConfig(h=32, blocks=2, heads=4, orbit_vectors=True))
    for index, block in enumerate(on.blocks):
        assert block.orbit_vec.shape == (4, BIAS_ROWS, 8), index
        # Zero-initialised, so the knob is inert until training moves it.
        assert torch.count_nonzero(block.orbit_vec) == 0, index
    assert sum("orbit_vec" in name for name in on.state_dict()) == 2


@pytest.mark.parametrize("knobs", [{}, _PRODUCTION, _ARM_B])
def test_the_knob_composes_with_the_other_live_knobs(knobs: dict):
    for orbit_vectors in (False, True):
        cfg = MantisConfig(h=32, blocks=1, heads=4, **knobs, orbit_vectors=orbit_vectors)
        assert cfg.orbit_vectors is orbit_vectors
        # Inferred from the presence of `orbit_vec`, not from a recorded field.
        inferred = infer_config(MantisNet(cfg).state_dict())
        assert inferred.orbit_vectors is orbit_vectors
        for field in ("window_attention", "cell_nodes", "action_tactical", "heads"):
            assert getattr(inferred, field) == getattr(cfg, field), field


def test_the_knob_round_trips_through_a_state_dict():
    for orbit_vectors in (False, True):
        cfg = MantisConfig(h=64, blocks=2, heads=2, orbit_vectors=orbit_vectors)
        assert infer_config(MantisNet(cfg).state_dict()) == cfg


def test_compose_row_bias_table_is_the_static_row_plus_query_content():
    p, t, d = 2, 5, 16
    generator = torch.Generator().manual_seed(11)
    static = torch.randn((_HEADS, BIAS_ROWS), generator=generator)
    vec = torch.randn((_HEADS, BIAS_ROWS, d), generator=generator)
    q = torch.randn((p, _HEADS, t, d), generator=generator)

    table = compose_row_bias_table(static, vec, q)
    assert table.shape == (p, _HEADS, t, BIAS_ROWS)
    # Written out per entry, off the definition rather than the batched form.
    expected = torch.empty_like(table)
    for pi in range(p):
        for hi in range(_HEADS):
            for mi in range(t):
                for b in range(BIAS_ROWS):
                    expected[pi, hi, mi, b] = (
                        static[hi, b] + (q[pi, hi, mi] * vec[hi, b]).sum()
                    )
    torch.testing.assert_close(table, expected, atol=2.0e-5, rtol=2.0e-5)

    # A zero vector is exactly the static table on every row: the content form
    # starts where the residual table stopped.
    zero = compose_row_bias_table(static, torch.zeros_like(vec), q)
    assert torch.equal(zero, static[None, :, None, :].expand(p, _HEADS, t, BIAS_ROWS))

    with pytest.raises(ValueError, match="bias_table"):
        compose_row_bias_table(static[:, :-1], vec, q)
    with pytest.raises(ValueError, match="orbit_vec"):
        compose_row_bias_table(static, vec[:, :-1], q)
    with pytest.raises(ValueError, match="orbit_vec"):
        compose_row_bias_table(static, vec[..., :-1], q)
    with pytest.raises(ValueError, match="q must have shape"):
        compose_row_bias_table(static, vec, q[0])


def _reference_inputs(p: int, t: int, d: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    q, k, v = (
        torch.randn((p, _HEADS, t, d), generator=generator) * 0.3 for _ in range(3)
    )
    coords = torch.stack(
        (
            torch.arange(t, dtype=torch.int32).remainder(7) - 3,
            torch.arange(t, dtype=torch.int32).div(7, rounding_mode="floor") - 2,
        ),
        dim=-1,
    ).expand(p, t, 2).contiguous()
    seq_lens = torch.tensor([t, max(1, t - 3)][:p] or [t], dtype=torch.int32)
    return q, k, v, coords, seq_lens


def test_reference_row_table_with_a_zero_vector_is_exactly_the_static_table():
    p, t, d = 2, 11, 8
    q, k, v, coords, seq_lens = _reference_inputs(p, t, d, seed=21)
    static = torch.randn((_HEADS, BIAS_ROWS), generator=torch.Generator().manual_seed(22))
    rows = compose_row_bias_table(static, torch.zeros(_HEADS, BIAS_ROWS, d), q)

    flat = _attention_reference(q, k, v, coords, seq_lens, static, orbit_lut("cpu"), 4)
    per_row = _attention_reference(q, k, v, coords, seq_lens, rows, orbit_lut("cpu"), 4)
    assert torch.equal(flat, per_row)


def test_reference_row_table_matches_a_hand_written_per_row_softmax():
    """Each pair must read the querying row's table, not a shared one and not
    the key's — a transposed gather is invisible to any zero-vector check."""
    p, t, d = 2, 11, 8
    q, k, v, coords, seq_lens = _reference_inputs(p, t, d, seed=31)
    generator = torch.Generator().manual_seed(32)
    rows = torch.randn((p, _HEADS, t, BIAS_ROWS), generator=generator) * 0.5

    actual = _attention_reference(q, k, v, coords, seq_lens, rows, orbit_lut("cpu"), 4)

    bucket, valid = _bucket_index(coords, seq_lens, t, orbit_lut("cpu"), 4)
    padded = torch.cat((rows, rows.new_full((p, _HEADS, t, 1), -3.0e4)), dim=3)
    scores = q @ k.transpose(-1, -2) / math.sqrt(d)
    for pi in range(p):
        for hi in range(_HEADS):
            for mi in range(t):
                for ni in range(t):
                    scores[pi, hi, mi, ni] += padded[pi, hi, mi, bucket[pi, mi, ni]]
    expected = (scores.softmax(-1) @ v).masked_fill(~valid[:, None, :, None], 0)
    torch.testing.assert_close(actual, expected, atol=2.0e-6, rtol=2.0e-6)
    assert padded.shape[-1] == TABLE_WIDTH


def test_a_query_rows_bias_moves_only_that_rows_output():
    p, t, d = 1, 9, 8
    q, k, v, coords, seq_lens = _reference_inputs(p, t, d, seed=41)
    rows = torch.zeros((p, _HEADS, t, BIAS_ROWS))
    base = _attention_reference(q, k, v, coords, seq_lens, rows, orbit_lut("cpu"), 4)

    moved = rows.clone()
    moved[0, :, 6, :] = 3.0  # one query row's whole table, every bucket
    after = _attention_reference(q, k, v, coords, seq_lens, moved, orbit_lut("cpu"), 4)
    changed = (after - base).abs().amax(dim=(1, 3))[0]
    # A constant added to every bucket shifts the row's scores uniformly and
    # cancels in its softmax, to rounding.
    assert changed[6] < 1.0e-6

    moved = rows.clone()
    moved[0, :, 6, 0] = 3.0  # a single bucket of a single query row
    after = _attention_reference(q, k, v, coords, seq_lens, moved, orbit_lut("cpu"), 4)
    changed = (after - base).abs().amax(dim=(1, 3))[0]
    assert changed[6] > 1.0e-3
    others = torch.cat((changed[:6], changed[7:]))
    assert torch.equal(others, torch.zeros_like(others))


@torch.no_grad()
def test_fresh_init_row_bias_is_the_static_table_on_every_query_row():
    """`orbit_vec` starts at zero, so a fresh block's per-row bias is exactly
    its static (distance, on-axis) + residual table repeated down the rows."""
    cfg = MantisConfig(h=32, blocks=1, heads=4, orbit_vectors=True)
    block = MantisNet(cfg).blocks[0]
    hd = cfg.h // cfg.heads

    q = torch.randn((3, cfg.heads, 7, hd))
    rows = block.attention_bias(q)
    assert rows.shape == (3, cfg.heads, 7, BIAS_ROWS)
    assert torch.equal(
        rows, block.bias_table()[None, :, None, :].expand(3, cfg.heads, 7, BIAS_ROWS)
    )
    # Off the knob the same call hands the kernel the static rows unchanged.
    off = MantisNet(MantisConfig(h=32, blocks=1, heads=4)).blocks[0]
    assert torch.equal(off.attention_bias(q), off.bias_table())

    # A nonzero vector makes the row depend on the query that asks.
    block.orbit_vec.normal_(std=0.5)
    varied = block.attention_bias(q)
    assert not torch.equal(varied[0, :, 0], varied[0, :, 1])


@torch.no_grad()
def test_the_knob_is_a_no_op_at_init():
    """With `orbit_vec` at its zero init the two models are the same function,
    bit for bit, on the same weights."""
    batch = _pin_batch()
    off = _pin_model()
    on = MantisNet(MantisConfig(**_PIN_CFG, orbit_vectors=True))
    missing, unexpected = on.load_state_dict(off.state_dict(), strict=False)
    assert not unexpected
    assert set(missing) == {f"blocks.{i}.orbit_vec" for i in range(_PIN_CFG["blocks"])}
    on.eval()

    expected = off(batch, 0.2)
    actual = on(batch, 0.2)
    for field in ("policy_logits", "q_score", "q_values", "value", "value_logits"):
        assert torch.equal(
            getattr(expected, field), getattr(actual, field)
        ), field

    # And it stops being a no-op the moment the vectors are not zero.
    for block in on.blocks:
        block.orbit_vec.normal_(std=0.5)
    assert not torch.equal(on(batch, 0.2).value_logits, expected.value_logits)


@torch.no_grad()
def test_off_knob_forward_matches_the_values_measured_before_the_knob():
    """The knob-off path is the pre-change code. These numbers were read off
    the tree at a7fb5d1 and must not move: they are the only record that
    adding the branch left the static path alone."""
    net = _pin_model()
    stones, _windows, latents, cells = net.trunk(_pin_batch())
    assert cells is None
    torch.testing.assert_close(
        stones[0, :6],
        torch.tensor(
            [-0.1042012, 0.0271382, -0.0758258, -0.2708996, -0.0194194, 0.0954184]
        ),
        atol=1.0e-6, rtol=0.0,
    )
    torch.testing.assert_close(
        stones[-1, :6],
        torch.tensor(
            [-0.0258550, 0.0136697, -0.1653055, -0.1638691, 0.0229257, 0.0972160]
        ),
        atol=1.0e-6, rtol=0.0,
    )
    torch.testing.assert_close(
        latents[1, :6],
        torch.tensor(
            [0.1025837, 0.0383512, -0.0752387, -0.1898969, 0.0070869, 0.2059588]
        ),
        atol=1.0e-6, rtol=0.0,
    )
    torch.testing.assert_close(
        net(_pin_batch(), 0.2).value_logits[0],
        torch.tensor([0.0437244, -0.0412518, 0.0192741, 0.0558072, -0.1149710]),
        atol=1.0e-6, rtol=0.0,
    )


def test_parameter_counts_pin_the_knob():
    def total(**knobs) -> int:
        return sum(p.numel() for p in MantisNet(MantisConfig(**knobs)).parameters())

    # The pins the knob must leave standing.
    assert total() == 4_804_581
    assert total(**_PRODUCTION) == 5_197_093
    assert total(**_ARM_B) == 5_215_141

    cfg = MantisConfig()
    per_block = cfg.heads * BIAS_ROWS * (cfg.h // cfg.heads)
    assert total(orbit_vectors=True) - total() == cfg.blocks * per_block == 26_112
    assert total(**_ARM_B, orbit_vectors=True) == 5_241_253
