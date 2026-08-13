"""Step 15 knob wiring: config contracts, the cell-latent trunk stream and
decoder read, the line-pass slot, and knob-off inertness."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from mantisnet import MantisConfig, MantisNet, collate, from_position
from mantisnet.builder import TERN_DEC_CLASSES
from mantisnet.cell_latents import cell_tables
from mantisnet.lab.families import infer_config

# Dot-anchored so "mlp_p.lin_a.weight" does not read as a line-pass key.
_STEP15_MARKS = ("cell_base", ".cr_", ".wr_", ".lp_", ".ln_cr", ".ln_wr", ".ln_lp")


def _config(**overrides) -> MantisConfig:
    values = dict(
        h=16,
        blocks=2,
        heads=2,
        value_queries=2,
        value_bins=5,
        policy_hidden=16,
        value_hidden=16,
        cell_latents=True,
        line_pass=True,
        window_attention=False,
    )
    values.update(overrides)
    return MantisConfig(**values)


def _batch(positions, count=6):
    return collate([from_position(position) for position in positions[:count]])


def test_the_knobs_are_path_selectors():
    assert MantisConfig().cell_latents is False
    assert MantisConfig().line_pass is False
    assert MantisConfig().claim_reach == 5
    for value in (-1, 1, 3, 4, 6):
        with pytest.raises(ValueError, match=r"claim_reach"):
            MantisConfig(claim_reach=value)
    with pytest.raises(ValueError, match="inert"):
        MantisConfig(claim_reach=0, window_attention=False)


def test_knobs_off_leave_no_step15_trace():
    state = MantisNet(MantisConfig()).state_dict()
    assert not any(mark in key for key in state for mark in _STEP15_MARKS)


def test_the_cell_stage_replaces_the_relay():
    state = MantisNet(_config()).state_dict()
    assert "cell_base" in state
    assert "blocks.0.cr_vclass.weight" in state
    assert "blocks.1.wr_bias" in state
    assert "blocks.0.lp_bias" in state
    relay = ("u_cp", "e_cp", "mlp_cp", "ln_cp_in", "ln_cp_w")
    assert not any(mark in key for key in state for mark in relay)
    line_only = MantisNet(_config(cell_latents=False, window_attention=True))
    assert "blocks.0.u_cp.weight" in line_only.state_dict()
    assert "blocks.0.lp_bias" in line_only.state_dict()


def test_every_step15_parameter_trains(positions):
    torch.manual_seed(0)
    model = MantisNet(_config())
    with torch.no_grad():
        model.mlp_p.out.weight.normal_(std=0.1)
        model.mlp_q.out.weight.normal_(std=0.1)
    out = model(_batch(positions), 0.2)
    loss = (
        out.policy_logits.float().square().mean()
        + out.q_values.square().mean()
        + out.value.square().mean()
    )
    loss.backward()
    step15 = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if any(mark in name for mark in _STEP15_MARKS)
    ]
    # Per block: 25 cell-stage tensors and 10 line-pass tensors, plus the base.
    assert len(step15) == 2 * (25 + 10) + 1
    for name, parameter in step15:
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert float(parameter.grad.abs().sum()) > 0, name


def test_uncovered_legal_cells_read_the_base_row(positions):
    torch.manual_seed(0)
    model = MantisNet(_config()).eval()
    batch = _batch(positions)
    with torch.no_grad():
        _s, _w, _g, cells = model.trunk(batch)
    n_cells = batch.cell_pos.shape[0]
    assert cells is not None and cells.shape == (n_cells, model.cfg.h)

    tables = cell_tables(
        batch.dec_cell,
        batch.dec_window,
        batch.dec_class,
        batch.window_feat.shape[0],
        TERN_DEC_CLASSES,
    )
    covered = torch.zeros(n_cells, dtype=torch.bool)
    covered[tables.covered] = True
    assert bool(covered.any()) and not bool(covered.all())

    with torch.no_grad():
        base = model.ln_out(model.cell_base)
    torch.testing.assert_close(
        cells[~covered], base.expand(int((~covered).sum()), model.cfg.h)
    )
    # Refined rows moved away from the base: the stage did something.
    assert float((cells[covered] - base).abs().amax()) > 1e-4


def test_the_decoder_input_contract_fails_loudly(positions):
    batch = _batch(positions, count=2)
    off = MantisNet(_config(cell_latents=False, window_attention=True)).eval()
    on = MantisNet(_config()).eval()
    with torch.no_grad():
        _s, w_off, g_off, cells_off = off.trunk(batch)
        _s, w_on, g_on, cells_on = on.trunk(batch)
        assert cells_off is None
        with pytest.raises(ValueError, match="cell_latents is off"):
            off.cell_head_logits(w_off, g_off, cells_on, batch)
        with pytest.raises(ValueError, match="were not passed"):
            on.cell_head_logits(w_on, g_on, None, batch)


def test_arm_shapes_infer_and_reload(positions):
    # Every measured Step 15 arm round-trips shape inference and a strict
    # state-dict load; claim_reach leaves no tensor trace, so arm E infers
    # as the baseline shape.
    arms = {
        "B": _config(),
        "C": _config(line_pass=False),
        "D": _config(cell_latents=False),
        "E": _config(
            cell_latents=False, line_pass=False,
            window_attention=True, claim_reach=0,
        ),
    }
    for arm, cfg in arms.items():
        state = MantisNet(cfg).state_dict()
        inferred = infer_config(state)
        expected = cfg if arm != "E" else replace(cfg, claim_reach=5)
        assert inferred == expected, arm
        MantisNet(inferred).load_state_dict(state, strict=True)
