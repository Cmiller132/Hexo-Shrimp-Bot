"""Historical checkpoint-family identification, loading, and composition."""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

import hexo_py
from mantisnet.builder import DEC_CLASSES, WINDOW_LEN, _DEC_CLASS, collate_prefixes
from mantisnet.lab.families import (
    BIPOLAR,
    FACTORED,
    FAMILIES,
    SCALAR,
    TRINOMIAL,
    FamilyMantisNet,
    infer_config,
    load_checkpoint,
)
from mantisnet.model import MantisConfig, MantisNet


TINY = MantisConfig(
    h=16,
    blocks=1,
    heads=2,
    ffn_factor=2,
    d_max=5,
    value_queries=2,
    value_bins=9,
    policy_hidden=12,
    value_hidden=10,
)


def _versions():
    return {
        "MODEL_REPR_VERSION": -1,
        "RULES_VERSION": hexo_py.RULES_VERSION,
        "ACTION_ORDER_VERSION": hexo_py.ACTION_ORDER_VERSION,
        "torch": "historical-test-build",
    }


def _independent_parent_rows() -> torch.Tensor:
    rows = np.full(DEC_CLASSES, -1, dtype=np.int64)
    for mask in range(1, 63):
        for slot in range(WINDOW_LEN):
            joint = int(_DEC_CLASS[mask, slot])
            if joint >= 0:
                slot_class = min(slot, WINDOW_LEN - 1 - slot)
                if rows[joint] not in {-1, slot_class}:
                    raise AssertionError("decoder orbit crosses slot classes")
                rows[joint] = slot_class
    assert np.all(rows >= 0)
    return torch.from_numpy(rows)


def _family_state(name: str, *, cfg=TINY):
    torch.manual_seed(4)
    state = copy.deepcopy(MantisNet(cfg).state_dict())
    entry = next(item for item in FAMILIES if item.name == name)
    width = {
        "trinomial-joint": 3,
        "bipolar-joint": 2,
        "scalar-joint": 1,
        "scalar-slot": 1,
        "bipolar-slot": 2,
        "factored-slot": 2,
        "tail-slot": 1,
        "duel-slot": 1,
    }[name]
    state["mlp_q.out.weight"] = torch.randn(width, cfg.policy_hidden)
    state["mlp_q.out.bias"] = torch.randn(width)
    if entry.table_rows == 3:
        parent = _independent_parent_rows()
        representatives = [int(torch.nonzero(parent == row)[0]) for row in range(3)]
        for key in ("e_pw.weight", "e_qw.weight"):
            state[key] = state[key].index_select(0, torch.tensor(representatives))
    if name == "tail-slot":
        fh = cfg.ffn_factor * cfg.h
        state.update(
            {
                "q_tail_ln.weight": torch.ones(cfg.h),
                "q_tail_ln.bias": torch.zeros(cfg.h),
                "q_tail.0.weight": torch.randn(fh, cfg.h),
                "q_tail.0.bias": torch.randn(fh),
                "q_tail.2.weight": torch.randn(cfg.h, fh),
                "q_tail.2.bias": torch.randn(cfg.h),
            }
        )
    if name == "duel-slot":
        state.update(
            {
                "mlp_qbase.0.weight": torch.randn(cfg.policy_hidden, cfg.h),
                "mlp_qbase.0.bias": torch.randn(cfg.policy_hidden),
                "mlp_qbase.2.weight": torch.randn(1, cfg.policy_hidden),
                "mlp_qbase.2.bias": torch.randn(1),
            }
        )
    return state


def _write(tmp_path, name: str, *, cfg=TINY):
    path = tmp_path / f"{name}.pt"
    torch.save(
        {"model": _family_state(name, cfg=cfg), "versions": _versions(), "iteration": 17},
        path,
    )
    return path


@pytest.mark.parametrize(
    ("name", "explicit"),
    (
        ("trinomial-joint", None),
        ("bipolar-joint", None),
        ("scalar-joint", None),
        ("scalar-slot", None),
        ("bipolar-slot", "bipolar-slot"),
        ("factored-slot", "factored-slot"),
    ),
)
def test_every_scoreable_family_identifies_loads_and_runs_full_forward(
    tmp_path, name, explicit
):
    loaded = load_checkpoint(_write(tmp_path, name), family=explicit, device="cpu")
    assert loaded.family.name == name
    assert loaded.config == TINY
    output = loaded.model(collate_prefixes([[]], [0]), 0.2)
    assert output.policy_logits.ndim == output.q_score.ndim == output.q_values.ndim == 1
    assert output.value.shape == (1,)
    assert output.q_values.dtype == output.q_score.dtype == torch.float32


def test_scalar_joint_matches_native_tanh_and_trigraft_transform(tmp_path):
    loaded = load_checkpoint(_write(tmp_path, "scalar-joint"), device="cpu")
    batch = collate_prefixes([[]], [0])
    _s, windows, token = loaded.model.trunk(batch)
    _policy, logits = loaded.model.cell_head_logits(windows, token, batch)
    scalar = loaded.composition.q_value(logits)
    grafted = torch.cat((logits, -logits, torch.full_like(logits, -20.0)), dim=-1)
    assert torch.equal(scalar, logits.squeeze(-1).tanh())
    assert torch.allclose(scalar, TRINOMIAL.q_value(grafted), rtol=0.0, atol=1e-6)


def test_slot_tables_expand_by_independently_derived_decoder_slot_class(tmp_path):
    state = _family_state("scalar-slot")
    path = tmp_path / "slot.pt"
    torch.save({"model": state, "versions": _versions(), "iteration": 1}, path)
    loaded = load_checkpoint(path)
    parent_rows = _independent_parent_rows()
    for key in ("e_pw.weight", "e_qw.weight"):
        expected = state[key].index_select(0, parent_rows)
        assert torch.equal(loaded.model.state_dict()[key], expected)

    direct_state = dict(state)
    for key in ("e_pw.weight", "e_qw.weight"):
        direct_state[key] = state[key].index_select(0, parent_rows)
    direct = FamilyMantisNet(TINY, SCALAR)
    direct.load_state_dict(direct_state, strict=True)
    batch = collate_prefixes([[], [(0, 0), (-8, 8)]], [0, 2])
    got, expected = loaded.model(batch, 0.2), direct(batch, 0.2)
    for field in ("policy_logits", "q_score", "q_values"):
        assert torch.equal(getattr(got, field), getattr(expected, field))


def test_bipolar_and_factored_compositions_match_their_formulas():
    torch.manual_seed(9)
    logits = torch.randn(31, 2)
    positive, negative = logits.sigmoid().unbind(dim=-1)
    assert torch.equal(BIPOLAR.q_value(logits), positive - negative)
    assert torch.equal(BIPOLAR.mass(logits), positive + negative)
    mover, magnitude = logits.unbind(dim=-1)
    mass = magnitude.sigmoid()
    assert torch.equal(FACTORED.q_value(logits), (2 * mover.sigmoid() - 1) * mass)
    assert torch.equal(FACTORED.mass(logits), mass)


def test_two_row_slot_tie_refuses_without_family_and_loads_with_either(tmp_path):
    path = _write(tmp_path, "bipolar-slot")
    with pytest.raises(ValueError, match="ambiguous.*bipolar-slot, factored-slot.*--family"):
        load_checkpoint(path)
    assert load_checkpoint(path, family="bipolar-slot").family.name == "bipolar-slot"
    assert load_checkpoint(path, family="factored-slot").family.name == "factored-slot"


@pytest.mark.parametrize(("name", "private"), (("tail-slot", "q_tail"), ("duel-slot", "mlp_qbase")))
def test_named_unscoreable_families_identify_and_refuse(tmp_path, name, private):
    path = _write(tmp_path, name)
    with pytest.raises(ValueError, match=rf"{name!s}.*not scoreable.*{private}.*composition-parity"):
        load_checkpoint(path)


def test_infer_config_recovers_a_nondefault_deep_shape_exactly():
    cfg = MantisConfig(
        h=64,
        blocks=6,
        heads=2,
        ffn_factor=3,
        d_max=9,
        value_queries=3,
        value_bins=33,
        policy_hidden=48,
        value_hidden=40,
        dropout=0.0,
    )
    assert infer_config(MantisNet(cfg).state_dict()) == cfg


@pytest.mark.parametrize(
    ("field", "variant_key"),
    (
        ("cell_pass", "blocks.0.e_cp.weight"),
        ("axis_bias", "blocks.0.axis_bias"),
    ),
)
def test_ablation_state_dicts_are_refused_by_family_registry(
    tmp_path, field, variant_key
):
    cfg = replace(TINY, **{field: True})
    state = _family_state("trinomial-joint", cfg=cfg)
    assert variant_key in state

    path = tmp_path / f"{field}.pt"
    torch.save({"model": state, "versions": _versions(), "iteration": 1}, path)
    with pytest.raises(ValueError, match="not identifiable by the family registry"):
        load_checkpoint(path)
    with pytest.raises(ValueError, match="does not structurally claim named family"):
        load_checkpoint(path, family="trinomial-joint")


def test_unidentifiable_state_names_registry_and_contract(tmp_path):
    path = tmp_path / "unknown.pt"
    torch.save({"model": {"mystery": torch.zeros(1)}, "versions": _versions()}, path)
    with pytest.raises(ValueError, match="family registry.*trinomial-joint.*docs/LAB_SPEC.md"):
        load_checkpoint(path)
