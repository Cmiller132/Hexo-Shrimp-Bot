"""Historical checkpoint-family identification, loading, and composition."""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import replace

import pytest
import torch

import hexo_py
from mantisnet.builder import collate_prefixes
from mantisnet.lab.families import (
    BIPOLAR,
    TRINOMIAL,
    infer_config,
    load_checkpoint,
)
from mantisnet.model import MantisConfig, MantisNet


TINY = MantisConfig(
    h=16,
    blocks=1,
    heads=2,
    ffn_factor=2,
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


def _family_state(name: str, *, cfg=TINY):
    torch.manual_seed(4)
    state = copy.deepcopy(MantisNet(cfg).state_dict())
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
    if name.endswith("-slot"):
        for key in ("e_pw.weight", "e_qw.weight"):
            state[key] = state[key][:3]
    if name not in {"trinomial-joint", "bipolar-joint", "scalar-joint"}:
        state["window_table.weight"] = state["window_table.weight"][:68]
        for index in range(cfg.blocks):
            for suffix in ("e_ws.weight", "e_sw.weight", "e_cp.weight"):
                key = f"blocks.{index}.{suffix}"
                state[key] = state[key][:93]
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
    _s, windows, token, cells = loaded.model.trunk(batch)
    _policy, logits = loaded.model.cell_head_logits(windows, token, cells, batch)
    scalar = loaded.composition.q_value(logits)
    grafted = torch.cat((logits, -logits, torch.full_like(logits, -20.0)), dim=-1)
    assert torch.equal(scalar, logits.squeeze(-1).tanh())
    assert torch.allclose(scalar, TRINOMIAL.q_value(grafted), rtol=0.0, atol=1e-6)


@torch.no_grad()
def test_historical_dead_key_biases_load_and_leave_outputs_identical(tmp_path):
    torch.manual_seed(41)
    direct = MantisNet(TINY).eval()
    state = copy.deepcopy(direct.state_dict())
    for index in range(TINY.blocks):
        state[f"blocks.{index}.wk.bias"] = torch.linspace(0.25, 1.25, TINY.h)
        state[f"blocks.{index}.wk_wa.bias"] = torch.linspace(-1.5, -0.5, TINY.h)

    path = tmp_path / "historical-key-biases.pt"
    torch.save(
        {
            "model": state,
            "model_config": dataclasses.asdict(TINY),
            "versions": _versions(),
            "iteration": 23,
        },
        path,
    )
    loaded = load_checkpoint(path, device="cpu")
    batch = collate_prefixes([[], [(0, 0), (-8, 8)]], [0, 2])
    expected, actual = direct(batch, 0.2), loaded.model(batch, 0.2)

    for field in vars(expected):
        assert torch.equal(getattr(actual, field), getattr(expected, field)), field


def test_bipolar_composition_matches_its_formula():
    torch.manual_seed(9)
    logits = torch.randn(31, 2)
    positive, negative = logits.sigmoid().unbind(dim=-1)
    assert torch.equal(BIPOLAR.q_value(logits), positive - negative)
    assert torch.equal(BIPOLAR.mass(logits), positive + negative)


@pytest.mark.parametrize(
    "name",
    ("scalar-slot", "bipolar-slot", "factored-slot", "tail-slot", "duel-slot"),
)
def test_binary_scope_families_are_cleanly_rejected(tmp_path, name):
    path = _write(tmp_path, name)
    with pytest.raises(ValueError, match="not identifiable by the family registry"):
        load_checkpoint(path)


def test_infer_config_recovers_a_nondefault_deep_shape_exactly():
    cfg = MantisConfig(
        h=64,
        blocks=6,
        heads=2,
        ffn_factor=3,
        value_queries=3,
        value_bins=33,
        policy_hidden=48,
        value_hidden=40,
        dropout=0.0,
    )
    assert infer_config(MantisNet(cfg).state_dict()) == cfg


def _write_production(tmp_path, cfg, *, record_config=True, model_config=None):
    torch.manual_seed(4)
    state = copy.deepcopy(MantisNet(cfg).state_dict())
    payload = {"model": state, "versions": _versions(), "iteration": 3}
    if record_config:
        payload["model_config"] = (
            dataclasses.asdict(cfg) if model_config is None else model_config
        )
    path = tmp_path / "production.pt"
    torch.save(payload, path)
    return path


def test_legacy_knob_recording_of_the_baked_architecture_loads(tmp_path):
    from mantisnet.model import LEGACY_BAKED_KNOBS

    cfg = replace(TINY, h=32, policy_hidden=24)
    legacy = {**dataclasses.asdict(cfg), **LEGACY_BAKED_KNOBS}
    loaded = load_checkpoint(
        _write_production(tmp_path, cfg, model_config=legacy), device="cpu"
    )
    assert loaded.family.name == "trinomial-joint"
    assert loaded.config == cfg


def test_legacy_knob_recording_of_another_architecture_refuses(tmp_path):
    from mantisnet.model import LEGACY_BAKED_KNOBS

    lying = {
        **dataclasses.asdict(TINY),
        **LEGACY_BAKED_KNOBS,
        "cell_pass_rounds": 2,
    }
    path = _write_production(tmp_path, TINY, model_config=lying)
    with pytest.raises(ValueError, match="no longer implements"):
        load_checkpoint(path)


def test_recorded_config_contradicting_tensors_is_refused(tmp_path):
    recorded = dataclasses.asdict(replace(TINY, heads=1))
    path = _write_production(tmp_path, TINY, model_config=recorded)
    with pytest.raises(ValueError, match="does not match the configuration inferred"):
        load_checkpoint(path)


def test_pre_baked_trunk_identifies_and_refuses_with_its_profile(tmp_path):
    state = _family_state("trinomial-joint")
    blocks = 1
    for index in range(blocks):
        prefix = f"blocks.{index}."
        for key in list(state):
            if key.startswith(prefix) and (
                "_cp" in key or "_wa" in key or key.endswith("wa_bias")
            ):
                del state[key]
        for key in (prefix + "e_ws.weight", prefix + "e_sw.weight"):
            state[key] = state[key][:3]
    path = tmp_path / "preknob.pt"
    torch.save({"model": state, "versions": _versions(), "iteration": 1}, path)
    with pytest.raises(ValueError, match="not the baked architecture"):
        load_checkpoint(path)


def test_baked_keys_on_only_some_blocks_are_refused(tmp_path):
    cfg = replace(TINY, blocks=2)
    state = _family_state("trinomial-joint", cfg=cfg)
    del state["blocks.1.orbit_bias"]
    path = tmp_path / "torn.pt"
    torch.save({"model": state, "versions": _versions(), "iteration": 1}, path)
    with pytest.raises(ValueError, match="not identifiable by the family registry"):
        load_checkpoint(path)


def test_unidentifiable_state_names_registry_and_contract(tmp_path):
    path = tmp_path / "unknown.pt"
    torch.save({"model": {"mystery": torch.zeros(1)}, "versions": _versions()}, path)
    with pytest.raises(ValueError, match="family registry.*trinomial-joint.*python/mantisnet/README.md"):
        load_checkpoint(path)
