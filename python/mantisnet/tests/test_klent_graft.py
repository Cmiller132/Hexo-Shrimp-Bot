"""The composed joint-class and bipolar-return-mass graft.

The synthetic parent has both old shapes: three slot-class rows per decoder and
one scalar-tanh critic row. Tests cover each tensor transform, their composed
function, both Adam remaps, both independent check batteries, and refusals.

The row map is checked against ``conftest``'s own orbit derivation rather than the
builder's table, and preservation is checked by transcribing §6 over the parent's
3-row tables here, so neither half of this file reads its expectation out of the
code it tests.
"""

from __future__ import annotations

import copy
import json

import hexo_py
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from mantisnet import MantisConfig, MantisNet, collate, from_position
from mantisnet.builder import DEC_CLASSES
from mantisnet.klent import graft as graft_module
from mantisnet.klent.graft import (
    MAX_ABS_DQ,
    MAX_MEAN_KL,
    PARENT_CLASSES,
    PARENT_REPR_VERSION,
    PROBE_PLIES,
    PROBE_POSITIONS,
    PROBE_TOP_K,
    _EXPANDED_KEYS,
    _PARENT_ROW,
    _READOUT_KEYS,
    _probe_prefixes,
    graft,
    main,
)
from mantisnet.klent.run import _versions, load_checkpoint

from .conftest import joint_class

TAU, LAM = 0.1, 0.01
_ITERATION = 151


def _parent_versions() -> dict:
    return {**_versions(), "MODEL_REPR_VERSION": PARENT_REPR_VERSION}


def _parent_checkpoint(unstepped: str | None = "value_queries") -> dict:
    """A slot-class/scalar-tanh checkpoint with one Adam step behind it.

    ``unstepped`` names a parameter the optimizer never stepped, which is how the
    state-value head KLENT does not train appears in every real checkpoint.
    """
    torch.manual_seed(30)
    model = MantisNet(MantisConfig())
    bound = model.cfg.policy_hidden**-0.5
    with torch.no_grad():
        for out in (model.mlp_p.out, model.mlp_q.out):
            out.weight.uniform_(-bound, bound)
            out.bias.uniform_(-bound, bound)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    names = [name for name, _p in model.named_parameters()]
    for index, (name, parameter) in enumerate(model.named_parameters()):
        if name != unstepped:
            parameter.grad = torch.full_like(parameter, 0.01 * (index % 5 + 1))
    optimizer.step()

    state = copy.deepcopy(optimizer.state_dict())
    entries = {
        name: state["state"][index]
        for index, name in enumerate(names)
        if index in state["state"]
    }

    # Cut the two class tables back to the parent's 3 rows, weights and moments
    # alike, with values of their own rather than a slice of the expansion.
    generator = torch.Generator().manual_seed(31)
    model_state = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }
    for key in _EXPANDED_KEYS:
        h = model_state[key].shape[1]
        model_state[key] = torch.randn(PARENT_CLASSES, h, generator=generator) * 0.02
        for field in ("exp_avg", "exp_avg_sq"):
            entries[key][field] = entries[key][field][:PARENT_CLASSES].clone()
    for key in _READOUT_KEYS:
        model_state[key] = model_state[key][:1].clone()
        for field in ("exp_avg", "exp_avg_sq"):
            entries[key][field] = entries[key][field][:1].clone()

    rng = np.random.default_rng(31)
    return {
        "model": model_state,
        "optimizer": {
            "state": {
                index: entries[name]
                for index, name in enumerate(names)
                if name in entries
            },
            "param_groups": [
                {**state["param_groups"][0], "params": list(range(len(names)))}
            ],
        },
        "iteration": _ITERATION,
        "rng_state": copy.deepcopy(rng.bit_generator.state),
        "versions": _parent_versions(),
    }


def _paths(tmp_path):
    return tmp_path / "parent.pt", tmp_path / "grafted.pt", tmp_path / "manifest.json"


def _write_parent(tmp_path, checkpoint=None):
    old, new, manifest = _paths(tmp_path)
    torch.save(checkpoint if checkpoint is not None else _parent_checkpoint(), old)
    return old, new, manifest


def test_every_joint_class_replicates_its_slot_class_row():
    # The map the conversion rests on, against conftest's own orbit derivation:
    # every (mask, slot) pair's class takes the parent row of that slot's class,
    # and mirrored slots — which share an orbit — agree on it.
    assert _PARENT_ROW.shape == (DEC_CLASSES,)
    for mask in range(1, 63):
        for slot in range(6):
            if (mask >> slot) & 1:
                continue
            assert _PARENT_ROW[joint_class(mask, slot)] == min(slot, 5 - slot)
    assert set(_PARENT_ROW.tolist()) == {0, 1, 2}


@torch.no_grad()
def test_graft_preserves_the_parent_decode(tmp_path, positions):
    old, new, manifest_path = _write_parent(tmp_path)
    parent = torch.load(old, map_location="cpu", weights_only=False)["model"]

    manifest = graft(old, new, TAU, LAM, manifest_path)
    assert manifest["preservation"]["holds"]
    # Preservation is exact, not merely inside a bound: the spec decode agrees
    # bit for bit on both sides. The bounded deltas below it are the folded head
    # GEMM's reassociation, which is not the graft's.
    assert manifest["spec_decode_bitwise_equal"] is True
    assert new.exists()
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    assert manifest["source_iteration"] == _ITERATION
    # The conversion is where the representation version moves, and the only
    # place: the parent's is refused by every loader, the child's is this build's.
    assert manifest["parent_versions"]["MODEL_REPR_VERSION"] == PARENT_REPR_VERSION
    assert manifest["versions"] == _versions()
    assert manifest["probe_positions"] == PROBE_POSITIONS
    assert manifest["probe_legal_cells"] > PROBE_POSITIONS * PROBE_TOP_K
    assert (min(manifest["probe_plies"]), max(manifest["probe_plies"])) == PROBE_PLIES
    assert 0 <= manifest["mean_abs_dq"] <= manifest["max_abs_dq"] <= MAX_ABS_DQ
    assert abs(manifest["mean_improved_kl"]) <= MAX_MEAN_KL

    model = MantisNet(MantisConfig())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    assert load_checkpoint(new, model, optimizer) == _ITERATION
    model.eval()

    # §6 transcribed here over the parent's 3-row tables and the slot class each
    # joint class stands for, which is the comparison the manifest also makes and
    # this repeats independently of it.
    batch = collate([from_position(p) for p in positions])
    _s, w, g = model.trunk(batch)
    policy, q = model.cell_heads(w, g, batch)
    slot_class = torch.from_numpy(_PARENT_ROW)[batch.dec_class]
    for scores, tail, proj, table, bg, mlp in (
        (policy, lambda x: x, "p.weight", "e_pw.weight", "e_bg.weight", "mlp_p"),
        (q, torch.tanh, "q.weight", "e_qw.weight", "e_qbg.weight", "mlp_q"),
    ):
        msg = F.linear(w, parent[proj])[batch.dec_window] + parent[table][slot_class]
        h = torch.zeros(batch.cell_pos.shape[0], model.cfg.h)
        h.index_add_(0, batch.dec_cell, msg)
        h[batch.bg_cell] = parent[bg][batch.bg_bucket]
        hidden = F.relu(
            F.linear(h, parent[f"{mlp}.lin_a.weight"], parent[f"{mlp}.lin_a.bias"])
            + F.linear(g, parent[f"{mlp}.lin_b.weight"])[batch.cell_pos]
        )
        expected = F.linear(
            hidden, parent[f"{mlp}.out.weight"], parent[f"{mlp}.out.bias"]
        ).squeeze(-1)
        torch.testing.assert_close(scores, tail(expected), rtol=0, atol=1e-4)

    # And the expansion is a row replication, bit for bit.
    grafted = torch.load(new, map_location="cpu", weights_only=False)["model"]
    for key in _EXPANDED_KEYS:
        assert grafted[key].shape == (DEC_CLASSES, model.cfg.h)
        for cls in range(DEC_CLASSES):
            assert torch.equal(grafted[key][cls], parent[key][_PARENT_ROW[cls]])
    for key in _READOUT_KEYS:
        assert torch.equal(
            grafted[key],
            torch.cat([2 * parent[key], -2 * parent[key]], dim=0),
        )
    for key, tensor in grafted.items():
        if key not in (*_EXPANDED_KEYS, *_READOUT_KEYS):
            assert torch.equal(tensor, parent[key]), key


def test_adam_moments_follow_their_rows_and_absence_is_carried(tmp_path):
    old, new, manifest_path = _write_parent(tmp_path)
    parent = torch.load(old, map_location="cpu", weights_only=False)
    graft(old, new, TAU, LAM, manifest_path)
    child = torch.load(new, map_location="cpu", weights_only=False)

    model = MantisNet(MantisConfig())
    names = [name for name, _p in model.named_parameters()]
    parent_state = {names[i]: e for i, e in parent["optimizer"]["state"].items()}
    child_state = {names[i]: e for i, e in child["optimizer"]["state"].items()}

    # Adam's state dict is sparse, and absence means unstepped: carried as
    # absence, not as a zero-filled entry that would claim a step happened.
    assert "value_queries" not in parent_state and "value_queries" not in child_state
    assert set(child_state) == set(parent_state)

    for name, entry in child_state.items():
        for field in ("exp_avg", "exp_avg_sq"):
            if name in _EXPANDED_KEYS:
                assert entry[field].shape == (DEC_CLASSES, model.cfg.h)
                for cls in range(DEC_CLASSES):
                    assert torch.equal(
                        entry[field][cls], parent_state[name][field][_PARENT_ROW[cls]]
                    )
            elif name in _READOUT_KEYS:
                if field == "exp_avg":
                    expected = torch.cat(
                        [2 * parent_state[name][field], -2 * parent_state[name][field]]
                    )
                else:
                    expected = torch.cat(
                        [parent_state[name][field], parent_state[name][field]]
                    )
                assert torch.equal(entry[field], expected)
            else:
                assert torch.equal(entry[field], parent_state[name][field])
        assert entry["step"] == parent_state[name]["step"]


def test_the_cli_writes_the_same_manifest(tmp_path):
    old, new, _manifest_path = _write_parent(tmp_path)
    manifest_path = new.with_suffix(".json")
    main([str(old), str(new), "--tau", str(TAU), "--lam", str(LAM)])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["arm"] == "joint-brm"
    assert manifest["classes"] == DEC_CLASSES == 93
    assert manifest["class_to_parent_row"] == _PARENT_ROW.tolist()
    assert manifest["preservation"]["holds"]


def test_a_parent_of_this_representation_is_refused(tmp_path):
    # A checkpoint already at this build's version is not a parent to convert:
    # its class tables are 93 rows wide and the transform has nothing to do.
    checkpoint = _parent_checkpoint()
    checkpoint["versions"] = _versions()
    old, new, manifest_path = _write_parent(tmp_path, checkpoint)
    with pytest.raises(ValueError, match="!= the parent build"):
        graft(old, new, TAU, LAM, manifest_path)
    assert not new.exists() and not manifest_path.exists()


def test_an_already_expanded_table_is_refused(tmp_path):
    checkpoint = _parent_checkpoint()
    for key in _EXPANDED_KEYS:
        checkpoint["model"][key] = torch.zeros(DEC_CLASSES, MantisConfig().h)
    old, new, manifest_path = _write_parent(tmp_path, checkpoint)
    with pytest.raises(ValueError, match="differ from this build only at"):
        graft(old, new, TAU, LAM, manifest_path)
    assert not new.exists() and not manifest_path.exists()


def test_an_unexpected_architecture_difference_is_refused(tmp_path):
    checkpoint = _parent_checkpoint()
    checkpoint["model"]["mlp_q.out.weight"] = torch.zeros(1, 7)
    old, new, manifest_path = _write_parent(tmp_path, checkpoint)
    with pytest.raises(ValueError, match="critic readout must be one row"):
        graft(old, new, TAU, LAM, manifest_path)
    assert not new.exists() and not manifest_path.exists()


def test_a_malformed_optimizer_entry_is_refused(tmp_path):
    checkpoint = _parent_checkpoint()
    entry = next(iter(checkpoint["optimizer"]["state"].values()))
    del entry["exp_avg_sq"]
    old, new, manifest_path = _write_parent(tmp_path, checkpoint)
    with pytest.raises(ValueError, match="missing: exp_avg_sq"):
        graft(old, new, TAU, LAM, manifest_path)
    assert not new.exists() and not manifest_path.exists()


def test_a_moment_of_the_wrong_width_is_refused(tmp_path):
    # A moment that is already 93 rows wide would sail through a shape check
    # written against the new parameter instead of the parent's.
    checkpoint = _parent_checkpoint()
    model = MantisNet(MantisConfig())
    names = [name for name, _p in model.named_parameters()]
    index = names.index("e_pw.weight")
    checkpoint["optimizer"]["state"][index]["exp_avg"] = torch.zeros(
        DEC_CLASSES, model.cfg.h
    )
    old, new, manifest_path = _write_parent(tmp_path, checkpoint)
    with pytest.raises(ValueError, match="exp_avg for e_pw.weight"):
        graft(old, new, TAU, LAM, manifest_path)
    assert not new.exists() and not manifest_path.exists()


def test_a_misplaced_expansion_row_is_caught(tmp_path, monkeypatch):
    # The row check is a detector, not a restatement: an expansion that writes
    # each class the row of its neighbour is refused by name.
    old, new, manifest_path = _write_parent(tmp_path)
    monkeypatch.setattr(
        graft_module, "_expand", lambda table: table[_PARENT_ROW].roll(1, dims=0)
    )
    with pytest.raises(ValueError, match="are not the parent row they replace"):
        graft(old, new, TAU, LAM, manifest_path)
    assert not new.exists() and not manifest_path.exists()


def test_a_failed_bound_writes_nothing(tmp_path, monkeypatch):
    old, new, manifest_path = _write_parent(tmp_path)
    monkeypatch.setattr(graft_module, "Q_TOLERANCE", -1.0)
    with pytest.raises(ValueError, match="joint decoder checks failed"):
        graft(old, new, TAU, LAM, manifest_path)
    assert not new.exists() and not manifest_path.exists()


def test_one_path_for_both_is_refused(tmp_path):
    old, _new, manifest_path = _write_parent(tmp_path)
    with pytest.raises(ValueError, match="must be different paths"):
        graft(old, old, TAU, LAM, manifest_path)


def test_brm_probe_is_deterministic_and_nonterminal():
    prefixes = _probe_prefixes()
    assert len(prefixes) == PROBE_POSITIONS
    lengths = [len(moves) for moves in prefixes]
    assert lengths == sorted(lengths)
    assert (lengths[0], lengths[-1]) == PROBE_PLIES
    assert _probe_prefixes() == prefixes
    for moves in prefixes[::8]:
        assert not hexo_py.Position.replay(moves).is_terminal


def test_a_wrong_brm_gain_is_caught(tmp_path, monkeypatch):
    """Gain one composes to tanh(z / 2), so the BRM battery must refuse it."""
    old, new, manifest_path = _write_parent(tmp_path)
    monkeypatch.setattr(graft_module, "READOUT_GAIN", 1.0)
    with pytest.raises(ValueError, match="checks failed|not function preserving"):
        graft(old, new, TAU, LAM, manifest_path)
    assert not new.exists() and not manifest_path.exists()


def test_a_mangled_shared_tensor_is_caught(tmp_path, monkeypatch):
    """The separate parent forward makes the BRM probe cover shared tensors."""
    victim = "blocks.0.ffn.0.weight"
    convert = graft_module._converted_state

    def mangle(old_model):
        state = convert(old_model)
        state[victim] = state[victim] * 1.05
        return state

    old, new, manifest_path = _write_parent(tmp_path)
    monkeypatch.setattr(graft_module, "_converted_state", mangle)
    with pytest.raises(ValueError, match="untouched parent tensor"):
        graft(old, new, TAU, LAM, manifest_path)
    assert not new.exists() and not manifest_path.exists()


def test_missing_brm_adam_state_is_refused(tmp_path):
    checkpoint = _parent_checkpoint()
    names = list(checkpoint["model"])
    index = names.index(_READOUT_KEYS[0])
    del checkpoint["optimizer"]["state"][index]
    old, new, manifest_path = _write_parent(tmp_path, checkpoint)
    with pytest.raises(ValueError, match="no Adam state for"):
        graft(old, new, TAU, LAM, manifest_path)
    assert not new.exists() and not manifest_path.exists()


def test_cli_requires_the_operating_point(tmp_path):
    paths = [str(tmp_path / "old.pt"), str(tmp_path / "new.pt")]
    for argv in (paths, paths + ["--tau", str(TAU)], paths + ["--lam", str(LAM)]):
        with pytest.raises(SystemExit):
            main(argv)
