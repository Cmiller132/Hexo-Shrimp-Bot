"""The baked all-nonempty scope: ternary laws, builder parity, and structure.

The ternary tables are the MANTIS_GRAFT_SPEC §4 class laws; the builder is
checked against the engine's window walk as the independent oracle, against
the Rust builder field for field, and with window attention on and off.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import hexo_py
from mantisnet import builder
from mantisnet.builder import (
    TERN_DEC_CLASSES,
    TERN_OCC_CLASSES,
    TERN_PATTERNS,
    WINDOW_LEN,
    _TERN_DEC_CLASS,
    _TERN_OCC_CLASS,
    _TERN_RANK,
    collate,
    collate_prefixes,
    from_position,
)
from mantisnet.model import MantisConfig, MantisNet


def _digits(pattern: int) -> list[int]:
    return [(pattern // 3**k) % 3 for k in range(WINDOW_LEN)]


def _reverse3(pattern: int) -> int:
    return sum(d * 3**k for k, d in enumerate(reversed(_digits(pattern))))


# Deterministic self-play-free fixture games: scripted prefixes of varied
# density, including adjacent opposite-colour stones so mixed windows exist.
_GAMES = [
    [(0, 0)],
    [(0, 0), (1, 0), (2, 0)],
    [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)],
    [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (0, 3), (0, 4), (3, 0)],
    [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, -1), (0, 2), (4, -1), (-1, 2)],
]


def _positions():
    return [hexo_py.Position.replay(moves) for moves in _GAMES]


def test_ternary_class_laws():
    """The spec's carried counts: 378/377 patterns, 2187/2184 joint orbits."""
    canon = {min(p, _reverse3(p)) for p in range(729)}
    assert len(canon) == 378
    assert TERN_PATTERNS == 377
    assert int(_TERN_RANK[0]) == -1 and int(_TERN_RANK.max()) + 1 == 377

    # Rank is reversal-invariant and dense over nonempty patterns.
    for p in range(1, 729):
        assert _TERN_RANK[p] == _TERN_RANK[_reverse3(p)] >= 0

    assert TERN_DEC_CLASSES == 726 and TERN_OCC_CLASSES == 1458
    assert TERN_DEC_CLASSES + TERN_OCC_CLASSES == 2184
    # Including the empty pattern's three orbits, the joint involution has
    # 2187 orbits over all 729 x 6 pairs.
    assert TERN_DEC_CLASSES + TERN_OCC_CLASSES + 3 == 2187

    for p in range(1, 729):
        rev = _reverse3(p)
        for s, digit in enumerate(_digits(p)):
            mirror = (rev, WINDOW_LEN - 1 - s)
            if digit == 0:
                assert _TERN_OCC_CLASS[p, s] == -1
                assert _TERN_DEC_CLASS[p, s] == _TERN_DEC_CLASS[mirror] >= 0
            else:
                assert _TERN_DEC_CLASS[p, s] == -1
                assert _TERN_OCC_CLASS[p, s] == _TERN_OCC_CLASS[mirror] >= 0


def test_scope_matches_the_engine_window_oracle():
    """The kept window set is exactly the engine's nonempty candidates, and
    each window's ternary pattern matches the engine masks slot for slot."""
    for pos in _positions():
        graph = from_position(pos)
        mover = pos.current_player
        oracle: dict[tuple[int, int, int], int] = {}
        for q, r, _player in pos.stones():
            for axis, sq, sr, m0, m1 in pos.windows_through(q, r):
                if (m0 | m1) == 0:
                    continue
                own, opp = (m0, m1) if mover == 0 else (m1, m0)
                pattern = sum(
                    ((own >> k & 1) + 2 * (opp >> k & 1)) * 3**k
                    for k in range(WINDOW_LEN)
                )
                oracle[(axis, sq, sr)] = pattern

        got = {tuple(map(int, row)) for row in graph.window_id}
        assert got == set(oracle)
        for row, feat in zip(graph.window_id, graph.window_feat):
            assert int(feat) == int(_TERN_RANK[oracle[tuple(map(int, row))]])


def test_rust_python_parity():
    """Rust and Python builders agree field for field."""
    rust = builder.collate_prefixes(_GAMES, [len(g) for g in _GAMES])
    python = collate([from_position(pos) for pos in _positions()])
    for name, value in vars(python).items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(getattr(rust, name), value), name
        else:
            assert getattr(rust, name) == value, name


def test_baked_model_forward_and_shapes():
    """The baked model builds, runs every head, and sizes its tables, and
    carries no §5.1c stage parameters."""
    cfg = MantisConfig(h=32, heads=2, blocks=2, policy_hidden=32, value_hidden=32)
    torch.manual_seed(0)
    model = MantisNet(cfg).eval()

    assert model.window_table.weight.shape[0] == cfg.window_vocab
    assert model.e_pw.weight.shape[0] == cfg.dec_classes
    assert model.blocks[0].e_ws.weight.shape[0] == cfg.occ_classes
    assert model.blocks[0].e_cp.weight.shape[0] == cfg.dec_classes
    assert not any("wa" in name for name, _ in model.named_parameters())

    batch = collate([from_position(pos) for pos in _positions()])
    with torch.no_grad():
        out = model(batch, 0.2)
    assert out.policy_logits.shape[0] == batch.n_cells
    assert out.q_values.shape[0] == batch.n_cells
    assert torch.isfinite(out.policy_logits).all()
    assert torch.isfinite(out.value).all()

    # Gradients reach the scope tables through the gathered class paths. The
    # decoder out layers zero-init (constant initial logits), which blocks
    # upstream flow — perturb them so the paths carry gradient.
    model.train()
    with torch.no_grad():
        for head in (model.mlp_p, model.mlp_q):
            torch.nn.init.normal_(head.out.weight, std=0.1)
    out = model(batch, 0.2)
    (out.policy_logits.sum() + out.q_values.sum() + out.value.sum()).backward()
    for name in ("window_table", "e_pw", "e_qw"):
        grad = getattr(model, name).weight.grad
        assert grad is not None and grad.abs().sum() > 0, name
    assert model.blocks[0].e_ws.weight.grad is not None
    assert model.blocks[0].e_ws.weight.grad.abs().sum() > 0


def _transform(move):
    # A non-identity member of the engine-consistent D6 family.
    from mantisnet.klent import telemetry

    return telemetry.D6_TRANSFORMS[1](move)


def test_baked_model_is_d6_invariant():
    """Value invariance and policy equivariance hold for the baked model."""
    cfg = MantisConfig(h=32, heads=2, blocks=2, policy_hidden=32, value_hidden=32)
    torch.manual_seed(1)
    model = MantisNet(cfg).eval()

    for moves in _GAMES[2:]:
        pos = hexo_py.Position.replay(moves)
        base = collate([from_position(pos)])
        turned_moves = [_transform(m) for m in moves]
        turned_pos = hexo_py.Position.replay(turned_moves)
        turned = collate([from_position(turned_pos)])
        with torch.no_grad():
            got = model(base, 0.2)
            got_turned = model(turned, 0.2)

        assert torch.allclose(got.value, got_turned.value, atol=1e-5)
        base_map = dict(zip(pos.legal_moves(), got.policy_logits.tolist()))
        turned_map = dict(
            zip(turned_pos.legal_moves(), got_turned.policy_logits.tolist())
        )
        assert set(turned_map) == {_transform(m) for m in base_map}
        for move, logit in base_map.items():
            assert turned_map[_transform(move)] == pytest.approx(logit, abs=1e-5)


def test_checkpoint_round_trips_through_the_family_registry(tmp_path):
    """A recorded-config checkpoint is identified, inferred, and loaded."""
    from mantisnet.lab.families import infer_config, load_checkpoint

    cfg = MantisConfig()
    torch.manual_seed(2)
    model = MantisNet(cfg)
    # With every per-head knob off no tensor carries the head count:
    # tensor-only inference refuses, and the recorded config supplies it.
    with pytest.raises(ValueError, match="per-head"):
        infer_config(model.state_dict())
    inferred = infer_config(model.state_dict(), heads=cfg.heads)
    assert inferred == cfg

    path = tmp_path / "mixed.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "versions": {
                "RULES_VERSION": hexo_py.RULES_VERSION,
                "ACTION_ORDER_VERSION": hexo_py.ACTION_ORDER_VERSION,
            },
            "model_config": {
                "mixed_windows": True,
                "window_attention": False,
            },
            "iteration": 7,
        },
        path,
    )
    loaded = load_checkpoint(path)
    assert loaded.family.name == "trinomial-joint"
    assert loaded.config == cfg

    batch = collate_prefixes(_GAMES[2:], [len(g) for g in _GAMES[2:]])
    with torch.no_grad():
        reference = MantisNet(cfg).eval()
        reference.load_state_dict(model.state_dict())
        expected = reference(batch, 0.2)
        got_policy, got_score, got_q = loaded.model.cell_heads(
            *loaded.model.trunk(batch), batch, 0.2
        )
    assert torch.allclose(got_policy, expected.policy_logits, atol=1e-6)
    assert torch.allclose(got_q, expected.q_values, atol=1e-6)


def test_recorded_binary_scope_is_refused():
    from mantisnet.model import strip_legacy_knobs

    with pytest.raises(ValueError, match="no longer implements"):
        strip_legacy_knobs({"mixed_windows": False})
