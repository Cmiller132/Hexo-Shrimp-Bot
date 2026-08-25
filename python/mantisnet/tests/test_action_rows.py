"""Action-row tables: class laws, successor-board oracle, D6, and collation.

The builder's 18 hypothetical post-placement windows per legal action are
checked against an independent oracle that actually plays each action on a
board copy and reads the successor's windows from the engine walk. The
collated views are recomputed here from the dense tables, and the model is held
to the same D6, initialization, and registry contracts as every other build.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import hexo_py
from mantisnet import builder
from mantisnet.builder import (
    ACTION_EMPTY,
    ACTION_EMPTY_CLASSES,
    ACTION_MIXED,
    ACTION_OPP,
    ACTION_OWN,
    TERN_POST1_CLASSES,
    WINDOW_LEN,
    _TERN_POST1_CLASS,
    _TERN_REV,
    collate,
    from_position,
)
from mantisnet.model import MantisConfig, MantisNet

AXES = ((1, 0), (0, 1), (1, -1))

_GAMES = [
    [],
    [(0, 0)],
    [(0, 0), (1, 0), (2, 0)],
    [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)],
    [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (0, 3), (0, 4), (3, 0)],
    [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, -1), (0, 2), (4, -1), (-1, 2)],
]


def test_ternary_post1_law():
    """729 orbits of own-slot (post, slot) pairs, reversal-invariant."""
    assert TERN_POST1_CLASSES == 729
    assert int(_TERN_POST1_CLASS.max()) + 1 == 729
    for post in range(729):
        rev = int(_TERN_REV[post])
        for s in range(WINDOW_LEN):
            own = (post // 3**s) % 3 == 1
            mirror = _TERN_POST1_CLASS[rev, WINDOW_LEN - 1 - s]
            if own:
                assert _TERN_POST1_CLASS[post, s] == mirror >= 0
            else:
                assert _TERN_POST1_CLASS[post, s] == -1


def _successor_windows(pos, move):
    """Engine oracle: play the move on a copy, return the successor's window
    masks keyed by (axis, start_q, start_r)."""
    succ = pos.copy()
    succ.advance(*move)
    return {
        (axis, sq, sr): (m0, m1)
        for axis, sq, sr, m0, m1 in succ.windows_through(*move)
    }


def _expected_row(pre_own: int, pre_opp: int, slot: int):
    """The class and status an action row must carry, from PRE-insert masks."""
    has_own, has_opp = pre_own > 0, pre_opp > 0
    if has_own and not has_opp:
        status = ACTION_OWN
    elif has_opp and not has_own:
        status = ACTION_OPP
    elif has_own and has_opp:
        status = ACTION_MIXED
    else:
        status = ACTION_EMPTY
    post = sum(
        (1 if (pre_own >> j) & 1 else 2 if (pre_opp >> j) & 1 else 0) * 3**j
        for j in range(WINDOW_LEN)
    ) + 3**slot
    return int(_TERN_POST1_CLASS[post, slot]), status


def test_action_rows_match_the_successor_board_oracle():
    """Every emitted row agrees with actually playing the action: the class
    recomputed from the successor's engine windows, the status from the pre
    masks, and the window index from the graph's own kept-window identity."""
    rows_checked = 0
    for moves in _GAMES:
        pos = hexo_py.Position.replay(moves)
        mover = pos.current_player
        graph = from_position(pos)
        window_ids = [tuple(map(int, row)) for row in graph.window_id]
        legal = pos.legal_moves()
        picks = range(len(legal)) if len(legal) <= 40 else range(0, len(legal), 7)
        for a in picks:
            move = legal[a]
            oracle = _successor_windows(pos, move)
            for axis, (dq, dr) in enumerate(AXES):
                for k in range(WINDOW_LEN):
                    start = (move[0] - k * dq, move[1] - k * dr)
                    got_class = int(graph.action_post1_class[a, axis, k])
                    got_status = int(graph.action_pre_status[a, axis, k])
                    got_index = int(graph.action_window_index[a, axis, k])

                    key = (axis, start[0], start[1])
                    if key not in oracle:
                        # Off the engine's valid-coordinate domain: the walk
                        # still emits the row from an all-empty line edge.
                        continue
                    m0, m1 = oracle[key]
                    own_post, opp_post = (m0, m1) if mover == 0 else (m1, m0)
                    assert (own_post >> k) & 1, "the played stone is missing"
                    pre_own = own_post & ~(1 << k)
                    want_class, want_status = _expected_row(pre_own, opp_post, k)
                    assert got_class == want_class, (moves, move, axis, k)
                    assert got_status == want_status, (moves, move, axis, k)

                    kept = want_status != ACTION_EMPTY
                    assert (got_index >= 0) == kept
                    if got_index >= 0:
                        assert window_ids[got_index] == key
                    rows_checked += 1
    assert rows_checked > 500


def test_action_row_tables_are_d6_invariant_as_multisets():
    """A transformed board's rows are the same multiset of (class, status)
    per action — the classes are reversal orbits, so the multiset survives
    any axis permutation or line reversal the transform induces."""
    from mantisnet.klent import telemetry

    for moves in _GAMES[2:]:
        pos = hexo_py.Position.replay(moves)
        graph = from_position(pos)
        base = {
            move: sorted(
                zip(
                    graph.action_post1_class[a].ravel().tolist(),
                    graph.action_pre_status[a].ravel().tolist(),
                )
            )
            for a, move in enumerate(pos.legal_moves())
        }
        for transform in telemetry.D6_TRANSFORMS[1:3]:
            turned_pos = hexo_py.Position.replay([transform(m) for m in moves])
            turned = from_position(turned_pos)
            turned_rows = {
                move: sorted(
                    zip(
                        turned.action_post1_class[a].ravel().tolist(),
                        turned.action_pre_status[a].ravel().tolist(),
                    )
                )
                for a, move in enumerate(turned_pos.legal_moves())
            }
            for move, rows in base.items():
                assert turned_rows[transform(move)] == rows


def test_ply0_rows_are_empty_inserts():
    graph = from_position(hexo_py.Position())
    assert graph.action_window_index.shape == (1, 3, 6)
    assert (graph.action_window_index == -1).all()
    assert (graph.action_pre_status == ACTION_EMPTY).all()
    classes = np.array([_expected_row(0, 0, slot)[0] for slot in range(WINDOW_LEN)])
    assert (graph.action_post1_class == classes[None, None, :]).all()


# --------------------------------------------------------------------------
# Collation


def _graphs():
    return [
        from_position(hexo_py.Position.replay(moves))
        for moves in _GAMES
    ]


def test_collated_action_fields_match_the_dense_tables():
    """`act_class`, `act_rev`, and `act_empty` recomputed independently from
    the per-position dense tables."""
    graphs = _graphs()
    batch = collate(graphs)

    classes, counts, cell_off = [], [], 0
    for graph in graphs:
        status = graph.action_pre_status
        for a in range(graph.n_legal):
            for axis in range(3):
                for k in range(WINDOW_LEN):
                    if status[a, axis, k] != ACTION_EMPTY:
                        classes.append(int(graph.action_post1_class[a, axis, k]))
        for a in range(graph.n_legal):
            row = [0, 0, 0]
            for axis in range(3):
                for k in range(WINDOW_LEN):
                    if status[a, axis, k] == ACTION_EMPTY:
                        row[min(k, WINDOW_LEN - 1 - k)] += 1
            counts.append(row)
        cell_off += graph.n_legal

    assert batch.act_class.tolist() == classes
    assert batch.act_empty.tolist() == counts
    # Every candidate row is kept or EMPTY: kept degree + empty count is 18.
    kept = torch.bincount(batch.dec_cell, minlength=batch.n_cells)
    assert torch.equal(kept + batch.act_empty.sum(dim=1), torch.full_like(kept, 18))

    # The reverse view is the stable window-major permutation.
    rev = batch.act_rev
    assert sorted(rev.tolist()) == list(range(len(batch.dec_window)))
    by_window = batch.dec_window.index_select(0, rev)
    assert torch.equal(by_window, by_window.sort(stable=True).values)
    same = by_window[1:] == by_window[:-1]
    assert (rev[1:][same] > rev[:-1][same]).all()


def test_rust_python_action_collation_parity():
    """Rust and Python builders agree on every field."""
    rust = builder.collate_prefixes(_GAMES, [len(g) for g in _GAMES])
    python = collate(_graphs())
    for name, value in vars(python).items():
        got = getattr(rust, name)
        if isinstance(value, torch.Tensor):
            assert got.dtype == value.dtype and torch.equal(got, value), name
        else:
            assert got == value, name


# --------------------------------------------------------------------------
# The model path


def _small_config(**kw):
    return MantisConfig(
        h=32, heads=2, blocks=2, policy_hidden=32, value_hidden=32,
        **kw,
    )


def test_empty_orbit_classes_are_the_three_empty_insert_orbits():
    assert len(ACTION_EMPTY_CLASSES) == 3
    for k in range(WINDOW_LEN):
        orbit = min(k, WINDOW_LEN - 1 - k)
        assert _TERN_POST1_CLASS[3**k, k] == ACTION_EMPTY_CLASSES[orbit]


def test_model_has_the_row_encoder():
    model = MantisNet(_small_config())
    names = {name for name, _ in model.named_parameters()}
    assert {"act_proj.weight", "act_proj.bias", "act_table.weight",
            "act_empty_base", "p_act.weight", "q_act.weight"} <= names
    assert model.act_table.weight.shape == (TERN_POST1_CLASSES, 32)


def test_forward_runs_and_initial_decoders_are_zero():
    torch.manual_seed(0)
    model = MantisNet(_small_config()).eval()
    batch = collate(_graphs())
    with torch.no_grad():
        out = model(batch, 0.2)
    assert out.policy_logits.shape[0] == batch.n_cells
    assert torch.isfinite(out.policy_logits).all()
    # Zero-initialized decoder outputs: exactly zero action values, and
    # constant policy logits within each position.
    assert torch.equal(out.q_values, torch.zeros_like(out.q_values))
    for lo, hi in zip(batch.legal_offsets[:-1], batch.legal_offsets[1:]):
        logits = out.policy_logits[lo:hi]
        assert torch.allclose(logits, logits[:1].expand_as(logits))


def test_gradients_reach_the_action_parameters():
    torch.manual_seed(3)
    model = MantisNet(_small_config())
    with torch.no_grad():
        for head in (model.mlp_p, model.mlp_q):
            torch.nn.init.normal_(head.out.weight, std=0.1)
    batch = collate(_graphs())
    out = model(batch, 0.2)
    (out.policy_logits.sum() + out.q_values.sum()).backward()
    for name in ("act_proj.weight", "act_table.weight", "act_empty_base",
                 "p_act.weight", "q_act.weight"):
        grad = dict(model.named_parameters())[name].grad
        assert grad is not None and grad.abs().sum() > 0, name


def test_outputs_are_d6_invariant():
    from mantisnet.klent import telemetry

    torch.manual_seed(1)
    model = MantisNet(_small_config()).eval()
    transform = telemetry.D6_TRANSFORMS[1]
    for moves in _GAMES[2:]:
        pos = hexo_py.Position.replay(moves)
        base = collate([from_position(pos)])
        turned_pos = hexo_py.Position.replay([transform(m) for m in moves])
        turned = collate([from_position(turned_pos)])
        with torch.no_grad():
            got = model(base, 0.2)
            got_turned = model(turned, 0.2)
        assert torch.allclose(got.value, got_turned.value, atol=1e-5)
        for head in ("policy_logits", "q_values"):
            base_map = dict(zip(pos.legal_moves(), getattr(got, head).tolist()))
            turned_map = dict(
                zip(turned_pos.legal_moves(), getattr(got_turned, head).tolist())
            )
            assert set(turned_map) == {transform(m) for m in base_map}
            for move, score in base_map.items():
                assert turned_map[transform(move)] == pytest.approx(score, abs=1e-5)


def test_checkpoint_round_trips_through_the_family_registry(tmp_path):
    from mantisnet.lab.families import infer_config, load_checkpoint

    cfg = MantisConfig()
    torch.manual_seed(2)
    model = MantisNet(cfg)
    inferred = infer_config(model.state_dict())
    assert inferred == cfg

    path = tmp_path / "rows.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "versions": {
                "RULES_VERSION": hexo_py.RULES_VERSION,
                "ACTION_ORDER_VERSION": hexo_py.ACTION_ORDER_VERSION,
            },
            "iteration": 3,
        },
        path,
    )
    loaded = load_checkpoint(path)
    assert loaded.config == cfg

    batch = collate(_graphs())
    with torch.no_grad():
        reference = MantisNet(cfg).eval()
        reference.load_state_dict(model.state_dict())
        expected = reference(batch, 0.2)
        got_policy, _score, got_q = loaded.model.cell_heads(
            *loaded.model.trunk(batch), batch, 0.2
        )
    assert torch.allclose(got_policy, expected.policy_logits, atol=1e-6)
    assert torch.allclose(got_q, expected.q_values, atol=1e-6)
