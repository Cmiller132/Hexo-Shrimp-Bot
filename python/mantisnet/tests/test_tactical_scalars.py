"""Step 5 tactical scalars: engine oracle, D6, collation, and knob wiring.

The eleven per-action values are recomputed here by actually playing each
action on a board copy and reading the successor's windows from the engine
walk — fully independent of the builder's mask arithmetic. The global threat
census is recomputed from the current position's own window walk. Bit
equality is demanded throughout: both builders divide small exact integers
in float32.
"""

from __future__ import annotations

import numpy as np
import torch

import hexo_py
from mantisnet.builder import (
    TACTICAL_FEATURES,
    WINDOW_LEN,
    _GLOBAL_THREAT_CAP,
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
    # Turn order is 1-then-2: plies 0,3,4,7,8 are P0, plies 1,2,5,6,9,10 P1.
    # P1 holds an open five (0,2)..(4,2) and P0 moves: the threat census and
    # both five-window ends fire.
    [(0, 0), (0, 2), (1, 2), (-3, -3), (-4, -4), (2, 2), (3, 2),
     (-5, -5), (-6, -6), (4, 2), (0, 5)],
    # As above with P0 already on (-1, 2): exactly one five-window remains,
    # so playing (5, 2) hits every immediate threat — the blocks-all flag.
    [(0, 0), (0, 2), (1, 2), (-1, 2), (-3, -3), (2, 2), (3, 2),
     (-4, -4), (-5, -5), (4, 2), (0, 5)],
    [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, -1), (0, 2), (4, -1), (-1, 2)],
]

# D6 generators in axial coordinates: a rotation and a reflection.
_D6 = [
    lambda q, r: (q, r),
    lambda q, r: (-r, q + r),
    lambda q, r: (r, q),
    lambda q, r: (-q - r, r),
]


def _popcount(mask: int) -> int:
    return bin(mask).count("1")


def _mover_windows(pos):
    """Every live window of the current position, mover-relative, deduped."""
    mover = pos.current_player
    seen = {}
    for stone in pos.stones():
        for axis, sq, sr, m0, m1 in pos.windows_through(stone[0], stone[1]):
            own, opp = (m0, m1) if mover == 0 else (m1, m0)
            if own | opp:
                seen[(axis, sq, sr)] = (own, opp)
    return seen


def _oracle_features(pos, move) -> np.ndarray:
    """The eleven values recomputed from engine walks alone."""
    mover = pos.current_player
    succ = pos.copy()
    succ.advance(*move)
    post = {
        (axis, sq, sr): (m0, m1)
        for axis, sq, sr, m0, m1 in succ.windows_through(*move)
    }

    live = _mover_windows(pos)
    five_remaining = sum(
        1 for own, opp in live.values() if own == 0 and _popcount(opp) == 5
    )
    four_remaining = sum(
        1 for own, opp in live.values() if own == 0 and _popcount(opp) == 4
    )

    immediate = False
    max_own_after = 0
    max_opp_before = 0
    own_five = own_four = 0
    opp_five_hit = opp_four_hit = 0
    nonempty = 0
    for axis, (dq, dr) in enumerate(AXES):
        for k in range(WINDOW_LEN):
            key = (axis, move[0] - k * dq, move[1] - k * dr)
            if key not in post:
                # Off the engine coordinate domain: an all-empty line edge.
                max_own_after = max(max_own_after, 1)
                continue
            m0, m1 = post[key]
            own_post, opp = (m0, m1) if mover == 0 else (m1, m0)
            assert (own_post >> k) & 1, "the played stone is missing"
            pre_own = own_post & ~(1 << k)
            own_count, opp_count = _popcount(pre_own), _popcount(opp)
            immediate |= opp == 0 and own_count == WINDOW_LEN - 1
            max_own_after = max(max_own_after, own_count + 1)
            max_opp_before = max(max_opp_before, opp_count)
            if opp == 0:
                own_five += own_count + 1 == 5
                own_four += own_count + 1 == 4
            if pre_own == 0 and opp != 0:
                opp_five_hit += opp_count == 5
                opp_four_hit += opp_count == 4
            nonempty += (pre_own | opp) != 0

    def frac(count: int, total: int) -> np.float32:
        return np.float32(count) / np.float32(total)

    rows = 3 * WINDOW_LEN
    return np.array(
        [
            np.float32(immediate),
            frac(max_own_after, WINDOW_LEN),
            frac(max_opp_before, WINDOW_LEN),
            frac(own_five, rows),
            frac(own_four, rows),
            frac(opp_five_hit, rows),
            frac(opp_four_hit, rows),
            frac(min(five_remaining, _GLOBAL_THREAT_CAP), _GLOBAL_THREAT_CAP),
            frac(min(four_remaining, _GLOBAL_THREAT_CAP), _GLOBAL_THREAT_CAP),
            np.float32(five_remaining > 0 and opp_five_hit == five_remaining),
            frac(nonempty, rows),
        ],
        dtype=np.float32,
    )


def test_tactical_matches_the_engine_oracle():
    rows_checked = 0
    threat_rows = 0
    blocks_all_rows = 0
    for moves in _GAMES:
        pos = hexo_py.Position.replay(moves)
        graph = from_position(pos)
        legal = pos.legal_moves()
        # Threat games sweep every action: the one blocking cell must not
        # fall between subsample strides.
        threat_game = len(moves) == 11
        picks = (
            range(len(legal))
            if threat_game or len(legal) <= 40
            else range(0, len(legal), 7)
        )
        for a in picks:
            want = _oracle_features(pos, legal[a])
            got = graph.action_tactical[a]
            assert got.dtype == np.float32
            assert np.array_equal(got, want), (moves, legal[a], got, want)
            rows_checked += 1
            threat_rows += want[7] > 0
            blocks_all_rows += want[9] == 1.0
    assert rows_checked > 100
    # The game list must actually exercise the census and the flag.
    assert threat_rows > 0
    assert blocks_all_rows > 0


def test_ply_zero_tactical_is_the_empty_board_constant():
    graph = from_position(hexo_py.Position())
    assert graph.action_tactical.shape == (graph.n_legal, TACTICAL_FEATURES)
    want = np.zeros(TACTICAL_FEATURES, dtype=np.float32)
    want[1] = np.float32(1) / np.float32(WINDOW_LEN)
    assert np.array_equal(graph.action_tactical, np.tile(want, (graph.n_legal, 1)))


def test_tactical_is_d6_invariant_as_a_multiset():
    for moves in _GAMES[2:]:
        base = None
        for transform in _D6:
            mapped = [transform(q, r) for q, r in moves]
            graph = from_position(hexo_py.Position.replay(mapped))
            rows = sorted(map(tuple, graph.action_tactical.tolist()))
            if base is None:
                base = rows
            else:
                assert rows == base, moves


def test_collated_tactical_concatenates_the_dense_tables():
    graphs = [from_position(hexo_py.Position.replay(m)) for m in _GAMES]
    batch = collate(graphs)
    want = np.concatenate([g.action_tactical for g in graphs])
    assert batch.act_tactical.dtype == torch.float32
    assert np.array_equal(batch.act_tactical.numpy(), want)


def _tiny(**overrides) -> MantisConfig:
    return MantisConfig(
        h=32, blocks=1, heads=2, policy_hidden=16, value_hidden=16, **overrides
    )


def test_knob_off_is_byte_identical_to_the_incumbent():
    torch.manual_seed(2026)
    incumbent = MantisNet(_tiny())
    torch.manual_seed(2026)
    explicit_off = MantisNet(_tiny(action_tactical=False))
    left, right = incumbent.state_dict(), explicit_off.state_dict()
    assert list(left) == list(right)
    for a, b in zip(left.values(), right.values()):
        assert torch.equal(a, b)
    assert not any("tactical" in name for name in left)

    batch = collate([from_position(hexo_py.Position.replay(m)) for m in _GAMES])
    incumbent.eval(), explicit_off.eval()
    with torch.no_grad():
        out_a = incumbent(batch, 0.2)
        out_b = explicit_off(batch, 0.2)
    for name in vars(out_a):
        va, vb = getattr(out_a, name), getattr(out_b, name)
        assert va.cpu().numpy().tobytes() == vb.cpu().numpy().tobytes(), name


def test_knob_on_starts_at_the_incumbent_function():
    """Zero-init on the tactical output makes the added term exactly zero:
    a knob-on model whose shared parameters equal a knob-off model's computes
    the identical function at initialization."""
    torch.manual_seed(7)
    off = MantisNet(_tiny())
    aligned = MantisNet(_tiny(action_tactical=True))
    aligned.load_state_dict(
        {**aligned.state_dict(), **off.state_dict()}, strict=True
    )
    batch = collate([from_position(hexo_py.Position.replay(m)) for m in _GAMES])
    off.eval(), aligned.eval()
    with torch.no_grad():
        out_off = off(batch, 0.2)
        out_aligned = aligned(batch, 0.2)
    for name in vars(out_off):
        va, vb = getattr(out_off, name), getattr(out_aligned, name)
        assert torch.equal(va, vb), name


def test_gradients_reach_the_tactical_parameters():
    torch.manual_seed(3)
    model = MantisNet(_tiny(action_tactical=True))
    # The zero-init decoder outputs block upstream flow, and the zero-init
    # tactical output likewise gates its own first layer; perturb both first.
    for head in (model.mlp_p, model.mlp_q):
        torch.nn.init.normal_(head.out.weight, std=0.05)
    torch.nn.init.normal_(model.tactical_out.weight, std=0.05)
    batch = collate([from_position(hexo_py.Position.replay(m)) for m in _GAMES])
    out = model(batch, 0.2)
    (out.policy_logits.sum() + out.q_values.sum()).backward()
    for name in ("tactical_a.weight", "tactical_a.bias", "tactical_out.weight"):
        grad = dict(model.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all(), name
        assert grad.abs().sum() > 0, name


def test_parameter_counts_pin_the_knob():
    base = sum(p.numel() for p in MantisNet(MantisConfig()).parameters())
    on = sum(
        p.numel()
        for p in MantisNet(MantisConfig(action_tactical=True)).parameters()
    )
    assert base == 4_804_213
    h = MantisConfig().h
    assert on - base == (h * TACTICAL_FEATURES + h) + (h * h + h)
    assert on == 4_822_261
