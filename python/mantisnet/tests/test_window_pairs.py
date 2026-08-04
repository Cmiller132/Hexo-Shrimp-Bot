"""§5.1c pair-table oracle, fold invariance, and the window-attention model.

The oracle re-derives every relation pairwise from the identity triples —
literal line intersection per pair, no sorted joins — so the vectorized
derivation and the brute force share nothing but the definition.
"""

from __future__ import annotations

import pytest
import torch

from mantisnet import MantisConfig, MantisNet, collate, from_position
from mantisnet.builder import AXES
from mantisnet.window_pairs import WA_CLASSES, pair_tables


def _fold(t: int) -> int:
    if 0 <= t <= 5:
        return min(t, 5 - t)
    return 2 + min(-t if t < 0 else t - 5, 3)


def _oracle_relation(a: tuple[int, int, int], b: tuple[int, int, int]):
    """The directed class of edge b -> a, or None. Brute force, one pair."""
    (ax_a, q_a, r_a), (ax_b, q_b, r_b) = a, b
    if ax_a == ax_b:
        # Same axis: related only when on one line, within offset 11.
        va = AXES[ax_a]
        # Colinear iff the start difference is an integer multiple of the axis.
        dq, dr = q_b - q_a, r_b - r_a
        if dq * int(va[1]) != dr * int(va[0]):
            return None
        offset = dq if int(va[0]) else dr
        if not 1 <= abs(offset) <= 11:
            return None
        return abs(offset) - 1
    # Crossing: solve start_a + t * v_a == start_b + u * v_b over the lattice.
    va, vb = AXES[ax_a], AXES[ax_b]
    det = int(va[0]) * int(vb[1]) - int(va[1]) * int(vb[0])
    assert det in (-1, 1), "hex axis pairs are unimodular"
    dq, dr = q_b - q_a, r_b - r_a
    t = (dq * int(vb[1]) - dr * int(vb[0])) // det
    u = -(int(va[0]) * dr - int(va[1]) * dq) // det
    assert (q_a + t * int(va[0]), r_a + t * int(va[1])) == (
        q_b + u * int(vb[0]),
        r_b + u * int(vb[1]),
    )
    if not (-5 <= t <= 10 and -5 <= u <= 10):
        return None
    return 11 + _fold(t) * 6 + _fold(u)


def test_forty_eight_classes():
    assert WA_CLASSES == 48  # 11 colinear + 6 * 6 crossing + SELF


def test_the_fold_is_invariant_under_line_reversal():
    # Reversal sends the span parameter t to 5 - t; each side of a crossing
    # class folds invariantly on its own, which is what makes the product
    # class D6-invariant under independent line reversals.
    for t in range(-5, 11):
        assert _fold(t) == _fold(5 - t)
    assert [_fold(t) for t in range(6)] == [0, 1, 2, 2, 1, 0]
    assert _fold(-1) == _fold(6) == 3
    assert _fold(-2) == _fold(7) == 4
    assert _fold(-5) == _fold(10) == 5


def test_pair_tables_match_the_brute_force(positions):
    for pos in positions:
        g = from_position(pos)
        ids = [tuple(map(int, row)) for row in g.window_id]
        n_w = len(ids)
        window_id = torch.from_numpy(g.window_id)
        ptr, src, cls = pair_tables(window_id, torch.zeros(n_w, dtype=torch.long))

        got = set()
        for dst in range(n_w):
            for e in range(int(ptr[dst]), int(ptr[dst + 1])):
                got.add((dst, int(src[e]), int(cls[e])))

        expected = {(w, w, 47) for w in range(n_w)}
        for i in range(n_w):
            for j in range(n_w):
                if i == j:
                    continue
                relation = _oracle_relation(ids[i], ids[j])
                if relation is not None:
                    expected.add((i, j, relation))
        assert got == expected


def test_pairs_never_cross_positions(positions):
    graphs = [from_position(pos) for pos in positions if from_position(pos).n_windows]
    batch = collate(graphs, pairs=True)
    # Stack the same graphs as one batch; every edge must stay inside its
    # position's window range.
    offsets = [0]
    for g in graphs:
        offsets.append(offsets[-1] + g.n_windows)
    dst = torch.repeat_interleave(
        torch.arange(batch.window_id.shape[0]), batch.wa_ptr[1:] - batch.wa_ptr[:-1]
    )
    lo = torch.tensor(offsets[:-1])
    hi = torch.tensor(offsets[1:])
    position = torch.searchsorted(hi, dst, right=True)
    assert torch.all(batch.wa_src >= lo[position])
    assert torch.all(batch.wa_src < hi[position])

    # Single-position tables agree with the batched ones for the first graph,
    # per destination as sets: the join visits edges in a different order when
    # other positions interleave the sort.
    single = collate([graphs[0]], pairs=True)
    n0 = graphs[0].n_windows
    assert torch.equal(single.wa_ptr, batch.wa_ptr[: n0 + 1])
    for w in range(n0):
        a, b = int(batch.wa_ptr[w]), int(batch.wa_ptr[w + 1])
        got = set(zip(batch.wa_src[a:b].tolist(), batch.wa_class[a:b].tolist()))
        want = set(zip(single.wa_src[a:b].tolist(), single.wa_class[a:b].tolist()))
        assert got == want


def test_pairs_are_opt_in():
    g = from_position(__import__("hexo_py").Position.replay([(0, 0), (1, 0), (0, 1)]))
    plain = collate([g])
    assert plain.wa_ptr is None and plain.wa_src is None and plain.wa_class is None
    with_pairs = collate([g], pairs=True)
    assert with_pairs.wa_ptr is not None
    # Every window has at least the SELF loop, exactly once.
    counts = with_pairs.wa_ptr[1:] - with_pairs.wa_ptr[:-1]
    assert torch.all(counts >= 1)
    self_edges = (
        with_pairs.wa_src
        == torch.repeat_interleave(torch.arange(g.n_windows), counts)
    ) & (with_pairs.wa_class == 47)
    assert int(self_edges.sum()) == g.n_windows


def _small_wa_config(**extra) -> MantisConfig:
    return MantisConfig(
        h=16,
        blocks=2,
        heads=2,
        value_queries=2,
        value_bins=5,
        policy_hidden=16,
        value_hidden=16,
        window_attention=True,
        **extra,
    )


@torch.no_grad()
def test_window_attention_refuses_a_batch_without_pair_tables(positions):
    net = MantisNet(_small_wa_config()).eval()
    batch = collate([from_position(positions[6])])
    with pytest.raises(RuntimeError, match="pairs=True"):
        net(batch, 0.2)


@torch.no_grad()
def test_window_attention_forward_and_batch_parity(positions):
    torch.manual_seed(11)
    net = MantisNet(_small_wa_config()).eval()
    graphs = [from_position(p) for p in positions]
    batch = collate(graphs, pairs=True)
    together = net(batch, 0.2)
    for tensor in vars(together).values():
        assert torch.isfinite(tensor).all()

    for i, g in enumerate(graphs):
        single = net(collate([g], pairs=True), 0.2)
        a, b = int(batch.legal_offsets[i]), int(batch.legal_offsets[i + 1])
        assert torch.allclose(
            together.policy_logits[a:b], single.policy_logits, atol=1e-6
        )
        assert torch.allclose(
            together.value[i : i + 1], single.value, atol=1e-6
        )


@torch.no_grad()
def test_window_attention_is_d6_invariant(positions, move_lists):
    import hexo_py

    from mantisnet.klent import telemetry

    torch.manual_seed(12)
    net = MantisNet(_small_wa_config()).eval()
    for moves in (move_lists[6], move_lists[8]):
        pos = hexo_py.Position.replay(moves)
        base = net(collate([from_position(pos)], pairs=True), 0.2)
        base_policy = dict(zip(pos.legal_moves(), base.policy_logits.tolist()))
        for transform in telemetry.D6_TRANSFORMS[1:]:
            t_pos = hexo_py.Position.replay([transform(m) for m in moves])
            got = net(collate([from_position(t_pos)], pairs=True), 0.2)
            assert torch.allclose(got.value, base.value, atol=1e-5)
            mapped = dict(zip(t_pos.legal_moves(), got.policy_logits.tolist()))
            for move, logit in base_policy.items():
                assert abs(mapped[transform(move)] - logit) <= 1e-5


def test_window_attention_gradients_reach_every_table(positions):
    torch.manual_seed(13)
    net = MantisNet(_small_wa_config())
    batch = collate([from_position(positions[8])], pairs=True)
    out = net(batch, 0.2)
    (out.value_logits.sum() + out.policy_logits.sum()).backward()
    for name in ("wq_wa", "wk_wa", "wv_wa", "wo_wa"):
        grad = getattr(net.blocks[0], name).weight.grad
        assert grad is not None and torch.isfinite(grad).all()
    bias_grad = net.blocks[0].wa_bias.grad
    assert bias_grad is not None and torch.isfinite(bias_grad).all()
