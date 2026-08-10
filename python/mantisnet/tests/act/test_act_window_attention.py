"""§16's typed window↔window attention: its edges, its classes, and its law.

What is being detected here, and why each detector is independent of the code
it watches:

- **The edge set (§16).** :func:`naive_pairs` re-derives every directed edge and
  its class from §16's own definitions — a collinear pair at a signed start
  offset of at most eleven, a crossing pair at the one lattice cell two
  non-parallel hex lines meet in, and the two per-side folds — by inspecting
  every ordered pair of windows and every cell each of them claims. It shares no
  table, no key packing and no join with `window_pairs`, whose whole method is
  the sorted joins this replaces with `for i: for j:`. A wrong reach, a wrong
  fold, a dropped direction or a group the join lets an offset escape from all
  show up as a set difference.
- **The class is a D6 invariant.** The same game is replayed through each of the
  twelve transforms and the two edge sets are matched window for window. That is
  the claim the axis stream's shared bias table rests on: if a class moved under
  a rotation that reverses the two lines independently, the arm would not be
  equivariant and no forward-shape test would say so.
- **The forward, against a dense oracle.** :func:`dense_attention` computes the
  module's own formula from an ``(N_w, N_w)`` class matrix built out of
  ``naive_pairs``, with plain torch ops and plain autograd. It is the parity
  reference for both streams and the autograd oracle for
  `window_pairs`' custom backward as this arm calls it — in particular for the
  claim that channel ``a`` of a destination attends to channel ``a`` of its
  sources through one shared set of projections.
- **§12.1 on the whole trunk under this arm**, through the builder and the real
  engine, which is what a wrong channel route would fail.

Positions come from seeded random playouts through the engine. As
``docs/MANTIS_ACT_DEVIATIONS.md`` records, random play is nothing like the
self-play density this model will see, so nothing here asserts a family size —
the counts are only asked to be nonzero where a path would otherwise be dead
code.
"""

from __future__ import annotations

import math
import random
from dataclasses import replace

import hexo_py
import numpy as np
import pytest
import torch

from mantisnet.models.mantis_act.builder import build
from mantisnet.models.mantis_act.config import PRESETS
from mantisnet.models.mantis_act.equivariant import (
    AXIS_CHANNELS,
    EquivariantState,
    permute_axis_channels,
)
from mantisnet.models.mantis_act.messages import (
    WINDOW_WINDOW_RELATIONS,
    TypedWindowAttention,
    window_window_edges,
)
from mantisnet.models.mantis_act.packed import collate
from mantisnet.models.mantis_act.state_trunk import StateTrunk, state_edges
from mantisnet.models.mantis_act.symmetry import (
    AXES,
    D6_TRANSFORMS,
    axis_permutation,
    transform_coords,
)
from mantisnet.models.mantis_act.windows import WINDOW_COORD_LIMIT, window_cells

SEED = 20260807

FULL = PRESETS["full_act_v4"]
TYPED = PRESETS["full_with_typed_window_attention"]

# Dense enough for both class families and both movers, small enough that the
# oracle's ordered-pair walk over the windows stays quick.
PLIES = (7, 21, 40)

# §16's own numbers, restated here because the oracle may not read the
# implementation's: eleven collinear classes at offsets 1..11, then the 6x6
# crossing fold product, then the self loop.
COLLINEAR_CLASSES = 11
REACH = 5
SELF_CLASS = COLLINEAR_CLASSES + 36


# --------------------------------------------------------------------------
# The independent oracle


def fold(t: int) -> int:
    """§16's per-side fold of a crossing's slot parameter.

    In the span, a slot folds to its distance from the nearer end, so a
    reflection — which maps ``t`` to ``5 - t`` — leaves it alone. Outside it,
    the distance past the end folds the same way and is offset above the
    in-span values, so the two cases never collide. Six values either way.
    """
    if 0 <= t <= 5:
        return min(t, 5 - t)
    return 2 + min(max(-t, t - 5), 3)


def naive_pairs(window_id: np.ndarray) -> set[tuple[int, int, int]]:
    """Every directed ``(dst, src, class)`` of §16, by ordered-pair inspection.

    Quadratic in the windows on purpose: it is the statement of what the edge
    set *is*, against which the sorted joins are the optimisation.
    """
    window_id = np.asarray(window_id, dtype=np.int64).reshape(-1, 3)
    claimed = []
    for axis, start_q, start_r in window_id.tolist():
        step = AXES[axis]
        claimed.append(
            {
                (start_q + t * int(step[0]), start_r + t * int(step[1])): t
                for t in range(-REACH, 6 + REACH)
            }
        )

    edges: set[tuple[int, int, int]] = set()
    for i, (axis_i, qi, ri) in enumerate(window_id.tolist()):
        edges.add((i, i, SELF_CLASS))
        for j, (axis_j, qj, rj) in enumerate(window_id.tolist()):
            if i == j:
                continue
            if axis_i == axis_j:
                step = AXES[axis_i]
                delta = (qj - qi, rj - ri)
                lead = 0 if int(step[0]) else 1
                offset, remainder = divmod(delta[lead], int(step[lead]))
                other = 1 - lead
                if remainder or offset * int(step[other]) != delta[other]:
                    continue  # parallel lines, not the same one
                if 1 <= abs(offset) <= COLLINEAR_CLASSES:
                    # dst i reads src j; the class is the unsigned offset,
                    # because a reflection re-signs it and nothing else does.
                    edges.add((i, j, abs(offset) - 1))
                continue
            # Two non-parallel hex lines meet in exactly one lattice cell, so
            # at most one shared claimed cell survives the intersection.
            shared = set(claimed[i]) & set(claimed[j])
            for cell in shared:
                edges.add(
                    (
                        i,
                        j,
                        COLLINEAR_CLASSES + fold(claimed[i][cell]) * 6 + fold(claimed[j][cell]),
                    )
                )
    return edges


def derived_pairs(batch) -> set[tuple[int, int, int]]:
    """The same set, as `window_window_edges` produces it."""
    edges = window_window_edges(batch)
    counts = edges.ptr[1:] - edges.ptr[:-1]
    dst = torch.repeat_interleave(torch.arange(edges.n_windows), counts)
    return {
        (int(d), int(s), int(c))
        for d, s, c in zip(dst.tolist(), edges.src.tolist(), edges.cls.tolist())
    }


def dense_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    classes: torch.Tensor,
) -> torch.Tensor:
    """``(N_w, heads, head_dim)`` attention over a dense masked class matrix.

    ``classes`` is ``(N_w, N_w)`` with ``-1`` where there is no edge. Plain
    torch, plain autograd, no CSR view and no kernel — the parity reference for
    the module's own formula.
    """
    head_dim = q.shape[2]
    score = torch.einsum("dhc,shc->dsh", q.float(), k.float()) / math.sqrt(head_dim)
    live = classes >= 0
    score = score + bias.t().float()[classes.clamp(min=0)]
    score = score.masked_fill(~live.unsqueeze(-1), float("-inf"))
    weight = torch.softmax(score, dim=1)
    return torch.einsum("dsh,shc->dhc", weight, v.float())


def class_matrix(window_id: np.ndarray) -> torch.Tensor:
    """``naive_pairs`` as the ``(N_w, N_w)`` matrix `dense_attention` reads."""
    n = len(window_id)
    matrix = torch.full((n, n), -1, dtype=torch.long)
    for dst, src, cls in naive_pairs(window_id):
        matrix[dst, src] = cls
    return matrix


# --------------------------------------------------------------------------
# Fixtures


def playout(plies: int, seed: int) -> list[tuple[int, int]]:
    """A seeded nonterminal random playout of exactly ``plies`` placements."""
    for attempt in range(100):
        rng = random.Random(seed * 7919 + attempt * 31 + plies)
        position = hexo_py.Position()
        moves: list[tuple[int, int]] = []
        for _ in range(plies):
            move = rng.choice(position.legal_moves())
            position.advance(*move)
            moves.append(move)
        if not position.is_terminal:
            return moves
    raise AssertionError(f"no nonterminal {plies}-ply playout in 100 seeds")


@pytest.fixture(scope="module")
def move_lists() -> dict[int, list[tuple[int, int]]]:
    return {plies: playout(plies, SEED) for plies in PLIES}


@pytest.fixture(scope="module")
def graphs(move_lists):
    return {
        plies: build(hexo_py.Position.replay(moves), TYPED)
        for plies, moves in move_lists.items()
    }


@pytest.fixture(scope="module")
def batch(graphs):
    return collate([graphs[plies] for plies in PLIES], TYPED)


# --------------------------------------------------------------------------
# The edge set (§16)


@pytest.mark.parametrize("plies", PLIES)
def test_the_pair_join_matches_the_ordered_pair_oracle(graphs, plies):
    """§16's two class families, against a quadratic statement of what they are."""
    graph = graphs[plies]
    assert graph.n_windows > 0
    got = derived_pairs(collate([graph], TYPED))
    want = naive_pairs(graph.window_id)
    assert got == want
    # Both families and the self loop are actually present, so a match is not a
    # match of two empty sets.
    kinds = {cls for _dst, _src, cls in got}
    assert any(cls < COLLINEAR_CLASSES for cls in kinds)
    assert any(COLLINEAR_CLASSES <= cls < SELF_CLASS for cls in kinds)
    assert SELF_CLASS in kinds
    assert max(kinds) < WINDOW_WINDOW_RELATIONS


def test_the_pair_join_is_a_batch_of_per_position_joins(graphs, batch):
    """§26: no edge may cross a position, and the batch is the concatenation."""
    offsets = batch.window_offsets.tolist()
    got = derived_pairs(batch)
    want = set()
    for index, plies in enumerate(PLIES):
        base = offsets[index]
        for dst, src, cls in naive_pairs(graphs[plies].window_id):
            want.add((dst + base, src + base, cls))
    assert got == want


def test_every_window_is_its_own_source(batch):
    """The self loop is what keeps every softmax segment nonempty."""
    edges = window_window_edges(batch)
    counts = edges.ptr[1:] - edges.ptr[:-1]
    assert int(counts.min()) >= 1
    dst = torch.repeat_interleave(torch.arange(edges.n_windows), counts)
    loops = edges.cls == SELF_CLASS
    assert torch.equal(dst[loops], torch.arange(edges.n_windows))
    assert torch.equal(edges.src[loops], torch.arange(edges.n_windows))


@pytest.mark.parametrize("transform_index", range(len(D6_TRANSFORMS)))
def test_every_relation_class_survives_every_d6_transform(move_lists, transform_index):
    """The invariance the shared per-class bias table rests on (§12.1, §16).

    A 60-degree rotation permutes the three axes and may reverse the two lines
    of a crossing independently. If either side's fold coupled to the other's
    orientation, the transformed board would carry a different class for the
    same pair of windows, and the arm would not be equivariant.
    """
    moves = move_lists[21]
    base = build(hexo_py.Position.replay(moves), TYPED)
    turned = build(
        hexo_py.Position.replay(
            [D6_TRANSFORMS[transform_index](move) for move in moves]
        ),
        TYPED,
    )
    assert base.n_windows == turned.n_windows > 0

    windows = window_correspondence(base, turned, transform_index)
    mapped = {
        (int(windows[dst]), int(windows[src]), cls)
        for dst, src, cls in derived_pairs(collate([base], TYPED))
    }
    assert mapped == derived_pairs(collate([turned], TYPED))


def window_correspondence(base, image, transform_index) -> np.ndarray:
    """Row of ``image`` holding the transform of each window of ``base``.

    A window is matched by its cell *set*: a transform may reverse a window's
    slot order, so its stored identity is not simply the transform of the
    original's, but the six coordinates it covers are.
    """

    def key(window_id):
        cells = window_cells(window_id)
        return [tuple(sorted(tuple(qr) for qr in row.tolist())) for row in cells]

    rows = {shape: index for index, shape in enumerate(key(image.window_id))}
    turned = transform_coords(
        transform_index, window_cells(base.window_id).reshape(-1, 2)
    ).reshape(-1, 6, 2)
    shapes = [tuple(sorted(tuple(qr) for qr in row.tolist())) for row in turned]
    return np.array([rows[shape] for shape in shapes], dtype=np.int64)


# --------------------------------------------------------------------------
# The attention itself


def state_of(n_windows: int, cfg, generator) -> EquivariantState:
    return EquivariantState(
        torch.randn(n_windows, cfg.d_inv, generator=generator),
        torch.randn(n_windows, AXIS_CHANNELS, cfg.d_axis, generator=generator),
    )


def test_both_streams_match_the_dense_oracle(graphs):
    """Parity, and the claim that a channel attends only within itself."""
    graph = graphs[21]
    packed = collate([graph], TYPED)
    generator = torch.Generator().manual_seed(SEED)
    torch.manual_seed(SEED)

    module = TypedWindowAttention(TYPED).eval()
    with torch.no_grad():
        module.bias_inv.normal_(0.0, 0.5, generator=generator)
        module.bias_axis.normal_(0.0, 0.5, generator=generator)
    windows = state_of(graph.n_windows, TYPED, generator)
    out = module(window_window_edges(packed), windows)

    classes = class_matrix(graph.window_id)
    z = module.norm(windows)
    heads, n = TYPED.num_heads, graph.n_windows

    def split(tensor, width):
        return tensor.reshape(n, heads, width // heads)

    want_inv = dense_attention(
        split(module.q_inv(z.inv), TYPED.d_inv),
        split(module.k_inv(z.inv), TYPED.d_inv),
        split(module.v_inv(z.inv), TYPED.d_inv),
        module.bias_inv,
        classes,
    ).reshape(n, TYPED.d_inv)
    expect_inv = windows.inv + module.residual.inv(module.out_inv(want_inv))
    torch.testing.assert_close(out.inv, expect_inv, atol=2e-5, rtol=1e-4)

    # The axis stream, one channel at a time and with the *same* bias rows —
    # which is the per-channel-parameter-free construction of §12.2.
    channels = []
    for channel in range(AXIS_CHANNELS):
        slice_ = z.axis[:, channel]
        channels.append(
            dense_attention(
                split(module.q_axis(slice_), TYPED.d_axis),
                split(module.k_axis(slice_), TYPED.d_axis),
                split(module.v_axis(slice_), TYPED.d_axis),
                module.bias_axis,
                classes,
            ).reshape(n, TYPED.d_axis)
        )
    want_axis = torch.stack(channels, dim=1)
    expect_axis = windows.axis + module.residual.axis(module.out_axis(want_axis))
    torch.testing.assert_close(out.axis, expect_axis, atol=2e-5, rtol=1e-4)


def test_the_gradients_match_the_dense_oracles(graphs):
    """The custom backward, as this arm calls it, against plain autograd."""
    graph = graphs[7]
    packed = collate([graph], TYPED)
    classes = class_matrix(graph.window_id)
    n, heads = graph.n_windows, 2
    head_dim = 5

    def parameters():
        torch.manual_seed(SEED)
        return [
            torch.randn(n, heads, head_dim, requires_grad=True) for _ in range(3)
        ] + [torch.randn(heads, WINDOW_WINDOW_RELATIONS, requires_grad=True)]

    from mantisnet.window_pairs import edge_attention

    fast = parameters()
    edges = window_window_edges(packed)
    edge_attention(*fast, *edges.views()).square().sum().backward()

    slow = parameters()
    dense_attention(*slow, classes).square().sum().backward()

    for got, want, name in zip(fast, slow, ("q", "k", "v", "bias")):
        torch.testing.assert_close(got.grad, want.grad, atol=2e-4, rtol=1e-3, msg=name)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the flash path needs a real device"
)
def test_the_device_path_matches_the_host_path(graphs):
    """The Triton kernels against the sliced reference, on the same weights.

    `window_pairs`' `_supported` guard sends a CPU tensor to the reference
    composition and a CUDA one to the flash kernels, so on this hardware the two
    are different code and only a cross-device comparison holds them together.
    """
    graph = graphs[21]
    packed = collate([graph], TYPED)
    torch.manual_seed(SEED)
    module = TypedWindowAttention(TYPED).eval()
    with torch.no_grad():
        module.bias_inv.normal_(0.0, 0.5)
        module.bias_axis.normal_(0.0, 0.5)
    windows = state_of(graph.n_windows, TYPED, torch.Generator().manual_seed(SEED))

    host = module(window_window_edges(packed), windows)
    device = module.cuda()(
        window_window_edges(packed.to("cuda")),
        EquivariantState(windows.inv.cuda(), windows.axis.cuda()),
    )
    torch.testing.assert_close(device.inv.cpu(), host.inv, atol=2e-5, rtol=1e-4)
    torch.testing.assert_close(device.axis.cpu(), host.axis, atol=2e-5, rtol=1e-4)


def test_the_axis_bias_is_one_table_shared_by_the_three_channels():
    """Three tables would be §12.2's forbidden per-absolute-axis parameter."""
    module = TypedWindowAttention(TYPED)
    assert module.bias_axis.shape == (TYPED.num_heads, WINDOW_WINDOW_RELATIONS)
    assert module.bias_inv.shape == (TYPED.num_heads, WINDOW_WINDOW_RELATIONS)
    # Zero at init, so every destination starts on uniform weights (§27).
    assert torch.count_nonzero(module.bias_axis) == 0
    assert torch.count_nonzero(module.bias_inv) == 0


def test_permuting_the_channels_permutes_the_axis_output(graphs):
    """§12.1 on the stage alone, without the builder in the way."""
    graph = graphs[21]
    edges = window_window_edges(collate([graph], TYPED))
    generator = torch.Generator().manual_seed(SEED)
    torch.manual_seed(SEED)
    module = TypedWindowAttention(TYPED).eval()
    windows = state_of(graph.n_windows, TYPED, generator)

    before = module(edges, windows)
    for permutation in ((1, 2, 0), (2, 0, 1), (0, 2, 1), (2, 1, 0), (1, 0, 2)):
        after = module(edges, windows.permute_axes(permutation))
        torch.testing.assert_close(after.inv, before.inv, atol=2e-5, rtol=1e-4)
        torch.testing.assert_close(
            after.axis,
            permute_axis_channels(before.axis, permutation),
            atol=2e-5,
            rtol=1e-4,
        )


def test_the_module_refuses_a_state_or_an_edge_set_that_is_not_its_own(graphs):
    module = TypedWindowAttention(TYPED)
    edges = window_window_edges(collate([graphs[21]], TYPED))
    generator = torch.Generator().manual_seed(SEED)
    with pytest.raises(ValueError, match="windows against the edge family"):
        module(edges, state_of(edges.n_windows + 1, TYPED, generator))
    with pytest.raises(ValueError, match="d_inv"):
        module(edges, state_of(edges.n_windows, replace(TYPED, d_inv=32), generator))
    with pytest.raises(ValueError, match="does not ask for typed window attention"):
        TypedWindowAttention(FULL)


def test_an_axis_width_the_heads_do_not_divide_is_refused():
    """A silent floor here would drop a slice of the axis stream."""
    with pytest.raises(ValueError, match="must divide into num_heads"):
        TypedWindowAttention(replace(TYPED, d_axis=25))


# --------------------------------------------------------------------------
# The trunk under this arm (§12.1, §18.5)


@pytest.mark.parametrize("transform_index", range(len(D6_TRANSFORMS)))
def test_the_typed_trunk_maps_under_every_d6_transform(move_lists, transform_index):
    """§31.4-§31.7 with step 5 on, through the builder and the real engine."""
    moves = move_lists[21]
    base = build(hexo_py.Position.replay(moves), TYPED)
    turned = build(
        hexo_py.Position.replay(
            [D6_TRANSFORMS[transform_index](move) for move in moves]
        ),
        TYPED,
    )
    torch.manual_seed(SEED)
    trunk = StateTrunk(TYPED)
    with torch.no_grad():
        for name, parameter in trunk.named_parameters():
            if name.endswith("gamma") and parameter.dim() == 1:
                parameter.normal_(0.5, 0.2)
    trunk.eval()

    before = trunk(collate([base], TYPED))
    after = trunk(collate([turned], TYPED))
    windows = torch.from_numpy(window_correspondence(base, turned, transform_index))
    permutation = axis_permutation(transform_index)

    torch.testing.assert_close(
        after.windows.inv.index_select(0, windows), before.windows.inv, atol=2e-5, rtol=1e-4
    )
    torch.testing.assert_close(
        after.windows.axis.index_select(0, windows),
        permute_axis_channels(before.windows.axis, permutation),
        atol=2e-5,
        rtol=1e-4,
    )


def test_the_arm_changes_the_state_it_is_added_to(batch):
    """A stage that moved nothing would make ablation 13 compare two controls."""
    torch.manual_seed(SEED)
    typed = StateTrunk(TYPED).eval()
    with torch.no_grad():
        for block in typed.blocks:
            block.window_attention.bias_inv.normal_(0.0, 0.5)
    out = typed(batch)
    assert torch.isfinite(out.windows.inv).all()

    stripped = StateTrunk(TYPED).eval()
    stripped.load_state_dict(typed.state_dict())
    for block in stripped.blocks:
        block.window_attention = None
    assert not torch.allclose(stripped(batch).windows.inv, out.windows.inv)


def test_the_edge_family_is_derived_only_by_the_arm_that_reads_it(batch):
    """An arm that does not attend pays neither the join nor its syncs."""
    assert state_edges(batch, FULL).window_window is None
    assert state_edges(batch, TYPED).window_window is not None
    for name, cfg in PRESETS.items():
        typed = cfg.window_window_mode == "typed_collinear_crossing"
        assert (StateTrunk(cfg).blocks[0].window_attention is not None) == typed, name


# --------------------------------------------------------------------------
# The bound the join needs (§7, §16)


def test_a_window_outside_the_addressable_range_is_refused(graphs):
    """The pair key packs a *sum* of coordinates; both must stay inside 2**15.

    Past it the QR axis's line key wraps into the neighbouring axis's band and
    two windows on different lines silently become collinear partners — a fault
    that applies identically on both sides of the join and that no round trip
    over the edge set can see.
    """
    graph = graphs[7]
    outside = graph.window_id.copy()
    outside[0, 1] = WINDOW_COORD_LIMIT
    with pytest.raises(ValueError, match=r"window_id must be <="):
        replace(graph, window_id=outside)


def test_a_window_whose_two_axis_columns_disagree_is_refused(graphs):
    """The embedding routes by one column and the pair join reads the other.

    Neither is a shape or an index, so a disagreement would simply have the two
    stages describe different boards, and every tensor would keep its size.
    """
    graph = graphs[7]
    with pytest.raises(ValueError, match=r"in window_axis and .* in its identity"):
        replace(graph, window_axis=(graph.window_axis + 1) % 3)
