"""§5.1/§5.2 CUDA message-passing parity, gradients, and compilation.

The oracle is the literal torch formulation that preceded the CUDA path:
gather the source rows, add the incidence-class embedding, then ``index_add_``
into the destination rows.  The production decomposition since split the class
term into ``incidence_counts @ class_weight``; every composition test here
still compares against that one historical formula, so the split cannot
quietly change the §5 math.  The oracle deliberately does not use the
implementation's run discovery or any reordered incidence table.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import mantisnet.message_passing as message_impl
from mantisnet import TERN_OCC_CLASSES, collate, from_position
from mantisnet.builder import ACTION_EMPTY
from mantisnet.message_passing import (
    STONE_RUN,
    WINDOW_RUN,
    aggregate_to_stones,
    aggregate_to_windows,
    class_row_sum,
    incidence_plan,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the message-passing parity cases require CUDA"
)

_DEVICE = torch.device("cuda")
_H = 128


def _old_scatter(
    values: torch.Tensor,
    class_weight: torch.Tensor,
    source: torch.Tensor,
    destination: torch.Tensor,
    inc_class: torch.Tensor,
    n_dest: int,
) -> torch.Tensor:
    """The pre-kernel gather/add/scatter, independent of kernel preprocessing."""
    msg = values.index_select(0, source) + class_weight.index_select(0, inc_class)
    return msg.new_zeros((n_dest, values.shape[1])).index_add_(
        0, destination, msg
    )


def _values_scatter(
    values: torch.Tensor,
    source: torch.Tensor,
    destination: torch.Tensor,
    n_dest: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """The oracle's projected-values term: fp32 accumulate, one output cast."""
    msg = values.index_select(0, source).float()
    out = msg.new_zeros((n_dest, values.shape[1])).index_add_(
        0, destination, msg
    )
    return out.to(dtype)


def _composed(
    direction: str,
    values: torch.Tensor,
    class_weight: torch.Tensor,
    views: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    class_view: tuple[torch.Tensor, torch.Tensor, int],
    n_dest: int,
) -> torch.Tensor:
    """The production composition: values aggregation plus the class term."""
    fn = aggregate_to_windows if direction == "windows" else aggregate_to_stones
    rows = fn(values, *views, n_dest)
    gather, runs, run_len = class_view
    term = class_row_sum(class_weight, gather, runs, n_dest, run_len)
    return rows + term.to(rows.dtype)


def _class_view(batch, plan, direction: str) -> tuple[torch.Tensor, torch.Tensor, int]:
    """The production class-term views: window-major native, stone-major reordered."""
    if direction == "windows":
        return (batch.inc_class, batch.inc_window, WINDOW_RUN)
    return (plan.run_class, plan.run_stone, STONE_RUN)


def _batch_for_case(positions, case: str):
    graphs = [from_position(pos) for pos in positions]
    empty = [graph for graph in graphs if graph.n_stones == 0]
    single = [graph for graph in graphs if graph.n_stones == 1]
    assert len(empty) == len(single) == 1

    # Ply zero is the no-window trunk case and consists entirely of EMPTY
    # action rows. Pin both facts so fixture drift cannot delete the edge case.
    zero = empty[0]
    assert zero.n_windows == len(zero.inc_stone) == 0
    assert len(zero.dec_cell) == 0
    assert (zero.action_pre_status == ACTION_EMPTY).all()

    # A single stone owns all 18 candidate windows.  Thus window <- stone has
    # 18 one-entry reductions while stone <- window sends all 18 entries to one
    # destination, the two opposite degree extremes in one real position.
    one = single[0]
    assert one.n_windows == len(one.inc_stone) == 18
    assert set(one.inc_stone.tolist()) == {0}
    assert sorted(one.inc_window.tolist()) == list(range(18))

    if case == "empty":
        selected = [zero]
    elif case == "single":
        selected = [one]
    elif case == "small":
        selected = sorted(graphs, key=lambda graph: graph.n_stones)[:5]
    elif case == "ragged":
        selected = graphs
    else:  # pragma: no cover - test parametrization owns the case names
        raise ValueError(case)
    return collate(selected).to(_DEVICE)


def _plan_for(batch):
    return incidence_plan(batch.inc_stone, batch.inc_window, batch.inc_class)


def _views_for(batch, plan) -> tuple[torch.Tensor, ...]:
    return (batch.inc_stone, batch.inc_window, plan.run_stone, plan.run_window)


def _direction(batch, name: str):
    if name == "windows":
        return (
            int(batch.stone_own.shape[0]),
            batch.inc_stone,
            batch.inc_window,
            int(batch.window_feat.shape[0]),
        )
    if name == "stones":
        return (
            int(batch.window_feat.shape[0]),
            batch.inc_window,
            batch.inc_stone,
            int(batch.stone_own.shape[0]),
        )
    raise ValueError(name)  # pragma: no cover - parametrization owns the names


def _inputs(n_source: int, source_dtype: torch.dtype, seed: int):
    generator = torch.Generator(device=_DEVICE).manual_seed(seed)
    values = torch.randn(
        (n_source, _H),
        device=_DEVICE,
        dtype=source_dtype,
        generator=generator,
    )
    # This is the production autocast signature: linear projections may be
    # bf16, while nn.Embedding keeps its fp32 weight and lookup result.  The old
    # add therefore promotes the messages and accumulator to fp32.  The table
    # spans every joint class a batch can carry.
    class_weight = torch.randn(
        (TERN_OCC_CLASSES, _H), device=_DEVICE, dtype=torch.float32, generator=generator
    )
    return values, class_weight


@pytest.mark.parametrize("case", ["empty", "single", "ragged"])
@pytest.mark.parametrize("direction", ["windows", "stones"])
@pytest.mark.parametrize("source_dtype", [torch.float32, torch.bfloat16])
def test_cuda_values_term_matches_the_old_scatter(
    positions, case: str, direction: str, source_dtype: torch.dtype
):
    batch = _batch_for_case(positions, case)
    n_source, source, destination, n_dest = _direction(batch, direction)
    values, _class_weight = _inputs(
        n_source,
        source_dtype,
        seed=10_000 + 100 * len(case) + len(direction),
    )
    if case != "empty":
        # Otherwise an absent Triton install would make every parity assertion
        # exercise the torch fallback and prove nothing about the CUDA path.
        assert message_impl.triton is not None
    views = _views_for(batch, _plan_for(batch))

    if direction == "windows":
        actual = aggregate_to_windows(values, *views, n_dest)
        expected = _values_scatter(
            values, source, destination, n_dest, values.dtype
        )
    else:
        actual = aggregate_to_stones(values, *views, n_dest)
        expected = _values_scatter(
            values, source, destination, n_dest, torch.float32
        )

    assert actual.shape == (n_dest, _H)
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-5)


def test_stone_scatter_accepts_nonmonotone_destinations_and_zero_degree_rows():
    """Stage 2's table stays window-major; the stone view carries its runs."""
    generator = torch.Generator(device=_DEVICE).manual_seed(21)
    values = torch.randn((3, _H), device=_DEVICE, generator=generator)
    # Window/source runs are contiguous, as the builder guarantees.  Stone
    # destinations deliberately run backwards, repeat across windows, and
    # leave rows 1 and 4 untouched.
    inc_window = torch.tensor([0, 0, 1, 1, 1, 2], device=_DEVICE)
    inc_stone = torch.tensor([2, 0, 2, 3, 0, 2], device=_DEVICE)
    assert torch.all(inc_window[1:] >= inc_window[:-1])
    assert torch.any(inc_stone[1:] < inc_stone[:-1])
    order = torch.argsort(inc_stone, stable=True)
    views = (inc_stone, inc_window, inc_stone[order], inc_window[order])

    actual = aggregate_to_stones(values, *views, 5)
    expected = _values_scatter(values, inc_window, inc_stone, 5, torch.float32)

    torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-5)
    assert torch.equal(actual[[1, 4]], torch.zeros_like(actual[[1, 4]]))


@pytest.mark.parametrize("direction", ["windows", "stones"])
def test_source_and_class_gradients_match_the_old_scatter(positions, direction: str):
    batch = _batch_for_case(positions, "ragged")
    # Enough distinct joint classes that the class-weight gradient rows are a
    # real reduction, not a degenerate one-row case.
    assert batch.inc_class.unique().numel() > 3
    n_source, source, destination, n_dest = _direction(batch, direction)
    base_values, base_class = _inputs(n_source, torch.float32, seed=31)
    plan = _plan_for(batch)
    views = _views_for(batch, plan)
    generator = torch.Generator(device=_DEVICE).manual_seed(32)
    # Small integer-valued gradients remain nonuniform but sum exactly in fp32,
    # so the two reduction orders cannot make this comparison flaky merely by
    # reaching the same sum in different orders.
    upstream = torch.randint(
        -3, 4, (n_dest, _H), device=_DEVICE, generator=generator
    ).float()

    fast_values = base_values.detach().clone().requires_grad_()
    fast_class = base_class.detach().clone().requires_grad_()
    fast = _composed(
        direction,
        fast_values,
        fast_class,
        views,
        _class_view(batch, plan, direction),
        n_dest,
    )
    fast_grads = torch.autograd.grad(
        fast, (fast_values, fast_class), grad_outputs=upstream
    )

    ref_values = base_values.detach().clone().requires_grad_()
    ref_class = base_class.detach().clone().requires_grad_()
    reference = _old_scatter(
        ref_values, ref_class, source, destination, batch.inc_class, n_dest
    )
    ref_grads = torch.autograd.grad(
        reference, (ref_values, ref_class), grad_outputs=upstream
    )

    for name, actual, expected in zip(
        ("source values", "class weight"), fast_grads, ref_grads
    ):
        assert torch.isfinite(actual).all(), name
        torch.testing.assert_close(
            actual, expected, rtol=2.0e-5, atol=2.0e-5
        )


@torch.no_grad()
def test_block_call_sites_match_the_old_formulas(positions, model, monkeypatch):
    """The integrated block passes each primitive the §5 direction and counts.

    Primitive parity alone would not catch swapping two incidence arguments at
    the call site.  Replace only the primitives with the old literal formulas,
    while pinning every argument by identity, then compare the whole block.
    """
    batch = _batch_for_case(positions, "ragged")
    model = model.to(_DEVICE)
    block = model.blocks[0]
    pairs = model._pair_tables(batch)
    plan = incidence_plan(batch.inc_stone, batch.inc_window, batch.inc_class)
    seen: set[str] = set()

    def windows(values, inc_stone, inc_window, run_stone, run_window, n_dest):
        assert inc_stone is batch.inc_stone
        assert inc_window is batch.inc_window
        assert run_stone is plan.run_stone
        assert run_window is plan.run_window
        assert n_dest == batch.window_feat.shape[0]
        seen.add("windows")
        return _values_scatter(
            values, inc_stone, inc_window, n_dest, values.dtype
        )

    def stones(values, inc_stone, inc_window, run_stone, run_window, n_dest):
        assert inc_stone is batch.inc_stone
        assert inc_window is batch.inc_window
        assert run_stone is plan.run_stone
        assert run_window is plan.run_window
        assert n_dest == batch.stone_own.shape[0]
        seen.add("stones")
        return _values_scatter(
            values, inc_window, inc_stone, n_dest, torch.float32
        )

    try:
        with torch.autocast("cuda", torch.bfloat16):
            s = model.stone_table(batch.stone_own)
            w = model.window_table(batch.window_feat)
            g = model.token_base + model.token_moves(batch.moves_idx)
            seq_lens = batch.attn_valid.sum(dim=1, dtype=torch.int32)
            fast = block(s, w, g, batch, seq_lens, plan, pairs, None)

            monkeypatch.setattr(message_impl, "aggregate_to_windows", windows)
            monkeypatch.setattr(message_impl, "aggregate_to_stones", stones)
            reference = block(s, w, g, batch, seq_lens, plan, pairs, None)
    finally:
        model.to("cpu")

    assert seen == {"windows", "stones"}
    for actual, expected in zip(fast, reference):
        torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-5)


@pytest.mark.parametrize("direction", ["windows", "stones"])
@torch.no_grad()
def test_dynamic_fullgraph_compile_matches_the_old_scatter(positions, direction: str):
    def run(values, class_weight, inc_stone, inc_window, run_stone, run_window, gather, runs, run_len, n_dest):
        return _composed(
            direction,
            values,
            class_weight,
            (inc_stone, inc_window, run_stone, run_window),
            (gather, runs, run_len),
            n_dest,
        )

    compiled = torch.compile(run, dynamic=True, fullgraph=True)
    for index, case in enumerate(("single", "small", "ragged")):
        batch = _batch_for_case(positions, case)
        n_source, source, destination, n_dest = _direction(batch, direction)
        values, class_weight = _inputs(
            n_source, torch.bfloat16, seed=41 + index
        )
        plan = _plan_for(batch)

        actual = compiled(
            values,
            class_weight,
            *_views_for(batch, plan),
            *_class_view(batch, plan, direction),
            n_dest,
        )
        expected = _old_scatter(
            values, class_weight, source, destination, batch.inc_class, n_dest
        )
        expected = expected.to(actual.dtype)
        torch.testing.assert_close(actual, expected, rtol=2.0e-2, atol=2.0e-2)


def test_dynamic_compiled_window_backward_matches_the_old_scatter(positions):
    def loss(values, class_weight, inc_stone, inc_window, run_stone, run_window, inc_class, upstream, n_dest):
        rows = aggregate_to_windows(
            values, inc_stone, inc_window, run_stone, run_window, n_dest
        )
        rows = rows + class_row_sum(
            class_weight, inc_class, inc_window, n_dest, WINDOW_RUN
        ).to(rows.dtype)
        return (rows.float() * upstream).sum()

    compiled = torch.compile(loss, dynamic=True, fullgraph=True)
    for index, case in enumerate(("single", "ragged")):
        batch = _batch_for_case(positions, case)
        n_source, source, destination, n_dest = _direction(batch, "windows")
        values, class_weight = _inputs(
            n_source, torch.bfloat16, seed=51 + index
        )
        plan = _plan_for(batch)
        generator = torch.Generator(device=_DEVICE).manual_seed(61 + index)
        upstream = torch.randint(
            -3,
            4,
            (n_dest, _H),
            device=_DEVICE,
            generator=generator,
        ).float()

        fast_values = values.detach().requires_grad_()
        fast_class = class_weight.detach().requires_grad_()
        fast_grads = torch.autograd.grad(
            compiled(
                fast_values,
                fast_class,
                *_views_for(batch, plan),
                batch.inc_class,
                upstream,
                n_dest,
            ),
            (fast_values, fast_class),
        )

        ref_values = values.detach().requires_grad_()
        ref_class = class_weight.detach().requires_grad_()
        reference = _old_scatter(
            ref_values, ref_class, source, destination, batch.inc_class, n_dest
        ).to(torch.bfloat16)
        ref_grads = torch.autograd.grad(
            (reference.float() * upstream).sum(),
            (ref_values, ref_class),
        )

        for actual, expected in zip(fast_grads, ref_grads):
            torch.testing.assert_close(
                actual, expected, rtol=2.0e-2, atol=2.0e-2
            )
