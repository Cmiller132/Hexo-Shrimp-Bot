"""Shared fitting-engine packing, grouping, RNG, and refusal contracts."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mantisnet.builder import PaddedPairChunkCost
from mantisnet.fitloop import ChunkCost, fit_epoch, pack_chunks


class _UnitCost:
    """One unit per sample: the loop's tests are about the loop, not a law."""

    def __init__(self, count: int, limit: int) -> None:
        self._count = count
        self._limit = limit
        self._taken = 0

    def sort_key(self, index: int) -> int:
        return 0

    def open(self) -> None:
        self._taken = 0

    def accepts(self, index: int, size: int) -> bool:
        return self._taken + 1 <= self._limit

    def take(self, index: int) -> None:
        self._taken += 1


class _ScalarModel(torch.nn.Module):
    def __init__(self, value: float = 0.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(value))


class _ObservedLock:
    def __init__(self):
        self.held = False

    def __enter__(self):
        assert not self.held
        self.held = True
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.held = False


def test_packing_respects_all_limits_and_keeps_oversized_singletons():
    lengths = [11, 5, 5, 5, 4, 4, 4, 4]
    cells = [1, 7, 4, 2, 2, 2, 2, 2]
    batch_size, pair_budget, cell_budget = 3, 100, 6
    cost = PaddedPairChunkCost(lengths, cells, pair_budget, cell_budget)

    chunks = pack_chunks(range(len(lengths)), batch_size, cost)

    assert isinstance(cost, ChunkCost)
    assert chunks == [[0], [1], [2, 3], [4, 5, 6], [7]]
    assert sorted(i for chunk in chunks for i in chunk) == list(range(len(lengths)))
    assert chunks[0] == [0]  # individually over the pair budget
    assert chunks[1] == [1]  # individually over the cell budget
    for chunk in chunks:
        assert len(chunk) <= batch_size
        if len(chunk) > 1:
            width = max(lengths[i] for i in chunk)
            assert len(chunk) * width * width <= pair_budget
            assert sum(cells[i] for i in chunk) <= cell_budget


def test_packing_is_the_costs_law_and_nothing_of_the_packers_own():
    """A cost with no memory term packs by the position cap alone.

    The loop owns the position cap, the deterministic descending order, and the
    singleton rule; every other limit is the architecture's. A cost that always
    accepts must therefore produce exactly ``batch_size``-sized chunks, which no
    padded-pair or graph-cell term is left in the packer to shorten.
    """

    class _Boundless:
        def sort_key(self, index):
            return 0

        def open(self):
            pass

        def accepts(self, index, size):
            return True

        def take(self, index):
            pass

    chunks = pack_chunks(range(7), 3, _Boundless())
    assert chunks == [[0, 1, 2], [3, 4, 5], [6]]


def test_groups_reach_batch_size_and_use_sample_weighted_means():
    model = _ScalarModel()
    lock = _ObservedLock()
    pending: list[int] = []

    class RecordingOptimizer:
        def __init__(self):
            self.groups: list[list[int]] = []
            self.gradients: list[float] = []

        def zero_grad(self, *, set_to_none):
            assert set_to_none
            model.weight.grad = None

        def step(self):
            assert not lock.held, "optimizer.step() ran under the fitting lock"
            self.groups.append(pending.copy())
            pending.clear()
            self.gradients.append(float(model.weight.grad))

    optimizer = RecordingOptimizer()
    values = torch.arange(1.0, 11.0)
    progress = []

    def prepare(indices):
        return list(indices)

    def step(indices):
        pending.extend(indices)
        mean = values[indices].mean()
        return model.weight * mean, {"sample_mean": mean}

    result = fit_epoch(
        model,
        optimizer,
        np.random.default_rng(14),
        sample_count=len(values),
        batch_size=5,
        cost=_UnitCost(len(values), 3),
        prepare=prepare,
        step=step,
        lock=lock,
        progress=lambda consumed, total: progress.append((consumed, total)),
    )

    assert sorted(i for group in optimizer.groups for i in group) == list(range(10))
    assert all(len(group) >= 5 for group in optimizer.groups[:-1])
    assert len(optimizer.groups[-1]) < 5  # the final partial group still steps
    for group, gradient in zip(optimizer.groups, optimizer.gradients):
        assert gradient == pytest.approx(float(values[group].mean()))
    assert result["sample_mean"] == pytest.approx(float(values.mean()))
    assert result["fit_steps"] == len(optimizer.groups)
    assert progress == [(1, 4), (2, 4), (3, 4), (4, 4)]


def test_nonfinite_refusal_names_the_optimizer_step_and_parameter():
    model = _ScalarModel(1.0)
    lock = _ObservedLock()

    class NonfiniteOptimizer:
        def zero_grad(self, *, set_to_none):
            model.weight.grad = None

        def step(self):
            assert not lock.held
            with torch.no_grad():
                model.weight.fill_(float("nan"))

    with pytest.raises(
        ValueError,
        match=r"optimizer step 1 left 1 parameter tensors non-finite: weight",
    ):
        fit_epoch(
            model,
            NonfiniteOptimizer(),
            np.random.default_rng(0),
            sample_count=1,
            batch_size=1,
            cost=_UnitCost(1, 1),
            prepare=lambda indices: indices,
            step=lambda _payload: (model.weight.square(), {}),
            lock=lock,
        )


def test_rng_consumes_sample_then_chunk_permutations_only():
    class FixedPermutationRng:
        def __init__(self):
            self.calls: list[int] = []
            self.outputs = [
                np.array([7, 0, 3, 1, 6, 4, 2, 5]),
                np.array([2, 0, 1]),
            ]

        def permutation(self, n):
            self.calls.append(n)
            result = self.outputs.pop(0)
            assert len(result) == n
            return result

    rng = FixedPermutationRng()
    model = _ScalarModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    prepared: list[list[int]] = []

    def prepare(indices):
        prepared.append(list(indices))
        return indices

    fit_epoch(
        model,
        optimizer,
        rng,
        sample_count=8,
        batch_size=3,
        cost=PaddedPairChunkCost([5, 5, 4, 4, 3, 3, 2, 2], [1] * 8, 1_000, 1_000),
        prepare=prepare,
        step=lambda _payload: (model.weight * 0.0, {}),
        lock=_ObservedLock(),
    )

    # Stable descending packing after the first fixed permutation gives
    # [[0, 1, 3], [2, 4, 5], [7, 6]]; the second permutation [2, 0, 1] submits
    # them in consumption order. Which chunks exist is the packing contract;
    # the order `prepare`'s calls land in is not one, because all three are
    # submitted at once to a `_PREFETCH_DEPTH`-wide pool and run concurrently.
    assert sorted(prepared) == [[0, 1, 3], [2, 4, 5], [7, 6]]
    assert rng.calls == [8, 3]
    assert rng.outputs == []


def test_empty_epoch_preserves_the_production_fit_refusal():
    model = _ScalarModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    with pytest.raises(ZeroDivisionError, match="float division by zero"):
        fit_epoch(
            model,
            optimizer,
            np.random.default_rng(0),
            sample_count=0,
            batch_size=1,
            cost=_UnitCost(0, 1),
            prepare=lambda indices: indices,
            step=lambda _payload: (model.weight * 0.0, {}),
            lock=_ObservedLock(),
        )
