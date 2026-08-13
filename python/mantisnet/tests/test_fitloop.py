"""Shared fitting-engine packing, grouping, RNG, and refusal contracts."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mantisnet.fitloop import FitBudgets, fit_epoch, pack_chunks


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
    budgets = FitBudgets(batch_size=3, pair_budget=100, cell_budget=6)

    chunks = pack_chunks(lengths, cells, range(len(lengths)), budgets)

    assert chunks == [[0], [1], [2, 3], [4, 5, 6], [7]]
    assert sorted(i for chunk in chunks for i in chunk) == list(range(len(lengths)))
    assert chunks[0] == [0]  # individually over the pair budget
    assert chunks[1] == [1]  # individually over the cell budget
    for chunk in chunks:
        assert len(chunk) <= budgets.batch_size
        if len(chunk) > 1:
            width = max(lengths[i] for i in chunk)
            assert len(chunk) * width * width <= budgets.pair_budget
            assert sum(cells[i] for i in chunk) <= budgets.cell_budget


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
        lengths=[1] * len(values),
        cells=[2] * len(values),
        budgets=FitBudgets(batch_size=5, pair_budget=1_000, cell_budget=6),
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
            lengths=[1],
            cells=[1],
            budgets=FitBudgets(batch_size=1, pair_budget=1, cell_budget=1),
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
    consumed: list[list[int]] = []

    def prepare(indices):
        prepared.append(list(indices))
        return indices

    def step(payload):
        consumed.append(list(payload))
        return model.weight * 0.0, {}

    fit_epoch(
        model,
        optimizer,
        rng,
        lengths=[5, 5, 4, 4, 3, 3, 2, 2],
        cells=[1] * 8,
        budgets=FitBudgets(batch_size=3, pair_budget=1_000, cell_budget=1_000),
        prepare=prepare,
        step=step,
        lock=_ObservedLock(),
    )

    # Stable descending packing after the first fixed permutation gives
    # [[0, 1, 3], [2, 4, 5], [7, 6]]; the second permutation is [2, 0, 1].
    # Preparation runs on the prefetch pool, so only its multiset is
    # guaranteed; consumption order is the packed permutation exactly.
    assert sorted(prepared) == [[0, 1, 3], [2, 4, 5], [7, 6]]
    assert consumed == [[7, 6], [0, 1, 3], [2, 4, 5]]
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
            lengths=[],
            cells=[],
            budgets=FitBudgets(batch_size=1, pair_budget=1, cell_budget=1),
            prepare=lambda indices: indices,
            step=lambda _payload: (model.weight * 0.0, {}),
            lock=_ObservedLock(),
        )
