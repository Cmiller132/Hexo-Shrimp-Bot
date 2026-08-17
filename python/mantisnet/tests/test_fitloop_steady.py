"""The §2.2 steady window measures without changing what the epoch computes."""

from __future__ import annotations

import threading

import numpy as np
import pytest
import torch
from torch import nn

from mantisnet.fitloop import FitBudgets, fit_epoch


def _epoch(steady, n=40, record=None):
    """One tiny CPU epoch over ``n`` two-length samples; ``record`` collects
    the prepare-order chunk indices so runs can be compared."""
    torch.manual_seed(11)
    model = nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    def prepare(indices):
        if record is not None:
            record.append(tuple(indices))
        g = torch.Generator().manual_seed(1000 + sum(indices))
        return torch.randn(len(indices), 4, generator=g)

    def step(payload):
        out = model(payload).square().mean()
        return out, {"loss": out.detach()}

    result = fit_epoch(
        model,
        optimizer,
        np.random.default_rng(3),
        lengths=[2] * n,
        cells=[3] * n,
        budgets=FitBudgets(batch_size=8, pair_budget=10**9, cell_budget=10**9),
        prepare=prepare,
        step=step,
        lock=threading.Lock(),
        steady=steady,
    )
    return model, result


def test_window_reports_rate_and_stops_the_epoch_early():
    _model, full = _epoch(None)
    assert "steady" not in full
    chunks_per_epoch = 40 // 8

    _model, measured = _epoch((2, 2))
    window = measured["steady"]
    assert window["warmup_chunks"] == 2
    assert window["measure_chunks"] == 2
    # Two 8-sample chunks were timed; the epoch stopped before finishing.
    assert window["samples"] == 16
    assert window["seconds"] > 0
    assert window["samples_per_s"] == pytest.approx(
        window["samples"] / window["seconds"]
    )
    assert window["chunk_ms_median"] >= 0
    assert window["chunk_ms_p95"] >= window["chunk_ms_median"]
    assert measured["fit_steps"] < full["fit_steps"] == chunks_per_epoch


def test_window_consumes_the_same_chunks_in_the_same_order(monkeypatch):
    # One prefetch worker makes the recorded prepare order the submission
    # order; with several workers the appends race under machine load, which
    # is not the property under test.
    import mantisnet.fitloop as fitloop

    monkeypatch.setattr(fitloop, "_PREFETCH_DEPTH", 1)
    full_order: list[tuple[int, ...]] = []
    _epoch(None, record=full_order)
    window_order: list[tuple[int, ...]] = []
    _epoch((1, 2), record=window_order)
    # Prefetch prepares a chunk past the stop, but everything prepared is
    # a prefix of the full epoch's identical packed order.
    assert window_order == full_order[: len(window_order)]


def test_impossible_windows_are_refused():
    with pytest.raises(ValueError, match="warmup >= 1"):
        _epoch((0, 2))
    with pytest.raises(ValueError, match="exceeds the"):
        _epoch((3, 1000))
