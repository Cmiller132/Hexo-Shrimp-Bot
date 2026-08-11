"""The lab contract gate and tiny CPU end-to-end smoke path."""

from __future__ import annotations

from dataclasses import replace
import json

import pytest
import torch

from mantisnet.lab.check import _check_batch_parity, run_check, smoke
from mantisnet.lab.cohort import CohortCase
from mantisnet.lab.variants import VARIANTS


def _tiny_model_kw():
    return {
        "h": 32,
        "blocks": 1,
        "heads": 2,
        "policy_hidden": 32,
        "value_hidden": 32,
        "value_queries": 2,
        "value_bins": 9,
    }


def test_contract_battery_passes_for_mantis(monkeypatch):
    declared = VARIANTS["mantis"]
    collate_calls = 0

    def tracked_collate(moves, ts, *, action_rows=False):
        nonlocal collate_calls
        collate_calls += 1
        return declared.collate(moves, ts, action_rows=action_rows)

    monkeypatch.setitem(VARIANTS, "mantis", replace(declared, collate=tracked_collate))
    result = run_check(
        variant="mantis",
        model_kw=_tiny_model_kw(),
        envs=1,
        steps=5,
        seed=3,
        device="cpu",
        compile=False,
    )
    assert result["d6_nonidentity_comparisons"] == 11
    assert result["batch_parity_positions"] == 1
    assert result["python_rust_positions"] == 1
    assert result["decoder_legal_cells"] > 0
    assert collate_calls > 0


@pytest.mark.parametrize("field", ("q_score", "value_logits"))
def test_batch_parity_covers_every_public_output(
    model, positions, move_lists, field
):
    class DriftOneOutput(torch.nn.Module):
        def forward(self, batch, mass_floor):
            output = model(batch, mass_floor)
            if batch.n_pos > 1:
                output = replace(
                    output, **{field: getattr(output, field) + 1e-3}
                )
            return output

    cases = [
        CohortCase(position, tuple(moves))
        for position, moves in zip(positions[:2], move_lists[:2], strict=True)
    ]
    with pytest.raises(ValueError, match=f"batched-vs-single {field} drift"):
        _check_batch_parity(
            DriftOneOutput(), cases, "cpu", VARIANTS["mantis"].collate
        )


def test_smoke_runs_end_to_end(tmp_path):
    result = smoke(tmp_path)
    assert result["device"] == "cpu"
    assert result["artifacts"] == 7
    assert (tmp_path / "report.json").is_file()
    # The final serialization is also a guard that no tensor or NumPy scalar
    # leaked through a public artifact summary.
    json.dumps(result, allow_nan=False)
