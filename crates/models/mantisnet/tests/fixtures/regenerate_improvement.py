"""Regenerate improvement.json from the production Python KLENT operator.

From ``python/mantisnet`` with its locked virtual environment active:

    python ../../crates/models/mantisnet/tests/fixtures/regenerate_improvement.py

The script is CPU-only and overwrites ``improvement.json`` beside itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from mantisnet.klent.improve import improved_policy


CASES = [
    {
        "name": "production_weights",
        "policy_logits": [0.25, -1.5, 2.0, 0.0],
        "q_values": [-0.75, 0.125, 0.9, -0.2],
        "tau": 0.1,
        "lambda": 0.03,
    },
    {
        "name": "single_legal_action",
        "policy_logits": [19.0],
        "q_values": [-0.3],
        "tau": 0.03,
        "lambda": 0.1,
    },
    {
        "name": "entropy_only",
        "policy_logits": [1_000.0, -1_000.0, 0.0],
        "q_values": [-1.0, 1.0, 0.25],
        "tau": 0.0,
        "lambda": 1.0,
    },
    {
        "name": "reverse_kl_only",
        "policy_logits": [-12.0, 4.0, 4.0, -0.5, 8.0, -7.0, 1.25],
        "q_values": [1.0, -1.0, 0.0, 0.5, -0.5, 0.75, -0.25],
        "tau": 1.0,
        "lambda": 0.0,
    },
    {
        "name": "small_weights_extreme_logits",
        "policy_logits": [1_000.0, 999.0, 0.0, -999.0, -1_000.0],
        "q_values": [-0.9, 0.8, 0.1, -0.2, 0.3],
        "tau": 0.0001,
        "lambda": 0.0003,
    },
]


def main() -> None:
    """Evaluate every scripted row and write stable, sorted JSON."""
    generated = []
    for case in CASES:
        logits = torch.tensor(case["policy_logits"], dtype=torch.float32, device="cpu")
        q_values = torch.tensor(case["q_values"], dtype=torch.float32, device="cpu")
        offsets = torch.tensor([0, logits.numel()], dtype=torch.int64, device="cpu")
        result = improved_policy(
            logits,
            q_values,
            offsets,
            tau=case["tau"],
            lam=case["lambda"],
            # Gain 1 is eq. 3 verbatim — the operator the Rust improve_policy
            # implements and the one these frozen fixtures were written at.
            q_scale=1.0,
        )
        generated.append(
            {
                **case,
                "expected_pi_prime": result.probs.tolist(),
                "expected_v_hat": result.v_hat.item(),
            }
        )

    output = Path(__file__).with_name("improvement.json")
    output.write_text(
        json.dumps({"cases": generated}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
