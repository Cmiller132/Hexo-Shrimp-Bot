"""Regenerate the cross-language Gumbel sequential-halving fixture.

Run from ``python/mantisnet`` with the project environment:

    .venv/Scripts/python.exe ../../crates/hexo-search/tests/fixtures/regenerate_gumbel.py --check

On Linux, use ``.venv/bin/python`` instead.  Pass ``--emit`` to print the
canonical JSON when the production reference intentionally changes.  The
script is CPU-only and imports ``mantisnet.klent.search.gumbel_choose``; it
does not carry a second implementation of the search.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import hexo_py


FIXTURE = Path(__file__).with_name("gumbel_parity.json")
ROOT_PREFIX = [(0, 0)]
SIMS = 32
CANDIDATES = 16
TAU = 1.0
LAM = 0.0

# The first sixteen ranks are the root candidates, in this order.  Values for
# every remaining legal rank are present too: the fixture never depends on an
# RNG stream shared between NumPy and Rust.
ROOT_GUMBELS = [
    *(10.0 - rank / 10.0 for rank in range(CANDIDATES)),
    *(-10.0 - rank / 1000.0 for rank in range(216 - CANDIDATES)),
]

# Values are in the root mover's frame.  The evaluator below converts them to
# the leaf side-to-move frame before returning raw Q values.
ROUND_0_ROOT_VALUES = {
    0: 0.12,
    1: 0.75,
    2: -0.10,
    3: 0.35,
    4: 0.90,
    5: 0.20,
    6: 0.55,
    7: -0.40,
    8: 0.65,
    9: 0.05,
    10: 0.45,
    11: -0.20,
    12: 0.80,
    13: 0.30,
    14: 0.60,
    15: 0.00,
}
ROUND_1_WAVE_0_ROOT_VALUES = {
    4: -0.40,
    12: 0.10,
    1: 0.30,
    8: -0.20,
    14: 0.20,
    6: 0.00,
    10: 0.40,
    3: -0.10,
}
ROUND_1_WAVE_1_ROOT_VALUES = {
    4: 0.10,
    12: 0.20,
    1: 0.40,
    8: 0.30,
    14: 0.70,
    6: 0.60,
    10: 0.50,
    3: 0.80,
}


def _load_search_module():
    try:
        from mantisnet.klent import search
    except (ImportError, ModuleNotFoundError) as exc:
        raise SystemExit(
            "production mantisnet.klent.search is unavailable; merge the "
            "MantisNet search reference before regenerating this fixture"
        ) from exc
    return search


def _compact_probs(probs: torch.Tensor, hot_rank: int | None):
    values = probs.detach().cpu().tolist()
    if hot_rank is None:
        fill = values[0]
        assert all(value == fill for value in values)
        return {"fill": float(fill)}
    fill_rank = 1 if hot_rank == 0 else 0
    fill = values[fill_rank]
    assert all(
        value == (values[hot_rank] if rank == hot_rank else fill)
        for rank, value in enumerate(values)
    )
    return {
        "fill": float(fill),
        "overrides": [
            {"rank": int(hot_rank), "value": float(values[hot_rank])}
        ],
    }


def _advance_rank(position, rank: int):
    move = position.nth_legal(rank)
    position.advance(*move)
    return [int(move[0]), int(move[1])]


def _row_position(root, root_rank: int, interior_steps: int):
    position = root.copy()
    prefix = [list(move) for move in ROOT_PREFIX]
    prefix.append(_advance_rank(position, root_rank))
    for _ in range(interior_steps):
        prefix.append(_advance_rank(position, 0))
    return position, prefix


def _columnarize_call(call: dict[str, Any]) -> dict[str, Any]:
    """Keep every scripted scalar while avoiding repeated JSON field names."""
    rows = call["rows"]
    out = {
        "phase": call["phase"],
        "raw_policy_logits": call["raw_policy_logits"],
        "position_prefixes": [row["position_prefix"] for row in rows],
        "root_ranks": [row["root_rank"] for row in rows],
        "legal_counts": [row["legal_count"] for row in rows],
        "raw_q_fill": [row["raw_q_fill"] for row in rows],
        "evaluation": {
            "priors_fill": [
                row["evaluation"]["priors"]["fill"] for row in rows
            ],
            "values": [row["evaluation"]["value"] for row in rows],
        },
    }
    if "overrides" in rows[0]["evaluation"]["priors"]:
        out["evaluation"]["priors_override_rank"] = 0
        out["evaluation"]["priors_override_values"] = [
            row["evaluation"]["priors"]["overrides"][0]["value"]
            for row in rows
        ]
    if "side_to_move" in rows[0]:
        out["side_to_move"] = [row["side_to_move"] for row in rows]
        out["moves_remaining"] = [row["moves_remaining"] for row in rows]
    return out


def generate() -> dict[str, Any]:
    search = _load_search_module()
    root = hexo_py.Position.replay(ROOT_PREFIX)
    assert root.legal_count == len(ROOT_GUMBELS)

    traced_searches = []

    class TracingSearch:
        """A compatible record whose property observes production mutations."""

        def __init__(self, root_player, lines, survivors, schedule):
            self.root_player = root_player
            self.lines = lines
            self.schedule = schedule
            self.survivor_root_rank_trace: list[list[int]] = []
            self.survivors = survivors
            traced_searches.append(self)

        @property
        def survivors(self):
            return self._survivors

        @survivors.setter
        def survivors(self, survivors):
            self._survivors = list(survivors)
            self.survivor_root_rank_trace.append(
                [int(self.lines[index].root_rank) for index in self._survivors]
            )

    class FixedRootRng:
        def gumbel(self, size):
            assert size == len(ROOT_GUMBELS)
            return np.asarray(ROOT_GUMBELS, dtype=np.float64)

    class FixedParentRng:
        def integers(self, low, high, size, dtype):
            assert low == 0 and size == 1 and dtype == np.uint64
            return np.asarray([0], dtype=np.uint64)

    calls: list[dict[str, Any]] = []
    phase_names = [
        "root",
        "round_0_wave_0",
        "round_1_wave_0",
        "round_1_wave_1",
    ]

    def evaluate(batch):
        call_index = len(calls)
        if call_index >= len(phase_names):
            raise AssertionError(f"unexpected evaluator call {call_index}")
        offsets = batch.legal_offsets.tolist()
        row_count = len(offsets) - 1
        if call_index == 0:
            root_ranks = [None]
            root_values = [0.0]
            root_sign = 1.0
            hot_rank = None
        else:
            assert len(traced_searches) == 1
            line_search = traced_searches[0]
            root_ranks = [
                int(line_search.lines[index].root_rank)
                for index in line_search.survivors
            ]
            value_maps = [
                ROUND_0_ROOT_VALUES,
                ROUND_1_WAVE_0_ROOT_VALUES,
                ROUND_1_WAVE_1_ROOT_VALUES,
            ]
            root_values = [value_maps[call_index - 1][rank] for rank in root_ranks]
            # With the one-stone root prefix, candidate leaves still belong to
            # the root mover.  Both later waves belong to the opponent.
            root_sign = 1.0 if call_index == 1 else -1.0
            hot_rank = 0
        assert row_count == len(root_values)

        logits = torch.empty(offsets[-1], dtype=torch.float32, device="cpu")
        q_values = torch.empty_like(logits)
        rows = []
        for row, (lo, hi) in enumerate(
            zip(offsets[:-1], offsets[1:], strict=True)
        ):
            logits[lo:hi] = 0.0 if hot_rank is None else -4.0
            if hot_rank is not None:
                logits[lo + hot_rank] = 4.0
            local_value = root_sign * root_values[row]
            q_values[lo:hi] = local_value

        improved = search.improved_policy(
            logits, q_values, batch.legal_offsets, TAU, LAM
        )
        for row, (lo, hi) in enumerate(
            zip(offsets[:-1], offsets[1:], strict=True)
        ):
            rows.append(
                {
                    "root_rank": root_ranks[row],
                    "legal_count": hi - lo,
                    "raw_q_fill": float(q_values[lo]),
                    "evaluation": {
                        "priors": _compact_probs(
                            improved.probs[lo:hi], hot_rank
                        ),
                        "value": float(improved.v_hat[row]),
                    },
                }
            )
        calls.append(
            {
                "phase": phase_names[call_index],
                "raw_policy_logits": {
                    "fill": 0.0 if hot_rank is None else -4.0,
                    **(
                        {}
                        if hot_rank is None
                        else {
                            "overrides": [
                                {"rank": hot_rank, "value": 4.0}
                            ]
                        }
                    ),
                },
                "rows": rows,
            }
        )
        return logits, q_values

    original_default_rng = search.np.random.default_rng
    original_search_record = search._Search
    try:
        search.np.random.default_rng = lambda _seed: FixedRootRng()
        search._Search = TracingSearch
        chooser = search.gumbel_choose(
            evaluate, tau=TAU, lam=LAM, sims=SIMS
        )
        chosen_move = chooser([root], FixedParentRng())[0]
    finally:
        search._Search = original_search_record
        search.np.random.default_rng = original_default_rng

    assert len(calls) == len(phase_names)
    assert len(traced_searches) == 1
    line_search = traced_searches[0]
    trace = line_search.survivor_root_rank_trace
    candidate_ranks = trace[0]

    # Attach exact position prefixes after the reference has established row
    # order.  Rust can replay these prefixes and feed the scripted Evaluation
    # for that canonical state without interpreting Python batch internals.
    calls[0]["rows"][0]["position_prefix"] = [list(move) for move in ROOT_PREFIX]
    for call_index, call in enumerate(calls[1:], start=1):
        interior_steps = call_index - 1
        for row in call["rows"]:
            position, prefix = _row_position(
                root, row["root_rank"], interior_steps
            )
            row["position_prefix"] = prefix
            row["side_to_move"] = int(position.current_player)
            row["moves_remaining"] = int(position.moves_remaining)

    chosen_rank = root.legal_moves().index(chosen_move)
    return {
        "schema_version": 1,
        "source": "python/mantisnet/mantisnet/klent/search.py:gumbel_choose",
        "notes": (
            "tau=1, lambda=0, and root Q=0 make log(pi_prime) differ "
            "from the Python root logits by one action-independent constant"
        ),
        "cases": [
            {
                "name": "opening_one_stone_sims_32_m_16",
                "position_prefix": [list(move) for move in ROOT_PREFIX],
                "config": {
                    "sims": SIMS,
                    "m": CANDIDATES,
                    "tau": TAU,
                    "lambda": LAM,
                    "c_visit": int(search.C_VISIT),
                    "c_scale": float(search.C_SCALE),
                },
                "root_gumbel_noise": ROOT_GUMBELS,
                "candidate_root_ranks": candidate_ranks,
                "scripted_calls": [_columnarize_call(call) for call in calls],
                "expected": {
                    "survivor_root_ranks_after_round": trace[1:],
                    "chosen_root_rank": int(chosen_rank),
                    "chosen_move": [int(chosen_move[0]), int(chosen_move[1])],
                },
            }
        ],
    }


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--emit", action="store_true")
    args = parser.parse_args()

    rendered = _canonical(generate())
    if args.emit:
        print(rendered, end="")
        return
    if not FIXTURE.exists():
        raise SystemExit(f"missing fixture: {FIXTURE}")
    if FIXTURE.read_text(encoding="utf-8") != rendered:
        raise SystemExit(
            f"{FIXTURE} is stale; run this script with --emit and review the diff"
        )
    print(f"{FIXTURE}: production Gumbel parity fixture is current")


if __name__ == "__main__":
    main()
