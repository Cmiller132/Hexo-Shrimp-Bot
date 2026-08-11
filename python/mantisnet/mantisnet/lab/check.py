"""Model-contract checks and the tiny end-to-end lab smoke run."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import torch

from ..builder import AXES, collate, from_position
from ..klent import telemetry
from .cohort import CohortCase, corpus_cohort, selfplay_cohort
from .families import load_checkpoint
from .variants import VARIANTS, build_variant, scoped_collate, variant_spec


_BATCH_TENSORS = (
    "stone_own",
    "window_feat",
    "window_id",
    "moves_idx",
    "inc_stone",
    "inc_window",
    "inc_class",
    "stone_slot",
    "coords",
    "attn_valid",
    "window_slot",
    "value_valid",
    "legal_offsets",
    "cell_pos",
    "dec_cell",
    "dec_window",
    "dec_class",
    "bg_cell",
    "bg_bucket",
)


def _forward(model, batch, device: str):
    batch = batch.to(device)
    with torch.no_grad(), torch.autocast(
        device, torch.bfloat16, enabled=device == "cuda"
    ):
        return model(batch, 0.2)


def _collate_cases(cases, collate_fn):
    moves = [case.moves for case in cases]
    return collate_fn(moves, [len(prefix) for prefix in moves])


def _transformed_batch(case: CohortCase, transform, collate_fn):
    pos = case.position
    moves = [transform(move) for move in case.moves]
    batch = collate_fn([moves], [len(moves)])
    # Engine legal order is lexicographic. Sorting the transformed set gives
    # the order exposed by replaying the transformed prefix.
    legal = np.asarray(
        sorted(transform(move) for move in pos.legal_moves()), dtype=np.int64
    ).reshape(-1, 2)
    return batch, [tuple(map(int, row)) for row in legal]


# Policy logits are unbounded — KLENT-sharpened checkpoints reach |logit| in
# the hundreds, where one fp32 ulp is ~3e-5 and a transformed or re-batched
# board reorders every scatter accumulation. Logit comparisons carry a
# relative term sized in ulps (5e-6 ~ 40 ulps) for this reason; bounded
# quantities (value, distributions) stay on the strict absolute tolerance.
_LOGIT_RTOL = 5e-6


def _check_d6(model, cases, device: str, collate_fn) -> int:
    comparisons = 0
    for case in cases:
        pos = case.position
        base = _forward(model, _collate_cases([case], collate_fn), device)
        base_policy = dict(zip(pos.legal_moves(), base.policy_logits.cpu().tolist()))
        for transform in telemetry.D6_TRANSFORMS[1:]:
            batch, transformed_legal = _transformed_batch(case, transform, collate_fn)
            got = _forward(model, batch, device)
            if not torch.allclose(
                got.value.cpu(), base.value.cpu(), rtol=0.0, atol=1e-5
            ):
                drift = float((got.value.cpu() - base.value.cpu()).abs().max())
                raise ValueError(f"D6 value invariance drift {drift:.3e}")
            mapped = dict(zip(transformed_legal, got.policy_logits.cpu().tolist()))
            if set(mapped) != {transform(move) for move in base_policy}:
                raise ValueError("D6 transformed policy legal set differs")
            for move, logit in base_policy.items():
                delta = abs(mapped[transform(move)] - logit)
                if delta > 1e-5 + _LOGIT_RTOL * abs(logit):
                    raise ValueError(
                        f"D6 policy invariance drift {delta:.3e} at move {move}"
                    )
            comparisons += 1
    return comparisons


def _check_batch_parity(model, cases, device: str, collate_fn) -> int:
    batch = _collate_cases(cases, collate_fn).to(device)
    with torch.no_grad(), torch.autocast(
        device, torch.bfloat16, enabled=device == "cuda"
    ):
        together = model(batch, 0.2)
    for i, case in enumerate(cases):
        single_batch = _collate_cases([case], collate_fn).to(device)
        with torch.no_grad(), torch.autocast(
            device, torch.bfloat16, enabled=device == "cuda"
        ):
            single = model(single_batch, 0.2)
        a, b = int(batch.legal_offsets[i]), int(batch.legal_offsets[i + 1])
        pairs = (
            ("policy", together.policy_logits[a:b], single.policy_logits),
            ("q_score", together.q_score[a:b], single.q_score),
            ("q", together.q_values[a:b], single.q_values),
            ("value", together.value[i : i + 1], single.value),
            ("value_dist", together.value_dist[i : i + 1], single.value_dist),
            (
                "value_logits",
                together.value_logits[i : i + 1],
                single.value_logits,
            ),
        )
        for name, expected, got in pairs:
            # Bounded heads (q, value, distributions) inherit the absolute
            # noise of the large logits behind them, so they share the 1e-5
            # floor rather than getting a tighter one.
            if not torch.allclose(expected, got, rtol=_LOGIT_RTOL, atol=1e-5):
                drift = float((expected - got).abs().max())
                raise ValueError(
                    f"batched-vs-single {name} drift {drift:.3e} at position {i}"
                )
    return len(cases)


def _covered_by_window(graph, move) -> bool:
    q, r = move
    for axis, start_q, start_r in graph.window_id:
        dq, dr = AXES[int(axis)]
        for slot in range(6):
            if (int(start_q + slot * dq), int(start_r + slot * dr)) == (q, r):
                return True
    return False


def _check_decoder_coverage(model, cases, device: str, collate_fn) -> int:
    checked = 0
    for case in cases:
        pos = case.position
        graph = from_position(pos, mixed_windows=model.cfg.mixed_windows)
        batch = _collate_cases([case], collate_fn)
        expected_bg = {
            i
            for i, move in enumerate(pos.legal_moves())
            if not _covered_by_window(graph, move)
        }
        got_bg = set(map(int, batch.bg_cell))
        covered = set(map(int, batch.dec_cell))
        if got_bg != expected_bg:
            raise ValueError(
                f"decoder background mismatch: {sorted(got_bg ^ expected_bg)[:8]}"
            )
        if got_bg & covered or got_bg | covered != set(range(graph.n_legal)):
            raise ValueError("decoder routes do not partition every legal cell")
        output = _forward(model, batch, device)
        if len(output.policy_logits) != graph.n_legal or len(output.q_values) != graph.n_legal:
            raise ValueError(
                f"decoder scored {len(output.policy_logits)} policy / "
                f"{len(output.q_values)} critic rows for {graph.n_legal} legal cells"
            )
        checked += graph.n_legal
    return checked


def _assert_batches_equal(rust, python) -> None:
    if rust.mixed_windows != python.mixed_windows:
        raise ValueError("Rust/Python builder disagreement in window scope")
    shape = (rust.n_pos, rust.max_t, rust.max_w, rust.n_cells)
    expected_shape = (python.n_pos, python.max_t, python.max_w, python.n_cells)
    if shape != expected_shape:
        raise ValueError(f"Rust/Python batch shapes {shape} != {expected_shape}")
    for name in _BATCH_TENSORS:
        a, b = getattr(rust, name), getattr(python, name)
        if a.dtype != b.dtype or not torch.equal(a, b):
            raise ValueError(f"Rust/Python builder disagreement in {name}")


def _check_builder_agreement(cases, collate_fn, mixed_windows: bool) -> int:
    batch = _collate_cases(cases, collate_fn)
    _assert_batches_equal(
        batch,
        collate(
            [from_position(case.position, mixed_windows=mixed_windows) for case in cases]
        ),
    )
    for case in cases:
        _assert_batches_equal(
            _collate_cases([case], collate_fn),
            collate([from_position(case.position, mixed_windows=mixed_windows)]),
        )
    return len(cases)


def contract_battery(
    model, cases, *, collate_fn, device: str = "cpu", rust_collate=True
) -> dict:
    """Run the contract battery and return its exact work counts."""
    if not cases:
        raise ValueError("the contract battery requires at least one real position")
    if any(case.position.is_terminal for case in cases):
        raise ValueError("the contract battery refuses terminal positions")
    model.eval()
    report = {
        "d6_nonidentity_comparisons": _check_d6(
            model, cases, device, collate_fn
        ),
        "batch_parity_positions": _check_batch_parity(
            model, cases, device, collate_fn
        ),
        "decoder_legal_cells": _check_decoder_coverage(
            model, cases, device, collate_fn
        ),
        "python_rust_positions": (
            _check_builder_agreement(
                cases, collate_fn, model.cfg.mixed_windows
            )
            if rust_collate
            else None
        ),
        "atol": 1e-5,
        "logit_rtol": _LOGIT_RTOL,
    }
    return report


def run_check(
    *,
    checkpoint=None,
    variant: str | None = None,
    model_kw: dict | None = None,
    corpus=None,
    split: str = "test",
    envs: int = 2,
    steps: int = 12,
    seed: int = 0,
    device: str = "cpu",
    compile: bool = False,
    family: str | None = None,
) -> dict:
    """Load a checkpoint or fresh variant and run the contract battery."""
    if (checkpoint is None) == (variant is None):
        raise ValueError("exactly one of checkpoint or variant is required")
    if envs <= 0 or steps < 0:
        raise ValueError(f"envs must be positive and steps nonnegative, got {envs}, {steps}")
    if checkpoint is not None and model_kw:
        raise ValueError("model_kw applies only to a fresh --variant, not --checkpoint")
    if checkpoint is None and family is not None:
        raise ValueError("--family applies only with --checkpoint")
    if checkpoint is not None:
        loaded = load_checkpoint(Path(checkpoint), family=family, device=device)
        model = loaded.model
        spec = variant_spec("mantis")
        identity = {"checkpoint": str(Path(checkpoint)), **loaded.metadata}
        variant = "mantis"
    else:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
        torch.manual_seed(seed)
        model, normalized_kw, spec = build_variant(variant, model_kw or {})
        model = model.to(device).eval()
        identity = {"variant": variant, "model_kw": normalized_kw}
    if corpus is None:
        # Cohort generation is representation-neutral: Collector owns the
        # production Rust batch path, so a not-yet-approved variant does not
        # act here before its declared collator has been validated.
        cases = selfplay_cohort(
            envs=envs,
            steps=steps,
            seed=seed,
            device=device,
            compile=compile,
            with_prefixes=True,
        )
    else:
        cases = corpus_cohort(
            corpus, split=split, count=envs, seed=seed, with_prefixes=True
        )
    result = {
        "mode": "check",
        "device": device,
        **identity,
        **contract_battery(
            model,
            cases,
            collate_fn=scoped_collate(variant, model),
            device=device,
            rust_collate=spec.rust_collate,
        ),
    }
    print(json.dumps(result, indent=2))
    return result


_FIRST_STONE_WIN = [
    (0, 0),
    (-8, 8),
    (-8, 9),
    (1, 0),
    (2, 0),
    (-8, 10),
    (-6, 8),
    (3, 0),
    (4, 0),
    (-6, 9),
    (-6, 10),
    (5, 0),
]


def _synthetic_episode(moves):
    """Build a complete telemetry Episode from a scripted terminal game."""
    import hexo_py

    from ..klent.selfplay import Episode

    pos = hexo_py.Position()
    episode = Episode()
    for move in moves:
        legal = pos.legal_moves()
        try:
            rank = legal.index(move)
        except ValueError as exc:  # pragma: no cover - fixed script guard.
            raise ValueError(f"synthetic smoke move {move} is not legal") from exc
        width = len(legal)
        episode.moves_remaining.append(pos.moves_remaining)
        episode.movers.append(pos.current_player)
        episode.ranks.append(rank)
        episode.improved.append(np.full(width, 1.0 / width, dtype=np.float32))
        episode.v_hats.append(0.0)
        episode.kls.append(0.0)
        episode.norm_entropies.append(1.0 if width > 1 else 0.0)
        episode.pi_top1.append(1.0 / width)
        episode.pi_chosen.append(1.0 / width)
        pos.advance(*move)
        episode.moves.append(move)
    if not pos.is_terminal:
        raise ValueError("synthetic smoke script did not terminate")
    episode.winner = pos.winner
    return episode


def _write_synthetic_run(run_dir: Path) -> int:
    episodes = []
    # The full D6 orbit supplies disjoint game rows while retaining an easy,
    # exactly replayable outcome.  Repeating it twice yields a few hundred
    # supervised positions and nonempty validation/test game splits.
    for _ in range(2):
        for transform in telemetry.D6_TRANSFORMS:
            episodes.append(_synthetic_episode([transform(m) for m in _FIRST_STONE_WIN]))
    versions = {
        "RULES_VERSION": __import__("hexo_py").RULES_VERSION,
        "ACTION_ORDER_VERSION": __import__("hexo_py").ACTION_ORDER_VERSION,
    }
    with telemetry.open_telemetry(run_dir) as writer:
        writer.begin_run({"iterations": 1}, versions, 0)
        writer.write_iteration(
            {
                "iteration": 0,
                "seconds": 1.0,
                "buffer_samples": sum(len(e.moves) for e in episodes),
            },
            episodes,
            {},
        )
    return len(episodes)


def smoke(work_dir: str | Path | None = None) -> dict:
    """Run freeze -> tiny train -> evaluate -> report entirely on CPU."""
    owned_tmp = tempfile.TemporaryDirectory(prefix="mantisnet-lab-smoke-") if work_dir is None else None
    root = Path(owned_tmp.name if owned_tmp is not None else work_dir)
    root.mkdir(parents=True, exist_ok=True)
    try:
        from .corpus import freeze
        from .evaluate import evaluate_cell
        from .report import build_report
        from .train import train_cell

        run_dir = root / "source-run"
        corpus_dir = root / "corpus"
        cell_dir = root / "cell" / "s0"
        games = _write_synthetic_run(run_dir)
        freeze(
            run_dir,
            corpus_dir,
            (0, 0),
            name="smoke",
            train_samples=240,
            val_samples=24,
            test_samples=24,
            seed=17,
        )
        trained = train_cell(
            corpus_dir,
            cell_dir,
            variant="mantis",
            model_kw={
                "h": 32,
                "blocks": 1,
                "heads": 2,
                "policy_hidden": 32,
                "value_hidden": 32,
                "value_queries": 2,
                "value_bins": 9,
            },
            seed=0,
            epochs=1,
            device="cpu",
            compile=False,
        )
        scores = evaluate_cell(cell_dir, corpus_dir, split="test", device="cpu")
        report_path = root / "report.json"
        report = build_report([cell_dir / "scores.json"], report_path)
        required = (
            corpus_dir / "manifest.json",
            corpus_dir / "corpus.npz",
            cell_dir / "config.json",
            cell_dir / "metrics.jsonl",
            cell_dir / "checkpoint_final.pt",
            cell_dir / "scores.json",
            report_path,
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(f"smoke artifacts missing: {missing}")

        def assert_finite(value, path="result"):
            if isinstance(value, dict):
                for key, item in value.items():
                    assert_finite(item, f"{path}.{key}")
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    assert_finite(item, f"{path}[{i}]")
            elif isinstance(value, float) and not math.isfinite(value):
                raise RuntimeError(f"smoke produced non-finite {path}: {value}")

        assert_finite(trained["metrics"], "training_metrics")
        assert_finite(scores, "scores")
        assert_finite(report, "report")
        result = {
            "mode": "smoke",
            "device": "cpu",
            "games": games,
            "artifacts": len(required),
            "scores": scores,
            "report": report,
        }
        print(json.dumps(result, indent=2))
        return result
    finally:
        if owned_tmp is not None:
            owned_tmp.cleanup()
