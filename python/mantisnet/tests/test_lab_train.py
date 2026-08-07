"""Supervised lab registry and cell-training contracts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import hexo_py
import numpy as np
import pytest
import torch

from mantisnet.lab.corpus import FrozenCorpus, SampleSplit
from mantisnet.lab.evaluate import evaluate_cell
from mantisnet.lab.train import TrainConfig, train_cell
from mantisnet.lab.variants import (
    VARIANTS,
    build_variant,
    count_parameters,
    derived_cell_name,
    parse_model_kw,
    refuse_param_budget,
)
from mantisnet.models.mantis_act import PRESETS as ACT_PRESETS

from .test_klent_returns import FIRST_STONE_WIN


TINY_MODEL_KW = {
    "h": 8,
    "blocks": 1,
    "heads": 1,
    "ffn_factor": 1,
    "value_queries": 1,
    "value_bins": 5,
    "policy_hidden": 8,
    "value_hidden": 8,
}


def _sample_split(game: int, moves: list[tuple[int, int]]) -> SampleSplit:
    pos = hexo_py.Position()
    rank = []
    mover = []
    z = []
    for move in moves:
        legal = pos.legal_moves()
        rank.append(legal.index(move))
        mover.append(pos.current_player)
        z.append(1 if pos.current_player == 0 else -1)
        pos.advance(*move)
    n = len(moves)
    return SampleSplit(
        game=np.full(n, game, dtype=np.int32),
        t=np.arange(n, dtype=np.int32),
        rank=np.asarray(rank, dtype=np.int32),
        mover=np.asarray(mover, dtype=np.int8),
        z=np.asarray(z, dtype=np.int8),
        dist=np.arange(n, 0, -1, dtype=np.int32),
    )


def tiny_corpus(tmp_path: Path, *, sha: str = "a" * 64) -> FrozenCorpus:
    games = [FIRST_STONE_WIN, FIRST_STONE_WIN, FIRST_STONE_WIN]
    flat = np.asarray([move for game in games for move in game], dtype=np.int16)
    offsets = np.arange(0, len(flat) + 1, len(FIRST_STONE_WIN), dtype=np.int64)
    return FrozenCorpus(
        path=tmp_path / "tiny-corpus",
        manifest={"name": "tiny", "corpus_sha256": sha},
        moves=flat,
        offsets=offsets,
        winner=np.zeros(3, dtype=np.int8),
        source_game_id=np.arange(3, dtype=np.int64),
        split=np.arange(3, dtype=np.int8),
        _samples={
            "train": _sample_split(0, games[0]),
            "val": _sample_split(1, games[1]),
            "test": _sample_split(2, games[2]),
        },
    )


def test_variant_registry_is_exact_and_overrides_are_typed():
    # MantisNet, plus one arm per §29 MantisNet-ACT preset. Each ACT arm's own
    # registration contract — its configuration dataclass, the collator bound
    # to its preset, and the builder-side overrides it refuses — is checked in
    # tests/act/test_act_lab_variants.py.
    assert sorted(VARIANTS) == sorted({"mantis", *ACT_PRESETS})
    assert VARIANTS["mantis"].rust_collate is True
    parsed = parse_model_kw(["h=16", "blocks=1", "dropout=0.25"])
    assert parsed == {"h": 16, "blocks": 1, "dropout": 0.25}
    model, normalized, spec = build_variant("mantis", {"h": 16, "heads": 2})
    assert normalized == {"h": 16, "heads": 2}
    assert model.cfg.h == 16 and spec is VARIANTS["mantis"]
    assert derived_cell_name("mantis", {"h": 16, "blocks": 1}) == "mantis+blocks1+h16"

    with pytest.raises(ValueError, match="unknown MantisConfig field"):
        parse_model_kw(["width=16"])
    with pytest.raises(ValueError, match="must be an int"):
        parse_model_kw(["h=wide"])
    with pytest.raises(ValueError, match="duplicate"):
        parse_model_kw(["h=8", "h=16"])


def test_tiny_cell_trains_all_heads_and_writes_complete_artifacts(tmp_path):
    corpus = tiny_corpus(tmp_path)
    cell = tmp_path / "sweep" / "tiny" / "s5"
    result = train_cell(
        corpus,
        cell,
        variant="mantis",
        model_kw=TINY_MODEL_KW,
        seed=5,
        config=TrainConfig(
            epochs=2,
            batch_size=12,
            pair_budget=100_000,
            cell_budget=100_000,
            collect_pair_budget=100_000,
            collect_cell_budget=100_000,
            lr=1e-3,
            device="cpu",
            autocast=False,
            compile=False,
        ),
    )
    assert {path.name for path in cell.iterdir()} == {
        "config.json",
        "metrics.jsonl",
        "checkpoint_final.pt",
    }
    rows = result["metrics"]
    assert len(rows) == 2
    for row in rows:
        for name in ("policy_loss", "critic_ce", "value_loss", "seconds", "samples_per_second"):
            assert np.isfinite(row[name])
        assert row["fit_steps"] > 0
        assert row["val"]["samples"] == len(corpus.split_samples("val"))
    first = sum(rows[0][name] for name in ("policy_loss", "critic_ce", "value_loss"))
    last = sum(rows[-1][name] for name in ("policy_loss", "critic_ce", "value_loss"))
    assert last < first

    config = json.loads((cell / "config.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(cell / "checkpoint_final.pt", map_location="cpu", weights_only=False)
    assert config["recipe"]["loss_weights"] == {
        "critic": 1.0,
        "policy": 1.0,
        "state_value": 1.0,
    }
    assert checkpoint["lab_cell_format"] == 1
    assert checkpoint["variant"] == "mantis"
    assert checkpoint["model_kw"] == TINY_MODEL_KW
    assert checkpoint["corpus_sha256"] == corpus.sha256
    assert set(checkpoint["model"]) == set(result["model"].state_dict())

    with pytest.raises(FileExistsError, match="nonempty"):
        train_cell(corpus, cell, model_kw=TINY_MODEL_KW, epochs=1)


TINY_ACT_MODEL_KW = {
    "d_inv": 16,
    "d_axis": 8,
    "d_rel": 8,
    "num_heads": 2,
    "ffn_mult": 1,
    "state_blocks": 1,
    "action_blocks": 1,
    "policy_private_blocks": 1,
    "critic_private_blocks": 1,
}


def _tiny_act_config() -> TrainConfig:
    return TrainConfig(
        epochs=2,
        batch_size=12,
        pair_budget=100_000,
        cell_budget=100_000,
        graph_cell_budget=100_000,
        collect_pair_budget=100_000,
        collect_cell_budget=100_000,
        collect_graph_cell_budget=100_000,
        lr=1e-3,
        device="cpu",
        autocast=False,
        compile=False,
    )


def test_an_act_cell_trains_the_two_terms_it_holds_and_records_only_those(tmp_path):
    """§29 gives MantisNet-ACT no state-value head, so the recipe drops term 3.

    The critic is not what is missing: `critic_ce` is the action-value
    categorical both architectures train, and it is present here. Nothing in
    this cell adds a head to the architecture to satisfy the harness.
    """

    corpus = tiny_corpus(tmp_path)
    cell = tmp_path / "sweep" / "act" / "s0"
    result = train_cell(
        corpus,
        cell,
        variant="full_act_v4",
        model_kw=TINY_ACT_MODEL_KW,
        seed=0,
        config=_tiny_act_config(),
    )

    config = json.loads((cell / "config.json").read_text(encoding="utf-8"))
    assert config["recipe"]["loss_weights"] == {"critic": 1.0, "policy": 1.0}

    for row in result["metrics"]:
        assert np.isfinite(row["policy_loss"]) and np.isfinite(row["critic_ce"])
        assert "value_loss" not in row
        assert np.isfinite(row["val"]["imitation_top1"])
        assert "value_sign_accuracy" not in row["val"]
        assert "value_mae" not in row["val"]
    first = sum(result["metrics"][0][name] for name in ("policy_loss", "critic_ce"))
    last = sum(result["metrics"][-1][name] for name in ("policy_loss", "critic_ce"))
    assert last < first

    scores = evaluate_cell(cell, corpus, device="cpu")
    assert set(scores["horizon"]) == {"v_hat"}
    assert scores["flags"]["state_value_scored"] is False
    assert scores["variant"] == "full_act_v4"


def test_an_enabled_state_value_head_is_refused_rather_than_left_unscored(tmp_path):
    """§23.3's head is a scalar auxiliary, not the binned readout term 3 reads.

    Instantiating it under this recipe would count parameters that nothing
    trains and no score channel reports, so the seam refuses instead.
    """

    corpus = tiny_corpus(tmp_path)
    cell = tmp_path / "sweep" / "act-sv" / "s0"
    with pytest.raises(ValueError, match="enable_state_value_head"):
        train_cell(
            corpus,
            cell,
            variant="full_act_v4",
            model_kw={**TINY_ACT_MODEL_KW, "enable_state_value_head": True},
            seed=0,
            config=_tiny_act_config(),
        )


def test_cosine_lr_schedule_anneals_and_is_recorded(tmp_path):
    with pytest.raises(ValueError, match="lr_schedule"):
        TrainConfig(lr_schedule="linear")

    cfg = TrainConfig(epochs=4, lr=1e-3, lr_schedule="cosine")
    rates = [cfg.epoch_lr(epoch) for epoch in range(1, 5)]
    assert rates[0] == pytest.approx(1e-3)
    assert rates == sorted(rates, reverse=True)
    assert rates[-1] == pytest.approx(1e-3 * 0.5 * (1 + math.cos(math.pi * 3 / 4)))
    assert TrainConfig(epochs=4, lr=1e-3).epoch_lr(4) == 1e-3

    corpus = tiny_corpus(tmp_path)
    cell = tmp_path / "sweep" / "tiny-cosine" / "s0"
    result = train_cell(
        corpus,
        cell,
        variant="mantis",
        model_kw=TINY_MODEL_KW,
        config=TrainConfig(
            epochs=2,
            batch_size=12,
            pair_budget=100_000,
            cell_budget=100_000,
            collect_pair_budget=100_000,
            collect_cell_budget=100_000,
            lr=1e-3,
            lr_schedule="cosine",
            device="cpu",
            autocast=False,
            compile=False,
        ),
    )
    rows = result["metrics"]
    assert [row["lr"] for row in rows] == [pytest.approx(1e-3), pytest.approx(5e-4)]
    config = json.loads((cell / "config.json").read_text(encoding="utf-8"))
    assert config["recipe"]["lr_schedule"] == "cosine"


def test_ema_decay_validation():
    with pytest.raises(ValueError, match="ema_decay"):
        TrainConfig(ema_decay=1.0)
    with pytest.raises(ValueError, match="ema_decay"):
        TrainConfig(ema_decay=-0.1)


def test_tiny_cell_writes_ema_checkpoint(tmp_path):
    corpus = tiny_corpus(tmp_path)
    cell = tmp_path / "sweep" / "tiny-ema" / "s0"
    train_cell(
        corpus,
        cell,
        variant="mantis",
        model_kw=TINY_MODEL_KW,
        config=TrainConfig(
            epochs=2,
            batch_size=12,
            pair_budget=100_000,
            cell_budget=100_000,
            collect_pair_budget=100_000,
            collect_cell_budget=100_000,
            lr=1e-3,
            ema_decay=0.5,
            device="cpu",
            autocast=False,
            compile=False,
        ),
    )

    final = torch.load(
        cell / "checkpoint_final.pt", map_location="cpu", weights_only=False
    )
    ema = torch.load(
        cell / "checkpoint_ema.pt", map_location="cpu", weights_only=False
    )
    assert set(ema) == set(final)
    assert set(ema["model"]) == set(final["model"])
    assert all(
        ema["model"][name].dtype == final["model"][name].dtype
        for name in final["model"]
    )
    assert any(
        not torch.equal(ema["model"][name], final["model"][name])
        for name in final["model"]
    )
    config = json.loads((cell / "config.json").read_text(encoding="utf-8"))
    assert config["recipe"]["ema_decay"] == 0.5


def test_parameter_budget_refuses_before_creating_a_cell(tmp_path):
    model, _overrides, _spec = build_variant("mantis", TINY_MODEL_KW)
    count = count_parameters(model)
    refuse_param_budget(count, count, 0.0)
    with pytest.raises(ValueError, match=rf"parameter count {count}.*bounds"):
        refuse_param_budget(count, count + 1, 0.0)

    destination = tmp_path / "never-created" / "s0"
    with pytest.raises(ValueError, match="parameter count"):
        train_cell(
            tiny_corpus(tmp_path),
            destination,
            model_kw=TINY_MODEL_KW,
            epochs=1,
            param_budget=1,
            param_tol=0.0,
        )
    assert not destination.exists()
