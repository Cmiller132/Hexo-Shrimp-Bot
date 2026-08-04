"""Packed metric blocks for lab-cell and production checkpoints."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from mantisnet import MantisConfig, MantisNet
from mantisnet.klent.run import save_checkpoint
from mantisnet.lab.evaluate import DISTANCE_BUCKETS, evaluate_cell, evaluate_checkpoint
from mantisnet.lab.train import TrainConfig, train_cell

from .test_lab_train import TINY_MODEL_KW, tiny_corpus


@pytest.fixture(scope="module")
def evaluated(tmp_path_factory):
    root = tmp_path_factory.mktemp("lab-evaluate")
    corpus = tiny_corpus(root)
    cell = root / "sweep" / "tiny" / "s0"
    train_cell(
        corpus,
        cell,
        model_kw=TINY_MODEL_KW,
        seed=0,
        config=TrainConfig(
            epochs=1,
            batch_size=12,
            pair_budget=100_000,
            cell_budget=100_000,
            collect_pair_budget=100_000,
            collect_cell_budget=100_000,
            device="cpu",
        ),
    )
    cell_scores = evaluate_cell(cell, corpus, split="test", device="cpu")

    torch.manual_seed(3)
    production_model = MantisNet(MantisConfig())
    optimizer = torch.optim.Adam(production_model.parameters(), lr=1e-3)
    production_path = root / "checkpoint_000001.pt"
    save_checkpoint(
        production_path,
        production_model,
        optimizer,
        iteration=1,
        rng=np.random.default_rng(4),
    )
    production_out = root / "production-scores.json"
    production_scores = evaluate_checkpoint(
        production_path,
        corpus,
        out=production_out,
        split="test",
        device="cpu",
    )
    return corpus, cell, cell_scores, production_path, production_out, production_scores


def _assert_metric_shape(scores, expected_n: int):
    labels = [label for label, _lower, _upper in DISTANCE_BUCKETS]
    imitation = scores["imitation"]
    assert imitation["overall"]["n"] == expected_n
    assert 0.0 <= imitation["overall"]["top1"] <= 1.0
    assert 0.0 <= imitation["overall"]["top3"] <= 1.0
    assert list(imitation["by_distance"]) == labels
    assert sum(bucket["n"] for bucket in imitation["by_distance"].values()) == expected_n
    for channel in scores["horizon"].values():
        assert list(channel) == labels
        assert sum(bucket["n"] for bucket in channel.values()) == expected_n
        for bucket in channel.values():
            if bucket["n"]:
                assert 0.0 <= bucket["sign_accuracy"] <= 1.0
                assert np.isfinite(bucket["mae"])
                assert np.isfinite(bucket["mean_prediction"])
                assert np.isfinite(bucket["mean_abs_prediction"])
            else:
                assert bucket == {
                    "n": 0,
                    "sign_accuracy": None,
                    "mae": None,
                    "mean_prediction": None,
                    "mean_abs_prediction": None,
                }


def test_lab_cell_scores_both_value_channels_and_writes_in_place(evaluated):
    corpus, cell, scores, _production_path, _production_out, _production_scores = evaluated
    assert (cell / "scores.json").is_file()
    assert "ema" not in scores
    assert scores["checkpoint"]["kind"] == "lab_cell"
    assert scores["checkpoint"]["param_count"] > 0
    assert scores["corpus"] == {"name": corpus.name, "sha256": corpus.sha256}
    assert scores["flags"]["state_value_scored"] is True
    assert set(scores["horizon"]) == {"state_value", "v_hat"}
    _assert_metric_shape(scores, len(corpus.split_samples("test")))


def test_production_checkpoint_exercises_v_hat_but_never_scores_state_value(evaluated):
    corpus, _cell, _cell_scores, checkpoint, out, scores = evaluated
    assert out.is_file()
    assert scores["checkpoint"]["kind"] == "production_klent"
    assert scores["checkpoint"]["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert scores["flags"]["state_value_scored"] is False
    assert set(scores["horizon"]) == {"v_hat"}
    _assert_metric_shape(scores, len(corpus.split_samples("test")))

    with pytest.raises(ValueError, match="--out is required"):
        evaluate_checkpoint(checkpoint, corpus, out=None, device="cpu")


def test_evaluate_ema_cell_and_refuse_missing_ema(tmp_path, evaluated):
    corpus = tiny_corpus(tmp_path)
    ema_cell = tmp_path / "sweep" / "tiny-ema" / "s0"
    train_cell(
        corpus,
        ema_cell,
        model_kw=TINY_MODEL_KW,
        config=TrainConfig(
            epochs=2,
            batch_size=12,
            pair_budget=100_000,
            cell_budget=100_000,
            collect_pair_budget=100_000,
            collect_cell_budget=100_000,
            ema_decay=0.5,
            device="cpu",
        ),
    )
    scores = evaluate_cell(ema_cell, corpus, ema=True, device="cpu")
    assert scores["ema"] is True
    assert (ema_cell / "scores_ema.json").is_file()

    no_ema_corpus, no_ema_cell, *_rest = evaluated
    with pytest.raises(FileNotFoundError, match=r"tiny.*without ema_decay"):
        evaluate_cell(no_ema_cell, no_ema_corpus, ema=True, device="cpu")
