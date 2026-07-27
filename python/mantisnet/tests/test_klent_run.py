"""Collection stats and the run driver: what a run persists, exactly."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from mantisnet.klent import Episode, collection_stats, play_episodes
from mantisnet.klent.run import load_checkpoint, main

from .test_klent_pipeline import heuristic_evaluate


def _episode(winner, movers, v_hats, seed_len=0, moves_remaining=None):
    n = len(movers)
    return Episode(
        moves=[(0, 0)] * (seed_len + n),
        seed_len=seed_len,
        winner=winner,
        moves_remaining=moves_remaining or [1] * n,
        movers=movers,
        ranks=[0] * n,
        improved=[np.ones(1, dtype=np.float32)] * n,
        v_hats=v_hats,
    )


def test_collection_stats_by_hand():
    won = _episode(
        winner=0,
        movers=[0, 1, 0],
        v_hats=[0.5, -0.5, 1.0],
        moves_remaining=[1, 1, 2],  # the win lands on a first stone
    )
    capped = _episode(winner=None, movers=[1], v_hats=[9.0], seed_len=4)
    stats = collection_stats([won, capped])

    assert stats["f_unseeded"] == 1.0  # the won one is unseeded
    assert stats["f_seeded"] == 0.0
    assert stats["p0_win_rate"] == 1.0
    assert stats["first_stone_win_rate"] == 1.0
    assert stats["seed_len_mean"] == 4.0 and stats["seed_len_max"] == 4
    # Calibration reads the won episode only: z = (+1, -1, +1) for movers
    # (0, 1, 0) with winner 0, so the capped episode's v = 9 never appears.
    assert stats["v_hat_winner_mean"] == pytest.approx((0.5 + 1.0) / 2)
    assert stats["v_hat_loser_mean"] == pytest.approx(-0.5)
    assert stats["v_hat_mae"] == pytest.approx((0.5 + 0.5 + 0.0) / 3)


def test_collection_stats_from_real_collection():
    rng = np.random.default_rng(21)
    episodes, _ = play_episodes(
        heuristic_evaluate, [[] for _ in range(6)], ply_cap=200, tau=0.1, lam=0.03, rng=rng
    )
    stats = collection_stats(episodes)
    assert 0.0 <= stats["p0_win_rate"] <= 1.0
    assert 0.0 <= stats["first_stone_win_rate"] <= 1.0
    assert stats["v_hat_mae"] >= 0.0
    assert stats["f_seeded"] != stats["f_seeded"] or stats["f_seeded"] >= 0  # nan ok


def test_run_resume_and_artifacts(tmp_path):
    out = tmp_path / "run"
    args = [
        "--out", str(out), "--games", "2", "--cap", "24", "--batch", "64",
        "--checkpoint-every", "1", "--device", "cpu", "--seed", "3",
        "--seed-cut", "1", "3",
    ]
    main(args + ["--iterations", "2"])

    config = json.loads((out / "config.json").read_text())
    assert config["klent"]["tau"] == 0.1 and config["klent"]["lam"] == 0.03
    assert "MODEL_REPR_VERSION" in config["versions"]
    lines = (out / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 2
    row = json.loads(lines[-1])
    for key in ("iteration", "seconds", "f_seeded", "acting_kl", "v_hat_mae"):
        assert key in row
    assert (out / "checkpoint_000002.pt").exists()

    # Resume finds the latest checkpoint and appends rather than restarting.
    main(args + ["--iterations", "3", "--resume"])
    lines = (out / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 3
    assert json.loads(lines[-1])["iteration"] == 2
    assert (out / "checkpoint_000003.pt").exists()

    # A fresh start into a used directory is refused.
    with pytest.raises(SystemExit, match="not empty"):
        main(args + ["--iterations", "1"])


def test_checkpoint_refuses_version_drift(tmp_path):
    from mantisnet import MantisConfig, MantisNet

    torch.manual_seed(0)
    model = MantisNet(MantisConfig(h=32, blocks=1, heads=2, value_queries=2, value_bins=5))
    optimizer = torch.optim.Adam(model.parameters())
    rng = np.random.default_rng(0)
    path = tmp_path / "ckpt.pt"
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": 1,
        "rng_state": rng.bit_generator.state,
        "versions": {"MODEL_REPR_VERSION": -1},
    }
    torch.save(ckpt, path)
    with pytest.raises(ValueError, match="versions"):
        load_checkpoint(path, model, optimizer, rng)
