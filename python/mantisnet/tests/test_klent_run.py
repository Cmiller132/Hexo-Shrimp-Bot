"""Collection stats and the run driver: what a run persists, exactly."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from mantisnet.klent import Collector, Episode, collection_stats
from mantisnet.klent.run import load_checkpoint, main

from .heuristic import heuristic_evaluate


def _episode(winner, movers, v_hats, moves_remaining=None):
    n = len(movers)
    return Episode(
        moves=[(0, 0)] * n,
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
    capped = _episode(winner=None, movers=[1], v_hats=[9.0])
    stats = collection_stats([won, capped])

    assert stats["f"] == 0.5
    assert stats["p0_win_rate"] == 1.0
    assert stats["first_stone_win_rate"] == 1.0
    assert stats["won_length_mean"] == 3.0
    # Calibration reads the won episode only: z = (+1, -1, +1) for movers
    # (0, 1, 0) with winner 0, so the capped episode's v = 9 never appears.
    assert stats["v_hat_winner_mean"] == pytest.approx((0.5 + 1.0) / 2)
    assert stats["v_hat_loser_mean"] == pytest.approx(-0.5)
    assert stats["v_hat_mae"] == pytest.approx((0.5 + 0.5 + 0.0) / 3)


def test_collection_stats_from_real_collection():
    rng = np.random.default_rng(21)
    episodes, _ = Collector(6, 200, 0.1, 0.03, rng).collect(heuristic_evaluate, 6)
    stats = collection_stats(episodes)
    assert 0.0 <= stats["f"] <= 1.0
    assert 0.0 <= stats["p0_win_rate"] <= 1.0
    assert 0.0 <= stats["first_stone_win_rate"] <= 1.0
    assert stats["v_hat_mae"] >= 0.0


def test_run_resume_and_artifacts(tmp_path):
    out = tmp_path / "run"
    args = [
        "--out", str(out), "--games", "2", "--envs", "2", "--cap", "24",
        "--batch", "64", "--checkpoint-every", "1", "--device", "cpu",
        "--seed", "3",
    ]
    main(args + ["--iterations", "2"])

    config = json.loads((out / "config.json").read_text())
    assert config["klent"]["tau"] == 0.1 and config["klent"]["lam"] == 0.03
    assert config["eval_time"] == 0.1
    assert config["eval_depth"] is None
    assert config["eval_sims"] == 32
    assert "MODEL_REPR_VERSION" in config["versions"]
    lines = (out / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 2
    row = json.loads(lines[-1])
    for key in ("iteration", "seconds", "f", "acting_kl", "v_hat_mae"):
        assert key in row
    assert (out / "checkpoint_000002.pt").exists()

    # Resume finds the latest checkpoint and appends rather than restarting;
    # a knob changed on resume lands on the record.
    main(args + [
        "--iterations", "3", "--resume", "--lam", "0.05",
        "--eval-time", "0.2", "--eval-depth", "3", "--eval-sims", "0",
    ])
    lines = (out / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 3
    assert json.loads(lines[-1])["iteration"] == 2
    assert (out / "checkpoint_000003.pt").exists()
    invocations = [
        json.loads(line)
        for line in (out / "invocations.jsonl").read_text().splitlines()
    ]
    assert [inv["start_iteration"] for inv in invocations] == [0, 2]
    assert invocations[0]["klent"]["lam"] == 0.03
    assert invocations[1]["klent"]["lam"] == 0.05
    assert invocations[1]["eval_time"] == 0.2
    assert invocations[1]["eval_depth"] == 3
    assert invocations[1]["eval_sims"] == 0

    # A fresh start into a used directory is refused.
    with pytest.raises(SystemExit, match="not empty"):
        main(args + ["--iterations", "1"])

    # In-driver eval is SealBot or nothing.
    with pytest.raises(SystemExit, match="sealbot"):
        main(["--out", str(tmp_path / "x"), "--iterations", "1", "--eval-every", "1"])
    with pytest.raises(SystemExit, match="eval-depth"):
        main(["--out", str(tmp_path / "y"), "--iterations", "1", "--eval-depth", "0"])
    with pytest.raises(SystemExit, match="eval-sims"):
        main(["--out", str(tmp_path / "z"), "--iterations", "1", "--eval-sims", "-1"])
    with pytest.raises(SystemExit, match="eval-time"):
        main(["--out", str(tmp_path / "w"), "--iterations", "1", "--eval-time", "0"])


def test_eval_in_driver_leaves_training_untouched(tmp_path):
    """``evaluate_fn`` reports into the metrics row without perturbing the
    training stream: the same run with eval off is bit-identical."""
    from mantisnet import MantisConfig, MantisNet
    from mantisnet.klent import run as run_mod
    from mantisnet.klent.train import KlentConfig

    def build():
        torch.manual_seed(2)
        model = MantisNet(MantisConfig(h=32, blocks=1, heads=2, value_queries=2, value_bins=5))
        return model, torch.optim.Adam(model.parameters())

    cfg = KlentConfig(games_per_iteration=2, envs=2, ply_cap=24, batch_size=64)
    fake_eval = lambda m, done, tel: {"eval_score": 0.5, "eval_capped": 0, "eval_games": 2}  # noqa: E731
    for name, eval_every in (("plain", 0), ("evaled", 1)):
        model, opt = build()
        run_mod.run_training(
            model, opt, cfg, iterations=2, out_dir=tmp_path / name,
            rng=np.random.default_rng(5), checkpoint_every=2,
            eval_every=eval_every, evaluate_fn=fake_eval if eval_every else None,
        )

    read = lambda name: [  # noqa: E731
        json.loads(line)
        for line in (tmp_path / name / "metrics.jsonl").read_text().splitlines()
    ]
    for plain, evaled in zip(read("plain"), read("evaled"), strict=True):
        assert evaled["eval_score"] == 0.5
        assert evaled["eval_games"] == 2 and evaled["eval_capped"] == 0
        assert evaled["eval_seconds"] >= 0
        for key in plain:
            if key != "seconds":
                assert evaled[key] == plain[key], key


def test_crossplay_plays_every_checkpoint_pair(tmp_path):
    """The A7 matrix plays every checkpoint pair of a run."""
    from mantisnet.klent.crossplay import cross_play

    out = tmp_path / "run"
    main(["--out", str(out), "--seed", "4", "--checkpoint-every", "1",
          "--iterations", "2", "--games", "2", "--cap", "16", "--batch", "64",
          "--device", "cpu"])

    rows = cross_play(out, games=2, ply_cap=12, device="cpu", seed=0)
    assert [(r["a"], r["b"]) for r in rows] == [
        ("checkpoint_000001.pt", "checkpoint_000002.pt")
    ]
    assert 0.0 <= rows[0]["score_a"] <= 1.0
    assert rows[0]["capped"] <= 2


def _tiny_run(out_dir, iterations, checkpoint_every=100, seed=2):
    from mantisnet import MantisConfig, MantisNet
    from mantisnet.klent import run as run_mod
    from mantisnet.klent.train import KlentConfig

    torch.manual_seed(seed)
    model = MantisNet(MantisConfig(h=32, blocks=1, heads=2, value_queries=2, value_bins=5))
    run_mod.run_training(
        model,
        torch.optim.Adam(model.parameters()),
        KlentConfig(games_per_iteration=2, envs=2, ply_cap=24, batch_size=64),
        iterations=iterations,
        out_dir=out_dir,
        rng=np.random.default_rng(seed),
        checkpoint_every=checkpoint_every,
    )


def test_status_heartbeat(tmp_path):
    """status.json is the deck's contract: per-lane, nulled at exit."""
    _tiny_run(tmp_path, iterations=2)
    status = json.loads((tmp_path / "status.json").read_text())
    assert set(status) == {"updated", "iteration", "collect", "fit", "eval"}
    assert status["iteration"] == 2
    assert status["collect"] is None and status["fit"] is None and status["eval"] is None
    assert status["updated"] is not None


def test_stop_sentinel(tmp_path):
    """STOP ends the run after the current iteration, durably and consumed."""
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "STOP").touch()
    _tiny_run(tmp_path, iterations=5)
    assert len((tmp_path / "metrics.jsonl").read_text().splitlines()) == 1
    assert (tmp_path / "checkpoint_000001.pt").exists()
    assert not (tmp_path / "STOP").exists()


def test_checkpoint_sentinel(tmp_path):
    """CHECKPOINT forces one durable write at the next commit point and the
    run continues to its requested end."""
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "CHECKPOINT").touch()
    _tiny_run(tmp_path, iterations=2)
    assert (tmp_path / "checkpoint_000001.pt").exists()  # the sentinel's
    assert (tmp_path / "checkpoint_000002.pt").exists()  # end-of-run
    assert not (tmp_path / "CHECKPOINT").exists()
    assert len((tmp_path / "metrics.jsonl").read_text().splitlines()) == 2


def test_starvation_stops_the_run(tmp_path, monkeypatch):
    """The unattended-run guard: consecutive empty-buffer iterations end the
    run with a checkpoint instead of collecting dead games until morning."""
    from mantisnet import MantisConfig, MantisNet
    from mantisnet.klent import run as run_mod
    from mantisnet.klent.train import KlentConfig

    starved = {"f": 0.0, "acting_kl": 0.0, "acting_norm_entropy": 0.99}
    monkeypatch.setattr(run_mod, "collect_episodes", lambda *a, **k: ([], dict(starved)))

    torch.manual_seed(0)
    model = MantisNet(MantisConfig(h=32, blocks=1, heads=2, value_queries=2, value_bins=5))
    run_mod.run_training(
        model,
        torch.optim.Adam(model.parameters()),
        KlentConfig(games_per_iteration=8),
        iterations=50,
        out_dir=tmp_path,
        rng=np.random.default_rng(0),
        checkpoint_every=100,
        starve_limit=3,
    )
    lines = (tmp_path / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 3  # the limit, not the 50 requested
    assert (tmp_path / "checkpoint_000003.pt").exists()


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
