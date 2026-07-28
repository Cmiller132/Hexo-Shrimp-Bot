"""CPU heuristic run driver used only by the Compose deck smoke."""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch

from mantisnet import MantisConfig, MantisNet
from mantisnet.klent.run import _versions, save_checkpoint
from mantisnet.klent.selfplay import Collector, collection_stats, episode_samples
from mantisnet.klent.telemetry import open_telemetry
from mantisnet.klent.train import KlentConfig
from tests.heuristic import heuristic_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--envs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args, _unknown = parser.parse_known_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} exists and is not empty")
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = KlentConfig(
        games_per_iteration=args.games, envs=args.envs, ply_cap=64,
        batch_size=64, device="cpu", autocast=False, compile=False,
    )
    model_cfg = MantisConfig(
        h=32, blocks=1, heads=2, value_queries=2, value_bins=5,
        policy_hidden=32, value_hidden=32,
    )
    config = {
        "klent": dataclasses.asdict(cfg), "model": dataclasses.asdict(model_cfg),
        "iterations": args.iterations, "checkpoint_every": 1, "eval_every": 0,
        "seed": args.seed, "versions": _versions(),
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (args.out / "invocations.jsonl").write_text(
        json.dumps({"start_iteration": 0, **config}) + "\n", encoding="utf-8"
    )
    rng = np.random.default_rng(args.seed)
    collector = Collector(
        args.envs, cfg.ply_cap, cfg.tau, cfg.lam, rng,
        pair_budget=cfg.collect_pair_budget, cell_budget=cfg.collect_cell_budget,
    )
    model = MantisNet(model_cfg)
    optimizer = torch.optim.Adam(model.parameters())
    with open_telemetry(args.out) as writer:
        writer.begin_run(config, _versions(), 0)
        for iteration in range(args.iterations):
            started = time.monotonic()
            episodes, acting = collector.collect(heuristic_evaluate, args.games)
            metrics = collection_stats(episodes) | acting | {
                "iteration": iteration,
                "buffer_samples": sum(
                    len(episode_samples(ep, cfg.lam_ret)) for ep in episodes
                ),
                "seconds": time.monotonic() - started,
            }
            writer.write_iteration(metrics, episodes, {})
            done = iteration + 1
            save_checkpoint(
                args.out / f"checkpoint_{done:06d}.pt",
                model, optimizer, done, rng,
            )
            (args.out / "status.json").write_text(json.dumps({
                "updated": "smoke", "iteration": done,
                "collect": None, "fit": None, "eval": None,
            }), encoding="utf-8")
            print(f"iteration {iteration}: heuristic fixture complete", flush=True)


if __name__ == "__main__":
    main()
