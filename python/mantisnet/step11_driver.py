"""Step 11 single-seed cell: one (sweep, seed) per invocation.

The exact Step 5/6 arm-B recipe (production architecture: cell latents +
cell nodes at scope all, window attention off, tactical scalars on) on the
same corpus, subset, budgets, and epochs, so the cell pairs directly with
``runs/lab/step56-epochs4/armB/s<seed>`` — the only difference between the
two is the tree this driver runs on. The sweep name records which:
``step11-orbit48`` (be1dda8, plain orbit rows), ``step11b-residual``
(ab6de77, coarse table + orbit residual), ``step11b-control`` (the residual
tree with the residual frozen at zero — the old vocabulary on the new
kernel path).

    python step11_driver.py train <sweep> <seed> [epochs]
    python step11_driver.py evaluate <sweep> <seed> [epochs]
"""

import json
import sys
from pathlib import Path

MODEL_KW = dict(
    cell_latents=True,
    window_attention=False,
    cell_nodes=True,
    cell_node_scope="all",
    action_tactical=True,
    action_latents=False,
)
CORPUS = Path("runs/corpora/cn1-late-v1")
SWEEPS = ("step11-orbit48", "step11b-residual", "step11b-control")
FREEZE = {"step11b-control": ("orbit_bias",)}

CELL_BUDGET = 125_000
PAIR_BUDGET = 2_000_000
COLLECT_CELL_BUDGET = 600_000
COLLECT_PAIR_BUDGET = 6_000_000
TRAIN_SUBSET = 400_000


def main() -> None:
    mode, sweep, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    if sweep not in SWEEPS:
        raise SystemExit(f"unknown sweep {sweep!r}; expected one of {SWEEPS}")
    cell_dir = Path("runs/lab") / sweep / "armB" / f"s{seed}"

    if mode == "train":
        from mantisnet.lab.train import TrainConfig, train_cell

        cfg = TrainConfig(
            epochs=epochs,
            device="cuda",
            autocast=True,
            compile=True,
            cell_budget=CELL_BUDGET,
            pair_budget=PAIR_BUDGET,
            collect_cell_budget=COLLECT_CELL_BUDGET,
            collect_pair_budget=COLLECT_PAIR_BUDGET,
            train_subset=TRAIN_SUBSET,
            freeze=FREEZE.get(sweep, ()),
        )
        result = train_cell(
            CORPUS,
            cell_dir,
            model_kw=MODEL_KW,
            seed=seed,
            config=cfg,
        )
        print(json.dumps({
            "cell": str(cell_dir),
            "param_count": result["config"]["param_count"],
        }))
    elif mode == "evaluate":
        from mantisnet.lab.evaluate import evaluate_cell

        evaluate_cell(
            cell_dir,
            CORPUS,
            split="val",
            device="cuda",
            compile=True,
            pair_budget=COLLECT_PAIR_BUDGET,
            cell_budget=COLLECT_CELL_BUDGET,
        )
        print(json.dumps({"cell": str(cell_dir), "evaluated": True}))
    else:
        raise SystemExit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main()
