#!/bin/bash
# confirm-1: the Round-3 confirmation run of the round-1 integration tree —
# the tactical-1 configuration on the tree carrying the orbit48 residual bias
# vocabulary and the S7 cell-table trim (MODEL_REPR_VERSION 9). Round 1 of
# screen v2 kept no new knob, so the model configuration is unchanged from
# tactical-1; what this run confirms is the tree.
#
# Same recipe as tactical-1 wherever the recipe is the science:
# tau/lam/mass_floor/lam_ret/gamma/lr, games/envs/batch/cap, eval cadence,
# lean chunk budgets (2M/125k fit, 6M/600k collect — chunking changes memory
# and speed only; gradients and metrics are identical for any chunking).
#
# The initialization is the screen-v2 fixture seed-0 cell: 4 epochs over the
# full 1M-position train split on the integration tree, the strongest of the
# six fixture seeds on the scored val split (policy NLL 1.8270, critic CE
# 0.6168, top-1 0.4923) — the same prefit-from-screen-cell pattern that
# seeded cellnodes-1 and tactical-1.
#
# hexo_py is rebuilt unconditionally: the image's venv may carry a wheel from
# an older tree, and an importable-but-stale extension would otherwise pass
# the import probe and fail only at the version assert.
#
# STRIX_SEAT_ADDR must be set: the strix seat is a client of the WSL-side
# bridge, and an unset address is what killed newmodeltest at iteration 25.
set -eu
cd /workspace/python/mantisnet
export VIRTUAL_ENV=/opt/venv
export PATH=/opt/venv/bin:$PATH
export CARGO_TARGET_DIR=/workspace/target-wsl

maturin develop --release -m ../hexo-py/Cargo.toml
python -c "import hexo_py; assert hexo_py.MODEL_REPR_VERSION == 9, hexo_py.MODEL_REPR_VERSION; print('hexo_py MODEL_REPR_VERSION', hexo_py.MODEL_REPR_VERSION)"

echo "=== launching confirm-1 $(date +%T) (strix bridge ${STRIX_SEAT_ADDR}) ==="
exec python -m mantisnet.klent.run \
  --out /runs/confirm-1 \
  --iterations 2000 \
  --games 4096 --envs 1024 --cap 512 --batch 4096 \
  --tau 0.1 --lam 0.01 --mass-floor 0.2 --lam-ret 0.939 --gamma 0.99 --lr 0.001 \
  --pair-budget 2000000 --cell-budget 125000 \
  --collect-pair-budget 6000000 --collect-cell-budget 600000 \
  --checkpoint-every 5 \
  --eval-every 25 --eval-games 64 --eval-sims 32 --eval-time 0.1 \
  --sealbot /sealbot \
  --eval-seat /workspace/local/strix-eval/eval-seat.json \
  --seed 23 \
  --model-kw cell_latents=True window_attention=False cell_nodes=True cell_node_scope=all action_tactical=True \
  --init-lab-cell /runs/lab/v2/fixture/s0/checkpoint_final.pt \
  --device cuda
