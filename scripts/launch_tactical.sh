#!/bin/bash
# tactical-1: KLENT successor carrying Step 5 tactical scalars (screen arm B,
# the only arm positive on every seed of the step56 factorial) on the
# recompute-in-backward trunk.
#
# Same recipe as cellnodes-1 wherever the recipe is the science:
# tau/lam/mass_floor/lam_ret/gamma/lr, games/envs/batch/cap, eval cadence.
# The chunk budgets are the lean-instrument pair (2M/125k fit, 6M/600k
# collect): the selective-recompute trunk fits at 3.5 GiB on the pinned
# instrument where cellnodes-1's budgets were sized against a 6.3 GiB fit
# path on a 12 GiB card. Chunking changes memory and speed only — gradients
# and metrics are identical for any chunking.
#
# The initialization is the arm-B screen cell: 4 epochs on cn1-late-v1
# (cellnodes-1 self-play, iterations 302-401), production config plus
# action_tactical, seed 2 — the same prefit-from-screen-cell pattern that
# seeded cellnodes-1's line.
#
# STRIX_SEAT_ADDR must be set: the strix seat is a client of the WSL-side
# bridge, and an unset address is what killed newmodeltest at iteration 25.
set -eu
cd /workspace/python/mantisnet
export VIRTUAL_ENV=/opt/venv
export PATH=/opt/venv/bin:$PATH
export CARGO_TARGET_DIR=/workspace/target-wsl

python -c "import hexo_py" 2>/dev/null || maturin develop --release -m ../hexo-py/Cargo.toml
python -c "import hexo_py; assert hexo_py.MODEL_REPR_VERSION == 8, hexo_py.MODEL_REPR_VERSION; print('hexo_py MODEL_REPR_VERSION', hexo_py.MODEL_REPR_VERSION)"

echo "=== launching tactical-1 $(date +%T) (strix bridge ${STRIX_SEAT_ADDR}) ==="
exec python -m mantisnet.klent.run \
  --out /runs/tactical-1 \
  --iterations 2000 \
  --games 4096 --envs 1024 --cap 512 --batch 4096 \
  --tau 0.1 --lam 0.01 --mass-floor 0.2 --lam-ret 0.939 --gamma 0.99 --lr 0.001 \
  --pair-budget 2000000 --cell-budget 125000 \
  --collect-pair-budget 6000000 --collect-cell-budget 600000 \
  --checkpoint-every 5 \
  --eval-every 25 --eval-games 64 --eval-sims 32 --eval-time 0.1 \
  --sealbot /sealbot \
  --eval-seat /workspace/local/strix-eval/eval-seat.json \
  --seed 22 \
  --model-kw cell_latents=True window_attention=False cell_nodes=True cell_node_scope=all action_tactical=True \
  --init-lab-cell /runs/lab/step56-epochs4/armB/s2/checkpoint_final.pt \
  --device cuda
