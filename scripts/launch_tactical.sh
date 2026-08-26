#!/bin/bash
# Launch tactical-1: KLENT at the production cell-node configuration plus
# `action_tactical` (the §4.4 tactical scalars), at MODEL_REPR_VERSION 8,
# initialized from a supervised lab cell. The reference recipe is passed
# explicitly, as it must be on every launch: the CLI defaults are the paper's
# coefficients and differ.
#
# The chunk budgets change memory and speed only - gradients and metrics are
# identical for any chunking.
#
# STRIX_SEAT_ADDR must be set: the strix eval seat reaches the WSL-side bridge
# through it, and `set -u` aborts the launch when it is unset.
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
