#!/bin/bash
# cellnodes-1: KLENT successor carrying Step 13 cell nodes at the `all` scope.
#
# Same recipe as stack-939 and newmodeltest wherever the recipe is the science:
# tau/lam/mass_floor/lam_ret/gamma/lr, games/envs/batch/cap, eval cadence, seed.
# The chunk budgets are smaller than newmodeltest's because the measured fit
# path costs 1.64 GiB more with cell nodes (6.27 vs 4.64 GiB on the pinned
# instrument) and newmodeltest already ran at 11.2-11.9 GiB of a 12 GiB card.
# Chunking changes memory and speed only - gradients and metrics are identical
# for any chunking - so this buys headroom below the paging cliff and costs
# nothing else.
#
# STRIX_SEAT_ADDR must be set: the strix seat is a client of the WSL-side
# bridge, and an unset address is what killed newmodeltest at iteration 25.
set -eu
cd /workspace/python/mantisnet
export VIRTUAL_ENV=/opt/venv
export PATH=/opt/venv/bin:$PATH
export CARGO_TARGET_DIR=/workspace/target-wsl

python -c "import hexo_py" 2>/dev/null || maturin develop --release -m ../hexo-py/Cargo.toml
python -c "import hexo_py; print('hexo_py MODEL_REPR_VERSION', hexo_py.MODEL_REPR_VERSION)"

echo "=== launching cellnodes-1 $(date +%T) (strix bridge ${STRIX_SEAT_ADDR}) ==="
exec python -m mantisnet.klent.run \
  --out /runs/cellnodes-1 \
  --iterations 2000 \
  --games 4096 --envs 1024 --cap 512 --batch 4096 \
  --tau 0.1 --lam 0.01 --mass-floor 0.2 --lam-ret 0.939 --gamma 0.99 --lr 0.001 \
  --pair-budget 1400000 --cell-budget 80000 \
  --collect-pair-budget 4000000 --collect-cell-budget 400000 \
  --checkpoint-every 5 \
  --eval-every 25 --eval-games 64 --eval-sims 32 --eval-time 0.1 \
  --sealbot /sealbot \
  --eval-seat /workspace/local/strix-eval/eval-seat.json \
  --seed 21 \
  --model-kw cell_latents=True window_attention=False cell_nodes=True cell_node_scope=all \
  --init-lab-cell /runs/lab/step13-prefit/all/s3/checkpoint_final.pt \
  --device cuda
