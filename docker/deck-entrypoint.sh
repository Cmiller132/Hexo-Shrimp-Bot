#!/usr/bin/env bash
set -euo pipefail

cd /workspace/python/mantisnet

# Unconditional, like the train launchers: an import-only check kept a
# stale extension alive across representation bumps (the long-lived deck
# container served MODEL_REPR_VERSION 2 against v3 checkpoints), and cargo
# makes an up-to-date rebuild cheap.
maturin develop --release -m ../hexo-py/Cargo.toml

sealbot_root="${SEALBOT_ROOT:-/sealbot}"
sealbot_variant="${SEALBOT_VARIANT:-current}"
if ! compgen -G "${sealbot_root}/${sealbot_variant}/minimax_cpp*.so" >/dev/null; then
  if [[ ! -f "${sealbot_root}/${sealbot_variant}/setup.py" ]]; then
    echo "SealBot extension is missing and ${sealbot_root}/${sealbot_variant}/setup.py does not exist" >&2
    exit 1
  fi
  (
    cd "${sealbot_root}/${sealbot_variant}"
    python setup.py build_ext --inplace
  )
fi

if [[ ! -f /workspace/frontend/dist/index.html ]]; then
  echo "frontend/dist is missing; run: docker compose run --rm frontend-build" >&2
  exit 1
fi

exec uvicorn mantisnet.deck.app:app --host 0.0.0.0 --port 8000
