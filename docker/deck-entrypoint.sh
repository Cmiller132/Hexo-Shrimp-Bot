#!/usr/bin/env bash
set -euo pipefail

cd /workspace/python/mantisnet

if ! python -c 'import hexo_py' >/dev/null 2>&1; then
  maturin develop --release -m ../hexo-py/Cargo.toml
fi

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
