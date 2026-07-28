# The training/self-play environment: CUDA-enabled Linux with the Python
# toolchain and a Rust toolchain for the hexo_py extension. The repository
# is bind-mounted at /workspace (see compose.yaml) — this image holds
# dependencies, never source, so code iteration needs no rebuild.
#
# The dependency venv lives at /opt/venv, outside the mount, so the
# Windows-side .venv in the repo tree never shadows it.

FROM nvidia/cuda:12.8.0-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

# Rust, for building hexo_py (maturin) inside the container.
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH=/root/.cargo/bin:/root/.local/bin:$PATH

# uv (which also provides the exact CPython the lock was made with), and
# the locked dependency set. Only the lock inputs are copied, so this
# layer caches until the lock changes.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
COPY python/mantisnet/pyproject.toml python/mantisnet/uv.lock /tmp/lock/
# hexo-py is a path dependency in the lock; it is built from the mounted
# tree by maturin at run time, so the deps layer skips it.
RUN cd /tmp/lock && uv venv /opt/venv --python 3.13 \
    && uv sync --frozen --no-install-project --no-install-package hexo-py \
    && uv pip install --python /opt/venv maturin \
    && rm -rf /tmp/lock

WORKDIR /workspace/python/mantisnet
ENV PATH=/opt/venv/bin:$PATH \
    CARGO_TARGET_DIR=/workspace/target-wsl \
    VIRTUAL_ENV=/opt/venv

# The extension is built from the mounted tree at run time:
#   maturin develop --release -m ../hexo-py/Cargo.toml
# (documented in docker/README.md; kept out of the image because it
# compiles mounted source).
CMD ["/bin/bash"]
