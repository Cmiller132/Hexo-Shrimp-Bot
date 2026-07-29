# Container environment

## Purpose

`docker/` defines the CUDA-enabled Linux environment for MantisNet training and
the control deck. The image contains toolchains and locked dependencies; source
and run artifacts come from repository bind mounts. Compose also provides the
one-shot frontend build service.

## Public surface

[`compose.yaml`](compose.yaml) defines three services:

| Service | Contract |
| --- | --- |
| `train` | Interactive MantisNet shell with CUDA, Python, Rust, and SealBot |
| `frontend-build` | Node 22 build that writes `frontend/dist` |
| `deck` | Port-8000 FastAPI service, static SPA, GPU inference, and run control |

[`Dockerfile`](Dockerfile) builds `mantisnet-train` from CUDA 12.8 on Ubuntu
24.04. It installs Rust, `uv`, CPython 3.13, all locked MantisNet dependency
groups, maturin, setuptools, and pybind11 into `/opt/venv`.

The repository is mounted at `/workspace`. The host SealBot checkout is mounted
at `/sealbot`. Linux Rust artifacts use `/workspace/target-wsl`.

The `deck` entry point:

1. builds `hexo_py` when it is not importable;
2. builds the configured SealBot extension when its Linux module is absent;
3. requires `/workspace/frontend/dist/index.html`;
4. starts Uvicorn on `0.0.0.0:8000`.

Relevant Compose environment variables are:

| Variable | Meaning |
| --- | --- |
| `REPO_ROOT` | Host repository or worktree mounted at `/workspace` |
| `SEALBOT_ROOT` | In-container SealBot root |
| `SEALBOT_VARIANT` | SealBot subdirectory selected by the entry point |
| `DECK_RUNS_ROOT` | Run directory scanned by the deck |
| `DECK_FRONTEND_DIST` | Static frontend directory |
| `DECK_DEVICE` | Inference device, default `cuda` |
| `DECK_RUN_COMMAND` | Optional deck training command override |

## Run / test

From the repository root in the WSL environment:

```sh
docker compose -f docker/compose.yaml build train
docker compose -f docker/compose.yaml run --rm train
```

Inside the `train` service, build the mounted extension and run tests:

```sh
maturin develop --release -m ../hexo-py/Cargo.toml
python -m pytest tests -q
```

Build and start the control deck:

```sh
docker compose -f docker/compose.yaml run --rm frontend-build
docker compose -f docker/compose.yaml up -d deck
docker compose -f docker/compose.yaml logs -f deck
```

Run the CPU deck receipt:

```sh
docker compose -f docker/compose.yaml down
DECK_DEVICE=cpu \
DECK_RUN_COMMAND="python /workspace/docker/smoke-driver.py" \
docker compose -f docker/compose.yaml up -d deck
docker compose -f docker/compose.yaml exec -T deck \
  python /workspace/docker/smoke-deck.py
```

Stop the services:

```sh
docker compose -f docker/compose.yaml down
```

Use a different checkout without editing Compose:

```sh
REPO_ROOT=/mnt/d/Hexo-Shrimp-Bot-worktree \
docker compose -f docker/compose.yaml run --rm frontend-build
```

## Connections

- `python/mantisnet` supplies the Python package, lockfile, tests, and runs.
- `python/hexo-py` supplies the mounted maturin extension source.
- `frontend` supplies the SPA built into `frontend/dist`.
- `mantisnet.deck` serves that directory and the `/api` surface.
- `docker/smoke-driver.py` produces bounded receipt artifacts.
- `docker/smoke-deck.py` checks the published deck API.
- Container obligations are in
  [`docs/CONTAINER_SPEC.md`](../docs/CONTAINER_SPEC.md).
- Deck deployment obligations are in [`docs/DECK_SPEC.md`](../docs/DECK_SPEC.md).

## Invariants & gotchas

- The image contains dependencies and toolchains, not the working source tree.
- Source changes under the bind mount do not require an image rebuild.
- Changes to the Dockerfile, lock inputs, or dependency groups require a
  rebuild.
- `/opt/venv` is outside the repository mount.
- A `docker compose run --rm train` container does not preserve an extension
  installed into `/opt/venv` after the container exits.
- `target-wsl` separates Linux Rust artifacts from host-platform `target`.
- The `train` and `deck` services require the NVIDIA container runtime.
- The `deck` service refuses startup when the frontend bundle is absent.
- The SealBot mount must be writable when its Linux extension needs building.
- Deck-launched training processes are children of the deck service.
- No Docker socket is mounted and the deck does not start nested containers.
- Port 8000 is the published API, SSE, and SPA endpoint.
- `frontend-node-modules` is a named volume; `frontend/dist` remains on the
  repository mount.
- The smoke run name must not already identify a run directory.
- Add `sudo` to Docker commands only when the local Docker installation
  requires it.
