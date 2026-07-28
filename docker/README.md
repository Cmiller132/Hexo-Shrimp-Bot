# The container environment

The owner-stated destination: training, self-play, and everything else run
in Docker under WSL. This directory is that environment. The image holds
*dependencies only* — the repository bind-mounts at `/workspace`, so code
changes never need a rebuild; only a lock change does.

Why Linux for the GPU work, measured rather than assumed: the win is the
compiler stack — `torch.compile`'s generated kernels and hand-written
Triton are first-class on Linux and a port on Windows. The Windows-native
path stays usable for tests and quick checks.

## One-time

Docker (docker-ce) and the NVIDIA container toolkit live inside the WSL
Ubuntu distro (installed 2026-07-28; CDI spec generated at
`/etc/cdi/nvidia.yaml`). GPU access uses the `nvidia` runtime, wired in
`compose.yaml`.

```sh
cd /mnt/d/Hexo-Shrimp-Bot
sudo docker compose -f docker/compose.yaml build train
```

## Using it

```sh
sudo docker compose -f docker/compose.yaml run --rm train
```

drops into `/workspace/python/mantisnet` with `/opt/venv` (the locked
dependency set, uv-managed CPython 3.13) on PATH and SealBot mounted at
`/sealbot`. First use per container — build the engine
extension from the mounted tree:

```sh
maturin develop --release -m ../hexo-py/Cargo.toml
```

`CARGO_TARGET_DIR=/workspace/target-wsl` keeps the Linux build artifacts
out of the Windows `target/` (both are gitignored). Then the usual:

```sh
python -m pytest tests/ -q
python bench/bench_loop.py collect --checkpoint runs/<r>/checkpoint_N.pt --compile
python -m mantisnet.klent.run --out runs/<name> ...
```

Note that `maturin develop` installs into `/opt/venv`, which lives in the
container layer — a `--rm` container rebuilds it next time. For a
long-lived workflow, `docker compose up -d train` once and `exec` into it.

SealBot's `minimax_cpp` on the mount is a Windows build; evals inside the
container need a Linux build of it first. Until that lands, run SealBot
evals from the Windows side.

## Control deck

Two services complete the LAN dashboard. Both use the same repository bind
mount, so the generated frontend and run artifacts stay visible on Windows.

```sh
cd /mnt/d/Hexo-Shrimp-Bot
docker compose -f docker/compose.yaml run --rm frontend-build
docker compose -f docker/compose.yaml up -d deck
```

`frontend-build` is a one-shot Node 22 service. It runs `npm ci && npm run
build` in `frontend/`, with `node_modules` in the
`frontend-node-modules` named volume and `frontend/dist` on the repository
mount.

`deck` uses the `mantisnet-train` image, publishes port 8000, has the GPU and
SealBot mount, and restarts unless stopped. Its entrypoint refuses to serve
without `frontend/dist/index.html`. On a cold container it builds `hexo_py`
with:

```sh
maturin develop --release -m ../hexo-py/Cargo.toml
```

and, when the Linux SealBot artifact is absent, runs `python setup.py
build_ext --inplace` in `/sealbot/current`. It then starts Uvicorn on
`0.0.0.0:8000`. Training runs launched by the API are ordinary child processes
inside this service; there is no Docker socket and no Docker-in-Docker.

For a worktree, substitute it without editing Compose:

```sh
REPO_ROOT=/mnt/d/Hexo-Shrimp-Bot-ui-draft docker compose \
  -f docker/compose.yaml run --rm frontend-build
REPO_ROOT=/mnt/d/Hexo-Shrimp-Bot-ui-draft docker compose \
  -f docker/compose.yaml up -d deck
```

The CPU end-to-end receipt uses the same Compose service with a bounded
heuristic fixture driver, then talks only to the published API:

```sh
docker compose -f docker/compose.yaml down deck
DECK_DEVICE=cpu \
DECK_RUN_COMMAND="python /workspace/docker/smoke-driver.py" \
  docker compose -f docker/compose.yaml up -d deck
docker compose -f docker/compose.yaml exec -T deck \
  python /workspace/docker/smoke-deck.py
```

It launches the run through `POST /api/runs`, waits for its real heuristic
telemetry and checkpoint, and asserts the run list, iteration series, game
plies, first SSE events, a play move, checkpoint inspection, and the SPA.
Remove `python/mantisnet/runs/deck-smoke` before repeating it; the lifecycle
correctly refuses to overwrite an existing run.
