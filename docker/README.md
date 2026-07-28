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
dependency set, uv-managed CPython 3.13) on PATH and SealBot mounted
read-only at `/sealbot`. First use per container — build the engine
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
