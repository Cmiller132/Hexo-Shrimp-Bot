# Hexo-Shrimp-Bot

A Hexo engine and bot-training framework. The Rust workspace provides an
authoritative rules engine and match loop; the Python side provides the
MantisNet neural network, the KLENT training pipeline, and a LAN control deck.

## The game

An infinite hex board in axial coordinates `(q, r)`. `P0` opens at the origin;
after that each player places two stones per turn, with a win check after
*every* placement. Six or more of one player's stones in a row along one of the
three axes wins. A placement must be an empty cell within 8 hex steps of an
occupied cell. No draws, passes, or captures; stones are permanent.
The [hexo-engine README](crates/hexo-engine/README.md) states the rules
normatively.

## Components

### `crates/`

The Rust workspace. All crates share a version, edition, and lint policy
declared in the root `Cargo.toml`.

| Crate | Purpose |
| --- | --- |
| `hexo-engine` | Authoritative rules, game state, and storage. Compiles to `wasm32`. |
| `hexo-runner` | Game adjudication: advances the authoritative position and enforces the turn protocol. |
| `hexo-player` | The player seam that connects search to a game in progress. |
| `hexo-search` | The evaluator seam and nonblocking search sessions. |
| `hexo-records` | On-disk shard format for game records. One writer, strict reader. |
| `hexo-model` | The model-package trait (`ModelPackage`), checkpoint manifest, and probe. |
| `models/mock` | A deterministic mock model package for testing. |
| `models/mantisnet` | The Rust-side MantisNet model package; bridges to the Python evaluator at runtime. |
| `hexo-bot` | The binary entry point: init, train, and match. Embeds the Python interpreter. |

### `python/`

| Package | Purpose |
| --- | --- |
| `hexo-py` | PyO3 bindings over the engine and shared encoder. A detached leaf crate (not in the Rust workspace). |
| `mantisnet` | The MantisNet model, KLENT trainer, supervised lab harness, and control-deck server. Depends on `hexo-py`. |

`mantisnet` contains several subsystems:

- The MantisNet network and its `models/mantis_act` variant (MantisNet-ACT).
- The KLENT self-play training loop (`klent/`).
- The supervised lab bench (`lab/`) for offline architecture comparison.
- The control deck (`deck/`) that launches, monitors, and inspects training runs.

### `frontend/`

A React + Vite SPA for the control deck. Built once and served by the deck
server in `python/mantisnet`.

### `docker/`

Dockerfile, Compose configuration, and entrypoint scripts for running the
training environment. The deck and training driver both run inside this
container.

### `xtask/`

Cargo xtask binary that defines the verification gates. `cargo xtask verify`
runs every gate CI runs; `cargo xtask` lists them.

### `scripts/`

Utility scripts for the development environment.

### `docs/`

The project's document set. [docs/README.md](docs/README.md) is the full
index. Normative specifications:

| Document | Governs |
| --- | --- |
| [MANTIS_ACT_SPEC.md](docs/MANTIS_ACT_SPEC.md) | MantisNet-ACT: the v4 architecture |
| [CONTAINER_SPEC.md](docs/CONTAINER_SPEC.md) | Container crates: packaging, deployment, lifecycle |
| [KLENT_FOR_HEXO.md](docs/KLENT_FOR_HEXO.md) | KLENT training path for Hexo |

Reference documents: [KLENT_PAPER.md](docs/KLENT_PAPER.md) (the source
algorithm) and [ABLATIONS.md](docs/ABLATIONS.md) (experimental record).

### `.github/workflows/`

CI definitions: `ci.yml` and `nightly-smoke.yml`.

## Build and verify

Rust 1.88+ (declared and gated in `Cargo.toml`).

```sh
cargo xtask verify     # every gate CI runs
cargo xtask            # list the gates
```

Python:

```sh
cd python/mantisnet
uv sync                # venv, hexo-py wheel via maturin, torch
uv run pytest          # the whole suite
```

Containers, training runs, and the deck: [docker/README.md](docker/README.md).
