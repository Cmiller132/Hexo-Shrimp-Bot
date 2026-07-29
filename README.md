# Hexo-Shrimp-Bot

A Hexo engine and bot-training framework: an authoritative Rust rules engine
and match loop, a model-package container stack, and a Python training path
(the MantisNet network trained by KLENT) with a LAN control deck.

## The game

An infinite hex board in axial coordinates `(q, r)`. `P0` opens at the origin;
after that each player places two stones per turn, with a win check after
*every* placement. Six or more of one player's stones in a row along one of the
three axes wins. A placement must be an empty cell within 8 hex steps of an
occupied cell. No draws, passes, or captures; stones are permanent.
[docs/ENGINE_SPEC.md](docs/ENGINE_SPEC.md) states the rules normatively.

## Layout

```
Hexo-Shrimp-Bot/
  Cargo.toml              workspace root; shared version, edition, lint policy
  crates/
    hexo-engine/          authoritative rules and game state
    hexo-runner/          the authoritative game and adjudication policy
    hexo-player/          the player seam, and the loop that drives games
    hexo-search/          the evaluator seam, and the nonblocking sessions
    hexo-records/         the on-disk shard format, written once, read strictly
    hexo-model/           the model-package trait, the manifest, and the probe
    models/               model packages, one crate each: mock, mantisnet
    hexo-bot/             the binary: init, train, and match
  xtask/                  the verification gates, defined once; cargo xtask
  python/
    hexo-py/              PyO3 engine/shared-encoder bindings; detached leaf crate
    mantisnet/            the MantisNet model, KLENT trainer, and control deck
  frontend/               the deck's React SPA, built once and served by the deck
  docker/                 the mantisnet-train image and WSL compose environment
  docs/                   the document set; docs/README.md is the index
  .github/workflows/      which runner executes which gate, and nothing more
```

Every crate and module directory carries a `README.md` on one template:
purpose, public surface, run/test, connections, invariants.

## Architecture

- **Rust owns the engine and the match loop; PyO3 is a leaf.** Only the
  `hexo-bot` executable embeds Python, crossing the interpreter once per
  evaluation batch. Logic crates are Python-free, and `hexo-engine` compiles
  to `wasm32`. `python/hexo-py` is the detached binding over the same Rust
  encoder core.
- **The engine owns canonical state; players search their own copy** with
  make/unmake. Only the runner advances the authoritative position, and
  exactly one authority exists per game.
- **A model is a package.** One crate under `crates/models/`, one
  `ModelPackage` trait covering encoder, evaluator, sessions, diagnostics,
  weights, and `fit`. A checkpoint is weights plus a manifest; loading one
  recomputes a probe hash over the evaluator's actual output and refuses a
  mismatch.
- **The evaluator seam never blocks.** Package encoders run worker-side,
  evaluators answer whole batches, and a seat's search is a state machine —
  a game in flight costs bytes, not a thread.
- **Records are one binary format with one implementation, read strictly.**
  `verify` replays a record through the engine. Python reads shards through
  this code rather than growing a second parser.
- **A remote seat replays a move prefix, then applies a move stream.**
  `Position` deliberately has no serde impl — board-shaped construction would
  bypass the turn rules. The position-only `zobrist()` is exchanged per ply to
  detect desync across processes.
- **Training is Python.** `python/mantisnet` holds the MantisNet model
  ([docs/MODEL_SPEC.md](docs/MODEL_SPEC.md)) and the KLENT training loop
  ([docs/KLENT_FOR_HEXO.md](docs/KLENT_FOR_HEXO.md)); the control deck
  ([docs/DECK_SPEC.md](docs/DECK_SPEC.md)) launches and watches runs inside
  the `docker/` environment. The container-side MantisNet package declines
  `fit`; production fitting is the Python loop.

## Build and verify

Rust 1.88+ (the floor is declared in `Cargo.toml` and gated).

```sh
cargo xtask verify     # every gate CI runs, in the order CI runs it
cargo xtask            # list the gates and what each one catches
```

The gates are defined in `xtask/src/main.rs` and nowhere else. A green
`cargo test` is not a green build: the release profile, rustdoc, the MSRV
floor, and the `wasm32` target each catch a class the test run cannot see.
Two gates need extra toolchains: `rustup target add wasm32-unknown-unknown`
and the MSRV toolchain named in `Cargo.toml`.

Python:

```sh
cd python/mantisnet
uv sync                # venv, hexo-py wheel via maturin, torch
uv run pytest          # the whole suite
```

Building the same tree from Windows and WSL collides on `target/`; set
`CARGO_TARGET_DIR=target-wsl` on the WSL side. Containers, training runs, and
the deck: [docker/README.md](docker/README.md).

## Docs

[docs/README.md](docs/README.md) indexes the set. Normative documents —
where code and document disagree, that is a finding to raise:

| Doc | Governs |
| --- | --- |
| [ENGINE_SPEC.md](docs/ENGINE_SPEC.md) | `crates/hexo-engine`: rules, state, storage, invariants, test obligations |
| [MODEL_SPEC.md](docs/MODEL_SPEC.md) | the MantisNet network: inputs, trunk, heads, obligations |
| [KLENT_FOR_HEXO.md](docs/KLENT_FOR_HEXO.md) | the KLENT training path, with its deviations from the paper |
| [CONTAINER_SPEC.md](docs/CONTAINER_SPEC.md) | the container crates: packaging, deployment, run lifecycle |
| [DECK_SPEC.md](docs/DECK_SPEC.md) | the control deck: API, lifecycle, SPA |

Reference documents: [KLENT_PAPER.md](docs/KLENT_PAPER.md) is the source
algorithm's paper, converted; [ABLATIONS.md](docs/ABLATIONS.md) is the
experimental record — every training run and engineering experiment, with
measurements and dispositions.
