# hexo-bot

## Purpose

`hexo-bot` is the native container executable and its reusable orchestration
library. It initializes package checkpoints, runs the epoch loop, drives
batched nonblocking decision sessions, writes records and metrics, and plays
fixed-checkpoint matches. The package registry contains `mock` and `mantisnet`.

## Public surface

The binary exposes three subcommands:

```text
hexo-bot init --package <name> --checkpoint <dir> [--package-config <string>]

hexo-bot train --run-dir <dir> --run-id <id> --package <name>
               --epochs <n> --games <n>
               [--package-config <string>] [--batch <n>] [--threads <n>]
               [--batch-wait-ms <n>] [--ply-cap <n>] [--eval-every <n>]
               [--eval-games <n>] [--resume]

hexo-bot match --games <n> --seat <spec> --seat <spec>
               [--batch <n>] [--threads <n>] [--batch-wait-ms <n>]
               [--ply-cap <n>] [--report <path>]
```

A match seat is a semicolon-separated value:

```text
package=<name>;checkpoint=<dir>[;config=<string>][;variant=<name>]
```

The library surface includes:

| Item | Contract |
| --- | --- |
| `parse`, `seat`, `USAGE` | Strict CLI and seat-spec parsers |
| `Command` | Parsed `Init`, `Train`, or `Match` request |
| `InitConfig`, `TrainConfig`, `MatchConfig`, `SeatSpec` | Typed command inputs |
| `registry::PackageRegistry`, `registry::PACKAGES` | Package lookup and construction |
| `init_checkpoint` | Atomic epoch-zero checkpoint placement |
| `train` | Long-lived epoch loop |
| `play_match`, `MatchRun`, `MatchReport`, `SeatReport` | Fixed-weight match |
| `Outcome` | Completion, signal stop, or failure exit classification |
| `BotError` | CLI, package, driver, record, and filesystem errors |

`PackageRegistry::new` receives the MantisNet forward loader.
`PackageRegistry::without_mantisnet_runtime` supports parsing and mock-only
operations without a live Python runtime.

The executable-only PyO3 layer loads `mantisnet.klent.run.load_model`, converts
the shared Rust `RawBatch` to Torch inputs, and returns flat policy and
per-legal-cell action-value outputs.

## Run / test

From the repository root:

```sh
cargo test -p hexo-bot
cargo doc -p hexo-bot --no-deps
```

Initialize and use the mock package:

```sh
cargo run -p hexo-bot -- init \
  --package mock \
  --package-config search=policy \
  --checkpoint tmp/mock-epoch-0

cargo run -p hexo-bot -- match \
  --games 2 \
  --seat "package=mock;checkpoint=tmp/mock-epoch-0;config=search=policy" \
  --seat "package=mock;checkpoint=tmp/mock-epoch-0;config=search=policy"
```

Run a bounded mock training job:

```sh
cargo run -p hexo-bot -- train \
  --run-dir tmp \
  --run-id mock-smoke \
  --package mock \
  --package-config search=policy \
  --epochs 1 \
  --games 2
```

Run all repository gates:

```sh
cargo xtask verify
```

When building the MantisNet runtime, `PYO3_PYTHON` selects the build
interpreter and `HEXO_PYTHON` selects the runtime environment.

## Connections

- `crates/hexo-engine` owns rules and canonical state.
- `crates/hexo-runner` supplies each game state machine.
- `crates/hexo-search` supplies sessions, leaf IDs, encoders, and evaluators.
- `crates/hexo-model` supplies the package lifecycle and manifests.
- `crates/hexo-records` supplies self-play shard writing.
- `crates/models/mock` and `crates/models/mantisnet` implement registered
  packages.
- `python/mantisnet` supplies the live Torch module used by the MantisNet
  forward loader.
- The process and artifact contract is
  [`docs/CONTAINER_SPEC.md`](../../docs/CONTAINER_SPEC.md).
- Container commands are documented in [`docker/README.md`](../../docker/README.md).

## Invariants & gotchas

- Only `init`, `train`, and `match` are accepted subcommands.
- Package configuration strings are passed to their package unchanged.
- `init` writes to an incomplete sibling and renames the complete checkpoint
  into place; an existing destination is refused.
- MantisNet initialization requires `source=PATH` in its package configuration.
- A run lives under `<run-dir>/runs/<run-id>`.
- Run IDs are one path component containing ASCII letters, digits, `.`, `_`,
  or `-`.
- `--games` is the number of concurrent self-play games and completed games per
  epoch; `--batch` is the maximum evaluator batch width.
- The driver encodes transient leaf positions on worker threads.
- One batcher owns the evaluator and pairs answers back to leaf IDs.
- Self-play writes record shards; evaluation rounds and `match` do not.
- A successful fit is followed by checkpoint placement and a proving load.
- MantisNet returns `Unsupported` for container-side `fit`; its training entry
  point is `python -m mantisnet.klent.run`.
- `--resume` validates the stored run manifest before continuing.
- Raising the configured epoch count on resume is allowed; lowering it is not.
- Signal stop is represented by exit code 2 after the current safe boundary.
- Exit code 0 means completion and exit code 1 means failure.
- A seat specification requires one package and one checkpoint and may contain
  one config and one variant.
- The first match competitor occupies P0 in even lanes and P1 in odd lanes.
- `--report` adds a JSON file; the match report is written to stdout in all
  cases.
