# hexo-bot

The native container binary and its reusable orchestration library. `hexo-bot`
initializes package checkpoints, runs the epoch loop (self-play, fit,
checkpoint, evaluation), drives batched nonblocking decision sessions, writes
records and metrics, plays fixed-checkpoint matches, and serves long-lived
native seats over a strict JSON-lines protocol. The package registry contains
`mock` and `mantisnet`.

## Subcommands

The binary exposes four subcommands:

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

hexo-bot serve --package <name> [--package-config <string>]
```

A match seat is a semicolon-separated value:

```text
package=<name>;checkpoint=<dir>[;config=<string>][;variant=<name>]
```

## Components

### CLI and configuration (`cli`)

Parses command-line arguments into typed configuration structs: `InitConfig`,
`TrainConfig`, `MatchConfig`, and `ServeConfig`. A `Command` enum wraps all
four. The `seat` parser reads `--seat` specs into `SeatSpec` values. Each
configuration carries a shared `AtomicBool` stop flag that the signal handler
sets.

### Package registry (`registry`)

Maps the `--package` name string to a constructed `ModelPackage`. The registry
holds two entries (`mock` and `mantisnet`) and receives a `ForwardLoader` for
MantisNet at construction. `PackageRegistry::without_mantisnet_runtime` provides
a mock-only registry for Python-free testing.

### Checkpoint initialization (`init`)

`init_checkpoint` asks a package to write epoch-zero contents into a temporary
directory, then atomically renames it to the destination. Uses the `ModelPackage`
returned by the registry.

### Training loop (`train`)

`train` runs the complete epoch loop. Each epoch performs self-play through the
shared sweep driver, fits the package, places and proves the resulting
checkpoint, optionally evaluates the new checkpoint against the anchor (epoch 0)
and the previous checkpoint, and appends one metrics line. The stop flag is
checked between epochs and inside sweeps.

### Match play (`matches`)

`play_match` loads two packages from their respective seat specs, constructs
sessions (either a named variant or the package's evaluation mode), and runs the
shared sweep driver with no record sink. The first `--seat` occupies P0 in
even-numbered lanes and P1 in odd lanes. Returns a `MatchReport` with per-seat
win counts broken out by colour.

### Native seat server (`serve`)

`serve` implements the JSON-lines native seat protocol over stdin/stdout. After
a `hello` handshake that validates protocol, rules, and action-order versions,
loads the checkpoint and constructs an evaluator, the seat holds position/session
slots. `open` allocates slots from an opening action line, `decide` applies
incremental moves to cloned mirrors and runs all named slots through one batched
evaluator call per pump round, `close` releases slots, and `bye` terminates
cleanly. Refusals carry a machine-readable cause code.

### Sweep driver (`driver`)

The shared batched game driver used by self-play, evaluation, and match. A
sweep takes a set of lanes (each holding one `Game` and two
`DecisionSession`s), a worker pool, a batcher thread, and an optional record
writer thread. Workers check out lanes, deliver evaluations, pump sessions, and
encode leaves. The batcher merges encoded leaves per evaluator slot, crosses the
evaluator, and scatters answers back. Finished games are optionally sent to the
record writer. A pool of encoding arenas avoids allocation in steady state.

### Run layout (`run`)

`RunLayout` computes the directory paths for a run's manifest, metrics file,
checkpoints, and record shards. `place_checkpoint` writes to a temporary
directory and renames atomically. `resume_point` cleans partial artifacts, finds
the highest loadable checkpoint, and loads it. `check_resume` validates a
resumed run's manifest against the current flags.

### Metrics (`metrics`)

`Tally` counts wins (by slot and colour), draws, and no-contests across a set
of finished games. `EpochMetrics` combines a tally with timing, batch
statistics, and evaluation results into one JSON line appended to
`metrics.jsonl`.

### Error types (`error`)

`BotError` is the unified error enum covering CLI usage, unknown packages,
package and record format errors, filesystem and transport I/O, seat protocol
refusals, run manifest problems, resume mismatches, and thread panics.

### PyO3 bridge (`mantisnet_python`)

The executable-only module that implements `ForwardLoader` via PyO3. Discovers
the runtime Python interpreter from `HEXO_PYTHON`, initializes the embedded
interpreter, loads `mantisnet.klent.run.load_model`, and converts the package's
`RawBatch` to CPU tensors. Returns flat policy-logit and action-value vectors.

### Binary entry point (`main`)

Parses the command line, constructs the `PackageRegistry` with the live PyO3
forward loader, installs the `ctrlc` signal handler, dispatches to the
appropriate library function, and maps `Outcome` to an exit code (0 completed,
2 stopped, 1 failed).

## Public surface

| Item | Contract |
| --- | --- |
| `parse`, `seat`, `USAGE` | Strict CLI and seat-spec parsers |
| `Command` | Parsed `Init`, `Train`, `Match`, or `Serve` request |
| `InitConfig`, `TrainConfig`, `MatchConfig`, `ServeConfig`, `SeatSpec` | Typed command inputs |
| `registry::PackageRegistry`, `registry::PACKAGES` | Package lookup and construction |
| `init_checkpoint` | Atomic epoch-zero checkpoint placement |
| `train` | Long-lived epoch loop |
| `play_match`, `MatchRun`, `MatchReport`, `SeatReport` | Fixed-weight match |
| `serve` | One strict JSON-lines native seat connection |
| `Outcome` | Completion, signal stop, or failure exit classification |
| `BotError` | CLI, package, driver, record, filesystem, transport, and protocol errors |

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

## Files

| File | Description |
| --- | --- |
| `src/lib.rs` | Crate root: module declarations, public re-exports, and the `Outcome` enum |
| `src/main.rs` | Binary entry point: CLI parse, signal handler, dispatch, exit code |
| `src/cli.rs` | Command-line parser and typed configuration structs for all four subcommands |
| `src/error.rs` | `BotError` enum covering every failure mode the crate produces |
| `src/registry.rs` | `PackageRegistry` mapping package names to constructed `ModelPackage` instances |
| `src/init.rs` | `init_checkpoint`: atomic epoch-zero checkpoint creation |
| `src/train.rs` | The epoch loop: self-play, fit, checkpoint placement, evaluation, metrics |
| `src/matches.rs` | `play_match`: two-seat head-to-head over fixed weights, producing a `MatchReport` |
| `src/serve.rs` | Native seat server: JSON-lines protocol, slot lifecycle, batched decision rounds |
| `src/driver.rs` | Shared batched sweep: lane/worker/batcher/writer topology for all game-playing modes |
| `src/run.rs` | `RunLayout` and run-directory operations: manifest, resume validation, checkpoint placement |
| `src/metrics.rs` | `Tally`, `EpochMetrics`, and per-epoch JSON-line writing to `metrics.jsonl` |
| `src/mantisnet_python.rs` | PyO3 bridge: Python discovery, interpreter setup, `RawBatch`-to-tensor conversion |
