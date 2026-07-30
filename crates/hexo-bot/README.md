# hexo-bot

## Purpose

`hexo-bot` is the native container executable and its reusable orchestration
library. It initializes package checkpoints, runs the epoch loop, drives
batched nonblocking decision sessions, writes records and metrics, and plays
fixed-checkpoint matches. It also serves long-lived native seats that hold
position/session slots without holding or adjudicating games. The package
registry contains `mock` and `mantisnet`.

## Public surface

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

The library surface includes:

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

`PackageRegistry::new` receives the MantisNet forward loader.
`PackageRegistry::without_mantisnet_runtime` supports parsing and mock-only
operations without a live Python runtime.

The executable-only PyO3 layer loads `mantisnet.klent.run.load_model`, converts
the shared Rust `RawBatch` to Torch inputs, and returns flat policy and
per-legal-cell action-value outputs.

## Native seat protocol

`hexo-bot serve --package <name> [--package-config <string>]` selects package
code and passes the optional configuration string to that package unchanged.
The first wire message supplies the checkpoint directory and session variant.
`serve` owns only slots, each consisting of one `Position` mirror, its assigned
side, and one `DecisionSession`. It never constructs a `Game`, receives a
result, checks action legality, or adjudicates an outcome.

### Transport and scalar encodings

The transport is UTF-8 JSON Lines: exactly one JSON object followed by `\n` per
stdin request, and exactly one flushed JSON object followed by `\n` on stdout
for each complete request. Responses are emitted in request order. Stderr is
diagnostic only. Request objects are strict: missing, repeated, unknown, or
wrongly typed members make the message malformed. JSON object member order has
no meaning; the order of every `slots`, `opening`, `moves`, `decisions`, and
`diagnostics` array does.

The message forms below use these terminals:

| Terminal | Exact JSON encoding |
| --- | --- |
| `U32` | Integer from 0 through 4,294,967,295 |
| `U64` | Integer from 0 through 18,446,744,073,709,551,615; slot ids are numbers, not strings |
| `BYTE` | Integer from 0 through 255 |
| `ACTION` | `U32`, preserving `ActionId.0 = ((q as u16 ^ 0x8000) << 16) \| (r as u16 ^ 0x8000)` |
| `HASH` | String `0x` followed by exactly sixteen lowercase hexadecimal digits |
| `SIDE` | Exactly the lowercase string `"p0"` or `"p1"` |
| `STRING` | A JSON string |
| `VALUE` | Any JSON value, used only in a refusal's typed comparison |

The textual member order shown is the seat's response emission order. A reader
must still treat object member order as semantically irrelevant.

```text
hello =
  {"type":"hello","protocol_version":U32,"rules_version":U32,
   "action_order_version":U32,"checkpoint":STRING,"variant":STRING}

welcome =
  {"type":"welcome","name":STRING,"version":U32,
   "encoder_version":U32?,"resolved_variant":STRING,"digest":HASH,
   "restriction":STRING?}

open =
  {"type":"open","slots":[OPEN_SLOT,...]}
OPEN_SLOT =
  {"slot":U64,"side":SIDE,"opening":[ACTION,...]}

ok(open) =
  {"type":"ok","message":"open"}

decide =
  {"type":"decide","slots":[DECIDE_SLOT,...]}
DECIDE_SLOT =
  {"slot":U64,"moves":[ACTION,...],"zobrist":HASH}

decided =
  {"type":"decided","decisions":[WIRE_DECISION,...]}
WIRE_DECISION =
  {"slot":U64,"action":ACTION,"zobrist":HASH,
   "diagnostics":null|[BYTE,...]}

close =
  {"type":"close","slots":[U64,...]}

ok(close) =
  {"type":"ok","message":"close"}

bye =
  {"type":"bye"}

ok(bye) =
  {"type":"ok","message":"bye"}

refuse =
  {"type":"refuse","message":STRING[,"slot":U64],
   "cause":{"code":STRING,"detail":STRING
            [,"expected":VALUE,"got":VALUE]}}
```

`hello`, `open`, `decide`, `close`, and `bye` accept no fields beyond those
shown. `open`, `decide`, and `close` require at least one array entry.
`diagnostics` is always present in a decision: it is `null` when the session
authored no diagnostics and otherwise the package's opaque bytes as a JSON byte
array. No weights, priors, values, seeds, outcomes, or records cross this
protocol.

### Handshake

`hello` must be the first request and can occur exactly once. Its three version
numbers must equal this binary's `hexo_runner::PROTOCOL_VERSION`,
`hexo_engine::RULES_VERSION`, and
`hexo_engine::ACTION_ORDER_VERSION`; they are checked in that order and the
first disagreement is refused. `checkpoint` is a nonempty, platform-native
path string naming the checkpoint directory to load; a relative path is
resolved from the seat process's working directory. `variant` is a nonempty
package variant string.

The seat loads and proves the checkpoint, constructs the requested variant and
evaluator, and only then returns `welcome`. `name` and `version` identify the
loaded package and `encoder_version` its encoder; `digest` is the proved
manifest probe hash. `resolved_variant` is the exact accepted `hello.variant`
string, without normalization or substitution. An orchestrator checks the three
shared versions independently for each seat. It must not require two seats'
`welcome` values to agree on name, version, encoder version, resolved variant,
or digest.

`welcome` has one shape for every seat, because an orchestrator reads native and
foreign seats through the same parser. A seat that is not a `hexo-model` package
— an independent engine reaching this protocol through an adapter of its own —
omits `encoder_version`, which it does not have, and puts a content digest of
its own weights in `digest`. `restriction`, absent here, is how such a seat
declares that it will not propose some legal actions; `hexo-bot` proposes every
action in the canonical order, so it never sends one.

### Slot lifecycle and batching

`open` allocates each new slot from its complete `opening` action line and
stores its `side`. Opening actions are replayed in array order from a new
position. The line must be valid and nonterminal, and a slot id cannot already
be open or occur twice in one request. Prospective slots are staged and become
visible only after every entry succeeds, so a failed multi-slot `open` never
installs only a prefix. A successfully closed or retired numeric id can later
start a fresh lifecycle.

`decide.slots` is one batch and `decided.decisions` answers all of it in the
same order. For each entry, `moves` contains only the accepted placements since
that slot's preceding `open` or successful `decide`, in placement order.
Applying them must produce the entry's `zobrist`, a live position, and the
stored side to move. The returned action is deliberately not applied by the
seat: if the orchestrator accepts it, the next `moves` delta includes that
action along with any later accepted placements.

Every entry is first applied to a cloned mirror. No session starts until all
deltas, hashes, terminal checks, and side checks pass. All sessions then begin
together. In each pump round, leaves from every still-active slot form one
`EncodedBatch` and cross through exactly one `Evaluator::evaluate` call before
answers are scattered back by slot and `LeafId`. A search may need several such
rounds, but never one evaluator call per slot.

Each session authors its complete `Decision`. `serve` preserves its raw
`ActionId`, fixed-width zobrist attestation, and optional diagnostics bytes. It
checks the attestation against the searched mirror but does not check whether
the action is legal and does not apply it. The host submits the decision
verbatim to its authoritative `Game`, which adjudicates an illegal action.

A successful batch commits every cloned mirror only after all sessions have
decided and all attestations match. A failed batch returns one `refuse`, no
partial `decided`, and commits none of the other entries' mirror deltas. The
offending slot lifecycle is terminal; other slots remain usable.

`close` requires distinct, currently open ids. It prevalidates the whole list
and then releases all named mirrors and sessions together. Success is
acknowledged by `ok(close)`.

### Refusals and connection termination

Every refusal names the request in `message` and carries a stable `cause.code`
plus human-readable `cause.detail`. A slot-local refusal includes `slot` and
leaves the connection available for other slots. A connection refusal omits
`slot`, is flushed once, and terminates the process with exit code 1. The
orchestrator must not retry a refused message.

`expected` and `got` are present together only when the refusal compares
values. Version disagreements encode both as `U32`. `zobrist_mismatch` encodes
the orchestrator's claimed hash as `expected` and the mirror's computed hash as
`got`. `attestation_mismatch` encodes the mirror hash as `expected` and the
session's attestation as `got`. Hash comparisons use `HASH` strings.

| Scope | `cause.code` values |
| --- | --- |
| Connection framing or sequencing | `malformed_line`, `malformed_message`, `handshake_required`, `unexpected_message` |
| Handshake | `protocol_version`, `rules_version`, `action_order_version`, `checkpoint`, `variant`, `evaluator` |
| Request shape with no attributable slot | `empty_slots` |
| Slot lifecycle | `duplicate_slot`, `slot_already_open`, `unknown_slot` |
| Mirror input | `opening_line`, `incremental_line`, `zobrist_mismatch`, `terminal_position`, `wrong_side` |
| Session construction or authorship | `variant`, `attestation_mismatch` |

For an existing slot, duplicate use, reopening while open, an invalid
incremental line, a zobrist mismatch, a terminal mirror, a wrong side to move,
or an attestation mismatch retires that slot rather than resynchronizing it.
Distinct `close` ids are all validated before any is removed; a duplicate-id
refusal retires the duplicated id.

`bye` is valid only after `welcome`; the seat flushes `ok(bye)` and exits 0.
Clean stdin EOF exits 0 without an additional response (or 2 when it observes a
signal stop). EOF after a partial, non-newline-terminated request writes
`malformed_line` with `"message":"line"` and exits 1. Any newline-terminated
line that cannot decode as one of the strict request objects writes
`malformed_message` and exits 1; its `message` is the line's string `type` when
recoverable, otherwise `"line"`.
`play` remains deliberately absent because
[`docs/CONTAINER_SPEC.md`](../../docs/CONTAINER_SPEC.md) §15 does not name the
foreign harness protocol its adapter would need to implement.

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

Serve the same mock checkpoint:

```sh
cargo run -p hexo-bot -- serve \
  --package mock \
  --package-config search=policy
```

The orchestrator then sends `hello` with
`"checkpoint":"tmp/mock-epoch-0"` and `"variant":"policy"` on stdin.

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

- Package configuration strings are passed to their package unchanged.
- `init` writes to an incomplete sibling and renames the complete checkpoint
  into place; an existing destination is refused.
- MantisNet initialization requires `source=PATH` in its package configuration.
- A run lives under `<run-dir>/runs/<run-id>`.
- Run IDs are one path component containing ASCII letters, digits, `.`, `_`,
  or `-`.
- `--games` is the number of concurrent self-play games and completed games per
  epoch; `--batch` is the maximum evaluator batch width.
- The self-play/match driver encodes transient leaf positions on worker
  threads.
- Its one batcher owns the evaluator and pairs answers back to leaf IDs;
  `serve` owns one evaluator and calls it once per aggregate pump round.
- Self-play writes record shards; evaluation rounds and `match` do not.
- `serve` writes no records and is not part of the self-play path.
- A successful fit is followed by checkpoint placement and a proving load.
- MantisNet returns `Unsupported` for container-side `fit`; its training entry
  point is `python -m mantisnet.klent.run`.
- `--resume` validates the stored run manifest before continuing.
- Raising the configured epoch count on resume is allowed; lowering it is not.
- Signal stop is represented by exit code 2 after the current safe boundary.
- Exit code 0 means completion and exit code 1 means failure.
- Only `init`, `train`, `match`, and `serve` are accepted subcommands. `play`
  remains forbidden until its foreign harness protocol is specified.
- A serving seat holds `Position`/`DecisionSession` slots and never a `Game`.
- One `decide` request is one multi-slot batch, and one response answers every
  slot in request order.
- A seat never rules on legality; the host's runner adjudicates the returned
  `ActionId`.
- A seat specification requires one package and one checkpoint and may contain
  one config and one variant.
- The first match competitor occupies P0 in even lanes and P1 in odd lanes.
- `--report` adds a JSON file; the match report is written to stdout in all
  cases.
