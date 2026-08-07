# hexo-model-mock

A deterministic, network-free implementation of the `ModelPackage` trait from
`hexo-model`. The package exercises the full container surface -- encoding,
evaluation, session creation, diagnostics, checkpoint lifecycle, record reading,
and fitting -- without Python, a GPU, or learned parameters. Its entire weight
state is a single `u64` salt.

## Public surface

The crate exports two items:

| Item | Kind |
| --- | --- |
| `MockPackage` | Concrete `ModelPackage` implementation |
| `ENCODER_VERSION` | The version of the encoding format the mock encoder writes |

`MockPackage` is constructed from a configuration string via
`MockPackage::from_config`. The configuration grammar is `search=<shape>`, where
a shape is either `policy` or `mcts:visits=N,inflight=N,cpuct=F`.

## Components

### config

Parses the package configuration string and the search-shape syntax. Defines the
`Search` enum with two variants -- `Policy` (one root evaluation per move) and
`Mcts` (PUCT with visit budget, in-flight cap, and exploration constant). The
same search-shape grammar is reused by variant session names, and parse failures
are classified as either unknown-shape or bad-parameters for distinct error
reporting paths.

### seam

Implements `Encoder` and `Evaluator` from `hexo-search`. The encoder
(`MockEncoder`) writes a twelve-byte item per position: the eight-byte Zobrist
hash followed by a four-byte legal count, both little-endian. The evaluator
(`MockEvaluator`) holds the loaded salt and derives normalized priors and a
bounded value from each encoded item using deterministic mixing. Priors are
strictly positive and sum to one; values lie strictly inside `(-1, 1)`.

### select

Provides four selector types -- `SelfPlaySearch`, `SelfPlayPolicy`,
`EvalSearch`, and `EvalPolicy` -- implementing `SelectFromSearch` and
`SelectFromPolicy` from `hexo-search`. Self-play selectors sample proportional
to visits or priors and emit diagnostics (a visit table or prior table encoded
with per-action keys). Evaluation selectors sample proportional to the cube of
visits or priors and emit no diagnostics. The module also provides
`session_seed`, which derives a deterministic per-session RNG seed from the
loaded salt and a monotonic serial.

### weights

Defines the checkpoint weight format: a single file `weights.mock` containing
one little-endian `u64` salt. Provides read/write functions for the weight file,
the fixed initial salt used by epoch-0 checkpoints, the SplitMix64 finalizer
(`mix`) used throughout the crate for salt derivation and evaluation, a unit-
interval conversion (`unit`), and `next_salt` which derives the next epoch's salt
from the prior salt, epoch number, and training-data digest.

### package

Implements `ModelPackage` on `MockPackage`. Tracks the configured search shape,
the loaded salt (absent until a checkpoint loads), and a session serial counter.
`init` writes an epoch-0 checkpoint with the fixed initial salt. `load` reads the
manifest and weight file, validates versions, and recomputes the probe hash
against the evaluator before publishing the salt. `self_play_session` and
`eval_session` build sessions from the configured search shape with self-play or
evaluation selectors respectively. `variant_session` parses the variant name as a
search shape and builds an evaluation-mode session. `fit` reads record shards,
replay-verifies every game, accumulates a digest from terminal hashes and ply
counts, and writes a new checkpoint whose salt is derived from the prior salt,
epoch, and digest.

## Connections

- `hexo-model` supplies the `ModelPackage` trait, `Manifest`, `PackageError`,
  and `probe_hash`.
- `hexo-search` supplies `Encoder`, `Evaluator`, `PolicySession`,
  `MctsSession`, `DecisionSession`, selector traits, and `SplitMix64`.
- `hexo-engine` supplies the position type, Zobrist hash, and legal-move
  enumeration.
- `hexo-records` supplies `ShardReader` for reading record files and `verify`
  for replay verification.
- `hexo-bot` registers this package under the name `mock` in its model registry.

## Files

| File | Description |
| --- | --- |
| `src/lib.rs` | Crate root; exports `MockPackage` and `ENCODER_VERSION`, declares internal version constants and the registry name |
| `src/config.rs` | Configuration and search-shape parsing with error classification |
| `src/seam.rs` | Twelve-byte encoder and deterministic salt-based evaluator |
| `src/select.rs` | Self-play and evaluation selectors with diagnostics encoding and session seeding |
| `src/weights.rs` | Salt format, read/write, mixing functions, and epoch derivation |
| `src/package.rs` | `ModelPackage` implementation: checkpoint lifecycle, sessions, and fitting |
