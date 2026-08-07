# hexo-model

The object-safe package boundary used by the container. `hexo-model` defines
the `ModelPackage` trait, checkpoint manifests, compatibility validation, and
the frozen probe that binds a checkpoint to evaluator behavior. It contains no
model architecture, feature layout, loss function, optimizer, or package
registry.

Concrete packages (`hexo-model-mock`, `hexo-model-mantisnet`) implement
`ModelPackage` in their own crates. The container consumes them as trait objects
through this crate's public surface.

## Components

### `ModelPackage` trait

The lifecycle and capability boundary a package exposes to the container:

- **Identity** -- `name`, `package_version`, `encoder_version`.
- **Checkpoint I/O** -- `init` writes an epoch-zero checkpoint; `load` validates
  a manifest, loads weights, and verifies their probe hash.
- **Encoding and evaluation** -- `encoder` returns a position encoder (available
  before weights are loaded); `evaluator` returns a batch evaluator for the
  currently loaded weights.
- **Sessions** -- `self_play_session` and `eval_session` are separate required
  methods; `variant_session` handles named evaluation variants (default rejects
  all names).
- **Fitting** -- `fit` consumes record shards and writes a new checkpoint.
  Packages without container-side fitting return `Unsupported`.

### `Manifest`

Checkpoint identity stored as `manifest.json` inside each checkpoint directory.
Fields:

- Package name, package version, encoder version.
- Rules version, action-order version, protocol version (filled from linked
  crate constants at construction time).
- Checkpoint epoch.
- Package-owned JSON metadata (opaque to this crate).
- Probe hash, serialized as a `0x`-prefixed, zero-padded, 16-digit lowercase
  hex string.

`Manifest::validate` checks every common version field against the running
build. Package metadata validation is left to the owning package.

### Probe

A frozen set of ten engine positions spanning plies 0 through 21. The positions
cover the opening, turn boundaries, mid-turn states, near-terminal states, and
a wide scattered frontier. `probe_positions` replays fixed move-list prefixes
to produce the positions; `probe_hash` encodes all ten into one batch, evaluates
in one call, and folds the exact little-endian prior and value bytes through
FNV-1a.

### `PackageError`

The error enum for all `ModelPackage` operations. Variants cover filesystem I/O,
manifest parsing, six version-mismatch checks (package name, package version,
encoder, rules, action-order, protocol), package metadata mismatch, probe-hash
mismatch, not-loaded state, unknown variant, invalid configuration, malformed
weights, empty training data, unsupported operations, and wrapped
package-internal failures.

## Connections

- `hexo-search` supplies the `Encoder`, `Evaluator`, and `DecisionSession`
  traits that `ModelPackage` methods return.
- `hexo-engine` supplies versioned probe positions, action ordering, and rules
  version constants embedded in manifests.
- `hexo-runner` supplies the protocol version embedded in manifests.
- `crates/hexo-bot/src/registry.rs` maps CLI package names to constructors.
- `crates/models/mock` implements the full boundary without a network.
- `crates/models/mantisnet` implements the network-backed boundary.

## Files

| File | Description |
| --- | --- |
| `src/lib.rs` | Crate root; declares modules and re-exports `ModelPackage`, `Manifest`, `MANIFEST_FILE`, `PackageError`, `probe_positions`, and `probe_hash`. |
| `src/package.rs` | The `ModelPackage` trait definition with all required and default methods. |
| `src/manifest.rs` | `Manifest` struct, its `new`/`with_package_metadata`/`write`/`read`/`validate` methods, the `MANIFEST_FILE` constant, and the `hex_u64` serde module for probe-hash serialization. |
| `src/probe.rs` | Frozen probe move lists (`PACKED`, `WIN_IN_TWO`, `WIN_IN_ONE`, `SCATTERED`), `probe_positions` construction, `probe_hash` computation via FNV-1a, and golden-vector tests. |
| `src/error.rs` | `PackageError` enum with Display and Error implementations and the `failed` convenience constructor. |
