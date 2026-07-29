# hexo-model

## Purpose

`hexo-model` defines the object-safe package boundary used by the container.
It also defines checkpoint manifests, compatibility validation, and the frozen
probe used to bind a checkpoint to evaluator behavior. The crate has no model
architecture, feature layout, loss, optimizer, or package registry.

## Public surface

The crate root re-exports:

| Item | Contract |
| --- | --- |
| `ModelPackage` | Lifecycle, encoder, evaluator, sessions, and fitting |
| `Manifest` | Checkpoint identity, versions, epoch, probe, and metadata |
| `MANIFEST_FILE` | Checkpoint manifest filename, `manifest.json` |
| `PackageError` | Typed package, manifest, version, probe, and I/O failures |
| `probe_positions` | Frozen ordered set of engine positions |
| `probe_hash` | Hash over exact evaluator answers on the probe set |

`ModelPackage` is object-safe and provides these methods:

- `name`, `package_version`, and `encoder_version`.
- `init(dir)` to write an epoch-zero checkpoint.
- `load(dir)` to validate and publish a checkpoint.
- `encoder()` for worker-side position encoding.
- `evaluator()` for batch evaluation under loaded weights.
- `self_play_session()` and `eval_session()`.
- `variant_session(name)` for package-defined evaluation variants.
- `fit(shards, out_dir, epoch)` for package-owned training.

`Manifest` records:

- package name and package version;
- encoder, rules, action-order, and protocol versions;
- checkpoint epoch;
- the fixed-width hexadecimal probe hash;
- package-owned JSON metadata.

Its public operations are `new`, `with_package_metadata`, `write`, `read`, and
`validate`.

An implementation is consumed through a trait object:

```rust
use hexo_model::ModelPackage;

fn encoder_for(package: &dyn ModelPackage) -> Box<dyn hexo_search::Encoder> {
    package.encoder()
}
```

The concrete packages are `hexo-model-mock` and
`hexo-model-mantisnet`; their constructors and configuration grammars remain
in their own crates.

## Run / test

From the repository root:

```sh
cargo test -p hexo-model
cargo test -p hexo-model --test manifest
cargo test -p hexo-model --test probe
cargo test -p hexo-model --test package
cargo doc -p hexo-model --no-deps
```

Run all workspace documentation and lint gates:

```sh
cargo xtask verify
```

Build the package API and its two implementations:

```sh
cargo check -p hexo-model
cargo check -p hexo-model-mock
cargo check -p hexo-model-mantisnet
```

## Connections

- The normative package and checkpoint contract is
  [`docs/CONTAINER_SPEC.md`](../../docs/CONTAINER_SPEC.md).
- `crates/hexo-bot/src/registry.rs` maps CLI package names to constructors.
- `crates/hexo-search` supplies `Encoder`, `Evaluator`, and
  `DecisionSession`.
- `crates/hexo-engine` supplies versioned probe positions and action order.
- `crates/hexo-runner` supplies the protocol version embedded in manifests.
- `crates/models/mock` implements the complete boundary without a network.
- `crates/models/mantisnet` implements the network-backed boundary.
- Checkpoint directories contain `manifest.json` plus package-owned weight
  files.

## Invariants & gotchas

- Every package name, package version, and encoder version is package-owned.
- `ModelPackage` methods use `&Path` so the trait remains object-safe.
- `self_play_session` and `eval_session` are separate required methods.
- The default `variant_session` refuses every name with `UnknownVariant`.
- An encoder is available independently of loaded weights.
- `evaluator` and session construction may refuse with `NotLoaded`.
- A successful `load` validates all manifest versions and the probe hash before
  publishing candidate weights.
- A failed `load` must leave the previously loaded state usable and unchanged.
- Consumers obtain a fresh evaluator after each successful load.
- The lifetime of evaluators created before a later load is package-defined.
- Probe hashing includes exact prior and value bytes in a frozen position order.
- Probe output uses the `hexo-search::Evaluation` conventions.
- `Manifest::write` writes one file; atomic placement of a whole checkpoint
  directory belongs to the caller.
- Package metadata is compared according to the concrete package contract.
- `fit` receives record-shard paths and the epoch of the checkpoint it must
  write.
- A package may return `Unsupported` when it has no container-side fitting
  implementation.
- The model registry does not live in this crate, so adding a package does not
  add a dependency here.
- Changes to probe positions or probe folding are checkpoint compatibility
  changes and require fixture review.
