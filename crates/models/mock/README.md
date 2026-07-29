# hexo-model-mock

## Purpose

`hexo-model-mock` is a deterministic, network-free implementation of the
complete `ModelPackage` boundary. It exercises encoding, evaluation, both
session modes, diagnostics, checkpoint validation, record reading, and fitting
without Python or a GPU. Its weights consist of one package salt.

## Public surface

The crate root exports:

| Item | Contract |
| --- | --- |
| `MockPackage` | Concrete `ModelPackage` implementation |
| `ENCODER_VERSION` | Mock encoding version |

Construct a package with one required search configuration:

```rust
use hexo_model_mock::MockPackage;

let policy = MockPackage::from_config("search=policy")?;
let mcts = MockPackage::from_config(
    "search=mcts:visits=64,inflight=8,cpuct=1.5"
)?;
# let _ = (policy, mcts);
# Ok::<(), hexo_model::PackageError>(())
```

The accepted grammar is:

```text
search=policy
search=mcts:visits=N,inflight=N,cpuct=F
```

The same bare search shapes are accepted as evaluation variant names:

```text
policy
mcts:visits=N,inflight=N,cpuct=F
```

The private package components are:

| Module | Contract |
| --- | --- |
| `config` | Strict package and variant grammar |
| `seam` | Twelve-byte encoder and deterministic evaluator |
| `select` | Self-play/evaluation selectors and diagnostics |
| `weights` | Salt format and deterministic mixing |
| `package` | Lifecycle, sessions, and fit |

The encoder stores the position Zobrist hash and legal count. The evaluator
derives canonical priors and a bounded side-to-move value from the encoded item
and the loaded salt.

The checkpoint weight file is `weights.mock`, containing one little-endian
`u64`.

## Run / test

From the repository root:

```sh
cargo test -p hexo-model-mock
cargo doc -p hexo-model-mock --no-deps
cargo check -p hexo-model-mock
```

Exercise the package through the container binary:

```sh
cargo run -p hexo-bot -- init \
  --package mock \
  --package-config search=policy \
  --checkpoint tmp/mock-checkpoint
```

Run all workspace gates:

```sh
cargo xtask verify
```

## Connections

- `crates/hexo-model` supplies `ModelPackage`, `Manifest`, and probe hashing.
- `crates/hexo-search` supplies policy and MCTS sessions and evaluator seams.
- `crates/hexo-engine` supplies the position hash and legal count.
- `crates/hexo-records` supplies strict shard reading and replay verification.
- `crates/hexo-bot/src/registry.rs` registers the package as `mock`.
- The package obligations are in
  [`docs/CONTAINER_SPEC.md`](../../../docs/CONTAINER_SPEC.md).

## Invariants & gotchas

- Configuration has exactly one `search` key and no default.
- Whitespace is not trimmed and unknown or repeated fields are errors.
- MCTS visits and in-flight counts are nonzero.
- `cpuct` is finite and nonnegative.
- Sessions cannot be created until a checkpoint has loaded successfully.
- Each newly created session receives a distinct deterministic seed derived
  from the loaded salt and session serial.
- Self-play and evaluation use separate selectors.
- Self-play samples proportional to priors or visits and writes diagnostics.
- Evaluation samples using cubed priors or cubed visits and writes no
  diagnostics.
- Diagnostics encode either the canonical root prior table or visit table.
- Diagnostics action keys are engine `ActionId` values.
- `init` writes a fixed initial salt and an epoch-zero manifest.
- `load` validates manifest versions and recomputes the evaluator probe before
  publishing the salt.
- A failed load preserves the previous salt.
- `fit` requires at least one decoded game across the supplied shards.
- Every fitted game is replay-verified before it contributes to the digest.
- Every shard consumed by `fit` must name the `mock` package.
- `fit` derives the next salt from prior salt, epoch, game count, position
  count, and terminal hashes.
- The package provides no network, tensor dependency, or learned parameters.
