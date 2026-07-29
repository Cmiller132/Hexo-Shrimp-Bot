# hexo-model-mantisnet

## Purpose

`hexo-model-mantisnet` is the Python-free Rust model package for MantisNet.
It owns the shared position encoder, package configuration, improved evaluator,
decision sessions, diagnostics, checkpoint semantics, and the abstract forward
boundary. A runtime supplies Torch execution by implementing `ForwardLoader`
and `Forward`.

## Public surface

The crate root exports:

| Item | Contract |
| --- | --- |
| `MantisPackage` | `ModelPackage` implementation |
| `WEIGHTS_FILE` | Sealed Torch checkpoint name, `weights.pt` |
| `ForwardLoader` | Loads a `Forward` from a checkpoint path |
| `Forward` | Answers one collated `RawBatch` |
| `RawOutputs` | Flat policy logits and action values |
| `BoxError` | Runtime loader and forward error type |
| `improvement` | `improve_policy`, `ImprovedPolicy`, and `ImprovementError` |
| `MODEL_REPR_VERSION` | Shared encoder layout and semantics version |
| `PACKAGE_VERSION` | Package behavior version |
| `PACKAGE_NAME` | Registry name, `mantisnet` |

The public `encoder` module provides:

- `Graph` and `RawBatch`;
- `encode_position` and `decode_batch`;
- `build`, `build_batch`, and `build_batch_prefixes`;
- `collate`;
- `WireError`;
- `NUM_PATTERNS`.

`MantisPackage::from_config` accepts:

```text
tau=F,lambda=F[,source=PATH]
```

Both coefficients are required, finite, and nonnegative, and their sum must be
positive. `source` is required by `init` and omitted for normal checkpoint
loads.

Session variant names are:

```text
policy
mcts:visits=N,inflight=N,cpuct=F
gumbel:sims=N,m=N
```

The forward result contains one policy logit and one scalar action value per
legal cell, ragged by `RawBatch::legal_offsets`. The evaluator applies the
configured KLENT improvement and returns canonical priors plus their expected
action value.

## Run / test

From the repository root:

```sh
cargo test -p hexo-model-mantisnet
cargo doc -p hexo-model-mantisnet --no-deps
cargo check -p hexo-model-mantisnet
```

Regenerate the improvement fixture from the Python implementation:

```sh
cd python/mantisnet
uv run python ../../crates/models/mantisnet/tests/fixtures/regenerate_improvement.py
```

Run all workspace gates:

```sh
cargo xtask verify
```

The crate is a library package; use `hexo-bot init`, `hexo-bot match`, or the
Python training entry point to operate on checkpoints.

## Connections

- The architecture and representation contract is
  [`docs/MODEL_SPEC.md`](../../../docs/MODEL_SPEC.md).
- The Hexo-specific KLENT contract is
  [`docs/KLENT_FOR_HEXO.md`](../../../docs/KLENT_FOR_HEXO.md).
- Measured variants and outcomes are indexed in
  [`docs/ABLATIONS.md`](../../../docs/ABLATIONS.md).
- `crates/hexo-model` supplies the package lifecycle and checkpoint manifest.
- `crates/hexo-search` supplies evaluator and session interfaces.
- `crates/hexo-bot/src/mantisnet_python.rs` implements the live Torch loader.
- `python/hexo-py` binds this crate's encoder for Python.
- `python/mantisnet` owns model weights and production training.

## Invariants & gotchas

- The Rust encoder is the production encoder and the Python builder is its
  independent parity oracle.
- Encoded items are versioned, little-endian, and fully validated before
  allocation and collation.
- Legal-cell rows are in engine canonical legal order.
- Policy and action-value outputs have identical ragged legal offsets.
- The action-value head produces one `tanh`-bounded scalar per legal cell.
- The action-value decoder has its own projection, embeddings, and MLP
  parameters.
- The evaluator returns improved priors and their expected action value.
- Runtime output length mismatches and non-finite values are errors.
- Runtime forward failures are not replaced with synthetic evaluations.
- Self-play uses `PolicySession` and records nine-byte diagnostics.
- Fixed evaluation uses `GumbelSession` with 32 simulations and 16 candidates.
- Named variants use evaluation selection and do not record self-play
  diagnostics.
- A sealed checkpoint contains `weights.pt` and `manifest.json`.
- Manifest package metadata contains the configured `tau` and `lambda`.
- Load validates metadata, versions, and the evaluator probe before publishing
  the candidate forward module.
- A failed load preserves the previously published module.
- Evaluator access to the live forward is serialized by its mutex.
- `fit` returns `PackageError::Unsupported`; production training runs through
  `python -m mantisnet.klent.run`.
- Encoder meaning or wire-layout changes require `MODEL_REPR_VERSION` review.
