# hexo-model-mantisnet

The Rust-side model runner for MantisNet. This crate owns the position encoder,
the KLENT policy improvement, checkpoint packaging, and the decision-session
factory. It does not contain a neural-network runtime; a runtime adapter
implements the `ForwardLoader` and `Forward` traits to supply Torch (or another
backend) execution.

## Components

### Position encoder (`encoder`)

Converts an `engine::Position` into the graph representation MantisNet consumes.
Each position becomes a `Graph` of stones, windows, stone-to-window
incidence edges, a decoder table mapping legal cells back to windows, and a
background-distance table for legal cells that no kept window covers. Every
nonempty candidate window is kept under the ternary slot patterns
(MANTIS_GRAFT_SPEC Step 12, baked): the encoder assigns reversal-invariant
joint pattern/slot classes to both incidence and decoder edges in the
377/726/1458 ternary vocabulary, and the wire format speaks that scope under
`MODEL_REPR_VERSION` 4.

`build(position, action_rows)` additionally emits the Step 4 dense action-row
tables — each legal action's 18 hypothetical post-placement windows with
their 729 joint `(post, slot)` classes and pre-insert statuses — behind the
campaign knob. Collation derives the model's views from them: the kept rows
are asserted to be the decoder incidence in the same order (`act_class`), the
stable window-major edge permutation (`act_rev`), and per-cell EMPTY-row
counts by slot orbit (`act_empty`). The wire format does not carry them while
the knob is live; a bake extends it under a `MODEL_REPR_VERSION` bump.

The encoder has three output paths:

- **`build` / `build_batch` / `build_batch_prefixes`** produce in-memory
  `Graph` values (the last two in parallel via rayon) and collate them into a
  `RawBatch`.
- **`encode_position`** serialises a `Graph` into a versioned little-endian wire
  format for worker-to-batcher transport.
- **`decode_batch`** deserialises and validates a sequence of wire items, then
  collates them into a `RawBatch`.

`RawBatch` is the flat, globally-indexed tensor layout the forward boundary
accepts.

### Forward boundary (`forward`)

Defines the runtime-independent interface between this crate and whatever
executes the neural network:

- `Forward` -- runs both cell heads (policy logits and action values) on a
  `RawBatch` and returns `RawOutputs`.
- `ForwardLoader` -- constructs a `Forward` from a weight-file path.
- `BoxError` -- the error type both traits use.

No runtime-specific type crosses these traits.

### KLENT improvement (`improvement`)

Implements the closed-form KLENT policy improvement (equation 3 of the model
specification) in `f32`. `improve_policy` takes one position's raw policy logits
and action values together with `tau` and `lambda`, and returns `ImprovedPolicy`:
the improved action probabilities and their expected action value. All inputs are
validated for finiteness and range before computation.

### Package (`package`)

`MantisPackage` implements the `hexo-model::ModelPackage` trait, connecting
MantisNet to the generic container checkpoint lifecycle:

- `init` seals a raw Python `.pt` checkpoint into a container directory with a
  manifest.
- `load` reads and validates a sealed checkpoint, runs a probe verification, and
  publishes the loaded forward module.
- `encoder` / `evaluator` / `self_play_session` / `eval_session` /
  `variant_session` produce the objects the container runtime calls during play,
  training data generation, and evaluation.
- `fit` returns `Unsupported`; production training runs through the Python
  `mantisnet.klent.run` entry point.

Session construction seeds are derived from the loaded checkpoint's probe hash
and a monotonic serial, mixed through a SplitMix64 finaliser.

### Evaluator and encoder seam (`seam`)

`MantisEncoder` wraps `encoder::encode_position` behind the `hexo-search::Encoder`
trait. `MantisEvaluator` holds a mutex-guarded `Forward`, decodes an
`EncodedBatch` into a `RawBatch`, runs the forward pass, applies KLENT
improvement to each position's ragged row, and returns `Evaluation` values.

### Action selection (`select`)

`ActingPolicy` samples from the improved policy for `PolicySession` and
optionally records a nine-byte diagnostic payload (version, `v_hat`, entropy).
`MaxVisits` selects the most-visited root child for `MctsSession`.

### Configuration (`config`)

Parses the two string grammars the container passes into MantisNet:

- Package configuration: `tau=F,lambda=F[,source=PATH]`.
- Session variants: `policy`, `mcts:visits=N,inflight=N,cpuct=F`,
  `gumbel:sims=N,m=N[,temp=F]`.

## Connections

- **`hexo-engine`** supplies `Position`, `Action`, legal-move iteration, and
  window queries.
- **`hexo-model`** supplies `ModelPackage`, `Manifest`, and the checkpoint
  lifecycle.
- **`hexo-search`** supplies `Encoder`, `Evaluator`, `DecisionSession`,
  `PolicySession`, `MctsSession`, `GumbelSession`, and the selection traits.
- **`crates/hexo-bot/src/mantisnet_python.rs`** implements `ForwardLoader` and
  `Forward` using the live Torch runtime.
- **`python/hexo-py`** binds this crate's encoder for use in Python training and
  parity tests.
- **`python/mantisnet`** owns model weights, architecture definition, and
  production KLENT training.
- **`docs/KLENT_FOR_HEXO.md`** describes the Hexo-specific KLENT training path.

## Files

| File | Description |
| --- | --- |
| `lib.rs` | Crate root; re-exports the public surface and declares `MODEL_REPR_VERSION`, `PACKAGE_VERSION`, and `PACKAGE_NAME`. |
| `encoder.rs` | Position-to-graph builder, wire serialisation/deserialisation, batch collation, and the `RawBatch` layout. |
| `forward.rs` | The `Forward` and `ForwardLoader` traits plus the `RawOutputs` and `BoxError` types. |
| `improvement.rs` | Closed-form KLENT policy improvement (`improve_policy`, `ImprovedPolicy`, `ImprovementError`). |
| `package.rs` | `MantisPackage` (`ModelPackage` implementation), checkpoint init/load, session factory, and `WEIGHTS_FILE`. |
| `seam.rs` | `MantisEncoder` and `MantisEvaluator` adapters bridging the encoder and forward into `hexo-search` traits. |
| `select.rs` | `ActingPolicy` (policy sampling with optional diagnostics) and `MaxVisits` (MCTS selection). |
| `config.rs` | Strict parsers for the package configuration and session-variant grammars. |
