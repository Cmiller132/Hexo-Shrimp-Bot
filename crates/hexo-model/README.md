# hexo-model

The model-package API: what every model crate provides to the container, and the
checkpoint manifest and probe that hold it honest.

**Status: implemented.** The trait, the manifest, and the probe ship. No model
ships — this crate is the shape of one, and it never learns what a feature, a
layer, or a loss is. `crates/models/mock` and `crates/models/mantisnet` are
the packages built to it.

## Shape

Pure Rust library crate, depending on `hexo-engine`, `hexo-runner`,
`hexo-search`, `serde`, and `serde_json` (the manifest's package-owned
metadata is a `serde_json::Value` in the public API). No threads, no I/O
beyond two files in a checkpoint directory, and no knowledge of any
particular model.

```
crates/hexo-model/
  Cargo.toml
  README.md
  src/
    lib.rs        # crate root, flat re-exports
    package.rs    # ModelPackage
    manifest.rs   # Manifest, MANIFEST_FILE, the hex probe-hash encoding
    probe.rs      # probe_positions, probe_hash, and the FNV-1a fold
    error.rs      # PackageError
  tests/
    manifest.rs   # the round trip, and every version mismatch by name
    probe.rs      # the frozen set, determinism, and what the hash notices
    package.rs    # object safety, and the one method a package inherits
```

## Module map

| Module | Role |
| --- | --- |
| `package` | `ModelPackage`: the whole of what the container knows a model by. Object-safe, because the registry holds `Box<dyn ModelPackage>`. |
| `manifest` | `Manifest`: which package wrote these weights, under which versions, at which epoch, and what they answer — plus `package_metadata`, a `serde_json::Value` the package owns. `new`, `with_package_metadata`, `write`, `read`, `validate`. |
| `probe` | The frozen probe set and the hash over the evaluator's exact output bytes — `docs/CONTAINER_SPEC.md` §10.2's detector. |
| `error` | `PackageError`, carrying the path, the pair of versions, or the package's own error that locates the problem. |

## Design notes

- **The two session modes are required methods with no defaults.** This is
  `crates/hexo-player/README.md`'s argument for `Model`'s two methods, one level
  up and unchanged: a single constructor taking a `Mode` can be written to
  ignore it, and that compiles, passes, and produces a self-play run in which
  every game is identical. No downstream stage can detect it, because the data is
  well-formed — well-shaped records of the same game, a loss curve that looks
  fine, and a bot that learned one line. The cost of the rule is that every
  package writes two constructors; the cost of not having it is a training run
  nobody can tell apart from a working one.

- **`variant_session` may default, because its default refuses.** The rule is
  not "no defaults" — it is that a default may never *answer*. A default that
  declines names the thing the operator asked for and stops, so nothing runs
  against a choice nobody made. Variants exist for search comparisons and
  benchmark matches, the vocabulary is the package's, and a package that defines
  none inherits an honest "no such variant".

- **Loading is proving.** `load` reads the manifest, checks every version against
  this build, loads the weights, recomputes the probe hash over what *actually
  answers*, and refuses on a mismatch. After it returns `Ok` the weights that
  answer are the weights the manifest promised — which is a different and much
  stronger claim than "a file was read". It is worth the forward pass because
  every failure it excludes is silent: none of them crash, and all of them train
  or play against the wrong weights indefinitely.

  A package that refuses mid-load must leave what it had loaded alone. Half a
  load is worse than none, because the process keeps running against weights it
  can no longer name.

- **A fresh evaluator after every load.** The container calls `evaluator()` again
  rather than reusing a handle. Whether an older handle survives a load is
  package-defined and nothing may rely on it either way — which is what lets one
  package back its evaluator with a live module whose parameter storage is
  written in place (`CONTAINER_SPEC.md` §10.1) and another with an owned
  snapshot, without the container knowing which it has.

- **The manifest does not describe the architecture.** How many layers there are
  and what shape they have lives inside the package's own weight file. The
  manifest answers which package wrote this, which version, which epoch, and
  whether it is compatible with this binary — plus a `package_metadata` value
  whose meaning is entirely the package's (semantic knobs like τ/λ live there,
  opaque to the container). That is what keeps "add a GNN
  package" a new crate and one registry entry instead of a schema change here —
  the moment the container can describe an architecture, it has an opinion about
  models, and every package afterwards has to fit the opinion.

  The three versions that are not the package's — `RULES_VERSION`,
  `ACTION_ORDER_VERSION`, `PROTOCOL_VERSION` — are filled in by `Manifest::new`
  from the crates this build links, not passed in. A package cannot write a
  manifest claiming rules it was not built against.

- **The probe hash is over the bytes, not over the weights.** Three properties
  make the number mean something. The whole set is **one batch forwarded once**,
  because batch shape decides which kernel runs and a probe split in two could
  produce two hashes from one set of weights. The hash is over the **evaluator's
  exact output bytes** — every prior's `to_le_bytes` and every value's — because
  hashing the weight file would miss every failure that leaves the file intact
  and answers with something else. And the positions are **fixed and derived from
  nothing**: hardcoded move lists replayed through the engine, no RNG, no
  dependence on the caller.

  What it catches, all of it silent: the wrong checkpoint loaded, a swap that
  constant-folding turned into a no-op, a mismatched encoder version, a scrambled
  action ordering, a runtime that drifted between build and run.

  The set is ten positions spanning plies 0, 1, 2, 5, 9, 10, 11, 12, 13, and 21,
  and the spread is deliberate: both movers, all three turn phases, both stones
  of a turn, two positions one turn from a decided game, and frontier widths from
  1 legal action to 748. A set of ten openings would agree with itself under an
  encoder bug that only appears once there are stones to encode.

  `probe_hash` also checks that the evaluator answered every position and that
  each answer's prior count is its position's `legal_count`, and **panics** if
  not. A probe that hashed a misaligned answer would still produce a stable
  number, so it would go on agreeing with itself while describing nothing — the
  detector deleting itself rather than firing.

- **The FNV-1a fold is written out here.** Five lines and two constants, rather
  than a dependency, because the constants are part of the checkpoint format: a
  hash function that changed under someone else's version bump would invalidate
  every manifest on disk without anything in this workspace having moved. The
  published FNV-1a vectors are pinned in a unit test, which also pins the
  xor-then-multiply order — swapping it produces FNV-1, a different function that
  every determinism test in this crate would still pass.

- **`&Path`, not `impl AsRef<Path>`.** Object safety is the constraint: the
  registry holds `Box<dyn ModelPackage>`, and a generic method would make the
  trait unusable as the one thing the container is allowed to know.

- **`PackageError` is designed so a package never flattens something that had
  structure.** `hexo-records` is deliberately not a dependency of this crate, so
  a package's `fit` has an error type this crate cannot name; `Failed` boxes it
  as a source, intact and downcastable, with a `doing` clause so the message
  reads as a sentence. `NoTrainingData` is a variant of its own rather than a
  message because "the fit consumed nothing" is the exact silent failure
  `CONTAINER_SPEC.md` §5 builds the mock to catch. `InvalidConfig` and
  `MalformedWeights` carry the package's own words, because config syntax and
  weight format are the package's and a shared enum that enumerated them would be
  the container having an opinion about a model. `PackageMetadata` locates a
  manifest whose package-owned metadata the package itself cannot read, and
  `Unsupported` is a package declining an operation it deliberately does not
  implement — how `mantisnet` answers `fit` while training lives in Python.

## Deliberately absent

| Omitted | Why |
| --- | --- |
| A registry | It is one map from name to constructor, and it belongs in `hexo-bot` with the flag that reads it. A registry here would make this crate depend on every package. |
| Anything about architecture | See above. The container stores a file and a manifest, not a description of layers. |
| A checkpoint-reference resolver | A reference may be a path, a `<run-id>/<epoch>` pair, or `latest` (`CONTAINER_SPEC.md` §10). Resolving one needs the run directory layout, which is `hexo-bot`'s — and nothing resolves all three forms yet, because `train` finds its own checkpoints and a `match` seat is handed a directory. |
| Atomic checkpoint placement | Writing to a temporary name and renaming is a decision about a whole directory, and belongs to whoever is making one. `Manifest::write` writes a file into a directory the caller owns. |
| A `Mode` enum | There is no method that takes one, on purpose. |
| A seed field on the manifest | `CONTAINER_SPEC.md` §12 and `OPEN_DECISIONS.md` B4: nothing mints a per-game seed, so a field for one would read as a guarantee nobody checked. |

## Connections

- Packages implement `ModelPackage`. `crates/models/mock` is the first, and is
  the one every CI run exercises.
- `hexo-bot` consumes it: a name registry, `--package` to pick one, and a `train`
  loop that calls `init`, `load`, the two session constructors, and `fit`. Its
  `match` subcommand is the caller of `variant_session`, which is where a
  refusing default is the honest answer.
- `hexo-search` supplies the seam types the trait hands back — `Encoder`,
  `Evaluator`, `DecisionSession` — and owns the two normative conventions on
  `Evaluation` that every package is held to.
- `hexo-engine` and `hexo-runner` supply the three version constants a manifest
  pins, and the position type the probe set is built from.
