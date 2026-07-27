# hexo-model-mock

The mock model package: a deterministic, weightless evaluator that exercises
every seam the container has, with no network, no GPU, and no Python.

**Status: implemented.** It is the first `ModelPackage`, and the one the
container is built against.

## What it is, and why it stays

**Not a placeholder to be deleted.** `docs/CONTAINER_SPEC.md` §5 makes the
argument and it is not re-argued here, only pointed at: this package drives the
encoder, the evaluator, both session kinds, both selection policies, the
diagnostics channel, shard writing, checkpoint write and load, the probe hash,
and `fit` — with no network, no GPU, and no Python. It is what makes the whole
loop testable in CI, and it is the package the container is built against first
precisely because it can be wrong in none of the ways a real model can. A
package the entire container is exercised against on every run earns its place
permanently.

Its games are not good. That is not what it is for: what it can say is that the
loop ran, that the bytes crossed every seam intact, that a checkpoint proved on
the way in, and that a fit consumed the games it claims to have.

## Shape

Pure Rust library crate depending on `hexo-engine`, `hexo-runner`,
`hexo-search`, `hexo-records`, and `hexo-model`. The public surface is one
constructor, one trait implementation, and one constant; everything else is
private, because all of it is what a package *owns* and none of it is what a
package *exposes*.

```
crates/models/mock/
  Cargo.toml
  README.md
  src/
    lib.rs        # crate root, MockPackage re-export, ENCODER_VERSION
    package.rs    # MockPackage: config in, checkpoints and sessions out
    config.rs     # the configuration grammar, parsed in one place
    seam.rs       # MockEncoder and MockEvaluator
    select.rs     # the four selectors, the diagnostics format, the session seed
    weights.rs    # the salt, its file, and the one mixing function
  tests/
    common/mod.rs # the pump/resume loop that drives whole games
    config.rs     # what the grammar accepts and what it refuses by name
    checkpoint.rs # init, load-is-proving, and what refuses before a load
    train.rs      # play, record, fit, and load what the fit wrote
```

## Module map

| Module | Role |
| --- | --- |
| `package` | `MockPackage` and its `ModelPackage` impl: `init`, `load`, `fit`, the two required session modes, and `variant_session`. |
| `config` | `search=policy` and `search=mcts:...`, parsed once and reached from two entry points. |
| `seam` | The encoder (zobrist + legal count) and the evaluator (priors and a value, from the salt). |
| `select` | Four selectors — self-play and eval, per session shape — the diagnostics encoding, and the seed a session is constructed with. |
| `weights` | `weights.mock`, the salt arithmetic, and the splitmix64 finalizer every other module draws from. |

## Configuration

One key, required, with no default:

```
search=policy
search=mcts:visits=64,inflight=8,cpuct=1.5
```

A search shape is a model choice, so there is nothing to fall back on: a missing
`search` key, an unknown key, an unknown shape, a missing or repeated `mcts`
parameter, a number that is not one, a zero budget, a zero in-flight cap, and a
`c_puct` that is not finite and non-negative are each refused by name.
Whitespace is not trimmed anywhere — one grammar is easier to state than one
grammar plus a lenience policy, so `search = policy` is refused rather than
guessed at.

The three `mcts` parameters are all required for the reason
`hexo_search::MctsConfig` has no `Default`: the budget is the compute a seat is
allowed, the cap is how much of a batch it may occupy, and `c_puct` trades
exploration against the value head.

**A session variant name is a search shape in the same grammar**, read by the
same parser. `"policy"` and `"mcts:visits=128,inflight=4,cpuct=1.0"` are both
valid variant names, so a match harness can pit two search shapes against each
other over one set of weights. A name that is no shape at all comes back as
`UnknownVariant` — the honest answer to "do you have a variant called `greedy`" —
while a name that *is* a shape with bad parameters comes back naming the
parameter, because that is the answer to the mistake actually made.

Variants select the way an **evaluating** seat does, not a self-play one: they
exist to compare search shapes and play benchmark matches, and a variant that
sampled and annotated like a self-play seat would be a third mode wearing the
name of a comparison.

## Weights

`weights.mock`, eight bytes, one little-endian `u64` **salt**. That is the whole
model.

The trick is that a salt is a complete stand-in for weights at this seam. Every
answer the evaluator gives is a pure function of the salt and the position's
zobrist, so:

- **the same salt answers the same position identically, forever** — which is
  what makes the probe hash a detector here rather than noise, since nothing is
  timing-dependent, device-dependent, or order-dependent;
- **a different salt answers everything differently** — which is what makes the
  probe hash *move* across epochs, so a `fit` that changed nothing is visible;
- **one flipped bit anywhere in the file is a different salt** — which is what
  makes tampering a `ProbeMismatch` on the next load rather than a training run
  that quietly never converges.

`init` writes a fixed documented constant, not entropy: two fresh runs of the
same build produce byte-identical epoch-0 checkpoints, and a probe hash that
moved between them would be reporting the initialisation rather than the
weights.

### The encoder and the evaluator

`ENCODER_VERSION` is 1, and the encoding is twelve bytes: the position's zobrist
as `u64` LE, then its legal count as `u32` LE. Enough to be a real encoding — the
zobrist distinguishes every position the evaluator will see, and the legal count
is what says how many priors to produce — and deliberately not a feature *plan*,
because an encoder with planes would be pretending to be a model.

The evaluator produces, for each item, priors that are strictly positive and
normalised to sum to one, and a value strictly inside `[-1, 1]`. Strictly: the
seam allows ±1, but ±1 is the value of a *decided* position and a network that
has not seen the game end has no business claiming it. Every prior weight lands
in `[1, 2)` before normalisation, so no prior can be zero and a sampling selector
is never handed a table it cannot draw from.

## Diagnostics

Written by the self-play selectors, read by the same crate, and stored verbatim
by `hexo-runner`. Both kinds share an eight-byte stride, so a reader walks one
loop.

| Offset | Field | Bytes | Notes |
| --- | --- | --- | --- |
| 0 | kind | u8 | `0` visit table, `1` prior table |
| 1 | entry count | u32 LE | one entry per root child, in canonical legal order |
| `5 + 8i` | action | u32 LE | the placement's `ActionId` — the record encoding |
| `9 + 8i` | value | 4 | visits as `u32` LE for kind `0`, the prior's `f32` bits for kind `1` |

The kind tag is one byte and it earns it: a shard header states the *mode* a game
was played in, not the search shape, so a reader that only had the mode could not
tell a visit table from a prior table without guessing.

**Eval seats record nothing, on purpose.** Annotations exist to be trained on and
an eval game trains nothing. Its shard is written to be read for results — which
checkpoint beat which, and how — and diagnostics on it would be bytes nobody
consumes occupying the largest field in the format. `hexo-records` keeps absent
and present-but-empty distinguishable, so "this seat answered with nothing" stays
a fact somebody can check.

## Selection

Four selectors, because the two modes are two contracts.

| Seat | Draws proportional to | Records |
| --- | --- | --- |
| self-play, tree search | visits | the visit table |
| self-play, policy | priors | the prior table |
| eval, tree search | visits³ | nothing |
| eval, policy | priors³ | nothing |

Self-play is proportional and unsharpened because the visit distribution *is* the
policy target, and a run that only played its own argmax would collect a target
it never explored around.

Eval is the **cube**, and it is neither argmax nor a fourth power. Sharp enough
that the move the search actually preferred wins the overwhelming majority of the
time, so a match measures the checkpoint rather than the sampler; soft enough that
a close second is played often enough for a thousand-game match to be a thousand
different games. `crates/hexo-player/README.md` argues the second half of that:
two deterministic seats replay one game.

## `fit`, and why it must read the shards

`fit` opens every path with `ShardReader`, refuses a shard whose header names
another package, runs `hexo_records::verify` on every game — which replays the
move list through the engine, the detector parsing cannot be — and counts games
and positions. Zero shards or zero games is `PackageError::NoTrainingData`, and
no checkpoint is written.

**A fit that consumed nothing and produced weights anyway is the silent failure
this whole design exists to catch.** It is indistinguishable downstream from a
fit that worked: the checkpoint is well-formed, the manifest validates, the probe
hash matches the weights that were written, and the loop runs for a hundred more
epochs.

So the next salt is a function of the old salt, the epoch, **and a digest folded
over every game actually read**. The count catches a fit that read nothing; the
digest catches one that read *some* — a fit handed two shards and reading one
writes different weights than a fit reading both, which is a thing a test can say
out loud, unlike "it opened a file".

**`fit` does not load what it wrote.** The container loads it, through the same
`load` as any other checkpoint, which is what puts the fit's own output behind
the probe.

## Seeds

Sessions are constructed with a seed derived from the loaded salt and a
per-package serial. `docs/CONTAINER_SPEC.md` §12 leaves seeding to the driver —
a session takes a seed at construction and exposes `reseed`, and nothing above
that seam exists yet — so a package may construct with any seed it likes. What it
may not do is hand two concurrent sessions the same stream, which is what the
serial is for: a driver that forgets to reseed gets sessions that differ from
each other, rather than a self-play run of one repeated game.

## Deliberately absent

| Omitted | Why |
| --- | --- |
| Any actual learning | The salt moves; nothing gets better. A model that learned would need a network, and the point of this one is that it needs nothing. |
| A driver | The pump/resume loop lives in `hexo-bot`. The one in `tests/common` is a test harness, and a package that shipped a driver would be the second implementation of it. |
| A public encoder, evaluator, or selector | All of them are what a package owns. Exposing them would invite something above the package to have an opinion about one. |
| Atomic checkpoint placement | Writing under a temporary name and renaming is a decision about a whole directory (`CONTAINER_SPEC.md` §9), and belongs to whoever is making one. |
| A diagnostics decoder | The tests carry one written from this README rather than derived from the encoder — a decoder built from the encoder would agree with it by construction and could never notice the format drifting from what is documented. |

## Connections

- Implements `hexo-model`'s `ModelPackage`, and is the package its probe and
  manifest are exercised through.
- Uses `hexo-search` for both session shapes and for the two traits a package
  implements at the evaluator seam, plus the `SplitMix64` its selectors sample
  from.
- Reads `hexo-records` shards in `fit`, through the format's one reader, and
  holds them to `verify`.
- `hexo-bot` registers it by name (`mock`) and runs the `train` loop against it.
