# Suggestions

Design proposals that are **not** decided. Each carries a status and, where it
matters, an explicit statement of what would change relative to the previous
implementation in `Hexo-BotTrainer-hexgt`.

Decided items live in the root `README.md`. When a suggestion here is accepted,
move it there and delete it from this file — this doc should only ever contain
open questions.

| # | Suggestion | Status |
| --- | --- | --- |
| S1 | Dense action indexing | Deferred — explained below, awaiting decision |
| S2 | Symmetry operations in the engine | Deferred — probably not yet |
| S3 | The evaluator seam | Deferred — explained below |
| S4 | Differential test against the old engine | Later |
| S5 | Read-surface contract for model encoders | Open — needed once models start |
| S6 | Containerised model players | Open — needs planning |
| S7 | Tooling: CI, linting, formatting, type checking | Proposal below |
| S8 | Zobrist position hashing | Open — cheap now, awkward later |

---

## S1. Dense action indexing

**Status: deferred. This section exists to explain the proposal properly.**

### What the old implementation does

`legal.rs` encodes an action as a packed coordinate:

```rust
pub const LEGAL_RADIUS: i16 = 8;
const COORD_OFFSET: i32 = 1 << 15;   // 32768
const COORD_MASK:   i32 = 0xffff;

pub const fn pack_coord(coord: HexCoord) -> PackedCoord
pub const fn unpack_coord(action_id: PackedCoord) -> HexCoord
```

Each of `q` and `r` is an `i16`, biased by 32768 into an unsigned half, and the
two halves are packed into one 32-bit word. It is a clean, exactly invertible
encoding of *any* coordinate on the infinite board, and as an identifier it is
correct and permanent.

The problem is not correctness. It is that this identifier is also being asked
to serve as a **neural network output index**, and it cannot.

A policy head is a fixed-width vector: the network emits `N` numbers and the
`i`-th one is the prior for action `i`. `pack_coord` produces values spread
across a 2^32 space to represent maybe a few hundred reachable cells. No head
can be 4.3 billion wide. So somewhere between the engine and the network,
something must be compressing "the action IDs that exist right now" down to
"indices 0..N-1".

In the old repo that compression is real but implicit — it lives outside the
engine, in encoding and model code, and it is rebuilt per position. Three
consequences:

1. **It can drift.** Self-play, training, and serving each need the identical
   mapping. Nothing structurally guarantees they agree; only matching code does.
   This is the classic train/serve skew bug, and it fails silently — the model
   simply learns the wrong thing.
2. **It is dynamic.** If the index of a cell depends on the current legal-move
   set, then "index 7" means a different cell in different positions. A policy
   target recorded in one position is not comparable to one recorded in another,
   and the network cannot learn a stable spatial prior.
3. **It is invisible.** The mapping is the single most load-bearing constant in
   the whole system — every trained network is permanently married to it — and
   it is not written down in one place.

### What would change

Give the engine **two** action encodings with clearly separate jobs, instead of
one encoding doing both jobs badly.

**Keep `pack_coord` as-is, for records.** Unbounded, rules-faithful, exactly
invertible. This is action *identity*: what goes in a game record, what the
runner validates, what a replay reads. Nothing about it changes.

**Add a dense index, for model I/O.** Hexo always opens with the centre stone
at `(0, 0)`, so the origin is canonical and fixed forever — which means a
coordinate can be given a fixed index relative to it, decided once and never
recomputed:

- Pick a radius `R` covering the playable region. A hex of radius `R` holds
  `3R² + 3R + 1` cells: `R=15` → 721, `R=20` → 1261, `R=25` → 1951.
- Enumerate that hex in a fixed order. `coord_to_index` and `index_to_coord`
  become `const fn`s in `hexo-engine`, computed, not stored.
- Legal-move generation yields dense indices directly. The legal mask is an
  `N`-bit bitset. The policy head is exactly `N` wide. No remapping anywhere.
- A golden test hashes the entire table so the mapping can never silently
  change. If someone reorders the enumeration, the test fails loudly instead of
  every existing checkpoint quietly becoming wrong.

`R` is not `LEGAL_RADIUS`. `LEGAL_RADIUS = 8` is a *rule* — how far from an
occupied cell a placement may be. `R` is a *representation bound* — how far from
the origin the model can address. They are unrelated numbers that both happen to
be radii.

### The trade-off, stated honestly

This bounds what the model can address, while the game itself stays unbounded.
The rules do not change (per your instruction); the engine still plays on an
infinite board. But a position whose stones escape radius `R` cannot be encoded.

Three ways to handle that:

- **(a)** Make out-of-region placements illegal. Rejected — that changes the rules.
- **(b)** Engine stays infinite; the action encoding covers radius `R`; escaping
  `R` is an operational error the runner reports. Pick `R` with margin and
  instrument the maximum radius ever reached so the margin is measured, not
  assumed.
- **(c)** Recentre the index dynamically. Rejected — it reintroduces exactly the
  instability described above.

**(b) is the recommendation.** The honest caveat: I do not know how far real
games actually spread. With a frontier rule of 8, the worst case grows fast, but
the worst case is not the typical case, and the typical case is what sets `R`.
That number should come from measurement before `R` is fixed — it is a cheap
statistic to collect from the old repo's existing game records, and worth
collecting before this is decided.

---

## S2. Symmetry operations in the engine

**Status: deferred — your call was "maybe don't do this yet." Recorded for later.**

The board has a fixed centre at the origin, which gives the position a D6
symmetry group: six rotations times a reflection, twelve elements. That is
useful twice over — as training data augmentation, and as a weight-tying
constraint in the network trunk.

The suggestion is only about *where the operation lives*: coordinate transforms
and the corresponding action-index permutations in one place in the engine, with
golden tests, rather than one implementation for augmentation and another for
the model. Two implementations of the same permutation that disagree is a
genuinely nasty bug to find.

Nothing forces this now. It becomes relevant when the first model appears, and
it is coupled to S1 — the action-index permutation only exists if a dense index
exists.

---

## S3. The evaluator seam

**Status: deferred. You flagged this as unclear, so here it is properly.**

### The problem it solves

This only matters once there is a search (MCTS or similar). Search works by
walking down the tree to a leaf position and asking the network "how good is
this, and what moves look promising?" — a policy vector and a value. A serious
search does this tens of thousands of times per move.

If the search lives in Rust and the network lives in PyTorch, every one of those
leaf evaluations is a crossing from Rust into Python. Each crossing has to
acquire the GIL, which is a global lock: while one thread holds it, no other
thread can run Python. So a multithreaded Rust search would spend nearly all its
time queueing for a lock. That is the wall.

### What the seam is

Just an interface. Define a trait — one method, "here is a batch of positions,
give me back a policy and a value for each" — and have the search call only
that. The search never knows where the numbers come from.

```rust
trait Evaluator {
    fn evaluate(&self, batch: &[Position]) -> Vec<(Policy, Value)>;
}
```

Two things fall out of it:

**Batching.** Because the interface takes a batch, the search is forced to
collect many leaves before asking. One crossing per 256 leaves instead of 256
crossings. That alone turns the GIL from a wall into a rounding error, and it
is also what GPUs want anyway.

**Substitutability.** The same search runs against any implementation:

1. *Python callback* — Rust hands a batch over, PyTorch evaluates it, hands
   results back. Easy to build, and it lets you keep iterating on the model in
   Python where that is pleasant.
2. *In-process Rust inference* — ONNX Runtime, `tch`, or `candle` running the
   exported network inside the same process. No Python in the loop at all, so
   self-play saturates every core. This is where the throughput ends up.
3. *Mock* — returns fixed or uniform numbers. Makes the search deterministically
   unit-testable with no model and no GPU.

### Why it is mentioned now, with search out of scope

Only one thing needs to happen today, and it is a thing not to do: **keep every
model and tensor concept out of `hexo-engine`.** If the engine stays ignorant of
evaluation, all three implementations remain available later. If evaluation
concepts leak into the engine, option 2 quietly stops being reachable.

That is the same conclusion as your position on encoding — models own their own
encoding, the engine exposes state and nothing more.

---

## S4. Differential test against the old engine

**Status: later, per your call. Noting the expiry date.**

`Hexo-BotTrainer-hexgt` contains a working, battle-tested implementation of
exactly these rules. A property test that drives random legal move sequences
through both engines and asserts identical legality sets, terminal results, and
window states would turn "the rewrite is correct" from a belief into a checkable
property.

The only reason to raise it now: this option is available exactly as long as the
old engine still builds and the rules have not diverged. It gets weaker over
time, so it is worth doing before the new engine has been tuned much, not after.

---

## S5. Read-surface contract for model encoders

**Status: open. Follows from your decision that models own their own encoding.**

If models encode features themselves, the engine's obligation is a *stable,
cheap, read-only view* of a position — and nothing else. Sketch of what that
view needs to expose:

- Occupied cells with owner, iterable in a deterministic order.
- Per-player occupancy, ideally as bitboards so an encoder can copy planes out
  rather than walk cells.
- Current player and turn phase.
- Legal moves, as coordinates and (if S1 lands) dense indices, plus a mask.
- Terminal status.
- Window and threat masks — derived, but expensive to recompute, so worth
  exposing rather than making every encoder rediscover.
- Move history.

Expressed as a Rust trait in the engine, so a model crate depends on
`hexo-engine` and never on `hexo-runner`. Worth designing when the first model
lands, not before — but worth *not foreclosing* now, which mostly means keeping
the state representation clonable and readable without going through the runner.

---

## S6. Containerised model players

**Status: open, needs planning. Recording the constraint it puts on the runner.**

The requirement — models developed independently, on a shared framework, each
containerised — mostly lands on the player interface, and it argues for one
specific decision early:

**The runner should speak "initial position plus move stream", not "full state
per turn."** A container cannot be handed a Rust struct. If the protocol is a
move stream, then the same interface serves an in-process player, a subprocess,
and a container, with no second implementation. It also costs O(1) per ply
rather than O(board), and the move stream *is* the game record, so recording
comes free.

Consequences worth planning through when this is picked up: how a container gets
the rules (depend on `hexo-engine` and keep its own mirrored state — with a
position hash exchanged per ply to catch desync, which is the strongest argument
for S8); versioning between runner and player; and how model-side resource
limits interact with the adjudication policy.

---

## S7. Tooling: CI, linting, formatting, type checking

**Status: proposal, answering "what is the benefit?"**

Taking these one at a time, because they are not equally worth it and the
generic answer ("code quality") is not a real answer for a solo project.

**Formatter — `cargo fmt`, and `ruff format` later.** Worth it, cost ≈ zero.
The benefit is not pretty code, it is that diffs contain only semantic changes.
When you come back to a rewrite after two weeks, `git log -p` is the fastest way
to remember what you did — and it is useless if half of every diff is whitespace.
Rust already has one canonical style, so there is nothing to argue about.
**Recommend: on, from the first commit.**

**Linter — `clippy`.** The highest-value item on this list, and not for style
reasons. Clippy catches real defects in exactly the code this repo is about:
needless clones in hot loops, redundant allocation, iterator misuse,
`unwrap` on paths that can fail, integer-overflow patterns. For a
performance-sensitive engine where the whole point is avoiding copies, it is a
second pair of eyes on the thing you care most about. The workspace
`Cargo.toml` already sets `correctness = deny` and `perf = warn`.
**Recommend: on, from the first commit.**

**Type checking — mypy or pyright.** Value scales with how much Python exists,
and right now the answer is none. When the Python boundary lands it will be a
thin bridge plus glue, where pyright in `basic` mode is enough. Strict typing
over a few hundred lines of facade is not worth the friction.
**Recommend: skip for now. Revisit when Python is more than glue.**

**Python linting — `ruff`.** Same reasoning. Cheap to add with the Python
package; nothing to lint until then.
**Recommend: add with the first Python code, not before.**

**CI.** The one that sounds like team ceremony but is not. For a solo project on
this machine, the concrete benefits are:

- *It builds on a clean checkout.* The old repo cannot honestly claim this —
  packages are not installed, ~120 test files each do their own
  `sys.path.insert`, and a shell script hand-assembles `PYTHONPATH` from every
  `packages/*/python` directory. That is a build that works because of the
  machine it is on. CI is what makes that class of problem impossible to ignore,
  and it is far cheaper to never acquire it than to unpick it later.
- *Tests run on every commit, not when you remember.* During a long rewrite,
  this is the difference between finding a rules regression in the commit that
  caused it and finding it three weeks later.
- *It is the enforcement for everything above.* `fmt` and `clippy` configured
  but not run are decoration.

Cost is one YAML file and a few minutes of GitHub's compute per push.

**Concrete proposal:** one workflow, three steps — `cargo fmt --check`,
`cargo clippy -- -D warnings`, `cargo test`. Roughly twenty lines. Add the
Python steps when there is Python. Nothing else, no coverage gates, no release
automation, no badges.

---

## S8. Zobrist position hashing

**Status: open. Raised because it is cheap now and awkward later.**

An incrementally-maintained 64-bit hash of the position: XOR a random constant
per (cell, player) on placement, XOR the same one back on undo. A handful of
lines, essentially free at runtime, and it wants to live inside the same delta
machinery as `apply`/`undo` — which is why retrofitting it is more annoying than
it sounds.

What it buys, none of it needed today:

- **Desync detection** between the runner and a remote or containerised player —
  exchange the hash each ply and a divergence is caught immediately rather than
  as inexplicable behaviour later. This is the near-term one, and it is the
  reason S6 points here.
- **Transposition tables** in a future search.
- **Repetition and deduplication** across recorded positions.

Worth deciding now only because it is small and touches the undo path. Perfectly
reasonable to defer if the undo machinery is designed to accommodate it.
