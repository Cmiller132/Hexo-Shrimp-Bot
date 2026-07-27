# Suggestions

*Optional* design proposals — things worth doing that nothing forces. Each
carries a status and, where it matters, an explicit statement of what it would
change in the crates that already exist.

Questions that **must** be answered to build at all live in
[OPEN_DECISIONS.md](OPEN_DECISIONS.md).

Decided items live with the thing they decided. When a suggestion here is
accepted, its *reasoning* moves to the relevant `README.md` and the body of the
item is deleted from this file — the table keeps a one-line row saying where the
answer went, exactly as `OPEN_DECISIONS.md` does, so that a cited number always
resolves to something. Numbering is stable: a closed item's number is never
reused, because other documents cite it.

| # | Suggestion | Status |
| --- | --- | --- |
| S1 | Dense action indexing | **Closed.** The diagnosis was right and the proposed fix was wrong; what survived is in [crates/hexo-engine/README.md](../crates/hexo-engine/README.md) |
| S2 | Symmetry operations in the engine | Deferred — probably not yet |
| S3 | The evaluator seam | Deferred — explained below |
| S4 | ~~Differential test against the old engine~~ | **Retired.** It was built, the two engines agreed, and the crate that held it was deleted along with the old engine. Its job is now done by the independent oracles in `crates/hexo-engine/tests/common` and the frozen golden vectors, which do not depend on a second implementation existing |
| S5 | Read-surface contract for model encoders | Open — needed once models start |
| S6 | Containerised bots | **Closed.** Accepted and specified in [CONTAINER_SPEC.md](CONTAINER_SPEC.md); the wire format it leaves open is `OPEN_DECISIONS.md` C1/C2 |
| S7 | Python-side tooling (ruff, type checking) | Deferred — no Python yet |

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

Nothing forces this now. It becomes relevant when the first model appears. The
half that used to block it no longer does: the canonical ordering shipped, so the
action-index permutation is expressible — it is the permutation `legal_rank`
undergoes when the board is transformed.

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

## S5. Read-surface contract for model encoders

**Status: open. Follows from your decision that models own their own encoding.**

If models encode features themselves, the engine's obligation is a *stable,
cheap, read-only view* of a position — and nothing else. Sketch of what that
view needs to expose:

- Occupied cells with owner, iterable in a deterministic order.
- Per-player occupancy, ideally as bitboards so an encoder can copy planes out
  rather than walk cells.
- Current player and turn phase.
- Legal moves, as coordinates and as ranks in the canonical ordering, plus a
  mask.
- Terminal status.
- Window and threat masks — derived, but expensive to recompute, so worth
  exposing rather than making every encoder rediscover.
- Move history.

Most of that list already exists as a *scalar* surface. What is open is the bulk
form: methods that fill a caller-provided buffer for a caller-named region, so an
encoder walking a few thousand cells per position does not pay per-cell address
arithmetic. `ENGINE_SPEC.md` §12 sanctions the shape in advance and sets the one
hard constraint — the caller names the region in **coordinates**, so no row,
word, plane, or stride escapes and the arena stays replaceable.

Deliberately not built yet. The shape is dictated by a consumer that does not
exist, and this workspace does not keep two versions of anything, so guessing
wrong is expensive. The narrowest piece — reading a stone's owner out of the
bit-scan slot instead of mapping the coordinate back — shipped, because it
changed no API.

Expressed as a Rust trait in the engine, so a model crate depends on
`hexo-engine` and never on `hexo-runner`. Worth designing when the first model
lands, not before — but worth *not foreclosing* now, which mostly means keeping
the state representation clonable and readable without going through the runner.

---

## S7. Python-side tooling

**Status: deferred. The Rust half is accepted and landed — see the root
`README.md` and `.github/workflows/ci.yml`.**

`cargo fmt`, `clippy` (`correctness = deny`, `perf = warn` at the workspace
root), and the gates in `cargo xtask` are in place. What remains is deferred only
because there is no Python yet:

- **`ruff`** for linting and formatting — cheap to add alongside the first
  Python code, pointless before it.
- **Type checking** — value scales with how much Python exists. When the
  boundary lands it will be a thin bridge plus glue, where pyright in `basic`
  mode is enough; strict typing over a few hundred lines of facade is friction
  without a payoff. Revisit if Python grows past glue.
