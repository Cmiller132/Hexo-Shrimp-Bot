# Suggestions

*Optional* design proposals — things worth doing that nothing forces. Each
carries a status and, where it matters, an explicit statement of what would
change relative to the previous implementation in `Hexo-BotTrainer-hexgt`.

Questions that **must** be answered to build at all live in
[OPEN_DECISIONS.md](OPEN_DECISIONS.md).

Decided items live with the thing they decided. When a suggestion here is
accepted, its reasoning moves to the relevant `README.md` and it is deleted from
this file — this doc should only ever contain open questions. Numbering is
stable: a closed item's number is not reused, because other documents cite it.

| # | Suggestion | Status |
| --- | --- | --- |
| S1 | Dense action indexing | **Closed.** The diagnosis was right and the proposed fix was wrong; what survived is in [crates/hexo-engine/README.md](../crates/hexo-engine/README.md) |
| S2 | Symmetry operations in the engine | Deferred — probably not yet |
| S3 | The evaluator seam | Deferred — explained below |
| S4 | Differential test against the old engine | **Closed.** Built, and it agrees; findings are in [crates/hexo-reference/README.md](../crates/hexo-reference/README.md) |
| S5 | Read-surface contract for model encoders | Open — needed once models start |
| S6 | Containerised bots | Open — scope set, protocol undecided |
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

## S6. Containerised bots

**Status: scope decided, protocol open.**

A container is a *complete bot*, not a move oracle. It carries `hexo-engine` and
`hexo-runner` inside it and must cover four jobs:

| Job | Who is the authority | What the container does |
| --- | --- | --- |
| Self-play | itself | drives whole games internally, emits records |
| Training | n/a | consumes records, emits checkpoints |
| Eval | a host orchestrator | plays as one seat |
| External tournament | the tournament harness | plays as one seat, in *their* protocol |

Three consequences fall out of that table.

**Exactly one authority per game.** Two containers each running a full runner
means two authorities, which is a desync waiting to happen. The container needs
an explicit mode — drive the game, or answer as a seat — and it must never
adjudicate when it is not the authority.

**Modes imply a binary.** The workspace is libraries only today. A container
needs an entry point with subcommands along the lines of `selfplay`, `serve`,
`train`. Worth deciding when that crate appears, and what it is called.

**External protocols get adapters, not accommodation.** Tournament harnesses
have their own wire formats. Design the native protocol for this system's needs
and translate at the edge; letting a foreign protocol's assumptions into the
runner is how the runner ends up serving two masters.

Still to decide: transport and wire format (a line-oriented stdio protocol is
the obvious default — trivial to containerise, trivial to debug by hand, and
close to what tournament harnesses already expect); the handshake, which should
pin protocol version, rules version, and action-encoding version before the
first move; and how model-side resource limits interact with adjudication.

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
