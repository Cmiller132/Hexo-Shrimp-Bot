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
| S3 | The evaluator seam | **Closed.** Built as three types in [crates/hexo-search/README.md](../crates/hexo-search/README.md), which argues them; `CONTAINER_SPEC.md` §6 states what the container needs from them |
| S4 | ~~Differential test against the old engine~~ | **Retired.** It was built, the two engines agreed, and the crate that held it was deleted along with the old engine. Its job is now done by the independent oracles in `crates/hexo-engine/tests/common` and the frozen golden vectors, which do not depend on a second implementation existing |
| S5 | Read-surface contract for model encoders | Open, and now half of what it was — the ownership half is built; the bulk read surface still waits for a real encoder to shape it |
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

## S5. Read-surface contract for model encoders

**Status: open, and now half of what it was.** The ownership half is settled and
built: an encoder belongs to a model package, behind `hexo_search::Encoder`,
which runs worker-side and writes bytes into a caller-provided reusable batch —
`crates/hexo-search/README.md` and `crates/hexo-model/README.md` argue it, and
the engine still exposes state and nothing else. What is open is the *bulk
position-read surface* below that seam.

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

Move order is deliberately not on that list. It is not a property of a position,
and an encoder that wants it reads the game record — `hexo_runner::Game::plies`,
which is what a seat is handed — rather than asking the engine for a second copy.

Most of that list already exists as a *scalar* surface. What is open is the bulk
form: methods that fill a caller-provided buffer for a caller-named region, so an
encoder walking a few thousand cells per position does not pay per-cell address
arithmetic. `ENGINE_SPEC.md` §12 sanctions the shape in advance and sets the one
hard constraint — the caller names the region in **coordinates**, so no row,
word, plane, or stride escapes and the arena stays replaceable.

Deliberately still not built. The first encoder exists but does not dictate the
shape: `crates/models/mock` encodes twelve bytes — a zobrist and a legal count —
which the scalar surface already answers, so a bulk API built for it would be
built for nothing. The shape is dictated by a consumer that does not exist yet,
and this workspace does not keep two versions of anything, so guessing wrong is
expensive. The narrowest piece — reading a stone's owner out of the bit-scan
slot instead of mapping the coordinate back — shipped, because it changed no
API.

Inherent methods on `Position`, as `ENGINE_SPEC.md` §12 states them, rather than
a trait: `Position` as a trait is refused there for its own reasons, and an
additive `copy_planes_into` needs none. Worth designing when a package's encoder
actually walks planes, not before — but worth *not foreclosing* now, which
mostly means keeping the state representation clonable and readable without
going through anything above the engine.

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
