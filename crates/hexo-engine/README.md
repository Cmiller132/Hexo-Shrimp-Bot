# hexo-engine

Authoritative Hexo rules and game state. Owns what the game *is*; owns nothing
about how it is played, recorded, or learned from.

**Status: MVP implemented.** Built to `docs/ENGINE_SPEC.md`, which is the
normative contract for this crate. Zero runtime dependencies; `proptest` and
`criterion` are dev-dependencies only.

## Shape

Pure Rust library crate. No PyO3, no I/O, no threads, no async, no tensors.
Those constraints are the point: they keep this crate `wasm32`-compilable (so a
web frontend can run real rules) and testable with `cargo test` alone.

```
crates/hexo-engine/
  Cargo.toml
  README.md
  src/
    lib.rs              # crate root, flat re-exports, RULES_VERSION, MAX_GRID_CELLS
    coord.rs
    player.rs
    action.rs
    window.rs
    grid.rs             # private
    position.rs
    position_tests.rs   # #[path]-included unit tests for the rule machine
    search.rs
    zobrist.rs          # private
  tests/
    common/mod.rs       # brute-force oracles + a dependency-free PRNG
    boundary.rs         # the coordinate domain at all six faces
    fixtures.rs         # hand-built win shapes
    golden.rs           # frozen Zobrist, move-ordering, and action-index vectors
    properties.rs       # proptest properties
    smoke.rs            # random playouts
  benches/
    common/mod.rs       # deterministic fixtures, over the shared PRNG
    engine.rs           # the criterion suite
  testkit/
    rng.rs              # the one splitmix64, `#[path]`-included by both of the above
```

## Module map

| Module | Public? | Role |
| --- | --- | --- |
| `coord` | yes | Axial `HexCoord` (`q`, `r`, derived `s`), the three line `Axis` values, `hex_distance`, the radius-8 disk table, and the coordinate domain bound. |
| `player` | yes | `Player` and `TurnPhase` — who moves and where they are inside the two-placement turn. |
| `action` | yes | `Action` (the move atom), `ActionId` (the unbounded, exactly invertible record encoding), and `ACTION_ORDER_VERSION`. |
| `window` | yes | `Window` (pure six-cell line geometry, including the `cell_index` / `contains` / `intersects` / `touches` relations), `WindowMask`, `WindowRef`, `WinningWindows` — the exposed win-detection surface. |
| `grid` | **no** | The dense recentred arena: two occupancy bit planes, one frontier bit plane, one coverage byte plane, the row-run disk update, the frontier rank/select scan, and the growth policy. **Zero items escape the crate.** |
| `position` | yes | `Position`, `Applied`, `Outcome`, `Stones`, `LegalActions`, the rule machine, replay, every read accessor, and `audit`. |
| `search` | yes | `Search<'p>` — the borrow-scoped make/unmake session — and the crate-private `Undo` token. |
| `zobrist` | **no** | The `const` mixing function and the twelve turn keys. Reachable only through `Position::zobrist()`. |
| `error` | yes | `MoveError`, `ReplayError`, `IntegrityError`, `IntegrityCheck`. |

`lib.rs` re-exports the public set flat, so consumers write
`use hexo_engine::{Position, Search, Action, Player};`.

There is deliberately no `board.rs`, `rules.rs`, `legal.rs`, `windows.rs`, or
`snapshot.rs`. A public `Board` type is what re-opened the rule-bypassing
construction path in the previous engine; a free `is_legal_placement` invites
non-atomic check-then-place; and neither the legal set nor the window masks are
stored, so neither gets a module.

## Design notes

- **`undo` is a first-class operation, not an afterthought.** Every field is
  restored by exactly one of three named mechanisms: re-running a self-inverse
  operation (occupancy, coverage, frontier, hash, counters), copying a snapshot
  out of the delta (phase, mover), or deliberately not at all (arena geometry,
  which is private and therefore unobservable). Nothing is restored by
  re-derivation.
- **The frontier bit plane *is* the legal set.** Membership is one bit, the
  count is a maintained `u32`, and enumeration is a bit scan that produces the
  canonical `(q, r)` order for free. `cover` is a byte count rather than a bit
  because an OR of radius-8 disks cannot be undone.

- **The disk is written as 17 row runs, not 217 coordinates.** A radius-8 disk is
  `dq`-major, so each of its rows is one contiguous run: contiguous bytes of
  `cover`, one or two words of every bit plane. `Grid::disk_runs` produces them
  and one placement maps 17 row bases, where the per-cell form mapped each of the
  217 cells three or four times. It lives in `grid` because it is a claim about
  the layout, and it is the only writer of coverage.

  The `DISK8` offset table is still there, and is worth more now than when it was
  load-bearing: it is the *independent* statement of the same cell set, walked
  offset by offset by the tier-C frontier assertion on every apply and undo, and
  compared against the runs directly in `grid`'s tests. A wrong run and a wrong
  offset are both symmetric, so neither can check itself.

  Worth 45–60% of an edge `apply`+`undo` pair and 45% of a full `replay` —
  `docs/ENGINE_RL_AUDIT.md` has the table.
- **Window masks are derived on read**, by an O(1) bit gather over an 11x11
  strip. Nothing about windows is stored, so there is no growth path, no delta,
  and no "stored mask disagrees with the board" bug class.

  Window *identity* is separate from that and reads nothing: `Window` is
  `(start, axis)`, and `cell`, `cells`, `cell_index`, `contains`, `intersects`,
  and `touches` are arithmetic on six coordinates with no `Position` in hand. A
  consumer building a cell/window incidence graph therefore pays the engine
  nothing for structure — only for the masks of windows that actually hold a
  stone, which it can enumerate exactly once each with no hash set by keeping a
  window only from the stone at the lowest set bit of `mask.occupied()`.
- **One placement is the atom, not one turn.** A turn is two placements, but a
  win is checked after each, so a turn can end after the first — and when it
  does, the phase and the mover *freeze*.
- **The position carries its own move history, and there is exactly one hash.**
  `history()` is what makes a position writable as a game record and rebuildable
  by `replay()`, and it is the only thing on the type not derivable from the
  board. `zobrist()` stays *position-only*: Hexo transposes structurally, since a
  turn's two stones are playable in either order and reach the same position, so
  a history-sensitive key would forfeit a 2x merge per turn of search. A model
  whose features depend on move order reads `history()` and mixes it into its own
  cache key rather than asking the engine for a second hash.
- **`PartialEq` means *same position*, not *same game*.** It ignores arena
  geometry and history alike, matching `zobrist()`, `audit()`, and the oracles.
  Compare `history()` explicitly when the question is whether two positions came
  from the same game.
- **Clone copies five buffers**, four grid planes plus the history — it was never
  a single `memcpy`, and the shorthand only ever meant "no pointer chasing, no
  per-cell work". History is four bytes per ply against an arena that is tens of
  kilobytes.
- **One canonical action ordering, owned here, in both directions.** `legal_rank`
  and `nth_legal` exist so self-play, training, and serving cannot each derive a
  private copy of the mapping a policy head is indexed by. A divergence there is
  silent: the network keeps training, against scrambled targets. Both directions
  are needed, not one — training records "the move played was index *k*", serving
  asks "the argmax is index *k*, which move is that?" — and shipping only the
  forward map would leave every model to write the inverse, which is the same
  drift in a different place.

- **The action region is unbounded, and that is load-bearing.** The obvious
  alternative is to index a fixed hex disk around the origin, since the opening is
  always at the centre. That has already been run in production and failed: a
  radius-20 crop excluded out-of-crop legal moves from policy and search, froze
  out-of-rim wins, and caused the previous repo's `main_3` training collapse. A
  larger radius narrows the failure without removing it, because the crop is still
  there. So the policy head is sized by the legal set instead, and the two jobs get
  two encodings — `ActionId` is action *identity*, unbounded and exactly
  invertible, for records and validation; this ordering is the *index*, for model
  I/O. The previous `pack_coord` was doing both, and could not do the second.

  The bijection did not hold at first: `legal_actions` offered 136 coordinates
  that `advance` refused, so `legal_rank` was assigning policy indices to
  unplayable moves. Fixed at the source — `place` no longer writes coverage
  outside the coordinate domain — and pinned by `tests/boundary.rs`. A dense index
  over a region that does not match the legal set is exactly the failure this
  design exists to prevent, so it is worth recording that the first version of the
  fix had it too.
- **Symmetric bugs are the real hazard.** A wrong disk offset, a wrong shear in
  the QR fold, a wrong hash constant, or a growth copy with the same wrong index
  on both sides all apply and un-apply identically, so round-trip tests are blind
  to them. `Position::audit()`, the independent oracles in `tests/common`, and
  the frozen golden vectors are the only detectors — they are not redundant.

## Testing

`cargo xtask verify` runs every gate, and `cargo xtask` says what each one
catches; they are defined in `xtask/src/main.rs`. The whole set takes well under
a minute, of which `cargo xtask test` is about 20 seconds.

Four of them exist for this crate specifically — the release lint, the rustdoc
gate, and the two `wasm32` gates each see a class the debug test run cannot.

The deep smoke run is `cargo xtask smoke`: an order of magnitude more games,
release profile, scheduled nightly rather than per-push. Worth running by hand
after a change to hashing, ordering, growth, or win detection.

The suite is also driven by the environment for a one-off:

```
HEXO_SMOKE_GAMES=200000 cargo test --release -p hexo-engine --test smoke
```

## Benchmarks

```
cargo bench -p hexo-engine                   # ~4 min, all groups
cargo bench -p hexo-engine -- ordering       # one group
cargo bench -p hexo-engine -- 'apply_undo/edge'
```

`criterion` is a dev-dependency of this crate only, with `default-features =
off` so it pulls in neither `rayon` nor `plotters`. Nothing about the runtime
dependency set changes, and the `wasm32` gate is unaffected.

The suite exists so that every optimisation `docs/ENGINE_SPEC.md` §12 and
`docs/ENGINE_RL_AUDIT.md` defer "if profiling ever shows it" is answerable with
a number rather than an argument. The groups are `advance`, `apply_undo`,
`clone`, `enumerate`, `ordering`, `windows`, `replay`, and `new_game`, each
reported at plies 1, 32, 96, and 256.

Three fixture choices carry the weight:

- **Uniform random play**, from one seed, through the normal rule machine.
  Fixtures nest — the ply-256 move list starts with the ply-32 one — and a game
  that ends early is a panic, not a shorter fixture.
- **Interior versus edge placements**, the legal cells nearest to and furthest
  from the centroid of the stones. The interior cell's radius-8 disk is already
  fully covered and flips no frontier bits; the edge cell's flips 136. That is
  the split `apply_undo` is measured over, and it is where the disk walk's cost
  actually lives.
- **An inflated arena**: ply 96 after a search excursion walked 32 placements
  out along `+q` and unwound. `undo` restores every observable field and keeps
  the allocation, so it holds the same position and the same legal set in four
  times the words. It is the only fixture that separates "cost per item" from
  "cost per arena word".

Benches are separate targets from tests and cannot `use` `tests/common`, so both
`#[path]`-include `testkit/rng.rs`. That file exists rather than a copy on each
side because "a fixture named by ply here is the position the test corpus builds
at that ply" is only true while the two generators agree constant for constant,
and two hand-matched copies would make it a coincidence.

## Connections

- `hexo-runner` holds the one canonical state and is the only crate that
  advances it. It also owns everything this crate refuses to model: ply caps,
  non-win match results, adjudication, and the game record (which is a move
  list, not a serialised position).
- Future model crates depend on this crate's read surface — `windows_through`,
  `stones`, `legal_actions` — for their own feature encoding. This crate never
  learns what a feature is.
