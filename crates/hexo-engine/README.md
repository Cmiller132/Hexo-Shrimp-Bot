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
    common/mod.rs       # deterministic fixtures, same PRNG as tests/common
    engine.rs           # the criterion suite
```

## Module map

| Module | Public? | Role |
| --- | --- | --- |
| `coord` | yes | Axial `HexCoord` (`q`, `r`, derived `s`), the three line `Axis` values, `hex_distance`, the radius-8 disk table, and the coordinate domain bound. |
| `player` | yes | `Player` and `TurnPhase` — who moves and where they are inside the two-placement turn. |
| `action` | yes | `Action` (the move atom), `ActionId` (the unbounded, exactly invertible record encoding), and `ACTION_ORDER_VERSION`. |
| `window` | yes | `Window` (pure six-cell line geometry), `WindowMask`, `WindowRef`, `WinningWindows` — the exposed win-detection surface. |
| `grid` | **no** | The dense recentred arena: two occupancy bit planes, one frontier bit plane, one coverage byte plane, the frontier rank/select scan, and the growth policy. **Zero items escape the crate.** |
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
- **Window masks are derived on read**, by an O(1) bit gather over an 11x11
  strip. Nothing about windows is stored, so there is no growth path, no delta,
  and no "stored mask disagrees with the board" bug class.
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
  silent: the network keeps training, against scrambled targets.
- **Symmetric bugs are the real hazard.** A wrong disk offset, a wrong shear in
  the QR fold, a wrong hash constant, or a growth copy with the same wrong index
  on both sides all apply and un-apply identically, so round-trip tests are blind
  to them. `Position::audit()`, the independent oracles in `tests/common`, and
  the frozen golden vectors are the only detectors — they are not redundant.

## Testing

```
cargo test --workspace                       # ~20 s, debug profile
cargo clippy --all-targets -- -D warnings
cargo clippy --release --all-targets -- -D warnings
cargo fmt --all --check
cargo build -p hexo-engine --target wasm32-unknown-unknown
```

The `wasm32` build is a real gate, not a formality: nothing in the native build
would catch a `std::time` call, a threading primitive, or a PyO3 dependency
creeping in, and any of those would silently cost this crate its ability to run
the real rules in a browser.

The release lint is a separate gate, not a duplicate: `debug_assertions` is off in
release, which deletes the only callers of the helpers the tier-C assertions use, so
the debug lint cannot see a dead-code regression there.

The smoke test scales via the environment for a nightly or release run:

```
HEXO_SMOKE_GAMES=10000 HEXO_SMOKE_UNIFORM=500 \
    cargo test --release -p hexo-engine --test smoke
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

Benches are separate targets from tests and cannot `use` `tests/common`, so
`benches/common/mod.rs` carries its own copy of the same splitmix64 with the
same constants.

## Connections

- `hexo-runner` holds the one canonical state and is the only crate that
  advances it. It also owns everything this crate refuses to model: ply caps,
  non-win match results, adjudication, and the game record (which is a move
  list, not a serialised position).
- Future model crates depend on this crate's read surface — `windows_through`,
  `stones`, `legal_actions` — for their own feature encoding. This crate never
  learns what a feature is.
