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
    error.rs
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
| `window` | yes | `Window` (pure six-cell line geometry, including the `cell_index` / `contains` / `intersects` / `touches` relations), `WindowMask`, `WindowRef`, `Win` (a completed run) — the exposed win-detection surface. |
| `grid` | **no** | The dense recentred arena: two occupancy bit planes, one coverage bit plane, the derived frontier and its rank/select scan, the row-run disk update, the undo dilation, and the growth policy. **Zero items escape the crate.** |
| `position` | yes | `Position`, `Applied`, `Outcome`, `Stones`, `LegalActions`, the rule machine, replay, every read accessor, and `audit`. |
| `search` | yes | `Search<'p>` — the borrow-scoped make/unmake session — and the crate-private `Undo` token. |
| `zobrist` | **no** | The `const` mixing function and the twelve turn keys. Reachable only through `Position::zobrist()`. |
| `error` | yes | `MoveError`, `ReplayError`, `IntegrityError`, `IntegrityCheck`. |

`lib.rs` re-exports the public set flat, so consumers write
`use hexo_engine::{Position, Search, Action, Player};`.

There is deliberately no `board.rs`, `rules.rs`, `legal.rs`, `windows.rs`, or
`snapshot.rs`. A public `Board` type re-opens the rule-bypassing construction
path, since anything that can build a board from cells can build one no sequence
of legal turns could reach; a free `is_legal_placement` invites non-atomic
check-then-place; and neither the legal set nor the window masks are stored, so
neither gets a module.

## Design notes

- **The board is a recentred dense grid, not a sparse map.** An infinite board
  reads like an argument for a `HashMap<HexCoord, _>`, but play is contiguous — a
  placement must be within 8 steps of a stone — and every operation here is a
  neighbourhood query: the radius-8 disk, the window gather, the run scan, the
  frontier scan. Dense, each is address arithmetic over words already in cache; sparse,
  each is hundreds of independent hashes, and the frontier scan loses canonical
  order and has to sort.

  The price is that the arena moves: recentred on the live stone box, grown when a
  placement would break the padding margin. `COORD_LIMIT` and `MAX_GRID_CELLS` are
  where it admits a bound, and both report as representation limits rather than
  rule violations, so a runner can tell "you broke a rule" from "I cannot hold
  this board".
- **`undo` is a first-class operation, not an afterthought.** Every field is
  restored by exactly one of four named mechanisms: re-running a self-inverse
  operation (occupancy, hash, counters), recomputing from the stones over the
  affected disk (coverage, the frontier count), copying a snapshot out of the
  delta (phase, mover), or deliberately not at all (arena geometry, which is
  private and therefore unobservable). Nothing is restored by re-derivation from
  the bookkeeping under test.
- **Coverage is one bit per cell, and the frontier is derived, not stored.**
  `covered` is the occupancy dilated by the radius-8 disk — a pure function of
  the stones. Apply ORs the placed disk in word-wide runs; undo recomputes the
  removed stone's disk from occupancy by a separable dilation over a 33x33
  window (the disk is a zonogon, so the dilation factors into three log-shift
  passes — spec §5.4 is normative). The legal set is then `covered & !occupied`,
  composed per word on read: membership is two bit probes, the count is one
  maintained `u32`, and enumeration is a bit scan that produces the canonical
  `(q, r)` order for free. There is no stored frontier plane left to disagree
  with its own definition.

- **The disk is written as 17 row runs, not 217 coordinates.** A radius-8 disk is
  `dq`-major, so each of its rows is one contiguous run — one or two words of a
  plane. `Grid::disk_runs` produces them, clipped exactly to the coordinate
  domain, and every disk-shaped read and write goes through them.

  The `DISK8` offset table is still there, and earns its place: it is the
  *independent* statement of the same cell set, walked offset by offset by the
  tier-C coverage recount on every debug undo, and compared against the runs
  directly in `grid`'s tests — membership and order both. A wrong run and a
  wrong offset are both symmetric, so neither can check itself.
- **Window masks are derived on read**, by an O(1) gather of the 11 cells a
  window can reach along each axis. Nothing about windows is stored, so there is
  no growth path, no delta, and no "stored mask disagrees with the board" bug
  class.

  Window *identity* is separate from that and reads nothing: `Window` is
  `(start, axis)`, and `cell`, `cells`, `cell_index`, `contains`, `intersects`,
  and `touches` are arithmetic on six coordinates with no `Position` in hand. A
  consumer building a cell/window incidence graph therefore pays the engine
  nothing for structure — only for the masks of windows that actually hold a
  stone, which it can enumerate exactly once each with no hash set by keeping a
  window only from the stone at the lowest set bit of `mask.occupied()`.
- **A win is a run, and it is found by walking.** After each placement the engine
  steps outward from the placed cell along all three axes, counting the mover's
  stones, and reports `Win { axis, start, len }` per axis in `Applied::wins`.
  That is one fact — *this line, this long* — where a set of overlapping
  six-windows was several, and the walk stops after two or three probes in almost
  every position. The earlier strided bit fold was faster in the worst case and
  bought nothing on a check that runs once per placement, while concentrating the
  crate's worst symmetric-bug risk (a QR shear and an index table, neither
  visible to any round-trip test) on the rule path. The debug cross-check in
  `apply_raw` re-answers the same question the other way, through
  `Position::window`, and the two must agree.
- **One placement is the atom, not one turn.** A turn is two placements, but a
  win is checked after each, so a turn can end after the first — and when it
  does, the phase and the mover *freeze*.
- **A position is a value, and it carries no move history.** Everything on the
  type is the stones or derived from them; the placement sequence belongs to
  whoever keeps the record, and `replay()` is the way back from a record to a
  board. A board that carried its own copy made every consumer that already had a
  record hold two, and the engine-side copy was the one nothing could check — it
  can only ever agree with the moves it was just handed. `hexo-runner`'s
  `Game::plies` is the record for a match.
- **There is exactly one hash, and it stays *position-only*.** Hexo transposes
  structurally, since a turn's two stones are playable in either order and reach
  the same position, so a history-sensitive key would forfeit a 2x merge per turn
  of search. A model whose features depend on move order hashes the record into
  its own cache key rather than asking the engine for a second hash.
- **`PartialEq` means *same position*.** It ignores arena geometry, matching
  `zobrist()`, `audit()`, and the oracles. Whether two games are the same game is
  a question about two records, not about two boards.
- **Clone copies three buffers**, one per grid plane — it was never a single
  `memcpy`, and the shorthand only ever meant "no pointer chasing, no per-cell
  work". At 3 bits per cell it is ~175 ns at 256 stones, 4.5x cheaper than the
  byte-per-cell design it replaced, and clone cost is what a search mirror
  actually pays.
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
  always at the centre. It fails on a board that has no edge: a radius-20 crop
  excludes out-of-crop legal moves from policy and search and freezes out-of-rim
  wins, so the model is trained on an action space that does not match the game
  it is playing. A larger radius narrows the failure without removing it, because
  the crop is still there. So the policy head is sized by the legal set instead,
  and the two jobs get two encodings — `ActionId` is action *identity*, unbounded
  and exactly invertible, for records and validation; this ordering is the
  *index*, for model I/O. One encoding cannot serve both: an identity that is
  unbounded is not a dense index, and a dense index is not invertible outside its
  region.

  The coordinate-domain clip in `Grid` is what keeps the ordering a bijection —
  coverage is never written outside the domain, so `legal_actions` cannot offer a
  coordinate `advance` would refuse. `tests/boundary.rs` pins it.
- **Symmetric bugs are the real hazard.** A wrong disk offset, a wrong offset in
  the window gather, a wrong hash constant, or a growth copy with the same wrong
  index on both sides all apply and un-apply identically, so round-trip tests are
  blind to them. `Position::audit()`, the independent oracles in `tests/common`, and
  the frozen golden vectors are the only detectors — they are not redundant.

## Testing

`cargo xtask verify` runs every gate, and `cargo xtask` says what each one
catches; they are defined in `xtask/src/main.rs`. The whole set takes well under
a minute, of which `cargo xtask test` is about 20 seconds.

Four of them see a class the debug test run cannot — the release lint, the
rustdoc gate, and the two `wasm32` gates. The `wasm32` pair is scoped to this
crate; the release lint and rustdoc gates run workspace-wide.

The deep smoke run is `cargo xtask smoke`: ten times as many line-building
games and more than sixteen times as many uniform games, release profile,
scheduled nightly rather than per-push. Worth running by hand after a change
to hashing, ordering, growth, or win detection.

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
`clone`, `enumerate`, `ordering`, `windows`, `replay`, and `new_game`; the
first six are each reported at plies 1, 32, 96, and 256, while `replay` is
parameterised by its record's length and `new_game` is a single benchmark.

Three fixture choices carry the weight:

- **Uniform random play**, from one seed, through the normal rule machine.
  Fixtures nest — the ply-256 move list starts with the ply-32 one — and a game
  that ends early is a panic, not a shorter fixture.
- **Interior versus edge placements**, the legal cells nearest to and furthest
  from the centroid of the stones. The interior cell's radius-8 disk is already
  fully covered, so its OR changes nothing; the edge cell's newly covers 136
  cells. That is the split `apply_undo` is measured over, and it is where the
  disk walk's cost actually lives.
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
