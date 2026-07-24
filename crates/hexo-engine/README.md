# hexo-engine

Authoritative Hexo rules and game state. Owns what the game *is*; owns nothing
about how it is played, recorded, or learned from.

**Status: MVP implemented.** Built to `docs/ENGINE_SPEC.md`, which is the
normative contract for this crate. Zero runtime dependencies; `proptest` is a
dev-dependency only.

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
    fixtures.rs         # hand-built win shapes
    golden.rs           # frozen Zobrist and move-ordering vectors
    properties.rs       # proptest properties
    smoke.rs            # random playouts
```

## Module map

| Module | Public? | Role |
| --- | --- | --- |
| `coord` | yes | Axial `HexCoord` (`q`, `r`, derived `s`), the three line `Axis` values, `hex_distance`, the radius-8 disk table, and the coordinate domain bound. |
| `player` | yes | `Player` and `TurnPhase` — who moves and where they are inside the two-placement turn. |
| `action` | yes | `Action` (the move atom), `ActionId` (the unbounded, exactly invertible record encoding), and `ACTION_ORDER_VERSION`. |
| `window` | yes | `Window` (pure six-cell line geometry), `WindowMask`, `WindowRef` — the exposed win-detection surface. |
| `grid` | **no** | The dense recentred arena: two occupancy bit planes, one frontier bit plane, one coverage byte plane, and the growth policy. **Zero items escape the crate.** |
| `position` | yes | `Position`, `Applied`, `Outcome`, `Stones`, `LegalActions`, the rule machine, every read accessor, and `audit`. |
| `search` | yes | `Search<'p>` — the borrow-scoped make/unmake session — and the crate-private `Undo` token. |
| `zobrist` | **no** | The `const` mixing function and the twelve turn keys. Reachable only through `Position::zobrist()`. |
| `error` | yes | `MoveError`, `IntegrityError`, `IntegrityCheck`. |

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
- **Clone is a memcpy.** Players search on their own copies, so the cost of
  copying a position is a first-order design constraint. The state holds no move
  history.
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
```

The release lint is a separate gate, not a duplicate: `debug_assertions` is off in
release, which deletes the only callers of the helpers the tier-C assertions use, so
the debug lint cannot see a dead-code regression there.

The smoke test scales via the environment for a nightly or release run:

```
HEXO_SMOKE_GAMES=10000 HEXO_SMOKE_UNIFORM=500 \
    cargo test --release -p hexo-engine --test smoke
```

## Connections

- `hexo-runner` holds the one canonical state and is the only crate that
  advances it. It also owns everything this crate refuses to model: ply caps,
  non-win match results, adjudication, and the game record (which is a move
  list, not a serialised position).
- Future model crates depend on this crate's read surface — `windows_through`,
  `stones`, `legal_actions` — for their own feature encoding. This crate never
  learns what a feature is.
