# hexo-engine

Authoritative Hexo rules and game state. Owns what the game *is*; owns nothing
about how it is played, recorded, or learned from.

**Status: EMPTY — scaffold only.** The rules rebuild has not started.

## Shape

Pure Rust library crate. No PyO3, no I/O, no threads, no tensors. Those
constraints are the point: they keep this crate `wasm32`-compilable (so a web
frontend can run real rules) and testable with `cargo test` alone.

```
crates/hexo-engine/
  Cargo.toml
  README.md
  src/
    lib.rs        # crate root and public surface
```

## Planned module map

Proposed split, not yet built. Roughly tracks the previous engine, with the
board representation and action identity changed (see `docs/SUGGESTIONS.md`).

| Module | Role |
| --- | --- |
| `coord.rs` | Axial hex coordinate math (`q`, `r`, derived `s`), distance, neighbours, the three line axes. |
| `board.rs` | Position storage: recentered dense grid plus per-player bitboards. Replaces the previous sparse `AHashMap` board. |
| `windows.rs` | Incremental six-cell window tracking; win and threat masks. The previous `tactics.rs`. |
| `legal.rs` | Incremental legal-placement set and the frontier radius rule. |
| `rules.rs` | Placement validity predicate — the smallest statement of legality. |
| `state.rs` | Position, turn phase (`Opening` / `FirstStone` / `SecondStone`), and `apply` / `undo` over an explicit delta stack. |
| `action.rs` | Action identity. Two encodings with separate jobs: unbounded coordinate IDs for records, dense indices for model I/O. |
| `zobrist.rs` | Incremental 64-bit position hash, maintained through the same delta stack as `apply` / `undo`. |
| `error.rs` | Error types for illegal placement and malformed state. |

## Design notes

- **`undo` is a first-class operation, not an afterthought.** The board, the
  window store, and the legal-move set are all maintained incrementally, so
  each must be exactly restorable. `apply(m); undo()` returning a state
  identical to the original is a property test, not a comment.
- **One placement is the atom, not one turn.** A turn is two placements, but a
  win is checked after each, so a turn can end after the first. Making the
  placement the unit avoids special-casing that everywhere downstream.
- **Clone is a memcpy.** Players search on their own copies, so the cost of
  copying a position is a first-order design constraint on how it is stored.

## Connections

- `hexo-runner` holds the one canonical state and is the only crate that
  advances it.
- Future model crates depend on this crate's read surface for their own feature
  encoding. This crate never learns what a feature is.
