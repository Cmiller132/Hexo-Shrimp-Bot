# hexo-engine

## Purpose

`hexo-engine` is the authoritative Hexo rules machine and position value.
It defines legal placements, turn progression, wins, canonical action order,
position hashing, replay, and reversible search state. It has no runtime
dependencies and contains no match policy, persistence, I/O, or model logic.

## Public surface

The crate root re-exports the consumer-facing types and constants from these
modules:

| Module | Consumer-facing contract |
| --- | --- |
| `action` | `Action`, `ActionId`, and `ACTION_ORDER_VERSION` |
| `coord` | `HexCoord`, `Axis`, `hex_distance`, and board-domain constants |
| `error` | `MoveError`, `ReplayError`, `IntegrityError`, `IntegrityCheck` |
| `player` | `Player` and `TurnPhase` |
| `position` | `Position`, `Applied`, `Outcome`, `Stones`, `LegalActions` |
| `search` | Borrow-scoped `Search` make/unmake session |
| `window` | `Window`, `WindowRef`, `WindowMask`, `Win` |

The crate-level compatibility constants are:

- `RULES_VERSION`: rule-machine semantics.
- `ACTION_ORDER_VERSION`: legal-action indexing semantics.
- `MAX_GRID_CELLS`: the internal representation limit.
- `LEGAL_RADIUS`, `DISK_CELLS`, `WINDOW_LEN`, and `COORD_LIMIT`: geometry
  bounds.

Typical state transitions use the flat re-exports:

```rust
use hexo_engine::{Action, HexCoord, Position};

let mut position = Position::new();
let applied = position.advance(Action::new(HexCoord::ORIGIN))?;
assert_eq!(applied.action.coord(), HexCoord::ORIGIN);
# Ok::<(), hexo_engine::MoveError>(())
```

`Position` provides:

- `new`, `replay`, and `replay_from` for construction through legal moves.
- `advance` for the only public mutation of game state.
- `current_player`, `phase`, `outcome`, and `is_terminal` for turn state.
- `stones`, `get`, and stone counts for occupancy reads.
- `legal_actions`, `legal_count`, `legal_rank`, `nth_legal`, and `is_legal`.
- `windows_through` and `window` for six-cell line reads.
- `zobrist` for the position-only key.
- `audit` for an independent integrity check.

`Search::new(&mut position)` supports `apply`, `undo`, `unwind`, and `commit`;
dropping an uncommitted search restores its borrowed position.

## Run / test

From the repository root:

```sh
cargo test -p hexo-engine
cargo test -p hexo-engine --test golden
cargo test -p hexo-engine --test properties
cargo test -p hexo-engine --test boundary
cargo doc -p hexo-engine --no-deps
```

Run the workspace gates that include formatting, lints, documentation, and
platform checks:

```sh
cargo xtask verify
```

Run the extended randomized engine suite:

```sh
cargo xtask smoke
```

Run benchmarks or one benchmark filter:

```sh
cargo bench -p hexo-engine
cargo bench -p hexo-engine -- ordering
cargo bench -p hexo-engine -- "apply_undo/edge"
```

## Connections

- The normative rule contract is [`docs/ENGINE_SPEC.md`](../../docs/ENGINE_SPEC.md).
- `crates/hexo-runner` owns match orchestration around one canonical
  `Position`.
- `crates/hexo-search` consumes `Position`, `Search`, and canonical legal
  ordering.
- `crates/hexo-records` replays stored `ActionId` values through this crate.
- `crates/models/mantisnet` reads stones, windows, and legal actions for its
  encoder.
- `python/hexo-py` exposes a restricted PyO3 read and replay surface.
- `src/grid.rs` and `src/zobrist.rs` are implementation-private.

## Invariants & gotchas

- One placement is the mutation atom; a turn may contain two placements.
- A win is checked after every placement, including the first placement of a
  two-placement turn.
- Terminal positions freeze the mover and phase and refuse further moves.
- Positions are created empty or by replay; no public board-shaped constructor
  exists.
- `Position` stores no move history. Match history belongs to the runner.
- Legal actions are frontier cells within radius eight of an existing stone;
  the opening action is the origin.
- Legal enumeration order is canonical and versioned. Policy indices must use
  `legal_rank` and `nth_legal`.
- `ActionId` is an exactly invertible coordinate identity, not a dense policy
  index.
- The board has no game-rule edge. `COORD_LIMIT` and `MAX_GRID_CELLS` are
  representation limits and their errors are not rule violations.
- `zobrist` and `PartialEq` describe position state, not move history or arena
  allocation geometry.
- Consecutive plies may have the same current player; depth parity is not a
  valid mover test.
- Window masks and the legal frontier are derived from occupancy.
- `Search` keeps its own undo depth; `commit` makes the current state the new
  floor and `unwind` returns to that floor.
- `audit`, golden vectors, and independent test oracles detect errors that
  apply/undo round trips can preserve symmetrically.
- Changes to rules, action ordering, or encoded coordinate identity require
  the corresponding version and fixture review.
