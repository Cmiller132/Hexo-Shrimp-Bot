# hexo-engine — specification

**Status: normative.** This is the implementation contract for
`crates/hexo-engine`: its public surface, rules, representation invariants,
ordering, hashing, and verification obligations.

---

## 1. Module map

Nine modules. `lib.rs` re-exports the public set flat, so consumers write
`use hexo_engine::{Position, Search, Action, Player};`.

| Module | Public? | One-line responsibility |
| --- | --- | --- |
| `coord` | yes | Axial coordinate, the three axes, hex distance, the radius-8 disk offsets, the coordinate domain bound. |
| `player` | yes | `Player` and `TurnPhase` — who moves and where they are inside the two-placement turn. |
| `action` | yes | `Action` (the move atom) and `ActionId` (the invertible record encoding) and the ordering version. |
| `window` | yes | `Window` (pure six-cell line geometry), `WindowMask`, `WindowRef`, and `Win` (a completed run) — the exposed win-detection surface. |
| `grid` | **no** | The dense recentred arena: two occupancy bit planes, one coverage bit plane, the derived frontier, and the growth policy. **Zero public items.** |
| `position` | yes | `Position`, `Applied`, `Outcome`, the rule machine (`advance` / `apply_raw` / `undo_raw`), every read accessor, `audit`. |
| `search` | yes | `Search<'p>` — the borrow-scoped make/unmake session — and the private `Undo` token. |
| `zobrist` | **no** | The const mixing function and the turn-key table. Reachable only through `Position::zobrist()`. |
| `error` | yes | `MoveError`, `ReplayError`, `IntegrityError`, `IntegrityCheck`. |

There is no `board.rs`, `rules.rs`, `legal.rs`, `windows.rs`, or `snapshot.rs`.
`Position` is the only public rule-bearing state. Legality mutation is exposed through
`Position::advance` and `Search::apply`; `Position::is_legal` is the read-only predicate.
The legal set and window masks are derived (§5–§6).

`grid` is `mod grid;`, not `pub mod grid;`. Grid geometry is entirely private, enforced by
the module system: **no public item in this crate may expose a row, word, plane, stride,
or storage index.**

---

## 2. Constants and versioning

```rust
// coord
/// Cells in a win window.
pub const WINDOW_LEN: usize = 6;
/// A non-opening placement must lie within this many hex steps of some stone.
pub const LEGAL_RADIUS: u32 = 8;
/// Cells in a radius-`LEGAL_RADIUS` hex disk: `3 * 8 * 9 + 1`.
pub const DISK_CELLS: usize = 217;
/// Largest magnitude allowed for any of `q`, `r`, `s` on a placed or queried cell.
///
/// A representation bound, not a rule. Every internal coordinate walk
/// (`+-8` for the disk, `+-5` for a window) stays inside `i16`.
pub const COORD_LIMIT: i16 = 16_000;

// window
/// Windows touched by one placement: 3 axes x 6 offsets.
pub const WINDOWS_PER_PLACEMENT: usize = 18;

// action
/// Version of the canonical legal-move ordering (§9).
///
/// Bumping this invalidates every trained checkpoint that indexed a policy head
/// by legal-move position.
pub const ACTION_ORDER_VERSION: u32 = 1;

// lib
/// Version of the rules and of the Zobrist mixing function (§8).
///
/// Bumping this invalidates cross-process hash agreement and every stored game
/// record. It is the `rules version` field of the container handshake (C2).
pub const RULES_VERSION: u32 = 1;
/// Hard ceiling on dense arena cells. Three bit planes use at most ~6 MiB.
///
/// A representation limit, not a rule: a placement that would push the arena past
/// this is legal, and the engine reports that it cannot represent it. It bounds the
/// AREA of the padded stone box, so a walk along one axis is bounded by COORD_LIMIT
/// instead (§5.6).
pub const MAX_GRID_CELLS: u64 = 1 << 24;
```

`RULES_VERSION` covers both rules and the Zobrist function because either change
invalidates the same artifacts.

**Dependencies.** `[dependencies]` is empty and stays empty. `[dev-dependencies]` is
`proptest = "1"` and `criterion` (default features off, `cargo_bench_support` only),
which back the property suite and benchmarks respectively; neither is reachable from a
non-test build. `MoveError` implements `core::fmt::Display` and
`core::error::Error` without a proc-macro dependency. The workspace Rust floor is 1.88.

The crate is not `no_std`; no `#![no_std]` attribute or gate establishes that contract.
The `wasm32` gate constrains the dependency graph and must reject native-only
dependencies.

---

## 3. Public types

### 3.1 `coord`

```rust
/// One cell on the unbounded hex board, in axial coordinates.
///
/// The third cube axis is derived: `s = -q - r`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct HexCoord {
    /// Axial `q` coordinate. Ordered first by `Ord`.
    pub q: i16,
    /// Axial `r` coordinate. Ordered second by `Ord`.
    pub r: i16,
}

impl HexCoord {
    /// The board centre `(0, 0)`, and the only legal opening placement.
    pub const ORIGIN: Self = Self { q: 0, r: 0 };

    /// Construct an axial coordinate. Total; performs no validation.
    pub const fn new(q: i16, r: i16) -> Self;

    /// The derived cube axis `-q - r`.
    ///
    /// Returns `i32`, not `i16`: `-q - r` overflows `i16` for extreme inputs, and
    /// this function is total over every `(i16, i16)` pair.
    pub const fn s(self) -> i32;

    /// Whether `q`, `r`, and `s` all lie within [`COORD_LIMIT`].
    ///
    /// The documented domain of every geometry function in this crate.
    pub const fn is_valid(self) -> bool;

    /// This coordinate stepped `n` cells along `axis`.
    ///
    /// # Panics
    /// Debug builds assert `self.is_valid()` and `|n| <= 8`. In release the
    /// arithmetic wraps; the crate forbids `unsafe`, so this is memory-safe but
    /// meaningless. No reachable engine state can supply an invalid coordinate.
    pub const fn step(self, axis: Axis, n: i16) -> Self;
}

impl core::ops::Add for HexCoord { type Output = Self; }
impl core::ops::Sub for HexCoord { type Output = Self; }

/// The three straight-line axes a win window can run along.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Axis {
    /// `(1, 0)`.
    Q,
    /// `(0, 1)`.
    R,
    /// `(1, -1)`.
    QR,
}

impl Axis {
    /// All three axes, in canonical order: `Q`, `R`, `QR`.
    pub const ALL: [Self; 3] = [Self::Q, Self::R, Self::QR];
    /// The unit step along this axis.
    pub const fn vector(self) -> HexCoord;
    /// Canonical index: `Q = 0`, `R = 1`, `QR = 2`.
    pub const fn index(self) -> usize;
}

/// Distance in hex steps between two cells.
///
/// Total over every pair of `i16` coordinates; computed in `i32` and returned as
/// `u32`, because the maximum representable separation exceeds `i16::MAX`.
pub const fn hex_distance(a: HexCoord, b: HexCoord) -> u32;
```

`Ord` on `HexCoord` is lexicographic `(q, r)`, the canonical ordering of §9, and must
agree with `ActionId` ordering.

`DISK8` — the `[(i8, i8); DISK_CELLS]` table of radius-8 offsets — lives in `coord` and is
`pub(crate)`. Its order is fixed:

```
for dq in -8..=8 { for dr in max(-8, -dq - 8) ..= min(8, -dq + 8) { yield (dq, dr) } }
```

That is `dq`-major, `dr`-minor, exactly 217 entries. The machine itself walks the disk as
`Grid::disk_runs` row runs (§5.4); this table is the *independent* statement of the same
cell set that those runs are tested against, and the `dq`-major order is what makes each
`dq` one contiguous run so the two formulations correspond row for row. The tier-C
coverage recount (C2) and the off-domain classification probe (§3.7) walk it directly.

### 3.2 `player`

```rust
/// One of the two players. `P0` places the opening stone.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Player {
    /// Player 0. Moves first, at the origin.
    P0 = 0,
    /// Player 1. Moves second; takes plies 1 and 2.
    P1 = 1,
}

impl Player {
    /// The opposing player.
    pub const fn other(self) -> Self;
    /// `0` for `P0`, `1` for `P1`.
    pub const fn index(self) -> usize;
}

/// Where the mover is inside the two-placement turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TurnPhase {
    /// Ply 0 only. `P0` must place at [`HexCoord::ORIGIN`].
    Opening,
    /// The mover places the first stone of its turn.
    FirstStone,
    /// The mover places the second stone of its turn.
    SecondStone,
}

impl TurnPhase {
    /// Canonical kind index: `Opening = 0`, `FirstStone = 1`, `SecondStone = 2`.
    ///
    /// Used by the Zobrist turn key (§8).
    pub const fn kind_index(self) -> usize;
}
```

`SecondStone` carries no payload. The first placement remains occupied, so reuse is
reported as `Occupied`. Placement order belongs to the game record, not `Position`.

### 3.3 `action`

```rust
/// Unbounded, exactly invertible identity of a placement. The record encoding.
///
/// The packing is order-preserving: comparing the inner `u32` is exactly
/// lexicographic `(q, r)` comparison on the signed coordinate (§9).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ActionId(
    /// The wire value. Public so a record writer needs no accessor.
    pub u32,
);

impl ActionId {
    /// `((q as u16 ^ 0x8000) as u32) << 16 | (r as u16 ^ 0x8000) as u32`.
    ///
    /// Total and injective over every `HexCoord`.
    pub const fn from_coord(c: HexCoord) -> Self;
    /// The exact inverse of [`ActionId::from_coord`]. Total over every `u32`.
    pub const fn coord(self) -> HexCoord;
}

impl From<HexCoord> for ActionId {}
impl From<ActionId> for HexCoord {}

/// A single placement — the atom of play.
///
/// Carries no legality claim; validation happens in [`Position::advance`] and
/// [`Search::apply`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Action(HexCoord);

impl Action {
    /// Wrap a coordinate as a placement.
    pub const fn new(coord: HexCoord) -> Self;
    /// The cell this placement targets.
    pub const fn coord(self) -> HexCoord;
    /// The record encoding of this placement.
    pub const fn id(self) -> ActionId;
    /// Recover a placement from its record encoding. Total over every `u32`.
    pub const fn from_id(id: ActionId) -> Self;
}

impl From<HexCoord> for Action {}
impl From<Action> for HexCoord {}
impl From<Action> for ActionId {}
impl From<ActionId> for Action {}
```

`ActionId` is a newtype, not a `u32` alias, and is distinct from dense policy indices.
`Action`'s field is private; construction is through coordinate or id conversion.

`Action` derives `Ord`, and it must agree with `ActionId::cmp` and `HexCoord::cmp`. Three
orderings, one meaning, pinned by a unit test over a fixed coordinate set including all
four sign quadrants.

### 3.4 `window`

```rust
/// Ownership of one six-cell window, as two six-bit masks.
///
/// Bit `i` refers to the window's cell `i`, which is a statement about the
/// infinite board (`start + axis.vector() * i`) and never about storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct WindowMask([u8; 2]);

impl WindowMask {
    /// The empty window.
    pub const EMPTY: Self = Self([0, 0]);
    /// Bit `i` set iff cell `i` of the window holds a stone of `player`. Low six bits.
    pub const fn mask(self, player: Player) -> u8;
    /// Stones `player` holds in this window, `0..=6`.
    pub const fn count(self, player: Player) -> u32;
    /// Either player's stones. `mask(P0) | mask(P1)`.
    pub const fn occupied(self) -> u8;
    /// Complement of [`WindowMask::occupied`] within the low six bits.
    pub const fn empty(self) -> u8;
    /// Whether `player` owns all six cells — the win condition for this window.
    pub const fn is_full_for(self, player: Player) -> bool;
}

/// Identity of one six-cell window: its first cell and the axis it runs along.
///
/// Pure geometry. Constructible and interpretable with no `Position` in hand, and
/// valid forever regardless of how the engine stores the board.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Window {
    /// Cell `0` of the window.
    pub start: HexCoord,
    /// The direction cells `1..6` run in.
    pub axis: Axis,
}

impl Window {
    /// Coordinate of cell `index`.
    ///
    /// # Panics
    /// Panics if `index >= WINDOW_LEN`. Debug builds also assert
    /// `self.start.is_valid()`.
    pub const fn cell(self, index: usize) -> HexCoord;
    /// All six coordinates, in bit order.
    ///
    /// # Panics
    /// Debug builds assert `self.start.is_valid()`.
    pub const fn cells(self) -> [HexCoord; WINDOW_LEN];
    /// Which of the six cells `coord` is, or `None`. The inverse of `cell`.
    /// Total, and computed in `i32` rather than by walking `cells`.
    pub const fn cell_index(self, coord: HexCoord) -> Option<usize>;
    /// Whether `coord` is one of this window's six cells.
    pub const fn contains(self, coord: HexCoord) -> bool;
    /// Whether the two windows share at least one cell. Symmetric.
    pub const fn intersects(self, other: Self) -> bool;
    /// Whether the two windows are disjoint but have a pair of adjacent cells.
    /// **Exclusive of overlap**, so `intersects` / `touches` / neither partition
    /// every pair.
    pub const fn touches(self, other: Self) -> bool;
}

/// A window paired with its current ownership.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WindowRef {
    /// Which window.
    pub window: Window,
    /// Who owns which of its cells.
    pub mask: WindowMask,
}

/// A maximal run of one player's stones along one axis.
///
/// Produced by win detection (§6.4), so `len >= WINDOW_LEN`. Maximal in both
/// directions: the cells one step before `start` and one step past the end are not
/// that player's. Plain data with no methods — the run is `start` stepped `0..len`
/// along `axis`, which every consumer can walk directly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Win {
    /// The axis the run lies along.
    pub axis: Axis,
    /// The first cell of the run — its end furthest back along `axis`.
    pub start: HexCoord,
    /// Cells in the run, `start` stepped `0..len` along `axis`. At most 11 (§6.4).
    pub len: u8,
}
```

`Win` has no `Default` and no empty value. "No win" is `Option<Win>`, so a placement that
did not win cannot be confused with one that won a run of length zero.

`WindowMask`'s inner `[u8; 2]` is private: the player-to-lane mapping is an internal
convention, and `mask(Player::P1)` is the contract.

There is no `is_win_for`, `threat_player`, `is_active`,
`stone_cells() -> Vec<_>`, or `intersects_or_touches`. Ownership predicates derive from
`WindowMask`; coordinate relationships live on `Window`. `cell_index` is the primitive,
`contains` is its `is_some()`, and `intersects` uses `contains`.

### 3.5 `position`

```rust
/// A Hexo position: board, turn phase, mover, hash, terminal status.
///
/// A value type. It carries no placement sequence and no undo stack (§7): a game is
/// rebuilt by `replay` from a record its keeper holds, which for a match is
/// `hexo_runner::Game::plies`.
///
/// `PartialEq` is content-based and ignores arena geometry: two
/// positions with the same stones, phase, mover, and terminal status are equal even
/// if one's arena grew larger getting there. Equality means *same position*,
/// matching `zobrist`, the oracles, and `audit`. It is `O(arena extent)`.
///
/// This type does **not** implement `core::hash::Hash`. Use
/// [`Position::zobrist`] as the map key: a *derived* `Hash` would fold in the arena
/// geometry that `PartialEq` ignores, breaking the `Eq`/`Hash` contract, and `Grid`
/// implements neither trait so that the derive cannot compile (§5.7).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Position { /* private, see §5 */ }

impl Default for Position { /* = Position::new() */ }

/// What one placement did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Applied {
    /// The placement that was made.
    pub action: Action,
    /// Who made it. Equals `phase`-independent `current_player()` before the call.
    pub mover: Player,
    /// The phase before the placement.
    pub phase_before: TurnPhase,
    /// The phase after. Equals `phase_before` exactly when this placement won.
    pub phase_after: TurnPhase,
    /// `Some` iff this placement completed a six-window.
    pub outcome: Option<Outcome>,
    /// The run this placement completed on each axis, indexed by `Axis::index()`.
    ///
    /// Some entry is `Some` iff `outcome.is_some()`; two can be, when the placement
    /// completes two crossing lines at once. Every `Some` entry contains `action`
    /// and its `axis` field equals the axis it is indexed by.
    pub wins: [Option<Win>; 3],
}

/// How the game ended. Win only.
///
/// Non-win match results (ply caps, crashes, adjudication) belong to the runner.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Outcome {
    /// The player who completed a window.
    pub winner: Player,
}

/// Occupied cells with their owners, in canonical `(q, r)` order.
#[derive(Clone, Debug)]
pub struct Stones<'a> { /* private */ }
impl<'a> Iterator for Stones<'a> { type Item = (HexCoord, Player); }
impl<'a> ExactSizeIterator for Stones<'a> {}
impl<'a> core::iter::FusedIterator for Stones<'a> {}

/// Legal placements in canonical order (§9). Allocation-free.
#[derive(Clone, Debug)]
pub struct LegalActions<'a> { /* private */ }
impl<'a> Iterator for LegalActions<'a> { type Item = Action; }
impl<'a> ExactSizeIterator for LegalActions<'a> {}
impl<'a> core::iter::FusedIterator for LegalActions<'a> {}
```

`Outcome` carries only the winner. Placement count is `Position::stone_count()`.

`wins` is a plain array with no accessor methods, because the array *is* the answer:
`wins.iter().flatten()` is every completed run and `wins[axis.index()]` is the one on a
named axis. `Win` is declared in §3.4.

There is at most one run per axis: a placement lies on exactly one line of each axis, and
the reported run is that line's maximal run through it.

### 3.6 `search`

```rust
/// Exclusive make/unmake session over a position. The only path to `undo`.
///
/// Seeding a `Search` fixes the **undo floor**: the position as it was at
/// [`Search::new`]. Nothing in the API can rewind past it (§7.3).
///
/// On `Drop` the session unwinds to the floor, so a position lent to a search is
/// returned in its seeded state on every exit path, including `?` and panic.
#[derive(Debug)]
pub struct Search<'p> { /* private */ }
```

`Undo` is `pub(crate)` and appears in no public signature. See §7.

### 3.7 `error`

```rust
/// Why a placement was rejected. On `Err` the position is bit-identical to before.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoveError {
    /// The game is over; no placement is legal.
    TerminalState,
    /// The opening placement must be at [`HexCoord::ORIGIN`].
    IllegalOpening,
    /// The coordinate is outside [`COORD_LIMIT`], yet within [`LEGAL_RADIUS`] of
    /// a stone — a placement the rules allow but the engine cannot represent.
    ///
    /// A representation limit, not a rule. See [`MoveError::is_rule_violation`].
    /// An off-domain cell far from every stone is [`MoveError::TooFarFromStones`]
    /// instead: a rule violation is reported as one even when the cell is also
    /// unrepresentable.
    CoordOutOfBounds(HexCoord),
    /// The cell already holds a stone.
    Occupied(HexCoord),
    /// The cell is further than [`LEGAL_RADIUS`] from every stone — on the
    /// coordinate domain or off it.
    TooFarFromStones(HexCoord),
    /// The dense arena would exceed [`MAX_GRID_CELLS`].
    ///
    /// A representation limit, not a rule: the placement is legal and the engine
    /// cannot represent the board it would produce.
    BoardExtentExceeded {
        /// Cells the arena would have needed.
        cells: u64,
    },
}

impl MoveError {
    /// Whether this rejection is a rule violation rather than an engine limit.
    ///
    /// `false` for [`MoveError::CoordOutOfBounds`] and
    /// [`MoveError::BoardExtentExceeded`], `true` for the rest. A runner
    /// adjudicating an illegal move should treat `false` as an engine fault, not
    /// as a player fault.
    pub const fn is_rule_violation(self) -> bool;
}

impl core::fmt::Display for MoveError {}
impl core::error::Error for MoveError {}

/// A placement sequence that stopped being legal partway through.
///
/// The error type of `replay` and `replay_from` (§4.2). `ply` identifies where
/// the sequence stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplayError {
    /// Index into the replayed slice, counting from zero.
    pub ply: usize,
    /// The placement that was refused.
    pub action: Action,
    /// Why it was refused.
    pub cause: MoveError,
}

impl core::fmt::Display for ReplayError {}
impl core::error::Error for ReplayError {
    /// `Some(&self.cause)`.
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)>;
}

/// A failed [`Position::audit`] check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IntegrityError {
    /// Which invariant failed.
    pub check: IntegrityCheck,
    /// The cell it failed at, when the check is per-cell.
    pub coord: Option<HexCoord>,
}

/// The invariant that [`Position::audit`] found broken. See §10.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum IntegrityCheck {
    /// A cell is owned by both players.
    DoubleOwned,
    /// A per-player stone count disagrees with its plane. Also covers the total:
    /// `stone_count` is the sum of the per-player counts.
    StoneCountForPlayer,
    /// The `covered` plane disagrees with a recount of stones within `LEGAL_RADIUS`.
    Coverage,
    /// The derived frontier's population disagrees with the maintained counter.
    FrontierCount,
    /// `hash_cells` disagrees with a from-scratch recomputation.
    Zobrist,
    /// Terminal status disagrees with a brute-force six-in-a-row scan.
    Terminal,
    /// The reported winner is not the owner of the completed window.
    Winner,
    /// Phase or mover disagrees with the closed form of §10.2.
    TurnClosedForm,
    /// A stone lies within `LEGAL_RADIUS` of the arena boundary.
    ArenaMargin,
}
```

There is no `StateLoadError`; positions are reconstructed by replay (§4.2, §12).

**Error precedence is observable and pinned by tests**, in this exact order:

```
TerminalState
  -> IllegalOpening
  -> CoordOutOfBounds
  -> Occupied
  -> TooFarFromStones
  -> BoardExtentExceeded
```

The order is observable through `advance` and must not change without a contract change.
The first two checks require no board access. Domain validation precedes coordinate
arithmetic. `Occupied` precedes `TooFarFromStones`; `BoardExtentExceeded` follows every
rule check (§7.1).

**An off-domain coordinate classifies by the rules, not only by the domain.** Off-domain
and within
`LEGAL_RADIUS` of a stone, the placement is rule-legal and merely unrepresentable —
`CoordOutOfBounds`, an engine limit. Off-domain anywhere else it is plain
`TooFarFromStones`, a rule violation. (Off-domain `Occupied` cannot arise: stones exist
only on valid cells.) Runners distinguish rule violations from engine limits through
`is_rule_violation()`. Classification uses a `DISK8` walk over neighbours in `i32` and is
reached only for off-domain input (§7.1).

---

## 4. Public function signatures and contracts

### 4.1 Construction and turn state

```rust
impl Position {
    /// The empty position: `P0` to move, [`TurnPhase::Opening`], no arena allocated.
    pub fn new() -> Self;

    /// Whose turn it is. Frozen at the winner once terminal.
    pub fn current_player(&self) -> Player;

    /// Where the mover is inside its turn. Frozen once terminal.
    pub fn phase(&self) -> TurnPhase;

    /// The winner, if the game is over.
    pub fn outcome(&self) -> Option<Outcome>;

    /// Whether the game is over. Equivalent to `outcome().is_some()`.
    pub fn is_terminal(&self) -> bool;

    /// Incremental Zobrist hash (§8).
    ///
    /// Identical across builds, machines, and processes for a given
    /// [`RULES_VERSION`]. Covers stones and owners, the mover, the phase *kind*,
    /// and the terminal bit.
    pub fn zobrist(&self) -> u64;
}
```

Every branch on `phase()` in consumer code must test `is_terminal()` first: a terminal
position carries whichever phase it froze at, and a second-stone win freezes at
`SecondStone` with the turn's first stone already on the board.

### 4.2 Advancing the canonical position

```rust
impl Position {
    /// Advance the position irreversibly by one placement.
    ///
    /// The only mutating method on `Position`.
    ///
    /// # Errors
    /// Any [`MoveError`], in the precedence of §3.7. **Atomic:** on `Err` the
    /// position is bit-identical to before the call, including arena geometry.
    pub fn advance(&mut self, action: Action) -> Result<Applied, MoveError>;

    /// Rebuild a position by replaying a placement sequence from the empty board.
    ///
    /// Replaying the record of a game reproduces the position it reached; any
    /// prefix reproduces the position at that ply.
    ///
    /// # Errors
    /// `ReplayError { ply, action, cause }`. A sequence continuing past a win
    /// fails with `TerminalState` at the first surplus ply.
    pub fn replay(actions: &[Action]) -> Result<Self, ReplayError>;

    /// Apply a placement sequence to an existing position, continuing from where it
    /// stands.
    ///
    /// # Errors
    /// As `replay`, with `ply` relative to `actions`. **Not atomic:** placements
    /// before the failure are not rolled back.
    pub fn replay_from(&mut self, actions: &[Action]) -> Result<(), ReplayError>;
}
```

**Replay is the only position-loading interface.** Every replayed placement goes through
`advance`, so every constructed position is reachable through the rule machine. There is
no `serde` implementation or board-shaped deserialization for `Position`.

`ReplayError::ply` identifies the first rejected placement in the supplied slice.

A mirror must call `advance` and must not re-derive the phase transition. The
transition is `if won { phase_before } else { advance_turn(phase_before, mover) }`, and
the win branch freezes the turn state.

`advance` builds the same `Undo` that `Search::apply` does and drops it. `Undo` has no
heap allocation or `Drop` implementation.

### 4.3 Legal moves

```rust
impl Position {
    /// Number of legal placements. `0` if and only if the position is terminal.
    ///
    /// `1` in [`TurnPhase::Opening`]. Otherwise the number of empty cells inside the
    /// coordinate domain and within
    /// [`LEGAL_RADIUS`] of at least one stone.
    pub fn legal_count(&self) -> usize;

    /// Legal placements in canonical order (§9). Allocation-free.
    ///
    /// Yields exactly [`HexCoord::ORIGIN`] in [`TurnPhase::Opening`], nothing when
    /// terminal, and `legal_count()` items otherwise.
    pub fn legal_actions(&self) -> LegalActions<'_>;

    /// Whether `action` is legal right now: phase, occupancy, and radius.
    ///
    /// Returns `bool`, not `Result`; `advance` supplies the error variant.
    pub fn is_legal(&self, action: Action) -> bool;

    /// Where `action` sits in `legal_actions()` order, or `None` if it is not
    /// legal here.
    pub fn legal_rank(&self, action: Action) -> Option<usize>;

    /// The legal placement at `index` in `legal_actions()` order, or `None` if
    /// `index >= legal_count()`.
    pub fn nth_legal(&self, index: usize) -> Option<Action>;
}
```

`legal_rank` and `nth_legal` are inverse directions of the canonical ordering used by
policy consumers. The engine is the sole implementation of this mapping, and a golden
test pins the rank of each played move.

Both are a popcount prefix and a select scan over the derived frontier — each word is
`covered & !occ0 & !occ1`, composed on read (§5.1) — which the `q`-major/`r`-minor layout
makes equal to position within the canonical enumeration. Each touches only the words up
to its target, so the cost is `O(word index)`.

`legal_actions` never special-cases at the call site: callers do not branch on phase to
generate moves. A caller reusing a buffer writes
`out.clear(); out.extend(pos.legal_actions());` — the same allocation behaviour as a
`write_into` method, with no engine API for it. `ExactSizeIterator` means `collect`
pre-sizes exactly.

Callers derive action ids with `.map(Action::id)`; there is no second iterator.

### 4.4 Windows

```rust
impl Position {
    /// Ownership of the 18 windows through `coord`, in the canonical slot order
    /// of §6.3: axis-major (`Q`, `R`, `QR`), then offset `0..6`, where offset `k`
    /// means `coord` sits at bit `k` of the window.
    ///
    /// Total over the coordinate domain: defined for any valid coordinate,
    /// occupied or not, inside the arena or far outside it. Cells outside the
    /// arena read as empty. Returns a stack array; allocates nothing.
    ///
    /// A returned `Window::start` may lie up to `WINDOW_LEN - 1` cells **outside**
    /// the coordinate domain, because slot `k` starts `k` cells back along its
    /// axis. Such a slot's mask is correct and its identity is well defined as
    /// geometry, but it may not be fed back into [`Position::window`] or
    /// [`Window::cells`] — see below.
    ///
    /// # Panics
    /// Debug builds assert `coord.is_valid()`.
    pub fn windows_through(&self, coord: HexCoord) -> [WindowRef; WINDOWS_PER_PLACEMENT];

    /// Ownership of one specific window.
    ///
    /// Total over the coordinate domain: a window no stone has ever been near
    /// reads as [`WindowMask::EMPTY`]. There is no `Option` — "no stone has ever
    /// been here" and "empty" are the same answer, and an `Option` would force a
    /// branch on every caller to say the same thing.
    ///
    /// # Panics
    /// Debug builds assert `window.start.is_valid()`.
    pub fn window(&self, window: Window) -> WindowMask;
}
```

`windows_through` returns the eighteen six-bit ownership patterns incident to a
candidate cell.

**The window-start domain is narrower than the queried-coordinate domain.** Within five
steps of a `COORD_LIMIT` face, up to ten of the eighteen slots `windows_through` returns
have `!start.is_valid()`, and up to fifteen at a corner where two of the three cube
bounds are tight, such as `(-COORD_LIMIT, 0)`. Their masks are correct, but `Position::window` and
`Window::cells` assert a valid start in debug, so round-tripping one panics there and
works in release. That is the contract: a caller re-querying a slot must skip invalid
starts.

### 4.5 Occupancy

```rust
impl Position {
    /// Owner of `coord`, or `None` if empty.
    ///
    /// Total over every `(i16, i16)` pair, including coordinates far outside the
    /// arena — those are simply empty. This totality is what keeps geometry
    /// private: a caller can probe anywhere without discovering where the arrays
    /// end.
    pub fn get(&self, coord: HexCoord) -> Option<Player>;

    /// Whether no stone occupies `coord`. Total, as [`Position::get`].
    pub fn is_empty_cell(&self, coord: HexCoord) -> bool;

    /// Total stones placed. Equals the ply count.
    pub fn stone_count(&self) -> u32;

    /// Stones held by one player.
    pub fn stone_count_for(&self, player: Player) -> u32;

    /// Every occupied cell with its owner, in canonical `(q, r)` order (§9).
    ///
    /// Position-only and route-independent: two positions with the same stones
    /// yield the same sequence regardless of the order the stones were played or
    /// how the arena grew.
    pub fn stones(&self) -> Stones<'_>;
}
```

There is no `history()`: records own placement order and reconstruct positions through
`replay`. Move identity in a record is `ActionId`.

There is no `bounds()`, insertion-ordered `occupied_cells()`, or raw bitboard/plane
accessor (§12).

### 4.6 Integrity audit

```rust
impl Position {
    /// Recompute every derived structure from the stones alone and compare.
    ///
    /// A normal method, not a `cfg` or cargo feature.
    /// `O(arena extent * DISK_CELLS)`; not intended for a search loop.
    ///
    /// Its recomputation takes a different route than the incremental path — a
    /// stone-by-stone repaint where the increment edits one disk, a cell-by-cell
    /// window scan where the increment walks runs. It necessarily shares the
    /// *definitions*: `cell_key` is the hash, so its constants are verified by
    /// the frozen golden vectors (§8).
    ///
    /// # Errors
    /// The first broken invariant found, in the order listed in §10.4.
    pub fn audit(&self) -> Result<(), IntegrityError>;
}
```

### 4.7 The search session

```rust
impl<'p> Search<'p> {
    /// Begin a make/unmake session. The position's current state becomes the
    /// undo floor.
    pub fn new(position: &'p mut Position) -> Self;

    /// Read the position at the current depth. Never handed out mutably.
    pub fn position(&self) -> &Position;

    /// Plies applied above the floor.
    pub fn depth(&self) -> usize;

    /// Whether no plies have been applied above the floor.
    pub fn at_floor(&self) -> bool;

    /// The placements applied above the floor, oldest first.
    pub fn path(&self) -> impl Iterator<Item = Action> + '_;

    /// Apply one placement, recording how to reverse it.
    ///
    /// # Errors
    /// As [`Position::advance`], with the same precedence and the same atomicity
    /// guarantee. On `Err` the depth is unchanged.
    pub fn apply(&mut self, action: Action) -> Result<Applied, MoveError>;

    /// Reverse the most recent [`Search::apply`], restoring the board, coverage,
    /// frontier, hash, phase, mover, and terminal status exactly.
    ///
    /// Returns the placement that was undone, or `None` at the floor. At the
    /// floor it is the identity: the position is not touched.
    pub fn undo(&mut self) -> Option<Action>;

    /// Undo every ply back to the floor.
    pub fn unwind(&mut self);

    /// Move the floor to the current depth: the applied plies become permanent
    /// for this session and can no longer be undone.
    pub fn commit(&mut self);
}

impl Drop for Search<'_> {
    /// Unwinds to the floor.
    fn drop(&mut self);
}
```

`undo` is total over session states: it returns `None` and leaves the position unchanged
at the floor.

`unwind` — and therefore `Drop` — performs no fallible work and cannot panic in release.
In debug it runs the undo assertions of §10.1.

---

## 5. Private representation

### 5.1 What is stored, and what is not

```rust
pub struct Position {
    grid: Grid,                 // §5.2
    phase: TurnPhase,
    current: Player,
    terminal: Option<Outcome>,
    hash_cells: u64,            // XOR of cell keys only; turn key applied on read (§8)
    stones_by: [u32; 2],
}

#[derive(Clone, Debug)]
struct Grid {
    rows: usize,                // extent along q
    row_words: usize,           // u64 words per row, i.e. extent along r is 64 * row_words
    origin_q: i32,              // q of row 0
    origin_r: i32,              // r of bit 0; always a multiple of 64
    occ: [Vec<u64>; 2],         // rows * row_words words each; stones of P0 / P1
    covered: Vec<u64>,          // rows * row_words words; cells within LEGAL_RADIUS of a stone
    frontier_cells: u32,        // popcount of the derived frontier, maintained
}
```

**There is no stored legal-move set, window-mask table, or frontier plane.**

- The legal set is the **derived frontier**: word `i` is
  `covered[i] & !occ[0][i] & !occ[1][i]`, composed on read and never stored. Membership
  is one bit probe; enumeration is a bit scan in canonical order (§9); the count is the
  maintained `u32`.
- The window masks are **derived on read** from the occupancy planes by an O(1) cell
  gather (§6.2), and win detection is an independent run scan (§6.4). The public surface
  exposes masks as specified in §4.4.

Cell `(q, r)` maps to row `i = q - origin_q` and bit `j = r - origin_r`. **All index
arithmetic is performed in `i32`**, so a coordinate anywhere in the `i16` range produces
an in-range-or-out-of-range answer and never wraps. Out-of-range reads return zero; the
internal write path never sees one, because growth (§5.5) runs first.

Bit-per-cell layout is `q`-major, `r`-minor; a row scan therefore produces canonical
ascending `(q, r)` order.

**No placement sequence is stored.** The record keeper owns the move list, and
`Position::replay` rebuilds a position from it (§4.2). Every `Position` field is board
state or a value derived from board state.

`stone_count()` is `stones_by[0] + stones_by[1]`; there is no third count field.

Memory is three bits per arena cell: two occupancy planes and `covered`. The first
placement allocates a 32x128 arena with 1,536 bytes of plane payload. `clone` allocates
and copies the three grid planes.

### 5.2 Coverage representation and undo recomputation

`covered[c]` states that some stone lies within `LEGAL_RADIUS` of `c` — the occupancy
dilated by the radius-8 disk. It is a **pure function of the stone set**.

An OR of radius-8 disks cannot be un-ORed — removing one stone cannot clear cells another
stone also covers. Apply ORs the placed disk in word-wide runs (§5.4); undo recomputes the
removed stone's disk from occupancy. Removing the stone at `c` changes coverage only
inside `c`'s disk, and every stone covering a cell of that disk lies within
`2 * LEGAL_RADIUS` of `c`. The recomputation is the 33x33 separable dilation of §5.4.

> **The frontier invariant.** `c` is a frontier cell if and only if
> `covered[c] == 1 && occ[0][c] == 0 && occ[1][c] == 0`. The engine evaluates this
> definition on read.

A non-opening placement at `c` is legal iff `c` is covered and empty. The lookup performs
no coordinate arithmetic beyond the `i32` index map. Off-domain coordinates use the
classification path of §3.7.

### 5.3 Empty position

`Position::new()` allocates nothing: `rows = 0`, `row_words = 0`, all three `Vec`s empty.
Every read is out-of-range and answers empty. The first `advance` grows the arena to the
initial geometry before touching anything:

```
MIN_ROWS      = 32     // q in [-15, 16] initially
MIN_ROW_WORDS = 2      // r in [-64, 63] initially
```

This contains the radius-8 disk around the origin with margin. Step 4 of §5.5 computes
the `q` span as `[-15, 16]`; a unit test pins it.

### 5.4 The placement pair

`Grid::place_stone` and `Grid::unplace_stone` are the only writers of occupancy and
coverage; `Position::place` / `unplace` wrap them with the hash and the stone counters.
Both halves have the same shape — **measure the frontier, mutate, measure again**:

```rust
pub(crate) fn place_stone(&mut self, c: HexCoord, p: Player) {
    let runs = self.disk_runs(c);                  // 17 domain-clipped row runs
    let before = self.frontier_pop_runs(&runs);
    /* set the occupancy bit at c */
    for run in runs { /* covered |= run, word-wide */ }
    let after = self.frontier_pop_runs(&runs);
    self.frontier_cells = self.frontier_cells - before + after;
}

pub(crate) fn unplace_stone(&mut self, c: HexCoord, p: Player) {
    let runs = self.disk_runs(c);
    let before = self.frontier_pop_runs(&runs);
    /* clear the occupancy bit at c */
    self.recompute_covered_disk(c, &runs);         // the dilation below
    let after = self.frontier_pop_runs(&runs);
    self.frontier_cells = self.frontier_cells - before + after;
}
```

Either half changes the derived frontier
only inside the placed cell's disk: occupancy changes at `c`, which is a member of its own
disk, and coverage changes nowhere outside it. So popcounting the derived frontier over
the disk runs before and after the mutation is the complete `frontier_cells` update.

**The disk is walked as 17 contiguous row runs, inside `Grid`.** `DISK8` is `dq`-major
and `dr`-minor, so each `dq` is one run of consecutive cells in storage order — one or two
words of each plane. `Grid::disk_runs` produces those `(start, len)` runs, and every
disk-shaped read and write — the coverage OR, the recomputation's writeback, the frontier
popcounts — goes through them. The run walk visits exactly the `DISK8` cells in exactly
the `DISK8` order. `grid` tests compare run membership and order directly with the
independent `DISK8` table, and the tier-C coverage recount (C2) walks the table on every
undo.

**Coverage is written only inside the coordinate domain.** The clip lives in `disk_runs`
and is applied once per row. At a fixed `q`, `is_valid` reduces to `r` lying
in `[max(-LIM, -LIM - q), min(LIM, LIM - q)]`, because `s = -q - r` is the only one of the
three axes whose bound depends on both. Every disk operation reads the same runs, so the
halves cannot disagree about the clip. Pinned by `tests/boundary.rs` at all six faces, and
compared against `DISK8` plus a per-cell `is_valid` in `grid`'s tests.

**The undo recomputation, normatively.** Coverage is occupancy dilated by the radius-8
hex disk, and the disk is a *zonogon* — the Minkowski sum of three segments, translated:

```
{ (dq, dr) : |dq| <= 8, |dr| <= 8, |dq + dr| <= 8 }
    = { a·(1,0) + b·(0,1) + c·(1,-1) : a, b, c in 0..=8 } + (-8, 0)
```

(Writing a target as `(a + c - 8, b - c)`: `a in 0..=8` forces `c in [dq, dq + 8]` and
`b in 0..=8` forces `c in [-dr, 8 - dr]`, and those intervals intersect exactly when the
three disk inequalities hold.) Dilation by a Minkowski sum factors into successive
dilations by each summand, and a 1-D dilation by a 9-cell segment decomposes into
doubling shifts — spans `[1, 2, 4, 1]` accumulate `0..=8`. Over a window `t` with
`t[i]` bit `j` = occupancy at `(c.q - 16 + i, c.r - 16 + j)`:

1. **+Q rows:** `for d in [1, 2, 4, 1] { for i descending { t[i] |= t[i - d] } }`
2. **+R bits:** per row, `v |= v << 1; v |= v << 2; v |= v << 4; v |= v << 1`
3. **+QR diagonal:** `for d in [1, 2, 4, 1] { for i descending { t[i] |= t[i - d] >> d } }`

The descending row order makes each span read the previous pass's output rather than its
own. After the three passes the `(-8, 0)` translation lands covered row `c.q - 8 + k` at
`t[16 + k]` with the same bit origin, and writeback replaces exactly the bits selected by
the domain-clipped runs, leaving other bits of shared words untouched. The window is
33x33 (`4 * LEGAL_RADIUS + 1`: every stone covering a cell of `c`'s disk lies within
`2 * LEGAL_RADIUS` of `c`), 33 bits fit one `u64` with room for the shifts, and the whole
recomputation is a few hundred word operations with no allocation.

The dilation is independent of both the `DISK8` offset table and the run-OR of apply.
`grid` tests hold all three formulations equal through place/unplace/growth scripts
(§11), C2 recounts the disk against `DISK8` on every debug undo, and `audit` repaints
coverage from stones at each test checkpoint.

`place` reads nothing about the win and is called unconditionally (§7.1).

### 5.5 Growth policy

**Trigger.** Before any mutation, `advance` requires the arena to contain
`[c.q - 8, c.q + 8] x [c.r - 8, c.r + 8]`. Padding is 8, not 5, because the coverage disk
— not the window neighbourhood — is the widest write. Padding by 8 also guarantees that
every window containing an occupied cell is fully in-arena, so no window read can be
truncated by the arena edge.

**Policy, exactly.**

```rust
const fn floor64(x: i32) -> i32 { x & !63 }   // two's complement; -1 -> -64, 63 -> 0

fn reserve_around(&mut self, c: HexCoord) -> Result<(), MoveError> {
    let (cq, cr) = (c.q as i32, c.r as i32);
    if self.rows != 0
        && cq - 8 >= self.origin_q && cq + 8 < self.origin_q + self.rows as i32
        && cr - 8 >= self.origin_r && cr + 8 < self.origin_r + 64 * self.row_words as i32
    { return Ok(()); }

    // 1. Union the requested cell and live-stone box, each padded by 8.
    //    The current arena extent is not an input.
    let (mut lo_q, mut hi_q, mut lo_r, mut hi_r) = (cq - 8, cq + 8, cr - 8, cr + 8);
    if let Some((sq0, sq1, sr0, sr1)) = self.stone_bounds() {
        lo_q = min(lo_q, sq0 - 8); hi_q = max(hi_q, sq1 + 8);
        lo_r = min(lo_r, sr0 - 8); hi_r = max(hi_r, sr1 + 8);
    }
    let need_rows  = (hi_q - lo_q + 1) as usize;
    let base_r     = floor64(lo_r);                       // origin_r is a multiple of 64
    let need_words = ((hi_r - base_r) as usize / 64) + 1; // exact, no slack

    // 2. Refuse before allocating if the smallest sufficient arena exceeds the ceiling.
    let least_rows  = max(MIN_ROWS,      need_rows);
    let least_words = max(MIN_ROW_WORDS, need_words);
    let least_cells = least_rows as u64 * least_words as u64 * 64;
    if least_cells > MAX_GRID_CELLS {
        return Err(MoveError::BoardExtentExceeded { cells: least_cells });
    }

    // 3. Size dimensions independently, capped at 4x their content requirement.
    //    Use a tighter shape when geometric growth would exceed the ceiling.
    let bump = |have, need, min_| max(min_, if need > have {
        max(2 * have, need).next_power_of_two()
    } else {
        min(have, need.next_power_of_two() * 4)
    });
    let mut new_rows  = bump(self.rows,      need_rows,  MIN_ROWS);
    let mut new_words = bump(self.row_words, need_words, MIN_ROW_WORDS);
    if !fits(new_rows, new_words) {
        new_words = max(MIN_ROW_WORDS, need_words.next_power_of_two());
        if !fits(least_rows, new_words) { new_words = least_words; }
        let budget = (MAX_GRID_CELLS / (64 * new_words as u64)) as usize;
        new_rows = max(min(new_rows, budget), least_rows);
    }

    // 4. Re-centre the required box. origin_r stays a multiple of 64.
    let new_origin_q = lo_q   - ((new_rows  - need_rows ) / 2) as i32;
    let new_origin_r = base_r - 64 * ((new_words - need_words) / 2) as i32;

    // 5. Allocate zeroed and copy the padded live-stone region.
    //    Row and word offsets are exact because both origins are multiples of 64.
    ...
}
```

Representation properties:

- `origin_r` only ever moves by multiples of 64, so **every row copy is a word-aligned
  `memcpy`; there is no bit-shifted copy.**
- `rows` may take any value.
- Growth performs three allocations and `3 * live rows` row copies. Doubling the short
  dimension gives amortized O(1) growth.
- **The shape is a function of the stones, never of the allocation history.** The
  required box, the refusal predicate, and therefore `BoardExtentExceeded` all read the
  live stone box. Two positions holding the same stones accept and refuse exactly the
  same placements regardless of arena growth history.
- **`undo` does not shrink the arena.** Allocation size may survive rewind, but semantic
  behavior does not. The next growth reduces any dimension above 4x the content need.
- Dimensions grow independently; a sufficient dimension remains unchanged subject to
  the 4x cap.
- Growth happens *after* every rule check and *before* every class-I mutation, so a
  `BoardExtentExceeded` leaves the position untouched and a successful growth is followed
  only by infallible work.

Containment follows from `base_r = floor64(lo_r) <= lo_r`, and
`base_r + 64 * need_words - 1 >= hi_r` by the definition of `need_words`. Step 4 shifts
the origin down by at most half the surplus `new_words - need_words`, so the top of the
box still lands at `base_r + 64 * (new_words - k) - 1` with `new_words - k >= need_words`
and containment survives it. The same argument in `q` uses `new_rows >= need_rows`. A
`debug_assert` re-checks containment after every growth.

Step 2 makes the ceiling history-independent. Any arena that already contains the
required box has `rows >= need_rows`
and, because `origin_r` is 64-aligned, `row_words >= need_words`, so its cell count is at
least `least_cells`. A position whose arena already contains the box therefore cannot be
holding an allocation the ceiling would have refused, and the early `contains_padded`
return cannot disagree with the predicate.

### 5.6 Representation limits

Memory scales with the **bounding box**; stones scale with **plies**. Legality permits a
stone 8 cells from the nearest stone. At `N` plies split between two directions, the box
can reach `~16 N^2` cells.

- Memory is `O(bbox)`, not `O(stones)`.
- A `Q` or `QR`
  window gather (§6.2) touches 11 rows, `row_words * 8` bytes apart. At `row_words = 1`
  that is two cache lines; at `row_words = 512`, each row occupies a separate page.
- `clone()` cost scales with the bounding box.
- A growth event may copy the current live region despite amortized O(1) growth.

The guards are `COORD_LIMIT` and `MAX_GRID_CELLS`, both surfaced as typed errors, both
documented as **representation limits, not rules**, and both distinguishable from rule
violations via `MoveError::is_rule_violation()`. `MAX_GRID_CELLS = 1 << 24` limits the
three planes to approximately 6.3 MB.

**The ceiling is on the area of the padded stone box, not on either span.** Because the
arena is shaped to that box (5.5), a game spreading along one axis is bounded by
`COORD_LIMIT` rather than `MAX_GRID_CELLS`; multidirectional spread is bounded by the
area ceiling. Tests cover both cases:

- Property 8 requires random legal games of a few hundred plies to return neither limit.
- Property 6b requires a maximal spreader to reach the area limit and verifies that
  `is_rule_violation()` is `false`, the position is bit-identical to before the call,
  `is_legal()` still reports the refused placement as legal, and play continues elsewhere.

A runner may impose a match-level bounding-box cap. The engine imposes no match rule
beyond its typed representation limits.

### 5.7 `PartialEq` and geometry independence

`Grid` implements **neither `PartialEq` nor `Hash`**. A derived `PartialEq` on `Position`
therefore cannot compile; `Position` uses the content-based implementation below.

`PartialEq for Position` compares `stones_by`, `phase`, `current`, `terminal`,
`zobrist()`, and then zips `self.stones()` with `other.stones()`. `stone_count()` is not
compared separately, because it is the sum of `stones_by`. It ignores `rows`,
`row_words`, `origin_q`, `origin_r`, and the `covered` plane — `covered` is a pure function
of the stones, so comparing it would be redundant, and comparing geometry would make
`apply; undo` unequal to a fresh replay.

Every semantic observable must be extent-independent. Legal enumeration reads
`covered && !occupied`; stone iteration uses canonical coordinate order. There is no
insertion-ordered `occupied: Vec<HexCoord>`; recency comes from the record.

`Debug` is exempt: `Position` derives it, so output includes rows, origins, and raw plane
words. Debug text is unstable diagnostic output and must not be parsed, persisted, or
compared as a semantic value.

### 5.8 Representation replacement boundary

The current representation is the flat grid specified above. Any replacement must
preserve every public contract, ordering, error, and invariant in this document. No
public item may expose a row, word, plane, or storage index.

---

## 6. Windows and win detection

Nothing about windows is stored. Everything below is derived from the two occupancy
planes on read, in constant time, with no allocation.

Window reporting (§6.2, §6.3) and win detection (§6.4) are independent computations over
the same board so C12 (§10.1) compares separate formulations.

### 6.1 Window identity

A window is six consecutive cells along one axis: `Window { start, axis }`, with
`cell(i) = start + axis.vector() * i`. The 18 windows through a cell `c` are indexed by
`(axis, offset)` with `offset = k` meaning **`c` sits at bit `k`**, so

```
window(c, axis, k) = Window { start: c - axis.vector() * k, axis }
```

and `window(c, axis, k).cell(k) == c`.

### 6.2 The per-axis line gather

Every window through `c` on one axis lies inside the 11 cells `c` stepped `-5..=+5` along
that axis, because slot `k` starts `k` cells back and ends `5 - k` cells forward. So the
18 masks are three gathers of 11 cells, 33 `Position::get` probes in all:

```rust
for axis in Axis::ALL {
    // line[i] is the owner of `coord` stepped `i - 5` along `axis`.
    let mut line = [None; 11];
    for i in 0..11 { line[i] = self.get(coord.step(axis, i as i16 - 5)); }
    // Bit m of slot k is the cell at offset m - k, which is line[m + 5 - k].
    ...
}
```

For `k, m` in `0..6` the index `m + 5 - k` lands in `0..11`, and at `m == k` it is 5 —
the queried cell itself, which is the check that the indexing is right.

`get` is total over every `(i16, i16)` pair, so the gather needs no bounds test and no
clamping: a cell the arena has never covered, or one outside the coordinate domain
entirely, reads as empty. That is what makes `windows_through` and `window` total over the
*arena*. Totality over the *coordinate domain* is a separate and narrower claim: see §4.4,
where `windows_through` may return a `Window` whose `start` is outside the domain and
therefore may not be fed back to `window`.

This gather serves `windows_through` alone. Win detection does not use it (§6.4).

### 6.3 Canonical slot order

Slot index is `axis.index() * 6 + offset`, giving 18 slots in the order

```
Q/0 Q/1 Q/2 Q/3 Q/4 Q/5  R/0 R/1 R/2 R/3 R/4 R/5  QR/0 QR/1 QR/2 QR/3 QR/4 QR/5
```

This order indexes `Position::windows_through`'s return array, and nothing else.

Win reporting does not use it: `Applied::wins` is a three-element array indexed by
`Axis::index()`, holding the run each axis contributed rather than a set of window slots.
A seven-cell run is reported once as `Win { len: 7 }`.

The canonical *legal-move* ordering of §9 is a different thing entirely and is untouched
by this: it orders actions, not windows, and `ACTION_ORDER_VERSION` still pins it.

### 6.4 Win detection

Win detection is a run scan from the placed cell, done inside `apply_raw` after `place`.
For each axis, walk one step at a time away from `c` in each direction, counting cells the
mover owns:

```rust
let mut wins: [Option<Win>; 3] = [None; 3];
for axis in Axis::ALL {
    let mut back = 0u8;
    let mut probe = c.step(axis, -1);
    while self.get(probe) == Some(mover) { back += 1; probe = probe.step(axis, -1); }

    let mut fwd = 0u8;
    let mut probe = c.step(axis, 1);
    while self.get(probe) == Some(mover) { fwd += 1; probe = probe.step(axis, 1); }

    let len = back + fwd + 1;
    if len as usize >= WINDOW_LEN {
        wins[axis.index()] = Some(Win { axis, start: c.step(axis, -(back as i16)), len });
    }
}
```

The position is won iff some axis is `Some`. Six *or more* in a row wins, with no overline
rule; the run scan reports the whole run, so seven in a row is `len == 7` rather than two
overlapping windows.

**Walk bounds.** `get` is total, so the loop needs no bounds test — it stops at
the first cell that is not the mover's, and a cell that *is* the mover's is an occupied
cell and therefore a valid coordinate. So the walk never steps off a coordinate
`HexCoord::step` would refuse, and each step is `±1`, well inside that function's `±8`
debug assertion.

**The length bound.** Before this placement no six-in-a-row existed — one would have ended
the game on the ply that made it — so the run reaching `c` from either side is at most 5.
Hence `back <= 5`, `fwd <= 5`, and `len <= 11`, which is why `u8` holds it and why
`c.step(axis, -(back as i16))` stays inside the `±8` step bound. A caller walking a `Win`'s
cells must step one at a time for the same reason: `start.step(axis, 10)` would trip that
assertion.

Only the mover's stones are examined; the winner is always the mover.

Two independent formulations are required: this scan on the
apply path, and the brute-force six-window scans in `audit()` (A7/A8) and in the Tier-C
debug assert C12 (§10.1), which read cell by cell through `Position::window`. They must
agree. The scan must not share the §6.2 line-gather implementation.

---

## 7. Delta, undo, and the undo floor

### 7.1 The structural law

> **Every field is restored by exactly one of four mechanisms, named in
> the code.**

| Class | Mechanism | Fields |
| --- | --- | --- |
| **I — involutive** | re-run a self-inverse operation | `occ` bits (set/clear), `hash_cells` (xor/xor), `stones_by` (+-1, and with it `stone_count()`) |
| **R — recomputed** | re-derive from the primary store, over the affected disk only | `covered` (disk-run OR forward, separable-dilation recompute backward — §5.4), `frontier_cells` (a before/after popcount of the derived frontier over the disk runs, on both halves) |
| **II — snapshot** | verbatim copy out of the delta | `phase`, `current` |
| **III — not restored** | unobservable *by construction*, not by privacy (`Debug` output excepted — §5.7): the growth policy sizes and refuses from the live stone box (§5.5), so a rewound arena behaves exactly like a freshly grown one and only its size differs | arena `rows`, `row_words`, `origin_q`, `origin_r`, allocation |

**Nothing is restored by re-derivation from the bookkeeping under test.** Class R derives
from occupancy, the class-I primary store. `covered` is checked against C2's `DISK8`
recount, the run-OR, and `audit`'s repaint. For `terminal`,
`advance` returns `Err(TerminalState)` before any mutation, so every successful apply had
`terminal == None`; `undo` assigns `None` unconditionally and stores nothing. That is a
theorem (P1 in §10.1) enforced by a debug assertion.

The ordering law is:

> **`advance` runs all fallible checks, then growth, then the entire class-I/R mutation
> *unconditionally*, and only then reads the win out of the freshly updated planes and
> branches. The mutation half never observes the win. `undo` restores class II first,
> then reverses classes I and R in exact reverse statement order.**

The winning branch must not skip the coverage update.

```rust
fn apply_raw(&mut self, action: Action) -> Result<(Applied, Undo), MoveError> {
    let c = action.coord();

    // 1. FALLIBLE RULE CHECKS. No mutation above the end of this block, so a
    //    rejected placement leaves the position bit-identical.
    if self.terminal.is_some() { return Err(MoveError::TerminalState); }
    match self.phase {
        TurnPhase::Opening => {
            if c != HexCoord::ORIGIN { return Err(MoveError::IllegalOpening); }
        }
        // The second stone takes the same checks as the first: reuse is already
        // `Occupied`.
        TurnPhase::FirstStone | TurnPhase::SecondStone => self.check_placement(c)?,
    }
    let player_before = self.current;
    let phase_before = self.phase;

    // 2. FALLIBLE GEOMETRY. Class III: reallocation only, no observable change.
    self.grid.reserve_around(c)?;

    // 3. THE MUTATION — classes I and R. Unconditional and infallible. Must not
    //    read the win.
    #[cfg(debug_assertions)] let audit = UndoAudit::capture(self);
    self.place(c, player_before);

    // 4. RULE BRANCH. Reads only what step 3 wrote.
    let wins = /* the per-axis run scan of §6.4 */;
    let outcome = if wins.iter().any(Option::is_some) {
        let o = Outcome { winner: player_before };
        self.terminal = Some(o);
        // Freeze phase and current on a win.
        Some(o)
    } else {
        let (p, ph) = advance_turn(phase_before, player_before);  // sole transition
        self.current = p;
        self.phase = ph;
        None
    };
    ...
}

/// The only phase transition. Private, called from exactly one site.
const fn advance_turn(before: TurnPhase, current: Player) -> (Player, TurnPhase);
//   Opening     -> (P1,            FirstStone)
//   FirstStone  -> (same player,   SecondStone)
//   SecondStone -> (other player,  FirstStone)
```

The transition is a function only of the phase and mover; the placed coordinate is not an
input. The call site passes `player_before`, which is also the delta's mover source.

`check_placement(c)` is, in order: the off-domain classification of §3.7 if
`!c.is_valid()`, `Occupied` if `get(c).is_some()`, `TooFarFromStones` if `c` is not
covered. In-domain checks are index-mapped table lookups with no geometry walk. The
classification's `DISK8` probe is reachable only for off-domain input and runs in `i32`.

The opening arm checks only `c != ORIGIN`, because `Opening` implies the board is empty,
so "occupied" is unreachable there. `IllegalOpening` is therefore the single reason an
opening can be refused, and §11 records the resulting precedence pairs as unreachable
rather than untested.

### 7.2 The delta type

```rust
/// Undo authority for one placement. Unforgeable, unclonable, consumed on use.
#[must_use]
#[derive(Debug)]
pub(crate) struct Undo {
    action: Action,           // class-I key for occupancy, cover, frontier, hash
    phase_before: TurnPhase,  // class II
    player_before: Player,    // class II, and *also the mover* — one source of truth
    #[cfg(debug_assertions)]
    audit: UndoAudit,
}

#[cfg(debug_assertions)]
#[derive(Debug)]
struct UndoAudit {
    zobrist_before: u64,
    zobrist_after: u64,      // the LIFO / wrong-position detector
    hash_cells_before: u64,
    frontier_before: u32,
    stones_before: u32,
    stones_by_before: u32,   // stone_count_for(mover)
}
```

`hash_cells_before` and `stones_by_before` are captured for C10 (§10.1), which checks
`hash_cells == hash_cells_before ^ cell_key(c, mover)`, which is exactly the statement a
wrong or duplicated cell key breaks.

`Undo` is not `Clone`, `Copy`, `Default`, `PartialEq`, or public. Its release
representation has no heap allocation or `Drop`.

The type has:

- **No separate `mover` field.** The mover is `player_before`, read from `current` before
  mutation.
- **No `terminal_before`.** Provably `None`; `undo` assigns `None`.
- **No `stones_before` / `frontier_before` in release.** Both are class I and move by
  exactly one on each side. They appear in the debug audit as **assertions, not
  assignments**.

```rust
fn undo_raw(&mut self, u: Undo) {
    #[cfg(debug_assertions)]
    debug_assert_eq!(self.zobrist(), u.audit.zobrist_after,
        "undo applied to the wrong position, or out of LIFO order");

    self.phase = u.phase_before;              // class II, before class I
    self.current = u.player_before;
    self.terminal = None;                     // theorem P1
    self.unplace(u.action.coord(), u.player_before);

    #[cfg(debug_assertions)] {
        debug_assert_eq!(self.zobrist(), u.audit.zobrist_before);
        debug_assert_eq!(self.grid.frontier_cells(), u.audit.frontier_before);
        debug_assert_eq!(self.stone_count(), u.audit.stones_before);
        self.debug_assert_turn_closed_form();
    }
}
```

The `zobrist_after` check at entry is the misuse detector: handing a delta from a
different `Position`, out of order, or twice is caught with 1-in-2^64 miss probability, in
debug, at O(1) cost.

### 7.3 The undo floor

**The undo stack is not in `Position`.** It lives in the borrow-scoped `Search` (§4.7),
where the borrow *is* the floor: the session cannot outlive it and cannot rewind past it,
with no extra machinery to enforce either.

The stack contains consumed reversal deltas, not a retained game record. A cloned
position carries neither.

```rust
pub struct Search<'p> {
    position: &'p mut Position,
    stack: Vec<Undo>,        // the ONLY undo authority in the crate
}
```

Five layers enforce the floor:

1. **`Undo` is `pub(crate)` and unforgeable.** There is no public `undo(token)` and no way
   for a consumer to obtain a token. The `Vec` makes undo count `<=` apply count.
2. **`Undo: !Clone`** — a delta cannot be duplicated and replayed twice. **`Vec` LIFO** —
   deltas cannot be reordered.
3. **`&'p mut Position`** — no other path may mutate the position while a session exists,
   so the stack cannot go stale and no second `Search` can alias it.
4. **`Drop` unwinds.** A position lent to a search is returned in its seeded state on
   every exit path, including `?` and panic. `commit()` is the explicit opt-out and moves
   the floor to the current depth.
5. **A mirror builds `Search` over its own `Position`.** It holds no deltas below the
   seed position.

The floor is `stack.is_empty()`. `Search::new` sets it, `commit` moves it, `unwind`
returns to it.

The runner calls `Position::advance`; both forward entry points use the same
`apply_raw`.

### 7.4 Incremental consistency obligations

The following obligations require independent checks in addition to round-trip tests:

- **H1 — `phase_after` is not a function of `phase_before` alone.** It is
  `if won { phase_before } else { advance_turn(phase_before, mover) }`.
  `advance_turn` is private and called from the `else` arm; mirrors call `advance`; the
  Zobrist turn key includes the terminal bit.
- **H2 — a terminal position can carry either placement phase.** (`Opening` cannot
  terminate — §10.3.) If the *second* stone
  wins, the frozen phase is `SecondStone`.
  **Every branch on `phase` must test `terminal` first.** `frontier_cells` is a geometric
  count and can remain nonzero in a terminal position. Keep it distinct from
  `legal_count()`: `frontier_cells` is private and geometric; `legal_count()` is public
  and rule-level (`0` when terminal, `1` in `Opening`, otherwise `frontier_cells`).
- **H3 — `Opening` legality is not in the `covered` plane.** At ply 0 the plane is all-zero
  and `frontier_cells == 0`. A `legal_count()` that reads the counter without the
  `Opening` arm reports zero legal moves at game start. Match on phase first.
- **H4 — undoing the winning ply must un-freeze.** An implementation that *inverts* the
  transition is ambiguous under freeze, where `phase_after == phase_before`.
  `phase` and `current` are class II.
- **H5 — the win check must not gate the mutation** (§7.1). If `apply` skipped the
  coverage OR on a winning ply, the plane is stale under the winning stone's disk, and
  the stale plane remains live state and `audit` fails.
- **H6 — a placement can win on more than one axis.** Two lines crossing at the placed
  cell fill two entries of `Applied::wins`. Nothing may assume exactly one. Seven in a
  row is one `Win` with `len == 7`.
- **H7 — snapshot-restore hides forward drift.** Any field both maintained forward and
  snapshot-restored backward cannot be checked by round trip. Therefore
  `frontier_cells`, `stone_count()`, and `stones_by` are class I with debug *assertions*
  rather than class II assignments. `phase` and `current`, which have no involutive form,
  are covered by the closed form of §10.2.
- **H8 — symmetric bugs are invisible to the entire round-trip machinery.** A wrong
  `DISK8` offset, a wrong offset in the §6.2 window gather, a wrong `cell_key` constant,
  or a growth copy that uses the same wrong index for read and write all apply and
  un-apply identically. `audit()`, the brute-force oracles, and the frozen golden vectors are the
  required independent detectors.
- **H9 — geometry leaking into observables.** The arena grows and never shrinks, so
  `apply; undo` leaves a larger arena than a fresh replay of the same prefix. §5.7 is the
  enforcement.
- **H10 — atomicity on error.** All fallible checks precede all mutation. Every error
  variant has an explicit atomicity test.

---

## 8. Zobrist

The following definition is normative.

```rust
/// splitmix64 finalizer. A bijection on `u64`; wrapping arithmetic only.
const fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^= x >> 31;
    x
}

/// Domain tag bit for cell keys. Disjoint from `TURN_DOMAIN`.
const CELL_DOMAIN: u64 = 1 << 16;
/// Domain tag bit for turn keys. Disjoint from `CELL_DOMAIN`.
const TURN_DOMAIN: u64 = 1 << 17;

/// Hash contribution of one stone.
///
/// Input layout: `q` in bits 48..64, `r` in bits 32..48, `CELL_DOMAIN` at bit 16,
/// owner at bit 0. Every other bit is zero, so no two `(q, r, player)` triples
/// share an input, and `mix64` is injective, so no two share a key.
const fn cell_key(c: HexCoord, p: Player) -> u64 {
    mix64(((c.q as u16 as u64) << 48)
        | ((c.r as u16 as u64) << 32)
        | CELL_DOMAIN
        | (p as u64))
}

/// Hash contribution of the turn state. `slot` is `0..12` (see `turn_slot`).
///
/// Input layout: `slot` in bits 1..5, `TURN_DOMAIN` at bit 17. Bit 16 is clear,
/// so no turn-key input can collide with a cell-key input.
const fn turn_key_of(slot: usize) -> u64 {
    mix64(TURN_DOMAIN | ((slot as u64) << 1))
}

/// 12 constants: 3 phase kinds x 2 players x 2 terminal states. Baked at compile
/// time. No startup generation, no RNG, no endianness dependence.
const TURN_KEY: [u64; 12] = {
    let mut t = [0u64; 12];
    let mut i = 0;
    while i < 12 { t[i] = turn_key_of(i); i += 1; }
    t
};

impl Position {
    /// `phase.kind_index() * 4 + current.index() * 2 + terminal.is_some() as usize`.
    const fn turn_slot(&self) -> usize;

    /// Full position hash. Derived on read; there is nothing to un-XOR on undo.
    pub fn zobrist(&self) -> u64 {
        self.hash_cells ^ TURN_KEY[self.turn_slot()]
    }
}
```

Only wrapping `u64` arithmetic over an explicitly packed key. No float, no pointer, no
`Hasher`, no RNG, no startup generation, no endianness dependence — identical across
builds, machines, and processes, which is the container-boundary requirement (A5).

Hash-state obligations:

- **Derived on read, not accumulated.** `hash_cells` holds only the stone contributions;
  the turn key is XOR-ed in at read time. Undo therefore restores the turn component
  by restoring `phase` and `current`.
- **The turn key covers `terminal`.**
- **The turn key covers the whole of the turn state.** `TurnPhase` is three unit variants
  (§3.2) and the mover is one bit, so `kind * 4 + mover * 2 + terminal` is a total,
  injective encoding of `(phase, current, terminal)`.

**Golden vectors are required.** Freeze in the repository: sixteen
`(q, r, player) -> u64` vectors spanning all four sign quadrants and both players, the
twelve `TURN_KEY` entries, and `Position::zobrist()` after each ply of a fixed 40-ply
game.

---

## 9. The canonical legal-move ordering

> **CANONICAL ORDERING v1.** Legal placements are yielded in strictly ascending
> `ActionId`, which is exactly ascending lexicographic `(q, r)` on the signed axial
> coordinate: `q` first, then `r`, both increasing.

The encoding that makes those two statements the same statement:

```rust
pub const fn from_coord(c: HexCoord) -> ActionId {
    ActionId((((c.q as u16 ^ 0x8000) as u32) << 16) | ((c.r as u16 ^ 0x8000) as u32))
}
pub const fn coord(self) -> HexCoord {
    HexCoord {
        q: ((self.0 >> 16) as u16 ^ 0x8000) as i16,
        r: ((self.0 & 0xFFFF) as u16 ^ 0x8000) as i16,
    }
}
```

XOR by `0x8000` is the order-preserving bias: it maps `i16::MIN..=i16::MAX` onto
`0..=u16::MAX` monotonically. So unsigned `u32` comparison of `ActionId` is signed
lexicographic comparison of `(q, r)`, and `HexCoord`'s derived `Ord` (field order `q` then
`r`) agrees with it by construction. Round-trip is exact for every `u32` and for every
`HexCoord`.

This ordering is unbounded and position-independent. **It imposes no region, crop, or
fixed-width mask**; it is a sort, not a table index. A cell's index within a position's
legal list remains position-dependent.

**Enumeration.** `LegalActions` walks the derived
frontier — each word composed as `covered & !occ` on read (§5.1) — in storage order:
rows ascending (row `i` is `q = origin_q + i`), and within a row,
words ascending, and within a word, bits ascending from bit 0 (bit `j` is
`r = origin_r + j`). Because the layout is `q`-major and `r`-minor (§5.1), storage order
*is* ascending `(q, r)`. Enumeration uses no sort, comparator, or allocation.

The three special cases, in the order they are tested:

1. `terminal.is_some()` — yield nothing. `len() == 0`.
2. `phase == Opening` — yield exactly `Action(HexCoord::ORIGIN)`. `len() == 1`. The
   frontier plane is not consulted; at ply 0 it is empty.
3. otherwise — the bit scan. `len() == frontier_cells`.

`SecondStone` needs no special case at all — it enumerates exactly as `FirstStone` does.
The turn's first stone is occupied, so its frontier bit is clear and it is already absent
from the scan.

Pinned by `ACTION_ORDER_VERSION` and by a golden test that hashes the full ordering
emitted at every ply of a fixed replayed game.

---

## 10. Invariants

Four tiers, by cost. Every invariant is marked with where it runs. Nothing in Tier C
allocates; nothing in Tier C is `O(arena)`.

### 10.1 Tier C — `debug_assert`, inside `apply_raw` and `undo_raw`

Every apply-side assert is O(1), O(18), or O(217); they run on every ply of every debug
build and property test. The undo-side coverage recount C2 is the exception at
`O(DISK_CELLS^2)` and supplies a from-scratch coverage check on every unwind.

| # | Invariant | Where |
| --- | --- | --- |
| C1 | After `place`: every in-domain cell of the placed disk reads covered. The necessary direction only; C2 on undo and tier A at checkpoints close the sufficient direction (no cell covered without a stone in range). | apply |
| C2 | After `unplace`: for every in-domain cell of the undone disk, the covered bit equals an independent stone recount — `DISK8` offsets probed against occupancy directly, sharing nothing with the run-OR of apply or the dilation of undo. | undo |
| C3 | `occ[0]` and `occ[1]` are never both set at the placed cell. | after `place` |
| C4 | `zobrist() == hash_cells ^ TURN_KEY[turn_slot()]`. | both |
| C5 | The turn closed form of §10.2. | both |
| C6 | `legal_count() == 0` iff `terminal.is_some()`. | both |
| C7 | **P1:** `terminal.is_none()` on entry to `apply_raw`. Structural, from the first check. | apply |
| C8 | The placed cell is at least `LEGAL_RADIUS` from every arena boundary. | after growth |
| C9 | The reserve-around containment check after every growth (§5.5). | growth |
| C10 | `stone_count() == before + 1`; `stones_by[mover] == before + 1`; `get(placed) == Some(mover)`; `hash_cells == before ^ cell_key(placed, mover)`. | apply |
| C11 | `outcome.is_some()` iff some entry of `wins` is `Some`; `outcome.winner == mover`. | apply |
| C12 | **The two win formulations agree, per axis:** `wins[axis.index()].is_some()` equals "some one of the six windows through `c` on that axis is full for the mover", read cell by cell through `Position::window`. Windows whose `start` is off-domain are skipped — such a window holds a cell no stone can occupy, so it is never full. | apply |
| C13 | On entry to `undo_raw`: `zobrist() == audit.zobrist_after` (LIFO / wrong-position detector). | undo |
| C14 | On exit from `undo_raw`: `zobrist()`, `frontier_cells`, and `stone_count()` equal their pre-apply values. **Asserted, never assigned** (H7). | undo |

C11 stops at the winner's identity. C5 independently pins the transition and freeze from
the stone count and terminal bit.

The frontier has no stored per-cell plane. C14 restores its counter exactly on undo, A5
recounts it in `audit`, and both placement halves measure it over the same domain-clipped
runs.

C12's two formulations must stay independent: the run scan walks
outward from the placed cell and never materialises a window, while C12 reads six named
windows cell by cell. Neither may use the other's helper.

C12 checks *whether* each axis won, not the run's `start` and `len`. Those are pinned in
Tier T instead, where `tests/fixtures.rs` states each expected run by hand and the smoke
test checks every reported run for ownership, containment of the placement, and
maximality at both ends.

### 10.2 The turn closed form (C5)

Let `n = stone_count()` and `m = n - (terminal.is_some() as u32)`.

```
n == 0                 <=>  phase == Opening && current == P0

m >= 1:
    phase kind    = if m is odd { FirstStone } else { SecondStone }
    current       = if (m - 1) / 2 is even { P1 } else { P0 }
```

Walk it: ply 0 gives `n = 1, m = 1` -> `FirstStone` / `P1`; ply 1 gives `m = 2` ->
`SecondStone` / `P1`; ply 2 gives `m = 3` -> `FirstStone` / `P0`; ply 3 gives `m = 4` ->
`SecondStone` / `P0`. That is the documented pattern: ply 0 is P0, plies 1-2 are P1, plies
3-4 are P0.

Freeze is covered by `m`, not by a special case. A win on the mover's first stone leaves
`n = k, terminal`, so `m = k - 1` and the form reports the *pre-move* phase and player,
which is exactly what freeze preserves. `m == 0` with `terminal` would mean a one-stone
win, which is impossible, so that case need not be handled.

The O(1) helper is called after every apply and undo and checks phase, player, and freeze.

### 10.3 Derived theorems

- **The frontier is never empty once a stone exists.** Take any stone on the convex hull
  of the occupied set; the cell one step outward from it is at distance 1, which is
  `<= LEGAL_RADIUS`, and is unoccupied. So `stones >= 1` implies `frontier_cells >= 1`,
  so zero legal moves implies terminal. C6 asserts the contrapositive.
- **A terminal `Opening` phase is impossible** — one stone cannot fill a six-window — so
  the `Opening` arm never has to reason about freeze.

### 10.4 Tier A — `Position::audit()`, and the order it checks in

`audit()` is `O(arena * DISK_CELLS)` in the worst case. Its recomputations take a
different *route* than the incremental path — coverage is repainted stone by stone into a
scratch plane where the increment edits one disk; the win scan reads windows cell by cell
where the increment walks runs. It shares the required definitions: `cell_key` is the
hash, so A6 uses the same constants. The frozen golden vectors verify those constants
(§8).

Checked in this order, returning the first failure:

| # | `IntegrityCheck` | Recomputation |
| --- | --- | --- |
| A1 | `DoubleOwned` | `occ[0] & occ[1] == 0` in every word. |
| A2 | `StoneCountForPlayer` | `stones_by[p]` equals the popcount of plane `p`. The total needs no check of its own: `stone_count()` is the sum of the two fields this pins. |
| A3 | `ArenaMargin` | every stone is at least `LEGAL_RADIUS` from every arena boundary. |
| A4 | `Coverage` | a scratch bit plane is repainted by iterating stones and marking every in-domain cell within `hex_distance <= LEGAL_RADIUS`, then compared word for word with `covered`. |
| A5 | `FrontierCount` | `frontier_cells` equals the popcount of the derived frontier words. |
| A6 | `Zobrist` | `hash_cells` equals the XOR of `cell_key(c, owner)` over `stones()`. |
| A7 | `Terminal` | `terminal.is_some()` iff some window is fully owned, by a brute-force scan over every stone, axis, and offset. |
| A8 | `Winner` | when terminal, the reported winner owns a completed window, and no window is completed by the other player. |
| A9 | `TurnClosedForm` | the closed form of §10.2. |

There is no stored-window-mask or move-history check because neither structure exists
(§5.1). There is no separate `StoneCount` check because A2 pins its source fields, and no
`FrontierBit` check because the frontier plane is derived on read.

### 10.5 Tier T — test-only oracles

These independent, `O(stones^2)`-or-worse oracles live only in `tests/`.

| # | Oracle |
| --- | --- |
| T1 | Legal set: the brute-force union of radius-8 disks over all occupied cells, minus occupied cells, minus cells outside the coordinate domain (§3.1's `is_valid`), minus (in `Opening`) everything but the origin, minus (when terminal) everything. Compared to `legal_actions()` as an ordered sequence at **every ply**. The domain clip is mandatory and independent of the arena row clip. |
| T2 | Zobrist: recomputed from scratch as `XOR cell_key(c, owner) ^ TURN_KEY[slot]`, compared at every ply and after every undo. |
| T3 | Win: a brute-force six-in-a-row scan over every stone, every axis, and every offset, compared to `is_terminal()` at every ply. |
| T4 | Turn sequence: the `(player, phase)` stream compared against the literal documented pattern `P0; P1 P1; P0 P0; P1 P1; ...` with freeze applied at the terminal ply. |
| T5 | Replay parity: for a random game and every prefix length `k`, a fresh `Position` advanced `k` times is `PartialEq` to a `Search` that applied `n` plies and then undid `n - k`. This states the exactness theorem against a construction path that shares no incremental bookkeeping with the comparison. |

---

## 11. Test obligations

`cargo xtask verify` must pass. Its gates are defined in `xtask/src/main.rs`, including
the release-profile lint. This section specifies the engine-specific test obligations.

**Unit tests, per module.**

- `coord`: `s()` and `hex_distance` totality at `i16` extremes; `DISK8` has exactly 217
  distinct offsets, all at distance `<= 8`, and its order is `dq`-major/`dr`-minor;
  `is_valid` boundaries.
- `action`: `ActionId` round-trips for every `HexCoord` in a grid sample and for a set of
  raw `u32`s including `0`, `u32::MAX`, and both bias boundaries; `ActionId` ordering,
  `HexCoord` ordering, and `Action` ordering agree over a set spanning all four sign
  quadrants.
- `window`: `Window::cell(k)` of `windows_through(c)[axis*6 + k]` equals `c`, for every
  slot whose `start` is inside the coordinate domain — all 18 of them in the interior.
  Separately, the mask round-trip against `Position::window` must include a face
  coordinate, where up to 15 slots start off-domain and are skipped under §4.4's
  contract rather than by the test avoiding the region. Also `WindowMask` accessor
  algebra (`occupied == mask(P0) | mask(P1)`, `empty == !occupied & 0x3F`).
- `grid`: growth from empty; growth in each of four directions; that `origin_r` stays a
  multiple of 64; that a grown arena reads back every previously written cell and reads
  zero everywhere new; `MAX_GRID_CELLS` refusal before allocation. And the coverage
  detectors, which are the tier-T half of §5.4's three-formulation argument: through a
  scripted mix of places, unplaces, and growths, the `covered` plane equals an
  independent `DISK8`-table recount at every step; a place followed by its unplace
  restores **every plane bit for bit**; `disk_runs` visits exactly the `DISK8` cells in
  exactly the `DISK8` order, and clips exactly the cells a per-cell `is_valid` excludes;
  and the maintained `frontier_cells` tracks a brute-force count of the derived plane.
- `zobrist`: the frozen golden vectors (§8) and the twelve `TURN_KEY` entries.
- `position`: one test per `MoveError` variant, and one per **ordered pair of
  simultaneously violated conditions that is actually reachable**, pinning the precedence
  table of §3.7. Three pairs in that table are *not* reachable and have no test, because
  no position can violate both halves at once, not because they were skipped:
  `TerminalState` over `IllegalOpening` (one stone cannot fill a six-window, so a terminal
  `Opening` is impossible — §10.3), `CoordOutOfBounds` over `Occupied` (a stone exists
  only on a valid cell), and `Occupied` over `TooFarFromStones` (an occupied cell is
  inside its own radius-8 disk, so it is covered). The reachable pairs — terminal over
  each of `Occupied` / `TooFarFromStones` / off-domain, `IllegalOpening` over an
  off-domain coordinate and over `TooFarFromStones`, and `TooFarFromStones` over
  `BoardExtentExceeded` — each get one. The off-domain **classification** of §3.7 gets
  two of its own: a far off-domain cell is `TooFarFromStones` and `is_rule_violation()`,
  and — in `tests/boundary.rs`, where a face is reachable — an off-domain cell beside a
  face stone is `CoordOutOfBounds` and not a rule violation. Separately, one test pins that the second
  stone of a turn is refused at the turn's first stone as `Occupied`, which is the whole
  of the reuse rule (§3.2). Also atomicity —
  the position is `PartialEq` to its clone after every rejected placement.
- `search`: floor behaviour — `undo()` at depth 0 returns `None` and leaves `zobrist()`
  unchanged; `commit()` then `unwind()` is a no-op; `Drop` restores the caller's position
  after an early `?`.
- `window`: additionally, `Win` — a run's cells are `start` stepped `0..len` along its
  axis, walked one step at a time and cross-checked against `hex_distance` from `start`.
- `window` geometry: `cell_index` inverts `cell` on every window of a corpus spanning all
  three axes; `contains`, `intersects`, and `touches` each agree with a **brute-force walk
  of the materialised `cells()` arrays** over that corpus. The closed-form index
  arithmetic and linear scan must remain independent. Additionally, `intersects` and `touches` are symmetric
  and never both true; same-axis windows overlap exactly when their starts are within six
  steps (a property the cell walk does not state); and the corpus is checked to contain all
  three of overlapping, touching, and disjoint pairs, so the partition claim is not vacuous.
- `position`, counting and replay: `stone_count` tracks the ply at every placement; `undo`
  restores it and names the placement it reversed; `replay` of a move list reproduces the
  position and of every prefix reproduces that ply; `ReplayError.ply` names the failing
  index for an illegal opening, an occupied cell, and a sequence continuing past a win; and
  two move orders reaching the same board are `PartialEq`. The move lists are held by the
  tests, as a record-keeper holds them — a position offers none.
- `position`, the ordering: `legal_rank` and `nth_legal` agree with the iterator at every
  index, `legal_rank(a).is_some() == is_legal(a)` over a neighbourhood, the opening ranks
  only the origin, a terminal position ranks nothing, and the second stone of a turn cannot
  rank the first.
- `grid`: additionally, `frontier_rank` and `nth_frontier` against a brute-force
  coordinate-by-coordinate walk of the arena, across `u64` word boundaries and over a full
  word; `None` off the plane and on an unallocated arena.

**Property tests (`proptest`, dev-dependency only).**

1. **Round trip.** Over random legal games: `apply` then `undo` restores a `PartialEq`
   state, and `audit()` passes after every apply *and* every undo.
2. **Legal-set oracle (T1)** at every ply.
3. **Zobrist oracle (T2)** at every ply, and exact restoration after undo.
4. **Win oracle (T3)** at every ply.
5. **Turn sequence (T4)** for whole games.
6. **Growth invariance**, in three parts.

   - **6a, geometry does not leak (H9).** Two positions are seeded identically, but one
     has its arena pre-grown far out by a `Search` that walks along a **single** axis and
     unwinds; `undo` never shrinks it, so the two arenas differ in `rows`, `row_words`,
     `origin_q`, `origin_r`, and allocation size while holding the same stones. Replaying
     an identical random game into both must then produce, at **every** ply, the same
     `Ok`/`Err` from `advance`, equal `Applied` values, equal `Position` (`PartialEq`),
     equal `zobrist()`, equal `legal_count()`, and byte-identical `legal_actions()`
     sequences. The pre-growth axis is part of the generated case.
   - **6b, spreading never panics.** A driver that plays the legal cell furthest along a
     rotating axis runs for ~140 plies with the oracles
     re-checked periodically and `audit()` at the end. The property asserts that a
     refusal, if one happens, is not a rule violation, is atomic, leaves the refused
     placement `is_legal`, and permits play elsewhere.
   - **6c, the budget is content, not history.** A position that is searched along a
     *single* axis and fully unwound, and a fresh replay of the same moves, are driven in
     lockstep down a spreading diagonal until `BoardExtentExceeded`. They must agree on
     every acceptance and on the refusal itself.

7. **Replay parity (T5)** for every prefix.
8. **Bounded random play.** Random legal games of up to a few hundred plies never return
   `CoordOutOfBounds` or `BoardExtentExceeded`; spreading behavior is covered by 6b.
9. **The canonical ordering is a bijection at every ply**, checked inside the same
   per-ply oracle pass as 2–5: `legal_rank` and `nth_legal` agree with the enumeration at
   every index, and `nth_legal(legal_count())` is `None`.
`Position` has no history-replay property (§4.5). `hexo-runner` owns the corresponding
record contract:
`the_prefix_replays_into_the_canonical_position` replays `Game::prefix()` into an equal
`Position`.

**Boundary tests (`tests/boundary.rs`).**

The coordinate domain is a hexagon with six faces, and `legal_actions`, `legal_count`,
`legal_rank`, `nth_legal`, `is_legal`, and `advance` must agree at all six faces. The four axis-aligned walks —
`(1,0)`, `(-1,0)`, `(0,1)`, `(0,-1)`, each ~2000 plies in `LEGAL_RADIUS` steps — reach all
six faces between them, because each drives two cube coordinates to their limits. At each:
every enumerated action is `is_valid` and `is_legal`; rank and select agree with the
enumeration over a sample weighted to the face; advancing an enumerated action never
yields `CoordOutOfBounds`; `audit()` passes; and an apply/undo taken *at* the face restores
exactly, which is what pins `place` and `unplace` to the same domain filter.

The two diagonal walks widen both arena dimensions, so the padded bounding box grows as an
area and `MAX_GRID_CELLS` refuses at
around `|q| = 4000`, long before `COORD_LIMIT` would. The test asserts that this is what
happens, that the refusal is a representation limit rather than a rule violation, and that
the position survives it intact.

**Fixtures.**

- A first-stone win: frozen at `FirstStone`; applied, audited, undone, audited, re-applied,
  hash-compared.
- A second-stone win: frozen at `SecondStone`, with the turn's first stone still on the
  board; same cycle.
- A seven-in-a-row, asserting one run of `len == 7` rather than two of six; and two
  crossing lines, asserting a run on each of two axes (H6).
- A win that completes a run the placed stone is *not* at the end of, so the walk is
  exercised in both directions at once rather than only forward or only backward.

Each fixture states its expected `[Option<Win>; 3]` **derived by hand from the move
list**, not captured from the implementation.

**Smoke test.** At least 10 000 full playouts, each to termination or a test-local ply
bound of 512, with no panics, `audit()` on the final position of each, and an assertion
that the terminal ply and the winner agree with T3. Every reported `Win` is checked over
its own cells: all `len` of them are the winner's, one is the placement, `len >= 6`, and
the cells one step off each end are not the winner's.

The default mix uses a line-building driver with swept noise plus a smaller pure-uniform
slice; both slices apply the same assertions. `HEXO_SMOKE_GAMES` and
`HEXO_SMOKE_UNIFORM` scale the slices and reject malformed values. `cargo xtask smoke`
runs the release-profile gate with `HEXO_SMOKE_GAMES=100000` and
`HEXO_SMOKE_UNIFORM=500`. Changes to hashing, ordering, growth, or win detection must run
that gate in addition to `verify`.

---

## 12. Excluded surface

The following items are not part of `hexo-engine`:

| Excluded item | Contract or owner |
| --- | --- |
| Public `Board` | `Position` is the sole constructible rule-bearing state. |
| Any `serde` implementation | Records are move lists owned by the runner. |
| `snapshot.rs`, `StateLoadError`, or position loading | Reconstruct with `Position::replay` through the rule machine. |
| `bounds()` | Arena geometry is private (§1). |
| Threat predicates (`is_threat_for`, `threat_player`, `is_active`) | Derive them from `WindowMask`. |
| `touched_windows()` | Derive it from `stones()` and `windows_through`. |
| Stored window-mask table | Window masks are derived in O(1) (§6.2). |
| Stored legal set (`Vec<ActionId>`, `AHashSet`) | The derived frontier is the legal set. |
| Insertion-ordered `occupied_cells()` | `stones()` uses canonical order; records own placement order. |
| Move history on `Position` | The record keeper owns history; `replay` reconstructs state (§13.16). |
| Raw occupancy planes or bitboard slices | Origins, strides, and arena geometry remain private. |
| `Position` as a trait | The specified methods are inherent. |
| Draw or non-win `Outcome` | The engine rules have no draw; the runner owns match results. |
| Engine ply cap | Ply caps are runner-owned match rules. |
| `row_any` summary bits | Enumeration uses `BitScan::next_slot` and `Grid::owner_at` directly. |
| Tiled arena | The current representation is the flat grid (§5.8). |
| Recursive-search RAII sub-guard | Recursive and iterative search use `&mut Search`. |
| History-sensitive engine hash | `zobrist()` is position-only; record-aware consumers own any history key. |
| `serde` for move lists | Record encoding belongs to the record writer; actions expose `ActionId`. |
| Runtime public `Undo` token validation | `Undo` is private; debug builds check `zobrist_after`. |
| PyO3 types, dictionary marshalling, or lazy action views | Bindings belong in a leaf crate that depends on this crate. |

---

## 13. Contract ledger

This ledger summarizes cross-cutting representation and ownership contracts.

1. **Coverage is one bit per cell and undo recomputes it.** Coverage is a pure function
   of stones and is recomputed by the separable zonogon dilation (§5.4).
2. **The frontier is derived on read, never stored.** `covered & !occ` defines membership
   and canonical enumeration; one `u32` stores the count.
3. **Window masks are derived from occupancy.** No stored mask table exists.
4. **Win detection is a per-axis run scan from the placed cell.** Debug checks compare it
   with an independent brute-force six-window scan, and `Win` reports the maximal run.
5. **`q`-major, `r`-minor bit layout.** It makes storage order identical to canonical
   `(q, r)` order.
6. **Undo stack in a borrow-scoped `Search`, with `Undo` `pub(crate)` and `!Clone`.**
   The API cannot undo past the floor, duplicate a delta, or use a delta on another
   position.
7. **Two forward entry points — `Position::advance` and `Search::apply` — over one
   internal `apply_raw`.** There is one rule implementation.
8. **`Position::advance` is irreversible; `Search::apply` is reversible.**
9. **All index and distance arithmetic in `i32`, `HexCoord::s() -> i32`,
   `hex_distance -> u32`.** These public functions are total over their documented
   domains.
10. **`COORD_LIMIT` and `MAX_GRID_CELLS` are typed representation limits, not rules.**
    `MoveError::is_rule_violation()` distinguishes them.
11. **`CoordOutOfBounds` sits below `TerminalState` and `IllegalOpening` in the precedence
     order.** The preceding checks require no board access or coordinate arithmetic
     (§3.7).
12. **Zobrist derived on read (`hash_cells ^ TURN_KEY[slot]`), with `terminal` in the turn
     key.** Restoring `phase` and `current` restores the turn contribution.
13. **`Position` does not implement `Hash`.** A derived one would fold in the arena
     geometry that `PartialEq` ignores; `zobrist()` is the key to use, and `Grid`
     implements neither trait so the derive cannot compile (§5.7).
14. **`Outcome` contains only `winner`.** Placement count is `stone_count()`.
15. **`audit()` is a normal public method, not a cargo feature.**
16. **Placement history belongs to the record-keeper, not to `Position`.** The board is a
     value; `replay` rebuilds it from a record. Move-order consumers read the record.
17. **One hash, and it stays position-only.** `zobrist()` covers stones, owners, mover,
     phase kind, and terminal status, not placement history. Consumers may mix a record
     into a separate process-internal key.
18. **`PartialEq` is positional.** The type is called `Position`; its equality matches
     `zobrist`, `audit`, and the oracles. Different move orders reaching the same position
     compare equal.
19. **Coverage is written only inside the coordinate domain (§5.4).** Legal enumeration,
     rank, selection, and `advance` therefore share the same domain.
20. **A win is reported as a run, not as a set of window slots.** `Applied::wins` is
     `[Option<Win>; 3]` indexed by axis. `Win { axis, start, len }` identifies the
     completed maximal line; no win is `None`.
21. **`TurnPhase::SecondStone` carries no coordinate and there is no `ReusedFirstStone`
     error.** Occupancy reports reuse as `Occupied` (§3.2).
22. **Player and model selection interfaces receive `&Game`.** The game supplies the
     position, record, and budget; `hexo-engine` defines no seat abstraction.
23. **Off-domain placements classify by the rules, not by the domain (§3.7).**
     `CoordOutOfBounds` is reserved for a placement the rules allow and the engine cannot
     represent — off-domain within `LEGAL_RADIUS` of a stone. Every other off-domain cell
     is `TooFarFromStones`, a rule violation.
24. **`MAX_GRID_CELLS = 1 << 24`.** At three bits per cell, the grid-plane ceiling is
     approximately 6.3 MB (§5.6).
