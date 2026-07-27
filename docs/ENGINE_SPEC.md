# hexo-engine — MVP specification

**Status: normative.** This is the single implementation target for `crates/hexo-engine`.
Where it conflicts with `README.md`'s "planned module map" or with `docs/SUGGESTIONS.md`,
this document wins. Where it conflicts with the audited rules, the rules win — but they
should not conflict, because every rule below was transcribed from the audit.

Everything here is a decision. There are no alternatives in this document by design; the
reasoning behind the contested ones is collected in §13 so the body stays unambiguous.

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

There is deliberately **no** `board.rs`, `rules.rs`, `legal.rs`, `windows.rs`, or
`snapshot.rs`. There is no `board` module because a public `Board` type is a second,
rule-free way to construct a position: whatever the type can be built from becomes a
construction path that never runs the turn rules. Occupancy hangs off `Position`, which
can only be advanced one legal placement at a time (§12). There is no `rules` module
because a free `is_legal_placement(&state, coord)` is a second entry point that invites
non-atomic check-then-place; legality is a private method plus the public
`Position::is_legal`. `legal` and `windows` are absent because neither the legal set nor
the window masks are stored — see §5 and §6.

`grid` is `mod grid;`, not `pub mod grid;`. Grid geometry is entirely private, enforced by
the module system rather than by discipline: **no public item in this crate may ever
expose a row, a word, a plane, a stride, or an index.** That constraint is what makes the
arena replaceable without touching a caller (§5.8).

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
/// A representation bound, not a rule. It is chosen so that every internal
/// coordinate walk (`+-8` for the disk, `+-5` for a window) stays inside `i16`
/// with room to spare, and so that no legal game ever reaches it: spreading is
/// capped at 8 cells per ply, so the first placement that could violate it is
/// ply ~2000.
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
/// Hard ceiling on dense arena cells. With three bit planes this caps a position
/// at ~6 MiB, and a one-seat spreading walk cannot reach it inside a real game.
///
/// A representation limit, not a rule: a placement that would push the arena past
/// this is legal, and the engine reports that it cannot represent it. It bounds the
/// AREA of the padded stone box, so a walk along one axis is bounded by COORD_LIMIT
/// instead; unreachable without a deliberate ~thousand-ply walk that spreads in
/// every direction at once (§5.6).
pub const MAX_GRID_CELLS: u64 = 1 << 24;
```

Two version constants, not three. The Zobrist function rides inside `RULES_VERSION`
because a hash change and a rule change invalidate the same artefacts, and a third
constant is a third thing to forget to bump.

**Dependencies.** `[dependencies]` is empty and stays empty. `[dev-dependencies]` is
`proptest = "1"` and `criterion` (default features off, `cargo_bench_support` only),
which back the property suite and the benchmarks respectively; neither is reachable from
a non-test build. The workspace takes no proc-macro error crate: `MoveError` implements
`core::fmt::Display` and `core::error::Error` by hand, because six variants do not pay
for `thiserror`. `core::error::Error` is stable from 1.81, well under the workspace floor
of 1.88 (`Position::audit`'s winner check uses a let-chain, which is what actually sets
that floor).

The crate uses `std` only in `#[cfg(test)]` code today, but it is **not** `no_std`: there
is no `#![no_std]` attribute and no gate enforces one. Do not describe it as `no_std +
alloc` compatible — that would be a promise nothing checks. The `wasm32` gate does not
substitute for one either; that target ships a full `std`, so a `std::time` or threading
call compiles there and fails at run time. What the gate does insure is the dependency
graph: a PyO3 or other native-only dependency creeping into this crate fails the build.

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

`Ord` on `HexCoord` is load-bearing: it is lexicographic `(q, r)`, which is the canonical
ordering of §9, and it must agree with `ActionId` ordering by construction.

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

**`SecondStone` carries no payload, because occupancy already implies the rule it would
serve.** "The second stone of a turn may not reuse the first" needs no state: the first
stone occupies its cell and stones are permanent, so in every reachable position the reuse
placement is refused as `Occupied`. A stored coordinate could therefore only change *which*
error variant reported the refusal, and it made `PartialEq` split positions that the Zobrist
hash and the game dynamics treat as one (§13).

A consumer that wants a "played this turn" plane reads the last entry of the game record it
already keeps, which is the same coordinate.

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

`ActionId` is a newtype, never `pub type PackedCoord = u32`: a bare alias makes a raw
index and an action id the same type to the compiler, so a dense policy index and a board
coordinate become silently interchangeable — and a crop or a re-indexing then goes
undetected because nothing is mistyped. `Action`'s field is private so a move is
constructed only through the coordinate or id paths.

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

`Win` lives here rather than with `Applied`, which is its only producer: it is a
statement about line geometry, in the same category as `Window`.

`Win` has no `Default` and no empty value. "No win" is `Option<Win>`, so a placement that
did not win cannot be confused with one that won a run of length zero.

`WindowMask`'s inner `[u8; 2]` is private: the player-to-lane mapping is an internal
convention, and `mask(Player::P1)` is the contract.

There is no `is_win_for`, no `threat_player`, no `is_active`, no
`stone_cells() -> Vec<_>`. A mask is strictly more information than any predicate over it,
and every one of those is a one-liner over `mask()` and `Window::cells()`; the
`Vec`-returning form is worse than redundant, because it allocates inside a search loop.

`contains`, `intersects`, and `touches` are the exception that proves where the line is,
and they sit on `Window` rather than on anything carrying ownership. The rule is about
*ownership*: a predicate that collapses a mask is strictly less than the mask. These
collapse no mask — they are statements about six coordinates on the infinite board, in the
same category as `cell` and `cells`, and they read no `Position` at all. They exist
because a cell/window incidence graph needs them: window nodes need identity arithmetic
and no board access. `cell_index` is the primitive and `contains` is its `is_some()`,
because an incidence edge has to carry *which* bit of the mask a cell is; the mask is
positional, so "somewhere in this window" cannot be read back from it.
`intersects_or_touches` stays out: it is a union of two answers, not a third fact.
`intersects` is six `contains` calls rather than a closed form over the parallel and
crossing cases, because two branches of a case analysis that can be wrong in the same way
is precisely the symmetric-bug hazard, and the price of avoiding it here is six cheap
tests.

### 3.5 `position`

```rust
/// A Hexo position: board, turn phase, mover, hash, terminal status.
///
/// A value type. It carries no placement sequence and no undo stack (§7): a game is
/// rebuilt by `replay` from a record its keeper holds, which for a match is
/// `hexo_runner::Game::plies`.
///
/// `PartialEq` is content-based and deliberately ignores arena geometry: two
/// positions with the same stones, phase, mover, and terminal status are equal even
/// if one's arena grew larger getting there. Equality means *same position*,
/// matching `zobrist`, the oracles, and `audit`. It is `O(arena extent)`.
///
/// This type deliberately does **not** implement `core::hash::Hash`. Use
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

`Outcome` carries the winner and nothing else. The obvious second field — how many
placements the game took — is `Position::stone_count()`, which is already exact and
already restored by undo; a second copy is a field pair that can disagree.

`wins` is a plain array with no accessor methods, because the array *is* the answer:
`wins.iter().flatten()` is every completed run and `wins[axis.index()]` is the one on a
named axis. `Win` is declared in §3.4.

There is at most one run per axis by construction, not by choice: a placement lies on
exactly one line of each axis, and the run reported is that line's maximal run through it.

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
/// The error type of `replay` and `replay_from` (§4.2). `ply` is load-bearing: a
/// rejected sequence has to say *where* it stopped, or a caller cannot tell a
/// corrupt record from one truncated mid-turn.
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

There is no `StateLoadError`: there is no loader (§12).

**Error precedence is observable and pinned by tests**, in this exact order:

```
TerminalState
  -> IllegalOpening
  -> CoordOutOfBounds
  -> Occupied
  -> TooFarFromStones
  -> BoardExtentExceeded
```

The order is not arbitrary and is not free to change: it is observable through `advance`,
so a caller that branches on the variant sees a different answer if it moves. The first
two are the checks that need no board access at all — a terminal flag and one equality test
against a coordinate already known valid (`ORIGIN`). The domain test comes next because
everything after it does coordinate arithmetic in `i16` and must not be handed a
coordinate that could overflow it. `Occupied` precedes `TooFarFromStones` so that the more
specific reason wins when both hold. `BoardExtentExceeded` is last because growth runs
after every rule check (§7.1) — a placement is refused for breaking a rule before it is
refused for being unrepresentable.

**An off-domain coordinate classifies by the rules, not by the domain.** The domain test
does not report `CoordOutOfBounds` unconditionally: the rules do not know the domain
exists, so the refusal states what the rules would say. Off-domain and within
`LEGAL_RADIUS` of a stone, the placement is rule-legal and merely unrepresentable —
`CoordOutOfBounds`, an engine limit. Off-domain anywhere else it is plain
`TooFarFromStones`, a rule violation. (Off-domain `Occupied` cannot arise: stones exist
only on valid cells.) The distinction is load-bearing for adjudication: a runner treats a
rule violation as the seat's forfeit and an engine limit as no-contest, so before this
classification a losing seat could submit `(20000, 0)` and have the match voided rather
than lose it. The classification probe is a cold path — a `DISK8` walk over the
neighbours, in `i32` so nothing wraps — reached only for off-domain input; the hot path
never runs it (§7.1).

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
    /// The only mutating method on `Position`. Used by the runner, which owns the
    /// canonical state and never rewinds it, and by a player's mirror consuming
    /// the move stream.
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
    /// The catching-up path: a mirror consuming buffered moves, or a seat joining
    /// from the move prefix in its handshake.
    ///
    /// # Errors
    /// As `replay`, with `ply` relative to `actions`. **Not atomic:** placements
    /// before the failure are not rolled back.
    pub fn replay_from(&mut self, actions: &[Action]) -> Result<(), ReplayError>;
}
```

**Replay is the only way to load a position, and it is not a second code path.** Every
placement goes through `advance`, so a position is expressible exactly when it is
reachable by a legal game — which is decision A3, expressed as an API instead of a
convention. There is no `serde` impl on `Position` and no board-shaped deserialisation:
any deserialiser that accepts a bare cell list reconstructs a position without ever
running the turn rules that could have produced it, which makes every invariant in §10 a
statement about how the position was *built* rather than about the type.

`ReplayError::ply` is the load-bearing field. A record that fails to replay is
untriageable without knowing *where* it diverged, and "illegal move" alone does not let a
caller bisect a corrupt game.

A mirror must call `advance` and must never re-derive the phase transition itself. The
transition is `if won { phase_before } else { advance_turn(phase_before, mover) }`, and
anything that mirrors it without the win check diverges on exactly one ply per game — the
last one.

`advance` builds the same `Undo` that `Search::apply` does and drops it. That is free:
`Undo` is a ~12-byte POD with no heap and no `Drop`, which is what lets there be exactly
one forward code path.

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
    /// Returns `bool`, not `Result`. The error variant is what `advance` produces;
    /// a rich-error predicate tempts callers into check-then-apply, which
    /// double-validates on the hot path. To learn *why*, call `advance` and read
    /// the error.
    pub fn is_legal(&self, action: Action) -> bool;

    /// Where `action` sits in `legal_actions()` order, or `None` if it is not
    /// legal here.
    pub fn legal_rank(&self, action: Action) -> Option<usize>;

    /// The legal placement at `index` in `legal_actions()` order, or `None` if
    /// `index >= legal_count()`.
    pub fn nth_legal(&self, index: usize) -> Option<Action>;
}
```

**`legal_rank` and `nth_legal` are the two directions of the canonical ordering, and they
are in the engine for one reason.** A policy head is indexed by that ordering, so
self-play, training, and serving must all use the *same* mapping. Deriving it separately
in each makes them agree only by coincidence, and a divergence is silent — the network
keeps training, against scrambled targets. The fix is that the ordering has exactly one
implementation and a golden test over the rank of each played move, **not** that the action
space is bounded — a bounded one has already been tried and collapsed a training run. That
argument is recorded in full in `crates/hexo-engine/README.md`.

Both are a popcount prefix and a select scan over the derived frontier — each word is
`covered & !occ0 & !occ1`, composed on read (§5.1) — which the `q`-major/`r`-minor layout
makes equal to position within the canonical enumeration. Each touches only the words up
to its target, so the cost is `O(word index)` — it *does* grow with the rank, roughly
1.6 ns per 64-cell word against 3.0 ns per item for a walk of `legal_actions`.

Measured at 256 stones (7,349 legal): `legal_rank` is 5.6 ns at rank 0, 106 ns at the
middle, 195 ns at the last, against 5.4 ns / 11.0 us / 22.2 us for the walk — **34x to
129x**. The two cross over only at rank ~0. Quadrupling the arena words behind the same
legal set slows the prefix by 97% and the walk by 2%, and the prefix still wins by 67x;
the real crossover is a frontier density below about 0.13 legal cells per word, against
3.3 to 14.4 measured. (Measured against the stored frontier plane; the derived read is
two extra AND-NOTs per word and does not change the shape of the argument.) The boundary
tests sample rather than sweep because a ~16,000-row arena makes the *walk* expensive,
not the prefix.

`legal_actions` never special-cases at the call site: callers do not branch on phase to
generate moves. A caller reusing a buffer writes
`out.clear(); out.extend(pos.legal_actions());` — the same allocation behaviour as a
`write_into` method, with no engine API for it. `ExactSizeIterator` means `collect`
pre-sizes exactly.

One iterator covers every shape a caller wants: coordinates, action ids, and either of
them written into a caller's buffer. `.map(Action::id)` is the caller's business, not a
second accessor.

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

`windows_through` is what a model encoder actually wants: hand it a candidate cell, get
the eighteen six-bit ownership patterns a stone there would interact with. That is
strictly more than any threat predicate could report, computed from data already present.

**The window domain is narrower than the coordinate domain, deliberately.** Within five
steps of a `COORD_LIMIT` face, up to ten of the eighteen slots `windows_through` returns
have `!start.is_valid()`, and up to fifteen at a corner where two of the three cube
bounds are tight, such as `(-COORD_LIMIT, 0)`. Their masks are correct, but `Position::window` and
`Window::cells` assert a valid start in debug, so round-tripping one panics there and
works in release. That is the contract: a caller re-querying a slot must skip invalid
starts.

Making the read path total over all of `i16` was rejected — it requires dropping the
`is_valid` assertion from `HexCoord::step`, a live detector on the placement paths, where
walking off the domain is a real bug rather than an expected edge. A second,
face-tolerant accessor is likewise out: two accessors for one question is a dual path.

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

**There is no `history()`.** A position does not know the sequence that produced it; the
record-keeper does, and hands it back through `replay`. A model whose features depend on
move order reads the record — `hexo_runner::Game::plies` or `Game::prefix` — which is the
one authoritative history and is what a seat is handed. Move *identity* in a record is
`ActionId`, which is frozen.

Deliberately absent: `bounds()`, which answers a question no caller has and leaks arena
geometry to do it (§1); `occupied_cells() -> &[HexCoord]` in placement order, because
placement order is not a property of a position at all; and any raw bitboard or plane
accessor (§12).

### 4.6 Integrity audit

```rust
impl Position {
    /// Recompute every derived structure from the stones alone and compare.
    ///
    /// A normal method, not a `cfg` or a cargo feature: feature unification makes
    /// `cfg`-gated correctness machinery unreliable, and the runner may want a
    /// paranoid mode. `O(arena extent * DISK_CELLS)` — for tests, fuzzing, and
    /// deliberate paranoia, not for a search loop.
    ///
    /// Its recomputation takes a different route than the incremental path — a
    /// stone-by-stone repaint where the increment edits one disk, a cell-by-cell
    /// window scan where the increment walks runs. It necessarily shares the
    /// *definitions*: `cell_key` is the hash, so a wrong constant inside it is
    /// invisible here by construction. That class belongs to the frozen golden
    /// vectors (§8), which is why they are not optional.
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

`undo` returns `Option` rather than panicking, which makes it total in the mathematical
sense as well as the "restores everything" sense: it is defined on every state of the
session, and it is the identity below the floor.

A recursive alpha-beta passes `&mut Search` down the call chain; an iterative MCTS holds
one `Search` and walks it. Both are covered without a second type.

`unwind` — and therefore `Drop` — performs no fallible work and cannot panic in release.
In debug it runs the undo assertions of §10.1; a failure during unwind-on-panic aborts,
which is the correct outcome for a corrupted position.

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

**There is no legal-move set, no window-mask store, and no stored frontier.** The obvious
design keeps each as a separate incremental structure, and each then carries its own
delta, its own growth handling, and its own class of invariants — copies of the board
that can disagree. Here:

- The legal set is the **derived frontier**: word `i` is
  `covered[i] & !occ[0][i] & !occ[1][i]`, composed on read and never stored. Membership
  is one bit probe; enumeration is a bit scan that produces canonical order for free
  (§9); the count is the one maintained `u32`. No `Vec<ActionId>`, no sorted insert, no
  hash set, no delta — and no stored plane whose bits could disagree with their own
  definition, because the definition is the read.
- The window masks are **derived on read** from the occupancy planes by an O(1) cell
  gather (§6.2), and win detection is an independent run scan (§6.4). Nothing is
  stored, so there is nothing to grow, nothing to undo, and no "stored mask disagrees
  with the board" bug class. What the *public surface* exposes is masks, not predicates
  over them (§3.4), and it is exactly as specified in §4.4.

Cell `(q, r)` maps to row `i = q - origin_q` and bit `j = r - origin_r`. **All index
arithmetic is performed in `i32`**, so a coordinate anywhere in the `i16` range produces
an in-range-or-out-of-range answer and never wraps. Out-of-range reads return zero; the
internal write path never sees one, because growth (§5.5) runs first.

Bit-per-cell layout is `q`-major, `r`-minor because that makes a row scan produce
ascending `(q, r)` — the canonical order — with no sort.

**No placement sequence is stored.** A game's move list belongs to whoever keeps the
record — for a match, `hexo_runner::Game::plies` — and a position is rebuilt from it by
`Position::replay` (§4.2). Storing a second copy on the board made every consumer that
already held a record carry two, and made a value type answer a question about a *game*.
Everything on the type is now either the board or derived from it, which is what makes
`PartialEq` (§5.7) a total statement about the value rather than one that ignores a field.

`stone_count()` is `stones_by[0] + stones_by[1]`, not a third field: a redundant field is
one that can disagree.

Memory is 3 bits per cell in the bounding box — two occupancy planes and `covered`, one
bit each. The first opened position allocates a 32x128 arena, which is 1,536 bytes of
plane payload; a realistic 200-stone game commonly pads and rounds to a 128x256 arena,
~12 KB. **`clone` performs three allocations**, one per grid plane, and copies their
contents. "Clone is a memcpy" is shorthand for "no pointer chasing and no per-cell work",
not a literal single `memcpy`, because `Grid` owns three separate buffers. Measured at
256 stones, a clone is ~175 ns — the cost a search mirror pays per position, and the
number the byte-per-cell design this replaced paid 4.5x more for.

### 5.2 Why `covered` is one bit, and undo a recomputation

`covered[c]` states that some stone lies within `LEGAL_RADIUS` of `c` — the occupancy
dilated by the radius-8 disk. It is a **pure function of the stone set**, and that is the
whole design: a structure that is a function of the stones needs no inverse operation,
because undo can *recompute* it from the stones that remain.

An OR of radius-8 disks cannot be un-ORed — removing one stone cannot clear cells another
stone also covers — and the previous design answered that with a byte-per-cell refcount
(`+1` on apply, `-1` on undo), which bought exact invertibility at 8x the memory and a
217-byte read-modify-write on every placement half. The recomputation answer deletes the
refcount instead: apply ORs the placed disk in word-wide runs (§5.4), and undo recomputes
the removed stone's disk from occupancy alone, which is possible locally because removing
the stone at `c` changes coverage only inside `c`'s disk, and every stone covering a cell
of that disk lies within `2 * LEGAL_RADIUS` of `c`. The recomputation is the separable
dilation of §5.4 — a few hundred word operations over a 33x33 window, not a per-cell
geometry walk.

The frontier then needs no storage at all:

> **The frontier invariant.** `c` is a frontier cell if and only if
> `covered[c] == 1 && occ[0][c] == 0 && occ[1][c] == 0` — and the engine *evaluates* this
> definition on read rather than maintaining a plane that must be held equal to it.

The legality rule falls straight out of it: a non-opening placement at `c` is legal iff
`c` is covered and empty. That is a pair of total table lookups with **no coordinate
arithmetic beyond the i32 index map**, which is what lets a wild in-domain coordinate
from an untrusted player be rejected without ever walking geometry. (A wild *off-domain*
coordinate takes the cold classification path of §3.7 instead.)

### 5.3 Empty position

`Position::new()` allocates nothing: `rows = 0`, `row_words = 0`, all three `Vec`s empty.
Every read is out-of-range and answers empty. The first `advance` grows the arena to the
initial geometry before touching anything:

```
MIN_ROWS      = 32     // q in [-15, 16] initially
MIN_ROW_WORDS = 2      // r in [-64, 63] initially
```

which contains the radius-8 disk around the origin with margin. (The `q` span is
`[-15, 16]`, not the `[-16, 15]` an earlier draft wrote: step 4 of §5.5 recentres by
`lo_q - (new_rows - need_rows) / 2 = -8 - (32 - 17) / 2 = -15`, and the truncating
division puts the odd cell on the high side. Nothing depends on which side it lands;
the value is pinned by a unit test so it cannot drift silently.)

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

**The counter delta needs no case analysis.** Either half changes the derived frontier
only inside the placed cell's disk: occupancy changes at `c`, which is a member of its own
disk, and coverage changes nowhere outside it. So popcounting the derived frontier over
the disk runs before and after the mutation *is* the whole of `frontier_cells`
maintenance. The involutive pair this replaced maintained a stored frontier bit by bit and
therefore needed an exact statement-order mirror — occupancy before the disk update on the
way in, after it on the way out, with a one-cell corruption waiting in any reordering.
Measuring instead of mirroring deletes that hazard along with the plane; there is no
ordering constraint left between the occupancy write and the coverage write.

**The disk is walked as 17 contiguous row runs, inside `Grid`.** `DISK8` is `dq`-major
and `dr`-minor, so each `dq` is one run of consecutive cells in storage order — one or two
words of each plane. `Grid::disk_runs` produces those `(start, len)` runs, and every
disk-shaped read and write — the coverage OR, the recomputation's writeback, the frontier
popcounts — goes through them. The run walk visits exactly the `DISK8` cells in exactly
the `DISK8` order, which leaves `DISK8` as a genuinely **independent** statement of the
same cell set rather than the one the machine follows: `grid`'s tests compare the two
formulations directly (membership and order), and the tier-C coverage recount (C2) walks
the table offset by offset on every undo. A wrong row run and a wrong offset are both
symmetric bugs, so neither can be checked against itself.

**Coverage is written only inside the coordinate domain.** If the disk update painted an
off-domain cell covered, `legal_actions` would offer a placement `advance` refuses — and
`legal_rank` would give it a policy index. That was a real defect: a walk to `q = 16000`
put 136 unaddressable coordinates into the legal set, all of which `is_legal` rejected.
The clip lives in `disk_runs` and is applied once per row rather than once per cell, and
it is **exact rather than a fast path**: at a fixed `q`, `is_valid` reduces to `r` lying
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

Why a shear or span error here cannot survive: the dilation is a third formulation of
coverage, independent of both the `DISK8` offset table and the run-OR of apply. `grid`'s
tests hold all three equal through place/unplace/growth scripts (§11), C2 recounts the
disk against `DISK8` on every debug undo, and `audit` repaints coverage from the stones
at every test checkpoint. Wrong in one formulation is a symmetric bug; wrong in three
independent ones identically is not a plausible accident.

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

    // 1. The required box: the requested cell padded by 8, unioned with the LIVE STONE BOX
    //    padded by 8. Content only. The current arena extent is deliberately not an input:
    //    if it were, the shape - and with it the ceiling - would depend on how the position
    //    was reached rather than on what is on the board.
    let (mut lo_q, mut hi_q, mut lo_r, mut hi_r) = (cq - 8, cq + 8, cr - 8, cr + 8);
    if let Some((sq0, sq1, sr0, sr1)) = self.stone_bounds() {
        lo_q = min(lo_q, sq0 - 8); hi_q = max(hi_q, sq1 + 8);
        lo_r = min(lo_r, sr0 - 8); hi_r = max(hi_r, sr1 + 8);
    }
    let need_rows  = (hi_q - lo_q + 1) as usize;
    let base_r     = floor64(lo_r);                       // origin_r is a multiple of 64
    let need_words = ((hi_r - base_r) as usize / 64) + 1; // exact, no slack

    // 2. Refuse before allocating, on the SMALLEST arena that could hold the required box.
    let least_rows  = max(MIN_ROWS,      need_rows);
    let least_words = max(MIN_ROW_WORDS, need_words);
    let least_cells = least_rows as u64 * least_words as u64 * 64;
    if least_cells > MAX_GRID_CELLS {
        return Err(MoveError::BoardExtentExceeded { cells: least_cells });
    }

    // 3. Size each dimension INDEPENDENTLY. A dimension that is short doubles; one that is
    //    not is left alone, but never at more than 4x what the content needs. Fall back to
    //    a tighter shape when the geometric one would break the ceiling.
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

    // 5. Allocate zeroed, then copy the LIVE REGION - the padded stone box, outside which
    //    every plane is zero. Copying that rather than the old allocation is what lets the
    //    arena be re-shaped, not merely extended, so step 3 can hand a dimension back
    //    capacity the content no longer needs.
    //    Row and word offsets are exact because both origins are multiples of 64.
    ...
}
```

Consequences, all of them intended:

- `origin_r` only ever moves by multiples of 64, so **every row copy is a word-aligned
  `memcpy`. There is never a bit-shifted copy.** That eliminates the entire class of
  off-by-one-bit growth bugs, which are symmetric and therefore invisible to round-trip
  tests.
- `rows` may take any value; row growth is free.
- Cost is three allocations — one per plane — plus `3 * live rows` `memcpy`s. Doubling the short dimension
  makes it amortised O(1).
- **The shape is a function of the stones, never of the allocation history.** The
  required box, the refusal predicate, and therefore `BoardExtentExceeded` all read the
  live stone box. Two positions holding the same stones accept and refuse exactly the
  same placements however much either arena grew getting there: a `Search` that applies a
  wide line and unwinds it does not consume the position's extent budget.
- **`undo` still does not shrink the arena**, and it does not have to. Only the size of
  the allocation survives the rewind, never a behavioural difference - which is what
  makes the geometry genuinely unobservable rather than merely private, and it is still
  why `PartialEq` must be logical rather than a byte comparison. The next growth hands
  back any dimension holding more than 4x what the content needs, so a searched mirror
  does not stay bloated either.
- **Only one dimension grows at a time.** Growing both on every event quadruples the
  arena per growth and freezes its aspect ratio at the initial `32 : 128`, which makes a
  straight walk along `q` - a perfectly legal game - refuse at ply 65, and breaks the
  `(q, r) -> (r, q)` symmetry of the rules by refusing the `q` walk 4.5x sooner than the
  identical `r` walk. This was shipped and fixed; the tests pin it.
- Growth happens *after* every rule check and *before* every class-I mutation, so a
  `BoardExtentExceeded` leaves the position untouched and a successful growth is followed
  only by infallible work.

Containment is guaranteed by construction, and it is worth checking the `r` arithmetic
once here so the implementer does not have to: `base_r = floor64(lo_r) <= lo_r`, and
`base_r + 64 * need_words - 1 >= hi_r` by the definition of `need_words`. Step 4 shifts
the origin down by at most half the surplus `new_words - need_words`, so the top of the
box still lands at `base_r + 64 * (new_words - k) - 1` with `new_words - k >= need_words`
and containment survives it. The same argument in `q` uses `new_rows >= need_rows`. A
`debug_assert` re-checks containment after every growth anyway.

Step 2's predicate is what makes the ceiling history-independent, and it is worth stating
the argument: any arena that already contains the required box has `rows >= need_rows`
and, because `origin_r` is 64-aligned, `row_words >= need_words`, so its cell count is at
least `least_cells`. A position whose arena already contains the box therefore cannot be
holding an allocation the ceiling would have refused, and the early `contains_padded`
return cannot disagree with the predicate.

### 5.6 Where this design is wrong, and the guard rails

Memory scales with the **bounding box**; stones scale with **plies**. Legality permits a
stone 8 cells from the nearest stone, so two bots that both like to extend can widen the
span by 8 per ply with no malice involved. At `N` plies split between two directions the
box reaches `~16 N^2` cells. This is the honest failure mode and it has four faces:

- Memory is `O(bbox)`, not `O(stones)`. At 3 bits per cell dense wins while
  `bbox_cells < ~110 * stones`; a real 200-stone game in an 80x80 box is at 32x — a
  2,400-byte plane bill — comfortably inside, but the margin is a constant factor, not
  an order.
- **The strided gather degrades before memory does — this bites first.** A `Q` or `QR`
  window gather (§6.2) touches 11 rows, `row_words * 8` bytes apart. At `row_words = 1`
  that is two cache lines; at `row_words = 512` each row is its own page and the "two
  cache lines" selling point becomes eleven TLB misses, with the same number of stones on
  the board. This is a `windows_through` cost, not a per-placement one: win detection
  walks only as far as the mover's run reaches (§6.4).
- `clone()` tracks the box, not the stones, which attacks the central premise directly.
- A single `advance` that doubles a large arena is a millisecond latency spike at an
  unpredictable ply. Amortised O(1) is the wrong metric for a player under a clock.

The guards are `COORD_LIMIT` and `MAX_GRID_CELLS`, both surfaced as typed errors, both
documented as **representation limits, not rules**, and both distinguishable from rule
violations via `MoveError::is_rule_violation()`. `MAX_GRID_CELLS = 1 << 24` is a
~6.3 MB position — the same envelope the byte-plane design spent on `1 << 22` cells, so
the bit plane funded a 4x ceiling raise for free.

**The ceiling is on the area of the padded stone box, not on either span.** Because the
arena is shaped to that box (5.5), a game spreading along one axis is bounded by
`COORD_LIMIT` and not by this constant - a straight walk reaches `|q| = 16000` inside a
32768 x 128 arena, still under the ceiling - while a game spreading in every direction at
once refuses once its padded box passes roughly 4096x4096, which takes a deliberate
~1000-ply maximally spreading walk. A *single* seat cannot reach that against a
non-cooperating opponent inside any real match: the spreading placements must be its own,
which puts the refusal past 2000 total plies. Two seats spreading cooperatively can still
trip it, and thereby void only their own match. An earlier implementation grew both
dimensions on every growth event and so reached the ceiling at **ply 65**, inside the
length of a real game; that is fixed and pinned by tests. Two things follow, and both are
tested:

- Random legal games of a few hundred plies never come near it (property 8).
- The maximal spreader *does* trip it, and the refusal is clean: `is_rule_violation()`
  is `false`, the position is bit-identical to before the call, `is_legal()` still
  reports the refused placement as legal, and play continues elsewhere (property 6b).

A runner that wants a hard guarantee rather than a typed error should cap the bounding
box itself; the engine will not, because a ply cap and a board cap are both match rules.

### 5.7 `PartialEq` and the geometry leak

`Grid` deliberately implements **neither `PartialEq` nor `Hash`**. That is a compile-time
trap, not a comment: a future `#[derive(PartialEq)]` on `Position` fails to build, forcing
whoever adds it to write the content-based impl.

`PartialEq for Position` compares `stones_by`, `phase`, `current`, `terminal`,
`zobrist()`, and then zips `self.stones()` with `other.stones()`. `stone_count()` is not
compared separately, because it is the sum of `stones_by`. It ignores `rows`,
`row_words`, `origin_q`, `origin_r`, and the `covered` plane — `covered` is a pure function
of the stones, so comparing it would be redundant, and comparing geometry would make
`apply; undo` unequal to a fresh replay.

Everything observable must be extent-independent. Legal enumeration is (only
`covered && !occupied` cells appear). Stone iteration is, because it is in canonical
coordinate order rather than insertion order — which is why there is no
`occupied: Vec<HexCoord>` alongside the planes: an insertion-ordered stone list is
route-dependent, so it would make two positions with the same stones compare unequal,
and it needs a `truncate` in undo. Recency comes from the record instead.

One deliberate exemption: **`Debug`**. `Position` derives it and `Grid`'s output rides
inside, so `{:?}` prints rows, origins, and raw plane words — two equal positions with
different growth histories print differently. That is intended, not a leak: `Debug`
exists to inspect the representation, and a geometry-hiding hand-written impl would make
exactly the arena bugs this spec worries about undiagnosable. "Observable" throughout
this document means the semantic surface a program can branch on; `Debug` text is
unstable diagnostic output, never parsed, persisted, or compared. The derive that *is*
banned is `PartialEq`, above, because equality is semantic.

### 5.8 The escape hatch, specified in advance

If the p99 bounding box over recorded real games ever exceeds `2^18` cells (512x512),
replace the flat plane with a **64x64-cell tiled arena**: each tile is 64 rows x 1 `u64`
per plane — three planes, 1.5 KB, exactly eight cache lines each — held in a `Vec` arena
with an open-addressed `(tile_q, tile_r) -> u32` directory. Memory returns to
`O(stones)`, `clone` stays a memcpy of the arena, and the directory clones flat. The cost
is one indirection per row access and a boundary case when a window gather, the 17-row
disk, or the 33-row recomputation window crosses a tile edge.

Until that trigger fires the flat grid is strictly better and the tiling is unjustified
complexity. **This is a swap, not a rewrite, only because no public item exposes a row, a
word, a plane, or an index.** Protect that.

---

## 6. Windows and win detection

Nothing about windows is stored. Everything below is derived from the two occupancy
planes on read, in constant time, with no allocation.

Window *reporting* (§6.2, §6.3) and win *detection* (§6.4) are two separate computations
over the same board, deliberately: the first answers "what does a stone here interact
with" for a feature encoder, the second answers "did this placement win" for the rules.
Fusing them would leave C12 (§10.1) asserting one computation against itself.

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
A run is one fact where a slot set was several — seven in a row is one `Win { len: 7 }`
where it was two slot bits, and the caller no longer has to reconstruct the line by
unioning windows.

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

**Why the walk is safe.** `get` is total, so the loop needs no bounds test — it stops at
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

Only the mover's stones are examined: you cannot complete an opponent's line with your own
stone, so the winner is always the mover.

Two independent formulations of the same question remain in the crate: this scan on the
apply path, and the brute-force six-window scans in `audit()` (A7/A8) and in the Tier-C
debug assert C12 (§10.1), which read cell by cell through `Position::window`. They must
agree. What the scan deliberately does *not* do is share the §6.2 line gather: a shared
gather would make the two formulations one, and C12 would then assert a computation
against itself.

---

## 7. Delta, undo, and the undo floor

### 7.1 The structural law

> **Every field is restored by exactly one of four mechanisms, chosen once and named in
> the code.**

| Class | Mechanism | Fields |
| --- | --- | --- |
| **I — involutive** | re-run a self-inverse operation | `occ` bits (set/clear), `hash_cells` (xor/xor), `stones_by` (+-1, and with it `stone_count()`) |
| **R — recomputed** | re-derive from the primary store, over the affected disk only | `covered` (disk-run OR forward, separable-dilation recompute backward — §5.4), `frontier_cells` (a before/after popcount of the derived frontier over the disk runs, on both halves) |
| **II — snapshot** | verbatim copy out of the delta | `phase`, `current` |
| **III — not restored** | unobservable *by construction*, not by privacy (`Debug` output excepted — §5.7): the growth policy sizes and refuses from the live stone box (§5.5), so a rewound arena behaves exactly like a freshly grown one and only its size differs | arena `rows`, `row_words`, `origin_q`, `origin_r`, allocation |

**Nothing is restored by re-derivation from the bookkeeping under test.** Class R is
re-derivation from *occupancy* — the class-I primary store, already restored when the
recomputation reads it — and the hazard the rule guards is untouched: deriving a value
from the structure it is supposed to be checked against erases the detector, and `covered`
is instead checked against three formulations that share nothing with the dilation (C2's
`DISK8` recount, the run-OR, `audit`'s repaint). `terminal` is the degenerate case:
`advance` returns `Err(TerminalState)` before any mutation, so every successful apply had
`terminal == None`; `undo` assigns `None` unconditionally and stores nothing. That is a
theorem (P1 in §10.1), not a guess, and it carries a debug assert.

And the ordering law, which is where the terminal-freeze bug would otherwise live:

> **`advance` runs all fallible checks, then growth, then the entire class-I/R mutation
> *unconditionally*, and only then reads the win out of the freshly updated planes and
> branches. The mutation half never observes the win. `undo` restores class II first,
> then reverses classes I and R in exact reverse statement order.**

Any optimisation of the form "skip the coverage update because we already know this wins"
breaks undo silently. That sentence belongs in a comment at the branch.

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
        // phase and current FREEZE: no assignment here, deliberately.
        Some(o)
    } else {
        let (p, ph) = advance_turn(phase_before, player_before);  // the ONLY transition
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

The transition is a function of the phase and the mover, and of nothing else — the placed
coordinate is not an input. The `current` parameter is load-bearing: "same player" and
"other player" are not recoverable from `before` alone. The call site passes
`player_before`, which is the same value the delta stores, so there is still exactly one
source of truth for the mover.

`check_placement(c)` is, in order: the off-domain classification of §3.7 if
`!c.is_valid()`, `Occupied` if `get(c).is_some()`, `TooFarFromStones` if `c` is not
covered. On the domain — every coordinate a rational player submits — these are
index-mapped table lookups with no geometry walk, so a wild in-domain coordinate from an
untrusted player is rejected before any disk or window arithmetic runs. The one geometry
walk, the classification's `DISK8` probe, is a cold path that only off-domain input can
reach, and it runs in `i32` so nothing wraps.

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

`hash_cells_before` and `stones_by_before` are what C10 (§10.1) asserts against on the
*forward* side, so they are captured here rather than recomputed: C10 checks
`hash_cells == hash_cells_before ^ cell_key(c, mover)`, which is exactly the statement a
wrong or duplicated cell key breaks and which no round-trip can see.

Deliberately **not** `Clone`, `Copy`, `Default`, `PartialEq`, and **not `pub`**. Release
size ~12 bytes, no heap, no `Drop`.

Three things are absent on purpose:

- **No separate `mover` field.** The mover is `current` read before any mutation, which is
  `player_before`. Storing it twice creates a field pair that can disagree.
- **No `terminal_before`.** Provably `None`; `undo` assigns `None`.
- **No `stones_before` / `frontier_before` in release.** Both are class I and move by
  exactly one on each side. They appear in the debug audit as **assertions, not
  assignments**, and that distinction is the entire point: snapshot-*restoring* a value
  that is also maintained incrementally erases the forward bug, while
  snapshot-*asserting* it detects the forward bug at zero release cost.

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

The stack is *deltas*, not a record: it says how to reverse each speculative ply and is
consumed doing so, where a record says what was played and is kept. A cloned position
carries neither, which is what a player mirror needs — it replays the record it was
handed and can undo only what its own `Search` applied.

```rust
pub struct Search<'p> {
    position: &'p mut Position,
    stack: Vec<Undo>,        // the ONLY undo authority in the crate
}
```

Five layers of enforcement, none of them by convention:

1. **`Undo` is `pub(crate)` and unforgeable.** There is no public `undo(token)` and no way
   for a consumer to obtain a token. Undo count `<=` apply count *by construction of the
   `Vec`*. Undoing past the floor is not discouraged, it is **inexpressible**.
2. **`Undo: !Clone`** — a delta cannot be duplicated and replayed twice. **`Vec` LIFO** —
   deltas cannot be reordered.
3. **`&'p mut Position`** — no other path may mutate the position while a session exists,
   so the stack can never go stale and no second `Search` can alias it. Borrow checking,
   not documentation.
4. **`Drop` unwinds.** A position lent to a search is returned in its seeded state on
   every exit path, including `?` and panic. `commit()` is the explicit opt-out and moves
   the floor to the current depth.
5. **A player's mirror is its own `Position`** — maintained from the move stream or
   replayed from the record the seat is handed (`hexo_runner::Game`) — and it builds its
   own `Search` over that mirror. It holds no deltas for the plies that produced the
   mirror, so those plies are unreachable. "Cannot be undone past the position it was
   seeded at" falls out for free.

The floor is `stack.is_empty()`. `Search::new` sets it, `commit` moves it, `unwind`
returns to it.

The runner does not use `Search` at all: it calls `Position::advance`, which builds the
same `Undo` and drops it. One forward code path, two entry points.

### 7.4 Where incremental-versus-recomputed divergence hides

Nine places. Every one of them is a bug that a naive round-trip test cannot see.

- **H1 — `phase_after` is not a function of `phase_before` alone.** It is
  `if won { phase_before } else { advance_turn(phase_before, mover) }`. Anything that
  mirrors
  the transition without the win check diverges on exactly one ply per game, the last one.
  Mitigation: `advance_turn` is private and called from exactly one site, inside the
  `else` arm; mirrors call `advance`; the terminal bit in the Zobrist turn key makes a
  missed freeze a *same-ply* hash mismatch rather than a next-ply one.
- **H2 — a terminal position can carry either placement phase.** (`Opening` cannot
  terminate — §10.3.) If the *second* stone
  wins, the frozen phase is `SecondStone` even though the turn will never be completed.
  **Every branch on `phase` must test `terminal` first.** The concrete trap: `frontier_cells` is a *geometric* count and is emphatically
  **not zero** in a terminal position — roughly 200 cells still satisfy
  `covered && !occupied`. Keep the names disjoint: `frontier_cells` (private, geometric,
  incremental) versus `legal_count()` (public, rule-level, `0` when terminal, `1` in
  `Opening`, else `frontier_cells`). This is precisely the spot where a later optimiser
  writes `if frontier_cells == 0` and silently deletes the freeze.
- **H3 — `Opening` legality is not in the `covered` plane.** At ply 0 the plane is all-zero
  and `frontier_cells == 0`. A `legal_count()` that reads the counter without the
  `Opening` arm reports zero legal moves at game start, which by the theorem in §10.3
  would declare a false terminal. Match on phase first.
- **H4 — undoing the winning ply must un-freeze.** An implementation that *inverts* the
  transition ("`FirstStone` came from `Opening` or `SecondStone`") is ambiguous in general
  and outright wrong under freeze, where `phase_after == phase_before`. This is the whole
  reason `phase` and `current` are class II.
- **H5 — the win check must not gate the mutation** (§7.1). If `apply` skipped the
  coverage OR on a winning ply, the plane is stale under the winning stone's disk, and
  the stale plane is live state: `audit` fails, and `advance` — which never undoes —
  keeps the corruption on the canonical position forever. That the recomputing undo
  would happen to repair it in a search is exactly what would keep the bug invisible
  there, not a defence.
- **H6 — a placement can win on more than one axis.** Two lines crossing at the placed
  cell fill two entries of `Applied::wins`. Nothing may assume exactly one. The related
  case — seven in a row — is now one `Win` with `len == 7` rather than several windows,
  so the length, not the count, is what must not be assumed.
- **H7 — snapshot-restore hides forward drift.** Any field both maintained forward and
  snapshot-restored backward has its forward bug erased by `undo`. That is why
  `frontier_cells`, `stone_count()`, and `stones_by` are class I with debug *assertions* rather
  than class II with assignments, and why `phase` and `current` — which have no involutive
  form — are covered instead by the closed form of §10.2.
- **H8 — symmetric bugs are invisible to the entire round-trip machinery.** A wrong
  `DISK8` offset, a wrong offset in the §6.2 window gather, a wrong `cell_key` constant,
  or a growth copy that uses the same wrong index for read and write all apply and
  un-apply identically. `audit()`, the brute-force oracles, and the frozen golden vectors are the
  only detectors. Say so in the module doc so nobody deletes them as redundant.
- **H9 — geometry leaking into observables.** The arena grows and never shrinks, so
  `apply; undo` leaves a larger arena than a fresh replay of the same prefix. §5.7 is the
  enforcement.
- **H10 — atomicity on error.** All fallible checks precede all mutation. Worth an
  explicit test per error variant, because a future "validate lazily" refactor breaks it
  invisibly.

---

## 8. Zobrist

Written out in full. This is normative source, not a sketch.

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

Three deliberate calls:

- **Derived on read, not accumulated.** `hash_cells` holds only the stone contributions;
  the turn key is XOR-ed in at read time. Undo therefore restores the turn component
  *automatically* by restoring `phase` and `current`. The alternative — one accumulated
  `u64` with XOR-out/XOR-in on every ply — is correct only if the freeze path XORs out and
  back in the *same* key, which is correct by accident and breaks the moment a
  `if phase_changed` guard is added. This removes the failure mode instead of testing
  for it.
- **The turn key covers `terminal`.** This is one step beyond what the hash strictly
  needs to distinguish positions, taken because it is exactly the freeze-desync detector:
  a mirror that fails to freeze diverges on the *same* ply rather than the next one.
- **The turn key covers the whole of the turn state.** `TurnPhase` is three unit variants
  (§3.2) and the mover is one bit, so `kind * 4 + mover * 2 + terminal` is a total,
  injective encoding of `(phase, current, terminal)`. Nothing about whose turn it is, or
  where in the turn, can be silently dropped from the hash.

**Golden vectors are not optional.** A wrong `cell_key` is a symmetric bug that no
round-trip or invariant test can see. Freeze in the repository: sixteen
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

This is unbounded and position-independent. **It imposes no region, crop, or fixed-width
mask** — it is a *sort*, not an index into a table. The index of a cell within one
position's legal list is still position-dependent, which is unavoidable without a crop,
but the *rule producing that list* is global and fixed, so self-play, training, and
serving all read the same ordering and nothing downstream can accidentally invent a
different one. A fixed radius-20 crop, which would exclude out-of-crop legal moves from
policy and search and freeze out-of-rim wins, is structurally unreachable through this
API.

**How the engine produces it, and why it is free.** `LegalActions` walks the derived
frontier — each word composed as `covered & !occ` on read (§5.1) — in storage order:
rows ascending (row `i` is `q = origin_q + i`), and within a row,
words ascending, and within a word, bits ascending from bit 0 (bit `j` is
`r = origin_r + j`). Because the layout is `q`-major and `r`-minor (§5.1), storage order
*is* ascending `(q, r)`. No sort, no comparator, no allocation. Two implementers following
§5.1 and this paragraph produce byte-identical sequences.

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
build and every property test. The one exception is deliberate: the undo-side coverage
recount C2 is `O(DISK_CELLS^2)` — ~47k probes — paid on the rarer half so that every
unwind still gets a from-scratch coverage check without making every apply pay for it.

| # | Invariant | Where |
| --- | --- | --- |
| C1 | After `place`: every in-domain cell of the placed disk reads covered. The necessary direction only; C2 on undo and tier A at checkpoints close the sufficient direction (no cell covered without a stone in range). | apply |
| C2 | After `unplace`: for every in-domain cell of the undone disk, the covered bit equals an independent stone recount — `DISK8` offsets probed against occupancy directly, sharing nothing with the run-OR of apply or the dilation of undo. | undo |
| C3 | `occ[0]` and `occ[1]` are never both set at the placed cell. | after `place` |
| C4 | `zobrist() == hash_cells ^ TURN_KEY[turn_slot()]`. Free — it is the definition. | both |
| C5 | The turn closed form of §10.2. **The most valuable assert in the crate.** | both |
| C6 | `legal_count() == 0` iff `terminal.is_some()`. | both |
| C7 | **P1:** `terminal.is_none()` on entry to `apply_raw`. Structural, from the first check. | apply |
| C8 | The placed cell is at least `LEGAL_RADIUS` from every arena boundary. | after growth |
| C9 | The reserve-around containment check after every growth (§5.5). | growth |
| C10 | `stone_count() == before + 1`; `stones_by[mover] == before + 1`; `get(placed) == Some(mover)`; `hash_cells == before ^ cell_key(placed, mover)`. | apply |
| C11 | `outcome.is_some()` iff some entry of `wins` is `Some`; `outcome.winner == mover`. | apply |
| C12 | **The two win formulations agree, per axis:** `wins[axis.index()].is_some()` equals "some one of the six windows through `c` on that axis is full for the mover", read cell by cell through `Position::window`. Windows whose `start` is off-domain are skipped — such a window holds a cell no stone can occupy, so it is never full. | apply |
| C13 | On entry to `undo_raw`: `zobrist() == audit.zobrist_after` (LIFO / wrong-position detector). | undo |
| C14 | On exit from `undo_raw`: `zobrist()`, `frontier_cells`, and `stone_count()` equal their pre-apply values. **Asserted, never assigned** (H7). | undo |

C11 deliberately stops at the winner's identity. It once also asserted the transition and
the freeze, re-derived through `advance_turn` — which is the function the forward path
had just called, so the assert compared the transition against itself and could not fail.
Both facts are exactly what C5's closed form pins from the stone count and the terminal
bit, independently. Deleting the tautological half was a detector *gained*, not lost.

The old per-cell frontier assert died with the stored frontier plane: a derived read
cannot disagree with its own definition. What remains of frontier state is the counter,
and it is pinned three ways — C14 restores it exactly on undo, A5 recounts it in `audit`,
and both placement halves measure it over the same domain-clipped runs.

C12 earns its place because a win-detection bug is symmetric — a wrong axis step or a
wrong run boundary applies and un-applies identically — so only a second, independent
computation can catch it. The two formulations must stay independent: the run scan walks
outward from the placed cell and never materialises a window, while C12 reads six named
windows cell by cell. Rewriting either in terms of the other's helper deletes the
detector.

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

**One O(1) helper, called after every apply and every undo, catches every phase, player,
and freeze bug** — including the ones a snapshot-restore of `phase` would otherwise hide.
This is the answer to "where could freeze divergence hide": nowhere, if this assert
exists.

### 10.3 Two theorems worth stating in the docs

- **The frontier is never empty once a stone exists.** Take any stone on the convex hull
  of the occupied set; the cell one step outward from it is at distance 1, which is
  `<= LEGAL_RADIUS`, and is unoccupied. So `stones >= 1` implies `frontier_cells >= 1`,
  and "zero legal moves implies terminal" is a theorem rather than a hope. C6 asserts the
  contrapositive.
- **A terminal `Opening` phase is impossible** — one stone cannot fill a six-window — so
  the `Opening` arm never has to reason about freeze.

### 10.4 Tier A — `Position::audit()`, and the order it checks in

`audit()` is `O(arena * DISK_CELLS)` in the worst case. Its recomputations take a
different *route* than the incremental path — coverage is repainted stone by stone into a
scratch plane where the increment edits one disk; the win scan reads windows cell by cell
where the increment walks runs. What it necessarily shares with that path is the
**definitions**: `cell_key` *is* the hash, so recomputing the XOR runs the same constants,
and a wrong constant inside `cell_key` is invisible to A6 by construction. That class of
bug belongs to the frozen golden vectors (§8) — which is why re-baselining them is a
detector deletion, not a fix.

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

Note what is *not* here. No "the stored window masks agree with the board" check, because
no window masks are stored, and no "the move history agrees with the board" check, because
no history is stored (§5.1). No `StoneCount` tier: it could only ever fail when A2 fails,
so it was a second name for one detector. No `FrontierBit` tier: the frontier plane it
checked no longer exists, and a derived read cannot disagree with its own definition.
Every one of these invariant classes was deleted **with its structure**, rather than
weakened against a structure that still exists.

### 10.5 Tier T — test-only oracles

These live in `tests/`, never in the library, because they are `O(stones^2)` or worse and
they exist to disagree with the implementation.

| # | Oracle |
| --- | --- |
| T1 | Legal set: the brute-force union of radius-8 disks over all occupied cells, minus occupied cells, minus cells outside the coordinate domain (§3.1's `is_valid` — a placement there is refused as `CoordOutOfBounds`, so it was never legal), minus (in `Opening`) everything but the origin, minus (when terminal) everything. Compared to `legal_actions()` as an ordered sequence, at **every ply**. The domain clip is not optional and is not a restatement of the arena's row clip: without it the oracle over-reports by 136 cells at a `COORD_LIMIT` face, and the tempting response is to loosen the comparison — which deletes the detector. |
| T2 | Zobrist: recomputed from scratch as `XOR cell_key(c, owner) ^ TURN_KEY[slot]`, compared at every ply and after every undo. |
| T3 | Win: a brute-force six-in-a-row scan over every stone, every axis, and every offset, compared to `is_terminal()` at every ply. |
| T4 | Turn sequence: the `(player, phase)` stream compared against the literal documented pattern `P0; P1 P1; P0 P0; P1 P1; ...` with freeze applied at the terminal ply. |
| T5 | Replay parity: for a random game and every prefix length `k`, a fresh `Position` advanced `k` times is `PartialEq` to a `Search` that applied `n` plies and then undid `n - k`. This states the exactness theorem against a construction path that shares no incremental bookkeeping with the comparison. |

---

## 11. Test obligations

`cargo xtask verify` must pass. It is the whole obligation, and the gates it runs are
defined in `xtask/src/main.rs` — including the ones that are easy to mistake for
duplicates, such as the release-profile lint, which sees dead code the debug profile
deletes. This section does not restate them; a list here would drift from the one CI
runs, which is what it did before.

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
  of the materialised `cells()` arrays** over that corpus — the closed-form index
  arithmetic and a linear scan being two independent statements of the same geometry, which
  is the only reason comparing them detects anything. A QR window whose off-line constraint
  read `dq - dr` instead of `dq + dr` is an H8-class bug: self-consistent, invisible to any
  round trip, and caught only here. Additionally: `intersects` and `touches` are symmetric
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
6. **Growth invariance**, in two halves.

   An earlier draft asked for "a game translated far from the origin — every coordinate
   offset by a random `(dq, dr)`". **That test cannot exist.** Ply 0 is
   `TurnPhase::Opening` and the only legal opening placement is `HexCoord::ORIGIN`
   (§7.1), so a translated game's first ply is `IllegalOpening` and no translated game
   is reachable at all. The forced origin is a rule; the translation was a leftover from
   a design without one. What replaces it states the property the translation was
   reaching for — *arena geometry must never reach an observable* — without requiring an
   unreachable position:

   - **6a, geometry does not leak (H9).** Two positions are seeded identically, but one
     has its arena pre-grown far out by a `Search` that walks along a **single** axis and
     unwinds; `undo` never shrinks it, so the two arenas differ in `rows`, `row_words`,
     `origin_q`, `origin_r`, and allocation size while holding the same stones. Replaying
     an identical random game into both must then produce, at **every** ply, the same
     `Ok`/`Err` from `advance`, equal `Applied` values, equal `Position` (`PartialEq`),
     equal `zobrist()`, equal `legal_count()`, and byte-identical `legal_actions()`
     sequences. The axis is part of the generated case: a walk in all four directions
     re-centres the arena roughly where the flat position's own growth lands, so a
     balanced pre-growth cannot expose an asymmetric leak.
   - **6b, spreading never panics.** A driver that plays the legal cell furthest along a
     rotating axis — the worst case §5.6 describes — runs for ~140 plies with the oracles
     re-checked periodically and `audit()` at the end. The property asserts that a
     refusal, if one happens, is clean rather than asserting it never happens: not a rule
     violation, atomic, the refused placement still `is_legal`, and play continues.
   - **6c, the budget is content, not history.** A position that is searched along a
     *single* axis and fully unwound, and a fresh replay of the same moves, are driven in
     lockstep down a spreading diagonal until `BoardExtentExceeded`. They must agree on
     every accept and on the refusal itself. A balanced pre-growth re-centres the arena
     roughly where the flat position's own growth lands and cannot expose this, which is
     why the direction is part of the case and why 6a alone missed the shipped leak.

7. **Replay parity (T5)** for every prefix.
8. **Limits are unreachable in normal play.** *Random* legal games of up to a few hundred
   plies never return `CoordOutOfBounds` or `BoardExtentExceeded`. Deliberately
   adversarial spreading is a different claim and is covered by 6b, not here.
9. **The canonical ordering is a bijection at every ply**, checked inside the same
   per-ply oracle pass as 2–5: `legal_rank` and `nth_legal` agree with the enumeration at
   every index, and `nth_legal(legal_count())` is `None`.
There is no "the position's own history replays into it" property, because a position has
no history to replay (§4.5). The claim it was making — *the record and the board agree* —
is a claim about a record-keeper and is pinned where one exists: `hexo-runner`'s
`the_prefix_replays_into_the_canonical_position` replays `Game::prefix()`, which is built
from `Game::plies` and never from the position, into an equal `Position`. Held inside the
engine it was comparing `advance` against a loop of `advance` over the list it had just
recorded, and only the second copy of the list gave it any content.

**Boundary tests (`tests/boundary.rs`), because the properties above never travel far
enough to reach them.**

The coordinate domain is a hexagon with six faces, and `legal_actions`, `legal_count`,
`legal_rank`, `nth_legal`, `is_legal`, and `advance` are six implementations of "what may
be played here" that can only disagree there. The four axis-aligned walks —
`(1,0)`, `(-1,0)`, `(0,1)`, `(0,-1)`, each ~2000 plies in `LEGAL_RADIUS` steps — reach all
six faces between them, because each drives two cube coordinates to their limits. At each:
every enumerated action is `is_valid` and `is_legal`; rank and select agree with the
enumeration over a sample weighted to the face; advancing an enumerated action never
yields `CoordOutOfBounds`; `audit()` passes; and an apply/undo taken *at* the face restores
exactly, which is what pins `place` and `unplace` to the same domain filter.

The two diagonal walks are a separate assertion, not a gap: a diagonal widens both arena
dimensions, so the padded bounding box grows as an area and `MAX_GRID_CELLS` refuses at
around `|q| = 4000`, long before `COORD_LIMIT` would. The test asserts that this is what
happens, that the refusal is a representation limit rather than a rule violation, and that
the position survives it intact.

**Fixtures, because the property generator will not find these on its own.**

- A first-stone win: frozen at `FirstStone`; applied, audited, undone, audited, re-applied,
  hash-compared.
- A second-stone win: frozen at `SecondStone`, with the turn's first stone still on the
  board; same cycle.
- A seven-in-a-row, asserting one run of `len == 7` rather than two of six; and two
  crossing lines, asserting a run on each of two axes (H6).
- A win that completes a run the placed stone is *not* at the end of, so the walk is
  exercised in both directions at once rather than only forward or only backward.

Each fixture states its expected `[Option<Win>; 3]` **derived by hand from the move
list**, not captured from a run of the implementation. A captured expectation is the
implementation asserting against itself, which is the same deletion of a detector as
re-baselining a golden vector.

**Smoke test.** At least 10 000 full playouts, each to termination or a test-local ply
bound of 512, with no panics, `audit()` on the final position of each, and an assertion
that the terminal ply and the winner agree with T3. Every reported `Win` is checked over
its own cells: all `len` of them are the winner's, one of them is the placement, `len >= 6`,
and the cells one step off each end are *not* the winner's — the maximality half, which no
window-shaped assertion could state.

Two corrections to an earlier draft of this paragraph, both found by running it:

- **Uniformly random play essentially never terminates.** Measured: *zero* of 20
  uniform 512-ply playouts produced six in a row. A purely uniform smoke test therefore
  hammers the rules and never once exercises the win, the freeze, or the terminal legal
  set — which is the half of the machine most worth smoking. The default 10 000 games
  are consequently driven by a **line-building driver with a swept noise level**, which
  terminates essentially all of them in tens of plies; a smaller slice of pure-uniform
  full-length games runs alongside it to keep the deep, wide-arena case covered. Both
  slices assert the same things.
- **Ply bound versus game count is a real budget.** A debug build runs the full Tier-C
  assertion set on every placement — a disk-wide coverage check plus a second,
  independent win computation on every apply, and the `DISK8`-squared coverage recount
  on every undo — so deep debug playouts are dominated by the assertions, not by the
  engine. The default mix is sized to keep a debug `cargo test --workspace` in the tens
  of seconds. `HEXO_SMOKE_GAMES` and `HEXO_SMOKE_UNIFORM` scale the two slices — a
  malformed value refuses loudly rather than silently falling back to the default — and
  `cargo xtask smoke` is the scaled-up nightly gate: release profile, 10x the
  line-building games, ~17x the uniform ones. That gate is deliberately not part of
  `verify`; hashing, ordering, growth, and win detection are the classes that only
  surface over many thousands of playouts, and a change touching them runs `smoke` too.

---

## 12. Deliberately omitted from the MVP

Each line is a thing someone will ask for. Each has one reason.

| Omitted | Why |
| --- | --- |
| `Board` as a public type | A public board is a field of the state promoted to a constructible thing, and whatever it can be built from becomes a construction path that never runs the turn rules. |
| Any `serde` impl at all | A position is expressible only if it is reachable by a legal game; serialisation of a position is the same hole in a different coat. Records are move lists, and the runner owns them. |
| `snapshot.rs` / `StateLoadError` / position loading | If loading is ever wanted it is replay of a move prefix through the normal rule machine, which needs no new engine code. |
| A `bounds()` accessor | It answers a question no caller has, and it leaks arena geometry to do it (§1). |
| Threat predicates (`is_threat_for`, `threat_player`, `is_active`) | A mask is strictly more information than any predicate over it, and each is a one-liner over `WindowMask`. |
| `touched_windows()` — every window with a stone | No consumer, and it is the one window accessor that would force a stored index. Derivable as `stones()` crossed with `windows_through`. |
| A stored window-mask table | Derived on read in O(1) (§6.2), which deletes a growth path, a delta, and an entire invariant class. |
| A stored legal-move set (`Vec<ActionId>`, `AHashSet`) | The derived frontier is the legal set, and the bit scan yields canonical order for free. |
| `occupied_cells() -> &[HexCoord]` in placement order | Placement order is not a property of a position. `stones()` answers in canonical order; the order they were played in is the record's. |
| A move history on `Position` | The record-keeper has one, and a board that carried a second copy made every consumer hold two that could drift. `replay` is the way back from a record to a board (§13.16). |
| Raw occupancy planes / bitboard slices | A bit layout *is* the origin offset and stride. Exposing it makes growth visible and permanently freezes the arena. If profiling ever demands it, add `copy_planes_into(&self, out, origin, radius)` where the *caller* names the region in coordinates — additive, and it never leaks geometry. |
| `Position` as a trait | Dynamic dispatch or generic infection on the hottest path in the system for the benefit of a second implementer that should not exist. Every method above is inherent and non-conflicting, so a trait with a blanket impl can be added later at zero cost. |
| A draw / non-win `Outcome` variant | The rules have no draw. Ply caps and adjudication are *match* rules; the runner owns them and needs its own result type regardless. |
| A ply cap inside the engine | Same. |
| `row_any` row-summary bits for skipping empty rows in enumeration | **Measured, and it does not pay.** Enumeration is item-bound, not word-bound: `legal_actions` holds 326–338 M items/s across every ply and both a compact and a 4x-inflated arena, and tripling the empty words costs it **3.0%**. `stones` looked more sensitive (+70%) only because it has 35x fewer items, and its real cost was `Stones::next` looking the owner up a second time after the bit scan had already located the cell. That was the one worth building, and it is built: `BitScan::next_slot` hands back the `(word, bit)` slot and `Grid::owner_at` reads the owner straight out of it, alongside an early-out when the maintained population count is spent. `stones` improves 14–57%, at a measured 3–6% cost to `legal_actions` for sharing the scan — the trade is argued in `ENGINE_RL_AUDIT.md`. `row_any` itself stays unbuilt. |
| The 64x64 tiled arena | Unjustified complexity until the p99 bounding box exceeds `2^18` cells; specified in advance (§5.8) so it stays a swap rather than a rewrite. |
| A `Scoped<'s, 'p>` RAII guard for recursive search | `&mut Search` down the call chain already covers recursive alpha-beta and iterative MCTS, with fewer public items and the same guarantees. |
| A second, history-sensitive hash | One hash. `zobrist()` is position-only, which is what makes transposition tables merge — every turn's two stones are playable in either order and reach the same position, so a history-sensitive key would forfeit a 2x merge per turn. A model whose features depend on move order has the record and hashes it into its own cache key. |
| `serde` on a move list | A record is `[Action]`, and `Action` is `ActionId` in a newtype. A record writer emits `u32`s; that is the whole format, and it belongs to the writer. |
| Runtime `Undo`-token validation | The debug `zobrist_after` check catches misuse at 1-in-2^64; making `undo` fallible would force every search to handle an impossible case in its hot loop. |
| Any PyO3 type, dict marshalling, or lazy action view | Bindings live in a leaf crate that depends on this one, never behind a feature flag here — that is what keeps `cargo test` free of a Python toolchain and this crate compilable to `wasm32`. |

---

## 13. Decisions ledger

The decisions that could reasonably have gone the other way, and why they went this way.
One line each. A decision recorded here is settled; reopening one means arguing against
the reason, not rediscovering the question.

1. **Coverage as one bit per cell restored by recomputation, not a `u8` refcount
   restored by decrement.** The refcount existed only because an OR-ed plane cannot be
   un-ORed; but coverage is a pure function of the stones and local to the removed
   stone's disk, so undo recomputes it by the separable zonogon dilation (§5.4) and gets
   the same exactness from the definition, at 8x less memory. This reversed the shipped
   refcount design. Measured: apply+undo 5–15% slower, `clone` — the cost a search
   mirror actually pays — 36–78% faster.
2. **The frontier is derived on read, never stored.** `covered & !occ` per word makes
   the legal set, the count, and canonical enumeration exactly as free as the stored
   plane did, and deletes that plane along with its bit-by-bit maintenance ordering
   (§5.4) and its per-cell invariant class (§10.4), leaving one maintained `u32`.
3. **No stored window masks; derive them from the occupancy planes.** The public surface
   is unchanged either way; storing them adds a growth path, a delta, and an invariant
   class for no observable gain.
4. **Win detection by a per-axis run scan from the placed cell, cross-checked in debug
   against a brute-force six-window scan.** The scan replaced a strided bit fold over an
   11x11 strip: the fold was the crate's densest symmetric-bug habitat — a QR shear, a
   slot-to-strip index table, and two fold widths, none of which any round-trip test could
   see through — and it bought speed on a check that runs once per placement, against a
   walk that stops after two or three probes in the overwhelming majority of positions.
   The scan is also the stronger answer: it reports the whole run, which the fold could
   only express as a set of overlapping windows.
5. **`q`-major, `r`-minor bit layout.** It makes storage order identical to canonical
   `(q, r)` order, so enumeration needs no sort and two implementers cannot disagree.
6. **Undo stack in a borrow-scoped `Search`, with `Undo` `pub(crate)` and `!Clone`.**
   Undoing past the floor becomes inexpressible rather than merely discouraged, and
   `Position::clone` stays flat. Chosen over the public-token design, which allowed
   replaying a token twice and applying one position's token to another.
7. **Two forward entry points — `Position::advance` and `Search::apply` — over one
   internal `apply_raw`.** The runner pays nothing for undo machinery it never uses, and
   there is still exactly one rule implementation.
8. **`advance` rather than `apply` on `Position`.** Irreversibility deserves a distinct
   verb from the reversible `Search::apply`.
9. **All index and distance arithmetic in `i32`, `HexCoord::s() -> i32`,
   `hex_distance -> u32`.** `-q - r` overflows `i16` at the extremes, and these are public
   total functions.
10. **`COORD_LIMIT` and `MAX_GRID_CELLS` shipped as typed errors, labelled not-rules, with
    `is_rule_violation()`.** The alternative was silent `i16` wrap and unbounded memory;
    the claim that no arithmetic runs on unvalidated coordinates is true of the legality
    lookup but false of the public window geometry.
11. **`CoordOutOfBounds` sits below `TerminalState` and `IllegalOpening` in the precedence
    order.** The two checks above it need no board access and no coordinate arithmetic, so
    they cannot be handed a value that would overflow them; everything below it can be
    (§3.7).
12. **Zobrist derived on read (`hash_cells ^ TURN_KEY[slot]`), with `terminal` in the turn
    key.** Deriving makes undo restore the turn component automatically, and `terminal`
    turns a missed freeze into a same-ply mismatch rather than a next-ply one.
13. **`Position` does not implement `Hash`.** A derived one would fold in the arena
    geometry that `PartialEq` ignores; `zobrist()` is the key to use, and `Grid`
    implements neither trait so the derive cannot compile (§5.7).
14. **`Outcome { winner }` only.** The obvious second field, the placement count, is
    `stone_count()` — already exact and already restored by undo; a second copy is a field
    pair that can disagree.
15. **`audit()` is a normal public method, not a cargo feature.** Feature unification makes
    `cfg`-gated correctness machinery unreliable, and symmetric bugs (§7.4 H8) have no
    other detector.
16. **Placement history belongs to the record-keeper, not to `Position`.** The board is a
    value: everything on it is the stones or derived from them, and `replay` is the way
    back from a record to a board. The alternative — a `history: Vec<Action>` on the type
    — was shipped and reversed, because every consumer that keeps a record then keeps two
    copies of it, and the engine-side copy is the one nothing checks: a runner already
    records each ply with its seat and its hash, and a position's own list can only ever
    agree with the moves it was just handed. A model whose features depend on move order
    reads the record, which is `hexo_runner::Game::plies` for a match.
17. **One hash, and it stays position-only.** `zobrist()` covers stones, owners, mover,
    phase kind, and the terminal bit — not history. Hexo transposes structurally: a
    turn's two stones are playable in either order and reach the same position, so a
    history-sensitive key forfeits a 2x merge per turn of search. A model whose features
    depend on move order reads the record and mixes it into its own cache key. The two
    are easy to conflate: a hash that folds in placement order — to serve recency planes
    in a model's encoder, say — has to be documented as process-internal and never
    persisted. This one crosses the container boundary, so it cannot be that hash.
18. **`PartialEq` is positional.** The type is called `Position`; its equality matches
    `zobrist`, `audit`, and the oracles. Two games that reach the same board by different
    move orders compare equal, and *same game* is a question about two records rather
    than about two boards.
19. **Coverage is written only inside the coordinate domain (§5.4).** Otherwise
    `legal_actions` offers placements `advance` refuses with `CoordOutOfBounds` and
    `legal_rank` assigns them a policy index — measured at 136 such coordinates after a
    walk to `q = 16000`. The alternative, filtering at every read, would cost `legal_count`
    its `O(1)` and leave four accessors to keep in agreement instead of one writer.
20. **A win is reported as a run, not as a set of window slots.** `Applied::wins` is
    `[Option<Win>; 3]` indexed by axis — plain data, no accessors. The slot-set form it
    replaced needed a bit layout, a newtype to hide the layout, and an iterator to resolve
    slots back into `Window`s, all to express something a caller then had to re-derive:
    *which line was completed*. `Win { axis, start, len }` is that fact directly, and
    "no win" is `None` rather than an empty set.
21. **`TurnPhase::SecondStone` carries no coordinate and there is no `ReusedFirstStone`
    error.** Occupancy implies the rule — the turn's first stone is on the board and stones
    are permanent — so the payload bought a different error variant for a placement already
    refused, at the price of a `PartialEq` distinction the hash and the dynamics do not
    make (§3.2).
22. **A seat is handed the game, not a board.** Because the record is the history, a model
    that wants move order as an input needs the record, so `hexo-player`'s `Player::choose`
    and `Model`'s two methods take `&Game` — from which the position, the record, and the
    budget all read. This crate is not involved and does not learn what a seat is; the
    entry is here because it is the consequence of 16 that a reader of 16 will ask about,
    and `crates/hexo-player/README.md` argues it.
23. **Off-domain placements classify by the rules, not by the domain (§3.7).**
    `CoordOutOfBounds` is reserved for a placement the rules allow and the engine cannot
    represent — off-domain within `LEGAL_RADIUS` of a stone. Every other off-domain cell
    is `TooFarFromStones`, a rule violation. The alternative reported every off-domain
    coordinate as an engine limit, which let a losing seat submit `(20000, 0)` and have
    the runner void the match as no-contest instead of forfeiting the seat.
24. **`MAX_GRID_CELLS = 1 << 24`, up from `1 << 22`.** The bit plane cut a cell from 11
    bits to 3, so a 4x ceiling raise fits inside the old memory envelope (~6.3 MB worst
    case). The raise moves the cooperative spreading refusal to ~1000 plies and puts a
    single seat's reach past 2000 (§5.6), turning the ceiling from a reachable
    adjudication surface into a genuine backstop.
