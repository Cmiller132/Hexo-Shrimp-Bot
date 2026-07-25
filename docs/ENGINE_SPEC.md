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
| `window` | yes | `Window` (pure six-cell line geometry), `WindowMask`, `WindowRef` — the exposed win-detection surface. |
| `grid` | **no** | The dense recentred arena: two occupancy bit planes, one frontier bit plane, one coverage byte plane, and the growth policy. **Zero public items.** |
| `position` | yes | `Position`, `Applied`, `Outcome`, the rule machine (`advance` / `apply_raw` / `undo_raw`), every read accessor, `audit`. |
| `search` | yes | `Search<'p>` — the borrow-scoped make/unmake session — and the private `Undo` token. |
| `zobrist` | **no** | The const mixing function and the turn-key table. Reachable only through `Position::zobrist()`. |
| `error` | yes | `MoveError`, `IntegrityError`, `IntegrityCheck`. |

There is deliberately **no** `board.rs`, `rules.rs`, `legal.rs`, `windows.rs`, or
`snapshot.rs`. `board` is deleted because a public `Board` type is what re-opened the
rule-bypassing construction path in the reference; occupancy hangs off `Position`.
`rules` is deleted because a free `is_legal_placement(&state, coord)` is a second entry
point that invites non-atomic check-then-place; legality is a private method plus the
public `Position::is_legal`. `legal` and `windows` are deleted because neither the legal
set nor the window masks are stored — see §5 and §6.

`grid` is `mod grid;`, not `pub mod grid;`. Ruling 2's "grid geometry is ENTIRELY
PRIVATE" is enforced by the module system, not by discipline: **no public item in this
crate may ever expose a row, a word, a plane, a stride, or an index.** That constraint is
what makes the arena replaceable without touching a caller (§5.8).

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
/// Hard ceiling on dense arena cells.
///
/// A representation limit, not a rule: a placement that would push the arena past
/// this is legal, and the engine reports that it cannot represent it. It bounds the
/// AREA of the padded stone box, so a walk along one axis is bounded by COORD_LIMIT
/// instead; unreachable without a deliberate multi-hundred-ply walk that spreads in
/// every direction at once.
pub const MAX_GRID_CELLS: u64 = 1 << 22;
```

Two version constants, not three. The Zobrist function rides inside `RULES_VERSION`
because a hash change and a rule change invalidate the same artefacts, and a third
constant is a third thing to forget to bump.

**Dependencies.** `[dependencies]` is empty and stays empty. `[dev-dependencies]` is
`proptest = "1"` only. `MoveError` implements `core::fmt::Display` and
`core::error::Error` by hand — `thiserror` is declared in the workspace but this crate
does not take it; seven variants do not pay for a proc macro, and `core::error::Error`
(stable 1.81, workspace is 1.85) keeps the crate `no_std + alloc` compatible, which is
the cheapest available insurance for the wasm32 target.

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
`pub(crate)`. Its order is fixed and is depended on by `undo` (§7):

```
for dq in -8..=8 { for dr in max(-8, -dq - 8) ..= min(8, -dq + 8) { yield (dq, dr) } }
```

That is `dq`-major, `dr`-minor, exactly 217 entries, and it makes the coverage loop of §5.4
walk contiguous byte runs.

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
    /// The mover places the second stone of its turn and may not reuse `first`.
    SecondStone {
        /// The cell this turn's first stone went to.
        first: HexCoord,
    },
}

impl TurnPhase {
    /// Canonical kind index: `Opening = 0`, `FirstStone = 1`, `SecondStone = 2`.
    ///
    /// Ignores the `first` payload; used by the Zobrist turn key (§8).
    pub const fn kind_index(self) -> usize;
}
```

`SecondStone { first }` keeps the coordinate public: encoders want a "played this turn"
plane, the `ReusedFirstStone` rule needs it, and hiding it would force an accessor that
returns exactly the same thing.

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
index and an action id the same type to the compiler, which is precisely the confusion
that produced the reference's crop failure. `Action`'s field is private so a move is
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
}

/// A window paired with its current ownership.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WindowRef {
    /// Which window.
    pub window: Window,
    /// Who owns which of its cells.
    pub mask: WindowMask,
}
```

`WindowMask`'s inner `[u8; 2]` is private: the player-to-lane mapping is an internal
convention, and `mask(Player::P1)` is the contract.

There is no `is_win_for`, no `threat_player`, no `is_active`, no `intersects`, no
`stone_cells() -> Vec<_>`. Ruling 3: masks are strictly more information than any
predicate, and every one of those is a one-liner over `mask()` and `Window::cells()`. The
`Vec`-returning ones were actively harmful — they allocated inside a search loop.

### 3.5 `position`

```rust
/// A Hexo position: board, move history, turn phase, mover, hash, terminal status.
///
/// Carries the placement sequence that produced it (ruling 1), so any position can
/// be written out as a game and rebuilt with `replay`. It holds no undo stack (§7).
///
/// `PartialEq` is content-based and deliberately ignores **both** arena geometry
/// and history: two positions with the same stones, phase, mover, and terminal
/// status are equal even if one's arena grew larger getting there and even if the
/// two games reached the board by different move orders. Equality means *same
/// position*, matching `zobrist`, the oracles, and `audit`. It is
/// `O(arena extent)`.
///
/// This type deliberately does **not** implement `core::hash::Hash`. Use
/// [`Position::zobrist`], which excludes `SecondStone::first` and would therefore
/// violate the `Eq`/`Hash` contract if wired up as `Hash`.
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
    /// Which of the 18 windows through `action` this placement completed.
    ///
    /// Non-empty iff `outcome.is_some()`. More than one can be set.
    pub winning: WinningWindows,
}

impl Applied {
    /// The completed windows as geometry rather than slots, in the canonical
    /// slot order of §6.3. Empty unless this placement won.
    pub fn winning_windows(&self) -> impl Iterator<Item = Window> + '_;
}

/// A set over the 18 window slots of §6.3.
///
/// The bit layout is `axis.index() * 6 + offset`, and **nothing outside this type
/// needs to know that** — which is the reason it is a type rather than a `u32`.
/// `bits()` is the escape hatch for a record writer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct WinningWindows(u32);

impl WinningWindows {
    pub const EMPTY: Self;
    pub const fn is_empty(self) -> bool;
    pub const fn count(self) -> u32;                              // can exceed 1
    pub const fn contains(self, axis: Axis, offset: usize) -> bool;  // panics past WINDOW_LEN
    pub const fn bits(self) -> u32;
    pub const fn iter(self) -> WinningSlots;                      // (Axis, usize), ascending
}

impl IntoIterator for WinningWindows { /* = iter() */ }

/// How the game ended. Win only — ruling 6.
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

`Outcome` carries the winner and nothing else. The reference's second field,
`placements`, is `Position::stone_count()` — already exact, already restored by undo, and
a second copy is a field pair that can disagree.

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
    /// The second stone of a turn may not reuse the first.
    ReusedFirstStone(HexCoord),
    /// The coordinate is outside [`COORD_LIMIT`].
    ///
    /// A representation limit, not a rule. See [`MoveError::is_rule_violation`].
    CoordOutOfBounds(HexCoord),
    /// The cell already holds a stone.
    Occupied(HexCoord),
    /// The cell is empty but further than [`LEGAL_RADIUS`] from every stone.
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
    /// `stone_count` disagrees with the occupancy planes.
    StoneCount,
    /// A cell is owned by both players.
    DoubleOwned,
    /// A per-player stone count disagrees with its plane.
    StoneCountForPlayer,
    /// `cover` disagrees with a recount of stones within `LEGAL_RADIUS`.
    Coverage,
    /// A frontier bit disagrees with `cover > 0 && !occupied`.
    FrontierBit,
    /// The frontier population count disagrees with the maintained counter.
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
  -> ReusedFirstStone
  -> CoordOutOfBounds
  -> Occupied
  -> TooFarFromStones
  -> BoardExtentExceeded
```

The first three positions match the reference exactly. `CoordOutOfBounds` is inserted
after them because both preceding checks are pure equality tests against a coordinate
that is known valid (`ORIGIN`, or a stored `first`) and therefore do no arithmetic;
everything after it does. `BoardExtentExceeded` is last because growth runs after every
rule check (§7.1).

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
    /// and the terminal bit. Deliberately excludes `SecondStone::first`, which
    /// cannot affect the legal set, any successor, or any value.
    pub fn zobrist(&self) -> u64;
}
```

Every branch on `phase()` in consumer code must test `is_terminal()` first: a terminal
position carries whichever phase it froze at, and a second-stone win freezes at
`SecondStone { first }` with `first` pointing at a live stone.

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
    /// `Position::replay(p.history())` reproduces `p`; any prefix reproduces the
    /// position at that ply.
    ///
    /// # Errors
    /// `ReplayError { ply, action, cause }`. A sequence continuing past a win
    /// fails with `TerminalState` at the first surplus ply.
    pub fn replay(actions: &[Action]) -> Result<Self, ReplayError>;

    /// Apply a placement sequence to an existing position, continuing its history.
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
reachable by a legal game — which is decision A3, finally expressed as an API instead of
a convention. There is no `serde` impl on `Position` and no board-shaped
deserialisation; the reference had both, and its `Board` deserialiser bypassed the turn
rules entirely.

`ReplayError::ply` is the load-bearing field. A record that fails to replay is
untriageable without knowing *where* it diverged, and "illegal move" alone does not let a
caller bisect a corrupt game.

A mirror must call `advance` and must never re-derive the phase transition itself. The
transition is `if won { phase_before } else { advance_turn(phase_before, coord) }`, and
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
    /// `1` in [`TurnPhase::Opening`]. Otherwise the number of empty cells within
    /// [`LEGAL_RADIUS`] of at least one stone.
    pub fn legal_count(&self) -> usize;

    /// Legal placements in canonical order (§9). Allocation-free.
    ///
    /// Yields exactly [`HexCoord::ORIGIN`] in [`TurnPhase::Opening`], nothing when
    /// terminal, and `legal_count()` items otherwise.
    pub fn legal_actions(&self) -> LegalActions<'_>;

    /// Whether `action` is legal right now: phase, occupancy, radius, and the
    /// first-stone reuse rule.
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
keeps training, against scrambled targets. This is the defect `SUGGESTIONS.md` S1
diagnosed; the fix is that the ordering has exactly one implementation and a golden test
over the rank of each played move, not that the action space is bounded.

Both are a popcount prefix and a select scan over the `frontier` plane, which the
`q`-major/`r`-minor layout makes equal to position within the canonical enumeration. Each
touches only the words up to its target, so the cost is `O(word index)` — it *does* grow
with the rank, roughly 1.6 ns per 64-cell word against 3.0 ns per item for a walk of
`legal_actions`.

Measured at 256 stones (7,349 legal): `legal_rank` is 5.6 ns at rank 0, 106 ns at the
middle, 195 ns at the last, against 5.4 ns / 11.0 us / 22.2 us for the walk — **34x to
129x**. The two cross over only at rank ~0. Quadrupling the arena words behind the same
legal set slows the prefix by 97% and the walk by 2%, and the prefix still wins by 67x;
the real crossover is a frontier density below about 0.13 legal cells per word, against
3.3 to 14.4 measured. The boundary tests sample rather than sweep because a ~16,000-row
arena makes the *walk* expensive, not the prefix.

`legal_actions` never special-cases at the call site: callers do not branch on phase to
generate moves. A caller reusing a buffer writes
`out.clear(); out.extend(pos.legal_actions());` — the same allocation behaviour as a
`write_into` method, with no engine API for it. `ExactSizeIterator` means `collect`
pre-sizes exactly.

The reference's four accessors (`write_legal_moves`, `write_legal_action_ids`, `coords`,
`action_ids`) collapse to this one iterator; `.map(Action::id)` is the caller's business.

### 4.4 Windows

```rust
impl Position {
    /// Ownership of the 18 windows through `coord`, in the canonical slot order
    /// of §6.3: axis-major (`Q`, `R`, `QR`), then offset `0..6`, where offset `k`
    /// means `coord` sits at bit `k` of the window.
    ///
    /// Total: defined for any coordinate, occupied or not, inside the arena or
    /// far outside it. Cells outside the arena read as empty. Returns a stack
    /// array; allocates nothing.
    ///
    /// # Panics
    /// Debug builds assert `coord.is_valid()`.
    pub fn windows_through(&self, coord: HexCoord) -> [WindowRef; WINDOWS_PER_PLACEMENT];

    /// Ownership of one specific window.
    ///
    /// Total: a window no stone has ever been near reads as [`WindowMask::EMPTY`].
    /// There is no `Option` — "no stone has ever been here" and "empty" are the
    /// same answer, and the reference's `Option` forced a branch on every caller.
    ///
    /// # Panics
    /// Debug builds assert `window.start.is_valid()`.
    pub fn window(&self, window: Window) -> WindowMask;
}
```

`windows_through` is what a model encoder actually wants: hand it a candidate cell, get
the eighteen six-bit ownership patterns a stone there would interact with. That is
strictly more than any threat predicate could report, computed from data already present.

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

```rust
impl Position {
    /// Every placement that produced this position, oldest first. Length is
    /// always `stone_count()`. Feeding it to `replay` rebuilds an equal
    /// position, and any prefix rebuilds the position at that ply.
    ///
    /// Inside a `Search` this includes the speculative plies applied above the
    /// floor, and each `undo` removes one.
    ///
    /// Scope: records, replay, and debugging. Not part of the read-surface
    /// contract a model encoder should build features on — the representation
    /// may change. Move *identity* is `ActionId`, which is frozen.
    pub fn history(&self) -> &[Action];
}
```

Deliberately absent: `bounds()` (zero callers in the reference; the brief kills it),
`occupied_cells() -> &[HexCoord]` in placement order (`history()` is the placement-order
answer and `stones()` the canonical-order one), and any raw bitboard or plane accessor
(§12).

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
    /// Its recomputation is written independently of the incremental path and
    /// shares no helpers with it. A bug in a shared helper would be invisible to
    /// both, which would defeat the entire purpose of this method.
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
    history: Vec<Action>,       // every placement, oldest first (ruling 1)
    phase: TurnPhase,
    current: Player,
    terminal: Option<Outcome>,
    hash_cells: u64,            // XOR of cell keys only; turn key applied on read (§8)
    stones: u32,                // == history.len(); see below
    stones_by: [u32; 2],
}

#[derive(Clone, Debug)]
struct Grid {
    rows: usize,                // extent along q
    row_words: usize,           // u64 words per row, i.e. extent along r is 64 * row_words
    origin_q: i32,              // q of row 0
    origin_r: i32,              // r of bit 0; always a multiple of 64
    occ: [Vec<u64>; 2],         // rows * row_words words each; stones of P0 / P1
    frontier: Vec<u64>,         // rows * row_words words; empty cells with cover > 0
    cover: Vec<u8>,             // rows * row_words * 64 bytes; stones within LEGAL_RADIUS
    frontier_cells: u32,        // popcount(frontier), maintained
}
```

**There is no legal-move set and no window-mask store.** Both were separate incremental
structures in the reference, each with its own delta, its own growth handling, and its own
class of invariants. Here:

- The legal set **is** the `frontier` plane. Membership is one bit; enumeration is a bit
  scan that produces canonical order for free (§9); the count is a maintained `u32`. No
  `Vec<ActionId>`, no sorted insert, no hash set, no delta.
- The window masks are **derived on read** from the occupancy planes by an O(1) bit
  gather (§6.2), and win detection is a bit fold over the same gather (§6.4). Nothing is
  stored, so there is nothing to grow, nothing to undo, and no "stored mask disagrees
  with the board" bug class. Ruling 3 governs the *public surface* — masks exposed,
  predicates not — and the public surface is exactly as specified in §4.4.

Cell `(q, r)` maps to row `i = q - origin_q` and bit `j = r - origin_r`. **All index
arithmetic is performed in `i32`**, so a coordinate anywhere in the `i16` range produces
an in-range-or-out-of-range answer and never wraps. Out-of-range reads return zero; the
internal write path never sees one, because growth (§5.5) runs first.

Bit-per-cell layout is `q`-major, `r`-minor because that makes a row scan produce
ascending `(q, r)` — the canonical order — with no sort.

**`history` is stored, and `stones` is kept alongside it.** Ruling 1 was reversed: the
placement sequence is what makes a position writable as a game record and rebuildable by
`Position::replay` (§4.2), and it is the only thing on the type that is not derivable
from the board. It costs four bytes per ply. `stones` is redundant with `history.len()`
and is kept so `stone_count` stays a `const fn` on the crate's MSRV; it is *asserted*
equal on every apply and undo (C15) and never restored from the history, per the §7.4
rule that a maintained value is snapshot-asserted rather than snapshot-restored.

History joins the involutive class of §5.4 rather than the delta of §7.2: a push is
exactly inverted by a pop, so `Undo` carries no history field.

Memory is 11 bits per cell in the bounding box — 3 planes at 1 bit plus `cover` at 1
byte — plus 4 bytes per ply of history. The first opened position allocates a 32x128
arena, which is 5,632 bytes of plane payload; a realistic 200-stone game commonly pads
and rounds to a 128x256 arena, ~45 KB, against 800 bytes of history. **`clone` performs
five allocations** — the four grid planes plus the history — and copies their contents.
"Clone is a memcpy" is shorthand for "no pointer chasing and no per-cell work", not a
literal single `memcpy`; it was never one, because `Grid` has always owned four separate
buffers.

### 5.2 Why `cover` is a byte count and not a bit

`cover[cell]` is the number of stones within `LEGAL_RADIUS` of `cell`, in `0..=DISK_CELLS`
(217 fits in `u8` with room). It is **the only structure that makes the frontier exactly
invertible.** A frontier maintained as an OR of radius-8 disks cannot be undone: removing
one stone cannot clear cells that another stone also covers. A `+1` on apply and a `-1` on
undo is self-inverse by construction, and the frontier bit is a pure function of `cover`
and occupancy:

> **The frontier invariant.** `frontier[c] == 1` if and only if `cover[c] > 0 && occ[0][c] == 0 && occ[1][c] == 0`.

The legality rule falls straight out of it: a non-opening placement at `c` is legal iff
`frontier[c] == 1`. That is a total table lookup with **no coordinate arithmetic beyond
the i32 index map**, which is what lets a wild coordinate from an untrusted player be
rejected without ever walking geometry.

### 5.3 Empty position

`Position::new()` allocates nothing: `rows = 0`, `row_words = 0`, all four `Vec`s empty.
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

### 5.4 The involutive placement pair

This is the sharpest code in the crate. `place` and `unplace` are exact statement-order
mirrors, and the ordering is load-bearing, not stylistic.

```rust
fn place(&mut self, c: HexCoord, p: Player) {
    debug_assert!(self.grid.is_empty_cell(c));
    // (a) c is about to stop being a frontier cell, if it was one.
    if self.grid.cover(c) > 0 { self.grid.clear_frontier(c); }
    // (b) occupancy BEFORE the disk loop. c is a member of its own disk.
    self.grid.set_owner(c, p);
    // (c) coverage, in DISK8 order, skipping cells outside the domain.
    let interior = disk_is_interior(c);
    for d in DISK8 {
        let cell = c + d;
        if !interior && !cell.is_valid() { continue; }
        debug_assert!(self.grid.cover(cell) < DISK_CELLS as u8);
        self.grid.bump_cover(cell, 1);
        if self.grid.cover(cell) == 1 && self.grid.is_empty_cell(cell) {
            self.grid.set_frontier(cell);
        }
    }
    // (d) hash, (e) counters, (f) history.
    self.hash_cells ^= zobrist::cell_key(c, p);
    self.stones += 1;
    self.stones_by[p.index()] += 1;
    self.history.push(Action::new(c));
}

fn unplace(&mut self, c: HexCoord, p: Player) {
    self.history.pop();                                          // (f')
    self.stones_by[p.index()] -= 1;                              // (e')
    self.stones -= 1;
    self.hash_cells ^= zobrist::cell_key(c, p);                  // (d')
    let interior = disk_is_interior(c);                          // (c')
    for d in DISK8.iter().rev() {
        let cell = c + *d;
        if !interior && !cell.is_valid() { continue; }
        debug_assert!(self.grid.cover(cell) > 0);
        if self.grid.cover(cell) == 1 && self.grid.is_empty_cell(cell) {
            self.grid.clear_frontier(cell);
        }
        self.grid.bump_cover(cell, -1);
    }
    self.grid.clear_owner(c, p);                                 // (b')
    if self.grid.cover(c) > 0 { self.grid.set_frontier(c); }     // (a')
}
```

`set_frontier` / `clear_frontier` also maintain `frontier_cells`.

**`c` is a member of its own disk**, so (a)/(b) and (c) interact. In `place`, occupancy is
set *before* the disk loop, so when the loop reaches `c` it fails `is_empty_cell` and does
not mark `c` as frontier. In `unplace`, the disk loop runs *before* `clear_owner`, so `c`
again fails it. Steps (a) and (a') read `cover(c)` at the same value on both sides
(pre-increment on the way in, post-decrement on the way out). The mirror is exact only
because of that ordering; swapping (b) and (c) independently would corrupt
`frontier_cells` by exactly one on every ply, in a way that no round-trip test would
catch, because it would be symmetric.

**Coverage is written only inside the coordinate domain.** `check_placement` refuses a
coordinate outside `COORD_LIMIT` with `CoordOutOfBounds`, so if the disk loop marked such
a cell as frontier, `legal_actions` would offer a placement `advance` refuses — and
`legal_rank` would give it a policy index. That was a real defect: a walk to `q = 16000`
put 136 unaddressable coordinates into the legal set, all of which `is_legal` rejected.
Both halves of the pair apply the same `is_valid` filter, computed from the same
coordinate, so they stay exact inverses. The `disk_is_interior` hoist is a fast path for
the predicate, not a second implementation of it: it is a sufficient condition for *all*
217 cells being in-domain, so the per-cell test runs only within `LEGAL_RADIUS` of a face
— which no ordinary game reaches. Pinned by `tests/boundary.rs` at all six faces.

`place` reads nothing about the win and is called unconditionally (§7.1).

### 5.5 Growth policy

**Trigger.** Before any mutation, `advance` requires the arena to contain
`[c.q - 8, c.q + 8] x [c.r - 8, c.r + 8]`. Padding is 8, not 5, because the coverage disk
— not the win strip — is the widest write. Padding by 8 also guarantees that every window
containing an occupied cell is fully in-arena, so the internal gather path (§6) needs no
bounds checks; only the public `window()` / `windows_through()` queries do.

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
- Cost is one allocation plus `4 * live rows` `memcpy`s. Doubling the short dimension
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

- Memory is `O(bbox)`, not `O(stones)`. Dense wins while `bbox_cells < ~30 * stones`; a
  real 200-stone game in an 80x80 box is at 32x, inside the envelope but with ~3x
  headroom, not 100x.
- **The strided gather degrades before memory does — this bites first.** The 11-row win
  strip costs `11 * row_words * 8` bytes of stride. At `row_words = 1` it is two cache
  lines; at `row_words = 512` each row is its own page and the "two cache lines" selling
  point becomes eleven TLB misses per plane, with the same number of stones on the board.
- `clone()` tracks the box, not the stones, which attacks the central premise directly.
- A single `advance` that doubles a large arena is a millisecond latency spike at an
  unpredictable ply. Amortised O(1) is the wrong metric for a player under a clock.

The guards are `COORD_LIMIT` and `MAX_GRID_CELLS`, both surfaced as typed errors, both
documented as **representation limits, not rules**, and both distinguishable from rule
violations via `MoveError::is_rule_violation()`. `MAX_GRID_CELLS = 1 << 22` is a
~5.8 MB position.

**The ceiling is on the area of the padded stone box, not on either span.** Because the
arena is shaped to that box (5.5), a game spreading along one axis is bounded by
`COORD_LIMIT` and not by this constant - a straight walk reaches `|q| = 16000` inside a
32768 x 128 arena, still under the ceiling - while a game spreading in every direction at
once refuses once its padded box passes roughly 2048x2048, which takes a deliberate
~500-ply maximally spreading walk. An earlier implementation grew both dimensions on
every growth event and so reached the ceiling at **ply 65**, inside the length of a real
game; that is fixed and pinned by tests. Two things follow, and both are tested:

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

`PartialEq for Position` compares `stones`, `stones_by`, `phase` (including
`SecondStone::first`), `current`, `terminal`, `zobrist()`, and then zips `self.stones()`
with `other.stones()`. It ignores `rows`, `row_words`, `origin_q`, `origin_r`, and the
`cover` plane — `cover` is a pure function of the stones, so comparing it would be
redundant, and comparing geometry would make `apply; undo` unequal to a fresh replay.

Everything observable must be extent-independent. Legal enumeration is (only
`cover > 0 && !occupied` cells appear). Stone iteration is, because it is in canonical
coordinate order rather than insertion order — which is the other reason the reference's
`occupied: Vec<HexCoord>` is deleted: it was history-dependent, it needed a `truncate` in
undo, and recency now comes from the move stream.

### 5.8 The escape hatch, specified in advance

If the p99 bounding box over recorded real games ever exceeds `2^18` cells (512x512),
replace the flat plane with a **64x64-cell tiled arena**: each tile is 64 rows x 1 `u64`
per plane (512 B, exactly eight cache lines) plus 4 KB of `cover`, held in a `Vec` arena
with an open-addressed `(tile_q, tile_r) -> u32` directory. Memory returns to
`O(stones)`, `clone` stays a memcpy of the arena, and the directory clones flat. The cost
is one indirection per row access and a boundary case when the 11-row strip or the 17-row
disk crosses a tile edge.

Until that trigger fires the flat grid is strictly better and the tiling is unjustified
complexity. **This is a swap, not a rewrite, only because no public item exposes a row, a
word, a plane, or an index.** Protect that.

---

## 6. Windows and win detection

Nothing about windows is stored. Everything below is derived from the two occupancy
planes by a fixed bit gather, in constant time, with no allocation.

### 6.1 Window identity

A window is six consecutive cells along one axis: `Window { start, axis }`, with
`cell(i) = start + axis.vector() * i`. The 18 windows through a cell `c` are indexed by
`(axis, offset)` with `offset = k` meaning **`c` sits at bit `k`**, so

```
window(c, axis, k) = Window { start: c - axis.vector() * k, axis }
```

and `window(c, axis, k).cell(k) == c`.

### 6.2 The 11x11 strip

Every window through `c` lies inside the 11x11 neighbourhood `q in c.q +- 5`,
`r in c.r +- 5`. Gather it once per plane:

```rust
/// `strip[i]` bit `j` = plane bit at `(c.q - 5 + i, c.r - 5 + j)`, for `i, j` in `0..11`.
/// Bits 11..16 are always zero. `c` is at row 5, bit 5.
fn strip11(plane: &[u64], c: HexCoord) -> [u16; 11];
```

Each row is one or two word loads:

```rust
let bit = (c.r - 5) - origin_r;          // i32
let (w, sh) = ((bit >> 6) as usize, (bit & 63) as u32);
let mut v = words[row_base + w] >> sh;
if sh + 11 > 64 { v |= words[row_base + w + 1] << (64 - sh); }   // sh >= 54, so shift <= 10
let out = (v & 0x7FF) as u16;
```

On the internal path the padding-8 invariant guarantees every row and both words are
in-arena. The public path (`windows_through`, `window`) clamps out-of-arena rows and bits
to zero, which is what makes those functions total.

### 6.3 Canonical slot order

Slot index is `axis.index() * 6 + offset`, giving 18 slots in the order

```
Q/0 Q/1 Q/2 Q/3 Q/4 Q/5  R/0 R/1 R/2 R/3 R/4 R/5  QR/0 QR/1 QR/2 QR/3 QR/4 QR/5
```

This order indexes `Position::windows_through`'s return array and `Applied::winning_windows`.

Bit `m` of slot `(axis, k)`'s mask reads from the strip at:

| axis | strip row | strip bit |
| --- | --- | --- |
| `Q` `(1, 0)` | `5 - k + m` | `5` |
| `R` `(0, 1)` | `5` | `5 - k + m` |
| `QR` `(1, -1)` | `5 - k + m` | `5 + k - m` |

For `k, m` in `0..6` every index lands in `0..11`. At `m == k` all three reduce to row 5,
bit 5 — the placed cell — which is the check that the table is right.

### 6.4 Win detection

Win detection is a fold over the mover's strip. Two independent formulations of the same
question exist in the crate: this fold on the hot path, and a brute-force scan of the 18
masks in `audit()` and in a Tier-C debug assert (§10.1). They must agree.

```rust
/// Bit `t` set iff bits `t..t+6` of `x` are all set.
const fn run6_u16(x: u16) -> u16 {
    let a = x & (x >> 1);      // runs of 2
    let b = a & (a >> 2);      // runs of 4
    b & (b >> 2)               // runs of 6
}

/// `out[i]` bit `j` set iff rows `i..i+6` all have bit `j` set.
fn fold6_u16(s: &[u16; 11]) -> [u16; 6] {
    let mut a = [0u16; 10];
    for i in 0..10 { a[i] = s[i] & s[i + 1]; }
    let mut b = [0u16; 8];
    for i in 0..8 { b[i] = a[i] & a[i + 2]; }
    let mut c = [0u16; 6];
    for i in 0..6 { c[i] = b[i] & b[i + 2]; }
    c
}
/// Identical shape over `u32`, for the sheared plane.
fn fold6_u32(s: &[u32; 11]) -> [u32; 6];

/// Bitmask over the 18 slots of §6.3: which windows through `c` are fully `p`'s.
fn winning_windows(&self, c: HexCoord, p: Player) -> u32 {
    let s = strip11(self.grid.occ_plane(p), c);
    let mut out = 0u32;

    // Axis Q (1, 0): six consecutive rows at column 5.
    let cq = fold6_u16(&s);
    for k in 0..6 { if (cq[5 - k] >> 5) & 1 == 1 { out |= 1 << k; } }

    // Axis R (0, 1): six consecutive bits inside row 5.
    let cr = run6_u16(s[5]);
    for k in 0..6 { if (cr >> (5 - k)) & 1 == 1 { out |= 1 << (6 + k); } }

    // Axis QR (1, -1): shear so anti-diagonals become columns, then fold.
    // Cell (i, j) moves to bit j + i; a QR line has i + j constant, and the
    // window through `c` at offset k lands on bit 10 of rows 5-k ..= 10-k.
    let mut sh = [0u32; 11];
    for i in 0..11 { sh[i] = (s[i] as u32) << i; }
    let cd = fold6_u32(&sh);
    for k in 0..6 { if (cd[5 - k] >> 10) & 1 == 1 { out |= 1 << (12 + k); } }

    out
}
```

A player wins iff `winning_windows(c, mover) != 0`. Six *or more* in a row wins, with no
overline rule, because a run of seven contains a fully-owned six-window and the fold finds
it. **Nothing may assume exactly one bit is set** — seven in a row, or two lines crossing
at the placed cell, sets several.

Only the mover's plane is examined: you cannot complete an opponent's window with your own
stone, so the winner is always the mover.

---

## 7. Delta, undo, and the undo floor

### 7.1 The structural law

> **Every field is restored by exactly one of three mechanisms, chosen once and named in
> the code.**

| Class | Mechanism | Fields |
| --- | --- | --- |
| **I — involutive** | re-run a self-inverse operation | `occ` bits (set/clear), `cover` (+1/-1), `frontier` bits (set/clear), `frontier_cells` (+-1), `hash_cells` (xor/xor), `stones` (+-1), `stones_by` (+-1) |
| **II — snapshot** | verbatim copy out of the delta | `phase`, `current` |
| **III — not restored** | unobservable *by construction*, not by privacy: the growth policy sizes and refuses from the live stone box (§5.5), so a rewound arena behaves exactly like a freshly grown one and only its size differs | arena `rows`, `row_words`, `origin_q`, `origin_r`, allocation |

**Nothing is restored by re-derivation.** `terminal` is a fourth, degenerate case:
`advance` returns `Err(TerminalState)` before any mutation, so every successful apply had
`terminal == None`; `undo` assigns `None` unconditionally and stores nothing. That is a
theorem (P1 in §10.1), not a guess, and it carries a debug assert.

And the ordering law, which is where the terminal-freeze bug would otherwise live:

> **`advance` runs all fallible checks, then growth, then the entire class-I mutation
> *unconditionally*, and only then reads the win out of the freshly updated planes and
> branches. The class-I half never observes the win. `undo` restores class II first, then
> inverts class I in exact reverse statement order.**

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
        TurnPhase::FirstStone => self.check_placement(c)?,
        TurnPhase::SecondStone { first } => {
            if c == first { return Err(MoveError::ReusedFirstStone(c)); }
            self.check_placement(c)?;
        }
    }
    let player_before = self.current;
    let phase_before = self.phase;

    // 2. FALLIBLE GEOMETRY. Class III: reallocation only, no observable change.
    self.grid.reserve_around(c)?;

    // 3. CLASS I. Unconditional and infallible. Must not read the win.
    #[cfg(debug_assertions)] let audit = UndoAudit::capture(self);
    self.place(c, player_before);

    // 4. RULE BRANCH. Reads only what step 3 wrote.
    let winning = self.winning_windows(c, player_before);
    let outcome = if winning != 0 {
        let o = Outcome { winner: player_before };
        self.terminal = Some(o);
        // phase and current FREEZE: no assignment here, deliberately.
        Some(o)
    } else {
        let (p, ph) = advance_turn(phase_before, c);   // the ONLY transition function
        self.current = p;
        self.phase = ph;
        None
    };
    ...
}

/// The only phase transition. Private, called from exactly one site.
const fn advance_turn(before: TurnPhase, current: Player, c: HexCoord) -> (Player, TurnPhase);
//   Opening            -> (P1,            FirstStone)
//   FirstStone         -> (same player,   SecondStone { first: c })
//   SecondStone { .. } -> (other player,  FirstStone)
```

The `current` parameter is load-bearing and was missing from an earlier draft of
this signature: "same player" and "other player" are not recoverable from
`before` and `c` alone. The call site passes `player_before`, which is the same
value the delta stores, so there is still exactly one source of truth for the
mover.

`check_placement(c)` is, in order: `CoordOutOfBounds` if `!c.is_valid()`, `Occupied` if
`get(c).is_some()`, `TooFarFromStones` if `cover(c) == 0`. All three are index-mapped
table lookups with no geometry walk, so a wild coordinate from an untrusted player is
rejected before any disk or window arithmetic runs.

The opening arm checks only `c != ORIGIN`, because `Opening` implies the board is empty,
so "occupied" is unreachable there. `IllegalOpening` therefore covers exactly what the
reference covered.

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
    zobrist_after: u64,
    frontier_before: u32,
    stones_before: u32,
}
```

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
        debug_assert_eq!(self.stones, u.audit.stones_before);
        self.debug_assert_turn_closed_form();
    }
}
```

The `zobrist_after` check at entry is the misuse detector: handing a delta from a
different `Position`, out of order, or twice is caught with 1-in-2^64 miss probability, in
debug, at O(1) cost.

### 7.3 The undo floor

**The undo stack is not in `Position`.** It lives in the borrow-scoped `Search` (§4.7),
which delivers ruling 4's floor with no extra machinery.

The stack is *deltas*, not history, and the two are unrelated: `history` records what was
played and is restored by a pop, while the stack records how to reverse each ply and is
consumed. A cloned position carries its history and no deltas, which is exactly what a
player mirror needs — it can name every move of the game so far and undo none of them.

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
5. **A player receives a `Position` by value** and builds its own `Search` over it. It
   holds no deltas for the plies that produced that position, so those plies are
   unreachable. "Cannot be undone past the position it was seeded at" falls out for free.

The floor is `stack.is_empty()`. `Search::new` sets it, `commit` moves it, `unwind`
returns to it.

The runner does not use `Search` at all: it calls `Position::advance`, which builds the
same `Undo` and drops it. One forward code path, two entry points.

### 7.4 Where incremental-versus-recomputed divergence hides

Nine places. Every one of them is a bug that a naive round-trip test cannot see.

- **H1 — `phase_after` is not a function of `phase_before` alone.** It is
  `if won { phase_before } else { advance_turn(phase_before, c) }`. Anything that mirrors
  the transition without the win check diverges on exactly one ply per game, the last one.
  Mitigation: `advance_turn` is private and called from exactly one site, inside the
  `else` arm; mirrors call `advance`; the terminal bit in the Zobrist turn key makes a
  missed freeze a *same-ply* hash mismatch rather than a next-ply one.
- **H2 — a terminal position can carry any of the three phases, and `first` can point at
  a live stone.** If the *second* stone wins, the frozen phase is `SecondStone { first }`
  with `first` legitimately occupied. **Every branch on `phase` must test `terminal`
  first.** The concrete trap: `frontier_cells` is a *geometric* count and is emphatically
  **not zero** in a terminal position — roughly 200 cells still satisfy
  `!occupied && cover > 0`. Keep the names disjoint: `frontier_cells` (private, geometric,
  incremental) versus `legal_count()` (public, rule-level, `0` when terminal, `1` in
  `Opening`, else `frontier_cells`). This is precisely the spot where a later optimiser
  writes `if frontier_cells == 0` and silently deletes the freeze.
- **H3 — `Opening` legality is not in the `cover` plane.** At ply 0 the plane is all-zero
  and `frontier_cells == 0`. A `legal_count()` that reads the counter without the
  `Opening` arm reports zero legal moves at game start, which by the theorem in §10.3
  would declare a false terminal. Match on phase first.
- **H4 — undoing the winning ply must un-freeze.** An implementation that *inverts* the
  transition ("`FirstStone` came from `Opening` or `SecondStone`") is ambiguous in general
  and outright wrong under freeze, where `phase_after == phase_before`. This is the whole
  reason `phase` and `current` are class II.
- **H5 — the win check must not gate the class-I mutation** (§7.1). `undo` clears
  coverage unconditionally; if `apply` ever skipped writing it, `undo` corrupts a
  *different* cell's count.
- **H6 — multiple winning windows.** Seven in a row, or two lines crossing, sets several
  bits in `winning_windows`. Nothing may assume exactly one.
- **H7 — snapshot-restore hides forward drift.** Any field both maintained forward and
  snapshot-restored backward has its forward bug erased by `undo`. That is why
  `frontier_cells`, `stones`, and `stones_by` are class I with debug *assertions* rather
  than class II with assignments, and why `phase` and `current` — which have no involutive
  form — are covered instead by the closed form of §10.2.
- **H8 — symmetric bugs are invisible to the entire round-trip machinery.** A wrong
  `DISK8` offset, a wrong shear in the QR fold, a wrong `cell_key` constant, or a growth
  copy that uses the same wrong index for read and write all apply and un-apply
  identically. `audit()`, the brute-force oracles, and the frozen golden vectors are the
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
- **The turn key covers `terminal`.** This is one step beyond the literal wording of
  ruling 5, taken because it is exactly the freeze-desync detector: a mirror that fails to
  freeze diverges on the *same* ply rather than the next one.
- **The turn key does not cover `SecondStone::first`.** `first` is always occupied, so
  `ReusedFirstStone` and `Occupied` forbid the same cell: `first` cannot change the legal
  set, any successor, or any value, and hashing it would split transposition entries for
  no gain. It is still restored exactly, because it rides inside class-II `phase`.
  Consequence: two positions differing only in `first` are `!=` but hash-equal, so
  **`Position` must not implement `core::hash::Hash`** (§3.5).

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
different one. The reference's radius-20 crop, which caused the main_3 training collapse,
is structurally unreachable through this API.

**How the engine produces it, and why it is free.** `LegalActions` walks the `frontier`
plane in storage order: rows ascending (row `i` is `q = origin_q + i`), and within a row,
words ascending, and within a word, bits ascending from bit 0 (bit `j` is
`r = origin_r + j`). Because the layout is `q`-major and `r`-minor (§5.1), storage order
*is* ascending `(q, r)`. No sort, no comparator, no allocation. Two implementers following
§5.1 and this paragraph produce byte-identical sequences.

The three special cases, in the order they are tested:

1. `terminal.is_some()` — yield nothing. `len() == 0`.
2. `phase == Opening` — yield exactly `Action(HexCoord::ORIGIN)`. `len() == 1`. The
   frontier plane is not consulted; at ply 0 it is empty.
3. otherwise — the bit scan. `len() == frontier_cells`.

`SecondStone { first }` needs no special case: `first` is occupied, so its frontier bit is
clear.

Pinned by `ACTION_ORDER_VERSION` and by a golden test that hashes the full ordering
emitted at every ply of a fixed replayed game.

---

## 10. Invariants

Four tiers, by cost. Every invariant is marked with where it runs. Nothing in Tier C
allocates; nothing in Tier C is `O(arena)`.

### 10.1 Tier C — `debug_assert`, inside `apply_raw` and `undo_raw`

Every one of these is O(1), O(18), or O(217). They run on every ply of every debug build
and every property test.

| # | Invariant | Where |
| --- | --- | --- |
| C1 | `cover(cell) < DISK_CELLS` before each increment; `cover(cell) > 0` before each decrement. | disk loop |
| C2 | For every cell touched by the disk loop, and for the placed cell: `frontier bit == (cover > 0 && !occupied)`. | after `place` / `unplace` |
| C3 | `occ[0]` and `occ[1]` are never both set at the placed cell. | after `place` |
| C4 | `zobrist() == hash_cells ^ TURN_KEY[turn_slot()]`. Free — it is the definition. | both |
| C5 | The turn closed form of §10.2. **The most valuable assert in the crate.** | both |
| C6 | `legal_count() == 0` iff `terminal.is_some()`. | both |
| C7 | **P1:** `terminal.is_none()` on entry to `apply_raw`. Structural, from the first check. | apply |
| C8 | The placed cell is at least `LEGAL_RADIUS` from every arena boundary. | after growth |
| C9 | The reserve-around containment check after every growth (§5.5). | growth |
| C10 | `stones == before + 1`; `stones_by[mover] == before + 1`; `get(placed) == Some(mover)`; `hash_cells == before ^ cell_key(placed, mover)`. | apply |
| C11 | `outcome.is_some()` iff `winning_windows != 0`; `outcome.winner == mover`; `outcome.is_some()` implies `(phase, current)` unchanged, else `(phase, current) == advance_turn(phase_before, placed)`. | apply |
| C12 | **The two win formulations agree:** `winning_windows(c, mover)` from the bit fold equals the brute-force scan of `windows_through(c)` for `mask.is_full_for(mover)`, slot for slot. | apply |
| C13 | On entry to `undo_raw`: `zobrist() == audit.zobrist_after` (LIFO / wrong-position detector). | undo |
| C14 | On exit from `undo_raw`: `zobrist()`, `frontier_cells`, and `stones` equal their pre-apply values. **Asserted, never assigned** (H7). | undo |
| C15 | `history.len() == stones`, and after an apply `history.last() == Some(placed)`. On undo the popped entry is the placement being undone. **Asserted, never assigned** — `stones` is maintained, not derived from the history. | both |

C12 earns its place because the QR shear (§6.4) is the single most error-prone line in the
crate and it is a symmetric bug: a wrong shear applies and un-applies identically, so only
a second, independent computation can catch it.

### 10.2 The turn closed form (C5)

Let `n = stones` and `m = n - (terminal.is_some() as u32)`.

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

`audit()` is `O(arena * DISK_CELLS)` in the worst case. Its recomputation is written
independently of the incremental path and **shares no helper functions with it**; a bug in
a shared helper would be invisible to both, which would defeat the method entirely.

Checked in this order, returning the first failure:

| # | `IntegrityCheck` | Recomputation |
| --- | --- | --- |
| A1 | `StoneCount` | `stones` equals the total popcount of both occupancy planes. |
| A2 | `DoubleOwned` | `occ[0] & occ[1] == 0` in every word. |
| A3 | `StoneCountForPlayer` | `stones_by[p]` equals the popcount of plane `p`. |
| A4 | `ArenaMargin` | every stone is at least `LEGAL_RADIUS` from every arena boundary. |
| A5 | `Coverage` | for every cell in the arena, `cover[cell]` equals the number of stones `s` with `hex_distance(s, cell) <= LEGAL_RADIUS`, counted by iterating stones and their disks. |
| A6 | `FrontierBit` | for every cell, `frontier bit == (cover > 0 && !occupied)`. |
| A7 | `FrontierCount` | `frontier_cells` equals the popcount of the frontier plane. |
| A8 | `Zobrist` | `hash_cells` equals the XOR of `cell_key(c, owner)` over `stones()`. |
| A9 | `Terminal` | `terminal.is_some()` iff some window is fully owned, by a brute-force scan of `windows_through` over every stone. |
| A10 | `Winner` | when terminal, the reported winner owns a completed window, and no window is completed by the other player. |
| A11 | `TurnClosedForm` | the closed form of §10.2. |
| A12 | `History` | `history.len() == stone_count()`, and the history names exactly the occupied cells — compared as a *set* against the plane scan, because order is history's own business and nothing else records it. |

Note what is *not* here: there is no "the stored window masks agree with the board" check,
because no window masks are stored (§5.1). That entire invariant class was deleted with
the structure.

### 10.5 Tier T — test-only oracles

These live in `tests/`, never in the library, because they are `O(stones^2)` or worse and
they exist to disagree with the implementation.

| # | Oracle |
| --- | --- |
| T1 | Legal set: the brute-force union of radius-8 disks over all occupied cells, minus occupied cells, minus (in `Opening`) everything but the origin, minus (when terminal) everything. Compared to `legal_actions()` as an ordered sequence, at **every ply**. |
| T2 | Zobrist: recomputed from scratch as `XOR cell_key(c, owner) ^ TURN_KEY[slot]`, compared at every ply and after every undo. |
| T3 | Win: a brute-force six-in-a-row scan over every stone, every axis, and every offset, compared to `is_terminal()` at every ply. |
| T4 | Turn sequence: the `(player, phase)` stream compared against the literal documented pattern `P0; P1 P1; P0 P0; P1 P1; ...` with freeze applied at the terminal ply. |
| T5 | Replay parity: for a random game and every prefix length `k`, a fresh `Position` advanced `k` times is `PartialEq` to a `Search` that applied `n` plies and then undid `n - k`. This states the exactness theorem against a construction path that shares no incremental bookkeeping with the comparison. |

---

## 11. Test obligations

All of these must pass from the workspace root: `cargo build`, `cargo test --workspace`,
`cargo fmt --all --check`, `cargo clippy --all-targets -- -D warnings`, and
`cargo clippy --release --all-targets -- -D warnings`. The release lint is a separate
obligation, not a duplicate: `debug_assertions` is off in release, which deletes the only
callers of the helpers the tier-C assertions use, so a dead-code regression there is
invisible to the debug lint.

**Unit tests, per module.**

- `coord`: `s()` and `hex_distance` totality at `i16` extremes; `DISK8` has exactly 217
  distinct offsets, all at distance `<= 8`, and its order is `dq`-major/`dr`-minor;
  `is_valid` boundaries.
- `action`: `ActionId` round-trips for every `HexCoord` in a grid sample and for a set of
  raw `u32`s including `0`, `u32::MAX`, and both bias boundaries; `ActionId` ordering,
  `HexCoord` ordering, and `Action` ordering agree over a set spanning all four sign
  quadrants.
- `window`: `Window::cell(k)` of `windows_through(c)[axis*6 + k]` equals `c`, for all 18
  slots; `WindowMask` accessor algebra (`occupied == mask(P0) | mask(P1)`,
  `empty == !occupied & 0x3F`).
- `grid`: growth from empty; growth in each of four directions; that `origin_r` stays a
  multiple of 64; that a grown arena reads back every previously written cell and reads
  zero everywhere new; `MAX_GRID_CELLS` refusal before allocation.
- `zobrist`: the frozen golden vectors (§8) and the twelve `TURN_KEY` entries.
- `position`: one test per `MoveError` variant, and one per **ordered pair of
  simultaneously violated conditions that is actually reachable**, pinning the precedence
  table of §3.7. Three pairs in that table are *not* reachable and have no test, because
  no position can violate both halves at once, not because they were skipped:
  `TerminalState` over `IllegalOpening` (one stone cannot fill a six-window, so a terminal
  `Opening` is impossible — §10.3), `CoordOutOfBounds` over `Occupied` (a cell outside
  `COORD_LIMIT` has never been written), and `Occupied` over `TooFarFromStones` (an
  occupied cell is inside its own radius-8 disk, so its `cover` is at least 1). The
  reachable pairs — terminal over each of `Occupied` / `TooFarFromStones` /
  `CoordOutOfBounds` / `ReusedFirstStone`, `IllegalOpening` over `CoordOutOfBounds` and
  over `TooFarFromStones`, `ReusedFirstStone` over `Occupied`, `CoordOutOfBounds` over
  `TooFarFromStones`, and `TooFarFromStones` over `BoardExtentExceeded` — each get one.
  Also atomicity —
  the position is `PartialEq` to its clone after every rejected placement.
- `search`: floor behaviour — `undo()` at depth 0 returns `None` and leaves `zobrist()`
  unchanged; `commit()` then `unwind()` is a no-op; `Drop` restores the caller's position
  after an early `?`.
- `window`: additionally, `WinningWindows` — every one of the 18 slots round-trips through
  `contains` and `iter` one at a time with no leakage into its neighbours, `iter` and
  `contains` agree over every single- and double-slot mask, and the multi-slot case yields
  in ascending order.
- `position`, history and replay: history length tracks `stone_count` at every ply; a
  rejected placement leaves it untouched; `undo` pops it; `replay` of a history reproduces
  the position and of every prefix reproduces that ply; `ReplayError.ply` names the failing
  index for an illegal opening, an occupied cell, and a sequence continuing past a win; and
  two games reaching the same board by different move orders are `PartialEq` while their
  histories differ.
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
10. **History replays at every ply.** `Position::replay(pos.history())` equals `pos` and
    has the same `zobrist()` and the same history. This is the only property that crosses
    the incremental path against a from-scratch replay — property 7 compares two
    incremental paths with each other.

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
around `|q| = 1984`, long before `COORD_LIMIT` would. The test asserts that this is what
happens, that the refusal is a representation limit rather than a rule violation, and that
the position survives it intact.

**Fixtures, because the property generator will not find these on its own.**

- A first-stone win: frozen at `FirstStone`; applied, audited, undone, audited, re-applied,
  hash-compared.
- A second-stone win: frozen at `SecondStone { first }` with `first` occupied; same cycle.
- A seven-in-a-row and two crossing lines, both asserting `winning_windows.count_ones() > 1`
  (H6).
- A win that completes a window the placed stone is *not* at the end of (offset 2 or 3),
  so the offset arithmetic is exercised away from its boundaries.

**Smoke test.** At least 10 000 full playouts, each to termination or a test-local ply
bound of 512, with no panics, `audit()` on the final position of each, and an assertion
that the terminal ply and the winner agree with T3.

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
  assertion set on every placement — a 217-cell coverage recheck plus a second,
  independent win computation — so a uniform 512-ply playout costs ~125 ms and 10 000 of
  them would be ~20 minutes. The default mix is sized to keep a debug
  `cargo test --workspace` around 20 seconds; `HEXO_SMOKE_GAMES` and
  `HEXO_SMOKE_UNIFORM` raise both counts for a release-profile or nightly run.

---

## 12. Deliberately omitted from the MVP

Each line is a thing someone will ask for. Each has one reason.

| Omitted | Why |
| --- | --- |
| `Board` as a public type | It only ever existed to be a field of the state, and exposing it re-opens the rule-bypassing construction path. |
| The serde impl on `Board` (`board.rs:179-212`) | It deserialises a bare cell list and skips turn rules — a second, rule-free way to build a position. |
| Any `serde` impl at all | A position is expressible only if it is reachable by a legal game; serialisation of a position is the same hole in a different coat. Records are move lists, and the runner owns them. |
| `snapshot.rs` / `StateLoadError` / position loading | If loading is ever wanted it is replay of a move prefix through the normal rule machine, which needs no new engine code. |
| `Board::bounds()` | Zero callers in the entire reference, and it leaks geometry. |
| Threat predicates (`is_threat_for`, `threat_player`, `is_active`) | Ruling 3: masks are strictly more information, and each predicate is a one-liner over `WindowMask`. |
| `touched_windows()` — every window with a stone | No consumer, and it is the one window accessor that would force a stored index. Derivable as `stones()` crossed with `windows_through`. |
| A stored window-mask table | Derived on read in O(1) (§6.2), which deletes a growth path, a delta, and an entire invariant class. |
| A stored legal-move set (`Vec<ActionId>`, `AHashSet`) | The frontier bit plane is the legal set, and the bit scan yields canonical order for free. |
| `occupied_cells() -> &[HexCoord]` in placement order | `history()` is the placement-order answer and `stones()` the canonical-order one; a third accessor over the same data earns nothing. |
| Raw occupancy planes / bitboard slices | A bit layout *is* the origin offset and stride. Exposing it makes growth visible and permanently freezes the arena. If profiling ever demands it, add `copy_planes_into(&self, out, origin, radius)` where the *caller* names the region in coordinates — additive, and it never leaks geometry. |
| `Position` as a trait | Dynamic dispatch or generic infection on the hottest path in the system for the benefit of a second implementer that should not exist. Every method above is inherent and non-conflicting, so a trait with a blanket impl can be added later at zero cost. |
| A draw / non-win `Outcome` variant | Ruling 6. Ply caps and adjudication are match rules; the runner owns them and needs its own result type regardless. |
| A ply cap inside the engine | Same. |
| `row_any` row-summary bits for skipping empty rows in enumeration | **Measured, and it does not pay.** Enumeration is item-bound, not word-bound: `legal_actions` holds 326–338 M items/s across every ply and both a compact and a 4x-inflated arena, and tripling the empty words costs it **3.0%**. `stones` is more sensitive (+70%) only because it has 35x fewer items — but its real cost is `Stones::next` looking the owner up a second time after the bit scan already located the cell, worth ~33% on *every* arena against `row_any`'s 3% on an inflated one. If one of the two gets built, it is the owner lookup. |
| The 64x64 tiled arena | Unjustified complexity until the p99 bounding box exceeds `2^18` cells; specified in advance (§5.8) so it stays a swap rather than a rewrite. |
| A `Scoped<'s, 'p>` RAII guard for recursive search | `&mut Search` down the call chain already covers recursive alpha-beta and iterative MCTS, with fewer public items and the same guarantees. |
| A second, history-sensitive hash | One hash. `zobrist()` is position-only, which is what makes transposition tables merge — every turn's two stones are playable in either order and reach the same position, so a history-sensitive key would forfeit a 2x merge per turn. A model whose features depend on move order has `history()` and hashes it into its own cache key. |
| `serde` on `history()` | It is `&[Action]`, and `Action` is `ActionId` in a newtype. A record writer emits `u32`s; that is the whole format. |
| Runtime `Undo`-token validation | The debug `zobrist_after` check catches misuse at 1-in-2^64; making `undo` fallible would force every search to handle an impossible case in its hot loop. |
| Every PyO3 type (`PythonBoard`, `PythonHexoState`, the dict marshalling, the lazy action view) | Ruling 8. Bindings live in a leaf crate that depends on this one. |

---

## 13. Decisions ledger

Where the three input designs disagreed, this is what was chosen and why. One line each.

1. **Frontier as a `cover: u8` count plane, not an OR-ed bitmask.** An OR is not
   invertible — removing a stone cannot clear cells another stone also covers — and exact
   undo is ruling 4.
2. **Frontier *also* kept as a bit plane derived from `cover`.** It makes the legal set,
   the legal count, and canonical enumeration free, and it is involutive because it is
   driven only by involutive `cover` and occupancy transitions.
3. **No stored window masks; derive them from the occupancy planes.** Ruling 3 governs the
   public surface, which is unchanged; storing them adds a growth path, a delta, and an
   invariant class for no observable gain.
4. **Win detection by bit fold on an 11x11 strip, cross-checked in debug against a
   brute-force 18-mask scan.** The fold is the fast path; the shear arithmetic is a
   symmetric bug waiting to happen, so a second independent formulation asserts against it.
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
11. **`CoordOutOfBounds` sits after `ReusedFirstStone` in the precedence order.** The three
    checks above it are pure equality tests against coordinates already known valid, so the
    reference's observable precedence is preserved exactly.
12. **Zobrist derived on read (`hash_cells ^ TURN_KEY[slot]`), with `terminal` in the turn
    key and `SecondStone::first` out of it.** Deriving makes undo restore the turn
    component automatically; `terminal` turns a missed freeze into a same-ply mismatch;
    `first` cannot affect any value, so hashing it would split transposition entries.
13. **`Position` does not implement `Hash`.** `zobrist()` excludes `first` while
    `PartialEq` includes it, so wiring it up as `Hash` would violate the `Eq`/`Hash`
    contract.
14. **`Outcome { winner }` only.** The reference's second field is `stone_count()`, already
    exact and already restored; a second copy is a field pair that can disagree.
15. **`audit()` is a normal public method, not a cargo feature.** Feature unification makes
    `cfg`-gated correctness machinery unreliable, and symmetric bugs (§7.4 H8) have no
    other detector.
16. **Placement history lives in `Position` (reverses ruling 1's original form).** A
    position that cannot name the game that produced it is not writable as a record, and
    every consumer would otherwise carry a parallel move list that can drift from the
    board. It costs four bytes per ply against an arena already tens of kilobytes, on a
    type whose `clone` was never a single `memcpy` — `Grid` has always owned four
    buffers, so history is a fifth. Push/pop is involutive, so it needs no delta.
17. **One hash, and it stays position-only.** `zobrist()` covers stones, owners, mover,
    phase kind, and the terminal bit — not history. Hexo transposes structurally: a
    turn's two stones are playable in either order and reach the same position, so a
    history-sensitive key forfeits a 2x merge per turn of search. A model whose features
    depend on move order reads `history()` and mixes it into its own cache key. The
    reference conflated these — `hexo_utils/rust/src/state_hash.rs` hashes placement
    order *because* dense-cnn used recency planes, and is explicitly process-internal and
    never persisted, while this hash crosses the container boundary.
18. **`PartialEq` stays positional and excludes history.** The type is called `Position`;
    its equality matches `zobrist`, `audit`, and the oracles. Two games that reach the
    same board by different move orders compare equal. Compare `history()` explicitly
    when *same game* is the question.
19. **Coverage is written only inside the coordinate domain (§5.4).** Otherwise
    `legal_actions` offers placements `advance` refuses with `CoordOutOfBounds` and
    `legal_rank` assigns them a policy index — measured at 136 such coordinates after a
    walk to `q = 16000`. The alternative, filtering at every read, would cost `legal_count`
    its `O(1)` and leave four accessors to keep in agreement instead of one writer.
20. **`WinningWindows` is a type, not a `u32`.** The §6.3 slot layout was documented and
    then reproduced by hand at every call site. A newtype with `iter()`, plus
    `Applied::winning_windows()` resolving slots into real `Window`s, means no consumer
    needs the layout at all.
