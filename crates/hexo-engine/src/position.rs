//! The rule machine: `Position`, its read surface, `advance`, and `audit`.
//!
//! # Where incremental-versus-recomputed divergence hides
//!
//! Several hazards in this module are **symmetric bugs**: a wrong disk row run,
//! a wrong shear in the QR fold, a wrong `cell_key` constant, or a growth copy
//! that uses the same wrong index for read and write all apply and un-apply
//! identically. Round-trip tests cannot see them. [`Position::audit`],
//! the brute-force oracles in `tests/`, and the frozen golden vectors in
//! `zobrist` are the only detectors — **do not delete them as redundant**.
//!
//! The other traps, all guarded here:
//!
//! - `phase_after` is not a function of `phase_before` alone; it is
//!   `if won { phase_before } else { advance_turn(..) }`.
//! - A terminal position can carry any of the three phases, and
//!   `SecondStone::first` can point at a live stone. **Every branch on `phase`
//!   must test `terminal` first.**
//! - `frontier_cells` is a *geometric* count and is emphatically **not** zero
//!   in a terminal position. `legal_count()` is the rule-level answer.
//! - `Opening` legality is not in the `cover` plane; at ply 0 the plane is
//!   all-zero.
//! - Undoing the winning ply must un-freeze, which is why `phase` and `current`
//!   are snapshot-restored rather than re-derived.
//! - The win check must not gate the class-I mutation.

use crate::action::Action;
use crate::coord::{Axis, DISK_CELLS, HexCoord, LEGAL_RADIUS, WINDOW_LEN, hex_distance};
use crate::error::{IntegrityCheck, IntegrityError, MoveError, ReplayError};
use crate::grid::Grid;
use crate::player::{Player, TurnPhase};
use crate::search::Undo;
// The offset table is no longer how the disk is *written* — `Grid` walks it as
// contiguous row runs — so it survives only as the tier-C assertion's second,
// independent statement of the same cell set.
#[cfg(debug_assertions)]
use crate::coord::{DISK8, offset};
#[cfg(debug_assertions)]
use crate::search::UndoAudit;
use crate::window::{WINDOWS_PER_PLACEMENT, Window, WindowMask, WindowRef, WinningWindows};
use crate::zobrist::{TURN_KEY, cell_key};
use core::iter::FusedIterator;

#[cfg(test)]
#[path = "position_tests.rs"]
mod tests;

/// A Hexo position: board, move history, turn phase, mover, hash, terminal status.
///
/// Carries the placement sequence that produced it, so any position can be
/// written out as a game and rebuilt with [`Position::replay`]. It holds no
/// undo stack — that lives in [`crate::Search`].
///
/// `PartialEq` is content-based and deliberately ignores **both** arena
/// geometry and history: two positions with the same stones, phase, mover, and
/// terminal status are equal even if one's arena grew larger getting there and
/// even if the two games reached the board by different move orders. Equality
/// on this type means *same position*, matching [`Position::zobrist`], the
/// oracles, and [`Position::audit`]. Compare [`Position::history`] explicitly
/// when *same game* is what you mean.
///
/// This type deliberately does **not** implement [`core::hash::Hash`]. Use
/// [`Position::zobrist`], which excludes `SecondStone::first` and would
/// therefore violate the `Eq`/`Hash` contract if wired up as `Hash`.
#[derive(Clone, Debug)]
pub struct Position {
    grid: Grid,
    /// Every placement, oldest first. Pushed by `place`, popped by `unplace`.
    ///
    /// Also the ply counter: [`Position::stone_count`] is its length, so there is
    /// no second field that could disagree with it (ruling 14). A push is exactly
    /// inverted by a pop, which puts it in the involutive class and keeps it out
    /// of [`crate::search::Undo`].
    ///
    /// Four bytes per ply against an arena that is already tens of kilobytes,
    /// and `clone` already performs four allocations for the grid planes; this
    /// is a fifth. Deliberately **not** part of `PartialEq` and **not** hashed
    /// into [`Position::zobrist`] — see the type docs and spec §8.
    history: Vec<Action>,
    phase: TurnPhase,
    current: Player,
    terminal: Option<Outcome>,
    /// XOR of cell keys only; the turn key is applied on read.
    hash_cells: u64,
    stones_by: [u32; 2],
}

impl Default for Position {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialEq for Position {
    fn eq(&self, other: &Self) -> bool {
        self.stone_count() == other.stone_count()
            && self.stones_by == other.stones_by
            && self.phase == other.phase
            && self.current == other.current
            && self.terminal == other.terminal
            && self.zobrist() == other.zobrist()
            && self.stones().eq(other.stones())
    }
}

impl Eq for Position {}

/// What one placement did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Applied {
    /// The placement that was made.
    pub action: Action,
    /// Who made it. Equals `current_player()` before the call.
    pub mover: Player,
    /// The phase before the placement.
    pub phase_before: TurnPhase,
    /// The phase after. Equals `phase_before` exactly when this placement won.
    pub phase_after: TurnPhase,
    /// `Some` iff this placement completed a six-window.
    pub outcome: Option<Outcome>,
    /// Which of the 18 windows through `action` this placement completed.
    ///
    /// Non-empty iff `outcome.is_some()`. **More than one can be set** — seven
    /// in a row, or two lines crossing at the placed cell.
    ///
    /// [`Applied::winning_windows`] resolves these into real [`Window`] values.
    pub winning: WinningWindows,
}

impl Applied {
    /// The windows this placement completed, as geometry rather than slots.
    ///
    /// Empty unless this placement won. Yielded in the canonical slot order of
    /// spec §6.3, which is what keeps that order from having to be reproduced
    /// by every consumer.
    pub fn winning_windows(&self) -> impl Iterator<Item = Window> + '_ {
        let start = self.action.coord();
        self.winning.iter().map(move |(axis, offset)| Window {
            start: start.step(axis, -(offset as i16)),
            axis,
        })
    }
}

/// How the game ended. Win only — ruling 6.
///
/// Non-win match results (ply caps, crashes, adjudication) belong to the runner.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Outcome {
    /// The player who completed a window.
    pub winner: Player,
}

// ---------------------------------------------------------------------------
// Construction and turn state
// ---------------------------------------------------------------------------

impl Position {
    /// The empty position: `P0` to move, [`TurnPhase::Opening`], no arena
    /// allocated and no history.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            grid: Grid::new(),
            history: Vec::new(),
            phase: TurnPhase::Opening,
            current: Player::P0,
            terminal: None,
            hash_cells: 0,
            stones_by: [0, 0],
        }
    }

    /// Whose turn it is. Frozen at the winner once terminal.
    #[inline]
    #[must_use]
    pub const fn current_player(&self) -> Player {
        self.current
    }

    /// Where the mover is inside its turn. Frozen once terminal.
    ///
    /// A terminal position carries whichever phase it froze at, so test
    /// [`Position::is_terminal`] before branching on this.
    #[inline]
    #[must_use]
    pub const fn phase(&self) -> TurnPhase {
        self.phase
    }

    /// The winner, if the game is over.
    #[inline]
    #[must_use]
    pub const fn outcome(&self) -> Option<Outcome> {
        self.terminal
    }

    /// Whether the game is over. Equivalent to `outcome().is_some()`.
    #[inline]
    #[must_use]
    pub const fn is_terminal(&self) -> bool {
        self.terminal.is_some()
    }

    /// `phase.kind_index() * 4 + current.index() * 2 + terminal.is_some()`.
    #[inline]
    const fn turn_slot(&self) -> usize {
        self.phase.kind_index() * 4 + self.current.index() * 2 + self.terminal.is_some() as usize
    }

    /// Incremental Zobrist hash.
    ///
    /// Identical across builds, machines, and processes for a given
    /// [`crate::RULES_VERSION`]. Covers stones and owners, the mover, the phase
    /// *kind*, and the terminal bit. Deliberately excludes
    /// `SecondStone::first`, which cannot affect the legal set, any successor,
    /// or any value.
    ///
    /// Derived on read, not accumulated: undo restores the turn component
    /// automatically by restoring `phase` and `current`.
    #[inline]
    #[must_use]
    pub const fn zobrist(&self) -> u64 {
        self.hash_cells ^ TURN_KEY[self.turn_slot()]
    }

    /// Maintained geometric frontier population. Crate-private: this is **not**
    /// [`Position::legal_count`].
    ///
    /// Read only by [`crate::search::UndoAudit`] and by tests, so it is dead
    /// code in a release build — which `cargo clippy --release` reports.
    #[inline]
    #[cfg_attr(not(debug_assertions), allow(dead_code))]
    pub(crate) const fn frontier_cells(&self) -> u32 {
        self.grid.frontier_cells()
    }

    /// The stone-only half of the hash, before the turn key is folded in.
    ///
    /// Read only by [`crate::search::UndoAudit`] and by tests, so it is dead
    /// code in a release build — which `cargo clippy --release` reports.
    #[inline]
    #[cfg_attr(not(debug_assertions), allow(dead_code))]
    pub(crate) const fn hash_cells(&self) -> u64 {
        self.hash_cells
    }
}

// ---------------------------------------------------------------------------
// Occupancy
// ---------------------------------------------------------------------------

impl Position {
    /// Owner of `coord`, or `None` if empty.
    ///
    /// Total over every `(i16, i16)` pair, including coordinates far outside
    /// the arena — those are simply empty. This totality is what keeps geometry
    /// private: a caller can probe anywhere without discovering where the
    /// arrays end.
    #[inline]
    #[must_use]
    pub fn get(&self, coord: HexCoord) -> Option<Player> {
        self.grid.owner(coord)
    }

    /// Whether no stone occupies `coord`. Total, as [`Position::get`].
    #[inline]
    #[must_use]
    pub fn is_empty_cell(&self, coord: HexCoord) -> bool {
        self.grid.is_empty_cell(coord)
    }

    /// Total stones placed. Equals the ply count, and the length of
    /// [`Position::history`].
    #[inline]
    #[must_use]
    pub const fn stone_count(&self) -> u32 {
        self.history.len() as u32
    }

    /// Stones held by one player.
    #[inline]
    #[must_use]
    pub const fn stone_count_for(&self, player: Player) -> u32 {
        self.stones_by[player.index()]
    }

    /// Every occupied cell with its owner, in canonical `(q, r)` order.
    ///
    /// Position-only and route-independent: two positions with the same stones
    /// yield the same sequence regardless of the order the stones were played
    /// or how the arena grew.
    #[must_use]
    pub fn stones(&self) -> Stones<'_> {
        Stones {
            scan: BitScan::new(&self.grid, ScanPlane::Occupied, self.history.len()),
        }
    }

    /// Every placement that produced this position, oldest first.
    ///
    /// Length is always [`Position::stone_count`]. Feeding this straight back
    /// to [`Position::replay`] rebuilds an equal position, and any prefix of it
    /// rebuilds the position at that ply — which is what makes a game record a
    /// move list rather than a serialised board.
    ///
    /// Inside a [`crate::Search`] this includes the speculative plies applied
    /// above the floor, and each `undo` removes one. That is correct: those are
    /// real placements in the line currently being examined.
    ///
    /// > **Scope.** This exists for records, replay, and debugging. It is not
    /// > part of the read-surface contract a model encoder should build
    /// > features on; the engine reserves the right to change how history is
    /// > represented. Move *identity* is [`crate::ActionId`], which is frozen.
    #[inline]
    #[must_use]
    pub fn history(&self) -> &[Action] {
        &self.history
    }
}

// ---------------------------------------------------------------------------
// Replay
// ---------------------------------------------------------------------------

impl Position {
    /// Rebuild a position by replaying a placement sequence from the empty
    /// board.
    ///
    /// A position is expressible exactly when it is reachable by a legal game.
    /// There is no board-shaped deserialisation and no `serde` impl, so this is
    /// the only way to load one — and because it runs every placement through
    /// [`Position::advance`], there is still exactly one rule implementation.
    ///
    /// `Position::replay(p.history())` reproduces `p`.
    ///
    /// # Errors
    /// [`ReplayError`] naming the **ply index** that failed, the placement, and
    /// the [`MoveError`] it produced. A sequence that continues past a win
    /// fails with [`MoveError::TerminalState`] at the first surplus ply, which
    /// is the correct reading of a corrupt record rather than something to
    /// silently stop at.
    pub fn replay(actions: &[Action]) -> Result<Self, ReplayError> {
        let mut pos = Self::new();
        pos.replay_from(actions)?;
        Ok(pos)
    }

    /// Apply a placement sequence to an existing position, continuing its
    /// history.
    ///
    /// This is the catching-up path: a player's mirror consuming a batch of
    /// buffered moves, or a seat joining a game in progress from the move
    /// prefix in its handshake.
    ///
    /// # Errors
    /// As [`Position::replay`]. The `ply` field counts from the start of
    /// `actions`, not from the start of the game — add
    /// [`Position::stone_count`] for an absolute ply.
    ///
    /// **Not atomic.** Placements before the failure have been applied and are
    /// not rolled back; the position is left at the last good ply, which is the
    /// state a caller wants for diagnosis. Clone first if you need the original.
    pub fn replay_from(&mut self, actions: &[Action]) -> Result<(), ReplayError> {
        for (ply, &action) in actions.iter().enumerate() {
            self.advance(action)
                .map_err(|cause| ReplayError { ply, action, cause })?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Legal moves
// ---------------------------------------------------------------------------

impl Position {
    /// Number of legal placements. `0` if and only if the position is terminal.
    ///
    /// `1` in [`TurnPhase::Opening`]. Otherwise the number of empty cells
    /// within [`crate::LEGAL_RADIUS`] of at least one stone.
    #[must_use]
    pub const fn legal_count(&self) -> usize {
        if self.terminal.is_some() {
            0
        } else if matches!(self.phase, TurnPhase::Opening) {
            1
        } else {
            self.grid.frontier_cells() as usize
        }
    }

    /// Legal placements in canonical order (spec §9). Allocation-free.
    ///
    /// > **Canonical ordering v1.** Legal placements are yielded in strictly
    /// > ascending [`crate::ActionId`], which is exactly ascending lexicographic
    /// > `(q, r)` on the signed axial coordinate.
    ///
    /// Yields exactly [`HexCoord::ORIGIN`] in [`TurnPhase::Opening`], nothing
    /// when terminal, and `legal_count()` items otherwise.
    #[must_use]
    pub fn legal_actions(&self) -> LegalActions<'_> {
        let inner = if self.terminal.is_some() {
            LegalInner::Done
        } else if matches!(self.phase, TurnPhase::Opening) {
            LegalInner::Origin
        } else {
            LegalInner::Scan(BitScan::new(
                &self.grid,
                ScanPlane::Frontier,
                self.grid.frontier_cells() as usize,
            ))
        };
        LegalActions { inner }
    }

    /// Where `action` sits in [`Position::legal_actions`] order, or `None` if
    /// it is not legal here.
    ///
    /// This and [`Position::nth_legal`] are the two directions of the canonical
    /// ordering, and they are in the engine for one reason: a policy head is
    /// indexed by this ordering, so self-play, training, and serving must all
    /// use the *same* mapping. Deriving it separately in each of them makes
    /// them agree only by coincidence, and a divergence is silent — the network
    /// keeps training, against scrambled targets. The ordering is pinned by
    /// [`crate::ACTION_ORDER_VERSION`] and by a golden test.
    ///
    /// A popcount prefix over the frontier plane rather than a walk of
    /// [`Position::legal_actions`]. Cost is `O(words below the target)`, so it
    /// does grow with the rank — about **1.6 ns per 64-cell word** against the
    /// walk's **3.0 ns per item**. Measured at 256 stones: 5.6 ns at rank 0,
    /// 106 ns at the middle, 195 ns at the last of 7,349, against 5.4 ns /
    /// 11.0 us / 22.2 us for the walk. The walk wins only when the answer is
    /// the first legal move.
    #[must_use]
    pub fn legal_rank(&self, action: Action) -> Option<usize> {
        if self.terminal.is_some() {
            return None;
        }
        let c = action.coord();
        match self.phase {
            TurnPhase::Opening => (c == HexCoord::ORIGIN).then_some(0),
            TurnPhase::FirstStone => self.grid.frontier_rank(c),
            TurnPhase::SecondStone { first } => {
                if c == first {
                    return None;
                }
                self.grid.frontier_rank(c)
            }
        }
    }

    /// The legal placement at `index` in [`Position::legal_actions`] order, or
    /// `None` if `index >= legal_count()`.
    ///
    /// The inverse of [`Position::legal_rank`]: what a policy head's argmax
    /// needs in order to name a move.
    ///
    /// A select scan, `O(words up to the answer)`. Measured at 256 stones:
    /// 4.8 ns at index 0, 269 ns at the middle, 468 ns at the last of 7,349 —
    /// 41x to 47x faster than `legal_actions().nth(index)`, which the walk beats
    /// only for roughly the first ten indices.
    #[must_use]
    pub fn nth_legal(&self, index: usize) -> Option<Action> {
        if self.terminal.is_some() {
            return None;
        }
        match self.phase {
            TurnPhase::Opening => (index == 0).then(|| Action::new(HexCoord::ORIGIN)),
            _ => self.grid.nth_frontier(index).map(Action::new),
        }
    }

    /// Whether `action` is legal right now: phase, occupancy, radius, and the
    /// first-stone reuse rule.
    ///
    /// Returns `bool`, not `Result`. To learn *why* a placement is rejected,
    /// call [`Position::advance`] and read the error; a rich-error predicate
    /// tempts callers into check-then-apply, which double-validates on the hot
    /// path.
    #[must_use]
    pub fn is_legal(&self, action: Action) -> bool {
        if self.terminal.is_some() {
            return false;
        }
        let c = action.coord();
        match self.phase {
            TurnPhase::Opening => c == HexCoord::ORIGIN,
            TurnPhase::FirstStone => self.check_placement(c).is_ok(),
            TurnPhase::SecondStone { first } => c != first && self.check_placement(c).is_ok(),
        }
    }

    /// Occupancy and radius legality, in precedence order. Three index-mapped
    /// table lookups; no geometry walk, so a wild coordinate from an untrusted
    /// player is rejected before any disk or window arithmetic runs.
    fn check_placement(&self, c: HexCoord) -> Result<(), MoveError> {
        if !c.is_valid() {
            return Err(MoveError::CoordOutOfBounds(c));
        }
        if self.grid.owner(c).is_some() {
            return Err(MoveError::Occupied(c));
        }
        if self.grid.cover(c) == 0 {
            return Err(MoveError::TooFarFromStones(c));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Windows
// ---------------------------------------------------------------------------

/// `(strip row, strip bit)` that bit `m` of slot `(axis, k)` reads from.
///
/// At `m == k` all three reduce to row 5, bit 5 — the queried cell — which is
/// the check that this table is right.
#[inline]
const fn strip_slot(axis: Axis, k: usize, m: usize) -> (usize, usize) {
    match axis {
        Axis::Q => (5 - k + m, 5),
        Axis::R => (5, 5 - k + m),
        Axis::QR => (5 - k + m, 5 + k - m),
    }
}

impl Position {
    /// Ownership of the 18 windows through `coord`, in the canonical slot order
    /// of spec §6.3: axis-major (`Q`, `R`, `QR`), then offset `0..6`, where
    /// offset `k` means `coord` sits at bit `k` of the window.
    ///
    /// Total: defined for any coordinate, occupied or not, inside the arena or
    /// far outside it. Cells outside the arena read as empty. Returns a stack
    /// array; allocates nothing.
    ///
    /// # Panics
    /// Debug builds assert `coord.is_valid()`.
    #[must_use]
    pub fn windows_through(&self, coord: HexCoord) -> [WindowRef; WINDOWS_PER_PLACEMENT] {
        debug_assert!(coord.is_valid());
        let s0 = self.grid.strip11(self.grid.occ_plane(Player::P0), coord);
        let s1 = self.grid.strip11(self.grid.occ_plane(Player::P1), coord);
        let mut out = [WindowRef {
            window: Window {
                start: coord,
                axis: Axis::Q,
            },
            mask: WindowMask::EMPTY,
        }; WINDOWS_PER_PLACEMENT];
        for axis in Axis::ALL {
            for k in 0..WINDOW_LEN {
                let mut m0 = 0u8;
                let mut m1 = 0u8;
                for m in 0..WINDOW_LEN {
                    let (row, bit) = strip_slot(axis, k, m);
                    m0 |= (((s0[row] >> bit) & 1) as u8) << m;
                    m1 |= (((s1[row] >> bit) & 1) as u8) << m;
                }
                out[axis.index() * WINDOW_LEN + k] = WindowRef {
                    window: Window {
                        start: coord.step(axis, -(k as i16)),
                        axis,
                    },
                    mask: WindowMask::from_lanes(m0, m1),
                };
            }
        }
        out
    }

    /// Ownership of one specific window.
    ///
    /// Total: a window no stone has ever been near reads as
    /// [`WindowMask::EMPTY`]. There is no `Option` — "no stone has ever been
    /// here" and "empty" are the same answer.
    ///
    /// Computed cell by cell, independently of the strip gather that
    /// [`Position::windows_through`] uses, so the two are a genuine
    /// cross-check.
    ///
    /// # Panics
    /// Debug builds assert `window.start.is_valid()`.
    #[must_use]
    pub fn window(&self, window: Window) -> WindowMask {
        debug_assert!(window.start.is_valid());
        let mut m0 = 0u8;
        let mut m1 = 0u8;
        for (i, cell) in window.cells().into_iter().enumerate() {
            match self.grid.owner(cell) {
                Some(Player::P0) => m0 |= 1 << i,
                Some(Player::P1) => m1 |= 1 << i,
                None => {}
            }
        }
        WindowMask::from_lanes(m0, m1)
    }
}

// ---------------------------------------------------------------------------
// Win detection
// ---------------------------------------------------------------------------

/// Bit `t` set iff bits `t..t+6` of `x` are all set.
#[inline]
const fn run6(x: u32) -> u32 {
    let a = x & (x >> 1);
    let b = a & (a >> 2);
    b & (b >> 2)
}

/// `out[i]` bit `j` set iff rows `i..i+6` all have bit `j` set.
///
/// One implementation, at one width, for both the plain strip and the sheared
/// one. The shear pushes the top row up by 10 bits, so `u16` will not hold it —
/// and a second copy of this fold differing only in width is exactly the kind of
/// duplication the QR shear already makes dangerous, since the two would drift
/// silently and symmetrically.
#[inline]
const fn fold6(s: &[u32; 11]) -> [u32; 6] {
    let mut a = [0u32; 10];
    let mut i = 0;
    while i < 10 {
        a[i] = s[i] & s[i + 1];
        i += 1;
    }
    let mut b = [0u32; 8];
    i = 0;
    while i < 8 {
        b[i] = a[i] & a[i + 2];
        i += 1;
    }
    let mut c = [0u32; 6];
    i = 0;
    while i < 6 {
        c[i] = b[i] & b[i + 2];
        i += 1;
    }
    c
}

impl Position {
    /// Bitmask over the 18 slots of spec §6.3: which windows through `c` are
    /// fully `p`'s.
    ///
    /// Only the mover's plane is examined: you cannot complete an opponent's
    /// window with your own stone, so the winner is always the mover.
    ///
    /// Six *or more* in a row wins, with no overline rule, because a run of
    /// seven contains a fully-owned six-window and the fold finds it.
    /// **Nothing may assume exactly one bit is set.**
    ///
    /// Returns the raw slot mask; [`Position::apply_raw`] wraps it in
    /// [`WinningWindows`], which is the only type that escapes the crate.
    fn winning_slots(&self, c: HexCoord, p: Player) -> u32 {
        let strip = self.grid.strip11(self.grid.occ_plane(p), c);
        // Widened once, so the plain fold and the sheared fold are the same
        // function rather than two copies of it at different widths.
        let mut s = [0u32; 11];
        let mut i = 0;
        while i < 11 {
            s[i] = strip[i] as u32;
            i += 1;
        }
        let mut out = 0u32;

        // Axis Q (1, 0): six consecutive rows at column 5.
        let cq = fold6(&s);
        for k in 0..WINDOW_LEN {
            out |= ((cq[5 - k] >> 5) & 1) << k;
        }

        // Axis R (0, 1): six consecutive bits inside row 5.
        let cr = run6(s[5]);
        for k in 0..WINDOW_LEN {
            out |= ((cr >> (5 - k)) & 1) << (6 + k);
        }

        // Axis QR (1, -1): shear so anti-diagonals become columns, then fold.
        // Cell (i, j) moves to bit j + i; a QR line has i + j constant, and the
        // window through `c` at offset k lands on bit 10 of rows 5-k ..= 10-k.
        let mut sh = [0u32; 11];
        let mut i = 0;
        while i < 11 {
            sh[i] = s[i] << i;
            i += 1;
        }
        let cd = fold6(&sh);
        for k in 0..WINDOW_LEN {
            out |= ((cd[5 - k] >> 10) & 1) << (12 + k);
        }

        out
    }
}

// ---------------------------------------------------------------------------
// The rule machine
// ---------------------------------------------------------------------------

/// The only phase transition. Private, called from exactly one site.
///
/// The spec sketches this as `advance_turn(before, c)`; it also needs the
/// current player, because `FirstStone` keeps the mover and `SecondStone`
/// flips it.
#[inline]
const fn advance_turn(before: TurnPhase, current: Player, c: HexCoord) -> (Player, TurnPhase) {
    match before {
        TurnPhase::Opening => (Player::P1, TurnPhase::FirstStone),
        TurnPhase::FirstStone => (current, TurnPhase::SecondStone { first: c }),
        TurnPhase::SecondStone { .. } => (current.other(), TurnPhase::FirstStone),
    }
}

/// The turn closed form of spec §10.2: `(phase kind index, mover)` implied by
/// the stone count and the terminal bit alone.
///
/// `None` marks an unreachable combination (`stones == 0` while terminal, or a
/// one-stone win).
pub(crate) const fn turn_closed_form(stones: u32, terminal: bool) -> Option<(usize, Player)> {
    if stones == 0 {
        return if terminal {
            None
        } else {
            Some((0, Player::P0))
        };
    }
    let m = stones - terminal as u32;
    if m == 0 {
        return None;
    }
    let kind = if m.is_multiple_of(2) { 2 } else { 1 };
    let player = if ((m - 1) / 2).is_multiple_of(2) {
        Player::P1
    } else {
        Player::P0
    };
    Some((kind, player))
}

impl Position {
    /// The involutive half of a placement (spec §5.4). Exact statement-order
    /// mirror of [`Position::unplace`]; **the ordering is load-bearing, not
    /// stylistic**, because `c` is a member of its own disk.
    fn place(&mut self, c: HexCoord, p: Player) {
        debug_assert!(self.grid.is_empty_cell(c));
        // (a) `c` is about to stop being a frontier cell, if it was one.
        if self.grid.cover(c) > 0 {
            self.grid.clear_frontier(c);
        }
        // (b) Occupancy BEFORE the disk update, so the update does not re-mark
        // `c`: it reads the occupancy planes to decide which newly-covered cells
        // become frontier, and `c` is now among the occupied.
        self.grid.set_owner(c, p);
        // (c) Coverage and the frontier bits it creates, in `DISK8` order.
        //
        // Disk cells outside the coordinate domain are skipped, so the frontier
        // plane holds only valid coordinates and enumeration can never offer a
        // placement that `advance` would refuse with `CoordOutOfBounds`.
        self.grid.add_cover_disk(c);
        // (d) hash, (e) counters.
        self.hash_cells ^= cell_key(c, p);
        self.stones_by[p.index()] += 1;
        // (f) history. A push is exactly inverted by a pop, so history joins
        // the involutive class rather than needing a snapshot in `Undo`.
        self.history.push(Action::new(c));
    }

    /// The exact inverse of [`Position::place`], in reverse statement order.
    fn unplace(&mut self, c: HexCoord, p: Player) {
        let popped = self.history.pop(); // (f')
        debug_assert_eq!(
            popped,
            Some(Action::new(c)),
            "C15: history top is not the placement being undone"
        );
        self.stones_by[p.index()] -= 1; // (e')
        self.hash_cells ^= cell_key(c, p); // (d')
        // (c') The same runs walked in reverse, derived the same way from the
        // same coordinate, so the two remain exact inverses.
        self.grid.remove_cover_disk(c);
        self.grid.clear_owner(c, p); // (b')
        if self.grid.cover(c) > 0 {
            self.grid.set_frontier(c); // (a')
        }
    }

    /// The single forward code path. Two entry points call it:
    /// [`Position::advance`] (which drops the delta) and
    /// [`crate::Search::apply`] (which keeps it).
    ///
    /// > `advance` runs all fallible checks, then growth, then the entire
    /// > class-I mutation **unconditionally**, and only then reads the win out
    /// > of the freshly updated planes and branches.
    ///
    /// Any optimisation of the form "skip the coverage update because we
    /// already know this wins" breaks undo silently.
    pub(crate) fn apply_raw(&mut self, action: Action) -> Result<(Applied, Undo), MoveError> {
        let c = action.coord();

        // 1. FALLIBLE RULE CHECKS. No mutation above the end of this block, so
        //    a rejected placement leaves the position bit-identical.
        if self.terminal.is_some() {
            return Err(MoveError::TerminalState);
        }
        match self.phase {
            TurnPhase::Opening => {
                if c != HexCoord::ORIGIN {
                    return Err(MoveError::IllegalOpening);
                }
            }
            TurnPhase::FirstStone => self.check_placement(c)?,
            TurnPhase::SecondStone { first } => {
                if c == first {
                    return Err(MoveError::ReusedFirstStone(c));
                }
                self.check_placement(c)?;
            }
        }
        let player_before = self.current;
        let phase_before = self.phase;

        // 2. FALLIBLE GEOMETRY. Class III: reallocation only.
        self.grid.reserve_around(c)?;

        // 3. CLASS I. Unconditional and infallible. Must not read the win.
        #[cfg(debug_assertions)]
        let mut audit = UndoAudit::capture(self);
        self.place(c, player_before);

        // 4. RULE BRANCH. Reads only what step 3 wrote.
        let winning = self.winning_slots(c, player_before);
        let outcome = if winning != 0 {
            let o = Outcome {
                winner: player_before,
            };
            self.terminal = Some(o);
            // phase and current FREEZE: no assignment here, deliberately.
            Some(o)
        } else {
            let (p, ph) = advance_turn(phase_before, player_before, c);
            self.current = p;
            self.phase = ph;
            None
        };

        #[cfg(debug_assertions)]
        {
            audit.set_after(self.zobrist());
            self.debug_assert_tier_c(c, player_before, phase_before, winning, &audit);
        }

        let applied = Applied {
            action,
            mover: player_before,
            phase_before,
            phase_after: self.phase,
            outcome,
            winning: WinningWindows::from_bits(winning),
        };
        let undo = Undo {
            action,
            phase_before,
            player_before,
            #[cfg(debug_assertions)]
            audit,
        };
        Ok((applied, undo))
    }

    /// Reverse one [`Position::apply_raw`]. Class II first, then class I in
    /// exact reverse statement order.
    pub(crate) fn undo_raw(&mut self, u: Undo) {
        #[cfg(debug_assertions)]
        debug_assert_eq!(
            self.zobrist(),
            u.audit.zobrist_after,
            "C13: undo applied to the wrong position, or out of LIFO order"
        );

        self.phase = u.phase_before; // class II, before class I
        self.current = u.player_before;
        self.terminal = None; // theorem P1
        self.unplace(u.action.coord(), u.player_before);

        #[cfg(debug_assertions)]
        {
            // C14: asserted, never assigned.
            debug_assert_eq!(self.zobrist(), u.audit.zobrist_before, "C14: zobrist");
            debug_assert_eq!(
                self.frontier_cells(),
                u.audit.frontier_before,
                "C14: frontier_cells"
            );
            debug_assert_eq!(self.stone_count(), u.audit.stones_before, "C14: stones");
            self.debug_assert_frontier_around(u.action.coord());
            self.debug_assert_turn_closed_form();
            debug_assert_eq!(
                self.legal_count() == 0,
                self.terminal.is_some(),
                "C6: legal_count/terminal disagree"
            );
        }
    }

    /// Advance the position irreversibly by one placement.
    ///
    /// The only mutating method on `Position`. Used by the runner, which owns
    /// the canonical state and never rewinds it, and by a player's mirror
    /// consuming the move stream.
    ///
    /// A mirror must call this and must never re-derive the phase transition
    /// itself: the transition is
    /// `if won { phase_before } else { advance_turn(..) }`, and anything that
    /// mirrors it without the win check diverges on exactly one ply per game —
    /// the last one.
    ///
    /// # Errors
    /// Any [`MoveError`], in the documented precedence. **Atomic:** on `Err`
    /// the position is bit-identical to before the call, including arena
    /// geometry.
    pub fn advance(&mut self, action: Action) -> Result<Applied, MoveError> {
        // Builds the same `Undo` that `Search::apply` does and drops it. That
        // is free: `Undo` is a small POD with no heap and no `Drop`, which is
        // what lets there be exactly one forward code path.
        let (applied, _undo) = self.apply_raw(action)?;
        Ok(applied)
    }
}

// ---------------------------------------------------------------------------
// Tier-C debug assertions
// ---------------------------------------------------------------------------

#[cfg(debug_assertions)]
impl Position {
    /// C5: the turn closed form. The most valuable assert in the crate.
    fn debug_assert_turn_closed_form(&self) {
        let form = turn_closed_form(self.stone_count(), self.terminal.is_some());
        let (kind, player) = form.expect("C5: unreachable stones/terminal combination");
        debug_assert_eq!(self.phase.kind_index(), kind, "C5: phase kind");
        debug_assert_eq!(self.current, player, "C5: mover");
    }

    /// C2: the frontier invariant over the placed cell and its whole disk.
    fn debug_assert_frontier_around(&self, c: HexCoord) {
        let check = |cell: HexCoord| {
            let expect = self.grid.cover(cell) > 0 && self.grid.owner(cell).is_none();
            debug_assert_eq!(
                self.grid.frontier_bit(cell),
                expect,
                "C2: frontier invariant at ({}, {})",
                cell.q,
                cell.r
            );
        };
        check(c);
        for d in DISK8 {
            let cell = offset(c, d);
            // Out-of-domain cells are never covered and never in the frontier,
            // so `expect` would be trivially satisfied; skipping keeps the
            // assertion aligned with what `place` actually touches.
            if cell.is_valid() {
                check(cell);
            }
        }
    }

    /// C2, C3, C5, C6, C8, C10, C11, C12 after a successful apply.
    fn debug_assert_tier_c(
        &self,
        c: HexCoord,
        mover: Player,
        phase_before: TurnPhase,
        winning: u32,
        audit: &UndoAudit,
    ) {
        // C8: the placed cell is at least LEGAL_RADIUS from every boundary.
        debug_assert!(self.grid.contains_padded(c), "C8: arena margin");
        // C3: never double-owned.
        debug_assert!(!self.grid.is_double_owned(c), "C3: double-owned cell");
        // C10
        debug_assert_eq!(self.stone_count(), audit.stones_before + 1, "C10: stones");
        debug_assert_eq!(self.get(c), Some(mover), "C10: owner");
        // C15: history is the ply counter, so what is left to check is that the
        // placement it just recorded is the one that was made. The old pairing of
        // `history.len()` against a separate `stones` field went away with the
        // field (ruling 14); `audit()` now compares that length straight against
        // the occupancy planes, which is a stronger statement than either.
        debug_assert_eq!(
            self.history.last().copied(),
            Some(Action::new(c)),
            "C15: history top"
        );
        debug_assert_eq!(
            self.stones_by[mover.index()],
            audit.stones_by_before + 1,
            "C10: stones_by"
        );
        debug_assert_eq!(
            self.hash_cells,
            audit.hash_cells_before ^ cell_key(c, mover),
            "C10: hash_cells"
        );
        // C11
        debug_assert_eq!(
            self.terminal.is_some(),
            winning != 0,
            "C11: outcome/winning disagree"
        );
        if let Some(o) = self.terminal {
            debug_assert_eq!(o.winner, mover, "C11: winner is not the mover");
            debug_assert_eq!(self.phase, phase_before, "C11: phase did not freeze");
            debug_assert_eq!(self.current, mover, "C11: mover did not freeze");
        } else {
            let (p, ph) = advance_turn(phase_before, mover, c);
            debug_assert_eq!((self.current, self.phase), (p, ph), "C11: transition");
        }
        // C12: the two win formulations must agree, slot for slot. The QR shear
        // is a symmetric bug waiting to happen; only a second, independent
        // computation can catch it.
        let mut brute = 0u32;
        for (i, wr) in self.windows_through(c).iter().enumerate() {
            if wr.mask.is_full_for(mover) {
                brute |= 1 << i;
            }
        }
        debug_assert_eq!(brute, winning, "C12: win formulations disagree");
        // C2, C5, C6.
        self.debug_assert_frontier_around(c);
        self.debug_assert_turn_closed_form();
        debug_assert_eq!(
            self.legal_count() == 0,
            self.terminal.is_some(),
            "C6: legal_count/terminal disagree"
        );
    }
}

// ---------------------------------------------------------------------------
// Iterators
// ---------------------------------------------------------------------------

/// Which bit plane a [`BitScan`] walks.
#[derive(Clone, Copy, Debug)]
enum ScanPlane {
    /// The legal set.
    Frontier,
    /// The union of both occupancy planes.
    Occupied,
}

/// A canonical-order walk over one bit plane.
///
/// Rows ascend (`q`), then words within a row ascend, then bits within a word
/// ascend from bit 0 (`r`). Because the layout is `q`-major and `r`-minor,
/// storage order *is* ascending `(q, r)`. No sort, no comparator, no
/// allocation.
#[derive(Clone, Debug)]
struct BitScan<'a> {
    grid: &'a Grid,
    plane: ScanPlane,
    word: usize,
    cur: u64,
    remaining: usize,
}

impl<'a> BitScan<'a> {
    fn new(grid: &'a Grid, plane: ScanPlane, remaining: usize) -> Self {
        let cur = if grid.total_words() == 0 {
            0
        } else {
            Self::word_at(grid, plane, 0)
        };
        Self {
            grid,
            plane,
            word: 0,
            cur,
            remaining,
        }
    }

    fn word_at(grid: &Grid, plane: ScanPlane, i: usize) -> u64 {
        match plane {
            ScanPlane::Frontier => grid.frontier_plane()[i],
            ScanPlane::Occupied => grid.occupied_word(i),
        }
    }

    /// The next set bit, as the `(word, bit)` slot that holds it.
    ///
    /// Returning the slot rather than only the coordinate is what lets
    /// [`Stones`] read the owner out of the plane it has already indexed,
    /// instead of converting to a coordinate and mapping it back.
    ///
    /// `#[inline]` is load-bearing, not decoration: both iterators are consumed
    /// in tight folds, and leaving this as an out-of-line call costs enumeration
    /// roughly a fifth of its throughput.
    #[inline]
    fn next_slot(&mut self) -> Option<(usize, u32)> {
        // Every set bit has been yielded, so the rest of the plane is empty and
        // there is nothing left to scan for. Without this the last `next` walks
        // every trailing word — on an arena inflated by a rewound search that is
        // most of the arena, and it is what `stones` there was paying: 889 ns
        // against 529 ns for the same 96 stones.
        //
        // Tested at entry, deliberately, and not inside the loop where it would
        // be one branch per exhausted word instead of one per item. That looks
        // like the cheaper placement and measures far worse — `legal_actions`
        // loses 11-48% to it against 3-6% here — because the loop body is what
        // has to stay in the shape the optimiser already likes.
        if self.remaining == 0 {
            #[cfg(debug_assertions)]
            self.debug_assert_plane_exhausted();
            return None;
        }
        loop {
            if self.cur != 0 {
                let b = self.cur.trailing_zeros();
                self.cur &= self.cur - 1;
                self.remaining -= 1;
                return Some((self.word, b));
            }
            self.word += 1;
            if self.word >= self.grid.total_words() {
                // `remaining` is non-zero here, so this is only reachable if the
                // maintained count over-states the plane. Loud in debug, and a
                // clean end of iteration rather than a panic in release.
                debug_assert!(false, "the maintained population count exceeds the plane");
                self.remaining = 0;
                return None;
            }
            self.cur = Self::word_at(self.grid, self.plane, self.word);
        }
    }

    /// The next set bit as a coordinate, for the consumer that does not need the
    /// slot.
    ///
    /// Keeping this wrapper rather than having [`LegalActions`] call
    /// [`BitScan::next_slot`] itself is measured, not stylistic: the flatter form
    /// is consistently *slower*, 11.09 us against 10.48 us over 3,373 actions at
    /// ply 96.
    #[inline]
    fn next_coord(&mut self) -> Option<HexCoord> {
        let (word, bit) = self.next_slot()?;
        Some(self.grid.coord_of(word, bit))
    }

    /// The plane really is exhausted when `remaining` says so.
    ///
    /// The early-out above trusts a maintained count to decide there is nothing
    /// left; were that count ever short, enumeration would silently truncate,
    /// which is the one failure mode here worth paying for. `O(words)`, once per
    /// exhausted iterator, and out of line so its body cannot weigh on the
    /// inlining of the scan itself.
    #[cfg(debug_assertions)]
    #[cold]
    fn debug_assert_plane_exhausted(&self) {
        debug_assert_eq!(self.cur, 0, "unyielded bits in the current word");
        debug_assert!(
            (self.word + 1..self.grid.total_words())
                .all(|i| Self::word_at(self.grid, self.plane, i) == 0),
            "the maintained population count is short of the plane"
        );
    }
}

/// Occupied cells with their owners, in canonical `(q, r)` order.
#[derive(Clone, Debug)]
pub struct Stones<'a> {
    scan: BitScan<'a>,
}

impl Iterator for Stones<'_> {
    type Item = (HexCoord, Player);

    fn next(&mut self) -> Option<Self::Item> {
        let (word, bit) = self.scan.next_slot()?;
        let grid = self.scan.grid;
        Some((grid.coord_of(word, bit), grid.owner_at(word, bit)))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.scan.remaining, Some(self.scan.remaining))
    }
}

impl ExactSizeIterator for Stones<'_> {}
impl FusedIterator for Stones<'_> {}

/// Legal placements in canonical order (spec §9). Allocation-free.
#[derive(Clone, Debug)]
pub struct LegalActions<'a> {
    inner: LegalInner<'a>,
}

/// The three cases of spec §9, in the order they are tested.
#[derive(Clone, Debug)]
enum LegalInner<'a> {
    /// Terminal: nothing is legal.
    Done,
    /// `Opening`: exactly the origin. The frontier plane is not consulted.
    Origin,
    /// Everything else: the frontier bit scan.
    Scan(BitScan<'a>),
}

impl LegalActions<'_> {
    fn remaining(&self) -> usize {
        match &self.inner {
            LegalInner::Done => 0,
            LegalInner::Origin => 1,
            LegalInner::Scan(s) => s.remaining,
        }
    }
}

impl Iterator for LegalActions<'_> {
    type Item = Action;

    fn next(&mut self) -> Option<Action> {
        match &mut self.inner {
            LegalInner::Done => None,
            LegalInner::Origin => {
                self.inner = LegalInner::Done;
                Some(Action::new(HexCoord::ORIGIN))
            }
            LegalInner::Scan(s) => s.next_coord().map(Action::new),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.remaining();
        (n, Some(n))
    }
}

impl ExactSizeIterator for LegalActions<'_> {}
impl FusedIterator for LegalActions<'_> {}

// ---------------------------------------------------------------------------
// Integrity audit
// ---------------------------------------------------------------------------

/// Build an [`IntegrityError`].
#[inline]
const fn fail<T>(check: IntegrityCheck, coord: Option<HexCoord>) -> Result<T, IntegrityError> {
    Err(IntegrityError { check, coord })
}

impl Position {
    /// Recompute every derived structure from the stones alone and compare.
    ///
    /// A normal method, not a `cfg` or a cargo feature: feature unification
    /// makes `cfg`-gated correctness machinery unreliable, and the runner may
    /// want a paranoid mode. `O(arena * DISK_CELLS)` — for tests, fuzzing, and
    /// deliberate paranoia, not for a search loop.
    ///
    /// Its recomputation is written independently of the incremental path and
    /// shares no helpers with it. A bug in a shared helper would be invisible
    /// to both, which would defeat the entire purpose of this method.
    ///
    /// # Errors
    /// The first broken invariant found, in the order of spec §10.4.
    pub fn audit(&self) -> Result<(), IntegrityError> {
        let g = &self.grid;
        let total = g.total_words();
        let occ0 = g.occ_plane(Player::P0);
        let occ1 = g.occ_plane(Player::P1);

        // A1. `stone_count()` is `history.len()`, so this is also the length half
        // of A12 — and a stronger statement of it than a separate counter was.
        let pop0: u32 = occ0.iter().map(|w| w.count_ones()).sum();
        let pop1: u32 = occ1.iter().map(|w| w.count_ones()).sum();
        if self.stone_count() != pop0 + pop1 {
            return fail(IntegrityCheck::StoneCount, None);
        }

        // A2
        for i in 0..total {
            let both = occ0[i] & occ1[i];
            if both != 0 {
                return fail(
                    IntegrityCheck::DoubleOwned,
                    Some(g.coord_of(i, both.trailing_zeros())),
                );
            }
        }

        // A3
        if self.stones_by[0] != pop0 || self.stones_by[1] != pop1 {
            return fail(IntegrityCheck::StoneCountForPlayer, None);
        }

        // Independent stone list, read straight out of the planes.
        let mut stones: Vec<(HexCoord, Player)> = Vec::with_capacity(self.history.len());
        for i in 0..total {
            let mut w = occ0[i] | occ1[i];
            while w != 0 {
                let b = w.trailing_zeros();
                w &= w - 1;
                let owner = if (occ0[i] >> b) & 1 == 1 {
                    Player::P0
                } else {
                    Player::P1
                };
                stones.push((g.coord_of(i, b), owner));
            }
        }

        // A4
        let pad = LEGAL_RADIUS as i32;
        let (lo_q, hi_q) = (g.origin_q(), g.origin_q() + g.rows() as i32 - 1);
        let (lo_r, hi_r) = (g.origin_r(), g.origin_r() + 64 * g.row_words() as i32 - 1);
        for &(c, _) in &stones {
            let (q, r) = (c.q as i32, c.r as i32);
            if q - pad < lo_q || q + pad > hi_q || r - pad < lo_r || r + pad > hi_r {
                return fail(IntegrityCheck::ArenaMargin, Some(c));
            }
        }

        // A5: recount coverage from a 17x17 box filtered by hex distance —
        // deliberately not the DISK8 table the incremental path walks.
        let cells = total * 64;
        let mut recount = vec![0u8; cells];
        for &(s, _) in &stones {
            for dq in -(pad as i16)..=(pad as i16) {
                for dr in -(pad as i16)..=(pad as i16) {
                    let cell = HexCoord::new(s.q + dq, s.r + dr);
                    // Cells outside the coordinate domain carry no coverage;
                    // `place` skips them so enumeration cannot offer them.
                    if hex_distance(s, cell) > LEGAL_RADIUS || !cell.is_valid() {
                        continue;
                    }
                    let idx = match g.cell_index(cell) {
                        Some(i) => i,
                        None => return fail(IntegrityCheck::ArenaMargin, Some(cell)),
                    };
                    if recount[idx] as usize >= DISK_CELLS {
                        return fail(IntegrityCheck::Coverage, Some(cell));
                    }
                    recount[idx] += 1;
                }
            }
        }
        let cover = g.cover_plane();
        for (i, &want) in recount.iter().enumerate() {
            if cover[i] != want {
                return fail(
                    IntegrityCheck::Coverage,
                    Some(g.coord_of(i / 64, (i % 64) as u32)),
                );
            }
        }

        // A6
        let frontier = g.frontier_plane();
        for (i, &covered) in cover.iter().enumerate().take(cells) {
            let (word, bit) = (i / 64, (i % 64) as u32);
            let occupied = ((occ0[word] | occ1[word]) >> bit) & 1 == 1;
            let set = (frontier[word] >> bit) & 1 == 1;
            if set != (covered > 0 && !occupied) {
                return fail(IntegrityCheck::FrontierBit, Some(g.coord_of(word, bit)));
            }
        }

        // A7
        let fpop: u32 = frontier.iter().map(|w| w.count_ones()).sum();
        if fpop != g.frontier_cells() {
            return fail(IntegrityCheck::FrontierCount, None);
        }

        // A8
        let mut h = 0u64;
        for &(c, p) in &stones {
            h ^= cell_key(c, p);
        }
        if h != self.hash_cells {
            return fail(IntegrityCheck::Zobrist, None);
        }

        // A9: brute-force six-in-a-row over every stone, axis, and offset.
        let mut winners = [false; 2];
        for &(c, p) in &stones {
            for axis in Axis::ALL {
                for k in 0..WINDOW_LEN {
                    let mut all = true;
                    for m in 0..WINDOW_LEN {
                        let cell = c.step(axis, m as i16 - k as i16);
                        if self.get(cell) != Some(p) {
                            all = false;
                            break;
                        }
                    }
                    if all {
                        winners[p.index()] = true;
                    }
                }
            }
        }
        if (winners[0] || winners[1]) != self.terminal.is_some() {
            return fail(IntegrityCheck::Terminal, None);
        }

        // A10
        if let Some(o) = self.terminal
            && (!winners[o.winner.index()] || winners[o.winner.other().index()])
        {
            return fail(IntegrityCheck::Winner, None);
        }

        // A11
        match turn_closed_form(self.stone_count(), self.terminal.is_some()) {
            Some((kind, player)) if kind == self.phase.kind_index() && player == self.current => {}
            _ => return fail(IntegrityCheck::TurnClosedForm, None),
        }

        // A12: history's entries are exactly the occupied cells. Compared as a
        // *set* against the independent stone list above, because order is
        // history's own business and nothing else records it. Its length was
        // already checked against the planes by A1, which is where the ply
        // counter lives now.
        let mut replayed: Vec<HexCoord> = self.history.iter().map(|a| a.coord()).collect();
        replayed.sort_unstable();
        let mut occupied: Vec<HexCoord> = stones.iter().map(|&(c, _)| c).collect();
        occupied.sort_unstable();
        if let Some(bad) = replayed
            .iter()
            .zip(&occupied)
            .find(|(a, b)| a != b)
            .map(|(a, _)| *a)
        {
            return fail(IntegrityCheck::History, Some(bad));
        }

        Ok(())
    }
}
