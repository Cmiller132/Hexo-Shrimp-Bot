//! The rule machine: `Position`, its read surface, `advance`, and `audit`.

use crate::action::Action;
use crate::coord::{Axis, DISK_CELLS, HexCoord, LEGAL_RADIUS, WINDOW_LEN, hex_distance};
#[cfg(debug_assertions)]
use crate::coord::{DISK8, offset};
use crate::error::{IntegrityCheck, IntegrityError, MoveError, ReplayError};
use crate::grid::Grid;
use crate::player::{Player, TurnPhase};
use crate::search::Undo;
#[cfg(debug_assertions)]
use crate::search::UndoAudit;
use crate::window::{WINDOWS_PER_PLACEMENT, Window, WindowMask, WindowRef, WinningWindows};
use crate::zobrist::{TURN_KEY, cell_key};
use core::iter::FusedIterator;

#[cfg(test)]
#[path = "position_tests.rs"]
mod tests;

/// A Hexo position: board, move history, turn phase, mover, hash, terminal status.
#[derive(Clone, Debug)]
pub struct Position {
    grid: Grid,
    /// Every placement, oldest first. Pushed by `place`, popped by `unplace`.
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
    pub winning: WinningWindows,
}

impl Applied {
    /// The windows this placement completed, as geometry rather than slots.
    pub fn winning_windows(&self) -> impl Iterator<Item = Window> + '_ {
        let start = self.action.coord();
        self.winning.iter().map(move |(axis, offset)| Window {
            start: start.step(axis, -(offset as i16)),
            axis,
        })
    }
}

/// How the game ended. Win only.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Outcome {
    /// The player who completed a window.
    pub winner: Player,
}

impl Position {
    /// The empty position: `P0` to move, [`TurnPhase::Opening`], no arena allocated and
    /// no history.
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
    #[inline]
    #[must_use]
    pub const fn zobrist(&self) -> u64 {
        self.hash_cells ^ TURN_KEY[self.turn_slot()]
    }

    /// Maintained geometric frontier population, which is **not**
    /// [`Position::legal_count`].
    #[inline]
    #[cfg_attr(not(debug_assertions), allow(dead_code))]
    pub(crate) const fn frontier_cells(&self) -> u32 {
        self.grid.frontier_cells()
    }

    /// The stone-only half of the hash, before the turn key is folded in.
    #[inline]
    #[cfg_attr(not(debug_assertions), allow(dead_code))]
    pub(crate) const fn hash_cells(&self) -> u64 {
        self.hash_cells
    }
}

impl Position {
    /// Owner of `coord`, or `None` if empty.
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

    /// Total stones placed.
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
    #[must_use]
    pub fn stones(&self) -> Stones<'_> {
        Stones {
            scan: BitScan::new(&self.grid, ScanPlane::Occupied, self.history.len()),
        }
    }

    /// Every placement that produced this position, oldest first.
    #[inline]
    #[must_use]
    pub fn history(&self) -> &[Action] {
        &self.history
    }
}

impl Position {
    /// Rebuild a position by replaying a placement sequence from the empty board.
    pub fn replay(actions: &[Action]) -> Result<Self, ReplayError> {
        let mut pos = Self::new();
        pos.replay_from(actions)?;
        Ok(pos)
    }

    /// Apply a placement sequence to an existing position, continuing its history.
    pub fn replay_from(&mut self, actions: &[Action]) -> Result<(), ReplayError> {
        for (ply, &action) in actions.iter().enumerate() {
            self.advance(action)
                .map_err(|cause| ReplayError { ply, action, cause })?;
        }
        Ok(())
    }
}

impl Position {
    /// Number of legal placements. `0` if and only if the position is terminal.
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

    /// Where `action` sits in [`Position::legal_actions`] order, or `None` if it is not
    /// legal here.
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

    /// The legal placement at `index` in [`Position::legal_actions`] order, or `None`
    /// if `index >= legal_count()`.
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

    /// Whether `action` is legal right now: phase, occupancy, radius, and the first-
    /// stone reuse rule.
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

    /// Occupancy and radius legality, in precedence order.
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

/// `(strip row, strip bit)` that bit `m` of slot `(axis, k)` reads from.
#[inline]
const fn strip_slot(axis: Axis, k: usize, m: usize) -> (usize, usize) {
    match axis {
        Axis::Q => (5 - k + m, 5),
        Axis::R => (5, 5 - k + m),
        Axis::QR => (5 - k + m, 5 + k - m),
    }
}

impl Position {
    /// Ownership of the 18 windows through `coord`, in the canonical slot order of spec
    /// §6.3: axis-major (`Q`, `R`, `QR`), then offset `0..6`, where offset `k` means
    /// `coord` sits at bit `k` of the window.
    ///
    /// Near a coordinate-domain face, a returned window's start can be off-domain.
    /// Callers must skip those slots before passing the window to [`Position::window`].
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

/// Bit `t` set iff bits `t..t+6` of `x` are all set.
#[inline]
const fn run6(x: u32) -> u32 {
    let a = x & (x >> 1);
    let b = a & (a >> 2);
    b & (b >> 2)
}

/// `out[i]` bit `j` set iff rows `i..i+6` all have bit `j` set.
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
    fn winning_slots(&self, c: HexCoord, p: Player) -> u32 {
        let strip = self.grid.strip11(self.grid.occ_plane(p), c);
        let mut s = [0u32; 11];
        let mut i = 0;
        while i < 11 {
            s[i] = strip[i] as u32;
            i += 1;
        }
        let mut out = 0u32;

        let cq = fold6(&s);
        for k in 0..WINDOW_LEN {
            out |= ((cq[5 - k] >> 5) & 1) << k;
        }

        let cr = run6(s[5]);
        for k in 0..WINDOW_LEN {
            out |= ((cr >> (5 - k)) & 1) << (6 + k);
        }

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

/// The only phase transition. Private, called from exactly one site.
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
    /// The involutive half of a placement (spec §5.4).
    fn place(&mut self, c: HexCoord, p: Player) {
        debug_assert!(self.grid.is_empty_cell(c));
        if self.grid.cover(c) > 0 {
            self.grid.clear_frontier(c);
        }
        self.grid.set_owner(c, p);
        self.grid.add_cover_disk(c);
        self.hash_cells ^= cell_key(c, p);
        self.stones_by[p.index()] += 1;
        self.history.push(Action::new(c));
    }

    /// The exact inverse of [`Position::place`], in reverse statement order.
    fn unplace(&mut self, c: HexCoord, p: Player) {
        let popped = self.history.pop();
        debug_assert_eq!(
            popped,
            Some(Action::new(c)),
            "C15: history top is not the placement being undone"
        );
        self.stones_by[p.index()] -= 1;
        self.hash_cells ^= cell_key(c, p);
        self.grid.remove_cover_disk(c);
        self.grid.clear_owner(c, p);
        if self.grid.cover(c) > 0 {
            self.grid.set_frontier(c);
        }
    }

    /// The single forward code path, called by [`Position::advance`] and [`Search::apply`].
    pub(crate) fn apply_raw(&mut self, action: Action) -> Result<(Applied, Undo), MoveError> {
        let c = action.coord();

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

        self.grid.reserve_around(c)?;

        #[cfg(debug_assertions)]
        let mut audit = UndoAudit::capture(self);
        self.place(c, player_before);

        let winning = self.winning_slots(c, player_before);
        let outcome = if winning != 0 {
            let o = Outcome {
                winner: player_before,
            };
            self.terminal = Some(o);
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

    /// Reverse one [`Position::apply_raw`].
    pub(crate) fn undo_raw(&mut self, u: Undo) {
        #[cfg(debug_assertions)]
        debug_assert_eq!(
            self.zobrist(),
            u.audit.zobrist_after,
            "C13: undo applied to the wrong position, or out of LIFO order"
        );

        self.phase = u.phase_before;
        self.current = u.player_before;
        self.terminal = None;
        self.unplace(u.action.coord(), u.player_before);

        #[cfg(debug_assertions)]
        {
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
    pub fn advance(&mut self, action: Action) -> Result<Applied, MoveError> {
        let (applied, _undo) = self.apply_raw(action)?;
        Ok(applied)
    }
}

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
        debug_assert!(self.grid.contains_padded(c), "C8: arena margin");
        debug_assert!(!self.grid.is_double_owned(c), "C3: double-owned cell");
        debug_assert_eq!(self.stone_count(), audit.stones_before + 1, "C10: stones");
        debug_assert_eq!(self.get(c), Some(mover), "C10: owner");
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
        let mut brute = 0u32;
        for (i, wr) in self.windows_through(c).iter().enumerate() {
            if wr.mask.is_full_for(mover) {
                brute |= 1 << i;
            }
        }
        debug_assert_eq!(brute, winning, "C12: win formulations disagree");
        self.debug_assert_frontier_around(c);
        self.debug_assert_turn_closed_form();
        debug_assert_eq!(
            self.legal_count() == 0,
            self.terminal.is_some(),
            "C6: legal_count/terminal disagree"
        );
    }
}

/// Which bit plane a [`BitScan`] walks.
#[derive(Clone, Copy, Debug)]
enum ScanPlane {
    /// The legal set.
    Frontier,
    /// The union of both occupancy planes.
    Occupied,
}

/// A canonical-order walk over one bit plane.
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
    #[inline]
    fn next_slot(&mut self) -> Option<(usize, u32)> {
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
                debug_assert!(false, "the maintained population count exceeds the plane");
                self.remaining = 0;
                return None;
            }
            self.cur = Self::word_at(self.grid, self.plane, self.word);
        }
    }

    /// The next set bit as a coordinate, for the consumer that does not need the slot.
    #[inline]
    fn next_coord(&mut self) -> Option<HexCoord> {
        let (word, bit) = self.next_slot()?;
        Some(self.grid.coord_of(word, bit))
    }

    /// The plane really is exhausted when `remaining` says so.
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

/// Build an [`IntegrityError`].
#[inline]
const fn fail<T>(check: IntegrityCheck, coord: Option<HexCoord>) -> Result<T, IntegrityError> {
    Err(IntegrityError { check, coord })
}

impl Position {
    /// Recompute every derived structure from the stones alone and compare.
    pub fn audit(&self) -> Result<(), IntegrityError> {
        let g = &self.grid;
        let total = g.total_words();
        let occ0 = g.occ_plane(Player::P0);
        let occ1 = g.occ_plane(Player::P1);

        let pop0: u32 = occ0.iter().map(|w| w.count_ones()).sum();
        let pop1: u32 = occ1.iter().map(|w| w.count_ones()).sum();
        if self.stone_count() != pop0 + pop1 {
            return fail(IntegrityCheck::StoneCount, None);
        }

        for i in 0..total {
            let both = occ0[i] & occ1[i];
            if both != 0 {
                return fail(
                    IntegrityCheck::DoubleOwned,
                    Some(g.coord_of(i, both.trailing_zeros())),
                );
            }
        }

        if self.stones_by[0] != pop0 || self.stones_by[1] != pop1 {
            return fail(IntegrityCheck::StoneCountForPlayer, None);
        }

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

        let pad = LEGAL_RADIUS as i32;
        let (lo_q, hi_q) = (g.origin_q(), g.origin_q() + g.rows() as i32 - 1);
        let (lo_r, hi_r) = (g.origin_r(), g.origin_r() + 64 * g.row_words() as i32 - 1);
        for &(c, _) in &stones {
            let (q, r) = (c.q as i32, c.r as i32);
            if q - pad < lo_q || q + pad > hi_q || r - pad < lo_r || r + pad > hi_r {
                return fail(IntegrityCheck::ArenaMargin, Some(c));
            }
        }

        let cells = total * 64;
        let mut recount = vec![0u8; cells];
        for &(s, _) in &stones {
            for dq in -(pad as i16)..=(pad as i16) {
                for dr in -(pad as i16)..=(pad as i16) {
                    let cell = HexCoord::new(s.q + dq, s.r + dr);
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

        let frontier = g.frontier_plane();
        for (i, &covered) in cover.iter().enumerate().take(cells) {
            let (word, bit) = (i / 64, (i % 64) as u32);
            let occupied = ((occ0[word] | occ1[word]) >> bit) & 1 == 1;
            let set = (frontier[word] >> bit) & 1 == 1;
            if set != (covered > 0 && !occupied) {
                return fail(IntegrityCheck::FrontierBit, Some(g.coord_of(word, bit)));
            }
        }

        let fpop: u32 = frontier.iter().map(|w| w.count_ones()).sum();
        if fpop != g.frontier_cells() {
            return fail(IntegrityCheck::FrontierCount, None);
        }

        let mut h = 0u64;
        for &(c, p) in &stones {
            h ^= cell_key(c, p);
        }
        if h != self.hash_cells {
            return fail(IntegrityCheck::Zobrist, None);
        }

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

        if let Some(o) = self.terminal
            && (!winners[o.winner.index()] || winners[o.winner.other().index()])
        {
            return fail(IntegrityCheck::Winner, None);
        }

        match turn_closed_form(self.stone_count(), self.terminal.is_some()) {
            Some((kind, player)) if kind == self.phase.kind_index() && player == self.current => {}
            _ => return fail(IntegrityCheck::TurnClosedForm, None),
        }

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
