//! The dense recentred arena. **Private: zero items escape the crate.**
//!
//! Ruling 2's "grid geometry is entirely private" is enforced by the module
//! system, not by discipline: this module is `mod grid;`, and **no public item
//! in this crate may ever expose a row, a word, a plane, a stride, or an
//! index.** That constraint is what makes the arena replaceable (spec §5.8)
//! without touching a caller.
//!
//! Four planes over the same `rows × row_words × 64` cell box:
//!
//! - `occ[0]`, `occ[1]` — one bit per cell, stones of `P0` / `P1`.
//! - `frontier`         — one bit per cell, empty cells with `cover > 0`.
//! - `cover`            — one **byte** per cell, stones within `LEGAL_RADIUS`.
//!
//! `cover` is a count, not a bit, because it is the only structure that makes
//! the frontier exactly invertible: an OR of radius-8 disks cannot be undone,
//! but `+1` on apply and `-1` on undo is self-inverse by construction.
//!
//! > **The frontier invariant.** `frontier[c] == 1` if and only if
//! > `cover[c] > 0 && occ[0][c] == 0 && occ[1][c] == 0`.
//!
//! Cell `(q, r)` maps to row `q - origin_q` and bit `r - origin_r`. The layout
//! is `q`-major and `r`-minor, so **storage order is ascending `(q, r)`** — the
//! canonical order of spec §9 — and enumeration needs no sort.
//!
//! All index arithmetic is performed in `i32`, so a coordinate anywhere in the
//! `i16` range produces an in-range-or-out-of-range answer and never wraps.

use crate::MAX_GRID_CELLS;
use crate::coord::{HexCoord, LEGAL_RADIUS};
use crate::error::MoveError;
use crate::player::Player;

/// Rows the arena starts with: `q` spans 32 values.
const MIN_ROWS: usize = 32;

/// Words per row the arena starts with: `r` spans 128 values.
const MIN_ROW_WORDS: usize = 2;

/// Margin, in cells, kept between any written cell and the arena boundary.
///
/// 8, not 5, because the coverage disk — not the win strip — is the widest
/// write. Padding by 8 also guarantees every window containing an occupied cell
/// is fully in-arena.
const PAD: i32 = LEGAL_RADIUS as i32;

/// Round down to a multiple of 64. Two's complement: `-1 -> -64`, `63 -> 0`.
#[inline]
const fn floor64(x: i32) -> i32 {
    x & !63
}

/// The dense recentred arena.
///
/// Deliberately implements **neither `PartialEq` nor `Hash`**. That is a
/// compile-time trap, not a comment: a future `#[derive(PartialEq)]` on
/// `Position` fails to build, forcing whoever adds it to write the
/// content-based impl (spec §5.7).
#[derive(Clone, Debug)]
pub(crate) struct Grid {
    /// Extent along `q`.
    rows: usize,
    /// `u64` words per row; the extent along `r` is `64 * row_words`.
    row_words: usize,
    /// `q` of row 0.
    origin_q: i32,
    /// `r` of bit 0. Always a multiple of 64.
    origin_r: i32,
    /// Occupancy bit planes, `rows * row_words` words each.
    occ: [Vec<u64>; 2],
    /// Empty cells with `cover > 0`, `rows * row_words` words.
    frontier: Vec<u64>,
    /// Stones within [`LEGAL_RADIUS`], `rows * row_words * 64` bytes.
    cover: Vec<u8>,
    /// Maintained `popcount(frontier)`.
    ///
    /// Purely **geometric**: it is emphatically not zero in a terminal
    /// position. `Position::legal_count` is the rule-level answer.
    frontier_cells: u32,
}

impl Grid {
    /// The empty arena. Allocates nothing; every read is out of range and
    /// answers empty.
    pub(crate) const fn new() -> Self {
        Self {
            rows: 0,
            row_words: 0,
            origin_q: 0,
            origin_r: 0,
            occ: [Vec::new(), Vec::new()],
            frontier: Vec::new(),
            cover: Vec::new(),
            frontier_cells: 0,
        }
    }

    // ---- geometry, crate-private -----------------------------------------

    /// Rows currently allocated.
    #[inline]
    pub(crate) const fn rows(&self) -> usize {
        self.rows
    }

    /// `u64` words per row.
    #[inline]
    pub(crate) const fn row_words(&self) -> usize {
        self.row_words
    }

    /// `q` of row 0.
    #[inline]
    pub(crate) const fn origin_q(&self) -> i32 {
        self.origin_q
    }

    /// `r` of bit 0.
    #[inline]
    pub(crate) const fn origin_r(&self) -> i32 {
        self.origin_r
    }

    /// Total words in one bit plane.
    #[inline]
    pub(crate) const fn total_words(&self) -> usize {
        self.rows * self.row_words
    }

    /// One occupancy plane.
    #[inline]
    pub(crate) fn occ_plane(&self, p: Player) -> &[u64] {
        &self.occ[p.index()]
    }

    /// The frontier plane.
    #[inline]
    pub(crate) fn frontier_plane(&self) -> &[u64] {
        &self.frontier
    }

    /// The coverage plane.
    #[inline]
    pub(crate) fn cover_plane(&self) -> &[u8] {
        &self.cover
    }

    /// Maintained frontier population count.
    #[inline]
    pub(crate) const fn frontier_cells(&self) -> u32 {
        self.frontier_cells
    }

    /// Word `i` of the union of both occupancy planes.
    #[inline]
    pub(crate) fn occupied_word(&self, i: usize) -> u64 {
        self.occ[0][i] | self.occ[1][i]
    }

    /// The cell a `(word, bit)` slot addresses.
    #[inline]
    pub(crate) fn coord_of(&self, word: usize, bit: u32) -> HexCoord {
        let row = word / self.row_words;
        let w = word % self.row_words;
        HexCoord::new(
            (self.origin_q + row as i32) as i16,
            (self.origin_r + (w as i32) * 64 + bit as i32) as i16,
        )
    }

    /// `(word index, bit)` of `c`, or `None` if `c` is outside the arena.
    #[inline]
    fn locate(&self, c: HexCoord) -> Option<(usize, u32)> {
        if self.rows == 0 {
            return None;
        }
        let row = c.q as i32 - self.origin_q;
        if row < 0 || row >= self.rows as i32 {
            return None;
        }
        let bit = c.r as i32 - self.origin_r;
        if bit < 0 || bit >= 64 * self.row_words as i32 {
            return None;
        }
        Some((
            row as usize * self.row_words + (bit >> 6) as usize,
            (bit & 63) as u32,
        ))
    }

    /// `locate`, panicking. Only ever called on the write path, which runs
    /// after `reserve_around`.
    #[inline]
    fn locate_written(&self, c: HexCoord) -> (usize, u32) {
        match self.locate(c) {
            Some(x) => x,
            None => unreachable!("arena write outside the reserved region"),
        }
    }

    // ---- occupancy --------------------------------------------------------

    /// Owner of `c`, or `None`. Total over every coordinate.
    #[inline]
    pub(crate) fn owner(&self, c: HexCoord) -> Option<Player> {
        let (w, b) = self.locate(c)?;
        if (self.occ[0][w] >> b) & 1 == 1 {
            Some(Player::P0)
        } else if (self.occ[1][w] >> b) & 1 == 1 {
            Some(Player::P1)
        } else {
            None
        }
    }

    /// Whether both occupancy planes claim `c`. Always false in a sound arena.
    ///
    /// Read only by the tier-C debug assertions and by tests, so it is dead
    /// code in a release build — which `cargo clippy --release` reports.
    #[inline]
    #[cfg_attr(not(debug_assertions), allow(dead_code))]
    pub(crate) fn is_double_owned(&self, c: HexCoord) -> bool {
        match self.locate(c) {
            Some((w, b)) => ((self.occ[0][w] & self.occ[1][w]) >> b) & 1 == 1,
            None => false,
        }
    }

    /// Flat cell index of `c` within the byte-per-cell planes, or `None` if `c`
    /// is outside the arena.
    #[inline]
    pub(crate) fn cell_index(&self, c: HexCoord) -> Option<usize> {
        let (w, b) = self.locate(c)?;
        Some(w * 64 + b as usize)
    }

    /// Whether `c` holds no stone. Total over every coordinate.
    #[inline]
    pub(crate) fn is_empty_cell(&self, c: HexCoord) -> bool {
        match self.locate(c) {
            Some((w, b)) => ((self.occ[0][w] | self.occ[1][w]) >> b) & 1 == 0,
            None => true,
        }
    }

    /// Set `c` to `p`'s stone.
    #[inline]
    pub(crate) fn set_owner(&mut self, c: HexCoord, p: Player) {
        let (w, b) = self.locate_written(c);
        self.occ[p.index()][w] |= 1 << b;
    }

    /// Clear `p`'s stone from `c`.
    #[inline]
    pub(crate) fn clear_owner(&mut self, c: HexCoord, p: Player) {
        let (w, b) = self.locate_written(c);
        self.occ[p.index()][w] &= !(1u64 << b);
    }

    // ---- coverage and frontier -------------------------------------------

    /// Stones within [`LEGAL_RADIUS`] of `c`. Total: `0` outside the arena.
    #[inline]
    pub(crate) fn cover(&self, c: HexCoord) -> u8 {
        match self.locate(c) {
            Some((w, b)) => self.cover[w * 64 + b as usize],
            None => 0,
        }
    }

    /// `cover(c) += 1`.
    #[inline]
    pub(crate) fn inc_cover(&mut self, c: HexCoord) {
        let (w, b) = self.locate_written(c);
        let i = w * 64 + b as usize;
        debug_assert!(
            (self.cover[i] as usize) < crate::coord::DISK_CELLS,
            "C1: coverage overflow"
        );
        self.cover[i] += 1;
    }

    /// `cover(c) -= 1`.
    #[inline]
    pub(crate) fn dec_cover(&mut self, c: HexCoord) {
        let (w, b) = self.locate_written(c);
        let i = w * 64 + b as usize;
        debug_assert!(self.cover[i] > 0, "C1: coverage underflow");
        self.cover[i] -= 1;
    }

    /// Whether `c`'s frontier bit is set. Total: `false` outside the arena.
    ///
    /// Read only by the tier-C debug assertions and by tests, so it is dead
    /// code in a release build — which `cargo clippy --release` reports.
    #[inline]
    #[cfg_attr(not(debug_assertions), allow(dead_code))]
    pub(crate) fn frontier_bit(&self, c: HexCoord) -> bool {
        match self.locate(c) {
            Some((w, b)) => (self.frontier[w] >> b) & 1 == 1,
            None => false,
        }
    }

    /// Set `c`'s frontier bit, maintaining `frontier_cells`. Idempotent.
    #[inline]
    pub(crate) fn set_frontier(&mut self, c: HexCoord) {
        let (w, b) = self.locate_written(c);
        if (self.frontier[w] >> b) & 1 == 0 {
            self.frontier[w] |= 1 << b;
            self.frontier_cells += 1;
        }
    }

    /// Clear `c`'s frontier bit, maintaining `frontier_cells`. Idempotent.
    #[inline]
    pub(crate) fn clear_frontier(&mut self, c: HexCoord) {
        let (w, b) = self.locate_written(c);
        if (self.frontier[w] >> b) & 1 == 1 {
            self.frontier[w] &= !(1u64 << b);
            self.frontier_cells -= 1;
        }
    }

    // ---- the 11x11 strip --------------------------------------------------

    /// `strip[i]` bit `j` = `plane` bit at `(c.q - 5 + i, c.r - 5 + j)`, for
    /// `i, j` in `0..11`. Bits 11..16 are always zero; `c` is at row 5, bit 5.
    ///
    /// Total: rows and bits outside the arena read as zero, which is what makes
    /// the public window queries total. On the internal path the padding-8
    /// invariant guarantees everything is in-arena and the fast path is taken.
    pub(crate) fn strip11(&self, plane: &[u64], c: HexCoord) -> [u16; 11] {
        let mut out = [0u16; 11];
        if self.rows == 0 {
            return out;
        }
        let base_q = c.q as i32 - 5;
        let base_r = c.r as i32 - 5;
        let total_bits = 64 * self.row_words as i32;
        for (i, slot) in out.iter_mut().enumerate() {
            let row = base_q + i as i32 - self.origin_q;
            if row < 0 || row >= self.rows as i32 {
                continue;
            }
            let row_base = row as usize * self.row_words;
            let bit = base_r - self.origin_r;
            *slot = if bit >= 0 && bit + 11 <= total_bits {
                // Fast path: one or two word loads.
                let w = (bit >> 6) as usize;
                let sh = (bit & 63) as u32;
                let mut v = plane[row_base + w] >> sh;
                if sh + 11 > 64 {
                    v |= plane[row_base + w + 1] << (64 - sh);
                }
                (v & 0x7FF) as u16
            } else {
                // Clamped path: the strip hangs off an arena edge.
                let mut v = 0u16;
                for j in 0..11i32 {
                    let b = bit + j;
                    if b < 0 || b >= total_bits {
                        continue;
                    }
                    let word = plane[row_base + (b >> 6) as usize];
                    if (word >> (b & 63)) & 1 == 1 {
                        v |= 1 << j;
                    }
                }
                v
            };
        }
        out
    }

    // ---- growth -----------------------------------------------------------

    /// Whether `[c.q ± PAD] × [c.r ± PAD]` is already inside the arena.
    #[inline]
    pub(crate) fn contains_padded(&self, c: HexCoord) -> bool {
        if self.rows == 0 {
            return false;
        }
        let (cq, cr) = (c.q as i32, c.r as i32);
        cq - PAD >= self.origin_q
            && cq + PAD < self.origin_q + self.rows as i32
            && cr - PAD >= self.origin_r
            && cr + PAD < self.origin_r + 64 * self.row_words as i32
    }

    /// Bounding box of the stones actually on the board, `(lo_q, hi_q, lo_r,
    /// hi_r)`, or `None` when no stone has been placed.
    ///
    /// **This is the only input the growth policy takes from the arena**, and
    /// it is *content*, not geometry: it is unchanged by how large the arena
    /// happens to have grown. That is what makes capacity decisions — and
    /// therefore [`MoveError::BoardExtentExceeded`] — a function of the
    /// position rather than of its search history.
    ///
    /// Recomputed from the occupancy planes rather than cached, because a cache
    /// would have to be restored by `undo` to preserve exactly the property
    /// this function exists to provide. It is read only on the growth path,
    /// which runs a logarithmic number of times per game.
    fn stone_bounds(&self) -> Option<(i32, i32, i32, i32)> {
        let (mut lo_q, mut hi_q) = (i32::MAX, i32::MIN);
        let (mut lo_r, mut hi_r) = (i32::MAX, i32::MIN);
        for row in 0..self.rows {
            let base = row * self.row_words;
            let mut any = false;
            for w in 0..self.row_words {
                let bits = self.occ[0][base + w] | self.occ[1][base + w];
                if bits == 0 {
                    continue;
                }
                any = true;
                let word_r = self.origin_r + (w as i32) * 64;
                lo_r = lo_r.min(word_r + bits.trailing_zeros() as i32);
                hi_r = hi_r.max(word_r + 63 - bits.leading_zeros() as i32);
            }
            if any {
                let q = self.origin_q + row as i32;
                lo_q = lo_q.min(q);
                hi_q = hi_q.max(q);
            }
        }
        if hi_q == i32::MIN {
            None
        } else {
            Some((lo_q, hi_q, lo_r, hi_r))
        }
    }

    /// Grow, if needed, so `[c.q ± PAD] × [c.r ± PAD]` is inside the arena.
    ///
    /// Class III (spec §7.1): reallocation only, no observable change.
    ///
    /// > **The shape is a function of the stones, never of the allocation
    /// > history.** The required box is the live stone box padded by `PAD`,
    /// > unioned with the requested cell padded by `PAD`. The refusal predicate
    /// > is "the smallest arena holding that box is over
    /// > [`crate::MAX_GRID_CELLS`]". Two positions with the same stones
    /// > therefore accept and refuse exactly the same placements, however much
    /// > either one's arena grew getting there — a search that applies a wide
    /// > line and unwinds it does not consume the position's extent budget.
    ///
    /// Each dimension is sized independently: a dimension with enough capacity
    /// is left alone, so a straight walk along `q` keeps `row_words` at its
    /// minimum instead of quadrupling the arena on every step. The deficient
    /// dimension doubles, which keeps growth amortised O(1) per ply.
    ///
    /// `origin_r` only ever moves by multiples of 64, so **every row copy is a
    /// word-aligned `memcpy`; there is never a bit-shifted copy.** That
    /// eliminates the entire class of off-by-one-bit growth bugs, which are
    /// symmetric and therefore invisible to round-trip tests.
    ///
    /// The arena may be *re-shaped* rather than merely extended, so the copy
    /// moves the live region — the padded stone box, outside which every plane
    /// is zero — and not the whole old allocation.
    ///
    /// # Errors
    /// [`MoveError::BoardExtentExceeded`] — checked **before** allocating, so a
    /// refusal leaves the arena untouched.
    pub(crate) fn reserve_around(&mut self, c: HexCoord) -> Result<(), MoveError> {
        if self.contains_padded(c) {
            return Ok(());
        }
        let (cq, cr) = (c.q as i32, c.r as i32);
        let bounds = self.stone_bounds();

        // 1. The required box: the requested cell padded by PAD, unioned with
        //    the live stone box padded by PAD. Content only — the current
        //    arena extent is deliberately not part of this.
        let (mut lo_q, mut hi_q) = (cq - PAD, cq + PAD);
        let (mut lo_r, mut hi_r) = (cr - PAD, cr + PAD);
        if let Some((sq0, sq1, sr0, sr1)) = bounds {
            lo_q = lo_q.min(sq0 - PAD);
            hi_q = hi_q.max(sq1 + PAD);
            lo_r = lo_r.min(sr0 - PAD);
            hi_r = hi_r.max(sr1 + PAD);
        }
        let need_rows = (hi_q - lo_q + 1) as usize;
        // Exact, not slack: `origin_r` is a multiple of 64, so the box starts
        // at `floor64(lo_r)` and needs exactly this many words to reach `hi_r`.
        let base_r = floor64(lo_r);
        let need_words = ((hi_r - base_r) as usize / 64) + 1;

        // 2. Refuse before allocating, on the *smallest* arena that could hold
        //    the required box. Any arena holding that box is at least this
        //    large, so no reachable geometry can accept a placement this test
        //    refuses, nor refuse one it accepts.
        let least_rows = MIN_ROWS.max(need_rows);
        let least_words = MIN_ROW_WORDS.max(need_words);
        let least_cells = least_rows as u64 * least_words as u64 * 64;
        if least_cells > MAX_GRID_CELLS {
            return Err(MoveError::BoardExtentExceeded { cells: least_cells });
        }

        // 3. Size each dimension independently, doubling only where capacity is
        //    actually short.
        let fits = |rows: usize, words: usize| rows as u64 * words as u64 * 64 <= MAX_GRID_CELLS;
        let bump = |have: usize, need: usize, min: usize| {
            let want = if need > have {
                // Short: double, so growth stays amortised O(1) per ply.
                (2 * have).max(need).next_power_of_two()
            } else {
                // Not short: keep what is already allocated, but never more
                // than 4x what the content needs. A search that reached far out
                // and was rewound therefore cannot leave the position holding a
                // permanently oversized arena — `clone` has to stay cheap.
                have.min(need.next_power_of_two().saturating_mul(4))
            };
            min.max(want)
        };
        let mut new_rows = bump(self.rows, need_rows, MIN_ROWS);
        let mut new_words = bump(self.row_words, need_words, MIN_ROW_WORDS);
        if !fits(new_rows, new_words) {
            // Near the ceiling the slack has to come from somewhere: give `r`
            // only what the content needs, then spend the rest of the budget on
            // `q`. Never below `least_*`, which step 2 proved fits.
            new_words = MIN_ROW_WORDS.max(need_words.next_power_of_two());
            if !fits(least_rows, new_words) {
                new_words = least_words;
            }
            let budget = (MAX_GRID_CELLS / (64 * new_words as u64)) as usize;
            new_rows = new_rows.min(budget).max(least_rows);
        }
        debug_assert!(fits(new_rows, new_words), "chosen shape breaks the ceiling");
        debug_assert!(new_rows >= need_rows && new_words >= need_words);

        // 4. Re-centre the required box. `origin_r` stays a multiple of 64.
        let new_origin_q = lo_q - ((new_rows - need_rows) / 2) as i32;
        let new_origin_r = base_r - 64 * ((new_words - need_words) / 2) as i32;

        // 5. Allocate zeroed, then copy the live region. Every plane is zero
        //    outside the padded stone box, so that box is the whole of what has
        //    to move — and copying it, rather than the old allocation, is what
        //    lets step 3 hand a dimension back its unused capacity.
        let words = new_rows * new_words;
        let mut occ0 = vec![0u64; words];
        let mut occ1 = vec![0u64; words];
        let mut frontier = vec![0u64; words];
        let mut cover = vec![0u8; words * 64];

        if let Some((sq0, sq1, sr0, sr1)) = bounds {
            // Clamped to the old arena. The padding invariant makes the clamp a
            // no-op for any position the rule machine can produce; clamping
            // anyway keeps the copy in bounds unconditionally rather than by
            // appeal to an invariant enforced somewhere else.
            let live_lo_q = (sq0 - PAD).max(self.origin_q);
            let live_hi_q = (sq1 + PAD).min(self.origin_q + self.rows as i32 - 1);
            let live_base_r = floor64((sr0 - PAD).max(self.origin_r));
            let live_hi_r = (sr1 + PAD).min(self.origin_r + 64 * self.row_words as i32 - 1);
            let n_rows = (live_hi_q - live_lo_q + 1) as usize;
            let n_words = ((live_hi_r - live_base_r) as usize / 64) + 1;

            debug_assert_eq!((live_base_r - self.origin_r) % 64, 0);
            let src_row0 = (live_lo_q - self.origin_q) as usize;
            let src_word0 = ((live_base_r - self.origin_r) / 64) as usize;
            let dst_row0 = (live_lo_q - new_origin_q) as usize;
            let dst_word0 = ((live_base_r - new_origin_r) / 64) as usize;
            debug_assert!(src_row0 + n_rows <= self.rows && src_word0 + n_words <= self.row_words);
            debug_assert!(dst_row0 + n_rows <= new_rows && dst_word0 + n_words <= new_words);

            for i in 0..n_rows {
                let src = (src_row0 + i) * self.row_words + src_word0;
                let dst = (dst_row0 + i) * new_words + dst_word0;
                occ0[dst..dst + n_words].copy_from_slice(&self.occ[0][src..src + n_words]);
                occ1[dst..dst + n_words].copy_from_slice(&self.occ[1][src..src + n_words]);
                frontier[dst..dst + n_words].copy_from_slice(&self.frontier[src..src + n_words]);
                let (bsrc, bdst, n) = (src * 64, dst * 64, n_words * 64);
                cover[bdst..bdst + n].copy_from_slice(&self.cover[bsrc..bsrc + n]);
            }
        }

        self.rows = new_rows;
        self.row_words = new_words;
        self.origin_q = new_origin_q;
        self.origin_r = new_origin_r;
        self.occ = [occ0, occ1];
        self.frontier = frontier;
        self.cover = cover;

        // C9: containment must hold after every growth.
        debug_assert!(
            self.contains_padded(c),
            "C9: reserve_around failed to contain the requested cell"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grow(g: &mut Grid, q: i16, r: i16) {
        g.reserve_around(HexCoord::new(q, r)).expect("growth");
    }

    /// Reserve around `(q, r)` and put a stone there, the way `Position` does.
    fn place(g: &mut Grid, q: i16, r: i16) {
        grow(g, q, r);
        g.set_owner(HexCoord::new(q, r), Player::P0);
    }

    fn cells(g: &Grid) -> u64 {
        g.rows() as u64 * g.row_words() as u64 * 64
    }

    #[test]
    fn empty_grid_allocates_nothing_and_reads_empty() {
        let g = Grid::new();
        assert_eq!(g.rows(), 0);
        assert_eq!(g.row_words(), 0);
        assert_eq!(g.total_words(), 0);
        assert_eq!(g.frontier_cells(), 0);
        assert!(g.is_empty_cell(HexCoord::ORIGIN));
        assert_eq!(g.owner(HexCoord::new(500, -500)), None);
        assert_eq!(g.cover(HexCoord::ORIGIN), 0);
        assert!(!g.frontier_bit(HexCoord::ORIGIN));
        assert_eq!(
            g.strip11(g.occ_plane(Player::P0), HexCoord::ORIGIN),
            [0; 11]
        );
    }

    #[test]
    fn first_growth_reaches_the_documented_minimum() {
        let mut g = Grid::new();
        grow(&mut g, 0, 0);
        assert_eq!(g.rows(), MIN_ROWS);
        assert_eq!(g.row_words(), MIN_ROW_WORDS);
        assert_eq!(g.origin_r(), -64);
        assert_eq!(g.origin_r() % 64, 0);
        assert!(g.contains_padded(HexCoord::ORIGIN));
    }

    #[test]
    fn growth_in_each_of_four_directions_keeps_origin_r_aligned() {
        for (dq, dr) in [(400i16, 0i16), (-400, 0), (0, 400), (0, -400)] {
            let mut g = Grid::new();
            grow(&mut g, 0, 0);
            g.set_owner(HexCoord::ORIGIN, Player::P0);
            g.inc_cover(HexCoord::ORIGIN);
            g.set_frontier(HexCoord::new(1, 0));
            grow(&mut g, dq, dr);
            assert_eq!(g.origin_r() % 64, 0, "origin_r misaligned for {dq},{dr}");
            assert_eq!(g.owner(HexCoord::ORIGIN), Some(Player::P0));
            assert_eq!(g.cover(HexCoord::ORIGIN), 1);
            assert!(g.frontier_bit(HexCoord::new(1, 0)));
            assert!(g.contains_padded(HexCoord::new(dq, dr)));
            assert!(g.contains_padded(HexCoord::ORIGIN));
        }
    }

    #[test]
    fn grown_arena_reads_back_every_written_cell_and_zero_elsewhere() {
        let mut g = Grid::new();
        let written = [(0i16, 0i16), (3, -2), (-5, 7), (9, 9), (-11, -1)];
        for &(q, r) in &written {
            grow(&mut g, q, r);
            let c = HexCoord::new(q, r);
            g.set_owner(c, if q % 2 == 0 { Player::P0 } else { Player::P1 });
            g.inc_cover(c);
            g.inc_cover(c);
        }
        // Force several reallocations in both directions.
        for &(q, r) in &[(300i16, 300i16), (-300, -300), (300, -300), (-300, 300)] {
            grow(&mut g, q, r);
            for &(wq, wr) in &written {
                let c = HexCoord::new(wq, wr);
                assert_eq!(
                    g.owner(c),
                    Some(if wq % 2 == 0 { Player::P0 } else { Player::P1 })
                );
                assert_eq!(g.cover(c), 2);
            }
            // Newly reachable territory is zeroed.
            assert_eq!(g.cover(HexCoord::new(q, r)), 0);
            assert!(g.is_empty_cell(HexCoord::new(q, r)));
            assert!(!g.frontier_bit(HexCoord::new(q, r)));
        }
    }

    #[test]
    fn frontier_counter_tracks_the_plane() {
        let mut g = Grid::new();
        grow(&mut g, 0, 0);
        assert_eq!(g.frontier_cells(), 0);
        g.set_frontier(HexCoord::new(1, 1));
        assert_eq!(g.frontier_cells(), 1);
        g.set_frontier(HexCoord::new(1, 1)); // idempotent
        assert_eq!(g.frontier_cells(), 1);
        g.set_frontier(HexCoord::new(2, 1));
        assert_eq!(g.frontier_cells(), 2);
        g.clear_frontier(HexCoord::new(1, 1));
        assert_eq!(g.frontier_cells(), 1);
        g.clear_frontier(HexCoord::new(1, 1)); // idempotent
        assert_eq!(g.frontier_cells(), 1);
        let pop: u32 = g.frontier_plane().iter().map(|w| w.count_ones()).sum();
        assert_eq!(pop, g.frontier_cells());
    }

    #[test]
    fn max_grid_cells_is_refused_before_allocating() {
        let mut g = Grid::new();
        place(&mut g, 0, 0);
        let before_rows = g.rows();
        let before_words = g.row_words();
        // A box spanning ~9000 in both directions is far past 1 << 22 cells.
        let err = g
            .reserve_around(HexCoord::new(9000, 9000))
            .expect_err("must refuse");
        match err {
            MoveError::BoardExtentExceeded { cells } => assert!(cells > MAX_GRID_CELLS),
            other => panic!("wrong error: {other:?}"),
        }
        assert_eq!(g.rows(), before_rows, "arena mutated on refusal");
        assert_eq!(g.row_words(), before_words, "arena mutated on refusal");
    }

    /// Regression: the growth policy grew *both* dimensions on every event, so
    /// the arena quadrupled per growth, its aspect ratio froze at 32:128, and a
    /// straight walk along `q` blew the ceiling after six allocations.
    #[test]
    fn a_q_only_walk_never_widens_r() {
        let mut g = Grid::new();
        place(&mut g, 0, 0);
        let mut q = 0i16;
        for _ in 0..600 {
            q += 8;
            place(&mut g, q, 0);
            assert_eq!(
                g.row_words(),
                MIN_ROW_WORDS,
                "r widened at q = {q} for a walk that never leaves r = 0"
            );
            assert!(cells(&g) <= MAX_GRID_CELLS);
        }
        // A tall, one-word-wide arena, not a square one.
        assert!(g.rows() >= (q as usize) + 2 * PAD as usize);
        assert!(
            cells(&g) <= MAX_GRID_CELLS / 4,
            "{} cells for a one-row game",
            cells(&g)
        );
        // Every stone survived the reshaping.
        for k in 0..=(q / 8) {
            assert_eq!(g.owner(HexCoord::new(k * 8, 0)), Some(Player::P0));
        }
    }

    /// The mirror walk. `(q, r) -> (r, q)` is a symmetry of the rules, so the
    /// two walks must survive equally far; the old policy refused the `q` walk
    /// 4.5x sooner.
    #[test]
    fn q_and_r_walks_reach_the_same_extent() {
        fn walk(along_q: bool) -> usize {
            let mut g = Grid::new();
            place(&mut g, 0, 0);
            let mut n = 0usize;
            for k in 1..2000i16 {
                let (q, r) = if along_q { (k * 8, 0) } else { (0, k * 8) };
                if g.reserve_around(HexCoord::new(q, r)).is_err() {
                    break;
                }
                g.set_owner(HexCoord::new(q, r), Player::P0);
                n += 1;
            }
            n
        }
        // Neither direction may trip the ceiling inside the coordinate range a
        // real game can reach; `COORD_LIMIT` is the binding constraint.
        assert_eq!(walk(true), 1999, "the q walk hit the arena ceiling");
        assert_eq!(walk(false), 1999, "the r walk hit the arena ceiling");
    }

    /// The refusal predicate reads the stones, not the allocation history: an
    /// arena that grew large and then lost those stones must decide exactly as
    /// a fresh one holding the same stones does.
    #[test]
    fn the_ceiling_is_a_function_of_the_stones_not_of_past_growth() {
        // `inflated` is driven far out along q, then the stones are removed.
        let mut inflated = Grid::new();
        place(&mut inflated, 0, 0);
        let mut q = 0i16;
        for _ in 0..400 {
            q += 8;
            place(&mut inflated, q, 0);
        }
        for k in 1..=(q / 8) {
            inflated.clear_owner(HexCoord::new(k * 8, 0), Player::P0);
        }
        assert!(
            inflated.rows() > 1000,
            "the excursion did not grow the arena"
        );

        let mut fresh = Grid::new();
        place(&mut fresh, 0, 0);

        // Now push both, in lockstep, all the way to the ceiling along a
        // diagonal that grows the box in both dimensions.
        let (mut q, mut r) = (0i16, 0i16);
        let mut refused = false;
        for step in 0..800 {
            if step % 2 == 0 {
                q += 8;
            } else {
                r += 8;
            }
            let c = HexCoord::new(q, r);
            let a = inflated.reserve_around(c);
            let b = fresh.reserve_around(c);
            assert_eq!(
                a.is_err(),
                b.is_err(),
                "grown and fresh arenas disagree at ({q}, {r}): {a:?} vs {b:?}"
            );
            if a.is_err() {
                assert_eq!(a, b, "different refusals at ({q}, {r})");
                refused = true;
                break;
            }
            inflated.set_owner(c, Player::P0);
            fresh.set_owner(c, Player::P0);
        }
        assert!(refused, "the diagonal never reached the ceiling");
    }

    #[test]
    fn strip11_places_the_query_cell_at_row_5_bit_5() {
        let mut g = Grid::new();
        grow(&mut g, 0, 0);
        let c = HexCoord::new(2, -3);
        g.set_owner(c, Player::P0);
        let s = g.strip11(g.occ_plane(Player::P0), c);
        assert_eq!(s[5], 1 << 5);
        for (i, row) in s.iter().enumerate() {
            if i != 5 {
                assert_eq!(*row, 0);
            }
        }
    }

    #[test]
    fn strip11_reads_the_whole_11x11_neighbourhood() {
        let mut g = Grid::new();
        let c = HexCoord::new(0, 0);
        grow(&mut g, 0, 0);
        for dq in -5i16..=5 {
            for dr in -5i16..=5 {
                let cell = HexCoord::new(c.q + dq, c.r + dr);
                grow(&mut g, cell.q, cell.r);
            }
        }
        let mut expect = [0u16; 11];
        for dq in -5i16..=5 {
            for dr in -5i16..=5 {
                if (dq + dr) % 3 != 0 {
                    continue;
                }
                g.set_owner(HexCoord::new(c.q + dq, c.r + dr), Player::P1);
                expect[(dq + 5) as usize] |= 1 << (dr + 5);
            }
        }
        assert_eq!(g.strip11(g.occ_plane(Player::P1), c), expect);
        assert_eq!(g.strip11(g.occ_plane(Player::P0), c), [0u16; 11]);
    }

    #[test]
    fn strip11_is_total_off_the_arena_edge() {
        let mut g = Grid::new();
        grow(&mut g, 0, 0);
        // Far outside the arena in every direction: all zero, no panic.
        for c in [
            HexCoord::new(9000, 0),
            HexCoord::new(-9000, 0),
            HexCoord::new(0, 9000),
            HexCoord::new(0, -9000),
        ] {
            assert_eq!(g.strip11(g.occ_plane(Player::P0), c), [0u16; 11]);
        }
        // Straddling the edge: the in-arena part still reads correctly.
        let edge_q = (g.origin_q() + g.rows() as i32 - 1) as i16;
        g.set_owner(HexCoord::new(edge_q, 0), Player::P0);
        let s = g.strip11(g.occ_plane(Player::P0), HexCoord::new(edge_q, 0));
        assert_eq!(s[5], 1 << 5);
        let edge_r = (g.origin_r() + 64 * g.row_words() as i32 - 1) as i16;
        g.set_owner(HexCoord::new(0, edge_r), Player::P1);
        let s = g.strip11(g.occ_plane(Player::P1), HexCoord::new(0, edge_r));
        assert_eq!(s[5], 1 << 5);
    }

    #[test]
    fn strip11_fast_and_clamped_paths_agree_across_word_boundaries() {
        let mut g = Grid::new();
        grow(&mut g, 0, 200);
        // Sprinkle stones, then compare the word-load path against a
        // per-cell recomputation at every bit offset within a word.
        for q in -6i16..=6 {
            for r in 180i16..=220 {
                if (q as i32 * 7 + r as i32 * 3) % 5 == 0 {
                    grow(&mut g, q, r);
                    g.set_owner(HexCoord::new(q, r), Player::P0);
                }
            }
        }
        for r in 190i16..=210 {
            let c = HexCoord::new(0, r);
            let s = g.strip11(g.occ_plane(Player::P0), c);
            for (i, &row) in s.iter().enumerate() {
                for j in 0..11usize {
                    let cell = HexCoord::new(c.q - 5 + i as i16, c.r - 5 + j as i16);
                    let expect = g.owner(cell) == Some(Player::P0);
                    assert_eq!((row >> j) & 1 == 1, expect, "at {cell:?}");
                }
            }
        }
    }

    #[test]
    fn coord_of_inverts_locate() {
        let mut g = Grid::new();
        grow(&mut g, 5, -70);
        for q in -10i16..=10 {
            for r in -100i16..=0 {
                let c = HexCoord::new(q, r);
                if let Some((w, b)) = g.locate(c) {
                    assert_eq!(g.coord_of(w, b), c);
                }
            }
        }
    }

    #[test]
    fn floor64_rounds_toward_negative_infinity() {
        assert_eq!(floor64(0), 0);
        assert_eq!(floor64(63), 0);
        assert_eq!(floor64(64), 64);
        assert_eq!(floor64(-1), -64);
        assert_eq!(floor64(-64), -64);
        assert_eq!(floor64(-65), -128);
    }

    #[test]
    fn repeated_growth_never_shrinks_or_loses_alignment() {
        let mut g = Grid::new();
        let mut q = 0i16;
        let mut r = 0i16;
        let mut prev_cells = 0u64;
        let mut placed = Vec::new();
        for step in 0..60 {
            // Stones make the live box monotone, which is what a real game
            // does: the arena may be re-shaped but never loses a stone.
            place(&mut g, q, r);
            placed.push(HexCoord::new(q, r));
            let cells = cells(&g);
            assert!(cells >= prev_cells, "arena shrank at step {step}");
            assert!(cells <= MAX_GRID_CELLS);
            assert_eq!(g.origin_r() % 64, 0);
            prev_cells = cells;
            q = q.wrapping_add(if step % 2 == 0 { 8 } else { -8 });
            r = r.wrapping_add(if step % 3 == 0 { 8 } else { -8 });
        }
        for c in placed {
            assert_eq!(g.owner(c), Some(Player::P0), "lost the stone at {c:?}");
            assert!(g.contains_padded(c), "{c:?} lost its padding margin");
        }
    }

    /// A dimension the content does not need is handed back when the arena is
    /// re-shaped, so an excursion cannot leave a permanently bloated position.
    #[test]
    fn a_reshape_hands_back_capacity_the_content_no_longer_needs() {
        let mut g = Grid::new();
        place(&mut g, 0, 0);
        let mut q = 0i16;
        for _ in 0..200 {
            q += 8;
            place(&mut g, q, 0);
        }
        let tall = g.rows();
        assert!(
            tall >= 1024,
            "the q walk should have grown rows, got {tall}"
        );
        // Retract the walk, then force a growth along r.
        for k in 1..=(q / 8) {
            g.clear_owner(HexCoord::new(k * 8, 0), Player::P0);
        }
        grow(&mut g, 0, 400);
        assert!(
            g.rows() < tall,
            "rows stayed at {tall} for a one-stone board"
        );
        assert_eq!(g.owner(HexCoord::ORIGIN), Some(Player::P0));
        assert!(g.contains_padded(HexCoord::ORIGIN));
        assert!(g.contains_padded(HexCoord::new(0, 400)));
    }
}
