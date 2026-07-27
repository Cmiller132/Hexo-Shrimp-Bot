//! The dense recentred arena. **Private: zero items escape the crate.**

use crate::MAX_GRID_CELLS;
use crate::coord::{COORD_LIMIT, DISK_CELLS, HexCoord, LEGAL_RADIUS};
use crate::error::MoveError;
use crate::player::Player;

/// Rows the arena starts with: `q` spans 32 values.
const MIN_ROWS: usize = 32;

/// Rows a radius-[`LEGAL_RADIUS`] disk spans, and the length of its longest row.
const DISK_ROWS: usize = 2 * LEGAL_RADIUS as usize + 1;

/// Words per row the arena starts with: `r` spans 128 values.
const MIN_ROW_WORDS: usize = 2;

/// Margin, in cells, kept between any written cell and the arena boundary.
const PAD: i32 = LEGAL_RADIUS as i32;

/// Round down to a multiple of 64. Two's complement: `-1 -> -64`, `63 -> 0`.
#[inline]
const fn floor64(x: i32) -> i32 {
    x & !63
}

/// Bits `start .. start + n` of `plane`, as a mask whose bit `k` is cell `start + k`.
#[inline]
fn gather_run(plane: &[u64], start: usize, n: usize) -> u64 {
    debug_assert!(n > 0 && n <= DISK_ROWS, "run of {n} cells");
    let (w, sh) = (start >> 6, (start & 63) as u32);
    let mut v = plane[w] >> sh;
    if sh + n as u32 > 64 {
        v |= plane[w + 1] << (64 - sh);
    }
    v & ((1u64 << n) - 1)
}

/// The dense recentred arena.
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
    frontier_cells: u32,
}

impl Grid {
    /// The empty arena.
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

    /// `locate`, panicking.
    #[inline]
    fn locate_written(&self, c: HexCoord) -> (usize, u32) {
        match self.locate(c) {
            Some(x) => x,
            None => unreachable!("arena write outside the reserved region"),
        }
    }

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

    /// Owner of the stone at a `(word, bit)` slot of the occupancy planes.
    #[inline]
    pub(crate) fn owner_at(&self, word: usize, bit: u32) -> Player {
        debug_assert!(
            (self.occupied_word(word) >> bit) & 1 == 1,
            "an occupancy slot without a stone"
        );
        if (self.occ[0][word] >> bit) & 1 == 1 {
            Player::P0
        } else {
            Player::P1
        }
    }

    /// Whether both occupancy planes claim `c`. Always false in a sound arena.
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

    /// How many frontier cells precede `c` in canonical order, or `None` if `c`
    /// is not itself a frontier cell.
    pub(crate) fn frontier_rank(&self, c: HexCoord) -> Option<usize> {
        let (word, bit) = self.locate(c)?;
        if (self.frontier[word] >> bit) & 1 == 0 {
            return None;
        }
        let below: u32 = self.frontier[..word].iter().map(|w| w.count_ones()).sum();
        let within = (self.frontier[word] & ((1u64 << bit) - 1)).count_ones();
        Some((below + within) as usize)
    }

    /// The frontier cell at `index` in canonical order, or `None` if `index` is past
    /// the end.
    pub(crate) fn nth_frontier(&self, index: usize) -> Option<HexCoord> {
        let mut remaining = index;
        for (word, &bits) in self.frontier.iter().enumerate() {
            let pop = bits.count_ones() as usize;
            if remaining >= pop {
                remaining -= pop;
                continue;
            }
            let mut w = bits;
            for _ in 0..remaining {
                w &= w - 1;
            }
            return Some(self.coord_of(word, w.trailing_zeros()));
        }
        None
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

    /// Stones within [`LEGAL_RADIUS`] of `c`. Total: `0` outside the arena.
    #[inline]
    pub(crate) fn cover(&self, c: HexCoord) -> u8 {
        match self.locate(c) {
            Some((w, b)) => self.cover[w * 64 + b as usize],
            None => 0,
        }
    }

    /// The radius-[`LEGAL_RADIUS`] disk around `c` as one contiguous cell run per `q`
    /// row: `(first cell index, length)`, rows ascending in `q` and each run ascending
    /// in `r`.
    fn disk_runs(&self, c: HexCoord) -> [(usize, usize); DISK_ROWS] {
        debug_assert!(self.contains_padded(c), "disk outside the reserved region");
        let lim = COORD_LIMIT as i32;
        let rad = LEGAL_RADIUS as i32;
        let (cq, cr) = (c.q as i32, c.r as i32);
        let mut out = [(0usize, 0usize); DISK_ROWS];
        for (i, run) in out.iter_mut().enumerate() {
            let dq = i as i32 - rad;
            let q = cq + dq;
            if q < -lim || q > lim {
                continue;
            }
            let lo = (cr - rad).max(cr - dq - rad);
            let hi = (cr + rad).min(cr - dq + rad);
            let lo = lo.max(-lim).max(-lim - q);
            let hi = hi.min(lim).min(lim - q);
            if lo > hi {
                continue;
            }
            let row = (q - self.origin_q) as usize;
            let bit = (lo - self.origin_r) as usize;
            *run = (row * self.row_words * 64 + bit, (hi - lo + 1) as usize);
        }
        out
    }

    /// `cover += 1` across the disk around `c`, setting the frontier bit of every empty
    /// cell the increment brought to coverage `1`.
    pub(crate) fn add_cover_disk(&mut self, c: HexCoord) {
        for (start, n) in self.disk_runs(c) {
            if n == 0 {
                continue;
            }
            let mut fresh = 0u64;
            for (k, cell) in self.cover[start..start + n].iter_mut().enumerate() {
                debug_assert!((*cell as usize) < DISK_CELLS, "C1: coverage overflow");
                *cell += 1;
                if *cell == 1 {
                    fresh |= 1 << k;
                }
            }
            let occupied = gather_run(&self.occ[0], start, n) | gather_run(&self.occ[1], start, n);
            self.set_frontier_run(start, fresh & !occupied);
        }
    }

    /// The exact inverse of [`Grid::add_cover_disk`]: rows in reverse order, and each
    /// frontier bit cleared *before* the decrement that justifies it.
    pub(crate) fn remove_cover_disk(&mut self, c: HexCoord) {
        for (start, n) in self.disk_runs(c).into_iter().rev() {
            if n == 0 {
                continue;
            }
            let mut falling = 0u64;
            for (k, &cell) in self.cover[start..start + n].iter().enumerate() {
                if cell == 1 {
                    falling |= 1 << k;
                }
            }
            let occupied = gather_run(&self.occ[0], start, n) | gather_run(&self.occ[1], start, n);
            self.clear_frontier_run(start, falling & !occupied);
            for cell in &mut self.cover[start..start + n] {
                debug_assert!(*cell > 0, "C1: coverage underflow");
                *cell -= 1;
            }
        }
    }

    /// Set the frontier bits named by `bits`, where bit `k` is cell `start + k`,
    /// maintaining [`Grid::frontier_cells`].
    #[inline]
    fn set_frontier_run(&mut self, start: usize, bits: u64) {
        if bits == 0 {
            return;
        }
        let (w, sh) = (start >> 6, (start & 63) as u32);
        debug_assert_eq!(self.frontier[w] & (bits << sh), 0, "C2: bit already set");
        self.frontier[w] |= bits << sh;
        if sh != 0 {
            let high = bits >> (64 - sh);
            if high != 0 {
                debug_assert_eq!(self.frontier[w + 1] & high, 0, "C2: bit already set");
                self.frontier[w + 1] |= high;
            }
        }
        self.frontier_cells += bits.count_ones();
    }

    /// Clear the frontier bits named by `bits`, as [`Grid::set_frontier_run`].
    #[inline]
    fn clear_frontier_run(&mut self, start: usize, bits: u64) {
        if bits == 0 {
            return;
        }
        let (w, sh) = (start >> 6, (start & 63) as u32);
        let low = bits << sh;
        debug_assert_eq!(self.frontier[w] & low, low, "C2: bit not set");
        self.frontier[w] &= !low;
        if sh != 0 {
            let high = bits >> (64 - sh);
            if high != 0 {
                debug_assert_eq!(self.frontier[w + 1] & high, high, "C2: bit not set");
                self.frontier[w + 1] &= !high;
            }
        }
        self.frontier_cells -= bits.count_ones();
    }

    /// Whether `c`'s frontier bit is set. Total: `false` outside the arena.
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

    /// `strip[i]` bit `j` = `plane` bit at `(c.q - 5 + i, c.r - 5 + j)`, for `i, j` in
    /// `0..11`.
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
                let w = (bit >> 6) as usize;
                let sh = (bit & 63) as u32;
                let mut v = plane[row_base + w] >> sh;
                if sh + 11 > 64 {
                    v |= plane[row_base + w + 1] << (64 - sh);
                }
                (v & 0x7FF) as u16
            } else {
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
    pub(crate) fn reserve_around(&mut self, c: HexCoord) -> Result<(), MoveError> {
        if self.contains_padded(c) {
            return Ok(());
        }
        let (cq, cr) = (c.q as i32, c.r as i32);
        let bounds = self.stone_bounds();

        let (mut lo_q, mut hi_q) = (cq - PAD, cq + PAD);
        let (mut lo_r, mut hi_r) = (cr - PAD, cr + PAD);
        if let Some((sq0, sq1, sr0, sr1)) = bounds {
            lo_q = lo_q.min(sq0 - PAD);
            hi_q = hi_q.max(sq1 + PAD);
            lo_r = lo_r.min(sr0 - PAD);
            hi_r = hi_r.max(sr1 + PAD);
        }
        let need_rows = (hi_q - lo_q + 1) as usize;
        let base_r = floor64(lo_r);
        let need_words = ((hi_r - base_r) as usize / 64) + 1;

        let least_rows = MIN_ROWS.max(need_rows);
        let least_words = MIN_ROW_WORDS.max(need_words);
        let least_cells = least_rows as u64 * least_words as u64 * 64;
        if least_cells > MAX_GRID_CELLS {
            return Err(MoveError::BoardExtentExceeded { cells: least_cells });
        }

        let fits = |rows: usize, words: usize| rows as u64 * words as u64 * 64 <= MAX_GRID_CELLS;
        let bump = |have: usize, need: usize, min: usize| {
            let want = if need > have {
                (2 * have).max(need).next_power_of_two()
            } else {
                have.min(need.next_power_of_two().saturating_mul(4))
            };
            min.max(want)
        };
        let mut new_rows = bump(self.rows, need_rows, MIN_ROWS);
        let mut new_words = bump(self.row_words, need_words, MIN_ROW_WORDS);
        if !fits(new_rows, new_words) {
            new_words = MIN_ROW_WORDS.max(need_words.next_power_of_two());
            if !fits(least_rows, new_words) {
                new_words = least_words;
            }
            let budget = (MAX_GRID_CELLS / (64 * new_words as u64)) as usize;
            new_rows = new_rows.min(budget).max(least_rows);
        }
        debug_assert!(fits(new_rows, new_words), "chosen shape breaks the ceiling");
        debug_assert!(new_rows >= need_rows && new_words >= need_words);

        let new_origin_q = lo_q - ((new_rows - need_rows) / 2) as i32;
        let new_origin_r = base_r - 64 * ((new_words - need_words) / 2) as i32;

        let words = new_rows * new_words;
        let mut occ0 = vec![0u64; words];
        let mut occ1 = vec![0u64; words];
        let mut frontier = vec![0u64; words];
        let mut cover = vec![0u8; words * 64];

        if let Some((sq0, sq1, sr0, sr1)) = bounds {
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
    use crate::coord::{DISK8, offset};

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

    /// Every frontier cell in canonical order, read by walking the whole arena
    /// coordinate by coordinate — deliberately not the word scan that `frontier_rank`
    /// and `nth_frontier` use, so the two are a real cross-check rather than the same
    /// code twice.
    fn frontier_by_brute_force(g: &Grid) -> Vec<HexCoord> {
        let mut out = Vec::new();
        for row in 0..g.rows() {
            for bit in 0..(64 * g.row_words()) {
                let c = HexCoord::new(
                    (g.origin_q() + row as i32) as i16,
                    (g.origin_r() + bit as i32) as i16,
                );
                if g.frontier_bit(c) {
                    out.push(c);
                }
            }
        }
        out
    }

    /// Fix the arena to span both corners by planting a stone at each.
    fn anchor(g: &mut Grid, lo: (i16, i16), hi: (i16, i16)) {
        place(g, lo.0, lo.1);
        place(g, hi.0, hi.1);
    }

    /// Mark `(q, r)` as a frontier cell directly, bypassing `Position`, so the
    /// rank/select scan can be tested against arbitrary bit patterns.
    fn mark_frontier(g: &mut Grid, q: i16, r: i16) {
        let c = HexCoord::new(q, r);
        assert!(
            g.cell_index(c).is_some(),
            "({q}, {r}) is outside the anchored arena"
        );
        g.set_frontier(c);
    }

    #[test]
    fn frontier_rank_and_select_are_inverse_over_a_scattered_arena() {
        let mut g = Grid::new();
        anchor(&mut g, (-6, -80), (8, 220));
        let marks = [
            (0i16, 0i16),
            (0, 1),
            (0, 63),
            (0, 64),
            (0, 65),
            (1, -1),
            (1, 0),
            (-1, 7),
            (-3, 130),
            (5, -70),
            (5, 200),
        ];
        for &(q, r) in &marks {
            mark_frontier(&mut g, q, r);
        }

        let expected = frontier_by_brute_force(&g);
        assert_eq!(expected.len(), marks.len(), "every mark must be distinct");
        assert_eq!(expected.len() as u32, g.frontier_cells());

        for (i, &c) in expected.iter().enumerate() {
            assert_eq!(g.frontier_rank(c), Some(i), "rank of ({}, {})", c.q, c.r);
            assert_eq!(g.nth_frontier(i), Some(c), "nth_frontier({i})");
        }
        assert_eq!(g.nth_frontier(expected.len()), None);
        assert_eq!(g.nth_frontier(usize::MAX), None);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn frontier_select_rejects_an_index_above_u32_max() {
        let mut g = Grid::new();
        anchor(&mut g, (-1, -1), (1, 1));
        mark_frontier(&mut g, 0, 0);
        assert_eq!(g.nth_frontier(u32::MAX as usize + 1), None);
    }

    #[test]
    fn frontier_rank_is_ascending_and_matches_canonical_order() {
        let mut g = Grid::new();
        anchor(&mut g, (-6, -10), (4, 100));
        for &(q, r) in &[(2i16, 5i16), (-4, 90), (0, 0), (2, 4), (-4, 89)] {
            mark_frontier(&mut g, q, r);
        }
        let listed = frontier_by_brute_force(&g);
        let mut sorted = listed.clone();
        sorted.sort_unstable();
        assert_eq!(listed, sorted);
        let ranks: Vec<usize> = listed
            .iter()
            .map(|&c| g.frontier_rank(c).expect("marked"))
            .collect();
        assert_eq!(ranks, (0..listed.len()).collect::<Vec<_>>());
    }

    #[test]
    fn frontier_rank_is_none_off_the_plane() {
        let mut g = Grid::new();
        anchor(&mut g, (-4, -4), (4, 4));
        mark_frontier(&mut g, 0, 0);
        assert_eq!(g.frontier_rank(HexCoord::new(0, 1)), None);
        assert_eq!(g.frontier_rank(HexCoord::new(9000, 9000)), None);
        let empty = Grid::new();
        assert_eq!(empty.frontier_rank(HexCoord::ORIGIN), None);
        assert_eq!(empty.nth_frontier(0), None);
    }

    #[test]
    fn a_full_word_ranks_every_bit() {
        let mut g = Grid::new();
        anchor(&mut g, (-2, -2), (2, 70));
        for r in 0i16..64 {
            mark_frontier(&mut g, 0, r);
        }
        assert_eq!(g.frontier_cells(), 64);
        for r in 0i16..64 {
            assert_eq!(g.frontier_rank(HexCoord::new(0, r)), Some(r as usize));
            assert_eq!(g.nth_frontier(r as usize), Some(HexCoord::new(0, r)));
        }
        assert_eq!(g.nth_frontier(64), None);
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
        assert_eq!(g.origin_q(), -15);
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
            g.add_cover_disk(HexCoord::ORIGIN);
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
            g.add_cover_disk(c);
        }
        let expect: Vec<(HexCoord, Option<Player>, u8)> = written
            .iter()
            .map(|&(q, r)| {
                let c = HexCoord::new(q, r);
                (c, g.owner(c), g.cover(c))
            })
            .collect();
        assert!(
            expect
                .iter()
                .all(|&(_, owner, cover)| owner.is_some() && cover > 0)
        );

        for &(q, r) in &[(300i16, 300i16), (-300, -300), (300, -300), (-300, 300)] {
            grow(&mut g, q, r);
            for &(c, owner, cover) in &expect {
                assert_eq!(g.owner(c), owner, "owner lost at {c:?}");
                assert_eq!(g.cover(c), cover, "coverage lost at {c:?}");
            }
            assert_eq!(g.cover(HexCoord::new(q, r)), 0);
            assert!(g.is_empty_cell(HexCoord::new(q, r)));
            assert!(!g.frontier_bit(HexCoord::new(q, r)));
        }
    }

    /// Every cell index a walk of the row runs touches, in order.
    fn cells_by_runs(g: &Grid, c: HexCoord) -> Vec<usize> {
        g.disk_runs(c)
            .into_iter()
            .flat_map(|(start, n)| start..start + n)
            .collect()
    }

    /// The same set read through the `DISK8` offset table and `locate`.
    fn cells_by_table(g: &Grid, c: HexCoord) -> Vec<usize> {
        DISK8
            .iter()
            .map(|&d| offset(c, d))
            .filter(|cell| cell.is_valid())
            .map(|cell| g.cell_index(cell).expect("inside the reserved region"))
            .collect()
    }

    /// The row runs and the `DISK8` table are two independent statements of the same
    /// cell set: `add_cover_disk` walks the runs, and the tier-C frontier assertion
    /// walks the table on every apply and undo.
    #[test]
    fn disk_runs_visit_exactly_the_disk8_cells_in_disk8_order() {
        for &(q, r) in &[(0i16, 0i16), (5, -3), (-7, 11), (40, 40), (-40, 13)] {
            let mut g = Grid::new();
            let c = HexCoord::new(q, r);
            grow(&mut g, q, r);
            let by_runs = cells_by_runs(&g, c);
            assert_eq!(by_runs, cells_by_table(&g, c), "at ({q}, {r})");
            assert_eq!(by_runs.len(), DISK_CELLS, "at ({q}, {r})");
        }
    }

    /// The per-row domain clip must agree cell for cell with `is_valid`, which is only
    /// observable within `LEGAL_RADIUS` of a face.
    #[test]
    fn disk_runs_clip_exactly_what_the_coordinate_domain_excludes() {
        for &(q, r) in &[
            (COORD_LIMIT, -COORD_LIMIT),
            (COORD_LIMIT, 0),
            (0, COORD_LIMIT),
            (-COORD_LIMIT, 0),
        ] {
            let mut g = Grid::new();
            let c = HexCoord::new(q, r);
            assert!(c.is_valid());
            grow(&mut g, q, r);
            let by_runs = cells_by_runs(&g, c);
            assert_eq!(by_runs, cells_by_table(&g, c), "at ({q}, {r})");
            assert!(
                by_runs.len() < DISK_CELLS,
                "({q}, {r}) is on a face; the domain must clip part of its disk"
            );
        }
    }

    /// Applying and removing the same disk restores every plane exactly, at a
    /// coordinate whose disk the domain clips — the case where the two halves could
    /// disagree about which cells to skip.
    #[test]
    fn a_clipped_disk_round_trips() {
        let mut g = Grid::new();
        let c = HexCoord::new(COORD_LIMIT, -COORD_LIMIT + 3);
        grow(&mut g, c.q, c.r);
        g.set_owner(c, Player::P0);
        g.add_cover_disk(c);
        let covered = g.cover_plane().iter().filter(|&&v| v > 0).count();
        assert!(covered > 0 && covered < DISK_CELLS);
        assert_eq!(
            g.frontier_cells(),
            covered as u32 - 1,
            "the stone is not free"
        );

        g.remove_cover_disk(c);
        assert_eq!(g.frontier_cells(), 0);
        assert!(g.cover_plane().iter().all(|&v| v == 0));
        assert!(g.frontier_plane().iter().all(|&w| w == 0));
    }

    #[test]
    fn frontier_counter_tracks_the_plane() {
        let mut g = Grid::new();
        grow(&mut g, 0, 0);
        assert_eq!(g.frontier_cells(), 0);
        g.set_frontier(HexCoord::new(1, 1));
        assert_eq!(g.frontier_cells(), 1);
        g.set_frontier(HexCoord::new(1, 1));
        assert_eq!(g.frontier_cells(), 1);
        g.set_frontier(HexCoord::new(2, 1));
        assert_eq!(g.frontier_cells(), 2);
        g.clear_frontier(HexCoord::new(1, 1));
        assert_eq!(g.frontier_cells(), 1);
        g.clear_frontier(HexCoord::new(1, 1));
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

    /// Regression: the growth policy grew *both* dimensions on every event, so the
    /// arena quadrupled per growth, its aspect ratio froze at 32:128, and a straight
    /// walk along `q` blew the ceiling after six allocations.
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
        assert!(g.rows() >= (q as usize) + 2 * PAD as usize);
        assert!(
            cells(&g) <= MAX_GRID_CELLS / 4,
            "{} cells for a one-row game",
            cells(&g)
        );
        for k in 0..=(q / 8) {
            assert_eq!(g.owner(HexCoord::new(k * 8, 0)), Some(Player::P0));
        }
    }

    /// The mirror walk.
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
        assert_eq!(walk(true), 1999, "the q walk hit the arena ceiling");
        assert_eq!(walk(false), 1999, "the r walk hit the arena ceiling");
    }

    /// The refusal predicate reads the stones, not the allocation history: an arena
    /// that grew large and then lost those stones must decide exactly as a fresh one
    /// holding the same stones does.
    #[test]
    fn the_ceiling_is_a_function_of_the_stones_not_of_past_growth() {
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
        for c in [
            HexCoord::new(9000, 0),
            HexCoord::new(-9000, 0),
            HexCoord::new(0, 9000),
            HexCoord::new(0, -9000),
        ] {
            assert_eq!(g.strip11(g.occ_plane(Player::P0), c), [0u16; 11]);
        }
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

    /// A dimension the content does not need is handed back when the arena is re-
    /// shaped, so an excursion cannot leave a permanently bloated position.
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
