//! Six-cell window geometry and the exposed win-detection surface.
//!
//! A *window* is six consecutive cells along one of the three axes. A player
//! wins when any window is completely filled by that player's stones; six **or
//! more** in a row wins, because a run of seven contains a fully-owned
//! six-window. There is no overline rule.
//!
//! Nothing here is stored by the engine. [`WindowMask`] values are derived on
//! read from the occupancy planes (spec §6.2), so there is no stored mask table
//! to grow, to undo, or to disagree with the board.
//!
//! Ruling 3: this module exposes *masks*, not predicates. `is_threat_for`,
//! `threat_player`, and `is_active` are deliberately absent — a mask is
//! strictly more information, and each of those predicates is a one-liner over
//! [`WindowMask::mask`] and [`Window::cells`].

use crate::coord::{Axis, HexCoord, WINDOW_LEN, hex_distance};
use crate::player::Player;

/// Windows touched by one placement: 3 axes × 6 offsets.
pub const WINDOWS_PER_PLACEMENT: usize = 18;

/// Every bit position inside a window.
const FULL: u8 = 0x3F;

/// Ownership of one six-cell window, as two six-bit masks.
///
/// Bit `i` refers to the window's cell `i`, which is a statement about the
/// infinite board (`start + axis.vector() * i`) and never about storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct WindowMask([u8; 2]);

impl WindowMask {
    /// The empty window.
    pub const EMPTY: Self = Self([0, 0]);

    /// Build a mask from the two per-player lanes. Internal: the player-to-lane
    /// mapping is a private convention.
    #[inline]
    pub(crate) const fn from_lanes(p0: u8, p1: u8) -> Self {
        Self([p0 & FULL, p1 & FULL])
    }

    /// Bit `i` set iff cell `i` of the window holds a stone of `player`. Low six bits.
    #[inline]
    #[must_use]
    pub const fn mask(self, player: Player) -> u8 {
        self.0[player.index()]
    }

    /// Stones `player` holds in this window, `0..=6`.
    #[inline]
    #[must_use]
    pub const fn count(self, player: Player) -> u32 {
        self.0[player.index()].count_ones()
    }

    /// Either player's stones. `mask(P0) | mask(P1)`.
    #[inline]
    #[must_use]
    pub const fn occupied(self) -> u8 {
        self.0[0] | self.0[1]
    }

    /// Complement of [`WindowMask::occupied`] within the low six bits.
    #[inline]
    #[must_use]
    pub const fn empty(self) -> u8 {
        !self.occupied() & FULL
    }

    /// Whether `player` owns all six cells — the win condition for this window.
    #[inline]
    #[must_use]
    pub const fn is_full_for(self, player: Player) -> bool {
        self.0[player.index()] == FULL
    }
}

/// Identity of one six-cell window: its first cell and the axis it runs along.
///
/// Pure geometry. Constructible and interpretable with no
/// [`crate::Position`] in hand, and valid forever regardless of how the engine
/// stores the board.
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
    #[inline]
    #[must_use]
    pub const fn cell(self, index: usize) -> HexCoord {
        assert!(index < WINDOW_LEN, "window cell index out of range");
        self.start.step(self.axis, index as i16)
    }

    /// All six coordinates, in bit order.
    ///
    /// # Panics
    /// Debug builds assert `self.start.is_valid()`.
    #[inline]
    #[must_use]
    pub const fn cells(self) -> [HexCoord; WINDOW_LEN] {
        let mut out = [self.start; WINDOW_LEN];
        let mut i = 0;
        while i < WINDOW_LEN {
            out[i] = self.start.step(self.axis, i as i16);
            i += 1;
        }
        out
    }

    /// Which of the six cells `coord` is, or `None` if it is not one of them.
    ///
    /// The inverse of [`Window::cell`], and strictly more information than
    /// [`Window::contains`] — which is why it is the one that composes. An
    /// incidence edge in a cell/window graph has to carry *which* bit of
    /// [`WindowMask`] the cell is, because the mask is positional; "somewhere in
    /// this window" would not be enough to read the mask back.
    ///
    /// Total, and computed in `i32` rather than by walking [`Window::cells`], so
    /// a coordinate arbitrarily far from `start` answers `None` instead of
    /// wrapping.
    #[inline]
    #[must_use]
    pub const fn cell_index(self, coord: HexCoord) -> Option<usize> {
        let dq = coord.q as i32 - self.start.q as i32;
        let dr = coord.r as i32 - self.start.r as i32;
        // `coord == start + axis.vector() * i`, split into the constraint that
        // pins `coord` to the line and the index along it.
        let (off_line, i) = match self.axis {
            Axis::Q => (dr, dq),
            Axis::R => (dq, dr),
            Axis::QR => (dq + dr, dq),
        };
        if off_line != 0 || i < 0 || i >= WINDOW_LEN as i32 {
            return None;
        }
        Some(i as usize)
    }

    /// Whether `coord` is one of this window's six cells.
    #[inline]
    #[must_use]
    pub const fn contains(self, coord: HexCoord) -> bool {
        self.cell_index(coord).is_some()
    }

    /// Whether the two windows share at least one cell. Symmetric.
    ///
    /// Six [`Window::contains`] calls rather than a closed form over the
    /// parallel and crossing cases. Two windows on the same axis overlap when
    /// their starts are within six steps; on different axes they meet in at most
    /// one cell, found by solving the two lines. That is two branches of a case
    /// analysis that could be wrong in the same way — the exact hazard this crate
    /// treats as the one that matters — for six cheap tests.
    ///
    /// # Panics
    /// Debug builds assert `self.start.is_valid()`.
    #[inline]
    #[must_use]
    pub const fn intersects(self, other: Self) -> bool {
        let mut i = 0;
        while i < WINDOW_LEN {
            if other.contains(self.cell(i)) {
                return true;
            }
            i += 1;
        }
        false
    }

    /// Whether the two windows are disjoint but have a pair of adjacent cells.
    ///
    /// **Exclusive of overlap.** Two windows that share a cell do *not* touch, so
    /// `intersects`, `touches`, and neither partition every pair of windows.
    /// "Overlapping or adjacent" is `a.intersects(b) || a.touches(b)`, which is
    /// left to the caller because it is a union of two answers rather than a
    /// third fact.
    ///
    /// # Panics
    /// Debug builds assert both starts are valid.
    #[inline]
    #[must_use]
    pub const fn touches(self, other: Self) -> bool {
        if self.intersects(other) {
            return false;
        }
        let mine = self.cells();
        let theirs = other.cells();
        let mut i = 0;
        while i < WINDOW_LEN {
            let mut j = 0;
            while j < WINDOW_LEN {
                if hex_distance(mine[i], theirs[j]) == 1 {
                    return true;
                }
                j += 1;
            }
            i += 1;
        }
        false
    }
}

/// A window paired with its current ownership.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WindowRef {
    /// Which window.
    pub window: Window,
    /// Who owns which of its cells.
    pub mask: WindowMask,
}

/// Which of the 18 windows through a placement that placement completed.
///
/// A set over the canonical slot order of spec §6.3: axis-major (`Q`, `R`,
/// `QR`), then offset `0..6`, where offset `k` means the placed cell sits at
/// bit `k` of the window. The bit layout is `axis.index() * 6 + offset`, but
/// **nothing outside this type needs to know that** — that is the point of the
/// type. [`WinningWindows::bits`] is the escape hatch for a record writer.
///
/// **More than one slot can be set.** Seven in a row contains two fully-owned
/// six-windows, and two lines crossing at the placed cell complete both. Code
/// that assumes exactly one is wrong (spec §7.4 H6).
///
/// [`crate::Applied::winning_windows`] resolves these slots into real
/// [`Window`] values, which is the accessor most consumers want.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct WinningWindows(u32);

impl WinningWindows {
    /// No window was completed — the placement did not win.
    pub const EMPTY: Self = Self(0);

    /// Wrap a raw slot mask. Internal: the bit layout is a private convention.
    #[inline]
    pub(crate) const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    /// Whether no window was completed. Equivalent to `count() == 0`, and
    /// exactly the negation of "this placement won".
    #[inline]
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// How many of the 18 windows this placement completed. Can exceed one.
    #[inline]
    #[must_use]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    /// Whether the window at `offset` along `axis` was completed.
    ///
    /// # Panics
    /// Panics if `offset >= WINDOW_LEN`.
    #[inline]
    #[must_use]
    pub const fn contains(self, axis: Axis, offset: usize) -> bool {
        assert!(offset < WINDOW_LEN, "window offset out of range");
        (self.0 >> (axis.index() * WINDOW_LEN + offset)) & 1 == 1
    }

    /// The raw slot mask, for a record writer that persists it verbatim.
    ///
    /// Reading individual bits out of this is what [`WinningWindows::iter`]
    /// exists to make unnecessary.
    #[inline]
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// The completed slots as `(axis, offset)` pairs, in ascending slot order.
    #[inline]
    #[must_use]
    pub const fn iter(self) -> WinningSlots {
        WinningSlots(self.0)
    }
}

impl IntoIterator for WinningWindows {
    type Item = (Axis, usize);
    type IntoIter = WinningSlots;

    #[inline]
    fn into_iter(self) -> WinningSlots {
        self.iter()
    }
}

/// Iterator over the completed slots of a [`WinningWindows`], ascending.
#[derive(Clone, Debug)]
pub struct WinningSlots(u32);

impl Iterator for WinningSlots {
    type Item = (Axis, usize);

    #[inline]
    fn next(&mut self) -> Option<(Axis, usize)> {
        if self.0 == 0 {
            return None;
        }
        let slot = self.0.trailing_zeros() as usize;
        self.0 &= self.0 - 1;
        Some((Axis::ALL[slot / WINDOW_LEN], slot % WINDOW_LEN))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.0.count_ones() as usize;
        (n, Some(n))
    }
}

impl ExactSizeIterator for WinningSlots {}
impl core::iter::FusedIterator for WinningSlots {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_mask_algebra() {
        let m = WindowMask::EMPTY;
        assert_eq!(m.mask(Player::P0), 0);
        assert_eq!(m.mask(Player::P1), 0);
        assert_eq!(m.occupied(), 0);
        assert_eq!(m.empty(), FULL);
        assert_eq!(m.count(Player::P0), 0);
        assert!(!m.is_full_for(Player::P0));
        assert_eq!(WindowMask::default(), WindowMask::EMPTY);
    }

    #[test]
    fn mask_accessor_algebra_over_every_disjoint_pair() {
        for a in 0u8..64 {
            for b in 0u8..64 {
                if a & b != 0 {
                    continue;
                }
                let m = WindowMask::from_lanes(a, b);
                assert_eq!(m.mask(Player::P0), a);
                assert_eq!(m.mask(Player::P1), b);
                assert_eq!(m.occupied(), a | b);
                assert_eq!(m.empty(), !(a | b) & FULL);
                assert_eq!(m.count(Player::P0), a.count_ones());
                assert_eq!(m.count(Player::P1), b.count_ones());
                assert_eq!(m.is_full_for(Player::P0), a == FULL);
                assert_eq!(m.is_full_for(Player::P1), b == FULL);
            }
        }
    }

    #[test]
    fn from_lanes_clamps_to_six_bits() {
        let m = WindowMask::from_lanes(0xFF, 0xC0);
        assert_eq!(m.mask(Player::P0), FULL);
        assert_eq!(m.mask(Player::P1), 0);
    }

    #[test]
    fn window_cells_walk_the_axis() {
        for axis in Axis::ALL {
            let w = Window {
                start: HexCoord::new(-4, 6),
                axis,
            };
            let cells = w.cells();
            assert_eq!(cells[0], w.start);
            for (i, &cell) in cells.iter().enumerate() {
                assert_eq!(cell, w.cell(i));
                assert_eq!(
                    crate::coord::hex_distance(w.start, cell),
                    i as u32,
                    "axis {axis:?} index {i}"
                );
            }
        }
    }

    /// Every window whose start lies in a small box, on all three axes.
    ///
    /// Small because the relation tests are quadratic in it; wide enough that
    /// every pair of axes is represented at every relative offset that can
    /// overlap, touch, or miss.
    fn corpus() -> Vec<Window> {
        let mut out = Vec::new();
        for q in -3..=3 {
            for r in -3..=3 {
                for axis in Axis::ALL {
                    out.push(Window {
                        start: HexCoord::new(q, r),
                        axis,
                    });
                }
            }
        }
        out
    }

    /// The same three relations read off the materialised cell arrays.
    ///
    /// Deliberately *not* written in terms of `cell_index`: the shipped versions
    /// are closed-form index arithmetic, and a brute-force walk of `cells()` is
    /// the only independent statement of the same geometry. A wrong off-line
    /// constraint — `dq - dr` where the QR axis needs `dq + dr` — is symmetric
    /// under any round trip, so this comparison is the detector.
    fn brute_contains(w: Window, c: HexCoord) -> bool {
        w.cells().contains(&c)
    }

    fn brute_intersects(a: Window, b: Window) -> bool {
        let (x, y) = (a.cells(), b.cells());
        x.iter().any(|c| y.contains(c))
    }

    fn brute_touches(a: Window, b: Window) -> bool {
        if brute_intersects(a, b) {
            return false;
        }
        let (x, y) = (a.cells(), b.cells());
        x.iter()
            .any(|p| y.iter().any(|q| hex_distance(*p, *q) == 1))
    }

    #[test]
    fn cell_index_inverts_cell() {
        for w in corpus() {
            for i in 0..WINDOW_LEN {
                assert_eq!(w.cell_index(w.cell(i)), Some(i), "{w:?} cell {i}");
            }
        }
    }

    #[test]
    fn contains_agrees_with_a_cell_walk_over_a_whole_neighbourhood() {
        for w in corpus() {
            for q in -9..=9 {
                for r in -9..=9 {
                    let c = HexCoord::new(q, r);
                    assert_eq!(w.contains(c), brute_contains(w, c), "{w:?} vs {c:?}");
                    assert_eq!(w.contains(c), w.cell_index(c).is_some());
                }
            }
        }
    }

    /// A coordinate on the window's line but past either end is not in it, and
    /// one step off the line never is. Named separately from the sweep because
    /// these are the two ways `cell_index` can be wrong per axis.
    #[test]
    fn contains_rejects_off_line_and_past_the_ends() {
        for axis in Axis::ALL {
            let w = Window {
                start: HexCoord::ORIGIN,
                axis,
            };
            assert!(!w.contains(w.start.step(axis, -1)), "{axis:?} before start");
            assert!(
                !w.contains(w.start.step(axis, WINDOW_LEN as i16)),
                "{axis:?} past end"
            );
            for off in Axis::ALL {
                if off.index() == axis.index() {
                    continue;
                }
                for i in 0..WINDOW_LEN {
                    let beside = w.cell(i).step(off, 1);
                    assert_eq!(
                        w.contains(beside),
                        brute_contains(w, beside),
                        "{axis:?} cell {i} stepped along {off:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn intersects_agrees_with_a_cell_walk_and_is_symmetric() {
        let all = corpus();
        for &a in &all {
            for &b in &all {
                assert_eq!(a.intersects(b), brute_intersects(a, b), "{a:?} vs {b:?}");
                assert_eq!(a.intersects(b), b.intersects(a), "asymmetric {a:?} {b:?}");
            }
        }
    }

    /// A property the cell walk does not state: two windows on the same axis and
    /// the same line overlap exactly when their starts are within six steps.
    #[test]
    fn same_axis_windows_overlap_within_six_steps() {
        for axis in Axis::ALL {
            let a = Window {
                start: HexCoord::ORIGIN,
                axis,
            };
            for k in -8..=8i16 {
                let b = Window {
                    start: HexCoord::ORIGIN.step(axis, k),
                    axis,
                };
                assert_eq!(
                    a.intersects(b),
                    k.abs() < WINDOW_LEN as i16,
                    "{axis:?} offset {k}"
                );
            }
        }
    }

    #[test]
    fn touches_agrees_with_a_cell_walk_and_excludes_overlap() {
        let all = corpus();
        for &a in &all {
            for &b in &all {
                assert_eq!(a.touches(b), brute_touches(a, b), "{a:?} vs {b:?}");
                assert_eq!(a.touches(b), b.touches(a), "asymmetric {a:?} {b:?}");
                assert!(
                    !(a.intersects(b) && a.touches(b)),
                    "{a:?} and {b:?} both overlap and touch"
                );
            }
        }
        // The partition is not vacuous: the corpus contains all three cases.
        assert!(all.iter().any(|&a| all.iter().any(|&b| a.intersects(b))));
        assert!(all.iter().any(|&a| all.iter().any(|&b| a.touches(b))));
        assert!(
            all.iter()
                .any(|&a| all.iter().any(|&b| !a.intersects(b) && !a.touches(b)))
        );
    }

    /// A window always overlaps itself and never touches itself.
    #[test]
    fn a_window_intersects_itself() {
        for w in corpus() {
            assert!(w.intersects(w));
            assert!(!w.touches(w));
        }
    }

    /// Two windows six apart along the same axis are the adjacent-but-disjoint
    /// case by construction: cell 5 of the first neighbours cell 0 of the second.
    #[test]
    fn consecutive_collinear_windows_touch() {
        for axis in Axis::ALL {
            let a = Window {
                start: HexCoord::ORIGIN,
                axis,
            };
            let b = Window {
                start: HexCoord::ORIGIN.step(axis, WINDOW_LEN as i16),
                axis,
            };
            assert!(!a.intersects(b), "{axis:?}");
            assert!(a.touches(b), "{axis:?}");
        }
    }

    #[test]
    #[should_panic(expected = "window cell index out of range")]
    fn window_cell_panics_past_the_end() {
        let w = Window {
            start: HexCoord::ORIGIN,
            axis: Axis::Q,
        };
        let _ = w.cell(WINDOW_LEN);
    }

    #[test]
    fn windows_per_placement_is_three_axes_by_six_offsets() {
        assert_eq!(WINDOWS_PER_PLACEMENT, Axis::ALL.len() * WINDOW_LEN);
    }

    #[test]
    fn winning_windows_is_empty_by_default() {
        let w = WinningWindows::default();
        assert_eq!(w, WinningWindows::EMPTY);
        assert!(w.is_empty());
        assert_eq!(w.count(), 0);
        assert_eq!(w.bits(), 0);
        assert_eq!(w.iter().next(), None);
    }

    /// `contains` and `iter` must agree with the documented bit layout over
    /// every one of the 18 slots, one slot at a time.
    #[test]
    fn each_slot_round_trips_through_contains_and_iter() {
        for axis in Axis::ALL {
            for offset in 0..WINDOW_LEN {
                let slot = axis.index() * WINDOW_LEN + offset;
                let w = WinningWindows::from_bits(1 << slot);
                assert!(!w.is_empty());
                assert_eq!(w.count(), 1);
                assert!(w.contains(axis, offset), "{axis:?}+{offset}");
                assert_eq!(w.iter().collect::<Vec<_>>(), vec![(axis, offset)]);
                // Every other slot must read false.
                for other in Axis::ALL {
                    for k in 0..WINDOW_LEN {
                        if other.index() * WINDOW_LEN + k == slot {
                            continue;
                        }
                        assert!(!w.contains(other, k), "{other:?}+{k} leaked");
                    }
                }
            }
        }
    }

    /// The multi-window case (H6): seven in a row and crossing lines both set
    /// more than one bit, and `iter` must yield them all in ascending order.
    #[test]
    fn iter_yields_every_set_slot_in_ascending_order() {
        let bits = (1 << 0) | (1 << 1) | (1 << 7) | (1 << 17);
        let w = WinningWindows::from_bits(bits);
        assert_eq!(w.count(), 4);
        assert_eq!(w.iter().len(), 4);
        assert_eq!(
            w.iter().collect::<Vec<_>>(),
            vec![
                (Axis::Q, 0),
                (Axis::Q, 1),
                (Axis::R, 1),
                (Axis::QR, WINDOW_LEN - 1),
            ]
        );
        assert_eq!(w.into_iter().count(), 4);
    }

    /// `iter` and `contains` are two independent readings of the same bit
    /// layout, so they are only a cross-check if they are compared. Every
    /// single- and double-slot mask across all three axes, plus the extremes.
    #[test]
    fn iter_agrees_with_contains_over_every_mask() {
        let mut masks: Vec<u32> = vec![0, (1 << WINDOWS_PER_PLACEMENT) - 1];
        for i in 0..WINDOWS_PER_PLACEMENT {
            for j in 0..WINDOWS_PER_PLACEMENT {
                masks.push((1 << i) | (1 << j));
            }
        }
        for bits in masks {
            let w = WinningWindows::from_bits(bits);
            let from_iter: Vec<_> = w.iter().collect();
            let mut from_contains = Vec::new();
            for axis in Axis::ALL {
                for offset in 0..WINDOW_LEN {
                    if w.contains(axis, offset) {
                        from_contains.push((axis, offset));
                    }
                }
            }
            assert_eq!(from_iter, from_contains, "bits {bits:#x}");
            assert_eq!(w.count() as usize, from_iter.len());
            assert_eq!(w.bits(), bits);
            assert_eq!(w.is_empty(), bits == 0);
        }
    }

    #[test]
    #[should_panic(expected = "window offset out of range")]
    fn contains_panics_past_the_end() {
        let _ = WinningWindows::EMPTY.contains(Axis::Q, WINDOW_LEN);
    }
}
