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

use crate::coord::{Axis, HexCoord, WINDOW_LEN};
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
