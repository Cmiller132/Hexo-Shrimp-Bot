//! The move atom and its record encoding.
//!
//! [`ActionId`] is unbounded and exactly invertible, and its packing is
//! order-preserving: unsigned comparison of the inner `u32` is signed
//! lexicographic comparison of `(q, r)`. That is what makes the canonical
//! legal-move ordering of spec §9 a *sort* rather than an index into a table,
//! so no region, crop, or fixed-width mask is imposed on the action space.

use crate::coord::HexCoord;

/// Version of the canonical legal-move ordering (spec §9).
///
/// Bumping this invalidates every trained checkpoint that indexed a policy head
/// by legal-move position.
pub const ACTION_ORDER_VERSION: u32 = 1;

/// Unbounded, exactly invertible identity of a placement. The record encoding.
///
/// The packing is order-preserving: comparing the inner `u32` is exactly
/// lexicographic `(q, r)` comparison on the signed coordinate (spec §9).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ActionId(
    /// The wire value. Public so a record writer needs no accessor.
    pub u32,
);

impl ActionId {
    /// `((q as u16 ^ 0x8000) as u32) << 16 | (r as u16 ^ 0x8000) as u32`.
    ///
    /// Total and injective over every [`HexCoord`].
    #[inline]
    #[must_use]
    pub const fn from_coord(c: HexCoord) -> Self {
        Self((((c.q as u16 ^ 0x8000) as u32) << 16) | ((c.r as u16 ^ 0x8000) as u32))
    }

    /// The exact inverse of [`ActionId::from_coord`]. Total over every `u32`.
    #[inline]
    #[must_use]
    pub const fn coord(self) -> HexCoord {
        HexCoord {
            q: ((self.0 >> 16) as u16 ^ 0x8000) as i16,
            r: ((self.0 & 0xFFFF) as u16 ^ 0x8000) as i16,
        }
    }
}

impl From<HexCoord> for ActionId {
    #[inline]
    fn from(c: HexCoord) -> Self {
        Self::from_coord(c)
    }
}

impl From<ActionId> for HexCoord {
    #[inline]
    fn from(id: ActionId) -> Self {
        id.coord()
    }
}

/// A single placement — the atom of play.
///
/// Carries no legality claim; validation happens in [`crate::Position::advance`]
/// and [`crate::Search::apply`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Action(HexCoord);

impl Action {
    /// Wrap a coordinate as a placement.
    #[inline]
    #[must_use]
    pub const fn new(coord: HexCoord) -> Self {
        Self(coord)
    }

    /// The cell this placement targets.
    #[inline]
    #[must_use]
    pub const fn coord(self) -> HexCoord {
        self.0
    }

    /// The record encoding of this placement.
    #[inline]
    #[must_use]
    pub const fn id(self) -> ActionId {
        ActionId::from_coord(self.0)
    }

    /// Recover a placement from its record encoding. Total over every `u32`.
    #[inline]
    #[must_use]
    pub const fn from_id(id: ActionId) -> Self {
        Self(id.coord())
    }
}

impl From<HexCoord> for Action {
    #[inline]
    fn from(c: HexCoord) -> Self {
        Self::new(c)
    }
}

impl From<Action> for HexCoord {
    #[inline]
    fn from(a: Action) -> Self {
        a.coord()
    }
}

impl From<Action> for ActionId {
    #[inline]
    fn from(a: Action) -> Self {
        a.id()
    }
}

impl From<ActionId> for Action {
    #[inline]
    fn from(id: ActionId) -> Self {
        Self::from_id(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A coordinate set spanning all four sign quadrants plus the extremes.
    fn sample() -> Vec<HexCoord> {
        let vals = [i16::MIN, -30000, -1000, -1, 0, 1, 1000, 30000, i16::MAX];
        let mut out = Vec::new();
        for &q in &vals {
            for &r in &vals {
                out.push(HexCoord::new(q, r));
            }
        }
        out
    }

    #[test]
    fn action_id_round_trips_over_a_coordinate_grid() {
        for c in sample() {
            assert_eq!(ActionId::from_coord(c).coord(), c);
            assert_eq!(Action::from_id(Action::new(c).id()).coord(), c);
        }
        for q in -40i16..=40 {
            for r in -40i16..=40 {
                let c = HexCoord::new(q, r);
                assert_eq!(ActionId::from_coord(c).coord(), c);
            }
        }
    }

    #[test]
    fn action_id_round_trips_over_raw_u32s() {
        let raws = [
            0u32,
            1,
            0x0000_8000,
            0x8000_0000,
            0x8000_8000,
            0x7FFF_FFFF,
            0xFFFF_FFFF,
            0x1234_5678,
        ];
        for raw in raws {
            let id = ActionId(raw);
            assert_eq!(ActionId::from_coord(id.coord()).0, raw);
        }
    }

    #[test]
    fn action_id_is_injective() {
        let mut seen = std::collections::HashSet::new();
        for q in -60i16..=60 {
            for r in -60i16..=60 {
                assert!(seen.insert(ActionId::from_coord(HexCoord::new(q, r)).0));
            }
        }
    }

    #[test]
    fn three_orderings_agree() {
        let mut cs = sample();
        cs.extend([
            HexCoord::new(-7, 3),
            HexCoord::new(7, -3),
            HexCoord::new(-7, -3),
            HexCoord::new(7, 3),
        ]);
        for &a in &cs {
            for &b in &cs {
                let by_coord = a.cmp(&b);
                let by_id = ActionId::from_coord(a).cmp(&ActionId::from_coord(b));
                let by_action = Action::new(a).cmp(&Action::new(b));
                assert_eq!(by_coord, by_id, "{a:?} vs {b:?}");
                assert_eq!(by_coord, by_action, "{a:?} vs {b:?}");
            }
        }
    }

    #[test]
    fn conversions_are_consistent() {
        let c = HexCoord::new(-12, 34);
        let a: Action = c.into();
        let id: ActionId = a.into();
        assert_eq!(id, ActionId::from_coord(c));
        let back: Action = id.into();
        assert_eq!(back, a);
        let coord: HexCoord = a.into();
        assert_eq!(coord, c);
        let coord2: HexCoord = id.into();
        assert_eq!(coord2, c);
    }

    #[test]
    fn order_version_is_pinned() {
        assert_eq!(ACTION_ORDER_VERSION, 1);
    }
}
