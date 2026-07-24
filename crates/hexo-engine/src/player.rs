//! Who moves, and where they are inside the two-placement turn.
//!
//! The turn structure is: ply 0 is [`TurnPhase::Opening`] and belongs to
//! [`Player::P0`] at [`HexCoord::ORIGIN`]; after that each player places two
//! stones per turn, and a win is checked after each of them. The resulting ply
//! pattern is `P0; P1 P1; P0 P0; P1 P1; ...`.

use crate::coord::HexCoord;

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
    #[inline]
    #[must_use]
    pub const fn other(self) -> Self {
        match self {
            Self::P0 => Self::P1,
            Self::P1 => Self::P0,
        }
    }

    /// `0` for `P0`, `1` for `P1`.
    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// Where the mover is inside the two-placement turn.
///
/// A terminal position freezes whichever phase it reached, so **every branch on
/// a phase must test [`crate::Position::is_terminal`] first** (spec §7.4 H2).
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
    /// Ignores the `first` payload; used by the Zobrist turn key (spec §8).
    #[inline]
    #[must_use]
    pub const fn kind_index(self) -> usize {
        match self {
            Self::Opening => 0,
            Self::FirstStone => 1,
            Self::SecondStone { .. } => 2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn player_other_is_an_involution() {
        assert_eq!(Player::P0.other(), Player::P1);
        assert_eq!(Player::P1.other(), Player::P0);
        assert_eq!(Player::P0.other().other(), Player::P0);
        assert_eq!(Player::P1.other().other(), Player::P1);
    }

    #[test]
    fn player_index_matches_repr() {
        assert_eq!(Player::P0.index(), 0);
        assert_eq!(Player::P1.index(), 1);
        assert_eq!(Player::P0 as u8, 0);
        assert_eq!(Player::P1 as u8, 1);
    }

    #[test]
    fn kind_index_ignores_the_first_payload() {
        assert_eq!(TurnPhase::Opening.kind_index(), 0);
        assert_eq!(TurnPhase::FirstStone.kind_index(), 1);
        assert_eq!(
            TurnPhase::SecondStone {
                first: HexCoord::ORIGIN
            }
            .kind_index(),
            2
        );
        assert_eq!(
            TurnPhase::SecondStone {
                first: HexCoord::new(-9, 4)
            }
            .kind_index(),
            2
        );
    }

    #[test]
    fn second_stone_equality_includes_first() {
        let a = TurnPhase::SecondStone {
            first: HexCoord::new(1, 2),
        };
        let b = TurnPhase::SecondStone {
            first: HexCoord::new(1, 3),
        };
        assert_ne!(a, b);
    }
}
