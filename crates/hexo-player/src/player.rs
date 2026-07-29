//! Seat decision interface.

use hexo_runner::{Decision, Game};

/// Anything that can choose a placement. The mover is
/// `game.position().current_player()`.
///
/// The seat receives the complete game. Move-order features may use
/// `game.plies()` or `game.prefix()`, and the seat is responsible for honoring
/// `game.spec().budget`.
///
/// The seat authors the complete [`Decision`]. `zobrist` is the hash of the
/// position used to choose, and `diagnostics` are opaque seat-owned bytes.
///
/// An illegal choice is not an error: the driver submits it and the game
/// adjudicates it.
pub trait Player {
    /// Choose a placement in `game`.
    fn choose(&mut self, game: &Game) -> Decision;
}

impl<P: Player + ?Sized> Player for Box<P> {
    fn choose(&mut self, game: &Game) -> Decision {
        (**self).choose(game)
    }
}
