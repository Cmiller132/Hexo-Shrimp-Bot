//! What a driver needs from a seat.

use hexo_runner::{Decision, Game};

/// Anything that can choose a placement. The mover is
/// `game.position().current_player()`.
///
/// The seat is handed the whole game rather than a position and a budget: the
/// record is the game's history, and a model whose features depend on move order
/// reads it from `game.plies()` or `game.prefix()`. The budget is
/// `game.spec().budget`; it is stated by the game and never enforced by it, so
/// honouring it is the seat's job.
///
/// The seat returns the whole [`Decision`], not a bare action, because two of
/// its fields can only be authored here. The `zobrist` is an attestation — the
/// hash of the position the seat actually chose from, which is
/// `game.position().zobrist()` for a seat reading the canonical game and its
/// own mirror's hash for a seat that keeps one — and a driver that filled it in
/// on the seat's behalf would delete the desync detector. The `diagnostics` are
/// the seat's annotations for the record, and nothing downstream can invent
/// them.
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
