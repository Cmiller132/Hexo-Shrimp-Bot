//! What a driver needs from a seat.

use hexo_engine::Action;
use hexo_runner::Game;

/// Anything that can choose a placement. The mover is
/// `game.position().current_player()`.
///
/// The seat is handed the whole game rather than a position and a budget: the
/// record is the game's history, and a model whose features depend on move order
/// reads it from `game.plies()` or `game.prefix()`. The budget is
/// `game.spec().budget`; it is stated by the game and never enforced by it, so
/// honouring it is the seat's job.
///
/// An illegal choice is not an error: the driver submits it and the game
/// adjudicates it.
pub trait Player {
    /// Choose a placement in `game`.
    fn choose(&mut self, game: &Game) -> Action;
}

impl<P: Player + ?Sized> Player for Box<P> {
    fn choose(&mut self, game: &Game) -> Action {
        (**self).choose(game)
    }
}
