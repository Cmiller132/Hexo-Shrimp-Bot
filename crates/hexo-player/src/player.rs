//! What a driver needs from a seat.

use hexo_engine::{Action, Position};
use hexo_runner::Budget;

/// Anything that can choose a placement. The mover is [`Position::current_player`].
///
/// An illegal choice is not an error: the driver submits it and the game
/// adjudicates it. `budget` is stated by the game and never enforced by it, so
/// honouring it is the seat's job.
pub trait Player {
    /// Choose a placement in `pos`.
    fn choose(&mut self, pos: &Position, budget: Budget) -> Action;
}

impl<P: Player + ?Sized> Player for Box<P> {
    fn choose(&mut self, pos: &Position, budget: Budget) -> Action {
        (**self).choose(pos, budget)
    }
}
