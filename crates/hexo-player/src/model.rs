//! What a trainable player must provide, and how it becomes a [`Player`].

use crate::player::Player;
use hexo_engine::{Action, Position};
use hexo_runner::Budget;

/// How a model is being asked to play.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mode {
    /// Generating training data.
    SelfPlay,
    /// Measuring strength.
    Eval,
}

/// A trainable player. Encoding, search, and move selection are all its own.
///
/// Two methods rather than one taking a [`Mode`], because a mode argument can be
/// ignored: that compiles, passes, and yields a self-play run of one repeated
/// game.
pub trait Model {
    /// Choose by sampling. Must vary between calls, or the training set collapses.
    fn self_play_move(&mut self, pos: &Position, budget: Budget) -> Action;

    /// Choose near-best. Not argmax — two deterministic seats replay one game.
    fn eval_move(&mut self, pos: &Position, budget: Budget) -> Action;
}

/// A [`Model`] bound to a [`Mode`]. Dispatch only; no selection logic lives here.
#[derive(Clone, Copy, Debug)]
pub struct ModelPlayer<M> {
    model: M,
    mode: Mode,
}

impl<M> ModelPlayer<M> {
    /// Bind `model` to `mode`.
    pub const fn new(model: M, mode: Mode) -> Self {
        Self { model, mode }
    }

    /// The mode this seat plays in.
    #[inline]
    #[must_use]
    pub const fn mode(&self) -> Mode {
        self.mode
    }

    /// The bound model.
    #[inline]
    #[must_use]
    pub const fn model(&self) -> &M {
        &self.model
    }

    /// Take the model back, discarding the binding.
    #[must_use]
    pub fn into_model(self) -> M {
        self.model
    }
}

impl<M: Model> Player for ModelPlayer<M> {
    fn choose(&mut self, pos: &Position, budget: Budget) -> Action {
        match self.mode {
            Mode::SelfPlay => self.model.self_play_move(pos, budget),
            Mode::Eval => self.model.eval_move(pos, budget),
        }
    }
}
