//! Trainable-player modes and their [`Player`] adapter.

use crate::player::Player;
use hexo_runner::{Decision, Game};

/// How a model is being asked to play.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mode {
    /// Generating training data.
    SelfPlay,
    /// Measuring strength.
    Eval,
}

/// A trainable player with distinct self-play and evaluation policies.
///
/// Both methods return a complete [`Decision`], including the model-authored
/// position hash and optional diagnostics.
pub trait Model {
    /// Choose under the model's self-play policy.
    fn self_play_move(&mut self, game: &Game) -> Decision;

    /// Choose under the model's evaluation policy.
    fn eval_move(&mut self, game: &Game) -> Decision;
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
    fn choose(&mut self, game: &Game) -> Decision {
        match self.mode {
            Mode::SelfPlay => self.model.self_play_move(game),
            Mode::Eval => self.model.eval_move(game),
        }
    }
}
