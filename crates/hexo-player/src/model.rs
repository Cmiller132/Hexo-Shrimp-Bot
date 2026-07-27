//! What a trainable player must provide, and how it becomes a [`Player`].

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

/// A trainable player. Encoding, search, and move selection are all its own.
///
/// Two methods rather than one taking a [`Mode`], because a mode argument can be
/// ignored: that compiles, passes, and yields a self-play run of one repeated
/// game.
///
/// Both return the whole [`Decision`], because the model is where its extra
/// fields originate: the `diagnostics` are the training annotations — visit
/// counts, value targets, whatever the trainer wants on the record — and the
/// `zobrist` attests the position the model chose from (see [`Player::choose`]).
pub trait Model {
    /// Choose by sampling. Must vary between calls, or the training set collapses.
    fn self_play_move(&mut self, game: &Game) -> Decision;

    /// Choose near-best. Not argmax — two deterministic seats replay one game.
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
