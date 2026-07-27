//! The selection hooks: what a package is handed when a search is done, and
//! what it must hand back.
//!
//! This crate ships **no** implementation of either trait — no sampler, no
//! temperature, no argmax. Move selection is the model's, as its encoding is,
//! and `crates/hexo-player/README.md` argues why at length: a shared default
//! that can be inherited without being chosen compiles, passes, and yields a
//! self-play run in which every game is identical, and no downstream stage can
//! detect it because the data is well-formed.

use crate::rng::SplitMix64;
use crate::seam::Evaluation;
use hexo_engine::{Action, Position};

/// One root child as the search left it.
///
/// Children are in the engine's canonical legal order, so `children()[i]`
/// corresponds to `root.nth_legal(i)` and to prior `i` of the root's own
/// evaluation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Child {
    /// The placement this child plays.
    pub action: Action,
    /// How many visits the search spent below it. Virtual loss is settled by the
    /// time a decision exists, so this is a real count.
    pub visits: u32,
    /// Mean backed-up value, from the perspective of the **root's** mover, or
    /// `0.0` for an unvisited child.
    pub mean_value: f64,
    /// The prior the network gave this action at the root.
    pub prior: f32,
}

/// The root and its children, as the search left them.
///
/// A borrowed view, not a snapshot: it exists only for the duration of the
/// selector call. It is publicly constructible so a package can unit-test its
/// own selector against a table of children without standing up a search.
#[derive(Clone, Copy, Debug)]
pub struct SearchOutcome<'a> {
    root: &'a Position,
    children: &'a [Child],
}

impl<'a> SearchOutcome<'a> {
    /// A view over `root` and its `children`, which must be in the canonical
    /// legal order of `root`.
    #[must_use]
    pub const fn new(root: &'a Position, children: &'a [Child]) -> Self {
        Self { root, children }
    }

    /// The position the search started from — the session's own copy of the
    /// game's, which is what its [`hexo_runner::Decision`] attests.
    #[inline]
    #[must_use]
    pub const fn root(&self) -> &'a Position {
        self.root
    }

    /// The root's children, in canonical legal order.
    #[inline]
    #[must_use]
    pub const fn children(&self) -> &'a [Child] {
        self.children
    }

    /// Visits spent below the root, summed over its children.
    ///
    /// Equal to the configured visit budget once the search is done: the root's
    /// own evaluation is not one of them.
    #[must_use]
    pub fn total_visits(&self) -> u32 {
        self.children.iter().map(|c| c.visits).sum()
    }
}

/// Package-owned: turns a finished tree search into the seat's whole utterance.
///
/// Both methods are required. A defaulted `diagnostics` returning `None` would
/// be inherited by every package that never thought about it, and the training
/// annotations — the visit distribution a policy target is built from — would
/// vanish into a record that looks complete.
pub trait SelectFromSearch: Send {
    /// Choose the placement to play.
    ///
    /// `rng` is the session's stream. Using it is what makes a self-play seat
    /// vary; a selector that ignores it plays one game forever.
    fn select(&mut self, outcome: &SearchOutcome<'_>, rng: &mut SplitMix64) -> Action;

    /// The seat-owned diagnostics for the record, or `None` to record nothing.
    ///
    /// Stored verbatim by `hexo_runner::Game` and never interpreted by anything
    /// in this workspace.
    fn diagnostics(&mut self, outcome: &SearchOutcome<'_>) -> Option<Vec<u8>>;
}

/// Package-owned: turns one root evaluation into the seat's whole utterance.
///
/// The policy-only counterpart of [`SelectFromSearch`], and the reason
/// policy-only training is the same loop as MCTS rather than a second path
/// through the driver.
pub trait SelectFromPolicy: Send {
    /// Choose the placement to play. `evaluation.priors[i]` belongs to
    /// `root.nth_legal(i)`.
    fn select(&mut self, root: &Position, evaluation: &Evaluation, rng: &mut SplitMix64) -> Action;

    /// The seat-owned diagnostics for the record, or `None` to record nothing.
    fn diagnostics(&mut self, root: &Position, evaluation: &Evaluation) -> Option<Vec<u8>>;
}
