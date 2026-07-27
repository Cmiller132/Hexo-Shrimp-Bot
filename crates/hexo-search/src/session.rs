//! A seat's search as a nonblocking state machine.

use crate::seam::Evaluation;
use hexo_engine::Position;
use hexo_runner::{Decision, Game};

/// Session-scoped handle for one requested leaf evaluation.
///
/// Opaque and never reused: a session mints a fresh serial per leaf and keeps
/// minting across [`DecisionSession::begin`], so an answer that arrives for a
/// decision the session has already moved past is *unknown* rather than
/// plausible. That is the same reasoning as `hexo-runner`'s generation token,
/// one level down.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LeafId(u64);

impl LeafId {
    /// Mint the id for a session's `serial`-th requested leaf.
    #[inline]
    pub(crate) const fn from_serial(serial: u64) -> Self {
        Self(serial)
    }
}

/// Where a session is between [`DecisionSession::begin`] and its decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SessionStatus {
    /// The session has evaluations in flight and cannot progress until they are
    /// delivered with [`DecisionSession::resume`].
    ///
    /// `in_flight` is never zero: a session with nothing outstanding and work
    /// left to do keeps working rather than returning.
    AwaitingEvals {
        /// How many leaves are waiting for an answer.
        in_flight: usize,
    },
    /// The decision is ready to take.
    Decided,
}

/// A seat's search as a nonblocking state machine: `Send`, object-safe, and
/// never in possession of a thread while it waits.
///
/// The shape mirrors `hexo_runner::Game` one level down. A game does not call
/// its players; a session does not call its evaluator. Both invert the obvious
/// blocking design for the same reason: a search that blocks inside
/// `evaluate(one_position)` pins a thread per game *and* forecloses batching,
/// because the one thing the process wants to coalesce is exactly the thing
/// every thread is asleep on.
///
/// The loop is `begin`, then `pump`/`resume` until [`SessionStatus::Decided`],
/// then `take_decision`. Every call returns promptly; the driver decides how
/// many sessions to interleave and how large a batch to fill before it crosses
/// to the device.
pub trait DecisionSession: Send {
    /// Reset onto `game`'s current position and discard any previous search.
    ///
    /// The session takes its own copy of the position — a seat never holds a
    /// mutable handle to canonical state — and reuses the buffer it copied into
    /// last time, so a session driven for ten thousand decisions keeps its
    /// allocations flat.
    ///
    /// # Panics
    ///
    /// If `game` has already finished. A driver only asks a live game's mover,
    /// and a session asked to search a settled game has been handed the wrong
    /// game rather than an empty search.
    fn begin(&mut self, game: &Game);

    /// Run until the decision is ready, the in-flight cap is reached, or the
    /// visit budget is fully dispatched.
    ///
    /// For each leaf the search wants evaluated, `emit` is called with the
    /// leaf's handle and the leaf *position*. The caller encodes it right there:
    /// that position is transient make/unmake state on the session's own board,
    /// valid only for the duration of the callback, and there is nothing to
    /// queue but the bytes an [`crate::Encoder`] produces from it.
    ///
    /// Calling `pump` again after [`SessionStatus::Decided`] returns `Decided`
    /// and emits nothing.
    ///
    /// # Panics
    ///
    /// If `begin` has never been called.
    fn pump(&mut self, emit: &mut dyn FnMut(LeafId, &Position)) -> SessionStatus;

    /// Deliver one result.
    ///
    /// # Panics
    ///
    /// If `leaf` is not in flight — including an answer for a decision this
    /// session has already left — or if the evaluation breaks either convention
    /// on [`Evaluation`]. Both are silent corruption of a training run if
    /// tolerated: the first plays a move chosen for another position, the second
    /// indexes the policy head against an action set it does not match.
    fn resume(&mut self, leaf: LeafId, evaluation: Evaluation);

    /// Take the finished decision, or `None` until the session is
    /// [`SessionStatus::Decided`].
    ///
    /// Taking it resets nothing; `begin` does the resetting. The decision is
    /// authored once, at the moment the search completes, so taking it has no
    /// side effects and cannot draw from the RNG a second time.
    fn take_decision(&mut self) -> Option<Decision>;

    /// Replace the RNG seed.
    ///
    /// This is the deliberate seam for `docs/OPEN_DECISIONS.md` B4. Today a
    /// driver passes entropy at construction and games are non-deterministic,
    /// which is honest: nothing records a seed, so nothing promises a replay.
    /// When reproducible self-play is wanted, seeds minted from stable game and
    /// seat ids — so that scheduling cannot change a run — land here, and
    /// nothing else about a session has to move.
    fn reseed(&mut self, seed: u64);
}
