//! Why a submission was not usable.
//!
//! Everything here means **nothing happened**: the game is exactly as it was and
//! the same [`crate::Step::NeedDecision`] is still outstanding. That is the line
//! this type draws — a `SubmitError` is a statement about the *submission*, not
//! about the game.
//!
//! An illegal placement is deliberately not here. It is a perfectly usable
//! submission that happens to lose the game, and it comes back as a
//! [`crate::Transition`] carrying a [`crate::MatchResult`]. Putting it here
//! instead would make every driver decide what an illegal move costs, which is
//! precisely the adjudication policy the game exists to own.

/// A submission the game refused to act on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SubmitError {
    /// The game has already ended.
    Finished,
    /// The submission answers a decision the game has moved past.
    ///
    /// Reachable whenever decisions are in flight: a late reply from a seat that
    /// was already given up on, a duplicate from a retrying transport, or a
    /// response arriving after the batch it belonged to was abandoned. Applying
    /// it would silently play a move chosen for a different position.
    StaleGeneration {
        /// The token the game is waiting on.
        expected: u64,
        /// The token that was submitted.
        got: u64,
    },
    /// The seat's mirror disagrees with the canonical position.
    ///
    /// The seat echoed a hash that is not the one it was asked to move in, so
    /// its board has drifted. Caught on the ply it drifts rather than at the end
    /// of a corrupted game.
    ///
    /// A driver may resynchronise the mirror from [`crate::Game::prefix`] and
    /// retry, or give up and submit
    /// [`Reply::Failed`](crate::Reply::Failed)`(`[`Failure::Protocol`](crate::Failure::Protocol)`)`,
    /// which ends the game under the spec's failure policy. What it must not do
    /// is ignore this.
    Desync {
        /// The canonical hash.
        expected: u64,
        /// The hash the seat believed.
        got: u64,
    },
}

impl core::fmt::Display for SubmitError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Finished => f.write_str("the game has already ended"),
            Self::StaleGeneration { expected, got } => write!(
                f,
                "submission is for generation {got}, but the game is at {expected}"
            ),
            Self::Desync { expected, got } => write!(
                f,
                "the seat's position hash is {got:#018x}, but the canonical one is {expected:#018x}"
            ),
        }
    }
}

impl core::error::Error for SubmitError {}
