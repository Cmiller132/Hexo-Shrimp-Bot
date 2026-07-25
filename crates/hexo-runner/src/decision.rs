//! What a seat sends back, and what it was given to work with.
//!
//! A driver asks a seat for a move however it likes — a function call, a pipe, a
//! socket, a human clicking — and reports the outcome here. [`Reply`] covers
//! every outcome that can arise, including the ones that are not moves, so the
//! game never has to learn what a player is.

use hexo_engine::Action;
use std::time::Duration;

/// What a seat was given to think with.
///
/// The game **states** the budget and records it; it does not enforce one. A
/// deterministic budget is the seat's to honour, and a wall clock is the
/// driver's to police — the game has no clock, by construction.
///
/// The point is that a budget is *stated at all*. The previous implementation
/// had no budget in its contract anywhere, so one seat searched a fixed visit
/// count while another used a 0.05 s think time, and there was no way to say
/// whether an evaluation had given both seats the same thing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Budget {
    /// No stated limit. The seat answers when it answers.
    #[default]
    Unlimited,
    /// A search-node count. Reproducible across machines.
    Nodes(u64),
    /// A tree-visit count. Reproducible across machines.
    Visits(u64),
    /// Wall-clock time. Not reproducible; for tournaments, not for self-play.
    Wall(Duration),
}

impl Budget {
    /// Whether two runs under this budget should produce identical games.
    ///
    /// `false` for [`Budget::Wall`], which is what makes a wall clock unsuitable
    /// for self-play that is meant to be replayable.
    #[inline]
    #[must_use]
    pub const fn is_reproducible(self) -> bool {
        !matches!(self, Self::Wall(_))
    }
}

/// A seat's chosen placement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decision {
    /// Where to place.
    pub action: Action,
    /// The hash of the position the seat believed it was moving in.
    ///
    /// Echoed from [`crate::Step::NeedDecision`]. The game compares it against
    /// its own and refuses the submission on a mismatch, which is the whole
    /// desync check: one `u64` per ply catches a mirror that has drifted, on the
    /// ply it drifts, rather than at the end of a corrupted game.
    ///
    /// Mandatory rather than optional. A check a seat may skip is a check that
    /// gets skipped.
    pub zobrist: u64,
    /// Opaque, seat-owned bytes, persisted verbatim and never interpreted.
    ///
    /// This is where visit counts, value estimates, and principal variations go.
    /// The game does not parse it and has no opinion about its shape, which is
    /// what lets the runner stay ignorant of what a model is while still being
    /// the single writer of the record.
    ///
    /// The previous implementation had this field, documented it as
    /// "transported into the position record", and then never read it — so every
    /// model package wrote its own training shards on a path that bypassed the
    /// runner entirely. This one is actually persisted.
    pub diagnostics: Option<Vec<u8>>,
}

impl Decision {
    /// A placement with no diagnostics.
    #[must_use]
    pub const fn new(action: Action, zobrist: u64) -> Self {
        Self {
            action,
            zobrist,
            diagnostics: None,
        }
    }

    /// Attach seat-owned bytes to this decision.
    #[must_use]
    pub fn with_diagnostics(mut self, bytes: Vec<u8>) -> Self {
        self.diagnostics = Some(bytes);
        self
    }
}

/// Why a driver could not get a decision out of a seat.
///
/// These are observations, not verdicts. The driver reports what happened; the
/// game decides what it costs, in [`crate::outcome`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Failure {
    /// The seat did not answer within its budget.
    Timeout,
    /// The seat's process died, or its transport broke.
    Crashed,
    /// The seat answered, but the answer could not be understood.
    Protocol,
}

/// Everything a seat can come back with.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Reply {
    /// A placement.
    Place(Decision),
    /// The seat gives up.
    ///
    /// Supported from the first version deliberately. Resignation did not exist
    /// in the previous protocol, and adding a message to a wire format that
    /// containers already speak is far more expensive than carrying one variant
    /// that is initially unused.
    Resign,
    /// The driver could not get a decision. The game adjudicates it.
    Failed(Failure),
}

#[cfg(test)]
mod tests {
    use super::*;
    use hexo_engine::HexCoord;

    #[test]
    fn only_a_wall_clock_is_irreproducible() {
        assert!(Budget::Unlimited.is_reproducible());
        assert!(Budget::Nodes(1_000).is_reproducible());
        assert!(Budget::Visits(800).is_reproducible());
        assert!(!Budget::Wall(Duration::from_millis(50)).is_reproducible());
    }

    #[test]
    fn the_default_budget_is_unlimited() {
        assert_eq!(Budget::default(), Budget::Unlimited);
    }

    #[test]
    fn diagnostics_are_absent_unless_attached() {
        let a = Action::new(HexCoord::ORIGIN);
        let plain = Decision::new(a, 0x1234);
        assert_eq!(plain.diagnostics, None);
        assert_eq!(plain.zobrist, 0x1234);

        let annotated = Decision::new(a, 0x1234).with_diagnostics(vec![1, 2, 3]);
        assert_eq!(annotated.diagnostics.as_deref(), Some(&[1u8, 2, 3][..]));
    }
}
