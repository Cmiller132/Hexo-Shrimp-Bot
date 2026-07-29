//! The training loop behind the `hexo-bot` binary: batched self-play, fitting,
//! checkpoints, and evaluation, from one long-lived process.
//!
//! `docs/CONTAINER_SPEC.md` defines the normative contract. The binary parses a
//! command line, installs a stop handler, calls this library, and maps its
//! outcome to an exit code.
//!
//! # Public operations
//!
//! [`init_checkpoint`], [`train`], and [`play_match`] are the whole surface.
//! The wire-protocol `serve` and `play` operations named by §3 are not exposed.
//!
//! # The shape of a run
//!
//! ```text
//! for epoch in start..epochs:
//!     self-play   G concurrent games under frozen weights -> records/<epoch>/
//!     fit         the package consumes them -> checkpoints/<epoch+1>/
//!     load        which is what proves the fit's own output
//!     eval        every K epochs: against the anchor and against the previous
//!     metrics     one line, appended as it happens
//! ```
//!
//! One sweep implementation serves self-play, evaluation, and matches. The
//! crate README and `driver` module define its topology and queue invariants.

pub mod cli;
pub mod registry;

mod driver;
mod error;
mod init;
mod matches;
mod metrics;
mod run;
mod train;

pub use cli::{Command, InitConfig, MatchConfig, SeatSpec, TrainConfig, USAGE, parse, seat};
pub use error::BotError;
pub use init::init_checkpoint;
pub use matches::{MatchReport, MatchRun, SeatReport, play_match};
pub use train::train;

/// How a subcommand ended, when it did not fail.
///
/// `docs/CONTAINER_SPEC.md` §8.1 assigns exit code **0** to completion, **2**
/// to a clean signal-driven stop, and **1** to failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Outcome {
    /// Everything that was asked for happened.
    Completed,
    /// The stop flag was set, and the work in flight was wound up cleanly.
    Stopped,
}

impl Outcome {
    /// The process exit code this outcome maps to.
    #[must_use]
    pub const fn exit_code(self) -> u8 {
        match self {
            Self::Completed => 0,
            Self::Stopped => 2,
        }
    }
}
