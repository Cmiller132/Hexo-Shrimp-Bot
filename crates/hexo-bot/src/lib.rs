//! The training loop behind the `hexo-bot` binary: batched self-play, fitting,
//! checkpoints, and evaluation, from one long-lived process.
//!
//! `docs/CONTAINER_SPEC.md` is the normative target. The binary is a thin shell
//! over this library — it parses a command line, installs a stop handler, calls
//! one of three entry points, and maps what comes back onto an exit code — so the
//! loop is driven in-process by the test suite rather than by spawning a
//! process and reading its output.
//!
//! # Three subcommands, and why not five
//!
//! [`init_checkpoint`], [`train`], and [`play_match`] are the whole surface.
//! `serve` and `play` are named by §3 of the spec and are deliberately *not*
//! here, not even as stubs that parse a flag and exit: both are entirely wire
//! protocol, there is no wire protocol yet, and a stub would publish a command
//! line before the thing behind it is designed — after which the flags it
//! guessed become the constraint the real implementation has to argue its way
//! out of.
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
//! One implementation of the sweep serves all three phases that play games. The
//! topology, the queues, and the argument for why it cannot deadlock are in the
//! crate's `README.md` and in the `driver` module's own documentation.

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
/// `docs/CONTAINER_SPEC.md` §8.1 pins the three exits so that a supervisor, a
/// shell loop, or a person reading `docker inspect` does not have to infer them:
/// **0** ran to completion, **2** stopped by signal after finishing cleanly,
/// **1** failed. A run that ended is not a run that broke, and a `docker stop`
/// produces a 2 — which is why this is a return value rather than a log line.
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
