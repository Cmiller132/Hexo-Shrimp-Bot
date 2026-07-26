//! Hexo match orchestration: the authoritative game, and the policy that decides how a
//! match ends.
//!
//! ```
//! use hexo_runner::{Decision, Game, GameSpec, Reply, Step};
//!
//! let mut game = Game::new(GameSpec::default());
//! loop {
//!     match game.step() {
//!         Step::Finished(result) => break result,
//!         Step::NeedDecision { generation, zobrist, .. } => {
//!             // A real driver asks a seat here, over whatever transport it
//!             // likes, and may block for as long as it wants. The game does
//!             // not care, because it is not involved.
//!             let action = game.position().nth_legal(0).expect("a legal move");
//!             let reply = Reply::Place(Decision::new(action, zobrist));
//!             game.submit(generation, reply).expect("fresh generation");
//!         }
//!     }
//! };
//! ```

pub mod decision;
pub mod error;
pub mod game;
pub mod outcome;

pub use decision::{Budget, Decision, Failure, Reply};
pub use error::SubmitError;
pub use game::{FailurePolicy, Game, GameSpec, PlyRecord, Step, Transition};
pub use outcome::{DrawReason, MatchResult, NoContest, WinReason};

/// Version of the runner's decision and result model.
pub const PROTOCOL_VERSION: u32 = 1;
