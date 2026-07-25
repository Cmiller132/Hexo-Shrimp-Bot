//! Hexo match orchestration: the authoritative game, and the policy that
//! decides how a match ends.
//!
//! [`Game`] owns the one canonical [`hexo_engine::Position`] for a game and is
//! the only code permitted to advance it. It is a **state machine, not a loop**:
//! it never blocks, holds no player handle, and has no transport, clock, or I/O.
//! A caller asks it what it wants with [`Game::step`] and tells it what happened
//! with [`Game::submit`]. See the [`game`] module docs for why that shape rather
//! than a `Player` trait the runner calls.
//!
//! What it owns that the engine refuses to model: ply caps, non-win results,
//! adjudication, and the record. It still does not know what a model is.
//!
//! # Driving one game on one thread
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
//!
//! Driving ten thousand games instead means holding ten thousand `Game` values,
//! sweeping them for everyone in [`Step::NeedDecision`], handing that whole set
//! to a batched evaluator, and submitting what comes back. Same type, same two
//! methods, no threads.
//!
//! # The three things a driver must not get wrong
//!
//! - **Do not call `is_legal` before submitting.** [`Game::submit`] validates,
//!   and a rejection is a *result*, not an error. Pre-checking duplicates the
//!   engine's hot path and moves adjudication into the driver.
//! - **Echo the `zobrist`.** A seat maintaining its own mirror proves it has not
//!   drifted by sending back the hash it moved in. One `u64` per ply catches a
//!   desync on the ply it happens.
//! - **Submit the `generation` you were given.** It is what stops a late or
//!   duplicated reply from playing a move chosen for a position the game has
//!   moved past.

pub mod decision;
pub mod error;
pub mod game;
pub mod outcome;

pub use decision::{Budget, Decision, Failure, Reply};
pub use error::SubmitError;
pub use game::{FailurePolicy, Game, GameSpec, PlyRecord, Step, Transition};
pub use outcome::{DrawReason, MatchResult, NoContest, WinReason};

/// Version of the runner's decision and result model.
///
/// Distinct from [`hexo_engine::RULES_VERSION`]: the rules can hold still while
/// adjudication, the budget vocabulary, or the failure policy change, and a
/// stored record has to say which of both it was produced under.
pub const PROTOCOL_VERSION: u32 = 1;
