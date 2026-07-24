//! Hexo match orchestration: the authoritative game loop and player communication.
//!
//! The runner holds the one canonical [`hexo_engine`] state for a game and is
//! the only code permitted to advance it. Players never receive a mutable
//! handle to it; they get their own state to search on and submit candidate
//! moves back, which the runner validates before applying.
//!
//! Responsibilities:
//!
//! - Own the canonical position and the move record for a game.
//! - Drive the turn loop and decide when a game is over.
//! - Speak to players over a transport-agnostic interface, so a player may be
//!   an in-process function, a subprocess, or a container.
//! - Own the failure policy: illegal moves, timeouts, crashes, resignation.
//!
//! It does not know what a model is.
//!
//! Empty pending the rules rebuild. See `README.md` for the planned module map.
