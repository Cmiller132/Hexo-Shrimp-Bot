//! Engine error types.
//!
//! VENDORED CHANGE: the `thiserror` derive was replaced by hand-written
//! `Display`/`Error` impls reproducing the original `#[error(...)]` strings
//! verbatim, so this crate needs no dependencies. `StateLoadError` was dropped
//! along with `snapshot.rs`.

use crate::coord::HexCoord;
use std::fmt;

/// Errors produced when a placement violates the rules.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MoveError {
    /// `cannot apply a move to a terminal state`
    TerminalState,
    /// `opening placement must be at (0, 0)`
    IllegalOpening,
    /// `cell {0:?} is already occupied`
    Occupied(HexCoord),
    /// `cell {0:?} is not a legal placement`
    IllegalPlacement(HexCoord),
    /// `second placement cannot reuse the first placement`
    ReusedFirstStone,
}

impl fmt::Display for MoveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TerminalState => write!(f, "cannot apply a move to a terminal state"),
            Self::IllegalOpening => write!(f, "opening placement must be at (0, 0)"),
            Self::Occupied(coord) => write!(f, "cell {coord:?} is already occupied"),
            Self::IllegalPlacement(coord) => write!(f, "cell {coord:?} is not a legal placement"),
            Self::ReusedFirstStone => {
                write!(f, "second placement cannot reuse the first placement")
            }
        }
    }
}

impl std::error::Error for MoveError {}
