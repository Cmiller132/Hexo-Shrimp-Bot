//! Authoritative Hexo rules and game state.
//!
//! This crate owns the definition of a Hexo position and every legal
//! transition between positions. It is the single source of truth for what the
//! game *is*: coordinates, board storage, legality, and terminal detection.
//!
//! # The rules, in full
//!
//! - The board is an infinite hex grid in axial coordinates `(q, r)`, with the
//!   derived cube axis `s = -q - r`. Lines run along three axes:
//!   [`Axis::Q`] `(1, 0)`, [`Axis::R`] `(0, 1)`, [`Axis::QR`] `(1, -1)`.
//! - [`Player::P0`] moves first. Ply 0 is [`TurnPhase::Opening`]: `P0` must
//!   place at [`HexCoord::ORIGIN`]. After that each player places **two** stones
//!   per turn, giving the ply pattern `P0; P1 P1; P0 P0; P1 P1; ...`.
//! - A non-opening placement is legal iff the cell is empty and within
//!   [`LEGAL_RADIUS`] hex steps of **at least one** occupied cell — the union of
//!   radius-8 disks over *every* stone, not just the last one.
//! - A player wins when any six consecutive cells along one axis are all theirs.
//!   Six **or more** in a row wins; there is no overline rule.
//! - The win is checked after **every** placement, including the first stone of
//!   a two-stone turn. If the first stone wins, the second is never played: the
//!   phase and the mover freeze at their pre-move values.
//! - There is no draw, no pass, no resignation, no swap, and no ply cap. Stones
//!   are permanent. Zero legal moves can only occur in a terminal state.
//!
//! # Deliberate non-responsibilities
//!
//! Keeping these out is what lets this crate compile to `wasm32` and be tested
//! with `cargo test` alone:
//!
//! - No I/O, no clocks, no threads, no async. **Zero runtime dependencies.**
//! - No PyO3. Bindings live in a leaf crate that depends on this one.
//! - No model, tensor, or feature-encoding concepts. Encoders are consumers of
//!   the read surface this crate exposes, never part of it.
//! - No match orchestration or player communication. That is `hexo-runner`,
//!   which also owns non-win match results, ply caps, and game records.
//!
//! # Reading the position
//!
//! ```
//! use hexo_engine::{Action, HexCoord, Player, Position, TurnPhase};
//!
//! let mut pos = Position::new();
//! assert_eq!(pos.legal_count(), 1); // only the origin
//!
//! pos.advance(Action::new(HexCoord::ORIGIN)).unwrap();
//! assert_eq!(pos.current_player(), Player::P1);
//! assert_eq!(pos.phase(), TurnPhase::FirstStone);
//! assert_eq!(pos.legal_count(), 216); // the radius-8 disk, minus the origin
//!
//! // Legal moves come out in one canonical order: ascending `(q, r)`.
//! let first = pos.legal_actions().next().unwrap();
//! assert_eq!(first.coord(), HexCoord::new(-8, 0));
//! ```
//!
//! # Searching
//!
//! [`Position::advance`] is irreversible and is what the runner uses.
//! [`Search`] is the borrow-scoped make/unmake session, and the only path to
//! `undo`. It cannot rewind past the position it was seeded at.
//!
//! ```
//! use hexo_engine::{Action, HexCoord, Position, Search};
//!
//! let mut pos = Position::new();
//! pos.advance(Action::new(HexCoord::ORIGIN)).unwrap();
//! let floor = pos.clone();
//! {
//!     let mut search = Search::new(&mut pos);
//!     search.apply(Action::new(HexCoord::new(1, 0))).unwrap();
//!     assert_eq!(search.depth(), 1);
//! } // Drop unwinds to the floor.
//! assert_eq!(pos, floor);
//! ```

pub mod action;
pub mod coord;
pub mod error;
mod grid;
pub mod player;
pub mod position;
pub mod search;
pub mod window;
mod zobrist;

pub use action::{ACTION_ORDER_VERSION, Action, ActionId};
pub use coord::{Axis, COORD_LIMIT, DISK_CELLS, HexCoord, LEGAL_RADIUS, WINDOW_LEN, hex_distance};
pub use error::{IntegrityCheck, IntegrityError, MoveError};
pub use player::{Player, TurnPhase};
pub use position::{Applied, LegalActions, Outcome, Position, Stones};
pub use search::Search;
pub use window::{WINDOWS_PER_PLACEMENT, Window, WindowMask, WindowRef};

/// Version of the rules and of the Zobrist mixing function.
///
/// Bumping this invalidates cross-process hash agreement and every stored game
/// record. The Zobrist function rides inside this constant rather than getting
/// its own, because a hash change and a rule change invalidate the same
/// artefacts.
pub const RULES_VERSION: u32 = 1;

/// Hard ceiling on dense arena cells.
///
/// A representation limit, not a rule: a placement that would push the arena
/// past this is legal, and the engine reports that it cannot represent it.
///
/// The ceiling applies to the *area* of the padded bounding box of the stones,
/// not to either span on its own, because the arena is shaped to that box. So a
/// game that spreads along one axis is bounded by [`COORD_LIMIT`] rather than by
/// this constant — a straight walk reaches `|q| = 16000` at an arena of
/// 32768 x 128 cells, still under the ceiling — while a game that spreads in
/// every direction at once refuses once its padded box passes roughly
/// 2048 x 2048. Measured: a six-armed star extending every arm by
/// [`LEGAL_RADIUS`] per placement is refused at **ply 759**, with the outermost
/// stone at `|coord| = 1016`. Ordinary games are roughly an order of magnitude
/// shorter than that, and random play never approaches it.
///
/// It is a function of the position alone: rewinding a search never leaves a
/// position closer to the ceiling than a fresh replay of the same moves.
pub const MAX_GRID_CELLS: u64 = 1 << 22;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions_are_pinned() {
        assert_eq!(RULES_VERSION, 1);
        assert_eq!(ACTION_ORDER_VERSION, 1);
    }

    #[test]
    fn constants_are_self_consistent() {
        assert_eq!(WINDOW_LEN, 6);
        assert_eq!(LEGAL_RADIUS, 8);
        assert_eq!(DISK_CELLS, 3 * 8 * 9 + 1);
        assert_eq!(WINDOWS_PER_PLACEMENT, 18);
        assert_eq!(MAX_GRID_CELLS, 1 << 22);
        assert_eq!(COORD_LIMIT, 16_000);
    }
}
