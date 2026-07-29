//! Engine replay verification for decoded records.

use crate::error::RecordError;
use crate::record::GameRecord;
use hexo_engine::{Action, Position};
use hexo_runner::{DrawReason, MatchResult, NoContest, WinReason};

/// Replay a record through the engine and check that it says what it claims.
///
/// This replays the move list from the empty position and checks:
///
/// - each ply's recorded mover is the seat the replay has on turn;
/// - each placement is legal in the position it is played into;
/// - each ply's `zobrist_after` equals the replayed hash;
/// - a [`WinReason::SixInARow`] result matches the terminal winner;
/// - every other result ends on a nonterminal position.
///
/// This function does not re-run runner policy such as ply-cap adjudication.
///
/// # Errors
///
/// [`RecordError::SeatMismatch`], [`RecordError::ReplayRefused`], or
/// [`RecordError::ZobristMismatch`], each naming the ply; or
/// [`RecordError::NotTerminal`], [`RecordError::UnexpectedTerminal`], or
/// [`RecordError::WinnerMismatch`] for a result the board disagrees with.
pub fn verify(record: &GameRecord) -> Result<(), RecordError> {
    let mut position = Position::new();

    for (ply, entry) in record.plies.iter().enumerate() {
        let on_turn = position.current_player();
        if on_turn != entry.seat {
            return Err(RecordError::SeatMismatch {
                ply,
                recorded: entry.seat,
                replayed: on_turn,
            });
        }
        position
            .advance(Action::from_id(entry.action))
            .map_err(|cause| RecordError::ReplayRefused { ply, cause })?;
        let replayed = position.zobrist();
        if replayed != entry.zobrist_after {
            return Err(RecordError::ZobristMismatch {
                ply,
                recorded: entry.zobrist_after,
                replayed,
            });
        }
    }

    // Exhaustive matching requires every result variant to declare whether it
    // claims an engine-terminal board.
    let claimed_winner = match record.result {
        MatchResult::Decisive {
            winner,
            reason: WinReason::SixInARow,
        } => Some(winner),
        MatchResult::Decisive {
            reason:
                WinReason::Resignation
                | WinReason::IllegalMove { .. }
                | WinReason::Timeout
                | WinReason::Crash
                | WinReason::Protocol
                | WinReason::Desync { .. },
            ..
        }
        | MatchResult::Drawn {
            reason: DrawReason::PlyCap,
        }
        | MatchResult::NoContest(NoContest::EngineLimit { .. } | NoContest::SeatFailure { .. }) => {
            None
        }
    };

    match (claimed_winner, position.outcome()) {
        (Some(_), None) => Err(RecordError::NotTerminal),
        (Some(recorded), Some(outcome)) if outcome.winner != recorded => {
            Err(RecordError::WinnerMismatch {
                recorded,
                replayed: outcome.winner,
            })
        }
        (Some(_), Some(_)) | (None, None) => Ok(()),
        (None, Some(_)) => Err(RecordError::UnexpectedTerminal),
    }
}
