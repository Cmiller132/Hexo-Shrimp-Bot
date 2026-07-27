//! The independent detector: replaying a record through the engine.

use crate::error::RecordError;
use crate::record::GameRecord;
use hexo_engine::{Action, Position};
use hexo_runner::{DrawReason, MatchResult, NoContest, WinReason};

/// Replay a record through the engine and check that it says what it claims.
///
/// Parsing proves a shard is well-formed. It cannot prove the shard is *the game
/// that was played*: an action id, a hash, or a seat byte that drifted still
/// decodes into a perfectly valid field, and a training pipeline would learn
/// from it without noticing. This replays the move list from the empty position
/// and checks it against everything the record independently claims:
///
/// - each ply's recorded mover is the seat the replay has on turn;
/// - each placement is legal in the position it is played into;
/// - each ply's `zobrist_after` is the hash the replay reaches — the whole
///   chain, not just the last one, so a drifted move is located at the ply it
///   drifted rather than at the end of a corrupted game;
/// - a [`WinReason::SixInARow`] result lands on a terminal position won by the
///   seat the record names, and every other ending — a resignation, a forfeit, a
///   ply cap, a no-contest — lands on a position that is *not* terminal, because
///   the runner would have adjudicated a completed six-in-a-row as one.
///
/// It deliberately does not re-adjudicate. Whether a ply cap fell where the
/// spec's cap says it should is match policy, and `hexo-runner` owns that; a
/// second implementation of it here would be a second answer to a question that
/// already has one. What is checked instead is engine fact, which the record and
/// the engine state independently.
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

    // Exhaustive, with no catch-all: every way a game can end is either a claim
    // that the board is won or a claim that it is not, and a variant added to
    // either enum has to be classified here rather than defaulting to unchecked.
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
