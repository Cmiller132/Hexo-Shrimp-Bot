//! The byte layout, stated once: every tag constant, and the encoder and decoder
//! for each shape a shard carries.
//!
//! The tag numbers are the format. They are written here as named constants so
//! that the encoder and the decoder cannot drift apart, and so that changing one
//! is visibly a format change rather than an edit to a literal. Every `match`
//! over a source enum is exhaustive with no catch-all arm: a variant added to
//! `hexo-engine` or `hexo-runner` must break this build, because the alternative
//! is a writer that silently encodes the wrong thing.

use crate::codec::{Cursor, put_i16, put_str, put_u8, put_u32, put_u64};
use crate::error::RecordError;
use crate::record::{GameRecord, ShardHeader, ShardMode};
use crate::{MAGIC, RECORDS_VERSION};
use hexo_engine::{ACTION_ORDER_VERSION, ActionId, HexCoord, MoveError, Player, RULES_VERSION};
use hexo_runner::{
    Budget, DrawReason, Failure, FailurePolicy, GameSpec, MatchResult, NoContest, PROTOCOL_VERSION,
    PlyRecord, WinReason,
};
use std::num::NonZeroU32;
use std::time::Duration;

/// [`ShardMode::SelfPlay`].
const MODE_SELF_PLAY: u8 = 0;
/// [`ShardMode::Eval`].
const MODE_EVAL: u8 = 1;

/// [`Player::P0`].
const SEAT_P0: u8 = 0;
/// [`Player::P1`].
const SEAT_P1: u8 = 1;

/// [`Budget::Unlimited`]; carries nothing.
const BUDGET_UNLIMITED: u8 = 0;
/// [`Budget::Nodes`]; carries a `u64`.
const BUDGET_NODES: u8 = 1;
/// [`Budget::Visits`]; carries a `u64`.
const BUDGET_VISITS: u8 = 2;
/// [`Budget::Wall`]; carries `u64` nanoseconds.
const BUDGET_WALL: u8 = 3;

/// [`FailurePolicy::Forfeit`].
const POLICY_FORFEIT: u8 = 0;
/// [`FailurePolicy::NoContest`].
const POLICY_NO_CONTEST: u8 = 1;

/// [`MatchResult::Decisive`].
const RESULT_DECISIVE: u8 = 0;
/// [`MatchResult::Drawn`].
const RESULT_DRAWN: u8 = 1;
/// [`MatchResult::NoContest`].
const RESULT_NO_CONTEST: u8 = 2;

/// [`WinReason::SixInARow`].
const WIN_SIX_IN_A_ROW: u8 = 0;
/// [`WinReason::Resignation`].
const WIN_RESIGNATION: u8 = 1;
/// [`WinReason::IllegalMove`]; carries the action id and the refusal.
const WIN_ILLEGAL_MOVE: u8 = 2;
/// [`WinReason::Timeout`].
const WIN_TIMEOUT: u8 = 3;
/// [`WinReason::Crash`].
const WIN_CRASH: u8 = 4;
/// [`WinReason::Protocol`].
const WIN_PROTOCOL: u8 = 5;
/// [`WinReason::Desync`]; carries both hashes.
const WIN_DESYNC: u8 = 6;

/// [`DrawReason::PlyCap`].
const DRAW_PLY_CAP: u8 = 0;

/// [`NoContest::EngineLimit`].
const NO_CONTEST_ENGINE_LIMIT: u8 = 0;
/// [`NoContest::SeatFailure`].
const NO_CONTEST_SEAT_FAILURE: u8 = 1;

/// [`Failure::Timeout`].
const FAILURE_TIMEOUT: u8 = 0;
/// [`Failure::Crashed`].
const FAILURE_CRASHED: u8 = 1;
/// [`Failure::Protocol`].
const FAILURE_PROTOCOL: u8 = 2;
/// [`Failure::Desync`]; carries both hashes.
const FAILURE_DESYNC: u8 = 3;

/// [`MoveError::TerminalState`].
const MOVE_TERMINAL_STATE: u8 = 0;
/// [`MoveError::IllegalOpening`].
const MOVE_ILLEGAL_OPENING: u8 = 1;
/// [`MoveError::CoordOutOfBounds`]; carries a coordinate.
const MOVE_COORD_OUT_OF_BOUNDS: u8 = 2;
/// [`MoveError::Occupied`]; carries a coordinate.
const MOVE_OCCUPIED: u8 = 3;
/// [`MoveError::TooFarFromStones`]; carries a coordinate.
const MOVE_TOO_FAR_FROM_STONES: u8 = 4;
/// [`MoveError::BoardExtentExceeded`]; carries a `u64` cell count.
const MOVE_BOARD_EXTENT_EXCEEDED: u8 = 5;

/// A ply with no diagnostics.
const DIAGNOSTICS_ABSENT: u8 = 0;
/// A ply with diagnostics, which follow as a `u32` length and that many bytes.
/// Distinct from absent even when the length is zero.
const DIAGNOSTICS_PRESENT: u8 = 1;

/// The fixed run of a header before the strings: magic, four versions, mode.
const HEADER_FIXED_PREFIX: u64 = 4 + 4 + 4 + 4 + 4 + 1;
/// The fixed run of a header after the strings: epoch and game count.
const HEADER_FIXED_SUFFIX: u64 = 4 + 4;

/// The most bytes a header can occupy, with all three strings at their `u16` cap.
///
/// A reader peeks this much of the file before decoding, which keeps one decoder
/// for the header rather than a streaming second implementation of the layout.
pub(crate) const HEADER_MAX_BYTES: u64 =
    HEADER_FIXED_PREFIX + 3 * (2 + u16::MAX as u64) + HEADER_FIXED_SUFFIX;

/// The `u32` byte-length prefix every game entry opens with, so a reader can
/// skip a game without decoding it.
pub(crate) const ENTRY_PREFIX_BYTES: usize = 4;

/// The smallest a ply can encode to: seat, action, hash, and the presence byte.
///
/// A decoder checks a ply count against this before allocating for it, so a
/// corrupt count cannot become a huge allocation.
const MIN_PLY_BYTES: usize = 1 + 4 + 8 + 1;

/// Write a header, and answer with the offset of its game-count field.
///
/// The caller writes the header at the start of the file, so the offset it gets
/// back is a file offset — which is what [`crate::ShardWriter::finalize`] seeks
/// to when it patches the true count in.
pub(crate) fn encode_header(out: &mut Vec<u8>, header: &ShardHeader) -> Result<u64, RecordError> {
    out.extend_from_slice(&MAGIC);
    put_u32(out, RECORDS_VERSION);
    put_u32(out, RULES_VERSION);
    put_u32(out, ACTION_ORDER_VERSION);
    put_u32(out, PROTOCOL_VERSION);
    put_u8(out, mode_tag(header.mode));
    put_str(out, "run id", &header.run_id)?;
    put_str(out, "package", &header.package)?;
    put_str(out, "checkpoint", &header.checkpoint)?;
    put_u32(out, header.epoch);
    let count_offset = out.len() as u64;
    put_u32(out, header.game_count);
    Ok(count_offset)
}

/// Read a header, refusing every version this build does not link.
pub(crate) fn decode_header(cursor: &mut Cursor<'_>) -> Result<ShardHeader, RecordError> {
    let mut magic = [0u8; 4];
    magic.copy_from_slice(cursor.take(4)?);
    if magic != MAGIC {
        return Err(RecordError::BadMagic { found: magic });
    }

    let records = cursor.u32()?;
    if records != RECORDS_VERSION {
        return Err(RecordError::RecordsVersion {
            expected: RECORDS_VERSION,
            found: records,
        });
    }
    let rules = cursor.u32()?;
    if rules != RULES_VERSION {
        return Err(RecordError::RulesVersion {
            expected: RULES_VERSION,
            found: rules,
        });
    }
    let order = cursor.u32()?;
    if order != ACTION_ORDER_VERSION {
        return Err(RecordError::ActionOrderVersion {
            expected: ACTION_ORDER_VERSION,
            found: order,
        });
    }
    let protocol = cursor.u32()?;
    if protocol != PROTOCOL_VERSION {
        return Err(RecordError::ProtocolVersion {
            expected: PROTOCOL_VERSION,
            found: protocol,
        });
    }

    let mode = read_mode(cursor)?;
    let run_id = cursor.string("run id")?;
    let package = cursor.string("package")?;
    let checkpoint = cursor.string("checkpoint")?;
    let epoch = cursor.u32()?;
    let game_count = cursor.u32()?;

    Ok(ShardHeader {
        mode,
        run_id,
        package,
        checkpoint,
        epoch,
        game_count,
    })
}

/// Write one game entry's payload, without its length prefix.
pub(crate) fn encode_game(out: &mut Vec<u8>, record: &GameRecord) -> Result<(), RecordError> {
    encode_spec(out, record)?;
    encode_result(out, record.result);

    let plies = u32::try_from(record.plies.len()).map_err(|_| RecordError::TooManyPlies {
        count: record.plies.len(),
    })?;
    put_u32(out, plies);
    for ply in &record.plies {
        encode_ply(out, ply)?;
    }
    Ok(())
}

/// Read one game entry's payload, having already consumed its length prefix.
pub(crate) fn decode_game(cursor: &mut Cursor<'_>) -> Result<GameRecord, RecordError> {
    let spec = decode_spec(cursor)?;
    let result = decode_result(cursor)?;

    let count = cursor.u32()? as usize;
    // Every ply occupies at least `MIN_PLY_BYTES`, so a count the entry cannot
    // hold is truncation — caught here rather than as an allocation of whatever
    // a corrupt `u32` happened to say.
    let needed = count.saturating_mul(MIN_PLY_BYTES);
    if needed > cursor.remaining() {
        return Err(RecordError::Truncated {
            offset: cursor.offset(),
            needed,
            available: cursor.remaining(),
        });
    }
    let mut plies = Vec::with_capacity(count);
    for _ in 0..count {
        plies.push(decode_ply(cursor)?);
    }

    Ok(GameRecord {
        spec,
        result,
        plies,
    })
}

/// Write the match rules: ply cap, budget, failure policy.
fn encode_spec(out: &mut Vec<u8>, record: &GameRecord) -> Result<(), RecordError> {
    put_u32(out, record.spec.ply_cap.get());
    match record.spec.budget {
        Budget::Unlimited => put_u8(out, BUDGET_UNLIMITED),
        Budget::Nodes(n) => {
            put_u8(out, BUDGET_NODES);
            put_u64(out, n);
        }
        Budget::Visits(n) => {
            put_u8(out, BUDGET_VISITS);
            put_u64(out, n);
        }
        Budget::Wall(duration) => {
            let nanos = duration.as_nanos();
            let nanos =
                u64::try_from(nanos).map_err(|_| RecordError::WallBudgetOverflow { nanos })?;
            put_u8(out, BUDGET_WALL);
            put_u64(out, nanos);
        }
    }
    put_u8(
        out,
        match record.spec.on_failure {
            FailurePolicy::Forfeit => POLICY_FORFEIT,
            FailurePolicy::NoContest => POLICY_NO_CONTEST,
        },
    );
    Ok(())
}

/// Read the match rules.
fn decode_spec(cursor: &mut Cursor<'_>) -> Result<GameSpec, RecordError> {
    // A ply cap is nonzero by type in the runner, so a file stating zero is
    // corrupt at that field rather than describing a game with no placements.
    let offset = cursor.offset();
    let ply_cap = NonZeroU32::new(cursor.u32()?).ok_or(RecordError::ZeroPlyCap { offset })?;

    let offset = cursor.offset();
    let budget = match cursor.u8()? {
        BUDGET_UNLIMITED => Budget::Unlimited,
        BUDGET_NODES => Budget::Nodes(cursor.u64()?),
        BUDGET_VISITS => Budget::Visits(cursor.u64()?),
        BUDGET_WALL => Budget::Wall(Duration::from_nanos(cursor.u64()?)),
        tag => {
            return Err(RecordError::BadTag {
                field: "budget",
                tag,
                offset,
            });
        }
    };

    let offset = cursor.offset();
    let on_failure = match cursor.u8()? {
        POLICY_FORFEIT => FailurePolicy::Forfeit,
        POLICY_NO_CONTEST => FailurePolicy::NoContest,
        tag => {
            return Err(RecordError::BadTag {
                field: "failure policy",
                tag,
                offset,
            });
        }
    };

    Ok(GameSpec {
        ply_cap,
        budget,
        on_failure,
    })
}

/// Write how the match ended, with its whole payload.
fn encode_result(out: &mut Vec<u8>, result: MatchResult) {
    match result {
        MatchResult::Decisive { winner, reason } => {
            put_u8(out, RESULT_DECISIVE);
            put_u8(out, seat_tag(winner));
            encode_win_reason(out, reason);
        }
        MatchResult::Drawn { reason } => {
            put_u8(out, RESULT_DRAWN);
            put_u8(
                out,
                match reason {
                    DrawReason::PlyCap => DRAW_PLY_CAP,
                },
            );
        }
        MatchResult::NoContest(no_contest) => {
            put_u8(out, RESULT_NO_CONTEST);
            match no_contest {
                NoContest::EngineLimit { seat, error } => {
                    put_u8(out, NO_CONTEST_ENGINE_LIMIT);
                    put_u8(out, seat_tag(seat));
                    encode_move_error(out, error);
                }
                NoContest::SeatFailure { seat, failure } => {
                    put_u8(out, NO_CONTEST_SEAT_FAILURE);
                    put_u8(out, seat_tag(seat));
                    encode_failure(out, failure);
                }
            }
        }
    }
}

/// Read how the match ended.
fn decode_result(cursor: &mut Cursor<'_>) -> Result<MatchResult, RecordError> {
    let offset = cursor.offset();
    match cursor.u8()? {
        RESULT_DECISIVE => {
            let winner = read_seat(cursor)?;
            let reason = decode_win_reason(cursor)?;
            Ok(MatchResult::Decisive { winner, reason })
        }
        RESULT_DRAWN => {
            let offset = cursor.offset();
            let reason = match cursor.u8()? {
                DRAW_PLY_CAP => DrawReason::PlyCap,
                tag => {
                    return Err(RecordError::BadTag {
                        field: "draw reason",
                        tag,
                        offset,
                    });
                }
            };
            Ok(MatchResult::Drawn { reason })
        }
        RESULT_NO_CONTEST => {
            let offset = cursor.offset();
            let no_contest = match cursor.u8()? {
                NO_CONTEST_ENGINE_LIMIT => NoContest::EngineLimit {
                    seat: read_seat(cursor)?,
                    error: decode_move_error(cursor)?,
                },
                NO_CONTEST_SEAT_FAILURE => NoContest::SeatFailure {
                    seat: read_seat(cursor)?,
                    failure: decode_failure(cursor)?,
                },
                tag => {
                    return Err(RecordError::BadTag {
                        field: "no-contest",
                        tag,
                        offset,
                    });
                }
            };
            Ok(MatchResult::NoContest(no_contest))
        }
        tag => Err(RecordError::BadTag {
            field: "match result",
            tag,
            offset,
        }),
    }
}

/// Write how a seat won, keeping the evidence the runner attached to it.
fn encode_win_reason(out: &mut Vec<u8>, reason: WinReason) {
    match reason {
        WinReason::SixInARow => put_u8(out, WIN_SIX_IN_A_ROW),
        WinReason::Resignation => put_u8(out, WIN_RESIGNATION),
        WinReason::IllegalMove { action, cause } => {
            put_u8(out, WIN_ILLEGAL_MOVE);
            put_u32(out, action.0);
            encode_move_error(out, cause);
        }
        WinReason::Timeout => put_u8(out, WIN_TIMEOUT),
        WinReason::Crash => put_u8(out, WIN_CRASH),
        WinReason::Protocol => put_u8(out, WIN_PROTOCOL),
        WinReason::Desync { expected, got } => {
            put_u8(out, WIN_DESYNC);
            put_u64(out, expected);
            put_u64(out, got);
        }
    }
}

/// Read how a seat won.
fn decode_win_reason(cursor: &mut Cursor<'_>) -> Result<WinReason, RecordError> {
    let offset = cursor.offset();
    match cursor.u8()? {
        WIN_SIX_IN_A_ROW => Ok(WinReason::SixInARow),
        WIN_RESIGNATION => Ok(WinReason::Resignation),
        WIN_ILLEGAL_MOVE => Ok(WinReason::IllegalMove {
            action: ActionId(cursor.u32()?),
            cause: decode_move_error(cursor)?,
        }),
        WIN_TIMEOUT => Ok(WinReason::Timeout),
        WIN_CRASH => Ok(WinReason::Crash),
        WIN_PROTOCOL => Ok(WinReason::Protocol),
        WIN_DESYNC => Ok(WinReason::Desync {
            expected: cursor.u64()?,
            got: cursor.u64()?,
        }),
        tag => Err(RecordError::BadTag {
            field: "win reason",
            tag,
            offset,
        }),
    }
}

/// Write a driver-reported failure.
fn encode_failure(out: &mut Vec<u8>, failure: Failure) {
    match failure {
        Failure::Timeout => put_u8(out, FAILURE_TIMEOUT),
        Failure::Crashed => put_u8(out, FAILURE_CRASHED),
        Failure::Protocol => put_u8(out, FAILURE_PROTOCOL),
        Failure::Desync { expected, got } => {
            put_u8(out, FAILURE_DESYNC);
            put_u64(out, expected);
            put_u64(out, got);
        }
    }
}

/// Read a driver-reported failure.
fn decode_failure(cursor: &mut Cursor<'_>) -> Result<Failure, RecordError> {
    let offset = cursor.offset();
    match cursor.u8()? {
        FAILURE_TIMEOUT => Ok(Failure::Timeout),
        FAILURE_CRASHED => Ok(Failure::Crashed),
        FAILURE_PROTOCOL => Ok(Failure::Protocol),
        FAILURE_DESYNC => Ok(Failure::Desync {
            expected: cursor.u64()?,
            got: cursor.u64()?,
        }),
        tag => Err(RecordError::BadTag {
            field: "failure",
            tag,
            offset,
        }),
    }
}

/// Write the engine's refusal, coordinate and all.
fn encode_move_error(out: &mut Vec<u8>, error: MoveError) {
    match error {
        MoveError::TerminalState => put_u8(out, MOVE_TERMINAL_STATE),
        MoveError::IllegalOpening => put_u8(out, MOVE_ILLEGAL_OPENING),
        MoveError::CoordOutOfBounds(coord) => {
            put_u8(out, MOVE_COORD_OUT_OF_BOUNDS);
            put_coord(out, coord);
        }
        MoveError::Occupied(coord) => {
            put_u8(out, MOVE_OCCUPIED);
            put_coord(out, coord);
        }
        MoveError::TooFarFromStones(coord) => {
            put_u8(out, MOVE_TOO_FAR_FROM_STONES);
            put_coord(out, coord);
        }
        MoveError::BoardExtentExceeded { cells } => {
            put_u8(out, MOVE_BOARD_EXTENT_EXCEEDED);
            put_u64(out, cells);
        }
    }
}

/// Read the engine's refusal.
fn decode_move_error(cursor: &mut Cursor<'_>) -> Result<MoveError, RecordError> {
    let offset = cursor.offset();
    match cursor.u8()? {
        MOVE_TERMINAL_STATE => Ok(MoveError::TerminalState),
        MOVE_ILLEGAL_OPENING => Ok(MoveError::IllegalOpening),
        MOVE_COORD_OUT_OF_BOUNDS => Ok(MoveError::CoordOutOfBounds(read_coord(cursor)?)),
        MOVE_OCCUPIED => Ok(MoveError::Occupied(read_coord(cursor)?)),
        MOVE_TOO_FAR_FROM_STONES => Ok(MoveError::TooFarFromStones(read_coord(cursor)?)),
        MOVE_BOARD_EXTENT_EXCEEDED => Ok(MoveError::BoardExtentExceeded {
            cells: cursor.u64()?,
        }),
        tag => Err(RecordError::BadTag {
            field: "move error",
            tag,
            offset,
        }),
    }
}

/// Write one placement: seat, action, hash after, diagnostics.
fn encode_ply(out: &mut Vec<u8>, ply: &PlyRecord) -> Result<(), RecordError> {
    put_u8(out, seat_tag(ply.seat));
    put_u32(out, ply.action.0);
    put_u64(out, ply.zobrist_after);
    match &ply.diagnostics {
        None => put_u8(out, DIAGNOSTICS_ABSENT),
        Some(bytes) => {
            let len = u32::try_from(bytes.len())
                .map_err(|_| RecordError::DiagnosticsTooLong { len: bytes.len() })?;
            put_u8(out, DIAGNOSTICS_PRESENT);
            put_u32(out, len);
            out.extend_from_slice(bytes);
        }
    }
    Ok(())
}

/// Read one placement.
fn decode_ply(cursor: &mut Cursor<'_>) -> Result<PlyRecord, RecordError> {
    let seat = read_seat(cursor)?;
    let action = ActionId(cursor.u32()?);
    let zobrist_after = cursor.u64()?;

    let offset = cursor.offset();
    let diagnostics = match cursor.u8()? {
        DIAGNOSTICS_ABSENT => None,
        DIAGNOSTICS_PRESENT => {
            let len = cursor.u32()? as usize;
            Some(cursor.take(len)?.to_vec())
        }
        tag => {
            return Err(RecordError::BadTag {
                field: "diagnostics presence",
                tag,
                offset,
            });
        }
    };

    Ok(PlyRecord {
        seat,
        action,
        zobrist_after,
        diagnostics,
    })
}

/// Write a coordinate as two `i16`s.
fn put_coord(out: &mut Vec<u8>, coord: HexCoord) {
    put_i16(out, coord.q);
    put_i16(out, coord.r);
}

/// Read a coordinate.
fn read_coord(cursor: &mut Cursor<'_>) -> Result<HexCoord, RecordError> {
    let q = cursor.i16()?;
    let r = cursor.i16()?;
    Ok(HexCoord::new(q, r))
}

/// The tag for a shard mode.
const fn mode_tag(mode: ShardMode) -> u8 {
    match mode {
        ShardMode::SelfPlay => MODE_SELF_PLAY,
        ShardMode::Eval => MODE_EVAL,
    }
}

/// Read a shard mode.
fn read_mode(cursor: &mut Cursor<'_>) -> Result<ShardMode, RecordError> {
    let offset = cursor.offset();
    match cursor.u8()? {
        MODE_SELF_PLAY => Ok(ShardMode::SelfPlay),
        MODE_EVAL => Ok(ShardMode::Eval),
        tag => Err(RecordError::BadTag {
            field: "shard mode",
            tag,
            offset,
        }),
    }
}

/// The tag for a seat.
const fn seat_tag(seat: Player) -> u8 {
    match seat {
        Player::P0 => SEAT_P0,
        Player::P1 => SEAT_P1,
    }
}

/// Read a seat.
fn read_seat(cursor: &mut Cursor<'_>) -> Result<Player, RecordError> {
    let offset = cursor.offset();
    match cursor.u8()? {
        SEAT_P0 => Ok(Player::P0),
        SEAT_P1 => Ok(Player::P1),
        tag => Err(RecordError::BadTag {
            field: "seat",
            tag,
            offset,
        }),
    }
}
