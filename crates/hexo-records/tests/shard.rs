//! Shard round trips, reader refusals, and replay verification.

use hexo_engine::{Action, ActionId, HexCoord, MoveError, Player};
use hexo_records::{
    GameRecord, RecordError, ShardHeader, ShardMode, ShardReader, ShardWriter, verify,
};
use hexo_runner::{
    Budget, Decision, DrawReason, Failure, FailurePolicy, Game, GameSpec, MatchResult, NoContest,
    PlyRecord, Reply, Step, WinReason,
};
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::time::Duration;

// ---------------------------------------------------------------- fixtures --

fn act(q: i16, r: i16) -> Action {
    Action::new(HexCoord::new(q, r))
}

fn cap(value: u32) -> NonZeroU32 {
    NonZeroU32::new(value).expect("test caps are nonzero")
}

/// A header with distinct, short strings, and the zero game count the writer requires.
fn header(mode: ShardMode) -> ShardHeader {
    ShardHeader {
        mode,
        run_id: "run-7".to_owned(),
        package: "mock-package".to_owned(),
        checkpoint: "run-7/epoch-3".to_owned(),
        epoch: 3,
        game_count: 0,
    }
}

/// A scratch directory and a path inside it that no file occupies yet.
fn scratch() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("0003.hxr");
    (dir, path)
}

/// `P1` builds a six-line along axis `Q` while `P0` takes the lowest-ranked legal
/// move. Ends decisively for `P1` on the eleventh placement.
fn a_won_game() -> Game {
    const LINE: [(i16, i16); 6] = [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)];
    let mut game = Game::new(GameSpec {
        budget: Budget::Nodes(4096),
        ..GameSpec::default()
    });
    let mut next = 0;
    while let Step::NeedDecision {
        seat,
        generation,
        zobrist,
        ..
    } = game.step()
    {
        let action = if seat == Player::P1 && next < LINE.len() {
            let (q, r) = LINE[next];
            next += 1;
            act(q, r)
        } else {
            game.position().nth_legal(0).expect("a legal move")
        };
        game.submit(generation, Reply::Place(Decision::new(action, zobrist)))
            .expect("accepted");
    }
    game
}

/// Both seats take the lowest-ranked legal move until the cap draws the game.
fn a_capped_game(ply_cap: u32, budget: Budget) -> Game {
    let mut game = Game::new(GameSpec {
        ply_cap: cap(ply_cap),
        budget,
        on_failure: FailurePolicy::NoContest,
    });
    while let Step::NeedDecision {
        generation,
        zobrist,
        ..
    } = game.step()
    {
        let action = game.position().nth_legal(0).expect("a legal move");
        game.submit(generation, Reply::Place(Decision::new(action, zobrist)))
            .expect("accepted");
    }
    game
}

/// One placement, then `P1` gives up.
fn a_resigned_game() -> Game {
    let mut game = Game::new(GameSpec::default());
    let Step::NeedDecision {
        generation,
        zobrist,
        ..
    } = game.step()
    else {
        panic!("a new game wants a decision");
    };
    game.submit(generation, Reply::Place(Decision::new(act(0, 0), zobrist)))
        .expect("opening");
    let Step::NeedDecision { generation, .. } = game.step() else {
        panic!("the game is not over after one stone");
    };
    game.submit(generation, Reply::Resign).expect("accepted");
    game
}

/// Three placements carrying, in order, no diagnostics, empty diagnostics, and
/// four bytes of them.
fn a_game_with_every_diagnostics_state() -> Game {
    let mut game = Game::new(GameSpec {
        ply_cap: cap(3),
        ..GameSpec::default()
    });
    let attach: [Option<Vec<u8>>; 3] = [None, Some(Vec::new()), Some(vec![0xDE, 0xAD, 0xBE, 0xEF])];
    let mut next = attach.into_iter();
    while let Step::NeedDecision {
        generation,
        zobrist,
        ..
    } = game.step()
    {
        let action = game.position().nth_legal(0).expect("a legal move");
        let mut decision = Decision::new(action, zobrist);
        decision.diagnostics = next.next().expect("three placements, three states");
        game.submit(generation, Reply::Place(decision))
            .expect("accepted");
    }
    game
}

fn record_of(game: &Game) -> GameRecord {
    GameRecord::from_game(game).expect("a finished game is a record")
}

fn write_shard(path: &Path, head: &ShardHeader, records: &[GameRecord]) {
    let mut writer = ShardWriter::create(path, head).expect("a fresh path");
    for record in records {
        writer.append(record).expect("appended");
    }
    writer.finalize().expect("finalized");
}

/// The header, and every game, or the first error the reader raised.
fn read_shard(path: &Path) -> (ShardHeader, Result<Vec<GameRecord>, RecordError>) {
    let reader = ShardReader::open(path).expect("opened");
    let head = reader.header().clone();
    (head, reader.collect())
}

/// The game-count offset derived independently from the documented header
/// layout.
fn game_count_offset(head: &ShardHeader) -> usize {
    4 + 4 * 4
        + 1
        + (2 + head.run_id.len())
        + (2 + head.package.len())
        + (2 + head.checkpoint.len())
        + 4
}

/// The offset of the rules-version field, from the same layout.
const RULES_VERSION_OFFSET: usize = 4 + 4;

// -------------------------------------------------------------- round trips --

#[test]
fn a_shard_round_trips_every_game_it_was_given() {
    let (_dir, path) = scratch();
    let games = [
        a_won_game(),
        a_capped_game(12, Budget::Wall(Duration::from_millis(250))),
        a_resigned_game(),
        a_game_with_every_diagnostics_state(),
    ];
    let written: Vec<GameRecord> = games.iter().map(record_of).collect();

    let mut expected = header(ShardMode::SelfPlay);
    write_shard(&path, &expected, &written);

    let (head, records) = read_shard(&path);
    let records = records.expect("every game reads back");
    expected.game_count = 4;
    assert_eq!(
        head, expected,
        "the header is what was written, plus the count"
    );
    assert_eq!(records, written, "every record is what was written");

    for (i, record) in records.iter().enumerate() {
        verify(record).unwrap_or_else(|e| panic!("game {i} does not replay: {e}"));
    }
}

/// Fabricated, non-replayable plies used to cover every result payload during
/// wire-format round trips.
#[test]
fn every_result_arm_round_trips_with_its_payload() {
    let (_dir, path) = scratch();
    let coord = HexCoord::new(-13, 21);
    let move_errors = [
        MoveError::TerminalState,
        MoveError::IllegalOpening,
        MoveError::CoordOutOfBounds(coord),
        MoveError::Occupied(coord),
        MoveError::TooFarFromStones(coord),
        MoveError::BoardExtentExceeded {
            cells: 0x0123_4567_89AB_CDEF,
        },
    ];
    let failures = [
        Failure::Timeout,
        Failure::Crashed,
        Failure::Protocol,
        Failure::Desync {
            expected: 0xFEED_FACE_CAFE_BEEF,
            got: 0x0BAD_F00D_1234_5678,
        },
    ];

    let mut results = vec![
        MatchResult::Decisive {
            winner: Player::P0,
            reason: WinReason::SixInARow,
        },
        MatchResult::Decisive {
            winner: Player::P1,
            reason: WinReason::Resignation,
        },
        MatchResult::Decisive {
            winner: Player::P0,
            reason: WinReason::Timeout,
        },
        MatchResult::Decisive {
            winner: Player::P1,
            reason: WinReason::Crash,
        },
        MatchResult::Decisive {
            winner: Player::P0,
            reason: WinReason::Protocol,
        },
        MatchResult::Decisive {
            winner: Player::P1,
            reason: WinReason::Desync {
                expected: u64::MAX,
                got: 1,
            },
        },
        MatchResult::Drawn {
            reason: DrawReason::PlyCap,
        },
    ];
    for error in move_errors {
        results.push(MatchResult::Decisive {
            winner: Player::P1,
            reason: WinReason::IllegalMove {
                action: act(-13, 21).id(),
                cause: error,
            },
        });
        results.push(MatchResult::NoContest(NoContest::EngineLimit {
            seat: Player::P0,
            error,
        }));
    }
    for failure in failures {
        results.push(MatchResult::NoContest(NoContest::SeatFailure {
            seat: Player::P1,
            failure,
        }));
    }

    let budgets = [
        Budget::Unlimited,
        Budget::Nodes(u64::MAX),
        Budget::Visits(1),
        // The largest wall budget the format states, exactly.
        Budget::Wall(Duration::from_nanos(u64::MAX)),
    ];
    let policies = [FailurePolicy::Forfeit, FailurePolicy::NoContest];

    let written: Vec<GameRecord> = results
        .into_iter()
        .enumerate()
        .map(|(i, result)| GameRecord {
            spec: GameSpec {
                ply_cap: cap(i as u32 + 1),
                budget: budgets[i % budgets.len()],
                on_failure: policies[i % policies.len()],
            },
            result,
            plies: vec![PlyRecord {
                seat: Player::P1,
                action: ActionId(0xDEAD_BEEF),
                zobrist_after: 0x1122_3344_5566_7788,
                diagnostics: None,
            }],
        })
        .collect();

    write_shard(&path, &header(ShardMode::Eval), &written);
    let (head, records) = read_shard(&path);
    assert_eq!(head.mode, ShardMode::Eval);
    assert_eq!(head.game_count as usize, written.len());
    assert_eq!(records.expect("every arm reads back"), written);
}

#[test]
fn diagnostics_are_absent_empty_and_present_as_three_distinct_states() {
    let (_dir, path) = scratch();
    let written = record_of(&a_game_with_every_diagnostics_state());
    assert_eq!(written.plies.len(), 3);

    write_shard(
        &path,
        &header(ShardMode::SelfPlay),
        std::slice::from_ref(&written),
    );
    let (_, records) = read_shard(&path);
    let records = records.expect("read back");
    let plies = &records[0].plies;

    assert_eq!(plies[0].diagnostics, None);
    assert_eq!(plies[1].diagnostics, Some(Vec::new()));
    assert_eq!(plies[2].diagnostics, Some(vec![0xDE, 0xAD, 0xBE, 0xEF]));
    assert_ne!(plies[0].diagnostics, plies[1].diagnostics);
    assert_eq!(records[0], written);
    verify(&records[0]).expect("a real game replays");
}

// --------------------------------------------------------------- detection --

/// A syntactically valid action-id mutation must fail decoding or replay
/// verification.
#[test]
fn a_flipped_byte_in_a_move_list_is_caught_by_parsing_or_by_verify() {
    let (_dir, path) = scratch();
    write_shard(
        &path,
        &header(ShardMode::SelfPlay),
        &[record_of(&a_won_game())],
    );

    // `P1`'s fourth line stone, (3, 1), in the record encoding.
    let target = act(3, 1).id().0.to_le_bytes();
    let mut bytes = std::fs::read(&path).expect("read back");
    let hits: Vec<usize> = bytes
        .windows(4)
        .enumerate()
        .filter(|(_, w)| *w == target)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(hits.len(), 1, "the action id must occur exactly once");

    // (3, 1) becomes (3, 5): still empty, still within the legal radius, so the
    // move replays and only the hash chain disagrees.
    bytes[hits[0]] ^= 0x04;
    std::fs::write(&path, &bytes).expect("patched");

    let (_, records) = read_shard(&path);
    match records {
        Err(_) => {}
        Ok(records) => {
            let caught = records.iter().any(|r| verify(r).is_err());
            assert!(
                caught,
                "a flipped action id parsed cleanly and verify did not catch it"
            );
        }
    }
}

#[test]
fn verify_catches_a_hash_that_does_not_follow_the_moves() {
    let mut record = record_of(&a_capped_game(9, Budget::Unlimited));
    record.plies[4].zobrist_after ^= 1;

    match verify(&record) {
        Err(RecordError::ZobristMismatch { ply, .. }) => assert_eq!(ply, 4),
        other => panic!("expected a hash mismatch at ply 4, got {other:?}"),
    }
}

#[test]
fn verify_catches_a_seat_the_replay_disagrees_with() {
    let mut record = record_of(&a_capped_game(9, Budget::Unlimited));
    record.plies[2].seat = record.plies[2].seat.other();

    match verify(&record) {
        Err(RecordError::SeatMismatch { ply, .. }) => assert_eq!(ply, 2),
        other => panic!("expected a seat mismatch at ply 2, got {other:?}"),
    }
}

#[test]
fn verify_catches_a_result_the_board_does_not_show() {
    let mut claims_a_win = record_of(&a_capped_game(9, Budget::Unlimited));
    claims_a_win.result = MatchResult::Decisive {
        winner: Player::P0,
        reason: WinReason::SixInARow,
    };
    assert!(matches!(
        verify(&claims_a_win),
        Err(RecordError::NotTerminal)
    ));

    let mut claims_the_wrong_winner = record_of(&a_won_game());
    verify(&claims_the_wrong_winner).expect("the game as played verifies");
    claims_the_wrong_winner.result = MatchResult::Decisive {
        winner: Player::P0,
        reason: WinReason::SixInARow,
    };
    assert!(matches!(
        verify(&claims_the_wrong_winner),
        Err(RecordError::WinnerMismatch { .. })
    ));

    let mut denies_a_win = record_of(&a_won_game());
    denies_a_win.result = MatchResult::Drawn {
        reason: DrawReason::PlyCap,
    };
    assert!(matches!(
        verify(&denies_a_win),
        Err(RecordError::UnexpectedTerminal)
    ));
}

// ---------------------------------------------------------- refusing a file --

#[test]
fn a_truncated_shard_is_an_error_and_not_a_shorter_shard() {
    let (_dir, path) = scratch();
    let written = [
        record_of(&a_won_game()),
        record_of(&a_capped_game(11, Budget::Unlimited)),
    ];
    write_shard(&path, &header(ShardMode::SelfPlay), &written);

    let full = std::fs::metadata(&path).expect("metadata").len();
    std::fs::write(
        &path,
        &std::fs::read(&path).expect("read back")[..(full - 20) as usize],
    )
    .expect("truncated");

    let (_, records) = read_shard(&path);
    match records {
        Err(RecordError::Truncated { .. }) => {}
        other => panic!("expected a truncation error, got {other:?}"),
    }
}

#[test]
fn a_shard_from_another_rules_version_names_both_versions() {
    let (_dir, path) = scratch();
    write_shard(
        &path,
        &header(ShardMode::SelfPlay),
        &[record_of(&a_resigned_game())],
    );

    let mut bytes = std::fs::read(&path).expect("read back");
    bytes[RULES_VERSION_OFFSET..RULES_VERSION_OFFSET + 4].copy_from_slice(&99u32.to_le_bytes());
    std::fs::write(&path, &bytes).expect("patched");

    match ShardReader::open(&path) {
        Err(RecordError::RulesVersion { expected, found }) => {
            assert_eq!(expected, hexo_engine::RULES_VERSION);
            assert_eq!(found, 99);
        }
        other => panic!("expected a rules-version refusal, got {other:?}"),
    }
}

#[test]
fn a_file_that_does_not_open_with_the_magic_is_not_a_shard() {
    let (_dir, path) = scratch();
    std::fs::write(&path, b"not a shard at all, but long enough").expect("written");
    match ShardReader::open(&path) {
        Err(RecordError::BadMagic { .. }) => {}
        other => panic!("expected a magic refusal, got {other:?}"),
    }
}

#[test]
fn a_game_count_that_disagrees_with_the_entries_is_an_error() {
    let (_dir, path) = scratch();
    let head = header(ShardMode::SelfPlay);
    let written = [
        record_of(&a_resigned_game()),
        record_of(&a_capped_game(7, Budget::Unlimited)),
    ];
    write_shard(&path, &head, &written);

    let offset = game_count_offset(&head);
    let original = std::fs::read(&path).expect("read back");
    assert_eq!(
        u32::from_le_bytes(original[offset..offset + 4].try_into().expect("four bytes")),
        2,
        "the writer patched the true count in at the documented offset"
    );

    let mut overstated = original.clone();
    overstated[offset..offset + 4].copy_from_slice(&3u32.to_le_bytes());
    std::fs::write(&path, &overstated).expect("patched");
    match read_shard(&path).1 {
        Err(RecordError::GameCountMismatch { declared, found }) => {
            assert_eq!((declared, found), (3, 2));
        }
        other => panic!("expected a count mismatch, got {other:?}"),
    }

    let mut understated = original;
    understated[offset..offset + 4].copy_from_slice(&1u32.to_le_bytes());
    std::fs::write(&path, &understated).expect("patched");
    match read_shard(&path).1 {
        Err(RecordError::TrailingBytes { .. }) => {}
        other => panic!("expected trailing bytes, got {other:?}"),
    }
}

// -------------------------------------------------------------- the writer --

#[test]
fn an_unfinished_game_is_not_a_record() {
    let empty = Game::new(GameSpec::default());
    assert!(matches!(
        GameRecord::from_game(&empty),
        Err(RecordError::Unfinished)
    ));

    let mut started = Game::new(GameSpec::default());
    let Step::NeedDecision {
        generation,
        zobrist,
        ..
    } = started.step()
    else {
        panic!("a new game wants a decision");
    };
    started
        .submit(generation, Reply::Place(Decision::new(act(0, 0), zobrist)))
        .expect("opening");
    assert!(matches!(
        GameRecord::from_game(&started),
        Err(RecordError::Unfinished)
    ));
}

#[test]
fn the_shard_appears_only_when_the_writer_finalizes() {
    let (_dir, path) = scratch();
    let tmp = PathBuf::from(format!("{}.tmp", path.display()));

    let mut writer =
        ShardWriter::create(&path, &header(ShardMode::SelfPlay)).expect("a fresh path");
    writer
        .append(&record_of(&a_resigned_game()))
        .expect("appended");
    assert!(!path.exists(), "an unfinalized shard is not at its path");
    assert!(tmp.exists(), "it is at the temporary path instead");

    writer.finalize().expect("finalized");
    assert!(path.exists(), "the shard arrives on finalize");
    assert!(!tmp.exists(), "and the temporary file is gone");

    let (head, records) = read_shard(&path);
    assert_eq!(head.game_count, 1);
    assert_eq!(records.expect("read back").len(), 1);
}

#[test]
fn a_dropped_writer_leaves_nothing_behind() {
    let (_dir, path) = scratch();
    let tmp = PathBuf::from(format!("{}.tmp", path.display()));

    {
        let mut writer =
            ShardWriter::create(&path, &header(ShardMode::SelfPlay)).expect("a fresh path");
        writer
            .append(&record_of(&a_resigned_game()))
            .expect("appended");
        assert!(tmp.exists());
    }

    assert!(!tmp.exists(), "a dropped writer removes its temporary file");
    assert!(!path.exists(), "and never wrote the shard");
}

#[test]
fn a_shard_is_written_once_and_never_over() {
    let (_dir, path) = scratch();
    write_shard(
        &path,
        &header(ShardMode::SelfPlay),
        &[record_of(&a_resigned_game())],
    );
    match ShardWriter::create(&path, &header(ShardMode::SelfPlay)) {
        Err(RecordError::AlreadyExists { path: taken }) => assert_eq!(taken, path),
        other => panic!("expected a refusal to overwrite, got {other:?}"),
    }
}

#[test]
fn a_header_that_presets_the_game_count_is_refused() {
    let (_dir, path) = scratch();
    let head = ShardHeader {
        game_count: 12,
        ..header(ShardMode::SelfPlay)
    };
    match ShardWriter::create(&path, &head) {
        Err(RecordError::HeaderGameCount { found }) => assert_eq!(found, 12),
        other => panic!("expected a refusal of the preset count, got {other:?}"),
    }
    assert!(!path.exists());
}

#[test]
fn a_wall_budget_past_the_formats_nanoseconds_is_a_write_error() {
    let (_dir, path) = scratch();
    let mut writer =
        ShardWriter::create(&path, &header(ShardMode::SelfPlay)).expect("a fresh path");
    let mut record = record_of(&a_resigned_game());
    record.spec.budget = Budget::Wall(Duration::MAX);

    match writer.append(&record) {
        Err(RecordError::WallBudgetOverflow { nanos }) => {
            assert_eq!(nanos, Duration::MAX.as_nanos());
        }
        other => panic!("expected a loud overflow, got {other:?}"),
    }

    // The failed append writes no partial entry; finalization yields an empty
    // shard.
    writer.finalize().expect("finalized");
    let (head, records) = read_shard(&path);
    assert_eq!(head.game_count, 0);
    assert!(records.expect("read back").is_empty());
}
