//! The game state machine: adjudication, the guards, and the two drive shapes.

use hexo_engine::{Action, HexCoord, MoveError, Player, Position, TurnPhase};
use hexo_runner::{
    Budget, Decision, DrawReason, Failure, FailurePolicy, Game, GameSpec, MatchResult, NoContest,
    Reply, Step, SubmitError, WinReason,
};
use std::num::NonZeroU32;

const P1_WINNING_PLY: u32 = 11;

fn act(q: i16, r: i16) -> Action {
    Action::new(HexCoord::new(q, r))
}

fn cap(value: u32) -> NonZeroU32 {
    NonZeroU32::new(value).expect("test caps are nonzero")
}

/// The outstanding request, or a panic if the game is over.
fn need(game: &Game) -> (Player, u64, u64) {
    match game.step() {
        Step::NeedDecision {
            seat,
            generation,
            zobrist,
            ..
        } => (seat, generation, zobrist),
        Step::Finished(r) => panic!("expected a decision, game is finished: {r:?}"),
    }
}

/// Submit `action` as the seat on turn, echoing the hash correctly.
fn play(game: &mut Game, action: Action) -> Result<Option<MatchResult>, SubmitError> {
    let (_, generation, zobrist) = need(game);
    game.submit(generation, Reply::Place(Decision::new(action, zobrist)))
        .map(|t| t.result)
}

/// Drive until the game ends, with `P1` building a six-line along axis `Q` and `P0`
/// taking the lowest-ranked legal move.
fn drive_to_a_p1_win(game: &mut Game) -> MatchResult {
    const LINE: [(i16, i16); 6] = [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)];
    let mut next = 0;
    loop {
        match game.step() {
            Step::Finished(result) => return result,
            Step::NeedDecision {
                seat,
                generation,
                zobrist,
                ..
            } => {
                let action = if seat == Player::P1 && next < LINE.len() {
                    let (q, r) = LINE[next];
                    next += 1;
                    act(q, r)
                } else {
                    game.position().nth_legal(0).expect("a legal move exists")
                };
                game.submit(generation, Reply::Place(Decision::new(action, zobrist)))
                    .expect("accepted");
            }
        }
    }
}

/// Drive with the lowest-ranked legal move until the game ends.
fn drive_to_the_end(game: &mut Game) -> MatchResult {
    loop {
        match game.step() {
            Step::Finished(result) => return result,
            Step::NeedDecision {
                generation,
                zobrist,
                ..
            } => {
                let action = game.position().nth_legal(0).expect("a legal move exists");
                game.submit(generation, Reply::Place(Decision::new(action, zobrist)))
                    .expect("a fresh generation and a matching hash");
            }
        }
    }
}

#[test]
fn a_new_game_asks_p0_to_open() {
    let game = Game::new(GameSpec::default());
    match game.step() {
        Step::NeedDecision {
            seat,
            generation,
            ply,
            zobrist,
            budget,
        } => {
            assert_eq!(seat, Player::P0);
            assert_eq!(generation, 0);
            assert_eq!(ply, 0);
            assert_eq!(zobrist, Position::new().zobrist());
            assert_eq!(budget, Budget::Unlimited);
        }
        Step::Finished(r) => panic!("a new game is not finished: {r:?}"),
    }
}

#[test]
fn step_is_pure() {
    let mut game = Game::new(GameSpec::default());
    assert_eq!(game.step(), game.step());
    play(&mut game, act(0, 0)).expect("opening");
    assert_eq!(game.step(), game.step());
    assert_eq!(game.position().stone_count(), 1);
}

#[test]
fn generation_advances_only_on_an_accepted_placement() {
    let mut game = Game::new(GameSpec::default());
    let (_, g0, z0) = need(&game);
    assert_eq!(g0, 0);

    assert!(
        game.submit(99, Reply::Place(Decision::new(act(0, 0), z0)))
            .is_err()
    );
    assert_eq!(need(&game).1, 0, "a refusal must not advance the token");

    assert!(
        game.submit(0, Reply::Place(Decision::new(act(0, 0), z0 ^ 1)))
            .is_err()
    );
    assert_eq!(need(&game).1, 0);

    let t = game
        .submit(0, Reply::Place(Decision::new(act(0, 0), z0)))
        .expect("legal");
    assert_eq!(t.generation, 1);
    assert_eq!(need(&game).1, 1);
}

#[test]
fn a_stale_generation_is_refused_and_changes_nothing() {
    let mut game = Game::new(GameSpec::default());
    play(&mut game, act(0, 0)).expect("opening");
    let before = game.position().clone();
    let (_, generation, zobrist) = need(&game);

    let err = game
        .submit(
            generation - 1,
            Reply::Place(Decision::new(act(1, 0), zobrist)),
        )
        .expect_err("a stale token must be refused");
    assert_eq!(
        err,
        SubmitError::StaleGeneration {
            expected: generation,
            got: generation - 1
        }
    );
    assert_eq!(game.position(), &before);
    assert_eq!(game.plies().len(), 1);
    assert!(game.result().is_none());
}

#[test]
fn a_drifted_mirror_is_refused_and_changes_nothing() {
    let mut game = Game::new(GameSpec::default());
    play(&mut game, act(0, 0)).expect("opening");
    let before = game.position().clone();
    let (_, generation, zobrist) = need(&game);

    let err = game
        .submit(
            generation,
            Reply::Place(Decision::new(act(1, 0), zobrist ^ 0xDEAD)),
        )
        .expect_err("a mismatched hash must be refused");
    assert_eq!(
        err,
        SubmitError::Desync {
            expected: zobrist,
            got: zobrist ^ 0xDEAD
        }
    );
    assert_eq!(game.position(), &before);
    assert!(game.result().is_none());

    play(&mut game, act(1, 0)).expect("legal");
    assert_eq!(game.position().stone_count(), 2);
}

#[test]
fn nothing_is_accepted_after_the_game_ends() {
    let mut game = Game::new(GameSpec {
        ply_cap: cap(3),
        ..GameSpec::default()
    });
    play(&mut game, act(0, 0)).expect("opening");
    play(&mut game, act(1, 0)).expect("legal");
    let result = play(&mut game, act(2, 0)).expect("legal");
    assert_eq!(
        result,
        Some(MatchResult::Drawn {
            reason: DrawReason::PlyCap
        })
    );

    assert!(matches!(game.step(), Step::Finished(_)));
    for reply in [
        Reply::Place(Decision::new(act(3, 0), game.position().zobrist())),
        Reply::Resign,
        Reply::Failed(Failure::Crashed),
    ] {
        assert_eq!(game.submit(3, reply), Err(SubmitError::Finished));
    }
    assert_eq!(game.position().stone_count(), 3);
}

#[test]
fn six_in_a_row_is_a_decisive_win() {
    let mut game = Game::new(GameSpec::default());
    let result = drive_to_a_p1_win(&mut game);
    assert_eq!(
        result,
        MatchResult::Decisive {
            winner: Player::P1,
            reason: WinReason::SixInARow
        }
    );
    assert!(result.is_contested());
    assert!(game.position().is_terminal());
    assert_eq!(
        game.position().outcome().expect("terminal").winner,
        Player::P1
    );
    assert_eq!(game.position().current_player(), Player::P1);
    assert_eq!(game.plies().len(), game.position().stone_count() as usize);
}

#[test]
fn a_win_on_the_capping_placement_is_a_win() {
    let mut game = Game::new(GameSpec {
        ply_cap: cap(P1_WINNING_PLY),
        ..GameSpec::default()
    });
    assert_eq!(
        drive_to_a_p1_win(&mut game),
        MatchResult::Decisive {
            winner: Player::P1,
            reason: WinReason::SixInARow,
        }
    );
    assert_eq!(game.position().stone_count(), P1_WINNING_PLY);
}

#[test]
fn the_ply_cap_is_a_draw_and_not_an_abort() {
    let mut game = Game::new(GameSpec {
        ply_cap: cap(12),
        ..GameSpec::default()
    });
    let result = drive_to_the_end(&mut game);
    assert_eq!(
        result,
        MatchResult::Drawn {
            reason: DrawReason::PlyCap
        }
    );
    // Turns end after placements 1, 3, 5, ...; 12 falls mid-turn, so the game
    // stops at the next boundary rather than leaving P1 a stone short.
    assert_eq!(game.position().stone_count(), 13);
    assert_eq!(
        game.position().phase(),
        TurnPhase::FirstStone,
        "the cap fired on a completed turn"
    );
    assert!(!game.position().is_terminal(), "the engine knows no draw");
    assert!(result.is_contested(), "a capped game is usable data");
    assert_eq!(result.winner(), None);
}

/// The odd cap the boundary rule lands on exactly, as the counterpart to the even
/// cap above.
#[test]
fn an_odd_ply_cap_is_reached_exactly() {
    let mut game = Game::new(GameSpec {
        ply_cap: cap(13),
        ..GameSpec::default()
    });
    assert_eq!(
        drive_to_the_end(&mut game),
        MatchResult::Drawn {
            reason: DrawReason::PlyCap
        }
    );
    assert_eq!(game.position().stone_count(), 13);
}

/// Reaching the cap on a turn's first stone does not end the game mid-turn.
#[test]
fn the_cap_never_ends_a_turn_after_its_first_stone() {
    for ply_cap in 2..=16 {
        let mut game = Game::new(GameSpec {
            ply_cap: cap(ply_cap),
            ..GameSpec::default()
        });
        assert_eq!(
            drive_to_the_end(&mut game),
            MatchResult::Drawn {
                reason: DrawReason::PlyCap
            },
            "cap {ply_cap}"
        );
        let stones = game.position().stone_count();
        assert_eq!(stones % 2, 1, "cap {ply_cap} stopped mid-turn at {stones}");
        assert!(
            (ply_cap..=ply_cap + 1).contains(&stones),
            "cap {ply_cap} overshot to {stones}"
        );
    }
}

#[test]
fn resignation_loses() {
    let mut game = Game::new(GameSpec::default());
    play(&mut game, act(0, 0)).expect("opening");
    let (seat, generation, _) = need(&game);
    assert_eq!(seat, Player::P1);

    let t = game.submit(generation, Reply::Resign).expect("accepted");
    assert!(t.applied.is_none());
    assert_eq!(
        t.result,
        Some(MatchResult::Decisive {
            winner: Player::P0,
            reason: WinReason::Resignation
        })
    );
    assert_eq!(game.position().stone_count(), 1, "no stone was placed");
}

#[test]
fn an_illegal_placement_loses_rather_than_aborting() {
    let mut game = Game::new(GameSpec::default());
    play(&mut game, act(0, 0)).expect("opening");
    play(&mut game, act(1, 0)).expect("legal");

    let (seat, generation, zobrist) = need(&game);
    assert_eq!(seat, Player::P1);
    let t = game
        .submit(generation, Reply::Place(Decision::new(act(0, 0), zobrist)))
        .expect("an illegal move is an accepted submission, not a submit error");
    assert!(t.applied.is_none());
    assert_eq!(
        t.result,
        Some(MatchResult::Decisive {
            winner: Player::P0,
            reason: WinReason::IllegalMove {
                action: act(0, 0).id(),
                cause: MoveError::Occupied(HexCoord::ORIGIN),
            }
        })
    );
    assert!(t.result.expect("ended").is_contested());
    assert_eq!(game.position().stone_count(), 2, "nothing was placed");
}

#[test]
fn a_far_placement_also_loses() {
    let mut game = Game::new(GameSpec::default());
    play(&mut game, act(0, 0)).expect("opening");
    let (_, generation, zobrist) = need(&game);
    let t = game
        .submit(
            generation,
            Reply::Place(Decision::new(act(300, 300), zobrist)),
        )
        .expect("accepted");
    assert_eq!(
        t.result,
        Some(MatchResult::Decisive {
            winner: Player::P0,
            reason: WinReason::IllegalMove {
                action: act(300, 300).id(),
                cause: MoveError::TooFarFromStones(HexCoord::new(300, 300)),
            }
        })
    );
}

#[test]
fn an_engine_limit_is_a_no_contest_and_blames_nobody() {
    let mut game = Game::new(GameSpec {
        ply_cap: cap(u32::MAX),
        ..GameSpec::default()
    });
    play(&mut game, act(0, 0)).expect("opening");

    let mut k: i16 = 1;
    let result = loop {
        let step = 8 * k;
        match play(&mut game, act(step, -step)) {
            Ok(None) => k += 1,
            Ok(Some(result)) => break result,
            Err(e) => panic!("submit refused the walk: {e}"),
        }
        assert!(k < 2000, "the walk never reached the arena ceiling");
    };

    match result {
        MatchResult::NoContest(NoContest::EngineLimit { error, .. }) => {
            assert!(
                matches!(error, MoveError::BoardExtentExceeded { .. }),
                "expected the arena ceiling, got {error:?}"
            );
            assert!(!error.is_rule_violation(), "not the seat's fault");
        }
        other => panic!("expected a no-contest engine limit, got {other:?}"),
    }
    assert!(
        !result.is_contested(),
        "a game the engine could not finish is not evidence about either seat"
    );
    assert_eq!(result.winner(), None);
}

#[test]
fn every_failure_kind_obeys_each_failure_policy() {
    for policy in [FailurePolicy::Forfeit, FailurePolicy::NoContest] {
        for failure in [
            Failure::Timeout,
            Failure::Crashed,
            Failure::Protocol,
            Failure::Desync {
                expected: 0xFEED,
                got: 0xF00D,
            },
        ] {
            let mut game = Game::new(GameSpec {
                on_failure: policy,
                ..GameSpec::default()
            });
            play(&mut game, act(0, 0)).expect("opening");
            let (seat, generation, _) = need(&game);
            assert_eq!(seat, Player::P1);
            let expected = match policy {
                FailurePolicy::Forfeit => MatchResult::Decisive {
                    winner: seat.other(),
                    reason: match failure {
                        Failure::Timeout => WinReason::Timeout,
                        Failure::Crashed => WinReason::Crash,
                        Failure::Protocol => WinReason::Protocol,
                        Failure::Desync { expected, got } => WinReason::Desync { expected, got },
                    },
                },
                FailurePolicy::NoContest => {
                    MatchResult::NoContest(NoContest::SeatFailure { seat, failure })
                }
            };
            let t = game
                .submit(generation, Reply::Failed(failure))
                .expect("accepted");
            assert_eq!(
                t.result,
                Some(expected),
                "policy {policy:?}, failure {failure:?}"
            );
        }
    }
}

#[test]
fn diagnostics_are_persisted_verbatim() {
    let mut game = Game::new(GameSpec {
        budget: Budget::Visits(800),
        ..GameSpec::default()
    });
    let blob = vec![0xDE, 0xAD, 0xBE, 0xEF];
    let (_, generation, zobrist) = need(&game);
    game.submit(
        generation,
        Reply::Place(Decision::new(act(0, 0), zobrist).with_diagnostics(blob.clone())),
    )
    .expect("opening");

    assert_eq!(game.plies().len(), 1);
    let ply = &game.plies()[0];
    assert_eq!(ply.diagnostics.as_deref(), Some(blob.as_slice()));
    assert_eq!(ply.seat, Player::P0);
    assert_eq!(ply.action, act(0, 0).id());
    assert_eq!(game.spec().budget, Budget::Visits(800));
    assert_eq!(ply.zobrist_after, game.position().zobrist());
}

#[test]
fn the_record_tracks_the_position_ply_for_ply() {
    let mut game = Game::new(GameSpec {
        ply_cap: cap(20),
        ..GameSpec::default()
    });
    drive_to_the_end(&mut game);

    assert_eq!(game.plies().len(), game.position().stone_count() as usize);

    let mut replay = Position::new();
    for (i, ply) in game.plies().iter().enumerate() {
        assert_eq!(replay.current_player(), ply.seat, "ply {i}: recorded seat");
        replay
            .advance(Action::from_id(ply.action))
            .unwrap_or_else(|e| panic!("ply {i} failed to replay: {e}"));
        assert_eq!(replay.zobrist(), ply.zobrist_after, "ply {i}");
    }
}

#[test]
fn the_prefix_replays_into_the_canonical_position() {
    let mut game = Game::new(GameSpec {
        ply_cap: cap(15),
        ..GameSpec::default()
    });
    drive_to_the_end(&mut game);

    let prefix = game.prefix();
    assert_eq!(prefix.len(), game.plies().len());
    let mirror = Position::replay(&prefix).expect("the prefix must replay");
    assert_eq!(&mirror, game.position());
    assert_eq!(mirror.zobrist(), game.position().zobrist());
}

#[test]
fn many_games_interleave_on_one_thread() {
    const GAMES: usize = 200;
    let mut games: Vec<Game> = (0..GAMES)
        .map(|i| {
            Game::new(GameSpec {
                ply_cap: cap(8 + (i as u32 % 5)),
                ..GameSpec::default()
            })
        })
        .collect();

    let mut finished = 0;
    let mut rounds = 0;
    while finished < GAMES {
        let pending: Vec<(usize, u64, u64)> = games
            .iter()
            .enumerate()
            .filter_map(|(i, g)| match g.step() {
                Step::NeedDecision {
                    generation,
                    zobrist,
                    ..
                } => Some((i, generation, zobrist)),
                Step::Finished(_) => None,
            })
            .collect();

        for (i, generation, zobrist) in pending {
            let action = games[i].position().nth_legal(0).expect("a legal move");
            let t = games[i]
                .submit(generation, Reply::Place(Decision::new(action, zobrist)))
                .expect("accepted");
            if t.result.is_some() {
                finished += 1;
            }
        }

        rounds += 1;
        assert!(rounds < 1000, "the sweep is not making progress");
    }

    for (i, game) in games.iter().enumerate() {
        let result = game.result().expect("finished");
        assert!(result.is_contested(), "game {i}: {result:?}");
        assert_eq!(game.plies().len(), game.position().stone_count() as usize);
    }
}

/// A garbage coordinate is a rule violation and forfeits the seat. It must not
/// escape adjudication as a blameless engine-limit no-contest.
#[test]
fn a_garbage_coordinate_forfeits_rather_than_voiding_the_game() {
    let mut game = Game::new(GameSpec::default());
    play(&mut game, act(0, 0)).expect("opening");
    let (_, generation, zobrist) = need(&game);
    let t = game
        .submit(
            generation,
            Reply::Place(Decision::new(act(20000, 0), zobrist)),
        )
        .expect("an illegal move is an accepted submission");
    assert_eq!(
        t.result,
        Some(MatchResult::Decisive {
            winner: Player::P0,
            reason: WinReason::IllegalMove {
                action: act(20000, 0).id(),
                cause: MoveError::TooFarFromStones(HexCoord::new(20000, 0)),
            }
        })
    );
}
