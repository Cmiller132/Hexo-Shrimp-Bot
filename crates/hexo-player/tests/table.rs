//! Player dispatch, table submission, and sweep contracts.

use hexo_engine::{Action, HexCoord, MoveError, Player as Seat, Position};
use hexo_player::{Mode, Model, ModelPlayer, Player, Table, sweep};
use hexo_runner::{
    Decision, DrawReason, Failure, FailurePolicy, Game, GameSpec, MatchResult, NoContest, Reply,
    Step, WinReason,
};
use std::num::NonZeroU32;

fn cap(value: u32) -> NonZeroU32 {
    NonZeroU32::new(value).expect("test caps are nonzero")
}

fn spec(ply_cap: u32) -> GameSpec {
    GameSpec {
        ply_cap: cap(ply_cap),
        ..GameSpec::default()
    }
}

/// A game with only its opening placement made.
fn opened_game() -> Game {
    let mut game = Game::new(GameSpec::default());
    let Step::NeedDecision {
        generation,
        zobrist,
        ..
    } = game.step()
    else {
        panic!("a fresh game wants a decision");
    };
    game.submit(
        generation,
        Reply::Place(Decision::new(Action::new(HexCoord::ORIGIN), zobrist)),
    )
    .expect("the opening is legal");
    game
}

/// The attestation of a seat that chose from the canonical game it was handed.
fn attest(game: &Game, action: Action) -> Decision {
    Decision::new(action, game.position().zobrist())
}

/// Takes the lowest-ranked legal placement, every time.
#[derive(Clone, Debug)]
struct Lowest;

impl Player for Lowest {
    fn choose(&mut self, game: &Game) -> Decision {
        attest(
            game,
            game.position()
                .nth_legal(0)
                .expect("a running game has a legal placement"),
        )
    }
}

/// Plays a six-line along `Q` as fast as it is asked to, then falls back.
#[derive(Clone, Debug)]
struct Liner {
    next: usize,
}

impl Liner {
    const LINE: [(i16, i16); 6] = [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)];
}

impl Player for Liner {
    fn choose(&mut self, game: &Game) -> Decision {
        match Self::LINE.get(self.next) {
            Some(&(q, r)) => {
                self.next += 1;
                attest(game, Action::new(HexCoord::new(q, r)))
            }
            None => Lowest.choose(game),
        }
    }
}

/// Always names the origin, which is occupied from the first placement onward.
#[derive(Clone, Debug)]
struct AlwaysOrigin;

impl Player for AlwaysOrigin {
    fn choose(&mut self, game: &Game) -> Decision {
        attest(game, Action::new(HexCoord::ORIGIN))
    }
}

/// Records which side it was asked as, then plays the lowest-ranked legal placement.
#[derive(Clone, Debug, Default)]
struct Recorder {
    seen: Vec<Seat>,
}

impl Player for Recorder {
    fn choose(&mut self, game: &Game) -> Decision {
        self.seen.push(game.position().current_player());
        Lowest.choose(game)
    }
}

/// Selects from a mirror rebuilt from the record and attests the mirror hash.
#[derive(Clone, Debug, Default)]
struct Mirrorer {
    plies_seen: Vec<usize>,
}

impl Player for Mirrorer {
    fn choose(&mut self, game: &Game) -> Decision {
        let prefix = game.prefix();
        self.plies_seen.push(prefix.len());
        let mirror = Position::replay(&prefix).expect("the record must replay");
        let action = mirror
            .nth_legal(0)
            .expect("a running game has a legal placement");
        Decision::new(action, mirror.zobrist())
    }
}

/// Annotates every move with the ply index it saw, as a model would with search
/// statistics.
#[derive(Clone, Debug)]
struct Annotator;

impl Player for Annotator {
    fn choose(&mut self, game: &Game) -> Decision {
        let ply = game.plies().len() as u8;
        Lowest.choose(game).with_diagnostics(vec![ply])
    }
}

/// Attests a hash that is not its position's, as a diverged mirror would.
#[derive(Clone, Debug)]
struct WrongEcho;

impl Player for WrongEcho {
    fn choose(&mut self, game: &Game) -> Decision {
        let mut decision = Lowest.choose(game);
        decision.zobrist ^= 1;
        decision
    }
}

/// Distinguishable per mode: self-play walks the legal ordering, eval pins index 0.
#[derive(Clone, Debug, Default)]
struct TwoFaced {
    self_play_calls: usize,
    eval_calls: usize,
}

impl Model for TwoFaced {
    fn self_play_move(&mut self, game: &Game) -> Decision {
        self.self_play_calls += 1;
        let pos = game.position();
        let n = pos.legal_count();
        attest(
            game,
            pos.nth_legal(self.self_play_calls % n)
                .expect("index is below the legal count"),
        )
    }

    fn eval_move(&mut self, game: &Game) -> Decision {
        self.eval_calls += 1;
        attest(
            game,
            game.position()
                .nth_legal(0)
                .expect("a running game has a legal placement"),
        )
    }
}

#[test]
fn a_table_drives_a_game_to_the_ply_cap() {
    let mut table = Table::new(spec(24), [Lowest, Lowest]);
    let result = table.run();

    assert_eq!(
        result,
        MatchResult::Drawn {
            reason: DrawReason::PlyCap
        }
    );
    // The cap is tested only where a turn ended, and turns end at odd placement
    // counts, so an even cap of 24 stops at 25.
    assert_eq!(table.game().position().stone_count(), 25);
    assert_eq!(table.game().plies().len(), 25);
    assert_eq!(table.result(), Some(result));
}

#[test]
fn a_table_reports_the_win_it_ended_on() {
    let seats: [Box<dyn Player>; 2] = [Box::new(Lowest), Box::new(Liner { next: 0 })];
    let mut table = Table::new(GameSpec::default(), seats);
    let result = table.run();

    assert_eq!(
        result,
        MatchResult::Decisive {
            winner: Seat::P1,
            reason: WinReason::SixInARow
        }
    );
    assert!(table.game().position().is_terminal());
    assert!(result.is_contested());
}

/// The driver submits the seat's action without pre-validating legality.
#[test]
fn an_illegal_choice_is_adjudicated_rather_than_prevented() {
    let seats: [Box<dyn Player>; 2] = [Box::new(Lowest), Box::new(AlwaysOrigin)];
    let mut table = Table::new(GameSpec::default(), seats);
    let result = table.run();

    assert_eq!(
        result,
        MatchResult::Decisive {
            winner: Seat::P0,
            reason: WinReason::IllegalMove {
                action: Action::new(HexCoord::ORIGIN).id(),
                cause: MoveError::Occupied(HexCoord::ORIGIN),
            }
        }
    );
    assert_eq!(
        table.game().position().stone_count(),
        1,
        "the refused placement was not applied"
    );
}

/// Seat-authored diagnostics reach the record verbatim.
#[test]
fn seat_diagnostics_reach_the_record() {
    let seats: [Box<dyn Player>; 2] = [Box::new(Annotator), Box::new(Lowest)];
    let mut table = Table::new(spec(8), seats);
    table.run();

    let plies = table.game().plies();
    assert_eq!(plies.len(), 9, "the cap of 8 falls mid-turn");
    for (i, ply) in plies.iter().enumerate() {
        match ply.seat {
            Seat::P0 => assert_eq!(
                ply.diagnostics.as_deref(),
                Some(&[i as u8][..]),
                "ply {i}: the annotation was lost or rewritten"
            ),
            Seat::P1 => assert_eq!(ply.diagnostics, None, "ply {i}: an annotation appeared"),
        }
    }
}

/// A wrong attestation is reported as a desync failure and adjudicated by policy.
#[test]
fn a_wrong_attestation_forfeits_the_seat_under_the_default_policy() {
    let seats: [Box<dyn Player>; 2] = [Box::new(WrongEcho), Box::new(Lowest)];
    let mut table = Table::new(GameSpec::default(), seats);
    let result = table.run();

    let canonical = Position::new().zobrist();
    assert_eq!(
        result,
        MatchResult::Decisive {
            winner: Seat::P1,
            reason: WinReason::Desync {
                expected: canonical,
                got: canonical ^ 1,
            },
        }
    );
    assert_eq!(
        table.game().plies().len(),
        0,
        "the desynced placement must not reach the record"
    );
}

#[test]
fn a_wrong_attestation_is_a_no_contest_under_that_policy() {
    let seats: [Box<dyn Player>; 2] = [Box::new(WrongEcho), Box::new(Lowest)];
    let mut table = Table::new(
        GameSpec {
            on_failure: FailurePolicy::NoContest,
            ..GameSpec::default()
        },
        seats,
    );
    let result = table.run();

    let canonical = Position::new().zobrist();
    assert_eq!(
        result,
        MatchResult::NoContest(NoContest::SeatFailure {
            seat: Seat::P0,
            failure: Failure::Desync {
                expected: canonical,
                got: canonical ^ 1,
            },
        })
    );
    assert!(!result.is_contested());
}

#[test]
fn a_step_after_the_end_changes_nothing() {
    let mut table = Table::new(spec(4), [Lowest, Lowest]);
    let result = table.run();
    let plies = table.game().plies().len();

    assert_eq!(table.step(), Some(result));
    assert_eq!(table.step(), Some(result));
    assert_eq!(table.game().plies().len(), plies);
}

/// Record the seat identity used for each call.
#[test]
fn each_seat_is_only_ever_asked_as_its_own_side() {
    let mut table = Table::new(spec(9), [Recorder::default(), Recorder::default()]);
    assert!(table.seat(Seat::P0).seen.is_empty());
    assert!(table.seat(Seat::P1).seen.is_empty());

    table.run();
    let [p0, p1] = table.into_seats();

    assert!(
        p0.seen.iter().all(|&s| s == Seat::P0),
        "the P0 slot was asked as another side: {:?}",
        p0.seen
    );
    assert!(
        p1.seen.iter().all(|&s| s == Seat::P1),
        "the P1 slot was asked as another side: {:?}",
        p1.seen
    );
    // P0 opens alone, then the turns alternate in pairs: 1, 2, 2, 2, 2.
    assert_eq!((p0.seen.len(), p1.seen.len()), (5, 4));
}

/// A seat can rebuild move-order features from the game's ply record.
#[test]
fn a_seat_can_replay_the_record_it_is_handed() {
    let mut table = Table::new(spec(9), [Mirrorer::default(), Mirrorer::default()]);
    table.run();

    let [p0, p1] = table.into_seats();
    assert_eq!(p0.plies_seen, vec![0, 3, 4, 7, 8]);
    assert_eq!(p1.plies_seen, vec![1, 2, 5, 6]);
}

#[test]
fn a_model_player_dispatches_on_its_mode() {
    let mut self_play = ModelPlayer::new(TwoFaced::default(), Mode::SelfPlay);
    let mut eval = ModelPlayer::new(TwoFaced::default(), Mode::Eval);
    assert_eq!(self_play.mode(), Mode::SelfPlay);
    assert_eq!(eval.mode(), Mode::Eval);

    let game = opened_game();
    let sampled = self_play.choose(&game);
    let greedy = eval.choose(&game);

    assert_eq!(greedy.action, game.position().nth_legal(0).expect("legal"));
    assert_ne!(
        sampled.action, greedy.action,
        "the two modes must be distinguishable"
    );

    assert_eq!(self_play.model().self_play_calls, 1);
    assert_eq!(self_play.model().eval_calls, 0);
    assert_eq!(eval.model().eval_calls, 1);
    assert_eq!(eval.model().self_play_calls, 0);
}

#[test]
fn only_the_bound_mode_is_ever_called() {
    let mut table = Table::new(
        spec(20),
        [
            ModelPlayer::new(TwoFaced::default(), Mode::SelfPlay),
            ModelPlayer::new(TwoFaced::default(), Mode::Eval),
        ],
    );
    table.run();

    let [p0, p1] = table.into_seats();
    assert_eq!(p0.model().eval_calls, 0, "a self-play seat never evaluated");
    assert!(p0.model().self_play_calls > 0);
    assert_eq!(p1.model().self_play_calls, 0, "an eval seat never sampled");
    assert!(p1.model().eval_calls > 0);
}

/// `Box<dyn Player>` permits two different player implementations in one table.
#[test]
fn a_plain_seat_can_face_a_model_backed_one() {
    let seats: [Box<dyn Player>; 2] = [
        Box::new(Lowest),
        Box::new(ModelPlayer::new(TwoFaced::default(), Mode::Eval)),
    ];
    let mut table = Table::new(spec(14), seats);

    assert!(table.run().is_contested());
    assert_eq!(
        table.game().plies().len(),
        15,
        "the cap of 14 falls mid-turn"
    );
}

#[test]
fn a_sweep_drives_every_table_and_counts_the_rest() {
    const TABLES: usize = 200;
    let mut tables: Vec<_> = (0..TABLES)
        .map(|i| Table::new(spec(8 + (i as u32 % 5)), [Lowest, Lowest]))
        .collect();

    let mut rounds = 0;
    let mut previous = TABLES;
    loop {
        let running = sweep(&mut tables);
        assert!(running <= previous, "a finished table restarted");
        previous = running;
        rounds += 1;
        assert!(rounds < 100, "the sweep is not making progress");
        if running == 0 {
            break;
        }
    }

    // One placement per table per round, so the round count is the longest game.
    // Caps 8..=12 stop at the next turn boundary — 9, 9, 11, 11, 13.
    assert_eq!(rounds, 13, "the longest game is the last to finish");
    for (i, table) in tables.iter().enumerate() {
        let result = table
            .result()
            .unwrap_or_else(|| panic!("table {i} unfinished"));
        assert!(result.is_contested(), "table {i}: {result:?}");
        assert_eq!(
            table.game().plies().len(),
            table.game().position().stone_count() as usize
        );
    }
}

#[test]
fn a_sweep_over_finished_tables_is_a_no_op() {
    let mut tables = vec![Table::new(spec(3), [Lowest, Lowest])];
    while sweep(&mut tables) > 0 {}

    let plies = tables[0].game().plies().len();
    assert_eq!(sweep(&mut tables), 0);
    assert_eq!(tables[0].game().plies().len(), plies);
}
