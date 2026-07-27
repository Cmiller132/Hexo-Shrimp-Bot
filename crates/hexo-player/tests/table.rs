//! The player seam: dispatch, the driver, and what the driver deliberately does not do.

use hexo_engine::{Action, HexCoord, MoveError, Player as Seat, Position};
use hexo_player::{Mode, Model, ModelPlayer, Player, Table, sweep};
use hexo_runner::{Budget, DrawReason, GameSpec, MatchResult, WinReason};
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

/// Takes the lowest-ranked legal placement, every time.
#[derive(Clone, Debug)]
struct Lowest;

impl Player for Lowest {
    fn choose(&mut self, pos: &Position, _budget: Budget) -> Action {
        pos.nth_legal(0)
            .expect("a running game has a legal placement")
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
    fn choose(&mut self, pos: &Position, budget: Budget) -> Action {
        match Self::LINE.get(self.next) {
            Some(&(q, r)) => {
                self.next += 1;
                Action::new(HexCoord::new(q, r))
            }
            None => Lowest.choose(pos, budget),
        }
    }
}

/// Always names the origin, which is occupied from the first placement onward.
#[derive(Clone, Debug)]
struct AlwaysOrigin;

impl Player for AlwaysOrigin {
    fn choose(&mut self, _pos: &Position, _budget: Budget) -> Action {
        Action::new(HexCoord::ORIGIN)
    }
}

/// Records which side it was asked as, then plays the lowest-ranked legal placement.
#[derive(Clone, Debug, Default)]
struct Recorder {
    seen: Vec<Seat>,
}

impl Player for Recorder {
    fn choose(&mut self, pos: &Position, budget: Budget) -> Action {
        self.seen.push(pos.current_player());
        Lowest.choose(pos, budget)
    }
}

/// Distinguishable per mode: self-play walks the legal ordering, eval pins index 0.
#[derive(Clone, Debug, Default)]
struct TwoFaced {
    self_play_calls: usize,
    eval_calls: usize,
}

impl Model for TwoFaced {
    fn self_play_move(&mut self, pos: &Position, _budget: Budget) -> Action {
        self.self_play_calls += 1;
        let n = pos.legal_count();
        pos.nth_legal(self.self_play_calls % n)
            .expect("index is below the legal count")
    }

    fn eval_move(&mut self, pos: &Position, _budget: Budget) -> Action {
        self.eval_calls += 1;
        pos.nth_legal(0)
            .expect("a running game has a legal placement")
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
    assert_eq!(table.game().position().stone_count(), 24);
    assert_eq!(table.game().plies().len(), 24);
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

/// The driver submits what the seat returned. Checking legality first would be a
/// second implementation of the rules, and the game already adjudicates it.
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

#[test]
fn a_step_after_the_end_changes_nothing() {
    let mut table = Table::new(spec(4), [Lowest, Lowest]);
    let result = table.run();
    let plies = table.game().plies().len();

    assert_eq!(table.step(), Some(result));
    assert_eq!(table.step(), Some(result));
    assert_eq!(table.game().plies().len(), plies);
}

/// A seat that is only ever asked as one side is the whole indexing claim, so this
/// records which side it saw rather than merely counting calls.
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

#[test]
fn a_model_player_dispatches_on_its_mode() {
    let mut self_play = ModelPlayer::new(TwoFaced::default(), Mode::SelfPlay);
    let mut eval = ModelPlayer::new(TwoFaced::default(), Mode::Eval);
    assert_eq!(self_play.mode(), Mode::SelfPlay);
    assert_eq!(eval.mode(), Mode::Eval);

    let mut pos = Position::new();
    pos.advance(Action::new(HexCoord::ORIGIN)).expect("opening");

    let sampled = self_play.choose(&pos, Budget::Unlimited);
    let greedy = eval.choose(&pos, Budget::Unlimited);

    assert_eq!(greedy, pos.nth_legal(0).expect("legal"));
    assert_ne!(sampled, greedy, "the two modes must be distinguishable");

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

/// The mixed-kind case the design exists for: a plain seat against a model-backed
/// one. `[P; 2]` cannot hold two types, so this is what `Box<dyn Player>` is for.
#[test]
fn a_plain_seat_can_face_a_model_backed_one() {
    let seats: [Box<dyn Player>; 2] = [
        Box::new(Lowest),
        Box::new(ModelPlayer::new(TwoFaced::default(), Mode::Eval)),
    ];
    let mut table = Table::new(spec(14), seats);

    assert!(table.run().is_contested());
    assert_eq!(table.game().plies().len(), 14);
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

    assert_eq!(rounds, 12, "the longest game is the last to finish");
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
