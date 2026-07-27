//! One question per move, and nothing else: what makes policy-only training the
//! same loop as a tree search rather than a second path.

mod common;

use common::{
    HighestPrior, SampleByPrior, Stamp, Uniform, WIN_IN_ONE, WIN_IN_ONE_CELL, decide, game_after,
    uniform_evaluation,
};
use hexo_engine::HexCoord;
use hexo_runner::{GameSpec, Reply, Step};
use hexo_search::{DecisionSession, Evaluation, PolicySession, SessionStatus};

#[test]
fn a_decision_costs_exactly_one_evaluation() {
    let game = game_after(&WIN_IN_ONE, GameSpec::default());
    let mut session = PolicySession::new(Box::new(HighestPrior), 1);
    let run = decide(&mut session, &game, &mut Uniform);

    assert_eq!(run.rounds.len(), 1, "one round of leaves");
    assert_eq!(run.evaluations(), 1, "one leaf in it");
    assert_eq!(run.peak_in_flight(), 1);
}

#[test]
fn the_session_is_decided_after_one_resume() {
    let game = game_after(&WIN_IN_ONE, GameSpec::default());
    let mut session = PolicySession::new(Box::new(HighestPrior), 1);
    session.begin(&game);

    let mut leaf = None;
    let status = session.pump(&mut |id, position| {
        assert_eq!(
            position.zobrist(),
            game.position().zobrist(),
            "the leaf is the root itself",
        );
        leaf = Some(id);
    });
    assert_eq!(status, SessionStatus::AwaitingEvals { in_flight: 1 });
    assert!(session.take_decision().is_none());

    // Pumping again while the answer is outstanding must not ask twice.
    let status = session.pump(&mut |_id, _position| panic!("the root was emitted a second time"));
    assert_eq!(status, SessionStatus::AwaitingEvals { in_flight: 1 });

    session.resume(
        leaf.expect("the root"),
        uniform_evaluation(game.position().legal_count()),
    );
    assert_eq!(
        session.pump(&mut |_id, _position| panic!("a decided session emits nothing")),
        SessionStatus::Decided,
    );
    assert!(session.take_decision().is_some());
    assert!(session.take_decision().is_none(), "and only once");
}

#[test]
fn the_selector_reads_the_canonical_order_it_was_promised() {
    let game = game_after(&WIN_IN_ONE, GameSpec::default());
    let n = game.position().legal_count();
    let winning = game
        .position()
        .legal_rank(hexo_engine::Action::new(WIN_IN_ONE_CELL))
        .expect("the winning cell is legal here");

    // A one-hot policy on the winning cell's canonical index. `HighestPrior`
    // never sees a coordinate; it maps the index back through `nth_legal`, which
    // is the only reason this lands on the intended cell.
    let mut priors = vec![0.0f32; n];
    priors[winning] = 1.0;

    let mut session = PolicySession::new(Box::new(HighestPrior), 1);
    session.begin(&game);
    let mut leaf = None;
    session.pump(&mut |id, _position| leaf = Some(id));
    session.resume(
        leaf.expect("the root"),
        Evaluation {
            priors: priors.into(),
            value: 0.0,
        },
    );
    let decision = session.take_decision().expect("decided");
    assert_eq!(decision.action.coord(), WIN_IN_ONE_CELL);
}

#[test]
fn the_decision_attests_the_position_and_carries_the_seats_bytes() {
    let game = game_after(&WIN_IN_ONE, GameSpec::default());
    let mut session = PolicySession::new(Box::new(Stamp(vec![1, 2, 3, 4])), 1);
    let run = decide(&mut session, &game, &mut Uniform);

    assert_eq!(run.decision.zobrist, game.position().zobrist());
    assert_eq!(
        run.decision.diagnostics.as_deref(),
        Some(&[1u8, 2, 3, 4][..]),
    );
}

#[test]
#[should_panic(expected = "priors but the evaluated position has")]
fn resuming_with_the_wrong_number_of_priors_panics() {
    let game = game_after(&WIN_IN_ONE, GameSpec::default());
    let mut session = PolicySession::new(Box::new(HighestPrior), 1);
    session.begin(&game);
    let mut leaf = None;
    session.pump(&mut |id, _position| leaf = Some(id));
    session.resume(
        leaf.expect("the root"),
        Evaluation {
            priors: vec![0.5; 2].into(),
            value: 0.0,
        },
    );
}

#[test]
#[should_panic(expected = "nothing in flight")]
fn resuming_before_the_root_was_asked_about_panics() {
    let game = game_after(&WIN_IN_ONE, GameSpec::default());
    let mut donor = PolicySession::new(Box::new(HighestPrior), 1);
    let mut session = PolicySession::new(Box::new(HighestPrior), 2);

    donor.begin(&game);
    let mut leaf = None;
    donor.pump(&mut |id, _position| leaf = Some(id));

    session.begin(&game);
    session.resume(
        leaf.expect("the donor asked"),
        uniform_evaluation(game.position().legal_count()),
    );
}

#[test]
#[should_panic(expected = "pump before begin")]
fn pumping_before_begin_panics() {
    let mut session = PolicySession::new(Box::new(HighestPrior), 1);
    session.pump(&mut |_id, _position| {});
}

#[test]
#[should_panic(expected = "a game that finished")]
fn beginning_on_a_finished_game_panics() {
    let mut moves = WIN_IN_ONE.to_vec();
    moves.push((WIN_IN_ONE_CELL.q, WIN_IN_ONE_CELL.r));
    let game = game_after(&moves, GameSpec::default());
    let mut session = PolicySession::new(Box::new(HighestPrior), 1);
    session.begin(&game);
}

#[test]
fn the_same_seed_and_the_same_answers_give_the_same_moves() {
    let line = play_out(3, None);
    assert_eq!(line, play_out(3, None));
    assert_ne!(line, play_out(4, None));
}

#[test]
fn reseeding_moves_a_session_onto_another_seeds_stream() {
    assert_eq!(play_out(4, Some(3)), play_out(3, None));
    assert_ne!(play_out(4, Some(3)), play_out(4, None));
}

/// Play eight placements with a sampling selector, optionally reseeding first.
fn play_out(seed: u64, reseed: Option<u64>) -> Vec<HexCoord> {
    let mut game = game_after(&[(0, 0)], GameSpec::default());
    let mut session = PolicySession::new(Box::new(SampleByPrior), seed);
    if let Some(other) = reseed {
        session.reseed(other);
    }

    let mut line = Vec::new();
    for _ in 0..8 {
        let run = decide(&mut session, &game, &mut Uniform);
        line.push(run.decision.action.coord());
        let Step::NeedDecision { generation, .. } = game.step() else {
            break;
        };
        game.submit(generation, Reply::Place(run.decision))
            .expect("the seat attested the canonical position");
    }
    line
}
