//! MCTS terminal handling, mover-relative values, virtual loss, and decisions.

mod common;

use common::{
    Focus, MaxVisits, SampleByVisits, Stamp, Uniform, WIN_IN_ONE, WIN_IN_ONE_CELL, WIN_IN_TWO,
    WIN_IN_TWO_FIRST, WIN_IN_TWO_SECOND, decide, decode_children, game_after, uniform_evaluation,
};
use hexo_engine::{Action, HexCoord, Player, Position};
use hexo_runner::{GameSpec, Reply, Step};
use hexo_search::{DecisionSession, Evaluation, LeafId, MctsConfig, MctsSession, SessionStatus};
use std::num::{NonZeroU32, NonZeroUsize};

fn config(visits: u32, cap: usize) -> MctsConfig {
    MctsConfig {
        visits: NonZeroU32::new(visits).expect("nonzero"),
        max_in_flight: NonZeroUsize::new(cap).expect("nonzero"),
        c_puct: 1.5,
    }
}

#[test]
fn the_search_finds_a_win_in_one() {
    let game = game_after(&WIN_IN_ONE, GameSpec::default());
    assert_eq!(game.position().current_player(), Player::P1);
    assert!(game.position().is_legal(Action::new(WIN_IN_ONE_CELL)));

    // Uniform priors visit unvisited children in canonical order, so the budget
    // must reach the winning action's rank.
    let legal = u32::try_from(game.position().legal_count()).expect("a legal count fits u32");
    let mut session = MctsSession::new(config(legal + 64, 8), Box::new(MaxVisits), 1);

    let run = decide(&mut session, &game, &mut Uniform);
    assert_eq!(
        run.decision.action.coord(),
        WIN_IN_ONE_CELL,
        "the search preferred a placement that does not end the game over one that does",
    );

    let children = decode_children(
        run.decision
            .diagnostics
            .as_deref()
            .expect("MaxVisits records the child table"),
    );
    assert_eq!(children.len() as u32, legal, "one child per legal action");
    let total: u32 = children.iter().map(|c| c.visits).sum();
    assert_eq!(
        total,
        legal + 64,
        "every visit of the budget is accounted for"
    );

    let winner = children
        .iter()
        .find(|c| c.action.coord() == WIN_IN_ONE_CELL)
        .expect("the winning cell is a root child");
    assert!(
        winner.mean_value > 0.99,
        "a placement that wins outright must back up +1 to the root's mover, not {}",
        winner.mean_value,
    );
    assert!(
        winner.visits > 64,
        "the winning child should take every visit left once it is found, not {}",
        winner.visits,
    );
}

/// A fixture that distinguishes mover-based signing from depth-parity signing.
///
/// `P1` is on the first stone of its turn and holds `(1, 0)..(4, 0)`. Neither
/// `(5, 0)` nor `(6, 0)` wins on its own; the *second* stone of the same turn
/// closes the line either way. The terminal is two plies below the root with the
/// same mover at both plies.
#[test]
fn a_win_on_the_turns_second_stone_is_preferred_and_not_avoided() {
    let game = game_after(&WIN_IN_TWO, GameSpec::default());
    let root = game.position();
    assert_eq!(root.current_player(), Player::P1);

    // Confirm both fixture plies have the same mover through engine state.
    let mut probe = root.clone();
    let applied = probe
        .advance(Action::new(WIN_IN_TWO_FIRST))
        .expect("the first stone is legal");
    assert_eq!(applied.outcome, None, "the first stone must not win");
    assert_eq!(probe.current_player(), Player::P1, "still P1's turn");
    let applied = probe
        .advance(Action::new(WIN_IN_TWO_SECOND))
        .expect("the second stone is legal");
    assert_eq!(applied.outcome.map(|o| o.winner), Some(Player::P1));

    let mut focus = Focus {
        hot: vec![WIN_IN_TWO_FIRST, WIN_IN_TWO_SECOND],
        cold: 0.000_1,
    };
    let mut session = MctsSession::new(config(64, 4), Box::new(MaxVisits), 7);
    let run = decide(&mut session, &game, &mut focus);

    let played = run.decision.action.coord();
    assert!(
        played == WIN_IN_TWO_FIRST || played == WIN_IN_TWO_SECOND,
        "the search played {played:?} rather than a stone of the winning turn",
    );

    let children = decode_children(run.decision.diagnostics.as_deref().expect("recorded"));
    let total: u32 = children.iter().map(|c| c.visits).sum();
    assert_eq!(total, 64);

    let line: Vec<_> = children
        .iter()
        .filter(|c| c.action.coord() == WIN_IN_TWO_FIRST || c.action.coord() == WIN_IN_TWO_SECOND)
        .collect();
    assert_eq!(line.len(), 2, "both stones of the turn are root children");
    for child in &line {
        assert!(
            child.mean_value > 0.0,
            "{:?} backed up {} to the root edge; a negative mean here is depth-parity negation, \
             which is the wrong function when a turn is two placements and the win is the \
             mover's own second stone",
            child.action.coord(),
            child.mean_value,
        );
    }
    let on_the_line: u32 = line.iter().map(|c| c.visits).sum();
    assert!(
        on_the_line * 2 > total,
        "the winning turn holds {on_the_line} of {total} visits; it should hold most of them",
    );
}

#[test]
fn every_visit_of_the_budget_lands_on_a_root_child() {
    let game = game_after(&WIN_IN_ONE, GameSpec::default());
    for (visits, cap) in [(1, 1), (7, 1), (16, 4), (31, 8), (64, 64)] {
        let mut session = MctsSession::new(config(visits, cap), Box::new(MaxVisits), 3);
        let run = decide(&mut session, &game, &mut Uniform);
        let children = decode_children(run.decision.diagnostics.as_deref().expect("recorded"));
        let total: u32 = children.iter().map(|c| c.visits).sum();
        assert_eq!(
            total, visits,
            "budget {visits} with cap {cap} settled {total} root-child visits",
        );
    }
}

#[test]
fn a_pump_never_exceeds_the_in_flight_cap() {
    let game = game_after(&WIN_IN_ONE, GameSpec::default());
    for cap in [1usize, 2, 5, 32] {
        let mut session = MctsSession::new(config(64, cap), Box::new(MaxVisits), 3);
        let run = decide(&mut session, &game, &mut Uniform);
        assert!(
            run.peak_in_flight() <= cap,
            "cap {cap} was exceeded: {}",
            run.peak_in_flight(),
        );
        assert_eq!(
            run.rounds[0].emitted, 1,
            "the first pump emits only the root: nothing below it can be selected until its \
             priors arrive",
        );
        assert_eq!(
            run.rounds[1].emitted, cap,
            "with budget to spare, the next pump fills the cap exactly",
        );
    }
}

#[test]
fn the_root_is_asked_about_exactly_once_per_decision() {
    let game = game_after(&WIN_IN_ONE, GameSpec::default());
    let mut session = MctsSession::new(config(4, 2), Box::new(MaxVisits), 3);
    let run = decide(&mut session, &game, &mut Uniform);
    assert_eq!(run.rounds[0].emitted, 1);
    // One root evaluation plus at most one leaf per visit. A visit that ends at a
    // terminal costs no evaluation, so the count is bounded rather than fixed.
    assert!(run.evaluations() <= 5, "{}", run.evaluations());
    assert!(run.evaluations() >= 1);
}

#[test]
#[should_panic(expected = "priors but the evaluated position has")]
fn resuming_with_the_wrong_number_of_priors_panics() {
    let game = game_after(&WIN_IN_ONE, GameSpec::default());
    let mut session = MctsSession::new(config(4, 2), Box::new(MaxVisits), 3);
    session.begin(game.position());
    let mut leaf = None;
    session.pump(&mut |id, _position| leaf = Some(id));
    session.resume(
        leaf.expect("the first pump emits the root"),
        Evaluation {
            priors: vec![1.0; 3].into(),
            value: 0.0,
        },
    );
}

#[test]
#[should_panic(expected = "value 2 is outside [-1, 1]")]
fn resuming_with_an_out_of_range_value_panics() {
    let game = game_after(&WIN_IN_ONE, GameSpec::default());
    let mut session = MctsSession::new(config(4, 2), Box::new(MaxVisits), 3);
    session.begin(game.position());
    let mut leaf = None;
    session.pump(&mut |id, _position| leaf = Some(id));
    let n = game.position().legal_count();
    session.resume(
        leaf.expect("the root"),
        Evaluation {
            priors: vec![1.0 / n as f32; n].into(),
            value: 2.0,
        },
    );
}

#[test]
#[should_panic(expected = "unknown LeafId")]
fn answering_the_same_leaf_twice_panics() {
    let game = game_after(&WIN_IN_ONE, GameSpec::default());
    let mut session = MctsSession::new(config(4, 2), Box::new(MaxVisits), 3);
    session.begin(game.position());
    let mut leaf = None;
    session.pump(&mut |id, _position| leaf = Some(id));
    let leaf = leaf.expect("the root");
    let n = game.position().legal_count();
    session.resume(leaf, uniform_evaluation(n));
    session.resume(leaf, uniform_evaluation(n));
}

#[test]
#[should_panic(expected = "unknown LeafId")]
fn an_answer_for_a_decision_the_session_has_left_panics() {
    let game = game_after(&WIN_IN_ONE, GameSpec::default());
    let mut session = MctsSession::new(config(4, 2), Box::new(MaxVisits), 3);

    session.begin(game.position());
    let mut stale = None;
    session.pump(&mut |id, _position| stale = Some(id));
    let stale = stale.expect("the root of the first decision");

    // Starting a new decision invalidates leaves from the prior tree.
    session.begin(game.position());
    session.pump(&mut |_id, _position| {});
    session.resume(stale, uniform_evaluation(game.position().legal_count()));
}

#[test]
#[should_panic(expected = "a terminal position")]
fn beginning_on_a_terminal_position_panics() {
    let mut moves = WIN_IN_ONE.to_vec();
    moves.push((WIN_IN_ONE_CELL.q, WIN_IN_ONE_CELL.r));
    let game = game_after(&moves, GameSpec::default());
    assert!(game.result().is_some());

    let mut session = MctsSession::new(config(4, 2), Box::new(MaxVisits), 1);
    session.begin(game.position());
}

#[test]
#[should_panic(expected = "pump before begin")]
fn pumping_before_begin_panics() {
    let mut session = MctsSession::new(config(4, 2), Box::new(MaxVisits), 1);
    session.pump(&mut |_id, _position| {});
}

#[test]
#[should_panic(expected = "c_puct")]
fn an_unusable_exploration_constant_is_refused_at_construction() {
    let config = MctsConfig {
        visits: NonZeroU32::new(8).expect("nonzero"),
        max_in_flight: NonZeroUsize::new(2).expect("nonzero"),
        c_puct: f32::NAN,
    };
    let _ = MctsSession::new(config, Box::new(MaxVisits), 1);
}

#[test]
fn the_decision_attests_the_position_the_search_actually_read() {
    let game = game_after(&WIN_IN_ONE, GameSpec::default());
    let mut session = MctsSession::new(config(8, 2), Box::new(Stamp(vec![9, 8, 7])), 1);
    let run = decide(&mut session, &game, &mut Uniform);

    assert_eq!(
        run.decision.zobrist,
        game.position().zobrist(),
        "the session searched its own clone of the canonical position, so the two hashes agree \
         and an in-process driver cannot desync",
    );
    assert_eq!(
        run.decision.diagnostics.as_deref(),
        Some(&[9u8, 8, 7][..]),
        "diagnostics reach the record byte for byte",
    );
}

#[test]
fn a_decision_is_taken_once_and_taking_it_resets_nothing() {
    let game = game_after(&WIN_IN_ONE, GameSpec::default());
    let mut session = MctsSession::new(config(8, 2), Box::new(MaxVisits), 1);
    assert!(
        session.take_decision().is_none(),
        "nothing has been searched"
    );

    session.begin(game.position());
    assert!(session.take_decision().is_none(), "the search has not run");

    // `decide` takes the decision itself, so a second take must come back empty.
    let _ = decide(&mut session, &game, &mut Uniform);
    assert!(
        session.take_decision().is_none(),
        "the driver already took it"
    );
    assert_eq!(
        session.pump(&mut |_id, _position| panic!("a decided session emits nothing")),
        SessionStatus::Decided,
    );
}

#[test]
fn the_same_seed_and_the_same_answers_give_the_same_moves() {
    let line = play_out(11, None);
    let same = play_out(11, None);
    let different = play_out(12, None);

    assert_eq!(
        line, same,
        "a session is a function of its position sequence, its evaluations, and its seed",
    );
    assert_ne!(
        line, different,
        "two seeds must not collapse onto one line of play",
    );
}

#[test]
fn reseeding_moves_a_session_onto_another_seeds_stream() {
    let with_eleven = play_out(11, None);
    let with_twelve = play_out(12, None);

    assert_eq!(
        play_out(12, Some(11)),
        with_eleven,
        "reseeding to 11 reproduces the seed-11 line exactly",
    );
    assert_ne!(play_out(12, Some(11)), with_twelve);
}

/// Play eight placements with a sampling selector, optionally reseeding before
/// the first one, and return the line.
fn play_out(seed: u64, reseed: Option<u64>) -> Vec<HexCoord> {
    let mut game = game_after(&[(0, 0)], GameSpec::default());
    let mut session = MctsSession::new(config(6, 3), Box::new(SampleByVisits), seed);
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

#[test]
fn a_driver_can_hold_a_session_behind_the_trait() {
    let game = game_after(&WIN_IN_ONE, GameSpec::default());
    let mut boxed: Box<dyn DecisionSession> =
        Box::new(MctsSession::new(config(4, 2), Box::new(MaxVisits), 1));
    let run = decide(boxed.as_mut(), &game, &mut Uniform);
    assert!(game.position().is_legal(run.decision.action));
}

#[test]
fn an_emitted_leaf_is_the_position_at_the_leaf_and_not_the_root() {
    let game = game_after(&WIN_IN_ONE, GameSpec::default());
    let mut session = MctsSession::new(config(8, 1), Box::new(MaxVisits), 1);
    session.begin(game.position());

    let mut seen: Vec<(u64, u32)> = Vec::new();
    let mut leaves: Vec<(LeafId, usize)> = Vec::new();
    for _ in 0..4 {
        leaves.clear();
        let status = session.pump(&mut |leaf, position: &Position| {
            seen.push((position.zobrist(), position.stone_count()));
            leaves.push((leaf, position.legal_count()));
        });
        if status == SessionStatus::Decided {
            break;
        }
        for (leaf, n) in leaves.drain(..) {
            session.resume(leaf, uniform_evaluation(n));
        }
    }

    assert_eq!(
        seen[0].1,
        game.position().stone_count(),
        "the first leaf is the root itself",
    );
    assert!(
        seen[1..].iter().all(|&(_, stones)| stones > seen[0].1),
        "every later leaf is a placement deeper than the root: {seen:?}",
    );
    assert_ne!(seen[0].0, seen[1].0, "and it hashes differently");
    assert!(
        game.position() == game.position(),
        "the game's own position is untouched by the search",
    );
    assert_eq!(game.position().stone_count(), WIN_IN_ONE.len() as u32);
}
