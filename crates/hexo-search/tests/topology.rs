//! Single-threaded multi-game batching topology.
//!
//! Thirty-two games, sixty-four seats, one batch. Every game holds two sessions;
//! every session hands out the leaves it wants evaluated and returns; the sweep
//! collects those leaves from *all* the games into one [`EncodedBatch`], crosses
//! to the evaluator once, and hands the answers back.

mod common;

use common::{Ragged, SampleByPrior, SampleByVisits, Uniform};
use hexo_runner::{Game, GameSpec, Reply, Step};
use hexo_search::{
    DecisionSession, EncodedBatch, Evaluator, LeafId, MctsConfig, MctsSession, PolicySession,
    SessionStatus,
};
use std::collections::BTreeSet;
use std::num::{NonZeroU32, NonZeroUsize};

const GAMES: usize = 32;
const PLY_CAP: u32 = 16;

/// One game and the two sessions filling its seats.
struct Lane {
    game: Game,
    seats: [Box<dyn DecisionSession>; 2],
    /// The seat that is mid-decision and the generation its decision must carry,
    /// or `None` when the lane is between decisions.
    open: Option<(usize, u64)>,
}

fn mcts(visits: u32, cap: usize, seed: u64) -> Box<dyn DecisionSession> {
    let config = MctsConfig {
        visits: NonZeroU32::new(visits).expect("nonzero"),
        max_in_flight: NonZeroUsize::new(cap).expect("nonzero"),
        c_puct: 1.4,
    };
    Box::new(MctsSession::new(config, Box::new(SampleByVisits), seed))
}

fn policy(seed: u64) -> Box<dyn DecisionSession> {
    Box::new(PolicySession::new(Box::new(SampleByPrior), seed))
}

/// Three lane shapes covering search-only, mixed, and policy-only seats.
fn lane(index: usize, spec: GameSpec) -> Lane {
    let seed = |seat: usize| (index * 2 + seat) as u64 + 1;
    let seats: [Box<dyn DecisionSession>; 2] = match index % 3 {
        0 => [mcts(6, 3, seed(0)), mcts(6, 3, seed(1))],
        1 => [mcts(5, 2, seed(0)), policy(seed(1))],
        _ => [policy(seed(0)), policy(seed(1))],
    };
    Lane {
        game: Game::new(spec),
        seats,
        open: None,
    }
}

#[test]
fn every_game_of_a_mixed_sweep_finishes_through_one_shared_batch() {
    let spec = GameSpec {
        ply_cap: NonZeroU32::new(PLY_CAP).expect("nonzero"),
        ..GameSpec::default()
    };
    let mut lanes: Vec<Lane> = (0..GAMES).map(|i| lane(i, spec)).collect();

    let mut evaluator = Uniform;
    let mut batch = EncodedBatch::with_capacity(128, 64 * 1024);
    let mut leaves: Vec<(usize, usize, LeafId)> = Vec::new();
    let mut answers = Vec::new();

    let mut sweeps = 0usize;
    let mut evaluations = 0usize;
    let mut decisions = 0usize;
    let mut widest_batch = 0usize;
    let mut most_games_in_one_batch = 0usize;

    loop {
        batch.clear();
        leaves.clear();
        let mut live = 0usize;

        for (index, lane) in lanes.iter_mut().enumerate() {
            // Start a decision for any live lane that is between them.
            if lane.open.is_none() {
                let Step::NeedDecision {
                    seat, generation, ..
                } = lane.game.step()
                else {
                    continue;
                };
                lane.seats[seat.index()].begin(&lane.game);
                lane.open = Some((seat.index(), generation));
            }
            live += 1;

            let (seat, generation) = lane.open.expect("the lane has an open decision");
            let status = lane.seats[seat].pump(&mut |leaf, position| {
                leaves.push((index, seat, leaf));
                batch.push_with(&Ragged, position);
            });

            if status == SessionStatus::Decided {
                // Submit the session-authored decision verbatim.
                let decision = lane.seats[seat]
                    .take_decision()
                    .expect("a decided session has a decision");
                assert_eq!(
                    decision.zobrist,
                    lane.game.position().zobrist(),
                    "game {index}: the seat attested a position that is not the game's",
                );
                lane.game
                    .submit(generation, Reply::Place(decision))
                    .unwrap_or_else(|e| panic!("game {index}: submission refused: {e}"));
                lane.open = None;
                decisions += 1;
            }
        }

        if live == 0 {
            break;
        }

        // Evaluate the complete sweep batch in one call.
        answers.clear();
        evaluator.evaluate(&batch, &mut answers);
        assert_eq!(answers.len(), batch.len(), "one answer per batch item");

        sweeps += 1;
        evaluations += batch.len();
        widest_batch = widest_batch.max(batch.len());
        most_games_in_one_batch = most_games_in_one_batch.max(
            leaves
                .iter()
                .map(|&(index, ..)| index)
                .collect::<BTreeSet<_>>()
                .len(),
        );

        for (&(index, seat, leaf), evaluation) in leaves.iter().zip(answers.drain(..)) {
            lanes[index].seats[seat].resume(leaf, evaluation);
        }
    }

    for (index, lane) in lanes.iter().enumerate() {
        let result = lane.game.result().unwrap_or_else(|| {
            panic!("game {index} never finished; the ply cap should have stopped it")
        });
        assert!(
            result.is_contested(),
            "game {index} ended as {result:?}, which means a seat failed rather than played",
        );
        assert!(
            lane.game.position().stone_count() <= PLY_CAP + 1,
            "game {index} ran past the cap",
        );
        assert!(lane.open.is_none(), "game {index} left a decision open");
    }

    assert_eq!(
        decisions,
        lanes
            .iter()
            .map(|lane| lane.game.plies().len())
            .sum::<usize>(),
        "every decision the sweep took reached the record",
    );
    assert!(
        most_games_in_one_batch > 1,
        "the batches were never assembled across games, so nothing here was tested",
    );
    assert!(
        widest_batch > 8,
        "the widest batch held {widest_batch} items; batching this narrow is not batching",
    );
    assert!(
        sweeps * 4 < evaluations,
        "{evaluations} evaluations took {sweeps} crossings; the batch is not doing its job",
    );
}
