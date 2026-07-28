//! Cross-language receipts for the shared Gumbel line session.

use hexo_engine::{Action, HexCoord, Position};
use hexo_runner::{Decision, Game, GameSpec, Reply, Step};
use hexo_search::{DecisionSession, Evaluation, GumbelConfig, GumbelSession, SessionStatus};
use serde::Deserialize;
use std::num::{NonZeroU32, NonZeroUsize};

#[derive(Deserialize)]
struct Fixture {
    schema_version: u32,
    source: String,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    position_prefix: Vec<[i16; 2]>,
    config: Config,
    root_gumbel_noise: Vec<f64>,
    candidate_root_ranks: Vec<usize>,
    scripted_calls: Vec<ScriptedCall>,
    expected: Expected,
}

#[derive(Deserialize)]
struct Config {
    sims: u32,
    m: usize,
}

#[derive(Deserialize)]
struct Expected {
    survivor_root_ranks_after_round: Vec<Vec<usize>>,
    chosen_root_rank: usize,
    chosen_move: [i16; 2],
}

#[derive(Deserialize)]
struct ScriptedCall {
    phase: String,
    position_prefixes: Vec<Vec<[i16; 2]>>,
    root_ranks: Vec<Option<usize>>,
    legal_counts: Vec<usize>,
    evaluation: ScriptedEvaluation,
}

#[derive(Deserialize)]
struct ScriptedEvaluation {
    priors_fill: Vec<f32>,
    #[serde(default)]
    priors_override_rank: Option<usize>,
    #[serde(default)]
    priors_override_values: Option<Vec<f32>>,
    values: Vec<f32>,
}

#[test]
fn the_rust_session_matches_the_python_reference_fixture() {
    let fixture: Fixture = serde_json::from_str(include_str!("fixtures/gumbel_parity.json"))
        .expect("the committed fixture is valid JSON");
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(
        fixture.source,
        "python/mantisnet/mantisnet/klent/search.py:gumbel_choose",
    );

    for case in fixture.cases {
        run_case(case);
    }
}

fn run_case(case: Case) {
    let mut game = game_after(&case.position_prefix);
    let root = game.position().clone();
    let mut session = GumbelSession::with_gumbels(
        GumbelConfig {
            simulations: NonZeroU32::new(case.config.sims)
                .expect("fixture simulations are positive"),
            candidates: NonZeroUsize::new(case.config.m).expect("fixture candidates are positive"),
        },
        case.root_gumbel_noise,
    );
    session.begin(&game);

    for call in &case.scripted_calls {
        assert_eq!(
            call.position_prefixes.len(),
            call.legal_counts.len(),
            "{} / {}: one legal count per row",
            case.name,
            call.phase,
        );
        assert_eq!(
            call.position_prefixes.len(),
            call.root_ranks.len(),
            "{} / {}: one root rank per row",
            case.name,
            call.phase,
        );
        assert_eq!(
            call.position_prefixes.len(),
            call.evaluation.values.len(),
            "{} / {}: one value per row",
            case.name,
            call.phase,
        );

        let expected_positions: Vec<Position> = call
            .position_prefixes
            .iter()
            .map(|prefix| replay(prefix))
            .collect();
        let mut leaves = Vec::new();
        let status = session.pump(&mut |leaf, position| {
            let row = leaves.len();
            assert_eq!(
                position, &expected_positions[row],
                "{} / {} / row {row}: emitted the reference position",
                case.name, call.phase,
            );
            assert_eq!(
                position.legal_count(),
                call.legal_counts[row],
                "{} / {} / row {row}: legal count",
                case.name,
                call.phase,
            );
            leaves.push(leaf);
        });
        assert_eq!(
            status,
            SessionStatus::AwaitingEvals {
                in_flight: expected_positions.len(),
            },
            "{} / {}: one in-flight receipt per row",
            case.name,
            call.phase,
        );

        // Reverse delivery proves batching order is not a resume-order
        // dependency. The scripted row remains attached to its LeafId.
        for row in (0..leaves.len()).rev() {
            let mut priors = vec![call.evaluation.priors_fill[row]; call.legal_counts[row]];
            if let Some(rank) = call.evaluation.priors_override_rank {
                let overrides = call
                    .evaluation
                    .priors_override_values
                    .as_ref()
                    .expect("override rank has row values");
                priors[rank] = overrides[row];
            }
            session.resume(
                leaves[row],
                Evaluation {
                    priors: priors.into(),
                    value: call.evaluation.values[row],
                },
            );
        }
    }

    assert_eq!(
        session.pump(&mut |_leaf, _position| panic!("fixture has no more calls")),
        SessionStatus::Decided,
        "{}: scripted calls exhaust the search",
        case.name,
    );
    let decision = session.take_decision().expect("the fixture decides");
    assert_eq!(
        decision.zobrist,
        root.zobrist(),
        "{}: root attestation",
        case.name
    );
    assert_eq!(
        root.legal_rank(decision.action),
        Some(case.expected.chosen_root_rank),
        "{}: chosen canonical root rank",
        case.name,
    );
    assert_eq!(
        decision.action.coord(),
        HexCoord::new(case.expected.chosen_move[0], case.expected.chosen_move[1]),
        "{}: chosen move",
        case.name,
    );

    let trace = session
        .last_trace()
        .expect("the completed fixture has a trace");
    assert_eq!(
        trace.candidate_root_ranks(),
        case.candidate_root_ranks,
        "{}: Gumbel-top candidates",
        case.name,
    );
    let survivors: Vec<Vec<usize>> = trace
        .survivor_root_ranks()
        .iter()
        .map(|round| round.to_vec())
        .collect();
    assert_eq!(
        survivors, case.expected.survivor_root_ranks_after_round,
        "{}: stable survivor trace",
        case.name,
    );

    // Keep the game live until after every comparison so the fixture is also
    // exercising the normal driver-owned position rather than a direct root.
    let Step::NeedDecision { generation, .. } = game.step() else {
        panic!("fixture root is live")
    };
    game.submit(generation, Reply::Place(decision))
        .expect("the chosen fixture move is legal");
}

fn replay(prefix: &[[i16; 2]]) -> Position {
    let actions: Vec<Action> = prefix
        .iter()
        .map(|&[q, r]| Action::new(HexCoord::new(q, r)))
        .collect();
    Position::replay(&actions).expect("fixture prefix is legal")
}

fn game_after(prefix: &[[i16; 2]]) -> Game {
    let mut game = Game::new(GameSpec::default());
    for &[q, r] in prefix {
        let Step::NeedDecision { generation, .. } = game.step() else {
            panic!("fixture prefix finished early")
        };
        let decision = Decision::new(Action::new(HexCoord::new(q, r)), game.position().zobrist());
        game.submit(generation, Reply::Place(decision))
            .expect("fixture prefix move is legal");
    }
    game
}
