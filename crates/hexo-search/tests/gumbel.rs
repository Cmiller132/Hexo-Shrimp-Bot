//! Cross-language parity vectors for the Gumbel line session.

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
        search_config(case.config.sims, case.config.m, 1.0),
        case.root_gumbel_noise,
    );
    session.begin(game.position());

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

    // Keep the fixture live through every comparison.
    let Step::NeedDecision { generation, .. } = game.step() else {
        panic!("fixture root is live")
    };
    game.submit(generation, Reply::Place(decision))
        .expect("the chosen fixture move is legal");
}

#[test]
fn zero_temperature_ignores_root_noise() {
    let game = game_after(&[[0, 0]]);
    let legal_count = game.position().legal_count();
    let mut priors = vec![0.0; legal_count];
    priors[2] = 0.7;
    priors[5] = 0.3;
    let mut noise = vec![100.0; legal_count];
    noise[2] = -100.0;
    let mut session = GumbelSession::with_gumbels(search_config(4, 2, 0.0), noise);

    session.begin(game.position());
    let mut root_leaf = None;
    assert_eq!(
        session.pump(&mut |leaf, _position| root_leaf = Some(leaf)),
        SessionStatus::AwaitingEvals { in_flight: 1 },
    );
    session.resume(
        root_leaf.expect("the root evaluation receipt"),
        Evaluation {
            priors: priors.into(),
            value: 0.0,
        },
    );
    finish_uniform(&mut session);

    assert_eq!(
        session
            .last_trace()
            .expect("the completed search has a trace")
            .candidate_root_ranks(),
        [2, 5],
        "T=0 ranks by root opinion even when the raw Gumbels prefer every other action",
    );
    let decision = session
        .take_decision()
        .expect("the deterministic search authored a decision");
    assert_eq!(
        game.position().legal_rank(decision.action),
        Some(2),
        "with equal searched values, zero temperature keeps the highest-prior root action",
    );
}

#[test]
fn invalid_temperatures_are_refused_at_construction() {
    for temperature in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let result =
            std::panic::catch_unwind(|| GumbelSession::new(search_config(2, 1, temperature), 7));
        assert!(
            result.is_err(),
            "temperature {temperature} unexpectedly constructed a session",
        );
    }
}

fn search_config(simulations: u32, candidates: usize, temperature: f64) -> GumbelConfig {
    GumbelConfig {
        simulations: NonZeroU32::new(simulations).expect("test simulations are positive"),
        candidates: NonZeroUsize::new(candidates).expect("test candidates are positive"),
        temperature,
    }
}

fn finish_uniform(session: &mut GumbelSession) {
    loop {
        let mut leaves = Vec::new();
        match session.pump(&mut |leaf, position| {
            leaves.push((leaf, position.legal_count()));
        }) {
            SessionStatus::Decided => {
                assert!(leaves.is_empty());
                return;
            }
            SessionStatus::AwaitingEvals { in_flight } => {
                assert_eq!(in_flight, leaves.len());
            }
        }
        for (leaf, legal_count) in leaves {
            session.resume(
                leaf,
                Evaluation {
                    priors: vec![1.0 / legal_count as f32; legal_count].into(),
                    value: 0.0,
                },
            );
        }
    }
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
