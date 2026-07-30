//! Test encoder, evaluators, selectors, and session driver.
//!
//! These fixtures represent package-owned implementations.

#![allow(dead_code)]

use hexo_engine::{Action, HexCoord, Player, Position};
use hexo_runner::{Decision, Game, GameSpec, Reply, Step};
use hexo_search::{
    Child, DecisionSession, EncodedBatch, Encoder, Evaluation, Evaluator, SearchOutcome,
    SelectFromPolicy, SelectFromSearch, SessionStatus, SplitMix64,
};

/// The one test encoder: the mover, then the legal actions in canonical order.
///
/// Item length varies with the legal count.
pub struct Ragged;

impl Encoder for Ragged {
    fn encode(&self, position: &Position, out: &mut Vec<u8>) {
        out.push(position.current_player().index() as u8);
        out.extend_from_slice(&(position.legal_count() as u32).to_le_bytes());
        for action in position.legal_actions() {
            out.extend_from_slice(&action.coord().q.to_le_bytes());
            out.extend_from_slice(&action.coord().r.to_le_bytes());
        }
    }
}

/// What [`Ragged`] wrote, read back on the batcher side.
pub struct Decoded {
    pub mover: Player,
    pub actions: Vec<Action>,
}

/// Decode one test-encoder batch item.
pub fn decode(item: &[u8]) -> Decoded {
    let mover = match item[0] {
        0 => Player::P0,
        1 => Player::P1,
        other => panic!("encoded mover byte {other}"),
    };
    let count = u32::from_le_bytes(item[1..5].try_into().expect("four bytes")) as usize;
    let mut actions = Vec::with_capacity(count);
    for i in 0..count {
        let at = 5 + i * 4;
        let q = i16::from_le_bytes(item[at..at + 2].try_into().expect("two bytes"));
        let r = i16::from_le_bytes(item[at + 2..at + 4].try_into().expect("two bytes"));
        actions.push(Action::new(HexCoord::new(q, r)));
    }
    assert_eq!(item.len(), 5 + count * 4, "the item has trailing bytes");
    Decoded { mover, actions }
}

/// Equal priors and zero value.
pub struct Uniform;

impl Evaluator for Uniform {
    fn evaluate(&mut self, batch: &EncodedBatch, out: &mut Vec<Evaluation>) {
        for item in batch.iter() {
            let n = decode(item).actions.len();
            out.push(Evaluation {
                priors: vec![1.0 / n as f32; n].into(),
                value: 0.0,
            });
        }
    }
}

/// One uniform answer, for a test that resumes a leaf by hand rather than
/// through an [`Evaluator`].
pub fn uniform_evaluation(legal_count: usize) -> Evaluation {
    Evaluation {
        priors: vec![1.0 / legal_count as f32; legal_count].into(),
        value: 0.0,
    }
}

/// Uniform except on the named cells, which take almost all of the mass.
///
/// A distribution concentrated on named cells for focused search tests.
pub struct Focus {
    pub hot: Vec<HexCoord>,
    /// Relative weight of a cold cell against a hot one.
    pub cold: f32,
}

impl Evaluator for Focus {
    fn evaluate(&mut self, batch: &EncodedBatch, out: &mut Vec<Evaluation>) {
        for item in batch.iter() {
            let decoded = decode(item);
            let raw: Vec<f32> = decoded
                .actions
                .iter()
                .map(|a| {
                    if self.hot.contains(&a.coord()) {
                        1.0
                    } else {
                        self.cold
                    }
                })
                .collect();
            let total: f32 = raw.iter().sum();
            out.push(Evaluation {
                priors: raw.into_iter().map(|w| w / total).collect(),
                value: 0.0,
            });
        }
    }
}

/// Plays the most-visited root child and records the child table.
pub struct MaxVisits;

impl SelectFromSearch for MaxVisits {
    fn select(&mut self, outcome: &SearchOutcome<'_>, _rng: &mut SplitMix64) -> Action {
        outcome
            .children()
            .iter()
            .max_by_key(|c| c.visits)
            .expect("a live root has children")
            .action
    }

    fn diagnostics(&mut self, outcome: &SearchOutcome<'_>) -> Option<Vec<u8>> {
        Some(encode_children(outcome.children()))
    }
}

/// Samples a root child in proportion to its visits, so the seat varies.
pub struct SampleByVisits;

impl SelectFromSearch for SampleByVisits {
    fn select(&mut self, outcome: &SearchOutcome<'_>, rng: &mut SplitMix64) -> Action {
        let total = f64::from(outcome.total_visits());
        let mut ticket = rng.next_f64() * total;
        for child in outcome.children() {
            ticket -= f64::from(child.visits);
            if ticket < 0.0 {
                return child.action;
            }
        }
        outcome
            .children()
            .last()
            .expect("a live root has children")
            .action
    }

    fn diagnostics(&mut self, _outcome: &SearchOutcome<'_>) -> Option<Vec<u8>> {
        None
    }
}

/// Plays the first legal action and stamps fixed bytes on the record, so a test
/// can check that the bytes reach it untouched.
pub struct Stamp(pub Vec<u8>);

impl SelectFromSearch for Stamp {
    fn select(&mut self, outcome: &SearchOutcome<'_>, _rng: &mut SplitMix64) -> Action {
        outcome.children()[0].action
    }

    fn diagnostics(&mut self, _outcome: &SearchOutcome<'_>) -> Option<Vec<u8>> {
        Some(self.0.clone())
    }
}

impl SelectFromPolicy for Stamp {
    fn select(&mut self, root: &Position, _e: &Evaluation, _rng: &mut SplitMix64) -> Action {
        root.nth_legal(0)
            .expect("a live position has a legal action")
    }

    fn diagnostics(&mut self, _root: &Position, _e: &Evaluation) -> Option<Vec<u8>> {
        Some(self.0.clone())
    }
}

/// Plays the highest-prior action.
pub struct HighestPrior;

impl SelectFromPolicy for HighestPrior {
    fn select(&mut self, root: &Position, e: &Evaluation, _rng: &mut SplitMix64) -> Action {
        let (index, _) = e
            .priors
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .expect("a live position has a legal action");
        root.nth_legal(index)
            .expect("an index into the canonical legal order")
    }

    fn diagnostics(&mut self, _root: &Position, _e: &Evaluation) -> Option<Vec<u8>> {
        None
    }
}

/// Samples an action in proportion to its prior.
pub struct SampleByPrior;

impl SelectFromPolicy for SampleByPrior {
    fn select(&mut self, root: &Position, e: &Evaluation, rng: &mut SplitMix64) -> Action {
        let total: f32 = e.priors.iter().sum();
        let mut ticket = rng.next_f64() * f64::from(total);
        for (index, &p) in e.priors.iter().enumerate() {
            ticket -= f64::from(p);
            if ticket < 0.0 {
                return root.nth_legal(index).expect("a canonical index");
            }
        }
        root.nth_legal(e.priors.len() - 1)
            .expect("a canonical index")
    }

    fn diagnostics(&mut self, _root: &Position, _e: &Evaluation) -> Option<Vec<u8>> {
        None
    }
}

/// The child table, as [`MaxVisits`] writes it into the record.
pub fn encode_children(children: &[Child]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + children.len() * 20);
    out.extend_from_slice(&(children.len() as u32).to_le_bytes());
    for child in children {
        out.extend_from_slice(&child.action.coord().q.to_le_bytes());
        out.extend_from_slice(&child.action.coord().r.to_le_bytes());
        out.extend_from_slice(&child.visits.to_le_bytes());
        out.extend_from_slice(&child.mean_value.to_le_bytes());
        out.extend_from_slice(&child.prior.to_le_bytes());
    }
    out
}

/// The exact inverse of [`encode_children`].
pub fn decode_children(bytes: &[u8]) -> Vec<Child> {
    let count = u32::from_le_bytes(bytes[0..4].try_into().expect("four bytes")) as usize;
    let mut children = Vec::with_capacity(count);
    for i in 0..count {
        let at = 4 + i * 20;
        let q = i16::from_le_bytes(bytes[at..at + 2].try_into().expect("two bytes"));
        let r = i16::from_le_bytes(bytes[at + 2..at + 4].try_into().expect("two bytes"));
        let visits = u32::from_le_bytes(bytes[at + 4..at + 8].try_into().expect("four bytes"));
        let mean_value =
            f64::from_le_bytes(bytes[at + 8..at + 16].try_into().expect("eight bytes"));
        let prior = f32::from_le_bytes(bytes[at + 16..at + 20].try_into().expect("four bytes"));
        children.push(Child {
            action: Action::new(HexCoord::new(q, r)),
            visits,
            mean_value,
            prior,
        });
    }
    children
}

/// What one `pump` did.
pub struct Round {
    /// Leaves emitted by this pump.
    pub emitted: usize,
    /// What the status reported was outstanding.
    pub in_flight: usize,
}

/// One decision, and the shape of the pump/resume traffic that produced it.
pub struct Run {
    pub decision: Decision,
    pub rounds: Vec<Round>,
}

impl Run {
    /// Every leaf this decision asked about.
    pub fn evaluations(&self) -> usize {
        self.rounds.iter().map(|r| r.emitted).sum()
    }

    /// The largest number of leaves outstanding at once.
    pub fn peak_in_flight(&self) -> usize {
        self.rounds.iter().map(|r| r.in_flight).max().unwrap_or(0)
    }
}

/// Drive one session to its decision, assembling a real [`EncodedBatch`] each
/// round and answering it with one evaluator call.
pub fn decide(
    session: &mut dyn DecisionSession,
    game: &Game,
    evaluator: &mut dyn Evaluator,
) -> Run {
    session.begin(game.position());
    let mut batch = EncodedBatch::new();
    let mut leaves = Vec::new();
    let mut answers = Vec::new();
    let mut rounds = Vec::new();

    loop {
        batch.clear();
        leaves.clear();
        let status = session.pump(&mut |leaf, position| {
            leaves.push(leaf);
            batch.push_with(&Ragged, position);
        });
        let SessionStatus::AwaitingEvals { in_flight } = status else {
            break;
        };
        assert_eq!(
            in_flight,
            leaves.len(),
            "this driver resumes everything each round, so what it emitted is what is outstanding",
        );
        assert!(
            !leaves.is_empty(),
            "a pump that is awaiting evaluations and emitted nothing cannot make progress",
        );
        rounds.push(Round {
            emitted: leaves.len(),
            in_flight,
        });

        answers.clear();
        evaluator.evaluate(&batch, &mut answers);
        assert_eq!(answers.len(), batch.len(), "one answer per batch item");
        for (leaf, evaluation) in leaves.drain(..).zip(answers.drain(..)) {
            session.resume(leaf, evaluation);
        }
    }

    Run {
        decision: session
            .take_decision()
            .expect("a decided session has a decision"),
        rounds,
    }
}

/// A game with `moves` already played, each attested as a seat would attest it.
pub fn game_after(moves: &[(i16, i16)], spec: GameSpec) -> Game {
    let mut game = Game::new(spec);
    for &(q, r) in moves {
        let Step::NeedDecision { generation, .. } = game.step() else {
            panic!("the fixture ended the game before its move list did")
        };
        let action = Action::new(HexCoord::new(q, r));
        let zobrist = game.position().zobrist();
        game.submit(generation, Reply::Place(Decision::new(action, zobrist)))
            .unwrap_or_else(|e| panic!("({q}, {r}) was refused: {e}"));
    }
    game
}

/// The move list from `crates/hexo-engine/tests/fixtures.rs`, stopped one
/// placement short: `P1` is on the second stone of its turn and `(3, 0)` closes
/// `(1, 0)..(6, 0)` along `Q`.
pub const WIN_IN_ONE: [(i16, i16); 10] = [
    (0, 0),
    (1, 0),
    (2, 0),
    (0, 1),
    (0, 3),
    (4, 0),
    (5, 0),
    (0, 5),
    (0, 7),
    (6, 0),
];

/// The winning placement for [`WIN_IN_ONE`].
pub const WIN_IN_ONE_CELL: HexCoord = HexCoord { q: 3, r: 0 };

/// The same fixture stopped two placements short: `P1` is on the *first* stone
/// of its turn, holds `(1, 0)..(4, 0)`, and wins by playing `(5, 0)` and then
/// `(6, 0)` — two plies with the **same** mover.
pub const WIN_IN_TWO: [(i16, i16); 9] = [
    (0, 0),
    (1, 0),
    (2, 0),
    (0, 1),
    (0, 3),
    (3, 0),
    (4, 0),
    (0, 5),
    (0, 7),
];

/// The first and second placements of [`WIN_IN_TWO`]'s winning turn.
pub const WIN_IN_TWO_FIRST: HexCoord = HexCoord { q: 5, r: 0 };
/// The placement that completes [`WIN_IN_TWO`].
pub const WIN_IN_TWO_SECOND: HexCoord = HexCoord { q: 6, r: 0 };
