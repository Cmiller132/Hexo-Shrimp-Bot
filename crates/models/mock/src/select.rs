//! Move selection and the diagnostics format: the two things `hexo-search`
//! ships none of, and this package owns.
//!
//! Four selectors, two per session shape, because the two modes are two
//! contracts. A self-play seat samples broadly and writes its distribution into
//! the record; an eval seat samples sharply and writes nothing. Neither is
//! argmax: two deterministic seats replay one game, so a thousand-game match
//! would carry no more information than one.

use crate::weights::mix;
use hexo_engine::{Action, Position};
use hexo_search::{
    Child, Evaluation, SearchOutcome, SelectFromPolicy, SelectFromSearch, SplitMix64,
};

/// Diagnostics kind: a visit table, written by a self-play tree search.
pub(crate) const TAG_VISITS: u8 = 0;

/// Diagnostics kind: a prior table, written by a self-play policy session.
pub(crate) const TAG_PRIORS: u8 = 1;

/// How sharply an eval seat samples: it draws proportional to the **cube** of
/// each candidate's weight.
///
/// A cube rather than an argmax, and rather than a fourth power, for two
/// reasons. Sharp enough that the move a search actually preferred wins the
/// overwhelming majority of the time, so an eval match measures the checkpoint
/// and not the sampler; soft enough that a close second is played often enough
/// for a thousand-game match to be a thousand different games. An odd power also
/// needs no absolute value: visits and priors are both non-negative, so the
/// ordering survives untouched.
const EVAL_POWER: i32 = 3;

/// Draw an index in proportion to `weights`.
///
/// # Panics
///
/// If the weights do not sum to something positive. Every table this package
/// hands in is either a visit count summing to the search budget or a prior from
/// [`crate::seam::MockEvaluator`], which is strictly positive — so a
/// non-positive total means the search or the evaluator broke, and silently
/// falling back to a uniform draw would turn that into a seat that still plays.
fn sample(weights: &[f64], rng: &mut SplitMix64) -> usize {
    let total: f64 = weights.iter().sum();
    assert!(
        total > 0.0 && total.is_finite(),
        "the mock selector was handed {} candidates totalling {total}; there is nothing to draw \
         from",
        weights.len(),
    );
    let mut ticket = rng.next_f64() * total;
    for (index, &weight) in weights.iter().enumerate() {
        ticket -= weight;
        if ticket < 0.0 {
            return index;
        }
    }
    // Reached only by floating-point drift on the last few ulps of the total.
    weights.len() - 1
}

/// The visit table, as a self-play tree search writes it into the record.
fn encode_visits(children: &[Child]) -> Vec<u8> {
    let mut out = Vec::with_capacity(5 + children.len() * 8);
    out.push(TAG_VISITS);
    let count = u32::try_from(children.len()).expect("a root's child count fits u32");
    out.extend_from_slice(&count.to_le_bytes());
    for child in children {
        out.extend_from_slice(&child.action.id().0.to_le_bytes());
        out.extend_from_slice(&child.visits.to_le_bytes());
    }
    out
}

/// The prior table, as a self-play policy session writes it into the record.
fn encode_priors(root: &Position, evaluation: &Evaluation) -> Vec<u8> {
    let mut out = Vec::with_capacity(5 + evaluation.priors.len() * 8);
    out.push(TAG_PRIORS);
    let count = u32::try_from(evaluation.priors.len()).expect("a legal count fits u32");
    out.extend_from_slice(&count.to_le_bytes());
    for (index, prior) in evaluation.priors.iter().enumerate() {
        let action = root
            .nth_legal(index)
            .expect("a prior is indexed by the canonical legal order");
        out.extend_from_slice(&action.id().0.to_le_bytes());
        out.extend_from_slice(&prior.to_le_bytes());
    }
    out
}

/// A self-play seat behind a tree search: samples proportional to visits, and
/// records the whole visit table.
///
/// Proportional rather than sharpened because the visit distribution *is* the
/// policy target, and a self-play run that only ever played its own argmax would
/// collect a target it never explored around.
pub(crate) struct SelfPlaySearch;

impl SelectFromSearch for SelfPlaySearch {
    fn select(&mut self, outcome: &SearchOutcome<'_>, rng: &mut SplitMix64) -> Action {
        let weights: Vec<f64> = outcome
            .children()
            .iter()
            .map(|child| f64::from(child.visits))
            .collect();
        outcome.children()[sample(&weights, rng)].action
    }

    fn diagnostics(&mut self, outcome: &SearchOutcome<'_>) -> Option<Vec<u8>> {
        Some(encode_visits(outcome.children()))
    }
}

/// An eval seat behind a tree search: samples proportional to the cube of the
/// visits, and records nothing.
///
/// Nothing, because the annotations exist to be trained on and an eval game
/// trains nothing. Its shard is written to be read for results — which
/// checkpoint beat which, and how — and diagnostics on it would be bytes nobody
/// consumes taking up the largest field in the format.
pub(crate) struct EvalSearch;

impl SelectFromSearch for EvalSearch {
    fn select(&mut self, outcome: &SearchOutcome<'_>, rng: &mut SplitMix64) -> Action {
        let weights: Vec<f64> = outcome
            .children()
            .iter()
            .map(|child| f64::from(child.visits).powi(EVAL_POWER))
            .collect();
        outcome.children()[sample(&weights, rng)].action
    }

    fn diagnostics(&mut self, _outcome: &SearchOutcome<'_>) -> Option<Vec<u8>> {
        None
    }
}

/// A self-play seat behind a policy session: samples proportional to the priors,
/// and records the whole prior table.
pub(crate) struct SelfPlayPolicy;

impl SelectFromPolicy for SelfPlayPolicy {
    fn select(&mut self, root: &Position, evaluation: &Evaluation, rng: &mut SplitMix64) -> Action {
        let weights: Vec<f64> = evaluation.priors.iter().map(|&p| f64::from(p)).collect();
        root.nth_legal(sample(&weights, rng))
            .expect("a prior is indexed by the canonical legal order")
    }

    fn diagnostics(&mut self, root: &Position, evaluation: &Evaluation) -> Option<Vec<u8>> {
        Some(encode_priors(root, evaluation))
    }
}

/// An eval seat behind a policy session: samples proportional to the cube of the
/// priors, and records nothing.
pub(crate) struct EvalPolicy;

impl SelectFromPolicy for EvalPolicy {
    fn select(&mut self, root: &Position, evaluation: &Evaluation, rng: &mut SplitMix64) -> Action {
        let weights: Vec<f64> = evaluation
            .priors
            .iter()
            .map(|&p| f64::from(p).powi(EVAL_POWER))
            .collect();
        root.nth_legal(sample(&weights, rng))
            .expect("a prior is indexed by the canonical legal order")
    }

    fn diagnostics(&mut self, _root: &Position, _evaluation: &Evaluation) -> Option<Vec<u8>> {
        None
    }
}

/// The seed a session is constructed with, derived from the loaded salt and a
/// per-package serial.
///
/// `docs/CONTAINER_SPEC.md` §12 leaves seeding to the driver — a session takes a
/// seed at construction and exposes `reseed`, and nothing above that seam exists
/// yet — so the package may construct with any seed it likes. What it may not do
/// is hand two concurrent sessions the same stream, which is why the serial is
/// here: a driver that forgets to reseed gets sessions that differ from each
/// other, rather than a self-play run of one repeated game.
pub(crate) const fn session_seed(salt: u64, serial: u64) -> u64 {
    mix(salt ^ serial.wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(23))
}
