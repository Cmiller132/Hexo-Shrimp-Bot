//! Mock-package move selection and diagnostics encoding.
//!
//! Self-play selectors sample the source distribution and emit diagnostics.
//! Evaluation selectors sample cubed weights and emit no diagnostics.

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
/// Visits and priors are non-negative, so the odd power preserves ordering
/// without an absolute-value transform.
const EVAL_POWER: i32 = 3;

/// Draw an index in proportion to `weights`.
///
/// # Panics
///
/// If the weights do not have a finite, positive sum.
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

/// A self-play tree-search selector that samples visits and records the table.
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

/// An evaluation tree-search selector that samples cubed visits and records no
/// diagnostics.
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

/// Derive a session's initial seed from the loaded salt and package serial.
pub(crate) const fn session_seed(salt: u64, serial: u64) -> u64 {
    mix(salt ^ serial.wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(23))
}
