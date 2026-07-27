//! The mock's encoder and evaluator: everything that stands in for a network.
//!
//! Both sides are pure functions of the position and the salt, which is what
//! makes the probe hash a real detector here: nothing is timing-dependent,
//! nothing is device-dependent, and two builds of the same source answer
//! identically. What moves the hash is the salt, and the salt only moves when a
//! `fit` writes a new one.

use crate::weights::{mix, unit};
use hexo_engine::Position;
use hexo_search::{EncodedBatch, Encoder, Evaluation, Evaluator};

/// How many bytes the encoder writes per position.
pub(crate) const ITEM_BYTES: usize = 12;

/// Keeps the value stream out of the prior stream.
const VALUE_TWEAK: u64 = 0x1234_5678_9abc_def0;

/// A margin that keeps a value strictly inside `[-1, 1]`.
///
/// `hexo_search::Evaluation` allows the endpoints, but ±1 is the value of a
/// *decided* position, and a network that has not seen the game end has no
/// business claiming it. The margin also survives the `f64` to `f32` rounding,
/// which on its own would turn `1 - 2^-53` into exactly `1.0`.
const VALUE_MARGIN: f64 = 255.0 / 256.0;

/// The position's hash and its legal count, and nothing else.
///
/// Twelve bytes is the whole feature set. It is enough to be a real encoding —
/// the zobrist distinguishes every position the evaluator will ever see, and the
/// legal count is what tells the evaluator how many priors to produce — and it is
/// deliberately not a feature *plan*: this package exists to exercise the seam,
/// and an encoder with planes would be pretending to be a model.
pub(crate) struct MockEncoder;

impl Encoder for MockEncoder {
    fn encode(&self, position: &Position, out: &mut Vec<u8>) {
        out.extend_from_slice(&position.zobrist().to_le_bytes());
        let legal = u32::try_from(position.legal_count()).expect("a legal count fits u32");
        out.extend_from_slice(&legal.to_le_bytes());
    }
}

/// A deterministic evaluator whose whole state is the salt.
///
/// Priors are strictly positive and normalised to sum to one; the value lands
/// strictly inside `[-1, 1]`. Both are functions of the salt and the position's
/// zobrist, so the same weights answer the same position the same way forever,
/// and two salts disagree everywhere.
pub(crate) struct MockEvaluator {
    /// The loaded weights, whole.
    salt: u64,
}

impl MockEvaluator {
    /// An evaluator holding `salt`.
    pub(crate) const fn new(salt: u64) -> Self {
        Self { salt }
    }

    /// The unnormalised weight of the `index`-th legal action.
    ///
    /// In `[1, 2)`, so it is strictly positive whatever the salt is and the
    /// normalised priors of an `n`-action position stay within a factor of two
    /// of `1/n`. A weight that could reach zero would let the evaluator hand a
    /// sampling selector a table it cannot draw from.
    fn weight(&self, zobrist: u64, index: usize) -> f64 {
        let stream = mix(self.salt
            ^ zobrist.rotate_left(17)
            ^ (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        1.0 + unit(stream)
    }

    /// The value of a position, from the side to move at it.
    fn value(&self, zobrist: u64) -> f32 {
        let stream = mix(self.salt.rotate_left(32) ^ zobrist ^ VALUE_TWEAK);
        // The half-step keeps the draw off both ends of the unit interval, so
        // the mapped value is strictly inside `(-1, 1)` before the margin.
        let draw = unit(stream) + 0.5 * (1.0 / 9_007_199_254_740_992.0);
        (draw.mul_add(2.0, -1.0) * VALUE_MARGIN) as f32
    }

    /// One whole answer for one encoded item.
    ///
    /// # Panics
    ///
    /// If the item is not [`ITEM_BYTES`] long, or states no legal actions. Both
    /// mean the batch was filled by something other than [`MockEncoder`], and
    /// answering it anyway would put this package's priors against another
    /// package's action set.
    fn answer(&self, item: &[u8]) -> Evaluation {
        assert_eq!(
            item.len(),
            ITEM_BYTES,
            "the mock evaluator was handed a {}-byte item; its encoder writes {ITEM_BYTES}",
            item.len(),
        );
        let zobrist = u64::from_le_bytes(item[0..8].try_into().expect("eight bytes"));
        let legal = u32::from_le_bytes(item[8..12].try_into().expect("four bytes")) as usize;
        assert!(
            legal > 0,
            "the mock evaluator was handed a position with no legal actions; a terminal position \
             is answered by the engine, not by a network",
        );

        let weights: Vec<f64> = (0..legal).map(|i| self.weight(zobrist, i)).collect();
        let total: f64 = weights.iter().sum();
        Evaluation {
            priors: weights.into_iter().map(|w| (w / total) as f32).collect(),
            value: self.value(zobrist),
        }
    }
}

impl Evaluator for MockEvaluator {
    fn evaluate(&mut self, batch: &EncodedBatch, out: &mut Vec<Evaluation>) {
        out.reserve(batch.len());
        for item in batch.iter() {
            out.push(self.answer(item));
        }
    }
}
