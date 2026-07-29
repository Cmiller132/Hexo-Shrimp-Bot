//! Mock position encoding and deterministic evaluation.
//!
//! Encodings are functions of position state; outputs are functions of the
//! encoded state and loaded salt.

use crate::weights::{mix, unit};
use hexo_engine::Position;
use hexo_search::{EncodedBatch, Encoder, Evaluation, Evaluator};

/// How many bytes the encoder writes per position.
pub(crate) const ITEM_BYTES: usize = 12;

/// Keeps the value stream out of the prior stream.
const VALUE_TWEAK: u64 = 0x1234_5678_9abc_def0;

/// A margin that keeps a value strictly inside `[-1, 1]`.
///
/// The margin keeps the converted `f32` strictly inside the interval.
const VALUE_MARGIN: f64 = 255.0 / 256.0;

/// The position's hash and its legal count, and nothing else.
///
/// The encoding is exactly twelve bytes: an eight-byte zobrist followed by a
/// four-byte legal count, both little-endian.
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
/// Priors are strictly positive and normalized; values lie strictly inside
/// `[-1, 1]`. Outputs are deterministic in the salt and encoded zobrist.
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
    /// The returned weight lies in `[1, 2)`.
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
    /// If the item is not [`ITEM_BYTES`] long or states no legal actions.
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
