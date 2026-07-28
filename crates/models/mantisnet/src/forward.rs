//! The Python-free forward boundary injected by the binary leaf.
//!
//! MantisNet owns the typed batch and output semantics. The executable owns how
//! those values cross into a live Torch module. Keeping the trait here and its
//! PyO3 implementation in `hexo-bot` resolves the container's leaf rule without
//! making either side invent the other's model contract.

use crate::encoder::RawBatch;
use std::path::Path;

/// A forward/load failure whose concrete source survives the package boundary.
pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

/// The two MantisNet cell-head outputs, concatenated by position.
///
/// Both arrays are in engine canonical legal order within each position. Their
/// ragged row boundaries are [`RawBatch::legal_offsets`] from the input batch.
#[derive(Clone, Debug, PartialEq)]
pub struct RawOutputs {
    /// Raw policy logits, one per legal action.
    pub policy_logits: Vec<f32>,
    /// Bounded action values in `[-1, 1]`, one per legal action.
    pub q_values: Vec<f32>,
}

/// One live MantisNet module, called once per collated evaluator batch.
///
/// This trait deliberately contains no Python, Torch, tensor, or device type.
/// A binary adapter converts the public [`RawBatch`] arrays to its runtime and
/// returns ordinary Rust vectors. The package then owns every semantic decision
/// made from those vectors.
pub trait Forward: Send {
    /// Run both cell heads for `batch`.
    ///
    /// Implementations must preserve the input's concatenated canonical legal
    /// order. The evaluator validates both output lengths and every value before
    /// using them.
    fn forward(&mut self, batch: &RawBatch) -> Result<RawOutputs, BoxError>;
}

/// Construct a live forward boundary from one package weight file.
///
/// Loading is separate from forwarding because checkpoint proof must build a
/// candidate module, answer the frozen probe through it, and only then publish
/// it as the package's loaded state.
pub trait ForwardLoader: Send + Sync {
    /// Load `weights`, including all package/runtime version checks.
    fn load(&self, weights: &Path) -> Result<Box<dyn Forward>, BoxError>;
}
