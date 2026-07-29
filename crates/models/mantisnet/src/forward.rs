//! Runtime-independent MantisNet forward boundary.
//!
//! This crate owns typed batch and output semantics; the executable supplies the
//! adapter to its model runtime.

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
/// Implementations convert [`RawBatch`] arrays to their runtime and return Rust
/// vectors; no runtime-specific type crosses this trait.
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
/// Loaders produce a candidate module that can be probe-verified before
/// publication.
pub trait ForwardLoader: Send + Sync {
    /// Load `weights`, including all package/runtime version checks.
    fn load(&self, weights: &Path) -> Result<Box<dyn Forward>, BoxError>;
}
