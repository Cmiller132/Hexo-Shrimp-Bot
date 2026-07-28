//! The MantisNet model package.
//!
//! [`encoder`] is the Python-free implementation of MantisNet's position
//! representation. The PyO3 extension and the container package both consume
//! this same core, so the representation has one implementation and one
//! version owner.

mod config;
pub mod encoder;
mod forward;
pub mod improvement;
mod package;
mod seam;
mod select;

pub use forward::{BoxError, Forward, ForwardLoader, RawOutputs};
pub use package::{MantisPackage, WEIGHTS_FILE};

/// Version of the MantisNet position representation.
///
/// Bumped whenever the bytes or index tables produced by [`encoder`] change
/// meaning. Checkpoints are not compatible across versions.
pub const MODEL_REPR_VERSION: u32 = 1;

/// Version of the MantisNet package semantics outside the representation.
///
/// This covers the KLENT improvement, session choices, diagnostics bytes, and
/// checkpoint metadata. It is intentionally separate from
/// [`MODEL_REPR_VERSION`], whose one job is naming the feature representation.
pub const PACKAGE_VERSION: u32 = 1;

/// The registry and checkpoint name of this package.
pub const PACKAGE_NAME: &str = "mantisnet";
