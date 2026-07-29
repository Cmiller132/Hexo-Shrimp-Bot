//! Deterministic mock model package for container integration.
//!
//! The package implements encoding, evaluation, self-play and evaluation
//! sessions, diagnostics, checkpoints, probe verification, and fitting without
//! a network or external runtime.
//!
//! Its public surface is [`MockPackage`] and [`ENCODER_VERSION`]. Encoding,
//! evaluation, selection, weights, and configuration parsing remain
//! package-private.
//!
//! ```
//! use hexo_model::ModelPackage;
//! use hexo_model_mock::MockPackage;
//!
//! let dir = tempfile::tempdir()?;
//! let checkpoint = dir.path().join("epoch-0");
//!
//! let mut package = MockPackage::from_config("search=mcts:visits=16,inflight=2,cpuct=1.4")?;
//! let written = package.init(&checkpoint)?;
//!
//! // Loading recomputes and verifies the checkpoint's probe hash.
//! let loaded = package.load(&checkpoint)?;
//! assert_eq!(loaded, written);
//!
//! // Required modes and named variants use separate constructors.
//! let _self_play = package.self_play_session()?;
//! let _eval = package.eval_session()?;
//! let _variant = package.variant_session("policy")?;
//! assert!(package.variant_session("greedy").is_err());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod config;
mod package;
mod seam;
mod select;
mod weights;

pub use package::MockPackage;

/// The version of the bytes [`MockPackage`]'s encoder writes.
///
/// The encoding is one little-endian zobrist and one little-endian legal count.
/// Increment this version whenever either field changes shape, order, or meaning.
pub const ENCODER_VERSION: u32 = 1;

/// This package's own version, bumped when its behaviour changes meaning.
///
/// This version covers salt arithmetic, selection, and diagnostics independently
/// of [`ENCODER_VERSION`].
pub(crate) const PACKAGE_VERSION: u32 = 1;

/// The registry name, written into every shard header and every checkpoint
/// manifest.
///
/// The registry name is `mock`.
pub(crate) const NAME: &str = "mock";
