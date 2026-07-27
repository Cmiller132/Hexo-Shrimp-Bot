//! The mock model package: a deterministic, weightless evaluator that exercises
//! every seam the container has, with no network, no GPU, and no Python.
//!
//! **Not a placeholder.** `docs/CONTAINER_SPEC.md` §5 makes the argument: this
//! package drives the encoder, the evaluator, both session kinds, both selection
//! policies, the diagnostics channel, shard writing, checkpoint write and load,
//! the probe hash, and `fit` — the whole loop — and it can be wrong in none of
//! the ways a real model can. It is what makes the container testable in CI, and
//! it is the package the container is built against first for that reason. A
//! package the entire container is exercised against on every run earns its
//! place permanently.
//!
//! The public surface is one constructor, one trait implementation, and one
//! constant. Everything else — the encoder, the evaluator, the four selectors,
//! the salt arithmetic, the configuration grammar — is private, because all of
//! it is what a package owns and none of it is what a package exposes.
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
//! // Loading is proving: the probe hash is recomputed over what the weights
//! // actually answer, and compared against what the manifest promised.
//! let loaded = package.load(&checkpoint)?;
//! assert_eq!(loaded, written);
//!
//! // Both modes exist, and neither is the other.
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
/// One zobrist and one legal count, little-endian. It moves the day either of
/// those changes meaning, and moving it means every checkpoint written before is
/// refused by name rather than reinterpreted — weights are indexed against a
/// feature layout, and a layout that changed under them is not a layout they can
/// be read against.
pub const ENCODER_VERSION: u32 = 1;

/// This package's own version, bumped when its behaviour changes meaning.
///
/// Separate from [`ENCODER_VERSION`]: the salt arithmetic, the selectors, and the
/// diagnostics format can each move without a feature byte changing, and each of
/// them makes an older checkpoint mean something different.
pub(crate) const PACKAGE_VERSION: u32 = 1;

/// The registry name, written into every shard header and every checkpoint
/// manifest.
///
/// `mock`, not the crate name: `--package mock` is what an operator types, and
/// the name in an artefact is the one that has to match it.
pub(crate) const NAME: &str = "mock";
