//! The model-package API: what every model crate provides to the container, and
//! the checkpoint manifest and probe that hold it honest.
//!
//! A model is a crate. Packages live under `crates/models/<name>/`, each one
//! implements [`ModelPackage`], and the container's entire knowledge of a model
//! is that trait plus a name registry in `hexo-bot`
//! (`docs/CONTAINER_SPEC.md` §5). Nothing here knows what a feature, a layer, or
//! a loss is, and nothing here is allowed to acquire an opinion about one:
//! the moment the container can describe an architecture, adding a package stops
//! being a new crate and one registry entry.
//!
//! Three things live in this crate rather than in a package, because they are
//! what the container needs *from* a package rather than what a package decides:
//!
//! - [`ModelPackage`], the surface — including the rule that the two session
//!   modes are separate required methods, which is `hexo-player`'s argument
//!   about defaults applied one level up.
//! - [`Manifest`], the checkpoint's account of itself: which package, which
//!   versions, which epoch, and what these weights are supposed to answer. It
//!   deliberately does not describe the architecture.
//! - [`probe_hash`], the detector for every way a checkpoint can be wrong
//!   without anything crashing.
//!
//! ```
//! use hexo_model::{Manifest, probe_positions};
//!
//! // A manifest is written by whoever produced the weights, and validated by
//! // whoever is about to run them.
//! let manifest = Manifest::new("mock", 1, 1, 0, 0x0123_4567_89ab_cdef);
//! manifest.validate("mock", 1, 1)?;
//! assert!(manifest.validate("gnn", 1, 1).is_err());
//!
//! // The probe set is fixed, legal, and live: every position has actions to
//! // carry priors for.
//! for position in probe_positions() {
//!     assert!(!position.is_terminal());
//!     assert!(position.legal_count() > 0);
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod error;
pub mod manifest;
pub mod package;
pub mod probe;

pub use error::PackageError;
pub use manifest::{MANIFEST_FILE, Manifest};
pub use package::ModelPackage;
pub use probe::{probe_hash, probe_positions};
