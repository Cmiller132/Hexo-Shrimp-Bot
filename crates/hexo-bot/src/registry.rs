//! The one place a package name becomes code.

use crate::error::BotError;
use hexo_model::ModelPackage;
use hexo_model_mantisnet::{BoxError, ForwardLoader, MantisPackage};
use hexo_model_mock::MockPackage;
use std::fmt;
use std::path::Path;
use std::sync::Arc;

/// Every model package this build carries, in the spelling `--package` takes.
pub const PACKAGES: &[&str] = &["mock", "mantisnet"];

/// Package construction plus the binary-owned runtime boundary it needs.
///
/// The registry itself is Python-free. The executable injects its private PyO3
/// loader; in-process logic tests inject a loader that only refuses if a test
/// actually attempts to load MantisNet.
pub struct PackageRegistry {
    mantisnet_loader: Arc<dyn ForwardLoader>,
}

impl PackageRegistry {
    /// A registry backed by the executable's live MantisNet runtime.
    #[must_use]
    pub fn new(mantisnet_loader: Arc<dyn ForwardLoader>) -> Self {
        Self { mantisnet_loader }
    }

    /// A registry whose MantisNet runtime fails loudly when used.
    ///
    /// This exists for Python-free container tests that exercise the mock. It
    /// still parses and constructs the MantisNet package, so the registry name
    /// and configuration grammar remain testable; only loading a live module is
    /// unavailable.
    #[must_use]
    pub fn without_mantisnet_runtime() -> Self {
        Self::new(Arc::new(UnavailableForwardLoader))
    }

    /// Build the named package from its configuration string.
    ///
    /// `config` is handed to the package verbatim and this crate has no opinion
    /// about it, including whether an empty string is usable.
    ///
    /// # Errors
    ///
    /// [`BotError::UnknownPackage`] for a name this build does not have, and
    /// [`BotError::Package`] for a configuration the package refuses.
    pub fn construct(&self, name: &str, config: &str) -> Result<Box<dyn ModelPackage>, BotError> {
        match name {
            "mock" => Ok(Box::new(MockPackage::from_config(config)?)),
            "mantisnet" => Ok(Box::new(MantisPackage::from_config(
                config,
                Arc::clone(&self.mantisnet_loader),
            )?)),
            _ => Err(BotError::UnknownPackage {
                name: name.to_owned(),
            }),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct UnavailableForwardLoader;

impl ForwardLoader for UnavailableForwardLoader {
    fn load(&self, _weights: &Path) -> Result<Box<dyn hexo_model_mantisnet::Forward>, BoxError> {
        Err(Box::new(UnavailableRuntime))
    }
}

#[derive(Clone, Copy, Debug)]
struct UnavailableRuntime;

impl fmt::Display for UnavailableRuntime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(
            "this PackageRegistry has no MantisNet forward runtime; the hexo-bot executable \
             injects PyO3, while Python-free tests may only load mock",
        )
    }
}

impl std::error::Error for UnavailableRuntime {}
