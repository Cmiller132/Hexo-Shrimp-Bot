//! The one place a package name becomes code.

use crate::error::BotError;
use hexo_model::ModelPackage;
use hexo_model_mock::MockPackage;

/// Every model package this build carries, in the spelling `--package` takes.
///
/// `docs/CONTAINER_SPEC.md` §5: adding a package is a new crate under
/// `crates/models/` and one arm of [`construct`], and neither the engine, nor
/// the runner, nor the record format learns anything. The registry is also not a
/// museum — a package that no longer builds against the current record format
/// and the current seam comes out in the same change that obsoletes it.
pub const PACKAGES: &[&str] = &["mock"];

/// Build the named package from its configuration string.
///
/// `config` is handed to the package verbatim and this crate has no opinion
/// about it, including whether an empty string is usable: the grammar is the
/// package's, so the package is what refuses. `mock` has one required key and no
/// default shape, so `""` comes back as a loud `InvalidConfig` rather than as a
/// search nobody chose.
///
/// # Errors
///
/// [`BotError::UnknownPackage`] for a name this build does not have, listing the
/// ones it does, and [`BotError::Package`] for a configuration the package
/// refuses.
pub fn construct(name: &str, config: &str) -> Result<Box<dyn ModelPackage>, BotError> {
    match name {
        "mock" => Ok(Box::new(MockPackage::from_config(config)?)),
        _ => Err(BotError::UnknownPackage {
            name: name.to_owned(),
        }),
    }
}
