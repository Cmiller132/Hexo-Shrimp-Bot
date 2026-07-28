//! Why a model package refused.

use std::path::PathBuf;

/// Everything a [`crate::ModelPackage`] can refuse to do.
///
/// The shape follows the same rule `hexo-records` follows: a variant carries
/// what it takes to locate the problem without opening a file, and every
/// version failure names *both* numbers, because "version mismatch" alone does
/// not say which side is old.
///
/// Two variants exist so that a package never has to flatten something that had
/// structure. [`PackageError::Failed`] keeps a package's own error as a boxed
/// source — a `RecordError` from a shard read, a Python exception from a
/// trainer — so a caller can still downcast to it, and
/// [`PackageError::NoTrainingData`] is a first-class variant because "the fit
/// consumed nothing" is the exact silent failure `docs/CONTAINER_SPEC.md` §5
/// builds the mock package to catch, and it deserves better than a message.
#[derive(Debug)]
pub enum PackageError {
    /// The filesystem refused an operation.
    Io {
        /// The file or directory it was refused on.
        path: PathBuf,
        /// What the filesystem said.
        source: std::io::Error,
    },
    /// A `manifest.json` is not JSON, or is not a [`crate::Manifest`].
    ///
    /// Missing and unknown fields both land here: the manifest is written and
    /// read by this crate alone, so a field this build does not know is a file
    /// from another build rather than a forward-compatible extension.
    ManifestParse {
        /// The manifest that could not be read.
        path: PathBuf,
        /// What the deserialiser said, with its line and column.
        source: serde_json::Error,
    },
    /// An artefact names a different package than the one running.
    ///
    /// Raised by [`crate::Manifest::validate`] on a checkpoint, and by a
    /// package's `fit` on a record shard whose header names someone else:
    /// either way the artefact was produced by code that is not this code, and
    /// its bytes mean whatever that code decided they mean.
    PackageName {
        /// The running package's registry name.
        expected: String,
        /// What the artefact states.
        found: String,
    },
    /// The checkpoint was written by a different version of this package.
    PackageVersion {
        /// What this build is.
        expected: u32,
        /// What the checkpoint states.
        found: u32,
    },
    /// The weights were trained against different encoder bytes than this build
    /// writes, so every feature they learned is indexed against a layout that no
    /// longer exists.
    EncoderVersion {
        /// What this build's encoder is.
        expected: u32,
        /// What the checkpoint states.
        found: u32,
    },
    /// The weights were produced under different rules than this build links.
    RulesVersion {
        /// `hexo_engine::RULES_VERSION` as linked.
        expected: u32,
        /// What the checkpoint states.
        found: u32,
    },
    /// The weights were produced under a different canonical action ordering,
    /// so the policy head's index *i* does not mean the same placement it did.
    ActionOrderVersion {
        /// `hexo_engine::ACTION_ORDER_VERSION` as linked.
        expected: u32,
        /// What the checkpoint states.
        found: u32,
    },
    /// The weights were produced under a different runner decision and result
    /// model than this build links.
    ProtocolVersion {
        /// `hexo_runner::PROTOCOL_VERSION` as linked.
        expected: u32,
        /// What the checkpoint states.
        found: u32,
    },
    /// Package-owned checkpoint metadata disagrees with this package instance.
    ///
    /// The JSON is opaque to this crate; carrying both values lets the package
    /// report the mismatch without flattening its configuration to a string or
    /// teaching the container what any field means.
    PackageMetadata {
        /// What the running package requires.
        expected: serde_json::Value,
        /// What the checkpoint states.
        found: serde_json::Value,
    },
    /// The loaded weights do not answer the probe the way the manifest says
    /// they should.
    ///
    /// This is the detector `docs/CONTAINER_SPEC.md` §10.2 exists for, and it is
    /// the one failure in this enum that nothing else in the system would ever
    /// report: the wrong checkpoint, a swap that folded away to a no-op, or a
    /// runtime that drifted between build and run all load cleanly and then
    /// train against weights nobody chose.
    ProbeMismatch {
        /// What the manifest promised.
        expected: u64,
        /// What the loaded weights actually produced.
        computed: u64,
    },
    /// Something that needs weights was asked for before [`crate::ModelPackage::load`]
    /// succeeded.
    NotLoaded {
        /// The package that has no weights.
        package: &'static str,
    },
    /// The package has no session variant by that name.
    ///
    /// The variant vocabulary is the package's, so this is the only thing a
    /// package that defines none can say — which is why
    /// [`crate::ModelPackage::variant_session`] may default to producing it.
    UnknownVariant {
        /// The package that was asked.
        package: &'static str,
        /// The name it does not have.
        variant: String,
    },
    /// The package's configuration string is not one it can use.
    ///
    /// The syntax is package-defined, so the description is the package's own:
    /// nothing above the package knows what keys it has, and a shared enum that
    /// tried to enumerate them would be the container having an opinion about a
    /// model.
    InvalidConfig {
        /// The package that refused.
        package: &'static str,
        /// What is wrong with the string, in the package's own words.
        problem: String,
    },
    /// A weight file is present but does not hold weights this build can read.
    ///
    /// The description is the package's, because the format is: the container
    /// stores a file and a manifest, not a description of layers
    /// (`docs/CONTAINER_SPEC.md` §10). A package whose loader raises structured
    /// failures of its own reports them through [`PackageError::Failed`]
    /// instead, which keeps the source error intact.
    MalformedWeights {
        /// The weight file.
        path: PathBuf,
        /// What is wrong with it, in the package's own words.
        problem: String,
    },
    /// A `fit` was handed nothing to fit on.
    ///
    /// A first-class variant rather than a message because it is the failure the
    /// whole checkpoint design exists to catch: a fit that consumed no games and
    /// produced weights anyway is indistinguishable, downstream, from a fit that
    /// worked. Every count that was zero is carried, so the caller can tell an
    /// empty shard list from a shard list of empty shards.
    NoTrainingData {
        /// The package whose fit refused.
        package: &'static str,
        /// How many shards it was handed.
        shards: usize,
        /// How many games it found in them.
        games: usize,
    },
    /// The package deliberately does not implement this operation.
    ///
    /// This is distinct from a failed attempt: no partial work was started and
    /// the reason names the owner decision required before the operation can
    /// exist honestly.
    Unsupported {
        /// The package that declined.
        package: &'static str,
        /// The operation it does not implement.
        operation: &'static str,
        /// Why the package declines it.
        reason: &'static str,
    },
    /// A package-internal operation failed, carrying the package's own error.
    ///
    /// This is the escape hatch for the errors a package's dependencies raise —
    /// `hexo-records` is not a dependency of this crate, and neither is
    /// PyTorch — and it is deliberately not a string: the source is boxed so it
    /// survives whole and can be downcast back to the concrete type by anything
    /// that links it. `doing` says which operation, in the package's words, so
    /// the message reads as a sentence without the source having to repeat it.
    Failed {
        /// The package that failed.
        package: &'static str,
        /// What it was doing, as a gerund clause: `"reading a record shard"`.
        doing: &'static str,
        /// The error it failed with, intact.
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

impl PackageError {
    /// Wrap a package-internal error as [`PackageError::Failed`].
    ///
    /// The convenience exists because the boxing is the whole point and a
    /// package should not have to write it out at every call site — the
    /// alternative a package reaches for otherwise is `to_string`, which throws
    /// away the error it took trouble to produce.
    pub fn failed<E>(package: &'static str, doing: &'static str, source: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::Failed {
            package,
            doing,
            source: Box::new(source),
        }
    }
}

impl core::fmt::Display for PackageError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "{}: {source}", path.display()),
            Self::ManifestParse { path, source } => {
                write!(f, "{}: {source}", path.display())
            }
            Self::PackageName { expected, found } => write!(
                f,
                "the artefact was produced by package {found:?}, but this build runs {expected:?}"
            ),
            Self::PackageVersion { expected, found } => write!(
                f,
                "the checkpoint was written by package version {found}, but this build is version \
                 {expected}"
            ),
            Self::EncoderVersion { expected, found } => write!(
                f,
                "the weights were trained against encoder version {found}, but this build's \
                 encoder is version {expected}"
            ),
            Self::RulesVersion { expected, found } => write!(
                f,
                "the checkpoint states rules version {found}, but this build links rules version \
                 {expected}"
            ),
            Self::ActionOrderVersion { expected, found } => write!(
                f,
                "the checkpoint states action-order version {found}, but this build links \
                 action-order version {expected}"
            ),
            Self::ProtocolVersion { expected, found } => write!(
                f,
                "the checkpoint states runner protocol version {found}, but this build links \
                 protocol version {expected}"
            ),
            Self::PackageMetadata { expected, found } => write!(
                f,
                "the checkpoint's package metadata {found} does not match the running package's \
                 required metadata {expected}"
            ),
            Self::ProbeMismatch { expected, computed } => write!(
                f,
                "the loaded weights answer the probe with {computed:#018x}, but the manifest \
                 promises {expected:#018x}; these are not the weights the checkpoint describes"
            ),
            Self::NotLoaded { package } => write!(
                f,
                "{package} has no weights loaded; a checkpoint has to be loaded before anything \
                 can answer"
            ),
            Self::UnknownVariant { package, variant } => {
                write!(f, "{package} has no session variant named {variant:?}")
            }
            Self::InvalidConfig { package, problem } => {
                write!(f, "{package} cannot use this configuration: {problem}")
            }
            Self::MalformedWeights { path, problem } => {
                write!(f, "{}: {problem}", path.display())
            }
            Self::NoTrainingData {
                package,
                shards,
                games,
            } => write!(
                f,
                "{package} was asked to fit on {shards} shard(s) holding {games} game(s); a fit \
                 that consumed nothing would produce weights nothing trained"
            ),
            Self::Unsupported {
                package,
                operation,
                reason,
            } => write!(f, "{package} does not support {operation}: {reason}"),
            Self::Failed {
                package,
                doing,
                source,
            } => write!(f, "{package} failed {doing}: {source}"),
        }
    }
}

impl std::error::Error for PackageError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::ManifestParse { source, .. } => Some(source),
            Self::Failed { source, .. } => Some(source.as_ref()),
            _ => None,
        }
    }
}
