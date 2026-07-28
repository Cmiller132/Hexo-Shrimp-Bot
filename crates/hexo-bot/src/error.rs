//! Why the bot refused.

use crate::registry;
use std::path::PathBuf;

/// Everything `hexo-bot` can fail with.
///
/// The shape follows the rule `hexo-records` and `hexo-model` follow: a variant
/// carries what it takes to locate the problem without opening a file, and a
/// disagreement names *both* sides. Two of them exist because this crate is the
/// one place where an operator's typing meets the workspace's libraries —
/// [`BotError::Usage`] and [`BotError::ResumeMismatch`] are the only errors a
/// correct binary produces in the course of a working day, and both of them are
/// answers to something somebody typed.
#[derive(Debug)]
pub enum BotError {
    /// The command line could not be read.
    ///
    /// Every message names the flag it is about. `docs/CONTAINER_SPEC.md` §9
    /// requires it: an injected value is never guessed, so a missing one, an
    /// unparseable one, and one stated twice are each a startup failure rather
    /// than a default quietly chosen on the operator's behalf.
    Usage {
        /// What is wrong, naming the flag.
        problem: String,
    },
    /// `--package` named something the registry does not have.
    UnknownPackage {
        /// The name that was asked for.
        name: String,
    },
    /// A model package refused.
    Package(hexo_model::PackageError),
    /// The record format refused.
    Record(hexo_records::RecordError),
    /// The filesystem refused an operation.
    Io {
        /// The file or directory it was refused on.
        path: PathBuf,
        /// What the filesystem said.
        source: std::io::Error,
    },
    /// A JSON document this crate wrote could not be read back as JSON.
    Json {
        /// The document.
        path: PathBuf,
        /// What the deserialiser said, with its line and column.
        source: serde_json::Error,
    },
    /// A run manifest is JSON, but not a run manifest this build wrote.
    ///
    /// Separate from [`BotError::Json`] because a field that is missing or of
    /// the wrong type is a file from another build rather than a corrupt one,
    /// and the two want different answers from an operator.
    RunManifest {
        /// The manifest.
        path: PathBuf,
        /// Which field, and what is wrong with it.
        problem: String,
    },
    /// The run directory already holds a run, and `--resume` was not given.
    RunExists {
        /// The run directory.
        path: PathBuf,
    },
    /// `init` was pointed at a checkpoint directory that already exists.
    CheckpointExists {
        /// The destination that was deliberately left untouched.
        path: PathBuf,
    },
    /// `--resume` was given for a run directory that does not exist.
    NoRun {
        /// Where the run was looked for.
        path: PathBuf,
    },
    /// A resumed run's manifest disagrees with the flags it is being resumed
    /// with.
    ///
    /// Every field but `epochs` has to match exactly, and `epochs` may only
    /// grow: a run whose games, batch, package, or ply cap changed halfway is
    /// two runs sharing a directory, and the checkpoints it produces afterwards
    /// would carry no record of which half made them.
    ResumeMismatch {
        /// The manifest field that disagrees.
        field: String,
        /// What the run manifest states.
        recorded: String,
        /// What the flags say now.
        given: String,
    },
    /// A checkpoint directory exists but this build cannot load it.
    ///
    /// Deliberately not repaired by deleting it. A checkpoint is placed by
    /// renaming a finished directory into position, so one that is present and
    /// unloadable was not left half-written by this code, and throwing away
    /// weights nobody asked to throw away is not a decision a resume gets to
    /// make.
    UnloadableCheckpoint {
        /// The checkpoint directory.
        path: PathBuf,
        /// Why it would not load.
        source: hexo_model::PackageError,
    },
    /// A worker, the batcher, or the record writer panicked.
    ///
    /// The panic message itself has already gone to stderr through the panic
    /// hook; this is the sweep reporting that it cannot describe its own result.
    ThreadPanicked {
        /// Which thread, as a noun phrase.
        what: &'static str,
    },
    /// The signal handler could not be installed.
    Signal(ctrlc::Error),
}

impl BotError {
    /// A filesystem refusal, with the path it happened on.
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }

    /// A command line that could not be read, in words that name the flag.
    pub(crate) fn usage(problem: impl Into<String>) -> Self {
        Self::Usage {
            problem: problem.into(),
        }
    }
}

impl From<hexo_model::PackageError> for BotError {
    fn from(source: hexo_model::PackageError) -> Self {
        Self::Package(source)
    }
}

impl From<hexo_records::RecordError> for BotError {
    fn from(source: hexo_records::RecordError) -> Self {
        Self::Record(source)
    }
}

impl core::fmt::Display for BotError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Usage { problem } => write!(f, "{problem}"),
            Self::UnknownPackage { name } => write!(
                f,
                "no model package is registered as {name:?}; this build has {:?}",
                registry::PACKAGES,
            ),
            Self::Package(source) => write!(f, "{source}"),
            Self::Record(source) => write!(f, "{source}"),
            Self::Io { path, source } => write!(f, "{}: {source}", path.display()),
            Self::Json { path, source } => write!(f, "{}: {source}", path.display()),
            Self::RunManifest { path, problem } => write!(f, "{}: {problem}", path.display()),
            Self::RunExists { path } => write!(
                f,
                "{} already holds a run; pass --resume to continue it, or choose another --run-id",
                path.display(),
            ),
            Self::CheckpointExists { path } => write!(
                f,
                "{} already exists; checkpoint init never overwrites an artefact",
                path.display(),
            ),
            Self::NoRun { path } => write!(
                f,
                "--resume was given, but there is no run at {}",
                path.display(),
            ),
            Self::ResumeMismatch {
                field,
                recorded,
                given,
            } => write!(
                f,
                "this run was started with {field} = {recorded}, and is being resumed with \
                 {given}; a resume continues a run, it does not redefine one",
            ),
            Self::UnloadableCheckpoint { path, source } => write!(
                f,
                "the checkpoint at {} does not load: {source}",
                path.display(),
            ),
            Self::ThreadPanicked { what } => write!(
                f,
                "{what} panicked; the sweep cannot report a result it did not finish",
            ),
            Self::Signal(source) => write!(f, "the stop handler could not be installed: {source}"),
        }
    }
}

impl std::error::Error for BotError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Package(source) | Self::UnloadableCheckpoint { source, .. } => Some(source),
            Self::Record(source) => Some(source),
            Self::Io { source, .. } => Some(source),
            Self::Json { source, .. } => Some(source),
            Self::Signal(source) => Some(source),
            _ => None,
        }
    }
}
