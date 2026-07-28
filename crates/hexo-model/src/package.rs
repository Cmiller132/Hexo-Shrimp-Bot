//! The one trait the container knows a model by.

use crate::error::PackageError;
use crate::manifest::Manifest;
use hexo_search::{DecisionSession, Encoder, Evaluator};
use std::path::{Path, PathBuf};

/// A model, as everything above it sees one.
///
/// `docs/CONTAINER_SPEC.md` §5 is the argument; this is the surface. The
/// container's *entire* knowledge of a model is this trait plus a name registry
/// in `hexo-bot`, so adding a GNN package or a CNN package is a new crate and one
/// registry entry, and neither the engine, nor the runner, nor the record format
/// learns anything.
///
/// Object-safe on purpose: the registry holds `Box<dyn ModelPackage>`, which is
/// why the path arguments are `&Path` rather than `impl AsRef<Path>` — a generic
/// method would make the trait unusable as the one thing the container is
/// allowed to know.
///
/// # What the package owns
///
/// Its encoder and encoder version, its evaluator, its two session
/// constructors, its move-selection policies, its diagnostics format, its
/// weight format, and its `fit`. The container has an opinion about none of
/// them. What it does hold every package to is the pair of conventions on
/// [`hexo_search::Evaluation`] — priors in the engine's canonical legal order,
/// value from the side to move — because both are indexed or signed by something
/// a package cannot see from its own side, and a package that gets either wrong
/// trains happily against nonsense.
///
/// # Design notes
///
/// - **The two session constructors are required and have no defaults.** A
///   single constructor taking a `Mode` can be written to ignore it; that
///   compiles, passes, and produces a self-play run in which every game is
///   identical, and no downstream stage can detect it because the data is
///   well-formed. `crates/hexo-player/README.md` argues this at length for
///   `Model`'s two methods and the reasoning transfers unchanged — it is about
///   what a default lets a package *not* decide, not about search.
///
/// - **[`ModelPackage::variant_session`] may default, because its default
///   refuses.** A refusal is not a silent substitution: nothing runs, nothing is
///   recorded, and the operator is told the name they asked for does not exist.
///   That is the line — a default that answers is forbidden, a default that
///   declines is honest.
///
/// - **Loading is proving.** [`ModelPackage::load`] validates the manifest
///   against this build, loads the weights, recomputes the probe hash over what
///   actually answers, and refuses on a mismatch. After it returns `Ok`, the
///   weights that answer are the weights the manifest promised — which is a
///   stronger statement than "a file was read", and it is the only statement
///   worth having, because every way this goes wrong is silent
///   (`docs/CONTAINER_SPEC.md` §10.2).
///
/// - **A fresh evaluator after every load.** The container constructs one by
///   calling [`ModelPackage::evaluator`] again; whether a handle taken before a
///   load stays valid is package-defined and nothing may rely on it either way.
///   Stating it as a rule is what lets a package back its evaluator with a live
///   module whose parameter storage is written in place (§10.1) *or* with an
///   owned snapshot, without the container having to know which.
///
/// - **Sessions come out seeded by the package, and the driver reseeds them.**
///   `docs/CONTAINER_SPEC.md` §12: nothing above `DecisionSession::reseed` exists
///   yet, games are deliberately non-deterministic, and no seed is recorded. A
///   package may therefore construct with any seed it likes; what it may not do
///   is hand two concurrent sessions the same stream, and what the driver must
///   do is reseed from entropy before a run.
pub trait ModelPackage {
    /// The registry name, written into every shard header and every checkpoint
    /// manifest, so no artefact on disk is ambiguous about what produced it.
    fn name(&self) -> &'static str;

    /// The package's own version, bumped when its behaviour changes meaning.
    ///
    /// Separate from the encoder version because a package can change how it
    /// fits, selects, or searches without moving a single byte of its features —
    /// and can change its features without any of the rest moving.
    fn package_version(&self) -> u32;

    /// The version of the bytes [`ModelPackage::encoder`] writes.
    ///
    /// Bumped whenever those bytes change *meaning*, which includes reordering
    /// planes and reinterpreting a field, not only resizing them. Weights are
    /// indexed against a feature layout; a checkpoint whose encoder version
    /// disagrees with this build is refused rather than reinterpreted.
    fn encoder_version(&self) -> u32;

    /// Write an epoch-0 checkpoint — untrained weights and a manifest — into
    /// `dir`.
    ///
    /// The manifest carries the probe hash computed over the fresh weights, so
    /// the checkpoint a run starts from is held to exactly the standard every
    /// later one is: a package whose initialisation is not deterministic has to
    /// hash what it actually wrote, not what it meant to.
    ///
    /// `&self` rather than `&mut self`: initialising is not loading, and a
    /// package that has written epoch 0 still has no weights in hand until it
    /// loads them.
    ///
    /// # Errors
    ///
    /// [`PackageError::Io`] if the directory or its files cannot be written, and
    /// whatever the package's own initialisation can fail with.
    fn init(&self, dir: &Path) -> Result<Manifest, PackageError>;

    /// Load the checkpoint in `dir`, proving it on the way in.
    ///
    /// Read the manifest, validate every version against this build, load the
    /// weights, recompute the probe hash over what they actually answer, and
    /// refuse on any disagreement. A package that refuses must leave whatever it
    /// had loaded before untouched: half a load is worse than none, because the
    /// process goes on running against weights it cannot name.
    ///
    /// # Errors
    ///
    /// [`PackageError::Io`] or [`PackageError::ManifestParse`] if the checkpoint
    /// cannot be read, any of the version variants if it disagrees with this
    /// build, and [`PackageError::ProbeMismatch`] if the weights do not answer
    /// the way the manifest says they do.
    fn load(&mut self, dir: &Path) -> Result<Manifest, PackageError>;

    /// The package's encoder.
    ///
    /// Infallible and available whether or not weights are loaded: an encoder is
    /// a description of a feature layout, not a thing that holds parameters. It
    /// runs worker-side, so a driver hands one to every thread in the pool.
    fn encoder(&self) -> Box<dyn Encoder>;

    /// A handle onto the currently loaded weights.
    ///
    /// The container constructs a fresh evaluator after every load. Whether an
    /// older handle stays valid across one is package-defined and nothing may
    /// rely on it.
    ///
    /// # Errors
    ///
    /// [`PackageError::NotLoaded`] before a successful [`ModelPackage::load`].
    fn evaluator(&self) -> Result<Box<dyn Evaluator>, PackageError>;

    /// A session for a self-play game: the mode whose games become training
    /// data.
    ///
    /// Required, with no default, and paired with [`ModelPackage::eval_session`]
    /// rather than folded into one constructor taking a mode — see the trait's
    /// design notes.
    ///
    /// # Errors
    ///
    /// [`PackageError::NotLoaded`] if the package needs weights it does not
    /// have, and [`PackageError::InvalidConfig`] if its configured search shape
    /// cannot be built.
    fn self_play_session(&self) -> Result<Box<dyn DecisionSession>, PackageError>;

    /// A session for an evaluation game: the mode whose games measure one
    /// checkpoint against another and train nothing.
    ///
    /// # Errors
    ///
    /// As [`ModelPackage::self_play_session`].
    fn eval_session(&self) -> Result<Box<dyn DecisionSession>, PackageError>;

    /// A named session variant beyond the two modes, for search comparisons and
    /// benchmark matches.
    ///
    /// The vocabulary and the syntax are the package's — `"policy"`,
    /// `"mcts:visits=128"`, whatever it defines — because what varies between
    /// two of a package's own sessions is a modelling question that nothing
    /// above it can name. A match harness pits two variants of one package
    /// against each other over the same weights.
    ///
    /// The default refuses every name. That is a default that answers nothing,
    /// which is the only kind this workspace allows: a package with no variants
    /// inherits a loud "no such variant" rather than a session nobody chose.
    ///
    /// # Errors
    ///
    /// [`PackageError::UnknownVariant`] for a name the package does not define,
    /// which is what the default always returns.
    fn variant_session(&self, name: &str) -> Result<Box<dyn DecisionSession>, PackageError> {
        Err(PackageError::UnknownVariant {
            package: self.name(),
            variant: name.to_owned(),
        })
    }

    /// Consume an epoch's record shards and write the next checkpoint.
    ///
    /// `shards` are the files to fit on, `out_dir` is where the new weights and
    /// manifest go, and `epoch` is the epoch of the checkpoint *being written* —
    /// the one after the games in `shards` were played under.
    ///
    /// What fitting means is entirely the package's: the objective, the
    /// optimiser, and the data pipeline are behind this call, and a
    /// Python-backed package crosses its boundary here. Two obligations are
    /// not the package's to choose. It must actually read the shards — a fit
    /// that consumed nothing and produced weights anyway is indistinguishable
    /// downstream from one that worked, which is why
    /// [`PackageError::NoTrainingData`] is a variant of its own — and the
    /// checkpoint it writes must carry the probe hash of the weights it wrote,
    /// so that the load which follows proves them.
    ///
    /// Writing a checkpoint is not loading it. The container loads what `fit`
    /// wrote, through the same [`ModelPackage::load`] as any other checkpoint,
    /// which is what puts the fit's own output behind the probe.
    ///
    /// A package may deliberately decline container-side fitting with
    /// [`PackageError::Unsupported`] when its production training loop remains
    /// elsewhere. That refusal must name the real loop and the owner decision
    /// required to move it; a partial trainer that accepts shards but does not
    /// implement the production algorithm is not an implementation.
    ///
    /// # Errors
    ///
    /// [`PackageError::NoTrainingData`] if there were no shards or no games,
    /// [`PackageError::Io`] for a shard or checkpoint the filesystem refuses,
    /// [`PackageError::Failed`] wrapping whatever the package's own reader or
    /// trainer raised, and [`PackageError::NotLoaded`] if the package fits from
    /// weights it does not have. [`PackageError::Unsupported`] when the package
    /// deliberately keeps fitting in an external production loop.
    fn fit(
        &mut self,
        shards: &[PathBuf],
        out_dir: &Path,
        epoch: u32,
    ) -> Result<Manifest, PackageError>;
}
