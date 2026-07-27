//! The package itself: configuration in, checkpoints and sessions out.

use crate::config::{self, Search};
use crate::seam::{MockEncoder, MockEvaluator};
use crate::select::{EvalPolicy, EvalSearch, SelfPlayPolicy, SelfPlaySearch, session_seed};
use crate::weights;
use crate::{ENCODER_VERSION, NAME, PACKAGE_VERSION};
use hexo_model::{Manifest, ModelPackage, PackageError, probe_hash};
use hexo_records::{ShardReader, verify};
use hexo_search::{DecisionSession, Encoder, Evaluator, MctsSession, PolicySession};
use std::cell::Cell;
use std::path::{Path, PathBuf};

/// The mock model package.
///
/// Deterministic, weightless, and complete: it exercises the encoder, the
/// evaluator, both session kinds, both selection policies, the diagnostics
/// channel, checkpoint write and load, the probe hash, and `fit`, with no
/// network, no GPU, and no Python. `docs/CONTAINER_SPEC.md` §5 argues why that
/// makes it a permanent part of the workspace rather than a placeholder.
///
/// Constructed from a configuration string and nothing else — see
/// [`MockPackage::from_config`] for the syntax — and then loaded from a
/// checkpoint before it can answer anything.
pub struct MockPackage {
    /// The configured search shape, used by both required session modes.
    search: Search,
    /// The loaded salt, or `None` until a checkpoint has been proved.
    salt: Option<u64>,
    /// Serial for the next session's seed, so two sessions from one package
    /// never share a stream.
    next_serial: Cell<u64>,
}

impl MockPackage {
    /// Read a configuration string.
    ///
    /// The grammar is `search=<shape>`, where a shape is `policy` or
    /// `mcts:visits=N,inflight=N,cpuct=F`. Every part of it is required: the
    /// `search` key has no default because a search shape is a model choice, and
    /// the three `mcts` parameters have none for the reason
    /// `hexo_search::MctsConfig` has no `Default`.
    ///
    /// ```
    /// use hexo_model_mock::MockPackage;
    ///
    /// MockPackage::from_config("search=policy")?;
    /// MockPackage::from_config("search=mcts:visits=64,inflight=8,cpuct=1.5")?;
    ///
    /// // Nothing is guessed at: no key, an unknown key, and a half-stated
    /// // search shape are all refused by name.
    /// assert!(MockPackage::from_config("").is_err());
    /// assert!(MockPackage::from_config("seach=policy").is_err());
    /// assert!(MockPackage::from_config("search=mcts:visits=64").is_err());
    /// # Ok::<(), hexo_model::PackageError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`PackageError::InvalidConfig`] for a string this package cannot use,
    /// naming the key, the parameter, or the shape that is wrong.
    pub fn from_config(config: &str) -> Result<Self, PackageError> {
        Ok(Self {
            search: config::parse_config(config)?,
            salt: None,
            next_serial: Cell::new(0),
        })
    }

    /// The loaded salt, or the refusal that says nothing has been loaded.
    fn salt(&self) -> Result<u64, PackageError> {
        self.salt.ok_or(PackageError::NotLoaded { package: NAME })
    }

    /// The seed for the next session, and bump the serial.
    fn seed(&self) -> Result<u64, PackageError> {
        let salt = self.salt()?;
        let serial = self.next_serial.get();
        self.next_serial.set(serial.wrapping_add(1));
        Ok(session_seed(salt, serial))
    }

    /// A session of `search`'s shape, selecting the way a self-play seat does.
    fn playing(&self, search: Search) -> Result<Box<dyn DecisionSession>, PackageError> {
        let seed = self.seed()?;
        Ok(match search {
            Search::Policy => Box::new(PolicySession::new(Box::new(SelfPlayPolicy), seed)),
            Search::Mcts(config) => {
                Box::new(MctsSession::new(config, Box::new(SelfPlaySearch), seed))
            }
        })
    }

    /// A session of `search`'s shape, selecting the way an evaluating seat does.
    fn evaluating(&self, search: Search) -> Result<Box<dyn DecisionSession>, PackageError> {
        let seed = self.seed()?;
        Ok(match search {
            Search::Policy => Box::new(PolicySession::new(Box::new(EvalPolicy), seed)),
            Search::Mcts(config) => Box::new(MctsSession::new(config, Box::new(EvalSearch), seed)),
        })
    }

    /// Write a whole checkpoint — weights, then the manifest carrying the probe
    /// hash of exactly those weights.
    ///
    /// The manifest is written last, so a directory holding one is a directory
    /// whose weights are already there. Placing the directory itself is the
    /// container's: `docs/CONTAINER_SPEC.md` §9 writes a checkpoint under a
    /// temporary name and renames it in, which is a decision about the whole
    /// directory rather than about either file in it.
    fn write_checkpoint(
        &self,
        dir: &Path,
        salt: u64,
        epoch: u32,
    ) -> Result<Manifest, PackageError> {
        std::fs::create_dir_all(dir).map_err(|source| PackageError::Io {
            path: dir.to_path_buf(),
            source,
        })?;
        weights::write(dir, salt)?;
        let hash = probe_hash(&MockEncoder, &mut MockEvaluator::new(salt));
        let manifest = Manifest::new(NAME, PACKAGE_VERSION, ENCODER_VERSION, epoch, hash);
        manifest.write(dir)?;
        Ok(manifest)
    }
}

impl ModelPackage for MockPackage {
    fn name(&self) -> &'static str {
        NAME
    }

    fn package_version(&self) -> u32 {
        PACKAGE_VERSION
    }

    fn encoder_version(&self) -> u32 {
        ENCODER_VERSION
    }

    fn init(&self, dir: &Path) -> Result<Manifest, PackageError> {
        self.write_checkpoint(dir, weights::INITIAL_SALT, 0)
    }

    fn load(&mut self, dir: &Path) -> Result<Manifest, PackageError> {
        let manifest = Manifest::read(dir)?;
        manifest.validate(NAME, PACKAGE_VERSION, ENCODER_VERSION)?;
        let salt = weights::read(dir)?;

        // Loading is proving. The hash is recomputed over what the weights just
        // read actually answer, not over the file, so a flipped byte anywhere in
        // it is caught here rather than by a training run that never converges.
        let computed = probe_hash(&MockEncoder, &mut MockEvaluator::new(salt));
        if computed != manifest.probe_hash {
            // Deliberately before the assignment: a package that refused a load
            // keeps whatever it had, because half a load leaves the process
            // running against weights it can no longer name.
            return Err(PackageError::ProbeMismatch {
                expected: manifest.probe_hash,
                computed,
            });
        }

        self.salt = Some(salt);
        Ok(manifest)
    }

    fn encoder(&self) -> Box<dyn Encoder> {
        Box::new(MockEncoder)
    }

    fn evaluator(&self) -> Result<Box<dyn Evaluator>, PackageError> {
        Ok(Box::new(MockEvaluator::new(self.salt()?)))
    }

    fn self_play_session(&self) -> Result<Box<dyn DecisionSession>, PackageError> {
        self.playing(self.search)
    }

    fn eval_session(&self) -> Result<Box<dyn DecisionSession>, PackageError> {
        self.evaluating(self.search)
    }

    /// A variant name is a search shape in the same grammar the `search=` value
    /// uses, so `"policy"` and `"mcts:visits=128,inflight=4,cpuct=1.0"` are both
    /// valid names and a match harness can pit them against each other over the
    /// same weights.
    ///
    /// Variants select the way an **evaluating** seat does, not a self-play one.
    /// They exist to compare search shapes and to play benchmark matches, and
    /// both of those want the sharper sampler and no diagnostics; a variant that
    /// selected like a self-play seat would be a third mode wearing the name of a
    /// comparison.
    fn variant_session(&self, name: &str) -> Result<Box<dyn DecisionSession>, PackageError> {
        let search = config::parse_search(name).map_err(|f| f.into_variant_error(name))?;
        self.evaluating(search)
    }

    fn fit(
        &mut self,
        shards: &[PathBuf],
        out_dir: &Path,
        epoch: u32,
    ) -> Result<Manifest, PackageError> {
        let salt = self.salt()?;
        let mut games = 0usize;
        let mut positions = 0usize;
        let mut digest = 0u64;

        for path in shards {
            let reader = ShardReader::open(path)
                .map_err(|e| PackageError::failed(NAME, "opening a record shard", e))?;
            if reader.header().package != NAME {
                return Err(PackageError::PackageName {
                    expected: NAME.to_owned(),
                    found: reader.header().package.clone(),
                });
            }
            for record in reader {
                let record =
                    record.map_err(|e| PackageError::failed(NAME, "reading a record shard", e))?;
                // `verify` replays the move list through the engine, which is the
                // detector parsing cannot be. A fit that trained on a shard whose
                // action ids had drifted would produce weights nothing could
                // explain, and every field in it parses.
                verify(&record)
                    .map_err(|e| PackageError::failed(NAME, "verifying a recorded game", e))?;
                games += 1;
                positions += record.plies.len();
                let tail = record.plies.last().map_or(0, |ply| ply.zobrist_after);
                digest = weights::mix(digest ^ tail ^ (record.plies.len() as u64).rotate_left(32));
            }
        }

        if games == 0 {
            return Err(PackageError::NoTrainingData {
                package: NAME,
                shards: shards.len(),
                games,
            });
        }

        digest = weights::mix(digest ^ (games as u64) ^ (positions as u64).rotate_left(32));
        self.write_checkpoint(out_dir, weights::next_salt(salt, epoch, digest), epoch)
    }
}
