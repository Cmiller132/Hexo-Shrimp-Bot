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
/// It implements the complete [`ModelPackage`] surface with deterministic
/// salt-based outputs and no external model runtime.
///
/// Construct with [`MockPackage::from_config`], then load a checkpoint before
/// requesting an evaluator or session.
pub struct MockPackage {
    /// The configured search shape, used by both required session modes.
    search: Search,
    /// The loaded salt, or `None` until checkpoint verification succeeds.
    salt: Option<u64>,
    /// Serial used to derive distinct initial session streams.
    next_serial: Cell<u64>,
}

impl MockPackage {
    /// Read a configuration string.
    ///
    /// The grammar is `search=<shape>`, where a shape is `policy` or
    /// `mcts:visits=N,inflight=N,cpuct=F`. The `search` key and all three `mcts`
    /// parameters are required.
    ///
    /// ```
    /// use hexo_model_mock::MockPackage;
    ///
    /// MockPackage::from_config("search=policy")?;
    /// MockPackage::from_config("search=mcts:visits=64,inflight=8,cpuct=1.5")?;
    ///
    /// // Missing, unknown, and incomplete configurations are rejected.
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

    /// Write weights followed by their manifest and probe hash.
    ///
    /// The caller owns atomic placement of the complete checkpoint directory.
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

        // Verify the evaluator's output hash, not only the weight-file bytes.
        let computed = probe_hash(&MockEncoder, &mut MockEvaluator::new(salt));
        if computed != manifest.probe_hash {
            // Preserve the prior loaded state when verification fails.
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

    /// A variant name uses the same search-shape grammar as the `search=` value.
    ///
    /// Variants use evaluation selection and emit no diagnostics.
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
                // Replay verification checks engine semantics beyond wire
                // decoding.
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
