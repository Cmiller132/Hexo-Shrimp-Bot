//! The run directory: where everything a run accumulates lives, what the run
//! manifest says, and what `--resume` is allowed to change about it.

use crate::cli::TrainConfig;
use crate::error::BotError;
use hexo_model::{MANIFEST_FILE, Manifest, ModelPackage};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

/// The run manifest's file name.
const RUN_MANIFEST: &str = "manifest.json";

/// Suffix a checkpoint directory carries while it is being written.
const PARTIAL: &str = ".partial";

/// The layout of one run's accumulated state.
///
/// `docs/CONTAINER_SPEC.md` §9:
///
/// ```text
/// <run-dir>/runs/<run-id>/
///   manifest.json              run config and versions
///   metrics.jsonl              one line per epoch
///   checkpoints/<epoch>/       weights and the package's manifest
///   records/<epoch>/           transient; removed after a successful fit
/// ```
#[derive(Clone, Debug)]
pub(crate) struct RunLayout {
    /// `<run-dir>/runs/<run-id>`.
    root: PathBuf,
}

impl RunLayout {
    /// The layout of the run `run_id` under `run_dir`.
    pub(crate) fn new(run_dir: &Path, run_id: &str) -> Self {
        Self {
            root: run_dir.join("runs").join(run_id),
        }
    }

    /// The run's own directory.
    pub(crate) fn root(&self) -> &Path {
        &self.root
    }

    /// The run manifest.
    pub(crate) fn manifest(&self) -> PathBuf {
        self.root.join(RUN_MANIFEST)
    }

    /// The metrics file, one line per epoch.
    pub(crate) fn metrics(&self) -> PathBuf {
        self.root.join("metrics.jsonl")
    }

    /// The directory every checkpoint lands in.
    pub(crate) fn checkpoints(&self) -> PathBuf {
        self.root.join("checkpoints")
    }

    /// One epoch's checkpoint.
    pub(crate) fn checkpoint(&self, epoch: u32) -> PathBuf {
        self.checkpoints().join(epoch.to_string())
    }

    /// Where a checkpoint is written before it is renamed into place.
    ///
    /// §9: a crashed run leaves a partial epoch directory, never a corrupted
    /// checkpoint. The rename is what makes a checkpoint directory's existence
    /// mean the checkpoint is whole.
    fn partial(&self, epoch: u32) -> PathBuf {
        self.checkpoints().join(format!("{epoch}{PARTIAL}"))
    }

    /// The directory every epoch's records land in.
    pub(crate) fn all_records(&self) -> PathBuf {
        self.root.join("records")
    }

    /// One epoch's records.
    pub(crate) fn records(&self, epoch: u32) -> PathBuf {
        self.all_records().join(epoch.to_string())
    }

    /// The one shard an epoch's self-play writes.
    ///
    /// The single writer produces one shard. Its numbered name reserves the
    /// multi-shard naming form.
    pub(crate) fn shard(&self, epoch: u32) -> PathBuf {
        self.records(epoch).join("shard-0000.hxr")
    }

    /// The checkpoint reference written into shard headers and read back out of
    /// them.
    ///
    /// `<run-id>/<epoch>`, which is one of the reference forms
    /// `docs/CONTAINER_SPEC.md` §10 names, rather than an absolute path: a shard
    /// outlives the machine it was written on, and a path that only resolves on
    /// one host is not a reference.
    pub(crate) fn checkpoint_ref(run_id: &str, epoch: u32) -> String {
        format!("{run_id}/{epoch}")
    }
}

/// Write a checkpoint by building it under a temporary name and renaming it in.
///
/// `write` is handed the directory to fill — `ModelPackage::init` or
/// `ModelPackage::fit`. Placing the directory is the container's job and not the
/// package's (`CONTAINER_SPEC.md` §9), which is why neither of those calls does
/// it.
///
/// # Errors
///
/// [`BotError::Io`] if the temporary directory cannot be made, cleared, or
/// renamed, and whatever the package's own write failed with.
pub(crate) fn place_checkpoint<F>(layout: &RunLayout, epoch: u32, write: F) -> Result<(), BotError>
where
    F: FnOnce(&Path) -> Result<Manifest, hexo_model::PackageError>,
{
    let partial = layout.partial(epoch);
    if partial.exists() {
        remove_dir_all(&partial)?;
    }
    create_dir_all(&partial)?;
    write(&partial)?;
    rename(&partial, &layout.checkpoint(epoch))
}

/// Clean incomplete artifacts and return the highest loadable checkpoint epoch.
///
/// Partial and manifest-free checkpoint directories are removed. A placed
/// checkpoint that fails package loading is reported and retained. Record
/// directories at or above the returned epoch are removed before replay.
///
/// # Errors
///
/// [`BotError::Io`] for a directory that cannot be read or removed,
/// [`BotError::UnloadableCheckpoint`] for a checkpoint that will not prove, and
/// [`BotError::NoRun`] if there is no checkpoint at all.
pub(crate) fn resume_point(
    layout: &RunLayout,
    package: &mut dyn ModelPackage,
) -> Result<u32, BotError> {
    let checkpoints = layout.checkpoints();
    let mut highest: Option<u32> = None;
    for entry in read_dir(&checkpoints)? {
        let name = entry.file_name();
        let name = name.to_string_lossy().into_owned();
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        if name.ends_with(PARTIAL) {
            remove_dir_all(&path)?;
            continue;
        }
        let Ok(epoch) = name.parse::<u32>() else {
            continue;
        };
        if !path.join(MANIFEST_FILE).exists() {
            // A manifest-free checkpoint directory is incomplete.
            remove_dir_all(&path)?;
            continue;
        }
        highest = Some(highest.map_or(epoch, |seen: u32| seen.max(epoch)));
    }

    let Some(epoch) = highest else {
        return Err(BotError::NoRun {
            path: layout.root().to_path_buf(),
        });
    };

    // Loading is proving: the epoch this run continues from is the epoch whose
    // weights answered the probe the way their manifest says they should.
    let path = layout.checkpoint(epoch);
    package
        .load(&path)
        .map_err(|source| BotError::UnloadableCheckpoint { path, source })?;

    for entry in read_dir(&layout.all_records())? {
        let name = entry.file_name();
        let Ok(stale) = name.to_string_lossy().parse::<u32>() else {
            continue;
        };
        if stale >= epoch {
            remove_dir_all(&entry.path())?;
        }
    }

    Ok(epoch)
}

/// Write the run manifest.
///
/// Records artifact-affecting run configuration and the four format versions.
/// It has no seed field because §12 uses unrecorded entropy-derived streams.
///
/// `--batch-wait-ms` is omitted because it changes only partial-batch latency.
///
/// # Errors
///
/// [`BotError::Io`] if the file cannot be written.
pub(crate) fn write_manifest(layout: &RunLayout, config: &TrainConfig) -> Result<(), BotError> {
    let path = layout.manifest();
    let mut json = serde_json::to_string_pretty(&manifest_json(config))
        .expect("a run manifest built from `json!` serialises");
    json.push('\n');
    std::fs::write(&path, json).map_err(|source| BotError::io(&path, source))
}

/// The run manifest as JSON.
fn manifest_json(config: &TrainConfig) -> Value {
    json!({
        "run_id": config.run_id,
        "package": config.package,
        "package_config": config.package_config,
        "epochs": config.epochs.get(),
        "games": config.games.get(),
        "batch": config.batch.get(),
        "threads": config.threads.get(),
        "ply_cap": config.ply_cap.get(),
        "eval_every": config.eval_every,
        "eval_games": config.eval_games.get(),
        "rules_version": hexo_engine::RULES_VERSION,
        "action_order_version": hexo_engine::ACTION_ORDER_VERSION,
        "protocol_version": hexo_runner::PROTOCOL_VERSION,
        "records_version": hexo_records::RECORDS_VERSION,
    })
}

/// Hold a resumed run to the flags it was started with.
///
/// Every field must match except `epochs`, which may increase but not decrease.
/// A successful increase is written back to the manifest.
///
/// # Errors
///
/// [`BotError::Io`] or [`BotError::Json`] if the manifest cannot be read,
/// [`BotError::RunManifest`] if it is JSON but not a run manifest this build
/// wrote, and [`BotError::ResumeMismatch`] naming the first field that
/// disagrees.
pub(crate) fn check_resume(layout: &RunLayout, config: &TrainConfig) -> Result<(), BotError> {
    let path = layout.manifest();
    let text = read_to_string(&path)?;
    let recorded: Value = serde_json::from_str(&text).map_err(|source| BotError::Json {
        path: path.clone(),
        source,
    })?;
    let current = manifest_json(config);

    let object = current.as_object().expect("`json!` built an object");
    for (field, given) in object {
        let Some(was) = recorded.get(field) else {
            return Err(BotError::RunManifest {
                path,
                problem: format!("no {field:?} field; this is not a run manifest this build wrote"),
            });
        };
        if field == "epochs" {
            let (was, now) = (as_u64(was, field, &path)?, as_u64(given, field, &path)?);
            if now < was {
                return Err(BotError::ResumeMismatch {
                    field: field.clone(),
                    recorded: was.to_string(),
                    given: now.to_string(),
                });
            }
            continue;
        }
        if was != given {
            return Err(BotError::ResumeMismatch {
                field: field.clone(),
                recorded: was.to_string(),
                given: given.to_string(),
            });
        }
    }

    if recorded.get("epochs") != current.get("epochs") {
        write_manifest(layout, config)?;
    }
    Ok(())
}

/// One JSON field as a number, or the manifest error that says it is not one.
fn as_u64(value: &Value, field: &str, path: &Path) -> Result<u64, BotError> {
    value.as_u64().ok_or_else(|| BotError::RunManifest {
        path: path.to_path_buf(),
        problem: format!("{field:?} is {value}, which is not a count"),
    })
}

/// `std::fs::create_dir_all`, with the path in the error.
pub(crate) fn create_dir_all(path: &Path) -> Result<(), BotError> {
    std::fs::create_dir_all(path).map_err(|source| BotError::io(path, source))
}

/// `std::fs::remove_dir_all`, with the path in the error.
pub(crate) fn remove_dir_all(path: &Path) -> Result<(), BotError> {
    std::fs::remove_dir_all(path).map_err(|source| BotError::io(path, source))
}

/// `std::fs::rename`, with the destination in the error.
fn rename(from: &Path, to: &Path) -> Result<(), BotError> {
    std::fs::rename(from, to).map_err(|source| BotError::io(to, source))
}

/// `std::fs::read_to_string`, with the path in the error.
fn read_to_string(path: &Path) -> Result<String, BotError> {
    std::fs::read_to_string(path).map_err(|source| BotError::io(path, source))
}

/// Every entry of a directory, with the path in the error.
fn read_dir(path: &Path) -> Result<Vec<std::fs::DirEntry>, BotError> {
    let entries = std::fs::read_dir(path).map_err(|source| BotError::io(path, source))?;
    entries
        .map(|entry| entry.map_err(|source| BotError::io(path, source)))
        .collect()
}
