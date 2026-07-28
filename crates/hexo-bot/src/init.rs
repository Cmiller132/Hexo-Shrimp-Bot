//! Atomic creation of one package checkpoint outside a training run.

use crate::cli::InitConfig;
use crate::error::BotError;
use crate::registry::PackageRegistry;
use hexo_model::Manifest;
use std::path::Path;

/// Ask a package to write epoch zero, then atomically place the directory.
///
/// MantisNet uses this to seal a Python training `.pt` with the package
/// metadata and probe hash a container load requires. It is a checkpoint
/// operation, not a game mode: no runner, session, or evaluator loop is
/// constructed by the container.
pub fn init_checkpoint(
    config: &InitConfig,
    registry: &PackageRegistry,
) -> Result<Manifest, BotError> {
    let package = registry.construct(&config.package, &config.package_config)?;
    let destination = &config.checkpoint;
    if destination.exists() {
        return Err(BotError::CheckpointExists {
            path: destination.clone(),
        });
    }

    let name = destination.file_name().ok_or_else(|| {
        BotError::usage(format!(
            "--checkpoint {} has no final path component",
            destination.display()
        ))
    })?;
    let parent = destination.parent().ok_or_else(|| {
        BotError::usage(format!(
            "--checkpoint {} has no parent directory",
            destination.display()
        ))
    })?;
    std::fs::create_dir_all(parent).map_err(|source| BotError::io(parent, source))?;

    let mut partial_name = name.to_os_string();
    partial_name.push(".incomplete");
    let partial = parent.join(partial_name);
    clear_incomplete(&partial)?;
    std::fs::create_dir(&partial).map_err(|source| BotError::io(&partial, source))?;

    let manifest = package.init(&partial)?;
    std::fs::rename(&partial, destination).map_err(|source| BotError::io(destination, source))?;
    Ok(manifest)
}

/// Remove only the temporary directory this command itself names.
fn clear_incomplete(partial: &Path) -> Result<(), BotError> {
    if !partial.exists() {
        return Ok(());
    }
    std::fs::remove_dir_all(partial).map_err(|source| BotError::io(partial, source))
}
