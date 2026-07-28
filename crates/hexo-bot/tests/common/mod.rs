//! Shared scaffolding: parsing a command line the way the binary does, and
//! reading back what a run left on disk.
//!
//! The tests drive the library entry points in-process rather than spawning the
//! binary, which is why `main.rs` is thin: what is worth testing is the loop,
//! and a child process would only add a way for a failure to become a string.

#![allow(dead_code)]

use hexo_bot::registry::PackageRegistry;
use hexo_bot::{Command, InitConfig, MatchConfig, TrainConfig};
use hexo_model::{ModelPackage, PackageError};
use hexo_model_mock::MockPackage;
use serde_json::Value;
use std::path::{Path, PathBuf};

/// Parse a `train` command line, or say which flag it fell over on.
pub fn train_config(args: &[&str]) -> TrainConfig {
    match hexo_bot::parse(args.iter().copied()) {
        Ok(Command::Train(config)) => config,
        Ok(Command::Init(_)) => panic!("these flags parsed as checkpoint init"),
        Ok(Command::Match(_)) => panic!("these flags parsed as a match"),
        Err(error) => panic!("these flags do not parse: {error}"),
    }
}

/// Parse an `init` command line, or say which flag it fell over on.
pub fn init_config(args: &[&str]) -> InitConfig {
    match hexo_bot::parse(args.iter().copied()) {
        Ok(Command::Init(config)) => config,
        Ok(Command::Train(_)) => panic!("these flags parsed as a train run"),
        Ok(Command::Match(_)) => panic!("these flags parsed as a match"),
        Err(error) => panic!("these flags do not parse: {error}"),
    }
}

/// Parse a `match` command line, or say which flag it fell over on.
pub fn match_config(args: &[&str]) -> MatchConfig {
    match hexo_bot::parse(args.iter().copied()) {
        Ok(Command::Match(config)) => config,
        Ok(Command::Init(_)) => panic!("these flags parsed as checkpoint init"),
        Ok(Command::Train(_)) => panic!("these flags parsed as a train run"),
        Err(error) => panic!("these flags do not parse: {error}"),
    }
}

/// A Python-free registry for tests that load only the mock package.
pub fn registry() -> PackageRegistry {
    PackageRegistry::without_mantisnet_runtime()
}

/// Where a run's accumulated state lives.
pub fn run_root(run_dir: &Path, run_id: &str) -> PathBuf {
    run_dir.join("runs").join(run_id)
}

/// One epoch's checkpoint directory.
pub fn checkpoint(run_dir: &Path, run_id: &str, epoch: u32) -> PathBuf {
    run_root(run_dir, run_id)
        .join("checkpoints")
        .join(epoch.to_string())
}

/// Every line of `metrics.jsonl`, parsed.
pub fn metrics(run_dir: &Path, run_id: &str) -> Vec<Value> {
    let path = run_root(run_dir, run_id).join("metrics.jsonl");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    text.lines()
        .map(|line| serde_json::from_str(line).expect("every metrics line is JSON"))
        .collect()
}

/// Write an epoch-0 checkpoint the way a run's first step does, so a `match`
/// has weights to load.
pub fn init_checkpoint(dir: &Path, config: &str) -> Result<(), PackageError> {
    let package = MockPackage::from_config(config)?;
    package.init(dir)?;
    Ok(())
}

/// Prove a checkpoint the way the loop does: read its manifest, then load it.
pub fn prove(dir: &Path, config: &str) {
    hexo_model::Manifest::read(dir)
        .unwrap_or_else(|error| panic!("{}: the manifest does not read: {error}", dir.display()));
    let mut package = MockPackage::from_config(config).expect("the mock takes this config");
    package
        .load(dir)
        .unwrap_or_else(|error| panic!("{}: the checkpoint does not load: {error}", dir.display()));
}

/// One field of a JSON object, or a message naming what was missing.
pub fn field<'a>(value: &'a Value, name: &str) -> &'a Value {
    value
        .get(name)
        .unwrap_or_else(|| panic!("no {name:?} field in {value}"))
}

/// One field of a JSON object, as a count.
pub fn count(value: &Value, name: &str) -> u64 {
    field(value, name)
        .as_u64()
        .unwrap_or_else(|| panic!("{name:?} is not a count in {value}"))
}
