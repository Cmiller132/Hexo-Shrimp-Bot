//! `--resume`: what it continues, what it refuses, and what it must not redo.

mod common;

use common::{checkpoint, metrics, registry, run_root, train_config};
use hexo_bot::{BotError, Outcome};
use std::path::Path;

/// The flags a resumed run has to be resumed with, with `epochs` left open.
fn flags<'a>(run_dir: &'a str, epochs: &'a str, games: &'a str) -> Vec<&'a str> {
    vec![
        "train",
        "--run-dir",
        run_dir,
        "--run-id",
        "r",
        "--package",
        "mock",
        "--package-config",
        "search=policy",
        "--epochs",
        epochs,
        "--games",
        games,
        "--batch",
        "4",
        "--threads",
        "2",
        "--ply-cap",
        "15",
    ]
}

/// The bytes of one checkpoint's manifest, so a resume that re-fit an epoch it
/// should have kept is visible rather than inferred.
fn manifest_bytes(dir: &Path) -> Vec<u8> {
    std::fs::read(dir.join("manifest.json")).expect("a checkpoint has a manifest")
}

#[test]
fn resuming_extends_a_run_without_refitting_the_epochs_it_already_has() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let run_dir = dir.path().to_string_lossy().into_owned();

    let first = train_config(&flags(&run_dir, "1", "4"));
    assert_eq!(
        hexo_bot::train(&first, &registry()).expect("the first run completes"),
        Outcome::Completed,
    );
    let epoch_one = manifest_bytes(&checkpoint(dir.path(), "r", 1));
    assert_eq!(metrics(dir.path(), "r").len(), 1);

    let mut extended = flags(&run_dir, "2", "4");
    extended.push("--resume");
    let second = train_config(&extended);
    assert_eq!(
        hexo_bot::train(&second, &registry()).expect("the resumed run completes"),
        Outcome::Completed,
    );

    assert_eq!(
        manifest_bytes(&checkpoint(dir.path(), "r", 1)),
        epoch_one,
        "the resume re-fit an epoch it already had",
    );
    common::prove(&checkpoint(dir.path(), "r", 2), "search=policy");
    assert_eq!(
        metrics(dir.path(), "r").len(),
        2,
        "the resumed epoch appended its own line",
    );

    // The extension is written back, so the manifest states the run's current
    // target rather than the one it was started with.
    let manifest = std::fs::read_to_string(run_root(dir.path(), "r").join("manifest.json"))
        .expect("the run manifest");
    let manifest: serde_json::Value = serde_json::from_str(&manifest).expect("it is JSON");
    assert_eq!(common::count(&manifest, "epochs"), 2);
}

#[test]
fn resuming_with_a_changed_flag_refuses_and_names_the_field() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let run_dir = dir.path().to_string_lossy().into_owned();

    let first = train_config(&flags(&run_dir, "1", "4"));
    hexo_bot::train(&first, &registry()).expect("the first run completes");

    let mut changed = flags(&run_dir, "1", "8");
    changed.push("--resume");
    let error =
        hexo_bot::train(&train_config(&changed), &registry()).expect_err("the run was redefined");
    match &error {
        BotError::ResumeMismatch { field, .. } => assert_eq!(field, "games"),
        other => panic!("expected a resume mismatch, got {other}"),
    }
    assert!(
        error.to_string().contains("games"),
        "the message names the field: {error}",
    );
}

#[test]
fn resuming_may_not_shorten_a_run() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let run_dir = dir.path().to_string_lossy().into_owned();

    hexo_bot::train(&train_config(&flags(&run_dir, "2", "4")), &registry())
        .expect("the first run completes");

    let mut shorter = flags(&run_dir, "1", "4");
    shorter.push("--resume");
    let error = hexo_bot::train(&train_config(&shorter), &registry())
        .expect_err("a resume does not shrink");
    match error {
        BotError::ResumeMismatch { field, .. } => assert_eq!(field, "epochs"),
        other => panic!("expected a resume mismatch on epochs, got {other}"),
    }
}

#[test]
fn starting_a_run_on_top_of_one_refuses_rather_than_writing_into_it() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let run_dir = dir.path().to_string_lossy().into_owned();

    hexo_bot::train(&train_config(&flags(&run_dir, "1", "4")), &registry())
        .expect("the first run completes");

    let error = hexo_bot::train(&train_config(&flags(&run_dir, "1", "4")), &registry())
        .expect_err("the run exists");
    assert!(
        matches!(error, BotError::RunExists { .. }),
        "expected a refusal to overwrite, got {error}",
    );
    assert!(
        error.to_string().contains("--resume"),
        "the message says what to do instead: {error}",
    );
}

#[test]
fn resuming_a_run_that_does_not_exist_refuses() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let run_dir = dir.path().to_string_lossy().into_owned();

    let mut absent = flags(&run_dir, "1", "4");
    absent.push("--resume");
    let error = hexo_bot::train(&train_config(&absent), &registry())
        .expect_err("there is nothing to resume");
    assert!(
        matches!(error, BotError::NoRun { .. }),
        "expected a refusal to resume nothing, got {error}",
    );
}
