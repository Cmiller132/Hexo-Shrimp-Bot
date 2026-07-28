//! `init`: package-owned checkpoint creation and container-owned placement.

mod common;

use common::{init_config, registry};
use hexo_bot::BotError;
use hexo_model::MANIFEST_FILE;

#[test]
fn init_places_one_proved_checkpoint_and_never_overwrites_it() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let checkpoint = dir.path().join("sealed");
    let checkpoint_text = checkpoint.to_string_lossy().into_owned();
    let config = init_config(&[
        "init",
        "--package",
        "mock",
        "--package-config",
        "search=policy",
        "--checkpoint",
        &checkpoint_text,
    ]);

    let manifest =
        hexo_bot::init_checkpoint(&config, &registry()).expect("the checkpoint is placed");
    assert_eq!(manifest.package, "mock");
    assert!(checkpoint.join(MANIFEST_FILE).is_file());
    assert!(!dir.path().join("sealed.incomplete").exists());

    let error = hexo_bot::init_checkpoint(&config, &registry())
        .expect_err("an existing checkpoint is never overwritten");
    assert!(matches!(error, BotError::CheckpointExists { .. }));
}

#[test]
fn an_unknown_package_creates_no_partial_directory() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let checkpoint = dir.path().join("sealed");
    let checkpoint_text = checkpoint.to_string_lossy().into_owned();
    let config = init_config(&[
        "init",
        "--package",
        "absent",
        "--checkpoint",
        &checkpoint_text,
    ]);

    let error =
        hexo_bot::init_checkpoint(&config, &registry()).expect_err("there is no such package");
    assert!(matches!(error, BotError::UnknownPackage { .. }));
    assert!(!checkpoint.exists());
    assert!(!dir.path().join("sealed.incomplete").exists());
}
