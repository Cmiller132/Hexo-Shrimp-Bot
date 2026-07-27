//! Writing a checkpoint, proving one on the way in, and what a package refuses
//! to answer before it has one.

mod common;

use common::{answers, loaded};
use hexo_engine::Position;
use hexo_model::{MANIFEST_FILE, Manifest, ModelPackage, PackageError, probe_positions};
use hexo_model_mock::{ENCODER_VERSION, MockPackage};
use std::path::Path;

/// The mock's weight file, by the name its README states.
const WEIGHTS: &str = "weights.mock";

/// A configuration that constructs, for tests that are not about configuration.
const CONFIG: &str = "search=mcts:visits=8,inflight=2,cpuct=1.4";

/// Flip one bit of the weight file, leaving its length alone.
fn flip_a_weight_bit(dir: &Path) {
    let path = dir.join(WEIGHTS);
    let mut bytes = std::fs::read(&path).expect("the weights are there");
    bytes[3] ^= 0x01;
    std::fs::write(&path, bytes).expect("the weights are writable");
}

#[test]
fn init_writes_the_two_files_a_checkpoint_is() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let package = MockPackage::from_config(CONFIG).expect("a shape");
    let manifest = package.init(dir.path()).expect("initialised");

    assert!(dir.path().join(WEIGHTS).exists());
    assert!(dir.path().join(MANIFEST_FILE).exists());
    assert_eq!(manifest.package, "mock");
    assert_eq!(manifest.encoder_version, ENCODER_VERSION);
    assert_eq!(manifest.epoch, 0);
    assert_eq!(Manifest::read(dir.path()).expect("readable"), manifest);
}

#[test]
fn init_creates_the_checkpoint_directory_it_was_pointed_at() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let nested = dir
        .path()
        .join("runs")
        .join("demo")
        .join("checkpoints")
        .join("0");
    MockPackage::from_config(CONFIG)
        .expect("a shape")
        .init(&nested)
        .expect("initialised");
    assert!(nested.join(MANIFEST_FILE).exists());
}

#[test]
fn two_fresh_checkpoints_are_byte_identical() {
    // Epoch 0 is a fixed constant, not entropy: a probe hash that moved between
    // two fresh initialisations would be reporting the initialisation rather
    // than the weights.
    let dir = tempfile::tempdir().expect("a scratch directory");
    let one = dir.path().join("one");
    let two = dir.path().join("two");
    let package = MockPackage::from_config(CONFIG).expect("a shape");
    let first = package.init(&one).expect("initialised");
    let second = package.init(&two).expect("initialised");

    assert_eq!(first, second);
    assert_eq!(
        std::fs::read(one.join(WEIGHTS)).expect("read"),
        std::fs::read(two.join(WEIGHTS)).expect("read"),
    );
}

#[test]
fn a_checkpoint_this_package_wrote_loads_and_reports_the_manifest_it_wrote() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let mut package = MockPackage::from_config(CONFIG).expect("a shape");
    let written = package.init(dir.path()).expect("initialised");
    let read = package.load(dir.path()).expect("loaded");
    assert_eq!(read, written);
}

#[test]
fn a_flipped_weight_byte_is_caught_by_the_probe_rather_than_loaded() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let mut package = MockPackage::from_config(CONFIG).expect("a shape");
    let written = package.init(dir.path()).expect("initialised");
    flip_a_weight_bit(dir.path());

    let error = package
        .load(dir.path())
        .expect_err("these are not the weights the manifest describes");
    match error {
        PackageError::ProbeMismatch { expected, computed } => {
            assert_eq!(expected, written.probe_hash);
            assert_ne!(computed, expected);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_refused_load_leaves_the_weights_that_were_already_loaded_alone() {
    // Half a load is worse than none: the process would go on running against
    // weights it can no longer name.
    let dir = tempfile::tempdir().expect("a scratch directory");
    let good = dir.path().join("good");
    let bad = dir.path().join("bad");
    let mut package = MockPackage::from_config(CONFIG).expect("a shape");
    package.init(&good).expect("initialised");
    package.init(&bad).expect("initialised");
    package.load(&good).expect("loaded");
    let before = answers(&package);

    flip_a_weight_bit(&bad);
    package.load(&bad).expect_err("the probe refuses it");
    assert_eq!(answers(&package), before);
}

#[test]
fn a_weight_file_of_the_wrong_length_is_refused_by_length() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let mut package = MockPackage::from_config(CONFIG).expect("a shape");
    package.init(dir.path()).expect("initialised");
    std::fs::write(dir.path().join(WEIGHTS), [1, 2, 3]).expect("writable");

    let error = package
        .load(dir.path())
        .expect_err("three bytes is not a salt");
    match error {
        PackageError::MalformedWeights { path, problem } => {
            assert!(path.ends_with(WEIGHTS), "{path:?}");
            assert!(problem.contains("3 bytes"), "{problem}");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_missing_weight_file_is_an_io_error_naming_it() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let mut package = MockPackage::from_config(CONFIG).expect("a shape");
    package.init(dir.path()).expect("initialised");
    std::fs::remove_file(dir.path().join(WEIGHTS)).expect("removable");

    let error = package.load(dir.path()).expect_err("nothing to load");
    match error {
        PackageError::Io { path, .. } => assert!(path.ends_with(WEIGHTS), "{path:?}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_checkpoint_from_another_package_is_refused_before_its_weights_are_read() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let mut package = MockPackage::from_config(CONFIG).expect("a shape");
    package.init(dir.path()).expect("initialised");
    Manifest::new("gnn", 1, 1, 0, 0)
        .write(dir.path())
        .expect("writable");

    let error = package.load(dir.path()).expect_err("not ours");
    match error {
        PackageError::PackageName { expected, found } => {
            assert_eq!(expected, "mock");
            assert_eq!(found, "gnn");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn nothing_that_needs_weights_answers_before_a_load() {
    let package = MockPackage::from_config(CONFIG).expect("a shape");
    for error in [
        package.evaluator().err(),
        package.self_play_session().err(),
        package.eval_session().err(),
        package.variant_session("policy").err(),
    ] {
        let error = error.expect("nothing has been loaded");
        assert!(
            matches!(error, PackageError::NotLoaded { package: "mock" }),
            "{error:?}"
        );
    }
    // The encoder is a description of a feature layout, not a thing that holds
    // parameters, so it needs no checkpoint.
    let mut bytes = Vec::new();
    package.encoder().encode(&Position::new(), &mut bytes);
    assert_eq!(bytes.len(), 12);
}

#[test]
fn the_evaluator_is_a_pure_function_of_the_salt_and_the_position() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let package = loaded(dir.path(), CONFIG).expect("initialised and loaded");
    assert_eq!(answers(&package), answers(&package));
}

#[test]
fn every_answer_satisfies_the_seams_two_conventions() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let package = loaded(dir.path(), CONFIG).expect("initialised and loaded");
    for (position, evaluation) in probe_positions().iter().zip(answers(&package)) {
        assert_eq!(evaluation.priors.len(), position.legal_count());
        assert!(
            evaluation.value.is_finite() && evaluation.value.abs() < 1.0,
            "value {} is not strictly inside [-1, 1]",
            evaluation.value,
        );
        let total: f32 = evaluation.priors.iter().sum();
        assert!((total - 1.0).abs() < 1e-3, "priors sum to {total}");
        for &prior in &evaluation.priors {
            assert!(prior > 0.0 && prior.is_finite(), "prior {prior}");
        }
    }
}

#[test]
fn two_sessions_from_one_package_do_not_share_a_stream() {
    // `CONTAINER_SPEC.md` §12 leaves seeding to the driver, so the package may
    // construct with any seed — but it may not hand two concurrent sessions the
    // same one, or a driver that forgot to reseed would produce a self-play run
    // of one repeated game.
    let dir = tempfile::tempdir().expect("a scratch directory");
    let package = loaded(dir.path(), "search=policy").expect("initialised and loaded");
    let games = common::self_play_games(&package, 2, 11);
    assert_ne!(
        games[0]
            .plies()
            .iter()
            .map(|p| p.action)
            .collect::<Vec<_>>(),
        games[1]
            .plies()
            .iter()
            .map(|p| p.action)
            .collect::<Vec<_>>(),
    );
}
