//! The whole loop the container will run, with nothing above it: play games with
//! the package's own sessions, write a shard, fit on it, and load what the fit
//! wrote.

mod common;

use common::{answers, decode_diagnostics, loaded, play, self_play_games, short_game, write_shard};
use hexo_model::{ModelPackage, PackageError};
use hexo_model_mock::MockPackage;
use hexo_runner::MatchResult;
use std::path::PathBuf;

/// A tree search small enough for a test and big enough to have a distribution.
const MCTS: &str = "search=mcts:visits=8,inflight=2,cpuct=1.4";

/// The visits a decision under [`MCTS`] spends below the root.
const BUDGET: u32 = 8;

/// Diagnostics kind bytes, as the package's README states them.
const TAG_VISITS: u8 = 0;
/// See [`TAG_VISITS`].
const TAG_PRIORS: u8 = 1;

#[test]
fn a_game_of_mcts_self_play_finishes_and_every_ply_carries_its_visit_table() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let package = loaded(dir.path(), MCTS).expect("initialised and loaded");
    let game = self_play_games(&package, 1, 21).pop().expect("one game");

    let result = game.result().expect("the game finished");
    assert!(result.is_contested(), "{result:?}");
    assert!(!game.plies().is_empty());

    for (ply, record) in game.plies().iter().enumerate() {
        let bytes = record
            .diagnostics
            .as_ref()
            .unwrap_or_else(|| panic!("ply {ply} carries no annotations"));
        let table = decode_diagnostics(bytes);
        assert_eq!(table.tag, TAG_VISITS, "ply {ply}");
        assert!(!table.entries.is_empty(), "ply {ply}");
        let spent: u32 = table.entries.iter().map(|entry| entry.visits).sum();
        assert_eq!(
            spent, BUDGET,
            "ply {ply}: the budget is spent below the root"
        );
        assert!(
            table
                .entries
                .iter()
                .any(|entry| entry.action == record.action),
            "ply {ply}: the placement is not one of the children the table lists",
        );
    }
}

#[test]
fn a_game_of_policy_self_play_carries_its_prior_table_instead() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let package = loaded(dir.path(), "search=policy").expect("initialised and loaded");
    let game = self_play_games(&package, 1, 11).pop().expect("one game");

    for (ply, record) in game.plies().iter().enumerate() {
        let table = decode_diagnostics(record.diagnostics.as_ref().expect("annotations"));
        assert_eq!(table.tag, TAG_PRIORS, "ply {ply}");
        let total: f32 = table.entries.iter().map(|entry| entry.prior).sum();
        assert!(
            (total - 1.0).abs() < 1e-3,
            "ply {ply}: priors sum to {total}"
        );
    }
}

#[test]
fn an_eval_game_records_nothing_because_it_trains_nothing() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let package = loaded(dir.path(), MCTS).expect("initialised and loaded");
    let encoder = package.encoder();
    let mut evaluator = package.evaluator().expect("loaded");
    let mut seats = [
        package.eval_session().expect("loaded"),
        package.eval_session().expect("loaded"),
    ];
    let game = play(
        encoder.as_ref(),
        evaluator.as_mut(),
        &mut seats,
        short_game(11),
    );

    assert!(game.result().is_some());
    assert!(game.plies().iter().all(|ply| ply.diagnostics.is_none()));
}

#[test]
fn two_variants_of_one_package_play_each_other_over_the_same_weights() {
    // What variants are for: a benchmark match between two search shapes, with
    // nothing else different between the seats.
    let dir = tempfile::tempdir().expect("a scratch directory");
    let package = loaded(dir.path(), "search=policy").expect("initialised and loaded");
    let encoder = package.encoder();
    let mut evaluator = package.evaluator().expect("loaded");
    let mut seats = [
        package.variant_session("policy").expect("a shape"),
        package
            .variant_session("mcts:visits=8,inflight=2,cpuct=1.0")
            .expect("a shape"),
    ];
    let game = play(
        encoder.as_ref(),
        evaluator.as_mut(),
        &mut seats,
        short_game(11),
    );

    assert!(matches!(
        game.result().expect("the game finished"),
        MatchResult::Drawn { .. } | MatchResult::Decisive { .. }
    ));
    assert!(game.plies().iter().all(|ply| ply.diagnostics.is_none()));
}

#[test]
fn fit_writes_a_checkpoint_that_loads_and_whose_weights_actually_moved() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let epoch0 = dir.path().join("epoch-0");
    let epoch1 = dir.path().join("epoch-1");

    let mut package = MockPackage::from_config(MCTS).expect("a shape");
    let initial = package.init(&epoch0).expect("initialised");
    package.load(&epoch0).expect("loaded");
    let before = answers(&package);

    let games = self_play_games(&package, 2, 11);
    let shard = write_shard(&dir.path().join("epoch-0.hxr"), 0, &games);

    let fitted = package.fit(&[shard], &epoch1, 1).expect("fitted");
    assert_eq!(fitted.epoch, 1);
    assert_eq!(fitted.package, "mock");
    assert_ne!(fitted.probe_hash, initial.probe_hash);
    assert_ne!(
        std::fs::read(epoch0.join("weights.mock")).expect("read"),
        std::fs::read(epoch1.join("weights.mock")).expect("read"),
    );

    // The fit did not load what it wrote — the container does, through the same
    // load as any other checkpoint, which is what puts the fit's own output
    // behind the probe.
    assert_eq!(answers(&package), before);
    let loaded_back = package.load(&epoch1).expect("the new checkpoint proves");
    assert_eq!(loaded_back, fitted);
    assert_ne!(answers(&package), before);
}

#[test]
fn fit_on_the_same_shards_twice_writes_the_same_weights() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let package = loaded(&dir.path().join("epoch-0"), MCTS).expect("initialised and loaded");
    let games = self_play_games(&package, 1, 11);
    let shard = write_shard(&dir.path().join("epoch-0.hxr"), 0, &games);

    let mut one = loaded(&dir.path().join("a"), MCTS).expect("initialised and loaded");
    let mut two = loaded(&dir.path().join("b"), MCTS).expect("initialised and loaded");
    let first = one
        .fit(std::slice::from_ref(&shard), &dir.path().join("a1"), 1)
        .expect("fitted");
    let second = two
        .fit(std::slice::from_ref(&shard), &dir.path().join("b1"), 1)
        .expect("fitted");
    assert_eq!(first.probe_hash, second.probe_hash);
}

#[test]
fn fit_reads_every_shard_it_is_handed() {
    // The weights are a function of the games read, so a fit that quietly
    // stopped after the first shard would write different weights than one that
    // read both — which is a thing a test can say, unlike "it opened a file".
    let dir = tempfile::tempdir().expect("a scratch directory");
    let package = loaded(&dir.path().join("epoch-0"), MCTS).expect("initialised and loaded");
    let first = write_shard(
        &dir.path().join("first.hxr"),
        0,
        &self_play_games(&package, 1, 11),
    );
    let second = write_shard(
        &dir.path().join("second.hxr"),
        0,
        &self_play_games(&package, 1, 11),
    );

    let mut one = loaded(&dir.path().join("a"), MCTS).expect("initialised and loaded");
    let mut two = loaded(&dir.path().join("b"), MCTS).expect("initialised and loaded");
    let partial = one
        .fit(std::slice::from_ref(&first), &dir.path().join("a1"), 1)
        .expect("fitted");
    let whole = two
        .fit(&[first, second], &dir.path().join("b1"), 1)
        .expect("fitted");
    assert_ne!(partial.probe_hash, whole.probe_hash);
}

#[test]
fn fit_refuses_an_empty_shard_list() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let mut package = loaded(&dir.path().join("epoch-0"), MCTS).expect("initialised and loaded");
    let error = package
        .fit(&[], &dir.path().join("epoch-1"), 1)
        .expect_err("nothing to fit on");
    assert!(
        matches!(
            error,
            PackageError::NoTrainingData {
                package: "mock",
                shards: 0,
                games: 0
            }
        ),
        "{error:?}"
    );
    assert!(
        !dir.path().join("epoch-1").exists(),
        "no checkpoint was written"
    );
}

#[test]
fn fit_refuses_a_shard_holding_no_games() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let mut package = loaded(&dir.path().join("epoch-0"), MCTS).expect("initialised and loaded");
    let empty = write_shard(&dir.path().join("empty.hxr"), 0, &[]);

    let error = package
        .fit(&[empty], &dir.path().join("epoch-1"), 1)
        .expect_err("a shard of no games is no training data");
    assert!(
        matches!(
            error,
            PackageError::NoTrainingData {
                package: "mock",
                shards: 1,
                games: 0
            }
        ),
        "{error:?}"
    );
}

#[test]
fn fit_refuses_a_shard_another_package_wrote() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let mut package = loaded(&dir.path().join("epoch-0"), MCTS).expect("initialised and loaded");
    let games = self_play_games(&package, 1, 11);
    let foreign = common::write_shard_for(&dir.path().join("gnn.hxr"), 0, &games, "gnn");

    let error = package
        .fit(&[foreign], &dir.path().join("epoch-1"), 1)
        .expect_err("those are somebody else's games");
    match error {
        PackageError::PackageName { expected, found } => {
            assert_eq!(expected, "mock");
            assert_eq!(found, "gnn");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn fit_refuses_a_path_that_is_not_a_shard_with_the_readers_own_error() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let mut package = loaded(&dir.path().join("epoch-0"), MCTS).expect("initialised and loaded");
    let junk = dir.path().join("junk.hxr");
    std::fs::write(&junk, b"not a shard at all").expect("writable");

    let error = package
        .fit(&[junk], &dir.path().join("epoch-1"), 1)
        .expect_err("not a shard");
    match error {
        // The record error survives whole rather than being flattened to a
        // message: `source` still hands back the `RecordError` that said so.
        PackageError::Failed {
            package,
            doing,
            source,
        } => {
            assert_eq!(package, "mock");
            assert_eq!(doing, "opening a record shard");
            assert!(
                source.downcast_ref::<hexo_records::RecordError>().is_some(),
                "{source}"
            );
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn fit_refuses_before_a_checkpoint_has_been_loaded() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let mut package = MockPackage::from_config(MCTS).expect("a shape");
    let shards: Vec<PathBuf> = Vec::new();
    let error = package
        .fit(&shards, &dir.path().join("epoch-1"), 1)
        .expect_err("there are no weights to fit from");
    assert!(
        matches!(error, PackageError::NotLoaded { package: "mock" }),
        "{error:?}"
    );
}
