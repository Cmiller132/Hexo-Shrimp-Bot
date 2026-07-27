//! Whole training runs, driven in-process: the loop, the artefacts it leaves,
//! and the stop flag.

mod common;

use common::{checkpoint, count, field, metrics, run_root, train_config};
use hexo_bot::Outcome;
use std::sync::atomic::Ordering;

/// The self-play and evaluation games are short on purpose: what these tests
/// check is that the loop ran and that what it wrote can be read back, and a
/// full-length game would only spend minutes proving the engine works.
const PLY_CAP: &str = "21";

#[test]
fn a_policy_run_leaves_a_checkpoint_per_epoch_and_a_metrics_line_per_epoch() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let run_dir = dir.path().to_string_lossy().into_owned();
    let config = train_config(&[
        "train",
        "--run-dir",
        &run_dir,
        "--run-id",
        "policy",
        "--package",
        "mock",
        "--package-config",
        "search=policy",
        "--epochs",
        "2",
        "--games",
        "8",
        "--batch",
        "8",
        "--threads",
        "2",
        "--ply-cap",
        PLY_CAP,
        "--eval-every",
        "1",
        // More evaluation games than lanes, so a lane that finishes one plays
        // the next: the restart path, and the colour flip that goes with it.
        "--eval-games",
        "12",
    ]);

    let outcome = hexo_bot::train(&config).expect("the run completes");
    assert_eq!(outcome, Outcome::Completed);

    let root = run_root(dir.path(), "policy");
    assert!(root.join("manifest.json").is_file(), "the run manifest");

    let lines = metrics(dir.path(), "policy");
    assert_eq!(lines.len(), 2, "one metrics line per epoch");
    for (epoch, line) in lines.iter().enumerate() {
        assert_eq!(count(line, "epoch"), epoch as u64);
        assert_eq!(count(line, "games"), 8, "every epoch plays its whole quota");
        assert!(count(line, "positions") > 0, "the games had plies");
        assert!(count(line, "evaluations") > 0, "the network was asked");
        assert!(count(line, "batches") > 0, "the batcher crossed");
        let results = field(line, "results");
        let decided = count(results, "p0_wins")
            + count(results, "p1_wins")
            + count(results, "draws")
            + count(results, "no_contests");
        assert_eq!(decided, 8, "every game landed in exactly one bucket");
        assert!(
            field(line, "seconds").get("self_play").is_some(),
            "the phase timings are broken out",
        );
        let evals = field(line, "eval")
            .as_array()
            .expect("`eval` is always an array")
            .clone();
        assert!(!evals.is_empty(), "--eval-every 1 runs every epoch");
        for pairing in &evals {
            assert_eq!(count(pairing, "games"), 12);
            assert_eq!(
                count(pairing, "wins")
                    + count(pairing, "losses")
                    + count(pairing, "draws")
                    + count(pairing, "no_contests"),
                12,
                "every evaluation game landed in exactly one bucket",
            );
            assert!(field(pairing, "win_rate").is_number());
        }
    }

    // Epoch 0 plus one per fit, and every one of them proves on the way in.
    for epoch in 0..=2 {
        common::prove(&checkpoint(dir.path(), "policy", epoch), "search=policy");
    }

    // The records of an epoch go once the fit that consumed them has succeeded.
    let records = root.join("records");
    assert!(records.is_dir(), "the records directory itself stays");
    for epoch in 0..2 {
        assert!(
            !records.join(epoch.to_string()).exists(),
            "epoch {epoch}'s records outlived the fit that consumed them",
        );
    }
}

#[test]
fn a_tree_search_run_carries_multi_leaf_sessions_across_the_worker_pool() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let run_dir = dir.path().to_string_lossy().into_owned();
    let config = train_config(&[
        "train",
        "--run-dir",
        &run_dir,
        "--run-id",
        "mcts",
        "--package",
        "mock",
        "--package-config",
        "search=mcts:visits=12,inflight=4,cpuct=1.5",
        "--epochs",
        "2",
        "--games",
        "4",
        "--batch",
        "8",
        "--threads",
        "2",
        "--ply-cap",
        "11",
    ]);

    let outcome = hexo_bot::train(&config).expect("the run completes");
    assert_eq!(outcome, Outcome::Completed);

    let lines = metrics(dir.path(), "mcts");
    assert_eq!(lines.len(), 2);
    for line in &lines {
        assert_eq!(count(line, "games"), 4);
        assert!(
            field(line, "eval")
                .as_array()
                .expect("`eval` is always an array")
                .is_empty(),
            "--eval-every defaults to never",
        );
        // A tree search asks many questions per move, and a policy-only session
        // asks exactly one. This is what says the driver really carried a
        // multi-leaf session rather than degenerating to one leaf per ply.
        assert!(
            count(line, "evaluations") > count(line, "positions"),
            "{} evaluations for {} placements is not a search",
            count(line, "evaluations"),
            count(line, "positions"),
        );
    }

    for epoch in 0..=2 {
        common::prove(
            &checkpoint(dir.path(), "mcts", epoch),
            "search=mcts:visits=12,inflight=4,cpuct=1.5",
        );
    }
}

#[test]
fn a_stop_before_the_first_epoch_writes_nothing_past_the_run_setup() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let run_dir = dir.path().to_string_lossy().into_owned();
    let config = train_config(&[
        "train",
        "--run-dir",
        &run_dir,
        "--run-id",
        "halted",
        "--package",
        "mock",
        "--package-config",
        "search=policy",
        "--epochs",
        "3",
        "--games",
        "4",
        "--batch",
        "4",
        "--threads",
        "2",
        "--ply-cap",
        PLY_CAP,
    ]);

    // The flag the signal handler sets, set by hand. Testing the OS signal would
    // be testing `ctrlc`; what this crate owns is what happens once the flag is
    // true.
    config.stop.store(true, Ordering::Relaxed);

    let outcome = hexo_bot::train(&config).expect("a stopped run is not a failed one");
    assert_eq!(outcome, Outcome::Stopped);

    let root = run_root(dir.path(), "halted");
    assert!(root.join("manifest.json").is_file(), "the run was set up");
    common::prove(&checkpoint(dir.path(), "halted", 0), "search=policy");
    assert!(
        !checkpoint(dir.path(), "halted", 1).exists(),
        "no epoch ran, so no epoch produced a checkpoint",
    );
    assert!(
        !root.join("metrics.jsonl").exists(),
        "no epoch ran, so no epoch had anything to report",
    );
    assert!(
        std::fs::read_dir(root.join("records"))
            .expect("the records directory")
            .next()
            .is_none(),
        "no epoch ran, so no records were produced",
    );
}

#[test]
fn a_stop_during_a_run_leaves_no_epoch_half_written() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let run_dir = dir.path().to_string_lossy().into_owned();
    let config = train_config(&[
        "train",
        "--run-dir",
        &run_dir,
        "--run-id",
        "cut",
        "--package",
        "mock",
        "--package-config",
        "search=policy",
        "--epochs",
        "2",
        "--games",
        "64",
        "--batch",
        "16",
        "--threads",
        "2",
        "--ply-cap",
        "61",
    ]);

    // Set from another thread while the sweep is running, which is the only
    // difference between this and a `SIGTERM`.
    let stop = std::sync::Arc::clone(&config.stop);
    let arm = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(5));
        stop.store(true, Ordering::Relaxed);
    });

    let outcome = hexo_bot::train(&config).expect("a stopped run is not a failed one");
    arm.join().expect("the arming thread");
    assert_eq!(outcome, Outcome::Stopped);

    let root = run_root(dir.path(), "cut");

    // Whether the stop landed mid-self-play or after the fit had begun, the
    // contract is the same from the outside: an epoch's records are either
    // consumed by a fit or abandoned whole, and an abandoned shard was never
    // finalized so nothing of it survives.
    let leftovers: Vec<String> = std::fs::read_dir(root.join("records"))
        .expect("the records directory")
        .filter_map(Result::ok)
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect();
    assert!(
        leftovers.is_empty(),
        "records survived a stop: {leftovers:?}"
    );

    // A stop mid-self-play writes no line at all; a stop after the fit began
    // writes one and finishes the epoch behind it. Either way, every line that
    // is there is an epoch whose checkpoint is there and proves.
    if root.join("metrics.jsonl").exists() {
        for (epoch, _) in metrics(dir.path(), "cut").iter().enumerate() {
            common::prove(
                &checkpoint(dir.path(), "cut", epoch as u32 + 1),
                "search=policy",
            );
        }
    }
}
