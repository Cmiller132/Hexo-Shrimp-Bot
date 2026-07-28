//! `match`: two searches over one set of weights, and what the report says.

mod common;

use common::{count, field, init_checkpoint, match_config, registry};
use hexo_bot::Outcome;
use serde_json::Value;

/// The variant the second seat plays: a search shape in the package's own
/// grammar, which is what a variant name is.
const TREE_SEARCH: &str = "mcts:visits=8,inflight=2,cpuct=1.4";

#[test]
fn two_searches_over_the_same_weights_produce_a_report_that_accounts_for_every_game() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let weights = dir.path().join("epoch-0");
    init_checkpoint(&weights, "search=policy").expect("the mock writes a checkpoint");
    let weights = weights.to_string_lossy().into_owned();
    let report_path = dir.path().join("report.json");

    // Both seats name the *same* checkpoint. That is the subcommand's whole
    // point: same weights, two searches.
    let config = match_config(&[
        "match",
        "--games",
        "8",
        "--batch",
        "8",
        "--threads",
        "2",
        "--ply-cap",
        "15",
        "--report",
        &report_path.to_string_lossy(),
        "--seat",
        &format!("package=mock;checkpoint={weights};config=search=policy"),
        "--seat",
        &format!("package=mock;checkpoint={weights};config=search=policy;variant={TREE_SEARCH}"),
    ]);

    let played = hexo_bot::play_match(&config, &registry()).expect("the match runs");
    assert_eq!(played.outcome, Outcome::Completed);

    let report = played.report.to_json();
    assert_eq!(count(&report, "games"), 8);
    assert!(field(&report, "mean_plies").as_f64().expect("a number") > 0.0);

    let seats = field(&report, "seats")
        .as_array()
        .expect("two seats")
        .clone();
    assert_eq!(seats.len(), 2);
    let wins: u64 = seats.iter().map(|seat| count(seat, "wins")).sum();
    assert_eq!(
        wins + count(&report, "draws") + count(&report, "no_contests"),
        count(&report, "games"),
        "every game landed in exactly one bucket",
    );
    for seat in &seats {
        assert_eq!(field(seat, "package"), "mock");
        assert_eq!(
            count(seat, "wins_as_p0") + count(seat, "wins_as_p1"),
            count(seat, "wins"),
            "a win was won in one colour or the other",
        );
    }
    assert!(
        field(&seats[0], "variant").is_null(),
        "the first seat evaluates"
    );
    assert_eq!(field(&seats[1], "variant"), TREE_SEARCH);

    // `--report` writes the same document the binary prints.
    let written: Value =
        serde_json::from_str(&std::fs::read_to_string(&report_path).expect("the report file"))
            .expect("the report is JSON");
    assert_eq!(written, report);

    // No shards: an evaluation game trains nothing, and nothing consumes a
    // match's records.
    let stray: Vec<_> = std::fs::read_dir(dir.path())
        .expect("the scratch directory")
        .filter_map(Result::ok)
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.ends_with(".hxr"))
        .collect();
    assert!(stray.is_empty(), "a match wrote records: {stray:?}");
}

#[test]
fn a_seat_spec_with_an_unknown_key_refuses_and_names_it() {
    let error = hexo_bot::seat("package=mock;checkpoint=x;seach=search=policy")
        .expect_err("`seach` is not a key");
    let message = error.to_string();
    assert!(
        message.contains("seach"),
        "the message names the key: {message}"
    );
    assert!(
        message.contains("package"),
        "the message lists the keys there are: {message}",
    );
}

#[test]
fn a_seat_spec_missing_its_checkpoint_refuses() {
    let error = hexo_bot::seat("package=mock").expect_err("a seat needs weights");
    assert!(
        error.to_string().contains("checkpoint"),
        "the message names what is missing: {error}",
    );
}

#[test]
fn a_seat_spec_repeating_a_key_refuses_rather_than_keeping_one() {
    let error = hexo_bot::seat("package=mock;package=gnn;checkpoint=x").expect_err("stated twice");
    assert!(
        error.to_string().contains("twice"),
        "the message says what is wrong: {error}",
    );
}

#[test]
fn an_unknown_variant_name_surfaces_the_packages_own_refusal() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let weights = dir.path().join("epoch-0");
    init_checkpoint(&weights, "search=policy").expect("the mock writes a checkpoint");
    let weights = weights.to_string_lossy().into_owned();

    let config = match_config(&[
        "match",
        "--games",
        "2",
        "--threads",
        "1",
        "--ply-cap",
        "7",
        "--seat",
        &format!("package=mock;checkpoint={weights};config=search=policy"),
        "--seat",
        &format!("package=mock;checkpoint={weights};config=search=policy;variant=greedy"),
    ]);

    let error = hexo_bot::play_match(&config, &registry()).expect_err("the mock has no `greedy`");
    let message = error.to_string();
    assert!(
        message.contains("greedy") && message.contains("variant"),
        "the package's own refusal reaches the operator: {message}",
    );
}

#[test]
fn a_seat_naming_no_package_lists_the_packages_there_are() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let weights = dir.path().join("epoch-0");
    init_checkpoint(&weights, "search=policy").expect("the mock writes a checkpoint");
    let weights = weights.to_string_lossy().into_owned();

    let config = match_config(&[
        "match",
        "--games",
        "2",
        "--threads",
        "1",
        "--seat",
        &format!("package=gnn;checkpoint={weights}"),
        "--seat",
        &format!("package=mock;checkpoint={weights};config=search=policy"),
    ]);

    let error = hexo_bot::play_match(&config, &registry()).expect_err("there is no `gnn`");
    let message = error.to_string();
    assert!(
        message.contains("gnn") && message.contains("mock"),
        "the message names what was asked for and what exists: {message}",
    );
}
