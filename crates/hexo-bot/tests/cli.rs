//! The command line: every refusal names the flag it is about.

use hexo_bot::{BotError, Command, Outcome};

#[test]
fn the_exit_codes_are_the_ones_the_spec_pins() {
    // `docs/CONTAINER_SPEC.md` §8.1: 0 ran to completion, 2 stopped by signal
    // after finishing cleanly, 1 failed. A run that ended is not a run that
    // broke, and a `docker stop` produces a 2.
    assert_eq!(Outcome::Completed.exit_code(), 0);
    assert_eq!(Outcome::Stopped.exit_code(), 2);
}

/// Parse and expect a refusal, returning its message.
fn refused(args: &[&str]) -> String {
    match hexo_bot::parse(args.iter().copied()) {
        Ok(_) => panic!("{args:?} parsed, and it should not have"),
        Err(error) => error.to_string(),
    }
}

/// A `train` command line with everything required and nothing else.
fn minimal_train() -> Vec<&'static str> {
    vec![
        "train",
        "--run-dir",
        "d",
        "--run-id",
        "r",
        "--package",
        "mock",
        "--epochs",
        "2",
        "--games",
        "8",
    ]
}

#[test]
fn no_subcommand_says_which_subcommands_there_are() {
    let message = refused(&[]);
    assert!(
        message.contains("train") && message.contains("match"),
        "{message}"
    );
}

#[test]
fn an_unknown_subcommand_says_why_serve_and_play_are_not_among_them() {
    let message = refused(&["serve", "--seat", "x"]);
    assert!(message.contains("serve"), "{message}");
    assert!(
        message.contains("wire protocol"),
        "the refusal says what is missing rather than only that it is: {message}",
    );
}

#[test]
fn every_required_train_flag_is_named_when_it_is_missing() {
    for required in ["--run-dir", "--run-id", "--package", "--epochs", "--games"] {
        let mut args = minimal_train();
        let at = args
            .iter()
            .position(|arg| *arg == required)
            .expect("the minimal line states it");
        args.drain(at..at + 2);
        let message = refused(&args);
        assert!(
            message.contains(required),
            "dropping {required} produced {message}",
        );
    }
}

#[test]
fn a_value_that_is_not_a_number_names_the_flag_and_the_value() {
    let mut args = minimal_train();
    let at = args.iter().position(|arg| *arg == "2").expect("--epochs 2");
    args[at] = "several";
    let message = refused(&args);
    assert!(message.contains("--epochs"), "{message}");
    assert!(message.contains("several"), "{message}");
}

#[test]
fn a_zero_where_a_count_belongs_is_refused() {
    let mut args = minimal_train();
    let at = args.iter().position(|arg| *arg == "8").expect("--games 8");
    args[at] = "0";
    let message = refused(&args);
    assert!(message.contains("--games"), "{message}");
}

#[test]
fn a_flag_stated_twice_is_a_mistake_rather_than_a_last_one_wins() {
    let mut args = minimal_train();
    args.extend(["--games", "16"]);
    let message = refused(&args);
    assert!(message.contains("--games"), "{message}");
    assert!(message.contains("once"), "{message}");
}

#[test]
fn an_unknown_flag_is_named() {
    let mut args = minimal_train();
    args.extend(["--gpus", "2"]);
    let message = refused(&args);
    assert!(message.contains("--gpus"), "{message}");
}

#[test]
fn a_flag_missing_its_value_is_named() {
    let mut args = minimal_train();
    args.push("--batch");
    let message = refused(&args);
    assert!(message.contains("--batch"), "{message}");
}

#[test]
fn a_run_id_that_is_not_one_path_component_is_refused() {
    for bad in ["", "..", "a/b", "a\\b", "a b"] {
        let mut args = minimal_train();
        let at = args.iter().position(|arg| *arg == "r").expect("--run-id r");
        args[at] = bad;
        let message = refused(&args);
        assert!(
            message.contains("--run-id"),
            "run id {bad:?} produced {message}",
        );
    }
}

#[test]
fn the_documented_defaults_are_what_an_unstated_flag_gets() {
    let Ok(Command::Train(config)) = hexo_bot::parse(minimal_train()) else {
        panic!("the minimal train line parses");
    };
    assert_eq!(config.batch.get(), 64);
    assert_eq!(config.ply_cap.get(), 512);
    assert_eq!(config.eval_every, 0, "evaluation is off unless asked for");
    assert_eq!(config.eval_games.get(), 32);
    assert_eq!(config.batch_wait.as_millis(), 2);
    assert!(!config.resume);
    assert_eq!(
        config.package_config, "",
        "an absent package config is empty, not guessed at; the package decides",
    );
}

#[test]
fn a_match_needs_exactly_two_seats() {
    for seats in [0usize, 1, 3] {
        let mut args = vec!["match", "--games", "8"];
        let spec = "package=mock;checkpoint=x";
        for _ in 0..seats {
            args.extend(["--seat", spec]);
        }
        let message = refused(&args);
        assert!(
            message.contains("two seats"),
            "{seats} seats produced {message}",
        );
    }
}

#[test]
fn a_seat_spec_parses_its_four_keys_and_defaults_only_the_optional_ones() {
    let spec = hexo_bot::seat(
        "package=mock;checkpoint=/w/0;config=search=mcts:visits=8,inflight=2,cpuct=1.0;\
         variant=policy",
    )
    .expect("a well-formed seat");
    assert_eq!(spec.package, "mock");
    assert_eq!(spec.checkpoint.to_string_lossy(), "/w/0");
    assert_eq!(spec.config, "search=mcts:visits=8,inflight=2,cpuct=1.0");
    assert_eq!(spec.variant.as_deref(), Some("policy"));

    let bare = hexo_bot::seat("package=mock;checkpoint=/w/0").expect("a minimal seat");
    assert_eq!(bare.config, "", "an absent config is empty, not guessed");
    assert_eq!(
        bare.variant, None,
        "an absent variant is the package's mode"
    );
}

#[test]
fn a_seat_segment_that_is_not_a_pair_is_refused() {
    let message = hexo_bot::seat("package=mock;checkpoint")
        .expect_err("`checkpoint` alone is not a pair")
        .to_string();
    assert!(message.contains("checkpoint"), "{message}");
}

#[test]
fn an_unknown_package_name_lists_the_ones_there_are() {
    let Err(error) = hexo_bot::registry::construct("gnn", "") else {
        panic!("there is no `gnn` package");
    };
    assert!(matches!(error, BotError::UnknownPackage { .. }));
    let message = error.to_string();
    assert!(message.contains("gnn"), "{message}");
    for known in hexo_bot::registry::PACKAGES {
        assert!(message.contains(known), "{message} omits {known}");
    }
}

#[test]
fn an_empty_package_config_is_the_packages_refusal_and_not_the_registrys() {
    // Absence is not a guess: the container hands the string over and the
    // package decides. The mock has one required key and no default shape.
    let Err(error) = hexo_bot::registry::construct("mock", "") else {
        panic!("the mock has no default search shape");
    };
    assert!(matches!(error, BotError::Package(_)), "{error}");
    assert!(error.to_string().contains("search"), "{error}");
}
