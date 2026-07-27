//! The configuration grammar: what it accepts, and what it refuses by name.

mod common;

use common::loaded;
use hexo_model::{ModelPackage, PackageError};
use hexo_model_mock::MockPackage;

/// The message a refusal carried, whichever refusal it was.
fn refusal(config: &str) -> String {
    MockPackage::from_config(config)
        .err()
        .map(|error| error.to_string())
        .unwrap_or_else(|| panic!("{config:?} was accepted"))
}

#[test]
fn both_search_shapes_construct() {
    MockPackage::from_config("search=policy").expect("policy is a shape");
    MockPackage::from_config("search=mcts:visits=64,inflight=8,cpuct=1.5").expect("so is mcts");
}

#[test]
fn the_mcts_parameters_may_be_given_in_any_order() {
    MockPackage::from_config("search=mcts:cpuct=0,inflight=1,visits=1").expect("all three stated");
}

#[test]
fn a_zero_exploration_constant_is_meaningful_and_accepted() {
    // Zero `c_puct` is pure exploitation of the value estimate, which is a
    // choice; zero visits and zero in-flight are not.
    MockPackage::from_config("search=mcts:visits=8,inflight=1,cpuct=0").expect("zero is a value");
}

#[test]
fn a_missing_search_key_is_refused_rather_than_defaulted() {
    for config in ["", "policy", "mcts"] {
        let message = refusal(config);
        assert!(message.contains("search=<shape>"), "{config:?}: {message}");
        assert!(message.contains("no default"), "{config:?}: {message}");
    }
}

#[test]
fn an_unknown_configuration_key_is_named() {
    let message = refusal("shape=policy");
    assert!(message.contains("\"shape\""), "{message}");
    assert!(message.contains("the only key is `search`"), "{message}");
}

#[test]
fn whitespace_is_not_quietly_trimmed() {
    // One grammar, stated once, rather than a grammar plus a lenience policy.
    let message = refusal("search = policy");
    assert!(message.contains("unknown configuration key"), "{message}");
}

#[test]
fn an_unknown_search_shape_is_named() {
    let message = refusal("search=greedy");
    assert!(
        message.contains("unknown search shape \"greedy\""),
        "{message}"
    );
}

#[test]
fn policy_takes_no_parameters() {
    let message = refusal("search=policy:visits=8");
    assert!(
        message.contains("`policy` takes no parameters"),
        "{message}"
    );
}

#[test]
fn mcts_without_parameters_is_refused_because_there_is_no_default_shape() {
    let message = refusal("search=mcts");
    assert!(
        message.contains("`visits`, `inflight`, and `cpuct`"),
        "{message}"
    );
}

#[test]
fn each_missing_mcts_parameter_is_named() {
    for (config, missing) in [
        ("search=mcts:inflight=8,cpuct=1.5", "visits"),
        ("search=mcts:visits=64,cpuct=1.5", "inflight"),
        ("search=mcts:visits=64,inflight=8", "cpuct"),
    ] {
        let message = refusal(config);
        assert!(message.contains(missing), "{config:?}: {message}");
        assert!(message.contains("is missing"), "{config:?}: {message}");
    }
}

#[test]
fn an_unknown_mcts_parameter_is_named() {
    let message = refusal("search=mcts:visits=64,inflight=8,cpuct=1.5,temperature=2");
    assert!(message.contains("\"temperature\""), "{message}");
}

#[test]
fn a_parameter_stated_twice_is_a_mistake_not_a_last_one_wins() {
    let message = refusal("search=mcts:visits=64,visits=128,inflight=8,cpuct=1.5");
    assert!(message.contains("stated twice"), "{message}");
}

#[test]
fn a_field_that_is_not_a_key_value_pair_is_refused() {
    let message = refusal("search=mcts:visits=64,inflight,cpuct=1.5");
    assert!(message.contains("is not a `key=value` pair"), "{message}");
}

#[test]
fn numbers_parse_loudly() {
    for config in [
        "search=mcts:visits=lots,inflight=8,cpuct=1.5",
        "search=mcts:visits=64,inflight=eight,cpuct=1.5",
        "search=mcts:visits=64,inflight=8,cpuct=wide",
        "search=mcts:visits=-1,inflight=8,cpuct=1.5",
    ] {
        let message = refusal(config);
        assert!(message.contains("not a number"), "{config:?}: {message}");
    }
}

#[test]
fn a_zero_budget_or_a_zero_in_flight_cap_is_refused() {
    assert!(
        refusal("search=mcts:visits=0,inflight=8,cpuct=1.5").contains("zero"),
        "a zero budget is no search",
    );
    assert!(
        refusal("search=mcts:visits=64,inflight=0,cpuct=1.5").contains("zero"),
        "a zero cap could never emit a leaf",
    );
}

#[test]
fn an_exploration_constant_that_is_not_finite_and_non_negative_is_refused() {
    for config in [
        "search=mcts:visits=64,inflight=8,cpuct=-1",
        "search=mcts:visits=64,inflight=8,cpuct=NaN",
        "search=mcts:visits=64,inflight=8,cpuct=inf",
    ] {
        let message = refusal(config);
        assert!(
            message.contains("finite and non-negative"),
            "{config:?}: {message}"
        );
    }
}

#[test]
fn a_variant_name_is_a_search_shape_in_the_same_grammar() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let package = loaded(dir.path(), "search=policy").expect("initialised and loaded");

    // The same strings that are valid `search=` values are valid variant names,
    // which is what lets a match harness pit two search shapes against each
    // other over one set of weights.
    package.variant_session("policy").expect("a shape");
    package
        .variant_session("mcts:visits=128,inflight=4,cpuct=1.0")
        .expect("also a shape");
}

#[test]
fn a_name_that_is_no_shape_at_all_is_an_unknown_variant() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let package = loaded(dir.path(), "search=policy").expect("initialised and loaded");
    let Err(error) = package.variant_session("greedy") else {
        panic!("there is no such variant");
    };
    match error {
        PackageError::UnknownVariant { package, variant } => {
            assert_eq!(package, "mock");
            assert_eq!(variant, "greedy");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_variant_naming_a_real_shape_badly_says_which_parameter_is_wrong() {
    // Not `UnknownVariant`: the shape is one this package has, so the useful
    // answer names the mistake rather than the name.
    let dir = tempfile::tempdir().expect("a scratch directory");
    let package = loaded(dir.path(), "search=policy").expect("initialised and loaded");
    let Err(error) = package.variant_session("mcts:visits=128,inflight=4") else {
        panic!("cpuct is missing from that name");
    };
    match error {
        PackageError::InvalidConfig { package, problem } => {
            assert_eq!(package, "mock");
            assert!(problem.contains("cpuct"), "{problem}");
        }
        other => panic!("{other:?}"),
    }
}
