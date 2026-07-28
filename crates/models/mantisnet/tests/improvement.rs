//! Parity and validation tests for MantisNet's KLENT policy improvement.

use hexo_model_mantisnet::improvement::{ImprovementError, improve_policy};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Case {
    name: String,
    policy_logits: Vec<f32>,
    q_values: Vec<f32>,
    tau: f32,
    lambda: f32,
    expected_pi_prime: Vec<f32>,
    expected_v_hat: f32,
}

#[test]
fn matches_python_improvement_fixtures() {
    let fixture: Fixture = serde_json::from_str(include_str!("fixtures/improvement.json"))
        .expect("valid fixture JSON");
    let mut maximum_probability_error = 0.0_f32;
    let mut maximum_value_error = 0.0_f32;

    for case in fixture.cases {
        let actual = improve_policy(&case.policy_logits, &case.q_values, case.tau, case.lambda)
            .unwrap_or_else(|error| panic!("fixture {} failed: {error}", case.name));
        assert_eq!(
            actual.pi_prime.len(),
            case.expected_pi_prime.len(),
            "fixture {} returned the wrong row length",
            case.name
        );

        for (&actual_probability, &expected_probability) in
            actual.pi_prime.iter().zip(&case.expected_pi_prime)
        {
            maximum_probability_error =
                maximum_probability_error.max((actual_probability - expected_probability).abs());
        }
        maximum_value_error = maximum_value_error.max((actual.v_hat - case.expected_v_hat).abs());
    }

    eprintln!(
        "Python parity: max |pi_prime error|={maximum_probability_error:e}, \
         max |v_hat error|={maximum_value_error:e}"
    );
    assert!(
        maximum_probability_error <= 1.0e-6,
        "pi_prime parity error {maximum_probability_error:e} exceeded 1e-6"
    );
    assert!(
        maximum_value_error <= 1.0e-6,
        "v_hat parity error {maximum_value_error:e} exceeded 1e-6"
    );
}

#[test]
fn rejects_length_mismatch_and_empty_rows() {
    assert!(matches!(
        improve_policy(&[0.0], &[], 0.1, 0.03),
        Err(ImprovementError::LengthMismatch { .. })
    ));
    assert_eq!(
        improve_policy(&[], &[], 0.1, 0.03),
        Err(ImprovementError::Empty)
    );
}

#[test]
fn rejects_invalid_parameters() {
    for (tau, lambda) in [
        (0.0, 0.0),
        (-0.1, 0.1),
        (0.1, -0.1),
        (f32::NAN, 0.1),
        (0.1, f32::INFINITY),
        (f32::MAX, f32::MAX),
    ] {
        assert!(matches!(
            improve_policy(&[0.0], &[0.0], tau, lambda),
            Err(ImprovementError::InvalidParameters { .. })
        ));
    }
}

#[test]
fn rejects_nonfinite_inputs_and_out_of_range_values() {
    assert!(matches!(
        improve_policy(&[f32::INFINITY], &[0.0], 0.1, 0.03),
        Err(ImprovementError::NonFinitePolicyLogit { index: 0, .. })
    ));
    assert!(matches!(
        improve_policy(&[0.0], &[f32::NAN], 0.1, 0.03),
        Err(ImprovementError::NonFiniteQValue { index: 0, .. })
    ));
    for q_value in [-1.000_001, 1.000_001] {
        assert!(matches!(
            improve_policy(&[0.0], &[q_value], 0.1, 0.03),
            Err(ImprovementError::QValueOutOfRange { index: 0, .. })
        ));
    }
}

#[test]
fn preserves_boundary_q_values() {
    let improved =
        improve_policy(&[0.0, 0.0], &[-1.0, 1.0], 0.0, 0.5).expect("boundary q values are legal");
    assert_eq!(improved.pi_prime.len(), 2);
    assert!((-1.0..=1.0).contains(&improved.v_hat));

    // A long row is where f32 probability accumulation can round past the
    // convex hull even though every Q value is exactly on its boundary.
    let logits = vec![0.0; 18];
    let q_values = vec![1.0; logits.len()];
    let improved =
        improve_policy(&logits, &q_values, 0.1, 0.03).expect("a long boundary row is legal");
    assert!(
        (-1.0..=1.0).contains(&improved.v_hat),
        "v_hat {} escaped the Evaluation convention",
        improved.v_hat,
    );
}
