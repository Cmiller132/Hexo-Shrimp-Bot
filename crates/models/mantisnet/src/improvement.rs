//! MantisNet's closed-form KLENT policy improvement.
//!
//! The network supplies one policy logit and one action value for every legal
//! action in canonical order. [`improve_policy`] applies equation 3 from the
//! model specification to that single ragged row. Keeping this operation in
//! the model package makes `pi_prime` part of MantisNet's opinion rather than a
//! search-session concern.

use std::{error::Error, fmt};

/// The improved MantisNet opinion for one position.
#[derive(Clone, Debug, PartialEq)]
pub struct ImprovedPolicy {
    /// Improved action probabilities, in the input's canonical legal order.
    pub pi_prime: Vec<f32>,
    /// Expected action value under `pi_prime`, from the side-to-move view.
    pub v_hat: f32,
}

/// A malformed or numerically invalid KLENT improvement request.
#[derive(Clone, Debug, PartialEq)]
pub enum ImprovementError {
    /// Policy and action-value rows must describe the same legal actions.
    LengthMismatch {
        /// Number of policy logits.
        policy_logits: usize,
        /// Number of action values.
        q_values: usize,
    },
    /// A position must have at least one legal action to improve.
    Empty,
    /// `tau` and `lambda` must be finite, non-negative, and have a finite
    /// positive sum.
    InvalidParameters {
        /// Reverse-KL weight.
        tau: f32,
        /// Entropy weight.
        lambda: f32,
    },
    /// A policy logit was not finite.
    NonFinitePolicyLogit {
        /// Canonical legal-action index.
        index: usize,
        /// Rejected value.
        value: f32,
    },
    /// An action value was not finite.
    NonFiniteQValue {
        /// Canonical legal-action index.
        index: usize,
        /// Rejected value.
        value: f32,
    },
    /// An action value fell outside the side-to-move `[-1, 1]` convention.
    QValueOutOfRange {
        /// Canonical legal-action index.
        index: usize,
        /// Rejected value.
        value: f32,
    },
    /// Finite inputs overflowed or otherwise produced a non-finite result.
    NumericalFailure,
}

impl fmt::Display for ImprovementError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch {
                policy_logits,
                q_values,
            } => write!(
                formatter,
                "policy/action-value length mismatch: {policy_logits} logits, {q_values} q values"
            ),
            Self::Empty => formatter.write_str("cannot improve a position with no legal actions"),
            Self::InvalidParameters { tau, lambda } => write!(
                formatter,
                "need finite tau, lambda >= 0 with finite tau + lambda > 0, got ({tau}, {lambda})"
            ),
            Self::NonFinitePolicyLogit { index, value } => {
                write!(formatter, "policy logit {index} is not finite: {value}")
            }
            Self::NonFiniteQValue { index, value } => {
                write!(formatter, "q value {index} is not finite: {value}")
            }
            Self::QValueOutOfRange { index, value } => {
                write!(formatter, "q value {index} is outside [-1, 1]: {value}")
            }
            Self::NumericalFailure => formatter
                .write_str("KLENT improvement produced a non-finite intermediate or output"),
        }
    }
}

impl Error for ImprovementError {}

/// Apply the KLENT closed-form policy improvement to one position.
///
/// The calculation is performed in `f32`, in the same operation order as
/// `mantisnet.klent.improve.improved_policy`:
///
/// `pi_prime(a) proportional to exp((Q(a) + tau * log pi(a)) / (tau + lambda))`
///
/// `v_hat = sum_a pi_prime(a) * Q(a)`
///
/// Both input slices use canonical legal-action order. The returned
/// probabilities preserve that order.
pub fn improve_policy(
    policy_logits: &[f32],
    q_values: &[f32],
    tau: f32,
    lambda: f32,
) -> Result<ImprovedPolicy, ImprovementError> {
    if policy_logits.len() != q_values.len() {
        return Err(ImprovementError::LengthMismatch {
            policy_logits: policy_logits.len(),
            q_values: q_values.len(),
        });
    }
    if policy_logits.is_empty() {
        return Err(ImprovementError::Empty);
    }

    let denominator = tau + lambda;
    if !tau.is_finite()
        || !lambda.is_finite()
        || tau < 0.0
        || lambda < 0.0
        || !denominator.is_finite()
        || denominator <= 0.0
    {
        return Err(ImprovementError::InvalidParameters { tau, lambda });
    }

    for (index, &value) in policy_logits.iter().enumerate() {
        if !value.is_finite() {
            return Err(ImprovementError::NonFinitePolicyLogit { index, value });
        }
    }
    for (index, &value) in q_values.iter().enumerate() {
        if !value.is_finite() {
            return Err(ImprovementError::NonFiniteQValue { index, value });
        }
        if !(-1.0..=1.0).contains(&value) {
            return Err(ImprovementError::QValueOutOfRange { index, value });
        }
    }

    let log_pi = log_softmax(policy_logits)?;
    let mut improved_logits = Vec::with_capacity(policy_logits.len());
    for (&q, &log_probability) in q_values.iter().zip(&log_pi) {
        let improved = (q + tau * log_probability) / denominator;
        if !improved.is_finite() {
            return Err(ImprovementError::NumericalFailure);
        }
        improved_logits.push(improved);
    }

    let log_improved = log_softmax(&improved_logits)?;
    let mut pi_prime = Vec::with_capacity(log_improved.len());
    let mut v_hat = 0.0_f32;
    for (&log_probability, &q) in log_improved.iter().zip(q_values) {
        let probability = log_probability.exp();
        if !probability.is_finite() {
            return Err(ImprovementError::NumericalFailure);
        }
        pi_prime.push(probability);
        v_hat += probability * q;
    }
    if !v_hat.is_finite() {
        return Err(ImprovementError::NumericalFailure);
    }
    // Mathematically this is a convex combination of values in [-1, 1].
    // Sequential f32 accumulation can overshoot an endpoint by a few ulps
    // (e.g. eighteen identical Q=1 actions), but Evaluation's public
    // convention is exact and sessions deliberately refuse out-of-range
    // values. Project only that arithmetic noise back onto the closed interval.
    let v_hat = v_hat.clamp(-1.0, 1.0);

    Ok(ImprovedPolicy { pi_prime, v_hat })
}

fn log_softmax(values: &[f32]) -> Result<Vec<f32>, ImprovementError> {
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exponential_sum = 0.0_f32;
    for &value in values {
        exponential_sum += (value - maximum).exp();
    }
    let log_sum = exponential_sum.ln();
    if !log_sum.is_finite() {
        return Err(ImprovementError::NumericalFailure);
    }

    let mut output = Vec::with_capacity(values.len());
    for &value in values {
        let log_probability = (value - maximum) - log_sum;
        if !log_probability.is_finite() {
            return Err(ImprovementError::NumericalFailure);
        }
        output.push(log_probability);
    }
    Ok(output)
}
