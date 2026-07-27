//! The package's configuration syntax, parsed in one place.
//!
//! Two entry points, one grammar. [`parse_config`] reads the whole
//! configuration string the container was given; [`parse_search`] reads a search
//! shape on its own, which is what a session variant name is. They share a
//! parser rather than agreeing about one, so `"mcts:visits=128,inflight=4,\
//! cpuct=1.0"` cannot be a valid `search=` value and an invalid variant name at
//! the same time.

use crate::NAME;
use hexo_model::PackageError;
use hexo_search::MctsConfig;
use std::num::{NonZeroU32, NonZeroUsize};

/// The search shape a session is built with.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Search {
    /// One root evaluation per move.
    Policy,
    /// PUCT with the given budget, in-flight cap, and exploration constant.
    Mcts(MctsConfig),
}

/// Why a search shape could not be read.
///
/// The distinction is not cosmetic: it is what lets a *variant name* that names
/// no shape at all come back as [`PackageError::UnknownVariant`] — the honest
/// answer to "do you have a variant called `greedy`" — while a name that does
/// name a shape but mis-states its parameters comes back saying which parameter,
/// which is the answer to the mistake actually made.
pub(crate) enum ParseFailure {
    /// The leading word is not a search shape this package has.
    UnknownShape,
    /// The shape is known, but what follows it is not usable.
    BadParameters(String),
}

impl ParseFailure {
    /// How this failure reads when the string came from the `search=` value.
    pub(crate) fn into_config_error(self, shape: &str) -> PackageError {
        let problem = match self {
            Self::UnknownShape => format!(
                "unknown search shape {shape:?}; this package has `policy` and \
                 `mcts:visits=N,inflight=N,cpuct=F`"
            ),
            Self::BadParameters(problem) => problem,
        };
        PackageError::InvalidConfig {
            package: NAME,
            problem,
        }
    }

    /// How this failure reads when the string was a session variant name.
    pub(crate) fn into_variant_error(self, name: &str) -> PackageError {
        match self {
            Self::UnknownShape => PackageError::UnknownVariant {
                package: NAME,
                variant: name.to_owned(),
            },
            Self::BadParameters(problem) => PackageError::InvalidConfig {
                package: NAME,
                problem,
            },
        }
    }
}

/// Read the package's whole configuration string.
///
/// The grammar is `search=<shape>` and nothing else. There is exactly one key,
/// it is required, and there is no default: a search shape is a model choice,
/// and a package that silently picked one would be choosing how every game in
/// the run is played on behalf of whoever forgot to say.
///
/// Whitespace is not trimmed anywhere. The string comes from a flag, one
/// grammar is easier to state than one grammar plus a lenience policy, and
/// `"search = policy"` is refused by name rather than guessed at.
pub(crate) fn parse_config(config: &str) -> Result<Search, PackageError> {
    let Some((key, value)) = config.split_once('=') else {
        return Err(PackageError::InvalidConfig {
            package: NAME,
            problem: format!(
                "{config:?} holds no `key=value` pair; this package needs `search=<shape>`, and \
                 has no default shape to fall back on"
            ),
        });
    };
    if key != "search" {
        return Err(PackageError::InvalidConfig {
            package: NAME,
            problem: format!("unknown configuration key {key:?}; the only key is `search`"),
        });
    }
    parse_search(value).map_err(|failure| failure.into_config_error(value))
}

/// Read one search shape: `policy`, or `mcts:` and its three parameters.
pub(crate) fn parse_search(shape: &str) -> Result<Search, ParseFailure> {
    let (word, params) = match shape.split_once(':') {
        Some((word, params)) => (word, Some(params)),
        None => (shape, None),
    };
    match word {
        "policy" => match params {
            None => Ok(Search::Policy),
            Some(params) => Err(ParseFailure::BadParameters(format!(
                "`policy` takes no parameters, but {params:?} follows it"
            ))),
        },
        "mcts" => match params {
            None => Err(ParseFailure::BadParameters(
                "`mcts` needs `visits`, `inflight`, and `cpuct`, as in \
                 `mcts:visits=64,inflight=8,cpuct=1.5`"
                    .to_owned(),
            )),
            Some(params) => parse_mcts(params),
        },
        _ => Err(ParseFailure::UnknownShape),
    }
}

/// The three `mcts` parameters, all required and each stated at most once.
///
/// Required rather than defaulted for the reason `MctsConfig` has no `Default`:
/// the budget is the compute a seat is allowed, the cap is how much of a batch it
/// may occupy, and `c_puct` trades exploration against the value head. A package
/// that let any of the three go unstated would be training against a number
/// nobody picked.
fn parse_mcts(params: &str) -> Result<Search, ParseFailure> {
    let mut visits: Option<NonZeroU32> = None;
    let mut inflight: Option<NonZeroUsize> = None;
    let mut c_puct: Option<f32> = None;

    for field in params.split(',') {
        let Some((key, value)) = field.split_once('=') else {
            return Err(ParseFailure::BadParameters(format!(
                "{field:?} is not a `key=value` pair"
            )));
        };
        match key {
            "visits" => {
                reject_repeat(visits.is_some(), key)?;
                visits = Some(nonzero_u32(key, value)?);
            }
            "inflight" => {
                reject_repeat(inflight.is_some(), key)?;
                inflight = Some(nonzero_usize(key, value)?);
            }
            "cpuct" => {
                reject_repeat(c_puct.is_some(), key)?;
                c_puct = Some(exploration(key, value)?);
            }
            _ => {
                return Err(ParseFailure::BadParameters(format!(
                    "unknown `mcts` parameter {key:?}; it takes `visits`, `inflight`, and `cpuct`"
                )));
            }
        }
    }

    Ok(Search::Mcts(MctsConfig {
        visits: visits.ok_or_else(|| missing("visits"))?,
        max_in_flight: inflight.ok_or_else(|| missing("inflight"))?,
        c_puct: c_puct.ok_or_else(|| missing("cpuct"))?,
    }))
}

/// A parameter stated twice is a mistake, not a last-one-wins.
fn reject_repeat(already: bool, key: &str) -> Result<(), ParseFailure> {
    if already {
        return Err(ParseFailure::BadParameters(format!(
            "`mcts` parameter {key:?} is stated twice"
        )));
    }
    Ok(())
}

/// A parameter that was never stated.
fn missing(key: &str) -> ParseFailure {
    ParseFailure::BadParameters(format!(
        "`mcts` parameter {key:?} is missing, and this package has no default for it"
    ))
}

/// A count that has to be at least one.
fn nonzero_u32(key: &str, value: &str) -> Result<NonZeroU32, ParseFailure> {
    let parsed: u32 = value.parse().map_err(|_| {
        ParseFailure::BadParameters(format!(
            "`mcts` parameter {key:?} is {value:?}, not a number"
        ))
    })?;
    NonZeroU32::new(parsed).ok_or_else(|| {
        ParseFailure::BadParameters(format!(
            "`mcts` parameter {key:?} is zero, which is no search"
        ))
    })
}

/// A count that has to be at least one, sized for the host.
fn nonzero_usize(key: &str, value: &str) -> Result<NonZeroUsize, ParseFailure> {
    let parsed: usize = value.parse().map_err(|_| {
        ParseFailure::BadParameters(format!(
            "`mcts` parameter {key:?} is {value:?}, not a number"
        ))
    })?;
    NonZeroUsize::new(parsed).ok_or_else(|| {
        ParseFailure::BadParameters(format!(
            "`mcts` parameter {key:?} is zero, so the session could never emit a leaf"
        ))
    })
}

/// The PUCT constant: finite and non-negative, zero being meaningful.
fn exploration(key: &str, value: &str) -> Result<f32, ParseFailure> {
    let parsed: f32 = value.parse().map_err(|_| {
        ParseFailure::BadParameters(format!(
            "`mcts` parameter {key:?} is {value:?}, not a number"
        ))
    })?;
    if !parsed.is_finite() || parsed < 0.0 {
        return Err(ParseFailure::BadParameters(format!(
            "`mcts` parameter {key:?} is {parsed}; the exploration constant must be finite and \
             non-negative"
        )));
    }
    Ok(parsed)
}
