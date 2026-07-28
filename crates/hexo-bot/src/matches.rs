//! `match`: two seats, fixed weights, one JSON report.

use crate::Outcome;
use crate::cli::{MatchConfig, SeatSpec};
use crate::driver::{LaneSeats, Sweep, run_sweep};
use crate::error::BotError;
use crate::registry::PackageRegistry;
use hexo_model::ModelPackage;
use hexo_runner::{Budget, FailurePolicy, GameSpec};
use hexo_search::{DecisionSession, Encoder};
use serde_json::{Value, json};
use std::path::PathBuf;

/// A finished `match`, and what it found.
#[derive(Clone, Debug)]
pub struct MatchRun {
    /// Whether the games ran out or the stop flag cut them short.
    pub outcome: Outcome,
    /// The report, which the binary prints to stdout either way — a match that
    /// was stopped reports the games it did finish rather than nothing.
    pub report: MatchReport,
}

/// One seat's side of a match report.
#[derive(Clone, Debug)]
pub struct SeatReport {
    /// The registry name of the package that filled the seat.
    pub package: String,
    /// The checkpoint its weights came from.
    pub checkpoint: PathBuf,
    /// The package configuration string it was built with.
    pub config: String,
    /// The session variant it played, or `None` for the package's evaluation
    /// mode.
    pub variant: Option<String>,
    /// Games won, in either colour.
    pub wins: usize,
    /// Of those, the ones won moving first.
    pub wins_as_p0: usize,
    /// Of those, the ones won moving second.
    pub wins_as_p1: usize,
}

/// What a match came to.
///
/// Colour is reported separately because it is the question a two-seat result
/// cannot answer on its own: a competitor that only wins when it moves first has
/// not beaten the other one.
#[derive(Clone, Debug)]
pub struct MatchReport {
    /// Games that finished.
    pub games: usize,
    /// Mean placements per game.
    pub mean_plies: f64,
    /// Games that ended on the ply cap.
    pub draws: usize,
    /// Games that say nothing about either seat.
    pub no_contests: usize,
    /// The two competitors, in the order their `--seat` flags were given.
    pub seats: [SeatReport; 2],
}

impl MatchReport {
    /// The report as JSON.
    #[must_use]
    pub fn to_json(&self) -> Value {
        json!({
            "games": self.games,
            "mean_plies": self.mean_plies,
            "draws": self.draws,
            "no_contests": self.no_contests,
            "seats": self.seats.iter().map(seat_json).collect::<Vec<_>>(),
        })
    }
}

/// One seat as JSON.
fn seat_json(seat: &SeatReport) -> Value {
    json!({
        "package": seat.package,
        "checkpoint": seat.checkpoint.display().to_string(),
        "config": seat.config,
        "variant": seat.variant,
        "wins": seat.wins,
        "wins_as_p0": seat.wins_as_p0,
        "wins_as_p1": seat.wins_as_p1,
    })
}

/// Play one head-to-head.
///
/// Both seats may name the same checkpoint. That is the point of the
/// subcommand rather than a curiosity: same weights, two searches, which is how
/// a search shape is compared without a training run in between. Each seat gets
/// its own package instance even then, because a package holds the weights that
/// answer and the container has no way to know whether two of them could share
/// one — the mock's evaluators are independent copies of a salt, and a
/// Python-backed package's would be a live module that §10.1 pools instead.
///
/// **No shards are written.** `docs/CONTAINER_SPEC.md` §11 keeps records as
/// training data, and nothing consumes a match's games: they are evidence about
/// two checkpoints, which is what the report is for.
///
/// # Errors
///
/// [`BotError::UnknownPackage`] for a seat naming no package,
/// [`BotError::Package`] for a configuration, a checkpoint, or a variant name
/// the package refuses, and [`BotError::Io`] if `--report` cannot be written.
pub fn play_match(config: &MatchConfig, registry: &PackageRegistry) -> Result<MatchRun, BotError> {
    let packages = [
        seat_package(registry, &config.seats[0])?,
        seat_package(registry, &config.seats[1])?,
    ];

    let games = config.games.get();
    let mut lanes = Vec::with_capacity(games);
    for lane in 0..games {
        // Lane `i` seats the first `--seat` as `P0` when `i` is even, so the
        // first-move advantage is split rather than handed to whoever was typed
        // first.
        let first_moves = lane % 2 == 0;
        let one = session(packages[0].as_ref(), &config.seats[0])?;
        let two = session(packages[1].as_ref(), &config.seats[1])?;
        lanes.push(if first_moves {
            LaneSeats {
                sessions: [one, two],
                slots: [0, 1],
            }
        } else {
            LaneSeats {
                sessions: [two, one],
                slots: [1, 0],
            }
        });
    }

    let mut encoders: Vec<Vec<Box<dyn Encoder>>> = Vec::with_capacity(config.threads.get());
    for _ in 0..config.threads.get() {
        encoders.push(vec![packages[0].encoder(), packages[1].encoder()]);
    }
    let evaluators = vec![packages[0].evaluator()?, packages[1].evaluator()?];

    let played = run_sweep(Sweep {
        spec: GameSpec {
            ply_cap: config.ply_cap,
            budget: Budget::Unlimited,
            on_failure: FailurePolicy::Forfeit,
        },
        lanes,
        games,
        encoders,
        evaluators,
        batch: config.batch,
        batch_wait: config.batch_wait,
        threads: config.threads,
        stop: &config.stop,
        records: None,
    })?;

    let tally = played.tally;
    let report = MatchReport {
        games: tally.games,
        mean_plies: tally.mean_plies(),
        draws: tally.draws,
        no_contests: tally.no_contests,
        seats: [
            seat_report(&config.seats[0], tally.wins[0]),
            seat_report(&config.seats[1], tally.wins[1]),
        ],
    };

    if let Some(path) = &config.report {
        let mut json = serde_json::to_string_pretty(&report.to_json())
            .expect("a report built from `json!` serialises");
        json.push('\n');
        std::fs::write(path, json).map_err(|source| BotError::io(path, source))?;
    }

    Ok(MatchRun {
        outcome: played.outcome,
        report,
    })
}

/// Build and load one seat's package.
fn seat_package(
    registry: &PackageRegistry,
    spec: &SeatSpec,
) -> Result<Box<dyn ModelPackage>, BotError> {
    let mut package = registry.construct(&spec.package, &spec.config)?;
    package
        .load(&spec.checkpoint)
        .map_err(|source| BotError::UnloadableCheckpoint {
            path: spec.checkpoint.clone(),
            source,
        })?;
    Ok(package)
}

/// One session for one seat.
///
/// A named variant, when the spec gave one, and the package's evaluation mode
/// otherwise. The vocabulary is the package's, so a name it does not have comes
/// back as its own refusal rather than as anything this crate invented.
fn session(
    package: &dyn ModelPackage,
    spec: &SeatSpec,
) -> Result<Box<dyn DecisionSession>, BotError> {
    match &spec.variant {
        Some(name) => Ok(package.variant_session(name)?),
        None => Ok(package.eval_session()?),
    }
}

/// One seat's line of the report, from its `[colour]` win counts.
fn seat_report(spec: &SeatSpec, wins: [usize; 2]) -> SeatReport {
    SeatReport {
        package: spec.package.clone(),
        checkpoint: spec.checkpoint.clone(),
        config: spec.config.clone(),
        variant: spec.variant.clone(),
        wins: wins[0] + wins[1],
        wins_as_p0: wins[0],
        wins_as_p1: wins[1],
    }
}
