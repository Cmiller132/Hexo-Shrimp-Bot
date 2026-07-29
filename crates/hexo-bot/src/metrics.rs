//! What a sweep counted, and the one line an epoch appends to `metrics.jsonl`.

use crate::error::BotError;
use crate::run::RunLayout;
use hexo_runner::MatchResult;
use serde_json::{Value, json};
use std::fs::OpenOptions;
use std::io::Write as _;
use std::time::Duration;

/// Which evaluator answers a seat's leaves: an index into the sweep's slots.
///
/// Self-play uses one slot for both seats; an evaluation or a match uses two,
/// one per competitor.
pub(crate) type SlotId = usize;

/// What a set of finished games came to.
///
/// Wins are indexed by both the evaluator slot that won and the colour it was
/// playing, because both questions are asked of the same games: a self-play
/// epoch wants to know whether `P0` is running away with it, and a match wants
/// to know whether a competitor only wins when it moves first.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Tally {
    /// Games that finished.
    pub games: usize,
    /// Placements across all of them.
    pub plies: usize,
    /// Wins, indexed `[slot][colour]`.
    pub wins: [[usize; 2]; 2],
    /// Games that ended on the ply cap.
    pub draws: usize,
    /// Games that say nothing about either seat's strength.
    pub no_contests: usize,
}

impl Tally {
    /// Fold one finished game in.
    ///
    /// `slots` is the lane's seat-to-slot map at the moment the game ended,
    /// which is what makes a win attributable after colours have alternated.
    pub(crate) fn record(&mut self, result: MatchResult, plies: usize, slots: [SlotId; 2]) {
        self.games += 1;
        self.plies += plies;
        match result {
            MatchResult::Decisive { winner, .. } => {
                self.wins[slots[winner.index()]][winner.index()] += 1;
            }
            MatchResult::Drawn { .. } => self.draws += 1,
            MatchResult::NoContest(_) => self.no_contests += 1,
        }
    }

    /// Wins by the seat filling `slot`, in either colour.
    pub(crate) fn wins_for(&self, slot: SlotId) -> usize {
        self.wins[slot][0] + self.wins[slot][1]
    }

    /// Mean placements per game, or zero if nothing finished.
    pub(crate) fn mean_plies(&self) -> f64 {
        if self.games == 0 {
            0.0
        } else {
            self.plies as f64 / self.games as f64
        }
    }
}

/// How long each phase of an epoch took.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Timing {
    /// Driving `G` self-play games to completion.
    pub self_play: Duration,
    /// The package's `fit`, plus placing and proving the checkpoint it wrote.
    pub fit: Duration,
    /// Every evaluation pairing this epoch ran.
    pub eval: Duration,
}

/// One evaluation pairing's outcome, from the current checkpoint's side.
#[derive(Clone, Copy, Debug)]
pub(crate) struct EvalResult {
    /// The epoch of the checkpoint that was played against.
    pub opponent_epoch: u32,
    /// Games that finished.
    pub games: usize,
    /// Games the current checkpoint won.
    pub wins: usize,
    /// Games the opponent won.
    pub losses: usize,
    /// Games that ended on the ply cap.
    pub draws: usize,
    /// Games adjudicated as no-contest.
    pub no_contests: usize,
}

impl EvalResult {
    /// This pairing as JSON, with the win rate over the games that finished.
    fn to_json(self) -> Value {
        let rate = if self.games == 0 {
            Value::Null
        } else {
            json!(self.wins as f64 / self.games as f64)
        };
        json!({
            "opponent_epoch": self.opponent_epoch,
            "games": self.games,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "no_contests": self.no_contests,
            "win_rate": rate,
        })
    }
}

/// Everything one epoch has to say about itself.
pub(crate) struct EpochMetrics {
    /// The epoch whose self-play produced these games.
    pub epoch: u32,
    /// Where the wall time went.
    pub timing: Timing,
    /// The self-play games.
    pub tally: Tally,
    /// Evaluator calls the batcher made.
    pub batches: usize,
    /// Positions those calls answered.
    pub evaluations: usize,
    /// Every evaluation pairing this epoch ran; empty when none did.
    pub evals: Vec<EvalResult>,
}

impl EpochMetrics {
    /// The line, as JSON.
    ///
    /// `eval` is always present and always an array, so a reader walking
    /// `metrics.jsonl` has one shape to handle rather than two: an epoch that
    /// ran no evaluation says so with an empty list.
    fn to_json(&self) -> Value {
        let fill = if self.batches == 0 {
            0.0
        } else {
            self.evaluations as f64 / self.batches as f64
        };
        let total = self.timing.self_play + self.timing.fit + self.timing.eval;
        json!({
            "epoch": self.epoch,
            "seconds": {
                "self_play": self.timing.self_play.as_secs_f64(),
                "fit": self.timing.fit.as_secs_f64(),
                "eval": self.timing.eval.as_secs_f64(),
                "total": total.as_secs_f64(),
            },
            "games": self.tally.games,
            "positions": self.tally.plies,
            "results": {
                "p0_wins": self.tally.wins[0][0],
                "p1_wins": self.tally.wins[0][1],
                "draws": self.tally.draws,
                "no_contests": self.tally.no_contests,
            },
            "evaluations": self.evaluations,
            "batches": self.batches,
            "mean_batch_fill": fill,
            "eval": self.evals.iter().map(|e| e.to_json()).collect::<Vec<_>>(),
        })
    }

    /// Append this epoch's line to the run's `metrics.jsonl`.
    ///
    /// Each completed epoch is appended immediately, making committed metrics
    /// observable while the run continues (`docs/CONTAINER_SPEC.md` §8.1).
    ///
    /// # Errors
    ///
    /// [`BotError::Io`] if the file cannot be opened or extended.
    pub(crate) fn append(&self, layout: &RunLayout) -> Result<(), BotError> {
        let path = layout.metrics();
        let mut line = self.to_json().to_string();
        line.push('\n');
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|source| BotError::io(&path, source))?;
        file.write_all(line.as_bytes())
            .map_err(|source| BotError::io(&path, source))
    }
}
