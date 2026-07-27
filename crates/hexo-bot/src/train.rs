//! The epoch loop: self-play, fit, checkpoint, evaluate, and one metrics line,
//! in one long-lived process.

use crate::Outcome;
use crate::cli::TrainConfig;
use crate::driver::{LaneSeats, RecordSink, Sweep, SweepReport, run_sweep};
use crate::error::BotError;
use crate::metrics::{EpochMetrics, EvalResult, Timing};
use crate::registry;
use crate::run::{self, RunLayout};
use hexo_model::{MANIFEST_FILE, ModelPackage};
use hexo_records::{ShardHeader, ShardMode};
use hexo_runner::{Budget, FailurePolicy, GameSpec};
use hexo_search::Encoder;
use std::num::NonZeroUsize;
use std::sync::atomic::Ordering;
use std::time::Instant;

/// Self-play has one evaluator, and both seats draw from it.
const ONE_EVALUATOR: usize = 0;

/// The evaluator slot the contender occupies in an evaluation round.
const CONTENDER: usize = 0;

/// The evaluator slot the older checkpoint occupies in an evaluation round.
const OPPONENT: usize = 1;

/// Run a whole training run.
///
/// Self-play and fitting are one mode, not two (`docs/CONTAINER_SPEC.md` §8):
/// the fixed costs of standing a model up are paid once per process rather than
/// once per epoch, the two phases never run at the same time so the one device
/// is never contended, and the phases are deliberately not separately invocable
/// — there is one implementation of the loop, not a loop plus a set of pieces
/// that could drift from it.
///
/// # Stopping
///
/// [`TrainConfig::stop`] is checked between epochs and inside every sweep. A
/// stop that arrives during self-play abandons the partial epoch, records and
/// all: those games were on-policy and are worthless without the fit that was
/// going to consume them. A stop that arrives once the fit has begun lets the
/// fit, the checkpoint, the load that proves it, and the metrics line all
/// complete, because `docker stop` must never lose an epoch. Either way the
/// return is [`Outcome::Stopped`], which the binary turns into exit code 2.
///
/// # Errors
///
/// [`BotError::RunExists`] for a run directory that is already occupied without
/// `--resume`, [`BotError::NoRun`] for `--resume` into nothing,
/// [`BotError::ResumeMismatch`] for a resume that changes the run,
/// [`BotError::Package`] for anything the package refuses — including a
/// checkpoint that does not prove — [`BotError::Record`] for a shard that
/// cannot be written, and [`BotError::Io`] for the filesystem.
pub fn train(config: &TrainConfig) -> Result<Outcome, BotError> {
    let layout = RunLayout::new(&config.run_dir, &config.run_id);
    let mut package = registry::construct(&config.package, &config.package_config)?;

    if layout.root().exists() {
        if !config.resume {
            return Err(BotError::RunExists {
                path: layout.root().to_path_buf(),
            });
        }
        run::check_resume(&layout, config)?;
    } else {
        if config.resume {
            return Err(BotError::NoRun {
                path: layout.root().to_path_buf(),
            });
        }
        run::create_dir_all(&layout.checkpoints())?;
        run::create_dir_all(&layout.all_records())?;
        run::write_manifest(&layout, config)?;
    }

    // Epoch 0 is the untrained checkpoint every run starts from, written through
    // the same temporary-then-rename placement as every later one.
    if !layout.checkpoint(0).join(MANIFEST_FILE).exists() {
        run::place_checkpoint(&layout, 0, |dir| package.init(dir))?;
    }

    // Clears what a crash left, proves the checkpoint being continued from, and
    // is where the package acquires its weights for the first time.
    let start = run::resume_point(&layout, package.as_mut())?;

    for epoch in start..config.epochs.get() {
        if stopping(config) {
            return Ok(Outcome::Stopped);
        }
        let mut timing = Timing::default();

        // (1) Self-play under frozen weights, producing this epoch's records.
        let started = Instant::now();
        let records = layout.records(epoch);
        run::create_dir_all(&records)?;
        let played = self_play(config, package.as_ref(), &layout, epoch)?;
        timing.self_play = started.elapsed();
        if played.outcome == Outcome::Stopped {
            // Nothing consumed these games, and nothing ever will: the weights
            // they were played under are the weights the run would have moved
            // past. The unfinalized shard is already gone with its writer.
            run::remove_dir_all(&records)?;
            return Ok(Outcome::Stopped);
        }

        // (2) Fit, and place the checkpoint it wrote.
        let started = Instant::now();
        let shards = vec![layout.shard(epoch)];
        run::place_checkpoint(&layout, epoch + 1, |dir| {
            package.fit(&shards, dir, epoch + 1)
        })?;

        // (3) The records go only once the fit that consumed them has succeeded.
        // Not before, so a failed fit leaves its input to be inspected or re-run,
        // and not later, because on-policy records are worthless under the new
        // weights and keeping them would make disk growth a function of run
        // length.
        run::remove_dir_all(&records)?;

        // Loading is proving, and it is what puts the fit's own output behind
        // the probe. Every evaluator this run uses is built after a load, from
        // the package that performed it.
        package.load(&layout.checkpoint(epoch + 1))?;
        timing.fit = started.elapsed();

        // (4) Evaluation, if this epoch is one of them.
        let started = Instant::now();
        let evals = evaluate(config, package.as_ref(), &layout, epoch)?;
        timing.eval = started.elapsed();

        // (5) One line, appended as it happens.
        EpochMetrics {
            epoch,
            timing,
            tally: played.tally,
            batches: played.batches,
            evaluations: played.evaluations,
            evals,
        }
        .append(&layout)?;

        if stopping(config) {
            return Ok(Outcome::Stopped);
        }
    }

    Ok(Outcome::Completed)
}

/// Whether the operator has asked the run to stop.
fn stopping(config: &TrainConfig) -> bool {
    config.stop.load(Ordering::Relaxed)
}

/// The rules every game of this run is played under.
///
/// `FailurePolicy::Forfeit` is the runner's default and is kept: a driver-level
/// failure is a decisive, contested result and a real fact about a match, and
/// `hexo-records` keeps the whole adjudication payload so a consumer selecting
/// training data can tell one from a win on the board.
fn spec(config: &TrainConfig) -> GameSpec {
    GameSpec {
        ply_cap: config.ply_cap,
        budget: Budget::Unlimited,
        on_failure: FailurePolicy::Forfeit,
    }
}

/// Drive one epoch's self-play games and write them down.
///
/// Both seats are the same package at the same checkpoint, answered by the same
/// evaluator slot, and each gets its *own* session: two seats drawing from one
/// stream would correlate their choices, which is the failure a self-play run
/// cannot detect afterwards because the data it produces is well-formed.
fn self_play(
    config: &TrainConfig,
    package: &dyn ModelPackage,
    layout: &RunLayout,
    epoch: u32,
) -> Result<SweepReport, BotError> {
    let mut lanes = Vec::with_capacity(config.games.get());
    for _ in 0..config.games.get() {
        lanes.push(LaneSeats {
            sessions: [package.self_play_session()?, package.self_play_session()?],
            slots: [ONE_EVALUATOR; 2],
        });
    }
    let encoders = (0..config.threads.get())
        .map(|_| vec![package.encoder()])
        .collect();

    run_sweep(Sweep {
        spec: spec(config),
        lanes,
        games: config.games.get(),
        encoders,
        evaluators: vec![package.evaluator()?],
        batch: config.batch,
        batch_wait: config.batch_wait,
        threads: config.threads,
        stop: &config.stop,
        records: Some(RecordSink {
            path: layout.shard(epoch),
            header: ShardHeader {
                mode: ShardMode::SelfPlay,
                run_id: config.run_id.clone(),
                package: package.name().to_owned(),
                checkpoint: RunLayout::checkpoint_ref(&config.run_id, epoch),
                epoch,
                game_count: 0,
            },
        }),
    })
}

/// Run this epoch's evaluation round, if it has one.
///
/// The checkpoint just written is played against the epoch-0 anchor and against
/// the checkpoint it was fit from — two pairings, or one when those are the same
/// checkpoint. Nothing is recorded: `docs/CONTAINER_SPEC.md` §11 keeps records
/// as training data, an evaluation game trains nothing, and a shard nobody
/// consumes would be the largest artefact in the run written for no reader.
fn evaluate(
    config: &TrainConfig,
    package: &dyn ModelPackage,
    layout: &RunLayout,
    epoch: u32,
) -> Result<Vec<EvalResult>, BotError> {
    if config.eval_every == 0 || !(epoch + 1).is_multiple_of(config.eval_every) {
        return Ok(Vec::new());
    }

    let mut opponents = vec![0];
    if epoch != 0 {
        opponents.push(epoch);
    }

    let mut results = Vec::with_capacity(opponents.len());
    for opponent_epoch in opponents {
        if stopping(config) {
            break;
        }
        let report = pairing(config, package, layout, opponent_epoch)?;
        results.push(EvalResult {
            opponent_epoch,
            games: report.tally.games,
            wins: report.tally.wins_for(CONTENDER),
            losses: report.tally.wins_for(OPPONENT),
            draws: report.tally.draws,
            no_contests: report.tally.no_contests,
        });
        if report.outcome == Outcome::Stopped {
            break;
        }
    }
    Ok(results)
}

/// Play the current checkpoint against one older one.
///
/// The opponent is a second instance of the same package, loaded from the older
/// checkpoint: one package instance per side, because a package holds the
/// weights that answer and the two sides are two sets of weights. Sharing one
/// live module between them is the slot pool of §10.1, which arrives with the
/// first package whose modules are expensive enough to be worth pooling.
fn pairing(
    config: &TrainConfig,
    package: &dyn ModelPackage,
    layout: &RunLayout,
    opponent_epoch: u32,
) -> Result<SweepReport, BotError> {
    let mut opponent = registry::construct(&config.package, &config.package_config)?;
    let path = layout.checkpoint(opponent_epoch);
    opponent
        .load(&path)
        .map_err(|source| BotError::UnloadableCheckpoint { path, source })?;

    // Lanes are capped by the run's concurrency, which is the number the host's
    // memory was budgeted for; a pairing wanting more games than that plays them
    // in successive games on the same lanes.
    let games = config.eval_games.get();
    let lanes = NonZeroUsize::min(config.games, config.eval_games).get();
    let mut seats = Vec::with_capacity(lanes);
    for lane in 0..lanes {
        // Lane `i` seats the contender as `P0` when `i` is even, so a round
        // cancels the first-move advantage rather than measuring it.
        let contender_first = lane % 2 == 0;
        let mine = package.eval_session()?;
        let theirs = opponent.eval_session()?;
        seats.push(if contender_first {
            LaneSeats {
                sessions: [mine, theirs],
                slots: [CONTENDER, OPPONENT],
            }
        } else {
            LaneSeats {
                sessions: [theirs, mine],
                slots: [OPPONENT, CONTENDER],
            }
        });
    }

    let mut encoders: Vec<Vec<Box<dyn Encoder>>> = Vec::with_capacity(config.threads.get());
    for _ in 0..config.threads.get() {
        encoders.push(vec![package.encoder(), opponent.encoder()]);
    }

    run_sweep(Sweep {
        spec: spec(config),
        lanes: seats,
        games,
        encoders,
        evaluators: vec![package.evaluator()?, opponent.evaluator()?],
        batch: config.batch,
        batch_wait: config.batch_wait,
        threads: config.threads,
        stop: &config.stop,
        records: None,
    })
}
