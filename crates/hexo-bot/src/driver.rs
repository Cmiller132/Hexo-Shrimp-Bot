//! The batched sweep: many games in flight, a pool sized to the silicon, and
//! one evaluator crossing per batch.
//!
//! One implementation, shared verbatim by self-play, by evaluation, and by
//! `match` — `docs/CONTAINER_SPEC.md` §8's "one implementation of the loop, not
//! a loop plus a set of pieces that could drift from it". What differs between
//! the three is what a caller hands in: which sessions fill the seats, how many
//! evaluator slots there are, and whether finished games are written down.
//!
//! # Topology (§7.1)
//!
//! ```text
//!   lanes: one Game + two DecisionSessions each, held as data, not as threads
//!        |                                     ^
//!        | (slot, lane, leaf ids, bytes)       | the same lane, carrying its
//!        v  bounded sync_channel               | evaluations, re-armed
//!   batcher, 1 thread ---- Evaluator::evaluate ----> the device crossing
//!        |
//!        | finished games, bounded sync_channel
//!        v
//!   writer, 1 thread ---> records/<epoch>/shard-0000.hxr
//! ```
//!
//! **There is no thread per game.** A lane is a slot, not a stack: a session
//! with leaves outstanding is a struct holding a few vectors, so the number of
//! concurrent games is bounded by memory rather than by scheduler pressure. That
//! is the whole architecture, and it is what `hexo-runner` inverted its loop for
//! and what `hexo-search` made its sessions nonblocking for.
//!
//! # Why it cannot deadlock
//!
//! **A lane is a token, and there are exactly as many tokens as lanes.** A lane
//! is in exactly one place at any moment: the ready queue, one worker's hand,
//! the job channel, or the batcher's slate. So the ready queue's occupancy can
//! never exceed the lane count — it is bounded structurally rather than by a
//! capacity nobody could pick — and the batcher can therefore *always* hand a
//! lane back without blocking.
//!
//! That is the property the whole thing rests on. Only workers ever block on a
//! queue: on the job channel when the batcher is behind, and on the record
//! channel when the writer is. Both of those consumers run to completion without
//! ever blocking on a producer, so there is no cycle of waits to close. The
//! backpressure a saturated device applies therefore stops the workers, which is
//! the point, instead of growing a queue until the process is killed.
//!
//! Draining is the same argument from the other end. When the phase's quota is
//! met, a lane whose game ends retires instead of restarting; workers wait on a
//! condition variable — never a spin — until every lane has retired, which the
//! in-flight evaluations are free to make happen because the batcher is still
//! running. A worker leaves only when no lane exists anywhere.

use crate::Outcome;
use crate::error::BotError;
use crate::metrics::{SlotId, Tally};
use hexo_records::{GameRecord, RecordError, ShardHeader, ShardWriter};
use hexo_runner::{Decision, Failure, Game, GameSpec, Reply, Step, SubmitError};
use hexo_search::{
    DecisionSession, EncodedBatch, Encoder, Evaluation, Evaluator, LeafId, SessionStatus,
};
use std::collections::VecDeque;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, sync_channel};
use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// How many jobs and records may be queued per worker before it blocks.
///
/// Small on purpose. The queues exist to keep the batcher fed across a
/// scheduling hiccup, not to store work: a deep queue would hide the device
/// being the bottleneck by letting the workers run ahead into memory.
const QUEUE_PER_WORKER: usize = 2;

/// The two sessions filling one lane's seats, and where each of them gets its
/// answers.
pub(crate) struct LaneSeats {
    /// Indexed by `hexo_engine::Player::index`.
    pub sessions: [Box<dyn DecisionSession>; 2],
    /// The evaluator slot each seat's leaves are answered by.
    pub slots: [SlotId; 2],
}

/// Where finished games go.
pub(crate) struct RecordSink {
    /// The shard file, which does not exist yet.
    pub path: PathBuf,
    /// Its header. `game_count` must be zero; the writer counts what it wrote.
    pub header: ShardHeader,
}

/// Everything one sweep needs, and the whole of what varies between the three.
pub(crate) struct Sweep<'a> {
    /// The rules every game is played under.
    pub spec: GameSpec,
    /// One entry per lane.
    pub lanes: Vec<LaneSeats>,
    /// How many games the phase wants. Never fewer than the lane count: a lane
    /// whose game ends restarts while games are still owed and retires after.
    pub games: usize,
    /// `encoders[worker][slot]`. An encoder runs worker-side and takes `&self`,
    /// so every thread in the pool holds its own for every slot.
    pub encoders: Vec<Vec<Box<dyn Encoder>>>,
    /// One evaluator per slot. The batcher thread owns all of them, because the
    /// crossing has to be serialised anyway.
    pub evaluators: Vec<Box<dyn Evaluator>>,
    /// How many encoded items one evaluator call answers.
    pub batch: NonZeroUsize,
    /// How long the batcher waits for a partial batch to fill.
    pub batch_wait: Duration,
    /// Worker threads.
    pub threads: NonZeroUsize,
    /// Watched by every worker and by the batcher.
    pub stop: &'a AtomicBool,
    /// `Some` for self-play; `None` for evaluation and for `match`, neither of
    /// which produces training data.
    pub records: Option<RecordSink>,
}

/// What a sweep did.
pub(crate) struct SweepReport {
    /// Whether the quota was met or the stop flag cut it short.
    pub outcome: Outcome,
    /// The games that finished, however the sweep ended.
    pub tally: Tally,
    /// Evaluator calls the batcher made.
    pub batches: usize,
    /// Positions those calls answered.
    pub evaluations: usize,
}

/// Drive a whole phase.
///
/// # Errors
///
/// [`BotError::Record`] if the shard could not be created, extended, or closed,
/// and [`BotError::ThreadPanicked`] if a worker, the batcher, or the writer died
/// — in which case the sweep has no result it is entitled to describe.
///
/// # Panics
///
/// If the sweep is malformed: no lanes, more lanes than games, or an encoder
/// table that does not have one row per worker and one column per evaluator
/// slot. Those are this crate's own bugs, and each of them would otherwise
/// surface as a batch answered by the wrong network.
pub(crate) fn run_sweep(sweep: Sweep<'_>) -> Result<SweepReport, BotError> {
    let Sweep {
        spec,
        lanes,
        games,
        encoders,
        evaluators,
        batch,
        batch_wait,
        threads,
        stop,
        records,
    } = sweep;

    assert!(!lanes.is_empty(), "a sweep with no lanes plays no games");
    assert!(
        lanes.len() <= games,
        "{} lanes for {games} games; a lane would have to play a fraction of one",
        lanes.len(),
    );
    assert_eq!(
        encoders.len(),
        threads.get(),
        "the encoder table has one row per worker",
    );
    for row in &encoders {
        assert_eq!(
            row.len(),
            evaluators.len(),
            "the encoder table has one column per evaluator slot",
        );
    }

    // Held by reference from here on, so every scoped thread captures the same
    // flag by copying a borrow rather than trying to take the flag itself.
    let aborted = AtomicBool::new(false);
    let abort = &aborted;

    let mut queue = Ready {
        lanes: VecDeque::with_capacity(lanes.len()),
        outstanding: 0,
        remaining: games - lanes.len(),
        tally: Tally::default(),
        halted: false,
    };
    for (index, seats) in lanes.into_iter().enumerate() {
        let mut lane = Lane::new(index, spec, seats);
        lane.reseed();
        queue.lanes.push_back(lane);
    }
    let pool = Pool {
        queue: Mutex::new(queue),
        wake: Condvar::new(),
        arenas: Mutex::new(Vec::new()),
    };

    let depth = threads.get() * QUEUE_PER_WORKER;
    let (jobs_tx, jobs_rx) = sync_channel::<Job>(depth);
    let (records_tx, records_rx) = match records {
        Some(_) => {
            let (tx, rx) = sync_channel::<GameRecord>(depth);
            (Some(tx), Some(rx))
        }
        None => (None, None),
    };

    let (stats, written, workers_lived) =
        std::thread::scope(|scope| {
            let scribe = records.map(|sink| {
                let rx = records_rx.expect("a record sink implies its receiver");
                scope.spawn(|| write_shard(rx, sink, stop, abort))
            });
            let pool = &pool;
            let batcher = scope.spawn(move || {
                run_batcher(
                    jobs_rx,
                    evaluators,
                    pool,
                    Fill {
                        size: batch.get(),
                        wait: batch_wait,
                    },
                    stop,
                    abort,
                )
            });
            let mut workers = Vec::with_capacity(threads.get());
            for row in encoders {
                let jobs = jobs_tx.clone();
                let records = records_tx.clone();
                workers.push(scope.spawn(move || {
                    run_worker(pool, row, spec, Outbox { jobs, records }, stop, abort)
                }));
            }
            // The batcher learns the phase is over by the job channel disconnecting,
            // and the writer by the record channel disconnecting, so the sweep must
            // not keep a sender of either alive.
            drop(jobs_tx);
            drop(records_tx);

            let mut lived = true;
            for worker in workers {
                lived &= worker.join().is_ok();
            }
            (batcher.join(), scribe.map(|s| s.join()), lived)
        });

    if !workers_lived {
        return Err(BotError::ThreadPanicked { what: "a worker" });
    }
    let stats = stats.map_err(|_| BotError::ThreadPanicked {
        what: "the batcher",
    })?;
    if let Some(written) = written {
        written.map_err(|_| BotError::ThreadPanicked {
            what: "the record writer",
        })??;
    }

    let ready = pool
        .queue
        .into_inner()
        .expect("every thread that held the lane queue has been joined");
    let outcome = if stop.load(Ordering::Relaxed) {
        Outcome::Stopped
    } else {
        assert_eq!(
            ready.tally.games, games,
            "the sweep was not stopped and finished {} of {games} games",
            ready.tally.games,
        );
        Outcome::Completed
    };
    Ok(SweepReport {
        outcome,
        tally: ready.tally,
        batches: stats.batches,
        evaluations: stats.items,
    })
}

/// One game in flight, and everything about it that is not shared.
struct Lane {
    /// Which lane this is. Fixed for the sweep, and mixed into every seed.
    index: usize,
    /// The authoritative game. Only this lane's worker ever advances it.
    game: Game,
    /// The seats, indexed by `hexo_engine::Player::index`.
    sessions: [Box<dyn DecisionSession>; 2],
    /// Which evaluator slot answers each seat.
    slots: [SlotId; 2],
    /// The seat mid-decision and the generation its decision must carry, or
    /// `None` between decisions.
    open: Option<(usize, u64)>,
    /// Leaves emitted for the open decision, in the order they were emitted.
    leaves: Vec<LeafId>,
    /// Their answers, in the same order, delivered by the batcher.
    answers: Vec<Evaluation>,
    /// How many games this lane has already played, so a restart cannot reuse a
    /// seed.
    serial: u64,
}

impl Lane {
    /// A lane at the start of its first game.
    fn new(index: usize, spec: GameSpec, seats: LaneSeats) -> Self {
        Self {
            index,
            game: Game::new(spec),
            sessions: seats.sessions,
            slots: seats.slots,
            open: None,
            leaves: Vec::new(),
            answers: Vec::new(),
            serial: 0,
        }
    }

    /// Start the next game on this lane.
    ///
    /// The seats swap along with their evaluator slots, so a lane playing many
    /// games alternates colours across them exactly as the lanes of one round
    /// alternate colours across each other. Fixed colours would let a quota that
    /// is not a multiple of twice the lane count hand one competitor more first
    /// moves than the other — the bias the alternation exists to cancel.
    fn restart(&mut self, spec: GameSpec) {
        self.game = Game::new(spec);
        self.sessions.swap(0, 1);
        self.slots.swap(0, 1);
        self.open = None;
        self.leaves.clear();
        self.answers.clear();
        self.serial += 1;
        self.reseed();
    }

    /// Give both seats a fresh stream.
    ///
    /// `docs/CONTAINER_SPEC.md` §12: the driver seeds sessions from entropy,
    /// games are deliberately non-deterministic, and nothing mints or records a
    /// per-game seed. `OPEN_DECISIONS.md` B4 is the deferral, and
    /// `DecisionSession::reseed` is the seam it will land on — a recorded seed
    /// that does not reproduce the game is worse than none, because it reads as
    /// a guarantee nobody ever checked.
    fn reseed(&mut self) {
        for seat in 0..2 {
            let seed = entropy_seed(self.index, self.serial, seat);
            self.sessions[seat].reseed(seed);
        }
    }
}

/// A seed for one seat of one game, from the clock and the seat's identity.
///
/// The lane index and the seat are mixed in because the clock is not: two lanes
/// reseeded inside the same nanosecond, or the two seats of one game, must not
/// get the same stream. Two seats sharing an RNG would correlate their choices,
/// which is the failure `hexo-player` and this crate both refuse to allow by
/// construction.
fn entropy_seed(lane: usize, serial: u64, seat: usize) -> u64 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_nanos() as u64);
    let identity = mix(lane as u64)
        ^ mix(serial.rotate_left(21))
        ^ mix((seat as u64).rotate_left(43))
        ^ 0x9e37_79b9_7f4a_7c15;
    mix(now ^ identity)
}

/// The splitmix64 finalizer, which is what makes a counter into a seed.
const fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// What a worker sends the batcher: the lane, the slot to answer it from, and
/// the bytes its leaves encoded to.
struct Job {
    /// The lane, which travels with its own question.
    lane: Lane,
    /// Which evaluator answers.
    slot: SlotId,
    /// The encoded leaves, in `lane.leaves` order.
    batch: EncodedBatch,
}

/// The lanes waiting for a worker, and what the sweep has left to do.
struct Ready {
    /// Lanes a worker may check out. Never longer than the lane count.
    lanes: VecDeque<Lane>,
    /// Lanes checked out, in the job channel, or on the batcher's slate.
    outstanding: usize,
    /// Games not yet started.
    remaining: usize,
    /// Games that finished.
    tally: Tally,
    /// Set once the sweep is giving up; workers leave at their next checkout.
    halted: bool,
}

/// The shared state of one sweep.
struct Pool {
    /// The ready queue and the sweep's bookkeeping, under one lock.
    queue: Mutex<Ready>,
    /// Signalled whenever a lane becomes available or the sweep ends. Workers
    /// wait on it rather than polling: a spin here would burn the cores the
    /// sweep exists to keep busy.
    wake: Condvar,
    /// Encoding arenas in circulation, so a pump allocates nothing
    /// steady-state (`docs/CONTAINER_SPEC.md` §8.1: no cross-epoch growth).
    arenas: Mutex<Vec<EncodedBatch>>,
}

impl Pool {
    /// The ready queue, or a panic naming why it could not be taken.
    fn lock(&self) -> MutexGuard<'_, Ready> {
        self.queue
            .lock()
            .expect("a thread panicked while holding the lane queue")
    }

    /// Take a lane to work on, or `None` once there is nothing left to take.
    ///
    /// `None` means one of two things and no third: the sweep has been halted,
    /// or every lane has retired. A lane that is merely busy elsewhere is
    /// counted in `outstanding`, so its absence from the queue is a wait rather
    /// than an ending.
    fn checkout(&self) -> Option<Lane> {
        let mut ready = self.lock();
        loop {
            if ready.halted {
                return None;
            }
            if let Some(lane) = ready.lanes.pop_front() {
                ready.outstanding += 1;
                return Some(lane);
            }
            if ready.outstanding == 0 {
                return None;
            }
            ready = self
                .wake
                .wait(ready)
                .expect("a thread panicked while holding the lane queue");
        }
    }

    /// Hand a lane back with its evaluations attached.
    fn rearm(&self, lane: Lane) {
        let mut ready = self.lock();
        ready.lanes.push_back(lane);
        ready.outstanding -= 1;
        drop(ready);
        self.wake.notify_all();
    }

    /// Count a finished game, and either start the next one on that lane or
    /// retire it.
    fn settle(&self, mut lane: Lane, spec: GameSpec) {
        let result = lane
            .game
            .result()
            .expect("a lane is only settled once its game has a result");
        let plies = lane.game.plies().len();
        let slots = lane.slots;

        let mut ready = self.lock();
        ready.tally.record(result, plies, slots);
        if ready.remaining > 0 && !ready.halted {
            ready.remaining -= 1;
            lane.restart(spec);
            ready.lanes.push_back(lane);
        }
        ready.outstanding -= 1;
        drop(ready);
        self.wake.notify_all();
    }

    /// Stop the sweep, and wake everything waiting for work that is not coming.
    ///
    /// The one operation that takes the queue *tolerantly*, because it is the
    /// one that has to work when a thread has died holding it: halting is what
    /// wakes the workers waiting for a lane that is never coming back, so the
    /// sweep can report the death instead of hanging on it. Every other
    /// operation takes the lock strictly, because a poisoned queue is state
    /// nothing should go on computing from.
    fn halt(&self) {
        let mut ready = self.queue.lock().unwrap_or_else(PoisonError::into_inner);
        ready.halted = true;
        drop(ready);
        self.wake.notify_all();
    }

    /// An empty encoding arena.
    fn take_arena(&self) -> EncodedBatch {
        self.arenas
            .lock()
            .expect("a thread panicked while holding the arena pool")
            .pop()
            .unwrap_or_default()
    }

    /// Return an arena for the next pump to fill.
    fn give_arena(&self, mut arena: EncodedBatch) {
        arena.clear();
        self.arenas
            .lock()
            .expect("a thread panicked while holding the arena pool")
            .push(arena);
    }
}

/// Whether the sweep should wind up: the operator asked, or something failed.
fn winding_up(stop: &AtomicBool, abort: &AtomicBool) -> bool {
    stop.load(Ordering::Relaxed) || abort.load(Ordering::Relaxed)
}

/// Ends the sweep when the thread holding it unwinds.
///
/// Panicking is how the layers below report a broken evaluation — a prior count
/// that does not match the position's legal set, a value outside `[-1, 1]`, an
/// answer to a question no session asked — and every one of those is a package
/// bug worth stopping for. What must not happen is that stopping becomes
/// *hanging*: a worker that dies with a lane checked out leaves the queue
/// counting a lane that no longer exists, and every other worker would wait on
/// the condition variable for it to come back. So the thread that unwinds halts
/// the sweep and sets the abort flag on its way out — the flag as well as the
/// halt, because a shard whose phase died must not be finalized as though the
/// phase had finished.
///
/// Only on a panic. A worker that returns normally has already halted, and
/// setting the abort flag then would tell the writer to throw away a shard that
/// is complete.
struct EndOnPanic<'a> {
    /// The sweep to halt.
    pool: &'a Pool,
    /// The flag that tells the writer this phase did not finish.
    abort: &'a AtomicBool,
}

impl Drop for EndOnPanic<'_> {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            return;
        }
        self.abort.store(true, Ordering::Relaxed);
        self.pool.halt();
    }
}

/// Where a worker sends what it produces.
struct Outbox {
    /// Encoded leaves, to the batcher.
    jobs: SyncSender<Job>,
    /// Finished games, to the writer, when the phase records any.
    records: Option<SyncSender<GameRecord>>,
}

/// One worker: check out a lane, deliver its answers, pump it, and either send
/// its next question or settle its finished game.
fn run_worker(
    pool: &Pool,
    encoders: Vec<Box<dyn Encoder>>,
    spec: GameSpec,
    out: Outbox,
    stop: &AtomicBool,
    abort: &AtomicBool,
) {
    let _guard = EndOnPanic { pool, abort };
    while !winding_up(stop, abort) {
        let Some(mut lane) = pool.checkout() else {
            break;
        };
        deliver(&mut lane);
        let mut arena = pool.take_arena();
        match advance(&mut lane, &encoders, &mut arena) {
            Disposition::Await(slot) => {
                if out
                    .jobs
                    .send(Job {
                        lane,
                        slot,
                        batch: arena,
                    })
                    .is_err()
                {
                    // The batcher is gone, so this lane's question will never be
                    // answered and no later one would be either.
                    break;
                }
            }
            Disposition::Finished => {
                pool.give_arena(arena);
                if let Some(records) = &out.records {
                    let record = GameRecord::from_game(&lane.game)
                        .expect("a lane is only settled once its game has a result");
                    if records.send(record).is_err() {
                        // The writer failed and has already said so; the sweep
                        // is over, and the game just played is not going to be
                        // recorded by anyone else.
                        abort.store(true, Ordering::Relaxed);
                    }
                }
                pool.settle(lane, spec);
            }
        }
    }
    // Whatever ended this worker ends the sweep: either every lane has retired,
    // in which case halting is a no-op, or something asked it to stop and the
    // workers still waiting need to hear about it.
    pool.halt();
}

/// Deliver the evaluations the batcher attached to a lane.
fn deliver(lane: &mut Lane) {
    if lane.answers.is_empty() {
        return;
    }
    let (seat, _) = lane
        .open
        .expect("a lane carrying answers is mid-decision, which is what asked for them");
    // Taken out so the session and its two buffers are three disjoint borrows;
    // both vectors go back with their allocations intact.
    let mut leaves = core::mem::take(&mut lane.leaves);
    let mut answers = core::mem::take(&mut lane.answers);
    assert_eq!(
        leaves.len(),
        answers.len(),
        "lane {} was handed {} answers to {} leaves",
        lane.index,
        answers.len(),
        leaves.len(),
    );
    for (leaf, evaluation) in leaves.drain(..).zip(answers.drain(..)) {
        lane.sessions[seat].resume(leaf, evaluation);
    }
    lane.leaves = leaves;
    lane.answers = answers;
}

/// What a worker's visit to a lane produced.
enum Disposition {
    /// The mover emitted leaves, which the given slot has to answer.
    Await(SlotId),
    /// The game ended.
    Finished,
}

/// Run a lane as far as it goes without the network.
///
/// Decisions that need no evaluation — a `PolicySession` whose root has already
/// been answered, an `MctsSession` that spent its whole budget on placements the
/// engine settled on the spot — are taken and submitted here, so a lane can make
/// several plies of progress on one visit.
///
/// The arena therefore only ever holds one pump's leaves, and so only ever one
/// seat's: a pump that reports `Decided` emitted nothing, because a session with
/// an answer outstanding is not decided. That is what lets a job name a single
/// evaluator slot.
fn advance(
    lane: &mut Lane,
    encoders: &[Box<dyn Encoder>],
    arena: &mut EncodedBatch,
) -> Disposition {
    loop {
        let (seat, generation) = match lane.open {
            Some(open) => open,
            None => match lane.game.step() {
                Step::NeedDecision {
                    seat, generation, ..
                } => {
                    lane.sessions[seat.index()].begin(&lane.game);
                    let open = (seat.index(), generation);
                    lane.open = Some(open);
                    open
                }
                Step::Finished(_) => return Disposition::Finished,
            },
        };

        let slot = lane.slots[seat];
        let encoder = encoders[slot].as_ref();
        let status = {
            let Lane {
                sessions, leaves, ..
            } = &mut *lane;
            sessions[seat].pump(&mut |leaf, position| {
                leaves.push(leaf);
                arena.push_with(encoder, position);
            })
        };

        match status {
            SessionStatus::Decided => {
                assert!(
                    arena.is_empty(),
                    "lane {}: a decided pump emitted leaves, which nothing will answer",
                    lane.index,
                );
                let decision = lane.sessions[seat]
                    .take_decision()
                    .expect("a decided session has a decision");
                submit(&mut lane.game, generation, decision);
                lane.open = None;
            }
            SessionStatus::AwaitingEvals { .. } => {
                assert!(
                    !arena.is_empty(),
                    "lane {}: a session awaiting evaluations emitted nothing, so the lane could \
                     never progress",
                    lane.index,
                );
                return Disposition::Await(slot);
            }
        }
    }
}

/// Submit a seat's decision verbatim.
///
/// The seat authored all three fields — the placement, the hash of the position
/// it searched, and its diagnostics — and this never fills one in on its behalf:
/// writing the hash would delete the desync detector, and writing the
/// diagnostics would invent the training annotations.
fn submit(game: &mut Game, generation: u64, decision: Decision) {
    match game.submit(generation, Reply::Place(decision)) {
        Ok(_) => {}
        Err(SubmitError::Desync { expected, got }) => {
            // The seat chose from a position that is not the game's. This driver
            // has nothing to resync — a transport adapter that can does so before
            // the decision reaches here — so it gives up on the turn and lets
            // policy adjudicate. In-process sessions copy the canonical position
            // in `begin`, so this cannot fire without a bug; it is here because
            // the alternative to adjudicating is guessing.
            game.submit(generation, Reply::Failed(Failure::Desync { expected, got }))
                .expect("a refused submission leaves the generation unchanged");
        }
        Err(error) => unreachable!("the generation was read from this game above: {error}"),
    }
}

/// How wide a batch the batcher fills, and how long it waits for one.
#[derive(Clone, Copy)]
struct Fill {
    /// Items after which a slot crosses without waiting further.
    size: usize,
    /// How long a partially filled slot waits for more.
    wait: Duration,
}

/// What the batcher did, for the metrics line.
#[derive(Clone, Copy, Default)]
struct BatchStats {
    /// Evaluator calls.
    batches: usize,
    /// Items those calls answered.
    items: usize,
}

/// One slot's half-filled batch, and the lanes whose leaves are in it.
#[derive(Default)]
struct Slate {
    /// The merged items, in the order the lanes arrived.
    batch: EncodedBatch,
    /// The lanes, in the same order.
    lanes: Vec<Lane>,
}

/// The one thread that owns the evaluators, and therefore the one place in the
/// process that would touch a device or an interpreter.
fn run_batcher(
    jobs: Receiver<Job>,
    mut evaluators: Vec<Box<dyn Evaluator>>,
    pool: &Pool,
    fill: Fill,
    stop: &AtomicBool,
    abort: &AtomicBool,
) -> BatchStats {
    let _guard = EndOnPanic { pool, abort };
    let mut slates: Vec<Slate> = (0..evaluators.len()).map(|_| Slate::default()).collect();
    let mut answers: Vec<Evaluation> = Vec::new();
    let mut stats = BatchStats::default();

    loop {
        // The batcher is the sweep's clock. It is the one thread that is never
        // blocked for longer than the flush window, so it is where a stop is
        // noticed and turned into a halt every other thread can see.
        if winding_up(stop, abort) {
            pool.halt();
        }
        match jobs.recv_timeout(fill.wait) {
            Ok(job) => {
                let slot = job.slot;
                let slate = &mut slates[slot];
                for item in job.batch.iter() {
                    slate.batch.push_bytes(item);
                }
                slate.lanes.push(job.lane);
                pool.give_arena(job.batch);
                if slate.batch.len() >= fill.size {
                    cross(
                        slot,
                        slate,
                        evaluators[slot].as_mut(),
                        &mut answers,
                        &mut stats,
                        pool,
                    );
                }
            }
            Err(RecvTimeoutError::Timeout) => {
                for (slot, slate) in slates.iter_mut().enumerate() {
                    cross(
                        slot,
                        slate,
                        evaluators[slot].as_mut(),
                        &mut answers,
                        &mut stats,
                        pool,
                    );
                }
            }
            Err(RecvTimeoutError::Disconnected) => {
                // Every worker has left, so nothing more will arrive. What is
                // already here is answered rather than dropped: on a clean drain
                // those lanes still have to retire, and on a stop the lanes are
                // discarded a moment later either way.
                for (slot, slate) in slates.iter_mut().enumerate() {
                    cross(
                        slot,
                        slate,
                        evaluators[slot].as_mut(),
                        &mut answers,
                        &mut stats,
                        pool,
                    );
                }
                break;
            }
        }
    }
    stats
}

/// Answer one slot's batch and hand every lane in it back.
///
/// This call is the single device or interpreter crossing the whole topology is
/// built around, and everything above exists to make it wide.
fn cross(
    slot: SlotId,
    slate: &mut Slate,
    evaluator: &mut dyn Evaluator,
    answers: &mut Vec<Evaluation>,
    stats: &mut BatchStats,
    pool: &Pool,
) {
    if slate.batch.is_empty() {
        return;
    }
    answers.clear();
    evaluator.evaluate(&slate.batch, answers);
    assert_eq!(
        answers.len(),
        slate.batch.len(),
        "the evaluator on slot {slot} answered {} items of a batch of {}; a package produces one \
         evaluation per item, in batch order, and pairing them up any other way would hand a \
         session another position's priors",
        answers.len(),
        slate.batch.len(),
    );
    stats.batches += 1;
    stats.items += slate.batch.len();

    let mut supply = answers.drain(..);
    for mut lane in slate.lanes.drain(..) {
        let wanted = lane.leaves.len();
        lane.answers.clear();
        lane.answers.extend(supply.by_ref().take(wanted));
        assert_eq!(
            lane.answers.len(),
            wanted,
            "lane {} asked {wanted} questions and the batch ran out of answers",
            lane.index,
        );
        pool.rearm(lane);
    }
    slate.batch.clear();
}

/// The one thread that touches a record file.
///
/// A shard that was abandoned is never finalized, so `ShardWriter`'s temporary
/// file is removed and nothing appears at the shard's name at all. That is
/// `docs/CONTAINER_SPEC.md` §8.1's rule for a stop mid-self-play: the games were
/// on-policy and are worthless without the fit that was going to consume them,
/// so a half epoch of them is not a smaller epoch, it is nothing.
fn write_shard(
    records: Receiver<GameRecord>,
    sink: RecordSink,
    stop: &AtomicBool,
    abort: &AtomicBool,
) -> Result<(), RecordError> {
    let mut writer = match ShardWriter::create(&sink.path, &sink.header) {
        Ok(writer) => writer,
        Err(error) => {
            // Nothing can be recorded, so nothing should go on being played.
            abort.store(true, Ordering::Relaxed);
            return Err(error);
        }
    };
    for record in records {
        if let Err(error) = writer.append(&record) {
            abort.store(true, Ordering::Relaxed);
            return Err(error);
        }
    }
    if winding_up(stop, abort) {
        drop(writer);
        return Ok(());
    }
    writer.finalize()
}
