//! Whole-game test driver for mock-package sessions.
//!
//! Each round pumps leaves, encodes one batch, evaluates once, and resumes the
//! originating sessions.

#![allow(dead_code)]

use hexo_engine::ActionId;
use hexo_model::{ModelPackage, PackageError};
use hexo_model_mock::MockPackage;
use hexo_records::{GameRecord, ShardHeader, ShardMode, ShardWriter};
use hexo_runner::{Game, GameSpec, Reply, Step};
use hexo_search::{DecisionSession, EncodedBatch, Encoder, Evaluation, Evaluator, SessionStatus};
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

/// A game short enough for a test and long enough to be a game.
///
/// The cap is odd, so it falls on a placement that completed a turn rather than
/// a placement past one.
pub fn short_game(ply_cap: u32) -> GameSpec {
    GameSpec {
        ply_cap: NonZeroU32::new(ply_cap).expect("a nonzero cap"),
        ..GameSpec::default()
    }
}

/// A package configured with `config`, initialised into `dir`, and loaded.
pub fn loaded(dir: &Path, config: &str) -> Result<MockPackage, PackageError> {
    let mut package = MockPackage::from_config(config)?;
    package.init(dir)?;
    package.load(dir)?;
    Ok(package)
}

/// Play one whole game with two sessions and one evaluator.
///
/// Every round assembles one [`EncodedBatch`] and performs one `evaluate` call.
pub fn play(
    encoder: &dyn Encoder,
    evaluator: &mut dyn Evaluator,
    seats: &mut [Box<dyn DecisionSession>; 2],
    spec: GameSpec,
) -> Game {
    let mut game = Game::new(spec);
    let mut batch = EncodedBatch::new();
    let mut leaves = Vec::new();
    let mut answers: Vec<Evaluation> = Vec::new();

    loop {
        let Step::NeedDecision {
            seat, generation, ..
        } = game.step()
        else {
            break game;
        };

        let session = &mut seats[seat.index()];
        session.begin(game.position());
        loop {
            batch.clear();
            leaves.clear();
            let status = session.pump(&mut |leaf, position| {
                leaves.push(leaf);
                batch.push_with(encoder, position);
            });
            if status == SessionStatus::Decided {
                break;
            }
            assert!(
                !leaves.is_empty(),
                "a session awaiting evaluations that emitted nothing cannot make progress",
            );
            answers.clear();
            evaluator.evaluate(&batch, &mut answers);
            assert_eq!(answers.len(), batch.len(), "one answer per batch item");
            for (leaf, evaluation) in leaves.drain(..).zip(answers.drain(..)) {
                session.resume(leaf, evaluation);
            }
        }

        let decision = session
            .take_decision()
            .expect("a decided session has a decision");
        game.submit(generation, Reply::Place(decision))
            .expect("the driver submits against a fresh generation");
    }
}

/// Every probe position, encoded with `package`'s encoder and answered by its
/// evaluator: the package's whole observable behaviour, in one call.
pub fn answers(package: &MockPackage) -> Vec<Evaluation> {
    let encoder = package.encoder();
    let mut evaluator = package.evaluator().expect("the package is loaded");
    let mut batch = EncodedBatch::new();
    for position in hexo_model::probe_positions() {
        batch.push_with(encoder.as_ref(), &position);
    }
    let mut out = Vec::new();
    evaluator.evaluate(&batch, &mut out);
    out
}

/// Play `count` self-play games with one package, all with the same weights.
pub fn self_play_games(package: &MockPackage, count: usize, ply_cap: u32) -> Vec<Game> {
    let encoder = package.encoder();
    let mut evaluator = package.evaluator().expect("the package is loaded");
    (0..count)
        .map(|_| {
            let mut seats = [
                package.self_play_session().expect("loaded"),
                package.self_play_session().expect("loaded"),
            ];
            play(
                encoder.as_ref(),
                evaluator.as_mut(),
                &mut seats,
                short_game(ply_cap),
            )
        })
        .collect()
}

/// Write `games` into a shard at `path`, with a header this package would write.
pub fn write_shard(path: &Path, epoch: u32, games: &[Game]) -> PathBuf {
    write_shard_for(path, epoch, games, "mock")
}

/// The same, with the header naming `package`.
pub fn write_shard_for(path: &Path, epoch: u32, games: &[Game], package: &str) -> PathBuf {
    let header = ShardHeader {
        mode: ShardMode::SelfPlay,
        run_id: "test".to_owned(),
        package: package.to_owned(),
        checkpoint: format!("epoch-{epoch}"),
        epoch,
        game_count: 0,
    };
    let mut writer = ShardWriter::create(path, &header).expect("a fresh shard path");
    for game in games {
        writer
            .append(&GameRecord::from_game(game).expect("a finished game"))
            .expect("the game encodes");
    }
    writer.finalize().expect("the shard closes");
    path.to_path_buf()
}

/// One entry of a decoded diagnostics table.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Entry {
    /// The placement this row is about.
    pub action: ActionId,
    /// Visits, for a visit table.
    pub visits: u32,
    /// The prior, for a prior table.
    pub prior: f32,
}

/// A decoded diagnostics blob: the kind tag, and one entry per candidate.
#[derive(Clone, Debug, PartialEq)]
pub struct Diagnostics {
    /// `0` for a visit table, `1` for a prior table.
    pub tag: u8,
    /// One row per legal action at the root, in canonical order.
    pub entries: Vec<Entry>,
}

/// Decode the package's documented diagnostics format.
///
/// This independent test decoder freezes the documented byte layout.
pub fn decode_diagnostics(bytes: &[u8]) -> Diagnostics {
    let tag = bytes[0];
    let count = u32::from_le_bytes(bytes[1..5].try_into().expect("four bytes")) as usize;
    assert_eq!(bytes.len(), 5 + count * 8, "the table has trailing bytes");
    let entries = (0..count)
        .map(|i| {
            let at = 5 + i * 8;
            let action = ActionId(u32::from_le_bytes(
                bytes[at..at + 4].try_into().expect("four bytes"),
            ));
            let raw: [u8; 4] = bytes[at + 4..at + 8].try_into().expect("four bytes");
            Entry {
                action,
                visits: u32::from_le_bytes(raw),
                prior: f32::from_le_bytes(raw),
            }
        })
        .collect();
    Diagnostics { tag, entries }
}
