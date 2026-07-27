//! What a shard carries: the header it opens with, and one record per game.

use crate::error::RecordError;
use hexo_runner::{Game, GameSpec, MatchResult, PlyRecord};

/// Which phase of a run produced a shard.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShardMode {
    /// Games a package played against itself to produce training data.
    SelfPlay,
    /// Games played to measure one checkpoint against another.
    Eval,
}

/// The preamble of a shard file: what the games in it belong to.
///
/// The four version numbers the file also carries — the record format, the
/// rules, the action ordering, and the runner protocol — are deliberately not
/// fields here. A [`crate::ShardReader`] has already proved each one equals the
/// constant this build links against before it hands the header out, so a copy
/// could only repeat a constant, and a copy that could be read as data invites
/// code that branches on it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardHeader {
    /// Which phase produced these games.
    pub mode: ShardMode,
    /// The training run the games belong to.
    pub run_id: String,
    /// The model package that played them.
    pub package: String,
    /// The checkpoint reference the package was loaded from.
    pub checkpoint: String,
    /// The epoch within the run.
    pub epoch: u32,
    /// How many games the shard holds.
    ///
    /// [`crate::ShardWriter::create`] requires this to be zero and refuses a
    /// header that presets it: the writer counts what it wrote and patches the
    /// true value in on finalize. A reader reports what the file states, having
    /// already checked that the file holds exactly that many entries.
    pub game_count: u32,
}

/// One finished game, whole.
///
/// The spec says what the game was played under, the result carries the entire
/// adjudication payload — including the action and [`MoveError`](hexo_engine::MoveError)
/// behind an illegal-move loss, and both hashes behind a desync — and the plies
/// are the game's history. Nothing is summarised: a verdict whose reasons were
/// discarded is one an operator cannot debug and a training pipeline cannot
/// filter on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GameRecord {
    /// The match rules the game was played under.
    pub spec: GameSpec,
    /// How it ended.
    pub result: MatchResult,
    /// Every placement, oldest first.
    pub plies: Vec<PlyRecord>,
}

impl GameRecord {
    /// Take the record out of a finished game.
    ///
    /// # Errors
    ///
    /// [`RecordError::Unfinished`] if the game has no result yet. An unfinished
    /// game is not a record: its move list is a prefix, its adjudication has not
    /// happened, and writing it would put a game nobody finished into training
    /// data as though someone had.
    pub fn from_game(game: &Game) -> Result<Self, RecordError> {
        let result = game.result().ok_or(RecordError::Unfinished)?;
        Ok(Self {
            spec: *game.spec(),
            result,
            plies: game.plies().to_vec(),
        })
    }
}
