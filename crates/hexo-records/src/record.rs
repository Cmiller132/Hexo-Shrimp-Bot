//! Shard header and game-record types.

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
/// Version fields are validated by [`crate::ShardReader`] and are not repeated
/// in this value.
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
/// The record contains the game specification, complete adjudication payload,
/// and every accepted ply.
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
    /// [`RecordError::Unfinished`] if the game has no result.
    pub fn from_game(game: &Game) -> Result<Self, RecordError> {
        let result = game.result().ok_or(RecordError::Unfinished)?;
        Ok(Self {
            spec: *game.spec(),
            result,
            plies: game.plies().to_vec(),
        })
    }
}
