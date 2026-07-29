//! Versioned on-disk shards of finished games.
//!
//! One shard file holds every game produced by one (run, epoch, phase): a header
//! that pins the versions the games were played under, then one entry per game
//! carrying the [`GameSpec`](hexo_runner::GameSpec) it was played under, its
//! [`MatchResult`](hexo_runner::MatchResult) with the whole adjudication payload,
//! and its [`PlyRecord`](hexo_runner::PlyRecord)s.
//!
//! Readers require matching versions, exact entry counts, complete fields, and
//! no trailing bytes. [`verify`] additionally replays engine semantics.
//!
//! ```
//! use hexo_records::{GameRecord, ShardHeader, ShardMode, ShardReader, ShardWriter, verify};
//! use hexo_runner::{Decision, Game, GameSpec, Reply, Step};
//! use std::num::NonZeroU32;
//!
//! // Play one short game to its ply cap.
//! let mut game = Game::new(GameSpec {
//!     ply_cap: NonZeroU32::new(5).expect("nonzero"),
//!     ..GameSpec::default()
//! });
//! while let Step::NeedDecision { generation, zobrist, .. } = game.step() {
//!     let action = game.position().nth_legal(0).expect("a legal move");
//!     game.submit(generation, Reply::Place(Decision::new(action, zobrist)))
//!         .expect("accepted");
//! }
//!
//! let dir = tempfile::tempdir()?;
//! let path = dir.path().join("0000.hxr");
//!
//! let mut writer = ShardWriter::create(&path, &ShardHeader {
//!     mode: ShardMode::SelfPlay,
//!     run_id: "demo".to_owned(),
//!     package: "mock".to_owned(),
//!     checkpoint: "epoch-0".to_owned(),
//!     epoch: 0,
//!     game_count: 0,
//! })?;
//! writer.append(&GameRecord::from_game(&game)?)?;
//! writer.finalize()?;
//!
//! let reader = ShardReader::open(&path)?;
//! assert_eq!(reader.header().game_count, 1);
//! assert_eq!(reader.header().mode, ShardMode::SelfPlay);
//! for record in reader {
//!     verify(&record?)?;
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod codec;
pub mod error;
mod format;
pub mod reader;
pub mod record;
pub mod replay;
pub mod writer;

pub use error::RecordError;
pub use reader::ShardReader;
pub use record::{GameRecord, ShardHeader, ShardMode};
pub use replay::verify;
pub use writer::ShardWriter;

/// Version of the shard format itself: the byte layout, the tag numbering, and
/// which fields are present.
///
/// Increment this for any layout, tag, or field-set change. The reader accepts
/// only an exact version match.
pub const RECORDS_VERSION: u32 = 1;

/// The first four bytes of every shard file.
///
/// The format version follows in its own field.
pub const MAGIC: [u8; 4] = *b"HXRC";
