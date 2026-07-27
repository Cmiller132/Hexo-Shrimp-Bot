//! The on-disk game record: shards of finished games, written once and read strictly.
//!
//! One shard file holds every game produced by one (run, epoch, phase): a header
//! that pins the versions the games were played under, then one entry per game
//! carrying the [`GameSpec`](hexo_runner::GameSpec) it was played under, its
//! [`MatchResult`](hexo_runner::MatchResult) with the whole adjudication payload,
//! and its [`PlyRecord`](hexo_runner::PlyRecord)s.
//!
//! A record is training data and match evidence, so the reader refuses anything
//! it cannot account for: a version it was not built against, a game count the
//! file does not hold, a byte past the last game, a field the file ends inside.
//! [`verify`] goes further and replays the move list through the engine, which
//! is the only check that sees a shard whose bytes drifted somewhere every field
//! still parses from.
//!
//! ```
//! use hexo_records::{GameRecord, ShardHeader, ShardMode, ShardReader, ShardWriter, verify};
//! use hexo_runner::{Decision, Game, GameSpec, Reply, Step};
//! use std::num::NonZeroU32;
//!
//! // Play one short game out to its ply cap.
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
/// It moves when the layout moves. Formats here are not backward compatible — a
/// bump means the shards from the previous one are regenerated, not read by a
/// second decoder — so nothing in this crate branches on it beyond refusing a
/// file it does not equal.
pub const RECORDS_VERSION: u32 = 1;

/// The first four bytes of every shard file.
///
/// It answers "is this a shard at all", and nothing else: the version lives in
/// its own field immediately after it, so exactly one thing in the file states
/// the format version and a reader never has to adjudicate between two of them.
pub const MAGIC: [u8; 4] = *b"HXRC";
