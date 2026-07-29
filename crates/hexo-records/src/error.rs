//! Shard write, read, and verification errors.

use hexo_engine::{MoveError, Player};
use std::path::PathBuf;

/// Everything that can go wrong writing, reading, or verifying a shard.
///
/// Decoding failures carry absolute byte offsets, entry failures identify the
/// game index, and version failures carry expected and found values.
#[derive(Debug)]
pub enum RecordError {
    /// The filesystem refused an operation.
    Io {
        /// The file it was refused on.
        path: PathBuf,
        /// What the filesystem said.
        source: std::io::Error,
    },
    /// A path the writer would have created already exists. Shards are written
    /// once, never appended to and never silently replaced.
    AlreadyExists {
        /// The path that is already taken.
        path: PathBuf,
    },
    /// The file does not open with [`crate::MAGIC`].
    BadMagic {
        /// The four bytes that were there instead.
        found: [u8; 4],
    },
    /// The file states a shard format version this build does not implement.
    RecordsVersion {
        /// [`crate::RECORDS_VERSION`], which is what this build reads and writes.
        expected: u32,
        /// What the file states.
        found: u32,
    },
    /// The games were played under different rules than this build links.
    RulesVersion {
        /// `hexo_engine::RULES_VERSION` as linked.
        expected: u32,
        /// What the file states.
        found: u32,
    },
    /// The games were played under a different action ordering than this build
    /// links, so their action ids do not mean what this build would take them to.
    ActionOrderVersion {
        /// `hexo_engine::ACTION_ORDER_VERSION` as linked.
        expected: u32,
        /// What the file states.
        found: u32,
    },
    /// The games were played under a different runner decision and result model
    /// than this build links.
    ProtocolVersion {
        /// `hexo_runner::PROTOCOL_VERSION` as linked.
        expected: u32,
        /// What the file states.
        found: u32,
    },
    /// A header handed to [`crate::ShardWriter::create`] stated a nonzero game
    /// count; the writer owns and patches this field.
    HeaderGameCount {
        /// The count the caller stated.
        found: u32,
    },
    /// The file, or a game entry within it, ended inside a field.
    Truncated {
        /// Where the field starts.
        offset: u64,
        /// How many bytes it needs.
        needed: usize,
        /// How many are left.
        available: usize,
    },
    /// Bytes follow the last game the header declared.
    TrailingBytes {
        /// Where the surplus starts.
        offset: u64,
        /// How much of it there is.
        extra: u64,
    },
    /// A game entry decoded without consuming the whole length its prefix declared.
    EntryTrailingBytes {
        /// Which game entry, counting from zero.
        game: u32,
        /// Where the unread bytes start.
        offset: u64,
        /// How many of them there are.
        remaining: usize,
    },
    /// The header's game count is not the number of entries the file holds.
    GameCountMismatch {
        /// What the header declares.
        declared: u32,
        /// How many entries were actually there.
        found: u32,
    },
    /// A discriminant the format does not define.
    BadTag {
        /// Which enum it was read for.
        field: &'static str,
        /// The value that is not one of its tags.
        tag: u8,
        /// Where the tag byte is.
        offset: u64,
    },
    /// A game states a ply cap of zero, which the runner's nonzero-by-type cap
    /// cannot hold.
    ZeroPlyCap {
        /// Where the cap field is.
        offset: u64,
    },
    /// A length-prefixed string is not UTF-8.
    Utf8 {
        /// Which header string.
        field: &'static str,
        /// Where its bytes start.
        offset: u64,
    },
    /// A header string is longer than the format's `u16` length prefix can state.
    StringTooLong {
        /// Which header string.
        field: &'static str,
        /// How long it is.
        len: usize,
    },
    /// A ply's diagnostics are longer than the format's `u32` length prefix can state.
    DiagnosticsTooLong {
        /// How long they are.
        len: usize,
    },
    /// A game has more plies than the format's `u32` count can state.
    TooManyPlies {
        /// How many it has.
        count: usize,
    },
    /// A game encodes to more bytes than the format's `u32` entry prefix can state.
    GameTooLarge {
        /// How many bytes it encodes to.
        bytes: usize,
    },
    /// A shard would hold more games than the format's `u32` count can state.
    TooManyGames,
    /// A [`Budget::Wall`](hexo_runner::Budget::Wall) duration does not fit the
    /// format's `u64` nanoseconds. Truncating it would record a budget the game
    /// was not played under.
    WallBudgetOverflow {
        /// The duration's nanoseconds.
        nanos: u128,
    },
    /// [`GameRecord::from_game`](crate::GameRecord::from_game) was handed a game
    /// that has not ended.
    Unfinished,
    /// A ply names a mover the replayed position does not have on turn.
    SeatMismatch {
        /// Which ply, counting from zero.
        ply: usize,
        /// The seat the record names.
        recorded: Player,
        /// The seat the replay has on turn.
        replayed: Player,
    },
    /// A recorded placement is not legal in the position the record replays into.
    ReplayRefused {
        /// Which ply, counting from zero.
        ply: usize,
        /// Why the engine refused it.
        cause: MoveError,
    },
    /// A ply's recorded hash is not the hash the replay reaches.
    ZobristMismatch {
        /// Which ply, counting from zero.
        ply: usize,
        /// The hash the record states.
        recorded: u64,
        /// The hash replaying the move list produces.
        replayed: u64,
    },
    /// The record claims a six-in-a-row win the replayed position does not show.
    NotTerminal,
    /// The record claims an ending other than six in a row, but the replayed
    /// position is a win — which the runner would have adjudicated as one.
    UnexpectedTerminal,
    /// The record names a winner the replayed position disagrees with.
    WinnerMismatch {
        /// The seat the record names.
        recorded: Player,
        /// The seat that owns the completed window.
        replayed: Player,
    },
}

impl core::fmt::Display for RecordError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "{}: {source}", path.display()),
            Self::AlreadyExists { path } => write!(
                f,
                "{} already exists; a shard is written once, never appended to",
                path.display()
            ),
            Self::BadMagic { found } => write!(
                f,
                "not a hexo shard: opens with {found:02x?}, expected {:02x?}",
                crate::MAGIC
            ),
            Self::RecordsVersion { expected, found } => write!(
                f,
                "shard states record format version {found}, but this build reads and writes {expected}"
            ),
            Self::RulesVersion { expected, found } => write!(
                f,
                "shard was played under rules version {found}, but this build links rules version {expected}"
            ),
            Self::ActionOrderVersion { expected, found } => write!(
                f,
                "shard was played under action-order version {found}, but this build links action-order version {expected}"
            ),
            Self::ProtocolVersion { expected, found } => write!(
                f,
                "shard was played under runner protocol version {found}, but this build links protocol version {expected}"
            ),
            Self::HeaderGameCount { found } => write!(
                f,
                "a header handed to the writer must state a game count of 0, not {found}: the writer counts what it wrote"
            ),
            Self::Truncated {
                offset,
                needed,
                available,
            } => write!(
                f,
                "truncated at offset {offset}: {needed} bytes needed, {available} left"
            ),
            Self::TrailingBytes { offset, extra } => write!(
                f,
                "{extra} bytes at offset {offset} follow the last game the header declares"
            ),
            Self::EntryTrailingBytes {
                game,
                offset,
                remaining,
            } => write!(
                f,
                "game {game} decoded with {remaining} bytes of its entry unread, at offset {offset}"
            ),
            Self::GameCountMismatch { declared, found } => write!(
                f,
                "the header declares {declared} games, but the shard holds {found}"
            ),
            Self::BadTag { field, tag, offset } => {
                write!(f, "unknown {field} tag {tag} at offset {offset}")
            }
            Self::ZeroPlyCap { offset } => {
                write!(f, "the ply cap at offset {offset} is zero")
            }
            Self::Utf8 { field, offset } => {
                write!(f, "the {field} string at offset {offset} is not UTF-8")
            }
            Self::StringTooLong { field, len } => write!(
                f,
                "the {field} string is {len} bytes; the format states one in at most {}",
                u16::MAX
            ),
            Self::DiagnosticsTooLong { len } => write!(
                f,
                "a ply carries {len} bytes of diagnostics; the format states at most {}",
                u32::MAX
            ),
            Self::TooManyPlies { count } => write!(
                f,
                "a game has {count} plies; the format states at most {}",
                u32::MAX
            ),
            Self::GameTooLarge { bytes } => write!(
                f,
                "a game encodes to {bytes} bytes; an entry may be at most {}",
                u32::MAX
            ),
            Self::TooManyGames => {
                write!(f, "a shard holds at most {} games", u32::MAX)
            }
            Self::WallBudgetOverflow { nanos } => write!(
                f,
                "a wall budget of {nanos} ns does not fit the format's u64 nanoseconds"
            ),
            Self::Unfinished => f.write_str("the game has no result yet, so it is not a record"),
            Self::SeatMismatch {
                ply,
                recorded,
                replayed,
            } => write!(
                f,
                "ply {ply} records {recorded:?} as the mover, but the replay has {replayed:?} on turn"
            ),
            Self::ReplayRefused { ply, cause } => write!(f, "ply {ply} does not replay: {cause}"),
            Self::ZobristMismatch {
                ply,
                recorded,
                replayed,
            } => write!(
                f,
                "ply {ply} records hash {recorded:#018x}, but the replay reaches {replayed:#018x}"
            ),
            Self::NotTerminal => f.write_str(
                "the record claims a six-in-a-row win, but the replayed position is not terminal",
            ),
            Self::UnexpectedTerminal => f.write_str(
                "the record claims an ending other than six in a row, but the replayed position is a win",
            ),
            Self::WinnerMismatch { recorded, replayed } => write!(
                f,
                "the record names {recorded:?} the winner, but the replayed position is won by {replayed:?}"
            ),
        }
    }
}

impl core::error::Error for RecordError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::ReplayRefused { cause, .. } => Some(cause),
            _ => None,
        }
    }
}
