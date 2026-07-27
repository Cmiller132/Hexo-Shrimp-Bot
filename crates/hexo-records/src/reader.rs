//! Reading a shard, strictly.

use crate::codec::Cursor;
use crate::error::RecordError;
use crate::format::{ENTRY_PREFIX_BYTES, HEADER_MAX_BYTES, decode_game, decode_header};
use crate::record::{GameRecord, ShardHeader};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// Reads one shard, refusing anything it cannot account for.
///
/// Opening validates the magic and all four versions against the constants this
/// build links, so a shard produced by a different engine, action ordering,
/// runner protocol, or record format is refused before a single game is decoded
/// rather than silently reinterpreted. Iterating yields exactly the games the
/// header declares: a file that ends early, holds a different number of entries,
/// or carries a byte past the last of them is an error, never a short read that
/// looks like a smaller shard.
///
/// What none of that can see is a shard whose bytes drifted somewhere every
/// field still parses from. [`crate::verify`] is the detector for that, and it
/// is a separate call because it costs a replay per game.
pub struct ShardReader {
    /// The file, positioned at the next entry.
    file: BufReader<File>,
    /// Which file, for error context.
    path: PathBuf,
    /// The validated header.
    header: ShardHeader,
    /// The file's length, read once at open. A shard is written once and never
    /// appended to, so this does not go stale under the reader.
    len: u64,
    /// Where the next entry starts.
    offset: u64,
    /// How many entries have been decoded.
    games_read: u32,
    /// True once iteration has ended, by exhaustion or by error.
    done: bool,
    /// Reused entry buffer, so decoding a game allocates nothing steady-state.
    scratch: Vec<u8>,
}

impl ShardReader {
    /// Open a shard and validate its header.
    ///
    /// # Errors
    ///
    /// [`RecordError::BadMagic`] if the file is not a shard;
    /// [`RecordError::RecordsVersion`], [`RecordError::RulesVersion`],
    /// [`RecordError::ActionOrderVersion`], or [`RecordError::ProtocolVersion`]
    /// if it was written by a build that differs from this one, each naming both
    /// numbers; [`RecordError::Truncated`] if the file ends inside the header;
    /// [`RecordError::Io`] for anything the filesystem refuses.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, RecordError> {
        let path = path.as_ref().to_path_buf();
        let io = |source: std::io::Error| RecordError::Io {
            path: path.clone(),
            source,
        };

        let file = File::open(&path).map_err(io)?;
        let len = file.metadata().map_err(io)?.len();
        let mut file = BufReader::new(file);

        // The header is variable-length, so the whole of what one could possibly
        // occupy is read up front and decoded with the same cursor every other
        // shape uses. A streaming header reader would be a second statement of
        // the layout, and the peek is bounded by three `u16`-capped strings.
        let peek = len.min(HEADER_MAX_BYTES);
        let mut bytes = Vec::new();
        file.by_ref()
            .take(peek)
            .read_to_end(&mut bytes)
            .map_err(io)?;

        let mut cursor = Cursor::new(&bytes, 0);
        let header = decode_header(&mut cursor)?;
        let offset = cursor.offset();
        file.seek(SeekFrom::Start(offset)).map_err(io)?;

        Ok(Self {
            file,
            path,
            header,
            len,
            offset,
            games_read: 0,
            done: false,
            scratch: Vec::new(),
        })
    }

    /// The validated header.
    #[must_use]
    pub const fn header(&self) -> &ShardHeader {
        &self.header
    }

    /// The next game, `Ok(None)` at a clean end of the shard.
    fn next_entry(&mut self) -> Result<Option<GameRecord>, RecordError> {
        if self.games_read == self.header.game_count {
            if self.offset < self.len {
                return Err(RecordError::TrailingBytes {
                    offset: self.offset,
                    extra: self.len - self.offset,
                });
            }
            return Ok(None);
        }

        let left = self.len - self.offset;
        if left < ENTRY_PREFIX_BYTES as u64 {
            // A file that stops exactly on an entry boundary holds fewer games
            // than the header claims; one that stops inside a prefix is torn.
            return Err(if left == 0 {
                RecordError::GameCountMismatch {
                    declared: self.header.game_count,
                    found: self.games_read,
                }
            } else {
                RecordError::Truncated {
                    offset: self.offset,
                    needed: ENTRY_PREFIX_BYTES,
                    available: left as usize,
                }
            });
        }

        let mut prefix = [0u8; ENTRY_PREFIX_BYTES];
        self.file
            .read_exact(&mut prefix)
            .map_err(|source| RecordError::Io {
                path: self.path.clone(),
                source,
            })?;
        let entry = u32::from_le_bytes(prefix) as usize;

        // The declared length is checked against what the file actually holds
        // before it is used as an allocation size.
        let body = self.offset + ENTRY_PREFIX_BYTES as u64;
        let left = self.len - body;
        if (entry as u64) > left {
            return Err(RecordError::Truncated {
                offset: body,
                needed: entry,
                available: left as usize,
            });
        }

        self.scratch.resize(entry, 0);
        self.file
            .read_exact(&mut self.scratch)
            .map_err(|source| RecordError::Io {
                path: self.path.clone(),
                source,
            })?;

        let mut cursor = Cursor::new(&self.scratch, body);
        let record = decode_game(&mut cursor)?;
        if cursor.remaining() != 0 {
            return Err(RecordError::EntryTrailingBytes {
                game: self.games_read,
                offset: cursor.offset(),
                remaining: cursor.remaining(),
            });
        }

        self.offset = body + entry as u64;
        self.games_read += 1;
        Ok(Some(record))
    }
}

impl core::fmt::Debug for ShardReader {
    /// Which file, what it says, and how far through it the reader is. The entry
    /// buffer is left out: it is scratch, and it can be megabytes.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ShardReader")
            .field("path", &self.path)
            .field("header", &self.header)
            .field("len", &self.len)
            .field("offset", &self.offset)
            .field("games_read", &self.games_read)
            .field("done", &self.done)
            .finish()
    }
}

impl Iterator for ShardReader {
    type Item = Result<GameRecord, RecordError>;

    /// The next game. One error ends the iteration: a shard that failed to
    /// decode is not a shard some of whose games can still be trusted.
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        match self.next_entry() {
            Ok(Some(record)) => Some(Ok(record)),
            Ok(None) => {
                self.done = true;
                None
            }
            Err(error) => {
                self.done = true;
                Some(Err(error))
            }
        }
    }
}
