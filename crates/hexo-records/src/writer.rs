//! Atomic shard writer.

use crate::error::RecordError;
use crate::format::{ENTRY_PREFIX_BYTES, encode_game, encode_header};
use crate::record::{GameRecord, ShardHeader};
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, ErrorKind, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Writes one shard, once.
///
/// The bytes go to `<path>.tmp` and arrive at `path` only when
/// [`finalize`](ShardWriter::finalize) has patched the true game count in and
/// synced the file. Dropping an unfinalized writer removes its temporary file
/// best-effort and never creates `path`.
pub struct ShardWriter {
    /// `None` once the file has been closed, by `finalize` or by `drop`.
    file: Option<BufWriter<File>>,
    /// Where the bytes are being written.
    tmp: PathBuf,
    /// Where they will be renamed to.
    path: PathBuf,
    /// The file offset of the header's game-count field.
    count_offset: u64,
    /// How many games have been appended.
    games: u32,
    /// Reused encoding buffer, so appending a game allocates nothing steady-state.
    scratch: Vec<u8>,
}

impl ShardWriter {
    /// Start a shard at `path`, writing its header to `<path>.tmp`.
    ///
    /// # Errors
    ///
    /// [`RecordError::HeaderGameCount`] if `header` presets a game count;
    /// [`RecordError::AlreadyExists`] if `path` or `<path>.tmp` exists;
    /// [`RecordError::StringTooLong`] for a header string past the format's
    /// `u16` length; [`RecordError::Io`] for anything the filesystem refuses.
    pub fn create(path: impl AsRef<Path>, header: &ShardHeader) -> Result<Self, RecordError> {
        if header.game_count != 0 {
            return Err(RecordError::HeaderGameCount {
                found: header.game_count,
            });
        }

        let path = path.as_ref().to_path_buf();
        let mut tmp = path.clone().into_os_string();
        tmp.push(".tmp");
        let tmp = PathBuf::from(tmp);

        // Check the destination before writing to make overwrite behavior
        // platform-independent.
        if path.exists() {
            return Err(RecordError::AlreadyExists { path });
        }

        let file = match OpenOptions::new().write(true).create_new(true).open(&tmp) {
            Ok(file) => file,
            Err(source) if source.kind() == ErrorKind::AlreadyExists => {
                return Err(RecordError::AlreadyExists { path: tmp });
            }
            Err(source) => return Err(RecordError::Io { path: tmp, source }),
        };

        let mut bytes = Vec::new();
        let count_offset = encode_header(&mut bytes, header)?;

        let mut file = BufWriter::new(file);
        if let Err(source) = file.write_all(&bytes) {
            drop(file);
            let _ = fs::remove_file(&tmp);
            return Err(RecordError::Io { path: tmp, source });
        }

        Ok(Self {
            file: Some(file),
            tmp,
            path,
            count_offset,
            games: 0,
            scratch: Vec::new(),
        })
    }

    /// Append one finished game.
    ///
    /// # Errors
    ///
    /// [`RecordError::WallBudgetOverflow`] for a wall budget past `u64`
    /// nanoseconds, and the other encoding limits — [`RecordError::TooManyPlies`],
    /// [`RecordError::DiagnosticsTooLong`], [`RecordError::GameTooLarge`],
    /// [`RecordError::TooManyGames`] — each of which would otherwise have to be
    /// a truncation. [`RecordError::Io`] for anything the filesystem refuses.
    pub fn append(&mut self, record: &GameRecord) -> Result<(), RecordError> {
        let games = self.games.checked_add(1).ok_or(RecordError::TooManyGames)?;

        // The entry's length prefix is reserved before the payload is encoded and
        // patched afterwards, so an entry reaches the file as one write.
        self.scratch.clear();
        self.scratch.extend_from_slice(&[0u8; ENTRY_PREFIX_BYTES]);
        encode_game(&mut self.scratch, record)?;
        let payload = self.scratch.len() - ENTRY_PREFIX_BYTES;
        let len =
            u32::try_from(payload).map_err(|_| RecordError::GameTooLarge { bytes: payload })?;
        self.scratch[..ENTRY_PREFIX_BYTES].copy_from_slice(&len.to_le_bytes());

        let file = self.file.as_mut().expect("a live writer holds its file");
        file.write_all(&self.scratch)
            .map_err(|source| RecordError::Io {
                path: self.tmp.clone(),
                source,
            })?;

        self.games = games;
        Ok(())
    }

    /// Patch the true game count in, sync, and rename the shard into place.
    ///
    /// # Errors
    ///
    /// [`RecordError::Io`] for anything the filesystem refuses. A failed
    /// finalize leaves nothing behind: the temporary file is removed, exactly as
    /// dropping the writer would have.
    pub fn finalize(mut self) -> Result<(), RecordError> {
        let file = self.file.take().expect("a live writer holds its file");
        match Self::commit(file, &self.tmp, &self.path, self.count_offset, self.games) {
            Ok(()) => Ok(()),
            Err(error) => {
                let _ = fs::remove_file(&self.tmp);
                Err(error)
            }
        }
    }

    /// Finish the temporary file and move it to its final name.
    ///
    /// The handle is synced and closed before the rename.
    fn commit(
        mut file: BufWriter<File>,
        tmp: &Path,
        path: &Path,
        count_offset: u64,
        games: u32,
    ) -> Result<(), RecordError> {
        let io = |source: std::io::Error| RecordError::Io {
            path: tmp.to_path_buf(),
            source,
        };
        file.seek(SeekFrom::Start(count_offset)).map_err(io)?;
        file.write_all(&games.to_le_bytes()).map_err(io)?;
        file.flush().map_err(io)?;
        file.get_ref().sync_all().map_err(io)?;
        drop(file);
        fs::rename(tmp, path).map_err(io)
    }
}

impl core::fmt::Debug for ShardWriter {
    /// Report paths, game count, and open state without the scratch buffer.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ShardWriter")
            .field("path", &self.path)
            .field("tmp", &self.tmp)
            .field("games", &self.games)
            .field("open", &self.file.is_some())
            .finish()
    }
}

impl Drop for ShardWriter {
    /// Close and remove an unfinalized temporary file best-effort.
    fn drop(&mut self) {
        if let Some(file) = self.file.take() {
            drop(file);
            let _ = fs::remove_file(&self.tmp);
        }
    }
}
