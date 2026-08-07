# hexo-records

Versioned on-disk shard format for completed Hexo games. A shard file holds the
games produced by one (run, epoch, phase): a header that pins the versions the
games were played under, followed by one entry per game carrying the `GameSpec`
it was played under, its `MatchResult` with the full adjudication payload, and
its `PlyRecord`s. The crate provides a writer, a reader, and replay
verification through the authoritative engine.

## Public surface

| Item | Description |
| --- | --- |
| `ShardHeader` | Shard identity: mode, run ID, package, checkpoint, epoch, and finalized game count |
| `ShardMode` | `SelfPlay` or `Eval` — which phase of a run produced the shard |
| `GameRecord` | One finished game: its `GameSpec`, terminal `MatchResult`, and accepted plies |
| `GameRecord::from_game` | Extracts a record from a finished `Game` |
| `ShardWriter` | Creates, appends to, and atomically finalizes a shard |
| `ShardReader` | Validates the header and iterates game entries |
| `verify` | Replays a record through the engine and checks stored facts |
| `RecordError` | All write, read, and verification failures |
| `RECORDS_VERSION` | Binary layout and semantics version (`u32`) |
| `MAGIC` | Four-byte `HXRC` file marker |

## Components

### ShardWriter

Writes a shard to `<path>.tmp`, appending length-prefixed game entries one at a
time. `finalize` patches the true game count into the header, syncs the file,
and renames it to the final path. Dropping an unfinalized writer removes the
temporary file. Uses a reusable scratch buffer so appending allocates nothing
at steady state.

### ShardReader

Opens a shard file, validates magic and all four version fields (records, rules,
action order, protocol), and decodes the header. Implements `Iterator` over
`Result<GameRecord, RecordError>`, requiring exactly the declared game count and
rejecting truncation or trailing bytes. Uses a reusable scratch buffer for entry
decoding.

### verify

Replays the move list from the empty position and checks that each ply's
recorded mover matches the seat on turn, each placement is legal, each
`zobrist_after` matches the replayed hash, and the terminal result agrees with
the board state. Does not re-run runner-level policy (e.g. ply-cap
adjudication).

### Format

The header stores magic, four version fields (records, rules, action order,
protocol), the shard mode, three length-prefixed UTF-8 strings (run ID,
package, checkpoint), epoch, and game count. Each game entry is length-prefixed
and stores its `GameSpec` (ply cap, budget, failure policy), `MatchResult` (with
the full win/draw/no-contest payload), ply count, and plies. Each ply stores the
seat, `ActionId`, post-move Zobrist hash, and an optional diagnostics byte
string. All integers are fixed-width little-endian; every enum uses an explicit
one-byte tag.

### RecordError

Covers I/O failures, already-exists on write, bad magic, version mismatches
(records, rules, action order, protocol), truncation, trailing bytes, invalid
tags, zero ply cap, UTF-8 failures, overflow of format limits (strings,
diagnostics, plies, entries, wall budget), unfinished games, and replay
disagreements (seat mismatch, illegal replay, Zobrist mismatch, terminal/winner
conflicts). Decoding failures carry absolute byte offsets; entry failures
identify the game index.

## Connections

- `hexo-engine` provides `ActionId`, `HexCoord`, `Player`, `Position`,
  `MoveError`, replay rules, action ordering, and Zobrist hashing.
- `hexo-runner` provides `GameSpec`, `MatchResult`, `PlyRecord`, `Game`,
  `Budget`, `WinReason`, `DrawReason`, `NoContest`, `Failure`, and
  `FailurePolicy`.
- `hexo-bot` writes one self-play shard per epoch.
- Model crates consume shards through `ShardReader` during fitting.

## Files

| File | Description |
| --- | --- |
| `src/lib.rs` | Crate root: module declarations, re-exports, `RECORDS_VERSION`, and `MAGIC` |
| `src/record.rs` | `ShardMode`, `ShardHeader`, and `GameRecord` type definitions |
| `src/writer.rs` | `ShardWriter`: atomic write-once shard creation with temp-file-and-rename |
| `src/reader.rs` | `ShardReader`: header validation, buffered iteration, and entry decoding |
| `src/replay.rs` | `verify`: engine replay of a decoded record against stored facts |
| `src/format.rs` | Binary layout: tag constants, encode/decode functions for headers, games, plies, specs, results, and errors |
| `src/codec.rs` | Fixed-width little-endian primitive writers and a bounds-checked `Cursor` reader |
| `src/error.rs` | `RecordError` enum with `Display` and `Error` implementations |
