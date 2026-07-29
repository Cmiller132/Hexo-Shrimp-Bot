# hexo-records

## Purpose

`hexo-records` defines the strict, versioned on-disk shard format for completed
Hexo games. It provides one writer, one reader, and replay verification through
the authoritative engine. A shard represents one run, epoch, phase, package,
and checkpoint and is finalized once.

## Public surface

The crate root re-exports:

| Item | Contract |
| --- | --- |
| `ShardHeader` | Shard identity, versions, mode, and finalized game count |
| `ShardMode` | `SelfPlay` or `Eval` header mode |
| `GameRecord` | `GameSpec`, terminal result, and accepted plies |
| `GameRecord::from_game` | Converts one finished runner game |
| `ShardWriter` | Creates, appends, and atomically finalizes a shard |
| `ShardReader` | Strict header validation and game iteration |
| `verify` | Replays a record and checks stored facts |
| `RecordError` | Located format, I/O, version, and replay failures |
| `RECORDS_VERSION` | Binary layout and semantics version |
| `MAGIC` | Four-byte `HXRC` file marker |

The v1 header stores:

| Field | Encoding |
| --- | --- |
| magic | four bytes, `HXRC` |
| records, rules, action-order, protocol versions | four little-endian `u32` |
| mode | one-byte tag |
| run ID, package, checkpoint | `u16` byte length plus UTF-8 |
| epoch | little-endian `u32` |
| game count | little-endian `u32`, patched at finalize |

Each game entry is length-prefixed and stores its `GameSpec`, terminal
`MatchResult`, ply count, and plies. Each ply stores the seat, `ActionId`,
post-move Zobrist hash, and an optional diagnostics byte string.

Basic write/read flow:

```rust
use hexo_records::{ShardReader, ShardWriter};

# fn round_trip(
#   path: &std::path::Path,
#   header: &hexo_records::ShardHeader,
#   record: &hexo_records::GameRecord
# ) -> Result<(), hexo_records::RecordError> {
let mut writer = ShardWriter::create(path, header)?;
writer.append(record)?;
writer.finalize()?;
for game in ShardReader::open(path)? {
    hexo_records::verify(&game?)?;
}
# Ok(())
# }
```

## Run / test

From the repository root:

```sh
cargo test -p hexo-records
cargo test -p hexo-records --test shard
cargo doc -p hexo-records --no-deps
cargo check -p hexo-records
```

Run the complete workspace gates:

```sh
cargo xtask verify
```

The crate has no command-line target. Read and write shards through the library
API or through `hexo-bot`.

## Connections

- `crates/hexo-runner` owns `GameSpec`, `MatchResult`, and `PlyRecord`.
- `crates/hexo-engine` owns `ActionId`, replay rules, action order, and position
  hashes.
- `crates/hexo-bot` writes one self-play shard per epoch.
- `crates/models/mock` consumes shards through `ShardReader` during `fit`.
- `docs/CONTAINER_SPEC.md` defines shard lifecycle and ownership.
- `src/format.rs` is the single binary encoder and decoder.
- `src/codec.rs` owns bounded little-endian primitive reads and writes.

## Invariants & gotchas

- A record can be constructed from a `Game` only after the game has finished.
- The format is little-endian and every enum uses an explicit tag.
- Reader version checks are exact; unsupported versions are not interpreted.
- The reader rejects truncation, invalid tags, invalid UTF-8, impossible
  lengths, and trailing bytes.
- Entry lengths bound each game before its fields are decoded.
- `ShardWriter::create` refuses a header with a preset game count.
- `ShardWriter` writes to `<path>.tmp` and `finalize` renames it to the
  requested final path.
- `finalize` patches the number of successfully appended games.
- Dropping an unfinalized writer removes its incomplete file.
- Appending after finalization is impossible because `finalize` consumes the
  writer.
- Appending to an existing finalized shard is unsupported.
- An empty finalized shard is valid only if its header and count agree.
- Diagnostics distinguish absent, present-empty, and present-nonempty values.
- Oversized strings, diagnostics, durations, counts, and entries are errors;
  values are never truncated.
- `verify` independently replays actions and checks mover, hash, result, and
  stored transition facts.
- Parsing success does not imply replay validity; callers that consume game
  semantics must call `verify`.
- `ShardMode` is header metadata; the binary shape of a game is shared by both
  modes.
- The crate does not provide compression, a secondary index, partial recovery,
  or a Serde representation.
