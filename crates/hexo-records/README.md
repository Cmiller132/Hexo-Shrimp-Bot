# hexo-records

The on-disk game record: a shard of finished games, written once and read
strictly.

**Status: the format is implemented, and written in anger.** It settles
`OPEN_DECISIONS.md` C4. One shard file holds every game from one (run, epoch,
phase), and `hexo-bot`'s training loop writes one per epoch of self-play and
hands it to the package's `fit`.

## Shape

Pure Rust library crate, depends on `hexo-engine` and `hexo-runner`. No serde,
and no other dependency — the format is hand-rolled binary and the crate is the
only thing that speaks it.

```
crates/hexo-records/
  Cargo.toml
  README.md
  src/
    lib.rs        # crate root, flat re-exports, RECORDS_VERSION, MAGIC
    record.rs     # ShardHeader, ShardMode, GameRecord
    format.rs     # the byte layout: every tag constant, encode and decode
    codec.rs      # little-endian primitives and the bounds-checked cursor
    writer.rs     # ShardWriter: tmp file, patched count, rename
    reader.rs     # ShardReader: validated header, then Result<GameRecord, _>
    replay.rs     # verify: the record replayed through the engine
    error.rs      # RecordError
  tests/
    shard.rs      # round trips, the reader's refusals, and what only verify sees
```

## Module map

| Module | Role |
| --- | --- |
| `record` | The shapes a shard carries. `GameRecord::from_game` is the only way one is made from a `Game`, and it refuses an unfinished one. |
| `format` | The format, stated once: the tag numbering, and one encoder and one decoder per shape. Every `match` over a runner or engine enum is exhaustive with no catch-all, so a new variant upstream breaks this build instead of being silently mis-encoded. |
| `codec` | Fixed-width little-endian primitives, and the cursor that reads them back with a bounds check and an absolute file offset. Knows nothing about games. |
| `writer` | `ShardWriter`: header, appended entries, then the count patched in and the file renamed into place. |
| `reader` | `ShardReader`: validate, then iterate. Every way a file can disagree with itself is an error. |
| `replay` | `verify`: the record replayed through `hexo-engine`. The detector parsing cannot be. |
| `error` | `RecordError`, carrying the offset, ply, game index, or pair of versions that locates the problem. |

## Format v1

All integers are little-endian and fixed width. There are no varints, so a
field's size never depends on its value and an offset in an error message is an
offset in the file.

### Shard header

| Field | Bytes | Notes |
| --- | --- | --- |
| magic | 4 | `HXRC` |
| records version | u32 | `RECORDS_VERSION`; refused unless equal |
| rules version | u32 | `hexo_engine::RULES_VERSION`; refused unless equal |
| action-order version | u32 | `hexo_engine::ACTION_ORDER_VERSION`; refused unless equal |
| protocol version | u32 | `hexo_runner::PROTOCOL_VERSION`; refused unless equal |
| mode | u8 | `0` self-play, `1` eval |
| run id | u16 length + UTF-8 | |
| package | u16 length + UTF-8 | |
| checkpoint | u16 length + UTF-8 | |
| epoch | u32 | |
| game count | u32 | placeholder while writing, patched on finalize |

### One game entry

| Field | Bytes | Notes |
| --- | --- | --- |
| entry length | u32 | the payload below, so a reader can skip a game |
| ply cap | u32 | nonzero; a zero is a corrupt file |
| budget | u8 tag | `0` unlimited, `1` nodes + u64, `2` visits + u64, `3` wall + u64 ns |
| failure policy | u8 | `0` forfeit, `1` no-contest |
| result | u8 tag + payload | below |
| ply count | u32 | |
| plies | ply count × | seat u8, action u32, `zobrist_after` u64, diagnostics u8 presence, then u32 length + bytes if present |

### Result

| Tag | Arm | Payload |
| --- | --- | --- |
| `0` | `Decisive` | winner u8, then a win reason |
| `1` | `Drawn` | draw reason u8 — `0` ply cap |
| `2` | `NoContest` | `0` engine limit: seat u8 + move error; `1` seat failure: seat u8 + failure |

| Tag | Win reason | Payload |
| --- | --- | --- |
| `0` | six in a row | — |
| `1` | resignation | — |
| `2` | illegal move | action id u32, then a move error |
| `3` | timeout | — |
| `4` | crash | — |
| `5` | protocol | — |
| `6` | desync | expected u64, got u64 |

| Tag | Move error | Payload |
| --- | --- | --- |
| `0` | terminal state | — |
| `1` | illegal opening | — |
| `2` | coord out of bounds | q i16, r i16 |
| `3` | occupied | q i16, r i16 |
| `4` | too far from stones | q i16, r i16 |
| `5` | board extent exceeded | cells u64 |

| Tag | Failure | Payload |
| --- | --- | --- |
| `0` | timeout | — |
| `1` | crashed | — |
| `2` | protocol | — |
| `3` | desync | expected u64, got u64 |

## Design notes

- **Hand-rolled binary, not serde.** A record is `[Action]` plus a result, and
  an `Action` is a `u32` in a newtype — `ENGINE_SPEC.md` §12 says a record
  writer emits `u32`s and that is the whole format. What serde would add is a
  second way to describe the same bytes, and the derive would put the layout in
  three places: the struct, the attribute, and this README.

  It also buys nothing at the boundary that matters. Python will read records
  through *this* code, via an embedded interpreter, not through a parser of its
  own. One implementation of the format means a shard cannot mean one thing to
  the trainer and another to the reader that produced it, which is the class of
  bug a training run cannot see and cannot debug.

  Formats here are not backward compatible — the workspace rule is to bump a
  format and regenerate the data rather than teach a reader two shapes — so
  `RECORDS_VERSION` moves and last epoch's shards are refused by name, loudly,
  instead of being reinterpreted.

- **The magic says "shard", the version field says "which".** `HXRC` and a
  separate `RECORDS_VERSION` rather than a version baked into the magic: one
  thing in the file states the format version, so a reader never has to
  adjudicate between two claims about it, and a file that is not a shard at all
  gets a different error than one that is a shard from another version.

- **The reader is strict because a record that parses wrong trains a model
  wrong.** Every version is checked against the constant this build links, and
  each mismatch is its own error naming both numbers — "version mismatch" alone
  does not say which side is old. The game count must equal the entries the file
  holds; a byte past the last game is an error; a file that ends inside a field
  is an error rather than a shard that looks smaller than it is. There is no
  lenient mode, because the consumer of a shard is a training run that will not
  notice being handed half of one.

- **`verify` catches what parsing cannot.** An action id, a hash, or a seat byte
  that drifted still decodes into a perfectly valid field. `verify` replays the
  move list from the empty position through `hexo-engine` and checks the
  recorded mover, the legality of each placement, the *whole* `zobrist_after`
  chain rather than only its last link, and that a six-in-a-row result lands on
  a terminal position won by the seat the record names while every other
  ending — a resignation, a forfeit, a ply cap, a no-contest — lands on a
  position that is not terminal.

  It deliberately does not re-adjudicate. Whether the cap fell where
  `GameSpec::ply_cap` says it should is match policy and `hexo-runner` owns it;
  a second implementation here would be a second answer to a settled question.
  What is checked is engine fact, which the record and the engine state
  independently — which is the same reason the engine keeps oracles it did not
  derive from itself.

- **A crashed run leaves a partial directory, never a corrupt shard.** The
  writer writes to `<path>.tmp` with a placeholder count, and on `finalize`
  patches the true count in, syncs, closes, and renames — atomic on the same
  filesystem. Nothing ever appears at `path` in a half-written state. A writer
  that is dropped instead removes its temporary file; a `finalize` that fails
  does the same, so a failure and an abandonment leave the same nothing behind.
  `create` refuses a destination that already exists rather than leaving it to
  the rename, where the platforms disagree — POSIX replaces, Windows refuses —
  and it refuses before a whole shard has been written rather than after.

- **The count is the writer's to state.** `ShardWriter::create` refuses a header
  that presets `game_count`: the writer counts what it actually wrote. A preset
  count could only ever be a claim about a file that does not exist yet, and
  quietly overwriting it would be a silently substituted default.

- **Diagnostics keep three distinguishable states.** Absent, present-and-empty,
  and present-with-bytes are three different facts about a seat, so the presence
  byte is separate from the length. Collapsing empty into absent would make the
  format unable to say "this seat answered with nothing", which is exactly the
  signal a package debugging its own annotations is looking for.

- **Every write limit is an error, not a truncation.** A wall budget past `u64`
  nanoseconds, a header string past its `u16`, diagnostics past their `u32`: each
  is refused at the write. Recording a budget the game was not played under is
  worse than refusing to record the game.

## Deliberately absent

| Omitted | Why |
| --- | --- |
| Appending to an existing shard | A shard is one (run, epoch, phase) and is finished when that is. Reopening one would put the atomic rename back on the table for every append. |
| Compression | Records live for one epoch — `CONTAINER_SPEC.md` removes them after a successful fit — and a move list is already `u32`s. Compression would trade the format's fixed offsets for CPU the trainer wants. |
| An index of game offsets | Entries carry a length prefix, so skipping is already a seek per game, and every consumer so far reads the whole shard. |
| A `serde` impl | One implementation of the format is the point. Python reads through this code, not through a parser of its own. |
| A lenient or partial read mode | A shard that failed to decode is not a shard some of whose games can still be trusted. |

## Connections

- `hexo-runner` owns the in-memory shapes — `GameSpec`, `MatchResult`,
  `PlyRecord` — and this crate is only their byte layout. It adds no field the
  runner does not have and drops none it does.
- `hexo-engine` supplies the replay `verify` checks against, and two of the four
  version constants the header pins.
- `hexo-bot` writes one shard per epoch of self-play, and a model package reads
  it back in its `fit` phase. Only self-play writes one: an evaluation round and
  a `match` are evidence about two checkpoints rather than training data, and
  nothing would read a shard of them (`CONTAINER_SPEC.md` §11).
