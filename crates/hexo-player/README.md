# hexo-player

The player seam, and the loop that drives games.

**Status: the seam and the driver are implemented.** No player ships — the crate
is two traits and a loop until a model exists.

## Shape

Pure Rust library crate, depends on `hexo-engine` and `hexo-runner`. The arrow
points this way and never back, so the runner still never learns what a player is.

```
crates/hexo-player/
  Cargo.toml
  README.md
  src/
    lib.rs        # crate root, flat re-exports
    player.rs     # Player
    model.rs      # Model, Mode, ModelPlayer
    table.rs      # Table, sweep
  tests/
    table.rs      # dispatch, the driver, and what the driver refuses to do
```

## Module map

| Module | Role |
| --- | --- |
| `player` | `Player`: one method, `choose`. What a human, a scripted bot, or a transport adapter implements. |
| `model` | `Model`: two required methods, one per `Mode`. `ModelPlayer` binds a model to a mode and is the only bridge between the two traits. |
| `table` | `Table` owns one `Game` and its two seats; `sweep` drives many at once. |

## Design notes

- **Two traits, because there are two contracts.** A human seat has no meaningful
  self-play mode and a model has no meaningful "just play" mode, so one trait
  would force one of them to implement something it cannot define.

- **`Model`'s two methods have no defaults.** A single method taking a `Mode` can
  be written to ignore it — that compiles, passes, and silently produces a
  self-play run in which every game is identical. No downstream stage can detect
  it, because the data is well-formed. `eval_move` is greedier but not argmax for
  the mirror reason: two deterministic seats replay one game, so a thousand-game
  match carries no more information than one.

- **No sampler, no temperature, no argmax.** Those are move selection, and move
  selection is the model's, as its encoding is. This is also what keeps
  `OPEN_DECISIONS.md` B4 deferred: nothing here samples, so nothing here needs a
  seed.

- **A player is handed `&Position`, not a replay mirror.** In process the driver
  already holds the canonical position and ownership already makes it read-only.
  The mirror belongs in the same change as the transport that needs it; until
  then the echoed `zobrist` is the canonical one.

- **`choose` returns `Action`, not `Reply`.** `Reply::Failed` is driver-reported:
  a seat cannot declare itself crashed. Resignation is the first extension, and is
  absent until something can evaluate its own position well enough to give up.

- **The driver never checks legality.** An illegal placement is submitted as-is
  and the game adjudicates it, which is what `WinReason::IllegalMove` carries the
  action and cause for. A check here would be a second implementation of the rules.

- **`[P; 2]` for one kind of seat, `Box<dyn Player>` for two.** Two checkpoints of
  one model are `[M; 2]`; a human against a model is boxed. `Player` is
  object-safe and `Box<P>` forwards, so both are the same `Table`.

## Not built yet

| Thing | Blocked on |
| --- | --- |
| Any actual player | A model. The test suite carries its own seats; none is public |
| A human seat | C3 — it needs stdin, which arrives with the binary |
| A remote seat | C1, C2 — the wire protocol, and the replay mirror that comes with it |
| Recording which mode a game was played in | C4. `ModelPlayer` knows its mode; there is no record format to write it into |
| Batched evaluation | S3. A separate seam, depending only on `hexo-engine` |

## Connections

- Depends on `hexo-engine` for rules and state, and on `hexo-runner` for `Game`,
  `Budget`, and the result model.
