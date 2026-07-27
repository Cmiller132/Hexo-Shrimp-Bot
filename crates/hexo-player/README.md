# hexo-player

The player seam, and the loop that drives games.

**Status: the seam and the driver are implemented.** No player ships — the crate
is two traits and a loop. The model package that exists plays through
`hexo-search`'s session seam instead, which is the other seat shape and is
described below; this one is what a human, a scripted bot, or a transport
adapter implements.

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
| `player` | `Player`: one method, `choose`, taking the whole `&Game` and returning the whole `Decision`. What a human, a scripted bot, or a transport adapter implements. |
| `model` | `Model`: two required methods, one per `Mode`, on the same `&Game`. `ModelPlayer` binds a model to a mode and is the only bridge between the two traits. |
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
  selection is the model's, as its encoding is. Nothing here samples, so nothing
  here needs a seed — the sessions in `hexo-search` do sample and do take one,
  and `OPEN_DECISIONS.md` B4 stays open above that seam rather than this one.

- **A seat is handed the `Game` view, not a position and a budget.** The record
  is the game's history — `Position` keeps none — so a model whose features
  depend on move order has to be able to reach it. One argument carries all of
  it: `game.position()` for the board, `game.plies()` or `game.prefix()` for the
  record, `game.spec().budget` for what it may spend. Passing the position alone
  would force a parallel move list on every seat that wants recency, which is the
  duplication the engine change removed.

  `&Game` is a shared borrow with no mutable counterpart, so this hands out no
  new authority: `submit` remains the only way to advance anything.

- **A player is handed the canonical game, not a replay mirror.** In process the
  driver already holds it and ownership already makes it read-only. The mirror
  belongs in the same change as the transport that needs it — but a seat that
  keeps one anyway attests the mirror's hash, and the test suite's `Mirrorer`
  does exactly that.

- **`choose` returns the whole `Decision`, not a bare `Action`.** Two of its
  fields can only be authored by the seat. The `zobrist` is an attestation of
  the position the seat actually chose from — a driver that filled it in from
  the canonical game would be the desync check vouching for itself — and the
  `diagnostics` are the seat's training annotations, which nothing downstream
  could invent. The driver submits the decision verbatim. If the game refuses
  it as a desync, the driver reports `Failure::Desync` and the failure policy
  adjudicates, so one broken seat ends its own game rather than the sweep; a
  transport adapter that can resync its remote does so inside `choose`, before
  the decision is ever submitted.

- **`choose` still does not return `Reply`.** `Reply::Failed` is driver-reported:
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
| Any actual player | A seat that wants *this* shape. The test suite carries its own; none is public, and the model package that exists is a `hexo-search` session rather than a `Model` |
| A human seat | The binary exists and reads no moves from anyone. A human seat needs a way to be asked and to answer, which is the same stdin-and-line-protocol story as C1 and C2 |
| A remote seat | C1, C2 — the wire protocol, and the replay mirror that comes with it |

## Connections

- Depends on `hexo-engine` for the seat type and the position read surface, and
  on `hexo-runner` for `Game` and `Decision` — the seat's whole argument and its
  whole answer, carrying the record, the spec, and the result model with them.
- `hexo-search` is the *other* seat shape, and the one a model-backed seat
  wants: a `DecisionSession` may ask a question halfway through its answer, so
  it can hand its leaves to a batched evaluator, which a blocking `choose`
  cannot. The two express the same contract about who authors a `Decision` and
  differ only in that. Neither crate depends on the other.
