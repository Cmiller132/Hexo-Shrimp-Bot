# hexo-search

The evaluator seam and the nonblocking decision sessions: how a seat searches,
and how its network questions get batched across a thousand concurrent games.

**Status: implemented.** The seam, all three session shapes, and the RNG ship. No
model, no encoder, no evaluator, and no selector ships — this crate is the
machinery, and it never learns what a feature or a network is. It settles
`docs/SUGGESTIONS.md` S3.

## Shape

Pure Rust library crate, depending on `hexo-engine` and `hexo-runner`. No
threads, no async, no channels, no tensors, no I/O. The nonblocking shape is what
makes a threaded driver possible; it is not itself the threaded driver.

```
crates/hexo-search/
  Cargo.toml
  README.md
  src/
    lib.rs        # crate root, flat re-exports, the loop in one doctest
    seam.rs       # Evaluation, Encoder, Evaluator, EncodedBatch
    session.rs    # LeafId, SessionStatus, DecisionSession
    select.rs     # Child, SearchOutcome, SelectFromSearch, SelectFromPolicy
    policy.rs     # PolicySession
    mcts.rs       # MctsConfig, MctsSession, and the private tree
    gumbel.rs     # GumbelConfig, GumbelSession, and deterministic lines
    rng.rs        # SplitMix64
  tests/
    common/mod.rs # the encoder, evaluators, and selectors a package would own
    topology.rs   # the reference sweep: 32 games, 64 seats, one batch
    mcts.rs       # terminals, the mover comparison, the books, authorship
    policy.rs     # one question per move, and nothing else
    gumbel.rs     # the halving schedule, tie-breaks, and the parity fixture
    fixtures/     # gumbel_parity.json + regenerate_gumbel.py (--check/--emit)
```

## Module map

| Module | Role |
| --- | --- |
| `seam` | `Evaluation` (the two normative conventions), `Encoder` (worker-side), `Evaluator` (batcher-side), and `EncodedBatch` — a reusable ragged arena so assembling a batch costs no per-item allocation. |
| `session` | `DecisionSession`: `begin`, `pump`, `resume`, `take_decision`, `reseed`. `LeafId` and `SessionStatus`. |
| `select` | What a package is handed when a search is done (`SearchOutcome`, `Child`) and the two traits it implements to answer. |
| `policy` | `PolicySession`: one root evaluation per move. |
| `mcts` | `MctsSession`: PUCT with virtual loss and an in-flight cap, over a make/unmake walk of the session's own position. |
| `gumbel` | `GumbelSession`: Gumbel-top root candidates and fixed-budget sequential halving over independently deepened lines. |
| `rng` | `SplitMix64`, the stream a sampling selector draws from. |

## Design notes

- **Sessions are nonblocking because waiting is data.** The obvious search calls
  `evaluate(one_position)` and blocks. That makes a game equal to a thread — ten
  thousand self-play games is ten thousand OS threads — and, worse, it forecloses
  the one thing that makes a GPU worth having: every thread is asleep inside its
  search holding exactly the leaf that should have gone into a batch.

  Inverting it is the same move `hexo-runner` made one level up, and it gives the
  same two shapes from one type. A session hands out the leaves it wants and
  returns; a driver sweeps hundreds of them into one `EncodedBatch` and crosses
  once. The thread pool is then sized to the silicon rather than to the number of
  games, and the number of concurrent games `G` is bounded by RAM instead of by
  scheduler pressure. `tests/topology.rs` is that loop, single-threaded and
  complete; `hexo-bot`'s driver is the same loop with the `for` replaced by a
  worker pool and two bounded channels, and nothing else about the shape
  changed.

  Concretely, this is `docs/ENGINE_RL_AUDIT.md`'s recommended topology minus the
  threads: fixed CPU shards, many games and trees per shard, reusable encoded
  leaf slots, bounded queues carrying handles rather than positions, one batcher
  per device. The audit's first constraint — *allow multiple pending leaves per
  actor instead of blocking per inference* — is `MctsConfig::max_in_flight`.

- **The encoder runs worker-side, the evaluator batcher-side.** Two reasons, and
  the first is not an optimisation. The `&Position` a session hands to `emit` is
  transient make/unmake state on the session's own board, alive only for the
  duration of that callback; there is nothing to put in a queue but bytes. And
  encoding is the CPU half of the work, so on the actor threads it scales with
  cores while on the batcher thread it serialises the one place the whole process
  converges.

  The batch has one append per side of that split: `push_with` encodes a
  position into it worker-side, while the position still exists, and
  `push_bytes` appends an item that is already encoded — which is how a batcher
  merges arriving jobs into one device-bound batch rather than pretending to
  re-encode positions that stopped existing when the pump returned.

  `EncodedBatch` is deliberately ragged — one `Vec<u8>` plus CSR-style offsets.
  A fixed action crop is exactly what this workspace refuses: the board has no
  edge, so an encoder emitting a per-action row emits `n` of them here and `m`
  next ply.

- **`Evaluation` carries two conventions, and they are normative.** Priors are in
  the engine's canonical legal order — entry `i` is `nth_legal(i)` of the
  evaluated position, and the length is that position's `legal_count()`. Value is
  from the perspective of the **side to move at the evaluated position**. Both
  are checked on delivery and both failures panic, because neither is detectable
  downstream: scrambled priors keep training, and a value with the wrong sign
  produces a search that plays to lose while every test stays green.

- **Backpropagation compares movers; it does not count plies.** A turn is two
  placements, so consecutive tree plies can have the **same** mover. Every node
  stores the player to move at it, an evaluation is signed by comparing that
  player against the leaf's mover, and a terminal is signed by comparing it
  against the actual winner. Depth parity — the standard shortcut everywhere else
  — makes a search work to *avoid* winning on its own second stone here.

  `tests/mcts.rs::a_win_on_the_turns_second_stone_is_preferred_and_not_avoided`
  is built to fail under depth parity, and it does: it reports `-0.75` on the
  root edge of the winning line. This is a symmetric bug in the sense
  `CLAUDE.md` means — the sign is applied consistently and every internal
  invariant still holds — so a fixture that discriminates is the only detector.

- **A terminal inside the search is not a question for the network.** A placement
  that completes six in a row is answered by the engine on `apply`, so the
  descent backs up ±1 on the spot, emits nothing, and still spends one visit. A
  terminal root is a `begin` panic instead: the driver only asks a live game's
  mover.

- **Virtual loss is applied and removed as one operation with two signs.** On
  emit the whole selection path takes `N += 1, W -= 1`; on resume the same path
  takes `N -= 1, W += 1` and *then* the real value. Written as two functions so
  the inverse is visible, and guarded by a debug assert that recomputes the
  root's visit total from its edges and requires it to equal the visits
  dispatched — before, between, and after every resume. A virtual loss removed
  twice, or applied to a path that was then edited, cancels perfectly in any
  round-trip test; that assert is what sees it.

- **Make/unmake in the tree; clones on lines.** An `MctsSession` owns one
  `Position` — its own copy of the game's, taken with `clone_from` into the
  buffer it kept from last time — and every descent walks it with
  `hexo_engine::Search`, unwinding explicitly before it returns.
  `Search::drop` would unwind too, but a single `Search` serves every descent
  of one `pump`, so the unwind has to be the descent's job and not the call's.
  The reason to walk is that the budget times the branching factor of clones
  is memory the process does not have at a thousand games. `GumbelSession`
  clones one `Position` per candidate line instead and advances it in place:
  a line is a single path, never re-descended, so there is nothing to unwind
  — and the engine's 3-bit-plane rewrite made a clone (~175 ns at ply 256)
  cheaper than an apply+undo pair anyway.

- **Children are materialised at emit, priors at resume.** Expansion needs the
  leaf's legal set, and by the time an answer arrives the descent has unwound and
  the position is gone. So the edges are built inside the emit callback, where
  the position exists, and the priors are written onto them when the evaluation
  lands. Nothing can read the zeroed priors in between: the node is not
  `evaluated`, so it is a leaf, and selection stops there.

  The same node can be asked about twice, when a descent re-selects a leaf that
  is still in flight. The first answer expands it; any later one only backs its
  value up. Virtual loss makes that rare rather than impossible, and forbidding
  it would mean a descent that can fail.

- **Selection ships empty, and both hook methods are required.** No sampler, no
  temperature, no argmax — the same rule `crates/hexo-player/README.md` argues:
  a default that can be inherited without being chosen compiles, passes, and
  yields a self-play run in which every game is identical, and no downstream stage
  can detect it because the data is well-formed. `diagnostics` has no default for
  the same reason: a silent `None` would drop the visit distribution a policy
  target is built from into a record that still looks complete.

  Nothing here checks that the selector returned a *legal* action either. That is
  `hexo-player`'s rule as well — the game adjudicates a bad placement and
  `WinReason::IllegalMove` carries the evidence — and a check here would be a
  second implementation of the rules.

- **`MctsConfig` has no `Default`.** A search shape is a model choice. `visits`
  and `max_in_flight` are `NonZero` because zero is nonsense; `c_puct` is a plain
  `f32` because zero is meaningful, and it is validated as finite and
  non-negative at construction.

  The budget counts descents below the root. The root's own evaluation — the one
  that supplies its priors — is not one of them, so a budget of `n` dispatches
  `n` descents and at most `n + 1` evaluations, and "root-child visits sum to the
  budget" is exactly true rather than off by one.

- **Gumbel search is lines, not a second tree.** `GumbelSession` samples at
  most `min(candidates, simulations / 2, legal_count)` root actions by
  `g + ln(prior)`, then gives every survivor an equal number of one-ply
  deepenings before each stable halving. An interior line follows the evaluated
  prior argmax. Hexo transitions are deterministic, so revisiting an identical
  path cannot reveal new information; spending that evaluation on another ply
  buys depth instead. A terminal line freezes at exact ±1 and emits no network
  question.

  The root evaluation is not part of the simulation budget. The schedule leaves
  integer remainders unused rather than treating candidates unequally within a
  round. Final selection uses
  `g + ln(prior) + (50 + max_visits) * value`, with values signed by comparing
  the leaf and root movers rather than by depth parity. Zero priors have
  `ln(prior) = -inf`; all-zero ties therefore retain canonical legal order.

  Production noise comes from the crate's `SplitMix64`. Tests and cross-language
  fixtures use `with_gumbels` or `queue_gumbels` to inject one finite value per
  canonical root action, because matching RNG streams across implementations is
  not an algorithm invariant. `GumbelTrace` exposes the initial candidate ranks
  and the survivor ranks after every round without putting package diagnostics
  into this model-agnostic crate.

  `tests/fixtures/gumbel_parity.json` is generated by the production Python
  implementation, with every emitted position, injected Gumbel, scripted
  evaluation, chosen move, and survivor trace recorded. From
  `python/mantisnet`, verify it with
  `.venv/Scripts/python.exe ../../crates/hexo-search/tests/fixtures/regenerate_gumbel.py --check`
  on Windows (`.venv/bin/python` on Linux). For an intentional reference
  change, run the same script with `--emit`, replace the JSON with that reviewed
  output, and run the Rust parity test.

- **PUCT, written as it is written everywhere.** `Q + c_puct * P *
  sqrt(N_parent) / (1 + N_child)`, with `Q = 0` for an unvisited child, from the
  parent's side. `W` accumulates in `f64`: it sums thousands of terms of
  magnitude one and its mean is compared against a `c_puct * P` term that can be
  four orders of magnitude smaller.

  At `N_parent == 0` every term is zero and the canonically first child is taken.
  That costs one visit per node, on its first descent. The usual patch is
  `sqrt(max(N, 1))`, and it is deliberately not applied: first-play urgency is a
  tuning knob, and tuning knobs belong to the model.

- **This crate ships its own `SplitMix64`.** The engine has the same algorithm in
  `testkit/rng.rs`, but that file is `#[path]`-included by that crate's tests and
  benches and is in no library, so nothing at runtime can reach it. Same
  algorithm rather than a second choice — two different generators in one
  workspace would be the thing that needs explaining. It is not `Copy`, because a
  copied generator hands out the same stream twice.

- **`reseed` is the B4 seam.** Today a package constructs a session with a seed
  of its own and `hexo-bot` reseeds both of a lane's seats from entropy before
  every game, so games are non-deterministic — which is honest: nothing records
  a seed, so nothing promises a replay. A session *is* a function of its
  position sequence, its evaluations, and its seed — `tests/mcts.rs` and
  `tests/policy.rs` pin that
  — so when `docs/OPEN_DECISIONS.md` B4 is answered, seeds minted from stable
  game and seat ids land in `reseed` and nothing else about a session moves.

## Deliberately not built

| Thing | Why not, and where it would go |
| --- | --- |
| Subtree reuse between decisions | `begin` builds a fresh tree, keeping the arenas. Reuse means carrying a tree across an opponent reply and re-rooting it, which needs the moves actually played — and a stale subtree whose priors came from an older checkpoint is a correctness question, not a speed one. It would land in `MctsSession::begin`, as a re-root step keyed on the game's record. |
| Transposition table | v1 is a tree, not a DAG. Hexo transposes structurally — a turn's two stones are playable in either order — so the merge is worth roughly 2x per turn, and `Position::zobrist()` is position-only precisely so a table is possible later. It is not built now because a shared table is where a search stops being a pure function of its inputs, and the determinism tests above are worth more at this stage than the 2x. It would land as a map beside `Tree`. |
| Dirichlet root noise | It is exploration policy, which is the model's, and it needs a Dirichlet sampler this crate has no business owning. A package applies it to the root evaluation before `resume`, or inside its own `Evaluator`. |
| Batched `resume` | One leaf at a time. A slice-taking variant would save a bounds check per leaf against a network call, which is not a trade worth an API. |
| Threads, queues, backpressure | `hexo-bot` has them. This crate is what makes them possible and deliberately contains none of them. |
| Any encoder, evaluator, or selector | All three are what a *package* owns. `hexo-model` states the trait that binds them together and `crates/models/mock` implements them; `tests/common/mod.rs` carries test ones and none is public. |

## Not built yet

| Thing | Blocked on |
| --- | --- |
| Minted, recorded per-game seeds | B4. `reseed` is the seam and takes no position on the policy |

## Connections

- Depends on `hexo-engine` for the position read surface, the canonical legal
  ordering the priors are indexed by, and `Search` for the make/unmake walk.
- Depends on `hexo-runner` for `Game` — the session's whole input — and
  `Decision`, which the session authors in full: the placement, the hash of the
  position it searched, and the package's diagnostics.
- `hexo-model` binds an encoder, an evaluator, and the two session constructors
  into the one trait a container knows a package by; `crates/models/mock`
  supplies all of them, and `hexo-bot`'s driver is what pumps the sessions,
  encodes their leaves worker-side, and fills the batches one evaluator answers.
- `hexo-player` is the *other* seat shape: `Player::choose` is one blocking call
  that returns a whole `Decision`, which is what a human, a scripted bot, or a
  transport adapter wants. A model-backed seat wants this crate instead, because
  a blocking `choose` is the design that cannot batch. Neither crate depends on
  the other.
