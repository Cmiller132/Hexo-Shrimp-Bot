# Open decisions

Questions that must be answered to build the engine and the runner. Distinct
from `SUGGESTIONS.md`, which holds *optional* improvements — everything here has
to be decided one way or another before the code that depends on it exists.

Facts cited from the previous implementation were read from
`Hexo-BotTrainer-hexgt` and are quoted with file references.

---

## A. Blocks the engine

### A1. Termination — a cap exists, but it reports as a crash

Stones are only ever added, on an unbounded board, so a position can never
repeat and no stalemate exists. Nothing in the *rules* ends a game except a win:

> `Terminal result. Hexo has no normal draw under the current rules.`
> — `packages/hexo_engine/rust/src/state.rs:64`

and the outcome type cannot express anything else:

```rust
pub struct GameOutcome {
    pub winner: Player,
    pub placements: u32,
}
```

The *runner* does cap games — `GameSpec.max_actions = 1024`
(`hexo_runner/session.py:38`), enforced at `loop.py:104-114`. But it enforces it
by raising:

```python
if record_writer.action_count >= spec.max_actions:
    raise RunnerAbort(AbortRecord(stage="runner.max_actions",
        exception_type="MaxActionsExceeded", ...))
```

**That is the actual problem.** A capped game is recorded as `ABORTED`, the same
status as a player crash or an illegal move — the only statuses that exist are
`COMPLETED` and `ABORTED` (`records/results.py:17-21`, wire byte at
`records.rs:37-38`). There is no draw representation anywhere on the wire. So a
game that legitimately ran long is indistinguishable in the data from one where
a player segfaulted, and both are equally unusable as training signal.

Model harnesses then impose their own separate truncation on top —
`max_game_plies = 256` in several run configs, default `512`
(`hexfield_eq/config.py:75`) — so the real cap is a different number in a
different layer from the one the runner enforces.

**To decide:** the cap value and where it lives (recommended: runner, engine
stays pure), and — the load-bearing part — that a capped game is a *representable
result* rather than an error. `GameOutcome` must be able to say something other
than "someone won" from the first version; retrofitting a variant later touches
the record format, the wire protocol, and every consumer. Whether that result is
a draw or an adjudication (e.g. by threat count) is the open half.

### A2. Win condition — resolved, six-or-more

Not an open question after all. The old predicate is:

```rust
pub fn is_win_for(self, player: Player) -> bool {
    self.active_player() == Some(player) && self.count(player) == WINDOW_LEN as u8
}
```
— `packages/hexo_engine/rust/src/tactics.rs:206`

A window is any six consecutive cells along an axis, so seven in a row contains
two completely-filled windows and still wins. There is no exactly-six
restriction and no overline rule. Carry this behaviour forward unchanged.

### A3. Arbitrary start positions — the old design already agrees

Eval against an opening book needs games that do not start empty. Two ways:

- **Load a board.** Requires reconstructing windows, the legal set, phase, and
  the Zobrist hash from a bare position, *and* proving that reconstruction
  agrees with the incremental path. A second code path, and a rich source of
  bugs that only surface in eval.
- **Replay a move prefix.** A start position *is* a move list. Loading becomes
  replay, and there is one code path forever.

The old engine already chose the second, which is worth knowing before
relitigating it — `snapshot.rs:19-23` stores only
`{rules_version: u32, placements: Vec<HexCoord>}`, and `load_state`
(`state.rs:373-389`) replays each placement through the full rule machine,
failing on the first illegal one. A position is expressible only if it is
reachable by a legal game.

Two caveats on inheriting it. It is dormant — `snapshot.rs:3-7` says "No
production code serializes snapshots today", it is unreachable from Python
(`pybridge.rs:205-221` registers no load function), and the runner rejects
scenarios outright (`loop.py:67-68`). So opening-book eval is not actually being
served by it today, and whatever *is* serving that need should be checked before
assuming this design covered it. Separately, there is a dormant serde impl on
`Board` itself (`board.rs:179-212`) that deserializes a bare cell list and
**bypasses turn rules** — exactly the second code path this decision is meant to
avoid. Do not port it.

**Recommended: move-prefix only, and no board-shaped deserialization at all.**

### A4. Grid growth policy

Initial extent, growth trigger, growth factor for the recentred dense grid.

**Recommended:** grid geometry is entirely private. The public API is
coordinate-addressed only, so growth is invisible to callers and a player's
mirror cannot desync from it.

### A5. Zobrist scope

What is hashed: cells and owners, plus side-to-move and turn phase.

The constants must be a **fixed, baked-in table**, not generated at startup —
the hash crosses the container boundary, so two processes must agree on it.

---

## B. Blocks the runner

### B1. Player interface: sync or async

Container players mean blocking I/O. Sync plus one thread per game is far
simpler; async infects the whole crate.

The old interface is a fully synchronous `Protocol` — "Players are opaque
synchronous adapters" (`player.py:3`) — with the lifecycle
`setup_worker → start_game → (decide / observe_transition)* → finish_game →
close` and `decide(state) -> DecisionResult` as the move-producing call
(`player.py:113-134`). It receives a *clone*, never the primary (`loop.py:119`),
and answers with one stone, not a whole turn — two-stone turns are handled by
asking twice, and adapters that think in whole turns buffer internally
(`adapters/sealbot.py:85-90`).

**Recommended: sync**, and keep the one-placement-per-call shape.

### B2. What is recorded per move — the old runner drops it

Training needs visit distributions and value estimates, but the runner must not
know what a model is.

The old code already tried this and left it half-wired, which is worth seeing
before redesigning it. `DecisionResult` carries
`diagnostics: Mapping[str, Any]`, documented as "Player-owned debug data
transported into the position record" (`player.py:66-83`). Players populate it —
hexgt returns `root_value` and `visits`, SealBot returns its PV. **But
`loop.py` never reads `decision.diagnostics`.** The docstring is stale; the data
is dropped on the floor. The runner persists exactly one thing per move:

```python
_run_stage("record_writer.record_action",
    lambda action=decision.action: record_writer.record_action(action))
```
— `loop.py:129-132`, reduced to a `u32` action ID.

Consequently no policy targets, visit counts, or value estimates exist anywhere
in `hexo_runner`, and every model package writes its own `.npz` training shards
on a path that bypasses the runner entirely (`modes/match.py:5-8`). That is the
duplication the new design should remove.

**Recommended:** the player returns a placement plus an optional opaque blob,
and the runner *actually persists* the blob without interpreting it. Model-
agnostic, one record path, no parallel shard writers. Settle before `record.rs`.

### B3. Search budget

Reproducible self-play wants a deterministic budget (visits or nodes);
tournaments impose wall clocks.

No budget exists in the old runner contract at all — `GameSpec` has no
time/visit/node field (`session.py:20-43`), and `Timer` only measures elapsed
time for reporting, never compares it to a limit (`timing.py:13-24`). Budgets
are entirely player-internal and inconsistent across players: hexgt uses a visit
count, SealBot uses a 0.05 s minimax think time. So there is no way to state
"both players got the same budget", which quietly undermines eval comparability.

**Recommended:** a budget enum in the move request, so the runner can state and
record what each seat was given.

### B4. Seed ownership — currently vestigial end to end

For byte-reproducible self-play the runner should mint and record a seed and
hand per-seat seeds to players.

Today the plumbing exists and does nothing. The engine discards the seed
outright:

```rust
/// `seed` and `scenario` are accepted for API-shape stability but DISCARDED:
/// the engine has no randomness and no scenario loader...
```
— `pybridge.rs:62-76`. The runner does not generate seeds, only forwards a
caller-supplied one, persists it in the record, and passes it to players via
`GameContext.seed`. **No shipped player reads it** — each carries its own RNG
(hexgt derives `eval_seed * 1_000_003 + move_index` internally). Replay
determinism comes from the stored action list, not from the seed.

So the seed field is currently decorative. Either wire it end to end or do not
carry it — a recorded seed that does not reproduce the game is worse than none.

### B5. Adjudication policy — everything is an abort today

Illegal move — instant loss, or one retry? Timeout — loss or draw? Crash —
abort with no result, or loss? Is resignation supported?

The old policy is uniform and blunt: every failure is wrapped into a structured
`AbortRecord{stage, exception_type, message}` and the game is finalized
`ABORTED` (`loop.py:226-240`, applied to every stage from `new_game` through
`apply_action`). An illegal move aborts the game rather than losing it. There is
no runner-side clock, so a timeout only registers if the player itself raises.
Resignation does not exist — `player.py:72-73`: "There is no refusal/forfeit
path; errors abort the game."

The `AbortRecord` stage strings are a genuinely good triage contract and worth
keeping. What needs deciding is which of these should be *results* rather than
errors — an illegal move from a tournament opponent is arguably a loss, not a
crashed game.

Each becomes a protocol message, so these are decided alongside C1.

---

## C. Container-time, not code-time

| # | Decision |
| --- | --- |
| C1 | Transport and wire format. A line-oriented stdio protocol is the default: trivial to containerise, debuggable by hand, close to what tournament harnesses expect. |
| C2 | Handshake fields: protocol version, rules version, action-encoding version, seat, seed, budget. |
| C3 | The binary crate — its name, and its subcommands (`selfplay`, `serve`, `train`). |
| C4 | On-disk record format. |
| C5 | ~~`R` for dense action indexing~~ — **withdrawn.** A fixed radius-20 crop caused the main_3 training collapse (`docs/ARCHITECTURE.md:280-283`); see `SUGGESTIONS.md` S1 for what replaces it. |
| C6 | **One image or two.** Training is a far heavier image than play: a play-only bot is a static Rust binary, while a training image carries a deep-learning stack and GPU runtime, and is the likely re-entry point for Python. Whether one image serves all four container jobs or a play image is split from a train image changes the base image, its size, and where the Rust/Python line falls. |
