# Open decisions

Questions that must be answered to build the engine and the runner. Distinct
from `SUGGESTIONS.md`, which holds *optional* improvements — everything here has
to be decided one way or another before the code that depends on it exists.

Facts cited from the previous implementation were read from
`Hexo-BotTrainer-hexgt` and are quoted with file references.

---

## A. Blocks the engine

### A1. Termination — there is no draw and no ply cap

**Confirmed gap, and larger than it first looks.**

Stones are only ever added, on an unbounded board, so a position can never
repeat and no stalemate exists. Nothing makes a game end except a win. The old
implementation says so outright:

> `Terminal result. Hexo has no normal draw under the current rules.`
> — `packages/hexo_engine/rust/src/state.rs:64`

and its outcome type cannot express anything else:

```rust
pub struct GameOutcome {
    pub winner: Player,
    pub placements: u32,
}
```

There is also no ply cap, move limit, or adjudication timeout anywhere in the
old engine or runner.

So a self-play pair that never completes a window never terminates. This needs a
cap, and the cap needs a representable result — which means `GameOutcome` must
be able to say something other than "someone won" from the very first version.
Retrofitting a variant into that type later touches the record format, the wire
protocol, and every consumer.

**To decide:** the cap value; whether a capped game is a draw or is adjudicated
(e.g. by threat count); and whether the cap is a rule of the *game* (engine) or
of the *match* (runner). Recommended: match rule, engine stays pure, but the
outcome type is expressive enough on day one.

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

### A3. Arbitrary start positions

Eval against an opening book needs games that do not start empty. Two ways to
support that:

- **Load a board.** Requires reconstructing windows, the legal set, phase, and
  the Zobrist hash from a bare position, *and* proving that reconstruction
  agrees with the incremental path. A second code path, and a rich source of
  bugs that only appear in eval.
- **Replay a move prefix.** A start position *is* a move list. Loading becomes
  replay, and there is exactly one code path forever.

**Recommended: move-prefix only.**

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
simpler; async infects the whole crate. **Recommended: sync**, unless thousands
of concurrent games are foreseen.

### B2. What is recorded per move

Training needs visit distributions and value estimates, but the runner must not
know what a model is.

**Recommended:** the player returns a placement plus an optional opaque blob,
and the runner stores the blob in the record without interpreting it. Keeps the
runner model-agnostic while still producing training data. This is the contract
to settle before `record.rs` exists.

### B3. Search budget

Reproducible self-play wants a deterministic budget (visits or nodes);
tournaments impose wall clocks. Probably both, as an enum — but it shapes the
move-request message, so it precedes the protocol.

### B4. Seed ownership

For byte-reproducible self-play the runner mints and records a seed and hands
per-seat seeds to players. Needs a slot in the handshake.

### B5. Adjudication policy

Illegal move — instant loss, or one retry? Timeout — loss or draw? Crash —
abort with no result, or loss? Is resignation supported? Each becomes a protocol
message, so these are decided alongside C1.

---

## C. Container-time, not code-time

| # | Decision |
| --- | --- |
| C1 | Transport and wire format. A line-oriented stdio protocol is the default: trivial to containerise, debuggable by hand, close to what tournament harnesses expect. |
| C2 | Handshake fields: protocol version, rules version, action-encoding version, seat, seed, budget. |
| C3 | The binary crate — its name, and its subcommands (`selfplay`, `serve`, `train`). |
| C4 | On-disk record format. |
| C5 | `R` for dense action indexing (`SUGGESTIONS.md` S1) — should be measured from real games, not guessed. |
| C6 | **One image or two.** Training is a far heavier image than play: a play-only bot is a static Rust binary, while a training image carries a deep-learning stack and GPU runtime, and is the likely re-entry point for Python. Whether one image serves all four container jobs or a play image is split from a train image changes the base image, its size, and where the Rust/Python line falls. |
