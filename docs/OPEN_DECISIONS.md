# Open decisions

Questions that must be answered to build the engine and the runner. Distinct
from `SUGGESTIONS.md`, which holds *optional* improvements — everything here has
to be decided one way or another before the code that depends on it exists.

A settled question leaves this file. Its answer moves to the `README.md` of
whatever it decided, so this doc stays a list of what is *not* known rather than
a changelog of what is. Numbering is stable — a closed item's number is not
reused, because other documents cite it.

## Settled, and where the answer lives

| # | Question | Where it is now |
| --- | --- | --- |
| A1 | Termination: a cap that is a result rather than a crash | `crates/hexo-runner/README.md`; `GameSpec::ply_cap` and `DrawReason::PlyCap` |
| A2 | Win condition: six **or more**, no overline rule | `crates/hexo-engine/src/lib.rs`, `window.rs` |
| A3 | Arbitrary start positions: move-prefix replay only, no board-shaped deserialisation | root `README.md`; `Position::replay`; `ENGINE_SPEC.md` §12 |
| A4 | Grid growth policy, and the extent ceiling as a function of the position | `crates/hexo-engine/src/grid.rs`; `ENGINE_SPEC.md` §5.5 |
| A5 | Zobrist scope, and a baked-in rather than generated key table | `crates/hexo-engine/src/zobrist.rs`; `ENGINE_SPEC.md` §8 |
| B1 | Player interface | `crates/hexo-runner/README.md`. Answered by neither option offered: `Game` is a nonblocking state machine, so whether a caller blocks a thread or polls a thousand games is decided outside the crate |
| B2 | What is recorded per move | `PlyRecord::diagnostics` — an opaque seat-owned blob, and this one is actually persisted |
| B3 | Search budget | `Budget`, stated and recorded by the game, never enforced by it |
| B5 | Adjudication policy | `MatchResult`, `FailurePolicy`, `NoContest` |
| C3 | The binary crate, and its modes | `CONTAINER_SPEC.md` §3: `hexo-bot`, with `train`, `serve`, and `play`. Self-play is not a mode — it is the first phase of `train`, because one loop cannot be split into pieces that could drift from it |
| C5 | ~~`R` for dense action indexing~~ — **withdrawn**, not answered: a fixed radius-20 crop makes wins outside the crop unrepresentable, so the action space silently stops matching the game | `crates/hexo-engine/README.md`; the canonical unbounded ordering replaced it |
| C6 | One image or two | `CONTAINER_SPEC.md` §2. One: a play-only image would need a second implementation of the model's forward pass, which the no-dual-paths rule forbids and which could silently disagree with the first |

---

## B4. Seed ownership — deliberately absent until it is real

For byte-reproducible self-play the runner would mint and record a seed and hand
per-seat seeds to players.

Nothing in the workspace needs one yet. The engine has no randomness at all, and
replay determinism comes from the stored action list rather than from a seed: a
game is reproduced by replaying its moves. A seed field would therefore be
carried, persisted, and read by nobody.

Either wire a seed end to end or do not carry it — a recorded seed that does not
reproduce the game is worse than none, because it reads as a reproducibility
guarantee that was never checked. `hexo-runner` therefore ships with **no seed
field at all**, deliberately.

This stops being optional the moment a player samples rather than maximising:
seeds must then be minted and recorded, derived from stable game and seat ids so
that scheduling cannot change a run.

---

## C. Container-time, not code-time

| # | Decision |
| --- | --- |
| C1 | Transport and wire format. A line-oriented stdio protocol is the default: trivial to containerise, debuggable by hand, close to what tournament harnesses expect. |
| C2 | Handshake fields: protocol version, rules version, action-encoding version, seat, seed, budget. |
| C4 | On-disk record format. |

`CONTAINER_SPEC.md` §9 is the same list from the other side, and adds B4 to it:
`train` needs seeds the moment self-play samples rather than maximises.
