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
| B1 | Player interface | `crates/hexo-runner/README.md` for why the runner has none, and `crates/hexo-player/README.md` for the one a driver actually drives. Answered by neither option offered: `Game` is a nonblocking state machine, so whether a caller blocks a thread or polls a thousand games is decided outside the crate — and the seat contract lives outside it too, split into `Player` for anything that plays and `Model` for anything that trains |
| B2 | What is recorded per move | `PlyRecord`: seat, action, resulting hash, and `diagnostics` — an opaque seat-owned blob, and this one is actually persisted. `Game::plies` is the game's history; no other type keeps one |
| B3 | Search budget | `Budget`, stated once on `GameSpec` and never enforced by the game. Not copied per ply: it cannot vary within a game, so a per-`PlyRecord` copy would carry no information |
| B5 | Adjudication policy | `MatchResult`, `FailurePolicy`, `NoContest` |
| C3 | The binary crate, and its modes | `CONTAINER_SPEC.md` §3: `hexo-bot`, with `train`, `serve`, and `play`. Self-play is not a mode — it is the first phase of `train`, because one loop cannot be split into pieces that could drift from it |
| C5 | ~~`R` for dense action indexing~~ — **withdrawn**, not answered: a fixed radius-20 crop makes wins outside the crop unrepresentable, so the action space silently stops matching the game | `crates/hexo-engine/README.md`; the canonical unbounded ordering replaced it |
| C6 | One image or two | `CONTAINER_SPEC.md` §2. One: a play-only image would need a second implementation of the model's forward pass, which the no-dual-paths rule forbids and which could silently disagree with the first |
| B6 | What a seat returns — a bare action, or the whole decision | `crates/hexo-player/README.md`. The whole `Decision`: the hash attestation and the diagnostics can only be authored by the seat, so a driver that filled either in was deleting the desync detector and discarding the training annotations. `Failure::Desync` carries a refused attestation into adjudication |
| C4 | On-disk record format | `crates/hexo-records/README.md`: shard format v1, a writer that renames into place, a reader that refuses anything it cannot account for, and a `verify` that replays the record through the engine. `CONTAINER_SPEC.md` §11 states what the container needs from it |

---

## B4. Seed ownership — deliberately open, against a seam that is built

For byte-reproducible self-play something would mint and record a seed and hand
per-seat seeds to the seats.

The engine has no randomness at all, and replay determinism comes from the
stored action list rather than from a seed: a game is reproduced by replaying
its moves. Either wire a seed end to end or do not carry it — a recorded seed
that does not reproduce the game is worse than none, because it reads as a
reproducibility guarantee that was never checked. `hexo-runner` therefore ships
with **no seed field at all**, and neither the run manifest nor the shard header
has one.

**The seam exists; nothing above it does.** A `DecisionSession` takes a seed
when it is constructed and exposes `reseed`, because passing a seed in is a
different job from retrofitting one into a search already written without it —
`CONTAINER_SPEC.md` §12 argues that, and `crates/hexo-search/README.md` states
the seam. `hexo-bot` reseeds both of a lane's sessions from entropy before every
game, mixing the clock with the lane, the lane's game serial, and the seat, so
that two lanes reseeded in the same nanosecond and the two seats of one game
never share a stream. It records none of it. Sampling therefore arrived without
seeds, and self-play games are deliberately non-deterministic rather than
accidentally so.

B4 lands the day a run has to reproduce, and it lands as a small change with a
known shape: mint per-game seeds from stable game and seat ids so scheduling
cannot alter a run, hand them to the sessions that already accept them, and
record them. That is a record format version bump and a regeneration of the
data, which is how formats change here, rather than a redesign of the loop.

---

## C. Container-time, not code-time

| # | Decision |
| --- | --- |
| C1 | Transport and wire format. A line-oriented stdio protocol is the default: trivial to containerise, debuggable by hand, close to what tournament harnesses expect. |
| C2 | Handshake fields: protocol version, rules version, action-encoding version, seat, seed, budget. |

`CONTAINER_SPEC.md` §15 is the same list from the other side. It carries C1 and
C2, B4 above, and one item that is the container's rather than the code's: the
**Dockerfile**, which arrives with the first Python-backed package, because
until then it would carry a CUDA and Python stack for a loop that uses neither.
