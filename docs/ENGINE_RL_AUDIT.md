# Engine and Runner Audit for Massively Parallel RL

Status: review findings, each annotated with what happened to it. The P0 and the
coverage-update P1 are fixed, and are kept below as a record of what was wrong
and how it was closed. The other three P1s are still open — but no longer
unmeasured, which is what *Measured verdicts on the P1 list* settles.

This is a snapshot with annotations, not a live plan. A finding stays here once
it is closed; what it decided lives with the thing it decided.

## Verdict

`hexo-engine` is a strong deterministic scalar rules core, but the repository is
not yet a maximally performant massively parallel RL system.

The engine has good foundations: independent positions have no shared mutable
state or locks, legal actions have a deterministic order, Zobrist hashing is
incremental, and `Search` provides safe make/unmake. However, the runner is still
an empty scaffold, one engine correctness issue can produce unusable "legal"
actions, and several hot paths need measurement and optimization before the
engine is used at large rollout scale.

**Since the review**, the correctness issue is fixed, the runner is implemented,
and the hot paths are measured — so what is left of this verdict is the last
clause, narrowed to the three P1s below.

## P0: Legal action enumeration disagrees with validation — FIXED

`Position::place` marks every cell in the radius-8 disk as frontier without
checking `HexCoord::is_valid()`:

- `crates/hexo-engine/src/position.rs:587-601`

`Position::legal_actions` enumerates the frontier directly:

- `crates/hexo-engine/src/position.rs:269-307`

But `Position::is_legal` and `Position::advance` reject coordinates outside
`COORD_LIMIT`:

- `crates/hexo-engine/src/position.rs:317-340`
- `crates/hexo-engine/src/position.rs:638-665`

This is reproducible with a legal sparse walk to `q = 16000`. At that point:

- `legal_count()` reports 270,216 actions.
- 136 enumerated actions have invalid `q`, `r`, or `s`.
- The first observed invalid action was `(15993, 8)`.
- `is_legal()` returned `false`.
- `advance()` returned `MoveError::CoordOutOfBounds`.

A runner or policy may therefore sample from `legal_actions()` and receive an
action the engine immediately rejects.

### Resolution as shipped

Reproduced exactly: at `q = 16000`, `legal_count()` was 270,216 with 136
enumerated actions invalid, the first being `(15993, 8)` at `s = -16001`.

Fixed at the source rather than by filtering at each read. The disk update skips
cells outside the coordinate domain, so the frontier plane holds only valid
coordinates and every accessor over it inherits the guarantee — `legal_count()`
keeps its `O(1)`, and there is one writer to keep correct instead of four
readers. Both halves derive the same predicate from the same coordinate, so they
remain exact inverses. Post-fix count is 270,080, and no golden vector moved,
confirming the region is unreachable in ordinary play. (The filter has since
moved into `Grid` and become a per-row clip; see the P1 section below.)

`tests/boundary.rs` covers all six faces via the four axis-aligned walks — each
drives two cube coordinates to their limits — and asserts (1) every enumerated
action is valid, (2) every one passes `is_legal`, (3) enumeration length equals
`legal_count`, (4) advancing a boundary-adjacent sample never yields
`CoordOutOfBounds`, plus `audit()` and an apply/undo taken at the face, which is
what pins the two halves of the pair to the same filter.

One thing the original finding did not anticipate: **the two diagonal directions
cannot reach a face at all.** A diagonal widens both arena dimensions, so the
padded bounding box grows as an area and `MAX_GRID_CELLS` refuses at around
`|q| = 1984`. That is asserted as its own case — a clean representation limit,
not a rule violation, with the position intact afterwards.

`BoardExtentExceeded` remains a separate representation-limit case because the
`is_legal` contract deliberately does not allocate or test arena growth.

## P1: The 217-cell coverage update is the main scalar hot path — FIXED

Every placement and undo walked all 217 cells in `DISK8`, and for each cell
mapped the same coordinate through `locate` three or four times: once to
increment coverage, once to read it back, once to test occupancy, and once more
to set the frontier bit. Doubled by undo, that was roughly 75% of an interior
`apply`+`undo` pair and 84% of an edge one.

### Resolution as shipped

The recommended first optimization, essentially unchanged:

1. map the placed coordinate once;
2. represent the radius-8 disk as 17 contiguous row runs;
3. update coverage directly through slices;
4. update frontier words with masks and popcounts.

`Grid::disk_runs` produces the runs and `Grid::add_cover_disk` /
`Grid::remove_cover_disk` are now the only writers of coverage, so `position` no
longer knows the disk is a disk. Safe Rust, and the representation is unchanged.

Two things fell out that the recommendation did not anticipate.

**The per-cell domain filter became a per-row clip, and got more exact rather
than less.** `place` used to test `is_valid()` on each cell behind a
`disk_is_interior` fast path. At a fixed `q`, `is_valid` reduces to `r` lying in
`[max(-LIM, -LIM - q), min(LIM, LIM - q)]`, so the clip is now two `min`/`max`
pairs per row and the fast path is gone — one predicate rather than a predicate
plus a sufficient condition for it.

**`DISK8` is worth more now than when it was load-bearing.** It is no longer the
thing the machine follows, so it survives as an *independent* statement of the
same cell set: the tier-C frontier assertion walks it offset by offset on every
apply and undo, and `grid`'s tests compare the two formulations directly. A wrong
row run and a wrong offset are both symmetric bugs, so neither can be checked
against itself.

Measured against the pre-change baseline, at plies 1 / 32 / 96 / 256:

| Benchmark | Change |
| --- | ---: |
| `apply_undo/edge` | **−45.1% / −58.2% / −58.8% / −60.4%** |
| `apply_undo/interior` | −37.7% / −38.4% / −42.8% / −42.2% |
| `advance` | −35.4% / −14.9% / −36.4% / −34.0% |
| `replay/256` | −45.2% |

The edge case is the one to read, since real games play rim placements: an edge
`apply`+`undo` pair went from 1.09 us to 432 ns at ply 256. That is at the top of
the 2–3x ceiling the first measurement predicted. `replay` halving is the same
win seen end to end, since replaying 256 plies is 256 disk updates and their
growth events.

The golden vectors, the boundary tests, and the property suite are all unmoved.

### Higher-upside prototype

Prototype a one-bit `covered` plane plus a 217-bit `newly_covered` mask in each
`Undo`. LIFO undo can clear exactly the coverage introduced by the most recent
move.

Potential benefit:

- position storage falls from roughly 11 to 4 bits per arena cell;
- disk updates become row-mask operations;
- clones become substantially smaller.

Cost:

- a larger undo token, approximately 40-48 bytes instead of a small scalar
  delta.

Measure this trade-off using realistic MCTS depths before adopting it.

## P1: Search excursions can permanently inflate a worker

Undo deliberately retains arena geometry:

- `crates/hexo-engine/src/search.rs:157-165`

`Grid::reserve_around` returns immediately while an action fits in the retained
arena:

- `crates/hexo-engine/src/grid.rs:463-466`

Legal and stone iteration scan allocated words:

- `crates/hexo-engine/src/position.rs:892-930`

One deep, spreading MCTS rollout can therefore grow the arena, unwind to a small
root position, and leave all subsequent scans, clones, and memory usage paying
for the largest explored position. The documented "next growth" reshaping does
not help when later actions continue to fit in the oversized allocation.

### Recommended resolution

Benchmark and implement one of:

- thresholded compaction at a search-root boundary;
- `Search::compact_floor`;
- disposable/recycled scratch positions that never bloat the persistent player
  mirror;
- a caller-selected search arena budget.

Also make `BitScan::next_coord` return immediately when `remaining == 0` so the
final iterator call does not scan trailing empty words.

## P1: Position cloning has four allocations

`Grid` owns four independent heap buffers: two occupancy planes, frontier, and
coverage:

- `crates/hexo-engine/src/grid.rs:59-79`

Derived `Clone` therefore performs four allocations and four copies — five since
move history was added to `Position`. The documentation's "clone is a memcpy" and
"allocates once" descriptions were useful shorthand but not literal; both have
since been corrected in `README.md`, `crates/hexo-engine/README.md`, and
`docs/ENGINE_SPEC.md` §5.1 to say what actually happens.

The first opened position allocates 4,096 cells, or 5,632 bytes of plane payload
before allocator overhead. An 80 by 80 live region will commonly pad and round
to a 128 by 128 arena, or 22,528 payload bytes rather than the unpadded 8.8 KB
estimate.

At large actor counts, state copies and allocator contention can become material
before search trees and model buffers are counted.

### Recommended resolution

- Add explicit buffer-reusing clone/reset support for long-lived game slots.
- Reuse one `Search` per worker so its undo capacity is retained.
- Consider consolidating planes into one or two backing slabs.
- Avoid two player mirrors when a single in-process self-play session controls
  both seats.
- Measure allocations and copied bytes per clone and per completed game.

## P1: The read surface is too scalar for model encoding

`Stones::next` finds a bit slot, converts it to a coordinate, and then maps the
coordinate back into the grid to determine its owner:

- `crates/hexo-engine/src/position.rs:939-949`

There is no bulk, representation-independent export for occupancy planes,
features, or ragged legal actions. Repeated scalar calls will add overhead when
thousands of positions are encoded for inference.

### Recommended resolution

Add caller-buffer APIs that are defined in board coordinates and do not expose
arena origin or stride. Examples include:

- copying selected coordinate regions into reusable per-player planes;
- filling packed feature buffers directly;
- filling `action_ids` plus row offsets for ragged inference batches;
- returning owner information directly from the bit scan slot.

Do not queue or serialize `Position` clones for inference.

## Runner integration — SHIPPED

Built as recommended: `Game` is a nonblocking state machine privately owning the
canonical `Position`, results and aborts are separate types, and the annotation
blob is opaque to the runner. `crates/hexo-runner/README.md` is the live
statement of it; the recommendation is kept below as the record of what was
asked for.

Keep `hexo-runner` synchronous and deterministic at its core, but do not build
massive self-play as one blocking OS thread per game.

The core should be a nonblocking state machine:

```text
Game::request() -> NeedDecision
Game::submit(decision) -> Transition | Finished
```

`Game` privately owns the single canonical `Position`. Players receive one
initial mirror and then an accepted move stream.

For each placement:

1. verify game/request generation and the player's root Zobrist hash;
2. reserve bounded record capacity;
3. call canonical `Position::advance()` exactly once;
4. append the action, transition, budget usage, and player annotation;
5. broadcast the accepted action and post-move hash;
6. have player mirrors call `Position::advance` and acknowledge the hash;
7. resolve win, ply-cap truncation, forfeit, timeout, or abort.

Do not call `is_legal()` before `advance()` in the runner; that duplicates hot
path validation.

Keep completed match results separate from infrastructure aborts:

- win, resignation, illegal-move forfeit, timeout, and ply-cap truncation are
  completed results;
- representation limits, integrity failures, cancellation, recorder failure,
  and runner invariants are aborts;
- a failed final notification must not rewrite an already established result.

Player-owned training annotations should be generic and typed in-process. At a
remote boundary they should become bounded bytes with a schema/version tag.

## Execution topology for massive RL

Recommended topology:

```text
Fixed CPU actor shards
  -> many GameSlots and search trees per shard
  -> reusable encoded leaf slots
  -> bounded queues containing slot IDs
  -> one inference batcher per GPU
  -> responses keyed by game/tree/node/generation
  -> bounded record queue
  -> chunked asynchronous shard writer
```

Important constraints:

- allow multiple pending leaves per actor instead of blocking per inference;
- queue buffer handles, not `Position`, tensors, or Python objects;
- use ragged `action_ids + offsets`, never a fixed action crop;
- apply bounded backpressure at inference and recording;
- derive seeds from stable game and seat IDs so scheduling does not alter runs;
- use line-oriented stdio only for external interoperability, not self-play;
- avoid CPU oversubscription between actors, PyTorch, Rayon, and OpenMP.

## Current performance snapshot

**These predate the 17-row disk update.** They are kept as the "before" side of
that change; the deltas are in the P1 section above.

Read-only release microbenchmarks were run on a Ryzen 9 7950X with Rust 1.95.
For a random 200-stone position with 6,525 legal actions:

| Operation | Observed result |
| --- | ---: |
| Legal enumeration | 304.7 million action items/s |
| Complete legal-list scans | 46.7 thousand/s |
| Position clone and drop | approximately 0.91 microseconds |
| Interior apply and undo | approximately 0.64 microseconds |
| Edge apply and undo | approximately 0.89-0.98 microseconds |
| Opening apply and undo | approximately 1.06 microseconds |
| Thread scaling | 1.4M cycles/s at 1 thread to 23.9M at 32 threads |

These results show good scalar speed and useful independent-position scaling.
They do not establish maximal performance.

## Required benchmark gate

The repository currently has no checked-in benchmark or performance CI:

- `.github/workflows/ci.yml:23-36`

Add reproducible release benchmarks for:

- `Position::advance`;
- `Search::apply` plus `undo`;
- legal and stone enumeration;
- clone latency, allocation count, and copied bytes;
- new-game/reset allocation;
- compact and pre-grown-then-unwound positions;
- arena growth p50, p95, and p99 latency;
- representative plies such as 1, 32, 96, 256, and 512;
- same-process scaling at 1, 2, 4, 8, 16, and 32 workers;
- complete self-play using a mock batched evaluator;
- GPU batch fill, queue wait, inference latency, nodes/s, games/s, and generated
  training positions/s;
- RSS per 1K, 10K, and 100K active lanes.

The primary acceptance metric should be end-to-end generated training positions
per second at a fixed search budget and identical outputs, not isolated engine
placements alone.

## Measured verdicts on the P1 list

The benchmark suite (`crates/hexo-engine/benches/`) now exists, and it settles most
of the recommendations above. The snapshot at the bottom of this document
reproduced almost exactly on the same CPU family, so nothing in it needs
retracting — but several of the conclusions drawn *from* it do.

| Proposal | Verdict | Evidence |
| --- | --- | --- |
| Restructure the 217-cell disk walk into 17 row spans | **Done.** The right target, and it paid. | The walk was ~75% of an interior `apply`+`undo` pair and ~84% of an edge one — ~1.1 ns (~4 cycles) per disk cell, matching the audit's "2–4 mappings per cell". See the P0/P1 section above for what shipped. |
| Buffer-reusing clone / consolidating planes into one slab | **Not worth it on latency.** | A 44 KiB ply-256 clone is 794 ns, of which allocation is only ~50–100 ns (6–13%); the rest is the copy at ~65 GB/s. Consolidating planes buys ~7%, full reuse ~13%. More telling: copy-on-descend is 1,213 ns against make/unmake's 613 ns — only **2.0x**, so the lever is cloning *less often*, not cheaper. Justified only by allocator contention and RSS at high actor counts, which this suite does not measure. |
| `row_any` row-summary bits | **Not worth it.** | Enumeration is item-bound: throughput is flat at 326–338 M items/s across every ply and both arenas, and tripling the empty words costs `legal_actions` **3.0%**. |
| Return the owner from the bit-scan slot (read surface) | **Done, and it is a trade rather than a free win.** | `stones` cost 4.3 ns per stone against `legal_actions`' 2.9 ns per action; the 1.4 ns delta was a second `owner()` lookup re-deriving what the scan had already located. `BitScan::next_slot` now yields the `(word, bit)` slot and `Grid::owner_at` reads the owner out of it. `stones` improves 14–57% (42.9% on the inflated arena, where the exhaustion early-out below also lands) — but routing both iterators through a slot-returning scan costs `legal_actions` **3–6%**. See the note below the table. |
| Search excursions permanently inflating a worker | **Real, but it is an RSS problem, not a latency one.** | An unwound excursion costs `legal_actions` 3% and `stones` 70%, but 4x the memory — 22 → 88 KiB for the same position. Compaction should be argued from footprint, not speed. |
| `legal_rank` / `nth_legal` are "much worse at extreme arena sizes" | **Does not hold.** | Quadrupling the words slowed the prefix 97% and the walk 2% — and the prefix still won by 67x. It loses only below ~0.13 legal cells per word; measured densities are 3.3 to 14.4. |

### The enumeration trade, stated plainly

`stones` and `legal_actions` share one bit scan, and making it hand back the
`(word, bit)` slot is what lets `stones` stop looking the owner up twice. The
same change costs `legal_actions` 3–6%, and **that is not obviously a good deal
in absolute terms**: at ply 256 a full `legal_actions` walk is 22 us against
`stones`' 1.3 us, because there are twenty times more legal cells than stones. If
an encoder calls both once per position, +6% on the larger one outweighs −14% on
the smaller.

It is shipped anyway, for two reasons. The gap closes on the arena shape that
actually hurts — inflated, `stones` is −42.9% against `legal_actions`' +3.7% —
and the alternative is two bit scans, which is the duplication this crate refuses
everywhere else. Worth revisiting if a real encoder's call ratio turns out to be
anywhere near 1:1.

Three placements of the exhaustion early-out were measured and they are not
close: at function entry costs `legal_actions` 3–6%; inside the loop after the
yield costs it 11–48%, despite being one branch per *word* rather than per item;
and omitting it entirely still leaves 5–15%, so the early-out was never the source
of that cost. Likewise `LegalActions` calling `next_slot` directly rather than
through the coordinate wrapper measures 11.09 us against 10.48 us at ply 96. The
loop body staying in the shape the optimiser already chose dominates the branch
arithmetic in every one of these, which is the general lesson.

Two things the numbers say that nobody asked about:

- **`advance` does not scale with board size at all** — 377–419 ns across a 256x range
  in stones and an 8x range in arena words, and `windows_through` is flat at 52–59 ns.
- **Arena growth is amortized to nothing.** Replaying 256 plies from empty, including
  every reallocation and recentring copy, is 437 ns/ply — 4% above steady-state
  `advance`.

## Recommended order of work

1. ~~Fix the coordinate-limit enumeration contract and add boundary properties.~~
   Done — see the P0 section.
2. ~~Add benchmarks and preserve representative performance fixtures.~~ Done — see
   the verdicts above.
3. ~~Implement the 17-row disk update and small iterator optimizations.~~ Done —
   the row runs, the owner-from-slot read, and an exhaustion early-out in
   `BitScan` so a spent iterator stops scanning trailing empty words.
4. Add clone/reset reuse and a search-root compaction policy. Still open, and
   still the right framing: this is an RSS problem, not a latency one, so it
   should be argued from footprint. There is no `Search::compact` today, and
   `reserve_around` only re-shapes on a growth event — so an unwound excursion
   holds its arena until something forces a reallocation.
5. Add bulk encoder-facing read APIs. Deliberately still open: the shape is
   dictated by an encoder that does not exist yet, and the workspace does not
   keep two versions of anything.
6. Implement the deterministic runner state machine and result model.
7. Add bounded actor, evaluator, and record pipelines.
8. Prototype bit-covered storage or a batched SoA engine only if end-to-end
   profiling still shows the scalar engine as material.

Keep the current engine as the authoritative scalar oracle. Any optimized
batch/SoA implementation should be differentially tested against it on every
transition.
