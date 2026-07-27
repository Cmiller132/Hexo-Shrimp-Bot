# Container spec

**Status: design settled; being built.** The `train` loop is under construction
against the mock model package, in the five crates this document names —
`hexo-search`, `hexo-records`, `hexo-model`, `crates/models/mock`, and
`hexo-bot`. **There is no image and there is no Python.** No Dockerfile exists,
nothing in the workspace depends on PyO3, and no Python-backed model package has
been written. Where this document describes those, it states the target they
will be built to, not something that runs today.

This is the normative design target for the container and for those five crates;
a later phase trues it up against the code as built. Nothing here constrains
`hexo-engine`, whose contract remains `ENGINE_SPEC.md`. It still says nothing
about how any particular model works — the trunk, the encoder, and the training
objective belong to a model package (§5), and the container's entire knowledge
of them is one trait.

---

## 1. Target

One machine: Ryzen 9 7950X (16 cores / 32 threads), one RTX 4070 Ti (12 GB),
32 GB of system RAM, Windows 11 host, **Docker engine installed inside a WSL2
distro** — not Docker Desktop. GPU access through the NVIDIA Container Toolkit
installed in that distro; the CUDA driver comes from the Windows driver
projected into WSL.

The 32 GB is split deliberately rather than left to the default: `.wslconfig`
gives the distro roughly 20 GB and leaves 12 to Windows. WSL2's default is a
fraction of host RAM that moves with the Windows build, and the distro's
allocation is shared by the Docker engine and every container in it — so the
ceiling that decides how many concurrent games fit (§7.2) is a number this
machine is configured with before the first long run, not one discovered as an
OOM kill at hour nine.

Multi-GPU and multi-node are out of scope. Nothing here forecloses them, but
nothing here is shaped by them either.

### 1.1 The filesystem rule

Windows paths reach WSL over 9p and are an order of magnitude slower than ext4.
That difference dominates every other I/O decision in this spec.

- Build cache, records, checkpoints, and metrics live in **named Docker
  volumes**. Never a `/mnt/d` bind mount.
- Source is copied into the image at build time. A read-only bind mount of the
  working tree is a dev-loop convenience only.

---

## 2. One image

A single image serves every job. The alternative — a slim play-only image with a
different inference runtime — would mean a second implementation of the model's
forward pass, which the workspace rule against dual paths forbids, and which
could silently disagree with the first. The image is large; that is the price of
having one implementation.

Multi-stage build: a Rust stage compiles the binary, a runtime stage carries the
binary plus the CUDA and Python stack. No source, no toolchain, and no build
cache in the runtime image. Runs as a non-root user.

The image records the versions it was built against —
`hexo_engine::RULES_VERSION`, `ACTION_ORDER_VERSION`,
`hexo_runner::PROTOCOL_VERSION`, and the image's own build id. A checkpoint
whose manifest disagrees with any of them is a startup failure, not a warning.

**The image arrives with the first Python-backed model package.** Everything in
it that is not the Rust binary — CUDA, Python, PyTorch — is there to serve a
package written in Python, and the only package that exists is the mock, which
is Rust and wants none of it. Building the image now would produce a
multi-gigabyte runtime with nothing to serve, built against a boundary (§4) that
no code has crossed yet. Until then the loop runs as a native binary, which is
not a second deployment shape — it is the same binary before there is anything
to wrap it around.

**This settles `OPEN_DECISIONS.md` C6.**

---

## 3. One binary, three modes

`hexo-bot`, a binary crate depending on the engine, the runner, the search, the
records, and the model API. Modes are subcommands. **This settles C3.**

| Mode | Who adjudicates | Lifetime |
| --- | --- | --- |
| `train` | itself | long-lived; the whole run |
| `serve` | a host orchestrator | long-lived; many games |
| `play` | a foreign harness | one game |

**Exactly one authority per game.** `serve` and `play` never construct a `Game`
at all — they hold a `Position` mirror and answer requests. That is an ownership
property, not a config flag, so a seat-mode container is structurally incapable
of adjudicating.

`play` speaks a foreign protocol through an adapter at the edge. The native
protocol is designed for this system's needs and translated into; a foreign
protocol's assumptions never reach `hexo-runner`.

`train` is the mode being built. `serve` and `play` do not exist yet and are
blocked on C1 and C2: both are entirely wire protocol, and there is no wire
protocol. The binary therefore ships only the subcommands that work. A stub that
parses `--seat` and then exits with "not implemented" is a half-implementation
in the exact sense the workspace forbids — it publishes a command line before
the thing behind it is designed, and the flags it guessed become the constraint
the real implementation has to argue its way out of.

---

## 4. The Rust/Python boundary

Training is PyTorch in Python. That is where the ecosystem, the optimisers, and
the iteration speed are, and rewriting any of it in Rust buys nothing.

In the self-play hot loop **the forward pass is executed by embedded PyTorch
through PyO3**. `hexo-bot` embeds the interpreter; the exact `nn.Module` that
trains is the module that serves.

**That is what makes §2's argument literal rather than aspirational.** §2 rules
out a second image because it would mean a second implementation of the forward
pass — but exporting the network to ONNX or TorchScript and running it in a Rust
runtime is that second implementation under another name, whatever the file
extension says. Two graph compilers, two operator sets, two sets of numeric
behaviour, and an export step that can succeed while producing something that
does not match. The probe hash (§10.2) would then be catching export drift as a
matter of routine, which is not what it is for. With a live module there is no
export step, no second runtime, and nothing for the probe hash to catch except
real drift.

The cost is the GIL, and it is paid per crossing rather than per position.
`SUGGESTIONS.md` S3 diagnosed that correctly and the batch interface is the
whole of the answer: one crossing per batch, a few hundred crossings per second,
which amortises the lock to noise. §6 is that interface.

The swap in §10.1 is native to a live module for the same reason. Copying
weights into existing parameter storage is an operation on a module that is
already resident and already warm; there is no reload, no re-export, and nothing
to recompile.

**PyO3 stays a leaf dependency.** `hexo-bot` is a leaf binary crate that nothing
depends on, and no logic crate mentions Python, even optionally. This is the
root `README.md`'s standing rule and it is not re-argued here — it is what keeps
`cargo test` free of a Python toolchain and keeps `hexo-engine` compilable to
`wasm32`, which the `wasm32` gate checks on every run and which is precisely the
gate that would fail if a native-only dependency crept downward.

None of this is built. No crate in the workspace depends on PyO3 today, and the
mock package (§5) deliberately needs nothing from this section — which is what
makes it possible to build and test the entire loop before the boundary exists.

---

## 5. Model packages

A model is a crate. Packages live under `crates/models/<name>/`; the first is
`crates/models/mock`. Each one implements the `ModelPackage` trait from
`crates/hexo-model`, and the container's whole knowledge of a model is that
trait plus a name registry in `hexo-bot`.

A package owns, and the container never has an opinion about:

| What it owns | Why it is the package's |
| --- | --- |
| Its encoder, and the encoder version | Features are a modelling decision. The engine exposes state; nothing else may decide what a plane means (S5) |
| Its evaluator | The forward pass, and the only code that touches a network |
| Its two session constructors, one per mode | Search shape and parameters are part of the model, not the harness |
| Its move-selection policies | Sampling, temperature, and greediness are the model's, as its encoding is |
| Its diagnostics format | The bytes on `PlyRecord`; written and read by the same crate |
| Its checkpoint weight format | The container stores a file and a manifest, not a description of layers |
| Its `fit` | The training objective, the optimiser, and the data pipeline |

`--package <name>` selects one at startup, and the name is written into every
shard header and every checkpoint manifest, so no artefact on disk is ambiguous
about what produced it. Adding a GNN package or a CNN package is a new crate and
one registry entry. Neither the engine, nor the runner, nor the record format
learns anything, and no existing package is touched.

The registry is not a museum. A package that is no longer built against the
current record format and the current seam comes out in the same change that
obsoletes it, exactly as any other superseded implementation does.

**The two session constructors are required and have no defaults**, one for
self-play and one for evaluation, for the reason `hexo-player` gives for
`Model`'s two methods. A single constructor taking a `Mode` can be written to
ignore it; that compiles, passes, and produces a self-play run in which every
game is identical, and no downstream stage can detect it because the data is
well-formed.

**The canonical action ordering is the indexing contract.** The engine owns it
in both directions — `legal_rank` and `nth_legal` — and every package's policy
head is held to it: index *i* is the *i*th legal action of the position being
evaluated, and nothing else. That is what makes packages substitutable without
retraining drift. Two packages agree about what index 37 means because neither
of them gets to decide it, and the ordering is versioned
(`ACTION_ORDER_VERSION`) and pinned in every manifest and every shard header, so
a change to it invalidates the artefacts it would have silently corrupted.

**The mock package is not a placeholder to be deleted.** It is a deterministic,
weightless evaluator that exercises every seam the container has — encoder,
evaluator, both session kinds, diagnostics, shard writing, checkpoint write and
load, the probe hash, and `fit` — with no network, no GPU, and no Python. It is
what makes the whole loop testable in CI, and it is the package the container is
built against first precisely because it can be wrong in none of the ways a real
model can. A package the entire container is exercised against on every run
earns its place permanently.

---

## 6. The evaluator seam

Three types in `crates/hexo-search`, two of them implemented by the package.
**This settles `SUGGESTIONS.md` S3.**

**`Encoder` — package-owned, runs worker-side.** It turns a `&Position` into
bytes written into a caller-provided, reusable batch buffer. Worker-side is the
load-bearing part. A `Position` never crosses a thread, so nothing is cloned
into a queue, nothing has to be made `Send` that is not already, and the queues
carry buffer handles rather than positions, tensors, or Python objects.
Encoding is also the one part of an evaluation that is pure CPU work, so running
it on the worker parallelises it across the whole pool for free instead of
serialising it in front of the GPU.

**`Evaluator` — package-owned, runs batcher-side.** One call answers one whole
batch. That call is the single Python and GPU crossing (§4), and it is the only
place in the system where either is touched.

**`Evaluation` — the answer for one position.** Priors over that position's
legal actions **in canonical order**, and a value in `[-1, 1]` **from the
perspective of the side to move**.

Both conventions are normative for every package rather than conventions a
package may restate. Priors are ragged — one per legal action, in `nth_legal`
order, with no fixed crop; a fixed-radius crop was proposed as C5 and withdrawn
rather than answered, because it makes a win outside the crop unrepresentable
and so lets the action space stop matching the game. The value's sign
convention has to be stated once and obeyed everywhere, because a package that
flips it trains against targets that are exactly wrong and produces a bot that
plays to lose. No shape check sees that, no round-trip test sees it, and the
loss curve looks fine.

---

## 7. Sessions, and why nothing blocks

A seat's search is a nonblocking state machine: `DecisionSession`, in
`hexo-search`. Pump it and it emits the leaf evaluations it wants; resume it
with the results and it continues; eventually it yields the whole `Decision`.
The authorship rules are `hexo-player`'s and are unchanged — the session
authors the zobrist attestation of the position it actually searched and the
diagnostics bytes, and the driver submits the decision verbatim, because a
driver that filled in either field would be deleting the desync detector or
inventing the training annotations.

Two implementations behind one seam:

- **`MctsSession`** — PUCT, virtual loss on the path to each leaf in flight, and
  a cap on how many leaves one session may have outstanding at once.
- **`PolicySession`** — one root evaluation per move, then the package's
  move-selection policy. Policy-only self-play is the same loop with a
  degenerate search, not a second path. That matters because the first real
  training runs are policy-only, and a separate driver for them would leave the
  driver that eventually carries every run as the least exercised code in the
  system.

**Waiting is data, not a blocked thread.** A session with leaves outstanding is
a struct holding a few vectors, so a game in flight costs bytes rather than a
stack, and the number of concurrent games stops being a number of threads. This
is the driver `hexo-runner` inverted its loop for: `Game` is a state machine
precisely so that a worker can sweep many of them, answer every
`Step::NeedDecision` by pumping that slot's session, and hand whole batches of
leaves to one evaluator.

`hexo-player`'s `Player` remains the seam for a seat that answers one decision
at a time — a human, a scripted bot, a remote container behind an adapter. It is
not the self-play driver, and cannot be: `choose` returns a `Decision`, and a
session that wants to ask a question halfway through does not have one yet. The
two express the same contract about who authors a `Decision`; they differ only
in whether the seat is allowed to ask something mid-answer.

### 7.1 Topology

```
  G games, held as slots across ~13 worker threads
  +--------------------------------------+
  | Game (hexo-runner) + DecisionSession |
  |   pump    -> leaf requests           |
  |   Encoder -> bytes into a batch slot |
  |   resume(evaluations) -> Decision    |
  +--------------------------------------+
      |                                ^
      | slot ids + encoded bytes       | evaluations, keyed by slot
      v (bounded queue)                | (bounded queue)
  +--------------------------------------+
  | batcher, 1 thread                    |
  |   fills a batch of B                 |
  |   one Evaluator::evaluate call -----------> embedded PyTorch -> GPU
  |   scatters results back to slots     |
  +--------------------------------------+

  finished games --(bounded queue)--> writer, 1 thread --> records/<epoch>/
```

The worker pool is sized to the silicon — about 13 of the 16 cores — leaving the
batcher, the writer, and the interpreter's own pools somewhere to run (§13). One
batcher thread owns the GPU crossing, because that crossing has to be serialised
anyway and a second batcher would only contend for the same lock and the same
device. One writer thread owns records, so the format has exactly one writer and
no worker ever touches a file. Every queue is bounded in both directions: a
saturated GPU applies backpressure to the workers rather than growing a queue
until the process is killed.

### 7.2 G and B are different numbers

**The batch size B and the game count G are chosen from different constraints
and are two flags, `--batch` and `--games`, not one.** B comes from what the GPU
wants: the batch at which the 4070 Ti is saturated, and beyond which latency
grows faster than throughput. G comes from RAM, and from how many leaves per
game should be in flight. Tying
them together would mean either running the GPU at a batch size it does not want
or sizing host memory from a device number.

The second constraint on G is the one that is easy to miss, and it is the real
reason to run a thousand games rather than a hundred and twenty-eight. A batch
is filled from leaves and leaves come from games. With few games, each one has to
contribute many leaves at once to fill a batch — and every leaf in flight is a
virtual-loss penalty on a path whose true value is not known yet, so the search
is being steered by a placeholder. With many games, each contributes one or two
and the distortion nearly vanishes. Throughput is bounded by the GPU either way;
what more games buy is search quality per unit of compute.

For a policy-only package the arithmetic is simpler and the conclusion is the
same. Every game contributes exactly one root request per move, so filling a
batch of B needs a G of a few times B — a game is not requesting for the whole
time it exists, it is also applying moves, being recorded, and being replaced by
the next one. Same batcher, same queues, same code path.

---

## 8. `train` — the loop

Self-play and fitting are **one mode, not two**. One invocation runs a whole
training run and is expected to stay up for days.

```
hexo-bot train --run-dir <dir> --run-id <id> --package <name> --epochs N [--resume]

for epoch in 0..N:
    1. self-play   frozen weights, G concurrent games, produce records
    2. fit         consume this epoch's records, produce new weights
    3. checkpoint  write weights + manifest
    4. eval        every K epochs: current vs anchor and older checkpoints
```

**Why one long-lived process rather than one container per epoch.** The fixed
costs — CUDA context creation, interpreter startup, model load, kernel warm-up,
and whatever shape-keyed compilation the serve path accumulates — are paid once
per process, not once per epoch. Over hundreds of epochs that is the difference
between a minor overhead and a dominant one.

**Why the phases can share one process at all.** The two phases never run at the
same time, so the single GPU is never contended: no time-slicing, no MPS, no
VRAM partitioning. Records are produced and consumed within one epoch, so
nothing has to outlive the process except checkpoints and metrics.

The phases are not separately invocable. There is one implementation of the
loop, not a loop plus a set of pieces that could drift from it.

Records are written per epoch into that epoch's own directory, and the directory
is removed once the fit that consumed it has succeeded. Not before — a failed
fit leaves its input on disk to be inspected or re-run — and not later, because
on-policy records are worthless under the new weights and keeping them would
make disk growth a function of run length.

### 8.1 What "long-lived" requires

- **Bounded memory.** No cross-epoch growth. Game slots, encoded batch buffers,
  and the record buffer are allocated once and reused; the record buffer is
  cleared at the end of each epoch. Steady-state RSS after epoch 2 is a checked
  property, not a hope.
- **Resumable at epoch boundaries only.** `--resume` restarts from the last
  complete checkpoint. A crash mid-epoch discards that epoch's records, which
  costs nothing: they are on-policy and worthless under new weights.
- **Graceful stop.** `SIGTERM` finishes the phase's in-flight work, writes a
  checkpoint, and exits. `docker stop` must never lose an epoch.
- **Distinguishable exits**, pinned here so that a supervisor, a shell loop, or
  a person reading `docker inspect` does not have to infer them: **0** ran to
  completion, **2** stopped by signal after finishing cleanly, **1** failed. A
  run that ended is not a run that broke, and a `docker stop` produces a 2.
- **Observable without stopping.** One metrics line per epoch, appended to the
  run directory as it happens. No dashboard, no metrics service, no scraping
  endpoint.

---

## 9. State

Three kinds, three homes. Getting this wrong is the usual way container designs
fail.

| Kind | Examples | Where |
| --- | --- | --- |
| Baked | binary, CUDA/Python stack, version constants, the package registry | the image |
| Injected | run directory, run id, package name, checkpoint reference, epochs, G, B, game spec | flags and env |
| Accumulated | checkpoints, metrics, records | the run directory, on a volume |

Injected values are never guessed. A missing or unparseable one is a loud
startup failure.

`--run-dir` names the root of the accumulated state, and the container maps a
named volume onto it. A flag rather than a compiled-in `/var/lib/hexo`, because
where the volume lands is the operator's decision and not the binary's — and
because the same binary running natively on the host during development should
not have to be told a Docker-shaped lie about where its state lives. It is
required and never defaulted: silently choosing a directory to write days of
training into is exactly the substitution this workspace forbids.

```
<run-dir>/runs/<run-id>/
  manifest.json              run config and versions
  metrics.jsonl              one line per epoch
  checkpoints/<epoch>/
    weights.<ext>            the package's own format
    manifest.json
  records/<epoch>/           transient; removed after a successful fit
```

The run manifest has no seed field, deliberately (§12).

A crashed run leaves a partial epoch directory, never a corrupted checkpoint:
checkpoints are written to a temporary name and renamed into place.

---

## 10. Checkpoints

A checkpoint is weights plus a manifest. The manifest pins:

- the **package name** and **package version** — which crate produced these
  weights and which version of it;
- the **encoder version** — the package's own, bumped whenever the bytes its
  encoder writes change meaning;
- `RULES_VERSION`, `ACTION_ORDER_VERSION`, and `PROTOCOL_VERSION`;
- the **probe hash** (§10.2).

A checkpoint whose manifest disagrees with the binary on any of them is a
startup failure, not a warning (§2).

The manifest does not describe the architecture. How many layers there are and
what shape they have is the package's business, carried inside its own weight
file; the container's manifest answers only which package wrote this, which
version, and whether it is compatible with this binary.

`--checkpoint` takes a **reference**, resolved from a filesystem path, a
`<run-id>/<epoch>` pair, `latest`, or a baked-in default. Shipping a
tournament-ready bot is then a thin image layer that copies one checkpoint in
and sets the default — not a separate architecture.

### 10.1 Swapping

**Copy weights in place; never rebind the module.** Whatever graph capture or
compilation the serve path has accumulated records fixed pointers, so writing
into existing parameter storage keeps all of it valid and a swap costs one small
host-to-device copy. Replacing the module invalidates that state and re-pays the
entire warm-up. Embedding a live interpreter (§4) is what makes this the natural
operation rather than a trick.

Older checkpoints for evaluation use a **slot pool**: weights are small enough
to keep many on disk, but each live module carries expensive compiled state, so
the number of live slots is capped (current, opponent, anchor) and an opponent's
weights are copied into a slot on demand.

### 10.2 The probe hash

A fixed set of probe positions is forwarded after every load and every swap, and
the hash of the outputs is compared against the manifest. One forward.

`hexo-model` computes it, over the **exact bytes the `Evaluator` returns** —
not over the weights, and not over a re-derived summary. The probe set is a
fixed batch forwarded whole, so nothing about batch shape can change which
kernel runs, and the same binary with the same weights therefore produces the
same hash every time. Everything that could vary is then a real difference.

It exists because every failure it catches is silent: the wrong checkpoint
loaded, a swap that constant-folding turned into a no-op, a mismatched encoder
version, a scrambled action ordering, or a runtime that drifted between build
and run. None of these crash. All of them train or play against the wrong
weights indefinitely.

---

## 11. Records

`hexo-records` owns the on-disk game record: a versioned binary shard, written
once and read strictly. **This settles C4.**

A shard opens with a header pinning the record format version, `RULES_VERSION`,
`ACTION_ORDER_VERSION`, and `PROTOCOL_VERSION`, plus the run id, the epoch, the
mode the games were played in, the package name, and the checkpoint reference
the weights came from. Then, per game: the `GameSpec` it was played under, the
`MatchResult` with its full adjudication payload, and the plies — each a seat,
an action, the resulting zobrist, and optional diagnostics bytes.

The result is carried whole rather than flattened to a winner because a training
pipeline selects on it. A forfeit is a decisive, contested result and a real fact
about a match, but the stones on the board are an abandoned game; a consumer
that can only see "P0 won" cannot tell that from a win on the board.
`hexo-runner`'s README argues this at length, and the format's job is not to
throw away what it went to the trouble of keeping — including the `ActionId` and
`MoveError` on an illegal-move forfeit, and the seat and `Failure` on a
no-contest.

The mode lives in the header rather than on each game because a shard is written
by one phase of one epoch by one package; a per-game copy would carry no
information. This is also what closes `hexo-player`'s open item on recording
which mode a game was played in. The `GameSpec` goes the other way and is
recorded per game, because a result cannot be read without it — a `PlyCap` draw
means nothing unless the cap that produced it is in hand — and because a run is
free to vary the budget between games in a way it cannot vary the mode within a
phase.

**One implementation, in Rust.** When `fit` needs to read records, the Python
side calls this reader through the embedded interpreter (§4) rather than growing
a parser of its own. A second parser is a second definition of the format, and
the two would disagree exactly where it is hardest to notice — a diagnostics
blob's length prefix, the end of a ragged ply list, an unset optional. Binary
and versioned rather than JSON because a shard is the largest artefact this
system writes and the fastest thing it re-reads, and because a strict reader
that refuses an unknown version is worth more than one that tolerates two
shapes.

The detailed statement of the format belongs in `crates/hexo-records/README.md`;
this section says only what the container needs from it.

---

## 12. Seeds

**B4 stays deferred, and the deferral is designed rather than postponed.** A
session takes a seed when it is constructed and exposes `reseed`. That is the
entire seam B4 will need, and it is built now because passing a seed in is a
different job from retrofitting one into a search already written without it.

Nothing above that seam exists. The driver seeds sessions from entropy, games
are deliberately non-deterministic, nothing mints a per-game seed, nothing
records one, and neither the run manifest nor the shard header has a field for
one. `OPEN_DECISIONS.md` B4 argues why and it is not re-argued here: a recorded
seed that does not reproduce the game is worse than none, because it reads as a
guarantee nobody ever checked.

B4 lands the day reproducible self-play is actually wanted, and it lands as a
small change with a known shape — mint per-game seeds from stable game and seat
ids so that scheduling cannot alter a run, hand them to the sessions that
already accept them, and record them. That is a record format version bump and a
regeneration of the data, which is how formats change here, rather than a
redesign of the loop.

---

## 13. Runtime knobs

Four, and three of them bite in a container specifically.

- **Thread budget.** A container sees the host's 32 threads regardless of its
  quota, so the Rust worker pool, PyTorch's intra-op pool, and OpenMP will each
  claim all of them and thrash. Each is set explicitly and budgeted against the
  others, against the ~13 workers plus batcher plus writer of §7.1. `--cpus` is
  a quota, not a pinning; use `--cpuset-cpus` if pinning is wanted.
- **`--shm-size`.** The 64 MB default breaks anything using shared-memory IPC or
  dataloader workers.
- **WSL2 memory ceiling.** The distro, the Docker engine, and every container
  share the one budget set in `.wslconfig` — the ~20 GB of §1. It is what bounds
  G.
- **12 GB of VRAM** is the hard ceiling on the device side. It decides B, along
  with the resident modules of the slot pool (§10.1). It does not decide G.

---

## 14. Out of scope

No orchestrator, scheduler, dashboard, or experiment tracker. No model registry
or promotion logic. No multi-node, no multi-GPU. No inference service separate
from the process that uses it. No adjudication in seat modes. No hyperparameter
sweep and no multi-run scheduler: a run is one process and one directory.

## 15. Still open

- **C1, C2** — transport, wire format, and handshake fields for `serve` and
  `play`. A line-oriented stdio protocol remains the default assumption, and the
  constraint from `ENGINE_RL_AUDIT.md` stands with it: stdio is for external
  interoperability, never for self-play.
- **The Dockerfile itself.** It arrives with the first Python-backed package,
  for the reason §2 gives — until then it would carry a CUDA and Python stack
  for a loop that uses neither.
- **B4** — seed ownership, deferred against a seam that already exists (§12).
