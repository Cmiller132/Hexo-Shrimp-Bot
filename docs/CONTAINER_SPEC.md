# Container spec

**Status: built through the first real package.** `hexo-search`,
`hexo-records`, `hexo-model`, `crates/models/mock`, and `hexo-bot` are
implemented, and `train` and `match` run against the mock package. MantisNet is
now the first network-backed package, at `crates/models/mantisnet`: its Rust
encoder, KLENT-improved evaluator opinion, sessions, diagnostics, Torch
checkpoint sealing, and proving loads all run through the same container
seams. `hexo-bot` owns the live PyO3/Torch crossing, while every logic crate
remains Python-free. The `mantisnet-train` image and compose environment live
under `docker/`; Python training remains the production KLENT loop under
`python/mantisnet`, because the package deliberately declines container-side
`fit` rather than supplying a second, partial trainer.

This is the normative design target for the container crates,
and it has been trued up against the code as built. Nothing here constrains
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

- Long-lived build cache, records, checkpoints, and metrics belong in
  **named Docker volumes**, never a `/mnt/d` bind mount.
- The as-built dependency image bind-mounts source at `/workspace` for the
  development and proof loop. A source-free release image may copy the built
  binary later; that packaging change does not create another model runtime.

---

## 2. One image

A single image serves every job. The alternative — a slim play-only image with a
different inference runtime — would mean a second implementation of the model's
forward pass, which the workspace rule against dual paths forbids, and which
could silently disagree with the first. The image is large; that is the price of
having one implementation.

The as-built `docker/Dockerfile` produces the `mantisnet-train` image: CUDA,
the locked `/opt/venv` Python/PyTorch environment, and the Rust/maturin tools
needed to compile mounted source. `docker/compose.yaml` mounts the checkout at
`/workspace`; `REPO_ROOT` lets a worktree supply that checkout, and
`CARGO_TARGET_DIR=/workspace/target-wsl` keeps Linux artefacts separate from
the Windows target directory. This is the development and training image in
use today. A source-free, binary-only release image can harden deployment
later without introducing a second model runtime.

The binary and checkpoint manifests record the versions that govern behaviour:
`hexo_engine::RULES_VERSION`, `ACTION_ORDER_VERSION`, and
`hexo_runner::PROTOCOL_VERSION`, plus package and encoder versions. A
checkpoint whose manifest disagrees with any of them is a startup failure, not
a warning. The image carries the locked Python runtime that executes the same
MantisNet module as training; there is no exported ONNX or TorchScript graph.

**The image arrived with the first Python-backed model package.** Everything
large in it — CUDA, Python, and PyTorch — serves MantisNet's existing training
and live-forward implementation. The mock package still needs none of it and
continues to exercise the complete logic path without Python.

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

`train` ships. `serve` and `play` do not exist and remain blocked on C1 and C2:
both are entirely wire protocol, and there is no wire protocol. The binary
therefore ships only the subcommands that work. A stub that parses `--seat` and
then exits with "not implemented" is a half-implementation in the exact sense
the workspace forbids — it publishes a command line before the thing behind it
is designed, and the flags it guessed become the constraint the real
implementation has to argue its way out of.

**A fourth subcommand shipped alongside `train`, and it is not a fourth mode.**
`match` plays two seats — each named by package, checkpoint, package config, and
session variant — over the same driver `train`'s phases use, and prints a JSON
report. It adjudicates locally exactly as `train` does: it holds its own `Game`
values and both seats are in-process sessions, so "exactly one authority per
game" is untouched and no protocol is implied. It is a local benchmarking
harness — how one checkpoint is measured against another, or two search shapes
against each other over one set of weights — rather than a deployment shape, and
it writes no records (§11). `crates/hexo-bot/README.md` states its flags and its
seat grammar.

`init` is the other utility subcommand, also not a mode. It asks one package to
write an epoch-0 checkpoint under a sibling `.incomplete` directory and
atomically renames it into place; an existing destination is refused. For
MantisNet, package config
`tau=<F>,lambda=<F>,source=<training-checkpoint.pt>` copies the authoritative
Python `.pt`, loads that copy through the live evaluator, and writes the
manifest and probe hash that make later loads provable. It constructs no game
and claims no new authority.

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
root `README.md`'s standing rule and it is not re-argued here. Logic-crate tests
remain free of a Python toolchain, and `hexo-engine` remains compilable to
`wasm32`, which the `wasm32` gate checks on every run and which is precisely the
gate that would fail if a native-only dependency crept downward.

The boundary is now built. `crates/models/mantisnet` defines Python-free
`ForwardLoader` and `Forward` traits over its Rust `RawBatch` and flat ragged
`RawOutputs`. The package owns all evaluator logic; the executable injects the
only implementation that loads `mantisnet.klent.run.load_model`, constructs CPU
Torch tensors, performs one live-module call for the whole batch, and returns
the policy logits and Q values unchanged. The package then applies KLENT
equation 3 and produces the public `Evaluation`. No `Position`, evaluator
semantics, Python object, tensor, GIL token, or device type crosses that trait.

PyO3 is an unconditional dependency of the `hexo-bot` leaf rather than a Cargo
feature. Thus the ordinary binary and the container build have the same
registry and boundary; there is no feature-unification shape in which MantisNet
silently disappears. `PYO3_PYTHON` chooses the interpreter at build time.
`HEXO_PYTHON` may name the same interpreter at runtime so the binary discovers
its `sys.path` and refuses a major/minor mismatch; otherwise PyO3's configured
interpreter is used. The mock and every logic crate still run without loading
an interpreter.

---

## 5. Model packages

A model is a crate. Packages live under `crates/models/<name>/`; the registry
currently carries `mock` and `mantisnet`. Each one implements the
`ModelPackage` trait from `crates/hexo-model`, and the container's whole
knowledge of a model is that trait plus the registry in `hexo-bot`.

A package owns, and the container never has an opinion about:

| What it owns | Why it is the package's |
| --- | --- |
| Its encoder, and the encoder version | Features are a modelling decision. The engine exposes state; nothing else may decide what a plane means (S5) |
| Its evaluator | The model opinion and the raw-forward trait it delegates through |
| Its two session constructors, one per mode | Search shape and parameters are part of the model, not the harness |
| Its move-selection policies | Sampling, temperature, and greediness are the model's, as its encoding is |
| Its diagnostics format | The bytes on `PlyRecord`; written and read by the same crate |
| Its checkpoint weight format | The container stores a file and a manifest, not a description of layers |
| Its `fit`, or its explicit refusal | The training objective, optimiser, and data pipeline cannot be invented by the container |

`train --package <name>` selects one at startup, and a `match` seat names its
own; either way the name is written into every shard header and every checkpoint
manifest, so no artefact on disk is ambiguous about what produced it. Adding a
GNN package or a CNN package is a new crate and one registry entry. Neither the
engine, nor the runner, nor the record format learns anything, and no existing
package is touched.

The registry is not a museum. A package that is no longer built against the
current record format and the current seam comes out in the same change that
obsoletes it, exactly as any other superseded implementation does.

`fit` has no implementation default, but an implementation may return
`PackageError::Unsupported` when the production training loop still lives
elsewhere. That refusal is sanctioned only when it names the real loop and the
owner decision needed to move it. MantisNet does exactly that:
`mantisnet.klent.run` remains the production KLENT trainer, and the package
does not scaffold a container trainer that would consume records without
faithfully reproducing it.

**The two session constructors are required and have no defaults**, one for
self-play and one for evaluation, for the reason `hexo-player` gives for
`Model`'s two methods. A single constructor taking a `Mode` can be written to
ignore it; that compiles, passes, and produces a self-play run in which every
game is identical, and no downstream stage can detect it because the data is
well-formed.

A third constructor, `variant_session`, is optional and **may** default, because
its default refuses. A variant names a search shape for a comparison or a
benchmark match (§3), the vocabulary is the package's own, and a package that
defines none inherits an honest "no such variant" rather than answering with a
shape nobody chose. That is the line: a default that answers is forbidden, a
default that declines is not.

**The canonical action ordering is the indexing contract.** The engine owns it
in both directions — `legal_rank` and `nth_legal` — and every package's policy
head is held to it: index *i* is the *i*th legal action of the position being
evaluated, and nothing else. That is what makes packages substitutable without
retraining drift. Two packages agree about what index 37 means because neither
of them gets to decide it, and the ordering is versioned
(`ACTION_ORDER_VERSION`) and pinned in every manifest and every shard header, so
a change to it invalidates the artefacts it would have silently corrupted.

**The mock package is not a placeholder to be deleted.** It is a deterministic,
weightless evaluator that exercises every required package seam — encoder,
evaluator, both session modes, diagnostics, shard writing, checkpoint write and
load, the probe hash, and `fit` — with no network, no GPU, and no Python. It is
what makes the whole loop testable in CI, and it is the package the container is
built against first precisely because it can be wrong in none of the ways a real
model can. A package the entire container is exercised against on every run
earns its place permanently.

**MantisNet is the first real package.** Its worker encoder and the
`python/hexo-py` extension call one Rust representation implementation, and
`MODEL_REPR_VERSION` is owned by that package alone. Its self-play session
samples the improved policy π′; its evaluation session uses the shared
`GumbelSession` at 32 simulations and 16 root candidates. Its exact variant
grammar is `policy`, `mcts:visits=..,inflight=..,cpuct=..`, and
`gumbel:sims=..,m=..`. Its nine diagnostic bytes carry a version, v̂, and π′
entropy for the acted self-play position.

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
serialising it at the forward boundary.

**`Evaluator` — package-owned, runs batcher-side.** One call answers one whole
batch. That call contains the single Python/device crossing (§4), and it is the
only place in the system where either may be touched. The current MantisNet
boundary loads and forwards on CPU; changing device placement remains inside
the injected forward and does not move the seam.

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

MantisNet resolves the leaf rule by splitting mechanics from semantics. Its
encoder bytes decode and collate into a Python-free `RawBatch`; one injected
`Forward::forward(&RawBatch)` returns flat policy-logit and Q-value vectors,
ragged by the batch's canonical legal offsets. The evaluator refuses malformed
lengths and applies

`π′(a) ∝ exp((Q(a) + τ log π(a)) / (τ + λ))`

per position. It returns `priors = π′` and
`value = v̂ = E_π′[Q]`, still in the universal side-to-move convention.
Keeping this improvement in the package is what makes policy-session sampling
reproduce KLENT acting while leaving the forward boundary free of model
opinion.

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

Three implementations behind one seam:

- **`MctsSession`** — PUCT, virtual loss on the path to each leaf in flight, and
  a cap on how many leaves one session may have outstanding at once.
- **`PolicySession`** — one root evaluation per move, then the package's
  move-selection policy. Policy-only self-play is the same loop with a
  degenerate search, not a second path. That matters because the first real
  training runs are policy-only, and a separate driver for them would leave the
  driver that eventually carries every run as the least exercised code in the
  system.
- **`GumbelSession`** — package-agnostic Gumbel-top-*m* root candidates (capped
  by the legal count and half the simulation budget) followed by sequential
  halving. It consumes only an `Evaluation`; for MantisNet those priors happen
  to be π′, but the search knows nothing about KLENT or that package.

Gumbel search extends deterministic **lines**, not a tree: revisiting the same
transition reveals no new information, so budget buys depth. Root candidates
rank by `g_a + log prior_a`; the root evaluation is outside the simulation
budget. There are up to `ceil(log2 m)` rounds, each giving every survivor an
equal, nonzero deepening share before retaining the stronger ceiling-half.
Extension follows the child evaluation's prior argmax for the next ply, and
each leaf value is signed to the root mover; terminal lines freeze at their
exact root-frame value. The final score is
`g + log prior + (50 + max_visits) · 1.0 · q̂`. Injected Gumbel vectors make
cross-language fixture parity testable without pretending NumPy and SplitMix64
share an RNG stream.

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
  G games, held as slots across T worker threads
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
  |   one Evaluator::evaluate call -----------> embedded PyTorch -> device
  |   scatters results back to slots     |
  +--------------------------------------+

  finished games --(bounded queue)--> writer, 1 thread --> records/<epoch>/
```

The worker pool is sized to the silicon rather than to the game count:
`--threads`, defaulting to the host's parallelism less three so that the
batcher, the writer, and the interpreter's own pools have
somewhere to run (§13). On this machine that is a number to set deliberately
rather than inherit, at about 13 of the 16 cores, once PyTorch's pools are
budgeted against the same 32 threads. One batcher thread owns the forward
crossing, because that crossing has to be serialised anyway and a second
batcher would only contend for the same lock and the same device. One writer
thread owns records, so the format has exactly one writer and no worker ever
touches a file. Every queue is bounded in both directions: a saturated
evaluator applies backpressure to the workers rather than growing a queue
until the process is killed.

`crates/hexo-bot/README.md` states this topology in the shapes it took — lanes
checked out of a ready queue, two bounded channels, and the token argument that
is why it cannot deadlock.

### 7.2 G and B are different numbers

**The batch size B and the game count G are chosen from different constraints
and are two flags, `--batch` and `--games`, not one.** B comes from what the
evaluator wants: today the MantisNet forward is CPU, while a device-backed
forward would choose the batch that saturates that device without paying excess
latency. G comes from RAM, and from how many leaves per game should be in
flight. Tying them together would mean either running the evaluator at a batch
size it does not want or sizing host memory from an unrelated throughput
number.

The second constraint on G is the one that is easy to miss, and it is the real
reason to run a thousand games rather than a hundred and twenty-eight. A batch
is filled from leaves and leaves come from games. With few games, each one has to
contribute many leaves at once to fill a batch — and every leaf in flight is a
virtual-loss penalty on a path whose true value is not known yet, so the search
is being steered by a placeholder. With many games, each contributes one or two
and the distortion nearly vanishes. Throughput is bounded by the evaluator
either way; what more games buy is search quality per unit of compute.

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
hexo-bot train --run-dir <dir> --run-id <id> --package <name> --epochs N --games G [--resume]

for epoch in 0..N:
    1. self-play   frozen weights, G concurrent games, produce records
    2. fit         consume this epoch's records, produce new weights
    3. checkpoint  write weights + manifest, rename into place, then load it —
                   loading is proving (§10.2)
    4. eval        every K epochs: current vs the anchor and the checkpoint it
                   was fit from
    5. metrics     one line, appended as it happens
```

**Why one long-lived process rather than one container per epoch.** The fixed
costs — CUDA context creation, interpreter startup, model load, kernel warm-up,
and whatever shape-keyed compilation the serve path accumulates — are paid once
per process, not once per epoch. Over hundreds of epochs that is the difference
between a minor overhead and a dominant one.

**Why the phases can share one process at all.** The two phases never run at the
same time, so a package that uses the single GPU for both never contends it:
no time-slicing, no MPS, no VRAM partitioning. Records are produced and
consumed within one epoch, so nothing has to outlive the process except
checkpoints and metrics. The current MantisNet container forward is CPU and its
fit is external; neither fact changes that scheduling invariant.

The phases are not separately invocable. There is one implementation of the
loop, not a loop plus a set of pieces that could drift from it.

That loop is complete for packages which implement `fit`, including the mock.
MantisNet intentionally stops at the package's explicit unsupported-fit error:
its live encoder, evaluator, self-play, matches, sessions, checkpoints, and
proving loads are container-side, while production fitting remains
`mantisnet.klent.run`. This is a package capability refusal, not a second
container phase or a silent no-op.

Records are written per epoch into that epoch's own directory, and the directory
is removed once the fit that consumed it has succeeded. Not before — a failed
fit leaves its input on disk to be inspected or re-run — and not later, because
on-policy records are worthless under the new weights and keeping them would
make disk growth a function of run length.

### 8.1 What "long-lived" requires

- **Bounded memory.** No cross-epoch growth. Within a phase the encoding arenas
  circulate through a pool and a lane's session reuses the position buffer and
  the tree it kept from its last game, so a sweep of a thousand games allocates
  per game rather than per leaf; across epochs the lanes are rebuilt and each
  epoch's finished games go to the writer thread and out to a file rather than
  accumulating in the process. Nothing measures RSS, so this is a property of
  those shapes rather than a checked one.
- **Resumable at epoch boundaries only.** `--resume` restarts from the last
  complete checkpoint. A crash mid-epoch discards that epoch's records, which
  costs nothing: they are on-policy and worthless under new weights.
- **Graceful stop.** `SIGTERM` — and Ctrl-C, which is the same request — sets a
  flag that is checked between epochs and inside every sweep. A stop during
  self-play abandons the partial epoch, records and all: those games were
  on-policy and are worthless without the fit that was going to consume them, so
  half an epoch of them is not a smaller epoch. A stop that arrives once the fit
  has begun lets the fit, the checkpoint, the load that proves it, and the
  metrics line all complete. `docker stop` therefore never loses an epoch that
  was going to finish.
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
| Injected | run directory, run id, package name and its config, checkpoint reference, epochs, G, B, game spec | flags; no run behaviour is read from the environment |
| Accumulated | checkpoints, metrics, records | the run directory, on a volume |

Run-behaviour values are never guessed or read from the environment. A missing
or unparseable one is a loud startup failure. `PYO3_PYTHON` and `HEXO_PYTHON`
are deployment plumbing for selecting the binary's one interpreter (§4), not
run or model configuration; they cannot change a game, session, or checkpoint
meaning.

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
- opaque **package metadata** whose shape and validation belong to the package;
  MantisNet records the τ and λ that turn raw heads into its opinion;
- the **probe hash** (§10.2).

A checkpoint whose manifest disagrees with the binary on any of them is a
startup failure, not a warning (§2).

The manifest does not describe the architecture. How many layers there are and
what shape they have is the package's business, carried inside its own weight
file; package metadata records only semantic configuration required to
interpret those weights. `hexo-model` preserves that JSON without understanding
it, and the package refuses a mismatch. The common manifest answers which
package wrote this, which versions apply, and whether it is compatible with
this binary.

MantisNet's weight file is the authoritative Torch `.pt` the Python trainer
writes. `hexo-bot init` seals it as `weights.pt` beside the manifest, probes the
copy through the live evaluator, and atomically places the directory. Every
later load uses the production version-refusing
`mantisnet.klent.run.load_model` path before comparing the computed probe.

A checkpoint is named by a **reference**: a filesystem path, a
`<run-id>/<epoch>` pair, `latest`, or a baked-in default. Shipping a
tournament-ready bot is then a thin image layer that copies one checkpoint in
and sets the default — not a separate architecture.

No flag resolves all four forms yet, because nothing needs one: `train` resolves
its own checkpoints out of the run layout, and a `match` seat takes a directory.
The `<run-id>/<epoch>` form is already in use — it is what a shard header
records, because a shard outlives the machine it was written on and an absolute
path is not a reference. A resolver lands with the first caller that has to name
a checkpoint it did not write.

### 10.1 Swapping

**Copy weights in place; never rebind the module.** Whatever graph capture or
compilation the serve path has accumulated records fixed pointers, so writing
into existing parameter storage keeps all of it valid and a swap costs one small
host-to-device copy. Replacing the module invalidates that state and re-pays the
entire warm-up. Embedding a live interpreter (§4) is what makes this the natural
operation rather than a trick.

The current CPU MantisNet boundary has no captured serve graph to preserve. A
load therefore builds and probes a candidate live module, then publishes it
only after every refusal point; a failed load leaves the previously proved
module intact. In-place copying remains the required shape when a compiled
long-lived serve slot lands, rather than a claim that the present CPU loader
already maintains such a slot.

A compiled serve implementation will use a **slot pool** for older checkpoints:
weights are small enough to keep many on disk, but each live module carries
expensive compiled state, so the number of live slots is capped (current,
opponent, anchor) and an opponent's weights are copied into a slot on demand.
That optimisation is not present in the current per-package-instance CPU
loader.

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

**Only self-play writes a shard.** An evaluation round and a `match` produce
evidence about two checkpoints rather than training data, and the metrics line
and the match report are what carry it; a shard nobody reads would be the
largest artefact in a run written for no reader. Every shard written so far
therefore says self-play in its header.

**One implementation, in Rust.** The mock's container-side `fit` reads through
`hexo-records`; MantisNet currently declines container fitting, so no Python
record reader exists. If that production loop migrates, Python will call this
Rust reader through the embedded boundary (§4) rather than grow a parser of its
own. A second parser is a second definition of the format, and the two would
disagree exactly where it is hardest to notice — a diagnostics blob's length
prefix, the end of a ragged ply list, an unset optional. Binary and versioned
rather than JSON because a shard is the largest artefact this system writes and
the fastest thing it re-reads, and because a strict reader that refuses an
unknown version is worth more than one that tolerates two shapes.

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
- **12 GB of VRAM** is the hard ceiling for a future device-backed forward. It
  would decide B, along with resident modules in the slot pool (§10.1), but
  never G. The current MantisNet forward is CPU-only and consumes none of that
  budget.

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
- **B4** — seed ownership, deferred against a seam that already exists (§12).
