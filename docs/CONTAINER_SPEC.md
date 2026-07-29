# Container spec

This document defines the container, package, evaluator, session, checkpoint,
record, and process contracts implemented by `hexo-bot` and its supporting
crates. `hexo-engine` remains governed by `ENGINE_SPEC.md`. Model architecture,
encoding, and training objectives belong to model packages (§5).

`crates/models/mantisnet` is the network-backed package. Its Rust encoder,
KLENT-improved evaluator, sessions, diagnostics, Torch checkpoint sealing, and
verified loads use the seams defined here. `hexo-bot` owns the PyO3/Torch
boundary; logic crates remain Python-free. The image and Compose environment
live under `docker/`. The production MantisNet trainer remains
`python/mantisnet`, and the MantisNet package reports container-side `fit` as
unsupported.

---

## 1. Target

One machine: Ryzen 9 7950X (16 cores / 32 threads), one RTX 4070 Ti (12 GB),
32 GB of system RAM, Windows 11 host, **Docker engine installed inside a WSL2
distro** — not Docker Desktop. GPU access through the NVIDIA Container Toolkit
installed in that distro; the CUDA driver comes from the Windows driver
projected into WSL.

`.wslconfig` MUST give the distro roughly 20 GB and leave roughly 12 GB to
Windows. The distro allocation is shared by the Docker engine and all
containers; it bounds the concurrent game count (§7.2).

Multi-GPU and multi-node execution are out of scope.

### 1.1 The filesystem rule

Windows paths reach WSL over 9p; high-volume I/O MUST use Linux-native Docker
volumes.

- Long-lived build cache, records, checkpoints, and metrics belong in
  **named Docker volumes**, never a `/mnt/d` bind mount.
- The development image bind-mounts source at `/workspace`.
- A source-free image MAY copy the built binary, but MUST use the same model
  runtime.

---

## 2. One image

A single image MUST serve every job. All modes MUST use the same implementation
of each model's forward pass.

`docker/Dockerfile` produces the `mantisnet-train` image: CUDA,
the locked `/opt/venv` Python/PyTorch environment, and the Rust/maturin tools
needed to compile mounted source. `docker/compose.yaml` mounts the checkout at
`/workspace`; `REPO_ROOT` lets a worktree supply that checkout, and
`CARGO_TARGET_DIR=/workspace/target-wsl` keeps Linux artefacts separate from
the Windows target directory.

The binary and checkpoint manifests record the versions that govern behaviour:
`hexo_engine::RULES_VERSION`, `ACTION_ORDER_VERSION`, and
`hexo_runner::PROTOCOL_VERSION`, plus package and encoder versions. A
checkpoint whose manifest disagrees with any of them is a startup failure, not
a warning. The image carries the locked Python runtime that executes the same
MantisNet module as training; there is no exported ONNX or TorchScript graph.

The mock package MUST exercise the complete logic path without importing
Python.

---

## 3. One binary, three modes

`hexo-bot`, a binary crate depending on the engine, the runner, the search, the
records, and the model API. Modes are subcommands.

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

`train` is implemented. `serve` and `play` are not implemented because their
wire protocol is unspecified; the binary MUST NOT expose either subcommand
until its protocol is specified and implemented.

`match` plays two seats — each named by package, checkpoint, package config, and
session variant — over the same driver used by `train`, and prints a JSON
report. It adjudicates locally exactly as `train` does: it holds its own `Game`
values and both seats are in-process sessions, so "exactly one authority per
game" remains satisfied and no wire protocol is implied. It writes no records
(§11). `crates/hexo-bot/README.md` defines its flags and seat grammar.

`init` is also a utility subcommand, not a mode. It asks one package to
write an epoch-0 checkpoint under a sibling `.incomplete` directory and
atomically renames it into place; an existing destination is refused. For
MantisNet, package config
`tau=<F>,lambda=<F>,source=<training-checkpoint.pt>` copies the authoritative
Python `.pt`, loads that copy through the live evaluator, and writes the
manifest and probe hash that make subsequent loads provable. It constructs no game
and claims no new authority.

---

## 4. The Rust/Python boundary

Training uses PyTorch in Python. In self-play, `hexo-bot` embeds Python and
executes the training `nn.Module` through PyO3. ONNX, TorchScript, or another
forward runtime MUST NOT be introduced as an alternate serving path.

The evaluator seam (§6) MUST cross the Python boundary once per batch, not once
per position. A long-lived module swap (§10.1) MUST copy weights into existing
parameter storage rather than export or compile another module.

**PyO3 is a leaf dependency.** `hexo-bot` is a leaf binary crate that nothing
depends on, and no logic crate mentions Python, even optionally. This is the
root workspace rule. Logic-crate tests remain free of a Python toolchain, and
`hexo-engine` remains compilable to `wasm32`.

`crates/models/mantisnet` defines Python-free
`ForwardLoader` and `Forward` traits over its Rust `RawBatch` and flat ragged
`RawOutputs`. The package owns all evaluator logic; the executable injects the
only implementation that loads `mantisnet.klent.run.load_model`, constructs CPU
Torch tensors, performs one live-module call for the whole batch, and returns
the policy logits and Q values unchanged. The package then applies KLENT
equation 3 and produces the public `Evaluation`. No `Position`, evaluator
semantics, Python object, tensor, GIL token, or device type crosses that trait.

PyO3 is an unconditional dependency of the `hexo-bot` leaf rather than a Cargo
feature. Thus the ordinary binary and the container build have the same
registry and boundary; MantisNet availability is not feature-gated.
`PYO3_PYTHON` chooses the interpreter at build time.
`HEXO_PYTHON` may name the same interpreter at runtime so the binary discovers
its `sys.path` and refuses a major/minor mismatch; otherwise PyO3's configured
interpreter is used. The mock and every logic crate run without loading
an interpreter.

---

## 5. Model packages

A model is a crate. Packages live under `crates/models/<name>/`; the registry
carries `mock` and `mantisnet`. Each one implements the
`ModelPackage` trait from `crates/hexo-model`, and the container's whole
knowledge of a model is that trait plus the registry in `hexo-bot`.

A package owns, and the container never has an opinion about:

| What it owns | Contract boundary |
| --- | --- |
| Its encoder, and the encoder version | The engine exposes state; only the package assigns feature meaning |
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

A registered package MUST build against the current record format and package
seam.

`fit` has no implementation default, but an implementation may return
`PackageError::Unsupported` when the production training loop still lives
elsewhere. Such an error MUST identify the production loop and the authority
required to move it. MantisNet identifies `mantisnet.klent.run` and does not
provide a partial container trainer.

**The two session constructors are required and have no defaults**: one for
self-play and one for evaluation. They MUST preserve their distinct
move-selection contracts.

A third constructor, `variant_session`, is optional and **may** default only to
refusal. A variant names a package-owned search shape for `match` (§3). A
package without variants MUST return "no such variant"; it MUST NOT substitute
another session shape.

**The canonical action ordering is the indexing contract.** The engine owns it
in both directions — `legal_rank` and `nth_legal` — and every package's policy
head is held to it: index *i* is the *i*th legal action of the position being
evaluated, and nothing else. `ACTION_ORDER_VERSION` is pinned in every manifest
and shard header; changing the ordering invalidates those artefacts.

The mock package is a deterministic, weightless evaluator that exercises the
encoder, evaluator, both session modes, diagnostics, shard writing, checkpoint
write and load, probe hash, and `fit` without a network, GPU, or Python. It MUST
remain available for container-loop tests.

MantisNet's worker encoder and the
`python/hexo-py` extension call one Rust representation implementation, and
`MODEL_REPR_VERSION` is owned by that package alone. Its self-play session
samples the improved policy π′; its evaluation session uses the shared
`GumbelSession` at 32 simulations and 16 root candidates. Its exact variant
grammar is `policy`, `mcts:visits=..,inflight=..,cpuct=..`, and
`gumbel:sims=..,m=..`. Its nine diagnostic bytes carry a version, v̂, and π′
entropy for the acted self-play position.

---

## 6. The evaluator seam

The evaluator seam consists of three types in `crates/hexo-search`; packages
implement `Encoder` and `Evaluator`.

**`Encoder` — package-owned, runs worker-side.** It turns a `&Position` into
bytes written into a caller-provided, reusable batch buffer. A `Position` MUST
NOT cross the worker boundary; queues carry buffer handles rather than
positions, tensors, or Python objects. Worker-side encoding distributes CPU
work across the pool.

**`Evaluator` — package-owned, runs batcher-side.** One call answers one whole
batch. That call contains the single Python/device crossing (§4), and it is the
only place in the system where either may be touched. The MantisNet
boundary loads and forwards on CPU; changing device placement remains inside
the injected forward and does not move the seam.

**`Evaluation` — the answer for one position.** Priors over that position's
legal actions **in canonical order**, and a value in `[-1, 1]` **from the
perspective of the side to move**.

Both conventions are normative for every package rather than conventions a
package may restate. Priors are ragged — one per legal action, in `nth_legal`
order, with no fixed crop. The value sign is always relative to the side to
move.

MantisNet resolves the leaf rule by splitting mechanics from semantics. Its
encoder bytes decode and collate into a Python-free `RawBatch`; one injected
`Forward::forward(&RawBatch)` returns flat policy-logit and Q-value vectors,
ragged by the batch's canonical legal offsets. The evaluator refuses malformed
lengths and applies

`π′(a) ∝ exp((Q(a) + τ log π(a)) / (τ + λ))`

per position. It returns `priors = π′` and
`value = v̂ = E_π′[Q]`, still in the universal side-to-move convention.
The package owns this improvement; the raw forward boundary MUST NOT apply it.

---

## 7. Sessions and nonblocking execution

A seat's search is a nonblocking state machine: `DecisionSession`, in
`hexo-search`. Pump it and it emits the leaf evaluations it wants; resume it
with the results and it continues; eventually it yields the whole `Decision`.
The authorship rules are `hexo-player`'s and are unchanged — the session
authors the zobrist attestation of the position it actually searched and the
diagnostics bytes. The driver submits the decision verbatim and MUST NOT replace
either field.

Three implementations behind one seam:

- **`MctsSession`** — PUCT, virtual loss on the path to each leaf in flight, and
  a cap on how many leaves one session may have outstanding at once.
- **`PolicySession`** — one root evaluation per move, then the package's
  move-selection policy. Policy-only self-play MUST use the same driver loop as
  searched play.
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
cross-language fixtures independent of NumPy and SplitMix64 RNG streams.

**A waiting session MUST NOT block a thread.** A session with leaves outstanding
is stored state. `Game` is a state machine so a worker can sweep multiple games,
answer every
`Step::NeedDecision` by pumping that slot's session, and hand whole batches of
leaves to one evaluator.

`hexo-player`'s `Player` remains the seam for a seat that answers one decision
at a time — a human, scripted bot, or remote container behind an adapter. It is
not the self-play driver: `choose` returns a complete `Decision`, while a
`DecisionSession` may request evaluations before producing one. Both seams
require the seat to author the `Decision`.

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
capacity (§13). On the target machine, the configured budget is approximately
13 of 16 cores after reserving capacity for PyTorch pools, the batcher, and the
writer. Exactly one batcher thread owns the forward crossing. Exactly one writer
thread owns records, and workers MUST NOT touch record files. Queues are bounded
in both directions and MUST apply backpressure when saturated.

`crates/hexo-bot/README.md` defines the ready-lane pool, bounded channels, and
deadlock-prevention token invariant.

### 7.2 G and B are different numbers

**The batch size B and game count G are independent flags, `--batch` and
`--games`.** B is constrained by evaluator throughput and latency. G is
constrained by host RAM and the desired number of leaves in flight per game.

Search workloads SHOULD use enough games that a full batch does not require
many simultaneous leaves from each game, because every outstanding leaf adds
virtual loss before its value is known. Policy-only workloads SHOULD set G to a
multiple of B because a game contributes at most one root request per move and
does not request while applying or recording a move.

---

## 8. `train` — the loop

Self-play and fitting are **one mode, not two**. One invocation owns the whole
training run.

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

The process is long-lived across epochs. Interpreter startup, model loading,
device-context creation, and kernel warm-up occur once per invocation.
Self-play and fitting MUST NOT overlap; a package may use the single GPU for
both without time-slicing, MPS, or VRAM partitioning. Only checkpoints and
metrics persist across successful epochs.

The phases are not separately invocable. There is one implementation of the
loop, not a loop plus a set of pieces that could drift from it.

Packages that implement `fit`, including the mock, execute the complete loop.
MantisNet stops at the package's explicit unsupported-fit error:
its live encoder, evaluator, self-play, matches, sessions, checkpoints, and
proving loads are container-side, while production fitting remains
`mantisnet.klent.run`. Unsupported `fit` MUST fail explicitly; it MUST NOT
become a no-op or alternate container phase.

Records are written per epoch into that epoch's own directory, and the directory
is removed only after the consuming fit succeeds. A failed fit leaves its input
on disk. Successful fits MUST remove consumed on-policy records so record
storage does not grow across epochs.

### 8.1 What "long-lived" requires

- **Bounded memory.** No cross-epoch growth. Within a phase the encoding arenas
  circulate through a pool and a lane's session reuses the position buffer and
  the tree it kept from its last game, so a sweep of a thousand games allocates
  per game rather than per leaf; across epochs the lanes are rebuilt and each
  epoch's finished games go to the writer thread and out to a file rather than
  accumulating in the process. This is a structural invariant; the process does
  not enforce it through RSS measurement.
- **Resumable at epoch boundaries only.** `--resume` restarts from the last
  complete checkpoint. A crash mid-epoch discards that epoch's records.
- **Graceful stop.** `SIGTERM` — and Ctrl-C, which is the same request — sets a
  flag that is checked between epochs and inside every sweep. A stop during
  self-play abandons the partial epoch and its records. A stop that arrives once
  the fit has begun lets the fit, the checkpoint, the load that proves it, and
  the metrics line all complete.
- **Distinguishable exits.** **0** ran to completion, **2** stopped by signal
  after finishing cleanly, and **1** failed.
- **Observable without stopping.** One metrics line per epoch, appended to the
  run directory as it happens.

---

## 9. State

Runtime state has three ownership classes:

| Kind | Examples | Where |
| --- | --- | --- |
| Baked | binary, CUDA/Python stack, version constants, the package registry | the image |
| Injected | run directory, run id, package name and its config, checkpoint reference, epochs, G, B, game spec | flags; no run behaviour is read from the environment |
| Accumulated | checkpoints, metrics, records | the run directory, on a volume |

Run-behaviour values are never guessed or read from the environment. A missing
or unparseable one is a startup error. `PYO3_PYTHON` and `HEXO_PYTHON`
are deployment plumbing for selecting the binary's one interpreter (§4), not
run or model configuration; they cannot change a game, session, or checkpoint
meaning.

`--run-dir` names the root of the accumulated state, and the container maps a
named volume onto it. The flag is required and has no compiled-in or implicit
default. The same binary accepts a host path during native development.

```
<run-dir>/runs/<run-id>/
  manifest.json              run config and versions
  metrics.jsonl              one line per epoch
  checkpoints/<epoch>/
    weights.<ext>            the package's own format
    manifest.json
  records/<epoch>/           transient; removed after a successful fit
```

The run manifest has no seed field (§12).

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
subsequent load uses the production version-refusing
`mantisnet.klent.run.load_model` path before comparing the computed probe.

A checkpoint is named by a **reference**: a filesystem path, a
`<run-id>/<epoch>` pair, `latest`, or a baked-in default. An image MAY copy one
checkpoint and set the baked-in default; it MUST use the same architecture and
runtime.

The CLI does not expose one resolver for all four forms. `train`
resolves checkpoints from the run layout, `match` accepts a checkpoint
directory, and shard headers use `<run-id>/<epoch>` rather than absolute paths.

### 10.1 Swapping

**Copy weights in place; never rebind the module.** Whatever graph capture or
compilation the serve path has accumulated records fixed pointers, so writing
into existing parameter storage MUST preserve it. A compiled serve
implementation MUST NOT replace the resident module during a swap.

The CPU MantisNet boundary has no captured serve graph to preserve. A
load therefore builds and probes a candidate live module, then publishes it
only after every refusal point; a failed load leaves the previously proved
module intact.

A compiled serve implementation MUST use a bounded **slot pool** for live
checkpoints (current, opponent, anchor) and copy an opponent's weights into a
slot on demand. The CPU loader uses one module per package instance and
does not implement this pool.

### 10.2 The probe hash

A fixed set of probe positions is forwarded after every load and every swap, and
the hash of the outputs is compared against the manifest. One forward.

`hexo-model` computes it, over the **exact bytes the `Evaluator` returns** —
not over the weights, and not over a re-derived summary. The probe set is a
fixed batch forwarded whole, so nothing about batch shape can change which
kernel runs, and the same binary with the same weights therefore produces the
same hash every time.

A mismatch MUST refuse the load or swap. The probe covers the loaded checkpoint,
weight replacement, encoder version, action ordering, and runtime output.

---

## 11. Records

`hexo-records` owns the on-disk game record: a versioned binary shard, written
once and read strictly.

A shard opens with a header pinning the record format version, `RULES_VERSION`,
`ACTION_ORDER_VERSION`, and `PROTOCOL_VERSION`, plus the run id, the epoch, the
mode the games were played in, the package name, and the checkpoint reference
the weights came from. Then, per game: the `GameSpec` it was played under, the
`MatchResult` with its full adjudication payload, and the plies — each a seat,
an action, the resulting zobrist, and optional diagnostics bytes.

The result is carried whole rather than flattened to a winner. It preserves the
`ActionId` and `MoveError` on an illegal-move forfeit and the seat and `Failure`
on a no-contest, allowing consumers to distinguish adjudication causes.

The mode lives in the header because every game in a shard belongs to one phase,
epoch, and package. `GameSpec` is recorded per game because adjudication depends
on it and the run may vary game budgets within a phase.

**Only self-play writes a shard.** An evaluation round and a `match` produce
metrics and match reports rather than training shards. Every written shard MUST
identify self-play in its header.

**One implementation, in Rust.** The mock's container-side `fit` reads through
`hexo-records`; MantisNet declines container fitting, so no Python
record reader exists. Any Python consumer MUST call the Rust reader through the
embedded boundary (§4), not implement a second parser. Readers MUST refuse
unknown format versions and malformed diagnostics lengths, ragged ply lists,
or optional fields.

The detailed statement of the format belongs in `crates/hexo-records/README.md`;
this section says only what the container needs from it.

---

## 12. Seeds

A session accepts a seed when constructed and exposes `reseed`. The driver
seeds sessions from entropy. It does not mint or record per-game seeds, and
neither the run manifest nor shard header contains a seed field. Records
therefore make no reproducibility guarantee.

Adding reproducible self-play requires stable per-game and per-seat seed
derivation, session reseeding independent of scheduling, persisted seeds, a
record-format version bump, and regenerated data.

---

## 13. Runtime knobs

The container exposes four runtime resource controls.

- **Thread budget.** A container sees the host's 32 threads regardless of its
  quota. The Rust worker pool, PyTorch's intra-op pool, and OpenMP thread counts
  MUST each be set explicitly and budgeted against the ~13 workers plus batcher
  and writer of §7.1. `--cpus` is
  a quota, not a pinning; use `--cpuset-cpus` if pinning is wanted.
- **`--shm-size`.** The 64 MB default breaks anything using shared-memory IPC or
  dataloader workers.
- **WSL2 memory ceiling.** The distro, the Docker engine, and every container
  share the one budget set in `.wslconfig` — the ~20 GB of §1. It is what bounds
  G.
- **12 GB of VRAM** is the hard ceiling for a device-backed forward. It decides
  B, along with resident modules in the slot pool (§10.1), but never G. The
  MantisNet forward is CPU-only and consumes none of that
  budget.

---

## 14. Out of scope

No orchestrator, scheduler, dashboard, or experiment tracker. No model registry
or promotion logic. No multi-node, no multi-GPU. No inference service separate
from the process that uses it. No adjudication in seat modes. No hyperparameter
sweep and no multi-run scheduler: a run is one process and one directory.

## 15. Unspecified interfaces

- The transport, wire format, and handshake fields for `serve` and `play` are
  unspecified. Any stdio protocol is limited to external interoperability and
  MUST NOT become the self-play path.
- Reproducible self-play seed ownership is unspecified beyond the session seam
  in §12.
