# Container spec

**Status: design spec, not yet built.** Describes how a Hexo bot is packaged,
deployed, and run. Nothing here constrains `hexo-engine`, whose contract remains
`ENGINE_SPEC.md`. Deliberately says nothing about how a model works — the trunk,
the encoder, and the training objective are decided elsewhere.

This is a ground-up design. It is not a port of `Hexo-BotTrainer-hexgt`'s
workflow, and it does not inherit that repo's process supervision, shard
writers, or dashboard.

---

## 1. Target

One machine: Ryzen 9 7950X (16 cores / 32 threads), one RTX 4070 Ti (12 GB),
Windows 11 host, **Docker engine installed inside a WSL2 distro** — not Docker
Desktop. GPU access through the NVIDIA Container Toolkit installed in that
distro; the CUDA driver comes from the Windows driver projected into WSL.

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

**This settles `OPEN_DECISIONS.md` C6.**

---

## 3. One binary, three modes

`hexo-bot`, a new binary crate depending on `hexo-engine` and `hexo-runner`.
Modes are subcommands. **This settles C3.**

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
protocol's assumptions never reach `hexo-runner`. Transport and wire format
remain open (C1, C2).

---

## 4. `train` — the loop

Self-play and fitting are **one mode, not two**. One invocation runs a whole
training run and is expected to stay up for days.

```
hexo-bot train --run-id <id> --epochs N [--resume]

for epoch in 0..N:
    1. self-play   frozen weights, G concurrent games, produce records
    2. fit         consume this epoch's records, produce new weights
    3. checkpoint  write weights + manifest
    4. eval        every K epochs: current vs anchor and older checkpoints
```

**Why one long-lived process rather than one container per epoch.** The fixed
costs — CUDA context creation, model load, kernel warm-up, and whatever
shape-keyed compilation the serve path accumulates — are paid once per process,
not once per epoch. Over hundreds of epochs that is the difference between a
minor overhead and a dominant one.

**Why the phases can share one process at all.** The two phases never run at the
same time, so the single GPU is never contended: no time-slicing, no MPS, no
VRAM partitioning. Records are produced and consumed within one epoch, so
nothing has to outlive the process except checkpoints and metrics.

The phases are not separately invocable. There is one implementation of the
loop, not a loop plus a set of pieces that could drift from it.

### 4.1 What "long-lived" requires

- **Bounded memory.** No cross-epoch growth. Game slots, staging buffers, and
  the record buffer are allocated once and reused; the record buffer is cleared
  at the end of each epoch. Steady-state RSS after epoch 2 is a checked
  property, not a hope.
- **Resumable at epoch boundaries only.** `--resume` restarts from the last
  complete checkpoint. A crash mid-epoch discards that epoch's records, which
  costs nothing: they are on-policy and worthless under new weights.
- **Graceful stop.** `SIGTERM` finishes the phase's in-flight work, writes a
  checkpoint, and exits 0. `docker stop` must never lose an epoch.
- **Distinguishable exits.** Ran to completion, stopped by signal, and failed
  are three different exit codes. A run that ended is not a run that broke.
- **Observable without stopping.** One metrics line per epoch, appended to the
  volume as it happens. No dashboard, no metrics service, no scraping endpoint.

### 4.2 Concurrency

`--games G` sets the number of concurrent self-play games. G is also the
inference batch size, so it is chosen from what the GPU can hold rather than
picked as a load target. Games are `hexo_runner::Game` values swept for
`Step::NeedDecision` and answered in batches — no thread per game.

---

## 5. State

Three kinds, three homes. Getting this wrong is the usual way container designs
fail.

| Kind | Examples | Where |
| --- | --- | --- |
| Baked | binary, CUDA/Python stack, version constants | the image |
| Injected | checkpoint reference, run id, epochs, G, seeds, game spec | flags and env |
| Accumulated | checkpoints, metrics, records | a named volume |

Injected values are never guessed. A missing or unparseable one is a loud
startup failure.

Layout under the volume:

```
/var/lib/hexo/runs/<run-id>/
  manifest.json              run config, versions, seed
  metrics.jsonl              one line per epoch
  checkpoints/<epoch>/
    weights.<ext>
    manifest.json
  records/<epoch>/           transient; removed after the fit phase
```

A crashed run leaves a partial epoch directory, never a corrupted checkpoint:
checkpoints are written to a temporary name and renamed into place.

---

## 6. Checkpoints

A checkpoint is weights plus a manifest. The manifest pins the architecture
config, the encoder version, `RULES_VERSION`, `ACTION_ORDER_VERSION`,
`PROTOCOL_VERSION`, and a probe hash (§6.2).

`--checkpoint` takes a **reference**, resolved from a filesystem path, a
`<run-id>/<epoch>` pair, `latest`, or a baked-in default. Shipping a
tournament-ready bot is then a thin image layer that copies one checkpoint in
and sets the default — not a separate architecture.

### 6.1 Swapping

**Copy weights in place; never rebind the module.** Whatever graph capture or
compilation the serve path uses records fixed pointers, so writing into existing
parameter storage keeps it valid and a swap costs one small host-to-device copy.
Replacing the module invalidates all of it and re-pays the entire warm-up.

Older checkpoints for evaluation use a **slot pool**: weights are small enough
to keep many on disk, but each live module carries expensive compiled state, so
the number of live slots is capped (current, opponent, anchor) and an opponent's
weights are copied into a slot on demand.

### 6.2 The probe hash

A fixed set of probe positions is forwarded after every load and every swap, and
the hash of the outputs is compared against the manifest. One forward.

It exists because every failure it catches is silent: the wrong checkpoint
loaded, a swap that constant-folding turned into a no-op, a mismatched feature
version, a scrambled action ordering, or a runtime that drifted between build
and run. None of these crash. All of them train or play against the wrong
weights indefinitely.

---

## 7. Runtime knobs

Four, and three of them bite in a container specifically.

- **Thread budget.** A container sees the host's 32 threads regardless of its
  quota, so the Rust thread pool, PyTorch's intra-op pool, and OpenMP will each
  claim all of them and thrash. Each is set explicitly and budgeted against the
  others. `--cpus` is a quota, not a pinning; use `--cpuset-cpus` if pinning is
  wanted.
- **`--shm-size`.** The 64 MB default breaks anything using shared-memory IPC or
  dataloader workers.
- **WSL2 memory ceiling.** The distro, the Docker engine, and every container
  share one budget set in `.wslconfig`. Set it deliberately rather than
  discovering it as an OOM kill at an epoch boundary.
- **12 GB of VRAM** is the hard ceiling, and it is what decides `--games`.

---

## 8. Out of scope

No orchestrator, scheduler, dashboard, or experiment tracker. No model registry
or promotion logic. No multi-node, no multi-GPU. No inference service separate
from the process that uses it. No adjudication in seat modes.

## 9. Still open

- **C1, C2** — transport, wire format, and handshake fields for `serve` and
  `play`. A line-oriented stdio protocol remains the default assumption.
- **C4** — the on-disk record format.
- **B4** — seed ownership, which `train` needs before self-play samples rather
  than maximises.
- The inference runtime, which follows the trunk and is deliberately not decided
  here. The container is written against an evaluator seam so that a mock
  implementation makes the whole loop testable with no model and no GPU.
