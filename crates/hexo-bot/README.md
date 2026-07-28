# hexo-bot

The one binary: the training loop that drives batched self-play, fitting,
checkpoints, and evaluation from a single long-lived process.

**Status: implemented through the first real package.** `init`, `train`, and
`match` ship; the registry carries `mock` and `mantisnet`. MantisNet's
encoder/evaluator/checkpoint/session logic remains Python-free in its package,
and this leaf binary owns the one live PyO3/Torch crossing. The
`mantisnet-train` image and compose environment are under `docker/`.

## Shape

A thin `main.rs` over a library, so the loop is driven in-process by the test
suite rather than by spawning a child and parsing its output. `main` parses a
command line, installs the stop handler, injects the executable-only MantisNet
forward loader into the registry, calls one of three entry points, and maps
what comes back onto an exit code; everything else is here.

```
crates/hexo-bot/
  Cargo.toml
  README.md
  src/
    lib.rs        # crate root, Outcome, the three entry points
    main.rs       # argv, the ctrlc handler, exit codes, the error chain
    cli.rs        # the flags, the seat-spec grammar, and every refusal
    registry.rs   # names become packages; the binary injects the live loader
    mantisnet_python.rs # executable-only PyO3/Torch ForwardLoader + Forward
    init.rs       # seal and atomically place one package checkpoint
    run.rs        # the run directory: layout, run manifest, resume, placement
    driver.rs     # the batched sweep: lanes, workers, batcher, writer
    train.rs      # the epoch loop
    matches.rs    # the head-to-head and its report
    metrics.rs    # the tally, and one JSON line per epoch
    error.rs      # BotError
  tests/
    common/mod.rs # parsing a command line, and reading a run back off disk
    init.rs       # atomic checkpoint creation and every refusal
    train.rs      # two whole runs, and both stop paths
    resume.rs     # what a resume continues, refuses, and must not redo
    matches.rs    # two searches over one checkpoint, and the seat grammar
    cli.rs        # every flag refusal names its flag
```

## Module map

| Module | Role |
| --- | --- |
| `cli` | `parse`, `InitConfig`, `TrainConfig`, `MatchConfig`, `SeatSpec`. Hand-rolled: three subcommands and their small flag grammars do not need a parser generator, and the refusals are the part worth writing by hand. |
| `registry` | `PackageRegistry`, `PACKAGES`, and `construct`. One `match` from a name to a `Box<dyn ModelPackage>`, with the binary's MantisNet forward loader injected at construction. |
| `mantisnet_python` | The executable-only PyO3 boundary: interpreter discovery, the production version-refusing loader, CPU `Batch` tensors, one live-module call, and raw head extraction. |
| `init` | Atomic creation of one package checkpoint outside a training run. |
| `run` | `<run-dir>/runs/<run-id>` and everything under it: the paths, the run manifest, the resume check, temporary-then-rename checkpoint placement, and clearing what a crash left. |
| `driver` | The sweep. One implementation, shared verbatim by self-play, evaluation, and `match`. |
| `train` | The epoch loop and the stop contract. |
| `matches` | Two seats, fixed weights, one JSON report. |
| `metrics` | `Tally` — wins by slot *and* by colour — and the epoch line. |
| `error` | `BotError`. Every variant carries what locates the problem. |

## Registry and the Python boundary

`PackageRegistry` carries the runtime dependency that only one package needs.
The executable injects `PythonForwardLoader`; Python-free in-process tests use
`without_mantisnet_runtime()`, which still constructs and parses MantisNet but
fails loudly if a test actually tries to load it. There is no global Python
handle and no model-private alternate registry.

The MantisNet package owns its Rust encoder, strict batch decoder, KLENT
equation-3 improvement, sessions, diagnostics, and checkpoint rules. Its
injected `ForwardLoader`/`Forward` traits mention only paths, Rust batch data,
and flat `f32` heads. `mantisnet_python` supplies the executable-only
implementation:

1. load `weights.pt` through
   `mantisnet.klent.run.load_model(path, "cpu")`, including that production
   loader's version refusal;
2. build the Python `Batch` as CPU tensors;
3. call the live `MantisNet` once under inference mode for the whole encoded
   batch; and
4. return ragged policy logits and Q values to the package unchanged.

That is one interpreter attachment and one model call per batch. No logic crate
mentions PyO3, Torch, a GIL token, or a device type. PyO3 is an unconditional
dependency of the leaf executable — there is no Cargo feature that can produce a
different registry—while the `wasm32` engine gate remains Python-free.

`PYO3_PYTHON` selects the interpreter used while building `hexo-bot`.
`HEXO_PYTHON` selects the runtime environment whose `sys.path` is installed
before MantisNet is imported; the binary refuses it if its Python major/minor
does not match the embedded interpreter. Set both to the same executable.
These are the exact CPU-only setups.

Windows PowerShell, from the repository root:

```powershell
$python = (Resolve-Path .\python\mantisnet\.venv\Scripts\python.exe).Path
$pythonBase = & $python -c "import sys; print(sys.base_prefix)"
$pythonScripts = Split-Path -Parent $python
$env:Path = "$pythonBase;$pythonBase\DLLs;$pythonScripts;$env:Path"
$env:PYO3_PYTHON = $python
$env:HEXO_PYTHON = $python
$env:CUDA_VISIBLE_DEVICES = "-1"
cargo build -p hexo-bot
```

The base-Python `PATH` entries are a runtime requirement on Windows:
`hexo-bot.exe` must find `python313.dll` before Rust `main` can run. The venv's
own `python.exe` knows its base installation, but the Windows loader resolving
an embedded DLL does not.

The `/opt/venv` container, from `/workspace`:

```sh
export PYO3_PYTHON=/opt/venv/bin/python
export HEXO_PYTHON=/opt/venv/bin/python
python_lib="$("$PYO3_PYTHON" -c \
  'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')"
export LD_LIBRARY_PATH="$python_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=-1
cargo build -p hexo-bot
```

If `HEXO_PYTHON` is absent, PyO3's configured interpreter discovery is used.
The explicit form is preferred for a venv because it makes both the linked
minor version and import path visible at invocation.

## `init`

```
hexo-bot init --package <name> --checkpoint <dir> [--package-config <string>]
```

`init` asks the package to write epoch zero under
`<checkpoint>.incomplete`, then renames that directory into place. It refuses
an existing destination, and clears only its own incomplete sibling after an
interrupted attempt. Success prints the checkpoint, package, and probe hash as
JSON.

For MantisNet, the exact package config is
`tau=F,lambda=F[,source=PATH]`; `source` is required by `init` and names the
authoritative Python training `.pt`. Initialisation copies that file to
`weights.pt`, loads the copy through the live evaluator, and writes a manifest
carrying `tau`, `lambda`, every common version, and the computed probe. For
example, after setting the environment above:

```powershell
cargo run -p hexo-bot -- init `
  --package mantisnet `
  --checkpoint .\scratch\checkpoints\mantisnet `
  --package-config "tau=0.1,lambda=0.03,source=D:\path\to\checkpoint.pt"
```

Under `/opt/venv`, after the container environment block above, the same
operation is:

```sh
cargo run -p hexo-bot -- init \
  --package mantisnet \
  --checkpoint /workspace/scratch/checkpoints/mantisnet \
  --package-config \
  "tau=0.1,lambda=0.03,source=/workspace/path/to/checkpoint.pt"
```

Loading is proving: the package validates manifest metadata and versions,
builds a candidate live module, recomputes the probe through its real
evaluator, and publishes that module only after the hash agrees.

## The driver

`docs/CONTAINER_SPEC.md` §7.1's topology, as built.

```
   G lanes: one Game + two DecisionSessions each, held as data, not as threads
        |                                          ^
        | Job { lane, slot, encoded bytes }        | the same lane, carrying
        v  bounded sync_channel, depth 2T          | its evaluations
   batcher, 1 thread ---- Evaluator::evaluate ----> the device crossing
        |
        | finished GameRecords, bounded sync_channel, depth 2T
        v
   writer, 1 thread ---> records/<epoch>/shard-0000.hxr
```

A **lane** is one game and the two sessions filling its seats. `--threads T`
workers check lanes out of a ready queue, pump the mover's session, encode each
emitted leaf worker-side with that seat's `Encoder`, and send the lane — with its
bytes — to the batcher. The batcher owns every evaluator (one, for self-play;
two, for an evaluation round or a match), merges arriving jobs into a per-slot
batch, crosses once per filled batch, and hands each lane back with its answers
attached.

**There is no thread per game.** `--games` is a lane count, not a thread count:
a session with leaves outstanding is a struct holding a few vectors, so the
number of concurrent games is bounded by memory rather than by scheduler
pressure. That is the whole architecture, and it is what `hexo-runner` inverted
its loop for and what `hexo-search` made its sessions nonblocking for.

### Queues, and why this cannot deadlock

**A lane is a token, and there are exactly as many tokens as lanes.** A lane is
in exactly one place at any moment: the ready queue, one worker's hand, the job
channel, or the batcher's slate. The ready queue's occupancy therefore can never
exceed the lane count — bounded structurally, rather than by a capacity number
nobody could pick — so the batcher can *always* hand a lane back without
blocking.

That is the property everything rests on:

- Only workers ever block on a queue: on the job channel when the batcher is
  behind, and on the record channel when the writer is. A saturated device
  therefore applies backpressure to the workers, which is the point, instead of
  growing a queue until the process is killed.
- Neither consumer ever blocks on a producer. The batcher's only output is the
  ready queue; the writer's is a file. So there is no cycle of waits to close.
- A worker with no lane available waits on a condition variable, never a spin,
  and wakes when a lane is re-armed, when one retires, or when the sweep halts.

**Draining is the same argument from the other end.** Once the phase's quota is
met, a lane whose game ends retires instead of restarting. Workers keep waiting
while any lane is outstanding, so the evaluations in flight are free to complete
and their lanes retire cleanly; a worker leaves only when no lane exists
anywhere. The batcher learns the phase is over from the job channel
disconnecting, answers whatever is on its slate, and returns.

**A panicking thread ends the sweep rather than stranding it.** Panicking is how
the layers below report a broken evaluation — a prior count that does not match
the position's legal set, a value outside `[-1, 1]`, an answer to a question no
session asked — and all of those are package bugs worth stopping for. What must
not happen is that stopping becomes hanging: a worker that dies holding a lane
would leave the queue counting a lane that no longer exists, and everyone else
would wait for it. So a worker or batcher that unwinds halts the sweep and sets
the abort flag on its way out, the run fails with `ThreadPanicked`, and the shard
of the phase that died is never finalized. Halting is the one operation that
takes the queue tolerantly of poisoning, because it is the one that has to work
after a thread has died holding it.

**Stopping is prompt because the batcher is the sweep's clock.** It is the one
thread never blocked for longer than the flush window, so it checks the stop flag
on every `recv_timeout` and turns it into a halt every other thread sees. Workers
leave at their next checkout; lanes in flight are dropped, because a phase that
was abandoned has nothing to finish.

### Three details worth knowing

- **`--batch` is a threshold, not a ceiling.** A slot crosses when its batch
  reaches `B`, and the job that took it there is included whole, so a batch can
  exceed `B` by one lane's leaves. A lane's questions are answered in one call
  because they were asked in one pump, and splitting them across two crossings
  would buy nothing.
- **Encoders are per worker, evaluators are per slot.** `Encoder::encode` takes
  `&self` and runs worker-side, so the driver holds a `threads × slots` table of
  them and every thread has its own. `Evaluator::evaluate` takes `&mut self` and
  is the one crossing, so the batcher owns all of them and there is exactly one.
- **The batcher merges bytes it did not produce.** The positions a job's bytes
  came from stopped existing when the worker's `pump` callback returned — which
  is exactly why the encoder runs worker-side — so there is nothing for the
  batcher to re-encode. It appends the items themselves with
  `EncodedBatch::push_bytes`, which is the batcher-side half of that seam and
  exists for this caller.

## Seeds

The driver reseeds both of a lane's sessions from entropy before every game,
mixing the clock with the lane index, the lane's game serial, and the seat — the
clock alone would hand two lanes reseeded in the same nanosecond, or the two
seats of one game, the same stream, and two seats drawing from one stream
correlate their choices.

**Nothing mints or records a per-game seed, and neither the run manifest nor a
shard header has a field for one.** `docs/CONTAINER_SPEC.md` §12 and
`OPEN_DECISIONS.md` B4: games are deliberately non-deterministic, and a recorded
seed that does not reproduce the game is worse than none, because it reads as a
guarantee nobody ever checked. `DecisionSession::reseed` is the seam B4 will land
on, and it already exists.

## Stopping

One `Arc<AtomicBool>`, set by the `ctrlc` handler — `SIGTERM` in a container,
Ctrl-C in a shell, which are the same request. `unsafe_code = "forbid"` rules out
a hand-rolled handler, which is why that one small crate is a dependency.

The flag is checked between epochs and inside every sweep, and where it lands
decides what happens:

| The stop arrives… | What happens |
| --- | --- |
| between epochs | Nothing more is started. |
| during self-play | The partial epoch is abandoned: the shard was never finalized so `ShardWriter` removes its temporary file, and `records/<epoch>` goes with it. Those games were on-policy and are worthless without the fit that was going to consume them, so half an epoch of them is not a smaller epoch — it is nothing. |
| once the fit has begun | The fit, the checkpoint placement, the load that proves it, and the metrics line all complete. `docker stop` must never lose an epoch. |
| during an evaluation round | The pairings that finished are reported and the rest are skipped. Evaluation is metrics, and metrics are not worth an epoch. |

Either way the run returns `Outcome::Stopped`, and the binary exits **2**.

## Exit codes

Pinned by §8.1 so that a supervisor, a shell loop, or a person reading `docker
inspect` does not have to infer them. A run that ended is not a run that broke.

| Code | Meaning |
| --- | --- |
| 0 | Ran to completion. |
| 2 | Stopped by signal, after finishing cleanly. |
| 1 | Failed. The error and its whole `source` chain go to stderr. |

## `train`

```
hexo-bot train --run-dir <dir> --run-id <id> --package <name> --epochs <n> --games <n>
               [--package-config <string>] [--batch <n>] [--threads <n>]
               [--batch-wait-ms <n>] [--ply-cap <n>] [--eval-every <n>]
               [--eval-games <n>] [--resume]
```

| Flag | Default | What it is |
| --- | --- | --- |
| `--run-dir` | *required* | Root of the accumulated state. Never defaulted: silently choosing a directory to write days of training into is exactly the substitution this workspace forbids, and the operator — not the binary — decides where the volume lands. |
| `--run-id` | *required* | The run's name. One path component, and written into every shard header, so it is held to ASCII letters, digits, `-`, `_`, and `.`. |
| `--package` | *required* | A name from `registry::PACKAGES`. |
| `--epochs` | *required* | How many epochs the run is for. A resume may raise it. |
| `--games` | *required* | Concurrent games, which is also how many games one epoch produces. From RAM and from how many leaves per game should be in flight (§7.2) — a different number from `--batch`, from a different constraint. |
| `--package-config` | `""` | Handed to the package verbatim. Absence is not a guess: the package decides whether an empty string is usable, and `mock` refuses one because it has no default search shape. |
| `--batch` | 64 | Encoded items per `Evaluator::evaluate` call. From what the device wants. |
| `--threads` | `available_parallelism() - 3`, min 1 | Worker threads, over and above the batcher and the writer (§7.1, §13). If the host's parallelism cannot be read, the flag becomes required rather than silently 1. |
| `--batch-wait-ms` | 2 | How long a partial batch waits for more leaves. A latency/throughput knob, and a starting point rather than a measured optimum. |
| `--ply-cap` | 512 | `GameSpec::ply_cap` for every game in the run. |
| `--eval-every` | 0 (never) | Run an evaluation round every this many epochs. |
| `--eval-games` | 32 | Games per evaluation pairing. Lanes are capped at `--games`, so a larger number plays successive games on the same lanes. |
| `--resume` | off | Continue an existing run rather than refusing its directory. |

MantisNet implements the container's encoder, evaluator, self-play, checkpoint,
and evaluation phases, but deliberately returns `PackageError::Unsupported`
when this loop reaches `fit`. Its production trainer is
`mantisnet.klent.run`; migrating that KLENT loop requires an owner decision,
and a partial record-consuming trainer would be worse than a loud refusal.
Consequently a complete `train` run is currently the mock's proof path, while
MantisNet checkpoints and play are exercised through `init` and `match`.

### The run directory

```
<run-dir>/runs/<run-id>/
  manifest.json              the run config and the four versions
  metrics.jsonl              one line per epoch, appended as it happens
  checkpoints/<epoch>/       weights.<ext> and the package's manifest.json
  records/<epoch>/           transient; removed after the fit that consumed it
```

Checkpoints are built under `checkpoints/<epoch>.partial` and renamed into
place, so a crash leaves a partial epoch directory and never a corrupt
checkpoint — and so a checkpoint directory's *existence* means the checkpoint is
whole.

An epoch's records go once the fit that consumed them has succeeded. Not before,
so a failed fit leaves its input on disk to be inspected or re-run; and not
later, because on-policy records are worthless under the new weights and keeping
them would make disk growth a function of run length.

### One epoch

0. At run start: write `checkpoints/0/` if it is not there, then load it —
   **loading is proving**, and the probe runs inside it.
1. **Self-play.** `--games` games through the driver, both seats
   `self_play_session()`, one evaluator. The writer appends each finished
   `GameRecord` to `records/<epoch>/shard-0000.hxr` and finalizes at the quota.
2. **Fit.** `package.fit(&[shard], <partial dir>, epoch + 1)`, then the rename.
3. **Records removed**, the fit having succeeded.
4. **Load** the new checkpoint, which is what puts the fit's own output behind
   the probe. Every evaluator this run uses is built after a load, from the
   package that performed it — the rule `hexo-model` states, satisfied by
   construction because a sweep builds its own.
5. **Evaluate**, every `--eval-every` epochs: the new checkpoint against the
   epoch-0 anchor and against the one it was fit from, `--eval-games` each. The
   opponent is a second package instance loaded from the older checkpoint.
6. **One metrics line**: the epoch, wall seconds per phase, games, the results
   breakdown, positions, evaluations, batches, mean batch fill, and one entry per
   evaluation pairing. `eval` is always an array, so a reader has one shape to
   handle rather than two.

### The run manifest, and `--resume`

`manifest.json` records the package, its config, the epochs, games, batch,
threads, ply cap, eval-every, eval-games, and the rules, action-order, protocol,
and records versions. `--resume` requires every one of them to match, except
that `epochs` may grow — extending a run is what a resume is often for, and a
successful extension is written back so the manifest always states the run's
current target. A mismatch names the field.

`--batch-wait-ms` is deliberately **not** recorded: it is a flush window that
changes how long a partial batch waits and nothing about what the run produces,
so holding a resume to the value it started with would forbid retuning a knob
with no bearing on the artefacts. `--threads` *is* recorded, because it is the
run's shape rather than a window.

A resume clears what a crash left and continues from the highest checkpoint. The
two kinds of leftover are treated differently on purpose: a `*.partial`
directory, or a checkpoint directory with no manifest, was never renamed into
place and is removed, because nothing was ever promised about it. A checkpoint
that *is* in place but does not load is an error — it arrived by rename, so it
was whole when it landed, and weights that no longer prove are a fact an operator
has to see rather than an artefact a resume may quietly discard.

## `match`

```
hexo-bot match --games <n> --seat <spec> --seat <spec>
               [--batch <n>] [--threads <n>] [--batch-wait-ms <n>]
               [--ply-cap <n>] [--report <path>]
```

| Flag | Default | What it is |
| --- | --- | --- |
| `--games` | *required* | How many games, and also the lane count: every game is in flight at once, which is what makes the batches wide. |
| `--seat` | *required, exactly twice* | One competitor. |
| `--batch` | 64 | As `train`. |
| `--threads` | `available_parallelism() - 3`, min 1 | As `train`. |
| `--batch-wait-ms` | 2 | As `train`. |
| `--ply-cap` | 512 | As `train`. |
| `--report` | — | Also write the JSON report to this path. It goes to stdout either way. |

A seat spec is `;`-separated `key=value`:

```
package=<name>;checkpoint=<dir>[;config=<package config>][;variant=<name>]
```

Semicolons, not commas, because a package configuration string and a session
variant name both contain commas —
`config=search=mcts:visits=64,inflight=8,cpuct=1.5` is one value, not three. An
unknown key, a repeated key, a segment that is not a pair, and a missing
`package` or `checkpoint` are each refused by name. `config` defaults to the
empty string and the package decides what to make of it; `variant` defaults to
absent, which means the package's `eval_session()`.

MantisNet requires `config=tau=F,lambda=F` when loading a sealed checkpoint
(`source` is only for `init`). Its default evaluation session is shared Gumbel
search at 32 simulations and 16 candidates. Its exact variants are `policy`,
`mcts:visits=N,inflight=N,cpuct=F`, and `gumbel:sims=N,m=N`:

```
package=mantisnet;checkpoint=<dir>;config=tau=0.1,lambda=0.03;variant=gumbel:sims=32,m=16
```

**Both seats may name the same checkpoint**, and that is the point of the
subcommand rather than a curiosity: same weights, two searches, which is how a
search shape is compared without a training run in between. Each seat still gets
its own package instance, because a package holds the weights that answer and the
container has no way to know whether two of them could share one — the mock's
evaluators are independent copies of a salt, and MantisNet's is a live module,
which is what §10.1's slot pool exists for and what will make sharing an
optimisation rather than an assumption.

Colours alternate: the first `--seat` is `P0` in even-numbered lanes, and a lane
that plays more than one game swaps colours between them, so the first-move
advantage is split for any combination of lane count and quota rather than only
for the ones that divide evenly. The report says how many games each seat won in
each colour, because that is the question a bare win count cannot answer.

## Deliberately absent

| Omitted | Why |
| --- | --- |
| `serve` and `play` | `CONTAINER_SPEC.md` §3, blocked on C1 and C2. Both are entirely wire protocol and there is no wire protocol. A stub that parses `--seat` and exits with "not implemented" is a half-implementation in the exact sense this workspace forbids: it publishes a command line before the thing behind it is designed, and the flags it guessed become the constraint the real implementation has to argue its way out of. |
| Shards from `match` or from an evaluation round | §11 keeps records as *training data*, and nothing consumes an evaluation game: they are evidence about two checkpoints, which is what the report and the metrics line are for. A shard nobody reads would be the largest artefact in the run written for no reader. |
| A minted or recorded seed | §12 and B4. See **Seeds** above. |
| Separately invocable phases | §8: there is one implementation of the loop, not a loop plus a set of pieces that could drift from it. |
| A checkpoint-reference resolver for `latest` or `<run-id>/<epoch>` | §10 names those forms and nothing needs them yet: `train` resolves its own checkpoints from the run layout, and `match` takes a directory. It would land here, next to `RunLayout`. |
| A second batcher, or a batcher per worker | The crossing has to be serialised anyway; a second one would only contend for the same lock and the same device. |
| A slot pool sharing one live module between seats | §10.1. MantisNet currently proves and owns one CPU live module per package instance. Sharing compiled modules is a later optimisation and must not become an assumption about package internals. |
| A dashboard, a scraping endpoint, or a metrics service | §8.1 and §14: one line per epoch appended to a file is the whole of "observable without stopping". |

## Connections

- `hexo-runner` supplies `Game`, which the driver advances directly. `Table` in
  `hexo-player` is the *other* driver — one blocking `choose` per turn, for a
  human, a scripted bot, or a transport adapter — and is deliberately not this
  one, because a blocking `choose` is the design that cannot batch. The desync
  path here follows `Table::step`'s, and for the same reason.
- `hexo-search` supplies `DecisionSession`, `Encoder`, `Evaluator`, and
  `EncodedBatch`. Its `tests/topology.rs` is this driver single-threaded; this is
  that loop with the `for` replaced by a pool and two bounded queues, and nothing
  else about the shape changed.
- `hexo-model` supplies `ModelPackage`, `Manifest`, and the probe behind every
  load. `crates/models/mock` is the package the whole loop is exercised against
  on every CI run; `crates/models/mantisnet` supplies the first real encoder,
  improved evaluator opinion, sessions, checkpoint semantics, and the
  Python-free forward traits this executable implements.
- `hexo-records` supplies `ShardWriter` and the format; a package's `fit` reads
  the shards back through the same crate.
- `hexo-engine` supplies the rules underneath all of it, and three of the four
  version constants the artefacts pin.
