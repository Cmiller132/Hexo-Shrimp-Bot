# MantisNet Graft Campaign — Stepwise Specification

**Status:** normative plan for evolving the production MantisNet in place.

**Supersedes:** `docs/MANTIS_ACT_SPEC.md` v4.1, which specified MantisNet-ACT
as a separate selectable architecture. ACT hit its architectural FLOP floor
(~59.7 vs production's 20.5 GFLOP/step; 208 vs 1,020 positions/s on the same
harness) and was stopped. The full v4.1 text lives in git history (`b735d27`);
the complete ACT implementation lives on branch `mantisnet-act` (`cc3edef`).
Every donor payload a step needs is carried into that step below — the ACT
spec does not need to be consulted to execute this one.

**The shape of the campaign:** there is one model. Each step modifies it in
place, holds throughput within measurement noise by shipping the kernel and
plan work inside the same step, is measured against the tree it landed on, and
is baked in or reverted on an explicit owner verdict. No separate architecture,
no permanent dual paths, no registry of dead experiments.

---

## 1. Standing contracts

Unchanged by every step:

- engine rules, legal moves, engine legal-move order, and terminal detection;
- one placement is one MDP action; `moves_remaining == 2` means the mover
  places again, `== 1` means control changes after a nonterminal placement;
- policy, Q, acting scores, and returns are in the current mover's frame;
- stone colours are encoded relative to the side to move;
- terminal positions are refused by the builder, never silently defaulted;
- the KLENT evaluator contract: flat CPU `(policy_logits, q_score, q_value)`
  in exact engine legal order with `legal_offsets`;
- the KLENT return sign derives from the authoritative `moves_remaining`,
  never from any model feature;
- checkpoint loading is strict: exact config and `MODEL_REPR_VERSION`
  agreement, no conversion on the ordinary path;
- no absolute coordinates, board origin, move history, or recency features;
- fp32 parameters; bf16-autocast-safe forward; softmaxes, segment sums, and
  critic composition in fp32;
- every optimized kernel has a reference implementation and random-weight plus
  real-checkpoint parity tests against it. Numerical qualification is
  path-specific: each eager/compiled/fused path is compared with its declared
  reference and tolerance; one path's pass never stands in for another's.

The architecture seams KLENT and the lab consume (`collate_positions`,
`collate_prefixes`, the pack budgets, `trunk` + `cell_head_logits`) keep their
signatures; steps may change what flows through them, with the representation
version bumped accordingly.

---

## 2. Campaign protocol

Every step runs the same loop:

```text
implement behind one transient knob (kernels and plans in the same change)
→ verify        (cargo xtask verify, pytest, lab check, step oracles)
→ benchmark     (speed gate, §2.2)
→ ablate        (supervised screen, §2.3 — skipped only where §4 says so)
→ review packet (§2.4) → owner verdict
→ bake (delete the knob) or revert (delete the path)
```

### 2.1 Baseline chain

The baseline for step *N* is the tree with steps *0..N−1*'s verdicts applied —
accepted steps baked, rejected steps absent. Baselines are never restated from
memory: each step's packet cites the baseline commit and its freshly measured
numbers from the same session on the same machine.

### 2.2 Speed gate — hard

"Must not sacrifice speed" is a per-step acceptance rule, not an aspiration.
The feature's cost must be paid for by the step's own kernel, plan, and layout
work; a step that cannot hold the line does not land, regardless of strength.

Harness (pinned by owner ruling, 2026-08-10, after the Step 0 calibration
runs; Windows-native measurement showed ±40% same-arm swings and is
disqualified):

- **Environment:** the WSL ext4 clone of the graft branch on this machine
  (`~/graft-bench`, venv `~/graft-venv`), RTX 4070 Ti 12 GiB, no concurrent
  CPU- or GPU-intensive load.
- **Fit:** `lab bench fit --corpus <campaign corpus> --split val --device
  cuda --compile --seed 7 --steady-warmup 20 --steady-measure 50` at the
  default budgets — the §2.2 steady window: warm-up chunks absorb
  compilation untimed, then exactly 50 CUDA-synchronized chunks are timed.
  Baseline at Step 0: **≈205 samples/s, rep spread ±0.2%**.
- **Collect:** `lab bench collect` over a production-shaped cohort at the
  production collection budgets.
- Repeat ≥ 3 times per arm; identical seed so packing, chunk order, and
  recompile points pair exactly across arms.

The corpus-mode fit number is a **gate metric, not production throughput**:
it times the supervised path (all heads) over the frozen corpus
distribution. Production KLENT iterations pipeline collection with fitting
and report far higher combined rates; the gate compares like with like
across arms and steps, nothing else.

Pass rule, all four required:

1. median fit samples/s within **2%** of the baseline median;
2. fit p95 chunk latency not worse than baseline by more than **5%**;
3. median collect positions/s within **2%** of the baseline median;
4. peak reserved VRAM not above baseline + 256 MiB and never above
   **10.25 GiB** (owner-amended 2026-08-10: the Step 0 baseline peaks at
   10.06 GiB, and the WSL harness showed no paging cliff there; the ~10.5
   GiB WDDM cliff is a Windows-native phenomenon).

Removal-trial steps (marked in §4) invert the expectation: they must show a
speed *gain* to be worth their strength risk.

Record with every benchmark: commit, resolved config, GPU/driver/CUDA/PyTorch
versions, compile mode, prefetch worker count, corpus digest.

### 2.3 Strength screen — supervised, paired, multi-metric

The campaign corpus is **`mnorm-late-v1`** (owner ruling 2026-08-10): 1M
train / 100k val / 100k test realized samples from `joint-mnorm` iterations
100–149, archive SHA-256
`cd5f5d0a0bc53c76aba6a1fe6d9a02f57f51ce8a46732a0a5e0ea1c560cec80c`. It is
reused unchanged by every step.

Each screen:

- **Arms:** knob-off (incumbent) vs knob-on, built from the *same tree*.
- **Pairing:** ≥ 5 seeds, identical seed set for both arms; identical corpus
  stream, fit recipe, budgets, epochs, and optimizer settings. The knob is the
  only difference between arms.
- **Predeclared primary metrics:** validation policy top-1 imitation accuracy
  and validation critic sign accuracy (overall).
- **Secondary/guard metrics:** top-3 accuracy, policy CE, critic CE, v̂ MAE,
  and the per-distance-from-end-bucket and per-phase splits of the above. A
  step must not be promoted on an unregistered metric, and a secondary-metric
  collapse is grounds for the owner to reject a primary-metric win.
- **Statistics:** per-seed paired deltas; report mean Δ, 95% t-interval, and
  per-seed sign counts for every metric. "Statistically significant" means the
  interval excludes zero on a primary metric. The test split is consulted only
  after the verdict, never while tuning a step.

The screen *informs*; it does not decide (§2.4).

### 2.4 Owner verdict — nothing is auto-decided

Every step ends with a review packet: the speed table (§2.2), the full paired
metric table with intervals (§2.3), parameter and VRAM deltas, any step
oracle/diagnostic output, and a recommendation. The owner approves the bake or
orders the revert (or rework).

Standing rules for the verdict stage:

- A **neutral or barely-negative** result never auto-removes a candidate. The
  packet is presented and the owner rules; "it measured ≈0" is a finding, not
  a deletion warrant.
- A significant result on one metric with contradiction on another is
  presented as exactly that — no silent metric arbitration.
- Rejected steps are recorded in `docs/ABLATIONS.md` with their full numbers
  so the verdict is re-openable on new evidence without rerunning from memory.

### 2.5 Knob lifecycle

- Each step adds exactly **one** field to `MantisConfig`, defaulting to the
  incumbent behaviour. The knob exists to let the lab build both arms from one
  tree; it is not a supported configuration surface.
- **On accept:** the incumbent path is deleted, the knob field is removed, and
  the knob's name and accepted value are added to `LEGACY_BAKED_KNOBS` so
  checkpoints written during the measurement window still load (matching
  recorded value) or are refused loudly (any other value).
- **On reject:** the entire new path comes out — code, kernels, builder
  fields, tests. No dormant branches remain in the tree.

### 2.6 Versioning

- Any change to the builder output or `Batch` layout bumps
  `MODEL_REPR_VERSION` (Rust-owned, shared with `hexo_py`) in the same change.
- Any state-dict key-set or shape change invalidates checkpoints through the
  existing strict load; steps note this in their packet. Mid-campaign
  checkpoint continuity is not a goal — the lab retrains from the frozen
  corpus, and KLENT runs restart from fresh weights after the campaign.
- Golden vectors and independent oracles are updated by *bumping and
  regenerating*, never by re-baselining an oracle to agree with the
  implementation it is supposed to check.

### 2.7 Records

`docs/ABLATIONS.md` gains one row per step: step id, arms, seeds, primary and
secondary deltas with intervals, speed table summary, verdict, and date. The
packet's full tables live with the run artifacts; the ABLATIONS row is the
durable record.

---

## 3. Step index

| # | Slug | Candidate | Gate profile |
|---|---|---|---|
| 0 | `perf-foundation` | pinned prefetch, fused Adam, `segment_reduce` offsets | speed only |
| 1 | `dead-key-bias` | remove softmax key biases | parity + speed |
| 2 | `state-latents` | 4 invariant state latents replace the token | full |
| 3 | `wa-removal` | remove typed window-pair attention | full, speed-gain expected |
| 4 | `action-rows` | 18 counterfactual post-placement rows per legal action | full |
| 5 | `tactical-scalars` | deterministic per-action tactical features | full |
| 6 | `action-latents` | 2 action-set latents over the legal set | full |
| 7 | `gated-incidence` | relation-gated trunk messages | full |
| 8 | `phase-film` | two-way phase FiLM in every block | full |
| 9 | `global-numeric` | global scalar conditioning of the latents | full |
| 10 | `head-adapters` | private per-head residual depth | full |
| 11 | `orbit48` | exact 48-orbit D6 attention geometry | full |
| 12 | `mixed-windows` | ternary all-nonempty window nodes | full |
| 13 | `cell-nodes` | explicit relevant-cell nodes + geometry edges | full |
| 14 | `axis-channels` | three axis-equivariant channels | full |

**Owner ruling (2026-08-10): ACT-era ablation results are disregarded as
evidence.** The old ablations were coarse; no step is favoured or disfavoured
by them, and this spec cites them nowhere as grounds. The only evidence that
counts for any verdict is this campaign's own paired measurements.

Steps keep their numbers as stable identities. Execution follows the
owner-set order (2026-08-10, revised same day to elevate mixed windows —
an owner priority — out of the frontier), dependencies respected:

```text
0 → 1 → 12 → 4 → 2 → 5 → 3 → 6 → 7 → 9 → 8 → 10 → frontier
```

Step 12 precedes Step 4 so the counterfactual action tables are built
ternary-native (729 classes) once, rather than binary (189) and then
migrated; if Step 12's verdict is a revert, Step 4 proceeds with the
binary-graft tables as specified. Step 12 runs as an owner-amended matrix
(see its Gates) that folds a window-attention-off arm in as its principal
cost offset; a bake with that arm subsumes Step 3. (Step 5 presumes Step 4's verdict; Steps 3
and 9 require Step 2 to have landed.) The owner may revise the order between
steps. Steps 11, 13, and 14 (**structural frontier**) additionally require
an explicit owner decision *before work starts*, because each is a large
implementation whose hard speed gate is at serious risk from irreducible
work; Step 12's elevation on 2026-08-10 is exactly such a decision.

---

## 4. The steps

Every step description has five parts: **Donor** (what ACT established),
**Adaptation** (what lands in production MantisNet), **Performance work** (the
in-step kernel/plan work that pays for it), **Knob**, and **Gates & bake**.

### Step 0 — `perf-foundation`

**Donor.** The three architecture-independent performance commits on
`mantisnet-act` (verify hashes at execution time):

- `5044d90` — pin prepared chunks on the prefetch worker so H2D transfers
  overlap launches instead of serializing against them (+35% measured);
- `78d4204` — `mantisnet.optim` fused-Adam step;
- `bbe64ca` — pass precomputed `offsets=` to `torch.segment_reduce` instead
  of `lengths=`, avoiding a ~3 ms/call host-side scan-and-validate.

**Adaptation.** Cherry-pick onto the graft tree; resolve drift against the
current `fitloop.py`/`builder.py`. Also freeze the campaign corpus (§2.3) and
take the campaign's opening baseline measurements.

**Performance work.** This *is* the performance work.

**Knob.** None — the model function is unchanged bit-for-bit.

**Gates & bake.** Speed gate only, with the inequality reversed: fit and
collect throughput must **improve or hold**. Full verification suite green.
No strength screen (nothing the screen measures can change). Baked directly
on the owner's approval of the numbers.

### Step 1 — `dead-key-bias`

**Donor.** ACT §17.6: a key-projection bias adds one constant to every key;
for any query the score shift is uniform across that query's softmax row and
cancels exactly. Such parameters are structurally dead — they receive zero
gradient and never move.

**Adaptation.** Production has two softmax key projections per block: `wk`
(stone self-attention) and `wk_wa` (window-pair attention). Remove both
biases: **8 tensors, 1,024 scalars** over the 4 blocks. The census test
enumerates `named_parameters()` and asserts that no parameter whose owning
projection feeds a softmax as keys ends in `.bias` — a substring count is not
sufficient. (The value-head readout scores raw rows without a key projection;
the decoder heads have no key softmax; neither is affected.)

**Adaptation note:** this is a state-dict key-set change; checkpoints written
before it are refused by strict load (§2.6).

**Performance work.** None needed; two fewer bias adds.

**Knob.** None — removal is functionally exact, so there is nothing for a
screen to measure.

**Gates & bake.** Parity gate instead of a screen: forward outputs of the
same weights with biases dropped must match within pinned fp32 tolerance on
real batches, and the zero-gradient claim is asserted directly (bias grads
identically zero before removal). Speed gate applies pro forma. Owner
approves; baked.

### Step 2 — `state-latents`

**Donor.** ACT §17: multiple learned global latents with a
read → mix → broadcast cycle per block, in place of a single token. The
production case is mechanistic: today the token attends over stones only, so
window state — where the tactical content lives — reaches the global summary
only secondhand, and both decoder g-halves condition every per-cell score on
that bottleneck.

**Adaptation.** Replace the single global token with `K = 4` invariant state
latents:

- each latent has a distinct learned base (`N(0, 0.02)`) plus the shared
  `moves_remaining` embedding;
- **stone read/broadcast** rides the existing fused stone attention: latents
  occupy padded rows `0..K−1` (today's token is row 0), every latent↔stone
  and latent↔latent pair takes the TOKEN bias bucket, `max_t` grows by
  `K − 1`;
- **window read/broadcast** is new: per block, latents attend over the padded
  `(P, max_w)` window layout (the value head's layout) and windows receive
  latent context back, fp32 softmax, bias-free key projections (Step 1's
  rule);
- **mix**: self-attention across the K latents per position;
- heads that read the token (both decoder `g`-halves, the value-head query
  context) read the **mean-pooled latents** after the final LN.

Builder/collation change: latent rows in the padded attention layout →
`MODEL_REPR_VERSION` bump, Rust builder updated in the same change.

**Performance work.** The stone-side cost is `K − 1` extra live rows per
position inside the existing kernel — no new kernel. The window read/mix/
broadcast is small dense masked attention over already-padded layouts;
implement it as batched GEMMs + fp32 masked softmax (no Python loops, no new
Triton). Budget accounting: `max_t` growth enters the pair budget; the packer
constants are re-tuned in-step if the gate demands it.

**Knob.** `state_latents: int = 0` — `0` is the incumbent token path exactly;
the measured arm is `4`. (The knob selects a path, not a width; widths other
than 0/4 are refused during the campaign.)

**Gates & bake.** Full protocol. On accept: token path deleted,
`state_latents: 4` enters `LEGACY_BAKED_KNOBS`.

### Step 3 — `wa-removal` (removal trial)

**Donor.** ACT's default (`window_window_mode="none"`) holds that direct
typed window↔window attention is redundant once global latents and shared
structure carry cross-window context; ACT never enabled it by default.

**Adaptation.** Runs only after Step 2 lands (the latents are the replacement
path; measuring removal without them tests a strawman). The trial arm
disables the §5.1c typed window-pair attention stage. On accept, the removal
is deep: the `window_pairs.py` module, its four kernels, the per-block
`wq_wa/wk_wa/wv_wa/wo_wa/wa_bias` parameters, the on-device pair-table
derivation, and `window_id` from the batch and both builders (it exists only
to feed pair tables and tests) — with the `MODEL_REPR_VERSION` bump that
implies.

**Performance work.** None to add — the step *is* a performance candidate.
The packet reports the reclaimed time and memory explicitly.

**Knob.** `window_attention: bool = True` (returns from `LEGACY_BAKED_KNOBS`
to a live field for the trial; measured arm `False`).

**Gates & bake.** Inverted speed expectation: the trial must show a **speed
gain** (otherwise removal has no upside and the step is dropped without a
screen). Strength rule: non-inferiority — the owner sets the acceptable
paired-Δ lower bound in the packet review; the screen must show the removal
arm's primary metrics within it. On accept: removal executed as above and
`window_attention: False` is *not* recorded in `LEGACY_BAKED_KNOBS` — the
stage ceases to exist and old checkpoints are already invalidated by the
parameter removal. On reject: knob deleted, stage returns to baked-in status.

### Step 4 — `action-rows` (the big port)

**Donor.** ACT §19: every legal action is encoded from its 18 hypothetical
post-placement windows (3 axes × 6 candidate slots with an own stone inserted
at the action cell), replacing any nearest-distance background alias; ACT §33's
structural-alias diagnostic verifies the alias is gone. Donor code: the Rust
builder and plan modules on `mantisnet-act` (`act_encoder.rs`,
`act_plans.rs`) and the row-encoder kernels — reusable with the class tables
swapped.

**Adaptation.** Production windows are binary and live-only, so the joint
post-placement classes adapt from ACT's ternary 729 to:

```text
POST1_GRAFT_CLASSES = 2 * DEC_CLASSES + 3 = 2 * 93 + 3 = 189
```

- inserting into an **own live** window: joint `(mask, slot)` empty-slot
  reversal orbit — the existing 93-orbit table, own colour;
- inserting into an **opponent live** window: the same 93 orbits, opponent
  colour (the post-window is dead — this is the blocking signal);
- a candidate window with **no current stones**: only the slot survives
  reversal, `slot ↦ 5 − slot` folds 6 slots to 3 orbits.

Assert 189 and the orbit properties (reversal-invariant, distinct across
orbits) against an independent successor-board oracle, exactly as the builder's
existing class tables are tested.

Builder output, dense and engine-ordered (Rust, `MODEL_REPR_VERSION` bump):

```text
action_window_index: [N_legal, 3, 6]   # live-window index or -1
action_post1_class:  [N_legal, 3, 6]   # 0..188, or -1 on DEAD rows
action_pre_status:   [N_legal, 3, 6]   # OWN_LIVE / OPP_LIVE / EMPTY / DEAD
```

A candidate window already containing **both colours** is dead — placing
there cannot revive it. Its row carries `pre_status = DEAD` and class `-1`,
and is **masked out of the row encoder**: a dead line is absent tactical
potential, not an empty line waiting to be born, and mapping it to EMPTY
would mislabel spent lines as fresh ones. The builder's candidate walk
already enumerates these windows before filtering to live, so the status is
free to emit; consumers must mask before any class gather.

Model side: per row, gather the final trunk window state (or a shared learned
empty-window base when the index is −1), combine with the 189-class relation
embedding, apply a shared per-row MLP, sum the six rows per axis and the three
axes into one invariant action contribution added to the decoder aggregate
row. Both cell heads read it through their extended `head_matrix`. **This path
runs for every legal action**, including cells in no live window — on accept,
the background nearest-bucket path (`bg_cell`, `bg_bucket`, `e_bg`, `e_qbg`,
`NEAREST_BUCKETS`) is deleted; the alias it papered over no longer exists.

The §33 alias diagnostic is ported as a lab command in this step: builder-side
structural signatures per legal action, 64-bit hash as a bucket index only,
full-signature comparison within buckets, reporting alias groups before/after.

**Performance work.** The irreducible cost is ~18 gathers + one small MLP per
legal action; the step stands or falls on making that near-free:

- all row tables and their CSR views (row-major by action, source-window-major
  for the backward, class-major in ≤128-row blocks for the embedding gradient)
  are built at collation in Rust — the forward performs **no data-dependent
  index discovery**, matching the house rule;
- one fused Triton row-encoder kernel per direction (forward, input-gradient,
  table-gradient), in the style of `message_passing.py`'s single run-reduce
  kernel serving four directions; fp32 accumulation; reference path for
  parity;
- the row MLP is folded so its first layer runs as one GEMM over packed rows;
- `chunk_cost` gains the per-position row count so the packer charges
  honestly; budgets re-tuned in-step.

**Knob.** `action_rows: bool = False` (measured arm `True`).

**Gates & bake.** Full protocol; this step gets the most scrutiny on speed
gate item 3 (collection runs the decoder too). On accept: background path
deleted, `action_rows: True` into `LEGACY_BAKED_KNOBS`, alias diagnostic
output archived in the packet.

### Step 5 — `tactical-scalars`

**Donor.** ACT §19.3: a deterministic, search-free tactical feature vector
per legal action, encoded by a small invariant MLP, toggleable.

**Adaptation.** Computed in the Rust builder from the live-window tables and
Step 4's row tables (requires Step 4 accepted; if Step 4 was rejected this
step is respecified before work starts). Per legal action:

```text
immediate_win                       max own count after / 6
max opponent count before / 6       own five-window count after
own four-window count after         opponent five-windows hit
opponent four-windows hit           opponent five-windows remaining globally
opponent four-windows remaining     blocks-all-immediate-threats flag
nonempty post-windows / 18
```

Count-normalized, clipped, emitted as `[N_legal, 11]` fp32
(`MODEL_REPR_VERSION` bump), encoded by one small MLP into the action row.
Exact deterministic inputs are not double-claimed as auxiliary prediction
targets.

**Performance work.** Builder-side vectorized computation over tables the
Step 4 walk already produced; model-side one GEMM. Expected comfortably
within noise; measured anyway.

**Knob.** `action_tactical: bool = False` (measured arm `True`).

**Gates & bake.** Full protocol.

### Step 6 — `action-latents`

**Donor.** ACT §21: two invariant latent queries read the whole legal action
set, self-mix, and broadcast back — permutation-invariant context about the
alternatives without quadratic action attention. State latents and action
latents stay separate: state latents carry no post-placement information.

**Adaptation.** After the decoder aggregate rows (and Step 4/5 contributions
where accepted): 2 learned action latents per position read the action rows
via ragged segment attention (fp32 segment softmax over `legal_offsets` — the
machinery in `segments.py`), mix (2×2 per position), and broadcast back into
the rows both heads read. Bias-free keys.

**Performance work.** Segment-softmax attention over cells reuses the
existing segment primitives; no padding, no new kernel unless profiling
demands one — if it does, it lands in-step with parity tests.

**Knob.** `action_latents: bool = False` (measured arm `True`).

**Gates & bake.** Full protocol.

### Step 7 — `gated-incidence`

**Donor.** ACT §14: relation-gated messages — value scaled by a learned
per-relation gate plus a per-relation bias — against the additive control
`U·src + E_rel`. Production's trunk passes are exactly the additive control.

**Adaptation.** In the two trunk incidence passes (§5.1 windows←stones,
§5.2 stones←windows): per relation class `r`,
`msg = sigmoid(Wg·E_rel[r]) ⊙ (U·src) + Wb·E_rel[r]`. Gates and biases depend
only on the class, so both stay table-shaped: the gate table
`(OCC_CLASSES, H)` is computed once per forward, and the bias term remains
the existing dense `counts @ table` matmul.

**Performance work.** Extend the run-reduce kernel with one optional
per-entry gate-row operand (gathered by the entry's class) — same kernel
serving all four directions, reference path updated, parity tests. No new
launches.

**Knob.** `gated_incidence: bool = False` (measured arm `True`).

**Gates & bake.** Full protocol.

### Step 8 — `phase-film`

**Donor.** ACT §13: placement-phase FiLM (`h ← scale ⊙ h + bias` from a
phase-embedding MLP) on every block's residual streams, initialized to exact
identity. The phase is a model feature only; the KLENT sign contract is
untouched.

**Adaptation.** ACT's phase was three-way, but its OPENING class is vacuous
for production: the engine forces the opening placement at the origin
(`IllegalOpening`; the legal count is 1 during `Opening`), so the OPENING
phase contains one position with one legal move — π′ is trivial there and
v̂₀ is never read by the λ-return. The graft is therefore **two-way**:
FIRST (`moves_remaining == 2`) vs SECOND (`== 1`), read directly from
`moves_idx` — no builder change, no repr bump. FiLM applies per block to the
stone, window, and latent streams with per-stream projections; identity init
so the knob-on arm starts exactly at the incumbent function.

**Performance work.** A per-position scale/bias broadcast — negligible;
measured anyway.

**Knob.** `phase_film: bool = False` (measured arm `True`).

**Gates & bake.** Full protocol.

### Step 9 — `global-numeric`

**Donor.** ACT §13.3: initialize the global latents from an MLP over a small
set of stable state-derived scalars; no history, recency, or absolute
position.

**Adaptation.** Requires Step 2. Seven fp32 scalars, all derivable on device
from the existing batch (no builder change):

```text
log1p(total stones)      own stone fraction        opponent stone fraction
log1p(legal count)       log1p(live window count)
own-colour window fraction        opponent-colour window fraction
```

Encoded by one MLP added to the latent initialization.

**Performance work.** Negligible; measured anyway.

**Knob.** `global_numeric: bool = False` (measured arm `True`).

**Gates & bake.** Full protocol.

### Step 10 — `head-adapters`

**Donor.** ACT §23: policy and critic share a trunk but own private adapter
blocks, so their gradients decorrelate before the outputs.

**Adaptation.** Production heads are already fully private after the shared
parameter-free aggregation — the honest production analogue of "private
adapters" is **one private residual MLP block per head** on the decoder rows
before the existing output MLPs. Zero-init the residual output so the knob-on
arm starts at the incumbent function.

**Performance work.** Two GEMM blocks over cell rows; bounded by the cell
budget; measured.

**Knob.** `head_adapter_blocks: int = 0` (measured arm `1`).

**Gates & bake.** Full protocol.

---

### Structural frontier — steps 11, 13, 14 (and 12's origin here)

Each of these is a large build whose irreducible cost puts the hard speed
gate at genuine risk. Per the campaign rules they remain fully specified
candidates, but **starting one requires an explicit owner decision**, and
each carries its honest cost projection in the request. Step 12 received
that decision on 2026-08-10 (owner: all-nonempty windows are a priority
idea) and moved into the main execution line; its speed gate is unchanged.

### Step 11 — `orbit48`

**Donor.** ACT §11: exact D6 displacement orbits replace coarse distance
buckets. Generation law (carried verbatim): apply all 12 engine-consistent D6
transforms to `(dq, dr)`; the canonical form is the lexicographic minimum;
enumerate every nonzero displacement with hex distance 1..12; assign stable
IDs sorted by `(distance, canonical_dq, canonical_dr)`; **assert exactly 48
orbits**. Generated from the transform functions, never hand-enumerated;
radii above 12 are refused, not clipped.

**Adaptation.** The stone-attention bias vocabulary changes from
`(distance bucket 1..12, on-axis flag)` — 2·d_max+3 learned rows per head —
to `(orbit 0..47, SELF, TOKEN, PAD)` — 51 rows per head. The orbit id is a
pure function of `(dq, dr)` within radius 12; beyond radius 12 the existing
clamp row semantics are preserved via a FAR row. Checkpoint bias-table shape
change; no builder change (coordinates already ride the batch).

**Performance work.** The fused attention kernel currently computes buckets
from coordinates inline; it gains one gather into a constant 25×25 int8
orbit-grid table (device-resident, `.ca`-cached) in place of part of the
arithmetic. Forward and all three backward kernels updated together with the
dense reference; parity tests.

**Knob.** `orbit48: bool = False` (measured arm `True`).

**Gates & bake.** Full protocol.

### Step 12 — `mixed-windows`

**Donor.** ACT §9: every nonempty window is a node — own-only,
opponent-only, and mixed — under ternary slot patterns. Carried class laws:
slots ∈ {empty, own, opp}; canonicalize all 3⁶ = 729 patterns under slot
reversal; **assert 378 classes total and 377 nonempty**; joint
(pattern, slot) incidence classes under joint reversal
`(pattern, slot) ↦ (reverse(pattern), 5 − slot)`; **assert 2187 all-pattern
and 2184 nonempty-pattern classes**. A current six-own or six-opponent window
is terminal and refused; a full mixed window is legal.

**Adaptation.** The builder's window walk keeps every deduplicated nonempty
candidate instead of only one-colour ones; `window_feat` becomes the ternary
class (+ status OWN_LIVE/OPP_LIVE/MIXED); stone incidence and decoder classes
move from the binary 93-orbit tables to the ternary joint tables; the relay
and (if still present) pair tables widen accordingly; Step 4's classes extend
from 189 to the ternary 729. Full `MODEL_REPR_VERSION` bump and golden-vector
regeneration.

**Performance work.** Node count grows by the mixed-window fraction (measure
it on the campaign corpus *before* building — the packet requesting this step
must include the projected node/edge growth). Kernels are count-agnostic; the
speed line is held or lost on node economy, so this step's performance work
is measurement-driven layout tuning plus packer re-budgeting.

**Knob.** `mixed_windows: bool = False` (measured arm `True`), plus the
resurrected `window_attention: bool = True` for the matrix below.

**Gates & bake — owner-amended protocol (2026-08-10).** The owner expects
this step to cost speed and to be worth exploring anyway. It therefore runs
as a small predeclared matrix rather than one arm:

| Arm | mixed_windows | window_attention | Purpose |
|---|---|---|---|
| A (baseline) | off | on | the incumbent |
| B | on | on | the idea, undiluted |
| C | on | off | the main cost offset — pair-attention cost grows with the window count, so removal buys most exactly here |
| D… | on | — | further speed-recovery arms as profiling suggests (kernel/layout work, incidence trimming), added to the packet as measured |

All arms screen with the same paired seeds; speed is measured for every arm.
The hard 2% gate applies only to the **combination finally proposed for
bake**; exploratory arms may run slower during measurement without being
auto-rejected. The owner judges the speed/strength trade from the full
matrix. If the baked combination has `window_attention` off, Step 3's
removal trial is subsumed and its deletion work happens here.

### Step 13 — `cell-nodes`

**Donor.** ACT §8/§15: explicit relevant-cell nodes (occupied ∪ legal ∪
persistent-window cells), hex-adjacency edges, and occupied-to-cell radius
edges under the 48-orbit vocabulary; symmetry-safe neighbour selection
(radius thresholds, never coordinate-tie-broken top-K).

**Adaptation.** This is the ACT node economy the fork explicitly rejected —
it multiplies node and edge counts on exactly the dense late positions that
set the packing budgets. It is retained as a specified candidate because the
campaign's rules say measurement, not memory, closes questions; but the
request packet must project node/edge growth on the campaign corpus and
identify which existing structures (relay, background path, action rows) it
subsumes, and the expected outcome is that the hard speed gate fails without
extraordinary kernel offsets. Detailed sub-specification is deferred to that
packet (donor: ACT §8, §10, §15 in `b735d27`).

**Knob.** `cell_nodes: bool = False` (measured arm `True`).

**Gates & bake.** Full protocol; owner pre-approval mandatory.

### Step 14 — `axis-channels`

**Donor.** ACT §12, carried in full because any future attempt needs the law
exactly:

- every node carries `h_inv: [.., d_inv]` and `h_axis: [.., 3, d_axis]`; for
  transform `g` with node map `T_g` and induced axis permutation `π_g`:
  `h_inv′(T_g(i)) = h_inv(i)`, `h_axis′(T_g(i), π_g(a)) = h_axis(i, a)`;
- forbidden: concatenating the three channels in fixed order into an
  unconstrained MLP; any per-absolute-axis weights, biases, norms, or
  embeddings; absolute axis identity as a lookup;
- allowed: shared maps applied per channel, native-axis routing,
  permutation-equivariant cross-axis mixing, symmetric pooling;
- `AxisMix`:

  ```python
  u_a = LN_axis(x_a)
  total = sum_b u_b
  other_a = (total - u_a) / 2
  delta_a = MLP_axis([u_a, other_a, W_inv_to_axis(LN_inv(h_inv))])
  x_a += layer_scale_axis * delta_a
  axis_summary = sum_a phi_axis(u_a) / 3
  h_inv += layer_scale_inv * MLP_inv([LN_inv(h_inv), axis_summary])
  ```

  with all per-axis applications sharing parameters;
- invariant readouts pool axes symmetrically (mean, or learned attention
  whose scores permute with the channels);
- axis bases are learned once and replicated across the three channels.

**Adaptation.** Roughly triples trunk FLOPs — the candidate least likely to
survive the hard speed gate, listed last for exactly that reason. The request
packet must state the FLOP projection against the then-current baseline and
the kernel plan that would absorb it; detailed sub-specification is deferred
to that packet.

**Knob.** `axis_channels: bool = False` (measured arm `True`).

**Gates & bake.** Full protocol; owner pre-approval mandatory.

---

## 5. End-of-campaign confirmation

The per-step screens are supervised; the campaign closes with one self-play
confirmation of the accumulated result:

1. Train the final tree with the production KLENT recipe, stated explicitly
   as always: `--gamma 0.99 --lam 0.01 --lam-ret 0.939` (the flagless
   defaults remain paper-faithful and are not used).
2. Train the campaign-start tree (post-Step-0, pre-Step-1) under the
   identical recipe, budget, and seed policy.
3. `klent.headtohead` between the two resulting checkpoints: shared paired
   openings, seat-balanced, Gumbel search at the standard evaluation budget;
   report paired standard errors, sign test, and Elo difference. Optionally
   the SealBot curve for both.
4. The owner rules on the campaign outcome from that packet. The margin is
   predeclared in the packet *before* the matches run.

A supervised-screen win that fails self-play confirmation reopens the
relevant step verdicts rather than being argued away.

---

## 6. Decisions that must not be silently changed

- One model, modified in place; no separate selectable architectures.
- The speed gate is hard and per-step; kernel work ships inside the step.
- Every verdict is the owner's. Statistical significance is reported, never
  auto-acted-on; neutral results are presented, not deleted.
- Screens are paired: same tree, same seeds, same corpus stream; the knob is
  the only difference between arms.
- The campaign corpus is frozen once and reused; metrics are predeclared.
- Knobs are transient: one field per step, deleted at bake, recorded in
  `LEGACY_BAKED_KNOBS` on accept, path fully deleted on reject.
- Oracles and golden vectors are bumped and regenerated, never re-baselined
  to match the implementation.
- Class tables are generated from their defining transforms and asserted
  (93/189 binary-graft; 378/377, 2187/2184, 729, 48 if their steps land) —
  never hand-enumerated.
- Softmax key projections have no bias, in every stage present and future.
- Action rows, once accepted, cover **every** legal action; no background
  alias path returns.
- No absolute coordinates, fixed crops, or move-history features, in any
  step.
- The KLENT training recipe is always spelled out on the command line;
  defaults stay paper-faithful.
- Frontier steps (11–14) start only on an explicit owner decision with an
  honest cost projection in hand.
