# KLENT run plan — from here to a production run

**Status: operational plan.** `KLENT_DESIGN.md` describes the mechanism; this
document is the path to running it: the questions settled before the first
run, the shakeout that gates the first long run, the evaluation that is
deliberately deferred, and the ladder to production-ready. Items leave this
document as they complete — it is a plan, not a changelog.

---

## 1. Where things stand

The baseline is implemented and green (`python/mantisnet/README.md`): model,
KLENT loop, 61 tests on Windows. The run driver (`mantisnet.klent.run`)
persists what a run is: `config.json` with every knob and version,
`metrics.jsonl` with the §13 metrics per iteration, and resumable checkpoints
carrying model, optimizer, and RNG state. Evaluation runs in the driver
(`--eval-every`, §4). Nothing here is a `ModelPackage` (§5, rung 5).

**2026-07-28, the faithfulness reset (owner decision).** The in-loop
machinery the first training nights accreted — line-builder seeded starts,
the warm-start heuristic phase, the f-driven seed-cut anneal, and SealBot
grounding inside collection — was removed whole, after checking the
implementation against the authors' reference code
([KazukiOhta/klent](https://github.com/KazukiOhta/klent)). Self-play runs
from the empty board with no curriculum; the cold-start bootstrap, when
needed, is a prefit on a foreign corpus before KLENT starts (design doc
§5.2); SealBot is the only evaluation, in-driver via `--eval-every`.
The same check landed two model-side fidelity fixes from the reference
`PQNet`: the Q head is tanh-bounded to `(−1, 1)`, and *both* decoder output
layers zero-initialize (the Q-side half of which the first training night
had already discovered empirically). The §3 history below records the
seeded/grounded era as measured; its mechanics are gone from the code.

---

## 2. Questions settled before the first run

### 2.1 The τ/λ transposition — settled, from the paper's text

Checked 2026-07-27 against arXiv 2602.10894v2, eq. 2 and §6.1: the paper
weights the **reverse KL by β = 0.1** and the **entropy by α = 0.03**. In this
repo's notation that is `τ = 0.1, λ = 0.03` — `KLENT_DESIGN.md` had the pair
transposed, exactly the hazard `KLENT_PROPOSALS.md` flagged. Design doc §1/§8
and `KlentConfig` now carry the corrected pair. Consequence worth restating:
the prior exponent is `τ/(τ+λ) = 0.77` — updates are more conservative than
the transposed reading implied, and §8's expectation becomes "Hexo may want
the ratio *at or above* 0.77", to be read off the KL/entropy diagnostics
rather than swept blindly.

### 2.2 λ_ret — decided: `e^{-1/16} ≈ 0.939`

The paper's `e^{-1/8}` is a horizon of 8 transitions, which in its games is 8
turns. Eight transitions in Hexo is 4 turns, so carrying 0.883 would halve
the strategic horizon while appearing faithful (`KLENT_PROPOSALS.md` A1's
correction). `KlentConfig.lam_ret` defaults to 0.939; the paper's literal
0.883 stays one flag away, and if the λ_intra split (A1) ever lands the two
knobs separate cleanly.

### 2.3 Seed annealing — removed with the seeding (2026-07-28)

Settled twice and then deleted: manual annealing gave way to the `--anneal`
walk, and the whole seeding apparatus left with the faithfulness reset (§1).
Kept here because §3's history reads against it.

### 2.4 The value head in checkpoints — deliberately open

KLENT never trains the §7 value head, so checkpoints carry its untouched
init. Harmless for training runs; whether the deployed artefact keeps those
tensors is a packaging question that arrives with the `ModelPackage` wiring
(§5, rung 6), not before.

---

## 3. The shakeout — ran 2026-07-27, `runs/shakeout-1`

Purpose was stability and instrumentation, not strength, and it delivered
both — including one real bug and one real dynamics finding. The command:

```
uv run python -m mantisnet.klent.run --out runs/shakeout-1 \
    --iterations 100 --games 64 --checkpoint-every 10 --seed 1 \
    --batch 256 --eval-every 10 --eval-games 64
```

**The batch-size probe ran (2026-07-27), and the paper's 4096 is out.** The
probe's premise — "~1,000 legal cells per sample" — was itself wrong at
iteration 0: an untrained π′ is near-uniform, the seeded near-win almost
never gets completed, and games drift for ~160–500 plies while the radius-8
frontier balloons. The measured iteration-0 corpus (64 games, seed 0) is
13.5 k samples at **~5,600 legal cells per sample** (max ~14 k), and fit
memory is dominated by the attention's materialised per-pair bias, quadratic
in stone count. On the 4070 Ti (12 GiB), against that corpus:

| `--batch` | s/step | peak alloc | verdict |
| --- | --- | --- | --- |
| 128 | 0.23 | 2.5 GiB | fits |
| **256** | 1.41 | 5.7 GiB | **chosen: fits with headroom** |
| 512 | 88 | 17.1 GiB | silent spill to system RAM, ~60× slower |
| 1024–4096 | — | 25 GiB+ | OOM even with spill |

Two hazards worth naming beyond the numbers. **Windows fails slow, not
loud:** the driver spills VRAM overflow into system RAM at PCIe speed
instead of raising OOM, so a too-big batch reads as a mysterious 20–60×
slowdown — watch peak memory, not just for crashes. And **the first process
on a machine pays a cold `torch.compile`** of these graphs (~15 min under
Windows Triton at these sizes; cached on disk thereafter, and much faster on
the Linux deploy target).

> **Superseded the same evening: memory is now budgeted, not probed for.**
> Token-budget packing landed (`KlentConfig.pair_budget` / `cell_budget`,
> README Performance): every network batch is packed under both measured
> memory axes, `--batch` is a per-step maximum, and the table above is the
> record of why. At the defaults, the same worst-case corpus peaks at
> 0.36 GiB in collection and ~2.9 GiB in fit — and the fit epoch runs ~2×
> faster than the packed-by-count 256 batch, because homogeneous-length
> chunks stop paying padding. FlexAttention was built and measured against
> this as an alternative (exactly equivalent, 5× slower here) and deleted;
> the README's "deliberately absent" table records the numbers.

**The machinery verdict: solid.** 100 iterations completed; `config.json`,
114 metrics rows, and ten checkpoints landed as designed. The crash-resume
path was exercised twice *on real wreckage* — both crashes were the same
bug: the policy-target sum-to-1 guard accusing its own fp32 accumulator on
flat-phase positions ~14 k cells wide (|sum−1| = 1.3e-4, zero NaN/Inf —
the guard's error message now reports deviation, width, and non-finite
counts, which is what settled the diagnosis). Fixed by accumulating the
check in f64; the final stretch ran the same regime clean. No leaks and no
recompile churn: `seconds` tracks *game length* (0.3 s sharp-phase, ~2 min
flat-phase), which supersedes the old watch-table reading of "growing
seconds = leak".

**The dynamics verdict: the loop trains, and the baseline is unstable at
the paper's knobs.** One seed, so observations rather than conclusions:

- The iteration-0 transient dies by iteration 4 (won lengths 177 → 60 → 20
  plies; the initial +1.8 Q bias corrected in one fit).
- Then an **entropy-breathing cycle** (`H/log|A|` 0.13 ↔ 0.92, ~10-iteration
  period): seeds give Q spread near wins → π′ sharpens → games shorten →
  Q values compress → the λ-entropy term flattens π′ *below* π_θ — where
  Q spread ≪ τ+λ, eq. 3 makes π′ ∝ π_θ^0.77, strictly flatter — and the
  loop drifts toward uniform. From iteration ~65 it sat in that **uniform
  fixed point** for long stretches: `H ≈ 0.91`, `KL(π′‖π_θ) ≈ 0.003` (no
  improvement signal left), policy CE ≈ ln|A|, `f` down to 0.64.
- Eval against the anchor peaked at **0.688** (iteration 39), then **0.000
  from iteration 69 onward** — even after collection re-sharpened and every
  collection metric looked healthy again. The deployed artefact is the
  policy head, and collection health does not measure it: **eval in the
  driver is not optional.**
- The §9 instrument shows no runaway overestimation (winner-side v̂ means
  stay ≤ ~0.4 after iteration 1) but weak discrimination all run:
  `v_hat_mae` never below ~0.7, winner/loser means barely separated. Q
  learns that a win is near, not whose.

What this buys rung 1 (§5): the baseline curves that gate every rung-2
deviation now exist, and they point at named knobs — λ down or τ up (§2.1's
"at or above 0.77" now has a mechanism behind it: the flattening operator),
A2's Bernoulli critic for Q discrimination, A3's entropy controller aimed
squarely at the breathing. Watch-table amendments for the next run, learned
here: `H` pinned ≥ 0.9 **with `KL ≈ 0`** is the uniform-fixed-point
signature and the strongest abort-and-retune signal there is; `seconds`
spikes mean long games, not leaks; `eval_score` at 0 while `f = 1.00` means
the policy head died, not the run.

Sizing note for real runs: collection cost is dominated by the *longest
game* (lockstep), so flat phases cost ~2 min/iteration at 64 games while
sharp phases cost 0.3 s. Size from the shakeout's own `seconds` column, not
per-sample arithmetic.

### The first training night (2026-07-28) — five runs, five mechanisms

The λ sweep the shakeout called for ran, collapsed, and each collapse was
chased to a mechanism with a fix landed the same night. In causal order:

1. **Init-noise poisoning.** KLENT exponentiates `Q/(τ+λ)`, so at
   framework-default init π′ is *sharpened noise* (KL ≈ 4 from the policy);
   the first fitting epoch trains π_θ onto it and seeded games stop
   terminating. Both sweep configs died of this at birth. Fix: the Q
   decoder's output layer initializes to zero (`MODEL_SPEC.md` appendix B,
   a measured requirement).
2. **The honest bootstrap starves.** With `Q ≡ 0`, π′ opens near-uniform
   and finishes ~3% of seeded games — the §5.2 seeding assumed π′ could
   finish cut games at birth, and that had only ever held by the init-noise
   accident. Fix: `--warm-iterations N` — collection acts through the line
   builder's scores (the same evaluator seam the pipeline tests use), games
   finish, both heads train on dense real outcomes.
3. **Bootstrap self-erasure.** At λ_ret = 0.939 a return on a longer game is
   ~94% bootstrap, and a young Q's v̂ ≈ 0 drags Q back to zero — the warm
   investment erased itself within two iterations of handover. Fix for now:
   pure Monte-Carlo returns (λ_ret = 1.0) and lr 2.5e-4; the λ-return
   returns when v̂ has earned trust. Warm must also actually converge:
   30 warm steps left Q-loss at 0.87 (no skill); 300 drove it to ~0.05,
   and the handover then held with Q separating winners +0.61 / losers
   −0.37 — the healthiest iteration of the night.
4. **Corpus conditioning, §5.1, realized.** With a static (1,8) cut the
   corpus parks on trivial near-terminal stubs: self-play metrics stay
   perfect while strength against a real opponent dies — checkpoint 350
   lost 63/64 to the anchor with *all three* heads while f sat at 1.00.
   Fix: `--anneal`, the f-driven cut walk §5.2 always required. Eval went
   0.000 → 0.695 in 24 annealed iterations. Known gap: f measures
   *termination*, not *competence* — deep-cut races still terminate, so
   the ceiling can outrun the student; a competence-gated anneal (eval- or
   length-gated) is the successor.
5. **Exploration collapse.** At λ = 0.01, once the cut saturated, π′ went
   near-deterministic (H ≈ 0.03–0.09): self-play locked into 11-ply mutual
   races that Q predicted almost perfectly (calibration MAE 0.04) and the
   anchor beat 9:1 — a self-consistent, objectively weak equilibrium, and
   the strongest argument yet for opponent grounding and A3. Fix: λ
   restored to 0.03 — safe *now* because Q's ±1 spread dwarfs τ+λ, so the
   old flattening pathology cannot recur. Eval climbed from 0.02 back into
   a stable 0.60–0.72 band and held it for 500+ iterations of pure
   self-play.

Also collected on the way: the sum-to-1 guard fired honestly a second time
(the fp32 softmax *denominator* leaves |sum−1| ≈ N·1e-8 at 10⁴-cell widths
— π′ is now stored f64-renormalized at the source), and `--starve-limit`
ended every dead configuration with a checkpoint instead of a burned night.

**The resolved recipe** (what `runs/overnight-3` converged to):
`--games 256 --warm-iterations 300 --lam 0.03 --lam-ret 1.0 --lr 2.5e-4
--seed-fraction 0.9 --anneal --starve-limit 6 --eval-every 25`. End state:
stable self-play training at ≈ the warm clone's strength (eval ~0.65,
peaks 0.82) — the loop *works*; exceeding the clone is what the next
builds are for.

### The first pure runs (2026-07-28, after the §1 reset)

**`runs/pure-1`** forked from the overnight-3 endpoint at the paper's knobs
(`--lam-ret 0.939 --lr 1e-3`, games 1024, no seeding, no grounding, SealBot
depth-1 eval every 25). Two findings:

- **Pure self-play needs no crutch from a trained start.** `f = 1.00` from
  iteration 0, and the eval climbed 0 → 0.016 → 0.062 → **0.125** (8/64 off
  depth-1 SealBot by iteration 124) — past anything the seeded/grounded era
  posted.
- **Mechanism 3 recurs even from a trained fork.** From ~iteration 130 the
  λ-return's bootstrap ate the critic: winner-side and loser-side v̂ both
  collapsed to ≈ 0 *while q_loss fell* — Q accurately fitting its own
  collapsed targets, the diagnostic signature — and the ρ = 0.77 prior
  exponent then flattened the policy wherever the now-flat Q gave it
  nothing (H breathing 0.2 ↔ 0.96, won lengths 19 → 200 plies, 600 s
  iterations, eval back to 0.016). At `λ_ret = 0.939` a target's grounding
  in the terminal decays as `0.939^k`, so on Hexo's game lengths the
  targets are almost pure bootstrap, and a weak critic (§9: `v̂_mae` never
  below ~0.7) makes that a self-erasing fixed point. Stopped at ~175.

**`runs/pure-2`** forks `pure-1/checkpoint_000100` (post-climb,
pre-collapse) changing exactly one flag: `--lam-ret 1.0`. Monte-Carlo
returns ground every ply in a real ±1 — the setting the only long stable
run used. The λ-return returns when the critic earns it; the critic build
that would earn it (dueling `Q = V + A`, or A2's Bernoulli critic — §9's
named mitigations) is the next staged deviation if pure-2 holds its climb.

### Conversion diffusion (2026-07-28/29) — a sixth mechanism, and the γ fix

pure-2 held its climb to eval **0.844** at iteration 199, then dissolved:
entropy ratcheted 0.33 → 0.56, won lengths 49 → 83 plies, iterations to
500 s+, eval declining — replicated identically from `checkpoint_000200`
on Windows and in the container, so a training dynamic, not environment.
The ply telemetry located it exactly: by iterations 235–239 the `t ≥ 100`
plies had `|v̂| = 0.91` with winner/loser separation ±0.86 — a
near-perfect critic — and π′ top-1 mass **0.106** over ~2,700 legal
cells. Both at once mean Q is *flat at ±0.9 across actions*: in a decided
position every move wins, eq. 3 degenerates to `π′ ∝ π_θ^0.77`, and the
winner learns to wander instead of converting. Longer games put more
decided plies in the corpus and the loop compounds. The trigger is the
critic *maturing* (q_loss 0.84 → 0.53 over exactly that window): while Q
was noisy the flatness was invisible. Root cause is objective-level: at
`γ = −1` per ply a 5-ply and a 300-ply win are the same return, and λ·H
actively rewards wandering — the reference objective is degenerate on a
game whose winner controls termination. Note A3 as specced (fixed ρ,
adapt T) cannot touch this: `π_θ^ρ` flattening at flat Q is
T-independent.

The fix is `--gamma`, a per-ply return-discount *magnitude* in the
λ-return (the mover-change sign stays in `signs`; 1.0 is the reference
objective). Three arms forked from `pure-2/checkpoint_000200`, judged
against the two crashed control replicates:

- **conv-disc** (γ = 0.99, λ = 0.03, 100 its): the runaway is arrested —
  won lengths 50–60, flat buffers, critic learns (q_loss 0.46 → 0.30),
  eval 0.609/0.625/**0.734**/0.578 — but decided plies still flatten to a
  *bounded* H ≈ 0.70 plateau: at k ~ 40 plies from the terminal the
  win-sooner spread `γᵏ(1 − γ^Δ)` ≈ 0.12 only ties `τ+λ = 0.13`.
- **conv-rho1** (τ = 0.13, λ = 0, stopped at 51): decided plies *sharpen*
  (top-1 0.64 → 0.73) — λ's exponent is definitively the diffusion
  mechanism — but the run stagnates whole: q_loss pinned at 0.88,
  KL → 0.006, eval flat, sharp deterministic wandering (won lengths
  58 → 87). γ, not ρ, is what feeds the critic: undiscounted ±1 targets
  on long balanced games teach it nothing.
- **conv-disc-lam01** (γ = 0.99, λ = 0.01 → ρ = 0.909, 50 its): decided
  sharpness *stable* (H 0.28 → 0.29, top-1 0.61 → 0.61), contested plies
  sharpen without rho1's over-concentration (0.54 → 0.66), q_loss falls,
  H settles at 0.20 with healthy KL ≈ 0.007, and evals **0.750 / 0.703**
  — the strongest of the night.

**Resolved recipe: `--gamma 0.99 --lam 0.01 --lam-ret 1.0`, τ = 0.1**;
`runs/conv-disc-lam01` resumed to `--iterations 2000` as the live run.
Watch items: won lengths creep mildly (55 → 71) — not runaway, but the
starve trajectory if it compounds; and λ = 0.01's documented collapse
mode, hedged here by γ's near-terminal Q spread (abort signature: H
under ~0.1 with short mutual races). Judging caveat: `v_hat_mae` and the
winner/loser v̂ means are defined against ±1, so they read differently
under γ — compare critics on the ply-bucket telemetry (decided =
`|v̂| ≥ 0.5`), not on those columns.

---

## 4. Evaluation — SealBot, in the driver

`--eval-every N` plays a 32-simulation Gumbel sequential-halving line search
in seat-balanced paired matches against **SealBot** (`--sealbot <root>`).
The in-loop opponent is full-strength iterative deepening: no depth cap and
0.1 s per move by default. `--eval-depth N` is an optional weaker rung;
`--eval-sims 0` is exact policy argmax. All three resolved settings land in
`config.json` and `invocations.jsonl`, because an anchor whose strength
drifts is not an anchor.

At the root the model samples
`m = min(16, simulations // 2, |A_legal|)` candidates by policy logit plus
Gumbel noise, then spends the budget extending deterministic lines. Interior
moves take the same `π′` argmax used for KLENT acting, leaf values are mapped
back to the root mover, and sequential halving drops the worse half. There
are no tree statistics or PUCT: hundreds of legal root cells make full-width
tiny-budget PUCT hopeless, and revisiting a deterministic path learns
nothing. This search exists only in evaluation; collection and fitting are
unchanged and the KLENT operator remains the only training-time improvement.

Capped games score ½ and stay visible as `eval_capped`. The score joins that
iteration's `metrics.jsonl` row, and the eval RNG derives from (run seed,
iteration) rather than the training stream — including the Gumbel draws — so
a resumed run replays the same match and training stays bit-identical with
evaluation on or off. The line-builder anchor ("anchor zero") is gone with
the seeding: a self-made heuristic anchor was measured to flatter exactly
the failure it existed to catch.

Scale note and adjacent diagnostic:

- Games per point scale with budget (the paper used 1024; `--eval-games`
  defaults to 64, a health signal rather than a rating).
- `mantisnet.klent.crossplay` provides the A7 checkpoint matrix as a
  diagnostic for cyclic forgetting; it is separate from the anchored
  opponent score.

### The external yardstick — SealBot (measured 2026-07-28)

`mantisnet.klent.sealbot` plays checkpoints against
[SealBot](https://github.com/Ramora0/SealBot), an independent C++
alpha-beta bot for this exact game (owner-supplied, machine-local checkout,
`--sealbot <root>`). It shares nothing with this repo — separate rules
implementation (asserted to agree with `hexo-engine` on every placement and
every winner), a hand-tuned 729-pattern eval, real search — which makes it
the first strength measurement that self-play conditioning cannot flatter.
Games run in seat-balanced pairs from shared uniform-random short openings,
since model-vs-searcher is otherwise near-deterministic. The generic match accepts
an opponent identity/config plus a batched chooser; SealBot's independent
rules oracle and 16-game memory cap live behind that adapter. A future
champion network needs one adapter and can immediately use both in-loop and
offline evaluation. Offline `--max-depth` caps SealBot for weaker rungs,
`--sims` defaults to zero for historical comparability, and `--run <dir>`
sweeps a checkpoint curve to `sealbot_curve.jsonl`.

What it measured, 64 games per point:

| Player | vs SealBot depth-1 | vs SealBot 0.1 s/turn | survival (plies) |
| --- | --- | --- | --- |
| line builder (the anchor) | 0/64 | 1/64 | 16 |
| overnight-3 it 250 (≈ warm clone) | 0/64 | — | 15 |
| overnight-3 it 2000–2062 | 0–2/64 | 0/64 | 22–24 |

These are historical pre-search, depth-1 measurements. Reading: the run's
eval-vs-anchor climb (0.65 → 0.87) is real — survival
against SealBot moved from anchor-level to clearly above it, and late
checkpoints steal the occasional game from both seats. But even a
*depth-1* SealBot (one turn of search plus mate-threat quiescence over its
pattern eval) wins every game in near-minimal time: the self-play
equilibrium races and does not defend, exactly the §3 mechanism-5 story at
external resolution. The gap to close is tactical defense, and the
yardstick for closing it now exists: **score against depth-1 SealBot is
the next headline metric**, with survival plies as the gradient while the
score sits at zero.

### Opponent grounding — landed and removed the same day (2026-07-28)

> Removed in the §1 faithfulness reset; kept as the record of what was
> measured. The `abl-gnd*` ablation arms were stopped unfinished (~iteration
> 277 of 1300); `runs/abl-gnd25` keeps its checkpoints and metrics.

`--ground-fraction F --sealbot <root> --ground-depth 1` seats a
depth-capped SealBot in one (alternating) side of `F` of each iteration's
games, unseeded. Only the model's plies are recorded; grounded returns are
pure Monte-Carlo (the λ-return's bootstrap chain breaks at unrecorded
opponent plies) and a capped grounded game is a **draw, g = 0**, not a K4
drop — against a real opponent, surviving to the cap is an outcome, and
the only gradient toward defense while wins are out of reach. The f stats
stay self-play-only (an external opponent terminates games regardless of
what the policy knows — mixing them in would flatter exactly what the
anneal walks on); grounded games report `f_grounded` and a per-iteration
`grounded_score` instead — a free strength reading against the yardstick
in every metrics row. Grounding stays off during warm; `--init-from`
forks arms from a shared parent checkpoint.

Measured at landing: grounding at depth 1 costs ~nothing (~1 ms/turn, one
shared engine); a 15-iteration shakeout forked from the overnight-3
endpoint moved `gnd` 0.00 → 0.05 with acting entropy rising 0.136 → 0.204
— the corpus getting harder in real time. Games-per-iteration probe:
throughput plateaus at ~3.2 k samples/s from 256 through 1024 (the loop
is orchestration-bound); 512 chosen — double the grounded games per
iteration of 256, twice the improvement rounds per hour of 1024.

The first grounding ablation (running as this is written): three arms
forked from `overnight-3/checkpoint_002062`, 1300 iterations,
`--ground-fraction` 0 / 0.25 / 0.5 (`runs/abl-gnd0`, `runs/abl-gnd25`,
`runs/abl-gnd50`), judged by the depth-1 SealBot curve over each arm's
checkpoints plus the anchor eval for regression. The winner's setting
carries into the next long run.

### The pipelined loop (landed the same afternoon)

A phase profile found the sequential loop's ~3.2 k samples/s ceiling was
CPU orchestration — half of it seed-prefix generation playing games one
at a time — with the GPU at ~6% duty. The loop was rewritten as the only
path: seed games play in lockstep cohorts, iteration ``i+1`` collects on
a worker thread against a weight snapshot while iteration ``i`` fits (the
corpus runs **one fit behind** the paper's strict alternation — the
recorded algorithmic cost of the overlap), sampling is vectorized per
chunk, and fit preps chunks one ahead with on-device loss accumulation.
Measured: **~6.3 k samples/s at games 1024 (2.3 s/iteration, GPU bursts
90–98%, ~5 GiB VRAM)** — 2× — with games 1024 the new operating point
(~1,540 improvement rounds/hour at double the corpus of the old 512).
The mantisnet README's Performance section records the two threading
hazards this depends on (the compile lock and the sequential first
iteration).

### The auto-reset cohort (2026-07-28, the drain-tail fix)

The pure runs exposed what the pipelined loop had left: iteration wall
clock tracked the *single longest game*, not the corpus. Same-size
iterations measured 2.7 s → 144 s (correlation of seconds with mean game
length: −0.02 — it was never the average), and a step-level trace showed
~30 % of collection wall clock spent with under 6 % of the cohort alive,
each near-empty step still paying full collate/launch/sync overhead.

The fix is design doc §16 item 15: `Collector` — persistent slots,
auto-reset on the spot, a completion quota (owner setting: **4096 finished
games per iteration**, toward the paper's ~2M-transition buffers), carry
of in-flight games across calls, and a three-lane pipeline inside each
step (collate worker → GPU → sampling worker). Measured at 1024 slots /
4096-game quota on the pure-2 checkpoint: **145 k samples in 41.8 s cold
(~30 s warm, ~4.9 k samples/s steady) vs the old loop's ~671 samples/s
production average — ~7×** — with the tail structurally gone and ~96 % of
wall clock now in the forward path. The forward is the next target; peak
collection VRAM measured at 0.26 GiB against 12, so the memory headroom
for that work is wide.

---

## 4a. Telemetry — the dashboard's substrate

`metrics.jsonl` answers "is the run healthy". It cannot answer "which
games", "which plies", "what did the policy believe there", or "was the
card the bottleneck" — every question §3's post-mortems actually needed and
had to reconstruct by hand. So every run now also writes
`runs/<name>/telemetry.db`: one SQLite file, WAL, one transaction per
iteration, always on. **It is the substrate a web dashboard will be built
on**; this pass is capture and query only, with no frontend.

What it holds is in the mantisnet README's Telemetry section. The four
decisions worth recording here, because they constrain what can be asked
later:

- **π′ is not stored, it is recomputed.** Per ply the database keeps five
  scalars (v̂, π′'s KL to π_θ, its normalized entropy, its maximum, and its
  value at the sampled move); `klent.inspect.inspect_position` reproduces
  the whole array from a checkpoint and the move prefix, through the
  training path's own loader and closed form. Storing π′ would be kilobytes
  a ply for something derivable exactly. This is what makes a policy
  debugger and a branch-and-play view possible without a storage decision
  in front of them.
- **Per-ply KL and entropy, not just their iteration means.** The §13 row
  reports averages; the *distributions* are what distinguish a policy that
  is uniformly uncertain from one that is confident everywhere except the
  positions that matter. Both are columns now, so both are queries.
- **The opponent is a row, not a column.** Nothing SealBot-specific is in
  the schema. An `opponents` row is a name plus its config, so uncapped
  SealBot at 0.1 s and a depth-capped rung are two opponents; a stronger
  engine — the intended eventual replacement — needs no schema change to get
  a curve.
- **Calibration and blunders are derivable, not precomputed.** The §9 bias
  instrument is a `GROUP BY` over `plies` joined to its game's realized
  outcome, bucketed by v̂, by ply, or by game length; a "blunder" is a v̂
  swing across consecutive plies read in the mover's own frame. Neither is
  a stored metric, so neither needs a rerun when the question changes.

Measured cost at the operating point (1024 games, ~30 k plies): ~72 ms an
iteration to write, ~1% of the loop, and ~78 bytes a ply — about 1.4 GB an
hour. **That growth rate is the standing decision to revisit**: a multi-day
run at this setting produces tens of GB, and the levers are dropping a
scalar column, quantizing the four bounded ones, or recording plies for a
sample of games rather than for all of them.

Per-iteration hardware columns come from a background sampler (NVML +
psutil, 1 s period, mean/max per iteration): GPU utilization, power,
temperature, VRAM in both the NVML and torch-allocator views, process
CPU/threads/RSS, host RAM. The throughput plateau of §4 was diagnosed with
an ad-hoc profile; the next one will be a query.

Deliberately not built here: the frontend, corpus-novelty tracking (the
branching factor makes position repeats too rare to be a signal), and any
cross-run join key (runs are independent files; comparing two is opening
two connections).

---

## 5. The ladder to production-ready

In order; each rung is small and none blocks the one above it being useful.

1. **First real run** (256+ games/iteration, days of wall clock), pure
   self-play from a trained checkpoint (`--init-from`), watched through the
   SealBot eval.
2. **Measured deviations from the baseline**, each gated on the baseline's
   own curves and staged per `KLENT_PROPOSALS.md` A4 (screen at ~10% budget,
   promote on improvement): the λ_intra split (A1), the Bernoulli
   win-probability critic (A2, the cheapest §9 mitigation), the
   normalized-entropy dual controller (A3, one dimension at fixed ρ).
3. **Artefact hygiene**: the value-head decision (§2.4), a checkpoint
   retention policy, and pruning `runs/` deliberately.
4. **Provenance**: per-game seeds minted from stable ids (the B4 shape) so a
   run reproduces, and the buffer's records written through `hexo-records`
   (B2's per-move blob) instead of living only in memory — at which point
   dropped-episode records become revisitable data, as design doc §12 notes.
5. **Everything into Docker** (owner intent, stated 2026-07-28; pulled
   forward and **landed the same day** as part of the performance push):
   training, self-play, evaluation — the whole loop, not just serving —
   runs in a Linux container as its permanent home. `docker/` at the repo
   root is the environment (deps-only image, repo bind-mounted;
   `docker/README.md` is the workflow). Measured: the container runs
   collection **1.44× faster than Windows-native on identical code**
   (5.0 k vs 3.5 k samples/s) — the Linux compiler stack is the gain —
   and it is where the Triton kernel work happens. Still open inside
   this rung: a Linux build of SealBot's `minimax_cpp` so evals run
   in-container, and moving live runs there when the owner relaunches.
6. **The `ModelPackage`**: encoder/evaluator over the hexo-search seam,
   manifest + probe hash, the container image of `CONTAINER_SPEC.md` — the
   point where this Python stops being a side tree and becomes the package
   the Rust container drives. The forward, builder, and checkpoints all
   survive unchanged; what is added is the seam plumbing.
7. **Evaluation search and the opponent seam — landed 2026-07-28.**
   Gumbel root sampling plus sequential halving spends 32 simulations on
   deterministic lines; full-strength SealBot is the in-loop default, depth
   caps and zero-search remain offline rungs, and a future champion network
   supplies one chooser adapter rather than another match loop.

---

## 6. Standing risks, named

- **§9 overestimation bias** is the predicted failure of this design at
  b ≈ 1000. Its instrument (v̂ calibration) is in every iteration's metrics;
  its cheapest mitigations are rung 2's A2 and a higher ρ.
- **Corpus conditioning** (§5.1 of the design): everything trained on is a
  game somebody won within the cap. Watched via `f` and game lengths; not
  patched.
- **The iteration-0 transient** (§3): long drifting games, ballooned legal
  sets, quadratic attention memory. Bounded by the memory budgets
  (`pair_budget` / `cell_budget`) — a corpus can now cost time, never VRAM.
- **VRAM overruns fail slow on Windows** (driver spill to system RAM), and
  loud on Linux. A run that suddenly runs ~50× slower without erroring has
  overrun; peak-memory checks, not crash logs, are the detector. The
  budgets exist to keep runs out of that regime entirely.
- **Compile warmup** is paid once per machine, not per iteration (cold
  Windows-Triton compile of these graphs measured ~15 min; disk-cached
  thereafter). A recompile that recurs *per iteration* is a bug —
  `seconds` in the metrics is the detector.
