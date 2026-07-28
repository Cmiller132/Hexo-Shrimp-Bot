# KLENT run plan — from here to a production run

**Status: operational plan.** `KLENT_DESIGN.md` describes the mechanism; this
document is the path to running it: the questions settled before the first
run, the shakeout that gates the first long run, the evaluation that is
deliberately deferred, and the ladder to production-ready. Items leave this
document as they complete — it is a plan, not a changelog.

---

## 1. Where things stand

The baseline is implemented and green (`python/mantisnet/README.md`): model,
KLENT loop, 55 tests on Windows and WSL2. The run driver (`mantisnet.klent.run`)
persists what a run is: `config.json` with every knob and version,
`metrics.jsonl` with the §13 metrics per iteration, and resumable checkpoints
carrying model, optimizer, and RNG state. Evaluation runs in the driver
(`--eval-every`, §4). Nothing here is a `ModelPackage` (§5, rung 5).

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

### 2.3 Seed annealing — manual, for now

`seed_fraction` and `seed_cut` are static config, adjusted by the operator
against the reported `f_seeded`/`f_unseeded` — the design doc's requirement
is that the annealing be *driven by the measured terminating fraction*, and
for the first runs a human reading `metrics.jsonl` is that driver. An
automated schedule is rung 2 of §5, earned only if manual proves tedious.

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
the Linux deploy target). The whole constraint is a transient: once Q learns
to finish the seeded lines, games shorten, positions shrink, and larger
batches become trivially affordable — if a later corpus ever justifies it,
re-probe rather than trust this table. Token-budget batching (chunk by
Σcells / padded T² cost rather than sample count) is the principled fix if
the transient ever needs to be fast.

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

---

## 4. Evaluation — in the driver, and what it still isn't

`--eval-every N` plays `argmax π_θ` (no search) against **anchor zero: the
line builder at pinned noise** (`ANCHOR_NOISE` in `klent/evaluate.py` —
pinned because an anchor whose randomness drifts is not an anchor). Seat
balanced; capped games score ½ and stay visible as `eval_capped`. The score
joins that iteration's `metrics.jsonl` row, and the eval RNG derives from
(run seed, iteration) rather than the training stream — a run's training
trajectory is bit-identical with evaluation on or off, tested.

Still future, deliberately:

- The first strong checkpoint replaces the line builder as the frozen
  anchor the design doc §11 wants, never retrained.
- Games per point scale with budget (the paper used 1024; `--eval-games`
  defaults to 64, a health signal rather than a rating).
- `KLENT_PROPOSALS.md` A7: the checkpoint cross-play matrix, as a
  diagnostic for cyclic forgetting — checkpoints already exist, so it is
  cheap when wanted.

---

## 5. The ladder to production-ready

In order; each rung is small and none blocks the one above it being useful.

1. **First real run** (256+ games/iteration, days of wall clock), seed
   annealing by hand from the metrics; automate the anneal only if manual
   proves tedious.
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
5. **The `ModelPackage`**: encoder/evaluator over the hexo-search seam,
   manifest + probe hash, the container image of `CONTAINER_SPEC.md` — the
   point where this Python stops being a side tree and becomes the package
   the Rust container drives. The forward, builder, and checkpoints all
   survive unchanged; what is added is the seam plumbing.
6. **Test-time Gumbel MCTS** (design doc §15), once there is a checkpoint
   worth searching with — as a DAG, per K8, and outside every comparable
   number this plan produces before it.

---

## 6. Standing risks, named

- **§9 overestimation bias** is the predicted failure of this design at
  b ≈ 1000. Its instrument (v̂ calibration) is in every iteration's metrics;
  its cheapest mitigations are rung 2's A2 and a higher ρ.
- **Corpus conditioning** (§5.1 of the design): everything trained on is a
  game somebody won within the cap. Watched via `f` and game lengths; not
  patched.
- **The iteration-0 transient** (§3): long drifting games, ballooned legal
  sets, quadratic attention memory. Survived by batch sizing today;
  token-budget batching is the named fix if it ever needs to be fast.
- **VRAM overruns fail slow on Windows** (driver spill to system RAM), and
  loud on Linux. A run that suddenly runs ~50× slower without erroring has
  overrun; peak-memory checks, not crash logs, are the detector.
- **Compile warmup** is paid once per machine, not per iteration (cold
  Windows-Triton compile of these graphs measured ~15 min; disk-cached
  thereafter). A recompile that recurs *per iteration* is a bug —
  `seconds` in the metrics is the detector.
