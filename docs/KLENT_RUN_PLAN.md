# KLENT run plan — from here to a production run

**Status: operational plan.** `KLENT_DESIGN.md` describes the mechanism; this
document is the path to running it: the questions settled before the first
run, the shakeout that gates the first long run, the evaluation that is
deliberately deferred, and the ladder to production-ready. Items leave this
document as they complete — it is a plan, not a changelog.

---

## 1. Where things stand

The baseline is implemented and green (`python/mantisnet/README.md`): model,
KLENT loop, 54 tests on Windows and WSL2, ~3 s per steady-state iteration at
32 games on the 4070 Ti. The run driver (`mantisnet.klent.run`) persists what
a run is: `config.json` with every knob and version, `metrics.jsonl` with the
§13 metrics per iteration, and resumable checkpoints carrying model,
optimizer, and RNG state. No evaluation runs yet (§4), and nothing here is a
`ModelPackage` (§5, rung 6).

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

## 3. The shakeout run

Purpose: not strength — **stability and instrumentation**. One command,
~100 iterations, a few hours, gating the first long run.

```
uv run python -m mantisnet.klent.run --out runs/shakeout-1 \
    --iterations 100 --games 64 --checkpoint-every 10 --seed 1
```

(`--cap` defaults to the design's 512; the fitting batch defaults to 1024 —
see the probe below.)

Before it: **the batch-size probe.** The paper's fitting batch is 4096; at
Hexo's ~1,000 legal cells per sample that is ~4 M decoder rows per step and
untested against 12 GB. Run a few fit steps at `--batch 4096` and `2048` on
shakeout data; keep the largest that fits with headroom, record it here.

What to watch, per iteration in `metrics.jsonl`:

| Metric | Healthy | Alarming |
| --- | --- | --- |
| `f_seeded` | high from the start (near-terminal starts) | falling toward 0 — seeds too long or policy degrading |
| `f_unseeded` | 0 early; *any* sustained rise is real progress | — |
| `acting_norm_entropy` | drifting down from ~0.9 as Q sharpens | pinned ≥0.95 forever (Q never bites) or collapsed ≈0 while `v_hat_mae` stays bad (§9 noise-latching) |
| `acting_kl` | small and steady — updates are gradual | growing without bound |
| `v_hat_mae` / winner-vs-loser means | MAE falling; means separating (+/−) | means converging or inverted — the §9 bias, or a sign bug K1 would have caused (tests exclude the latter) |
| `p0_win_rate` | near the seeded games' natural split | pinned 0 or 1 |
| `first_stone_win_rate` | strictly between 0 and 1 | 0 or 1 exactly — K2's freeze path uncovered |
| `seconds` | flat after iteration ~2 | growing — recompile churn or a leak |
| `buffer_samples` | roughly `f · games · mean length` | 0 for many iterations — fitting starved |

Abort and diagnose on: OOM, any NaN loss, `seconds` trending up across tens
of iterations, or `f_seeded` collapsing. Afterwards: resume once from the
last checkpoint and confirm the metrics line continues where it stopped —
that exercises the crash path on purpose.

Wall-clock arithmetic for sizing runs, from measured steady state
(~3 ms/sample end to end): iteration cost ≈ `f · games · mean_length · 3 ms`
+ ~0.5 s overhead. 64 games at ~70-ply wins and `f ≈ 0.9` is ~13 s/iteration;
100 iterations ≈ 25 minutes. A first *real* run at 256 games is ~1 min per
iteration, ~1,400 iterations/day.

---

## 4. Evaluation — deferred, and what it will be

Nothing measures strength yet, on purpose: the shakeout's health metrics do
not need it, and bolting eval on later costs nothing (the pieces —
`play_match`, `argmax_choose`, the line builder as opponent — already exist
and are tested). When it lands, in the run driver as `--eval-every N`:

- **Anchor zero is the line builder at pinned noise** (an anchor whose
  randomness drifts is not an anchor); the first strong checkpoint replaces
  it as the frozen anchor the design doc §11 wants, never retrained.
- Agent plays `argmax π_θ`, no search; **seat balanced**; capped games score
  ½ and are reported separately, not folded in.
- Enough games per point for the interval to mean something (the paper used
  1024; scale to budget).
- Later, `KLENT_PROPOSALS.md` A7: the checkpoint cross-play matrix, as a
  diagnostic for cyclic forgetting — checkpoints already exist, so it is
  cheap when wanted.

---

## 5. The ladder to production-ready

In order; each rung is small and none blocks the one above it being useful.

1. **Eval in the driver** (§4), after the shakeout proves the loop stable.
2. **First real run** (256+ games/iteration, days of wall clock), seed
   annealing by hand from the metrics; automate the anneal only if manual
   proves tedious.
3. **Measured deviations from the baseline**, each gated on the baseline's
   own curves and staged per `KLENT_PROPOSALS.md` A4 (screen at ~10% budget,
   promote on improvement): the λ_intra split (A1), the Bernoulli
   win-probability critic (A2, the cheapest §9 mitigation), the
   normalized-entropy dual controller (A3, one dimension at fixed ρ).
4. **Artefact hygiene**: the value-head decision (§2.4), a checkpoint
   retention policy, and pruning `runs/` deliberately.
5. **Provenance**: per-game seeds minted from stable ids (the B4 shape) so a
   run reproduces, and the buffer's records written through `hexo-records`
   (B2's per-move blob) instead of living only in memory — at which point
   dropped-episode records become revisitable data, as design doc §12 notes.
6. **The `ModelPackage`**: encoder/evaluator over the hexo-search seam,
   manifest + probe hash, the container image of `CONTAINER_SPEC.md` — the
   point where this Python stops being a side tree and becomes the package
   the Rust container drives. The forward, builder, and checkpoints all
   survive unchanged; what is added is the seam plumbing.
7. **Test-time Gumbel MCTS** (design doc §15), once there is a checkpoint
   worth searching with — as a DAG, per K8, and outside every comparable
   number this plan produces before it.

---

## 6. Standing risks, named

- **§9 overestimation bias** is the predicted failure of this design at
  b ≈ 1000. Its instrument (v̂ calibration) is in every iteration's metrics;
  its cheapest mitigations are rung 3's A2 and a higher ρ.
- **Corpus conditioning** (§5.1 of the design): everything trained on is a
  game somebody won within the cap. Watched via `f` and game lengths; not
  patched.
- **Compile warmup** (~1 min per process) is paid once per run, not per
  iteration; a recompile that recurs *per iteration* is a bug — `seconds`
  in the metrics is the detector.
