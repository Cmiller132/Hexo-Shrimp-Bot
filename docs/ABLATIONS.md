# Ablation and experimental record

This file is the technical record of the retained KLENT training runs and engineering experiments. It records configurations, measurements, artifact limits, and dispositions. It does not rank runs or prescribe a next experiment.

The comparison reference used below is: \(\gamma=0.99\), \(\lambda=0.01\), \(\tau=0.1\), \(\lambda_{\mathrm{ret}}=0.939\), and a scalar tanh critic; 4096 completed games per iteration, 1024 environments, ply cap 512, and learning rate \(10^{-3}\). Unless a run configuration says otherwise, the recorded in-driver evaluations below are 64 seat-balanced games against uncapped SealBot at 0.1 s/move, using the 32-simulation Gumbel line search.

One such evaluation costs 69.6 s on an RTX 4070 Ti, measured on `joint-939/checkpoint_000050.pt` at seed 0 over 64 games averaging 45.3 plies, none capped. Repeating it at zero simulations — raw-policy argmax, 45.6 mean plies — costs 57.2 s, so SealBot's own turn budget sets a floor near 57 s and the 32-simulation line search adds about 12 s above it. The pair is a cost control and not a strength comparison; it locates an evaluation's wall time, which is mostly not in the model.

The current driver can evaluate against SealBot, one independent §3.1 subprocess seat, or both. `--eval-games` is a per-opponent count: when both are configured, each receives the same seat-balanced opening and model-RNG schedule derived from the run seed and completed iteration, rather than splitting the requested games. Metrics and telemetry attribute a separate result to each opponent and its strength-defining configuration.

Runs whose last artifact predates 2026-07-29 — everything up to and including `pure-2`, plus the two `.aborted-guard` directories — live under `runs/archive/<name>/`. The move is the whole of the archiving: nothing is deleted, every artifact this record cites is still readable at that path, and the deck lists only direct children of `runs/` that hold a `config.json`, so an archived run stops appearing without being removed.

Training metric iteration numbers are the zero-based values stored in `metrics.jsonl` and `iterations`. `eval_matches.iteration` is the completed-iteration count, so an evaluation on metrics row 24 is stored as evaluation 25. Legacy results are labeled “metrics row”; database results are labeled “@”. `H` means `acting_norm_entropy`, or the corresponding quantized per-ply `norm_entropy` after division by 10000. “Decided” means \(|\hat v| \ge 0.5\).

## Training runs

### Archived exploratory runs (2026-07-27 - 2026-07-28)

Eleven runs preceding the gamma-conversion arms live under `runs/archive/<name>/`
with their full artifacts. All belong to the historical gamma = 1.0 seeded /
grounded era and every configuration in them is superseded; this record keeps
one line per run and the artifacts remain the source of detail.

| Run | Question | Disposition |
| --- | --- | --- |
| `shakeout-1` | Loop stability, crash/resume, instrumentation | Completed 100 iterations; superseded by the packed-batch sweeps |
| `sweep-a` | Packed-batch lambda = .03 baseline | Stopped at iteration 10 with an empty buffer |
| `sweep2-a` | lambda = .03 arm at 256 games | Stopped at iteration 3; corpus loss |
| `sweep2-b` | lambda = .01 arm at 256 games | Stopped at iteration 9; the high-initial-Q / empty-buffer sequence persisted |
| `abl-zeroq-lam01` | Zero-initialized Q output | Removed the initial Q/KL spike; starvation stop at iteration 7 |
| `overnight-1` | 15-iteration warm phase | Post-handoff collapse by iteration 23 |
| `overnight-2` | 30-iteration warm phase, Monte-Carlo returns | Post-handoff collapse by iteration 39 |
| `overnight-3` | 300-iteration warm; static vs annealed seeding; both lambda regimes | 2,062 iterations across four branches; checkpoint 2062 is the later fork parent |
| `abl-gnd25` | Depth-1 SealBot seated in 25% of collection | Stopped at 277 of 1,300; opponent grounding removed as a training input |
| `pure-1` | Unseeded, ungrounded self-play from the overnight-3 fork | Stopped at 174 |
| `pure-2` | Reproduction after a checkpoint-100 fork; the reference operating point | Reached 239; checkpoint 200 is the common fork for the gamma-conversion arms |

The measured through-line: gamma = 1.0 with seeding produced a recurring
high-initial-Q then empty-buffer collapse; longer warm phases delayed it
without removing it; opponent grounding did not change it and was removed;
and the pure-self-play forks from `overnight-3` were the first stable
configurations. `pure-2` checkpoint 200 is the ancestor of every retained
arm below.

### `conv-disc`

| Field | Record |
| --- | --- |
| Dates | 2026-07-28–2026-07-29 local; 2026-07-29 03:40–05:50 UTC |
| Init/fork | `runs/pure-2/checkpoint_000200.pt`; seed 21 |
| Delta from reference | `--lam .03 --lam-ret 1.0`; all reference operating-point and evaluator flags present; 100 iterations |
| Question | Effect of changing \(\gamma\) from 1.0 to 0.99 while retaining \(\lambda=.03\) and \(\lambda_{\mathrm{ret}}=1.0\) |
| Disposition | Completed 100 iterations; \(\gamma=.99\) retained for subsequent arms; superseded by `conv-disc-lam01` |

| Completed iteration | Score | Wilson 95% CI | Elo 95% CI | Seat wins |
| ---: | ---: | --- | --- | --- |
| 25 | 39/64 = 0.609375 | 0.486917–0.719446 | 77.250 [-9.093, 163.593] | 22/17 |
| 50 | 40/64 = 0.625 | 0.502502–0.733342 | 88.739 [1.738, 175.741] | 23/17 |
| 75 | 47/64 = 0.734375 | 0.615169–0.827038 | 176.660 [81.490, 271.829] | 28/19 |
| 100 | 37/64 = 0.578125 | 0.456098–0.691304 | 54.735 [-30.585, 140.055] | 19/18 |

All four `eval_matches` rows contain 64 games and no caps or forfeits.

| Iteration range | Measurements |
| --- | --- |
| 0 → 74 → 99 | q-loss 0.455834 → 0.295219 → 0.318616 |
| 0–99 | Mean game length 41.13–60.35; H 0.334573 → 0.454542, maximum 0.546314 |
| Decided ply bucket, 0 → 49 → 99 | H 0.334511 → 0.704091 → 0.729728; top-1 probability 0.553107 → 0.243057 → 0.232766 |
| Decided bucket, 95–99 aggregate | H 0.713595; top-1 probability 0.247647 |

The run plan’s scale comparison \(\gamma^k\lambda_{\mathrm{ret}}\approx0.12\) versus \(\tau+\lambda=0.13\) is **(run plan, not re-derived)**.

### `conv-rho1`

| Field | Record |
| --- | --- |
| Date | 2026-07-29 UTC |
| Init/fork | `runs/pure-2/checkpoint_000200.pt`; seed 21 |
| Delta from reference | `--gamma 1.0 --lam 0 --tau .13 --lam-ret 1.0`; \(\tau+\lambda=.13\); all reference operating-point and evaluator flags present |
| Question | Effect of removing entropy regularization’s flattening term while holding \(\tau+\lambda=.13\) |
| Disposition | Stopped after completed iteration 50; `status.json` records iteration 51 incomplete in the next collection; shelved |

| Completed iteration | Score | Wilson 95% CI | Elo 95% CI | Seat wins |
| ---: | ---: | --- | --- | --- |
| 25 | 37/64 = 0.578125 | 0.456098–0.691304 | 54.735 [-30.585, 140.055] | 21/16 |
| 50 | 40/64 = 0.625 | 0.502502–0.733342 | 88.739 [1.738, 175.741] | 24/16 |

Both `eval_matches` rows contain 64 games and zero forfeits.

| Iteration range | Measurements |
| --- | --- |
| 0 → 50 | H 0.292128 → 0.108288; KL 0.038430 → 0.008324, with minimum 0.006236; q-loss 0.879319 → 0.890106, range 0.868207–0.911336; mean length 58.09 → 80.98, maximum 87.14; iteration time 93.68 → 245.09 s, maximum 327.18 s |
| Ply buckets, rows 0–4 → 45–50 | Decided top-1 0.619346 → 0.730996 and H 0.279468 → 0.197477; undecided top-1 0.550077 → 0.770883 and H 0.254379 → 0.105704 |

The run plan’s top-1 summary “0.64 → 0.73” **(run plan, not re-derived)** uses an unstated aggregation window; the explicit artifact aggregates above are used here.

### `conv-disc-lam01`

| Field | Record |
| --- | --- |
| Date | 2026-07-29, 09:00–13:23 UTC |
| Init/fork | `runs/pure-2/checkpoint_000200.pt`; seed 21; later resumed from this run’s checkpoint 50 |
| Delta from reference | `--lam-ret 1.0`; otherwise the reference recipe and operating point. Initial invocation: `--iterations 50 --checkpoint-every 25`; resume at checkpoint 50: `--iterations 2000 --checkpoint-every 25`. |
| Question | Effect of \(\lambda=.01\) under \(\gamma=.99\) while retaining \(\lambda_{\mathrm{ret}}=1.0\) |
| Pre-stated abort signature | Acting H below approximately 0.1 together with short races **(run plan, not re-derived)** |
| Disposition | Abort signature not observed; stopped after checkpoint 151, with the next collection incomplete; superseded by `lam-ret-939` |

| Completed iteration | Score | Wilson 95% CI | Elo 95% CI | Seat wins | Provenance |
| ---: | ---: | --- | --- | --- | --- |
| 25 | 48/64 = 0.75 | 0.631835–0.839852 | 190.849 [93.824, 287.873] | 29/19 | `eval_matches` |
| 50 | 45/64 = 0.703125 | — | — | — | metrics only; resume cleanup removed the match row |
| 75 | 52/64 = 0.8125 | 0.700254–0.889355 | 254.729 [147.401, 362.057] | 28/24 | `eval_matches` |
| 100 | 50/64 = 0.78125 | 0.665670–0.864978 | 221.137 [119.633, 322.640] | 23/27 | `eval_matches` |
| 125 | 48/64 = 0.75 | 0.631835–0.839852 | 190.849 [93.824, 287.873] | 24/24 | `eval_matches` |
| 150 | 48/64 = 0.75 | 0.631835–0.839852 | 190.849 [93.824, 287.873] | 22/26 | `eval_matches` |

Every retained database match row has zero forfeits.

| Iteration range | Measurements |
| --- | --- |
| Initial 0 → 49 | H 0.299091 → 0.197119; KL 0.030841 → 0.007220; q-loss 0.442303 → 0.374989; mean length 54.80 → 70.85 |
| Ply buckets, rows 0–4 → 45–49 | Decided H 0.291828 → 0.287828 and top-1 0.603515 → 0.613259; undecided top-1 0.523668 → 0.659543 and H 0.277218 → 0.179858 |
| Continued 50–150 | H median 0.208104, range 0.171031–0.240448; q-loss 0.404108 → 0.358927, minimum 0.331376; mean length range 56.49–77.46 |
| Six evaluations | Arithmetic mean 0.7578125 |

### `lam-ret-939`

| Field | Record |
| --- | --- |
| Date | 2026-07-29 |
| Init/fork | `runs/conv-disc-lam01/checkpoint_000151.pt`; seed 21 |
| Delta from reference | None; 50 iterations |
| Question | Effect of changing \(\lambda_{\mathrm{ret}}\) from 1.0 to 0.939 from the checkpoint-151 fork |
| Disposition | Completed 50 iterations; adopted as the scalar reference recipe |

| Completed iteration | Score | Wilson 95% CI | Elo 95% CI | Seat wins |
| ---: | ---: | --- | --- | --- |
| 25 | 52/64 = 0.8125 | 0.700254–0.889355 | 254.729 [147.401, 362.057] | 29/23 |
| 50 | 56/64 = 0.875 | 0.772252–0.935278 | 338.039 [212.122, 463.957] | 29/27 |

Both `eval_matches` rows contain 64 games and zero forfeits.

| Iteration range | Measurements |
| --- | --- |
| 0 → 49 | H 0.202198 → 0.237155, full range 0.192162–0.271776; q-loss 0.100018 → 0.083034, minimum 0.075008 |
| 1–5 versus 44–49 | Mean game-length ranges 63.98–65.45 versus 64.87–73.71 |
| Decided plies per game | Parent `conv-disc-lam01` rows 145–150: 12.5129; this run rows 0–4: 12.7165; rows 45–49: 12.3632; halves 0–24 and 25–49: 12.3714 and 11.8836; row 40: 11.0727 |

The run-plan description “H stable at 0.20–0.26 throughout” **(run plan, not re-derived)** is approximate; the artifact extrema are 0.192162 and 0.271776.

The run-plan summary “decided plies/game 12.4 → 11.0” has no matching natural start/end aggregation in the retained telemetry. Its endpoints are **(run plan, not re-derived)**; the explicit queries above show the available aggregations.

### `factored-939`

| Field | Record |
| --- | --- |
| Date | 2026-07-29 |
| Init/fork | `runs/grafts/ckpt151-factored.pt`; seed 21; later resumes from checkpoints 50 and 53 |
| Delta from reference | `--critic-factors 2`; factored sign/magnitude critic instead of the scalar tanh critic. Initial invocation used `--iterations 50`; resumes at checkpoints 50 and 53 used `--iterations 100`; all used `--checkpoint-every 25`. |
| Question | Measurements from replacing the scalar critic with \(p=\operatorname{sigmoid}(p_\ell)\), \(m=\operatorname{sigmoid}(m_\ell)\), \(Q=(2p-1)m\), trained with sign and magnitude binary cross-entropy terms |
| Disposition | Stopped after completed iteration 78, with the next collection incomplete; superseded by `factored-939-s2`; factored critic subsequently shelved |

| Completed iteration | Score | Wilson 95% CI | Elo 95% CI | Seat wins | Provenance |
| ---: | ---: | --- | --- | --- | --- |
| 25 | 52/64 = 0.8125 | 0.700254–0.889355 | 254.729 [147.401, 362.057] | 30/22 | `eval_matches` |
| 50 | 52/64 = 0.8125 | — | — | — | metrics only; resume cleanup removed the match row |
| 75 | 35/64 = 0.546875 | 0.425734–0.662707 | 32.668 [-51.990, 117.326] | 20/15 | `eval_matches` |

The retained database match rows have zero forfeits.

| Iteration range | Measurements |
| --- | --- |
| 1 → 24 → 49 → 74 | H 0.224310 → 0.280948 → 0.313316 → 0.447296; maximum 0.459299 |
| 1–24, 25–49, 53–78 | Median mean game lengths 52.37, 55.92, and 56.05; the final window’s maximum was 62.03. Median iteration times were 52.84, 76.78, and 81.94 s. |
| Scalar sibling timing | `lam-ret-939` first/second-half median iteration times 116.15/137.58 s |
| Undecided ply bucket, rows 4 → 26 → 74 → 78 | H/top-1 `0.220646/0.620199` → `0.307274/0.527330` → `0.448330/0.411954` → `0.424660/0.441843` |
| Full telemetry | Undecided fraction 0.791535; decided top-1 0.595958, versus 0.590929 for the scalar sibling |

The run-plan H summary “0.309 → 0.422” **(run plan, not re-derived)** uses unstated rolling windows. The artifact H series is not monotone: it contains 37 row-to-row decreases, including 0.459299 at row 72 → 0.351021 at row 73, contrary to the run-plan word “monotonically.”

The run-plan decided-position top-1 comparison, 0.623 versus scalar 0.596 **(run plan, not re-derived)**, differs from the full-run telemetry aggregates 0.595958 and 0.590929; no aggregation window is stated. The run-plan statement that Q-action spread was approximately 0.04 is **(run plan, not re-derived)**.

### `factored-939-s2`

| Field | Record |
| --- | --- |
| Date | 2026-07-29 |
| Init/fork | `grafts/ckpt151-factored.pt`; seed 21 |
| Delta from reference | `--q-scale 2.0` and the factored critic; otherwise the reference recipe; `--iterations 50 --checkpoint-every 25` |
| Question | Whether doubling the factored critic’s Q contribution matched the scalar sibling’s evaluation and acting-entropy measurements |
| Pre-stated kill criterion | Match 0.8125/0.875 with acting H stable at approximately 0.20–0.26 |
| Disposition | Criterion missed; stopped 2026-07-29 via `STOP` sentinel; shelved |

Manager-measured run report, 2026-07-29:

> factored-939-s2 (q_scale=2.0, fork `grafts/ckpt151-factored.pt`, seed 21, otherwise reference recipe): eval @25 = 0.625 (40/64, Wilson CI 0.503–0.733, Elo 89 [2, 176], seats 20/20, no forfeits). Kill criterion stated in advance in KLENT_RUN_PLAN §3 — match 0.8125/0.875 with acting H stable ≈0.20–0.26 — was missed. Acting H/log|A| climbed 0.208 (it 1) → 0.312 (it 24) → 0.381 (it 38); run stopped 2026-07-29 at ~iteration 38 via STOP sentinel.

Manager-measured ply-bucket decomposition:

> Ply-bucket decomposition (telemetry, decided = |v̂| ≥ 0.5): factored-939 undecided-bucket H 0.221 (it 4) → 0.307 (it 26), vs factored-939-s2 0.186 → 0.351 over the same window; the s2 decided bucket sharpened to H 0.155–0.177 / top-1 0.73–0.76 and held there.

The artifact boundary contains metrics rows 0–39 and checkpoint 40; `status.json` records iteration 40 incomplete in the next collection. The stop was issued at approximately iteration 38 and observed after in-flight work.

`runs/grafts/` contains only `ckpt151-factored.pt`; it has no config, metrics, status, telemetry, or transform manifest. Run-plan prose and downstream invocations identify it as the checkpoint-151 factored graft. It is not a training run.

### Critic ranking-stability probe

Manager-measured setup, verbatim: “Critic ranking-stability probe (64 contested positions, |v̂| < 0.2, ply 20–60, sampled from factored-939 iterations 48–52; plausible set = policy top-16 per position; Spearman per position, medians):”

| Comparison | Manager-measured result, verbatim |
| --- | --- |
| ckpt50 vs ckpt53 | Q 0.894 (q25 0.793), π 0.937; Q-top1 agreement 0.73, π-top1 0.81 |
| ckpt75 vs ckpt79 | Q 0.879 (q25 0.775), π 0.953; Q-top1 agreement 0.72 |
| ckpt50 vs ckpt75 | Q 0.800, π 0.844; Q-top1 agreement 0.56 |
| Spread | σ(Q) over the top-16, median per checkpoint: 0.020–0.032 (ckpts 50/53/75/79); full-legal max−min median 0.28–0.38. |

Measured reading: contested-position Q rankings are stable across 3–4 iterations (comparable to the policy head's own stability), and measured σ(Q) in contested positions is 0.02–0.03 against operator temperature τ+λ = 0.11.

### `brm-939`

| Field | Record |
| --- | --- |
| Date | 2026-07-30, 01:50–03:41 UTC |
| Init/fork | `runs/grafts/ckpt151-brm.pt`, the function-preserving graft of `runs/conv-disc-lam01/checkpoint_000151.pt`; seed 21 |
| Delta from reference | The bipolar return-mass critic: two logits per legal cell, $u^{\pm}=\sigma(z^{\pm})$, $Q=u^{+}-u^{-}$, trained with taken-action $(Q-G)^2$ plus $\tfrac{\eta}{2}$ times the soft-target binary cross-entropies of $z^{+}$ against $\max(G,0)$ and $z^{-}$ against $\max(-G,0)$, $\eta=0.25$. Otherwise the reference recipe; 50 iterations. Branch `brm-critic`. |
| Question | Whether storing positive and negative return mass, rather than a marginal sign times a marginal magnitude, matches the scalar critic's evaluation |
| Pre-stated abort signature | Evaluation at or below 0.625 at iteration 25, or acting $H/\log\lvert A\rvert$ leaving $[0.12,0.36]$ on a monotone trend |
| Disposition | Completed 50 iterations; abort signature not observed |

| Completed iteration | Score | Wilson 95% | Elo 95% | Seat scores | Capped |
| ---: | ---: | --- | --- | --- | ---: |
| 25 | 56/64 = 0.875 | 0.772252–0.935278 | 338.039 [212.122, 463.957] | 31/25 | 0 |
| 50 | 56/64 = 0.875 | 0.772252–0.935278 | 338.039 [212.122, 463.957] | 29/27 | 0 |

Both rows contain 64 games and zero forfeits.

| Iteration range | Measurements |
| --- | --- |
| 0 → first five → last five | H 0.20344 → 0.19872 → 0.23129, full range 0.19297–0.23799; q-loss 0.10175 → 0.09601 → 0.07289, minimum 0.06863; mean won length 55.74 → 59.74 → 78.97, maximum 83.56 |
| Acting KL | First five 0.00775, last five 0.00576, minimum 0.00397, maximum 0.07339 at iteration 32 |
| `mass_loss`, the unweighted binary-cross-entropy pair | 0.64005 at iteration 14 → 0.57394 at iteration 49 |
| Ply buckets, iterations 45–49 | Undecided H 0.2069 and top-1 0.6265 over 1,370,610 plies; decided H 0.3694 and top-1 0.5414 over 248,929 plies |
| Cost | 1.83 h of iteration time over 50 iterations |

The acting-KL maximum at iteration 32 is an excursion of one iteration, 8.3 times the scalar reference's 50-iteration maximum of 0.00884, coincident with that run's highest q-loss after iteration 0 (0.09991), its highest `mass_loss` (0.67389), and a fall in mean won length from 77.11 to 52.90. It decayed over four iterations (0.07339 → 0.02292 → 0.01830 → 0.00986 → 0.00803). A manager hypothesis that unbounded binary-cross-entropy logit growth drove it is contradicted by the head measurements below. No cause is established.

`checkpoint_000023.pt` exists because a manager placed the `CHECKPOINT` sentinel at that boundary for the head probe; it is not a scheduled artifact.

### `duel-939`

| Field | Record |
| --- | --- |
| Date | 2026-07-30, 03:49–07:35 UTC |
| Init/fork | `runs/grafts/ckpt151-duel.pt`, the order-preserving graft of the same checkpoint; seed 21 |
| Delta from reference | The dueling critic: $Q=\tanh\big(v(S)+A(S,a)-\sum_b\mathrm{sg}[\pi_\theta(b\mid S)]A(S,b)\big)$, with $A$ the existing legal-cell decoder score and $v$ a new per-position readout over the global token. Loss, coefficients, and metric keys unchanged; 50 iterations. Branch `dueling-critic`. |
| Question | Whether moving the position's value level into a per-position readout, leaving the legal-cell decoder to score advantages, improves evaluation |
| Pre-stated abort signature | As `brm-939` |
| Disposition | Completed 50 iterations; abort signature not observed |

| Completed iteration | Score | Wilson 95% | Elo 95% | Seat scores | Capped |
| ---: | ---: | --- | --- | --- | ---: |
| 25 | 52/64 = 0.8125 | 0.700254–0.889355 | 254.729 [147.401, 362.057] | 26/26 | 0 |
| 50 | 53/64 = 0.828125 | 0.717949–0.901175 | 273.216 [162.108, 384.324] | 29/24 | 0 |

| Iteration range | Measurements |
| --- | --- |
| Iteration 0 | Acting KL 0.10440 and `v_hat_mae` 1.02507, against 0.00868 and 0.79204 for `brm-939` at the same iteration. Decided-bucket KL over iterations 0–4 was 0.2052, against 0.0078 and 0.0058 for the other two arms. |
| 0 → first five → last five | H 0.16494 → 0.16833 → 0.29613, maximum 0.30520; q-loss 0.06269 → 0.07617 → 0.06315, minimum 0.06133; mean won length 55.32 → 60.61 → 91.27, maximum 96.59 |
| Recovery of the reset level | `v_hat_mae` 1.02507 → 0.91385 (first five) → 0.85992 (last five); acting KL 0.10440 → 0.00447 |
| Ply buckets, iterations 45–49 | Undecided H 0.2891 and top-1 0.5349; decided H 0.3310 and top-1 0.5600 |
| Cost | 3.75 h of iteration time over 50 iterations; samples per second 2514 (first five) → 781 (last five) |

This arm recorded the lowest q-loss of the four configurations, 0.06133, and the lowest evaluation total. Its acting entropy passed the scalar reference's 50-iteration maximum of 0.25374 at iteration 15 and ended at 0.30520, and its mean won length ended about 20 plies above the reference's range.

### `tail-939`

| Field | Record |
| --- | --- |
| Date | 2026-07-30, 07:46 UTC onward |
| Init/fork | `runs/grafts/ckpt151-tail.pt`, the exactly identity graft of the same checkpoint; seed 21 |
| Delta from reference | The private critic tail: a pre-norm residual FFN over the window rows and the global token, read only by the action-value head, so the two cell heads no longer share the incidence aggregation. `--cell-budget 500000 --collect-cell-budget 1600000`, memory only. Otherwise the reference recipe. Branch `critic-tail`. |
| Question | Whether critic-private trunk-side features improve evaluation |
| Pre-stated abort signature | As `brm-939` |
| Disposition | Completed 50 iterations; continued in place toward 150 |

| Completed iteration | Score | Wilson 95% | Elo 95% | Seat scores | Provenance |
| ---: | ---: | --- | --- | --- | --- |
| 25 | 53/64 = 0.828125 | 0.717949–0.901175 | 273.216 [162.108, 384.324] | 29/24 | `eval_matches` |
| 50 | 57/64 = 0.890625 | — | — | — | metrics only; the continuation's replay removed the database row |

| Iteration range | Measurements |
| --- | --- |
| 0 → first five → last five of 0–49 | H 0.20327 → 0.20316 → 0.24850, maximum 0.26224; q-loss 0.09997 → 0.09192 → 0.07586, minimum 0.07277; mean won length 55.82 → 62.01 → 75.24 |
| Acting KL, 0–49 | Maximum 0.00886, last five 0.00571; no excursion |
| Ply buckets, iterations 45–49 | Undecided H 0.2320 and top-1 0.6057; decided H 0.3377 and top-1 0.5682 |
| Cost | 1.73 h over iterations 0–49; iteration time 176.77 s over the last five against 190.29 s for `brm-939`, and samples per second 1770 against 1721 |

The second incidence pass is not measurable as a doubling of cost at these budgets: this arm's iteration time and throughput over the same window are within 8% of a single-pass arm's.

### The three arms against the scalar reference

The window is iterations 0–49 from `conv-disc-lam01/checkpoint_000151.pt` at the reference recipe with seed 21. `lam-ret-939` iterations 0–49 are that fork under the scalar critic, so its rows are the control.

| Configuration | Iteration 25 | Iteration 50 | Wins of 128 | Minimum q-loss | Final H | Won length, last five |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `lam-ret-939` (scalar) | 0.8125 | 0.875 | 108 | 0.07501 | 0.23716 | 70.9 |
| `brm-939` | 0.875 | 0.875 | 112 | 0.06863 | 0.23129 | 78.97 |
| `tail-939` | 0.828125 | 0.890625 | 110 | 0.07277 | 0.24850 | 75.24 |
| `duel-939` | 0.8125 | 0.828125 | 105 | 0.06133 | 0.29613 | 91.27 |
| `joint-939` | 0.78125 | 0.765625 | 99 | 0.06877 | 0.24413 | 74.42 |

Each evaluation is 64 games. At the observed rates one standard error on a 128-game total is about four games, so a difference between two arms carries about 5.8. Every arm is within 1.6 of those standard errors of the control — `brm` at 0.7, `tail` at 0.3, `duel` at 0.5, `joint` at 1.6 — so this evaluation separates none of the five configurations, and no ordering may be read from it. The two arms that lowered q-loss most, `duel-939` and `brm-939`, do not order the same way on evaluation. The paired head-to-head below is the instrument that does separate them.

| Configuration | Undecided H | Undecided top-1 | Decided H | Decided top-1 |
| --- | ---: | ---: | ---: | ---: |
| `lam-ret-939` | 0.2312 | 0.6035 | 0.3123 | 0.5922 |
| `brm-939` | 0.2069 | 0.6265 | 0.3694 | 0.5414 |
| `tail-939` | 0.2320 | 0.6057 | 0.3377 | 0.5682 |
| `duel-939` | 0.2891 | 0.5349 | 0.3310 | 0.5600 |

Iterations 45–49, decided at $\lvert\hat v\rvert\ge0.5$. The undecided bucket holds about five times the ply mass of the decided one in every run.

### Paired head-to-head against the control

Each arm's iteration-50 checkpoint against `lam-ret-939`'s iteration-50 checkpoint, converted into that arm's architecture by that arm's own graft. 64 shared openings of 2–6 plies, each played from both seats, so 128 games; both seats search at 32 simulations with $\tau=0.1$, $\lambda=0.01$; ply cap 512, seed 21. The pair is the unit: $d=(\text{A's wins in the pair})-1\in\{-1,0,+1\}$, and the reported interval comes from $\operatorname{sd}(d)/\sqrt{K}$ rather than from the marginal score.

| Arm | Score of 128 | Wilson | Pairs A / split / B | Sign test | Elo | Mean plies |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `joint-939` | 84.0 = 65.6% | 57.0–73.3% | 26 / 32 / 6 | 0.0005 | **+112** (+55, +177) | 72.8 |
| `tail-939` | 73.0 = 57.0% | 48.4–65.3% | 19 / 35 / 10 | 0.136 | +49 (−8, +109) | 74.7 |
| `brm-939` | 49.0 = 38.3% | 30.3–46.9% | 10 / 29 / 25 | 0.0167 | **−83** (−150, −21) | 74.6 |
| `duel-939` | 128.0 = 100% | 97.1–100% | 64 / 0 / 0 | — | not reported | 20.4 |

No game reached the ply cap in any match.

This instrument separates what the anchored evaluation could not. `joint-939` beats the control and `brm-939` loses to it, both beyond the sign test's 0.05, while their anchored totals sit 1.6 and 0.7 standard errors from the control's — that is, the anchored evaluation has no opinion about either, so the two measurements do not conflict. `brm-939` is the load-bearing case for reading these at all: an artifact that flattered whichever arm was newer would have flattered it too, and it measures negative.

What the matches measure is strength against one same-lineage sibling, not strength in general. Both models in every pairing descend from `conv-disc-lam01/checkpoint_000151.pt` and differ by 50 iterations under one changed term.

`duel-939`'s result is not a strength measurement and is excluded. Its graft is the one conversion that is order-preserving rather than function-preserving, so its opponent is the control with its value level removed rather than the control. Its games ended in 20.4 plies against 72.8–74.7 for the other three, and all 64 pairs returned the same $d$, leaving the paired variance unestimable; the harness reported no Elo interval and no standard-error ratio for it.

No better graft exists, and the dueling decomposition is the reason. The head computes $Q(s,a)=\tanh\big(A(s,a)+b(s)-\sum_{a'}\pi_\theta(a'\mid s)A(s,a')\big)$. Carrying a scalar parent's pre-tanh $z$ into $A$ is exact and is what the graft does; preservation then requires $b(s)=\mathbb{E}_{\pi}[z(s,\cdot)]$. But $b$ is `mlp_qbase(g)`, a readout over the global token alone, while the quantity it would have to equal is a function of the decoder's per-cell outputs and of the policy over them. No assignment of weights to that head produces it, so the graft sets $b=0$ and $Q$ becomes $\tanh(z-\mathbb{E}_\pi[z])$ — the recorded level removal. Separating level from advantage is this arm's whole hypothesis, so the obstruction is the architecture's and not the conversion's.

`duel-939` therefore cannot be measured against any converted control, and needs the genuine scalar control on the other side of the board: two architectures in one match, which one process holding one model cannot do. `CONTAINER_SPEC.md` §3.1 specifies the seat protocol that admits it, and this arm is blocked on that implementation rather than on more evaluation games.

Pairing on shared openings and seats gained little over independent games: the ratio of unpaired to paired standard error was 1.055, 1.059, and 0.976 for `joint`, `tail`, and `brm`, against 1.24 in simulation. At $d$'s observed dispersion the pairing removes almost no variance, so the design's value here is the sign test over decisive pairs rather than a narrower interval.

### Graft manifests

Each arm's conversion of `conv-disc-lam01/checkpoint_000151.pt` wrote a manifest beside its checkpoint in `runs/grafts/`, measured on that arm's own seeded probe set at $\tau=0.1$, $\lambda=0.01$.

| Arm | Stated property | Measured |
| --- | --- | --- |
| `brm` | Action values and $\pi'$ are the parent's, since $\sigma(2z)-\sigma(-2z)=\tanh z$ | max $\lvert\Delta Q\rvert$ 1.1920929e-07, mean 2.6443e-08; mean $D_{\mathrm{KL}}(\pi'_{\text{new}}\Vert\pi'_{\text{parent}})$ 1.844e-14, maximum 1.610e-13; median top-16 σ(Q) 0.065138791 against the parent's 0.065138798 |
| `duel` | Every position's ordering of its legal cells is the parent's; the level is reset | 135,939,358 comparable pairs, 0 discordant, rank agreement 1.0; 185 shared parameters carried bitwise; $v(S)=0$ exactly. max $\lvert\Delta Q\rvert$ 1.4836, mean 0.1596; median removed level −0.006225; median top-16 σ(Q) 0.069721 → 0.072019; mean improved-policy KL 0.001490, maximum 0.024504 |
| `tail` | Every parent tensor is the source file's bit for bit and the tail is the identity | max $\lvert\Delta Q\rvert$ 0.0, mean 0.0; readout input max $\lvert\Delta\rvert$ 0.0; KL 0.0 mean and maximum; 185 shared tensors unchanged under one SHA-256; median top-16 σ(Q) 0.063669 unchanged |
| `joint-brm` | The joint-class row replication preserves both cell decoders, and $\sigma(2z)-\sigma(-2z)=\tanh z$ preserves the scalar parent's action values | 181 shared tensors bit-identical; 186 replicated rows checked; §6 decode bitwise equal; `q_max_abs_delta` 6.5565e-07 against tolerance 1e-5; `max_abs_dq` 5.9605e-07 against 1e-5; mean improved-policy KL 2.9768e-12 against 1e-6 |

The composed conversion transforms disjoint tensors: `e_pw` and `e_qw` for
the joint decoder, versus `mlp_q.out` for the bipolar readout. Each arm's
detector therefore still covers its own side, and both run on the composed
output. BRM alone measured max abs dQ 1.19e-7; the composed graft measured
5.96e-7. The increase is the head GEMM's fp32 reassociation as its K grows
from 3 to 93, exactly the behavior predicted by the graft module's docstring.

Manager-measured on a shared 96-position set drawn from checkpoint 151's own self-play and replayed against every model, at τ+λ = 0.110:

| Model | Contested σ(Q) top-16 | Decided σ(Q) | Mean $\lvert Q\rvert$ | fp32 ties in top-16 |
| --- | ---: | ---: | ---: | ---: |
| `conv-disc-lam01` ckpt151 | 0.0623 | 0.0526 | 0.8540 | 0.0 |
| `lam-ret-939` ckpt50 | 0.0582 | 0.0383 | 0.8731 | 0.5 |
| `brm-939` ckpt23 | 0.0575 | 0.0286 | 0.7226 | 0.0 |
| `brm-939` ckpt50 | 0.0464 | 0.0324 | 0.7184 | 0.0 |
| `ckpt151-duel.pt` at the graft | 0.2216 | 0.1700 | 0.8854 | 0.0 |

The bipolar head's own quantities over the same set: $u^{+}+u^{-}$, its estimate of $\mathbb E\lvert G\rvert$, moved from exactly 1.0 everywhere at the graft to 0.275 over the policy's top-16 in contested positions and 0.955 in decided ones, against $(\gamma\lambda_{\mathrm{ret}})^k=0.9296^k$. $\lvert z^{+}+z^{-}\rvert$ moved 0.00 → 3.61 → 4.30 and $\operatorname{corr}(z^{+},z^{-})$ −1.000 → −0.960 → −0.903, so the two logits are not each other's negation after training. No probe cell at any checkpoint had both sigmoids saturated, and none reached $\lvert Q\rvert=1$.

### `joint-939`

| Field | Record |
| --- | --- |
| Date | 2026-07-30, 08:25–10:49 EDT |
| Init/fork | `runs/grafts/ckpt151-joint.pt`, the exact function-preserving graft of `conv-disc-lam01/checkpoint_000151.pt`; seed 21 |
| Delta from reference | The decoder's class is joint in the window's occupancy mask and the candidate cell's own slot, folded by a reversal acting on both halves — 93 classes where the slot class alone gave 3 (§4.3 joint decode). Both cell heads' class tables grow to 93 rows, +23,040 parameters; the stone incidence keeps the slot class. `MODEL_REPR_VERSION` 1 → 2. `--cell-budget 450000 --collect-cell-budget 1350000`, memory only: the aggregate row widens from $H+16$ to $H+128$, so the budgets come down by that same 1.78 to hold decoder memory at the reference's level. Otherwise the reference recipe. Branch `joint-slot-decoder`. |
| Question | Whether removing the decoder's action aliases improves evaluation, and which head uses the separation |
| Pre-stated abort signature | As `brm-939`: evaluation at or below 0.625 at iteration 25, or H outside [0.12, 0.36] on a monotone trend |
| Disposition | Ran iterations 0–49 and stopped at its configured limit. No abort signature fired: evaluation was 0.78125 at 25 against a floor of 0.625, and H stayed in [0.12, 0.36]. |

Pre-registered before iteration 0, against `lam-ret-939` iterations 0–49 as the control:

| Quantity | Control | Predicted | Measured |
| --- | ---: | --- | ---: |
| Wins of 128 over the two evaluations | 108 | 104–116 — not separable at this evaluation budget | 99 |
| Minimum q-loss | 0.07501 | 0.068–0.073 | 0.06877 |
| Undecided-bucket top-1 | 0.6035 | 0.61–0.63 | — |
| Iteration time | — | +5% to +15% from the wider head GEMM | — |

The anchored evaluation fell below the predicted band, by 1.6 standard errors of the difference. The prediction's own stated basis was that this evaluation cannot separate the arms, and it does not: the interval is consistent with no change. It is the only arm whose two evaluations declined across the window, 0.78125 to 0.765625, where the other four rose or held.

Against the same control, the paired head-to-head below measures `joint-939` as the strongest of the four arms at 128 games, which the anchored evaluation neither shows nor contradicts.

The reasoning, so a wrong prediction is diagnostic rather than merely wrong: adjacency is already
recoverable from window multiplicity, which the aliasing does not touch. A candidate adjacent to a
lone stone shares five live windows with it; one four cells away shares two. The model therefore
reconstructs contact from how many windows a cell shares with a stone, and the joint class makes
that direct rather than newly possible. The exact ties it removes are thin — about 1.5 pairs per
position, usually between two low-probability halo moves. So the expectation is a real but small
gain, concentrated in the critic, where per-move contact geometry is what an action value needs.

What the old key merged, measured over 2,800 positions from uniformly-random legal playouts before the change:

| Quantity | Measurement |
| --- | ---: |
| `(mask, slot)` pairs a live window and an empty candidate can form | 186 |
| Their orbits under the joint reversal | 93 |
| Classes `(canonical mask, slot class)` realizes | 75, merging 18 orbit pairs |
| Decoder entries lying in a merged class | 6,092,396 of 7,723,536 = 78.9% |
| Pairs of legal moves sharing one decoder row, hence one logit and one $Q$ | 4,321, every one separated by the joint key |
| Positions holding at least one such pair | 1,605 of 2,800 = 57% |

The merged classes are exactly the mirrored slots of a non-palindromic mask, so all 18 involve masks of one to three stones — the common case. An instance: a window holding one stone at slot 0, with one legal cell at slot 1 and another at slot 4, neither in any other live window. Under the slot class those two cells' decoder rows are equal, so no weights can rank a contiguous extension above a split one. The aliasing is exact, not approximate.

| Graft | Stated property | Measured |
| --- | --- | --- |
| `joint` | Each of the 93 rows is bit for bit the slot-class row it replaces, so the grafted model is the parent as a function | §6 decode over the grafted tables and joint classes is bitwise equal to the parent's over its own tables and slot classes, both heads; 183 shared tensors unchanged under one SHA-256; 186 expanded rows checked; median top-16 σ(Q) 0.0636686347424984 on both sides. The folded path's deltas are fp32 reassociation from the wider head GEMM: max $\lvert\Delta Q\rvert$ 6.557e-07, max $\lvert\Delta\text{logit}\rvert$ 1.526e-05, max improved-policy KL 2.994e-06 — against 1.5e-06, 1.6e-06, and 1.8e-06 for one unmodified model decoded both ways. |

Adam's moments for the two tables are replicated by the same rows as their weights, so each new row inherits the ratio its parent row had rather than taking one bias-corrected cold step.

### `joint-brm-939`

| Field | Record |
| --- | --- |
| Init/fork | `runs/grafts/ckpt151-joint-brm.pt`, the function-preserving graft of `conv-disc-lam01/checkpoint_000151.pt`; seed 21 |
| Delta from reference | The joint-class decoder and bipolar return-mass critic together. `--cell-budget 450000 --collect-cell-budget 1350000`, the joint arm's reduced budgets for the widened decoder aggregate row; otherwise the reference recipe. Branch `joint-brm`. |
| Target and resumes | 500 iterations, originally 150 and extended in place by `STOP` plus `--resume` at 111 and again at 253 |
| Question | Measurements from composing the joint decoder class and bipolar return-mass critic in one model |
| Disposition | **In flight**, currently past iteration 270; no concluded disposition |

In-driver SealBot evaluations, 64 games each, with zero capped games and zero
forfeits throughout:

| Completed iteration | Score | Capped | Forfeits |
| ---: | ---: | ---: | ---: |
| 25 | 53/64 = 0.828125 | 0 | 0 |
| 50 | 54/64 = 0.84375 | 0 | 0 |
| 75 | 62/64 = 0.96875 | 0 | 0 |
| 100 | 58/64 = 0.90625 | 0 | 0 |
| 125 | 56/64 = 0.875 | 0 | 0 |
| 150 | 63/64 = 0.984375 | 0 | 0 |
| 175 | 58/64 = 0.90625 | 0 | 0 |
| 200 | 60/64 = 0.9375 | 0 | 0 |
| 225 | 57/64 = 0.890625 | 0 | 0 |
| 250 | 64/64 = 1.000 | 0 | 0 |

Over the four evaluations the control `lam-ret-939` also ran, at completed
iterations 25/50/75/100, its scores were 0.8125, 0.875, 0.9219, and 0.9219.
The control won 226 of 256 games and `joint-brm-939` won 227 of 256: one game
apart. This separates nothing, consistent with the anchored evaluation's
inability above to distinguish the reference and ablation arms.

### Later runs without full entries

Six retained runs postdate this record's last full training-run entry. Their
configurations, telemetry databases, and eval curves live in their run
directories; they are listed here so the record states what exists.

| Run | What it is |
| --- | --- |
| `tri-939` | Critic-arm follow-on from the checkpoint-151 lineage |
| `joint-mnorm` | Joint-decoder arm under mass-normalized acting |
| `deep6-mnorm150` | Depth-6 window variant probe |
| `stack-939` | The pre-campaign production reference: the full reference-recipe run (seed 21, 483 iterations) whose strength curve is the baseline the campaign compares against |
| `newmodeltest` | KLENT successor carrying cell latents (the campaign's cell-latents arm C), paused at iteration 31 |
| `cellnodes-1` | The live KLENT successor carrying cell latents plus cell nodes at scope `all`; the graft-campaign carry run |

## Engineering experiments

### Fused Triton block attention

| Field | Record |
| --- | --- |
| Setup | RTX 4070 Ti; compiled BF16; fixed 64×64 tiles, four warps, three stages; distance-bucket bias and the live-prefix mask applied inside the kernel |
| Question | Runtime and allocation change from fusing block attention’s score, distance bias, masking, softmax, and value accumulation |
| Disposition | Retained for supported collection shapes. Dense SDPA remains for CPU, unsupported/failed shapes, and fit-time recomputation. Further custom attention kernels are deliberately absent. |

| Stones | Cohort | Dense (ms) | Fused (ms) | Ratio |
| ---: | ---: | ---: | ---: | ---: |
| 50 | 256 | 33.1 | 32.9 | 1.01× |
| 50 | 1024 | 136.0 | 134.0 | 1.01× |
| 200 | 256 | 112.9 | 111.1 | 1.02× |
| 200 | 1024 | 455.1 | 445.4 | 1.02× |
| 400 | 256 | 214.6 | 206.4 | 1.04× |
| 400 | 1024 | 860.1 | 822.8 | 1.05× |

At 400 stones, peak allocated memory changed from 2.32 GiB to 2.05 GiB. The 1.4× whole-forward latency target was not met; the largest measured ratio was 1.05×.

| End-to-end measurement | Dense | Fused | Reading |
| --- | --- | --- | --- |
| Collection, cold | 41.8 s; 3.48k samples/s | 59.88 s; 2.42k samples/s | Cold fused run was slower |
| Collection, warm | — | 17.55 s; 8.16k samples/s; 15.76 s in network execution | Warm result |
| Fit, different sampled corpora | 8,215 samples / 0.86 s = 9,569 samples/s | 7,784 / 0.82 s = 9,487 samples/s | Normalized throughput -0.9% |

### FlexAttention distance bias

| Field | Record |
| --- | --- |
| Setup | Distance-bucket bias expressed as a FlexAttention `score_mod`; dynamic lengths required padding to 128 and eager block-mask construction |
| Correctness | Maximum output difference approximately \(2\times10^{-6}\); maximum gradient difference approximately \(2\times10^{-8}\) |
| Performance | Approximately 5× slower in fit and 2.7× slower in collection; peak memory reduction at most 0.2 GiB |
| Disposition | Removed; the branch and generated cache were deleted |

### Shared decoder aggregation and Triton segment reduction

| Field | Record |
| --- | --- |
| Setup | Both cell heads share one incidence aggregation, with projection moved after aggregation; a single-warp Triton segment reduction operates on cell-sorted decoder entries. `inc_window` is sorted for this path, while `inc_stone` remains separate. |
| Question | Cost of repeated per-cell decoder aggregation and the gather/scatter component of that cost |
| Disposition | Retained |

| Stones | Cohort | Before (ms) | Shared/reduced (ms) | Ratio |
| ---: | ---: | ---: | ---: | ---: |
| 50 | 256 | 44.5 | 24.5 | 1.82× |
| 50 | 1024 | 142.0 | 100.5 | 1.41× |
| 200 | 256 | 116.6 | 83.1 | 1.40× |
| 200 | 1024 | 468.4 | 335.3 | 1.40× |
| 400 | 256 | 217.3 | 156.7 | 1.39× |
| 400 | 1024 | 871.3 | 624.8 | 1.39× |

Peak allocation changed from 3.04 to 1.96 GiB at 50 stones/cohort 1024, and from 2.07 to 1.66 GiB at 200 stones/cohort 256. Sharing only the index tensors, before the segment kernel, measured 1.12–1.15×.

| Stones | Gather/scatter before | Gather/scatter after | Total before | Total after |
| ---: | --- | --- | ---: | ---: |
| 50 | 9.51 ms, 29.1% | 0.62 ms, 2.8% | 32.7 ms | 22.1 ms |
| 400 | 49.2 ms, 22.9% | 3.04 ms, 2.0% | 214.9 ms | 154.4 ms |

Fit throughput changed from 32,864 samples / 4.18 s = 7,871 samples/s to 33,043 / 3.54 s = 9,346 samples/s, a normalized ratio of 1.19×. Collection changed from 15.61 s / 9.3k samples/s to 14.75–15.25 s / 9.5–9.8k samples/s. After the change, the trunk accounted for 18.8–19% of the measured profile.

### Loop pipelining: collection/fit overlap

| Field | Record |
| --- | --- |
| Baseline | Approximately 3.2k samples/s and 6% GPU utilization before overlap **(run plan, not re-derived)** |
| Setup | Collection for iteration \(i+1\) overlaps fit for iteration \(i\); training is one fit behind collection. Sampling uses CDF/searchsorted and prefetch. The first iteration remains sequential. |
| Warm operating point | Approximately 6.3k samples/s at the 1,024-game operating point, 2.3 s collection cadence, 90–98% GPU utilization, approximately 5 GiB resident allocation, approximately 2× throughput, and approximately 1,540 rounds/hour **(run plan, not re-derived)** |
| Additional measurement | Thread-local compilation increased startup from seconds to minutes and introduced a compilation lock. Cold reservation reached 18.9 GiB with 6.6 GiB allocated; the README records approximately 5 GiB after warm-up. |
| Disposition | Retained |

### Auto-reset cohort collector

| Field | Record |
| --- | --- |
| Baseline | Same-size synchronous cohorts ranged from 2.7 to 144 s; cohort duration/game-length correlation was -0.02 **(run plan, not re-derived)**; 30% of wall time had fewer than 6% of slots active; measured throughput was 671 samples/s |
| Setup | 1,024 persistent slots collect a 4,096-game quota with carry-over and reset a finished slot immediately. Three lanes pipeline the collate worker, GPU, and sampling worker. |
| Initial measurement | 145k samples in 41.8 s cold and approximately 30 s warm, about 4.9k samples/s; approximately 7× the baseline; 96% of time in forward execution; 0.26 GiB collector allocation **(run plan, not re-derived)** |
| Current fused-path measurement | 144k samples in 59.88 s cold and 17.55 s warm: 2.42k and 8.16k samples/s |
| Disposition | Retained |

### VRAM budget packing

The initial batch-size probe below is **(run plan, not re-derived)**.

| `--batch` | Time/step | Observed memory/result |
| ---: | ---: | --- |
| 128 | 0.23 s | 2.5 GiB |
| 256 | 1.41 s | 5.7 GiB |
| 512 | 88 s | 17.1 GiB and host spill |
| 1,024–4,096 | — | More than 25 GiB / out of memory |

| Field | Record |
| --- | --- |
| Setup | Separate pair/cell memory budgets and a batch maximum; fit batches sorted by length; collection retains game order |
| Long-state measurement | At approximately 5,500 legal cells per sample near ply 500, collection used 0.36 GiB and fit used approximately 2.9 GiB; throughput was approximately 2× the unpacked path |
| Host-spill measurement | Windows host spill was approximately 50× slower |
| Disposition | Retained |

### Telemetry schema, quantization, and browse-order indexes

| Field | Record |
| --- | --- |
| Baseline writer | Approximately 72 ms/iteration and approximately 1% overhead; approximately 78 bytes/ply and approximately 1.4 GB/hour **(run plan, not re-derived)** |
| Quantization setup | Schema v2 introduced five scalar ply columns stored as integers in units of \(10^{-4}\); readers divide by 10,000. The former REAL values accounted for 40 of 71 row bytes. |
| Quantization measurements | Maximum rounding error \(5\times10^{-5}\); a short synthetic workload at mean length 11 measured approximately 65 bytes/ply |
| Schema v3 | An iteration now evaluates against every configured opponent, so its score is not one number. The `iterations` table no longer carries `eval_score`, `eval_capped`, or `eval_games`, which were a single-opponent projection of the per-opponent data already carried by `eval_matches` rows. |
| Compatibility decision | There is no v2 converter. Schema-v2 databases written before the v3 bump are not readable by this build; that was deliberate, not an oversight. |
| Index benchmark | On 575,342 games, a warm lookup changed from 484 ms to 0.05 ms; a deck-order browse changed from 44 s to milliseconds; four indexes added approximately 56 MB |
| Writer/read behavior | Writers add indexes idempotently without changing the schema version; read-only browsing does not create them |
| Disposition | Schema v3, quantization, and browse-order indexes retained |

### Container-side training over the deck bind mount

| Field | Record |
| --- | --- |
| Finding | Training must run container-side while the deck container is up. A Windows-side driver fails because the deck holds `status.json` open across the bind mount and Windows `os.replace` then returns a permission error; Linux can rename over an open file. |
| Disposition | Run the training driver in the Linux container. |

### Opponent grounding and SealBot anchored evaluation

#### Opponent grounding

| Field | Record |
| --- | --- |
| Setup | `--ground-fraction F` selected games for a depth-1 SealBot opponent and alternated the model’s seat. Only model decision plies entered training; returns were pure Monte Carlo across unrecorded opponent plies; capped games were scored as draws; the remainder of the cohort stayed self-play. |
| Landing probe | Grounded score 0.00 → 0.05, H 0.136 → 0.204, approximately 1 ms per opponent turn over 15 iterations **(run plan, not re-derived)** |
| Longer run | `abl-gnd25` reached metrics row 277 of 1,300. Its grounded score and critic measurements are reported in the training-run section. |
| Disposition | Landed, then removed on 2026-07-28 |

#### SealBot as anchored evaluator

| Field | Record |
| --- | --- |
| Setup | Independent C++ SealBot evaluator; 64 seat-balanced games with shared openings. The later driver default is uncapped SealBot at 0.1 s/move with the 32-simulation Gumbel line search. |
| Early line-builder checks | Depth 1: 0/64; uncapped: 1/64; mean survival approximately 16 plies **(run plan, not re-derived)** |
| Disposition | Retained as an anchored evaluator; opponent grounding was not retained as a training input |

Offline `overnight-3` checkpoint curve, depth-1 SealBot at 0.1 s/move using the historical pre-search evaluator:

| Checkpoint | Score | Wilson 95% CI | Elo 95% CI | Seat wins | Mean plies |
| ---: | ---: | --- | --- | --- | ---: |
| 250 | 0/64 = 0 | 0–0.056626 | \(-\infty\) [\(-\infty\), -488.667] | 0/0 | 15.3125 |
| 500 | 1/64 = 0.015625 | 0.002763–0.083343 | -719.736 [-1022.937, -416.535] | 0/1 | 17.59375 |
| 750 | 0/64 = 0 | 0–0.056626 | \(-\infty\) [\(-\infty\), -488.667] | 0/0 | 20.546875 |
| 1000 | 0/64 = 0 | 0–0.056626 | \(-\infty\) [\(-\infty\), -488.667] | 0/0 | 15.796875 |
| 1250 | 0/64 = 0 | 0–0.056626 | \(-\infty\) [\(-\infty\), -488.667] | 0/0 | 14.875 |
| 1500 | 0/64 = 0 | 0–0.056626 | \(-\infty\) [\(-\infty\), -488.667] | 0/0 | 18.25 |
| 1750 | 0/64 = 0 | 0–0.056626 | \(-\infty\) [\(-\infty\), -488.667] | 0/0 | 23.6875 |
| 2000 | 2/64 = 0.03125 | 0.008612–0.106975 | -596.545 [-824.457, -368.633] | 1/1 | 22.375 |
| 2062 | 0/64 = 0 | 0–0.056626 | \(-\infty\) [\(-\infty\), -488.667] | 0/0 | 23.3125 |

At the later scalar-reference endpoint, `lam-ret-939` measured 56/64 = 0.875, Wilson CI 0.772252–0.935278, Elo 338.039 [212.122, 463.957].

The `joint-brm-939` curve above is noise against a ceiling after roughly
iteration 75 and reaches a perfect 64/64 at iteration 250. SealBot therefore
stopped measuring improvement well before the run ended. That saturation is
the engineering finding that motivated a second anchor.

#### Strix as a second anchored evaluator

The second anchor is a hexo-strix checkpoint served as a native §3.1
subprocess seat. Its reported identity is name `strix-seat`, digest
`0x351ed562065bed55`, variant `mcts:sims=32`. Like SealBot, it is an external
anchor not version-pinned in this repository; its checkpoint and serving
environment live outside the tree, so its provenance has the same limit.

Strix declares radius-6 candidates while the rules allow radius 8, and its
checkpoint search is limited to 300 placements. These scores are therefore
against a handicapped opponent, not peer ratings. The restriction was never
exhausted: every match below had zero forfeits and zero capped games.

Offline evaluation used 64 games, seed 0, and `mcts:sims=32`:

| Checkpoint | Score against strix | Capped | Forfeits |
| --- | ---: | ---: | ---: |
| `ckpt151-joint-brm.pt` (iteration 0) | 12/64 = 0.1875 | 0 | 0 |
| `checkpoint_000100.pt` | 21/64 = 0.328125 | 0 | 0 |
| `checkpoint_000253.pt` | 26/64 = 0.40625 | 0 | 0 |

The graft-to-253 change is about 2.8 standard errors, so it is real; adjacent
steps are not individually resolvable at 64 games. As a reference point under
the same measurement, `joint-939/checkpoint_000050.pt` scored 17/64 = 0.266
against strix while scoring 0.81 against SealBot.

A 64-game strix evaluation at `mcts:sims=32` cost 105.8 s on an RTX 4070 Ti.
The corresponding SealBot cost and its zero-simulation control are recorded
once in this file's preamble.

### Section 3.1 anchor across the VM boundary

| Field | Record |
| --- | --- |
| Deployment constraint | Strix's `hexo_rs` search extension exists only as a CPython 3.14 Windows build, while the driver runs in a Linux container, putting the seat and its orchestrator on opposite sides of a VM boundary. |
| Protocol consequence | CONTAINER_SPEC §3.1 makes transport unobservable to the protocol. The seat and driver remain unmodified; a relay in the WSL VM carries the seat's stdio, reaches the Windows interpreter through WSL interop, and reaches the container through its bridge gateway. |
| Exit ordering | The orchestrator gives the seat five seconds to exit after `ok(bye)`. The relay must close the connection before joining its pump threads, not after. |
| Stdin lifecycle | A stdin pump cannot be a thread: the orchestrator closes the child's stdin only after the child exits, which otherwise leaves the pump thread live inside a socket call during interpreter finalization. |
| Disposition | Retained as the transport for the in-driver strix anchor. |

### Segmented reduction under `torch.compile`

| Field | Record |
| --- | --- |
| Date | 2026-07-30 |
| Setup | `segment_sum` as `index_add_` over a row-to-segment index, inside the compiled `trunk` + `cell_heads` pass. Reached only by a configuration that puts a ragged reduction in that pass: the `dueling-critic` critic centers its advantage with a policy-weighted segment sum. |
| Measurement | Inductor's lowering drops the bounds mask on its atomic once it can prove the destination has a single row, so a single-position batch also sums padding lanes. Independently reproduced at 0.911 instead of 1.0 for a single-segment softmax at N = 7 on CUDA. |
| Disposition | On `dueling-critic`, `segment_sum` and the segment maximum became `torch.segment_reduce` over the CSR offsets, and the compiled-heads test gained a one-position batch. Equivalence of the two forms was checked over ragged shapes including one-position and all-singleton segments: exact in float64, and at fp32 reassociation level on CUDA where `index_add_` is itself order-dependent. |
| Exposure elsewhere | None measured on the other configurations: `policy_loss` and `improved_policy` call these helpers outside the compiled region, and no other configuration puts one inside it. |

### The improvement operator's expectations at fp32

| Field | Record |
| --- | --- |
| Date | 2026-07-30 |
| Setup | $\hat v=\sum_a\pi'(a)Q(a)$ evaluated in fp32 over a position's legal cells, with the $\lambda$-return's range check refusing $\lvert\hat v\rvert$ outside $[-1,1]$. |
| Measurement | The bound holds in real arithmetic and not in fp32: an fp32 segment softmax sums to one only to a few ulps, and where every legal move shares one saturated value — a lost endgame, every cell at exactly $Q=-1$ — nothing cancels that error. Measured over 287,140 acting plies of one collection: 3,909 plies outside the interval, p99.9 excursion 9.7e-06, maximum 2.97e-05. A 159-ply position at approximately three thousand legal cells produced 1.09e-04. Mixed-sign positions do not show it: at 400-ply random playouts with 88% of cells saturated, $\lvert\hat v\rvert$ never left the interval. |
| Disposition | `v_hat`, `kl`, and the entropy behind `norm_entropy` divide by their segment's summed $\pi'$, which is what an expectation under a normalized distribution means; the bound then holds to a few ulps at any segment width. Retained on every configuration. The range check keeps a 1e-4 slack. |
| Detection history | The check first refused a legitimate corpus at `brm-939` iteration 0 under an exact bound, then again at `tail-939` iteration 56 under the 1e-4 slack. Its message named neither the value nor which of its two conditions failed, and two manager hypotheses — a non-finite value from bf16 or `torch.compile`, and a slack too small for the sample size — were measured and contradicted before the third measurement located the cause. Both refusals now name the entry, its value, and the count of each condition, and every optimizer step is followed by a fused finiteness check over the parameters. |

## Graft campaign

Per-step records of `docs/MANTIS_GRAFT_SPEC.md`. Speed harness: WSL ext4
clone, `bench fit --corpus mnorm-late-v1 --split val --device cuda --compile
--seed 7 --steady-warmup 20 --steady-measure 50`, default budgets; the
campaign corpus is `mnorm-late-v1` (SHA-256 `cd5f5d0a…`, 1M/100k/100k
realized). The corpus-mode fit number is a gate metric over the frozen
distribution, not production throughput. Baseline at Step 0:
205.45 samples/s median (six runs across two Adam arms, spread ±0.2%),
peak 10.06 GiB.

### Step 0 — `perf-foundation`

| Field | Record |
| --- | --- |
| Commits | `5bab833` (ports), `b4d09cf` (§2.2 steady-window instrument) |
| Content | fused-Adam execution policy (`mantisnet.optim`, recorded and reapplied after checkpoint loads); fit batches pinned on the KLENT prefetch worker; `bbe64ca` not ported (production has no `torch.segment_reduce`) |
| Speed | fused 205.45 samples/s median (3 reps) vs foreach — the pre-port corpus-path equivalent — 205.42 (2 reps); spread ±0.2%; VRAM 10.057 GiB in all six. Fused-vs-foreach is a null at production's tensor layout; the donor's gain was ACT-layout-specific. The klent-path prefetch pin is mechanism-documented, not campaign-measured (owner ruling 2026-08-10: accepted without a pre-tree A/B). |
| Verification | full pytest and `cargo xtask verify` green at each commit |
| Disposition | **Baked** (owner, 2026-08-10). Windows-native measurement disqualified during calibration (same-arm 188→114 samples/s swings at 10.06 GiB under WDDM); harness pinned to the WSL environment and the VRAM ceiling amended to 10.25 GiB in the same ruling. |

### Step 1 — `dead-key-bias`

| Field | Record |
| --- | --- |
| Commit | `d2bab78` |
| Content | `wk.bias` and `wk_wa.bias` removed from every block (8 tensors, 1,024 scalars): a constant key bias shifts each query's score row uniformly and cancels in the softmax, so the parameters were gradient-dead. Census test refuses any softmax key bias by name; theorem test asserts bias-shifted keys leave both CPU attention reference paths unchanged; the lab family loader drops the two keys from historical checkpoints (exact), ordinary KLENT loading stays strict. |
| Gate | parity in place of a screen (functionally exact change); speed pro forma. Parameter count 1,944,165 → 1,943,141. The fp32 open-interval assertion on composed Q in `test_model.py` was corrected to closed-interval (strict bound retained on the float64 composition): fp32 softmax legitimately reaches ±1.0 past a ~17 logit gap, exposed by the shifted init stream. |
| Verification | full pytest (376) and `cargo xtask verify` green |
| Disposition | **Baked** (owner, 2026-08-10). |

### Step 12 — `mixed-windows`

| Field | Record |
| --- | --- |
| Commits | `9168639` (knob, both builders), `f92f500` (run-reduced mixed class/decoder sums), `ddcd2e5` (bake: ternary-only scope, `MODEL_REPR_VERSION` 4) |
| Screen | owner-amended 2×2 factorial × 3 seeds (A baseline / B mixed / C mixed+waOff / D waOff), 400k-sample cells on `mnorm-late-v1`, uniform matrix budgets (fit pair 4M / cell 250k, collect 12M / 1.2M). B s2 ran on `f92f500` at halved budgets (2M/125k, 6M/600k) after WDDM paging — budgets are chunking-only and recipe-recorded, gradients identical. |
| Primaries (paired vs A) | B top-1 +0.553 ±1.438 pp [2+/1−], critic sign +1.334 ±8.355 pp; top-3 +0.941 ±1.210 pp [3+/0−]. Horizon top-1 (intervals excluding zero, all [3+/0−]): moves 33–48 +0.482 ±0.465 pp, 49–64 +2.019 ±1.939 pp, 65+ +1.294 ±0.772 pp — the late-horizon gains carried the verdict. C top-1 −0.120; D −0.760 [0+/3−]. |
| Speed | matrix-budget steady window: A 3186.0 samples/s / 3.32 GiB; B 1068.1 / 9.17; C 390.3 / 2.42 (CPU-collate prefetch starvation, not attention cost); D 2388.5 / 2.42. Lean-budget B on `f92f500`: 1045.8 / 4.63 — ~2% under uniform-budget B with the §5.1c pair-density cliff gone. Node bill (`edbf760`): windows 2.22×, incidence 4.31×, decoder 1.69×. |
| Verification | full pytest (397, 3 skipped) and `cargo xtask verify` green at `ddcd2e5`; the arm-B matrix checkpoints are the reference state-dict shape and load unchanged |
| Disposition | **ACCEPTED and baked** (owner, 2026-08-11). Binary path deleted across Rust and Python; `mixed_windows: True` recorded in `LEGACY_BAKED_KNOBS`; binary-scope families are cleanly rejected by the registry; `window_attention` stays a live knob for Step 3. |

### §5.1c cell-mediated attention — measured negative (Step 12 in-step work)

| Field | Record |
| --- | --- |
| Commits | `b482346..2b5aec4` (claims-CSR derivation, three fused kernels, tunings), reverted at `ab3c3ae` |
| Content | §5.1c without the materialized pair edge list: crossing pairs enumerated through claimed cells at attention time. VRAM dramatically better — B lean 3.43 vs 4.63 GiB, A 3.17 vs 3.32, and the pair-density scaling cliff gone entirely. |
| Speed | 2.4–2.5× slower on both scopes (B lean 415.8 vs 1045.8 samples/s; A uniform 1355.4 vs 3186.0): the kernels sit at the random-gather bandwidth floor, so the cost belongs to the pair function, not the kernels. A line-blocked variant (claimant tiles serving 16 lanes) lost a further 1.8× to 41% lane occupancy, the fp32-parity ban on tf32/bf16 dots, and register pressure. |
| Disposition | **Reverted** (owner, 2026-08-11): the edge-list §5.1c is restored; lean budgets are the sanctioned VRAM lever (chunking-only, ~2% cost). The result motivates spec Step 15 (`cell-latents`), which changes the function instead of the kernels. |

### Step 4 — `action-rows`

| Field | Record |
| --- | --- |
| Commits | `7d62232` (screen flake fix), `adb5e8d` (knob: ternary-native 729 post-placement action classes end to end — both builders emit the per-action window rows, collation adds `act_class`/`act_rev`/`act_empty`, the row encoder folds into `act_proj`/`act_table`/`act_empty_base` plus bias-free `p_act`/`q_act`, Triton fused forward and backward, §33 alias diagnostic as a lab command), `403354b` (spec: §2.2 fit-regression bound owner-amended to 10%), `3fec94b` (collation perf), `22e7844` (bake: `MODEL_REPR_VERSION` 5) |
| Screen | 5 seeds × arms A (bg nearest-bucket) / B (action rows), 1-epoch lean recipe on `mnorm-late-v1`; plus an owner-ordered 4-epoch diagnostic pair at seed 5 (B then A) after the critic bistability surfaced. |
| Primaries (paired vs A) | top-1 +0.647 ±2.401 pp [3+/2−]; top-3 +0.841 ±2.200 pp [4+/1−]. Horizon top-1: every bucket's mean positive; moves 1–4 +1.408 ±1.214 pp [5+/0−] is the only interval excluding zero, 5–8 +0.747 ±0.888 pp [4+/1−] — the tactical band is where the hypothetical windows apply. Critic sign overall −0.253 ±2.512 pp: uninterpretable at one epoch (next row). |
| Critic bistability (finding) | The 1-epoch recipe gives the critic 95 optimizer steps; cells land **arm-independently** in an "optimist" basin (every v̂ positive — the tempo prior; signature \|mean_prediction\|/mean_abs_prediction > 0.97 at moves 1–4) or a discriminating one, on init/order luck. Screen split 5/10 optimist (A s1 s2; B s0 s2 s4); train CE separates the basins only in the third decimal (0.7048 vs 0.7035). Step 12's matrix retroactively shows 6/12 optimist cells, unnoticed at the time. 380 steps (4 epochs) escape decisively. Any future 1-epoch screen must basin-classify cells before reading a critic row. |
| 4-epoch pair (seed 5) | B wins critic sign at every decidable bucket (+1.5 to +4.3 pp through moves 25–32; moves 1–4 error 2.3% → 0.8%, sign 0.9924 vs 0.9773) and state-value sign everywhere (+0.5 to +5.3 pp — the 1-epoch state dip inverted); tactical top-1 +1.4/+0.9 pp at 1–4/5–8 with overall top-1 tied (0.4577 vs 0.4570). B's near-terminal v̂ magnitudes are conservative (moves 1–4 MAE +0.117). This pair carried the critic verdict. |
| Aliasing (§33) | 2,000 val positions, 1,199,458 legal actions: aliased actions 910,491 → 910,925 (+0.05%); distinct signatures 346,226 → 334,653; signature groups containing a bg-path action 13,115 → 1,976; largest group 213 → 1,185 — the designed far-ring merge: legal cells with no stone within 5 steps have all-18-EMPTY windows and share one row, where bg buckets gave a 1-D distance gradient. |
| Speed | lean-budget steady window (3 reps): fit 1045.5 vs 1063.8 samples/s (−1.72%), chunk p95 +1.4%, VRAM 4.636 vs 4.629 GiB (+0.15%); collect 162.4 vs 185.2 samples/s (−12.29%) — **owner-accepted** (2026-08-11), all remaining cost in collation (busy 15.0 → 25.8 s; an O(E) counting sort for the `act_rev` argsort was identified, not pursued). Production budgets (fit pair 8M / cell 400k) are off-card at this scope since Step 12's window growth, independent of Step 4: arm A demands 15.59 GiB against the 12 GiB card and pages to 17.5/14.1 samples/s across reps; arm B's pinned prefetch buffers additionally exceed the WSL pinned-host cap and refuse before the first step. Lean budgets remain the sanctioned lever (Step 12 disposition). A paired point at the largest on-card budgets (4M/250k) was attempted twice and is **not measurable with the pinned instrument**: the harness's 17.5k-sample draw packs into ~5 chunks at those budgets, below the 20+50 steady window, so the report degrades to compile-dominated whole-run time (A 26.3–26.5, B 20.0 samples/s at 9.17 GiB torch peak — reproduced identically on a clean card; these are not throughputs). |
| Verification | full pytest **398 passed, 3 skipped** and `cargo xtask verify` all gates green at `22e7844`; parameter pin 4,007,269 (`test_action_rows.py`; +140,672 over Step 12's 3,866,597 — the 729×h table plus projections, minus the bg embeddings); the 729-class encoding is checked against an engine-grounded successor-board oracle, including opponent digits. Knob-era arm-B checkpoints are the baked state-dict shape and load unchanged; pre-Step-4 dicts refuse loudly via `LEGACY_BAKED_KNOBS`. |
| Disposition | **ACCEPTED and baked** (owner, 2026-08-11: collect accepted at −12.29% mid-gate; approval on the third seed group). The bg nearest-bucket path (`e_bg`/`e_qbg`) is deleted in both languages; `action_rows: True` joins `LEGACY_BAKED_KNOBS`; `MODEL_REPR_VERSION` 5. `klent.graft` (the repr v1→v2 converter, three generations stale) retired with its tests; its probe helper is inlined into `trigraft`. |

### Window-attention removal — measured negative (Step 3 preview at Step 2 scope)

| Field | Record |
| --- | --- |
| Code | No model change: screen arms compose the two live knobs (C `state_latents: 4, window_attention: False`; D `window_attention: False`) at `0875eb0` — §5.1c and the window-latent cycle are independently gated. Parameter cross-check: C 4,538,341 = D 3,741,797 + 796,544 (the latent stack); A − D = 265,472 (per-block wa projections + `wa_bias`). |
| Screen | Owner-posed hypothesis (2026-08-12): the Step 2 latents' window read/broadcast might subsume §5.1c window attention. Arms C/D × seeds 0–4, 4 epochs, Step 2 lean recipe on `mnorm-late-v1`, same sweep as the completed A/B cells (`runs/lab/step2-epochs4`) → a wattn × latents 2×2 factorial with 5 paired seeds per contrast. C/D cells ran at `0875eb0`; model math is identical to the A/B rev (the ragged rewrite was reverted before launch). |
| Primaries | D−A (removal alone): top-1 −1.927 ±0.834 pp [0+/5−] and top-3 −1.688 ±1.115 pp [0+/5−] — the factorial's only intervals excluding zero; state MAE +1.224 ±1.369 e−2 [5+/0−]. C−B (removal given latents): critic sign −1.279 ±1.470 pp [1+/4−]; state MAE +0.714 ±1.682 e−2 [4+/1−] — Step 2's calibration gain largely erased; top-1 −1.221 ±3.272 pp on a 4.39 pp seed range. C−D: state MAE −1.905 ±2.113 e−2 [0+/5−] — the latents' own calibration contribution survives without wattn. The two mechanisms are complementary, not redundant. |
| Horizon | The damage concentrates near-terminal: D−A v̂ sign at moves 1–4 mean −9.31 pp (seed 4 −39.0); D−A state MAE at 1–4 +6.27 e−2; C−B state MAE at 1–4 +5.75 e−2. No optimist-basin cell (4-epoch screen), but all five D cells sit at mean_pred/mean_abs +0.4–0.6 — a uniform optimism tilt absent in A and B. |
| Stability | wattn also stabilizes policy training: per-arm top-1 seed ranges A 0.67 / B 1.36 / D 1.63 / C 4.39 pp. |
| Disposition | **wattn stays**; the latents-subsume-wattn hypothesis is rejected. **Step 3 RULED REJECTED on this record (owner, 2026-08-12)** — the factorial is Step 3's screen shape at 5 paired seeds and the removal is measured negative, so the trial does not run separately. Per the reject clause §5.1c returns to baked-in, with one deliberate exception: the `window_attention` knob stays live until Step 15's matrix completes (its arms B/C/D run §5.1c off), and the knob deletion executes at Step 15's bake whichever way that verdict goes. |

### Step 2 — `state-latents` — PROVISIONAL

| Field | Record |
| --- | --- |
| Commits | `1794036` (knob: K=4 invariant state latents replace the global token; stone-side rides fused stone attention, per-block window read/broadcast, latent self-attention mix, token readers get mean-pooled latents), `58ec601` (fused ragged Triton window-latent read/broadcast; padded path deleted; literal state-latent oracles 21/21), `54b962e` (bake: `MODEL_REPR_VERSION` 6, Rust-native 4-row prefix, parameter pin 4,803,813) |
| Screen | 4-epoch, 5 paired seeds (`runs/lab/step2-epochs4` A/B): top-1 dead null (+0.02 ±1.07 pp); critic sign +1.28 ±2.33 pp [4+/1−] with moves 1–4 +1.28 ±1.19 pp [5+/0−]; the real gain is calibration — state MAE −1.40 ±0.72 e−2 [0+/5−] overall, v̂ MAE significant at horizons 17–48; no optimist-basin cells; params +796,544 |
| Gate | The padded reference implementation measured fit −9.9% (inside the amended 10% bound) but fit VRAM +1.12 GiB (fails the +256 MiB bound) and collect −14.7% (fails); the fused ragged kernels were the in-step fix. The §2.2 re-pin of the fused implementation on the current NVIDIA driver was started and stopped to free the card; the driver changed 610.47 → 610.88 between baseline and gate, voiding the one completed comparison. A python-level ragged rewrite measured a separate hard negative (fit −42.6%) and was reverted (`565633a`). |
| Disposition | **APPROVED by owner (2026-08-12); merged to main 2026-08-17 PROVISIONALLY** — the speed gate on the current driver was never completed. The verdict is re-openable on that measurement; the sanctioned instrument and bounds are recorded in the spec §2.2. |

### Step 15 — `cell-latents` — PROVISIONAL

| Field | Record |
| --- | --- |
| Commits | `6dd10ff`/`d4ce69c` (cell/line tables + typed segment attention kernels, relay geometry, deterministic backward), `0f212e5`/`60a57bf` knob wiring (rebased `04d4a50`/`5bc3ae7`/`87ae3af`): knobs `cell_latents`/`line_pass`/`claim_reach`; the cell stage replaces the §5.1b relay when on; trunk returns refined legal-cell latents scattered via `covered`, uncovered cells keep the learned base row |
| Screen | 25/25 4-epoch cells, five arms × five seeds (`runs/lab/step15-epochs4`), zero failures, no optimist cells. Cell mediation replaces §5.1c for policy: C−A top-1 +1.16 ±1.37, B−A +0.95 ±1.68, where naked §5.1c removal was −1.93* (the factorial above). The line pass is a dud: B−C −0.21 ±0.33, D−A negative. Starred intervals: C−D top-1 +2.06 ±1.80 [5+/0−]; C−D v̂ MAE +1.31 ±0.52. Critic pays a small consistent price: B−A sign −0.76 ±0.78 [0+/5−], worst bucket −2.4 pp at moves 25+, no bucket collapse. Arm C trains in 25.9 min vs A's 47.9 (−46%) with seed spread 0.70 pp vs 3.01. |
| Disposition | **Merged to main 2026-08-17 PROVISIONALLY.** The owner ordered the successor run launched from the arm-C prefit (2026-08-14) — acceptance in practice; the formal verdict and bake ritual (including the deferred `window_attention` knob deletion from the Step 3 ruling) remain open. |

### Step 13 — `cell-nodes` — PROVISIONAL

| Field | Record |
| --- | --- |
| Commits | `fabef56` (cell nodes: every legal cell carries occupancy/legality/bucketed nearest-distance features plus stone→cell radius-8 edges typed by the frozen 48-class D6 orbit vocabulary; orbit table generated from axial transforms with an independent cube-coordinate oracle; D6 equivariance test replays transformed moves through the engine), `067ee18` (`cell_node_scope` `all`\|`uncovered`), `da06e0c` (test pins). `MODEL_REPR_VERSION` 7; knob-off byte-identical. |
| Gate | Pinned WSL instrument at live-run budgets, medians of 3: base fit 1277.5 samples/s / 4.638 GiB; `all` 1036.5 (−18.9%) / 6.274 GiB; `uncovered` 1082.5 (−15.3%) / 6.177 GiB. `uncovered` is barely cheaper than `all`: the scope filter runs inside the edge-table op, so both scopes pay derivation and per-cell expansion. Collect −11.4% / −12.5%. |
| Prefit | armC-recipe paired at seed 3 (cell nodes the only difference): train loss better (ep4 1.891 vs 1.990) but static validation worse on policy (top-1 −0.60 pp) and v̂ sign in 9/9 horizon buckets, state MAE better in 8/9 — a mild overfit signature at +369k params (5,196,965). |
| Live evidence | The `cellnodes-1` carry run (scope `all`, from the step13 prefit, reference recipe, seed 21) at iteration ~390: strix-anchor evals pooled over iterations 200–375 = 399/512 = 77.9% vs `stack-939`'s 293/512 = 57.2% at the same iterations against the byte-identical anchor (digest `0x351ed562065bed55`) — ≈ +170 elo, z ≈ 5.3. This measures the whole campaign tree plus prefit initialization against pre-campaign production, not this step's marginal. In-play knowledge-horizon critic sign accuracy beats `stack-939` in every bucket at the matched window; the prefit's 9/9 static critic-sign deficit does not reproduce in live self-play. SealBot is saturated (64/64 at iterations 225 and 375). |
| Disposition | **Merged to main 2026-08-17 PROVISIONALLY** (owner's launch condition was speed-only: "launch with all if it is within 30%"). The isolating comparator — `newmodeltest` (cell latents only) resumed past iteration 31 — has not run, so the step's marginal value against its 2× per-iteration cost is unmeasured and the verdict is open. |

### Steps 5+6 — `tactical-scalars` × `action-latents` — screen complete, verdicts open

| Field | Record |
| --- | --- |
| Commit | `9aeee88`, both knobs in one change. Step 5 (`action_tactical`): the 11 deterministic per-action tactical scalars (ACT §19.3 minus mixed-created), emitted bit-identically by both builders, validated on the wire, entering the action contribution through a zero-init two-layer MLP. Step 6 (`action_latents`): 2 invariant per-position action latents reading the legal set via fp32 segment softmax with a dense mix and zero-init broadcast, mirroring the §5.4 cycle; model-only. `MODEL_REPR_VERSION` 8; both knobs off byte-identical. Param pins 4,803,813 default, +18,048 tactical, +199,296 latents. |
| Screen | 2×2 × 3 seeds (A base / B tactical / C latents / D both) on the production config (`cell_latents` + `cell_nodes` scope `all`, wattn off; measured params A 5,196,965 / B 5,215,013 / C 5,396,261 / D 5,414,309), 4 epochs, corpus `cn1-late-v1` (SHA-256 `46db7644…`; cellnodes-1 self-play iterations 302–401, 410k games / 36.6M plies, 1M/100k/100k realized), deterministic 400k train subset, lean budgets, paired evaluation on the identical 100k val positions. 12/12 cells, zero failures. |
| Primaries (paired vs A) | B top-1 **+1.15 ±0.71 pp [3+/0−]**, top-3 **+1.73 ±0.85 pp [3+/0−]** — both intervals exclude zero; positive in every distance bucket on every seed, largest in the tactical band (moves 1–8). C top-1 −1.34 ±2.18 pp [1+/2−], top-3 −1.25 ±2.44. D top-1 −0.81 ±2.34 pp [1+/2−], top-3 −0.80 ±3.13. Factorial means (tactical +0.84, latents −1.65, interaction −0.31 pp) mislead: the latents rows are a pathology-rate story, not a mean shift (next row). |
| Stability (finding) | B's absolute top-1 seed spread is ±0.11 pp (42.84/42.65/42.84) against the baseline's own ±0.65 — when A s2 dropped 1.1 pp into a weaker basin, B did not follow. The scalars stabilize training beyond their mean gain, and B's critic carries the least tempo bias of any arm (v̂ \|mean\|/mean_abs at moves 1–4: 0.44–0.69 vs A's 0.61–0.84). The latents arms are the opposite: 3 of 6 cells pathological in two distinct modes — policy collapse (C s0 −3.70 pp, D s2 −3.31 pp) and a critic optimist basin (D s0, ratio 0.970, near-terminal sign 0.620) — the campaign's first optimist cell at 4 epochs, which the Step 4 record had as a 1-epoch phenomenon that 380 steps escape decisively. The modes do not co-occur: D s2's near-terminal critic is the cleanest in the screen (ratio 0.048, sign 0.992) under its collapsed policy. Non-pathological latents cells span −0.92 to +1.34 pp, and D s1 (+1.34) is the best cell in the screen. Latents-off: 0 pathological in 6. |
| Critic | Pooled sign-accuracy separates no arm from A on either channel (per-bucket spreads reach ±0.20 on the latents arms — basin noise dominates the channel at this seed count). One interval survives: C's `state_value` sign at moves 1–4, +2.10 ±1.25 pp [3+/0−] over A, positive on all three seeds including the collapsed one. |
| Speed | §2.2 pinned WSL instrument on `mnorm-late-v1`, medians of 3 (20+50 steady window), 24/24 benches clean: base fit 1041.1 samples/s / 6.280 GiB, collect 147.6. Tactical fit 995.9 (−4.35%) / +34 MiB, collect −0.65% — **passes every bound**. Latents fit 1017.4 (−2.28%) and collect −6.53% pass, but fit VRAM +309 MiB **fails the +256 MiB bound**; both-knobs likewise (+344 MiB; fit −4.44%, collect −6.95%). Per-arm rep spread is ±3–4%, so the small fit deltas are ordering-noise between arms; on the screen corpus the latents' fit cost is larger and consistent (`cn1-late-v1` mean epoch throughput: A 1040.3 ±16.6, B 1017.4 −2.2%, C 931.7 −10.4%, D 930.8 −10.5%). |
| Verification | full pytest **466 passed, 3 skipped** and `cargo xtask verify` all gates green at `9aeee88`. The scalars are checked against an engine-grounded successor oracle in both languages, all-action sweeps over handcrafted threat games, D6 multiset invariance, and wire refusal of corrupt payloads; the latents against permutation invariance, a bias-free key census, and knob-off byte identity. |
| Disposition | **Screen complete 2026-08-17; both verdicts open (owner).** |

## Provenance

| Source | Use in this record |
| --- | --- |
| `metrics.jsonl` | Legacy evaluation rows and iteration trajectories, including append-only/resumed branches |
| `telemetry.db` | `eval_matches` score, win rate, Wilson interval, Elo, seat scores, capped count, and forfeits per opponent; `iterations` trajectories; `plies` bucket queries. Every database was opened with a read-only URI, `mode=ro`, followed by `PRAGMA query_only=ON`; the five quantized ply scalars were divided by 10,000. |
| `sealbot_curve.jsonl` | Offline `overnight-3` depth-1 checkpoint scores, Wilson intervals, Elo, seat scores, and mean plies |
| `config.json`, `invocations.jsonl`, `status.json`, adjacent logs/errors | Forks, exact flags, branch boundaries, dates, completion state, and stop-sentinel evidence |
| `KLENT_RUN_PLAN.md` §§2–4a | Historical claims and experiment criteria not encoded in the run artifacts, labeled **(run plan, not re-derived)** where cited. The file itself is retired and no longer in the tree. |
| Manager measurements dated 2026-07-29 | `factored-939-s2` report, ply-bucket decomposition, and critic ranking-stability probe |
| Manager measurements dated 2026-07-30 | The three arms' shared 96-position critic probe, the bipolar head's logit and mass quantities, the fp32 expectation excursion counts, and the segmented-reduction reproduction |
| `runs/grafts/ckpt151-{brm,duel,tail,joint,joint-brm}.json` | Each arm's graft transform, probe, and preservation measurements, written by the conversion itself |
| Externally served strix checkpoint | Offline anchor measurements and the seat-reported name, digest, variant, and restriction; the checkpoint and serving environment are outside this tree and not version-pinned here |
| `python/mantisnet/README.md`, “Performance” and “Deliberately absent” | Engineering benchmark receipts and retained/removed implementation dispositions |

Every evaluation score was compared with `eval_matches` when the run retained a database row. No score mismatch was found. Resume cleanup left metrics-only evaluations at `pure-2` iteration 200 (0.84375), `conv-disc-lam01` iteration 50 (0.703125), and `factored-939` iteration 50 (0.8125); those rows consequently have no retained database CI or Elo.

Artifact limits relevant to verification:

| Scope | Limit |
| --- | --- |
| Archived runs (`runs/archive/`) | None of the archived exploratory runs retains a `telemetry.db`, so Wilson intervals/Elo and ply buckets cannot be re-derived for them; most lack `status.json`. |
| `runs/grafts/` | The cited checkpoint conversions have transform manifests; grafts are not training runs and carry no run metadata. |
| Engineering microbenchmarks | No raw benchmark-output files are retained in the cited source set; the README and run plan are the available receipts. |

Artifact/run-plan discrepancies are stated in the affected reports; unstated aggregation windows are flagged for `conv-rho1`, `lam-ret-939`, and `factored-939`.
