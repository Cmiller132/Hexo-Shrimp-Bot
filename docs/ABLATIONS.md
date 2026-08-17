# Ablation and experimental record

This file is the technical record of the retained KLENT training runs and engineering experiments. It records configurations, measurements, artifact limits, and dispositions. It does not rank runs or prescribe a next experiment.

The comparison reference used below is: \(\gamma=0.99\), \(\lambda=0.01\), \(\tau=0.1\), \(\lambda_{\mathrm{ret}}=0.939\), and a scalar tanh critic; 4096 completed games per iteration, 1024 environments, ply cap 512, and learning rate \(10^{-3}\). Unless a run configuration says otherwise, the recorded in-driver evaluations below are 64 seat-balanced games against uncapped SealBot at 0.1 s/move, using the 32-simulation Gumbel line search.

One such evaluation costs 69.6 s on an RTX 4070 Ti, measured on `joint-939/checkpoint_000050.pt` at seed 0 over 64 games averaging 45.3 plies, none capped. Repeating it at zero simulations — raw-policy argmax, 45.6 mean plies — costs 57.2 s, so SealBot's own turn budget sets a floor near 57 s and the 32-simulation line search adds about 12 s above it. The pair is a cost control and not a strength comparison; it locates an evaluation's wall time, which is mostly not in the model.

The current driver can evaluate against SealBot, one independent §3.1 subprocess seat, or both. `--eval-games` is a per-opponent count: when both are configured, each receives the same seat-balanced opening and model-RNG schedule derived from the run seed and completed iteration, rather than splitting the requested games. Metrics and telemetry attribute a separate result to each opponent and its strength-defining configuration.

Runs whose last artifact predates 2026-07-29 — everything up to and including `pure-2`, plus the two `.aborted-guard` directories — live under `runs/archive/<name>/`. The move is the whole of the archiving: nothing is deleted, every artifact this record cites is still readable at that path, and the deck lists only direct children of `runs/` that hold a `config.json`, so an archived run stops appearing without being removed.

Training metric iteration numbers are the zero-based values stored in `metrics.jsonl` and `iterations`. `eval_matches.iteration` is the completed-iteration count, so an evaluation on metrics row 24 is stored as evaluation 25. Legacy results are labeled “metrics row”; database results are labeled “@”. `H` means `acting_norm_entropy`, or the corresponding quantized per-ply `norm_entropy` after division by 10000. “Decided” means \(|\hat v| \ge 0.5\).

## Training runs

### `shakeout-1`

| Field | Record |
| --- | --- |
| Dates | 2026-07-27, 19:23–20:11 EDT from artifact timestamps |
| Init/fork | Fresh initialization; seed 1 |
| Delta from reference | Historical \(\gamma=1.0\) objective; `--lam 0.03`; `--games 64`; pre-auto-reset collector with no `--envs`; `--seed-fraction 1.0 --seed-cut 1 8 --seed-noise 0.1`; `--batch 256`; `--iterations 100 --checkpoint-every 10 --eval-every 10 --eval-games 64`; legacy line-builder anchor evaluation |
| Question | Loop stability, crash/resume behavior, instrumentation, and baseline training dynamics |
| Disposition | Completed 100 iterations; superseded by the packed-batch sweep runs and later recipe changes |

| Measurement | Artifact record |
| --- | --- |
| Run shape | 114 metrics rows, 100 unique iterations 0–99, duplicate resume spans 10–16 and 90–96, and ten checkpoints through `checkpoint_000100.pt` |
| Initial transient | Won length 176.704 at iteration 0, 60.063 at 4, and 18.875 at 8; acting KL 3.894 at 0; winner/loser \(\hat v\) 1.788/1.778 at 0, 0.321/0.324 at 1, and approximately zero at 2 |
| Entropy/KL interval | H ranged 0.1247–0.9173. During iterations 64–75, H was generally 0.89–0.92, KL was 0.0026–0.0122, and `f_seeded` reached 0.5469 at iteration 71. |
| Legacy-anchor evals | Metrics rows `9:.515625, 19:.328125, 29:.34375, 39:.6875, 49:.421875, 59:.609375, 69:0, 79:0, 89:0, 99:0`; 64 games each, zero capped |
| Crash/resume diagnostic | Two crashes were attributed to the fp32 policy-target sum check; worst \(\lvert\mathrm{sum}-1\rvert=1.3\times10^{-4}\) at width about 14k with zero NaN/Inf, followed by f64 accumulation for the check **(run plan, not re-derived)**. |

Measured relationship: the interval with H near 0.91 and KL near zero coincided with the first zero evaluation at metrics row 69; later evaluations remained zero after `f_seeded` returned to 1.0.

Artifact discrepancies:

- The run plan rounds the peak evaluation to 0.688 **(run plan, not re-derived)**; the artifact value is 0.6875.
- The run plan says `v_hat_mae` never fell below about 0.7 **(run plan, not re-derived)**; the metrics minimum is 0.590697 at iteration 74.
- The run plan says winner-side \(\hat v\) remained at or below about 0.4 after iteration 1 **(run plan, not re-derived)**; metrics contain 0.647, 0.860, 0.808, and 0.867 at iterations 65, 67, 68, and 74.
- The run-plan sequence “177 → 60 → 20 by iteration 4” **(run plan, not re-derived)** is stored at iterations 0, 4, and 8.

### `sweep-a`

| Field | Record |
| --- | --- |
| Dates | 2026-07-27, 21:47–21:59 EDT |
| Init/fork | Fresh initialization; seed 1 |
| Delta from reference | Historical \(\gamma=1.0\); `--lam 0.03`; `--games 64`; no `--envs`; full seeding at cut 1–8 with noise 0.1; `--batch 4096 --pair-budget 8000000 --cell-budget 400000`; `--iterations 100 --checkpoint-every 25 --eval-every 10 --eval-games 64`; legacy anchor evaluation |
| Question | Packed-batch continuation of the \(\lambda=0.03\) baseline and the initialization behavior of that setting |
| Disposition | Stopped after iteration 10; no stored stop criterion or reason; superseded by the 256-game sweep arms |

| Measurement | Artifact record |
| --- | --- |
| Run shape | Iterations 0–10; no checkpoint |
| Start | Iteration 0: acting KL 3.904, H 0.322, `f_seeded=0.406`, winner/loser \(\hat v=1.791/1.779\) |
| Corpus loss | By iteration 3, H was 0.983, `f_seeded=0`, and the buffer was empty. Iterations 3–10 remained near H 0.982 with an empty buffer. |
| Eval | Metrics row 9: 3/64 = 0.046875 against the legacy anchor; zero capped; no CI or Elo artifact |

Measured relationship: high initial KL and same-sign winner/loser \(\hat v\) preceded the empty-buffer interval.

### `sweep2-a`

| Field | Record |
| --- | --- |
| Dates | 2026-07-27, 22:00–22:11 EDT |
| Init/fork | Fresh initialization; seed 1 |
| Delta from reference | Historical \(\gamma=1.0\); `--lam 0.03`; `--games 256`; no `--envs`; `--seed-fraction 1.0 --seed-cut 1 8 --seed-noise 0.1`; `--batch 4096 --pair-budget 8000000 --cell-budget 400000`; `--iterations 100 --checkpoint-every 25 --eval-every 10 --eval-games 128`; legacy anchor evaluation |
| Question | \(\lambda=0.03\) arm of the 256-game comparison |
| Disposition | Stopped after iteration 3; no stored stop criterion or reason; superseded by the zero-Q initialization arm |

| Iteration | KL | H | `f_seeded` | Winner/loser \(\hat v\) | Won length | Seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3.894 | 0.322 | 0.473 | 1.765 / 1.770 | 210.64 | 97.0 |
| 1 | 4.951 | 0.710 | 1.000 | -0.887 / -0.880 | 72.79 | 12.2 |
| 3 | 0.978 | 0.9935 | 0.0586 | -0.156 / -0.156 | 143.27 | 573.9 |

No evaluation, checkpoint, status file, telemetry database, or run-specific log is present.

### `sweep2-b`

| Field | Record |
| --- | --- |
| Dates | 2026-07-27, 23:01–23:44 EDT |
| Init/fork | Fresh initialization; seed 1 |
| Delta from reference | Historical \(\gamma=1.0\); `--games 256`; no `--envs`; `--seed-fraction 1.0 --seed-cut 1 8 --seed-noise 0.1`; `--batch 4096 --pair-budget 8000000 --cell-budget 400000`; `--iterations 100 --checkpoint-every 25 --eval-every 10 --eval-games 128`; legacy anchor evaluation; \(\lambda=0.01\), \(\lambda_{\mathrm{ret}}=0.939\), and lr \(10^{-3}\) match the numerical reference |
| Question | \(\lambda=0.01\) arm of the 256-game comparison |
| Disposition | Stopped after iteration 9; no stored stop criterion or reason; superseded by `abl-zeroq-lam01` |

| Measurement | Artifact record |
| --- | --- |
| Start | Iteration 0: KL 4.336, H 0.259, `f_seeded=0.441`, winner/loser \(\hat v=1.784/1.784\) |
| Completion | `f_seeded` was 0.0078 at iterations 5–6, 0.0039 at 7, and 0 at 8–9. |
| Entropy/buffer | H was 0.964 at iterations 8–9 and the buffer was empty. |
| Eval | Metrics row 9: 0/128 against the legacy anchor; zero capped; no CI or Elo artifact |

Lowering \(\lambda\) did not remove the observed high-initial-Q/empty-buffer sequence in this arm.

### `abl-zeroq-lam01`

| Field | Record |
| --- | --- |
| Dates | 2026-07-27 23:47–2026-07-28 01:55 EDT |
| Init/fork | Fresh initialization; seed 1; zero-initialized Q output. The code-level initialization change is not serialized in `config.json`; iteration-0 winner/loser \(\hat v=0/0\) and the run plan identify it. |
| Delta from reference | Historical \(\gamma=1.0\); `--games 256`; no `--envs`; `--seed-fraction 1.0 --seed-cut 1 8 --seed-noise 0.1`; `--iterations 100 --checkpoint-every 25 --eval-every 10 --eval-games 128 --starve-limit 4`; legacy anchor evaluation |
| Question | Whether zero Q initialization removes the sweep initialization behavior at \(\lambda=0.01\) |
| Disposition | Stopped by the four-iteration starvation criterion; `checkpoint_000008.pt` written; warm collection adopted in the next runs |

| Measurement | Artifact record |
| --- | --- |
| Run shape | Iterations 0–7 |
| Start | KL 0.001653, H 0.984913, winner/loser \(\hat v=0/0\), `f_seeded=0.0546875` |
| Completion | `f_seeded` sequence `.05469, .00391, .15625, .01953, .00391, 0, .00391, .00391`; mean 0.030762 |
| Other metrics | H stayed 0.978–0.998; `v_hat_mae` stayed approximately 1.0 |
| Eval | None before stop |

Measured relationship: zero initialization removed the initial Q/KL spike present in the sweep arms, while the initial improved policy was near-uniform and completion remained low. The adjacent log records the criterion as four consecutive iterations under one sample per game.

### `overnight-1`

| Field | Record |
| --- | --- |
| Dates | 2026-07-28, 02:00–02:39 EDT |
| Init/fork | Fresh initialization; seed 2 |
| Delta from reference | Historical \(\gamma=1.0\); `--lam 0.03`; `--games 256`; no `--envs`; `--seed-fraction 0.9 --seed-cut 1 8 --seed-noise 0.1`; `--warm-iterations 15 --starve-limit 6`; `--iterations 3000 --checkpoint-every 25 --eval-every 25 --eval-games 128`; legacy anchor evaluation |
| Question | Whether a 15-iteration line-builder warm phase survives handoff with \(\lambda_{\mathrm{ret}}=0.939\) |
| Disposition | Stopped after metrics row 23 with `checkpoint_000024.pt`; superseded by `overnight-2`. The configured starvation limit is present, but no status or log independently records the stop path. |

| Interval | Artifact measurements |
| --- | --- |
| Warm, iterations 0–14 | `f_seeded=f_unseeded=1`; q-loss 1.000 → 0.863736 |
| Handoff, iteration 15 | q-loss 0.400856; winner/loser \(\hat v=0.1626/0.0857\); H 0.7050 |
| Iteration 16 | q-loss 0.130942; \(\hat v=0.0222/0.0142\); H 0.9620 |
| Iteration 18 | q-loss 0.056555; \(\hat v=-0.00664/-0.00083\); `f_seeded=.00420`; `f_unseeded=0` |
| Iterations 19–21 | Empty buffer |
| Final row 23 | `f_seeded=.00452`; `f_unseeded=0` |

Measured relationship: winner/loser \(\hat v\) converged to similar near-zero values while q-loss fell after handoff, followed by near-zero completion. The run plan characterizes a long-game target at this setting as about 94% bootstrap **(run plan, not re-derived)**.

### `overnight-2`

| Field | Record |
| --- | --- |
| Dates | 2026-07-28, 02:41–02:59 EDT |
| Init/fork | Fresh initialization; seed 3 |
| Delta from reference | Historical \(\gamma=1.0\); `--lam 0.03 --lam-ret 1.0 --lr 0.00025`; `--games 256`; no `--envs`; `--seed-fraction 0.9 --seed-cut 1 8 --seed-noise 0.1`; `--warm-iterations 30 --starve-limit 6`; `--iterations 3000 --checkpoint-every 25 --eval-every 10 --eval-games 128`; legacy anchor evaluation |
| Question | Monte-Carlo returns and reduced learning rate after a 30-iteration warm phase |
| Disposition | Stopped after metrics row 39; superseded by `overnight-3`. Only checkpoint 25 is present; the configured starvation limit is not confirmed by status or log. |

| Interval | Artifact measurements |
| --- | --- |
| Warm, iterations 0–29 | q-loss 1.000 → 0.868446 |
| Warm evals | Metrics rows `9:.671875` (86/128), `19:.71875` (92/128), `29:.734375` (94/128); zero capped |
| Handoff, iteration 30 | H 0.5067; q-loss 0.9062; winner/loser \(\hat v=0.1941/0.0666\) |
| Iteration 33 | `f_unseeded=0`; `f_seeded=.4526` |
| Iteration 39 | H 0.9020; q-loss 1.02795; \(\hat v=-0.1859/-0.1922\); `f_seeded=.01293`; eval 0/128 |

The 0.868446 q-loss at warm handoff is the artifact value behind the run plan’s rounded 0.87.

### `overnight-3`

| Field | Record |
| --- | --- |
| Dates | 2026-07-28, 03:00–07:40 EDT; offline SealBot curve written later that day |
| Init/fork | Fresh initialization; seed 4 |
| Config phases | (1) Initial: historical \(\gamma=1.0\), `--lam .01 --lam-ret 1.0 --lr .00025 --games 256 --seed-fraction .9 --seed-cut 1 8 --seed-noise .1 --warm-iterations 300 --starve-limit 6 --eval-every 25 --eval-games 128`; (2) resume at 350 adds `--anneal`; (3) resume at 1100 changes `--lam .03`; (4) final resume at 2050 keeps \(\lambda=.03\), uses `--eval-every 5 --iterations 2062`. All in-driver evals were against the legacy anchor. |
| Question | Warm-duration sufficiency, static versus annealed seed cuts, and the measured \(\lambda=.01\) and \(\lambda=.03\) regimes |
| Disposition | `checkpoint_002062.pt` retained as the later fork; superseded by the no-seeding/no-grounding `pure-1` reset |

The append-only metrics file contains superseded branches: 0–354 initial, 350–1122 annealed at \(\lambda=.01\), 1100–2088 annealed at \(\lambda=.03\), and a final 2050–2061 re-anneal that produced checkpoint 2062. Duplicate spans are 350–354, 1100–1122, and 2050–2061.

| Phase | Decisive artifact trajectory |
| --- | --- |
| Warm, 0–299 | q-loss 1.000 → 0.075586, with 0.059527 at 249 and 0.045994 at 274; 12 evals ranged 0.65625–0.78125 |
| First handoff, 300 | Winner/loser \(\hat v=0.613202/-0.369469\); `v_hat_mae=0.511420` |
| Static cut, 324–349 | Eval 0.6328125 → 0 while `f_seeded=f_unseeded=1` at 349 |
| Annealed \(\lambda=.01\), 350–1099 | `seed_cut_hi` 10 at 350, 508 at 599, then 512; eval 0 at parent row 349 → 0.6953125 at 374. At 899: H 0.09919, q-loss 0.04057, winner/loser \(\hat v=0.893/-0.846\), won length 11.00. At 1074: H 0.08632, q-loss 0.02338, MAE 0.04256, won length 11.21, eval 0.140625. At 1099: H 0.03171, won length 28.58, eval 0.03125. |
| Restarted \(\lambda=.03\), 1100 onward | H 0.06839 at 1100 → 0.29251 at 1122 → 0.30352 at 1374. Eval 0.0234375 at 1124 → 0.640625 at 1274 → 0.6953125 at 1324 → 0.7421875 at 1374; later values were variable. |
| Final branch, 2050–2061 | `seed_cut_hi` 10 → 32; H 0.23082 → 0.17051; eval 0.8046875 at 2054 and 0.8671875 at 2059 |

<details>
<summary>Complete legacy-anchor evaluation series</summary>

| Branch | Metrics rows and scores; 128 games each, zero capped |
| --- | --- |
| Initial | `24:.671875, 49:.65625, 74:.671875, 99:.71875, 124:.7421875, 149:.78125, 174:.703125, 199:.7109375, 224:.6953125, 249:.7265625, 274:.671875, 299:.7265625, 324:.6328125, 349:0` |
| Annealed \(\lambda=.01\) | `374:.6953125, 399:.140625, 424:.6328125, 449:.625, 474:.703125, 499:.65625, 524:.6640625, 549:.5859375, 574:.8828125, 599:.75, 624:.390625, 649:.21875, 674:.703125, 699:.171875, 724:.8125, 749:.0703125, 774:.0078125, 799:.2109375, 824:.09375, 849:.4765625, 874:.2421875, 899:.8125, 924:.0859375, 949:.25, 974:.1328125, 999:.109375, 1024:.1328125, 1049:.125, 1074:.140625, 1099:.03125` |
| \(\lambda=.03\), later superseded after 2050 | `1124:.0234375, 1149:.0625, 1174:.1953125, 1199:.4765625, 1224:.4609375, 1249:.484375, 1274:.640625, 1299:.59375, 1324:.6953125, 1349:.59375, 1374:.7421875, 1399:.4453125, 1424:.6953125, 1449:.578125, 1474:.1328125, 1499:.7109375, 1524:.4921875, 1549:.4453125, 1574:.6328125, 1599:.6484375, 1624:.8203125, 1649:.1171875, 1674:.65625, 1699:.609375, 1724:.625, 1749:.625, 1774:.6484375, 1799:.7109375, 1824:.640625, 1849:.65625, 1874:.6171875, 1899:.2109375, 1924:.75, 1949:.703125, 1974:.796875, 1999:.78125, 2024:.8984375, 2049:.8359375, 2074:.78125` |
| Final branch | `2054:.8046875, 2059:.8671875` |

</details>

Measured relationships:

- Extending warmup from 30 to 300 iterations changed the first-handoff measurements from `overnight-2`’s 0.194/0.067 winner/loser \(\hat v\) to 0.613/-0.369.
- At the static cut, completion remained 1.0 while evaluation fell to zero; the annealed fork from the same checkpoint measured 0.6953125 at row 374.
- In the annealed \(\lambda=.01\) branch, low H, short games, low q-loss/MAE, and low legacy-anchor evaluations occurred in the same interval.
- The \(\lambda=.03\) restart raised H during the restarted anneal; its evaluations were not confined to one narrow band.

Artifact discrepancies:

- Root `config.json` reflects the initial \(\lambda=.01\), no-anneal invocation. Later invocations define the final \(\lambda=.03\) checkpoint.
- The run plan’s “eval about 0.65, peaks 0.82” **(run plan, not re-derived)** does not match the artifact maximum: metrics contain 0.8828125 at row 574 and 0.8984375 at row 2024.
- The run-plan statement that evaluation held in 0.60–0.72 for 500+ iterations **(run plan, not re-derived)** is not literal in the metrics; the \(\lambda=.03\) branch includes 0.1171875 at 1649, 0.2109375 at 1899, 0.8203125 at 1624, and 0.8984375 at 2024.
- The checkpoint-350 result “lost 63/64 with all three heads” and the later 9:1 anchor result are **(run plan, not re-derived)**. The stored in-driver result at row 349 is 0/128.

### `abl-gnd25`

| Field | Record |
| --- | --- |
| Dates | 2026-07-28, 09:24–09:47 EDT |
| Init/fork | `runs/overnight-3/checkpoint_002062.pt`; seed 11 |
| Delta from reference | Historical \(\gamma=1.0\); `--lam .03 --lam-ret 1.0 --lr .00025`; `--games 1024`; no `--envs`; `--seed-fraction .9 --seed-cut 1 8 --seed-noise .1 --anneal`; `--ground-fraction .25 --ground-depth 1 --ground-time .05 --sealbot D:\SealBot`; `--cell-budget 800000 --pair-budget 8000000`; `--warm-iterations 0 --starve-limit 6`; `--iterations 1300 --checkpoint-every 100 --eval-every 25 --eval-games 64`; legacy anchor evaluation |
| Question | Measurements from seating depth-1 SealBot in 25% of collection games |
| Disposition | Stopped unfinished after metrics row 277 of 1300; opponent grounding removed on 2026-07-28; no final checkpoint or stored stop reason |

| Measurement | Artifact record |
| --- | --- |
| Run shape | Iterations 0–277; checkpoints 100 and 200 |
| Completion/cut | `f_grounded=f_seeded=f_unseeded=1.0` throughout; `seed_cut_hi` 10 at iteration 0, 508 at 249, and 512 from 251 |
| Grounded score / H | Iteration `0:0/.13523`, `23:.046875/.22404`, `49:.074219/.27545`, `99:.136719/.26961`, `149:0/.46375`, `199:0/.16169`, `249:.011719/.20922`, `274:.078125/.16505`, `277:.066406/.23078` |
| Legacy-anchor evals | Metrics rows `24:.65625, 49:.875, 74:.65625, 99:.703125, 124:.625, 149:.578125, 174:.640625, 199:.796875, 224:.8125, 249:.6875, 274:.78125`; 64 games each, zero capped |
| Critic trajectory | q-loss 0.8608 at 0 → 0.4265 at 124 → 0.1229 at 149 while winner/loser \(\hat v\) reached 0.889/-0.870 and MAE 0.162; q-loss returned to 0.8099 at 199 and 0.8562 at 274 |

The grounded score was not monotone, and the retained artifact set has no completed ungrounded control arm for a causal comparison.

The separate 15-iteration landing probe measured grounded score 0.00 → 0.05, H 0.136 → 0.204, and about 1 ms per opponent turn **(run plan, not re-derived)**. Those values are not from `abl-gnd25`: this run’s iterations 0→14 measure grounded score 0→0.01171875 and H 0.135235→0.214780; its grounded score first reaches 0.046875 at iteration 23.

### `pure-1`

| Field | Record |
| --- | --- |
| Date | 2026-07-28 |
| Init/fork | `runs/overnight-3/checkpoint_002062.pt`; seed 20 |
| Delta from reference | Historical \(\gamma=1.0\) (no `gamma` key); `--lam .03`; `--games 1024`; no `--envs`; `--eval-depth 1 --eval-time .05`; no `--eval-sims`; `--iterations 2000 --checkpoint-every 100 --eval-every 25 --eval-games 64`; `--starve-limit 10`; no seeding or grounding |
| Question | Behavior of unseeded, ungrounded self-play from the trained `overnight-3` fork under \(\lambda_{\mathrm{ret}}=.939\) |
| Disposition | Stopped after metrics row 174; no stored stop criterion or `status.json`; superseded by `pure-2` |

| Iteration range | Measurements |
| --- | --- |
| 0–174 | 175 metrics rows; only checkpoint 100 is retained |
| Evaluations | After iterations `25:0/64`, `50:1/64=.015625`, `75:0/64`, `100:4/64=.0625`, `125:8/64=.125`, `150:1/64=.015625`, `175:1/64=.015625`. No `telemetry.db` is retained, so Wilson intervals and Elo are unavailable. |
| 130 | H 0.184668; winner/loser \(\hat v=0.398711/-0.164323\); q-loss 0.314766; mean game length 24.36 |
| 130–174 | H range 0.152766–0.960126; mean game length reached 196.10; maximum iteration time 610.96 s |
| 159 and 174 | At 159, winner/loser \(\hat v=0.041083/-0.010698\), q-loss 0.070411. At 174, the same measurements were 0.050525/-0.146271 and 0.209785. |

The artifact minimum `v_hat_mae` is 0.507515 at row 61. This differs from the run-plan statement that it never fell below about 0.7 **(run plan, not re-derived)**.

### `pure-2`

| Field | Record |
| --- | --- |
| Date | 2026-07-28 |
| Init/fork | `runs/pure-1/checkpoint_000100.pt`; seed 21 |
| Delta from reference | Historical \(\gamma=1.0\); `--lam .03 --lam-ret 1.0`. Initial branch: `--games 1024`, no `--envs`, `--iterations 2000 --checkpoint-every 100 --eval-depth 1 --eval-time .05`, no `--eval-sims`. Resume from checkpoint 100: `--games 4096 --envs 1024`, depth-1 evaluation, checkpoint interval 100. Resume from checkpoint 200: `--games 4096 --envs 1024 --checkpoint-every 25 --eval-time .1 --eval-sims 32`, uncapped search evaluation. All branches used `--eval-every 25 --eval-games 64 --starve-limit 10`. |
| Question | Whether the pure-self-play direction seen in `pure-1` reproduced after a checkpoint-100 fork, and how it behaved at the reference operating point |
| Disposition | Latest branch stopped after completed iteration 239; `status.json` records iteration 240 incomplete in collection/fit. Checkpoint 200 became the common fork for the \(\gamma\) conversion arms; superseded by `conv-disc`. |

The append-only metrics contain 344 physical rows but 240 distinct iteration numbers because resumed branches overlap. `telemetry.db` represents the latest branch for iterations 100–239.

| Completed iteration | Evaluator | Score | Wilson 95% CI | Elo 95% CI | Seat wins | Provenance |
| ---: | --- | ---: | --- | --- | --- | --- |
| 25 | depth 1, .05 s | 6/64 = 0.09375 | — | — | — | metrics only |
| 50 | depth 1, .05 s | 1/64 = 0.015625 | — | — | — | metrics only |
| 75 | depth 1, .05 s | 9/64 = 0.140625 | — | — | — | metrics only |
| 100 | depth 1, .05 s | 30/64 = 0.46875 | — | — | — | metrics only |
| 125, initial branch | depth 1, .05 s | 20/64 = 0.3125 | — | — | — | metrics only |
| 150, initial branch | depth 1, .05 s | 29/64 = 0.453125 | — | — | — | metrics only |
| 125, resumed branch | depth 1, .05 s | 41/64 = 0.640625 | 0.518206–0.747118 | 100.422 [12.657, 188.188] | 22/19 | `eval_matches` |
| 150, resumed branch | depth 1, .05 s | 40/64 = 0.625 | 0.502502–0.733342 | 88.739 [1.738, 175.741] | 17/23 | `eval_matches` |
| 175 | depth 1, .05 s | 49/64 = 0.765625 | 0.648665–0.852502 | 205.642 [106.520, 304.764] | 23/26 | `eval_matches` |
| 200 | depth 1, .05 s | 54/64 = 0.84375 | — | — | — | metrics only; resume cleanup removed the match row |
| 225, older branch | depth 1, .05 s | 46/64 = 0.71875 | — | — | — | metrics only |
| 225, resumed branch | uncapped, .1 s, 32 simulations | 44/64 = 0.6875 | 0.566074–0.787691 | 136.969 [46.183, 227.755] | 21/23 | `eval_matches`; zero forfeits |

| Iteration range | Measurements |
| --- | --- |
| 199 | Eval 0.84375; H 0.329250; mean length 54.41; q-loss 0.863650 |
| Latest replay, 200–239 | H 0.333506 → 0.499105; mean length 48.84 → 71.01; q-loss 0.843716 → 0.528065 |
| 238 | H 0.559391; mean length 82.843; iteration time 548.227 s |
| Telemetry 235–239, games lasting at least 100 plies | 146,538 plies; mean \(\lvert\hat v\rvert=0.909462\); winner/loser \(\hat v=0.856889/-0.845826\); policy top-1 probability 0.105932; mean legal-action count 2660.69 |

The direction reproduced across the two stored branches, but the numerical trajectories were not identical. This is a discrepancy with the run plan’s word “identically.”

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

## Provenance

| Source | Use in this record |
| --- | --- |
| `metrics.jsonl` | Legacy evaluation rows and iteration trajectories, including append-only/resumed branches |
| `telemetry.db` | `eval_matches` score, win rate, Wilson interval, Elo, seat scores, capped count, and forfeits per opponent; `iterations` trajectories; `plies` bucket queries. Every database was opened with a read-only URI, `mode=ro`, followed by `PRAGMA query_only=ON`; the five quantized ply scalars were divided by 10,000. |
| `sealbot_curve.jsonl` | Offline `overnight-3` depth-1 checkpoint scores, Wilson intervals, Elo, seat scores, and mean plies |
| `config.json`, `invocations.jsonl`, `status.json`, adjacent logs/errors | Forks, exact flags, branch boundaries, dates, completion state, and stop-sentinel evidence |
| `KLENT_RUN_PLAN.md` §§2–4a | Historical claims and experiment criteria not encoded in the run artifacts. Such measurements are labeled **(run plan, not re-derived)**. |
| Manager measurements dated 2026-07-29 | `factored-939-s2` report, ply-bucket decomposition, and critic ranking-stability probe |
| Manager measurements dated 2026-07-30 | The three arms' shared 96-position critic probe, the bipolar head's logit and mass quantities, the fp32 expectation excursion counts, and the segmented-reduction reproduction |
| `runs/grafts/ckpt151-{brm,duel,tail,joint,joint-brm}.json` | Each arm's graft transform, probe, and preservation measurements, written by the conversion itself |
| Externally served strix checkpoint | Offline anchor measurements and the seat-reported name, digest, variant, and restriction; the checkpoint and serving environment are outside this tree and not version-pinned here |
| `python/mantisnet/README.md`, “Performance” and “Deliberately absent” | Engineering benchmark receipts and retained/removed implementation dispositions |

Every evaluation score was compared with `eval_matches` when the run retained a database row. No score mismatch was found. Resume cleanup left metrics-only evaluations at `pure-2` iteration 200 (0.84375), `conv-disc-lam01` iteration 50 (0.703125), and `factored-939` iteration 50 (0.8125); those rows consequently have no retained database CI or Elo.

Artifact limits relevant to verification:

| Scope | Limit |
| --- | --- |
| `shakeout-1`, `sweep-a`, `sweep2-a`, `sweep2-b`, `abl-zeroq-lam01`, `overnight-1`, `overnight-2`, `overnight-3`, `abl-gnd25` | No retained `telemetry.db`; Wilson intervals/Elo and ply buckets cannot be re-derived. All nine lack `status.json`; only `abl-gnd25` has adjacent log/error files. |
| `sweep2-a` | Four metrics rows only; no evaluation, checkpoint, run-specific log, status, or telemetry record |
| `pure-1` | No retained `telemetry.db` or `status.json`; evaluation counts are metrics-only and the stop criterion is not recorded. |
| `runs/grafts/` | The cited checkpoint conversions have transform manifests; grafts are not training runs and carry no run metadata. |
| Engineering microbenchmarks | No raw benchmark-output files are retained in the cited source set; the README and run plan are the available receipts. |

Artifact/run-plan discrepancies are stated in the affected reports. They include the `shakeout-1` rounding and critic summaries; `overnight-3` peak/range summaries; the separate grounding-probe values being unlike `abl-gnd25` rows 0–14; `pure-1`’s minimum MAE; `pure-2`’s non-identical repeated trajectory; and unstated aggregation windows for `conv-rho1`, `lam-ret-939`, and `factored-939`.
