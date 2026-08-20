# Ablation and experimental record

Reference recipe: γ = 0.99, λ = 0.01, τ = 0.1, λ_ret = 0.939, scalar tanh critic; 4096 completed games per iteration, 1024 environments, ply cap 512,
learning rate 1e-3. Unless a run says otherwise, an evaluation is 64 seat-balanced games against uncapped SealBot at 0.1 s/move with the 32-simulation
Gumbel line search; one costs 69.6 s on an RTX 4070 Ti (57.2 s at zero simulations), and a 64-game strix evaluation at `mcts:sims=32` costs 105.8 s.
Training-metric iteration numbers are the zero-based values in `metrics.jsonl`; `eval_matches.iteration` is the completed-iteration count. `H` is
`acting_norm_entropy`; "decided" means |v̂| ≥ 0.5. Runs whose last artifact predates 2026-07-29 live under `runs/archive/<name>/`, artifacts intact.

## Training runs

**Archived exploratory runs (2026-07-27 – 2026-07-28).** Eleven runs precede the gamma-conversion arms, all in the γ = 1.0 seeded/grounded era:
`shakeout-1` (loop stability, 100 iterations); `sweep-a`, `sweep2-a`, `sweep2-b` (packed-batch λ = .03/.03/.01 arms, stopped at iterations 10, 3, 9);
`abl-zeroq-lam01` (zero-initialized Q output, stopped at 7); `overnight-1`/`overnight-2` (15- and 30-iteration warm phases, collapse by iterations 23
and 39); `overnight-3` (2,062 iterations across four branches; checkpoint 2062 is the later fork parent); `abl-gnd25` (depth-1 SealBot seated in 25% of
collection, stopped at 277 of 1,300; opponent grounding removed as a training input); `pure-1` (stopped at 174); `pure-2` (reached 239; checkpoint 200
is the fork for the gamma-conversion arms). γ = 1.0 with seeding produced a recurring high-initial-Q then empty-buffer sequence; longer warm phases
delayed it without removing it. Offline `overnight-3` curve against depth-1 SealBot, 64 games each: 0/64 at checkpoints 250, 750, 1000, 1250, 1500,
1750 and 2062; 1/64 at 500; 2/64 at 2000; mean plies 14.9–23.7.

### Gamma-conversion arms — all fork `runs/pure-2/checkpoint_000200.pt` at seed 21; evaluations are 64 games each, zero forfeits

| Run | Date | Delta from reference | Evaluations | State |
| --- | --- | --- | --- | --- |
| `conv-disc` | 2026-07-29 | `--lam .03 --lam-ret 1.0`; 100 iterations | 25: 39/64 = 0.609375; 50: 40/64 = 0.625; 75: 47/64 = 0.734375; 100: 37/64 = 0.578125 | Completed 100 iterations; γ = .99 kept for later arms |
| `conv-rho1` | 2026-07-29 | `--gamma 1.0 --lam 0 --tau .13 --lam-ret 1.0` | 25: 37/64 = 0.578125; 50: 40/64 = 0.625 | Stopped after iteration 50; shelved |
| `conv-disc-lam01` | 2026-07-29 | `--lam-ret 1.0` | 25: 48/64 = 0.75; 50: 45/64 = 0.703125 (metrics only); 75: 52/64 = 0.8125; 100: 50/64 = 0.78125; 125 and 150: 48/64 = 0.75; mean 0.7578125 | Stopped after checkpoint 151 |
| `lam-ret-939` | 2026-07-29 | None; forks `conv-disc-lam01/checkpoint_000151.pt`; 50 iterations | 25: 52/64 = 0.8125; 50: 56/64 = 0.875 | Completed; adopted as the scalar reference recipe |

### Critic and decoder arms — all fork a graft of `conv-disc-lam01/checkpoint_000151.pt` in `runs/grafts/` at seed 21; `lam-ret-939` iterations 0–49 are that fork under the scalar critic and serve as the control

| Run | Date | Delta from reference | Evaluations (64 games each) | State |
| --- | --- | --- | --- | --- |
| `factored-939` | 2026-07-29 | `--critic-factors 2`; factored sign/magnitude critic | 25: 52/64 = 0.8125; 50: 52/64 = 0.8125 (metrics only); 75: 35/64 = 0.546875 | Stopped after iteration 78; factored critic shelved |
| `factored-939-s2` | 2026-07-29 | `--q-scale 2.0` with the factored critic | 25: 40/64 = 0.625 | Stopped near iteration 38 by `STOP` sentinel; shelved. H 0.208 → 0.312 → 0.381 at iterations 1/24/38 |
| `brm-939` | 2026-07-30 | Bipolar return-mass critic, Q = u⁺ − u⁻, η = 0.25; 50 iterations | 25: 56/64 = 0.875; 50: 56/64 = 0.875 | Completed 50 iterations; 1.83 h |
| `duel-939` | 2026-07-30 | Dueling critic with a per-position value readout; order-preserving graft; 50 iterations | 25: 52/64 = 0.8125; 50: 53/64 = 0.828125 | Completed 50 iterations; 3.75 h |
| `tail-939` | 2026-07-30 | Private critic tail; the cell heads no longer share the incidence aggregation | 25: 53/64 = 0.828125; 50: 57/64 = 0.890625 (metrics only) | Completed 50 iterations, continued toward 150; 1.73 h |
| `joint-939` | 2026-07-30 | Joint decoder class, 93 classes where the slot class gave 3; +23,040 parameters; `MODEL_REPR_VERSION` 1 → 2 | 25: 50/64 = 0.78125; 50: 49/64 = 0.765625 | Ran iterations 0–49 to its configured limit |
| `joint-brm-939` | 2026-07-30 | Joint decoder and bipolar return-mass critic together; target 500 iterations | 25 through 250 by 25: 0.828125, 0.84375, 0.96875, 0.90625, 0.875, 0.984375, 0.90625, 0.9375, 0.890625, 1.000; zero capped, zero forfeits | In flight past iteration 270 |

One standard error on a 128-game two-evaluation total is about four games and a difference between two arms about 5.8; all five arms lie within 1.6 of
those standard errors of the control. Over the four evaluations `joint-brm-939` shared with the control, the control scored 0.8125, 0.875, 0.9219 and
0.9219 and won 226 of 256 games against 227 of 256.

Paired head-to-head: each arm's iteration-50 checkpoint against the control's, converted by that arm's own graft; 64 shared 2–6 ply
openings played from both seats, 32 simulations, τ = 0.1, λ = 0.01, ply cap 512, seed 21.

| Arm | Score of 128 | Pairs arm / split / control | Sign test | Mean plies |
| --- | ---: | ---: | ---: | ---: |
| `joint-939` | 84.0 = 65.6% | 26 / 32 / 6 | 0.0005 | 72.8 |
| `tail-939` | 73.0 = 57.0% | 19 / 35 / 10 | 0.136 | 74.7 |
| `brm-939` | 49.0 = 38.3% | 10 / 29 / 25 | 0.0167 | 74.6 |
| `duel-939` | 128.0 = 100% | 64 / 0 / 0 | — | 20.4 |

No game reached the ply cap. `duel-939`'s row is not a strength measurement: its graft is order-preserving rather than function-preserving, so its
opponent is the control with its value level removed.

**Later runs**, artifacts in their run directories: `tri-939` (critic-arm follow-on from the checkpoint-151 lineage), `joint-mnorm` (joint decoder
under mass-normalized acting), `deep6-mnorm150` (depth-6 window probe), `stack-939` (pre-campaign production reference: full reference recipe, seed 21,
483 iterations), `newmodeltest` (cell-latents arm C, paused at iteration 31), `cellnodes-1` (cell latents plus cell nodes at scope `all`; carry run,
stopped at iteration ~400), `tactical-1` (cell latents, cell nodes, tactical scalars; seed 22; from the Step 5/6 arm-B seed-2 cell; stopped at iteration 149 on 2026-08-19).

## Engineering

| Item | Measurement | State |
| --- | --- | --- |
| Fused Triton block attention | 1.01×–1.05× over dense at 50/200/400 stones and cohorts 256/1024; peak allocated 2.32 → 2.05 GiB at 400 stones; fit throughput −0.9% | Retained for supported collection shapes; dense SDPA elsewhere |
| Shared decoder aggregation, Triton segment reduction | 1.39×–1.82× over the same grid; peak allocation 3.04 → 1.96 GiB at 50 stones/cohort 1024; gather/scatter 29.1% → 2.8% of total at 50 stones; fit 7,871 → 9,346 samples/s | Retained |
| Loop pipelining (collection/fit overlap) | ~6.3k samples/s at the 1,024-game operating point, 2.3 s collection cadence, 90–98% GPU utilization, ~5 GiB resident, ~1,540 rounds/hour | Retained |
| Auto-reset cohort collector | 144k samples in 59.88 s cold and 17.55 s warm on the fused path: 2.42k and 8.16k samples/s, against a 671 samples/s baseline | Retained |
| VRAM budget packing | Near ply 500 at ~5,500 legal cells per sample: collection 0.36 GiB, fit ~2.9 GiB, ~2× the unpacked path; unpacked batch probe 128 → 2.5 GiB, 256 → 5.7 GiB, 512 → 17.1 GiB with host spill ~50× slower | Retained |
| Telemetry schema v3, quantization, browse-order indexes | Five ply scalars stored as 1e−4 integers, maximum rounding error 5e−5, ~65 bytes/ply against ~78; on 575,342 games a warm lookup 484 ms → 0.05 ms and a deck-order browse 44 s → milliseconds; four indexes ~56 MB | Retained; no v2 converter, so schema-v2 databases are unreadable |
| Container-side training over the deck bind mount | A Windows-side driver fails: the deck holds `status.json` open across the bind mount and Windows `os.replace` returns a permission error | Driver runs in the Linux container |
| Strix as a second anchored evaluator | Served as a §3.1 subprocess seat through a WSL relay (its `hexo_rs` extension is a CPython 3.14 Windows build); `strix-seat`, digest `0x351ed562065bed55`, `mcts:sims=32`; declares radius-6 candidates against the rules' radius 8, search limited to 300 placements. Offline 64-game scores: `ckpt151-joint-brm.pt` 12/64, `joint-brm-939` ckpt100 21/64 and ckpt253 26/64, `joint-939` ckpt50 17/64; zero capped, zero forfeits | Retained; SealBot saturates against `joint-brm-939` after roughly iteration 75 |

## Graft campaign

Per-step records of `docs/MANTIS_GRAFT_SPEC.md`. Speed harness: WSL ext4 clone, `bench fit --corpus mnorm-late-v1 --split val --device
cuda --compile --seed 7 --steady-warmup 20 --steady-measure 50`. Step 0 baseline 205.45 samples/s median, peak 10.06 GiB; VRAM ceiling 10.25 GiB.

| Step | Content | Screen primaries, paired | Speed and VRAM | State |
| --- | --- | --- | --- | --- |
| 0 `perf-foundation` | Fused-Adam execution policy; fit batches pinned on the KLENT prefetch worker | Speed-only gate | Fused 205.45 vs foreach 205.42 samples/s, spread ±0.2%; VRAM 10.057 GiB in all six runs | Baked into main 2026-08-10; harness pinned to WSL |
| 1 `dead-key-bias` | `wk.bias` and `wk_wa.bias` removed from every block (8 tensors, 1,024 scalars); functionally exact | Parity in place of a screen | Pro forma; parameters 1,944,165 → 1,943,141 | Baked into main 2026-08-10 |
| 12 `mixed-windows` | Ternary all-nonempty window nodes; `MODEL_REPR_VERSION` 4; 2×2 factorial × 3 seeds, 400k-sample cells | B top-1 +0.553 ±1.438 pp [2+/1−], top-3 +0.941 ±1.210 [3+/0−], critic sign +1.334 ±8.355; horizon top-1 moves 33–48 +0.482 ±0.465, 49–64 +2.019 ±1.939, 65+ +1.294 ±0.772, all [3+/0−]; C −0.120; D −0.760 [0+/3−] | A 3186.0 samples/s / 3.32 GiB; B 1068.1 / 9.17; C 390.3 / 2.42; D 2388.5 / 2.42; lean-budget B 1045.8 / 4.63 | Baked into main 2026-08-11; binary path deleted |
| §5.1c cell-mediated attention | Crossing pairs enumerated through claimed cells at attention time, with no materialized pair edge list | — | 2.4–2.5× slower on both scopes (B lean 415.8 vs 1045.8 samples/s; A 1355.4 vs 3186.0); VRAM B lean 3.43 vs 4.63 GiB, A 3.17 vs 3.32, pair-density scaling cliff gone | Reverted 2026-08-11 |
| 4 `action-rows` | 729 ternary-native post-placement action classes; `MODEL_REPR_VERSION` 5; 5 seeds × arms A/B at 1 epoch, plus a 4-epoch pair at seed 5 | top-1 +0.647 ±2.401 pp [3+/2−], top-3 +0.841 ±2.200 [4+/1−]; moves 1–4 +1.408 ±1.214 [5+/0−]; critic sign −0.253 ±2.512. At 1 epoch (95 optimizer steps) cells land arm-independently in an optimist or a discriminating critic basin, 5 of 10 optimist; 4 epochs (380 steps) escape | Fit 1045.5 vs 1063.8 samples/s (−1.72%), VRAM 4.636 vs 4.629 GiB; collect 162.4 vs 185.2 (−12.29%) | Baked into main 2026-08-11; parameter pin 4,007,269, +140,672 over Step 12's 3,866,597 |
| 3 window-attention removal | No model change; arms compose the live knobs at `0875eb0` into a wattn × latents 2×2 factorial, 5 paired seeds, 4 epochs | D−A top-1 −1.927 ±0.834 pp [0+/5−], top-3 −1.688 ±1.115 [0+/5−], state MAE +1.224 ±1.369 e−2 [5+/0−]; C−B critic sign −1.279 ±1.470 [1+/4−], state MAE +0.714 ±1.682 e−2 [4+/1−]; C−D state MAE −1.905 ±2.113 e−2 [0+/5−] | — | Rejected 2026-08-12; `window_attention` stays a live knob through Step 15. Parameters C 4,538,341 = D 3,741,797 + 796,544; A − D = 265,472 |
| 2 `state-latents` | K = 4 invariant state latents replace the global token; `MODEL_REPR_VERSION` 6; 4 epochs, 5 paired seeds | top-1 +0.02 ±1.07 pp; critic sign +1.28 ±2.33 [4+/1−], moves 1–4 +1.28 ±1.19 [5+/0−]; state MAE −1.40 ±0.72 e−2 [0+/5−] | Padded reference fit −9.9%, fit VRAM +1.12 GiB, collect −14.7%; fused ragged kernels were the in-step fix, and the re-pin on the current driver was not completed | Merged to main 2026-08-17; parameter pin 4,803,813, +796,544 |
| 15 `cell-latents` | Knobs `cell_latents`/`line_pass`/`claim_reach`; the cell stage replaces the §5.1b relay; 25/25 cells, five arms × five seeds, 4 epochs | C−A top-1 +1.16 ±1.37, B−A +0.95 ±1.68, B−C −0.21 ±0.33; C−D top-1 +2.06 ±1.80 [5+/0−] and v̂ MAE +1.31 ±0.52; B−A critic sign −0.76 ±0.78 [0+/5−], worst bucket −2.4 pp at moves 25+ | Arm C trains in 25.9 min against A's 47.9, seed spread 0.70 pp against 3.01 | Merged to main 2026-08-17 |
| 13 `cell-nodes` | Legal-cell nodes with stone→cell radius-8 edges typed by the frozen 48-class D6 orbit vocabulary; `cell_node_scope` `all`/`uncovered`; `MODEL_REPR_VERSION` 7 | Prefit paired at seed 3: train loss 1.891 vs 1.990 at epoch 4, top-1 −0.60 pp, v̂ sign worse in 9/9 horizon buckets, state MAE better in 8/9. Live `cellnodes-1` strix evaluations pooled over iterations 200–375: 399/512 = 77.9% against `stack-939`'s 293/512 = 57.2% at the same anchor digest, ≈ +170 Elo, z ≈ 5.3; SealBot 64/64 at iterations 225 and 375 | Base fit 1277.5 samples/s / 4.638 GiB; `all` 1036.5 (−18.9%) / 6.274; `uncovered` 1082.5 (−15.3%) / 6.177; collect −11.4% and −12.5% | Merged to main 2026-08-17; parameters 5,196,965, +369k. The isolating comparator (`newmodeltest` resumed) has not run |
| Fit-path VRAM recompute-in-backward | Cell stage and window-latent cycle wrapped in non-reentrant checkpointing under a selective save policy; execution policy only, gradients bit-identical | Baseline peak live set: §5.1b cell stage 52.6% (3.40 GiB), §5.4 window latents 18.0% (1.16 GiB) | 6.280 GiB / 1041.1 samples/s → 3.538 GiB (−43.7%) at 1034.3 samples/s, same-session A/B; steady chunk-ms median 143.3 vs 144.0. Measured dead ends: head-fused kernel programs 766 vs 924 samples/s; blanket recompute −2.1% at 3.213 GiB | Retained |
| 11 `orbit48-bias` | The §4.1 stone-attention bias re-keyed by the 48-class D6 displacement-orbit vocabulary. Three forms, each 3 seeds at the Step 5/6 recipe on the arm-B config: plain (51-row table, `be1dda8`), residual (26-row coarse (distance, on-axis) table + zero-init 48-orbit residual, `ab6de77`), control (residual with the orbit rows frozen at zero — the old function class on the new kernel path) | Epoch-4 top-1 against the arm-B baselines 0.4284/0.4265/0.4284: plain 0.4125/0.4062/0.4208; residual 0.4050/0.4074/0.4047; control 0.4104/0.4153/0.4243; residual − control −1.1 pp [0+/3−] paired. Baseline reruns place the screen's per-run spread at ≈0.9 pp (seed 3/4 on the launch tree 0.4248/0.4318; on the pre-VRAM tree 0.4171/0.4455 — same seed differs up to 1.4 pp across gradient-bit-identical trees; the bias-table gradient's `tl.atomic_add` is the identified nondeterminism). Equivalence checks: kernels bit-exact old-vs-new (forward and dq/dk/dv 0; table grad 0.2% bf16 summation order); a baseline checkpoint transplanted into the new layout reproduces 0.4284/0.6623 exactly. Eval-time knock-out: zeroing every learned bias table moves top-1 ≤0.06 pp in all six arms including the baseline — the geometric bias channel is decorative in the trained function; all observed differences are training-dynamics effects at or near the screen's noise floor. Per-chunk fit trace at seed 0 (old path, new path with the orbit residual frozen, old path rerun; identical init digests): the paths agree to ≤3e-3 for the first ~500 chunks, then diverge chaotically — with the old-vs-old rerun diverging as far as old-vs-new (end-of-window mean policy loss 3.5062 / 3.5022 / 3.5159) — so the new path's trajectory is indistinguishable from a rerun of the old one | Same kernel cost as the incumbent (shared bucket helper + 25×25 LUT); residual +368 parameters on arm B (5,215,141) | Residual form retained on the integration tree 2026-08-19; measured under screen v2 (fixture + arms) from here. Block-3 stone↔stone bias rows are exactly zero in every trained cell, baseline included — the last block's stone self-attention output is consumed only by the latent rows |
| 5+6 `tactical-scalars` × `action-latents` | 11 deterministic per-action tactical scalars through a zero-init MLP; 2 invariant per-position action latents; `MODEL_REPR_VERSION` 8; 2×2 × 3 seeds on the production config, 4 epochs, corpus `cn1-late-v1`, 12/12 cells | B top-1 +1.15 ±0.71 pp [3+/0−], top-3 +1.73 ±0.85 [3+/0−]; C top-1 −1.34 ±2.18 [1+/2−]; D top-1 −0.81 ±2.34 [1+/2−]; C `state_value` sign at moves 1–4 +2.10 ±1.25 [3+/0−]. 3 of 6 latents cells showed policy collapse (C s0 −3.70 pp, D s2 −3.31 pp) or an optimist critic (D s0); 0 of 6 latents-off | Base fit 1041.1 samples/s / 6.280 GiB, collect 147.6; tactical fit 995.9 (−4.35%), +34 MiB, collect −0.65%; latents fit 1017.4 (−2.28%), collect −6.53%, fit VRAM +309 MiB; both knobs +344 MiB | Screen complete 2026-08-17; `action_tactical` on and `action_latents` off in the `tactical-1` run launched 2026-08-18. Parameters A 5,196,965 / B 5,215,013 / C 5,396,261 / D 5,414,309 |

## Provenance and artifact limits

Sources: `metrics.jsonl`, `telemetry.db` (`eval_matches`, `iterations`, `plies`; opened read-only, quantized ply scalars divided by 10,000),
`sealbot_curve.jsonl`, `config.json`/`invocations.jsonl`/`status.json`, the `runs/grafts/*.json` transform manifests, and the externally served strix
checkpoint, which lives outside this tree and is not version-pinned here. Archived runs retain no `telemetry.db`, so intervals, Elo, and ply buckets
cannot be re-derived for them; no raw benchmark-output files are retained for the engineering measurements. Resume cleanup left metrics-only
evaluations at `pure-2` iteration 200 (0.84375), `conv-disc-lam01` iteration 50 (0.703125), `factored-939` iteration 50 (0.8125), and `tail-939` 50.
