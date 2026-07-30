# KLENT for Hexo — normative implementation contract

This document specifies the KLENT implementation in `python/mantisnet/mantisnet/klent/`, together with the KLENT-facing contracts in `losses.py`, `segments.py`, and the model interface. It is normative in the same sense that [MODEL_SPEC.md](MODEL_SPEC.md) is normative for MantisNet. A disagreement between this document and code is a finding to raise; neither side is silently preferred.

The algorithmic source is [KLENT_PAPER.md](KLENT_PAPER.md). Measured outcomes and configuration selection are recorded only in [ABLATIONS.md](ABLATIONS.md). The paper's $(\alpha,\beta)$, respectively the entropy and reverse-KL coefficients, are this repository's $(\lambda,\tau)$; $\lambda_{\mathrm{ret}}$ below is the distinct $\lambda$-return mixing coefficient.

## 1. Placement-level MDP

### 1.1 State, action, and transition

| Term | Contract |
|---|---|
| $S_t$ | A nonterminal engine `Position` immediately before one placement, including the board, current mover $m_t$, and `moves_remaining`. |
| $A(S_t)$ | The complete legal-cell list in engine order. |
| $A_t$ | One legal cell, stored by its rank in that list. |
| Transition | Deterministic `Position.advance(q, r)`. A terminal successor ends the episode and is never evaluated. |
| Outcome | From any stored state's mover frame, $z_t=+1$ if $m_t$ is the eventual winner and $-1$ otherwise. Capped episodes have no outcome target. |

The opening consists of one placement with `moves_remaining == 1`. Later turns consist of a first placement with `moves_remaining == 2`, followed by a second placement with `moves_remaining == 1`. A win may end either placement.

### 1.2 Perspective and mover-change sign

Policy, $Q(S_t,\cdot)$, acting value $\hat v_t$, and return $G_t$ are in the frame of the mover at $S_t$. For every nonterminal transition,

$$s_t =
\begin{cases}
+1,&m_{t+1}=m_t\\
-1,&m_{t+1}\ne m_t
\end{cases}
=
\begin{cases}
+1,&\texttt{moves\_remaining}(S_t)=2\\
-1,&\texttt{moves\_remaining}(S_t)=1.
\end{cases}$$

`signs_from_moves_remaining` returns a float64 array and refuses any value outside `{1, 2}`. The sign converts a next-state mover-frame quantity into the current mover's frame; it is not derived from ply parity.

### 1.3 Returns

For a naturally terminated episode with acted-on states $0,\ldots,T$,

$$G_T=+1,\qquad
G_t=s_t\,\gamma\left[(1-\lambda_{\mathrm{ret}})\hat v_{t+1}
                  +\lambda_{\mathrm{ret}}G_{t+1}\right]\quad(t<T).$$

The last stored state precedes the winning placement, so its acting mover is the winner. No terminal position is bootstrapped. `v_hats[0]` and `signs[T]` are not read by the recursion. $\gamma$ is a per-placement magnitude; the perspective sign is carried only by $s_t$.

`lambda_returns(signs, v_hats, lam_ret, gamma)` returns float64 and refuses:

- `lam_ret` outside $[0,1]$;
- `gamma` outside $(0,1]$;
- inputs that are empty, non-1-D, or unequal in shape; or
- `v_hats` entries that are non-finite or outside $[-1,1]$ by more than the fp32 summation slack $10^{-4}$.

Both refusals name the failing entry, its value, and how many entries were non-finite versus merely outside the interval, because a non-finite value and a small excursion have different causes. The returned array is itself refused if any entry leaves that same widened interval, which bounded inputs make unreachable. The slack is not a tolerance on the mathematics: $\lvert\hat v\rvert\le\max_a\lvert Q(a)\rvert\le1$ exactly, but $\hat v$ is an fp32 sum over a position's legal cells, a saturated critic puts many of them at exactly $\pm1$, and an fp32 segment softmax sums to one only to a few ulps.

It does not separately validate sign values, finiteness, or the range of `v_hats`.

### 1.4 Placement test obligations

| ID | Obligation |
|---|---|
| K1 | Sign follows mover change, not ply parity. |
| K2 | First-placement and second-placement wins both produce correct mover-frame returns. |
| K3 | A terminal successor is never evaluated or bootstrapped. |
| K4 | A capped episode contributes no training sample. |
| K5 | $\hat v$ is the expectation under $\pi'$, not under $\pi_\theta$. |
| K6 | $\hat v$ is captured at acting time and is not recomputed during fitting. |

The two-placement Count Up test and engine-replayed first/second-placement wins are the executable obligations for this section.

## 2. Improvement operator

For each nonterminal state, the implemented objective is

$$\underset{\pi'}{\operatorname{maximize}}\quad
\mathbb E_{a\sim\pi'}[Q(S,a)]
-\tau D_{\mathrm{KL}}(\pi'\Vert\pi_\theta)
+\lambda H(\pi').$$

Paper equation 3 is implemented as

$$\log\pi_\theta=\operatorname{logsoftmax}(\ell),\qquad
\pi'=\operatorname{softmax}\left(
  \frac{Q+\tau\log\pi_\theta}{\tau+\lambda}
\right)$$

independently within every legal-action segment.

Boundary identities are part of the contract: `tau == 0` ignores the prior; constant Q gives $\pi'\propto\pi_\theta^{\tau/(\tau+\lambda)}$; and a one-action segment is a point mass with zero KL and normalized entropy.

### 2.1 Function contract

```python
improved_policy(
    policy_logits: Tensor,  # (N,)
    q_values: Tensor,       # (N,)
    offsets: Tensor,        # (P + 1,) CSR boundaries
    tau: float,
    lam: float,
) -> ImprovedPolicy
```

`policy_logits` and `q_values` are flat, aligned, and ordered by engine legal rank. `offsets` starts at zero, ends at $N$, and defines $P$ positive-width positions. The tensors share a device. Logit and Q arithmetic is fp32, and the whole operator is `no_grad`.

| Return field | Shape | Definition |
|---|---:|---|
| `probs` | `(N,)` | $\pi'$, summing to one within each segment. |
| `v_hat` | `(P,)` | $\sum_a\pi'(a)Q(a)$, divided by the segment's own probability mass. |
| `kl` | `(P,)` | $D_{\mathrm{KL}}(\pi'\Vert\pi_\theta)$. |
| `norm_entropy` | `(P,)` | $H(\pi')/\log|A(S)|$, defined as zero when $|A(S)|=1$. |

Each per-position expectation — `v_hat`, `kl`, and the entropy behind `norm_entropy` — divides by that segment's summed $\pi'$. The mass is one by definition, but an fp32 segment softmax sums to one only to a few ulps, and where every legal move shares one saturated $Q$ nothing cancels that error: it reached $1.1\times10^{-4}$ outside $[-1,1]$ on a 159-ply position, which §1.3's range check refuses. Dividing keeps $\lvert\hat v\rvert\le\max_a\lvert Q(a)\rvert$ at any segment width.
vert\le\max_a\lvert Q(a)
vert$ at any segment width.

### 2.2 Refusal boundary

The operator refuses `tau < 0`, `lam < 0`, and `tau + lam <= 0`. Shape, device, CSR, finiteness, and Q-range requirements are caller preconditions, not explicit refusals. Terminal positions are refused by the model builder, not by this function.

## 3. Critic and losses

The consumed model contract is the trunk and legal-cell decoder in [MODEL_SPEC.md](MODEL_SPEC.md): `cell_heads` returns one raw policy logit and one scalar $Q\in(-1,1)$, produced by `tanh`, for every legal cell in engine order. KLENT calls no state-value readout and applies no loss to it.

The two heads do not read one output representation. The policy decodes the trunk's rows; the critic decodes the appendix-B critic tail's, a pre-norm residual FFN over the window rows and the token that only the action-value head reads. The tail's nonlinearity does not commute with the decoder's incidence sum, so each head makes its own pass over the incidence table: the decoder aggregation and the rows it produces cost twice as much per batch, in both the fit and collection budgets of §5.1. The tail's parameters are trained by the loss below, like the rest of the action-value path.

For a fitting batch of positions,

$$L =
\frac1P\sum_{i=1}^{P}
\left[
  -\sum_{a\in A(S_i)}\pi'_i(a)\log\pi_\theta(a\mid S_i)
  +(Q_\theta(S_i,A_i)-G_i)^2
\right].$$

The policy cross-entropy covers the full legal set. The critic loss selects only the stored action rank. The two terms have unit weight.

| Check | Owner | Exact behavior |
|---|---|---|
| Policy target normalization | `policy_loss` | Accumulates each segment in float64 and refuses sums not `torch.allclose` to one with `atol=1e-4` and the default relative tolerance; NaN, infinity, and truncated targets fail. It does not separately refuse negative entries. |
| Outcome range | `value_target` | Refuses values outside $[-1,1]$. KLENT does not call this binned state-value helper. |
| Scalar return range | KLENT fitter | No explicit range or finiteness check; valid collection through tanh Q and the return recursion keeps $G$ bounded. |
| Stored-policy width | `_rebuild` | Refuses a stored $\pi'$ whose length differs from the replayed position's legal count. |

## 4. Collection and buffer

### 4.1 Evaluation seam and acting values

Collection accepts:

```python
evaluate(batch) -> (policy_logits, q_values)
```

Both outputs are flat fp32 CPU tensors in batch legal-cell order. `network_evaluate` implements the seam with `trunk` plus `cell_heads` under `no_grad`; it does not evaluate the state-value head.

For each acting position, collection computes $\pi'$, $\hat v$, KL, and normalized entropy before sampling. It renormalizes each probability segment in float64 for sampling and fp32 storage, draws one uniform per slot, advances the selected legal rank, and retains the operator's original fp32 diagnostics.

### 4.2 Episode lifecycle and cap rule

- `Collector` owns a fixed cohort of persistent slots.
- Every new episode starts from `Position()`, the empty board; collection has no seeded-prefix or alternate-start-state interface.
- Finished and capped slots reset immediately. A collection call returns at least its ended-game quota because all endings in the final lockstep step are included.
- Unfinished games remain in their slots across collection calls. One completed episode may therefore contain acting records made under successive weight snapshots.
- The pre-action state for the winning placement is a sample; the terminal successor is not.
- `episode_samples` returns an empty list for a capped episode, dropping its entire prefix.

A sanctioned cold start is completed outside the KLENT loop and enters a new run through `--init-from`. There is no in-loop seeding, grounding, curriculum, or prefit phase.

### 4.3 Sample storage and replay

Each sample is `(moves, t, rank, improved, g)`: the completed episode's move tuple, the prefix length defining $S_t$, the action's legal rank, the full legal-set $\pi'$, and $G_t$. `moves[:t]` is replayed in parallel during fitting; board objects and model observations are not stored.

An iteration's training list contains every placement of every naturally terminated episode returned by that call and no placement from a capped episode. The list is in memory, is traversed for one fitting epoch, and is then discarded. Persistent environment slots are not a persistent sample replay buffer.

## 5. Fitting and pipelining

### 5.1 Epoch and memory packing

`fit` traverses every supplied sample exactly once. It randomizes sample indices, sorts them by descending prefix length for packing with random tie-order, randomizes the resulting chunks for optimizer grouping, and uses Adam.

Each forward chunk is packed under three configured limits:

- position count `batch_size`;
- padded attention pairs $B(\max(t)+1)^2$; and
- total legal-cell decoder rows.

Collection uses separate pair and cell budgets and a derived position-count pipeline cap. Budget numbers are configuration, not constants of this contract. A single indivisible position that exceeds a pair or cell budget is retained as a singleton; therefore the budgets are hard peak caps only when each individual position fits them.

Fit chunks accumulate sample-weighted gradients until an optimizer group reaches or crosses `batch_size`; a group may exceed it, and the final group may be smaller. The gradient is the mean loss over the whole group. Preparation of the next replayed chunk runs one chunk ahead on a CPU worker.

Every optimizer step is followed by one fused finiteness check over the parameters. A non-finite weight makes every later evaluation non-finite, and without the check the first refusal comes from the return recursion one collection later, naming neither the step nor the tensor; the check refuses at the step, naming both.

### 5.2 Outer-loop ordering

The first processed iteration after process start or resume collects and fits sequentially. Thereafter, while fit $i$ mutates the live model, collection for corpus $i+1$ runs through a snapshot taken immediately before fit $i$. The steady-state corpus is therefore one fit behind the paper's strict collection-then-fit alternation. GPU calls remain serialized by the shared lock.

## 6. Evaluation

### 6.1 Gumbel sequential-halving line search

Search is evaluation-only: collection and fitting do not import it. `gumbel_choose(evaluate, tau, lam, sims)` is a batched line search, not a tree.

- `sims == 0` returns raw-policy-logit argmax and consumes no RNG.
- A positive budget forms `min(16, sims // 2, legal_count)` root candidates from raw logits plus Gumbels; budgets below two fall back to policy argmax.
- Sequential-halving rounds deepen surviving lines without exceeding `sims` neural expansions per root.
- Interior actions are improved-policy argmax. Nonterminal leaf values are $\hat v$, signed into the root mover's frame; terminal values are exact $\pm1$.

### 6.2 Anchored opponent and RNG

SealBot is the anchored external opponent. Its recorded identity is `sealbot` plus variant, per-turn time limit, and optional depth limit; the checkout commit and build identity are not version-pinned.

Every match, in-driver or offline, plays one seat-paired schedule: `games / 2` uniform-random openings, each replayed from both seats, so a match requires an even game count of at least two. An opening is one to ten placements, two through six by default; the bound is what makes an opening nonterminal, because at ten placements the leading player owns five stones. The ply cap counts the opening's placements. Caps score one half. A second consecutive unplayable SealBot proposal after one retry is a forfeit scored as a model win.

The in-driver evaluation generator is derived from `(run seed, completed iteration)` and never from the training generator. Enabling evaluation therefore does not consume training RNG state.

### 6.3 Checkpoint crossplay

Crossplay evaluates every unordered pair of sorted run checkpoints once, using raw-policy argmax for both. Pair RNG derives from `(seed, index_a, index_b)` and draws that pairing's openings; caps score one half. Both choosers are deterministic, so the openings are the whole source of a pairing's diversity and the game count is the count of distinct games. `crossplay.json` and the telemetry crossplay table are replaced wholesale by each invocation.

### 6.4 Paired head-to-head

A head-to-head compares two checkpoints from any two run directories directly, and is the resolution instrument: two independent anchored scores of sixty-four games cannot separate less than about eight percentage points, while the pairing below reports a standard error next to the unpaired one it replaces.

`pairs` openings are each played twice with the seats swapped, both models searching at the same `sims`, and a pair is one unit — it shares its opening and its whole generator, derived from `(seed, pair index)`, and is reproducible alone. Per pair, `d` is A's wins minus one and lies in `{-1, 0, +1}`; a one-one split is the seat-advantage component the pairing removes. The result states A's score with its marginal Wilson interval, the win/split/loss pair counts, the paired and unpaired standard errors of the same estimand, an exact two-sided binomial sign test over the decisive pairs, Elo with an interval from the paired standard error, the seat split, and the capped count.

A cap is not a decision: a capped pair is counted apart from the win/split/loss counts, excluded from the sign test, and named in the result's warnings. Elo bounds that reach a zero or unit score are unbounded and are recorded as absent rather than as infinities. Pairs that all carry one `d` — the shape an all-splits match takes — have no spread to estimate, so such a match has no standard-error ratio and no Elo interval either, and names that degeneracy in its warnings. A ply cap must exceed the longest opening, or neither model would move.

The two checkpoints must agree on `RULES_VERSION`, `ACTION_ORDER_VERSION`, and the Torch version, for which no conversion exists, and on `MODEL_REPR_VERSION`, for which `klent.graft` is the bridge. The output names both checkpoints' SHA-256, versions, and iteration, and the seed, sims, coefficients, pairs, opening range, ply cap, and device the match ran under; an unsearched match records absent coefficients, because it never consults them.

## 7. Run directory contract

A fresh CLI run requires an absent or empty `--out`. A nonempty directory requires `--resume`, and resume requires at least one `checkpoint_*.pt`.

| Artifact | Guarantee |
|---|---|
| `config.json` | Replaced on every invocation with the current resolved KLENT/model settings, target iteration, checkpoint/evaluation settings, seed, init source, SealBot path, and versions. It omits `starve_limit`. |
| `invocations.jsonl` | Appends every resolved invocation, including start iteration, `starve_limit`, changed resume settings, KLENT configuration, and versions. |
| `metrics.jsonl` | Appends and flushes one strict-JSON row per executed iteration; NaN statistics become `null`. Resume does not prune a tail beyond the restored checkpoint, so superseded or duplicate iteration numbers may remain. |
| `status.json` | Atomically replaced heartbeat with exactly `updated`, `iteration`, `collect`, `fit`, and `eval`; idle lanes are `null`. A clean, STOP, or starvation return clears the lanes. |
| `telemetry.db` | Schema-versioned WAL database. Each iteration's row, games, plies, and hardware aggregates commit in one transaction; evaluation and crossplay are also stored. Resume removes driver/self-play rows at and beyond the restored iteration before replay. |
| `checkpoint_NNNNNN.pt` | Atomic write containing model, optimizer, completed-iteration count, main NumPy RNG state, and versions. `NNNNNN` is the completed count. |
| `CHECKPOINT` | At the next iteration commit, forces a checkpoint, is consumed, and training continues. |
| `STOP` | After the current iteration, forces a checkpoint, is consumed, and returns normally; a starvation return takes precedence. |

Telemetry stores per-ply $\hat v$, KL, normalized entropy, top probability, and chosen probability quantized to $10^{-4}$, but not the complete $\pi'$. `inspect_position` can recompute a specified checkpoint's policy for a stored prefix; this is not a guarantee that the checkpoint equals the acting snapshot.

`metrics.jsonl["iteration"]` is the zero-based loop index. Checkpoint names and contents, `status.json["iteration"]`, and driver evaluation records use the completed count, one greater than that row index.

### 7.1 Resume, initialization, and refusal

- Resume loads the lexically latest checkpoint and restores model, optimizer, main NumPy RNG, and completed count. It permits changed invocation settings, recorded in `invocations.jsonl`.
- Collector RNG state, live slots, and unfinished episodes are not checkpointed; resumed collection starts with empty slots.
- `--init-from` version-checks and restores model plus optimizer into a fresh run, but not iteration or RNG state. The new invocation's learning rate overwrites the source optimizer's stored rate.
- Checkpoints require exact equality of `MODEL_REPR_VERSION`, `RULES_VERSION`, `ACTION_ORDER_VERSION`, and Torch version. Model loading constructs default `MantisConfig`; incompatible shapes fail strict state-dict loading.
- `telemetry.db` independently refuses a schema-version mismatch.

A checkpoint trained before the critic tail existed is converted by `python -m mantisnet.klent.graft OLD.pt NEW.pt --tau T --lam L --manifest OUT.json`, which is the only path between the two formats; the loaders above stay strict. The conversion:

- adds the tail's keys from a fresh model seeded by one recorded constant, rewrites no parent tensor, and refuses a parent that differs from the build in any other key, that already holds the tail, that is not in the build's parameter order, or whose versions are not the build's;
- remaps Adam state by parameter name, not by position: a shared parameter keeps its moments and its step, and each added parameter arrives with zero moments and a zero step, which is the state Adam would build for it on its first step; Adam's state dict is sparse: a parameter it never stepped — the state-value head KLENT does not train — has no entry, and the conversion carries that absence rather than inventing moments for it.
- preserves `iteration`, `rng_state`, and `versions`, and requires `--tau` and `--lam` because it measures $\pi'$ at that operating point;
- compares every parent tensor it is about to write, and every one the grafted network then reads, against the source file read a second time, bit for bit, records their count and SHA-256 in the manifest, and refuses a mismatch. This is the half of the property that covers the trunk, the embeddings, and the state-value head: the probe below reads the grafted trunk on both sides of its comparison and never reaches the state-value head;
- measures the grafted network against the parent's own readout on a fixed seeded probe set of nonterminal positions and writes those measurements to `--manifest`. The tail arrives with the zero output layer of [MODEL_SPEC.md](MODEL_SPEC.md) appendix B, so the conversion preserves the parent's action values exactly; a probe that finds otherwise is a failure that writes no checkpoint and no manifest.

## 8. Per-iteration metrics

Acting means cover every position evaluated during the collection call, including capped episodes and unfinished slots. Outcome-conditioned statistics cover only naturally terminated episodes returned by the call.

| Field | Definition |
|---|---|
| `iteration` | Zero-based corpus/fit index. |
| `acting_kl` | Mean $D_{\mathrm{KL}}(\pi'\Vert\pi_\theta)$ over acting positions. |
| `acting_norm_entropy` | Mean $H(\pi')/\log|A_{\mathrm{legal}}|$; one-action positions contribute zero. |
| `f` | Naturally terminal episodes divided by all ended episodes returned, including caps in the denominator. |
| `won_length_mean` | Mean placement count of naturally terminal episodes. |
| `p0_win_rate` | Player-0 wins divided by naturally terminal episodes. |
| `first_stone_win_rate` | Fraction of naturally terminal episodes whose winning placement was acted with `moves_remaining == 2`. |
| `v_hat_winner_mean` | Mean acting-time $\hat v$ where the acting mover is the eventual winner. |
| `v_hat_loser_mean` | Mean acting-time $\hat v$ where the acting mover is the eventual loser. |
| `v_hat_mae` | Mean $|\hat v-z|$ on naturally terminal episode plies, with mover-frame $z\in\{-1,+1\}$. |
| `buffer_samples` | Training samples after whole-dropping capped episodes. |
| `policy_loss` | Sample-weighted mean full-legal-set cross-entropy over the epoch. |
| `q_loss` | Sample-weighted mean taken-action $(Q-G)^2$ over the epoch. |
| `fit_steps` | Number of optimizer groups stepped. |
| `seconds` | Driver interval covering the collection wait and fit; evaluation is excluded. |
| `eval_score` | Evaluation score per game: win or opponent forfeit 1, cap $1/2$, loss 0. |
| `eval_capped`, `eval_games` | Capped-game count and total games in that evaluation. |
| `eval_seconds` | Wall time around evaluation and its telemetry recording. |

`policy_loss`, `q_loss`, and `fit_steps` are absent when the buffer is empty. Evaluation fields are absent when no evaluation runs. Empty conditional statistics are serialized as `null`. Telemetry additionally derives samples/second, game and ply counts, and per-iteration hardware means and maxima.

## 9. Deviations from the paper

### 9.1 Placement granularity and sign

**Paper:** One game action is one MDP step in games whose mover alternates per action. **Here:** One placement is one step, and $s_t$ follows `moves_remaining`. **Grounds:** Hexo places two stones per turn after its one-stone opening. **Measured outcomes:** See [ABLATIONS.md § Training runs](ABLATIONS.md#training-runs).

### 9.2 Return time constant

**Paper:** $\lambda_{\mathrm{ret}}=e^{-1/8}$ per game action. **Here:** $\lambda_{\mathrm{ret}}=0.939$, the rounded $e^{-1/16}$, per placement. **Grounds:** Two Hexo placements constitute one ordinary turn. **Measured outcomes:** See [ABLATIONS.md § `lam-ret-939`](ABLATIONS.md#lam-ret-939).

### 9.3 Return discount

**Paper:** Terminal returns are undiscounted with $\gamma=1$. **Here:** The reference recipe uses $\gamma=0.99$ as a per-placement magnitude. **Grounds:** At $\gamma=1$, terminal distance does not change return magnitude. **Measured outcomes:** See [ABLATIONS.md § `conv-disc`](ABLATIONS.md#conv-disc) and [§ `conv-disc-lam01`](ABLATIONS.md#conv-disc-lam01).

### 9.4 Entropy coefficient

**Paper:** The practical entropy coefficient is $0.03$. **Here:** The reference recipe uses $\lambda=0.01$. **Grounds:** The repository selects reference coefficients separately for Hexo. **Measured outcomes:** See [ABLATIONS.md § `conv-disc-lam01`](ABLATIONS.md#conv-disc-lam01).

### 9.5 Initial states and cold-start boundary

**Paper:** Self-play initializes the game's initial state inside each episode. **Here:** Every episode starts empty with no seeded prefix; any prefit finishes before KLENT and enters through `--init-from`. **Grounds:** The Hexo initial state is the empty board. **Measured outcomes:** See [ABLATIONS.md § `pure-1`](ABLATIONS.md#pure-1) and [§ `pure-2`](ABLATIONS.md#pure-2).

### 9.6 Output initialization

**Paper:** Policy and action-value networks start from random initialization. **Here:** The policy and scalar-Q output layers initialize exactly to zero. **Grounds:** The remaining model parameters retain their configured initialization. **Measured outcomes:** See [ABLATIONS.md § `abl-zeroq-lam01`](ABLATIONS.md#abl-zeroq-lam01).

### 9.7 Capped episodes

**Paper:** The reference collection runs fixed transition blocks and NaN-masks an unfinished tail. **Here:** A 512-placement cap marks an episode outcome-less and drops all its samples. **Grounds:** An unfinished Hexo episode has no terminal $\pm1$ target. **Measured outcomes:** See [ABLATIONS.md § Auto-reset cohort collector](ABLATIONS.md#auto-reset-cohort-collector).

### 9.8 Replay lifetime and fitting epochs

**Paper:** Algorithm 1 clears its buffer each outer iteration, while Appendix P calls stored improved policies a replay buffer “for reuse” without specifying epoch count or cross-iteration retention. **Here:** One in-memory corpus is traversed once and discarded. **Grounds:** The fitter owns no persistent sample store. **Measured outcomes:** See [ABLATIONS.md § Training runs](ABLATIONS.md#training-runs).

### 9.9 Collection/fit ordering

**Paper:** Self-play collection completes before its fitting phase begins. **Here:** In steady state, corpus $i+1$ is collected from weights snapshotted before fit $i$. **Grounds:** The driver overlaps collection with fitting after the first processed iteration. **Measured outcomes:** See [ABLATIONS.md § Loop pipelining](ABLATIONS.md#loop-pipelining-collectionfit-overlap).

### 9.10 Cross-boundary episodes

**Paper:** An episode is completed in the self-play phase before its return enters the buffer. **Here:** Unfinished episodes persist across collection calls and may contain acting records from multiple snapshots. **Grounds:** Collector slots auto-reset only when an episode ends or reaches the cap. **Measured outcomes:** See [ABLATIONS.md § Auto-reset cohort collector](ABLATIONS.md#auto-reset-cohort-collector).

### 9.11 Collection stopping rule

**Paper:** Collection stops when the sample buffer reaches a predefined capacity. **Here:** Collection stops after at least a configured number of ended episodes, so sample count is variable. **Grounds:** Capped episodes count toward the ended-game quota but contribute zero samples. **Measured outcomes:** See [ABLATIONS.md § Auto-reset cohort collector](ABLATIONS.md#auto-reset-cohort-collector).

### 9.12 Evaluation search

**Paper:** Main anchored evaluation uses a deterministic reactive policy without search; separate appendices use Gumbel MCTS. **Here:** In-driver evaluation uses a configurable Gumbel sequential-halving line search, with 32 simulations in the reference recipe. **Grounds:** Search is imported only by evaluation entry points. **Measured outcomes:** See [ABLATIONS.md § SealBot as anchored evaluator](ABLATIONS.md#sealbot-as-anchored-evaluator).

### 9.13 Anchored evaluator and match protocol

**Paper:** Anchored evaluation uses 1,024 games against a fixed pretrained Pgx checkpoint. **Here:** The reference recipe uses 64 seat-paired games from shared random prefixes against external SealBot. **Grounds:** SealBot exposes a separate rules state and alpha-beta chooser through the opponent seam. **Measured outcomes:** See [ABLATIONS.md § SealBot as anchored evaluator](ABLATIONS.md#sealbot-as-anchored-evaluator).

### 9.14 Model substrate

**Paper:** Experiments use a fixed-action ResNetV2 with policy and action-value heads and no state-value head for KLENT. **Here:** KLENT consumes MantisNet's ragged legal-cell policy/scalar-Q decoders; its model container has a state-value head that KLENT neither calls nor trains. **Grounds:** Hexo's board and legal-action count are variable. **Measured outcomes:** See [ABLATIONS.md § Shared decoder aggregation](ABLATIONS.md#shared-decoder-aggregation-and-triton-segment-reduction) and [§ Critic ranking-stability probe](ABLATIONS.md#critic-ranking-stability-probe).

### 9.15 Critic output representation

**Paper:** Separate policy and action-value heads read one shared backbone's output representation; appendix H's alternatives are a single head over that representation and two fully separate backbones at twice the computation. **Here:** The heads stay separate and the action-value head reads a critic-private tail over the trunk's output — one pre-norm residual FFN over the window rows and the token, specified in [MODEL_SPEC.md](MODEL_SPEC.md) appendix B — while the policy reads the trunk's rows. **Grounds:** One private block over a shared trunk lies between the paper's two appendix-H alternatives, and it costs a second decoder incidence pass rather than a second backbone. **Measured outcomes:** See [ABLATIONS.md § Training runs](ABLATIONS.md#training-runs).

## 10. Reference configuration

The current repository reference recipe is:

| Parameter | Value |
|---|---:|
| $\gamma$ | `0.99` |
| $\lambda$ | `0.01` |
| $\tau$ | `0.1` |
| $\lambda_{\mathrm{ret}}$ | `0.939` |
| Critic | scalar tanh Q on a private critic tail |

These are configuration facts recorded in [ABLATIONS.md](ABLATIONS.md), which also records their selection.

**Finding:** `KlentConfig` and the CLI currently default to $\gamma=1.0,\lambda=0.03$; the reference recipe therefore requires explicit flags, and `test_klent_run.py` pins the `0.03` CLI default.

## 11. Open questions

- What simulator-evaluation, wall-clock, and seed budget defines a faithful-scale Hexo run?
- How many fitting epochs per corpus should be assumed when the paper does not specify an epoch count?
- Should the current derived live-window observation be ablated against a rules-minimal observation?
