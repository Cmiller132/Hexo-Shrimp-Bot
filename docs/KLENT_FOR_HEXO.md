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

`policy_logits` and `q_values` are flat, aligned, and ordered by engine legal rank. `offsets` starts at zero, ends at $N$, and defines $P$ positive-width positions. The tensors share a device. Arithmetic runs in the promotion of the two input dtypes with fp32 as a floor, and every return field carries that dtype: acting's fp32 and bf16 both give fp32, and a float64 caller keeps float64. The whole operator is `no_grad`.

| Return field | Shape | Definition |
|---|---:|---|
| `probs` | `(N,)` | $\pi'$, summing to one within each segment. |
| `v_hat` | `(P,)` | $\sum_a\pi'(a)Q(a)$, divided by the segment's own probability mass. |
| `kl` | `(P,)` | $D_{\mathrm{KL}}(\pi'\Vert\pi_\theta)$. |
| `norm_entropy` | `(P,)` | $H(\pi')/\log|A(S)|$, defined as zero when $|A(S)|=1$. |

Each per-position expectation — `v_hat`, `kl`, and the entropy behind `norm_entropy` — divides by that segment's summed $\pi'$. The mass is one by definition, but an fp32 segment softmax sums to one only to a few ulps, and where every legal move shares one saturated $Q$ nothing cancels that error: it reached $1.1\times10^{-4}$ outside $[-1,1]$ on a 159-ply position, which §1.3's range check refuses. Dividing keeps $\lvert\hat v\rvert\le\max_a\lvert Q(a)\rvert$ at any segment width.

### 2.2 Refusal boundary

The operator refuses `tau < 0`, `lam < 0`, and `tau + lam <= 0`. Shape, device, CSR, finiteness, and Q-range requirements are caller preconditions, not explicit refusals. Terminal positions are refused by the model builder, not by this function.

## 3. Critic and losses

The consumed model contract is the trunk and legal-cell decoder in [MODEL_SPEC.md](MODEL_SPEC.md) appendix B. For every legal cell in engine order, `cell_head_logits` returns one raw policy logit and the critic's two return-mass logits $(z^{+},z^{-})$; `cell_heads` composes the second into the action value

$$u^{+}=\sigma(z^{+}),\qquad u^{-}=\sigma(z^{-}),\qquad Q=u^{+}-u^{-}\in(-1,1),$$

which is what every acting path consumes. Fitting reads the raw logits and composes only the taken action from the same pass. KLENT calls no state-value readout and applies no loss to it.

For a fitting batch of positions, with $G^{+}_i=\max(G_i,0)$ and $G^{-}_i=\max(-G_i,0)$,

$$L =
\frac1P\sum_{i=1}^{P}
\left[
  -\sum_{a\in A(S_i)}\pi'_i(a)\log\pi_\theta(a\mid S_i)
  +(Q_\theta(S_i,A_i)-G_i)^2
  +\frac{\eta}{2}\Big(
     \mathrm{BCE}\big(z^{+}_\theta(S_i,A_i),\,G^{+}_i\big)
    +\mathrm{BCE}\big(z^{-}_\theta(S_i,A_i),\,G^{-}_i\big)\Big)
\right].$$

The policy cross-entropy covers the full legal set. All three critic terms select only the stored action rank. $\mathrm{BCE}$ is the with-logits binary cross-entropy against a **soft** target in $[0,1]$: at its optimum $u^{+}=\mathbb E[G^{+}\mid S,A]$ and $u^{-}=\mathbb E[G^{-}\mid S,A]$, hence $Q=\mathbb E[G\mid S,A]$. The policy cross-entropy and the squared error have unit weight; only the mass pair carries a coefficient, $\eta$ = `KlentConfig.mass_weight` (§10). Composition and every loss term are fp32.

| Check | Owner | Exact behavior |
|---|---|---|
| Policy target normalization | `policy_loss` | Accumulates each segment in float64 and refuses sums not `torch.allclose` to one with `atol=1e-4` and the default relative tolerance; NaN, infinity, and truncated targets fail. It does not separately refuse negative entries. |
| Outcome range | `value_target` | Refuses values outside $[-1,1]$. KLENT does not call this binned state-value helper. |
| Scalar return range | KLENT fitter | No explicit range or finiteness check; valid collection through bounded $Q$ and the return recursion keeps $G$ in $[-1,1]$, which is also what makes $G^{\pm}$ admissible cross-entropy targets. |
| Stored-policy width | `_rebuild` | Refuses a stored $\pi'$ whose length differs from the replayed position's legal count. |

## 4. Collection and buffer

### 4.1 Evaluation seam and acting values

Collection accepts:

```python
evaluate(batch) -> (policy_logits, q_values)
```

Both outputs are flat fp32 CPU tensors in batch legal-cell order. `network_evaluate` implements the seam with `trunk` plus the cell heads under `no_grad`, composing $Q$ outside the autocast region; it does not evaluate the state-value head.

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

### 6.2 Anchored opponents and RNG

The driver accepts SealBot, one independent §3.1 subprocess seat, or both as anchored external opponents. SealBot's recorded identity is `sealbot` plus variant, per-turn time limit, and optional depth limit; the checkout commit and build identity are not version-pinned. `--eval-seat PATH` reads the same strict participant-list format as crossplay but requires exactly one entry, with an explicit referee ID, argv command, checkpoint, and requested variant. The seat's `welcome.name` is its opponent name, and its strength-defining configuration is exactly `welcome.resolved_variant` plus `welcome.digest`.

The seat opponent starts one subprocess for a match and delays each game's `open` until the model/opponent loop first asks that seat to choose; at that point the authoritative position's `current_player` fixes the seat's side. All newly waiting games are opened in one batch. Each chooser round sends one `decide` containing every waiting slot, its accepted moves since the slot's last committed message, and the resulting zobrist; `after_move` accumulates this delta for both players' moves. A slot-local `restriction_exhausted` refusal is a scored `FORFEIT` only when that seat declared a `welcome.restriction`. Because a refusal answers the whole request, the adapter removes the forfeiting slot and retries all survivors together without advancing their sent cursors. An illegal action is also a forfeit. Every other refusal, a malformed response, a bad attestation, or a dead subprocess raises a participant-naming `SeatError` and aborts the evaluation. For a normally ended or capped game, `finish_game` closes its slot and returns the seat diagnostics as opponent metadata.

`--h2h-ref CHECKPOINT` adds §6.4's paired head-to-head as a third in-driver opponent kind: every evaluation boundary plays `--h2h-pairs` shared openings from both seats between the live model and the fixed reference, both searching at the driver's `--eval-sims` and coefficients. It satisfies the evaluation-opponent requirement on its own. The reference is pinned by digest and iteration in its `opponents` row, and the match writes one `eval_matches` row whose Elo interval comes from the paired standard error. Its `eval_results` entry carries `elo`, `elo_lo`, `elo_hi`, `sign_test_p`, and `pair_counts`, which anchored entries do not have. One generator derived from `(run seed, completed iteration)` drives the whole match and the games play as one batch; pairing is by schedule structure — games `2i` and `2i + 1` share pair `i` — so per-pair reproducibility is the offline tool's property, not this one's.

In-driver evaluation, paired head-to-head, and each crossplay pairing use one seat-paired schedule: uniform-random openings are each replayed from both seats. An opening is one to ten placements, two through six by default; the bound is what makes an opening nonterminal, because at ten placements the leading player owns five stones. The ply cap counts the opening's placements and must exceed the longest opening. Caps score one half. An illegal proposal or an exhausted declared seat restriction is a forfeit, not a cap; other seat faults abort their match, while the SealBot adapter alone retries one unplayable proposal before forfeiting.

The in-driver evaluation generator is derived from `(run seed, completed iteration)` and never from the training generator. Enabling evaluation therefore does not consume training RNG state. `--eval-games` applies separately to every configured opponent, and each match starts from a fresh generator with that same derivation, so adding another anchor neither divides the game budget nor perturbs the other anchor's opening/model-RNG schedule.

### 6.3 Seat crossplay

`mantisnet.klent.crossplay` is the host referee outside `hexo-bot` required by `CONTAINER_SPEC.md` §14. A participant list gives each stable referee ID, an argv launch command, and the checkpoint and variant for its §3.1 `hello`; it never imports participant code. The same command repeated with different checkpoint references expresses a within-run checkpoint sweep. There is no run-directory scanner or in-process crossplay path.

The referee launches one subprocess per participant and holds every authoritative `hexo_py.Position`. It owns opening generation, seat swapping, the placement-count ply cap, legality, and outcomes; a seat receives neither an opponent identity nor a result. It opens both participants' mirrors from the complete shared prefix, sends only accepted-placement deltas thereafter, checks every pre-action zobrist attestation, submits the returned action to its authoritative position, and closes both live slots when the game ends. A seat's own returned action remains in the next delta because the seat does not apply it to its mirror.

All games in all pairings run concurrently. In each scheduling round the referee groups every game waiting on a participant into exactly one `decide`, sends those participant-wide batches before reading their responses, and requires one ordered decision per requested slot. A slot-local `restriction_exhausted` refusal from a seat that declared a welcome restriction commits none of that batch's mirror deltas: the named slot loses its game, the complete refusal is recorded verbatim, and the unaffected slots are sent together again in the next round with the same deltas. The retired slot is not closed. An illegal action likewise loses and records its raw `ActionId` and Python engine `MoveError` text. Any other slot-local fault, a connection-scoped refusal, malformed response, wrong attestation, or dead child aborts the tournament and names the participant; none is silently scored.

Every `hello` takes `PROTOCOL_VERSION`, `RULES_VERSION`, and `ACTION_ORDER_VERSION` from the loaded `hexo_py`, never configuration. A compliant disagreeing seat refuses the connection. Each `welcome` is validated independently, including that its resolved variant equals its own request, but two seats need not agree on name, package version, optional encoder version, variant, digest, or restriction. A restriction is copied into every pairing and game result in which the participant appears and never narrows the referee's legal game.

Each unordered pairing plays `pairs` shared openings from both seats and is summarized by the same `paired_statistics` used in §6.4. The standalone manifest contains the exact launches, hellos and welcomes, every game's complete adjudication, every paired summary, and a symmetric row-perspective matrix. It is strict JSON written through a sibling temporary file and atomically replaces the requested `--out`; seat crossplay does not write run telemetry. The opening seed derives each pairing's prefixes from `(seed, participant index A, participant index B)`, but §3.1 carries no participant seed, so it is not a whole-match reproducibility guarantee.

Bradley–Terry ratings fit the full matrix with one or more referee IDs fixed at explicit natural-log-odds anchors. For aggregate score \(w_{ab}\) over \(n_{ab}\) games,

\[
\Pr(a \mathbin{>} b)=\sigma(\beta_a-\beta_b),\qquad
\ell=\sum_{a<b} w_{ab}\log \sigma(\beta_a-\beta_b)
 +(n_{ab}-w_{ab})\log \sigma(\beta_b-\beta_a).
\]

A cap contributes one half to each side. Newton maximization uses the unregularized observed-information graph Laplacian; a free rating's standard error is the square root of the corresponding inverse-information diagonal, conditional on the fixed anchors. No pseudocount, ridge, clipping, or pseudoinverse turns separation into a finite estimate. The result explicitly names disconnected components, leaves an unanchored component absent, and uses directed outcome reachability to leave an all-win, all-loss, or more general separated rating and its standard error absent rather than reporting a large sentinel. Fixed anchors retain their nominated value and have conditional standard error zero.

### 6.4 Paired head-to-head

A head-to-head compares two checkpoints from any two run directories directly, and is the resolution instrument: two independent anchored scores of sixty-four games cannot separate less than about eight percentage points, while the pairing below reports a standard error next to the unpaired one it replaces. Crossplay reuses its `paired_statistics` calculation for each seat-swapped participant pairing, but not this in-process checkpoint-loading path.

`pairs` openings are each played twice with the seats swapped, both models searching at the same `sims`, and a pair is one unit — it shares its opening and its whole generator, derived from `(seed, pair index)`, and is reproducible alone. Per pair, `d` is A's wins minus one and lies in `{-1, 0, +1}`; a one-one split is the seat-advantage component the pairing removes. The result states A's score with its marginal Wilson interval, the win/split/loss pair counts, the paired and unpaired standard errors of the same estimand, an exact two-sided binomial sign test over the decisive pairs, Elo with an interval from the paired standard error, the seat split, and the capped count.

A cap is not a decision: a capped pair is counted apart from the win/split/loss counts, excluded from the sign test, and named in the result's warnings. Elo bounds that reach a zero or unit score are unbounded and are recorded as absent rather than as infinities. Pairs that all carry one `d` — the shape an all-splits match takes — have no spread to estimate, so such a match has no standard-error ratio and no Elo interval either, and names that degeneracy in its warnings. A ply cap must exceed the longest opening, or neither model would move.

`temperature` scales the root Gumbel vector, and because Gumbel is a scale family this is a temperature exactly: ranking by `logit + Gumbel(0, T)` draws the root order from `softmax(logits / T)`. `T = 1` is the unscaled draw and `T = 0` searches deterministically, leaving the openings as the only source of a pairing's diversity. It applies to both seats — an asymmetric one would report the difference between two search settings as a difference between two models — and it is refused at `sims = 0`, where argmax draws no Gumbel to scale. `T` weighs against `C_VISIT` and `C_SCALE` as well as against the logits, so it also sets how much a searched line must be worth to overturn the prior order; matches at different `T` are different measurements.

The two head-to-head checkpoints must agree on `RULES_VERSION`, `ACTION_ORDER_VERSION`, and the Torch version, for which no conversion exists, and on `MODEL_REPR_VERSION`, for which `klent.graft` is the bridge. The output names both checkpoints' SHA-256, versions, and iteration, and the seed, sims, coefficients, temperature, pairs, opening range, ply cap, and device the match ran under; an unsearched match records absent coefficients, because it never consults them. These are head-to-head compatibility rules, not crossplay welcome-equality rules.

## 7. Run directory contract

A fresh CLI run requires an absent or empty `--out`. A nonempty directory requires `--resume`, and resume requires at least one `checkpoint_*.pt`.

| Artifact | Guarantee |
|---|---|
| `config.json` | Replaced on every invocation with the current resolved KLENT/model settings, target iteration, checkpoint/evaluation settings, seed, init source, configured SealBot path, evaluation-seat source plus participant entry, and/or head-to-head reference plus pair count, and versions. It omits `starve_limit`. |
| `invocations.jsonl` | Appends every resolved invocation, including start iteration, `starve_limit`, changed resume settings, external-opponent configuration, KLENT configuration, and versions. |
| `metrics.jsonl` | Appends and flushes one strict-JSON row per executed iteration; NaN statistics become `null`. Resume does not prune a tail beyond the restored checkpoint, so superseded or duplicate iteration numbers may remain. |
| `status.json` | Atomically replaced heartbeat with exactly `updated`, `iteration`, `collect`, `fit`, and `eval`; idle lanes are `null`. A clean, STOP, or starvation return clears the lanes. |
| `telemetry.db` | Schema-versioned WAL database. Each iteration's row, games, plies, and hardware aggregates commit in one transaction; every in-driver opponent result is stored as its own attributed match row. Resume removes driver/self-play rows at and beyond the restored iteration before replay. Seat crossplay writes its standalone manifest instead. |
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

A checkpoint with the old three-row slot-class decoder tables and the single
`tanh`-scored scalar critic row is not loadable by this build.
`python -m mantisnet.klent.graft OLD.pt NEW.pt --tau T --lam L` converts both
changes once, writes its evidence as `NEW.json`, and every other loader stays
strict. There is no flag or artifact for either transform alone. Each of the 93
joint-class rows in `e_pw.weight` and `e_qw.weight` copies the old slot-class row
it replaces. The critic conversion is

$$W^{+}=2W_s,\quad b^{+}=2b_s,\quad W^{-}=-2W_s,\quad b^{-}=-2b_s,$$

which is exactly function preserving because $\sigma(2z)-\sigma(-2z)=\tanh(z)$.
The graft refuses a parent that differs from this build at any key other than
the two decoder tables and two critic-readout tensors, a malformed checkpoint,
the wrong parent representation version, an unexpected tensor shape, missing
Adam state for either readout tensor, or any collision among source, destination,
and sidecar paths. `--tau` and `--lam` are required: the evidence's $\pi'$
measurements are meaningless without the operating point they were taken at.

Adam state is remapped by parameter name onto a single group contiguous over
`named_parameters()`. The decoder tables' first and second moments replicate by
the same row map as their weights. A readout row scaled by $s$ takes
$m\leftarrow s\,m$ with $v$ and the step unchanged, so the first post-graft step
is $s$ times the parent's — the same step in function space that the readout
itself preserves. No parameter tensor is added, and no moment is zero-filled.
Adam's state dict is sparse: a parameter it never stepped — the state-value head
KLENT does not train — has no entry, and the conversion carries that absence
rather than inventing moments for it.

The graft is a detector, not a formality. Its joint battery compares every
untouched tensor against a second source-file read, checks all 186 expanded rows
bit for bit, and transcribes MODEL_SPEC §6 independently to compare the old
slot-class decode with the expanded joint-class decode bit for bit. It also
retains the joint arm's bounds on the folded decoder's fp32 reassociation. Its
BRM battery runs a separate fixed set of 64 nonterminal positions through the
complete grafted model and through the parent checkpoint strict-loaded into its
three-row, one-wide architecture. It writes neither checkpoint nor evidence
unless $\max|Q_{\text{new}}-Q_{\text{parent}}|\le10^{-5}$ and
$\bigl|\overline{D_{\mathrm{KL}}(\pi'_{\text{new}}\Vert\pi'_{\text{parent}})}
\bigr|\le10^{-6}$. Each model's $\pi'$ is taken from its own policy logits and
action values in float64, so the KL tolerance bounds a difference between the
models rather than the operator's own rounding. The evidence records both probe
sets, both transforms, every tolerance and measurement, the operating point,
versions, and the shared-tensor digest.

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
| `mass_loss` | Sample-weighted mean taken-action $\mathrm{BCE}(z^{+},G^{+})+\mathrm{BCE}(z^{-},G^{-})$ over the epoch, **unweighted** by $\eta$, so it reads as a calibration diagnostic independent of the coefficient. |
| `fit_steps` | Number of optimizer groups stepped. |
| `seconds` | Driver interval covering the collection wait and fit; evaluation is excluded. |
| `eval_results` | One entry per configured opponent, each containing `opponent_name`, strength-defining `opponent_config`, total `score`, `win_rate`, `games`, `capped`, and `forfeits`. A model win or opponent forfeit scores 1, a cap \(1/2\), and a loss 0; forfeits remain explicit rather than being folded into losses. A head-to-head entry additionally carries `elo`, `elo_lo`, `elo_hi`, `sign_test_p`, and `pair_counts` from §6.4's paired statistics. |
| `eval_seconds` | Wall time across all opponent matches and their telemetry recording. |

`policy_loss`, `q_loss`, `mass_loss`, and `fit_steps` are absent when the buffer is empty. Evaluation fields are absent when no evaluation runs. Empty conditional statistics are serialized as `null`. `mass_loss` has no queryable telemetry column and survives in `metrics.jsonl` and the telemetry `metrics_json` payload. Telemetry writes one `eval_matches` row per `eval_results` entry and additionally derives samples/second, game and ply counts, and per-iteration hardware means and maxima.

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

**Paper:** Policy and action-value networks start from random initialization. **Here:** The policy and action-value output layers initialize exactly to zero, so the initial policy logits and action values are exactly zero. **Grounds:** The remaining model parameters retain their configured initialization. **Measured outcomes:** See [ABLATIONS.md § `abl-zeroq-lam01`](ABLATIONS.md#abl-zeroq-lam01).

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

**Paper:** Anchored evaluation uses 1,024 games against a fixed pretrained Pgx checkpoint. **Here:** The reference recipe uses 64 seat-paired games per configured anchor from shared random prefixes; retained measurements use external SealBot, and the driver can instead or additionally anchor against one independent §3.1 subprocess seat. **Grounds:** The common opponent seam admits SealBot's separate rules state and alpha-beta chooser or a seat whose strength is fixed by its resolved variant and digest. **Measured outcomes:** See [ABLATIONS.md § SealBot as anchored evaluator](ABLATIONS.md#sealbot-as-anchored-evaluator).

### 9.14 Model substrate

**Paper:** Experiments use a fixed-action ResNetV2 with policy and action-value heads and no state-value head for KLENT. **Here:** KLENT consumes MantisNet's ragged legal-cell policy and action-value decoders; its model container has a state-value head that KLENT neither calls nor trains. **Grounds:** Hexo's board and legal-action count are variable. **Measured outcomes:** See [ABLATIONS.md § Shared decoder aggregation](ABLATIONS.md#shared-decoder-aggregation-and-triton-segment-reduction) and [§ Critic ranking-stability probe](ABLATIONS.md#critic-ranking-stability-probe).

### 9.15 Critic parameterization

**Paper:** The action-value head emits one scalar per action, fitted by squared error against the $\lambda$-return. **Here:** The head emits two logits per action whose sigmoids are the positive and the negative return mass; their difference is $Q$ and carries the same squared error, and each mass additionally carries a cross-entropy against its own part of the return with weight $\eta/2$. **Grounds:** Each mass is calibrated by its own cross-entropy while $Q$ stays their difference, so the head stores $\mathbb E[G^{+}]$ and $\mathbb E[G^{-}]$ instead of $\mathbb E[G]$ alone. **Measured outcomes:** See [ABLATIONS.md § Training runs](ABLATIONS.md#training-runs).

## 10. Reference configuration

The current repository reference recipe is:

| Parameter | Value |
|---|---:|
| $\gamma$ | `0.99` |
| $\lambda$ | `0.01` |
| $\tau$ | `0.1` |
| $\lambda_{\mathrm{ret}}$ | `0.939` |
| $\eta$ (`mass_weight`) | `0.25` |
| Critic | two return-mass logits, $Q=u^{+}-u^{-}$ |

These are configuration facts recorded in [ABLATIONS.md](ABLATIONS.md), which also records their selection.

**Finding:** `KlentConfig` and the CLI currently default to $\gamma=1.0,\lambda=0.03$; the reference recipe therefore requires explicit flags, and `test_klent_run.py` pins the `0.03` CLI default. `mass_weight` defaults to the $\eta$ above, so `--mass-weight` is needed only to depart from it.

## 11. Open questions

- What simulator-evaluation, wall-clock, and seed budget defines a faithful-scale Hexo run?
- How many fitting epochs per corpus should be assumed when the paper does not specify an epoch count?
- Should the current derived live-window observation be ablated against a rules-minimal observation?
