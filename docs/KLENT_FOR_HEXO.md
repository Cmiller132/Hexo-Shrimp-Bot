# KLENT for Hexo

KLENT (KL and Entropy Regularized Policy Optimization) is a model-free
reinforcement learning algorithm for two-player zero-sum games, introduced at
ICML 2026. It trains a policy and an action-value function through self-play
without tree search, using a closed-form improved policy derived from reverse-KL
regularization and entropy regularization.

This document describes how KLENT is adapted and configured for the Hexo game.
The algorithm is specified in [KLENT_PAPER.md](KLENT_PAPER.md); measured
outcomes and configuration selection live in [ABLATIONS.md](ABLATIONS.md). The
implementation lives in `python/mantisnet/mantisnet/klent/`.

The paper's coefficients $(\alpha, \beta)$ for entropy and reverse-KL are this
repository's $(\lambda, \tau)$. $\lambda_{\mathrm{ret}}$ is the distinct
$\lambda$-return mixing coefficient.

## 1. The MDP

### 1.1 State, action, and transition

Each MDP step is one stone placement. A Hexo state $S_t$ is the engine
`Position` immediately before one placement, carrying the board, the current
mover $m_t$, and `moves_remaining`. The action space $A(S_t)$ is the complete
legal-cell list in engine order. Transitions are deterministic: `Position.advance(q, r)`.

The opening consists of one placement (`moves_remaining == 1`). Later turns
consist of a first placement (`moves_remaining == 2`) followed by a second
placement (`moves_remaining == 1`). A win can end either placement.

For a terminal game, each placement's outcome target is $z_t = +1$ if $m_t$ is
the eventual winner and $-1$ otherwise.

### 1.2 Mover-change sign

Policy, action values, and returns are in the frame of the mover at $S_t$. When
consecutive states have different movers, quantities must be sign-flipped. The
sign is:

$$s_t =
\begin{cases}
+1, & \texttt{moves\_remaining}(S_t) = 2 \\
-1, & \texttt{moves\_remaining}(S_t) = 1
\end{cases}$$

This follows mover change, not ply parity. `signs_from_moves_remaining` in
`returns.py` computes the sign array.

### 1.3 Lambda-returns

For a naturally terminated episode with acted-on states $0, \ldots, T$:

$$G_T = +1, \qquad
G_t = s_t \, \gamma \left[(1 - \lambda_{\mathrm{ret}}) \hat{v}_{t+1}
+ \lambda_{\mathrm{ret}} G_{t+1}\right] \quad (t < T)$$

The last stored state precedes the winning placement, so its acting mover is the
winner. No terminal position is bootstrapped. $\gamma$ is a per-placement
discount magnitude; the perspective sign is carried by $s_t$.

`lambda_returns` in `returns.py` computes the array in float64.

## 2. Improved policy

At each acting position, the closed-form improved policy from paper equation 3
is computed:

$$\pi'(a \mid s) = \operatorname{softmax}\!\left(
\frac{\tilde{Q}(s,a) + \tau \log \pi_\theta(a \mid s)}{\tau + \lambda}
\right)$$

where $\tilde{Q}$ is the mass-normalized acting score (see section 3) and
$\log \pi_\theta$ comes from the network's policy logits. $\tau$ weighs reverse-KL
to the current policy; $\lambda$ weighs entropy of $\pi'$. The computation runs
independently within each position's legal-action set.

The improved policy also produces:

- $\hat{v} = \sum_a \pi'(a)\, Q(s,a)$: the acting value, averaging unscaled $Q$.
- $D_{\mathrm{KL}}(\pi' \| \pi_\theta)$: per-position reverse KL.
- $H(\pi') / \log |A|$: normalized entropy.

`improved_policy` in `improve.py` implements this operator.

## 3. Critic

The model's cell-level head emits three raw logits $(z^+, z^-, z^0)$ per legal
cell. Their softmax gives a categorical distribution:

$$p = \operatorname{softmax}(z), \qquad Q = p^+ - p^- \in (-1, 1), \qquad
M = p^+ + p^- = 1 - p^0 \in (0, 1)$$

$Q$ is the action value. $M$ is the committed return mass: how much probability
the critic assigns to nonzero outcomes.

Acting uses two derived quantities from these logits:

- **Acting score** $\tilde{Q} = Q \,/\, \max(\max_b M(s,b),\; \texttt{mass\_floor})$.
  The per-position divisor adapts sharpness to committed return mass without
  changing action order. `mass_floor` bounds the divisor when all actions assign
  most probability to zero return.
- **Action value** $Q = p^+ - p^-$, used to compute $\hat{v}$.

`compose_q` and `compose_acting_q` in `model.py` compute these.

## 4. Self-play collection

### 4.1 The collector

`Collector` in `selfplay.py` maintains a fixed cohort of persistent environment
slots. Every slot starts from the empty board. In each lockstep step, every
slot's position is evaluated by the network, the improved policy is computed,
one action is sampled, and the slot is advanced. Finished or capped slots reset
immediately.

Each ply records the legal rank of the chosen action, the improved-policy
vector, the acting-time $\hat{v}$, the mover, the phase, and diagnostic
quantities (KL, normalized entropy, top probability, chosen probability).

### 4.2 Episode lifecycle

- Every episode starts from the empty board.
- A game ends naturally when a placement wins, or is capped at a configurable
  ply limit (default 512).
- A capped episode contributes no training samples; all its placements are
  dropped.
- Unfinished games persist in their slots across `collect` calls. A single
  episode may therefore contain acting records from successive weight snapshots.
- A `collect` call returns at least its configured game quota of ended episodes
  (including caps in the count).

### 4.3 Sample storage

Each sample is `(moves, t, rank, improved, g)`: the completed episode's move
tuple, the prefix length defining $S_t$, the action's legal rank, the full
legal-set $\pi'$, and $G_t$. Board objects and model observations are not
stored; `moves[:t]` is replayed during fitting.

An iteration's sample list contains every placement from every naturally
terminated episode in that collection call. The list is in memory, traversed for
one fitting epoch, and then discarded. There is no persistent replay buffer.

## 5. Fitting

### 5.1 Loss

The fitting objective per sample is:

$$L = -\sum_{a \in A(S)} \pi'(a) \log \pi_\theta(a \mid S)
\;-\; \sum_{c \in \{+,-,0\}} y_c \log p_{\theta,c}(S, A)$$

The first term is policy cross-entropy over the full legal set, with the stored
$\pi'$ as the target. The second is categorical cross-entropy on the taken
action's critic logits, with target $y = (G^+, G^-, 1 - |G|)$ derived from the
lambda-return $G \in [-1,1]$.

The critic has one objective with optimum $Q = \mathbb{E}[G \mid S, A]$. There
is no separate squared-error Q loss in the trained objective; the detached
$(Q - G)^2$ is measured as a diagnostic only.

`policy_loss` in `losses.py` handles the policy term. The critic term is
computed in `train.py`'s fit step.

### 5.2 Batching and packing

Samples are shuffled, sorted by descending prefix length for memory packing,
then chunked under three configured limits:

- Position count (`batch_size`).
- Padded attention pairs.
- Total legal-cell decoder rows.

Chunks accumulate sample-weighted gradients until the effective `batch_size` is
reached, then Adam steps once. Preparation of the next replay chunk runs one
chunk ahead on a CPU worker.

### 5.3 Parameter check

Every optimizer step is followed by a fused finiteness check over the
parameters. A non-finite weight is refused at the step that produced it.

## 6. Outer loop

The training driver in `run.py` alternates collection and fitting iterations.

### 6.1 Pipelining

The first iteration after start or resume collects and fits sequentially.
Thereafter, while fit $i$ runs on the main thread, collection for corpus $i+1$
proceeds on a worker thread through a weight snapshot taken immediately before
fit $i$. GPU calls are serialized by a shared lock.

### 6.2 Iteration flow

Each iteration:

1. Collect episodes until the game quota is met.
2. Compute lambda-returns for each naturally terminated episode.
3. Fit the model for one epoch over the resulting samples.
4. Optionally evaluate against external opponents.
5. Write metrics and telemetry; checkpoint if scheduled.

### 6.3 Run directory

A run directory contains:

| Artifact | Content |
|---|---|
| `config.json` | Resolved KLENT, model, and evaluation settings. |
| `invocations.jsonl` | Append-only record of every resolved invocation. |
| `metrics.jsonl` | One JSON row per iteration with all metrics. |
| `telemetry.db` | Schema-versioned WAL database with per-ply and per-game detail. |
| `checkpoint_NNNNNN.pt` | Model, optimizer, iteration count, RNG state, versions. |
| `status.json` | Heartbeat with iteration progress and lane status. |
| `STOP` | Sentinel: requests checkpoint and clean exit at the next boundary. |
| `CHECKPOINT` | Sentinel: requests a checkpoint without stopping. |

### 6.4 Resume and initialization

- `--resume` restores the latest checkpoint (model, optimizer, RNG, iteration count).
  Collector slots start empty; in-flight episodes are not checkpointed.
- `--init-from` loads model and optimizer into a fresh run (iteration 0, own seed).
- `--init-lab-cell` initializes from a supervised lab cell's checkpoint (fresh
  optimizer, iteration 0).

## 7. Evaluation

Evaluation is separate from the training loop: the search module is not imported
by collection or fitting.

### 7.1 Gumbel sequential-halving search

`gumbel_choose` in `search.py` is a batched line search (not a tree). It
evaluates root candidates drawn from policy logits plus Gumbel noise, then
deepens surviving lines by sequential halving. Interior actions use
improved-policy argmax; leaf values are $\hat{v}$ (signed into the root mover's
frame) or exact $\pm 1$ for terminal positions. Zero-budget mode returns
raw-policy argmax without RNG.

### 7.2 External opponents

The driver can evaluate the model against one or more anchored opponents:

- **SealBot**: an external alpha-beta engine with a configurable time limit and
  optional depth cap.
- **Seat opponent**: an independent subprocess conforming to the container
  protocol, identified by its resolved variant and checkpoint digest.
- **Head-to-head reference**: a fixed checkpoint from any run, compared via
  seat-paired shared openings. This produces paired standard errors and Elo
  intervals that the single-anchor matches cannot.

Each opponent is evaluated independently with its own RNG derived from the run
seed and the completed iteration, so adding or removing an opponent does not
perturb another's schedule.

### 7.3 Crossplay

`crossplay.py` is a standalone referee for multi-participant matches outside the
training driver. It launches one subprocess per participant, owns opening
generation and legality, and produces a JSON manifest with game results,
per-pairing summaries, and Bradley-Terry ratings.

## 8. Per-iteration metrics

| Field | Definition |
|---|---|
| `iteration` | Zero-based corpus/fit index. |
| `acting_kl` | Mean $D_{\mathrm{KL}}(\pi' \| \pi_\theta)$ over acting positions. |
| `acting_norm_entropy` | Mean $H(\pi') / \log |A_{\mathrm{legal}}|$. |
| `f` | Fraction of ended episodes that terminated naturally (not capped). |
| `won_length_mean` | Mean placement count of naturally terminal episodes. |
| `p0_win_rate` | Player-0 win rate among naturally terminal episodes. |
| `first_stone_win_rate` | Fraction of wins on the turn's first placement. |
| `v_hat_winner_mean` | Mean acting-time $\hat{v}$ for the eventual winner. |
| `v_hat_loser_mean` | Mean acting-time $\hat{v}$ for the eventual loser. |
| `v_hat_mae` | Mean $|\hat{v} - z|$ on naturally terminal episode plies. |
| `buffer_samples` | Training samples after dropping capped episodes. |
| `policy_loss` | Mean policy cross-entropy over the epoch. |
| `q_loss` | Mean $(Q - G)^2$, measured only (not trained). |
| `critic_ce` | Mean categorical cross-entropy on the critic. |
| `fit_steps` | Number of optimizer steps. |
| `seconds` | Wall time for collection and fit (evaluation excluded). |
| `eval_results` | Per-opponent scores, win rates, caps, forfeits, and (for h2h) Elo. |

## 9. Adaptations from the paper

The paper targets fixed-action games evaluated on Pgx. Several adaptations apply
KLENT to Hexo's variable-action, placement-level structure:

| Adaptation | Paper | Hexo |
|---|---|---|
| MDP granularity | One game action per step | One stone placement per step, with mover-change sign following `moves_remaining` |
| Return time constant | $\lambda_{\mathrm{ret}} = e^{-1/8}$ per action | $\lambda_{\mathrm{ret}} = 0.939 \approx e^{-1/16}$ per placement (two placements per turn) |
| Return discount | $\gamma = 1$ | $\gamma = 0.99$ per placement |
| Entropy coefficient | $\alpha = 0.03$ | $\lambda = 0.01$ |
| Critic parameterization | One scalar per action, squared-error loss | Three-outcome categorical $(p^+, p^-, p^0)$, cross-entropy loss; $Q = p^+ - p^-$ |
| Acting temperature | Fixed $\tau + \lambda$ denominator | $\tilde{Q} = Q / \max(\max_b M, \texttt{mass\_floor})$; $\hat{v}$ averages unscaled $Q$ |
| Model substrate | Fixed-action ResNetV2 | MantisNet's ragged legal-cell decoders on a variable board |
| Collection | Fixed transition blocks, NaN-masked tails | Persistent cohort with a ply cap; capped episodes are dropped whole |
| Fitting | Buffer cleared each iteration (epoch count unspecified) | One in-memory corpus traversed once and discarded |
| Collection/fit overlap | Sequential | Steady-state pipelining: corpus $i+1$ collected during fit $i$ |
| Evaluation | Reactive policy against pretrained Pgx checkpoints | Gumbel sequential-halving line search against SealBot, subprocess seats, or paired head-to-head |
| Initial states | Per-game initialization | Every episode starts from the empty board; cold-start pretraining finishes before KLENT |
| Output initialization | Random | Policy and action-value output layers initialized to zero |

## 10. Reference configuration

| Parameter | Value |
|---|---:|
| $\gamma$ | `0.99` |
| $\lambda$ (entropy) | `0.01` |
| $\tau$ (reverse-KL) | `0.1` |
| $\lambda_{\mathrm{ret}}$ | `0.939` |
| `mass_floor` | `0.2` |
| Ply cap | `512` |
| Games per iteration | `4096` |
| Environment slots | `1024` |
| Effective batch size | `4096` |
| Learning rate | `1e-3` |
| Optimizer | Adam |
| Evaluation search sims | `32` |
| Evaluation games per opponent | `64` |

The CLI defaults to $\gamma = 1.0$ and $\lambda = 0.03$ (the paper's values).
The reference recipe requires explicit flags: `--gamma 0.99 --lam 0.01
--lam-ret 0.939`.

## 11. Module map

| Module | Role |
|---|---|
| `improve.py` | Closed-form improved policy (section 2) |
| `returns.py` | Mover-change sign and lambda-return computation (section 1) |
| `selfplay.py` | Persistent-slot collector, episode and sample types (section 4) |
| `train.py` | `KlentConfig`, network evaluation seam, fit epoch, collection driver (sections 5-6) |
| `run.py` | Outer loop, checkpointing, CLI, run-directory management (section 6) |
| `search.py` | Gumbel sequential-halving line search (section 7.1) |
| `evaluate.py` | In-driver evaluation orchestration |
| `opponents.py` | SealBot and seat opponent adapters (section 7.2) |
| `sealbot.py` | SealBot subprocess management and match recording |
| `seat.py` | Container-protocol seat subprocess adapter |
| `crossplay.py` | Standalone multi-participant referee (section 7.3) |
| `headtohead.py` | Paired head-to-head comparison (section 7.2) |
| `graft.py` | Checkpoint format conversion between critic/decoder generations |
| `telemetry.py` | Schema-versioned WAL telemetry database |
| `hardware.py` | GPU memory and utilization sampling |
| `inspect.py` | Checkpoint inspection and position recomputation |
