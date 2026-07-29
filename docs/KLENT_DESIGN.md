# KLENT for Hexo — specification sketch

**Status: design sketch, with a first implementation.** Not normative, and
nothing here constrains `hexo-engine`, whose contract remains `ENGINE_SPEC.md`.
This document describes how the KLENT algorithm would work on Hexo: the MDP it
defines, the value target, the action space, the training corpus, the
evaluation protocol, and what all of that asks of the engine and the runner.

An initial faithful implementation exists at
`python/mantisnet/mantisnet/klent/`, on MantisNet (`MODEL_SPEC.md`, appendix B
for the Q head), with §4.7's obligations as its tests — including the
two-placements-per-turn Count Up Game against a backward-induction oracle. It
runs the §2 baseline only: none of `KLENT_PROPOSALS.md`'s accepted items are in
it, and the record/runner integration of §12 and §14 is not either — the buffer
is in-memory per iteration, and games run through `hexo-py` directly.

The governing principle is **faithful first**. The paper's algorithm is adopted
unchanged wherever Hexo permits it, and every deviation in §3 names the property
of Hexo that forces it. Anything merely attractive sits in §15, outside the
baseline.

---

## 1. The source algorithm

*Revisiting Regularized Policy Optimization for Stable and Efficient
Reinforcement Learning in Two-Player Games.* Ota, Osa, Omura, Harada. ICML 2026,
PMLR 306. arXiv:2602.10894v2.

**KLENT** is AlphaZero with the tree search deleted and replaced by a closed-form
policy improvement step.

One shared trunk (ResNetV2, 128 channels, 6 residual blocks; 20 for 19x19 Go;
1.7–2.1M parameters) with **a policy head and an action-value head and no
state-value head**. Per decision it computes an improved policy analytically and
samples from it:

```
                    ⎡ Q_θ(s,a) + τ·log π_θ(a|s) ⎤
π′(a|s) = 1/Z(s)·exp⎢ ───────────────────────── ⎥          (paper eq. 3)
                    ⎣           τ + λ           ⎦

v̂_t = E_{A~π′(·|S_t)}[ Q_θ(S_t, A) ]
```

`τ` weights a **reverse** KL divergence to the current network policy — gradual
updates against a non-stationary opponent. `λ` weights the entropy of `π′` —
sustained exploration against unseen test-time opponents. The pairing is chosen
specifically because reverse KL + entropy is the combination admitting an
elementary closed form; forward KL + entropy needs the Lambert W function.

Value targets are λ-returns over `v̂`, with `γ = 1` and reward only at the
terminal ply. One loss trains both heads:

```
L(θ) = E_D[ −Σ_a π′(a|S)·log π_θ(a|S) + (Q_θ(S,A) − G)² ]   (paper eq. 4)
```

The outer loop alternates a self-play phase filling an **on-policy buffer,
discarded every iteration**, with a fitting phase over it. 1024 parallel
environments x up to 2048 transitions, so roughly 2M fresh samples per update
round. Adam, lr 1e-3, batch 4096. Hyperparameters unified across all five games
at `(τ, λ, λ_ret) = (0.1, 0.03, e^{-1/8} ≈ 0.883)`.

> **Corrected 2026-07-27 against the paper's text.** The paper writes eq. 2 as
> `−β·D_KL(π′‖π) + α·H(π′)` and sets `(α, β) = (0.03, 0.1)` — so in this
> document's notation the reverse-KL weight is `τ = 0.1` and the entropy weight
> is `λ = 0.03`. Earlier drafts of this document carried the pair transposed,
> which `KLENT_PROPOSALS.md` flagged as the one-line check to make before use.
> The temperature `τ+λ = 0.13` is unchanged; the prior exponent `τ/(τ+λ)` is
> **0.77**, a prior that mostly holds, not the 0.23 the transposition implied.
> §8 is corrected accordingly.

Results worth carrying: 4x the training efficiency of Gumbel AlphaZero at equal
simulator evaluations, and 77.2% average win rate against the anchored baseline
with test-time Gumbel MCTS attached, against 53.6% for Gumbel AlphaZero itself.
All three components are load-bearing in ablation. At evaluation the deployed
artefact is **`argmax π_θ`**, not `π′` — `π′` is a training-time construct.

Theory, in two settings. In normal-form games the update rule is locally linearly
convergent to a unique fixed point under a condition on `(τ, λ)` relative to the
payoff matrix norm. In finite-length games the policy converges to the
entropy-regularised (quantal response) equilibrium `π* ∝ exp(Q/λ)`, proved by
backward induction from terminal states, approaching Nash as `λ → 0`.

Two facts about that theory matter more here than the theorems do. The
finite-length result **assumes a finite `T_max` and an acyclic reachable state
graph** (§5). And the paper's own practical `(τ, λ) = (0.1, 0.03)` sits far
outside its normal-form guarantee, so the theory is directional guidance about
which knobs stabilise learning, not a property the implementation has.

### 1.1 Why this algorithm, for this game

The paper's stated mechanism for its own advantage is branching factor. MCTS
spends simulator evaluations proportional to `b`; KLENT spends one network
evaluation per decision regardless of `b`. Their measurements follow that
prediction:

| Game | Mean legal actions | Result vs search-based |
| --- | ---: | --- |
| Animal Shogi | 7.5 | competitive |
| Othello | 8.0 | KLENT more efficient |
| Gardner Chess | 9.5 | competitive |
| 9x9 Go | 42.3 | "clearer advantage" (89% w/ test-time MCTS) |
| Hex 11x11 | 90.6 | strongest result (98% w/ test-time MCTS) |
| **Hexo** | **~500–3000** | **the extrapolation this design tests** |

Hexo sits one to two orders of magnitude past the far end of that trend, on the
exact axis the paper credits for its gains. 216 legal placements at ply 1;
600–1500 through a typical mid-game; 6,525 in a measured random 200-stone
position.

A second, quieter benefit: KLENT does no search during training, so the training
loop calls `Position::advance` and never `Search::apply`/`undo`. Two of the three
P1 findings in `ENGINE_RL_AUDIT.md` — search excursions permanently inflating a
worker's arena, and clone cost — are therefore **off the training critical path
entirely**. They return only at test time (§14).

---

## 2. Fidelity ledger

| Paper element | Hexo status |
| --- | --- |
| Objective (eq. 2), closed form (eq. 3) | **Unchanged in form**; the operator carries a critic gain `s` (`π′ ∝ softmax[(s·Q + τ·log π_θ)/(τ+λ)]`), which is the same operator at `(τ/s, λ/s)` — a temperature re-tune, not a new mechanism (`KLENT_RUN_PLAN.md` §3). `s = 1` is the paper. |
| Loss (eq. 4) | **Unchanged for the policy**, ragged over the legal set instead of dense over `|A|`. The `(Q − G)²` term is replaced by A2's factored critic: taken-action BCE on `sign(G)` and `|G|` (`MODEL_SPEC.md` appendix B). Under judgment against the scalar head it replaced. |
| λ-returns as value target | **Tweaked — necessary.** Sign follows mover *change*, not ply parity (§4). |
| Reward `±1`, terminal-only | **Unchanged.** Hexo has no rules-draw, so the terminal reward is strictly `±1`. |
| `γ = 1` | **Tweaked — necessary.** `γ = 0.99` as a per-ply return-discount magnitude (§4.4). At `γ = 1` a winner that wanders scores the same as one that converts, and `π′` provably flattens across a decided position's moves — measured as conversion diffusion, not anticipated. |
| On-policy buffer, discarded per iteration | **Unchanged.** |
| Policy head + action-value head, no V head | **Unchanged.** Dueling `Q = V + A` sits in §15. |
| Shared trunk, separate heads | **Unchanged as a shape**; the trunk is a GNN, not a ResNet (§6). |
| Fixed `|A|`-dimensional heads | **Replaced — necessary.** `|A|` is unbounded; heads read out per node (§6). |
| ResNetV2 + BatchNorm | **Replaced — necessary.** Variable-size inputs (§6.3). |
| Dense `π′` vector per buffer sample | **Replaced — necessary.** Ragged storage (§12). |
| Natural game termination | **Replaced — necessary.** 512-ply cap; capped episodes are dropped (§5). |
| Self-play from the initial state | **Unchanged.** Every episode starts from the empty board; the cold-start bootstrap, if one is needed, is a prefit that finishes before KLENT starts (§5.2). |
| `(τ, λ) = (0.1, 0.03)`, verified against the paper | **Moved, as §8 expected**: `(0.1, 0.01)`, i.e. `ρ = 0.909` against the paper's 0.77. λ = 0.03's exponent is the conversion-diffusion mechanism. |
| `λ_ret = e^{-1/16} ≈ 0.939`, rescaled from the paper's `e^{-1/8}` | **Tweaked — necessary.** The paper's horizon is 8 transitions = 8 turns; 8 Hexo transitions is 4 turns, so carrying 0.883 would halve the strategic horizon (`KLENT_PROPOSALS.md` A1's correction). |
| Adam, lr 1e-3, batch 4096 | **Unchanged as starting values.** |
| Evaluate with `argmax π_θ`, no search | **Upgraded after training**: evaluation uses a 32-simulation Gumbel line search; zero simulations remains the parity anchor (§11). |
| Anchored pretrained opponent | **Unchanged** — a fixed pretrained checkpoint (§11). |
| Test-time Gumbel MCTS | **Landed as deterministic line search**, not a general tree: Gumbel root sampling plus sequential halving (§11). |

Nothing else changes. In particular there is no reward shaping, no adjudication,
no auxiliary task, no symmetry augmentation, and no search in collection or
fitting. The KLENT operator remains the only training-time policy improvement.

---

## 3. What Hexo forces, in one list

The seven necessary deviations, each traceable to a property of the game rather
than to preference:

1. **Ragged per-node heads** — the action space is unbounded (§6).
2. **Sign on mover change, not ply parity** — two placements per turn (§4.3).
3. **Turn phase in the observation** — `FirstStone` and `SecondStone` are
   different decisions (§7).
4. **A 512-placement cap, with capped episodes dropped** — Hexo does not
   terminate on its own (§5).
5. **Ragged/sparse `π′` storage** — a dense `|A|` vector does not exist (§12).
6. **GroupNorm or LayerNorm instead of BatchNorm** — inputs vary in size (§6.3).
7. **`γ < 1` in the return** — the winner controls termination and the board is
   unbounded, so an undiscounted return pays nothing for converting a won
   position (§4.4).

An earlier revision carried an eighth — self-play seeded from foreign game
prefixes, with a warm-start heuristic phase and an annealed cut behind it —
because item 4 leaves an untrained policy's buffer empty. It was removed
whole (owner decision, 2026-07-28): the machinery grew into a curriculum the
paper does not have, and its failure mode — self-play metrics perfect while
strength against a real opponent died — was measured twice. §5.2 records
what replaces it.

---

## 4. The placement-level MDP

This section is the one that must be exactly right. Everything else in this
document describes something that fails loudly. This describes the place where a
mistake produces a system that trains smoothly forever and never gets strong.

### 4.1 States, actions, movers

The MDP is at **placement** granularity, matching the engine's atom. A turn is
two placements, but the win is checked after each, so a turn can end after the
first — which is why the turn cannot be the atom. A turn-level action space would
also be `O(b²) ≈ 10⁶`.

- State `S_t`: the `Position` before the `t`-th placement, `t ∈ 0..=T`.
- Action `A_t`: one `Action`, from `S_t.legal_actions()`.
- Mover `m_t = S_t.current_player()`.
- `T`: the ply whose placement completes a six-window. `S_{T+1}` is terminal and
  is **not** a sample — `legal_count() == 0` there, so `π′` is undefined.
- Reward `r_t = +1` if `A_t` wins, else `0`, accruing to `m_t`. The winner is
  always the mover, since you cannot complete an opponent's window with your own
  stone, so no sign bookkeeping is needed on the reward itself.

The mover sequence is `P0; P1 P1; P0 P0; P1 P1; …`. Ply 0 is `TurnPhase::Opening`
and forced to `HexCoord::ORIGIN`, so it has exactly one legal action.

### 4.2 Perspective convention

**Everything is mover-relative.** Observations are "my stones" / "their stones";
`Q_θ(s,a) ∈ [−1, 1]` is the expected final outcome *for the player to move at
`s`*; `G_t` is in `m_t`'s frame. The network never sees seat identity.

Three consequences. Colour symmetry becomes structural rather than augmented —
there is nothing to augment, because the network cannot tell P0 from P1. The seat
asymmetry of the rules (P0 places one stone then two at a time, so stone counts
run P0: 1,3,5,… and P1: 2,4,6,…) is carried by the phase and count features
rather than by a seat flag. And evaluation must still be **seat balanced**,
because the *game* is asymmetric even though the *encoding* is not.

### 4.3 The sign function

Because the mover changes every two plies rather than every ply, the standard
"negate at every step" transformation is wrong. Define

```
s_t = +1 if m_{t+1} == m_t, else −1
```

and read it off the phase:

```
s_t = +1  iff  S_t.phase() == TurnPhase::FirstStone
s_t = −1  for  TurnPhase::Opening  and  TurnPhase::SecondStone { .. }
```

Walk it, which is also the test:

| `t` | `phase(S_t)` | `m_t` | `phase(S_{t+1})` | `m_{t+1}` | `s_t` |
| --- | --- | --- | --- | --- | --- |
| 0 | `Opening` | P0 | `FirstStone` | P1 | −1 |
| 1 | `FirstStone` | P1 | `SecondStone` | P1 | **+1** |
| 2 | `SecondStone` | P1 | `FirstStone` | P0 | −1 |
| 3 | `FirstStone` | P0 | `SecondStone` | P0 | **+1** |
| 4 | `SecondStone` | P0 | `FirstStone` | P1 | −1 |

`s_t` is never needed at `t = T`; the recursion terminates there. That matters,
because a terminal position freezes its phase, and a second-stone win freezes at
`SecondStone { first }` with `first` pointing at a live stone — so reading `s_T`
off a terminal position returns an answer derived from a frozen field.

**Do not derive `s_t` from ply parity.** As a function of `t` this is
`−,+,−,+,…` from `t = 1`, which *looks* like parity and is not: it is a function
of the phase, and the phase is a function of the position. The two agree on every
well-formed game, which is exactly why a parity implementation would survive
every test that does not deliberately break it.

### 4.4 The λ-return

Computed backward over a completed, **won** episode, using the `v̂_t` recorded at
acting time (paper Algorithm 1 line 9) rather than recomputed during fitting:

```
G_T = r_T = +1
G_t = s_t·γ·[ (1 − λ)·v̂_{t+1} + λ·G_{t+1} ]          for t < T
```

with `r_t = 0` at every non-final ply. There is no truncation case: episodes
that do not reach `T` are discarded whole (§5.1).

`γ` is a per-ply discount **magnitude**; the mover-change sign is carried
entirely by `s_t`. The paper's `γ = 1` is a degenerate objective on a game
whose winner controls termination — a win in 5 plies and a win in 300 score
identically, so every move of a decided position carries the same `Q` and
eq. 3 flattens `π′` there. That is measured, not predicted
(`KLENT_RUN_PLAN.md` §3, conversion diffusion), and `γ = 0.99` is the
resolved setting: it ranks faster wins above slower ones, which is what
keeps a gradient alive in won positions.

Two sanity identities, both cheap unit tests, stated at `γ = 1`:

- **`λ = 1`** collapses to the Monte Carlo return: `G_t = +1` where
  `m_t == m_T` and `−1` otherwise. Winner's plies `+1`, loser's `−1`.
  At `γ < 1` the same signs carry a `γ^{T−t}` magnitude.
- **`λ = 0`** collapses to the one-step bootstrap `G_t = s_t·γ·v̂_{t+1}`.

### 4.5 Buffer contents

Per sample, `(S_t, A_t, π′(·|S_t), G_t)`, as in the paper. `v̂_t` only exists to
compute `G` and is not stored. Storage form is §12.

Excluded:

- **Terminal positions.** No legal actions, no `π′`.
- **Every ply of an episode that hit the cap.** Not just the tail — the early
  plies' returns depend on a terminal that does not exist (§5.1).

Ply 0 is included and harmless: one legal action, so `π′` is a point mass, `π_θ`
masked to the legal set is a point mass, and the cross-entropy term is
identically zero. Its `Q` target is real and worth keeping.

### 4.6 Divergence hazards

In the style of `ENGINE_SPEC.md` §7.4, because these are the same kind of bug:
silent, symmetric, and invisible to any test that does not target them.

- **K1 — sign from ply parity instead of mover change.** Produces correctly
  shaped targets with half of them negated. Training proceeds, the loss
  decreases, strength plateaus at nothing. The most likely catastrophic bug in
  this design.
- **K2 — the terminal ply's half-of-turn is data-dependent.** A first-stone win
  ends the game at a `FirstStone` ply, a second-stone win at a `SecondStone` ply.
  Code assuming games end on a fixed half is wrong about half the time.
- **K3 — bootstrapping from a terminal position.** `π′` and therefore `v̂` are
  undefined where `legal_count() == 0`. The recursion must stop at `T`.
- **K4 — keeping a capped episode.** An episode with no terminal has no grounded
  return at any ply. Partial retention is worse than dropping, because the
  discarded suffix is exactly the part that would have justified the values.
- **K5 — computing `v̂` from `π_θ` instead of `π′`.** The paper bootstraps under
  the improved policy; using the network policy silently changes the operator
  being iterated and throws away the improvement step in the value target.
- **K6 — recomputing `v̂` at fitting time.** By then `θ` has moved. `v̂` must be
  captured during self-play.
- **K7 — the two samples of one turn are near-duplicates.** They differ by a
  single stone and share a mover, so consecutive plies are strongly correlated
  within an on-policy batch.
- **K8 — the turn's two stones commute.** `(a, b)` and `(b, a)` reach the same
  post-turn position but different intermediate positions with different `Q`
  rows. An exploitable consistency constraint (§15), and a trap for anyone who
  assumes the intermediate state is canonical.

### 4.7 What pins this down

- **A two-placements-per-turn Count Up Game.** The paper validates its theory on
  a 7-state, 2-action synthetic with a known quantal-response fixed point (their
  Figure 4). The same game with **two moves per turn** exercises K1, K2, K3, K5,
  the λ-return, and the loss, at a cost measured in seconds, with no Hexo-scale
  machinery in the picture.
- A hand-built Hexo fixture game with `G_t` computed by hand at `λ = 1`,
  `λ = 0`, and one intermediate `λ`, asserted ply by ply.
- Both win-shape fixtures the engine already has — first-stone win and
  second-stone win — driven through the return computation, asserting K2.
- `s_t` derived from the phase equals `s_t` derived from `m_{t+1} != m_t` over a
  few thousand random games, compared against the **engine's** reported movers
  rather than a reimplementation of the turn rule. This is the one test that
  catches a parity implementation.

---

## 5. Termination and the training corpus

Hexo has no natural termination. All five of the paper's games do — Hex and
Othello fill the board, Go scores, Chess and Shogi truncate by rule — and their
finite-length convergence proof assumes a finite `T_max`. Hexo's reachable state
graph is acyclic, since stones are only ever added and no position repeats, which
supplies the other half of their assumption; the length is what is missing.

Measured, in `ENGINE_SPEC.md` §11: **zero of 20 uniform-random 512-ply playouts
produced a six in a row.** Random play essentially never terminates.

**A hard cap of 512 placements**, enforced by the runner rather than the engine so
`hexo-engine` stays pure. This already exists and matches what this design needs:
`GameSpec::ply_cap` defaults to 512, and reaching it yields
`MatchResult::Draw { reason: DrawReason::PlyCap }` — a **representable result**,
distinct from a win and distinct from `NoContest`. That is the load-bearing half
of `OPEN_DECISIONS.md` A1 settled: a capped game does not share a status with a
crash, so the training pipeline can tell which it has and drop only the former.

The cap also keeps games inside the engine's tested envelope. Representation
limits stay effectively unreachable — the measured six-armed maximal spreader is
refused at ply 759 — and if `BoardExtentExceeded` or `CoordOutOfBounds` does fire
it is an abort, not a result: `MoveError::is_rule_violation()` is `false` for
both.

### 5.1 Capped episodes are dropped

**A capped episode contributes nothing. Every ply of it is discarded, and there
is no special-case value logic anywhere.** No adjudication, no bootstrapped
truncation value, no synthetic draw. The λ-return of §4.4 has exactly one case
because there is exactly one kind of episode in the buffer: one that somebody
won.

This is the simplest possible rule and it is the right one so long as capped
games are rare. Three consequences follow from it, none of which is a defect to
be patched, all of which are things to watch:

- **The corpus is conditioned on "won within 512 plies."** Positions from which
  no win was found in the remaining budget never appear. Since a position's value
  is defined by what follows it, this biases the value function toward positions
  where a decisive line exists.
- **Wasted work scales as `1/f`**, where `f` is the fraction of episodes that
  terminate. At `f = 0.5` you pay 2x per training sample; at `f = 0.1`, 10x. This
  is the term that dominates the compute budget (O1) and it is entirely a function
  of how often the agent actually wins.
- **At `f = 0` the buffer is empty.** Not noisy, not biased — empty. Which is
  why a cold start needs §5.2's prefit before the loop can begin.

### 5.2 The cold start: prefit, not in-loop seeding

An untrained policy essentially never finishes a game, so a from-scratch run
starves. The sanctioned answer is a **pretraining phase that finishes before
KLENT starts**: fit `π_θ` (and optionally `Q_θ`) to a foreign corpus of
finished games — human games, or another bot's — then hand the warmed
checkpoint to the unmodified loop. Behaviour cloning is legal *there*
precisely because it is not inside the loop; once KLENT begins, every episode
starts from the empty board and every action is drawn from `π′`. This mirrors
the paper's own evaluation anchor, a checkpoint pretrained outside the
algorithm (Pgx's baseline models).

An earlier revision instead seeded self-play episodes from foreign game
prefixes, warm-started collection through a scripted heuristic, and annealed
the seed cut against the measured `f`. All of it was removed (2026-07-28).
What the removal cost is the ability to train from scratch without a corpus;
what it bought is a loop with no curriculum machinery, no seeded/unseeded
split in the corpus, and no way for prefix scheduling to fake progress —
the measured failure of the seeded era was exactly self-play metrics
looking perfect while strength against a real opponent died.

The `--starve-limit` guard makes the starving case loud: a run whose buffer
stays empty stops with a checkpoint instead of burning the night.

---

## 6. Action space

The policy and action-value heads are **per-node scalars read out at legal-move
nodes** of a graph trunk over the live region of the board. What the trunk looks
like inside is out of scope for this document; what is in scope is the interface
KLENT needs from it, and why the paper's fixed-width alternative is unavailable.

### 6.1 Why the readout must be ragged

Every `|A|`-shaped structure in KLENT — two `|A|`-dimensional heads, a dense `π′`
vector per sample, a cross-entropy summed over `|A|` — assumes a fixed, small
action space. The paper's largest legal set is 121. Hexo's action *identity* space
is the whole `i16` lattice, and its legal set runs to thousands.

A per-node readout dissolves that rather than working around it. No crop, no
fixed window, no padding, no size bucketing, no maximum. That matters beyond
convenience: a fixed radius-20 crop excludes out-of-crop legal moves from policy
and MCTS and freezes out-of-rim wins (`crates/hexo-engine/README.md`), which
trains an agent against an action space that is not the game's. A node set
derived from the position cannot reintroduce that failure mode, because there is
no rim.

It also matches the shape of the data. The active region is a stone cluster plus
a thin frontier shell — `O(stones + frontier)`, roughly 1,000–3,500 nodes — inside
a bounding box that is `O(bbox)` and mostly empty. A dense spatial trunk pays for
the empty space; a graph does not. Ragged batching is a disjoint union, with none
of the padded-normalisation hazards of §6.3.

Two smaller consequences. `Position::bounds()` was deliberately deleted as a
geometry leak, and a windowed encoder would have needed it back; a graph encoder
never asks. And the engine's canonical action ordering — `legal_rank` /
`nth_legal`, ascending `(q, r)`, pinned by `ACTION_ORDER_VERSION` — becomes the
node ordering for the ragged readout, which is what it was built for and what
stops self-play, training, and serving from each inventing a private mapping.

### 6.2 What the engine already provides

- **The legal set, exactly.** The frontier bit plane *is* the legal set;
  `legal_actions()` enumerates it allocation-free in canonical order, and
  `legal_count()` is O(1).
- **Per-cell structure for any coordinate, occupied or not.**
  `windows_through(coord)` returns the 18 six-bit ownership masks of every window
  through a cell by an O(1) bit gather, and is total — cells nowhere near a stone
  read as empty.
- **Adjacency is pure geometry.** `coord` is public, so the encoder computes hex
  neighbours itself and needs nothing from the arena.

One constraint on whatever the trunk is: **absolute coordinates are never
features.** After ply 0 nothing in the rules references the origin, so absolute
position carries no information and could only be learned as spurious structure.
That also makes translation invariance automatic rather than something to enforce.

### 6.3 Normalisation

**BatchNorm is wrong here.** With variable-size inputs its statistics are
computed over a batch whose members contribute wildly different element counts,
and under any padded representation it averages over padding. GroupNorm or
LayerNorm. The paper's ResNetV2 has BatchNorm and fixed-size boards; this is a
necessary tweak, not a preference.

---

## 7. Observation

Mover-relative throughout (§4.2). The minimum the rules make necessary, per node:

- Owner: mine / theirs / empty.
- Legality: is this a legal placement right now.
- `SecondStone { first }`: is this the stone my side placed earlier this turn.
  The engine keeps `first` public specifically so encoders can build this plane.

Global: phase (`Opening` / `FirstStone` / `SecondStone`), ply index, both stone
counts. **Phase is not optional** — `FirstStone` and `SecondStone` are materially
different decisions, since in the first the mover still holds a follow-up and in
the second it is about to hand over the turn.

**Derived window features are deferred, not decided.** `windows_through` would
cheaply supply, per axis: longest own run through a cell, longest opponent run,
count of windows that are five-of-six own with the sixth empty, the same for the
opponent, and count of windows still winnable by each side. The argument for is
§5 — Hexo's reward is sparse enough that a network which cannot *see* the win
condition may never encounter one, and with capped episodes dropped, not
encountering one means having no data at all. The argument against is the
orthodox one: handcrafted features can cap ultimate strength and invite the agent
to optimise a proxy. They are removable either way, so this resolves as an
ablation rather than as an argument (O3).

Explicitly excluded regardless: absolute `(q, r)`, distance from origin, anything
derived from arena geometry, seat identity, and anything about whether the episode
was seeded.

---

## 8. Regularisation at large `|A|`

The paper unified `(τ, λ) = (0.1, 0.03)` across games with `b ∈ [7.5, 90.6]`
(the pair as corrected in §1). Two quantities in eq. 3 are functions of `|A|`:

- **The entropy term's scale.** Uniform entropy is `log b`: 4.5 nats at Hex's
  b=90, 6.9 at b=1000. With `λ = 0.03` the entropy term contributes up to ~0.21
  against `Q ∈ [−1, 1]` — real exploration pressure that grows with the action
  count, though less than the transposed reading once feared.
- **The prior's strength.** Rearranged, eq. 3 is
  `π′ ∝ π_θ^{τ/(τ+λ)} · exp(Q/(τ+λ))`, i.e. `π_θ^{0.77} · exp(Q/0.13)` at the
  paper's values — a prior that mostly holds. An exponent below 1 still
  flattens, and it flattens across ten times as many tail actions here; the
  reverse-KL term is what makes the update *gradual*, and its grip weakens as
  the action space grows, only from a stronger starting point than an exponent
  of 0.23 would have given.

So: treat **`τ/(τ+λ)`** as the knob rather than `τ` and `λ` separately, expect
Hexo to want it at or above the paper's 0.77, and diagnose with quantities
already normalised for action count — per-step `D_KL(π′ ‖ π_θ)` (their Figure 8)
and `H(π′)/log|A_legal|` (their Figure 9 analogue), targeting the values they
reached on Hex. Raw entropy is not comparable across positions in Hexo, because
`|A_legal|` moves by an order of magnitude inside a single game.

Their own sensitivity result is the evidence this is real rather than
theoretical: at b=42, shrinking to `(0.01, 0.03)` produced "a notable decline in
performance… likely due to the improved policy becoming overly sharp."

**Measured, and the prediction held.** Hexo runs at `ρ = 0.909`
(`(τ, λ) = (0.1, 0.01)`), above the paper's 0.77 as this section expected,
because λ's exponent `π_θ^ρ` is what flattens the policy wherever `Q` is flat
across actions — the diffusion mechanism of `KLENT_RUN_PLAN.md` §3. `ρ = 1`
(λ = 0) was tested and is worse than either: the run sharpens and then
stagnates whole. The other half of the ratio matters as much: the *temperature*
`τ + λ` is only meaningful relative to `Q`'s realised spread across plausible
actions, which a well-calibrated critic makes small (~0.04 in undecided
positions, against `τ + λ = 0.11`). That is the `q_scale` finding, and it means
`τ + λ` cannot be set once and read as a property of the game — it is a
property of the game *and* the critic's output scale.

---

## 9. Action-value target sparsity

The loss trains the policy head against a **full distribution** over legal
actions and the critic against **the action actually taken** — one `Q` entry,
or one `(sign, magnitude)` pair under the factored readout. Coverage of a state's `Q` row per visit is `1/b`: 1/90 on their
Hex, ~1/1000 here. And Hexo positions are essentially never revisited, since the
board is unbounded, stones are permanent, and no position repeats, so unlike Go
there is no revisit process filling the row in. Generalisation does all of the
work.

Weight sharing means that is not fatal — a sample updates a function, not a table
cell, the same reason AlphaZero's single-scalar value head works. The downstream
consequence is what deserves attention:

**`π′` is an argmax-like operator over `b` noisy `Q` estimates, and its bias
grows with `b`.** With `τ + λ = 0.11`, eq. 3 puts `Q` on a ±9 logit scale. The
expected maximum of estimation noise over 1000 candidates is materially larger
than over 90, so `π′` systematically concentrates mass on cells whose `Q` is
*overestimated by noise* rather than cells that are good. `v̂ = E_{π′}[Q]`
inherits the bias, and `v̂` is the bootstrap for the λ-return, so it compounds
through value learning. This is Q-learning's overestimation bias amplified by
action count, and the paper's sensitivity finding at b=42 is a milder form of it.

Predicted symptom: the policy latches onto junk cells, and normalised entropy
either stays pinned high or collapses onto noise. Both are visible in §13.

The opposite failure is also real, and it is the reason `Q`'s output scale is
now a stated parameter rather than an accident. A `tanh`-plus-MSE head is
systematically overconfident, and that overconfidence was an implicit gain
keeping `Q`'s spread above `τ + λ`; an honestly calibrated head undershoots the
temperature in undecided positions, `π′` degenerates to `π_θ^ρ`, and the fit
trains the policy toward a flatter copy of itself. Measured as calibration
undershoot in `KLENT_RUN_PLAN.md` §3; the response is the operator's critic
gain `s`.

Responses, cheapest first: **larger `τ/(τ+λ)`** (§8), which is a hyperparameter
rather than a design change; **the bounded factored critic** (A2, landed —
`MODEL_SPEC.md` appendix B), which makes an out-of-range noise estimate
unreachable without removing the ranking bias; **dueling `Q = V + A`** with `A`
zero-meaned over the legal set, so the value loss reaches `V` from every sample
regardless of which action was taken and `V` gets one dense sample per state
instead of `1/b` (§15); ensembling or pessimistic `Q`, held in reserve as expensive.

---

## 10. Symmetry

Hexo's rules are invariant under **D6 about any centre** — a 60° rotation
permutes the three axes and the win condition is symmetric across them — and,
after ply 0, under **translation**, since nothing in the rules references the
origin once the opening is played. With the mover-relative encoding making colour
swap structural, that is a 12x exact augmentation or 12x weight tying. The
paper's games have at most 8-fold symmetry and the paper uses no augmentation at
all, so on an efficiency-focused method this is the cheapest multiplier available.

`SUGGESTIONS.md` S2 deferred symmetry partly because the action-index permutation
"only exists if a dense index exists". Under a per-node readout that blocker is
gone: the action permutation *is* the node permutation. Whether it lands as
augmentation or as weight tying is a question about the trunk and therefore out of
scope here. Outside the baseline either way (§15).

---

## 11. Evaluation

The default anchor is **SealBot** (`mantisnet.klent.sealbot`), an independent C++
alpha-beta bot for this exact game: never trained, never a training opponent,
sharing no code or heuristic with this repo — the same role the paper fills
with Pgx's pretrained baseline checkpoints. A self-made heuristic anchor was
measured to flatter exactly the failure it existed to catch. In-loop SealBot
uses uncapped iterative deepening at 0.1 s per move; optional depth caps remain
offline ladder rungs. Time and depth are pinned per run in `config.json`,
because an anchor whose strength drifts is not an anchor.

Protocol, following the paper so the curves are comparable:

- The agent gets a **32-simulation Gumbel sequential-halving line search**.
  It samples `m = min(16, simulations // 2, |A_legal|)` root candidates by
  `g_a + logit_a`, deepens every survivor by the interior `π′` argmax, and
  halves by the leaf value mapped back to the root mover. Deterministic
  transitions make lines the useful unit: re-walking an identical tree path
  buys no information, while spending that visit on depth does. Zero
  simulations is exactly `argmax π_θ` and remains the offline comparison
  anchor.
- **Seat balanced.** Every match played from both seats, because Hexo's seats are
  structurally asymmetric even though the encoding is not (§4.2).
- Enough matches per point to make the interval meaningful — the paper used 1024
  matches and three seeds.
- Horizontal axis: placements consumed, the analogue of their "simulator
  evaluations". Worth noting the asymmetry, though: their simulator ran on-GPU in
  JAX under Pgx, ours is CPU Rust with a GPU batcher, so wall-clock and training
  positions per second are the metrics that decide what is affordable, per
  `ENGINE_RL_AUDIT.md`.
- Capped games during evaluation are scored as half-wins, the paper's draw
  convention, and reported separately so the number is visible rather than folded
  in.
- Opponents enter through one identity-plus-chooser seam. SealBot's independent
  `HexGame` assertions and memory-bounded waves are adapter details. A future
  champion network supplies one batched chooser adapter plus its name/config and
  can be seated by the same in-loop and offline match code.

---

## 12. Data path and buffer

The paper stores a dense `|A|`-vector of `π′` per sample and flags memory
pressure on an 80 GB A100 even at `|A| ≤ 1225`, suggesting sparse storage since
only legal actions are nonzero (their Appendix P, Table 13). Here that becomes
mandatory: ~2M samples per iteration x ~1000 legal actions x fp16 is roughly 4 GB
of policy targets per iteration, discarded and refilled each round.

Three things make it comfortable, all falling out of decisions the repo already
made:

- **A position is a move prefix.** Four bytes per ply, and replay is measured at
  437 ns/ply from empty *including* every reallocation and recentring copy, so a
  200-ply position reconstructs in ~90 µs on one core. Store
  `(game_id, ply, ragged π′, G)` and rebuild states by replay during fitting,
  parallel across cores. A3's replay-only construction path turns out to be a
  memory optimisation.
- **The record and the training sample are one artefact.**
  `OPEN_DECISIONS.md` B2 asks the runner to persist a per-move blob it does not
  interpret; that blob *is* `(ragged π′, G)`. No second shard writer — which is
  exactly the duplication B2 exists to remove: a dropped `decision.diagnostics`
  puts every model package on its own `.npz` writing path that bypasses the
  runner.
- **`π′` is sharply peaked** by construction (`exp(Q/0.13)`), so a top-k plus
  residual-mass form is likely to capture nearly all of it. That is a measurement
  (§13), not an assumption.

Capped episodes are dropped at the point where returns are computed, so their
samples never reach the buffer. Their *records* are still written — a capped game
is a result (§5) and its move list is data about how the agent plays even if it
carries no value target.

---

## 13. Metrics

The first three are not diagnostics, they are the experiment. Anything that
cannot report them is not ready to run.

| Metric | Why |
| --- | --- |
| **Terminating fraction `f` per iteration** | §5.1. Whether there is any training data, and the `1/f` multiplier on all compute. |
| **`D_KL(π′ ‖ π_θ)` per step** | §8. Whether updates are actually gradual. Their Figure 8. |
| **`H(π′)/log|A_legal|`** | §8. Exploration, normalised so it is comparable across plies. |
| Game length distribution, terminated games only | Sets `λ_ret` and buffer sizing. |
| Legal-count distribution; node and edge counts | Sets trunk cost, and confirms the `b` estimate this whole design rests on. |
| P0 vs P1 win rate under self-play | Seat imbalance. |
| First-stone vs second-stone win rate | Coverage of the freeze path (K2). |
| `π′` mass in top-1 / top-8 / top-64 | Decides the storage form (§12). |
| `v̂_t` versus realised outcome, bucketed by ply | Value calibration, and the visible face of the §9 bias. |
| Padded bounding box distribution | Arena behaviour; also the trigger condition for `ENGINE_SPEC.md` §5.8. |

---

## 14. What this asks of the engine and the runner

Engine, all additive, none of it touching the rules:

- **Bulk node-feature export.** The read surface is per-cell and scalar today:
  `windows_through` is O(1) but one cell at a time. The benchmarks put numbers on
  what that costs an encoder:
  `windows_through` is flat at 52–59 ns, so ~2,000 nodes is ~110 µs per position,
  and at ~2M positions per iteration that is a few minutes of single-core CPU per
  iteration before anything else runs. Parallel across cores it is seconds, so
  this is a real cost rather than a fatal one. The fix is a version filling a
  caller-provided buffer for a caller-named coordinate set, gathering row runs at
  once — `ENGINE_SPEC.md` §12's sanctioned additive escape hatch, where the
  *caller* names the region in coordinates so no row, word, plane, or stride
  escapes. Deliberately not built yet: its shape is dictated by an encoder that
  does not exist, and this workspace does not keep two versions of anything. The
  narrowest piece — returning the owner from the bit-scan slot rather than mapping
  the coordinate back — has shipped, because it changed no API.
- **Ragged legal action ids plus offsets**, per `ENGINE_RL_AUDIT.md`, so a batch
  of positions produces one flat id buffer.
- **An end-to-end throughput benchmark.** The engine-level suite now exists and
  settles most of the audit's P1 list, but it measures placements, not training
  samples. The acceptance metric for this design is training positions per second
  at a fixed budget with identical outputs, and nothing measures that yet. Two of
  its results bear directly on the numbers above: `advance` does not scale with
  board size at all (377–419 ns across a 256x range in stones), and arena growth
  amortises to nothing (437 ns/ply replaying 256 plies from empty). The simulator
  is not the bottleneck here; encoding and the trunk are.

Not required: `bounds()` (§6.1), any `serde` on `Position`, any change to the
rules, any tensor or model concept inside `hexo-engine` (`SUGGESTIONS.md` S3's
only live constraint).

Runner. Already there: the nonblocking `Game` state machine with the canonical
`Position` private to it, the 512-placement cap as
`MatchResult::Draw { reason: DrawReason::PlyCap }` distinct from `NoContest`
(§5), and a `Budget` able to express the reactive no-search decision this design
runs on. Still needed:

- Per-move blob persistence (B2) — the buffer sample of §12.
- Seeds minted and recorded, derived from stable game and seat ids so scheduling
  cannot change a run (B4). `π′` is sampled, so this is load-bearing for
  reproducibility rather than decorative.
- Replay of a seeded prefix as a first-class start condition, with prefix plies
  marked so they stay out of the buffer.
- An external-process player adapter, for the anchored opponent (§11).

---

## 15. Outside the baseline

Test-time Gumbel search moved into §11 on 2026-07-28. It is intentionally a
small deterministic line search rather than the previously proposed general
DAG: the evaluation budget buys depth, while training remains search-free.

| Item | Why it is not in the faithful version |
| --- | --- |
| **D6 / colour augmentation or equivariance** | §10. Free upside, but it multiplies an efficiency that has to exist first. |
| **Dueling `Q = V + A`** | §9. The known `1/b` coverage problem, written down rather than pre-emptively patched. Deviates from the paper's Table 4. |
| **Turn-commutativity consistency loss** | K8. A real constraint, but a second objective added before the first one works is a way to not know which is broken. |
| Behaviour cloning from foreign games | §5.2. Breaks on-policy. |
| Adjudicated or bootstrapped cap values | §5.1. Capped episodes are dropped; no special-case value logic exists. |
| A win-length curriculum (four, then five, then six) | Would need a generic win fold, a `RULES_VERSION` bump, and regenerated golden vectors — `WINDOW_LEN` is baked into `run6`, `fold6`, and `WINDOWS_PER_PLACEMENT = 18`. |
| Derived window/threat features in the observation | §7. Undecided rather than excluded; resolves as an ablation (O3). |

---

## 16. Decisions ledger

1. **Placement-level MDP**, matching the engine's atom. A turn-level action space
   would be `O(b²)`, and the win check after the first stone makes the turn
   non-atomic anyway.
2. **Mover-relative everything.** Colour symmetry becomes structural rather than
   augmented, and the network never learns a seat.
3. **`s_t` read from the phase, not from ply parity.** They agree on every
   well-formed game, which is exactly why the wrong one survives testing.
4. **512-placement cap in the runner, as a representable result.** The engine
   stays pure (A1), and a capped game must be distinguishable from a crash both
   for triage and so the pipeline can drop it.
5. **Capped episodes are dropped whole, with no special-case value logic.**
   Simplest rule; correct while capped games are rare; costs a `1/f` multiplier on
   compute and a stated selection bias, neither of which is patched.
6. **`f`, the terminating fraction, is a first-class metric.** Under rule 5 it is
   not a diagnostic — it is the quantity that determines whether training data
   exists at all.
7. **Self-play starts from the empty board; the cold start is a prefit, not
   in-loop seeding** (2026-07-28, reversing this ledger's earlier entry). The
   seeded-prefix curriculum, its warm start, and its anneal were removed
   whole; behaviour cloning on a foreign corpus is legal only as a phase that
   finishes before KLENT begins (§5.2).
8. **Per-node ragged heads over a graph: no crop, no bucket, no maximum.** A
   crop excludes out-of-crop legal moves and collapses training; a
   position-derived node set cannot reintroduce that.
9. **The engine's canonical action ordering is the node ordering.** One mapping,
   versioned by `ACTION_ORDER_VERSION`, shared by self-play, training, and
   serving — a divergence there is silent and trains against scrambled targets.
10. **GroupNorm or LayerNorm, never BatchNorm.** Variable-size inputs make batch
    statistics meaningless.
11. **`τ/(τ+λ)` is the knob, and diagnostics are normalised by `log|A_legal|`.**
    Both regularisers change strength with action count, and Hexo's action count
    moves inside a single game.
12. **No V head. Dueling is outside the baseline.** Faithful to the paper's
    Table 4, with the `1/b` coverage problem documented in §9 instead.
13. **Records and training samples are one artefact.** B2's opaque per-move blob
    *is* the buffer sample; a parallel shard writer alongside it is the thing
    being ruled out.
14. **The default anchor is SealBot, external and never a training opponent.**
    In-loop: uncapped iterative deepening at 0.1 s/move against the model's
    32-simulation Gumbel line search. Offline depth caps and zero-simulation
    argmax remain comparison rungs. The opponent identity/chooser seam keeps
    the same paired match available to a future champion network adapter
    (2026-07-28; supersedes the earlier depth-1/argmax wording).
15. **Collection is a persistent auto-reset cohort with a completion quota**
    (2026-07-28). The reference implementation's own shape — a fixed set of
    environment slots, a finished game's slot restarting from the empty
    board immediately — replacing the drain-to-empty lockstep, whose wall
    clock was set by the single longest game (measured: iterations with
    identical corpora varying 2.7 s → 144 s; ~30 % of collection spent with
    under 6 % of the cohort alive). An iteration's buffer is "at least N
    finished games". The one departure from the reference inside this: games
    in flight when the quota fills *carry to the next call* and finish under
    the next weights, where the reference NaN-drops its window tails. At our
    episode-to-window ratio dropping would waste ~a third of collection;
    carrying keeps episodes whole and adds only the one-fit-behind staleness
    class the pipelined driver already accepts. Mixed-vintage episodes are
    the recorded cost.
16. **`γ < 1` in the return, as a discount magnitude carried separately from
    the mover-change sign** (2026-07-29). `γ = 0.99`. The undiscounted
    objective is degenerate on a game whose winner controls termination, and
    the degeneracy is in `π′` rather than in the returns: with `Q` equal
    across a decided position's moves, eq. 3 reduces to `π_θ^ρ` and the
    winner learns to wander. Discounting is the smallest change that ranks
    faster wins above slower ones; reward shaping (R6) is not reopened.

---

## 17. Open questions

- **O1 — compute budget.** The paper's runs are 800M simulator evaluations,
  ~2,000 A100-hours for all experiments across five games and six methods. A Hexo
  position carries ~1,000–3,500 nodes against 81 cells for 9x9 Go, and ledger item
  5 multiplies everything by `1/f`. A faithful-scale run is plausibly in the
  thousands of A100-hours. That is an extrapolation, not a measurement. The dials
  if it lands badly: total placements, trunk size, and the cap.
- **O2 — epochs per iteration.** The paper says "update `θ` by minimising
  `L(θ)`" over a ~2M-sample buffer at batch 4096 and does not state how many
  passes. One epoch, ~512 gradient steps, is the assumption until contradicted.
- **O3 — derived window features in the observation.** §7. Deferred; an ablation,
  not an argument.
- **O4 — the critic's output parameterisation.** A2's factored `(p, m)` readout
  is implemented (`MODEL_SPEC.md` appendix B) and is measurably better at
  converting won positions, but at `s = 1` it undershoots the operator
  temperature and collapses the policy (§9). The open question is whether the
  critic gain `s` fixes it — `runs/factored-939-s2` is the arm, with the kill
  criterion stated in `KLENT_RUN_PLAN.md` §3. Resolves one of three ways: `s`
  folded into `(τ, λ)` and the factored head kept, the head shelved for the
  scalar champion, or the gain kept as a live knob, which would need an
  argument this design does not currently have.
