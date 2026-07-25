# KLENT proposals — an external suggestion set, reviewed

**Status: review. Nothing here is decided, and nothing here has been applied to
[KLENT_DESIGN.md](KLENT_DESIGN.md).**

A second model was asked how to strengthen KLENT for Hexo and produced a
~20-item programme: turn-consistent backups, twin dueling critics, off-policy
sequence replay with a signed Retrace, adaptive entropy/KL duals, a magnet
policy and an exploiter league, coherent ensemble exploration, tactical
auxiliary heads, a procedural curriculum, turn-level latent options, potential
shaping, and an eight-stage ablation programme.

This document restates each item in this repo's notation and reviews it against
the game, the engine's standing decisions, and `KLENT_DESIGN.md`. The source is
restated rather than quoted: its equations arrived with LaTeX damage (several
are unparseable as written) and its symbols are not this repo's, so quoting it
would import two notations and a set of broken formulae. Where a restatement
could be doing the source a disservice, that is flagged.

Seven items are worth taking. Most of the rest are one of three things: already
in `KLENT_DESIGN.md` under a different name, in direct conflict with a standing
ruling, or defeated by something specific about Hexo.

## Notation, and a discrepancy in the source

`KLENT_DESIGN.md` §1 follows the paper: `τ` weights the reverse KL to `π_θ`,
`λ` weights the entropy of `π′`, and

```
π′ ∝ exp[ (Q + τ·log π_θ) / (τ + λ) ]  =  π_θ^{τ/(τ+λ)} · exp(Q/(τ+λ))
```

The source uses `α` for entropy and `β` for reverse KL, so `α ↦ λ` and
`β ↦ τ`. Its structure matches; its numbers may not. It calibrates against a
"fixed `(α, β) = (0.03, 0.1)` baseline", i.e. entropy `0.03` and reverse KL
`0.1`, where the paper's pair is `(τ, λ) = (0.03, 0.1)` — reverse KL `0.03`,
entropy `0.1`. One of the two documents has the pair transposed.

This is not cosmetic. Under this repo's reading the prior exponent is
`τ/(τ+λ) = 0.23`, a prior that is substantially flattened, which is the whole
premise of §8 and §9. Under the source's it is `0.77`, a prior that mostly
holds. Those are different algorithms, and the source's entire §6 (adaptive
duals) is calibrated against whichever is right. **A single line of the paper
settles it, and it should be checked before any of the source's coefficient
recommendations are used.**

---

## Verdict table

| # | Proposal | Verdict |
| --- | --- | --- |
| P1 | Separate `λ` for intra-turn and inter-turn transitions | **Accept.** The best item in the set — but its recommended direction is wrong (A1) |
| P2 | Bernoulli win-probability critic, `Q = 2p − 1`, BCE on `(G+1)/2` | **Accept.** Under-ranked by the source (A2) |
| P3 | Normalised-entropy target adapted by a dual | **Accept, reparameterised** to one dimension (A3) |
| P4 | Staged budgets, screen-then-promote, ≥5 seeds | **Accept.** Addresses O1 directly (A4) |
| P5 | Do not use coordinate novelty as intrinsic reward | **Accept as an explicit exclusion.** Correct, and worse than stated (A5) |
| P6 | Dueling `Q = V + A` with `A` centred on the policy | **Accept the centring fix only** — `π′`, not `π_θ` (A6) |
| P7 | Cross-play matrix across checkpoints | **Accept as a diagnostic**, not as a league (A7) |
| P8 | Twin critics, cross-evaluated or clipped | **Reserve, downgraded.** Cheap version cannot work; effective version costs 2x (R1) |
| P9 | Off-policy sequence replay + signed Retrace + reanalysis + phasic optimisation | **Reject.** Traces die at `b ≈ 1000`; largest deviation in the set, ranked as if free (R2) |
| P10 | Magnet policy, MMD/R-NaD style | **Reject.** Imperfect-information theory on a perfect-information game (R3) |
| P11 | Exploiter league | **Reject** the league, keep P7 (R4) |
| P12 | Turn-level latent options | **Reject.** Breaks the closed form's one cheap normalisation (R5) |
| P13 | Potential-based reward shaping | **Reject**, as the source itself nearly does (R6) |
| P14 | Procedural tactical curriculum from generated positions | **Reject.** Collides with A3 head-on (R7) |
| P15 | Tactical auxiliary label set (12 labels per candidate) | **Reject as posed.** Three cost classes conflated; one label needs a search (R8) |
| P16 | Seat / "current player role" in the observation | **Reject.** Deletes a structural symmetry for information that is provably absent (R9) |
| P17 | Bootstrap unfinished episodes at the collection boundary | **Reject.** Contradicts a standing ruling (R10) |
| P18 | Explicit turn-consistency loss | **Reject.** It is P1's knob in disguise, with an implicit weight (R11) |
| P19 | Separate `Q_1`, `Q_2` value functions for the two half-turns | **Reject as posed.** Phase is already in the state; this is parameterisation, not semantics (R12) |
| P20 | Per-turn / per-game coherent exploration heads | **Interesting, underspecified.** No answer for which head receives the policy target (R13) |
| P21 | D6 consistency loss | **Already S2 / §10** — and a golden test is the better bug detector (R14) |
| P22 | Ragged candidate heads; sign on mover change; phase in the observation; shared trunk with separate heads; replay-state starts | **Already in `KLENT_DESIGN.md`** §6, §4.3, §7, fidelity ledger, §5.2 (R15) |
| P23 | KLENT for unbounded-length games; semi-alternating backup formalism | **Agreed as open**, already noted in §5; the formalism is a nice framing (R16) |

---

## Accepted

### A1. One `λ` per transition type — right idea, wrong direction

**The proposal.** Hexo has two kinds of transition: `FirstStone → SecondStone`
(same mover) and `SecondStone → opponent's FirstStone` (mover changes). Give
them separate `λ`, and set `λ_intra` low (`0.25` suggested) so a first-stone
target leans on the *expected* value of the second placement rather than on the
one second placement that was sampled.

**Why it survives.** This is the only item in the set that is both new and
cheap. §4.4's recursion becomes

```
G_t = r_t + s_t·[ (1 − λ_t)·v̂_{t+1} + λ_t·G_{t+1} ],    λ_t = λ_intra if s_t = +1 else λ_inter
```

which is one extra scalar and one branch on the sign that §4.3 already
computes. No architecture change, no new head, no new loss term. And it is
arguably *more* faithful than a single `λ`, not less: the paper has one `λ`
because its games have one kind of transition. Hexo has two. A single `λ` is
the accident of transplanting a one-move-per-turn hyperparameter.

Note also that the source's stated benefit — the first-stone target using
`E_{π′}[Q]` over the second placement instead of the sampled one — is *already*
what happens for any `λ < 1`, because `v̂ = E_{π′}[Q]` is exactly that
expectation. The novelty is only that the two `λ` can differ, not the
expectation.

**Where the source is wrong.** It argues for a *low* `λ_intra` on variance
grounds and does not mention bias. But `v̂` is the quantity §9 identifies as
systematically **over**estimated, by an amount that grows with `b`, because
`π′` is an argmax-like operator over `b ≈ 1000` noisy `Q` estimates. Driving
`λ_intra → 0` maximises reliance on `v̂` at *half of all plies*. That trades a
variance problem for the one bias problem this design already knows it has.

So: take the knob, sweep it in both directions, and read it against the metric
§13 already lists — `v̂_t` versus realised outcome, bucketed by ply. A high
`λ_intra` is the low-bias end and is not obviously worse.

**A correction of this repo's own, in the same area.** The paper's
`λ_ret = e^{-1/8} ≈ 0.883` is a horizon of 8 *transitions*, which in its games
is 8 turns. Eight transitions in Hexo is 4 turns, so carrying `0.883` across
halves the strategic horizon while appearing to preserve it. Matching the
paper's per-turn decay gives `λ_ret = e^{-1/16} ≈ 0.939` as the faithful
starting point for `λ_inter`. This is a better-motivated starting value than the
one currently in the fidelity ledger, and it is independent of whether the
`λ_intra` split is adopted.

### A2. Bernoulli win-probability critic

**The proposal.** Parameterise `Q(s,a) = 2·p(s,a) − 1` with `p` a sigmoid, and
train it with binary cross-entropy against the soft target `(G+1)/2` instead of
squared error against `G`.

**Why it survives, and why it is under-ranked at 9th.** Three things line up
here that the source does not connect:

1. **It is exact, not approximate, and only because of a ruling already made.**
   Terminal reward is strictly `±1` (no rules-draw), and capped episodes are
   dropped whole (§5.1), so every return in the buffer is a convex combination
   of `±1` outcomes. There is no third outcome to model. The win-probability
   parameterisation is not a simplification of the return distribution — it *is*
   the return distribution.
2. **It bounds `Q` by construction, which is the cheapest structural check on
   the §9 hazard.** `π′ ∝ exp(Q/0.13)` puts `Q` on a ±7.7 logit scale; an
   unbounded regression head can place a noise-driven estimate outside `[−1,1]`
   and hand `π′` a logit no true value could justify. A sigmoid link makes that
   failure mode unreachable. It does *not* remove the ranking bias — noise still
   reorders candidates inside the bound — so it caps the damage rather than
   fixing the cause. That is still the cheapest mitigation on the table, ahead of
   the ensemble §9 currently holds in reserve.
3. **It makes the §13 calibration metric a first-class output.** `v̂_t` versus
   realised outcome is already a listed metric; with a probability head it is a
   Brier score and an ECE rather than a scatter plot.

**Costs and conflicts.** A link function, so essentially free. But it does not
compose naively with A6: `Q = V + A` is additive in outcome space, while a
sigmoid link is additive in *logit* space. Decomposing in logits is fine, but
then zero-meaning `A` no longer makes `V = E[Q]`, which is A6's entire point.
These two are alternatives at the same seam, and if both are wanted the
decomposition must be in logits with the centring understood to be approximate.

### A3. A dual on the temperature — reparameterised to one dimension

**The proposal.** Normalise entropy by `log|A(s)|`, set a target for it, and
adapt the coefficients with a SAC-style dual update rather than fixing them.
Same for a target on `D_KL(π′ ‖ π_θ)`.

**Why it survives.** §8 already establishes that both regularisers change
strength with `|A|` and that `|A|` moves by an order of magnitude *inside a
single Hexo game*, and it already uses `H(π′)/log|A_legal|` and per-step
`D_KL(π′ ‖ π_θ)` as the diagnostics. Turning a diagnostic into a controller is
the obvious next step, and it matters more here than it would in the paper's
games because O1 — whether a faithful-scale run is affordable at all — makes
every hyperparameter sweep dimension expensive. A controller that hits a
normalised-entropy target removes a sweep dimension rather than adding one.

**Where the source is wrong.** It adapts `α` and `β` independently, which moves
the temperature `τ+λ` and the prior exponent `τ/(τ+λ)` at the same time. §8's
finding is that the *ratio* is the knob; adapting both duals confounds the two
effects, and the resulting run cannot answer the question §8 is asking.

The reparameterisation is straightforward: hold `τ/(τ+λ) = ρ` fixed at its
swept value, and let the dual adapt the single scalar `T = τ+λ` to hit a
normalised-entropy target. Then `τ = ρT` and `λ = (1−ρ)T`. One dual variable,
one learning rate, and the `ρ` sweep stays interpretable.

Per-phase targets (`FirstStone` vs `SecondStone`) are a reasonable second step
and cost one more dual scalar. Per-`|A|`-bucket targets are not, because
normalising by `log|A|` is precisely what is supposed to have made buckets
unnecessary; if buckets are still needed, the normalisation is wrong and that is
the finding.

### A4. Staged budgets and seed counts

**The proposal.** Screen variants at ~10% of intended budget, promote only on
improvement, confirm at 30–40% with more seeds, full budget for the strongest
few. Preselect primary metrics per stage. Five seeds for core results against
the paper's three.

**Why it survives.** It is free, it is good practice, and it is the only part of
the source's experimental programme that engages with O1. It also fits what §13
already sets up: the promotion criterion at 10% budget can be the terminating
fraction `f` and the calibration of `v̂`, both of which move long before win rate
does.

### A5. Coordinate novelty is excluded, and for a sharper reason than given

**The proposal.** Do not add RND or state-count novelty to the reward; on an
unbounded board it rewards expanding the occupied radius and playing away from
the relevant area. Use novelty for replay priority or curriculum selection
instead, if at all.

**Why it survives, amplified.** The source is right and understates it. Legality
in Hexo is a radius-8 frontier around the occupied set, so a novelty bonus is a
direct incentive toward the shape that maximises frontier growth — which is the
**maximal spreader**, the exact configuration `ENGINE_SPEC.md` measures as
refused at ply 759 with `BoardExtentExceeded`. A novelty bonus does not merely
waste plies; it points training at the engine's representation ceiling, and it
inflates `|A|` and the node count as it goes, so the compute cost of a position
rises as the agent games the bonus. Worth recording in §15 as an exclusion with
that reason attached, rather than left unmentioned.

### A6. If dueling lands, centre on `π′`, not `π_θ`

**The proposal.** `Q(s,a) = V(s) + A(s,a) − Σ_b w(b)·A(s,b)`, with `w` the
detached `π_θ`.

**Why the centring choice matters.** Under that decomposition `V(s) = E_{π_θ}[Q]`.
But KLENT's bootstrap is `v̂ = E_{π′}[Q]`, under the *improved* policy — so a
`π_θ`-centred `V` is a state value the algorithm never uses. Centring on a
detached `π′` instead makes `V(s) ≡ v̂(s)` exactly, which means the dueling `V`
head is a direct predictor of the bootstrap value and gets one dense sample per
state, which was the whole reason §9 lists dueling as the response to `1/b`
coverage. Uniform centring, the third option offered, is worse than both: it
weights thousands of irrelevant tail cells equally with the handful `π′`
actually plays.

This is an amendment to §15's dueling item, not a new item. It stays outside the
baseline for the reason already recorded — the paper's Table 4 has no `V` head.

### A7. Cross-play matrix, without the league

**The proposal.** Maintain a cross-play matrix among checkpoints and report
worst-case historical win rate, not just latest-vs-latest.

**Why it survives.** §11 anchors evaluation on strix and nothing else, which
measures strength but cannot see cyclic forgetting — a monotone anchor curve is
compatible with each checkpoint losing to one three iterations back. The matrix
is cheap because the games are already being played and the checkpoints already
exist. Accepting it does not require accepting R3 or R4: the matrix is a
*diagnostic* over checkpoints that exist anyway, whereas the league is a
training-time cost centre.

---

## Rejected

### R1. Twin critics: cheap or effective, not both

The mechanism is on target. `π′ ∝ exp(Q/0.13)` is near-argmax, so the
selection-evaluation coupling that Double Q-learning was built for does apply
here even though there is no explicit `max` — §9 says as much.

The problem is decorrelation. Overestimation bias comes from *noise in the `Q`
estimates*, and two heads reading out of one shared trunk share the trunk's
representation error almost entirely. Different head initialisation and
bootstrap masks decorrelate the last linear layer and nothing else. So:

- **Shared trunk:** nearly free, and buys a fraction of the advertised effect.
- **Separate trunks:** buys the effect, and costs 2x — which the source itself
  reports the paper measuring as a bad compute-normalised trade.

Against that, §9's existing first response — raise `τ/(τ+λ)`, so the prior
restrains a jump onto a noise-favoured cell — is a hyperparameter, and A2 bounds
the logit for the price of a sigmoid. Both are strictly cheaper than either twin
configuration. Twin critics stay where §9 put them: in reserve, expensive.

One correction if they are ever built. The source recommends cross-evaluation as
the default and clipped-minimum as a safety ablation. But pessimism belongs in
the **bootstrap only**, never inside the improvement operator: `Q` enters `π′`
through `exp(·)`, so building `π′` from `min(Q_A, Q_B)` iterates a quantal
response to a pessimistic surrogate rather than to `Q`, and changes the fixed
point the theory in §1 is about. Improve on one critic (or the mean); evaluate
the bootstrap on the other.

### R2. Replay, signed Retrace, reanalysis, phasic optimisation

Ranked third of twelve, "very high impact". This is the largest deviation in the
set — the paper's on-policy buffer discarded per iteration is listed as
**Unchanged** in the fidelity ledger — and it fails on its own terms at Hexo's
action count.

**The trace dies immediately.** Retrace cuts traces with
`c_t = λ_t·min(1, ρ_t)` where `ρ_t = π′_new(a_t|s_t) / μ(a_t|s_t)`. Both
policies are sharply peaked (`exp(Q/0.13)`) over `b ≈ 1000` candidates, and most
sampled actions come from the tail of that peak, where the two policies'
disagreement is largest in *ratio* terms. `min(1, ρ)` caps the upside and leaves
the downside: any fitting step that halves the mass on the median sampled action
leaves `E[min(1,ρ)] ≈ 0.5`, so a stored 16-step sequence contributes
`0.5¹⁶ ≈ 10⁻⁵` of its final residual. You pay sequence storage, per-step
behaviour probabilities, and a reanalysis forward pass, and receive one-step TD.
This is worse in Hexo than in the domains Retrace was validated on for exactly
the reason this whole design is interesting: `b` is 1000, not 18.

**No contraction to appeal to.** Hexo is undiscounted, `|σ| = 1`, and has no
rule-bounded length. Retrace's guarantees come from `γ < 1` or a finite horizon.
The source acknowledges this and files it under future theory — which makes it
research, not a v1 component.

**The pieces are all-or-nothing, and one of them contradicts K6.** Reanalysis
recomputes `v̂` from the current network, which K6 explicitly forbids. That
prohibition is correct *within the on-policy baseline*, where the trajectory is
assumed drawn from the policy being bootstrapped; reanalysis is only coherent
together with off-policy correction. So reanalysis cannot be cherry-picked as
the cheap half, and the expensive half is the half that does not work.

**And it is the deviation most likely to make a negative result uninterpretable.**
The paper's claim is that a closed-form improvement step plus on-policy fitting
beats search. Replacing the data path converts KLENT into a different algorithm;
if the resulting system is weak, nothing distinguishes "KLENT does not
extrapolate to `b ≈ 1000`" from "this replay stack is misconfigured".

The one part worth keeping is orthogonal to all of it: **a tactical reservoir is
not needed, but the *records* of dropped episodes are already retained** (§12),
so if terminal scarcity turns out to be the binding problem, the material to
revisit exists without a replay stack.

### R3. Magnet policy

The mathematics is right — adding `β_M·log π_M` to the numerator and `β_M` to
the denominator preserves the closed form, and the derivation checks out. The
motivation is transplanted from the wrong class of game.

MMD and R-NaD target last-iterate convergence in **imperfect-information** games,
where the equilibrium is genuinely mixed and self-play cycling is intrinsic.
Hexo is perfect-information, deterministic, no chance nodes, and has a
determinate game value — the object being converged to is not a mixed
equilibrium that cycling could orbit. The failure mode in perfect-information
self-play is forgetting and opponent-overfitting, which is what A7 measures and
what the league claims to address, not last-iterate convergence.

There is also a plain empirical point: AlphaZero trained against its own latest
policy in three perfect-information games with no magnet, no average policy, and
no pool. The burden is on the proposal to say why Hexo is different, and
"KLENT has no search" is not obviously that reason.

### R4. Exploiter league

Same class of objection, plus cost. Main exploiters and league exploiters mean
training additional agents whose only product is data about the main agent's
weaknesses — a multiplier on a compute budget that O1 already flags as possibly
out of reach for one run. The diagnostic value is real and is available far
cheaper as A7.

### R5. Turn-level latent options

Two objections, one of them decisive.

The soft one: the second placement **already** conditions on the first. The
state at a `SecondStone` ply contains `SecondStone { first }`, and §7 puts that
plane in the observation specifically so the network can see it. The "plan" the
latent is supposed to carry is already in the state.

The decisive one: KLENT's improvement step needs `log π_θ(a|s)` for a *single*
distribution. With a latent, `π_θ(a|s) = Σ_z π_Z(z|s)·π(a|s,z)` is a mixture,
and its log-marginal is no longer one head's log-softmax. It is computable with
`|Z|` forward passes, but the entropy term's meaning changes and the one
property that makes eq. 3 usable at `b ≈ 1000` — one exponential and one
normalisation over the legal set — is gone. Highest cost in the set, against a
problem the state representation already solves.

### R6. Potential-based reward shaping

The source nearly rejects this itself and the reasons it gives are the right
ones: the invariance result needs a *fixed* potential, terminal `Φ = 0`, and a
clean episode boundary, and the proposal has a learned potential, truncation,
and a sign convention to get right. `KLENT_DESIGN.md` §15 already excludes
reward shaping. Nothing here changes that.

### R7. Procedural tactical curriculum — collides with A3

"The state generator must preserve reachability, **or at least rule legality**"
is precisely the construction path `OPEN_DECISIONS.md` A3 closed. `Position` has
no `serde` impl and no board-shaped constructor, deliberately, because that is
the rule-bypass hole the previous engine had — its `Board` deserialiser skipped
the turn rules. `Position::replay` is the only way in, so a curriculum position
is a **move prefix**, and generating a prefix that produces a named threat
structure with correct mover, phase, and stone parity is a search problem, not a
board-filling exercise.

The half of this that works is the half already in the baseline: §5.2's seeded
starts replay prefixes of real games, which is where the source's own
"replay-state starts" idea lands, and it is already a required component rather
than an enhancement. The generated-position half needs a construction path the
repo removed on purpose.

The same constraint applies, more mildly, to the proposed tactical unit-test
suite in the source's evaluation section: a hand-built suite of prefixes is
feasible — the engine already carries first-stone-win and second-stone-win
fixtures — while a procedurally generated one is the same A3 problem.

### R8. The tactical auxiliary label set

The list is 12 labels per candidate, described as cheap because the engine
already computes windows. The count is right — `windows_through` returns
exactly 18 masks per cell, matching `WINDOWS_PER_PLACEMENT = 18` — but the list
mixes three cost classes as though they were one:

- **O(18) per candidate, genuinely cheap:** max own/opponent occupancy among
  touched windows, count of windows still winnable by each side, own threats
  created, opponent threats destroyed, immediate win. These are the §7 features
  already on the table as O3.
- **Superlinear or global:** "leaves a defensive hitting set of size 0/1/2/>2"
  is a set-cover computation over the opponent's threat set, per candidate, at
  ~1000 candidates per position.
- **Requires a search:** "whether the state is forced-won/lost tactically" is
  the answer a solver produces. Labelling it presumes the thing being trained.

And the cheap class is not new: it is O3, which the owner has already ruled
**deferred** — an ablation, not an argument. Adopting a 12-label auxiliary stack
would settle that question by fiat in the direction of more handcrafting, which
is the direction the deferral declined to take without evidence.

### R9. Seat identity in the observation

Listed among "phase information" as "current player role". It contradicts §4.2,
and the interesting part is *why* it is not merely redundant but unavailable.

Under a mover-relative encoding the mover's own counts are `(n−1, n)` at a
`FirstStone` ply and `(n, n)` at a `SecondStone` ply — **identical for both
seats**. After the forced opening, P0 and P1 face structurally identical
situations, so the seat is not recoverable from the encoding and adding it
injects information the rules do not condition on. What it costs is concrete:
colour symmetry stops being structural, the 2x that §4.2 gets for free is gone,
and the network acquires a feature it can only use to learn seat-specific
artefacts of the training distribution.

"Placements remaining in the turn", offered in the same list, is a relabelling
of phase (`FirstStone → 1`, `SecondStone → 0`) and harmless but not additional.

### R10. Bootstrapping unfinished episodes

"Collection chunks must not convert unfinished Hexo games into draws. Save the
environment state across actor chunks and bootstrap at a collection boundary."

Directly against the standing ruling: draws are dropped from training with **no
special logic**, which is what gives §4.4 exactly one case and §5.1 no
adjudication, no truncation value, and no synthetic draw. The ruling is recorded
as ledger item 5, and the alternative — including bootstrapped truncation — is
listed in §15 as excluded.

Worth separating the two claims, though: the *first* clause is about actor
plumbing, and persisting a game across collection chunks so it can finish is
compatible with the ruling and probably desirable. What is rejected is
bootstrapping a value at the boundary. An episode that hits 512 is dropped;
an episode that hits a *chunk* boundary should simply continue.

### R11. The explicit turn-consistency loss

`L_turn = [Q(s_t,a_t) − sg(r_t + v̂_{t+1})]²` on first-stone plies. That target
is the `λ_intra = 0` endpoint of A1's recursion. Adding it alongside the
λ-return target for the same transition fits `Q` to two points on the same line
simultaneously, which is arithmetically a third value of `λ_intra` set by the
ratio of the two loss weights. It is the same knob with an implicit,
harder-to-read setting. Take A1 instead.

### R12. Separate `Q_1` and `Q_2` value functions

The premise is that first- and second-placement values "differ in semantics".
They do — but the phase is *part of the state* (§7 makes it non-optional), so
`Q(s,a)` already distinguishes them. `Q_1` and `Q_2` are `Q(s,·)` restricted to
two disjoint sets of states, and splitting them into separate heads is a
parameterisation choice about how much capacity to share, not a correction of a
semantic error. The one-step target equations the source derives from the split
are the same equations §4.3's sign function already produces, and its `y_1`/`y_2`
signs agree with `s_t`.

Separate final adapters per phase remain a legitimate, mundane architecture
option. They are not a fix for anything.

### R13. Coherent exploration heads

The premise is good — a single exploratory blunder can lose a Hexo game
immediately, so per-placement dithering is riskier here than in a game with
recoverable positions — and the per-turn/per-game/per-N-turn comparison is a
reasonable experiment.

It is underspecified where it matters. With `K` policy/Q heads there are `K`
different `Q` rows per state and therefore `K` different `π′`, and the loss fits
`π_θ` to `π′` by cross-entropy. Which head's `π′` is the target, and which head's
`π_θ` receives it? If all heads train on the acting head's target they converge
and the ensemble collapses; if each trains only on its own data, each head sees
`1/K` of the buffer, which is the opposite of what an efficiency-focused method
wants. The proposal does not say, and the answer determines whether this is
cheap or fatal.

One Hexo-specific mitigation the proposal misses, in the other direction: K8 —
the turn's two stones commute. `(a,b)` and `(b,a)` reach the same post-turn
position, so per-placement sampling of an unordered pair is less incoherent than
per-placement sampling would be in a game where move order mattered. The
coherence problem is real but smaller than argued.

### R14. D6 consistency loss

`§10` and `SUGGESTIONS.md` S2 already cover symmetry; the specific proposal here
is a soft consistency *loss* rather than augmentation or weight tying, motivated
partly as a way to catch coordinate-transform bugs.

For that motivation it is the wrong tool. S2's stated hazard is two
implementations of the same permutation disagreeing, and the detector for that
is an exact assertion in a test — the same argument `CLAUDE.md` makes about
symmetric bugs needing dedicated detectors rather than soft agreement. A penalty
term that is merely small does not tell you the permutation is right. As a
*regulariser* it is a reasonable alternative to augmentation, at two forward
passes instead of 12x data; that is a trunk question and stays out of scope per
§10.

### R15. Items already in `KLENT_DESIGN.md`

Roughly a third of the source restates decisions already recorded: candidate-
conditioned ragged heads (§6), the sign on mover change (§4.3, and derived in
closed form off `TurnPhase` there rather than left as a state-pair predicate),
phase in the observation (§7), shared trunk with separate policy and Q heads
(fidelity ledger), replay-state starts (§5.2), the `1/b` `Q`-coverage problem
(§9), and the finite-`T_max` gap in the paper's convergence proof (§5).

That is not a criticism of the source, which was not given the design doc. It is
a criticism of the priority ranking: the top item of twelve, "turn-consistent
expected backups", decomposes into one thing already in the baseline (bootstrapping
on `E_{π′}[Q]`), one parameterisation choice (R12), one duplicate knob (R11), and
one genuinely new scalar (A1).

One item in this group carries new *evidence* worth recording if it can be
verified: that the paper measured a shared output head for policy and `Q` as
very poor, fully separate backbones as ~2x compute for a modest gain, and the
shared-trunk/separate-heads arrangement as the best compute-normalised choice.
The fidelity ledger records that shape as "unchanged" on faithfulness grounds;
if the paper measured it, it is unchanged on evidence, which is stronger.

### R16. The theory items

The three open theory questions — KLENT for games with no rule-bounded length,
a formalism for semi-alternating backups where each transition carries a
perspective operator `σ(s,a,s') ∈ {−1,+1}`, and convergence for a signed
off-policy operator — are honestly labelled as outside current guarantees. The
first is already noted in §5. The second is a genuinely nice framing: it
generalises §4.3 from "Hexo has two placements per turn" to "games with
multi-action turns", and it is the right level of abstraction for a proof. The
third only matters if R2 is adopted, which it should not be.

Worth noting that the source's signed Retrace formula is *correct* on the point
where it would be easy to be wrong: the perspective product `∏σ_i` converts
`δ_k` from the mover at `s_k`'s frame into the mover at `s_t`'s frame, which is
what adding it to `Q(s_t,a_t)` requires.

---

## The cost objection

The source's experimental programme is eight stages of ablations, each a table
of research questions crossed with 3–4 variants: on the order of 160
configurations, at the ≥5 seeds it recommends, i.e. roughly 800 training runs.

O1 estimates that **one** faithful-scale Hexo run is plausibly in the thousands
of A100-hours, from the paper's ~2,000 A100-hours for five games and six
methods, scaled by ~1,000–3,500 nodes per position against 81 cells for 9x9 Go
and multiplied by `1/f` for dropped capped episodes. The programme is presented
without any cost figure attached.

This is the largest single problem with the document. Its Stage 0 correctness
checks are cheap and mostly already specified as unit tests in §4.7; its Stage 1
critic ablations are defensible if narrowed; the remaining six stages are sized
for a laboratory with a cluster and a year, and adopting the ranking as given
would commit the project to a programme it cannot start. A4 is the part of the
source that takes this seriously, which is why it is accepted.

---

## Unverified claims

Load-bearing citations that could not be checked here, listed so they are not
absorbed silently:

- **The reference implementation "hardcodes `gamma = -1` for every transition."**
  Cited to a GitHub repository whose existence and contents are unverified. It
  does not matter for this design either way: §4.3 derives the sign from the
  Hexo rules independently, and K1 already names the parity look-alike as the
  most likely catastrophic bug. If the claim is true it is corroboration, not
  new information.
- **The 9x9 Go anchored-opponent comparison** — Discrete SAC and V-MPO below 1%,
  Munchausen DQN 2%, GRPO 3%, KLENT 89% after 800M simulator evaluations. The
  89% figure matches this repo's own extraction of the paper (§1.1) — but there
  it is the number **with test-time Gumbel MCTS attached**. If the baselines are
  search-free, part of the reported gap is the search rather than the learning
  algorithm. The conclusion the source draws from it — borrow components, do not
  swap KLENT for SAC/V-MPO/Munchausen/GRPO — is not in doubt at that margin, but
  the margin is probably not 89-vs-1.
- **"Data-Augmented Game Starts", arXiv 2605.14379.** Unverified.
- The classical citations spot-check as accurate: Dueling (1511.06581), Double
  Q-learning (1509.06461), C51 (1707.06887), Retrace (1606.02647), PER
  (1511.05952), PPG (2009.04416), SAC auto-temperature (1812.05905), MMD
  (2206.05825), NFSP (1603.01121), Bootstrapped DQN (1602.04621), NoisyNet
  (1706.10295), SPR (2007.05929).

---

## What this would change in `KLENT_DESIGN.md`

Not applied. Listed so the diff is a decision rather than a discovery:

1. **§4.4** — `λ_intra` / `λ_inter` in the return recursion, with the bias
   direction stated (A1), and `λ_inter` starting at `e^{-1/16} ≈ 0.939` on the
   per-turn-horizon argument rather than the paper's per-transition `0.883`.
2. **Fidelity ledger** — the `λ_ret` row changes from a carried constant to a
   rescaled one; a new row for the `Q` head's output parameterisation (A2).
3. **§9** — A2 inserted ahead of the reserve ensemble as the second-cheapest
   response, with its bound-the-damage/does-not-fix-the-bias limit stated; R1's
   shared-trunk decorrelation objection recorded against the ensemble option.
4. **§8** — the dual controller as the mechanism for reaching a normalised-entropy
   target, one-dimensional over `T = τ+λ` at fixed `ρ = τ/(τ+λ)` (A3).
5. **§11 / §13** — checkpoint cross-play matrix and worst-case historical win
   rate (A7).
6. **§15** — novelty-based intrinsic reward as an explicit exclusion with the
   maximal-spreader reason (A5); the dueling row amended to `π′` centring and
   its incompatibility with A2 noted (A6).
7. **§16** — the ledger gains the `λ_intra` decision and the `Q`
   parameterisation; **§17** gains nothing, and O3 is reaffirmed rather than
   settled by R8.
8. **Nothing** in §5, §6, §7, §12, or §14 changes.
