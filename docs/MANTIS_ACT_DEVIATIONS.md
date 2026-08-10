# MantisNet-ACT v4.1 — Deviation Register

This register lists known differences between the normative target in
`MANTIS_ACT_SPEC.md` and the current implementation. It is not an experiment
log. Measurements and ablation outcomes belong in `docs/ABLATIONS.md`.

A listed gap is not a compatibility promise. The implementation must converge
to the specification with a format or schema bump where the contract requires
one; it must not retain both shapes or silently accept an obsolete value.

## §6, §16, §29 — exact typed-attention control is not implemented

**Target:** `full_extra_ffn_control` uses a dedicated window-only FFN stage with
hidden widths 111 invariant and 97 axis. At the default widths it adds exactly
19,424 parameters per state block and exactly 77,696 across four blocks, equal
to `full_with_typed_window_attention`. Construction asserts equality.

**Current implementation:** the preset changes the shared `ffn_mult` from 2 to
3. The config has no dedicated extra-stage width fields and construction does
not assert exact parameter equality.

**Required convergence:** add the two fields and stage, replace the preset
override, add the exact assertion, and remove the multiplier-based control.
The production parameter count is only a ceiling/reference for the controls;
it is not a minimum, and `d_inv` remains an owner decision.

## §6, §8 — obsolete cell-scope alias remains accepted

**Target:** the only cell scopes are `occupied_only` and `window_and_legal`.
The latter name is retained because `action_relevant` windows can introduce
cells that are neither occupied nor legal: a legal cell at distance 8 can seed
a window extending to distance 13.

**Current implementation:** config validation and the Rust/Python builder
boundary still accept `occupied_and_legal` as a third value.

**Required convergence:** remove `occupied_and_legal` from all public enums,
presets, projections, and builder parsing. Unknown or old values must fail; do
not add an alias or compatibility shim.

## §6, §13 — dead configuration knobs remain

**Target:** `use_full_cell_attention` does not exist. FiLM is the sole phase
conditioning mechanism; the only phase ablation is `use_three_way_phase=False`,
which folds OPENING into SECOND. There is no `token_only` mode.

**Current implementation:** `MantisACTConfig` still carries
`use_full_cell_attention` and `phase_conditioning`, whose accepted values include
`token_only`. Variant guards refuse some uses, but the dead fields remain part
of the public configuration surface.

**Required convergence:** delete both dead knobs and their validation/hash/lab
plumbing. Removed values must fail deserialization rather than select a no-op.

## §6, §28 — dropout is excluded from strict identity

**Target:** every resolved config field, including `dropout`, participates in
the architecture hash and strict checkpoint config comparison. Loading or
resuming with a different dropout value fails.

**Current implementation:** `dropout` is in `UNHASHED_FIELDS` and is explicitly
permitted to differ during checkpoint config reconstruction.

**Required convergence:** remove the exemption and update strict-load tests.
This is strict identity, not a request to apply dropout during evaluation.

## §17.6, §28 — structurally dead key biases remain

**Target:** every key projection used under a softmax has `bias=False`. The
corrected named-parameter census identifies 34 forbidden tensors totaling 1,616
scalars: 24 state-latent key biases (1,056 scalars), 6 action-latent key biases
(384), and 4 state-to-action broadcast key biases (176).

**Current implementation:** those key projections still instantiate biases and
`ACT_CHECKPOINT_FORMAT` remains 1.

**Required convergence:** remove the 34 tensors and bump
`ACT_CHECKPOINT_FORMAT` atomically. Do not change model code or the format while
the current checkpoint-dependent ablation matrix is live; once that dependency
ends, the state-dict change and format bump are one reviewed operation.

## §24.1 — phase-stratified auxiliary census is pending

**Target:** any decision to remove, retain, or redefine an action auxiliary must
be based on a corpus census stratified over OPENING, FIRST, and SECOND phase.
FIRST-only evidence cannot establish whether a label is constant elsewhere.

**Current implementation:** the available census sampled only FIRST-phase plies.
It therefore does not justify changing §24.1's six labels. In particular, no
claim that exactly three or four labels are constant is accepted from that
sample.

**Required convergence:** run the phase-stratified census queued in `RUNLOG.md`,
record the result in `docs/ABLATIONS.md`, and amend §24.1 only if the stratified
numbers support a contract change. Until then the six-label contract stands.

## §25, §28 — packed-schema discriminator is missing

**Target:** every `PackedACTBatch` carries
`MANTIS_ACT_PACKED_SCHEMA_VERSION = 2`, and every consumer refuses a missing or
unequal value before graph or plan arithmetic.

**Current implementation:** packed batches already carry Rust-built execution
plans and the 13-field builder fingerprint, and plan consumers validate the
fingerprint. There is no packed-schema discriminator.

**Required convergence:** add the discriminator to the Rust wire result, Python
container, device transfer, dynamic marking, and all model entry checks. Bump
the discriminator for future field/plan-shape changes rather than accepting two
schemas.

## §26 — chunk cost is not valid for every preset

**Target:** cost units follow the resolved scopes:

- `occupied_only`: `2 * occupied_stones`;
- `window_and_legal` with `live` or `nonempty`: `2 * occupied_stones + legal_actions`;
- `window_and_legal` with `action_relevant`: `graph_cell_count + occupied_stones`,
  using validated metadata or an exact count-only builder projection.

Only `full_act_v4` owns the standard fitting budget thresholds. Other presets
must declare their own limits under their node law.

**Current implementation:** `ACTChunkCost.units` returns
`2 * occupied_stones + legal_actions` for every configuration. It overcharges
`occupied_only` and can undercharge `action_relevant`.

**Required convergence:** make the cost object configuration-aware and add
exact builder-count tests for all scope combinations. Refuse budgeted packing
when exact action-relevant metadata is unavailable.

## §29, §35 — two predeclared ablation presets are absent

**Target:** `full_output_only_separation` names the
`separate_output_mlps`/no-private-adapter arm, and
`full_no_axis_live_windows` names the combined no-axis plus live-window arm.

**Current implementation:** the underlying individual switches exist, but
neither composition is a named preset in the variant registry.

**Required convergence:** add both presets to the closed registry and its
builder-read/refused-override tests. The combined arm must be registered before
the supervised screen, not synthesized after seeing component results.

## §35 — survivor and confirmation rules are not enforced by the lab

**Target:** each v4.1 comparison predeclares its supervised metric, margin,
split, seeds, aggregation, maximum survivors, cutoff, and tie-break. Only those
survivors enter a separately predeclared self-play confirmation stage with fixed
KLENT/search/evaluator settings. The test split is not used for selection.

**Current implementation:** the lab can launch and summarize individual arms,
but it does not require a machine-readable survivor rule or prevent manual arm
promotion between supervised and self-play stages.

**Required convergence:** add a validated experiment manifest and stage
transition that records the predeclared rule, computes survivors mechanically,
and refuses undeclared confirmation arms. Results remain in
`docs/ABLATIONS.md`.

## §33 — hash buckets are treated as exact structural identity

**Target:** a 64-bit signature hash creates candidate groups only. Each bucket
is partitioned by exact comparison of the complete canonical signature before
an alias is reported. At 17,461 signatures, the birthday approximation is
`8.26e-12`; aggregate hash counts cannot make a collision visible.

**Current implementation:** diagnostics reduce each signature to a 64-bit value
and group equal values without retaining and exactly comparing the canonical
signature inside a bucket.

**Required convergence:** retain or reconstruct the complete signature for
candidate buckets, exact-compare it, and base alias counts/samples on the exact
subgroups. The hash remains an acceleration index only.

## §34.1 — performance thresholds await calibration

**Target:** the fixed full-path RTX 4070 Ti-class 12 GiB, compiled bf16,
batch-512 protocol in §34.1 is an acceptance gate with concrete median, p95,
throughput, and VRAM limits.

**Current state:** the contract contains named non-passing placeholders pending
the merged-tree calibration run.

**Required convergence:** put the measurement in `docs/ABLATIONS.md` and replace
all five placeholders with reviewed thresholds. Do not place the measured
result narrative in either contract document.

## §2, §37 — KLENT collection and search are not architecture-neutral

**Target:** each architecture supplies `collate_positions`, `collate_prefixes`,
`chunk_cost`, `policy_q`, `supervised_heads`, and `has_state_value_head`.
Collection, search, fitting, and corpus evaluation receive those seams and do
not import a concrete builder. External evaluation still returns flat CPU
policy logits, `q_score`, and `q_value` in engine legal order.

**Current implementation:** fitting/corpus paths expose part of this boundary,
but KLENT self-play and search import the legacy MantisNet collation path, and
MantisACT does not expose the complete model-level seam.

**Required convergence:** inject the architecture interface through collect,
search, and evaluate entry points; make both model families satisfy it; retain
the existing external result semantics and terminal handling.

## §27 — bf16 qualification is path-specific

The contract does not claim that two bf16 implementations are equidistant from
an fp32/fp64 reference. Fused, unfused, eager, and compiled paths each require
their own declared reference and tolerance. A result within one tolerance is
not evidence that another path has the same error. This is a qualification rule,
not permission for a silent dtype or accumulation change.
