# hexo-search

Decision sessions and an evaluator batching seam for model-driven play. A model
package supplies the encoder, evaluator, and selection policy; this crate owns
the session state machines that drive them.

Three session types implement `DecisionSession`:

- **`PolicySession`** evaluates the root once and selects from the resulting
  prior distribution.
- **`MctsSession`** runs PUCT tree search with virtual loss and an in-flight
  leaf cap, batching leaf evaluations across descents.
- **`GumbelSession`** applies Gumbel sequential halving: it samples root
  candidates by noised log-prior, extends each candidate along deterministic
  prior-argmax lines, and halves the survivor set after each deepening round.

All three are `Send`, object-safe, and nonblocking. A driver interleaves
sessions and batches the leaf evaluations they request.

## The drive loop

Every session follows the same protocol:

1. `begin(&position)` copies the position and resets any previous search.
2. `pump` runs until it needs network answers. For each leaf it emits a
   `(LeafId, &Position)` pair through a callback; the position is transient and
   must be encoded before the callback returns.
3. The driver batches encoded items via `EncodedBatch` and calls
   `Evaluator::evaluate`.
4. Each answer is delivered with `resume(leaf, evaluation)`.
5. Steps 2--4 repeat until `pump` returns `SessionStatus::Decided`.
6. `take_decision` yields the finished `Decision`.

## Evaluator seam

The `Encoder` and `Evaluator` traits are package-owned. `Encoder` appends one
position's bytes to a shared arena; `Evaluator` appends one `Evaluation` per
batch item. `EncodedBatch` is the reusable ragged byte arena that holds the
batch: a single contiguous byte buffer plus an offset array that cuts it into
variable-length items.

`Evaluation` carries a prior vector (in the engine's canonical legal order) and
a scalar value from the evaluated position's side to move.

## Selection

`SelectFromPolicy` and `SelectFromSearch` are the package-owned hooks that
convert a completed evaluation or search into a placement action and optional
diagnostics. `SearchOutcome` presents the root's children (each a `Child` with
action, visits, mean value, and prior) to the selector; `PolicySession` hands
the raw `Evaluation` instead.

## Connections

- `hexo-engine` supplies `Position`, `Action`, `Search`, and the canonical legal
  order that priors index into.
- `hexo-runner` supplies `Decision`, `Game`, and the game-step protocol that
  consumes decisions.
- Model packages (`crates/hexo-model`, `crates/models/*`) provide concrete
  encoders, evaluators, and selectors.
- `crates/hexo-bot` pumps sessions on workers and owns the device batcher.
- `python/mantisnet/mantisnet/klent/search.py` is the independent Python parity
  implementation for Gumbel sequential halving.

## Files

| File | Description |
| --- | --- |
| `src/lib.rs` | Crate root: module declarations, re-exports, and a full drive-loop doc example |
| `src/seam.rs` | `Evaluation`, `Encoder` trait, `Evaluator` trait, and the `EncodedBatch` ragged arena |
| `src/session.rs` | `DecisionSession` trait, `LeafId`, and `SessionStatus` enum |
| `src/policy.rs` | `PolicySession`: single root evaluation per move |
| `src/mcts.rs` | `MctsConfig` and `MctsSession`: PUCT tree search with virtual loss, in-flight cap, and arena reuse |
| `src/gumbel.rs` | `GumbelConfig`, `GumbelSession`, and `GumbelTrace`: Gumbel sequential halving with deterministic lines and a halving schedule |
| `src/select.rs` | `SelectFromPolicy` and `SelectFromSearch` traits, `Child`, and `SearchOutcome` |
| `src/rng.rs` | `SplitMix64`: seeded 64-bit PRNG with `next_u64`, `next_f64`, and `below` sampling helpers |
