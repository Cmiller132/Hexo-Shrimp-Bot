# Documentation index

Load this file first to choose the contract or evidence source relevant to a
change. Specifications define obligations; measured results belong only in
`ABLATIONS.md`.

## Specifications

- [`ENGINE_SPEC.md`](ENGINE_SPEC.md) answers how Hexo state, actions, wins,
  adjudication, serialization, and golden vectors behave. Read it before
  changing the engine or any consumer that interprets engine data.
- [`MODEL_SPEC.md`](MODEL_SPEC.md) answers how MantisNet encodes positions and
  computes its trunk, policy, Q, and value outputs. Read it before changing
  model architecture, tensor layouts, losses, or fixtures.
- [`CONTAINER_SPEC.md`](CONTAINER_SPEC.md) answers how packages, evaluators,
  sessions, processes, checkpoints, and records compose. Read it before
  changing container execution or Rust/Python ownership boundaries.
- [`DECK_SPEC.md`](DECK_SPEC.md) answers how the control deck reads run
  artifacts and exposes lifecycle, telemetry, inspection, play, and match
  interfaces. Read it before changing the deck API, frontend, or Compose
  services.

## KLENT references and evidence

- [`KLENT_PAPER.md`](KLENT_PAPER.md) states the KLENT method and its mathematical
  basis. Read it when the algorithm or notation itself is in question.
- [`KLENT_FOR_HEXO.md`](KLENT_FOR_HEXO.md) maps KLENT onto Hexo's model,
  self-play, targets, and evaluation pipeline. Read it before changing that
  integration.
- [`ABLATIONS.md`](ABLATIONS.md) records measured comparisons and outcomes.
  Read it for empirical evidence; specifications may link here but do not
  restate its results.

## Crate and module READMEs

Every crate or module README uses this order:

- **Purpose** — what the component is and the responsibility it owns.
- **Public surface** — the APIs and modules a consumer touches.
- **Run / test** — exact commands for exercising and verifying it.
- **Connections** — the paths and components it consumes or serves.
- **Invariants & gotchas** — compatibility rules and hazards a newcomer must
  preserve.
