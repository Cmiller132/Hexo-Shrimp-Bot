//! Authoritative Hexo rules and game state.
//!
//! This crate owns the definition of a Hexo position and every legal
//! transition between positions. It is the single source of truth for what the
//! game *is*: coordinates, board storage, legality, and terminal detection.
//!
//! Deliberate non-responsibilities — these belong to other crates, and keeping
//! them out is what lets this one compile to `wasm32` and be tested without a
//! Python toolchain:
//!
//! - No I/O, no clocks, no threads.
//! - No PyO3. Bindings live in a leaf crate that depends on this one.
//! - No model, tensor, or feature-encoding concepts. Encoders are consumers of
//!   the read surface this crate exposes, never part of it.
//! - No match orchestration or player communication. That is `hexo-runner`.
//!
//! Empty pending the rules rebuild. See `README.md` for the planned module map.
