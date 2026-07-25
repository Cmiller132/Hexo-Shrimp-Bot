//! FROZEN COPY — test oracle only. **Never edit this crate to make a test pass.**
//!
//! Verbatim copy of `Hexo-BotTrainer-hexgt/packages/hexo_engine/rust/src` at
//! commit `93d7a761`, vendored 2026-07-24. See `README.md` in this crate for
//! the exact list of mechanical changes (dependency stripping and test removal
//! only) and for why this file is never the one that gets fixed.
//!
//! ---
//!
//! Hexo rule engine.
//!
//! This crate owns the authoritative game state and state transitions. Model,
//! search, and sample code live outside this crate so the rules layer stays
//! small, deterministic, and easy to audit.
//!
//! Consumed two ways:
//! - As an rlib by the sibling workspace crates `hexo_models` (dense_cnn +
//!   hexgt subcrates, threats_shared.rs, plus the #[path]-included hexgnn
//!   crate) and `hexo_utils` (state_hash.rs, records.rs).
//! - With the `python` feature, as the maturin-built extension
//!   `hexo_engine._rust` (pybridge.rs) behind python/hexo_engine/api.py.
//! See README.md in this package for the full contract map.

pub mod board;
pub mod coord;
pub mod error;
pub mod legal;
pub mod rules;
pub mod state;
pub mod tactics;

pub use board::{Board, BoardDelta, Stone};
pub use coord::{hex_distance, HexCoord};
pub use error::MoveError;
pub use legal::{
    pack_coord, unpack_coord, LegalMoveDelta, LegalMoveStore, PackedCoord, LEGAL_RADIUS,
};
pub use rules::is_legal_placement;
pub use state::{
    apply_placement, ApplyDelta, ApplyResult, GameOutcome, HexoState, MoveRecord, Placement,
    PlacementRecord, Player, TurnPhase,
};
pub use tactics::{
    Axis, WindowEntry, WindowKey, WindowKeyList, WindowStore, WindowStoreDelta, WindowUpdate,
    WINDOWS_PER_PLACEMENT, WINDOW_LEN,
};
