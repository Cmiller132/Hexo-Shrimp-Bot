//! Player interfaces and synchronous table drivers.
//!
//! [`Player`] supplies decisions, [`Model`] separates self-play and evaluation
//! policies, and [`Table`] submits those decisions to a game.
//!
//! `hexo_engine::Player` is the seat — `P0` or `P1` — and [`Player`] here is what
//! fills it; a consumer importing both aliases one.
//!
//! ```
//! use hexo_player::{Player, Table, sweep};
//! use hexo_runner::{Decision, Game, GameSpec};
//! use std::num::NonZeroU32;
//!
//! /// Takes the lowest-ranked legal placement, every time.
//! struct Lowest;
//!
//! impl Player for Lowest {
//!     fn choose(&mut self, game: &Game) -> Decision {
//!         let action = game.position().nth_legal(0).expect("a running game has a legal placement");
//!         // The decision attests the position used for selection.
//!         Decision::new(action, game.position().zobrist())
//!     }
//! }
//!
//! let spec = GameSpec {
//!     ply_cap: NonZeroU32::new(32).expect("nonzero"),
//!     ..GameSpec::default()
//! };
//!
//! // Drive one game to completion.
//! let mut table = Table::new(spec, [Lowest, Lowest]);
//! assert!(table.run().is_contested());
//!
//! // Interleave one step from each unfinished table.
//! let mut tables: Vec<_> = (0..100).map(|_| Table::new(spec, [Lowest, Lowest])).collect();
//! while sweep(&mut tables) > 0 {}
//! assert!(tables.iter().all(|t| t.result().is_some()));
//! ```

pub mod model;
pub mod player;
pub mod table;

pub use model::{Mode, Model, ModelPlayer};
pub use player::Player;
pub use table::{Table, sweep};
