//! Game state and phase-aware move application.
//!
//! This is the heart of the rule engine. Hexo turns are represented
//! autoregressively:
//! - `Opening`: Player 0 places the center stone.
//! - `FirstStone`: current player places the first stone of a normal turn.
//! - `SecondStone`: the same player places the second stone, then turn passes.
//!
//! A win is checked after every single placement. If the first stone of a
//! two-stone turn wins, the second stone is never played.
//!
//! VENDORED CHANGE: `serde` derives removed; the state export/replay pair that
//! lived on top of `snapshot.rs` removed with it; unit tests removed.

use super::board::{Board, BoardDelta};
use super::coord::HexCoord;
use super::error::MoveError;
use super::legal::{pack_coord, PackedCoord};
use super::rules::is_legal_placement;
use super::tactics::WindowUpdate;

/// Player identifier and stone owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Player {
    Player0,
    Player1,
}

impl Player {
    /// Return the opponent.
    pub fn other(self) -> Self {
        match self {
            Self::Player0 => Self::Player1,
            Self::Player1 => Self::Player0,
        }
    }

    /// Stable zero-based index for arrays and tensors.
    pub fn index(self) -> usize {
        match self {
            Self::Player0 => 0,
            Self::Player1 => 1,
        }
    }
}

/// Where the current player is inside the autoregressive turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TurnPhase {
    /// Game start. Only Player 0 at `(0, 0)` is legal.
    Opening,
    /// First placement of a normal two-stone turn.
    FirstStone,
    /// Second placement of the same turn; stores the first coordinate so the
    /// same cell cannot be reused and encoders can mark it.
    SecondStone { first: HexCoord },
}

/// One single-stone action.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Placement {
    pub coord: HexCoord,
}

/// Terminal result. Hexo has no normal draw under the current rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GameOutcome {
    /// Winning player.
    pub winner: Player,
    /// Number of stones placed when the game ended.
    pub placements: u32,
}

/// Flat history record for encoders and training samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlacementRecord {
    /// Player who placed the stone.
    pub player: Player,
    /// Coordinate that was placed.
    pub coord: HexCoord,
    /// Phase before the stone was placed.
    pub phase: TurnPhase,
    /// One-based placement count after this stone is applied.
    pub placement_index: u32,
}

/// Human-sized record of the most recent logical turn.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MoveRecord {
    /// Player who took the turn.
    pub player: Player,
    /// One coordinate for opening, two coordinates for a full normal turn.
    pub placements: Vec<HexCoord>,
}

/// Complete Hexo game state.
#[derive(Clone, Debug)]
pub struct HexoState {
    /// Sparse unlimited board.
    board: Board,
    /// Player who chooses the next placement.
    current_player: Player,
    /// Current point in the opening/first/second placement sequence.
    phase: TurnPhase,
    /// Total number of stones placed.
    placements_made: u32,
    /// Set once a player has six in a line.
    terminal: Option<GameOutcome>,
    /// Most recent logical turn progress.
    last_turn: Option<MoveRecord>,
    /// Full single-placement history for encoding recent stones.
    placement_history: Vec<PlacementRecord>,
}

/// Summary returned after applying one placement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApplyResult {
    /// Coordinate that was placed.
    pub placed: HexCoord,
    /// Player who placed the stone.
    pub player: Player,
    /// Phase before applying the placement.
    pub phase_before: TurnPhase,
    /// Phase after applying the placement. Unchanged if the move ended game.
    pub phase_after: TurnPhase,
    /// Terminal outcome if this placement won immediately.
    pub outcome: Option<GameOutcome>,
    /// Windows changed by this placement plus any threat/win windows.
    pub window_update: WindowUpdate,
}

/// State and board changes made by one placement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApplyDelta {
    board: BoardDelta,
    previous_current_player: Player,
    previous_phase: TurnPhase,
    previous_placements_made: u32,
    previous_terminal: Option<GameOutcome>,
    previous_last_turn: Option<MoveRecord>,
    previous_history_len: usize,
}

impl Default for HexoState {
    fn default() -> Self {
        Self::new()
    }
}

impl HexoState {
    /// Create the initial empty game state.
    pub fn new() -> Self {
        Self {
            board: Board::new(),
            current_player: Player::Player0,
            phase: TurnPhase::Opening,
            placements_made: 0,
            terminal: None,
            last_turn: None,
            placement_history: Vec::new(),
        }
    }

    /// Read-only access to board occupancy.
    pub fn board(&self) -> &Board {
        &self.board
    }

    /// Player who must choose the next single placement.
    pub fn current_player(&self) -> Player {
        self.current_player
    }

    /// Current turn phase.
    pub fn phase(&self) -> TurnPhase {
        self.phase
    }

    /// Total stones placed so far.
    pub fn placements_made(&self) -> u32 {
        self.placements_made
    }

    /// Terminal result, if the game has ended.
    pub fn terminal(&self) -> Option<GameOutcome> {
        self.terminal
    }

    /// True once no more moves should be generated.
    pub fn is_terminal(&self) -> bool {
        self.terminal.is_some()
    }

    /// Most recent logical turn progress.
    pub fn last_turn(&self) -> Option<&MoveRecord> {
        self.last_turn.as_ref()
    }

    /// Complete single-placement history.
    pub fn placement_history(&self) -> &[PlacementRecord] {
        &self.placement_history
    }

    /// Number of legal single-stone moves in the current state.
    pub fn legal_move_count(&self) -> usize {
        if self.terminal.is_some() {
            return 0;
        }

        match self.phase {
            TurnPhase::Opening => usize::from(self.board.is_cell_empty(HexCoord::ZERO)),
            TurnPhase::FirstStone | TurnPhase::SecondStone { .. } => self.board.legal_moves().len(),
        }
    }

    /// Fill `out` with deterministic legal single-stone move coordinates.
    pub fn write_legal_moves(&self, out: &mut Vec<HexCoord>) {
        out.clear();

        if self.terminal.is_some() {
            return;
        }

        match self.phase {
            TurnPhase::Opening => {
                if self.board.is_cell_empty(HexCoord::ZERO) {
                    out.push(HexCoord::ZERO);
                }
            }
            TurnPhase::FirstStone | TurnPhase::SecondStone { .. } => {
                self.board.legal_moves().write_coords(out);
            }
        }
    }

    /// Fill `out` with deterministic compact legal action IDs.
    pub fn write_legal_action_ids(&self, out: &mut Vec<PackedCoord>) {
        out.clear();

        if self.terminal.is_some() {
            return;
        }

        match self.phase {
            TurnPhase::Opening => {
                if self.board.is_cell_empty(HexCoord::ZERO) {
                    out.push(pack_coord(HexCoord::ZERO));
                }
            }
            TurnPhase::FirstStone | TurnPhase::SecondStone { .. } => {
                self.board.legal_moves().write_action_ids(out);
            }
        }
    }

    /// Append a single-stone history entry after placement succeeds.
    fn push_history(&mut self, player: Player, coord: HexCoord, phase: TurnPhase) {
        self.placement_history.push(PlacementRecord {
            player,
            coord,
            phase,
            placement_index: self.placements_made,
        });
    }

    fn record_turn_progress(&mut self, player: Player, coord: HexCoord, phase: TurnPhase) {
        let placements = match phase {
            TurnPhase::Opening | TurnPhase::FirstStone => vec![coord],
            TurnPhase::SecondStone { first } => vec![first, coord],
        };
        self.last_turn = Some(MoveRecord { player, placements });
    }

    /// Apply one placement and return an explicit undo delta.
    ///
    /// This is the engine's MCTS hot path: the model crates
    /// (hexo_models/dense_cnn, hexo_models/hexgt, hexgnn) drive search via
    /// apply/undo on capsule-cloned states. Note `previous_last_turn` clones a
    /// heap Vec on every placement purely to support `undo`.
    pub fn apply_with_delta(
        &mut self,
        placement: Placement,
    ) -> Result<(ApplyResult, ApplyDelta), MoveError> {
        is_legal_placement(self, placement.coord)?;

        let previous_current_player = self.current_player;
        let previous_phase = self.phase;
        let previous_placements_made = self.placements_made;
        let previous_terminal = self.terminal;
        let previous_last_turn = self.last_turn.clone();
        let previous_history_len = self.placement_history.len();

        let player = self.current_player;
        let phase_before = self.phase;
        let (window_update, board_delta) = self.board.place_with_delta(placement.coord, player)?;
        self.placements_made += 1;
        self.push_history(player, placement.coord, phase_before);
        self.record_turn_progress(player, placement.coord, phase_before);

        let outcome = if window_update.has_win() {
            let outcome = GameOutcome {
                winner: player,
                placements: self.placements_made,
            };
            self.terminal = Some(outcome);
            Some(outcome)
        } else {
            match phase_before {
                TurnPhase::Opening => {
                    // Opening is a special one-stone turn by Player 0. After it,
                    // Player 1 starts the first normal two-stone turn.
                    self.current_player = Player::Player1;
                    self.phase = TurnPhase::FirstStone;
                }
                TurnPhase::FirstStone => {
                    // The same player remains to place the second stone.
                    self.phase = TurnPhase::SecondStone {
                        first: placement.coord,
                    };
                }
                TurnPhase::SecondStone { .. } => {
                    // A normal two-stone turn is complete, so control passes.
                    self.current_player = player.other();
                    self.phase = TurnPhase::FirstStone;
                }
            }
            None
        };

        let result = ApplyResult {
            placed: placement.coord,
            player,
            phase_before,
            phase_after: self.phase,
            outcome,
            window_update,
        };
        let delta = ApplyDelta {
            board: board_delta,
            previous_current_player,
            previous_phase,
            previous_placements_made,
            previous_terminal,
            previous_last_turn,
            previous_history_len,
        };

        Ok((result, delta))
    }

    /// Restore the exact state that existed before `apply_with_delta`.
    pub fn undo(&mut self, delta: ApplyDelta) {
        self.board.undo_place(delta.board);
        self.current_player = delta.previous_current_player;
        self.phase = delta.previous_phase;
        self.placements_made = delta.previous_placements_made;
        self.terminal = delta.previous_terminal;
        self.last_turn = delta.previous_last_turn;
        self.placement_history.truncate(delta.previous_history_len);
    }
}

/// Apply one single-stone placement and advance the phase machine.
///
/// The function performs the full rule sequence:
/// 1. Validate the coordinate against the current phase.
/// 2. Place the stone for the current player.
/// 3. Record history.
/// 4. Check for an immediate six-in-line win.
/// 5. If not terminal, advance phase/current player.
pub fn apply_placement(
    state: &mut HexoState,
    placement: Placement,
) -> Result<ApplyResult, MoveError> {
    state
        .apply_with_delta(placement)
        .map(|(result, _delta)| result)
}
