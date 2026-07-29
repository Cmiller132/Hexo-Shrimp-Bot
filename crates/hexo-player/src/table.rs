//! Two-seat game tables and interleaved table sweeps.

use crate::player::Player;
use hexo_engine::Player as Seat;
use hexo_runner::{Failure, Game, GameSpec, MatchResult, Reply, Step, SubmitError};

/// One game and the two seats playing it.
///
/// The table owns one player value per seat.
#[derive(Clone, Debug)]
pub struct Table<P> {
    game: Game,
    seats: [P; 2],
}

impl<P: Player> Table<P> {
    /// A new game under `spec`, with `seats` indexed by [`Seat::index`].
    pub fn new(spec: GameSpec, seats: [P; 2]) -> Self {
        Self {
            game: Game::new(spec),
            seats,
        }
    }

    /// The game, read-only.
    #[inline]
    #[must_use]
    pub const fn game(&self) -> &Game {
        &self.game
    }

    /// The result, if the game has ended.
    #[inline]
    #[must_use]
    pub const fn result(&self) -> Option<MatchResult> {
        self.game.result()
    }

    /// The player filling `seat`.
    #[inline]
    #[must_use]
    pub fn seat(&self, seat: Seat) -> &P {
        &self.seats[seat.index()]
    }

    /// Take both players back.
    #[must_use]
    pub fn into_seats(self) -> [P; 2] {
        self.seats
    }

    /// Ask the seat on turn for one placement and submit it — verbatim, hash
    /// attestation and diagnostics included. `Some` once the game has ended,
    /// whether on this placement or before the call.
    pub fn step(&mut self) -> Option<MatchResult> {
        let Step::NeedDecision {
            seat, generation, ..
        } = self.game.step()
        else {
            return self.game.result();
        };

        let decision = self.seats[seat.index()].choose(&self.game);
        match self.game.submit(generation, Reply::Place(decision)) {
            Ok(transition) => transition.result,
            Err(SubmitError::Desync { expected, got }) => {
                // This driver reports desync as a seat failure; transport-level
                // resynchronization must occur before submission.
                self.game
                    .submit(generation, Reply::Failed(Failure::Desync { expected, got }))
                    .expect("a refused submission leaves the generation unchanged")
                    .result
            }
            Err(e) => unreachable!("the generation was read from this game above: {e}"),
        }
    }

    /// Drive this game to its end, which [`GameSpec::ply_cap`] bounds.
    pub fn run(&mut self) -> MatchResult {
        loop {
            if let Some(result) = self.step() {
                return result;
            }
        }
    }
}

/// One placement for every unfinished table; returns how many are still running,
/// so `while sweep(&mut tables) > 0 {}` drives the whole set.
pub fn sweep<P: Player>(tables: &mut [Table<P>]) -> usize {
    let mut running = 0;
    for table in tables {
        if table.result().is_some() {
            continue;
        }
        if table.step().is_none() {
            running += 1;
        }
    }
    running
}
