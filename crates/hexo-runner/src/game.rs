//! The authoritative game: a state machine that never blocks and never calls
//! out.
//!
//! # Why this shape
//!
//! The obvious design is a `Player` trait the runner calls:
//! `fn decide(&mut self, ..) -> Decision`. The runner drives the loop and blocks
//! inside `decide` until the seat answers. It reads well, and for one game it is
//! fine.
//!
//! It makes a game equal to a thread. For a container seat that is a blocked
//! pipe read; for a human, a blocked UI wait. Ten thousand concurrent self-play
//! games is then ten thousand OS threads — and worse, it forecloses batching:
//! every thread is blocked inside `decide`, inside its search, on a
//! single-position evaluation, so there is nothing left to coalesce into the
//! batch of 512 a GPU wants.
//!
//! So the control flow is inverted. [`Game`] holds the canonical position and
//! **nothing else** — no player handle, no transport, no clock, no I/O. It
//! cannot block because it has nobody to block on. A caller asks it what it
//! wants with [`Game::step`] and tells it what happened with [`Game::submit`].
//! Both deployment shapes fall out of the one type:
//!
//! - *One game, one thread.* A loop that steps, blocks on its seat however it
//!   likes, and submits. Around fifteen lines, and nothing is lost versus the
//!   trait design.
//! - *Ten thousand games, a few threads.* One actor owns thousands of `Game`
//!   values, sweeps them for everyone sitting in [`Step::NeedDecision`], hands
//!   that whole set to a batched evaluator, and submits what comes back. No game
//!   owns a thread.
//!
//! This is *more* synchronous than the callback design, not less: no `async fn`,
//! no executor, no futures, no channels. Whether a caller blocks one thread or
//! polls a thousand games is decided entirely outside this crate.
//!
//! It also makes S6's "exactly one authority per game" structural. A `Game` that
//! cannot call out cannot accidentally adjudicate somebody else's game.

use crate::decision::{Budget, Failure, Reply};
use crate::error::SubmitError;
use crate::outcome::{DrawReason, MatchResult, NoContest, WinReason};
use hexo_engine::{Action, ActionId, Applied, Player, Position};

/// What a driver-reported [`Failure`] costs the seat it happened to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum FailurePolicy {
    /// The failing seat loses. Correct for tournaments and evaluation: a seat
    /// that cannot answer has lost, the same as one that cannot answer legally.
    #[default]
    Forfeit,
    /// The game is a no-contest. Correct for self-play, where a bot that crashed
    /// is a broken run rather than evidence about how well it plays — recording
    /// it as a loss would teach the network from its own infrastructure faults.
    NoContest,
}

/// The match rules a game is played under.
///
/// Everything here is a rule of the *match*, not of the game. The engine has no
/// counterpart for any of it and deliberately refuses to model them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GameSpec {
    /// Placements after which the game is [`DrawReason::PlyCap`].
    ///
    /// Checked after each accepted placement. Hexo has no natural ending short
    /// of a win — stones are only added, on an unbounded board — so without a
    /// cap a pathological pair of seats plays forever.
    ///
    /// One cap, in one place, and it is recorded. The previous implementation
    /// had `max_actions = 1024` in the runner and `max_game_plies` of 256 or 512
    /// in model configs, so the real limit was a different number in a different
    /// layer from the one being enforced.
    pub ply_cap: u32,
    /// What each seat is told it has to think with. Stated and recorded, never
    /// enforced here — this type has no clock.
    pub budget: Budget,
    /// What a driver-reported failure costs.
    pub on_failure: FailurePolicy,
}

impl Default for GameSpec {
    fn default() -> Self {
        Self {
            ply_cap: 512,
            budget: Budget::Unlimited,
            on_failure: FailurePolicy::Forfeit,
        }
    }
}

/// One placement as recorded.
///
/// The placement itself is also in [`Position::history`]; what is here is
/// everything the *match* knows that the position does not.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlyRecord {
    /// Who placed.
    pub seat: Player,
    /// What was placed, in the record encoding.
    pub action: ActionId,
    /// What that seat had been told it could spend.
    pub budget: Budget,
    /// The position hash after the placement. A replay that disagrees here has
    /// found its divergence at this exact ply.
    pub zobrist_after: u64,
    /// Seat-owned bytes, stored verbatim and never interpreted.
    pub diagnostics: Option<Vec<u8>>,
}

/// What the game wants next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    /// A seat must choose. **Nothing happens until [`Game::submit`] is called** —
    /// asking twice is free and changes nothing.
    NeedDecision {
        /// Who must choose.
        seat: Player,
        /// The token this decision must be submitted with.
        generation: u64,
        /// What this seat is told it may spend.
        budget: Budget,
        /// The hash of the position being decided in. The seat echoes it back.
        zobrist: u64,
        /// Placements made so far.
        ply: u32,
    },
    /// The game is over.
    Finished(MatchResult),
}

/// What an accepted submission did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Transition {
    /// The placement, if the submission was one. `None` for a resignation or a
    /// reported failure.
    ///
    /// A driver broadcasts this to both seats' mirrors, which apply it with
    /// `Position::advance`.
    pub applied: Option<Applied>,
    /// The position hash after the submission. Sent alongside `applied` so a
    /// mirror can check itself.
    pub zobrist: u64,
    /// The token the *next* decision must be submitted with.
    pub generation: u64,
    /// `Some` if the game just ended.
    pub result: Option<MatchResult>,
}

/// The one authoritative game.
///
/// Owns the canonical [`Position`] privately. It is handed out as a shared
/// borrow or copied, never as a mutable handle — a seat that could mutate it
/// would be a second authority.
#[derive(Clone, Debug)]
pub struct Game {
    spec: GameSpec,
    position: Position,
    generation: u64,
    result: Option<MatchResult>,
    plies: Vec<PlyRecord>,
}

impl Game {
    /// A new game under `spec`, at the empty position with `P0` to open.
    #[must_use]
    pub fn new(spec: GameSpec) -> Self {
        Self {
            spec,
            position: Position::new(),
            generation: 0,
            result: None,
            plies: Vec::new(),
        }
    }

    /// The match rules in force.
    #[inline]
    #[must_use]
    pub const fn spec(&self) -> &GameSpec {
        &self.spec
    }

    /// The canonical position, read-only.
    ///
    /// Never handed out mutably. [`Game::submit`] is the only way to advance it,
    /// which is what makes this type the single authority rather than a
    /// convention that it is.
    #[inline]
    #[must_use]
    pub const fn position(&self) -> &Position {
        &self.position
    }

    /// Every placement so far, oldest first, with what the match knows about it.
    #[inline]
    #[must_use]
    pub fn plies(&self) -> &[PlyRecord] {
        &self.plies
    }

    /// The result, if the game has ended.
    #[inline]
    #[must_use]
    pub const fn result(&self) -> Option<MatchResult> {
        self.result
    }

    /// The move prefix a seat needs to build its own mirror.
    ///
    /// A move list, not a serialised position: a container cannot be handed a
    /// [`Position`], and board-shaped construction is the rule-bypass hole the
    /// engine refuses to reopen. The seat calls `Position::replay` on this.
    #[inline]
    #[must_use]
    pub fn prefix(&self) -> &[Action] {
        self.position.history()
    }

    /// What the game wants next. Pure; call it as often as you like.
    #[must_use]
    pub fn step(&self) -> Step {
        match self.result {
            Some(result) => Step::Finished(result),
            None => Step::NeedDecision {
                seat: self.position.current_player(),
                generation: self.generation,
                budget: self.spec.budget,
                zobrist: self.position.zobrist(),
                ply: self.position.stone_count(),
            },
        }
    }

    /// Report what a seat came back with.
    ///
    /// `generation` must be the one from the [`Step::NeedDecision`] this reply
    /// answers. Once decisions are in flight — batched, queued, or across a
    /// process boundary — a reply can arrive for a position that has moved on,
    /// and applying it would silently corrupt the game. The token turns that
    /// into a typed refusal.
    ///
    /// # Errors
    /// [`SubmitError`], and on any of them **the game is unchanged**: the
    /// submission was unusable, so nothing happened and the same
    /// [`Step::NeedDecision`] is still outstanding.
    ///
    /// An **illegal placement is not an error here.** It is an accepted
    /// submission that adjudicates to a loss — a seat that cannot play legally
    /// has lost, and the driver did its job by delivering the answer. The one
    /// exception is a refusal the engine reports as its own limit rather than a
    /// rule violation, which is a [`NoContest::EngineLimit`] and blames nobody.
    /// [`hexo_engine::MoveError::is_rule_violation`] is what tells the two
    /// apart; it exists for this decision.
    pub fn submit(&mut self, generation: u64, reply: Reply) -> Result<Transition, SubmitError> {
        if self.result.is_some() {
            return Err(SubmitError::Finished);
        }
        if generation != self.generation {
            return Err(SubmitError::StaleGeneration {
                expected: self.generation,
                got: generation,
            });
        }

        let seat = self.position.current_player();
        match reply {
            Reply::Resign => Ok(self.finish(MatchResult::Decisive {
                winner: seat.other(),
                reason: WinReason::Resignation,
            })),
            Reply::Failed(failure) => Ok(self.finish(match self.spec.on_failure {
                FailurePolicy::Forfeit => MatchResult::Decisive {
                    winner: seat.other(),
                    reason: match failure {
                        Failure::Timeout => WinReason::Timeout,
                        Failure::Crashed => WinReason::Crash,
                        Failure::Protocol => WinReason::Protocol,
                    },
                },
                FailurePolicy::NoContest => MatchResult::NoContest(NoContest::Harness {
                    stage: match failure {
                        Failure::Timeout => "seat.timeout",
                        Failure::Crashed => "seat.crashed",
                        Failure::Protocol => "seat.protocol",
                    },
                }),
            })),
            Reply::Place(decision) => {
                let expected = self.position.zobrist();
                if decision.zobrist != expected {
                    return Err(SubmitError::Desync {
                        expected,
                        got: decision.zobrist,
                    });
                }
                match self.position.advance(decision.action) {
                    Ok(applied) => Ok(self.accept(seat, applied, decision.diagnostics)),
                    Err(error) if error.is_rule_violation() => {
                        Ok(self.finish(MatchResult::Decisive {
                            winner: seat.other(),
                            reason: WinReason::IllegalMove,
                        }))
                    }
                    Err(error) => Ok(self.finish(MatchResult::NoContest(NoContest::EngineLimit {
                        seat,
                        error,
                    }))),
                }
            }
        }
    }

    /// Record an accepted placement and decide whether it ended the game.
    fn accept(
        &mut self,
        seat: Player,
        applied: Applied,
        diagnostics: Option<Vec<u8>>,
    ) -> Transition {
        let zobrist = self.position.zobrist();
        self.plies.push(PlyRecord {
            seat,
            action: applied.action.id(),
            budget: self.spec.budget,
            zobrist_after: zobrist,
            diagnostics,
        });
        self.generation += 1;

        self.result = if let Some(outcome) = applied.outcome {
            Some(MatchResult::Decisive {
                winner: outcome.winner,
                reason: WinReason::SixInARow,
            })
        } else if self.position.stone_count() >= self.spec.ply_cap {
            Some(MatchResult::Drawn {
                reason: DrawReason::PlyCap,
            })
        } else {
            None
        };

        Transition {
            applied: Some(applied),
            zobrist,
            generation: self.generation,
            result: self.result,
        }
    }

    /// End the game without a placement.
    fn finish(&mut self, result: MatchResult) -> Transition {
        self.result = Some(result);
        Transition {
            applied: None,
            zobrist: self.position.zobrist(),
            generation: self.generation,
            result: Some(result),
        }
    }
}
