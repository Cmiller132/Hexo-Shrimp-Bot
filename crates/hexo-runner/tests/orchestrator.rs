//! The host-side seam: a driver submits a seat's answer without ruling on it.

use hexo_engine::{Action, ActionId, HexCoord, MoveError, Player};
use hexo_runner::{Decision, Game, GameSpec, MatchResult, Reply, Step, WinReason};

#[test]
fn an_orchestrator_preserves_an_illegal_seat_answer_in_the_forfeit() {
    let mut game = Game::new(GameSpec::default());
    let Step::NeedDecision {
        seat,
        generation,
        zobrist,
        ..
    } = game.step()
    else {
        panic!("a new game needs its opening decision");
    };
    assert_eq!(seat, Player::P0);

    // This is the seat-authored payload as a transport adapter would decode it.
    // The host does not preflight its legality; Game remains the one authority.
    let coordinate = HexCoord::new(1, -1);
    let wire_action = ActionId::from_coord(coordinate);
    let decision = Decision::new(Action::from_id(wire_action), zobrist);
    let transition = game
        .submit(generation, Reply::Place(decision))
        .expect("an illegal answer is adjudicated, not rejected as a submit error");

    assert_eq!(
        transition.result,
        Some(MatchResult::Decisive {
            winner: Player::P1,
            reason: WinReason::IllegalMove {
                action: wire_action,
                cause: MoveError::IllegalOpening,
            },
        })
    );
    assert!(
        transition.applied.is_none(),
        "a forfeiting action is not placed"
    );
    assert!(game.plies().is_empty(), "a forfeiting action is not a ply");
    assert_eq!(
        game.position().stone_count(),
        0,
        "adjudication leaves the canonical position untouched"
    );
}
