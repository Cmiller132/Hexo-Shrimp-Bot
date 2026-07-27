//! Hand-built win fixtures the property generator will not find on its own.

mod common;

use common::{check_all_oracles, winners_oracle};
use hexo_engine::{
    Action, Axis, HexCoord, MoveError, Outcome, Player, Position, Search, TurnPhase, Win,
};

fn act(q: i16, r: i16) -> Action {
    Action::new(HexCoord::new(q, r))
}

/// The run a fixture expects, written the way the fixture's comment reads it off the
/// move list: first cell, axis, length.
const fn win(q: i16, r: i16, axis: Axis, len: u8) -> Option<Win> {
    Some(Win {
        axis,
        start: HexCoord::new(q, r),
        len,
    })
}

/// Play every move but the last, returning the position and the final move.
fn setup(moves: &[(i16, i16)]) -> (Position, Action) {
    let mut pos = Position::new();
    let (last, rest) = moves.split_last().expect("non-empty move list");
    for &(q, r) in rest {
        pos.advance(act(q, r))
            .unwrap_or_else(|e| panic!("({q}, {r}) rejected: {e}"));
        check_all_oracles(&pos, pos.stone_count() as usize);
    }
    (pos, act(last.0, last.1))
}

/// Apply the winning move, audit, undo it, audit, re-apply, and require the hash and
/// the whole position to come back identical.
fn assert_win_cycle(
    moves: &[(i16, i16)],
    winner: Player,
    expect_phase_kind_frozen: fn(TurnPhase) -> bool,
    expect_wins: [Option<Win>; 3],
) {
    let (mut pos, winning_move) = setup(moves);
    let before = pos.clone();
    let zobrist_before = pos.zobrist();
    assert!(!pos.is_terminal(), "the setup must not already be won");

    let mut search = Search::new(&mut pos);
    let applied = search
        .apply(winning_move)
        .expect("the winning move is legal");

    assert_eq!(applied.outcome, Some(Outcome { winner }));
    assert_eq!(applied.mover, winner);
    assert_eq!(
        applied.phase_after, applied.phase_before,
        "a winning placement must freeze the phase"
    );
    assert_eq!(
        applied.wins, expect_wins,
        "the runs this placement completed"
    );
    assert!(
        applied.wins.iter().any(Option::is_some),
        "outcome is Some, so some axis must report a run"
    );
    for (i, w) in applied.wins.iter().enumerate() {
        let Some(w) = w else { continue };
        assert_eq!(w.axis.index(), i, "wins must be indexed by axis");
        let mut cell = w.start;
        let mut hit = false;
        for _ in 0..w.len {
            assert_eq!(
                search.position().get(cell),
                Some(winner),
                "run {w:?} names {cell:?}, which is not the winner's"
            );
            hit |= cell == applied.action.coord();
            cell = cell.step(w.axis, 1);
        }
        assert!(hit, "run {w:?} does not contain the placement");
    }

    let won = search.position();
    assert!(won.is_terminal());
    assert_eq!(won.current_player(), winner, "the mover freezes too");
    assert!(expect_phase_kind_frozen(won.phase()), "{:?}", won.phase());
    assert_eq!(won.legal_count(), 0);
    assert_eq!(won.legal_actions().count(), 0);
    assert_eq!(winners_oracle(won), vec![winner]);
    won.audit().expect("audit after the winning move");
    let zobrist_won = won.zobrist();
    assert_ne!(zobrist_won, zobrist_before);

    let mut terminal_probe = won.clone();
    assert_eq!(
        terminal_probe.advance(act(0, 0)),
        Err(MoveError::TerminalState)
    );

    assert_eq!(search.undo(), Some(winning_move));
    assert!(search.at_floor());
    search
        .position()
        .audit()
        .expect("audit after undoing the winning move");
    assert_eq!(search.position(), &before, "undo must restore exactly");
    assert_eq!(search.position().zobrist(), zobrist_before);
    assert!(!search.position().is_terminal(), "undo must un-freeze");

    let again = search.apply(winning_move).expect("still legal");
    assert_eq!(again, applied, "re-applying must be identical");
    assert_eq!(search.position().zobrist(), zobrist_won);
    drop(search);

    let mut fresh = Position::new();
    for &(q, r) in moves {
        fresh.advance(act(q, r)).expect("replay");
    }
    assert_eq!(fresh.zobrist(), zobrist_won);
    assert_eq!(fresh.outcome(), Some(Outcome { winner }));
    check_all_oracles(&fresh, fresh.stone_count() as usize);
}

#[test]
fn second_stone_win_freezes_mid_turn() {
    assert_win_cycle(
        &[
            (0, 0),
            (1, 0),
            (2, 0),
            (0, 1),
            (0, 3),
            (3, 0),
            (4, 0),
            (0, 5),
            (0, 7),
            (5, 0),
            (6, 0),
        ],
        Player::P1,
        |p| matches!(p, TurnPhase::SecondStone),
        // P1 holds (1, 0)..(5, 0) and closes at (6, 0); (0, 0) is P0's, so the run is
        // the six cells (1, 0)..(6, 0) along Q.
        [win(1, 0, Axis::Q, 6), None, None],
    );

    let mut pos = Position::new();
    for &(q, r) in &[
        (0, 0),
        (1, 0),
        (2, 0),
        (0, 1),
        (0, 3),
        (3, 0),
        (4, 0),
        (0, 5),
        (0, 7),
        (5, 0),
        (6, 0),
    ] {
        pos.advance(act(q, r)).expect("legal");
    }
    assert_eq!(pos.phase(), TurnPhase::SecondStone);
    // The turn's first stone is still on the board, which is what makes a frozen
    // `SecondStone` a trap for consumer code that branches on phase before terminal.
    assert_eq!(pos.get(HexCoord::new(5, 0)), Some(Player::P1));
}

#[test]
fn first_stone_win_with_seven_in_a_row_reports_a_run_of_seven() {
    assert_win_cycle(
        &[
            (0, 0),
            (1, 0),
            (2, 0),
            (0, 1),
            (0, 3),
            (3, 0),
            (4, 0),
            (0, 5),
            (0, 7),
            (6, 0),
            (7, 0),
            (0, 9),
            (0, 11),
            (5, 0),
        ],
        Player::P1,
        |p| matches!(p, TurnPhase::FirstStone),
        // P1 holds (1, 0)..(4, 0) and (6, 0), (7, 0); filling the gap at (5, 0) joins
        // them into seven along Q, from (1, 0) to (7, 0).
        [win(1, 0, Axis::Q, 7), None, None],
    );
}

#[test]
fn two_crossing_lines_report_a_run_on_each_axis() {
    assert_win_cycle(
        &[
            (0, 0),
            (1, 0),
            (2, 0),
            (0, 1),
            (0, 3),
            (3, 0),
            (4, 0),
            (0, 5),
            (0, 7),
            (5, 0),
            (6, 1),
            (0, 9),
            (0, 11),
            (6, 2),
            (6, 3),
            (0, 13),
            (0, 15),
            (6, 4),
            (6, 5),
            (0, 17),
            (0, 19),
            (6, 0),
        ],
        Player::P1,
        |p| matches!(p, TurnPhase::FirstStone),
        // (6, 0) is the crossing point: it closes (1, 0)..(6, 0) along Q and
        // (6, 0)..(6, 5) along R at once.
        [win(1, 0, Axis::Q, 6), win(6, 0, Axis::R, 6), None],
    );
}

#[test]
fn win_completed_away_from_the_window_ends() {
    assert_win_cycle(
        &[
            (0, 0),
            (1, 0),
            (2, 0),
            (0, 1),
            (0, 3),
            (4, 0),
            (5, 0),
            (0, 5),
            (0, 7),
            (6, 0),
            (3, 0),
        ],
        Player::P1,
        |p| matches!(p, TurnPhase::SecondStone),
        // The placement at (3, 0) is interior to the run it completes: two of P1's
        // stones sit behind it and three ahead, giving (1, 0)..(6, 0) along Q.
        [win(1, 0, Axis::Q, 6), None, None],
    );
}

#[test]
fn win_on_the_qr_axis_exercises_the_shear() {
    assert_win_cycle(
        &[
            (0, 0),
            (1, -1),
            (2, -2),
            (0, 1),
            (0, 3),
            (4, -4),
            (5, -5),
            (0, 5),
            (0, 7),
            (6, -6),
            (3, -3),
        ],
        Player::P1,
        |p| matches!(p, TurnPhase::SecondStone),
        // The same interior fill as above, on the QR diagonal: (1, -1)..(6, -6).
        [None, None, win(1, -1, Axis::QR, 6)],
    );
}

#[test]
fn win_on_the_r_axis() {
    assert_win_cycle(
        &[
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 2),
            (2, 4),
            (1, 2),
            (1, 3),
            (2, 6),
            (2, 8),
            (1, 4),
            (1, 5),
        ],
        Player::P1,
        |p| matches!(p, TurnPhase::SecondStone),
        // P1's column at q = 1 closes upward: (1, 0)..(1, 5) along R.
        [None, win(1, 0, Axis::R, 6), None],
    );
}

#[test]
fn p0_can_win_too() {
    assert_win_cycle(
        &[
            (0, 0),
            (5, 0),
            (5, 2),
            (-1, 0),
            (-2, 0),
            (5, 4),
            (5, 6),
            (-3, 0),
            (-4, 0),
            (5, 8),
            (5, 10),
            (-5, 0),
        ],
        Player::P0,
        |p| matches!(p, TurnPhase::FirstStone),
        // P0 runs its opening stone backwards to (-5, 0), which is the run's first cell:
        // (-5, 0)..(0, 0) along Q.
        [win(-5, 0, Axis::Q, 6), None, None],
    );
}
