//! Hand-built win fixtures the property generator will not find on its own.

mod common;

use common::{check_all_oracles, winners_oracle};
use hexo_engine::{
    Action, Axis, HexCoord, MoveError, Outcome, Player, Position, Search, TurnPhase, WINDOW_LEN,
};

fn act(q: i16, r: i16) -> Action {
    Action::new(HexCoord::new(q, r))
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
    expect_window_bits: u32,
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
        applied.winning.bits(),
        expect_window_bits,
        "winning window slots"
    );
    assert!(
        !applied.winning.is_empty(),
        "outcome is Some, so winning must be non-empty"
    );
    assert_eq!(
        applied.winning.count(),
        expect_window_bits.count_ones(),
        "count must agree with the raw mask"
    );
    let resolved: Vec<_> = applied.winning_windows().collect();
    assert_eq!(resolved.len(), applied.winning.count() as usize);
    for w in &resolved {
        assert!(
            search.position().window(*w).is_full_for(winner),
            "resolved window {w:?} is not full for the winner"
        );
        assert!(
            w.cells().contains(&applied.action.coord()),
            "resolved window {w:?} does not contain the placement"
        );
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

/// Slot index of a window in the canonical order of the spec.
const fn slot(axis_index: usize, offset: usize) -> u32 {
    1 << (axis_index * WINDOW_LEN + offset)
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
        |p| matches!(p, TurnPhase::SecondStone { .. }),
        slot(Axis::Q.index(), 5),
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
    match pos.phase() {
        TurnPhase::SecondStone { first } => {
            assert_eq!(first, HexCoord::new(5, 0));
            assert_eq!(pos.get(first), Some(Player::P1));
        }
        other => panic!("expected SecondStone, got {other:?}"),
    }
}

#[test]
fn first_stone_win_with_seven_in_a_row_sets_two_window_bits() {
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
        slot(Axis::Q.index(), 3) | slot(Axis::Q.index(), 4),
    );
}

#[test]
fn two_crossing_lines_set_two_window_bits() {
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
        slot(Axis::Q.index(), 5) | slot(Axis::R.index(), 0),
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
        |p| matches!(p, TurnPhase::SecondStone { .. }),
        slot(Axis::Q.index(), 2),
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
        |p| matches!(p, TurnPhase::SecondStone { .. }),
        slot(Axis::QR.index(), 2),
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
        |p| matches!(p, TurnPhase::SecondStone { .. }),
        slot(Axis::R.index(), 5),
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
        slot(Axis::Q.index(), 0),
    );
}
