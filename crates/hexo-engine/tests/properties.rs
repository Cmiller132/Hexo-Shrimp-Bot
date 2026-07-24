//! Property tests over random legal games.
//!
//! Games are driven by a shrinkable vector of choice indices rather than a raw
//! seed, so a failure shrinks to a short move list.
//!
//! Every oracle here is written to disagree with the engine — see
//! `tests/common/mod.rs`.

mod common;

use common::{
    Rng, check_all_oracles, legal_set_oracle, nth_legal, phase_kind, turn_oracle, winners_oracle,
    zobrist_oracle,
};
use hexo_engine::{Action, HexCoord, MoveError, Player, Position, Search};
use proptest::prelude::*;

/// Plies a driven game runs for before giving up on termination.
const MAX_PLIES: usize = 80;

/// Play a game driven by `choices`, calling `on_ply` after each placement.
///
/// Returns the move list actually played.
fn drive(choices: &[u32], mut on_ply: impl FnMut(&Position, usize)) -> Vec<Action> {
    let mut pos = Position::new();
    let mut moves = Vec::new();
    for (i, &c) in choices.iter().enumerate() {
        if pos.is_terminal() {
            break;
        }
        let n = pos.legal_count();
        assert!(n > 0, "a non-terminal position must have legal moves");
        let a = nth_legal(&pos, c as usize).expect("legal move");
        let applied = pos
            .advance(a)
            .unwrap_or_else(|e| panic!("ply {i} {a:?} rejected: {e}"));
        assert_eq!(applied.action, a);
        assert_eq!(applied.outcome.is_some(), applied.winning_windows != 0);
        moves.push(a);
        on_ply(&pos, moves.len());
    }
    moves
}

/// A shrinkable choice vector.
fn choices(max: usize) -> impl Strategy<Value = Vec<u32>> {
    prop::collection::vec(any::<u32>(), 1..=max)
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 24, max_shrink_iters: 2_000, ..ProptestConfig::default() })]

    /// Properties 2, 3, 4, 5 at **every ply**: the legal-set oracle, the
    /// Zobrist oracle, the win oracle, and the turn sequence — plus the full
    /// Tier-A audit.
    #[test]
    fn all_oracles_hold_at_every_ply(cs in choices(MAX_PLIES)) {
        drive(&cs, check_all_oracles);
    }

    /// Property 1: `apply` then `undo` restores a `PartialEq` state, and
    /// `audit()` passes after every apply *and* every undo.
    #[test]
    fn apply_then_undo_restores_exactly(cs in choices(MAX_PLIES)) {
        let moves = drive(&cs, |_, _| {});
        let mut pos = Position::new();
        for (i, &a) in moves.iter().enumerate() {
            let before = pos.clone();
            let zobrist_before = pos.zobrist();
            let legal_before: Vec<Action> = pos.legal_actions().collect();

            let mut search = Search::new(&mut pos);
            let applied = search.apply(a).expect("legal");
            search.position().audit().expect("audit after apply");
            prop_assert_eq!(search.depth(), 1);

            prop_assert_eq!(search.undo(), Some(a));
            search.position().audit().expect("audit after undo");
            prop_assert_eq!(search.position(), &before, "ply {} did not round-trip", i);
            prop_assert_eq!(search.position().zobrist(), zobrist_before);
            prop_assert_eq!(
                search.position().legal_actions().collect::<Vec<_>>(),
                legal_before
            );

            // Re-apply for real and check the second apply reports the same.
            let again = search.apply(a).expect("legal again");
            prop_assert_eq!(again, applied);
            search.commit();
            drop(search);
        }
    }

    /// Property 7 (T5): for every prefix length `k`, a fresh `Position`
    /// advanced `k` times equals a `Search` that applied `n` plies and undid
    /// `n - k`. The two construction paths share no incremental bookkeeping
    /// with the comparison.
    #[test]
    fn replay_parity_for_every_prefix(cs in choices(40)) {
        let moves = drive(&cs, |_, _| {});
        let n = moves.len();

        // Fresh replays, one per prefix.
        let mut prefixes = Vec::with_capacity(n + 1);
        let mut fresh = Position::new();
        prefixes.push(fresh.clone());
        for &a in &moves {
            fresh.advance(a).expect("legal");
            prefixes.push(fresh.clone());
        }

        let mut pos = Position::new();
        let mut search = Search::new(&mut pos);
        for &a in &moves {
            search.apply(a).expect("legal");
        }
        for k in (0..=n).rev() {
            prop_assert_eq!(search.depth(), k);
            prop_assert_eq!(search.position(), &prefixes[k], "prefix {}", k);
            prop_assert_eq!(search.position().zobrist(), prefixes[k].zobrist());
            prop_assert_eq!(
                search.position().legal_actions().collect::<Vec<_>>(),
                prefixes[k].legal_actions().collect::<Vec<_>>()
            );
            if k > 0 {
                prop_assert_eq!(search.undo(), Some(moves[k - 1]));
            }
        }
        prop_assert!(search.at_floor());
        prop_assert_eq!(search.undo(), None);
    }

    /// Property 6a: arena geometry never leaks into an observable (hazard H9).
    ///
    /// One position is pre-grown far out and rewound — `undo` never shrinks the
    /// arena — the other is not. The identical game must then be
    /// indistinguishable at every ply, **including on whether `advance`
    /// accepts the placement at all**, which is the observable the growth
    /// policy has to keep history-independent.
    ///
    /// The pre-growth runs along a *single* axis, chosen by the case. A
    /// balanced pre-growth in all four directions re-centres the arena roughly
    /// where the flat position's own growth lands and cannot expose an
    /// asymmetric leak.
    #[test]
    fn growth_never_leaks_into_an_observable(
        cs in choices(40),
        spread in 1u16..90,
        dir in 0usize..4,
    ) {
        let moves = drive(&cs, |_, _| {});

        let mut grown = Position::new();
        grown.advance(Action::new(HexCoord::ORIGIN)).expect("opening");
        {
            let mut s = Search::new(&mut grown);
            let step = [(8i16, 0i16), (-8, 0), (0, 8), (0, -8)][dir];
            let (mut q, mut r) = (0i16, 0i16);
            for _ in 0..spread {
                q += step.0;
                r += step.1;
                if s.apply(Action::new(HexCoord::new(q, r))).is_err() {
                    break;
                }
            }
            s.unwind();
        }
        let mut flat = Position::new();
        flat.advance(Action::new(HexCoord::ORIGIN)).expect("opening");
        prop_assert_eq!(&grown, &flat, "pre-grown arena is already distinguishable");

        for (i, &a) in moves.iter().enumerate().skip(1) {
            let ga = grown.advance(a);
            let fa = flat.advance(a);
            prop_assert_eq!(&ga, &fa, "accept/refuse differs at ply {}", i);
            prop_assert!(ga.is_ok(), "a driven game was refused at ply {}: {:?}", i, ga);
            prop_assert_eq!(&grown, &flat, "positions differ at ply {}", i);
            prop_assert_eq!(grown.zobrist(), flat.zobrist());
            prop_assert_eq!(grown.legal_count(), flat.legal_count());
            prop_assert_eq!(
                grown.legal_actions().collect::<Vec<_>>(),
                flat.legal_actions().collect::<Vec<_>>()
            );
        }
        // `audit()` is O(arena) and the grown arena is deliberately enormous,
        // so it runs once at the end rather than on every ply.
        grown.audit().expect("grown audit");
        flat.audit().expect("flat audit");
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 4, max_shrink_iters: 64, ..ProptestConfig::default() })]

    /// Property 6a, driven all the way to [`MoveError::BoardExtentExceeded`]:
    /// a searched-and-rewound position and a freshly replayed one must agree on
    /// every accept *and* on the refusal itself.
    ///
    /// Property 6a alone cannot reach this: it drives ~40 plies, nowhere near
    /// the ceiling, and the leak only becomes observable where the ceiling
    /// bites. The spreading diagonal below widens the bounding box in both
    /// dimensions, which is the only shape of game the ceiling can stop.
    #[test]
    fn accept_and_refuse_agree_all_the_way_to_the_ceiling(
        spread in 1u16..70,
        dir in 0usize..4,
    ) {
        let mut searched = Position::new();
        searched.advance(Action::new(HexCoord::ORIGIN)).expect("opening");
        let mut flat = searched.clone();
        {
            let mut s = Search::new(&mut searched);
            let step = [(8i16, 0i16), (-8, 0), (0, 8), (0, -8)][dir];
            let (mut q, mut r) = (0i16, 0i16);
            for _ in 0..spread {
                q += step.0;
                r += step.1;
                if s.apply(Action::new(HexCoord::new(q, r))).is_err() {
                    break;
                }
            }
            s.unwind();
        }
        prop_assert_eq!(&searched, &flat);

        let (mut q, mut r) = (0i16, 0i16);
        let mut refused = false;
        for i in 0..900usize {
            if i % 2 == 0 { q += 8; } else { r += 8; }
            let a = Action::new(HexCoord::new(q, r));
            let sa = searched.advance(a);
            let fa = flat.advance(a);
            prop_assert_eq!(&sa, &fa, "ply {} at ({}, {})", i, q, r);
            if let Err(e) = sa {
                prop_assert!(matches!(e, MoveError::BoardExtentExceeded { .. }), "{:?}", e);
                prop_assert!(!e.is_rule_violation());
                prop_assert!(searched.is_legal(a), "the refused move was still legal");
                refused = true;
                break;
            }
        }
        prop_assert!(refused, "the spreading diagonal never reached the ceiling");
        prop_assert_eq!(&searched, &flat);
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 8, max_shrink_iters: 200, ..ProptestConfig::default() })]

    /// Property 6b: a game that deliberately spreads as far from the origin as
    /// the rules allow stays consistent and never panics.
    ///
    /// This is the adversarial case the spec calls out: legality permits a
    /// stone eight cells from the nearest stone, so two players who both like
    /// to extend widen the bounding box by 8 per ply and the dense arena grows
    /// quadratically. Far enough out that *does* reach
    /// [`MoveError::BoardExtentExceeded`] — the documented, typed, non-rule
    /// outcome, not a failure; a game spreading in every direction at once
    /// refuses once its padded box passes roughly 2048 x 2048, which the ply
    /// bound here does not necessarily reach. What is tested is that a refusal,
    /// if one happens, is clean: it is not a rule violation, the refused move is
    /// still reported legal, the position is untouched, and play continues
    /// elsewhere.
    #[test]
    fn spreading_games_stay_consistent_and_never_panic(seed in any::<u64>()) {
        let mut rng = Rng::new(seed);
        let mut pos = Position::new();
        pos.advance(Action::new(HexCoord::ORIGIN)).expect("opening");
        for ply in 1..140usize {
            if pos.is_terminal() {
                break;
            }
            // Take the legal cell furthest along a rotating direction, so the
            // arena grows on almost every ply.
            let dir = rng.next_u64() % 4;
            let pick = pos
                .legal_actions()
                .max_by_key(|a| {
                    let c = a.coord();
                    match dir {
                        0 => i32::from(c.q),
                        1 => -i32::from(c.q),
                        2 => i32::from(c.r),
                        _ => -i32::from(c.r),
                    }
                })
                .expect("a non-terminal position has legal moves");
            let snapshot = pos.zobrist();
            match pos.advance(pick) {
                Ok(_) => {}
                Err(e @ MoveError::BoardExtentExceeded { .. }) => {
                    prop_assert!(!e.is_rule_violation());
                    prop_assert_eq!(pos.zobrist(), snapshot, "refusal was not atomic");
                    prop_assert!(pos.is_legal(pick), "the refused move was still legal");
                    let n = pos.legal_count();
                    let fallback = nth_legal(&pos, rng.below(n)).expect("legal move");
                    if pos.advance(fallback).is_err() {
                        break;
                    }
                }
                Err(e) => prop_assert!(false, "ply {}: {:?}", ply, e),
            }
            if ply % 10 == 0 {
                prop_assert_eq!(
                    pos.legal_actions().map(Action::coord).collect::<Vec<_>>(),
                    legal_set_oracle(&pos)
                );
                prop_assert_eq!(pos.zobrist(), zobrist_oracle(&pos));
                prop_assert_eq!(pos.is_terminal(), !winners_oracle(&pos).is_empty());
            }
        }
        // `audit()` is O(arena) and this arena is at the ceiling, so it runs
        // once, at the end.
        pos.audit().expect("audit after a maximal spread");
    }

    /// Property 8: **random** legal games never hit a representation limit.
    #[test]
    fn representation_limits_are_unreachable(seed in any::<u64>()) {
        let mut rng = Rng::new(seed);
        let mut pos = Position::new();
        for _ in 0..300usize {
            if pos.is_terminal() {
                break;
            }
            let n = pos.legal_count();
            let a = nth_legal(&pos, rng.below(n)).expect("legal move");
            match pos.advance(a) {
                Ok(_) => {}
                Err(e @ (MoveError::CoordOutOfBounds(_)
                    | MoveError::BoardExtentExceeded { .. })) => {
                    prop_assert!(false, "hit a representation limit in normal play: {:?}", e);
                }
                Err(e) => prop_assert!(false, "a legal move was rejected: {:?}", e),
            }
        }
        pos.audit().expect("audit");
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 48, ..ProptestConfig::default() })]

    /// Property 5 (T4) for whole games: `P0; P1 P1; P0 P0; P1 P1; ...` with
    /// freeze applied at the terminal ply.
    #[test]
    fn turn_sequence_matches_the_documented_pattern(cs in choices(MAX_PLIES)) {
        let mut pos = Position::new();
        let mut ply = 0usize;
        for &c in &cs {
            if pos.is_terminal() {
                break;
            }
            // Before the placement, the mover and phase are the oracle's.
            let (mover, phase) = turn_oracle(ply);
            prop_assert_eq!(pos.current_player(), mover, "before ply {}", ply);
            prop_assert_eq!(
                phase_kind(pos.phase()),
                phase_kind(phase),
                "before ply {}", ply
            );

            let a = nth_legal(&pos, c as usize).expect("legal move");
            let applied = pos.advance(a).expect("legal");
            prop_assert_eq!(applied.mover, mover);
            prop_assert_eq!(phase_kind(applied.phase_before), phase_kind(phase));
            ply += 1;

            if applied.outcome.is_some() {
                // Freeze: the position still reports the *pre-move* state.
                prop_assert_eq!(pos.current_player(), mover, "freeze at ply {}", ply - 1);
                prop_assert_eq!(
                    phase_kind(pos.phase()),
                    phase_kind(phase),
                    "freeze at ply {}", ply - 1
                );
                prop_assert_eq!(applied.phase_after, applied.phase_before);
                break;
            }
            // Not a win: the transition is the documented one.
            prop_assert_eq!(phase_kind(pos.phase()), phase_kind(turn_oracle(ply).1));
            prop_assert_eq!(pos.current_player(), turn_oracle(ply).0);
        }
    }

    /// A win is reported by exactly one player, and only ever by the mover.
    #[test]
    fn only_the_mover_can_win(cs in choices(MAX_PLIES)) {
        let mut pos = Position::new();
        for &c in &cs {
            if pos.is_terminal() {
                break;
            }
            let mover = pos.current_player();
            let a = nth_legal(&pos, c as usize).expect("legal move");
            let applied = pos.advance(a).expect("legal");
            if let Some(o) = applied.outcome {
                prop_assert_eq!(o.winner, mover);
                prop_assert_eq!(winners_oracle(&pos), vec![mover]);
                prop_assert_eq!(pos.outcome(), Some(o));
                // Nothing at all is legal now.
                prop_assert_eq!(pos.legal_count(), 0);
                for probe in [HexCoord::ORIGIN, HexCoord::new(1, 1), HexCoord::new(-3, 4)] {
                    prop_assert!(!pos.is_legal(Action::new(probe)));
                }
            } else {
                prop_assert!(winners_oracle(&pos).is_empty());
            }
        }
    }

    /// Stone counts stay consistent with the ply pattern.
    #[test]
    fn stone_counts_track_the_ply_pattern(cs in choices(MAX_PLIES)) {
        let moves = drive(&cs, |pos, ply| {
            assert_eq!(pos.stone_count() as usize, ply);
            assert_eq!(
                pos.stone_count_for(Player::P0) + pos.stone_count_for(Player::P1),
                pos.stone_count()
            );
            assert_eq!(pos.stones().len(), ply);
        });
        prop_assert!(!moves.is_empty());
    }
}
