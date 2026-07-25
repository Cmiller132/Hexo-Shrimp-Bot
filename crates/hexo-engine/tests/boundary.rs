//! The coordinate-domain contract at all six faces of the cube.
//!
//! [`COORD_LIMIT`] bounds `q`, `r`, and `s`, so the domain is a hexagon with six
//! faces, and a game can be walked into any of them. Every read that names a
//! legal placement must agree with every check that validates one:
//! `legal_actions`, `legal_count`, `legal_rank`, `nth_legal`, `is_legal`, and
//! `advance` are six different implementations of "what may be played here", and
//! the boundary is the only place they can disagree.
//!
//! They did. Coverage was written over the whole radius-8 disk without testing
//! the domain, so a stone placed on a face marked out-of-domain cells as
//! frontier: at `q = 16000` the frontier held 136 coordinates that
//! `legal_actions` offered and `advance` refused with `CoordOutOfBounds`. A
//! sampler reading the legal set would pick one; a policy head indexed by
//! `legal_rank` would score one. `place` now skips those cells.
//!
//! Ordinary play is nowhere near this — the walks below are ~2000 plies of
//! deliberate travel in one direction — so nothing here is reachable by
//! accident. That is exactly why it needs a test that goes looking.

use hexo_engine::{Action, COORD_LIMIT, HexCoord, LEGAL_RADIUS, MoveError, Position, Search};

/// Cube distance from the coordinate domain's boundary that counts as "at the
/// face": within one legal radius, so a placement's disk crosses it.
const NEAR: i32 = COORD_LIMIT as i32 - LEGAL_RADIUS as i32;

/// The four directions that widen only one arena dimension, so the walk is
/// bounded by [`COORD_LIMIT`] rather than by the arena ceiling.
///
/// Between them they reach all six faces: `(1, 0)` drives `q` to `+lim` and `s`
/// to `-lim`, `(-1, 0)` the reverse, `(0, 1)` drives `r` to `+lim` and `s` to
/// `-lim`, and `(0, -1)` the reverse.
const AXIS_DIRECTIONS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

/// The two directions that widen both dimensions at once.
///
/// These never reach a face: a diagonal walk squares the padded bounding box,
/// so `MAX_GRID_CELLS` refuses first, at roughly `|q| = 1984`.
const DIAGONAL_DIRECTIONS: [(i32, i32); 2] = [(1, -1), (-1, 1)];

/// Why a walk stopped.
#[derive(Debug, PartialEq, Eq)]
enum Stop {
    /// The next step would have left the coordinate domain. The walk reached a
    /// face.
    DomainEdge,
    /// The engine refused the next step.
    Refused(MoveError),
}

/// Walk from the origin along `dir` in [`LEGAL_RADIUS`] steps for as far as the
/// engine allows.
fn walk(dir: (i32, i32)) -> (Position, Stop) {
    let step = LEGAL_RADIUS as i32;
    let mut pos = Position::new();
    pos.advance(Action::new(HexCoord::ORIGIN)).expect("opening");
    let mut k = 1;
    loop {
        let c = HexCoord::new((k * step * dir.0) as i16, (k * step * dir.1) as i16);
        if !c.is_valid() {
            return (pos, Stop::DomainEdge);
        }
        if let Err(e) = pos.advance(Action::new(c)) {
            return (pos, Stop::Refused(e));
        }
        k += 1;
    }
}

/// Largest absolute cube coordinate over the stones.
fn reach(pos: &Position) -> i32 {
    pos.stones()
        .map(|(c, _)| {
            let (q, r) = (i32::from(c.q), i32::from(c.r));
            q.abs().max(r.abs()).max((-q - r).abs())
        })
        .max()
        .expect("the walk placed stones")
}

/// Whether `c` sits within one legal radius of the domain boundary.
fn at_the_face(c: HexCoord) -> bool {
    let (q, r) = (i32::from(c.q), i32::from(c.r));
    q.abs() > NEAR || r.abs() > NEAR || (-q - r).abs() > NEAR
}

/// Every way of naming a legal placement must name the same set.
///
/// The full enumeration is scanned for validity and legality, which are `O(1)`
/// each. `legal_rank` and `nth_legal` are `O(arena words)`, which at these arena
/// sizes is ~32k words per call, so they are checked over a bounded sample
/// weighted to the boundary — the only place they can differ.
fn assert_the_legal_set_is_self_consistent(pos: &Position, label: &str) {
    let listed: Vec<Action> = pos.legal_actions().collect();

    assert_eq!(
        listed.len(),
        pos.legal_count(),
        "{label}: legal_count disagrees with the enumeration"
    );

    for a in &listed {
        let c = a.coord();
        assert!(
            c.is_valid(),
            "{label}: enumerated ({}, {}) is outside the coordinate domain (s = {})",
            c.q,
            c.r,
            -i32::from(c.q) - i32::from(c.r)
        );
        assert!(
            pos.is_legal(*a),
            "{label}: enumerated ({}, {}) fails is_legal",
            c.q,
            c.r
        );
    }

    // Indices to check both directions of the ordering on: the two ends, and
    // every action at the face.
    let mut sample: Vec<usize> = (0..listed.len().min(32)).collect();
    sample.extend(listed.len().saturating_sub(32)..listed.len());
    sample.extend(
        listed
            .iter()
            .enumerate()
            .filter(|(_, a)| at_the_face(a.coord()))
            .map(|(i, _)| i)
            .take(256),
    );
    sample.sort_unstable();
    sample.dedup();

    for i in sample {
        let a = listed[i];
        assert_eq!(
            pos.legal_rank(a),
            Some(i),
            "{label}: legal_rank disagrees with the enumeration at index {i}"
        );
        assert_eq!(
            pos.nth_legal(i),
            Some(a),
            "{label}: nth_legal disagrees with the enumeration at index {i}"
        );
    }
    assert_eq!(
        pos.nth_legal(listed.len()),
        None,
        "{label}: nth_legal ran past the end"
    );
}

#[test]
fn the_four_axis_walks_reach_the_domain_boundary() {
    // Guards every test below: if a walk stopped early they would pass
    // vacuously.
    for dir in AXIS_DIRECTIONS {
        let (pos, stop) = walk(dir);
        assert_eq!(stop, Stop::DomainEdge, "direction {dir:?} stopped early");
        assert!(
            reach(&pos) > NEAR,
            "direction {dir:?}: reached only {}, short of the boundary",
            reach(&pos)
        );
    }
}

#[test]
fn enumeration_agrees_with_validation_at_every_face() {
    for dir in AXIS_DIRECTIONS {
        let (pos, _) = walk(dir);
        assert!(
            pos.legal_actions().any(|a| at_the_face(a.coord())),
            "direction {dir:?}: no legal action is at the face, so this proves nothing"
        );
        assert_the_legal_set_is_self_consistent(&pos, &format!("direction {dir:?}"));
    }
}

#[test]
fn no_enumerated_action_can_be_refused_for_naming_an_unaddressable_cell() {
    // The whole point of the fix: advancing an enumerated action may run out of
    // arena, but must never come back with `CoordOutOfBounds`.
    for dir in AXIS_DIRECTIONS {
        let (pos, _) = walk(dir);
        let near: Vec<Action> = pos
            .legal_actions()
            .filter(|a| at_the_face(a.coord()))
            .take(64)
            .collect();
        assert!(!near.is_empty(), "direction {dir:?}: nothing at the face");
        for a in near {
            let mut probe = pos.clone();
            if let Err(e) = probe.advance(a) {
                let c = a.coord();
                assert!(
                    matches!(e, MoveError::BoardExtentExceeded { .. }),
                    "direction {dir:?}: advancing enumerated ({}, {}) gave {e}",
                    c.q,
                    c.r
                );
            }
        }
    }
}

#[test]
fn a_position_on_a_face_still_audits() {
    for dir in AXIS_DIRECTIONS {
        let (pos, _) = walk(dir);
        pos.audit()
            .unwrap_or_else(|e| panic!("direction {dir:?}: audit failed: {e}"));
    }
}

#[test]
fn undo_restores_a_boundary_position_exactly() {
    // `place` skips out-of-domain disk cells and `unplace` must skip the same
    // ones. A mismatch would leave coverage or the frontier count wrong after an
    // undo taken at the boundary, which round-trip tests in the interior cannot
    // see.
    for dir in AXIS_DIRECTIONS {
        let (mut pos, _) = walk(dir);
        let before = pos.clone();
        let zobrist_before = pos.zobrist();
        let legal_before = pos.legal_count();

        let candidate = pos
            .legal_actions()
            .find(|a| at_the_face(a.coord()))
            .expect("a legal move at the face");
        {
            let mut search = Search::new(&mut pos);
            search.apply(candidate).expect("legal at the boundary");
            search.position().audit().expect("audit after apply");
            assert_eq!(search.undo(), Some(candidate));
        }
        pos.audit().expect("audit after undo");
        assert_eq!(pos, before, "direction {dir:?}: undo did not restore");
        assert_eq!(pos.zobrist(), zobrist_before);
        assert_eq!(pos.legal_count(), legal_before);
        assert_eq!(pos.history(), before.history());
    }
}

#[test]
fn a_diagonal_walk_is_stopped_by_the_arena_ceiling_not_the_domain() {
    // A diagonal widens both arena dimensions, so the padded bounding box grows
    // as an area and `MAX_GRID_CELLS` refuses long before `COORD_LIMIT` would.
    // The refusal must be a clean representation limit, not a rule violation,
    // and the position must survive it intact.
    for dir in DIAGONAL_DIRECTIONS {
        let (pos, stop) = walk(dir);
        match stop {
            Stop::Refused(e @ MoveError::BoardExtentExceeded { .. }) => {
                assert!(!e.is_rule_violation(), "{dir:?}: {e} is not a rule");
            }
            other => panic!("direction {dir:?}: expected the arena ceiling, got {other:?}"),
        }
        assert!(
            reach(&pos) < NEAR,
            "direction {dir:?}: reached {}, so the domain was the binding limit",
            reach(&pos)
        );
        pos.audit()
            .unwrap_or_else(|e| panic!("direction {dir:?}: audit failed: {e}"));
        assert_the_legal_set_is_self_consistent(&pos, &format!("diagonal {dir:?}"));
    }
}
