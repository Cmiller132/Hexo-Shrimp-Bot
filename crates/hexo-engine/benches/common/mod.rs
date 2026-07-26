//! Deterministic fixtures for the engine benchmarks.

use hexo_engine::{Action, HexCoord, LEGAL_RADIUS, Position, Search, hex_distance};

/// Game stages the suite reports at: the opened board, early game, middle game,
/// and a length past where an ordinary game ends.
pub const PLIES: [usize; 4] = [1, 32, 96, 256];

/// The one seed every fixture is built from.
pub const SEED: u64 = 0x1234_5678_9abc_def0;

#[path = "../../testkit/rng.rs"]
mod rng;

pub use rng::Rng;

/// The move list of a `plies`-ply game of uniformly random legal placements.
pub fn game(plies: usize) -> Vec<Action> {
    let mut rng = Rng::new(SEED);
    let mut pos = Position::new();
    let mut moves = Vec::with_capacity(plies);
    for ply in 0..plies {
        let n = pos.legal_count();
        assert!(n > 0, "fixture game ended at ply {ply}, short of {plies}");
        let action = pos
            .nth_legal(rng.below(n))
            .expect("index below legal_count");
        pos.advance(action).expect("a placement from the legal set");
        moves.push(action);
    }
    assert!(
        !pos.is_terminal(),
        "fixture game is terminal at ply {plies}"
    );
    moves
}

/// The position after [`game`]'s first `plies` placements.
pub fn position_at(plies: usize) -> Position {
    Position::replay(&game(plies)).expect("a replayed fixture game")
}

/// Centroid of the stones, rounded toward the origin.
fn centroid(pos: &Position) -> HexCoord {
    let (mut q, mut r, mut n) = (0i64, 0i64, 0i64);
    for (c, _) in pos.stones() {
        q += i64::from(c.q);
        r += i64::from(c.r);
        n += 1;
    }
    assert!(n > 0, "an empty position has no centroid");
    HexCoord::new((q / n) as i16, (r / n) as i16)
}

/// The legal placements nearest to and furthest from the centroid of the stones, as
/// `(interior, edge)`.
pub fn interior_and_edge(pos: &Position) -> (Action, Action) {
    let mid = centroid(pos);
    let mut interior: Option<(u32, Action)> = None;
    let mut edge: Option<(u32, Action)> = None;
    for action in pos.legal_actions() {
        let d = hex_distance(mid, action.coord());
        if interior.is_none_or(|(best, _)| d < best) {
            interior = Some((d, action));
        }
        if edge.is_none_or(|(best, _)| d > best) {
            edge = Some((d, action));
        }
    }
    let (_, interior) = interior.expect("a non-terminal position has legal placements");
    let (_, edge) = edge.expect("a non-terminal position has legal placements");
    (interior, edge)
}

/// The same position after a search excursion that grew the arena and never gave it
/// back: `steps` placements walking [`LEGAL_RADIUS`] at a time along `+q`, then
/// unwound.
pub fn inflated(pos: &Position, steps: usize) -> Position {
    let mut out = pos.clone();
    {
        let mut search = Search::new(&mut out);
        let mut c = search
            .position()
            .stones()
            .map(|(c, _)| c)
            .max_by_key(|c| c.q)
            .expect("a position with at least one stone");
        for _ in 0..steps {
            c = HexCoord::new(c.q + LEGAL_RADIUS as i16, c.r);
            search
                .apply(Action::new(c))
                .expect("a legal excursion step");
        }
    }
    assert_eq!(
        &out, pos,
        "the excursion changed the position, not just its arena"
    );
    out
}
