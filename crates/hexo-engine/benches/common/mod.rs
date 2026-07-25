//! Deterministic fixtures for the engine benchmarks.
//!
//! A bench is a separate target from an integration test, so this cannot `use`
//! the oracles in `tests/common`. It deliberately uses the *same* splitmix64
//! with the same constants and the same uniform-over-the-legal-set driver, so a
//! fixture named by ply here is the position the test corpus builds at that
//! ply. Nothing here re-implements a rule: every placement goes through
//! `Position::advance`.
//!
//! Uniform random play is the fixture policy because it is what the audit's
//! out-of-tree snapshot used, so the numbers are comparable. It is also the
//! adversarial case for everything measured here: a uniform choice over the
//! frontier spreads outward, which makes the arena as wide and as sparse as
//! honest play ever gets.

use hexo_engine::{Action, HexCoord, LEGAL_RADIUS, Position, Search, hex_distance};

/// Game stages the suite reports at: the opened board, early game, middle game,
/// and a length past where an ordinary game ends.
pub const PLIES: [usize; 4] = [1, 32, 96, 256];

/// The one seed every fixture is built from.
///
/// Fixtures nest: the move list for a longer game has the shorter game's move
/// list as its prefix, because the driver is a pure function of the seed and
/// the position.
pub const SEED: u64 = 0x1234_5678_9abc_def0;

/// splitmix64. Deterministic, seedable, and dependency-free — the generator
/// `tests/common` uses, with the same constants.
pub struct Rng(u64);

impl Rng {
    /// Seed the generator.
    pub const fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Next 64 bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform-ish value in `0..n`. `n` must be non-zero.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// The move list of a `plies`-ply game of uniformly random legal placements.
///
/// # Panics
/// If the game ends before `plies` — a fixture that quietly returned a shorter
/// game would silently change what every benchmark measures.
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

/// The legal placements nearest to and furthest from the centroid of the
/// stones, as `(interior, edge)`.
///
/// The two are the extremes the audit claims differ. The interior cell sits
/// among stones, so most of its radius-8 disk is already covered and the
/// placement flips few frontier bits. The edge cell sits on the outer rim, so
/// most of its disk is at coverage zero, every one of those cells becomes a new
/// frontier bit, and the write reaches into the arena's padding rows.
///
/// Ties go to the first placement in canonical order.
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

/// The same position after a search excursion that grew the arena and never
/// gave it back: `steps` placements walking [`LEGAL_RADIUS`] at a time along
/// `+q`, then unwound.
///
/// `Search::undo` restores every observable field and deliberately keeps the
/// allocation, so the result is `PartialEq` to the input and scans a much
/// larger arena. This is the worker-inflation case in the audit, and it is what
/// separates "enumeration is O(legal count)" from "enumeration is O(arena
/// words)".
///
/// # Panics
/// If the excursion did not round-trip.
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
            // Exactly LEGAL_RADIUS from the previous stone, so every step is
            // legal, lands on empty space, and cannot complete a six-run.
            c = HexCoord::new(c.q + LEGAL_RADIUS as i16, c.r);
            search
                .apply(Action::new(c))
                .expect("a legal excursion step");
        }
    } // Drop unwinds to the floor.
    assert_eq!(
        &out, pos,
        "the excursion changed the position, not just its arena"
    );
    out
}
