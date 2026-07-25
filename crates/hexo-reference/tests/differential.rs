//! Differential test: the rewrite against the previous implementation.
//!
//! `hexo-engine` is a ground-up rewrite of the rules engine that
//! `hexo-reference` froze a copy of. The rules are supposed to be identical;
//! only the architecture changed. Every other check in this workspace compares
//! the engine against oracles written by the same people who wrote the engine.
//! This one compares it against an independent implementation, which is the
//! only check here that a *shared misreading of the rules* could not survive.
//!
//! # What is compared, on every ply of every game
//!
//! - the legal move set, as a set of coordinates;
//! - terminal status, and the winner when terminal;
//! - whose turn it is, and the turn phase (including `SecondStone`'s payload);
//! - the stone count and the per-player stone counts;
//! - the 18 six-cell windows through the placed cell — their identity, both
//!   players' ownership masks, and the six-in-a-row verdict for each;
//! - which windows the placement completed, as a set;
//! - the legality *predicate* on cells the driver probes, which is a different
//!   question from the enumerated set.
//!
//! # If this fails
//!
//! **Report the divergence. Do not edit either engine to make it pass**, and in
//! particular do not assume the rewrite is the correct one — it is the one that
//! changed, but the old engine has bugs of its own. `crates/hexo-reference` is
//! a frozen photograph and is never the file that gets fixed.
//!
//! # What is deliberately *not* compared
//!
//! - **Enumeration order.** The two engines are free to yield legal moves in
//!   different orders; the contract is the set. The order is observed and
//!   reported (see `ORDER` in the output) rather than asserted.
//! - **Error variants.** The rewrite has `MoveError::CoordOutOfBounds` and
//!   `TooFarFromStones`; the old engine has one `IllegalPlacement` for both.
//!   Accept-versus-reject is compared; the reason is not.
//! - **Extreme coordinates.** The rewrite has a `COORD_LIMIT` coordinate domain
//!   and a `MAX_GRID_CELLS` arena ceiling that the old engine, being sparse and
//!   hash-backed, does not. Those are representation limits, not rules, so the
//!   games generated here stay central and of ordinary length and never
//!   approach either bound.
//!
//! # Scale
//!
//! Defaults keep a debug `cargo test --workspace` inside its budget. Every
//! placement in a debug build also runs the rewrite's full Tier-C assertion set,
//! so a ply here is expensive. Raise the counts for a nightly or release run:
//!
//! ```text
//! HEXO_DIFF_GAMES=20000 HEXO_DIFF_UNIFORM=200 HEXO_DIFF_PLIES=512 \
//!     cargo test --release -p hexo-reference --test differential
//! ```

use hexo_engine as rewrite;
use hexo_reference as legacy;

// ---------------------------------------------------------------------------
// Knobs
// ---------------------------------------------------------------------------

/// Line-building games played when `HEXO_DIFF_GAMES` is unset.
///
/// Sized so this file adds roughly four seconds to a debug
/// `cargo test --workspace`. The heavy sweep is opt-in through the environment.
const DEFAULT_GAMES: usize = 600;

/// Uniform-play games played when `HEXO_DIFF_UNIFORM` is unset.
const DEFAULT_UNIFORM_GAMES: usize = 20;

/// Ply bound for a uniform game when `HEXO_DIFF_PLIES` is unset.
///
/// A test artefact, not a rule. Kept small enough that the frontier — which
/// grows without bound under uniform play — stays a few thousand cells, and far
/// enough from the rewrite's representation limits that they never enter the
/// picture.
const DEFAULT_UNIFORM_PLIES: usize = 128;

/// Ply bound for a line-building game. They terminate in tens of plies.
const LINE_PLY_BOUND: usize = 400;

fn env_count(key: &str, fallback: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(fallback)
}

// ---------------------------------------------------------------------------
// Neutral vocabulary
// ---------------------------------------------------------------------------
//
// Neither engine's types are used as the comparison currency: each side is
// projected onto plain tuples first, so a comparison can never be satisfied by
// one engine's `PartialEq` doing something the other's does not.

/// A cell, as plain axial numbers.
type Cell = (i16, i16);

/// A player, as an index. `0` is the player who places the opening stone in
/// both engines, which is what fixes this mapping.
type Side = u8;

/// A window, as its first cell and its axis index.
type Win = (Cell, u8);

fn rw_cell(c: rewrite::HexCoord) -> Cell {
    (c.q, c.r)
}

fn lg_cell(c: legacy::HexCoord) -> Cell {
    (c.q, c.r)
}

fn rw_coord(c: Cell) -> rewrite::HexCoord {
    rewrite::HexCoord::new(c.0, c.1)
}

fn lg_coord(c: Cell) -> legacy::HexCoord {
    legacy::HexCoord::new(c.0, c.1)
}

fn rw_side(p: rewrite::Player) -> Side {
    p.index() as Side
}

fn lg_side(p: legacy::Player) -> Side {
    p.index() as Side
}

/// The two players of each engine, index-aligned.
const RW_SIDES: [rewrite::Player; 2] = [rewrite::Player::P0, rewrite::Player::P1];
const LG_SIDES: [legacy::Player; 2] = [legacy::Player::Player0, legacy::Player::Player1];

/// Turn phase, projected out of two unrelated enums.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Opening,
    First,
    /// Carries the first stone of the turn, which both engines track.
    Second(Cell),
}

fn rw_phase(p: rewrite::TurnPhase) -> Phase {
    match p {
        rewrite::TurnPhase::Opening => Phase::Opening,
        rewrite::TurnPhase::FirstStone => Phase::First,
        rewrite::TurnPhase::SecondStone { first } => Phase::Second(rw_cell(first)),
    }
}

fn lg_phase(p: legacy::TurnPhase) -> Phase {
    match p {
        legacy::TurnPhase::Opening => Phase::Opening,
        legacy::TurnPhase::FirstStone => Phase::First,
        legacy::TurnPhase::SecondStone { first } => Phase::Second(lg_cell(first)),
    }
}

fn rw_axis(a: rewrite::Axis) -> u8 {
    a.index() as u8
}

fn lg_axis(a: legacy::Axis) -> u8 {
    a.index()
}

/// Axis step vectors, index-aligned with `rw_axis`/`lg_axis`.
const AXIS_VECTORS: [Cell; 3] = [(1, 0), (0, 1), (1, -1)];

/// Displacements used to probe the legality predicate around a legal cell.
///
/// A mix of on-board, just-inside-the-radius and just-outside-the-radius
/// offsets — `(5, 4)` and `(-3, -6)` are nine hex steps away along the derived
/// `s` axis, which is the case a `q`/`r`-only distance check gets wrong. All
/// stay far inside both engines' coordinate range.
const PROBE_OFFSETS: [Cell; 6] = [(0, 0), (9, 0), (-9, 0), (0, 9), (5, 4), (-3, -6)];

// ---------------------------------------------------------------------------
// Deterministic PRNG (splitmix64, no dependencies)
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    const fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

// ---------------------------------------------------------------------------
// The comparison
// ---------------------------------------------------------------------------

/// Reusable scratch space, so a per-ply comparison allocates nothing.
#[derive(Default)]
struct Scratch {
    rw_legal: Vec<Cell>,
    lg_legal: Vec<Cell>,
    lg_raw: Vec<legacy::HexCoord>,
    lhs: Vec<Win>,
    rhs: Vec<Win>,
}

/// Running totals, reported at the end of a run.
#[derive(Default)]
struct Tally {
    games: usize,
    plies: usize,
    terminated: usize,
    wins: [usize; 2],
    /// Plies where the two engines listed the legal moves in the same order.
    same_order: usize,
    /// Plies where the legal sets matched but the orders did not.
    reordered: usize,
    /// Placements that completed more than one six-window.
    multi_window_wins: usize,
    /// Legality-predicate probes compared.
    predicate_probes: usize,
}

/// Every read the two engines share, compared at one ply.
fn compare_read_surface(rw: &rewrite::Position, lg: &legacy::HexoState, at: &str) {
    // Terminal status and the winner.
    assert_eq!(
        rw.is_terminal(),
        lg.is_terminal(),
        "{at}: terminal status disagrees"
    );
    assert_eq!(
        rw.outcome().map(|o| rw_side(o.winner)),
        lg.terminal().map(|o| lg_side(o.winner)),
        "{at}: winner disagrees"
    );
    if let Some(o) = lg.terminal() {
        // The old outcome also records how many stones were on the board when
        // the game ended; the rewrite reports that through `stone_count`.
        assert_eq!(
            o.placements,
            rw.stone_count(),
            "{at}: stones-at-termination disagrees"
        );
    }

    // Whose turn, and where inside it.
    assert_eq!(
        rw_side(rw.current_player()),
        lg_side(lg.current_player()),
        "{at}: mover disagrees"
    );
    assert_eq!(
        rw_phase(rw.phase()),
        lg_phase(lg.phase()),
        "{at}: turn phase disagrees"
    );

    // Stone counts, total and per player.
    assert_eq!(
        rw.stone_count(),
        lg.placements_made(),
        "{at}: stone count disagrees"
    );
    let board = lg.board();
    let mut lg_by_side = [0u32; 2];
    for c in board.occupied_cells() {
        let owner = board.get(*c).expect("an occupied cell without an owner");
        lg_by_side[lg_side(owner) as usize] += 1;
    }
    for (i, &rw_p) in RW_SIDES.iter().enumerate() {
        assert_eq!(
            rw.stone_count_for(rw_p),
            lg_by_side[i],
            "{at}: stone count for side {i} disagrees"
        );
    }

    assert_eq!(
        rw.legal_count(),
        lg.legal_move_count(),
        "{at}: legal move count disagrees"
    );
}

/// Compare the legal move **sets**. Returns whether the two enumerations also
/// happened to be in the same order, which is reported but never required.
///
/// Leaves the agreed set in `scratch.rw_legal`.
fn compare_legal_sets(
    rw: &rewrite::Position,
    lg: &legacy::HexoState,
    scratch: &mut Scratch,
    at: &str,
) -> bool {
    scratch.rw_legal.clear();
    scratch
        .rw_legal
        .extend(rw.legal_actions().map(|a| rw_cell(a.coord())));
    lg.write_legal_moves(&mut scratch.lg_raw);
    scratch.lg_legal.clear();
    scratch
        .lg_legal
        .extend(scratch.lg_raw.iter().copied().map(lg_cell));

    assert_eq!(
        scratch.rw_legal.len(),
        rw.legal_count(),
        "{at}: the rewrite's iterator and legal_count disagree"
    );
    assert_eq!(
        scratch.lg_legal.len(),
        lg.legal_move_count(),
        "{at}: the old engine's iterator and legal_move_count disagree"
    );

    if scratch.rw_legal == scratch.lg_legal {
        return true;
    }
    // Same set, different order is legitimate. Sorting decides which it is.
    scratch.rw_legal.sort_unstable();
    scratch.lg_legal.sort_unstable();
    assert_eq!(
        scratch.rw_legal, scratch.lg_legal,
        "{at}: LEGAL SET MISMATCH (not merely an ordering difference)"
    );
    false
}

/// The 18 windows through the placed cell: identity, both masks, and the
/// six-in-a-row verdict for each.
fn compare_windows(
    rw: &rewrite::Position,
    lg: &legacy::HexoState,
    placed: Cell,
    update: &legacy::WindowUpdate,
    at: &str,
) {
    let through = rw.windows_through(rw_coord(placed));
    let changed = update.changed.as_slice();
    assert_eq!(
        through.len(),
        legacy::WINDOWS_PER_PLACEMENT,
        "{at}: the rewrite reported a different number of windows through a placement"
    );
    assert_eq!(
        changed.len(),
        through.len(),
        "{at}: the old engine touched a different number of windows"
    );

    let store = lg.board().windows();
    for (i, wr) in through.iter().enumerate() {
        let key = changed[i];
        assert_eq!(
            (rw_cell(wr.window.start), rw_axis(wr.window.axis)),
            (lg_cell(key.start), lg_axis(key.axis)),
            "{at}: window slot {i} names a different window"
        );
        let entry = store
            .entry(key)
            .expect("a window a placement touched must be in the store");
        for (s, (&rw_p, &lg_p)) in RW_SIDES.iter().zip(LG_SIDES.iter()).enumerate() {
            assert_eq!(
                wr.mask.mask(rw_p),
                entry.mask(lg_p),
                "{at}: window slot {i} ownership mask for side {s} disagrees"
            );
            assert_eq!(
                wr.mask.is_full_for(rw_p),
                entry.is_win_for(lg_p),
                "{at}: window slot {i} six-in-a-row verdict for side {s} disagrees"
            );
        }
    }
}

/// Which windows the placement completed, compared as a set.
fn compare_winning_windows(
    applied: &rewrite::Applied,
    update: &legacy::WindowUpdate,
    scratch: &mut Scratch,
    at: &str,
) {
    scratch.lhs.clear();
    scratch.lhs.extend(
        applied
            .winning_windows()
            .map(|w| (rw_cell(w.start), rw_axis(w.axis))),
    );
    scratch.rhs.clear();
    scratch.rhs.extend(
        update
            .winning_windows
            .as_slice()
            .iter()
            .map(|k| (lg_cell(k.start), lg_axis(k.axis))),
    );
    scratch.lhs.sort_unstable();
    scratch.rhs.sort_unstable();
    assert_eq!(
        scratch.lhs, scratch.rhs,
        "{at}: the set of completed windows disagrees"
    );
}

/// Every stone and its owner, compared as a set.
fn compare_occupancy(rw: &rewrite::Position, lg: &legacy::HexoState, at: &str) {
    let mut lhs: Vec<(Cell, Side)> = rw.stones().map(|(c, p)| (rw_cell(c), rw_side(p))).collect();
    let board = lg.board();
    let mut rhs: Vec<(Cell, Side)> = board
        .occupied_cells()
        .iter()
        .map(|c| {
            (
                lg_cell(*c),
                lg_side(board.get(*c).expect("an occupied cell without an owner")),
            )
        })
        .collect();
    lhs.sort_unstable();
    rhs.sort_unstable();
    assert_eq!(lhs, rhs, "{at}: board occupancy disagrees");
}

/// Ask both engines whether `cell` may be played right now.
fn compare_legality(rw: &rewrite::Position, lg: &legacy::HexoState, cell: Cell, at: &str) -> bool {
    let by_rewrite = rw.is_legal(rewrite::Action::new(rw_coord(cell)));
    let by_legacy = legacy::is_legal_placement(lg, lg_coord(cell)).is_ok();
    assert_eq!(
        by_rewrite, by_legacy,
        "{at}: legality of {cell:?} disagrees (rewrite says {by_rewrite})"
    );
    by_rewrite
}

/// Apply one placement to both engines and compare everything the placement
/// produced.
fn step(
    rw: &mut rewrite::Position,
    lg: &mut legacy::HexoState,
    cell: Cell,
    scratch: &mut Scratch,
    tally: &mut Tally,
    at: &str,
) {
    let action = rewrite::Action::new(rw_coord(cell));
    let applied = rw
        .advance(action)
        .unwrap_or_else(|e| panic!("{at}: the rewrite refused {cell:?}: {e}"));
    let result = legacy::apply_placement(
        lg,
        legacy::Placement {
            coord: lg_coord(cell),
        },
    )
    .unwrap_or_else(|e| panic!("{at}: the old engine refused {cell:?}: {e}"));

    // The record encoding of the move.
    assert_eq!(
        action.id().0,
        legacy::pack_coord(lg_coord(cell)),
        "{at}: the action-ID encodings disagree"
    );

    assert_eq!(
        rw_cell(applied.action.coord()),
        lg_cell(result.placed),
        "{at}: the placed cell disagrees"
    );
    assert_eq!(
        rw_side(applied.mover),
        lg_side(result.player),
        "{at}: the mover disagrees"
    );
    assert_eq!(
        rw_phase(applied.phase_before),
        lg_phase(result.phase_before),
        "{at}: phase_before disagrees"
    );
    assert_eq!(
        rw_phase(applied.phase_after),
        lg_phase(result.phase_after),
        "{at}: phase_after disagrees"
    );
    assert_eq!(
        applied.outcome.is_some(),
        result.outcome.is_some(),
        "{at}: the two engines disagree on whether this placement ended the game"
    );
    assert_eq!(
        applied.outcome.is_some(),
        result.window_update.has_win(),
        "{at}: the old engine's win flag disagrees with its own outcome"
    );
    if let (Some(a), Some(b)) = (applied.outcome, result.outcome) {
        assert_eq!(rw_side(a.winner), lg_side(b.winner), "{at}: winner");
    }

    compare_windows(rw, lg, cell, &result.window_update, at);
    compare_winning_windows(&applied, &result.window_update, scratch, at);

    if applied.winning.count() > 1 {
        tally.multi_window_wins += 1;
    }
}

// ---------------------------------------------------------------------------
// Drivers
// ---------------------------------------------------------------------------

/// Per-player line state for the biased driver, mirroring `hexo-engine`'s
/// smoke test: uniformly random play essentially never makes six in a row, so
/// without a bias the terminal and win-detection paths go untested.
#[derive(Clone, Copy)]
struct Line {
    anchor: Cell,
    axis: usize,
    started: bool,
}

/// The nearest cell extending this player's line that **both** engines call
/// legal. The agreement is asserted on every probe, which is how the legality
/// predicate gets compared as well as the enumerated set.
fn line_move(
    rw: &rewrite::Position,
    lg: &legacy::HexoState,
    line: &Line,
    tally: &mut Tally,
    at: &str,
) -> Option<Cell> {
    if !line.started {
        return None;
    }
    let v = AXIS_VECTORS[line.axis];
    for k in 1..=8i16 {
        for sign in [1i16, -1] {
            let cell = (
                line.anchor.0 + v.0 * k * sign,
                line.anchor.1 + v.1 * k * sign,
            );
            tally.predicate_probes += 1;
            if compare_legality(rw, lg, cell, at) {
                return Some(cell);
            }
        }
    }
    None
}

/// Play one game through both engines, comparing at every ply.
///
/// `noise` in `0..=255` is the chance of ignoring the line and playing
/// uniformly; `255` is pure uniform play.
fn play_one(
    seed: u64,
    noise: u8,
    ply_bound: usize,
    scratch: &mut Scratch,
    tally: &mut Tally,
) -> Option<Side> {
    let mut rng = Rng::new(seed);
    let mut rw = rewrite::Position::new();
    let mut lg = legacy::HexoState::new();
    let mut lines = [
        Line {
            anchor: (0, 0),
            axis: (seed % 3) as usize,
            started: false,
        },
        Line {
            anchor: (0, 0),
            axis: ((seed / 3) % 3) as usize,
            started: false,
        },
    ];
    let mut winner = None;
    let mut ply = 0usize;

    while ply < ply_bound {
        let at = format!("seed {seed} ply {ply}");
        compare_read_surface(&rw, &lg, &at);
        if compare_legal_sets(&rw, &lg, scratch, &at) {
            tally.same_order += 1;
        } else {
            tally.reordered += 1;
        }

        if rw.is_terminal() {
            break;
        }
        let n = scratch.rw_legal.len();
        assert!(n > 0, "{at}: a non-terminal position with no legal moves");

        // Probe the legality predicate on cells that were not drawn from the
        // enumerated set. Enumeration agreement would not catch a predicate
        // that accepts a cell it never lists, and the line driver — the other
        // source of probes — never runs in the uniform games, which are the
        // ones with a wide arena and a frontier in the thousands.
        if ply.is_multiple_of(16) {
            let base = scratch.rw_legal[rng.below(n)];
            for d in PROBE_OFFSETS {
                tally.predicate_probes += 1;
                compare_legality(&rw, &lg, (base.0 + d.0, base.1 + d.1), &at);
            }
        }

        let mover = rw_side(rw.current_player()) as usize;
        let uniform = (rng.next_u64() & 0xFF) as u8 <= noise;
        let cell = if uniform {
            scratch.rw_legal[rng.below(n)]
        } else {
            line_move(&rw, &lg, &lines[mover], tally, &at)
                .unwrap_or_else(|| scratch.rw_legal[rng.below(n)])
        };

        step(&mut rw, &mut lg, cell, scratch, tally, &at);
        ply += 1;
        tally.plies += 1;

        let line = &mut lines[mover];
        if !line.started {
            line.anchor = cell;
            line.started = true;
        }

        if let Some(o) = rw.outcome() {
            winner = Some(rw_side(o.winner));
        }
    }

    // One last look at the final position, plus the whole board.
    let at = format!("seed {seed} final");
    compare_read_surface(&rw, &lg, &at);
    compare_legal_sets(&rw, &lg, scratch, &at);
    compare_occupancy(&rw, &lg, &at);

    tally.games += 1;
    if let Some(w) = winner {
        tally.terminated += 1;
        tally.wins[w as usize] += 1;
    }
    winner
}

fn report(label: &str, t: &Tally) {
    eprintln!(
        "diff[{label}]: {} games, {} plies, {} terminated (side0 {}, side1 {}), \
         {} multi-window wins, {} legality probes; \
         ORDER: {} plies identical, {} plies same-set-different-order",
        t.games,
        t.plies,
        t.terminated,
        t.wins[0],
        t.wins[1],
        t.multi_window_wins,
        t.predicate_probes,
        t.same_order,
        t.reordered,
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Line-building games, which actually reach a win, driven through both engines.
#[test]
fn line_building_games_agree_ply_by_ply() {
    let games = env_count("HEXO_DIFF_GAMES", DEFAULT_GAMES);
    let mut scratch = Scratch::default();
    let mut tally = Tally::default();

    for seed in 0..games as u64 {
        // Sweep the noise so the sample spans clean line races and messy
        // interrupted ones.
        let noise = ((seed % 16) * 14) as u8;
        play_one(seed, noise, LINE_PLY_BOUND, &mut scratch, &mut tally);
    }

    report("line", &tally);
    assert!(
        tally.terminated * 2 > tally.games,
        "the biased driver should terminate most games; got {}/{}",
        tally.terminated,
        tally.games
    );
    assert!(
        tally.wins[0] > 0 && tally.wins[1] > 0,
        "both sides should win sometimes, or the terminal path is half-covered"
    );
}

/// Uniform games, which almost never terminate but spread the board wide and
/// grow the legal set into the thousands.
#[test]
fn uniform_games_agree_ply_by_ply() {
    let games = env_count("HEXO_DIFF_UNIFORM", DEFAULT_UNIFORM_GAMES);
    let plies = env_count("HEXO_DIFF_PLIES", DEFAULT_UNIFORM_PLIES);
    let mut scratch = Scratch::default();
    let mut tally = Tally::default();

    for seed in 0..games as u64 {
        play_one(seed ^ 0x5eed_1234, u8::MAX, plies, &mut scratch, &mut tally);
    }

    report("uniform", &tally);
    assert!(tally.plies > 0);
}

/// The engines must reject the same placements, not merely accept the same ones.
///
/// Enumeration agreement alone would not catch a predicate that accepts a cell
/// it never lists.
#[test]
fn both_engines_reject_the_same_illegal_placements() {
    let mut scratch = Scratch::default();
    let mut tally = Tally::default();
    let mut rw = rewrite::Position::new();
    let mut lg = legacy::HexoState::new();

    // Ply 0: only the origin is playable.
    for probe in [(0, 0), (1, 0), (-1, 0), (0, 1), (5, -5), (100, 100)] {
        compare_legality(&rw, &lg, probe, "opening");
    }
    step(
        &mut rw,
        &mut lg,
        (0, 0),
        &mut scratch,
        &mut tally,
        "opening",
    );

    // Occupied cells, cells beyond the radius, and the exact radius boundary.
    for probe in [
        (0, 0), // occupied
        (8, 0), // exactly at the radius
        (9, 0), // one past it
        (0, 9),
        (-9, 0),
        (5, 4), // hex distance 9 along the s axis
        (300, -150),
        (-4000, 2000),
    ] {
        compare_legality(&rw, &lg, probe, "after opening");
    }

    // Enter SecondStone and confirm both refuse to reuse the first stone.
    step(
        &mut rw,
        &mut lg,
        (1, 0),
        &mut scratch,
        &mut tally,
        "first stone",
    );
    assert_eq!(rw_phase(rw.phase()), Phase::Second((1, 0)));
    for probe in [(1, 0), (0, 0), (2, 0), (40, 0)] {
        compare_legality(&rw, &lg, probe, "second stone");
    }

    // Terminal positions refuse everything. Build a six-in-a-row for side 0.
    let mut rw = rewrite::Position::new();
    let mut lg = legacy::HexoState::new();
    for (i, cell) in SIX_IN_A_ROW.iter().enumerate() {
        step(
            &mut rw,
            &mut lg,
            *cell,
            &mut scratch,
            &mut tally,
            &format!("win line ply {i}"),
        );
    }
    compare_read_surface(&rw, &lg, "terminal");
    assert!(rw.is_terminal(), "the fixture should have ended the game");
    for probe in [(0, 0), (6, 0), (-1, 0), (0, 3)] {
        compare_legality(&rw, &lg, probe, "terminal");
    }
    compare_legal_sets(&rw, &lg, &mut scratch, "terminal");
}

/// A hand-built game that side 0 wins with six in a row on the `q` axis.
///
/// Ply pattern `P0; P1 P1; P0 P0; ...`, so side 0 owns plies 0, 3, 4, 7, 8,
/// 11, 12 and side 1 owns 1, 2, 5, 6, 9, 10. Side 1's stones sit on a different
/// `r` row, in range of the origin but making no line of their own.
const SIX_IN_A_ROW: [Cell; 12] = [
    (0, 0), // 0  side 0
    (0, 3), // 1  side 1
    (1, 3), // 2  side 1
    (1, 0), // 3  side 0
    (2, 0), // 4  side 0
    (0, 5), // 5  side 1
    (1, 5), // 6  side 1
    (3, 0), // 7  side 0
    (4, 0), // 8  side 0
    (0, 7), // 9  side 1
    (1, 7), // 10 side 1
    (5, 0), // 11 side 0 — completes (0,0)..(5,0)
];

/// The same shape, but with the run's last gap filled last, so the winning
/// placement completes **two** six-windows at once.
///
/// Random play reaches this rarely; the multi-window case is where an engine
/// that assumes "exactly one winning window" breaks, so it gets a fixture.
const SEVEN_IN_A_ROW: [Cell; 13] = [
    (0, 0), // 0  side 0
    (0, 3), // 1  side 1
    (1, 3), // 2  side 1
    (1, 0), // 3  side 0
    (2, 0), // 4  side 0
    (0, 5), // 5  side 1
    (1, 5), // 6  side 1
    (3, 0), // 7  side 0
    (5, 0), // 8  side 0 — leaves the gap at (4, 0)
    (0, 7), // 9  side 1
    (1, 7), // 10 side 1
    (6, 0), // 11 side 0 — still no six, the gap is open
    (4, 0), // 12 side 0 — completes (0..5) and (1..6)
];

#[test]
fn hand_built_wins_agree_including_the_multi_window_case() {
    let mut scratch = Scratch::default();

    for (name, moves, expect_windows) in [
        ("six", &SIX_IN_A_ROW[..], 1usize),
        ("seven", &SEVEN_IN_A_ROW[..], 2usize),
    ] {
        let mut tally = Tally::default();
        let mut rw = rewrite::Position::new();
        let mut lg = legacy::HexoState::new();
        for (i, cell) in moves.iter().enumerate() {
            let at = format!("{name} ply {i}");
            compare_read_surface(&rw, &lg, &at);
            compare_legal_sets(&rw, &lg, &mut scratch, &at);
            assert!(
                !rw.is_terminal(),
                "{at}: the fixture ended earlier than intended"
            );
            step(&mut rw, &mut lg, *cell, &mut scratch, &mut tally, &at);
        }
        let at = format!("{name} final");
        compare_read_surface(&rw, &lg, &at);
        compare_occupancy(&rw, &lg, &at);
        assert!(rw.is_terminal(), "{at}: the fixture did not end the game");
        assert_eq!(
            rw.outcome().map(|o| rw_side(o.winner)),
            Some(0),
            "{at}: side 0 should have won"
        );
        // Both engines agreed on the completed-window set inside `step`; this
        // pins the fixture itself down so a later edit cannot quietly turn the
        // two-window case back into a one-window case.
        assert_eq!(
            tally.multi_window_wins,
            usize::from(expect_windows > 1),
            "{at}: expected {expect_windows} completed windows"
        );
    }
}

/// The record encoding did not change across the rewrite.
///
/// The old `pack_coord` is documented as the canonical action-ID encoding,
/// duplicated in Python and in the frontend and persisted in training shards
/// and game records. If the rewrite's `ActionId` had drifted from it, every
/// stored record would silently reinterpret.
#[test]
fn action_id_encoding_matches_the_old_packing() {
    for q in -40i16..=40 {
        for r in -40i16..=40 {
            let cell = (q, r);
            assert_eq!(
                rewrite::ActionId::from_coord(rw_coord(cell)).0,
                legacy::pack_coord(lg_coord(cell)),
                "action id for {cell:?}"
            );
        }
    }
    for &q in &[i16::MIN, -30000, -1, 0, 1, 30000, i16::MAX] {
        for &r in &[i16::MIN, -30000, -1, 0, 1, 30000, i16::MAX] {
            let cell = (q, r);
            assert_eq!(
                rewrite::ActionId::from_coord(rw_coord(cell)).0,
                legacy::pack_coord(lg_coord(cell)),
                "action id for {cell:?}"
            );
            assert_eq!(
                rw_cell(rewrite::ActionId(legacy::pack_coord(lg_coord(cell))).coord()),
                lg_cell(legacy::unpack_coord(legacy::pack_coord(lg_coord(cell)))),
                "action id round trip for {cell:?}"
            );
        }
    }
}

/// The constants both engines build their geometry from.
#[test]
fn shared_constants_agree() {
    assert_eq!(rewrite::LEGAL_RADIUS as i16, legacy::LEGAL_RADIUS);
    assert_eq!(rewrite::WINDOW_LEN as i16, legacy::WINDOW_LEN);
    assert_eq!(
        rewrite::WINDOWS_PER_PLACEMENT,
        legacy::WINDOWS_PER_PLACEMENT
    );
    assert_eq!(rewrite::Axis::ALL.len(), legacy::Axis::ALL.len());
    for (a, b) in rewrite::Axis::ALL.iter().zip(legacy::Axis::ALL.iter()) {
        assert_eq!(rw_axis(*a), lg_axis(*b), "axis order");
        assert_eq!(
            rw_cell(a.vector()),
            lg_cell(b.vector()),
            "axis {a:?} step vector"
        );
        assert_eq!(AXIS_VECTORS[rw_axis(*a) as usize], rw_cell(a.vector()));
    }
    // Hex distance is the function the radius rule is stated in.
    for q in -12i16..=12 {
        for r in -12i16..=12 {
            assert_eq!(
                rewrite::hex_distance(rewrite::HexCoord::ORIGIN, rw_coord((q, r))) as i16,
                legacy::hex_distance(legacy::HexCoord::ZERO, lg_coord((q, r))),
                "hex_distance to ({q}, {r})"
            );
        }
    }
}
