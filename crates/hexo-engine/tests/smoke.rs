//! Random-playout smoke test.

mod common;

use common::{Rng, check_all_oracles, nth_legal, winners_oracle};
use hexo_engine::{Action, Axis, HexCoord, Player, Position};

/// Test-local ply bound.
const PLY_BOUND: usize = 512;

/// Line-building games played when `HEXO_SMOKE_GAMES` is unset.
const DEFAULT_GAMES: usize = 10_000;

/// Full-length uniform games played when `HEXO_SMOKE_UNIFORM` is unset.
const DEFAULT_UNIFORM_GAMES: usize = 30;

fn env_count(key: &str, fallback: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(fallback)
}

/// Per-player line state for the biased driver.
#[derive(Clone, Copy)]
struct Line {
    anchor: HexCoord,
    axis: Axis,
    started: bool,
}

impl Line {
    const fn new(axis: Axis) -> Self {
        Self {
            anchor: HexCoord::ORIGIN,
            axis,
            started: false,
        }
    }
}

/// Outcome of one playout.
struct Playout {
    plies: usize,
    winner: Option<Player>,
}

/// Play one game.
fn playout(seed: u64, noise: u8) -> Playout {
    let mut rng = Rng::new(seed);
    let mut pos = Position::new();
    let mut lines = [
        Line::new(Axis::ALL[(seed % 3) as usize]),
        Line::new(Axis::ALL[((seed / 3) % 3) as usize]),
    ];
    let mut plies = 0usize;
    let mut winner = None;

    while plies < PLY_BOUND {
        if pos.is_terminal() {
            break;
        }
        let n = pos.legal_count();
        assert!(
            n > 0,
            "seed {seed}: no legal moves in a non-terminal position"
        );
        let mover = pos.current_player();
        let uniform = (rng.next_u64() & 0xFF) as u8 <= noise;

        let action = if uniform {
            nth_legal(&pos, rng.below(n)).expect("legal move")
        } else {
            line_move(&pos, &lines[mover.index()])
                .unwrap_or_else(|| nth_legal(&pos, rng.below(n)).expect("legal move"))
        };

        let applied = pos
            .advance(action)
            .unwrap_or_else(|e| panic!("seed {seed} ply {plies}: {e}"));
        plies += 1;

        let line = &mut lines[mover.index()];
        if !line.started {
            line.anchor = action.coord();
            line.started = true;
        }

        assert_eq!(
            applied.outcome.is_some(),
            pos.is_terminal(),
            "seed {seed} ply {plies}: Applied and Position disagree on termination"
        );
        if let Some(o) = applied.outcome {
            assert_eq!(
                o.winner, applied.mover,
                "seed {seed}: the winner is not the mover"
            );
            assert!(!applied.winning.is_empty());
            for w in applied.winning_windows() {
                assert!(
                    pos.window(w).is_full_for(o.winner),
                    "seed {seed}: reported window {w:?} is not full for the winner"
                );
                assert!(
                    w.cells().contains(&applied.action.coord()),
                    "seed {seed}: reported window {w:?} does not contain the placement"
                );
            }
            assert_eq!(
                applied.phase_after, applied.phase_before,
                "seed {seed}: the winning placement did not freeze the phase"
            );
            assert_eq!(pos.current_player(), applied.mover);
            assert_eq!(pos.legal_count(), 0);
            winner = Some(o.winner);
        }
    }

    let oracle = winners_oracle(&pos);
    match winner {
        Some(w) => assert_eq!(oracle, vec![w], "seed {seed}: T3 winner disagreement"),
        None => assert!(
            oracle.is_empty(),
            "seed {seed}: the oracle found a win the engine missed"
        ),
    }
    assert_eq!(pos.is_terminal(), winner.is_some());
    assert_eq!(pos.stone_count() as usize, plies);

    pos.audit()
        .unwrap_or_else(|e| panic!("seed {seed}: audit failed on the final position: {e}"));

    Playout { plies, winner }
}

/// The nearest legal cell that extends this player's line, if any.
fn line_move(pos: &Position, line: &Line) -> Option<Action> {
    if !line.started {
        return None;
    }
    for k in 1..=8i16 {
        for sign in [1i16, -1] {
            let c = HexCoord::new(
                line.anchor.q + line.axis.vector().q * k * sign,
                line.anchor.r + line.axis.vector().r * k * sign,
            );
            let a = Action::new(c);
            if pos.is_legal(a) {
                return Some(a);
            }
        }
    }
    None
}

#[test]
fn line_building_playouts_terminate_and_never_panic() {
    let games = env_count("HEXO_SMOKE_GAMES", DEFAULT_GAMES);
    let mut terminated = 0usize;
    let mut total_plies = 0usize;
    let mut wins = [0usize; 2];

    for seed in 0..games as u64 {
        let noise = ((seed % 16) * 14) as u8;
        let p = playout(seed, noise);
        total_plies += p.plies;
        if let Some(w) = p.winner {
            terminated += 1;
            wins[w.index()] += 1;
            assert!(
                p.plies >= 6,
                "a win needs at least six stones, got {}",
                p.plies
            );
        } else {
            assert_eq!(p.plies, PLY_BOUND, "a non-terminal game must hit the bound");
        }
    }

    eprintln!(
        "smoke: {games} biased games, {total_plies} plies, {terminated} terminated (P0 {}, P1 {})",
        wins[0], wins[1]
    );
    assert!(
        terminated * 2 > games,
        "the biased driver should terminate most games; got {terminated}/{games}"
    );
    assert!(
        wins[0] > 0 && wins[1] > 0,
        "both players should win sometimes"
    );
}

#[test]
fn uniform_playouts_never_panic() {
    let games = env_count("HEXO_SMOKE_UNIFORM", DEFAULT_UNIFORM_GAMES);
    let mut total_plies = 0usize;
    for seed in 0..games as u64 {
        let p = playout(seed ^ 0xdead_beef, u8::MAX);
        total_plies += p.plies;
        if p.winner.is_none() {
            assert_eq!(p.plies, PLY_BOUND);
        }
    }
    eprintln!("smoke: {games} uniform games, {total_plies} plies");
    assert!(total_plies > 0);
}

/// A handful of playouts with the *full* oracle set on every single ply.
#[test]
fn a_few_playouts_are_checked_against_every_oracle_at_every_ply() {
    for seed in 0..12u64 {
        let mut rng = Rng::new(seed ^ 0xa5a5_a5a5);
        let mut pos = Position::new();
        for ply in 0..90usize {
            if pos.is_terminal() {
                break;
            }
            let n = pos.legal_count();
            let a = nth_legal(&pos, rng.below(n)).expect("legal move");
            pos.advance(a)
                .unwrap_or_else(|e| panic!("seed {seed} ply {ply}: {e}"));
            check_all_oracles(&pos, ply + 1);
        }
    }
}

/// The engine imposes no ply cap: a game that reaches the test bound is still a live,
/// legal, auditable position with legal moves available.
#[test]
fn the_engine_has_no_ply_cap() {
    let mut rng = Rng::new(0x1234_5678);
    let mut pos = Position::new();
    let mut plies = 0usize;
    while plies < PLY_BOUND && !pos.is_terminal() {
        let n = pos.legal_count();
        let a = nth_legal(&pos, rng.below(n)).expect("legal move");
        pos.advance(a).expect("legal");
        plies += 1;
    }
    assert!(!pos.is_terminal(), "the fixed seed is expected to run long");
    assert_eq!(plies, PLY_BOUND);
    assert!(
        pos.legal_count() > 0,
        "the engine invented a terminal state"
    );
    let a: Action = pos.legal_actions().next().expect("still has moves");
    pos.advance(a).expect("still playable past the test bound");
    pos.audit().expect("audit");
}
