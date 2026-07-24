//! Brute-force oracles and a deterministic driver, shared by the integration
//! tests.
//!
//! Everything here is written to **disagree** with the engine. The oracles are
//! `O(stones^2)` or worse, they never touch a crate-private helper, and the
//! Zobrist oracle re-implements the mixing function from the specification
//! rather than calling the crate's. That independence is the whole point:
//! symmetric bugs — a wrong disk offset, a wrong shear, a wrong key constant —
//! apply and un-apply identically, so only a second formulation can catch them.

#![allow(dead_code)]

use hexo_engine::{
    Action, Axis, HexCoord, LEGAL_RADIUS, Player, Position, TurnPhase, WINDOW_LEN, hex_distance,
};

// ---------------------------------------------------------------------------
// Deterministic PRNG (no dependencies)
// ---------------------------------------------------------------------------

/// splitmix64. Deterministic, seedable, and dependency-free.
#[derive(Clone, Debug)]
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

// ---------------------------------------------------------------------------
// T1 — the legal-set oracle
// ---------------------------------------------------------------------------

/// The brute-force union of radius-8 disks over all occupied cells, minus
/// occupied cells, minus (in `Opening`) everything but the origin, minus (when
/// terminal) everything. In canonical `(q, r)` order.
pub fn legal_set_oracle(pos: &Position) -> Vec<HexCoord> {
    if pos.is_terminal() {
        return Vec::new();
    }
    if pos.phase() == TurnPhase::Opening {
        return vec![HexCoord::ORIGIN];
    }
    let stones: Vec<HexCoord> = pos.stones().map(|(c, _)| c).collect();
    let radius = LEGAL_RADIUS as i16;
    let mut out = std::collections::BTreeSet::new();
    for s in &stones {
        for dq in -radius..=radius {
            for dr in -radius..=radius {
                let cell = HexCoord::new(s.q + dq, s.r + dr);
                if hex_distance(*s, cell) > LEGAL_RADIUS {
                    continue;
                }
                if pos.get(cell).is_some() {
                    continue;
                }
                out.insert(cell);
            }
        }
    }
    out.into_iter().collect()
}

// ---------------------------------------------------------------------------
// T2 — the Zobrist oracle, re-derived from the specification
// ---------------------------------------------------------------------------

/// splitmix64 finalizer, transcribed from the specification.
const fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^= x >> 31;
    x
}

/// Hash contribution of one stone, transcribed from the specification.
fn oracle_cell_key(c: HexCoord, p: Player) -> u64 {
    let owner = match p {
        Player::P0 => 0u64,
        Player::P1 => 1u64,
    };
    mix64(((c.q as u16 as u64) << 48) | ((c.r as u16 as u64) << 32) | (1 << 16) | owner)
}

/// Hash contribution of the turn state, transcribed from the specification.
fn oracle_turn_key(slot: usize) -> u64 {
    mix64((1 << 17) | ((slot as u64) << 1))
}

/// The full position hash, recomputed from scratch.
pub fn zobrist_oracle(pos: &Position) -> u64 {
    let mut h = 0u64;
    for (c, p) in pos.stones() {
        h ^= oracle_cell_key(c, p);
    }
    let kind = match pos.phase() {
        TurnPhase::Opening => 0,
        TurnPhase::FirstStone => 1,
        TurnPhase::SecondStone { .. } => 2,
    };
    let mover = match pos.current_player() {
        Player::P0 => 0,
        Player::P1 => 1,
    };
    h ^ oracle_turn_key(kind * 4 + mover * 2 + usize::from(pos.is_terminal()))
}

// ---------------------------------------------------------------------------
// T3 — the win oracle
// ---------------------------------------------------------------------------

/// Brute-force six-in-a-row scan over every stone, every axis, and every
/// offset. Returns every player with a completed window.
pub fn winners_oracle(pos: &Position) -> Vec<Player> {
    let mut found = [false; 2];
    for (c, p) in pos.stones() {
        for axis in Axis::ALL {
            for k in 0..WINDOW_LEN {
                let mut all = true;
                for m in 0..WINDOW_LEN {
                    let cell = HexCoord::new(
                        c.q + axis.vector().q * (m as i16 - k as i16),
                        c.r + axis.vector().r * (m as i16 - k as i16),
                    );
                    if pos.get(cell) != Some(p) {
                        all = false;
                        break;
                    }
                }
                if all {
                    found[match p {
                        Player::P0 => 0,
                        Player::P1 => 1,
                    }] = true;
                }
            }
        }
    }
    let mut out = Vec::new();
    if found[0] {
        out.push(Player::P0);
    }
    if found[1] {
        out.push(Player::P1);
    }
    out
}

// ---------------------------------------------------------------------------
// T4 — the turn-sequence oracle
// ---------------------------------------------------------------------------

/// The literal documented pattern `P0; P1 P1; P0 P0; P1 P1; ...`: who moves at
/// ply `n`, and what phase they are in.
pub fn turn_oracle(ply: usize) -> (Player, TurnPhase) {
    if ply == 0 {
        return (Player::P0, TurnPhase::Opening);
    }
    // Plies 1.. are grouped in pairs: (1,2) -> P1, (3,4) -> P0, (5,6) -> P1 ...
    let pair = (ply - 1) / 2;
    let mover = if pair % 2 == 0 {
        Player::P1
    } else {
        Player::P0
    };
    let phase = if (ply - 1) % 2 == 0 {
        TurnPhase::FirstStone
    } else {
        // The caller checks the kind only; the payload is position-dependent.
        TurnPhase::SecondStone {
            first: HexCoord::ORIGIN,
        }
    };
    (mover, phase)
}

/// Canonical kind index of a phase, ignoring the `SecondStone` payload.
pub fn phase_kind(phase: TurnPhase) -> usize {
    match phase {
        TurnPhase::Opening => 0,
        TurnPhase::FirstStone => 1,
        TurnPhase::SecondStone { .. } => 2,
    }
}

// ---------------------------------------------------------------------------
// Drivers
// ---------------------------------------------------------------------------

/// Pick the `k`-th legal action, wrapping. Returns `None` when terminal.
pub fn nth_legal(pos: &Position, k: usize) -> Option<Action> {
    let n = pos.legal_count();
    if n == 0 {
        return None;
    }
    pos.legal_actions().nth(k % n)
}

/// Play a random legal game and return the move list.
///
/// Stops at termination or at `max_plies`.
pub fn random_game(seed: u64, max_plies: usize) -> Vec<Action> {
    let mut rng = Rng::new(seed);
    let mut pos = Position::new();
    let mut moves = Vec::new();
    while !pos.is_terminal() && moves.len() < max_plies {
        let n = pos.legal_count();
        assert!(n > 0, "a non-terminal position must have legal moves");
        let a = nth_legal(&pos, rng.below(n)).expect("legal move");
        pos.advance(a).expect("the chosen move must be legal");
        moves.push(a);
    }
    moves
}

/// Replay a move list, asserting every ply is accepted.
pub fn replay(moves: &[Action]) -> Position {
    let mut pos = Position::new();
    for (i, &a) in moves.iter().enumerate() {
        pos.advance(a)
            .unwrap_or_else(|e| panic!("ply {i} {a:?} rejected on replay: {e}"));
    }
    pos
}

/// Every per-ply invariant the oracles can check, run against `pos` at ply
/// `ply`.
pub fn check_all_oracles(pos: &Position, ply: usize) {
    // T1
    let listed: Vec<HexCoord> = pos.legal_actions().map(Action::coord).collect();
    assert_eq!(
        listed,
        legal_set_oracle(pos),
        "T1: legal set mismatch at ply {ply}"
    );
    assert_eq!(
        listed.len(),
        pos.legal_count(),
        "legal_count disagrees with the iterator at ply {ply}"
    );

    // T2
    assert_eq!(
        pos.zobrist(),
        zobrist_oracle(pos),
        "T2: zobrist mismatch at ply {ply}"
    );

    // T3
    let winners = winners_oracle(pos);
    assert_eq!(
        pos.is_terminal(),
        !winners.is_empty(),
        "T3: terminal mismatch at ply {ply}"
    );
    if let Some(o) = pos.outcome() {
        assert_eq!(winners, vec![o.winner], "T3: winner mismatch at ply {ply}");
    }

    // T4. `stone_count` is the ply count; a terminal position froze one ply
    // back, which is exactly what the closed form of the pattern reports.
    let n = pos.stone_count() as usize;
    let (mover, phase) = turn_oracle(n - usize::from(pos.is_terminal()));
    assert_eq!(
        pos.current_player(),
        mover,
        "T4: mover mismatch at ply {ply}"
    );
    assert_eq!(
        phase_kind(pos.phase()),
        phase_kind(phase),
        "T4: phase mismatch at ply {ply}"
    );

    // Structural: zero legal moves iff terminal.
    assert_eq!(
        pos.legal_count() == 0,
        pos.is_terminal(),
        "legal_count/terminal disagree at ply {ply}"
    );

    // The full Tier-A audit.
    pos.audit()
        .unwrap_or_else(|e| panic!("audit failed at ply {ply}: {e}"));
}
