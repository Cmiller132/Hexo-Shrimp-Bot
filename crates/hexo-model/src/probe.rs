//! The probe: a frozen set of positions, forwarded whole, hashed over the exact
//! bytes the evaluator returned.
//!
//! `docs/CONTAINER_SPEC.md` §10.2 states what this is for, and every failure in
//! its list is silent: the wrong checkpoint loaded, a swap that constant-folding
//! turned into a no-op, a mismatched encoder version, a scrambled action
//! ordering, or a runtime that drifted between build and run. None of them
//! crash. All of them train or play against the wrong weights indefinitely.
//!
//! Three properties make the number mean something, and each is load-bearing:
//!
//! - **The whole probe set is one batch, forwarded once.** Batch shape decides
//!   which kernel runs on a GPU, so a probe split across two batches could
//!   produce two different hashes from one set of weights and the detector would
//!   be reporting its own arithmetic.
//! - **The hash is over the evaluator's exact output bytes**, not over the
//!   weights and not over a re-derived summary. Hashing the weights would miss
//!   every failure that leaves the file intact and answers with something else.
//! - **The positions are fixed and derived from nothing.** They are hardcoded
//!   move lists replayed through the engine, with no RNG and no dependence on
//!   the caller, so the same binary with the same weights produces the same hash
//!   every time and everything that varies is a real difference.

use hexo_engine::{Action, HexCoord, Position};
use hexo_search::{EncodedBatch, Encoder, Evaluator};

/// FNV-1a's 64-bit offset basis.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;

/// FNV-1a's 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// A packed game whose prefixes supply seven of the probe positions.
///
/// One list rather than seven because the positions wanted are the *same* game
/// at different plies — the opening, the first stone, a stone into a turn, and
/// three mid-game boards — and prefixes state that directly. The line lengths
/// are deliberately capped at five: `P0` holds `(0, 0)..(0, 4)` along `R` and
/// `P1` holds `(1, 1)..(5, 1)` along `Q`, so both sides sit one placement from a
/// win and an evaluator with any opinion at all has somewhere to put it.
static PACKED: [(i16, i16); 21] = [
    (0, 0),
    (1, 0),
    (2, 0),
    (0, 1),
    (0, 2),
    (3, 0),
    (1, 1),
    (0, 3),
    (1, 2),
    (2, 1),
    (3, 1),
    (1, 3),
    (2, 2),
    (4, 0),
    (4, 1),
    (0, 4),
    (1, 4),
    (5, 1),
    (-1, 0),
    (2, -1),
    (3, -1),
];

/// `P1` on the *first* stone of a turn it can win with two placements: it holds
/// `(1, 0)..(4, 0)` and closes the window with `(5, 0)` and `(6, 0)`.
///
/// The interesting case for this game specifically. A turn is two placements, so
/// the two plies that win have the **same** mover, which is the shape a value
/// signed by depth parity gets exactly backwards.
static WIN_IN_TWO: [(i16, i16); 9] = [
    (0, 0),
    (1, 0),
    (2, 0),
    (0, 1),
    (0, 3),
    (3, 0),
    (4, 0),
    (0, 5),
    (0, 7),
];

/// `P1` on the *second* stone of a turn, one placement from six in a row: it
/// holds `(1, 0)`, `(2, 0)`, `(4, 0)`, `(5, 0)`, `(6, 0)`, and `(3, 0)` closes
/// the window.
static WIN_IN_ONE: [(i16, i16); 10] = [
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
];

/// Stones pushed out to the legality radius in every direction, so the frontier
/// is the union of thirteen disks rather than one.
///
/// This is the position whose `legal_count` is nothing like the others', which
/// is what makes it catch an encoder that writes a fixed-width row: the board has
/// no edge, and a crop that fits every other probe position does not fit this
/// one.
static SCATTERED: [(i16, i16); 13] = [
    (0, 0),
    (6, 0),
    (0, 6),
    (-6, 0),
    (0, -6),
    (6, -6),
    (-6, 6),
    (3, 3),
    (-3, -3),
    (8, 0),
    (0, 8),
    (-8, 0),
    (0, -8),
];

/// The move lists behind [`probe_positions`], in the order they are hashed.
fn probe_games() -> [&'static [(i16, i16)]; 10] {
    [
        // 0 plies: the opening, where the only legal placement is the origin
        // and `legal_count` is 1.
        &PACKED[..0],
        // 1 ply: `P1` on the first stone of the first full turn.
        &PACKED[..1],
        // 2 plies: `P1` a stone into its turn — the mid-turn state that a search
        // signing values by depth parity gets wrong.
        &PACKED[..2],
        // 5 plies: two turns in.
        &PACKED[..5],
        // 9 plies: `P1` to move, two placements from a win.
        &WIN_IN_TWO,
        // 10 plies: `P1` mid-turn, one placement from a win.
        &WIN_IN_ONE,
        // 11 plies: a packed mid-game, `P0` to move.
        &PACKED[..11],
        // 12 plies: the same board one placement later, `P0` mid-turn.
        &PACKED[..12],
        // 13 plies: the widest frontier in the set.
        &SCATTERED,
        // 21 plies: the deepest board, with both sides one placement from a win.
        &PACKED,
    ]
}

/// The probe positions, in the order [`probe_hash`] forwards them.
///
/// Ten of them, spanning plies 0, 1, 2, 5, 9, 10, 11, 12, 13, and 21: the
/// opening, the first stone of a turn, two mid-turn states, two positions one
/// turn from a decided game, a wide-frontier board, and three ordinary
/// mid-games. Varied on purpose — a probe set that was all openings would agree
/// with itself under an encoder bug that only shows up once there are stones to
/// encode.
///
/// **The set is frozen.** Changing it changes every probe hash, which invalidates
/// every checkpoint manifest on disk — that is a regeneration, in the way this
/// workspace changes formats, and not an edit to make in passing.
///
/// # Panics
///
/// Never, for the lists in this module: each one is replayed through the engine
/// and a list that is not a legal game, or that ends in a terminal position with
/// no legal action to have a prior for, is a bug in this file that the panic
/// names.
#[must_use]
pub fn probe_positions() -> Vec<Position> {
    probe_games()
        .iter()
        .map(|moves| {
            let actions: Vec<Action> = moves
                .iter()
                .map(|&(q, r)| Action::new(HexCoord::new(q, r)))
                .collect();
            let position = Position::replay(&actions)
                .unwrap_or_else(|e| panic!("probe game {moves:?} is not a legal game: {e}"));
            assert!(
                !position.is_terminal(),
                "probe game {moves:?} ends in a terminal position, which has no legal actions to \
                 carry priors for",
            );
            position
        })
        .collect()
}

/// The probe hash of the weights `evaluator` is holding.
///
/// Every probe position is encoded into **one** [`EncodedBatch`] and answered by
/// **one** [`Evaluator::evaluate`] call, and the hash folds the exact
/// little-endian bytes of every prior and every value, in order, through FNV-1a.
/// Nothing is rounded, bucketed, or summarised on the way: the point is to
/// notice a difference, and a summary is a place for one to hide.
///
/// # Panics
///
/// If the evaluator returns a different number of answers than the batch held,
/// or if any answer's prior count is not its position's `legal_count`. Both are
/// package bugs of the same kind `hexo_search::Evaluation`'s own checks panic
/// on, and both have to be loud here rather than skipped: a probe that hashed a
/// misaligned answer would still produce a stable number, so the detector would
/// go on agreeing with itself while describing nothing.
pub fn probe_hash(encoder: &dyn Encoder, evaluator: &mut dyn Evaluator) -> u64 {
    let positions = probe_positions();
    let mut batch = EncodedBatch::with_capacity(positions.len(), 0);
    for position in &positions {
        batch.push_with(encoder, position);
    }

    let mut answers = Vec::with_capacity(positions.len());
    evaluator.evaluate(&batch, &mut answers);
    assert_eq!(
        answers.len(),
        positions.len(),
        "the evaluator answered {} of {} probe positions; the probe is one whole batch and one \
         forward, so a short answer list is a package that lost part of it",
        answers.len(),
        positions.len(),
    );

    let mut hash = FNV_OFFSET_BASIS;
    for (index, (position, evaluation)) in positions.iter().zip(&answers).enumerate() {
        assert_eq!(
            evaluation.priors.len(),
            position.legal_count(),
            "probe position {index} has {} legal actions but its evaluation carries {} priors; \
             priors are indexed by the engine's canonical legal order",
            position.legal_count(),
            evaluation.priors.len(),
        );
        for prior in &evaluation.priors {
            hash = fold(hash, &prior.to_le_bytes());
        }
        hash = fold(hash, &evaluation.value.to_le_bytes());
    }
    hash
}

/// One FNV-1a step per byte: xor, then multiply.
///
/// Written out here rather than taken as a dependency because it is five lines
/// and because the constants are part of the checkpoint format — a hash function
/// that changed under a version bump of somebody else's crate would invalidate
/// every manifest on disk without anything in this workspace having moved.
fn fold(hash: u64, bytes: &[u8]) -> u64 {
    let mut hash = hash;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fold_matches_the_published_fnv1a_vectors() {
        // The reference vectors for FNV-1a 64. They pin the offset basis, the
        // prime, and the xor-then-multiply order all at once: swapping the two
        // operations produces FNV-1, which is a different function that would
        // otherwise pass every determinism test in this crate.
        assert_eq!(fold(FNV_OFFSET_BASIS, b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fold(FNV_OFFSET_BASIS, b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fold(FNV_OFFSET_BASIS, b"foobar"), 0x8594_4171_f739_67e8);
    }

    #[test]
    fn folding_in_two_steps_equals_folding_in_one() {
        let split = fold(fold(FNV_OFFSET_BASIS, b"foo"), b"bar");
        assert_eq!(split, fold(FNV_OFFSET_BASIS, b"foobar"));
    }

    #[test]
    fn the_probe_set_is_frozen() {
        // A golden vector over the *positions*, independent of any encoder or
        // evaluator: it folds each position's zobrist and legal count, both of
        // which the engine states and this file does not. Editing a move list
        // moves this number, and moving it invalidates every checkpoint manifest
        // that exists — which is the point of having to change the constant by
        // hand.
        let mut hash = FNV_OFFSET_BASIS;
        for position in probe_positions() {
            hash = fold(hash, &position.zobrist().to_le_bytes());
            hash = fold(hash, &(position.legal_count() as u64).to_le_bytes());
        }
        assert_eq!(hash, 0x656d_6f60_cb31_b861);
    }
}
