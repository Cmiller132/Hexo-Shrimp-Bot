//! The deterministic PRNG shared by `tests/common` and `benches/common`.
//!
//! Neither of those is a module of the other — a bench and an integration test
//! are separate targets and cannot `use` each other — so both pull this file in
//! with `#[path]`. It lives outside `src/` because it is not part of the crate,
//! and outside both `tests/` and `benches/` because it belongs to neither.
//!
//! Having one copy is the point rather than a tidiness preference. The bench
//! fixtures are documented as being *the same positions* the test corpus builds
//! at a given ply, and that is only true while the two generators agree
//! constant for constant. Two hand-matched copies of splitmix64 would make that
//! claim a coincidence, and a divergence would show up as benchmark numbers
//! quietly measuring a different board.

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
