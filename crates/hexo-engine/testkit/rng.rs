//! The deterministic PRNG shared by `tests/common` and `benches/common`.

/// splitmix64.
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

    /// Uniform-ish value in `0..n`.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}
