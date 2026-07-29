//! Mock weight format and deterministic mixing functions.

use crate::NAME;
use hexo_model::PackageError;
use std::path::Path;

/// The weight file inside a checkpoint directory.
///
/// The package owns the file name and contents.
pub(crate) const WEIGHTS_FILE: &str = "weights.mock";

/// How many bytes a weight file holds: one little-endian `u64`.
pub(crate) const WEIGHTS_BYTES: usize = 8;

/// The salt an epoch-0 checkpoint is written with: `"hexo_moc"` as ASCII.
///
/// The fixed value makes [`crate::MockPackage::init`] byte-reproducible.
pub(crate) const INITIAL_SALT: u64 = 0x6865_786f_5f6d_6f63;

/// The SplitMix64 finalizer used by salt derivation, evaluation, and session
/// seeding.
pub(crate) const fn mix(z: u64) -> u64 {
    let z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// A `[0, 1)` draw from 64 mixed bits, with 53 bits of mantissa.
pub(crate) fn unit(bits: u64) -> f64 {
    // 2^-53 times the top 53 bits: exactly representable, and never 1.0.
    (bits >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
}

/// The salt the next epoch's checkpoint is written with.
///
/// The output is deterministic in the prior salt, epoch, and training-data
/// digest.
pub(crate) const fn next_salt(salt: u64, epoch: u32, digest: u64) -> u64 {
    mix(salt ^ (epoch as u64) ^ digest)
}

/// Write a weight file into an existing directory.
pub(crate) fn write(dir: &Path, salt: u64) -> Result<(), PackageError> {
    let path = dir.join(WEIGHTS_FILE);
    std::fs::write(&path, salt.to_le_bytes()).map_err(|source| PackageError::Io { path, source })
}

/// Read a weight file back.
///
/// Any length other than [`WEIGHTS_BYTES`] is
/// [`PackageError::MalformedWeights`].
pub(crate) fn read(dir: &Path) -> Result<u64, PackageError> {
    let path = dir.join(WEIGHTS_FILE);
    let bytes = std::fs::read(&path).map_err(|source| PackageError::Io {
        path: path.clone(),
        source,
    })?;
    let Ok(bytes) = <[u8; WEIGHTS_BYTES]>::try_from(bytes.as_slice()) else {
        return Err(PackageError::MalformedWeights {
            path,
            problem: format!(
                "{NAME} weights are {WEIGHTS_BYTES} bytes holding one little-endian u64 salt; this \
                 file is {} bytes",
                bytes.len()
            ),
        });
    };
    Ok(u64::from_le_bytes(bytes))
}
