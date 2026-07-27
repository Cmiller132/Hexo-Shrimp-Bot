//! The mock's weights: one `u64` salt, and the one mixing function everything
//! in this package draws from.

use crate::NAME;
use hexo_model::PackageError;
use std::path::Path;

/// The weight file inside a checkpoint directory.
///
/// The extension is the package's, as the format is: the container stores a file
/// and a manifest, never a description of what is in it
/// (`docs/CONTAINER_SPEC.md` §10).
pub(crate) const WEIGHTS_FILE: &str = "weights.mock";

/// How many bytes a weight file holds: one little-endian `u64`.
pub(crate) const WEIGHTS_BYTES: usize = 8;

/// The salt an epoch-0 checkpoint is written with: `"hexo_moc"` as ASCII.
///
/// A fixed documented constant rather than entropy, because
/// [`crate::MockPackage::init`] has to be reproducible: two fresh runs of the
/// same build produce byte-identical epoch-0 checkpoints, and a probe hash that
/// moved between them would be reporting the initialisation rather than the
/// weights.
pub(crate) const INITIAL_SALT: u64 = 0x6865_786f_5f6d_6f63;

/// The splitmix64 finalizer, and the only mixing function this package has.
///
/// The salt derivation, the evaluator's priors and values, and the seeds handed
/// to sessions all draw from this one function, so there is exactly one place to
/// look when an output moves and no chance of two nearly-identical mixers
/// drifting apart. It is the same avalanche `hexo_search::SplitMix64` uses, for
/// the same reason that crate gives for not picking a second algorithm.
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
/// Deterministic in all three inputs. The `digest` is folded over every game the
/// fit actually read, which is what makes reading the shards load-bearing rather
/// than ceremonial: the game count catches a fit that consumed nothing, and the
/// digest catches one that consumed some of it — a fit handed two shards and
/// reading one produces different weights than a fit reading both, and a test can
/// say so.
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
/// A file of any other length is [`PackageError::MalformedWeights`] rather than a
/// short read: the format is eight bytes, and a file that is not eight bytes is
/// not this package's weights however plausible its first eight look.
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
