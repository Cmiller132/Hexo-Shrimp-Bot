//! The complete Python-free package contract, with an injected scripted forward.

use hexo_model::{ModelPackage, PackageError, probe_hash};
use hexo_model_mantisnet::{
    BoxError, Forward, ForwardLoader, MantisPackage, RawOutputs, WEIGHTS_FILE,
};
use std::path::Path;
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
struct FakeLoader;

impl ForwardLoader for FakeLoader {
    fn load(&self, weights: &Path) -> Result<Box<dyn Forward>, BoxError> {
        let bytes = std::fs::read(weights)?;
        if bytes.len() != 1 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "fake weights are exactly one salt byte",
            )
            .into());
        }
        Ok(Box::new(FakeForward { salt: bytes[0] }))
    }
}

struct FakeForward {
    salt: u8,
}

impl Forward for FakeForward {
    fn forward(
        &mut self,
        batch: &hexo_model_mantisnet::encoder::RawBatch,
    ) -> Result<RawOutputs, BoxError> {
        let cells = usize::try_from(*batch.legal_offsets.last().expect("initial offset"))
            .expect("non-negative");
        let q = f32::from(self.salt).mul_add(1.8 / 255.0, -0.9);
        Ok(RawOutputs {
            policy_logits: (0..cells)
                .map(|index| f32::from(((index + usize::from(self.salt)) % 17) as u8) / 8.0)
                .collect(),
            q_values: vec![q; cells],
        })
    }
}

fn package(config: &str) -> MantisPackage {
    MantisPackage::from_config(config, Arc::new(FakeLoader)).expect("valid package config")
}

fn source_config(source: &Path) -> String {
    format!("tau=0.1,lambda=0.03,source={}", source.display())
}

#[test]
fn init_seals_the_source_and_two_loads_prove_the_same_live_opinion() {
    let scratch = tempfile::tempdir().expect("a scratch directory");
    let source = scratch.path().join("training.pt");
    std::fs::write(&source, [41]).expect("fake training weights");
    let checkpoint = scratch.path().join("checkpoint");

    let written = package(&source_config(&source))
        .init(&checkpoint)
        .expect("checkpoint sealed");
    assert_eq!(std::fs::read(checkpoint.join(WEIGHTS_FILE)).unwrap(), [41]);
    assert_eq!(
        written.package_metadata,
        serde_json::json!({"lambda": 0.03, "tau": 0.1})
    );

    let mut first = package("tau=0.1,lambda=0.03");
    let mut second = package("lambda=0.03,tau=0.1");
    let first_manifest = first.load(&checkpoint).expect("first load proves");
    let second_manifest = second.load(&checkpoint).expect("second load proves");
    assert_eq!(first_manifest.probe_hash, written.probe_hash);
    assert_eq!(second_manifest.probe_hash, written.probe_hash);

    let first_hash = probe_hash(
        first.encoder().as_ref(),
        first.evaluator().unwrap().as_mut(),
    );
    let second_hash = probe_hash(
        second.encoder().as_ref(),
        second.evaluator().unwrap().as_mut(),
    );
    assert_eq!(first_hash, written.probe_hash);
    assert_eq!(second_hash, written.probe_hash);
}

#[test]
fn a_failed_probe_keeps_the_previously_loaded_forward() {
    let scratch = tempfile::tempdir().expect("a scratch directory");
    let source = scratch.path().join("training.pt");
    std::fs::write(&source, [7]).expect("fake training weights");
    let checkpoint = scratch.path().join("checkpoint");
    package(&source_config(&source))
        .init(&checkpoint)
        .expect("checkpoint sealed");

    let mut loaded = package("tau=0.1,lambda=0.03");
    let manifest = loaded.load(&checkpoint).expect("initial load");
    std::fs::write(checkpoint.join(WEIGHTS_FILE), [8]).expect("corrupt the test checkpoint");
    let error = loaded.load(&checkpoint).expect_err("the probe must move");
    assert!(matches!(error, PackageError::ProbeMismatch { .. }));

    let still_loaded = probe_hash(
        loaded.encoder().as_ref(),
        loaded.evaluator().unwrap().as_mut(),
    );
    assert_eq!(still_loaded, manifest.probe_hash);
}

#[test]
fn package_metadata_is_part_of_checkpoint_compatibility() {
    let scratch = tempfile::tempdir().expect("a scratch directory");
    let source = scratch.path().join("training.pt");
    std::fs::write(&source, [1]).expect("fake training weights");
    let checkpoint = scratch.path().join("checkpoint");
    package(&source_config(&source))
        .init(&checkpoint)
        .expect("checkpoint sealed");

    let error = package("tau=0.2,lambda=0.03")
        .load(&checkpoint)
        .expect_err("tau changes the model opinion");
    assert!(matches!(error, PackageError::PackageMetadata { .. }));
}

#[test]
fn sessions_and_evaluators_require_proved_weights() {
    let unproved = package("tau=0.1,lambda=0.03");
    assert!(matches!(
        unproved.evaluator(),
        Err(PackageError::NotLoaded { .. })
    ));
    assert!(matches!(
        unproved.self_play_session(),
        Err(PackageError::NotLoaded { .. })
    ));
    assert!(matches!(
        unproved.eval_session(),
        Err(PackageError::NotLoaded { .. })
    ));
    assert!(matches!(
        unproved.variant_session("policy"),
        Err(PackageError::NotLoaded { .. })
    ));
}

#[test]
fn fit_declines_without_touching_shards_or_output() {
    let scratch = tempfile::tempdir().expect("a scratch directory");
    let output = scratch.path().join("next");
    let mut mantis = package("tau=0.1,lambda=0.03");
    let error = mantis
        .fit(&[scratch.path().join("not-read.hxr")], &output, 1)
        .expect_err("container fitting is deliberately absent");
    assert!(matches!(
        error,
        PackageError::Unsupported {
            package: "mantisnet",
            operation: "fit",
            ..
        }
    ));
    assert!(error.to_string().contains("mantisnet.klent.run"));
    assert!(!output.exists());
}
