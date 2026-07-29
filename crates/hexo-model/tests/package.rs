//! Default `ModelPackage` behavior.
//!
//! `Bare` refuses every required operation and inherits only
//! `variant_session`.

use hexo_engine::Position;
use hexo_model::{Manifest, ModelPackage, PackageError};
use hexo_search::{DecisionSession, Encoder, Evaluator};
use std::path::{Path, PathBuf};

/// An encoder that writes nothing, so `Bare` can answer the one method that has
/// no failure case.
struct Nothing;

impl Encoder for Nothing {
    fn encode(&self, _position: &Position, _out: &mut Vec<u8>) {}
}

/// A package that defines no variants and holds no weights.
struct Bare;

impl Bare {
    fn refuse(&self) -> PackageError {
        PackageError::NotLoaded {
            package: self.name(),
        }
    }
}

impl ModelPackage for Bare {
    fn name(&self) -> &'static str {
        "bare"
    }

    fn package_version(&self) -> u32 {
        1
    }

    fn encoder_version(&self) -> u32 {
        1
    }

    fn init(&self, _dir: &Path) -> Result<Manifest, PackageError> {
        Err(self.refuse())
    }

    fn load(&mut self, _dir: &Path) -> Result<Manifest, PackageError> {
        Err(self.refuse())
    }

    fn encoder(&self) -> Box<dyn Encoder> {
        Box::new(Nothing)
    }

    fn evaluator(&self) -> Result<Box<dyn Evaluator>, PackageError> {
        Err(self.refuse())
    }

    fn self_play_session(&self) -> Result<Box<dyn DecisionSession>, PackageError> {
        Err(self.refuse())
    }

    fn eval_session(&self) -> Result<Box<dyn DecisionSession>, PackageError> {
        Err(self.refuse())
    }

    fn fit(
        &mut self,
        _shards: &[PathBuf],
        _out_dir: &Path,
        _epoch: u32,
    ) -> Result<Manifest, PackageError> {
        Err(self.refuse())
    }
}

#[test]
fn a_package_is_object_safe_because_the_registry_holds_boxed_ones() {
    let packages: Vec<Box<dyn ModelPackage>> = vec![Box::new(Bare)];
    assert_eq!(packages[0].name(), "bare");
}

#[test]
fn the_inherited_variant_session_refuses_every_name_and_says_whose_it_is() {
    let package = Bare;
    for name in ["policy", "mcts:visits=128", "", "anything at all"] {
        let Err(error) = package.variant_session(name) else {
            panic!("{name:?}: the default defines no variants");
        };
        match error {
            PackageError::UnknownVariant { package, variant } => {
                assert_eq!(package, "bare");
                assert_eq!(variant, name);
            }
            other => panic!("{name:?}: {other:?}"),
        }
    }
}

#[test]
fn the_inherited_refusal_reads_as_a_sentence() {
    let Err(error) = Bare.variant_session("greedy") else {
        panic!("no such variant");
    };
    assert_eq!(
        error.to_string(),
        r#"bare has no session variant named "greedy""#
    );
}

#[test]
fn an_unsupported_operation_says_why_it_does_not_exist() {
    let error = PackageError::Unsupported {
        package: "mantisnet",
        operation: "fit",
        reason: "training remains in mantisnet.klent.run",
    };
    assert_eq!(
        error.to_string(),
        "mantisnet does not support fit: training remains in mantisnet.klent.run"
    );
}

#[test]
fn package_metadata_mismatches_keep_both_opaque_values() {
    let error = PackageError::PackageMetadata {
        expected: serde_json::json!({"tau": 0.1, "lambda": 0.03}),
        found: serde_json::json!({"tau": 0.2, "lambda": 0.03}),
    };
    let rendered = error.to_string();
    assert!(rendered.contains(r#""tau":0.1"#), "{rendered}");
    assert!(rendered.contains(r#""tau":0.2"#), "{rendered}");
}
