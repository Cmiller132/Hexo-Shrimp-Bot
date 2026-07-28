//! What a checkpoint manifest says, what it refuses, and what it will not read.

use hexo_model::{MANIFEST_FILE, Manifest, PackageError};
use std::path::Path;

/// A manifest that agrees with this build in every respect.
fn agreeing() -> Manifest {
    Manifest::new("mock", 3, 7, 12, 0x0123_4567_89ab_cdef)
}

/// Write `json` as a checkpoint manifest in a fresh directory and read it back.
fn read_back(dir: &Path, json: &str) -> Result<Manifest, PackageError> {
    std::fs::write(dir.join(MANIFEST_FILE), json).expect("the scratch directory is writable");
    Manifest::read(dir)
}

#[test]
fn a_manifest_survives_a_write_and_a_read_unchanged() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let manifest = agreeing().with_package_metadata(serde_json::json!({
        "lambda": 0.03,
        "tau": 0.1
    }));
    manifest.write(dir.path()).expect("written");
    assert_eq!(Manifest::read(dir.path()).expect("read"), manifest);
}

#[test]
fn a_new_manifest_takes_its_linked_versions_from_the_build_it_was_made_in() {
    let manifest = agreeing();
    assert_eq!(manifest.rules_version, hexo_engine::RULES_VERSION);
    assert_eq!(
        manifest.action_order_version,
        hexo_engine::ACTION_ORDER_VERSION
    );
    assert_eq!(manifest.protocol_version, hexo_runner::PROTOCOL_VERSION);
    assert_eq!(manifest.package, "mock");
    assert_eq!(manifest.package_version, 3);
    assert_eq!(manifest.encoder_version, 7);
    assert_eq!(manifest.epoch, 12);
    assert_eq!(manifest.package_metadata, serde_json::json!({}));
}

#[test]
fn the_probe_hash_is_written_as_a_greppable_hex_string() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    agreeing().write(dir.path()).expect("written");
    let json = std::fs::read_to_string(dir.path().join(MANIFEST_FILE)).expect("read");
    assert!(
        json.contains("\"probe_hash\": \"0x0123456789abcdef\""),
        "{json}"
    );
}

#[test]
fn a_short_probe_hash_is_padded_to_sixteen_digits() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    Manifest::new("mock", 1, 1, 0, 1)
        .write(dir.path())
        .expect("written");
    let json = std::fs::read_to_string(dir.path().join(MANIFEST_FILE)).expect("read");
    assert!(json.contains("\"0x0000000000000001\""), "{json}");
}

#[test]
fn a_manifest_that_agrees_with_the_build_validates() {
    agreeing().validate("mock", 3, 7).expect("it agrees");
}

#[test]
fn a_manifest_from_another_package_is_refused_by_name() {
    let error = agreeing()
        .validate("gnn", 3, 7)
        .expect_err("a different package wrote it");
    match error {
        PackageError::PackageName { expected, found } => {
            assert_eq!(expected, "gnn");
            assert_eq!(found, "mock");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_manifest_from_another_package_version_is_refused_with_both_numbers() {
    let error = agreeing()
        .validate("mock", 4, 7)
        .expect_err("the package moved");
    assert!(
        matches!(
            error,
            PackageError::PackageVersion {
                expected: 4,
                found: 3
            }
        ),
        "{error:?}"
    );
}

#[test]
fn a_manifest_from_another_encoder_version_is_refused_with_both_numbers() {
    let error = agreeing()
        .validate("mock", 3, 8)
        .expect_err("the encoder moved");
    assert!(
        matches!(
            error,
            PackageError::EncoderVersion {
                expected: 8,
                found: 7
            }
        ),
        "{error:?}"
    );
}

#[test]
fn a_manifest_from_another_rules_version_is_refused_with_both_numbers() {
    let mut manifest = agreeing();
    manifest.rules_version = hexo_engine::RULES_VERSION + 1;
    let error = manifest
        .validate("mock", 3, 7)
        .expect_err("the rules moved");
    match error {
        PackageError::RulesVersion { expected, found } => {
            assert_eq!(expected, hexo_engine::RULES_VERSION);
            assert_eq!(found, hexo_engine::RULES_VERSION + 1);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_manifest_from_another_action_ordering_is_refused_with_both_numbers() {
    let mut manifest = agreeing();
    manifest.action_order_version = hexo_engine::ACTION_ORDER_VERSION + 1;
    let error = manifest
        .validate("mock", 3, 7)
        .expect_err("the ordering moved");
    match error {
        PackageError::ActionOrderVersion { expected, found } => {
            assert_eq!(expected, hexo_engine::ACTION_ORDER_VERSION);
            assert_eq!(found, hexo_engine::ACTION_ORDER_VERSION + 1);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_manifest_from_another_protocol_version_is_refused_with_both_numbers() {
    let mut manifest = agreeing();
    manifest.protocol_version = hexo_runner::PROTOCOL_VERSION + 1;
    let error = manifest
        .validate("mock", 3, 7)
        .expect_err("the protocol moved");
    match error {
        PackageError::ProtocolVersion { expected, found } => {
            assert_eq!(expected, hexo_runner::PROTOCOL_VERSION);
            assert_eq!(found, hexo_runner::PROTOCOL_VERSION + 1);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_missing_manifest_is_an_io_error_naming_the_file() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let error = Manifest::read(dir.path()).expect_err("nothing has been written");
    match error {
        PackageError::Io { path, .. } => assert!(path.ends_with(MANIFEST_FILE), "{path:?}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_manifest_missing_a_field_is_refused_rather_than_defaulted() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let error = read_back(
        dir.path(),
        r#"{"package":"mock","package_version":3,"encoder_version":7,
            "rules_version":1,"action_order_version":1,"protocol_version":1,
            "probe_hash":"0x0123456789abcdef"}"#,
    )
    .expect_err("the epoch is missing");
    assert!(
        matches!(error, PackageError::ManifestParse { .. }),
        "{error:?}"
    );
    assert!(error.to_string().contains("epoch"), "{error}");
}

#[test]
fn a_manifest_carrying_an_unknown_field_is_refused_rather_than_tolerated() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let mut json = serde_json::to_string(&agreeing()).expect("serialises");
    json.pop();
    json.push_str(r#","hidden_layers":12}"#);
    let error = read_back(dir.path(), &json).expect_err("the field is not ours");
    assert!(
        matches!(error, PackageError::ManifestParse { .. }),
        "{error:?}"
    );
    assert!(error.to_string().contains("hidden_layers"), "{error}");
}

#[test]
fn a_probe_hash_that_is_not_the_stated_shape_is_refused() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    for hash in [
        "0123456789abcdef",    // no prefix
        "0x123456789abcdef",   // fifteen digits
        "0x0123456789abcdefa", // seventeen
        "0x+123456789abcdef",  // a sign the integer parser would have taken
        "0xzzzzzzzzzzzzzzzz",  // not hex at all
    ] {
        let mut manifest = serde_json::to_value(agreeing()).expect("serialises");
        manifest["probe_hash"] = serde_json::Value::String(hash.to_owned());
        let json = manifest.to_string();
        let error = read_back(dir.path(), &json).expect_err("not the stated shape");
        assert!(
            matches!(error, PackageError::ManifestParse { .. }),
            "{hash}: {error:?}"
        );
    }
}
