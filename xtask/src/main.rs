//! The workspace verification gates, defined once.
//!
//! Every command CI runs lives in [`GATES`] and nowhere else. `cargo xtask
//! verify` runs the set that guards every push, in the order CI runs it;
//! `cargo xtask <gate>` runs one. The workflow files choose which machine runs
//! which gate, but never restate a command line.
//!
//! That single-source rule is not tidiness. The gate list had already been
//! copied into `README.md` and `crates/hexo-engine/README.md`, and both copies
//! had drifted: neither mentioned the rustdoc gate, the MSRV check, or the
//! `wasm32` lint. Following either one to a green result and calling the work
//! done left three CI failures behind.

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

/// Which toolchain a gate runs under.
enum Toolchain {
    /// Whatever `cargo` already resolves to — current stable, in practice.
    Current,
    /// The `rust-version` floor declared in the workspace manifest, read from
    /// that manifest so the gate cannot outlive the promise it checks.
    Msrv,
}

/// One gate: what it runs, and why it catches something no other gate does.
///
/// `why` is printed when the gate fails, because "clippy failed again" is not
/// the useful part — "this is the release profile, and it sees dead code the
/// debug profile cannot" is.
struct Gate {
    /// The subcommand name, as typed.
    name: &'static str,
    /// Whether `verify` and the per-push CI workflow include this gate.
    on_every_push: bool,
    toolchain: Toolchain,
    /// Environment set for this gate only.
    env: &'static [(&'static str, &'static str)],
    /// Arguments after `cargo`.
    args: &'static [&'static str],
    /// One line for the gate table, then why the gate is not redundant.
    why: &'static str,
}

const GATES: &[Gate] = &[
    Gate {
        name: "fmt",
        on_every_push: true,
        toolchain: Toolchain::Current,
        env: &[],
        args: &["fmt", "--all", "--check"],
        why: "Formatting. Run `cargo fmt --all` to fix.",
    },
    Gate {
        name: "lint",
        on_every_push: true,
        toolchain: Toolchain::Current,
        env: &[],
        args: &["clippy", "--all-targets", "--", "-D", "warnings"],
        why: "Clippy, debug profile, including tests and benches.",
    },
    Gate {
        name: "lint-release",
        on_every_push: true,
        toolchain: Toolchain::Current,
        env: &[],
        args: &[
            "clippy",
            "--release",
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ],
        why: "Clippy under the release profile. Not a duplicate of `lint`: \
              release turns `debug_assertions` off, which deletes the only \
              callers of the helpers the tier-C assertions use. Those helpers \
              become dead code that the debug lint cannot see.",
    },
    Gate {
        name: "test",
        on_every_push: true,
        toolchain: Toolchain::Current,
        env: &[],
        args: &["test", "--workspace"],
        why: "The test suite, debug profile — so the tier-C assertions run on \
              every placement. Includes the differential test against \
              `hexo-reference` at its default size.",
    },
    Gate {
        name: "docs",
        on_every_push: true,
        toolchain: Toolchain::Current,
        env: &[("RUSTDOCFLAGS", "-D warnings")],
        args: &["doc", "--no-deps", "--workspace"],
        why: "Rustdoc. `cargo clippy` does not check it: a broken intra-doc \
              link like [`Position::advance`] compiles fine and silently stops \
              resolving. The module docs cross-reference specific methods \
              heavily, so a warning here is a build failure, not a note.",
    },
    Gate {
        name: "msrv",
        on_every_push: true,
        toolchain: Toolchain::Msrv,
        env: &[],
        args: &["check", "--workspace", "--all-targets"],
        why: "The declared `rust-version` floor. CI otherwise only ever sees \
              current stable, so the promise drifts silently — it already had \
              once. `--all-targets` holds tests and benches to the floor too. \
              If the toolchain is missing, `rustup toolchain install` it.",
    },
    Gate {
        name: "wasm",
        on_every_push: true,
        toolchain: Toolchain::Current,
        env: &[],
        args: &[
            "build",
            "-p",
            "hexo-engine",
            "--target",
            "wasm32-unknown-unknown",
        ],
        why: "`hexo-engine` must stay `wasm32`-compilable, which is what would \
              let a web frontend run the real rules instead of a \
              reimplementation. Nothing in the native build catches a \
              `std::time` call, a threading primitive, or a PyO3 dependency \
              creeping in. Needs `rustup target add wasm32-unknown-unknown`.",
    },
    Gate {
        name: "wasm-lint",
        on_every_push: true,
        toolchain: Toolchain::Current,
        env: &[],
        args: &[
            "clippy",
            "-p",
            "hexo-engine",
            "--target",
            "wasm32-unknown-unknown",
            "--",
            "-D",
            "warnings",
        ],
        why: "Clippy for the `wasm32` target, which lints the cfg branches the \
              native build never compiles.",
    },
    Gate {
        name: "smoke",
        on_every_push: false,
        toolchain: Toolchain::Current,
        env: &[("HEXO_SMOKE_GAMES", "10000"), ("HEXO_SMOKE_UNIFORM", "500")],
        args: &["test", "--release", "-p", "hexo-engine", "--test", "smoke"],
        why: "The deep smoke run — an order of magnitude more games than the \
              defaults `test` uses. Release profile, because a debug build \
              runs the full tier-C assertion set on every placement. Scheduled \
              nightly rather than per-push; that is why `verify` omits it.",
    },
];

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let selected: Vec<&Gate> = match args.as_slice() {
        [one] if one == "verify" => GATES.iter().filter(|g| g.on_every_push).collect(),
        [one] => match GATES.iter().find(|g| g.name == one) {
            Some(gate) => vec![gate],
            None => return usage(&format!("unknown gate `{one}`")),
        },
        [] => return usage("expected a gate name, or `verify`"),
        _ => return usage("expected exactly one argument"),
    };

    for gate in selected {
        if !gate.run() {
            eprintln!("\n\u{2716} gate `{}` failed.\n  {}", gate.name, gate.why);
            return ExitCode::FAILURE;
        }
    }

    println!("\n\u{2714} all gates passed.");
    ExitCode::SUCCESS
}

impl Gate {
    /// Runs the gate from the workspace root, echoing the exact command first
    /// so a failure can be reproduced by hand.
    fn run(&self) -> bool {
        let root = workspace_root();
        let toolchain = match self.toolchain {
            Toolchain::Current => None,
            Toolchain::Msrv => Some(format!("+{}", declared_msrv(&root))),
        };

        let mut shown = String::new();
        for (key, value) in self.env {
            shown.push_str(&format!("{key}={value} "));
        }
        shown.push_str("cargo ");
        if let Some(toolchain) = &toolchain {
            shown.push_str(toolchain);
            shown.push(' ');
        }
        shown.push_str(&self.args.join(" "));
        println!(
            "\n\u{2500}\u{2500} {} \u{2500}\u{2500}\n$ {shown}",
            self.name
        );

        let mut command = Command::new("cargo");
        command.current_dir(&root);
        if let Some(toolchain) = &toolchain {
            command.arg(toolchain);
        }
        command.args(self.args);
        for (key, value) in self.env {
            command.env(key, value);
        }

        match command.status() {
            Ok(status) => status.success(),
            Err(err) => {
                eprintln!("could not run cargo: {err}");
                false
            }
        }
    }
}

/// The workspace root, resolved from this crate's own manifest directory so
/// `cargo xtask` works from any subdirectory.
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask/ always has a parent")
        .to_path_buf()
}

/// The `rust-version` the workspace declares. Read rather than hardcoded: the
/// point of the MSRV gate is that the declared floor and the checked floor are
/// the same number.
fn declared_msrv(root: &Path) -> String {
    let manifest_path = root.join("Cargo.toml");
    let manifest = std::fs::read_to_string(&manifest_path)
        .unwrap_or_else(|err| panic!("cannot read {}: {err}", manifest_path.display()));

    manifest
        .lines()
        .find_map(|line| line.trim().strip_prefix("rust-version = \""))
        .and_then(|rest| rest.strip_suffix('"'))
        .unwrap_or_else(|| panic!("no `rust-version = \"...\"` in {}", manifest_path.display()))
        .to_string()
}

fn usage(problem: &str) -> ExitCode {
    eprintln!("{problem}\n\nusage: cargo xtask <verify|gate>\n");
    eprintln!("  {:<13} every gate marked * below", "verify");
    for gate in GATES {
        let mark = if gate.on_every_push { '*' } else { ' ' };
        let summary = gate.why.split_once('.').map_or(gate.why, |(head, _)| head);
        eprintln!("{mark} {:<13} {summary}", gate.name);
    }
    ExitCode::FAILURE
}
