//! `serve`: strict JSON-lines framing around batched, incremental seat slots.

mod common;

use common::{init_checkpoint, registry, serve_config};
use hexo_bot::{BotError, Outcome};
use hexo_engine::{
    ACTION_ORDER_VERSION, Action, ActionId, HexCoord, Player, Position, RULES_VERSION,
};
use hexo_model::Manifest;
use hexo_runner::PROTOCOL_VERSION;
use serde_json::{Value, json};
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

struct Fixture {
    _dir: tempfile::TempDir,
    checkpoint: PathBuf,
}

fn fixture() -> Fixture {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let checkpoint = dir.path().join("checkpoint");
    init_checkpoint(&checkpoint, "search=policy").expect("the mock writes a checkpoint");
    Fixture {
        _dir: dir,
        checkpoint,
    }
}

fn hello(checkpoint: &Path) -> Value {
    json!({
        "type": "hello",
        "protocol_version": PROTOCOL_VERSION,
        "rules_version": RULES_VERSION,
        "action_order_version": ACTION_ORDER_VERSION,
        "checkpoint": checkpoint,
        "variant": "policy",
    })
}

fn frame(messages: impl IntoIterator<Item = Value>) -> Vec<u8> {
    let mut bytes = Vec::new();
    for message in messages {
        serde_json::to_writer(&mut bytes, &message).expect("a request serializes");
        bytes.push(b'\n');
    }
    bytes
}

fn responses(bytes: &[u8]) -> Vec<Value> {
    std::str::from_utf8(bytes)
        .expect("seat output is UTF-8")
        .lines()
        .map(|line| serde_json::from_str(line).expect("every response line is JSON"))
        .collect()
}

fn run(input: &[u8]) -> (Result<Outcome, BotError>, Vec<Value>) {
    let config = serve_config(&[
        "serve",
        "--package",
        "mock",
        "--package-config",
        "search=policy",
    ]);
    let mut output = Vec::new();
    let result = hexo_bot::serve(&config, &registry(), Cursor::new(input), &mut output);
    (result, responses(&output))
}

fn hash(value: u64) -> String {
    format!("{value:#018x}")
}

fn replay(ids: &[u32]) -> Position {
    let actions: Vec<Action> = ids
        .iter()
        .map(|&id| Action::from_id(ActionId(id)))
        .collect();
    Position::replay(&actions).expect("the fixture line is legal")
}

const fn side(player: Player) -> &'static str {
    match player {
        Player::P0 => "p0",
        Player::P1 => "p1",
    }
}

fn assert_ok(response: &Value, message: &str) {
    assert_eq!(
        response,
        &json!({"type": "ok", "message": message}),
        "{message} was not acknowledged",
    );
}

#[test]
fn a_successful_handshake_loads_the_checkpoint_and_reports_its_identity() {
    let fixture = fixture();
    let manifest = Manifest::read(&fixture.checkpoint).expect("the checkpoint has a manifest");
    let input = frame([hello(&fixture.checkpoint), json!({"type": "bye"})]);
    let (result, output) = run(&input);

    assert_eq!(result.expect("hello and bye succeed"), Outcome::Completed);
    assert_eq!(output.len(), 2);
    assert_eq!(
        output[0],
        json!({
            "type": "welcome",
            "name": manifest.package,
            "version": manifest.package_version,
            "encoder_version": manifest.encoder_version,
            "resolved_variant": "policy",
            "digest": hash(manifest.probe_hash),
            // A package proposes every legal action, so it declares no
            // restriction and the optional field is absent entirely.
        }),
    );
    assert_ok(&output[1], "bye");
}

#[test]
fn the_handshake_refuses_each_shared_version_disagreement() {
    let fixture = fixture();
    for (field, code, expected) in [
        ("protocol_version", "protocol_version", PROTOCOL_VERSION),
        ("rules_version", "rules_version", RULES_VERSION),
        (
            "action_order_version",
            "action_order_version",
            ACTION_ORDER_VERSION,
        ),
    ] {
        let mut request = hello(&fixture.checkpoint);
        request[field] = json!(expected + 1);
        let input = frame([request]);
        let (result, output) = run(&input);

        assert!(
            matches!(result, Err(BotError::SeatProtocol { .. })),
            "{field} disagreement did not terminate the connection: {result:?}",
        );
        assert_eq!(output.len(), 1);
        assert_eq!(output[0]["type"], "refuse");
        assert_eq!(output[0]["message"], "hello");
        assert!(output[0].get("slot").is_none());
        assert_eq!(output[0]["cause"]["code"], code);
        assert_eq!(output[0]["cause"]["expected"], expected);
        assert_eq!(output[0]["cause"]["got"], expected + 1);
    }
}

#[test]
fn one_policy_round_answers_every_open_slot_in_one_decided_message() {
    let fixture = fixture();
    let a0 = Action::new(HexCoord::new(0, 0)).id().0;
    let a1 = Action::new(HexCoord::new(5, 2)).id().0;
    let a2 = Action::new(HexCoord::new(9, 4)).id().0;
    let positions = [replay(&[a0]), replay(&[a0, a1]), replay(&[a0, a1, a2])];
    let hashes = positions
        .each_ref()
        .map(|position| hash(position.zobrist()));
    let sides = positions
        .each_ref()
        .map(|position| side(position.current_player()));
    let input = frame([
        hello(&fixture.checkpoint),
        json!({
            "type": "open",
            "slots": [
                {"slot": 11, "side": sides[0], "opening": []},
                {"slot": 12, "side": sides[1], "opening": [a0]},
                {"slot": 13, "side": sides[2], "opening": [a0, a1]},
            ],
        }),
        json!({
            "type": "decide",
            "slots": [
                {"slot": 11, "moves": [a0], "zobrist": hashes[0]},
                {"slot": 12, "moves": [a1], "zobrist": hashes[1]},
                {"slot": 13, "moves": [a2], "zobrist": hashes[2]},
            ],
        }),
        json!({"type": "bye"}),
    ]);
    let (result, output) = run(&input);

    assert_eq!(result.expect("the batch succeeds"), Outcome::Completed);
    assert_eq!(output.len(), 4, "one response belongs to each request");
    assert_ok(&output[1], "open");
    assert_eq!(output[2]["type"], "decided");
    let decisions = output[2]["decisions"]
        .as_array()
        .expect("decided carries an array");
    assert_eq!(decisions.len(), 3);
    for ((decision, slot), expected_hash) in decisions.iter().zip([11, 12, 13]).zip(hashes) {
        assert_eq!(decision["slot"], slot);
        assert!(decision["action"].is_u64(), "{decision}");
        assert_eq!(decision["zobrist"], expected_hash);
        assert_eq!(
            decision["diagnostics"],
            Value::Null,
            "the mock evaluation variant authors no diagnostics",
        );
    }
    assert_ok(&output[3], "bye");
}

#[test]
fn a_zobrist_mismatch_names_both_hashes_and_retires_without_resynchronizing() {
    let fixture = fixture();
    let computed = Position::new().zobrist();
    let claimed = computed ^ 1;
    let input = frame([
        hello(&fixture.checkpoint),
        json!({
            "type": "open",
            "slots": [{"slot": 41, "side": "p0", "opening": []}],
        }),
        json!({
            "type": "decide",
            "slots": [{"slot": 41, "moves": [], "zobrist": hash(claimed)}],
        }),
        json!({
            "type": "decide",
            "slots": [{"slot": 41, "moves": [], "zobrist": hash(computed)}],
        }),
        json!({"type": "bye"}),
    ]);
    let (result, output) = run(&input);

    assert_eq!(
        result.expect("slot refusals leave the connection usable"),
        Outcome::Completed,
    );
    assert_ok(&output[1], "open");
    assert_eq!(output[2]["type"], "refuse");
    assert_eq!(output[2]["message"], "decide");
    assert_eq!(output[2]["slot"], 41);
    assert_eq!(output[2]["cause"]["code"], "zobrist_mismatch");
    assert_eq!(output[2]["cause"]["expected"], hash(claimed));
    assert_eq!(output[2]["cause"]["got"], hash(computed));
    let detail = output[2]["cause"]["detail"]
        .as_str()
        .expect("the refusal has detail");
    assert!(detail.contains(&hash(claimed)), "{detail}");
    assert!(detail.contains(&hash(computed)), "{detail}");

    assert_eq!(output[3]["type"], "refuse");
    assert_eq!(output[3]["slot"], 41);
    assert_eq!(output[3]["cause"]["code"], "unknown_slot");
    assert_ok(&output[4], "bye");
}

#[test]
fn a_closed_slot_can_be_opened_again_as_a_fresh_lifecycle() {
    let fixture = fixture();
    let input = frame([
        hello(&fixture.checkpoint),
        json!({
            "type": "open",
            "slots": [{"slot": 73, "side": "p0", "opening": []}],
        }),
        json!({"type": "close", "slots": [73]}),
        json!({
            "type": "open",
            "slots": [{"slot": 73, "side": "p0", "opening": []}],
        }),
        json!({"type": "close", "slots": [73]}),
        json!({"type": "bye"}),
    ]);
    let (result, output) = run(&input);

    assert_eq!(result.expect("the lifecycle succeeds"), Outcome::Completed);
    assert_eq!(output.len(), 6);
    assert_ok(&output[1], "open");
    assert_ok(&output[2], "close");
    assert_ok(&output[3], "open");
    assert_ok(&output[4], "close");
    assert_ok(&output[5], "bye");
}

fn child(input: &[u8]) -> std::process::Output {
    let mut process = Command::new(env!("CARGO_BIN_EXE_hexo-bot"))
        .args([
            "serve",
            "--package",
            "mock",
            "--package-config",
            "search=policy",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("the test binary starts");
    process
        .stdin
        .take()
        .expect("stdin was piped")
        .write_all(input)
        .expect("the script reaches the child");
    process.wait_with_output().expect("the child exits")
}

#[test]
fn bye_flushes_its_acknowledgement_and_the_child_exits_zero() {
    let fixture = fixture();
    let input = frame([hello(&fixture.checkpoint), json!({"type": "bye"})]);
    let output = child(&input);

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr),
    );
    let lines = responses(&output.stdout);
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[0]["type"], "welcome");
    assert_ok(&lines[1], "bye");
}

#[test]
fn a_malformed_line_refuses_once_and_the_child_exits_nonzero() {
    let output = child(b"{this is not json}\n");

    assert!(!output.status.success());
    let lines = responses(&output.stdout);
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0]["type"], "refuse");
    assert_eq!(lines[0]["message"], "line");
    assert_eq!(lines[0]["cause"]["code"], "malformed_message");
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("seat protocol"),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr),
    );
}
