//! `serve`: one strict JSON-lines seat connection holding position mirrors.
//!
//! The host owns every `Game`. This module owns only slots, and a slot is
//! exactly one [`Position`] plus one [`DecisionSession`]. In particular, this
//! module does not import `hexo_runner::Game`, `Reply`, or any result type.

use crate::Outcome;
use crate::cli::ServeConfig;
use crate::driver::entropy_seed;
use crate::error::BotError;
use crate::registry::PackageRegistry;
use hexo_engine::{ACTION_ORDER_VERSION, Action, ActionId, Player, Position, RULES_VERSION};
use hexo_model::ModelPackage;
use hexo_runner::PROTOCOL_VERSION;
use hexo_search::{
    DecisionSession, EncodedBatch, Encoder, Evaluation, Evaluator, LeafId, SessionStatus,
};
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, Write};
use std::path::Path;
use std::sync::atomic::Ordering;

/// Run one native seat connection.
///
/// `input` and `output` are normally locked stdin and stdout. Keeping them as
/// arguments makes the exact transport loop testable without a child process;
/// the executable supplies no alternate transport.
///
/// The seat constructs one package, loads the checkpoint named by `hello`, and
/// then holds only position/session slots. It never constructs or receives a
/// runner [`hexo_runner::Game`].
///
/// # Errors
///
/// [`BotError::SeatProtocol`] after a connection-terminal refusal, package
/// errors while constructing or loading the requested seat, and
/// [`BotError::Transport`] when either stream breaks.
pub fn serve<R: BufRead, W: Write>(
    config: &ServeConfig,
    registry: &PackageRegistry,
    mut input: R,
    mut output: W,
) -> Result<Outcome, BotError> {
    let mut package = Some(registry.construct(&config.package, &config.package_config)?);
    let mut seat: Option<Seat> = None;
    let mut frame = Vec::new();

    loop {
        frame.clear();
        let read = input
            .read_until(b'\n', &mut frame)
            .map_err(|source| BotError::Transport {
                operation: "reading a seat request from stdin",
                source,
            })?;
        if read == 0 {
            return Ok(if config.stop.load(Ordering::Relaxed) {
                Outcome::Stopped
            } else {
                Outcome::Completed
            });
        }
        if frame.last() != Some(&b'\n') {
            let refusal = Refusal::connection(
                "line",
                Cause::plain(
                    "malformed_line",
                    "stdin closed after a partial request; every JSON object must end in a newline",
                ),
            );
            write_response(&mut output, &Response::from(refusal))?;
            return Err(BotError::seat_protocol(
                "stdin closed after a request without its terminating newline",
            ));
        }

        let request = match serde_json::from_slice::<Request>(&frame) {
            Ok(request) => request,
            Err(source) => {
                let answered = recover_message_name(&frame).unwrap_or_else(|| "line".to_owned());
                let detail = format!("the request is not a seat-protocol JSON object: {source}");
                let refusal = Refusal::connection(
                    answered,
                    Cause::plain("malformed_message", detail.clone()),
                );
                write_response(&mut output, &Response::from(refusal))?;
                return Err(BotError::seat_protocol(detail));
            }
        };

        match (&mut seat, request) {
            (
                None,
                Request::Hello {
                    protocol_version,
                    rules_version,
                    action_order_version,
                    checkpoint,
                    variant,
                },
            ) => {
                if let Some(refusal) =
                    version_refusal(protocol_version, rules_version, action_order_version)
                {
                    let detail = refusal.cause.detail.clone();
                    write_response(&mut output, &Response::from(refusal))?;
                    return Err(BotError::seat_protocol(detail));
                }
                if checkpoint.is_empty() {
                    let refusal = Refusal::connection(
                        "hello",
                        Cause::plain(
                            "checkpoint",
                            "`checkpoint` is empty; hello must name a checkpoint directory",
                        ),
                    );
                    write_response(&mut output, &Response::from(refusal))?;
                    return Err(BotError::seat_protocol(
                        "hello named an empty checkpoint reference",
                    ));
                }
                if variant.is_empty() {
                    let refusal = Refusal::connection(
                        "hello",
                        Cause::plain(
                            "variant",
                            "`variant` is empty; hello must name a package session variant",
                        ),
                    );
                    write_response(&mut output, &Response::from(refusal))?;
                    return Err(BotError::seat_protocol(
                        "hello named an empty session variant",
                    ));
                }

                let mut loaded = package
                    .take()
                    .expect("the package is consumed by exactly one successful hello");
                let manifest = match loaded.load(Path::new(&checkpoint)) {
                    Ok(manifest) => manifest,
                    Err(source) => {
                        let detail = format!("the checkpoint {checkpoint:?} was refused: {source}");
                        let refusal = Refusal::connection(
                            "hello",
                            Cause::plain("checkpoint", detail.clone()),
                        );
                        write_response(&mut output, &Response::from(refusal))?;
                        return Err(BotError::Package(source));
                    }
                };
                let first_session = match loaded.variant_session(&variant) {
                    Ok(session) => session,
                    Err(source) => {
                        let detail =
                            format!("the requested variant {variant:?} was refused: {source}");
                        let refusal = Refusal::connection("hello", Cause::plain("variant", detail));
                        write_response(&mut output, &Response::from(refusal))?;
                        return Err(BotError::Package(source));
                    }
                };
                let encoder = loaded.encoder();
                let evaluator = match loaded.evaluator() {
                    Ok(evaluator) => evaluator,
                    Err(source) => {
                        let detail =
                            format!("the loaded package could not create its evaluator: {source}");
                        let refusal =
                            Refusal::connection("hello", Cause::plain("evaluator", detail));
                        write_response(&mut output, &Response::from(refusal))?;
                        return Err(BotError::Package(source));
                    }
                };

                let welcome = Response::Welcome {
                    name: loaded.name(),
                    version: loaded.package_version(),
                    encoder_version: Some(loaded.encoder_version()),
                    resolved_variant: variant.clone(),
                    digest: hex_u64(manifest.probe_hash),
                    // A package proposes every legal action: the canonical
                    // ordering (§5) is the whole of its candidate set.
                    restriction: None,
                };
                seat = Some(Seat {
                    package: loaded,
                    variant,
                    encoder,
                    evaluator,
                    spare_session: Some(first_session),
                    slots: HashMap::new(),
                    batch: EncodedBatch::new(),
                    leaves: Vec::new(),
                    answers: Vec::new(),
                    next_session_serial: 0,
                });
                write_response(&mut output, &welcome)?;
            }
            (None, request) => {
                let message = request.name();
                let detail = format!("{message} was sent before the required hello handshake");
                let refusal = Refusal::connection(
                    message,
                    Cause::plain("handshake_required", detail.clone()),
                );
                write_response(&mut output, &Response::from(refusal))?;
                return Err(BotError::seat_protocol(detail));
            }
            (Some(_), Request::Hello { .. }) => {
                let detail =
                    "hello was sent after the connection had already completed its handshake";
                let refusal =
                    Refusal::connection("hello", Cause::plain("unexpected_message", detail));
                write_response(&mut output, &Response::from(refusal))?;
                return Err(BotError::seat_protocol(detail));
            }
            (Some(ready), Request::Open { slots }) => {
                if let Some(detail) = answer_slot_request(&mut output, ready.open(slots))? {
                    return Err(BotError::seat_protocol(detail));
                }
            }
            (Some(ready), Request::Decide { slots }) => {
                if let Some(detail) = answer_slot_request(&mut output, ready.decide(slots))? {
                    return Err(BotError::seat_protocol(detail));
                }
            }
            (Some(ready), Request::Close { slots }) => {
                if let Some(detail) = answer_slot_request(&mut output, ready.close(&slots))? {
                    return Err(BotError::seat_protocol(detail));
                }
            }
            (Some(_), Request::Bye) => {
                write_response(&mut output, &Response::Ok { message: "bye" })?;
                return Ok(Outcome::Completed);
            }
        }
    }
}

/// Write a successful response or a slot refusal.
fn answer_slot_request<W: Write>(
    output: &mut W,
    answer: Result<Response, Refusal>,
) -> Result<Option<String>, BotError> {
    match answer {
        Ok(response) => {
            write_response(output, &response)?;
            Ok(None)
        }
        Err(refusal) => {
            let terminal = refusal.slot.is_none().then(|| refusal.cause.detail.clone());
            write_response(output, &Response::from(refusal))?;
            Ok(terminal)
        }
    }
}

/// One loaded seat, still without any runner authority.
struct Seat {
    package: Box<dyn ModelPackage>,
    variant: String,
    encoder: Box<dyn Encoder>,
    evaluator: Box<dyn Evaluator>,
    /// The session built during handshake proves the variant before `welcome`.
    spare_session: Option<Box<dyn DecisionSession>>,
    slots: HashMap<u64, Slot>,
    batch: EncodedBatch,
    leaves: Vec<(usize, LeafId)>,
    answers: Vec<Evaluation>,
    /// Distinguishes session streams even when a slot id is later reused.
    next_session_serial: u64,
}

/// One game as mirrored by this seat.
struct Slot {
    position: Position,
    side: Player,
    session: Box<dyn DecisionSession>,
}

impl Seat {
    /// Open all entries atomically.
    fn open(&mut self, entries: Vec<OpenSlot>) -> Result<Response, Refusal> {
        if entries.is_empty() {
            return Err(Refusal::connection(
                "open",
                Cause::plain("empty_slots", "open must carry at least one slot"),
            ));
        }
        if let Some(slot) = repeated_slot(entries.iter().map(|entry| entry.slot)) {
            self.slots.remove(&slot);
            return Err(Refusal::slot(
                "open",
                slot,
                Cause::plain(
                    "duplicate_slot",
                    format!("slot {slot} occurs more than once in one open request"),
                ),
            ));
        }

        let mut opened = Vec::with_capacity(entries.len());
        for entry in entries {
            let side: Player = entry.side.into();
            if self.slots.contains_key(&entry.slot) {
                self.slots.remove(&entry.slot);
                return Err(Refusal::slot(
                    "open",
                    entry.slot,
                    Cause::plain(
                        "slot_already_open",
                        format!("slot {} is already open and has been retired", entry.slot),
                    ),
                ));
            }
            let actions = actions(&entry.opening);
            let position = match Position::replay(&actions) {
                Ok(position) => position,
                Err(source) => {
                    return Err(Refusal::slot(
                        "open",
                        entry.slot,
                        Cause::plain(
                            "opening_line",
                            format!(
                                "slot {} opening action {} ({}) was refused: {}",
                                entry.slot,
                                source.ply,
                                source.action.id().0,
                                source.cause,
                            ),
                        ),
                    ));
                }
            };
            if position.is_terminal() {
                return Err(Refusal::slot(
                    "open",
                    entry.slot,
                    Cause::plain(
                        "terminal_position",
                        format!(
                            "slot {} opening line is terminal; a seat is only asked about live positions",
                            entry.slot
                        ),
                    ),
                ));
            }
            let mut session = match self.spare_session.take() {
                Some(session) => session,
                None => match self.package.variant_session(&self.variant) {
                    Ok(session) => session,
                    Err(source) => {
                        return Err(Refusal::slot(
                            "open",
                            entry.slot,
                            Cause::plain(
                                "variant",
                                format!(
                                    "slot {} could not construct variant {:?}: {source}",
                                    entry.slot, self.variant
                                ),
                            ),
                        ));
                    }
                },
            };
            session.reseed(entropy_seed(
                usize::try_from(self.next_session_serial)
                    .expect("the seat exhausted this target's session identity space"),
                entry.slot,
                side.index(),
            ));
            self.next_session_serial = self
                .next_session_serial
                .checked_add(1)
                .expect("the seat exhausted its session identity space");
            opened.push((
                entry.slot,
                Slot {
                    position,
                    side,
                    session,
                },
            ));
        }

        for (id, slot) in opened {
            let previous = self.slots.insert(id, slot);
            debug_assert!(previous.is_none(), "open prevalidated every slot id");
        }
        Ok(Response::Ok { message: "open" })
    }

    /// Apply every delta transactionally, search every slot together, and
    /// return every seat-authored decision.
    fn decide(&mut self, entries: Vec<DecideSlot>) -> Result<Response, Refusal> {
        if entries.is_empty() {
            return Err(Refusal::connection(
                "decide",
                Cause::plain("empty_slots", "decide must carry at least one slot"),
            ));
        }
        if let Some(slot) = repeated_slot(entries.iter().map(|entry| entry.slot)) {
            self.slots.remove(&slot);
            return Err(Refusal::slot(
                "decide",
                slot,
                Cause::plain(
                    "duplicate_slot",
                    format!("slot {slot} occurs more than once in one decide request"),
                ),
            ));
        }

        // Work from clones until every mirror, search, and attestation succeeds.
        // A refused batch therefore cannot half-advance the other slots.
        let mut positions = Vec::with_capacity(entries.len());
        for entry in &entries {
            let Some(slot) = self.slots.get(&entry.slot) else {
                return Err(Refusal::slot(
                    "decide",
                    entry.slot,
                    Cause::plain("unknown_slot", format!("slot {} is not open", entry.slot)),
                ));
            };
            let side = slot.side;
            let mut position = slot.position.clone();
            let delta = actions(&entry.moves);
            if let Err(source) = position.replay_from(&delta) {
                self.slots.remove(&entry.slot);
                return Err(Refusal::slot(
                    "decide",
                    entry.slot,
                    Cause::plain(
                        "incremental_line",
                        format!(
                            "slot {} incremental action {} ({}) was refused: {}; the slot has been retired",
                            entry.slot,
                            source.ply,
                            source.action.id().0,
                            source.cause,
                        ),
                    ),
                ));
            }
            let computed = position.zobrist();
            if computed != entry.zobrist.0 {
                self.slots.remove(&entry.slot);
                return Err(Refusal::slot(
                    "decide",
                    entry.slot,
                    Cause::values(
                        "zobrist_mismatch",
                        format!(
                            "slot {} expected zobrist {} but computed {}; the slot has been retired",
                            entry.slot,
                            hex_u64(entry.zobrist.0),
                            hex_u64(computed),
                        ),
                        Value::String(hex_u64(entry.zobrist.0)),
                        Value::String(hex_u64(computed)),
                    ),
                ));
            }
            if position.is_terminal() {
                self.slots.remove(&entry.slot);
                return Err(Refusal::slot(
                    "decide",
                    entry.slot,
                    Cause::plain(
                        "terminal_position",
                        format!(
                            "slot {} is terminal after its incremental line; the slot has been retired",
                            entry.slot
                        ),
                    ),
                ));
            }
            if position.current_player() != side {
                self.slots.remove(&entry.slot);
                return Err(Refusal::slot(
                    "decide",
                    entry.slot,
                    Cause::plain(
                        "wrong_side",
                        format!(
                            "slot {} belongs to {} but its mirror has {} to move; the slot has been retired",
                            entry.slot,
                            side_name(side),
                            side_name(position.current_player()),
                        ),
                    ),
                ));
            }
            positions.push(position);
        }

        for (entry, position) in entries.iter().zip(&positions) {
            self.slots
                .get_mut(&entry.slot)
                .expect("every decide slot was prevalidated")
                .session
                .begin(position);
        }

        let mut decided = vec![false; entries.len()];
        while decided.iter().any(|done| !done) {
            self.batch.clear();
            self.leaves.clear();

            for (index, entry) in entries.iter().enumerate() {
                if decided[index] {
                    continue;
                }
                let before = self.batch.len();
                let slot = self
                    .slots
                    .get_mut(&entry.slot)
                    .expect("every active decision belongs to an open slot");
                let status = slot.session.pump(&mut |leaf, position| {
                    self.leaves.push((index, leaf));
                    self.batch.push_with(self.encoder.as_ref(), position);
                });
                match status {
                    SessionStatus::Decided => {
                        assert_eq!(
                            self.batch.len(),
                            before,
                            "slot {} emitted leaves from a decided pump",
                            entry.slot,
                        );
                        decided[index] = true;
                    }
                    SessionStatus::AwaitingEvals { .. } => {
                        assert!(
                            self.batch.len() > before,
                            "slot {} awaits evaluations but emitted no leaf",
                            entry.slot,
                        );
                    }
                }
            }

            if self.batch.is_empty() {
                assert!(
                    decided.iter().all(|done| *done),
                    "an unfinished decision round produced no evaluator work"
                );
                break;
            }

            // This is the batching invariant of the seat protocol: every leaf
            // emitted by every active slot in this round crosses through this
            // one call, then answers are scattered by (slot-index, LeafId).
            self.answers.clear();
            self.evaluator.evaluate(&self.batch, &mut self.answers);
            assert_eq!(
                self.answers.len(),
                self.batch.len(),
                "the seat evaluator answered {} items of a {}-item round",
                self.answers.len(),
                self.batch.len(),
            );
            assert_eq!(
                self.leaves.len(),
                self.answers.len(),
                "every encoded leaf has exactly one evaluator answer"
            );
            for ((index, leaf), evaluation) in self.leaves.drain(..).zip(self.answers.drain(..)) {
                let entry = &entries[index];
                self.slots
                    .get_mut(&entry.slot)
                    .expect("the slot remains open for the whole synchronous round")
                    .session
                    .resume(leaf, evaluation);
            }
        }

        let mut decisions = Vec::with_capacity(entries.len());
        for ((entry, position), done) in entries.iter().zip(&positions).zip(decided) {
            assert!(
                done,
                "the batched pump loop returns only when every slot decided"
            );
            let decision = self
                .slots
                .get_mut(&entry.slot)
                .expect("the slot remains open until its decision is checked")
                .session
                .take_decision()
                .expect("a decided session authors one complete decision");
            if decision.zobrist != position.zobrist() {
                self.slots.remove(&entry.slot);
                return Err(Refusal::slot(
                    "decide",
                    entry.slot,
                    Cause::values(
                        "attestation_mismatch",
                        format!(
                            "slot {} session attested {} but its mirror is {}; the slot has been retired",
                            entry.slot,
                            hex_u64(decision.zobrist),
                            hex_u64(position.zobrist()),
                        ),
                        Value::String(hex_u64(position.zobrist())),
                        Value::String(hex_u64(decision.zobrist)),
                    ),
                ));
            }
            // Deliberately no legality check and no mirror advance. The host
            // submits this Decision verbatim to its Game, and the next decide
            // delta tells us which placements were actually accepted.
            decisions.push(WireDecision {
                slot: entry.slot,
                action: decision.action.id().0,
                zobrist: hex_u64(decision.zobrist),
                diagnostics: decision.diagnostics,
            });
        }

        for ((entry, position), _) in entries.iter().zip(positions).zip(&decisions) {
            self.slots
                .get_mut(&entry.slot)
                .expect("a successful decide keeps every slot open")
                .position = position;
        }
        Ok(Response::Decided { decisions })
    }

    /// Release every named slot atomically.
    fn close(&mut self, ids: &[u64]) -> Result<Response, Refusal> {
        if ids.is_empty() {
            return Err(Refusal::connection(
                "close",
                Cause::plain("empty_slots", "close must carry at least one slot"),
            ));
        }
        if let Some(slot) = repeated_slot(ids.iter().copied()) {
            self.slots.remove(&slot);
            return Err(Refusal::slot(
                "close",
                slot,
                Cause::plain(
                    "duplicate_slot",
                    format!("slot {slot} occurs more than once in one close request"),
                ),
            ));
        }
        for &slot in ids {
            if !self.slots.contains_key(&slot) {
                return Err(Refusal::slot(
                    "close",
                    slot,
                    Cause::plain("unknown_slot", format!("slot {slot} is not open")),
                ));
            }
        }
        for slot in ids {
            self.slots
                .remove(slot)
                .expect("close prevalidated every slot id");
        }
        Ok(Response::Ok { message: "close" })
    }
}

/// A strict request object.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
enum Request {
    /// Establish versions, weights, and one package-owned session shape.
    Hello {
        protocol_version: u32,
        rules_version: u32,
        action_order_version: u32,
        checkpoint: String,
        variant: String,
    },
    /// Allocate new mirrors and sessions.
    Open { slots: Vec<OpenSlot> },
    /// Advance mirrors and search every named slot as one round.
    Decide { slots: Vec<DecideSlot> },
    /// Release mirrors and sessions.
    Close { slots: Vec<u64> },
    /// Flush the success response and exit zero.
    Bye,
}

impl Request {
    const fn name(&self) -> &'static str {
        match self {
            Self::Hello { .. } => "hello",
            Self::Open { .. } => "open",
            Self::Decide { .. } => "decide",
            Self::Close { .. } => "close",
            Self::Bye => "bye",
        }
    }
}

/// One slot in `open`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OpenSlot {
    slot: u64,
    side: WireSide,
    opening: Vec<u32>,
}

/// One slot in `decide`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DecideSlot {
    slot: u64,
    moves: Vec<u32>,
    zobrist: WireU64,
}

/// A side's exact wire spelling.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum WireSide {
    P0,
    P1,
}

impl From<WireSide> for Player {
    fn from(side: WireSide) -> Self {
        match side {
            WireSide::P0 => Self::P0,
            WireSide::P1 => Self::P1,
        }
    }
}

/// A fixed-width hexadecimal `u64`, avoiding JSON's interoperable integer
/// precision limit for hashes.
#[derive(Clone, Copy, Debug)]
struct WireU64(u64);

impl<'de> Deserialize<'de> for WireU64 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        let Some(digits) = text.strip_prefix("0x") else {
            return Err(D::Error::custom(format!(
                "{text:?} does not start with `0x`; expected `0x` and sixteen lowercase hex digits"
            )));
        };
        if digits.len() != 16
            || !digits
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(D::Error::custom(format!(
                "{text:?} is not `0x` followed by sixteen lowercase hex digits"
            )));
        }
        u64::from_str_radix(digits, 16)
            .map(Self)
            .map_err(D::Error::custom)
    }
}

/// Every successful response and the one refusal envelope.
#[derive(Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum Response {
    /// §3.1's seat identity. One shape serves every seat: a `hexo-model`
    /// package fills `encoder_version` and puts its probe hash (§10.2) in
    /// `digest`, while an independent engine reaching this protocol through its
    /// own adapter omits the encoder version and digests its own weights.
    /// `restriction` is absent for a seat that proposes every legal action.
    Welcome {
        name: &'static str,
        version: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        encoder_version: Option<u32>,
        resolved_variant: String,
        digest: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        restriction: Option<String>,
    },
    Ok {
        message: &'static str,
    },
    Decided {
        decisions: Vec<WireDecision>,
    },
    Refuse {
        message: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        slot: Option<u64>,
        cause: Cause,
    },
}

impl From<Refusal> for Response {
    fn from(refusal: Refusal) -> Self {
        Self::Refuse {
            message: refusal.message,
            slot: refusal.slot,
            cause: *refusal.cause,
        }
    }
}

/// One slot's complete seat-authored decision.
#[derive(Serialize)]
struct WireDecision {
    slot: u64,
    action: u32,
    zobrist: String,
    diagnostics: Option<Vec<u8>>,
}

/// Machine-readable refusal classification plus a diagnostic explanation.
#[derive(Serialize)]
struct Cause {
    code: &'static str,
    detail: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    expected: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    got: Option<Value>,
}

impl Cause {
    fn plain(code: &'static str, detail: impl Into<String>) -> Self {
        Self {
            code,
            detail: detail.into(),
            expected: None,
            got: None,
        }
    }

    fn values(code: &'static str, detail: impl Into<String>, expected: Value, got: Value) -> Self {
        Self {
            code,
            detail: detail.into(),
            expected: Some(expected),
            got: Some(got),
        }
    }
}

/// A refusal before serialization.
struct Refusal {
    message: String,
    slot: Option<u64>,
    cause: Box<Cause>,
}

impl Refusal {
    fn connection(message: impl Into<String>, cause: Cause) -> Self {
        Self {
            message: message.into(),
            slot: None,
            cause: Box::new(cause),
        }
    }

    fn slot(message: impl Into<String>, slot: u64, cause: Cause) -> Self {
        Self {
            message: message.into(),
            slot: Some(slot),
            cause: Box::new(cause),
        }
    }
}

/// Shared-version handshake checks, in protocol/rules/action-order order.
fn version_refusal(protocol: u32, rules: u32, action_order: u32) -> Option<Refusal> {
    for (code, label, expected, got) in [
        (
            "protocol_version",
            "PROTOCOL_VERSION",
            PROTOCOL_VERSION,
            protocol,
        ),
        ("rules_version", "RULES_VERSION", RULES_VERSION, rules),
        (
            "action_order_version",
            "ACTION_ORDER_VERSION",
            ACTION_ORDER_VERSION,
            action_order,
        ),
    ] {
        if expected != got {
            return Some(Refusal::connection(
                "hello",
                Cause::values(
                    code,
                    format!("{label} disagreement: seat has {expected}, hello carries {got}"),
                    json!(expected),
                    json!(got),
                ),
            ));
        }
    }
    None
}

/// Serialize exactly one object, append its frame newline, and make it visible.
fn write_response<W: Write>(output: &mut W, response: &Response) -> Result<(), BotError> {
    let mut line = serde_json::to_vec(response).expect("seat responses always serialize");
    line.push(b'\n');
    output
        .write_all(&line)
        .map_err(|source| BotError::Transport {
            operation: "writing a seat response to stdout",
            source,
        })?;
    output.flush().map_err(|source| BotError::Transport {
        operation: "flushing a seat response on stdout",
        source,
    })
}

/// Recover a valid string tag for a malformed-but-parseable object.
fn recover_message_name(frame: &[u8]) -> Option<String> {
    serde_json::from_slice::<Value>(frame)
        .ok()?
        .as_object()?
        .get("type")?
        .as_str()
        .map(str::to_owned)
}

/// Convert action ids without asking whether they are legal.
fn actions(ids: &[u32]) -> Vec<Action> {
    ids.iter()
        .map(|&raw| Action::from_id(ActionId(raw)))
        .collect()
}

/// The first repeated id, if any.
fn repeated_slot(ids: impl IntoIterator<Item = u64>) -> Option<u64> {
    let mut seen = HashSet::new();
    ids.into_iter().find(|slot| !seen.insert(*slot))
}

/// Fixed-width lowercase wire hash.
fn hex_u64(value: u64) -> String {
    format!("{value:#018x}")
}

/// Human spelling used only inside refusal diagnostics.
const fn side_name(side: Player) -> &'static str {
    match side {
        Player::P0 => "p0",
        Player::P1 => "p1",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hexo_model_mock::MockPackage;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;

    /// Makes the device-boundary count observable without changing the mock
    /// package's production evaluator.
    struct CountingEvaluator {
        inner: Box<dyn Evaluator>,
        calls: Arc<AtomicUsize>,
    }

    impl Evaluator for CountingEvaluator {
        fn evaluate(&mut self, batch: &EncodedBatch, out: &mut Vec<Evaluation>) {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.inner.evaluate(batch, out);
        }
    }

    #[test]
    fn one_policy_round_crosses_the_evaluator_once_for_every_slot() {
        let scratch = tempfile::tempdir().expect("a scratch directory");
        let checkpoint = scratch.path().join("checkpoint");
        let mut package: Box<dyn ModelPackage> =
            Box::new(MockPackage::from_config("search=policy").expect("mock config"));
        package.init(&checkpoint).expect("mock checkpoint");
        package.load(&checkpoint).expect("proved mock checkpoint");

        let calls = Arc::new(AtomicUsize::new(0));
        let first_session = package.variant_session("policy").expect("policy variant");
        let encoder = package.encoder();
        let evaluator = package.evaluator().expect("loaded evaluator");
        let mut seat = Seat {
            package,
            variant: "policy".to_owned(),
            encoder,
            evaluator: Box::new(CountingEvaluator {
                inner: evaluator,
                calls: Arc::clone(&calls),
            }),
            spare_session: Some(first_session),
            slots: HashMap::new(),
            batch: EncodedBatch::new(),
            leaves: Vec::new(),
            answers: Vec::new(),
            next_session_serial: 0,
        };

        assert!(
            seat.open(
                [1, 2, 3]
                    .map(|slot| OpenSlot {
                        slot,
                        side: WireSide::P0,
                        opening: Vec::new(),
                    })
                    .into(),
            )
            .is_ok(),
            "three slots open"
        );
        let zobrist = Position::new().zobrist();
        let response = match seat.decide(
            [1, 2, 3]
                .map(|slot| DecideSlot {
                    slot,
                    moves: Vec::new(),
                    zobrist: WireU64(zobrist),
                })
                .into(),
        ) {
            Ok(response) => response,
            Err(_) => panic!("one batched decision succeeds"),
        };

        let Response::Decided { decisions } = response else {
            panic!("a successful decide returns decided");
        };
        assert_eq!(decisions.len(), 3);
        assert_eq!(
            calls.load(Ordering::Relaxed),
            1,
            "three policy roots belong to one aggregate evaluator round",
        );
    }
}
