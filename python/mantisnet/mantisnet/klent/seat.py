"""Client for one independent section 3.1 subprocess seat.

Owns the JSON-lines transport, wire encodings, and response validation.
"""

from __future__ import annotations

import collections
import json
import math
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import hexo_py

_U32_MAX = (1 << 32) - 1
_U64_MAX = (1 << 64) - 1
_HASH_LENGTH = 18
_STDERR_CHUNKS = 64
_SHUTDOWN_SECONDS = 5.0


class SeatError(RuntimeError):
    """A participant connection failed, so no match result is sound."""


@dataclass(frozen=True)
class Participant:
    """One independently launched seat and the two configurable hello fields."""

    id: str
    command: tuple[str, ...]
    checkpoint: str
    variant: str

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("participant id must be a nonempty string")
        if isinstance(self.command, (str, bytes)):
            raise TypeError(f"{self.id}: command must be an argv sequence, not a string")
        command = tuple(self.command)
        if not command or any(not isinstance(arg, str) or not arg for arg in command):
            raise ValueError(
                f"{self.id}: command must contain one or more nonempty argv strings"
            )
        if not isinstance(self.checkpoint, str) or not self.checkpoint:
            raise ValueError(f"{self.id}: checkpoint must be a nonempty string")
        if not isinstance(self.variant, str) or not self.variant:
            raise ValueError(f"{self.id}: variant must be a nonempty string")
        object.__setattr__(self, "command", command)

    def hello(self) -> dict[str, Any]:
        """The exact first request, with versions from the loaded extension."""
        return {
            "type": "hello",
            "protocol_version": hexo_py.PROTOCOL_VERSION,
            "rules_version": hexo_py.RULES_VERSION,
            "action_order_version": hexo_py.ACTION_ORDER_VERSION,
            "checkpoint": self.checkpoint,
            "variant": self.variant,
        }


def _is_uint(value: Any, maximum: int) -> bool:
    return type(value) is int and 0 <= value <= maximum


def _is_hash(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _HASH_LENGTH
        and value.startswith("0x")
        and all(char in "0123456789abcdef" for char in value[2:])
    )


def zobrist(position: Any) -> str:
    return f"0x{position.zobrist:016x}"


def wire_action(move: tuple[int, int]) -> int:
    q, r = (int(move[0]), int(move[1]))
    if not -32768 <= q <= 32767 or not -32768 <= r <= 32767:
        raise ValueError(f"coordinate does not fit the wire ActionId: {(q, r)}")
    return ((((q & 0xFFFF) ^ 0x8000) << 16) | ((r & 0xFFFF) ^ 0x8000))


def move_from_wire(action: int) -> tuple[int, int]:
    if not _is_uint(action, _U32_MAX):
        raise ValueError(f"action is not a U32: {action!r}")
    q = ((action >> 16) ^ 0x8000) & 0xFFFF
    r = ((action & 0xFFFF) ^ 0x8000) & 0xFFFF
    return (
        q - 0x10000 if q & 0x8000 else q,
        r - 0x10000 if r & 0x8000 else r,
    )


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"repeated JSON member {key!r}")
        obj[key] = value
    return obj


def _invalid_constant(value: str) -> Any:
    raise ValueError(f"{value} is not a JSON number")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{value} is outside the finite JSON number range")
    return parsed


def _decode_json(text: str) -> Any:
    return json.loads(
        text,
        object_pairs_hook=_strict_object,
        parse_constant=_invalid_constant,
        parse_float=_finite_float,
    )


def _members(
    obj: Any,
    required: Iterable[str],
    optional: Iterable[str] = (),
    *,
    what: str,
) -> None:
    if type(obj) is not dict:
        raise ValueError(f"{what} must be an object")
    required_set, optional_set = set(required), set(optional)
    missing = required_set - set(obj)
    unknown = set(obj) - required_set - optional_set
    if missing:
        raise ValueError(f"{what} is missing {sorted(missing)}")
    if unknown:
        raise ValueError(f"{what} has unknown members {sorted(unknown)}")


def _validate_refusal(response: Any) -> None:
    _members(
        response,
        ("type", "message", "cause"),
        ("slot",),
        what="refuse response",
    )
    if response["type"] != "refuse" or not isinstance(response["message"], str):
        raise ValueError("refuse type/message is invalid")
    if "slot" in response and not _is_uint(response["slot"], _U64_MAX):
        raise ValueError("refuse.slot is not a U64")
    cause = response["cause"]
    _members(cause, ("code", "detail"), ("expected", "got"), what="refuse.cause")
    if not isinstance(cause["code"], str) or not isinstance(cause["detail"], str):
        raise ValueError("refuse cause code/detail must be strings")
    if ("expected" in cause) != ("got" in cause):
        raise ValueError("refuse cause expected/got must occur together")


def validate_welcome(response: Any, participant: Participant) -> None:
    _members(
        response,
        ("type", "name", "version", "resolved_variant", "digest"),
        ("encoder_version", "restriction"),
        what="welcome response",
    )
    if response["type"] != "welcome":
        raise ValueError(f"expected welcome, got {response.get('type')!r}")
    if not isinstance(response["name"], str) or not response["name"]:
        raise ValueError("welcome.name must be a nonempty string")
    if not _is_uint(response["version"], _U32_MAX):
        raise ValueError("welcome.version must be a U32")
    if "encoder_version" in response and not _is_uint(
        response["encoder_version"], _U32_MAX
    ):
        raise ValueError("welcome.encoder_version must be a U32")
    if response["resolved_variant"] != participant.variant:
        raise ValueError(
            "welcome.resolved_variant does not equal hello.variant: "
            f"{response['resolved_variant']!r} != {participant.variant!r}"
        )
    if not _is_hash(response["digest"]):
        raise ValueError("welcome.digest is not a fixed-width lowercase HASH")
    if "restriction" in response and not isinstance(response["restriction"], str):
        raise ValueError("welcome.restriction must be a string")


def validate_ok(response: Any, message: str) -> None:
    _members(response, ("type", "message"), what=f"ok({message}) response")
    if response != {"type": "ok", "message": message}:
        raise ValueError(f"expected ok({message}), got {response!r}")


def validate_decided(
    response: Any, requested_slots: Sequence[int]
) -> list[dict[str, Any]]:
    _members(response, ("type", "decisions"), what="decided response")
    if response["type"] != "decided" or type(response["decisions"]) is not list:
        raise ValueError("expected decided with a decisions array")
    decisions = response["decisions"]
    if len(decisions) != len(requested_slots):
        raise ValueError(
            f"decided returned {len(decisions)} decisions for "
            f"{len(requested_slots)} slots"
        )
    for index, (decision, slot) in enumerate(
        zip(decisions, requested_slots, strict=True)
    ):
        _members(
            decision,
            ("slot", "action", "zobrist", "diagnostics"),
            what=f"decision {index}",
        )
        if decision["slot"] != slot or not _is_uint(decision["slot"], _U64_MAX):
            raise ValueError(
                f"decision {index} answered slot {decision['slot']!r}, expected {slot}"
            )
        if not _is_uint(decision["action"], _U32_MAX):
            raise ValueError(f"decision {index} action is not a U32")
        if not _is_hash(decision["zobrist"]):
            raise ValueError(f"decision {index} zobrist is not a HASH")
        diagnostics = decision["diagnostics"]
        if diagnostics is not None and (
            type(diagnostics) is not list
            or any(not _is_uint(byte, 255) for byte in diagnostics)
        ):
            raise ValueError(f"decision {index} diagnostics is not null or bytes")
    return decisions


class Seat:
    """One strict request/response JSON-lines child connection."""

    def __init__(self, participant: Participant):
        self.participant = participant
        self.identity: dict[str, Any] | None = None
        self.hello_message = participant.hello()
        self.open_slots: set[int] = set()
        self._pending: tuple[str, tuple[int, ...], str] | None = None
        self._stderr: collections.deque[str] = collections.deque(
            maxlen=_STDERR_CHUNKS
        )
        try:
            self.process = subprocess.Popen(
                participant.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
            )
        except OSError as error:
            raise SeatError(
                f"{self._label()} could not launch: {error}"
            ) from error
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        assert self.process.stderr is not None
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            name=f"crossplay-stderr-{participant.id}",
            daemon=True,
        )
        self._stderr_thread.start()

    def _label(self) -> str:
        if self.identity is None:
            return (
                f"participant {self.participant.id!r} "
                f"(command={list(self.participant.command)!r})"
            )
        return (
            f"participant {self.participant.id!r} "
            f"(name={self.identity['name']!r}, digest={self.identity['digest']!r})"
        )

    def _drain_stderr(self) -> None:
        assert self.process.stderr is not None
        while True:
            try:
                chunk = self.process.stderr.read(4096)
            except (OSError, ValueError):
                return
            if not chunk:
                return
            self._stderr.append(chunk.decode("utf-8", errors="replace"))

    def _diagnostic(self) -> str:
        text = "".join(self._stderr).strip()
        return f"; stderr: {text[-8192:]}" if text else ""

    def fail(self, detail: str, context: str) -> None:
        raise SeatError(f"{self._label()} failed during {context}: {detail}")

    @staticmethod
    def _request_slots(message: Mapping[str, Any]) -> tuple[int, ...]:
        if message["type"] in ("open", "decide"):
            return tuple(entry["slot"] for entry in message["slots"])
        if message["type"] == "close":
            return tuple(message["slots"])
        return ()

    def send(self, message: dict[str, Any], context: str) -> None:
        if self._pending is not None:
            self.fail("a request is already awaiting its response", context)
        code = self.process.poll()
        if code is not None:
            self._stderr_thread.join(timeout=0.2)
            self.fail(
                f"subprocess exited with code {code}{self._diagnostic()}",
                context,
            )
        try:
            line = (
                json.dumps(message, separators=(",", ":"), allow_nan=False)
                + "\n"
            ).encode("utf-8")
            assert self.process.stdin is not None
            self.process.stdin.write(line)
            self.process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as error:
            code = self.process.poll()
            self.fail(
                f"could not write request ({error}); exit code {code}"
                f"{self._diagnostic()}",
                context,
            )
        self._pending = (
            str(message["type"]),
            self._request_slots(message),
            context,
        )

    def receive(self) -> dict[str, Any]:
        if self._pending is None:
            self.fail("no request is awaiting a response", "protocol read")
        message, requested_slots, context = self._pending
        try:
            assert self.process.stdout is not None
            wire = self.process.stdout.readline()
        except OSError as error:
            self.fail(f"could not read response: {error}", context)
        if not wire:
            code = self.process.poll()
            if code is None:
                try:
                    code = self.process.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    pass
            if code is not None:
                self._stderr_thread.join(timeout=0.2)
            self.fail(
                f"stdout closed; exit code {code}{self._diagnostic()}",
                context,
            )
        if not wire.endswith(b"\n"):
            self.fail(
                f"response ended without a newline{self._diagnostic()}",
                context,
            )
        try:
            response = _decode_json(wire[:-1].decode("utf-8"))
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
            self.fail(f"malformed JSON response: {error}", context)
        if type(response) is not dict:
            self.fail("response is not a JSON object", context)
        self._pending = None
        if response.get("type") != "refuse":
            return response
        try:
            _validate_refusal(response)
        except ValueError as error:
            self.fail(str(error), context)
        if response["message"] != message:
            self.fail(
                f"refusal names {response['message']!r}, expected {message!r}",
                context,
            )
        if "slot" not in response:
            self.fail(
                "connection refusal " + json.dumps(response, ensure_ascii=False),
                context,
            )
        if response["slot"] not in requested_slots:
            self.fail(
                f"refusal names slot {response['slot']}, which was not requested",
                context,
            )
        return response

    def exchange(self, message: dict[str, Any], context: str) -> dict[str, Any]:
        self.send(message, context)
        return self.receive()

    def invalid(self, error: ValueError, context: str) -> None:
        self.fail(f"invalid response: {error}", context)

    def abort(self) -> None:
        try:
            if self.process.stdin is not None:
                self.process.stdin.close()
        except (OSError, ValueError):
            pass
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=1.0)
        for stream in (self.process.stdout, self.process.stderr):
            try:
                if stream is not None:
                    stream.close()
            except (OSError, ValueError):
                pass
        self._stderr_thread.join(timeout=1.0)

    def finish(self) -> None:
        try:
            code = self.process.wait(timeout=_SHUTDOWN_SECONDS)
        except subprocess.TimeoutExpired:
            self.fail("did not exit after ok(bye)", "shutdown")
        if code != 0:
            self._stderr_thread.join(timeout=0.2)
            self.fail(
                f"exited with code {code} after ok(bye){self._diagnostic()}",
                "shutdown",
            )
        assert self.process.stdout is not None
        extra = self.process.stdout.read()
        if extra:
            self.fail(f"unsolicited stdout after ok(bye): {extra!r}", "shutdown")
        self.process.stdin.close()
        self.process.stdout.close()
        self.process.stderr.close()
        self._stderr_thread.join(timeout=1.0)


def load_participants(path: Path, *, minimum: int = 2) -> list[Participant]:
    """Read the strict participant-list format used by the CLI."""
    if type(minimum) is not int or minimum < 1:
        raise ValueError(f"minimum participants must be >= 1, got {minimum!r}")
    path = Path(path)
    try:
        raw = _decode_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read participants from {path}: {error}") from error
    if type(raw) is not list or len(raw) < minimum:
        count = "two" if minimum == 2 else str(minimum)
        raise ValueError(
            f"participants file must be an array of at least {count} entries"
        )
    participants: list[Participant] = []
    for index, entry in enumerate(raw):
        _members(entry, ("id", "command", "hello"), what=f"participant {index}")
        if type(entry["command"]) is not list:
            raise ValueError(f"participant {index}.command must be an argv array")
        _members(
            entry["hello"],
            ("checkpoint", "variant"),
            what=f"participant {index}.hello",
        )
        participants.append(
            Participant(
                id=entry["id"],
                command=tuple(entry["command"]),
                checkpoint=entry["hello"]["checkpoint"],
                variant=entry["hello"]["variant"],
            )
        )
    ids = [participant.id for participant in participants]
    duplicates = sorted(
        participant_id
        for participant_id, count in collections.Counter(ids).items()
        if count > 1
    )
    if duplicates:
        raise ValueError(f"participant ids must be unique: {duplicates}")
    return participants
