"""The subprocess-seat round robin, protocol failures, and logistic ratings."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import hexo_py
import pytest

from mantisnet.klent.crossplay import (
    Participant,
    SeatError,
    cross_play,
    fit_bradley_terry,
    main,
)
from mantisnet.klent.headtohead import paired_statistics


_SCRIPTED_SEAT = r'''
import argparse
import json
import os
import sys

import hexo_py


def signed(value):
    return value - 0x10000 if value & 0x8000 else value


def decode(action):
    return (
        signed((((action >> 16) & 0xffff) ^ 0x8000)),
        signed(((action & 0xffff) ^ 0x8000)),
    )


def encode(move):
    q, r = move
    return ((((q & 0xffff) ^ 0x8000) << 16) | ((r & 0xffff) ^ 0x8000))


def wire_hash(position):
    return f"0x{position.zobrist:016x}"


parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--version", required=True, type=int)
parser.add_argument("--rule", choices=("first", "middle", "last"), required=True)
parser.add_argument("--digest", required=True)
parser.add_argument("--log", required=True)
parser.add_argument("--restriction")
parser.add_argument("--refuse-once", action="store_true")
parser.add_argument("--fault-once", action="store_true")
parser.add_argument("--die-on-decide", action="store_true")
parser.add_argument(
    "--mismatch",
    choices=("protocol_version", "rules_version", "action_order_version"),
)
args = parser.parse_args()

log_handle = open(args.log, "a", encoding="utf-8")


def log(direction, message):
    log_handle.write(
        json.dumps(
            {"direction": direction, "message": message},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    )
    log_handle.flush()


def send(message):
    log("out", message)
    wire = (
        json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    sys.stdout.buffer.write(wire)
    sys.stdout.buffer.flush()


hello_wire = sys.stdin.buffer.readline()
if not hello_wire:
    raise SystemExit(0)
hello = json.loads(hello_wire)
log("in", hello)
assert set(hello) == {
    "type",
    "protocol_version",
    "rules_version",
    "action_order_version",
    "checkpoint",
    "variant",
}
assert hello["type"] == "hello"
assert hello["checkpoint"] and hello["variant"]

versions = {
    "protocol_version": hexo_py.PROTOCOL_VERSION,
    "rules_version": hexo_py.RULES_VERSION,
    "action_order_version": hexo_py.ACTION_ORDER_VERSION,
}
if args.mismatch is not None:
    expected = versions[args.mismatch] + 1
    send(
        {
            "type": "refuse",
            "message": "hello",
            "cause": {
                "code": args.mismatch,
                "detail": f"script expects {args.mismatch} {expected}",
                "expected": expected,
                "got": hello[args.mismatch],
            },
        }
    )
    raise SystemExit(1)
assert all(hello[field] == value for field, value in versions.items())

welcome = {
    "type": "welcome",
    "name": args.name,
    "version": args.version,
    "resolved_variant": hello["variant"],
    "digest": args.digest,
}
if args.version % 2:
    welcome["encoder_version"] = args.version + 100
if args.restriction is not None:
    welcome["restriction"] = args.restriction
send(welcome)

slots = {}
refused = False
while True:
    wire = sys.stdin.buffer.readline()
    if not wire:
        break
    message = json.loads(wire)
    log("in", message)
    kind = message["type"]
    if kind == "open":
        staged = {}
        for entry in message["slots"]:
            assert entry["slot"] not in slots and entry["slot"] not in staged
            position = hexo_py.Position.replay(
                [decode(action) for action in entry["opening"]]
            )
            assert not position.is_terminal
            staged[entry["slot"]] = {
                "position": position,
                "side": 0 if entry["side"] == "p0" else 1,
            }
        slots.update(staged)
        send({"type": "ok", "message": "open"})
    elif kind == "decide":
        if args.die_on_decide:
            sys.stderr.write("scripted seat died during decide\n")
            sys.stderr.flush()
            os._exit(23)
        if (args.refuse_once or args.fault_once) and not refused:
            refused = True
            retired = message["slots"][0]["slot"]
            slots.pop(retired)
            code = "variant" if args.fault_once else "restriction_exhausted"
            detail = (
                "scripted participant fault"
                if args.fault_once
                else "scripted restriction exhausted: π unavailable"
            )
            send(
                {
                    "type": "refuse",
                    "message": "decide",
                    "slot": retired,
                    "cause": {
                        "code": code,
                        "detail": detail,
                        "expected": {"available": True},
                        "got": {"available": False},
                    },
                }
            )
            continue
        staged = []
        for entry in message["slots"]:
            state = slots[entry["slot"]]
            position = state["position"].copy()
            for action in entry["moves"]:
                position.advance(*decode(action))
            assert wire_hash(position) == entry["zobrist"]
            assert not position.is_terminal
            assert position.current_player == state["side"]
            staged.append((entry["slot"], position))
        decisions = []
        for slot, position in staged:
            if args.rule == "first":
                rank = 0
            elif args.rule == "middle":
                rank = position.legal_count // 2
            else:
                rank = position.legal_count - 1
            decisions.append(
                {
                    "slot": slot,
                    "action": encode(position.nth_legal(rank)),
                    "zobrist": wire_hash(position),
                    "diagnostics": [rank & 0xff],
                }
            )
        for slot, position in staged:
            slots[slot]["position"] = position
        send({"type": "decided", "decisions": decisions})
    elif kind == "close":
        assert len(message["slots"]) == len(set(message["slots"]))
        assert all(slot in slots for slot in message["slots"])
        for slot in message["slots"]:
            slots.pop(slot)
        send({"type": "ok", "message": "close"})
    elif kind == "bye":
        assert not slots
        send({"type": "ok", "message": "bye"})
        break
    else:
        raise AssertionError(f"unexpected message {message}")

log_handle.close()
'''


class _SeatFactory:
    def __init__(self, root: Path):
        self.script = root / "scripted seat.py"
        self.script.write_text(_SCRIPTED_SEAT, encoding="utf-8")
        self.logs: dict[str, Path] = {}

    def participant(
        self,
        participant: str,
        *,
        rule: str,
        version: int,
        restriction: str | None = None,
        refuse_once: bool = False,
        fault_once: bool = False,
        die_on_decide: bool = False,
        mismatch: str | None = None,
    ) -> Participant:
        log = self.script.parent / f"{participant}.jsonl"
        self.logs[participant] = log
        command = [
            sys.executable,
            "-u",
            str(self.script),
            "--name",
            f"engine-{participant}",
            "--version",
            str(version),
            "--rule",
            rule,
            "--digest",
            f"0x{version:016x}",
            "--log",
            str(log),
        ]
        if restriction is not None:
            command.extend(("--restriction", restriction))
        if refuse_once:
            command.append("--refuse-once")
        if fault_once:
            command.append("--fault-once")
        if die_on_decide:
            command.append("--die-on-decide")
        if mismatch is not None:
            command.extend(("--mismatch", mismatch))
        return Participant(
            id=participant,
            command=tuple(command),
            checkpoint=f"checkpoint-{participant}",
            variant=f"variant-{participant}",
        )

    def messages(self, participant: str, direction: str = "in") -> list[dict]:
        return [
            row["message"]
            for row in map(
                json.loads,
                self.logs[participant].read_text(encoding="utf-8").splitlines(),
            )
            if row["direction"] == direction
        ]


@pytest.fixture
def seat_factory(tmp_path) -> _SeatFactory:
    return _SeatFactory(tmp_path)


def _decode(action: int) -> tuple[int, int]:
    def signed(value):
        return value - 0x10000 if value & 0x8000 else value

    return (
        signed((((action >> 16) & 0xFFFF) ^ 0x8000)),
        signed(((action & 0xFFFF) ^ 0x8000)),
    )


def _restriction(game: dict, participant: str) -> str | None:
    for seat in game["seats"].values():
        if seat["id"] == participant:
            return seat["welcome"].get("restriction")
    raise AssertionError(f"{participant} did not appear in game {game['game']}")


def test_three_participant_round_robin_batches_every_waiting_game(seat_factory):
    participants = [
        seat_factory.participant(
            "first",
            rule="first",
            version=1,
            restriction="canonical legal rank zero only",
        ),
        seat_factory.participant("middle", rule="middle", version=2),
        seat_factory.participant(
            "last",
            rule="last",
            version=3,
            restriction="canonical final rank only",
        ),
    ]

    result = cross_play(
        participants,
        pairs=2,
        anchors={"middle": 0.0},
        ply_cap=8,
        seed=9,
        opening_range=(2, 2),
    )

    assert len(result["pairings"]) == 3
    assert sum(len(pairing["games"]) for pairing in result["pairings"]) == 12
    assert [row["welcome"]["name"] for row in result["participants"]] == [
        "engine-first",
        "engine-middle",
        "engine-last",
    ]
    assert [row["welcome"]["version"] for row in result["participants"]] == [1, 2, 3]
    assert [
        row["welcome"].get("encoder_version") for row in result["participants"]
    ] == [101, None, 103]
    for pairing in result["pairings"]:
        assert pairing["statistics"] == paired_statistics(
            [
                pairing["games"][index : index + 2]
                for index in range(0, len(pairing["games"]), 2)
            ]
        )
        assert pairing["statistics"]["score"] == 0.5
        assert pairing["statistics"]["capped"] == 4
        assert [row["seats"] for row in pairing["pairs"]] == [[0, 1], [0, 1]]
        assert pairing["pairs"][0]["opening"] == pairing["games"][0]["opening"]
        assert pairing["games"][0]["opening"] == pairing["games"][1]["opening"]
        assert pairing["games"][2]["opening"] == pairing["games"][3]["opening"]
    for participant in ("first", "middle", "last"):
        for opponent in ("first", "middle", "last"):
            cell = result["matrix"][participant][opponent]
            if participant == opponent:
                assert cell is None
            else:
                assert cell["games"] == 4
                assert cell["score"] == 0.5
                assert cell["draws"] == cell["capped"] == 4

    # Reconstruct the initial waiting set from the seat's own open messages.
    # The first decide must be one request containing the entire set.
    for participant in ("first", "middle", "last"):
        messages = seat_factory.messages(participant)
        opened = [
            entry
            for message in messages
            if message["type"] == "open"
            for entry in message["slots"]
        ]
        waiting = set()
        for entry in opened:
            position = hexo_py.Position.replay(
                [_decode(action) for action in entry["opening"]]
            )
            side = 0 if entry["side"] == "p0" else 1
            if position.current_player == side:
                waiting.add(entry["slot"])
        decides = [message for message in messages if message["type"] == "decide"]
        assert len(waiting) == 4
        assert {entry["slot"] for entry in decides[0]["slots"]} == waiting
        assert all(entry["moves"] == [] for entry in decides[0]["slots"])
        assert all(len(message["slots"]) == 4 for message in decides)

    for pairing in result["pairings"]:
        for game in pairing["games"]:
            for participant, expected in (
                ("first", "canonical legal rank zero only"),
                ("last", "canonical final rank only"),
            ):
                if participant in (pairing["a"], pairing["b"]):
                    assert _restriction(game, participant) == expected


def test_slot_refusal_forfeits_once_and_preserves_the_cause(seat_factory):
    restriction = "rank zero until the scripted evaluator is unavailable"
    peer_restriction = "canonical final rank only"
    participants = [
        seat_factory.participant(
            "refuser",
            rule="first",
            version=5,
            restriction=restriction,
            refuse_once=True,
        ),
        seat_factory.participant(
            "peer",
            rule="last",
            version=6,
            restriction=peer_restriction,
        ),
    ]

    result = cross_play(
        participants,
        pairs=2,
        anchors={"peer": 0.0},
        ply_cap=8,
        opening_range=(2, 2),
    )
    pairing = result["pairings"][0]
    refusals = [
        game
        for game in pairing["games"]
        if game["adjudication"]["type"] == "seat_refusal"
    ]
    assert len(refusals) == 1
    refused = refusals[0]
    assert refused["winner"] == "peer"
    assert refused["score_a"] == 0.0
    assert refused["capped"] is False
    assert refused["adjudication"] == {
        "type": "seat_refusal",
        "loser": "refuser",
        "response": {
            "type": "refuse",
            "message": "decide",
            "slot": refused["game"],
            "cause": {
                "code": "restriction_exhausted",
                "detail": "scripted restriction exhausted: π unavailable",
                "expected": {"available": True},
                "got": {"available": False},
            },
        },
    }
    assert pairing["restrictions"] == {
        "a": restriction,
        "b": peer_restriction,
    }
    assert all(_restriction(game, "refuser") == restriction for game in pairing["games"])
    assert all(
        _restriction(game, "peer") == peer_restriction for game in pairing["games"]
    )

    messages = seat_factory.messages("refuser")
    decides = [message for message in messages if message["type"] == "decide"]
    first_slots = {
        entry["slot"]: entry for entry in decides[0]["slots"]
    }
    assert refused["game"] in first_slots
    survivor = next(slot for slot in first_slots if slot != refused["game"])
    next_entry = next(
        entry for entry in decides[1]["slots"] if entry["slot"] == survivor
    )
    assert next_entry == first_slots[survivor]
    assert all(
        refused["game"] not in message["slots"]
        for message in messages
        if message["type"] == "close"
    )


def test_slot_participant_fault_fails_the_pairing_loudly(seat_factory):
    participants = [
        seat_factory.participant(
            "faulty",
            rule="first",
            version=7,
            fault_once=True,
        ),
        seat_factory.participant("peer", rule="last", version=8),
    ]

    with pytest.raises(SeatError) as raised:
        cross_play(
            participants,
            pairs=1,
            anchors={"peer": 0.0},
            ply_cap=8,
            opening_range=(2, 2),
        )

    message = str(raised.value)
    assert "participant 'faulty'" in message
    assert "decide" in message and "faulty vs peer" in message
    assert "participant fault" in message
    assert '"code": "variant"' in message


def test_undeclared_restriction_exhaustion_fails_the_pairing_loudly(
    seat_factory,
):
    participants = [
        seat_factory.participant(
            "undeclared",
            rule="first",
            version=7,
            refuse_once=True,
        ),
        seat_factory.participant("peer", rule="last", version=8),
    ]

    with pytest.raises(SeatError) as raised:
        cross_play(
            participants,
            pairs=1,
            anchors={"peer": 0.0},
            ply_cap=8,
            opening_range=(2, 2),
        )

    message = str(raised.value)
    assert "participant 'undeclared'" in message
    assert "participant fault" in message
    assert '"code": "restriction_exhausted"' in message


def test_dead_seat_fails_the_pairing_loudly(seat_factory):
    participants = [
        seat_factory.participant("healthy", rule="first", version=7),
        seat_factory.participant(
            "dead",
            rule="last",
            version=8,
            die_on_decide=True,
        ),
    ]

    with pytest.raises(SeatError) as raised:
        cross_play(
            participants,
            pairs=1,
            anchors={"healthy": 0.0},
            ply_cap=8,
            opening_range=(2, 2),
        )

    message = str(raised.value)
    assert "participant 'dead'" in message
    assert "decide" in message and "healthy vs dead" in message
    assert "23" in message
    assert "scripted seat died during decide" in message


@pytest.mark.parametrize(
    "field",
    ("protocol_version", "rules_version", "action_order_version"),
)
def test_version_mismatch_is_a_connection_refusal(seat_factory, field):
    participants = [
        seat_factory.participant(
            "mismatch",
            rule="first",
            version=9,
            mismatch=field,
        ),
        seat_factory.participant("peer", rule="last", version=10),
    ]

    with pytest.raises(SeatError) as raised:
        cross_play(
            participants,
            pairs=1,
            anchors={"peer": 0.0},
            ply_cap=8,
            opening_range=(2, 2),
        )

    message = str(raised.value)
    assert "participant 'mismatch'" in message
    assert "hello" in message and "connection refusal" in message
    assert f'"code": "{field}"' in message
    inbound = seat_factory.messages("mismatch")
    assert [message["type"] for message in inbound] == ["hello"]
    hello = inbound[0]
    assert hello["protocol_version"] == hexo_py.PROTOCOL_VERSION
    assert hello["rules_version"] == hexo_py.RULES_VERSION
    assert hello["action_order_version"] == hexo_py.ACTION_ORDER_VERSION


def test_cli_reads_a_participant_list_and_atomically_writes_strict_json(
    seat_factory, tmp_path
):
    participants = [
        seat_factory.participant("one", rule="first", version=11),
        seat_factory.participant("two", rule="last", version=12),
    ]
    configured = [
        {
            "id": participant.id,
            "command": list(participant.command),
            "hello": {
                "checkpoint": participant.checkpoint,
                "variant": participant.variant,
            },
        }
        for participant in participants
    ]
    participant_path = tmp_path / "participants.json"
    participant_path.write_text(json.dumps(configured), encoding="utf-8")
    # An output already ending in .tmp exercises the guarantee that the sibling
    # staging path is still distinct from the final destination.
    output = tmp_path / "result.tmp"

    main(
        [
            "--participants",
            str(participant_path),
            "--pairs",
            "1",
            "--anchor",
            "one=0",
            "--cap",
            "8",
            "--opening-range",
            "2",
            "2",
            "--out",
            str(output),
        ]
    )

    text = output.read_text(encoding="utf-8")
    assert "Infinity" not in text and "NaN" not in text
    assert [row["id"] for row in json.loads(text)["participants"]] == ["one", "two"]
    assert not (tmp_path / "result.tmp.tmp").exists()


def test_bradley_terry_recovers_a_known_ordering():
    result = fit_bradley_terry(
        ("high", "anchor", "low"),
        [
            {"a": "high", "b": "anchor", "games": 20, "score_a": 0.75},
            {"a": "high", "b": "low", "games": 20, "score_a": 0.90},
            {"a": "anchor", "b": "low", "games": 20, "score_a": 0.75},
        ],
        {"anchor": 0.0},
    )

    assert result["connected"] is True
    assert result["ratings"]["high"]["rating"] == pytest.approx(math.log(3))
    assert result["ratings"]["anchor"] == {
        "rating": 0.0,
        "standard_error": 0.0,
        "fixed": True,
        "status": "fixed",
    }
    assert result["ratings"]["low"]["rating"] == pytest.approx(-math.log(3))
    for participant in ("high", "low"):
        assert math.isfinite(result["ratings"][participant]["standard_error"])
        assert result["ratings"][participant]["standard_error"] > 0
    assert result["warnings"] == []


def test_bradley_terry_disconnects_an_unanchored_component():
    result = fit_bradley_terry(
        ("a", "b", "c", "d"),
        [
            {"a": "a", "b": "b", "games": 10, "score_a": 0.5},
            {"a": "c", "b": "d", "games": 10, "score_a": 0.5},
        ],
        {"a": 0.0},
    )

    assert result["connected"] is False
    assert result["ratings"]["a"]["rating"] == 0.0
    assert result["ratings"]["b"]["rating"] == pytest.approx(0.0)
    for participant in ("c", "d"):
        assert result["ratings"][participant] == {
            "rating": None,
            "standard_error": None,
            "fixed": False,
            "status": "disconnected",
        }
    assert any("disconnected" in warning for warning in result["warnings"])
    assert any("'c'" in warning and "'d'" in warning for warning in result["warnings"])


def test_bradley_terry_reports_all_win_and_all_loss_as_unbounded():
    result = fit_bradley_terry(
        ("high", "anchor", "low"),
        [
            {"a": "high", "b": "anchor", "games": 10, "score_a": 1.0},
            {"a": "anchor", "b": "low", "games": 10, "score_a": 1.0},
        ],
        {"anchor": 0.0},
    )

    assert result["ratings"]["high"] == {
        "rating": None,
        "standard_error": None,
        "fixed": False,
        "status": "unbounded_above",
    }
    assert result["ratings"]["low"] == {
        "rating": None,
        "standard_error": None,
        "fixed": False,
        "status": "unbounded_below",
    }
    assert result["ratings"]["anchor"]["rating"] == 0.0
    assert any("won every" in warning for warning in result["warnings"])
    assert any("lost every" in warning for warning in result["warnings"])
