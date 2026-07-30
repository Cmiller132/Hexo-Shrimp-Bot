"""A native subprocess seat through the generic evaluation-opponent seam."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from mantisnet.klent.opponents import SeatOpponent, opponent_match
from mantisnet.klent.seat import Participant, SeatError


_DIGEST = "0x1020304050607080"
_VARIANT = "scripted-rank-zero"

_SCRIPTED_SEAT = r'''
import argparse
import json
import sys

import hexo_py


parser = argparse.ArgumentParser()
parser.add_argument("--log", required=True)
parser.add_argument("--name", required=True)
parser.add_argument("--digest", required=True)
parser.add_argument("--restriction")
parser.add_argument("--refuse-once", action="store_true")
parser.add_argument("--die-on-decide", action="store_true")
parser.add_argument("--illegal-once", action="store_true")
args = parser.parse_args()

# Far outside the growth radius of any short game, so it is illegal wherever
# the match has reached.
FAR_CELL = (60, 60)

log_handle = open(args.log, "w", encoding="utf-8", buffering=1)
slots = {}
refused = False
played_illegal = False


def record(direction, message):
    log_handle.write(
        json.dumps(
            {"direction": direction, "message": message},
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    )
    log_handle.flush()


def send(message):
    record("out", message)
    sys.stdout.write(
        json.dumps(message, separators=(",", ":"), ensure_ascii=False) + "\n"
    )
    sys.stdout.flush()


def decode(action):
    q = ((action >> 16) ^ 0x8000) & 0xFFFF
    r = ((action & 0xFFFF) ^ 0x8000) & 0xFFFF
    return (
        q - 0x10000 if q & 0x8000 else q,
        r - 0x10000 if r & 0x8000 else r,
    )


def encode(move):
    q, r = move
    return ((((q & 0xFFFF) ^ 0x8000) << 16) | ((r & 0xFFFF) ^ 0x8000))


def wire_hash(position):
    return f"0x{position.zobrist:016x}"


for line in sys.stdin:
    message = json.loads(line)
    record("in", message)
    kind = message["type"]
    if kind == "hello":
        assert not slots
        welcome = {
            "type": "welcome",
            "name": args.name,
            "version": 17,
            "resolved_variant": message["variant"],
            "digest": args.digest,
        }
        if args.restriction is not None:
            welcome["restriction"] = args.restriction
        send(welcome)
    elif kind == "open":
        staged = []
        for entry in message["slots"]:
            assert entry["slot"] not in slots
            position = hexo_py.Position.replay(
                [decode(action) for action in entry["opening"]]
            )
            staged.append(
                (
                    entry["slot"],
                    {
                        "position": position,
                        "side": 0 if entry["side"] == "p0" else 1,
                    },
                )
            )
        slots.update(staged)
        send({"type": "ok", "message": "open"})
    elif kind == "decide":
        if args.die_on_decide:
            sys.stderr.write("scripted seat exited during decide\n")
            sys.stderr.flush()
            sys.exit(23)
        if args.refuse_once and not refused:
            refused = True
            slot = message["slots"][0]["slot"]
            slots.pop(slot)
            send(
                {
                    "type": "refuse",
                    "message": "decide",
                    "slot": slot,
                    "cause": {
                        "code": "restriction_exhausted",
                        "detail": "scripted rank-zero restriction exhausted",
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
        decisions = [
            {
                "slot": slot,
                "action": encode(position.nth_legal(0)),
                "zobrist": wire_hash(position),
                "diagnostics": [slot & 0xFF],
            }
            for slot, position in staged
        ]
        if args.illegal_once and not played_illegal:
            played_illegal = True
            # A correct attestation with an unplayable action: the orchestrator
            # adjudicates legality, so only that check can catch this.
            decisions[0]["action"] = encode(FAR_CELL)
        for slot, position in staged:
            slots[slot]["position"] = position
        send({"type": "decided", "decisions": decisions})
    elif kind == "close":
        assert len(message["slots"]) == len(set(message["slots"]))
        for slot in message["slots"]:
            slots.pop(slot)
        send({"type": "ok", "message": "close"})
    elif kind == "bye":
        assert not slots
        send({"type": "ok", "message": "bye"})
        break
    else:
        raise AssertionError(f"unexpected message: {message}")

log_handle.close()
'''


class _SeatFactory:
    def __init__(self, root: Path):
        self.script = root / "scripted evaluation seat.py"
        self.script.write_text(_SCRIPTED_SEAT, encoding="utf-8")
        self.logs: dict[str, Path] = {}

    def participant(
        self,
        participant_id: str,
        *,
        restriction: str | None = None,
        refuse_once: bool = False,
        die_on_decide: bool = False,
        illegal_once: bool = False,
    ) -> Participant:
        log = self.script.parent / f"{participant_id}.jsonl"
        self.logs[participant_id] = log
        command = [
            sys.executable,
            "-u",
            str(self.script),
            "--log",
            str(log),
            "--name",
            "scripted-native-seat",
            "--digest",
            _DIGEST,
        ]
        if restriction is not None:
            command.extend(("--restriction", restriction))
        if refuse_once:
            command.append("--refuse-once")
        if die_on_decide:
            command.append("--die-on-decide")
        if illegal_once:
            command.append("--illegal-once")
        return Participant(
            id=participant_id,
            command=tuple(command),
            checkpoint="unused-scripted-checkpoint",
            variant=_VARIANT,
        )

    def messages(self, participant_id: str, direction: str) -> list[dict]:
        return [
            row["message"]
            for row in map(
                json.loads,
                self.logs[participant_id]
                .read_text(encoding="utf-8")
                .splitlines(),
            )
            if row["direction"] == direction
        ]


@pytest.fixture
def seat_factory(tmp_path: Path) -> _SeatFactory:
    return _SeatFactory(tmp_path)


def _choose_first(positions, _rng):
    return [position.nth_legal(0) for position in positions]


def test_full_match_batches_every_waiting_game_and_shuts_down(seat_factory):
    participant = seat_factory.participant("complete")
    opponent = SeatOpponent(participant)

    summary, games = opponent_match(
        _choose_first,
        opponent,
        games=4,
        ply_cap=8,
        rng=np.random.default_rng(29),
        opening_range=(2, 2),
    )

    assert summary["opponent_name"] == "scripted-native-seat"
    assert summary["opponent_config"] == {
        "resolved_variant": _VARIANT,
        "digest": _DIGEST,
    }
    assert opponent.name == summary["opponent_name"]
    assert opponent.config == summary["opponent_config"]
    assert summary["forfeits"] == 0
    assert len(games) == 4
    assert all(game["capped"] and not game["forfeit"] for game in games)

    received = seat_factory.messages("complete", "in")
    assert received[0]["type"] == "hello"
    assert received[1]["type"] == "open"
    assert received[2]["type"] == "decide"
    opens = [message for message in received if message["type"] == "open"]
    assert len(opens) == 1
    waiting = {entry["slot"] for entry in opens[0]["slots"]}
    assert len(waiting) == 4
    first_decide = received[2]
    assert {entry["slot"] for entry in first_decide["slots"]} == waiting

    closed = [
        slot
        for message in received
        if message["type"] == "close"
        for slot in message["slots"]
    ]
    assert sorted(closed) == sorted(waiting)
    assert received[-1] == {"type": "bye"}
    sent = seat_factory.messages("complete", "out")
    assert sent[0]["type"] == "welcome"
    assert sent[-1] == {"type": "ok", "message": "bye"}


def test_declared_restriction_forfeits_and_retry_preserves_cursors(
    seat_factory,
):
    participant = seat_factory.participant(
        "restricted",
        restriction="only canonical legal rank zero",
        refuse_once=True,
    )

    summary, games = opponent_match(
        _choose_first,
        SeatOpponent(participant),
        games=4,
        ply_cap=8,
        rng=np.random.default_rng(31),
        opening_range=(2, 2),
    )

    forfeited = [game for game in games if game["forfeit"]]
    assert summary["forfeits"] == len(forfeited) == 1
    assert forfeited[0]["score"] == 1.0
    assert forfeited[0]["winner"] == forfeited[0]["seat"]

    received = seat_factory.messages("restricted", "in")
    sent = seat_factory.messages("restricted", "out")
    refusal = next(message for message in sent if message["type"] == "refuse")
    assert refusal["cause"]["code"] == "restriction_exhausted"
    refused_slot = refusal["slot"]
    decides = [message for message in received if message["type"] == "decide"]
    initial_entries = {
        entry["slot"]: entry for entry in decides[0]["slots"]
    }
    retried_entries = {
        entry["slot"]: entry for entry in decides[1]["slots"]
    }
    assert refused_slot in initial_entries
    assert list(retried_entries) == [
        slot for slot in initial_entries if slot != refused_slot
    ]
    assert retried_entries == {
        slot: entry
        for slot, entry in initial_entries.items()
        if slot != refused_slot
    }

    closed = {
        slot
        for message in received
        if message["type"] == "close"
        for slot in message["slots"]
    }
    assert refused_slot not in closed
    assert received[-1] == {"type": "bye"}


def test_illegal_action_forfeits_its_game_and_closes_only_that_slot(
    seat_factory,
):
    participant = seat_factory.participant("unplayable", illegal_once=True)

    summary, games = opponent_match(
        _choose_first,
        SeatOpponent(participant),
        games=4,
        ply_cap=8,
        rng=np.random.default_rng(43),
        opening_range=(2, 2),
    )

    forfeited = [game for game in games if game["forfeit"]]
    assert summary["forfeits"] == len(forfeited) == 1
    assert forfeited[0]["score"] == 1.0
    assert forfeited[0]["winner"] == forfeited[0]["seat"]

    received = seat_factory.messages("unplayable", "in")
    sent = seat_factory.messages("unplayable", "out")
    decided = next(message for message in sent if message["type"] == "decided")
    unplayable_slot = decided["decisions"][0]["slot"]

    decides = [message for message in received if message["type"] == "decide"]
    assert unplayable_slot in {entry["slot"] for entry in decides[0]["slots"]}
    # The seat is not consulted about its own illegal action and is never asked
    # to decide that slot again; unlike a refusal, the slot is closed.
    assert len(decides[0]["slots"]) == 4
    assert all(
        unplayable_slot not in {entry["slot"] for entry in decide["slots"]}
        for decide in decides[1:]
    )
    closed = [
        slot
        for message in received
        if message["type"] == "close"
        for slot in message["slots"]
    ]
    assert unplayable_slot in closed
    assert sorted(closed) == sorted(
        entry["slot"] for entry in decides[0]["slots"]
    )
    assert received[-1] == {"type": "bye"}
    assert sent[-1] == {"type": "ok", "message": "bye"}


def test_undeclared_restriction_exhaustion_is_a_seat_fault(seat_factory):
    participant = seat_factory.participant(
        "undeclared",
        refuse_once=True,
    )

    with pytest.raises(SeatError) as raised:
        opponent_match(
            _choose_first,
            SeatOpponent(participant),
            games=2,
            ply_cap=8,
            rng=np.random.default_rng(37),
            opening_range=(2, 2),
        )

    message = str(raised.value)
    assert "participant 'undeclared'" in message
    assert "participant fault" in message
    assert "restriction_exhausted" in message
    received = seat_factory.messages("undeclared", "in")
    assert received[-1]["type"] == "decide"
    assert all(request["type"] != "bye" for request in received)


def test_dead_seat_raises_with_participant_identity(seat_factory):
    participant = seat_factory.participant(
        "dead-evaluator",
        die_on_decide=True,
    )

    with pytest.raises(SeatError) as raised:
        opponent_match(
            _choose_first,
            SeatOpponent(participant),
            games=2,
            ply_cap=8,
            rng=np.random.default_rng(41),
            opening_range=(2, 2),
        )

    message = str(raised.value)
    assert "participant 'dead-evaluator'" in message
    assert "decide evaluation slots" in message
    assert "exit code 23" in message
    assert "scripted seat exited during decide" in message
    received = seat_factory.messages("dead-evaluator", "in")
    assert received[-1]["type"] == "decide"
    assert all(request["type"] != "bye" for request in received)
