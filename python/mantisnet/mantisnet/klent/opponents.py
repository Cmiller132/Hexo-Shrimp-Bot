"""Evaluation-opponent contract, SealBot adapter, and native seat adapter.

An opponent provides an identity (``name`` and strength-defining ``config``)
and ``make_chooser(ply_cap)``. Both in-loop evaluation and offline sweeps use
this interface.

Chooser objects may expose per-game lifecycle hooks when they maintain
opponent state. The generic loop merely calls those hooks. SealBot uses them
for its independent ``HexGame`` oracle and alpha-beta instances; SeatOpponent
uses them to hold protocol slots and incremental move cursors.
"""

from __future__ import annotations

import importlib
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from .seat import (
    Participant,
    Seat,
    SeatError,
    move_from_wire,
    validate_decided,
    validate_ok,
    validate_welcome,
    wire_action,
    zobrist,
)

_COORD_LIMIT = 60
# MinimaxBot instances are scoped to a wave to bound transposition-table memory.
_SEALBOT_WAVE = 16
_loaded_variant: str | None = None


class Opponent(Protocol):
    """The complete extension seam for an evaluation opponent."""

    name: str
    config: dict

    def make_chooser(self, ply_cap: int):
        """Return ``choose(positions, rng) -> moves``."""


def load_sealbot(root: Path, variant: str = "current"):
    """Import ``(game module, MinimaxBot)`` from a SealBot checkout."""
    global _loaded_variant
    root = Path(root)
    bot_dir = root / variant
    if not (root / "game.py").exists():
        raise FileNotFoundError(f"{root} is not a SealBot checkout: no game.py")
    if not any(bot_dir.glob("minimax_cpp*.pyd")) and not any(
        bot_dir.glob("minimax_cpp*.so")
    ):
        raise FileNotFoundError(
            f"no built minimax_cpp in {bot_dir} — build it there with "
            "`python setup.py build_ext --inplace` (needs pybind11 + setuptools)"
        )
    if _loaded_variant is not None and _loaded_variant != str(bot_dir):
        raise RuntimeError(
            f"SealBot variant {_loaded_variant} is already loaded; "
            "C extensions load once per process"
        )
    for path in (str(bot_dir), str(root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    game_mod = importlib.import_module("game")
    minimax = importlib.import_module("minimax_cpp")
    _loaded_variant = str(bot_dir)
    return game_mod, minimax.MinimaxBot


def _mirror(game_mod, moves):
    """SealBot's ``HexGame`` holding exactly ``moves``."""
    game = game_mod.HexGame()
    for q, r in moves:
        if game.game_over or not game.make_move(int(q), int(r)):
            raise RuntimeError(
                f"rules mismatch: SealBot's HexGame refused move {(q, r)} "
                f"at placement {game.move_count}"
            )
    return game


# An unplayable opponent proposal ends the game as a recorded forfeit.
FORFEIT = object()


class _SealBotChooser:
    """Standard one-placement chooser around SealBot's whole-turn API."""

    max_games_per_wave = _SEALBOT_WAVE

    def __init__(self, game_mod, bot_type, time_limit, max_depth):
        self.game_mod = game_mod
        self.bot_type = bot_type
        self.time_limit = time_limit
        self.max_depth = max_depth
        self.games = {}
        self.retries = 0

    def start_game(self, position, moves):
        bot = self.bot_type(self.time_limit)
        if self.max_depth is not None:
            bot.max_depth = self.max_depth
        self.games[id(position)] = {
            "bot": bot,
            "moves": [tuple(move) for move in moves],
            "pending": [],
            "depths": [],
        }

    def before_move(self, _position, move):
        q, r = int(move[0]), int(move[1])
        if max(abs(q), abs(r)) > _COORD_LIMIT:
            raise RuntimeError(
                f"game left SealBot's coordinate range: {(q, r)} "
                f"exceeds ±{_COORD_LIMIT}"
            )

    def after_move(self, position, move):
        q, r = int(move[0]), int(move[1])
        self.games[id(position)]["moves"].append((q, r))

    def _consult(self, state, position):
        mirror = _mirror(self.game_mod, state["moves"])
        assert mirror.current_player.value - 1 == position.current_player
        assert mirror.moves_left_in_turn == position.moves_remaining
        turn = state["bot"].get_move(mirror)
        if not turn:
            raise RuntimeError("SealBot returned no moves for a live position")
        state["depths"].append(state["bot"].last_depth)
        return [(int(m[0]), int(m[1])) for m in turn]

    def __call__(self, positions, _rng):
        moves = []
        for position in positions:
            state = self.games[id(position)]
            if state["pending"]:
                move = state["pending"].pop(0)
            else:
                move, *state["pending"] = self._consult(state, position)
            legal = set(position.legal_moves())
            if move not in legal:
                # The second unusable proposal forfeits after one retry.
                self.retries += 1
                state["pending"].clear()
                move, *state["pending"] = self._consult(state, position)
            if move not in legal:
                state["bot"] = None  # Release the per-game transposition table.
                self.games.pop(id(position))
                moves.append(FORFEIT)
                continue
            moves.append(move)
        return moves

    def finish_game(self, position, capped):
        state = self.games.pop(id(position))
        final = _mirror(self.game_mod, state["moves"])
        if capped:
            assert not final.game_over
            assert final.current_player.value - 1 == position.current_player
            assert final.moves_left_in_turn == position.moves_remaining
        else:
            assert final.game_over and final.winner.value - 1 == position.winner, (
                "rules mismatch: the two implementations name different winners"
            )
        state["bot"] = None  # Release the per-game transposition table.
        return {"depths": state["depths"]}


@dataclass(frozen=True)
class SealBotOpponent:
    """SealBot at one uniquely identified strength setting."""

    root: Path
    variant: str = "current"
    time_limit: float = 0.1
    max_depth: int | None = None
    name: str = "sealbot"

    def __post_init__(self):
        if self.time_limit <= 0:
            raise ValueError(f"time_limit must be > 0, got {self.time_limit}")
        if self.max_depth is not None and self.max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {self.max_depth}")

    @property
    def config(self):
        return {
            "variant": self.variant,
            "time_limit": self.time_limit,
            "max_depth": self.max_depth,
        }

    def make_chooser(self, _ply_cap):
        game_mod, bot_type = load_sealbot(self.root, self.variant)
        return _SealBotChooser(
            game_mod, bot_type, self.time_limit, self.max_depth
        )


class _SeatChooser:
    """Map the generic opponent lifecycle onto one native seat connection."""

    def __init__(self, opponent: SeatOpponent):
        self.opponent = opponent
        self.seat = Seat(opponent.participant)
        self.games: dict[int, dict] = {}
        self.slots: dict[int, dict] = {}
        self.next_slot = 0
        self.finished = False
        try:
            response = self.seat.exchange(self.seat.hello_message, "hello")
            validate_welcome(response, opponent.participant)
            self.seat.identity = response
            opponent._identity = response
        except (SeatError, ValueError) as error:
            self.seat.abort()
            if isinstance(error, SeatError):
                raise
            self.seat.invalid(error, "hello")

    def _fail_response(
        self, response: dict, *, request: str, context: str
    ) -> None:
        self.seat.fail(
            f"slot refusal while {request}: "
            + json.dumps(response, ensure_ascii=False),
            context,
        )

    def _ok(self, message: dict, *, request: str, context: str) -> None:
        response = self.seat.exchange(message, context)
        if response.get("type") == "refuse":
            self._fail_response(
                response,
                request=request,
                context=context,
            )
        try:
            validate_ok(response, request)
        except ValueError as error:
            self.seat.invalid(error, context)

    def _shutdown_if_done(self) -> None:
        if self.games or self.finished:
            return
        if self.seat.open_slots:
            self.seat.fail(
                f"retained open slots {sorted(self.seat.open_slots)}",
                "evaluation shutdown",
            )
        self._ok(
            {"type": "bye"},
            request="bye",
            context="bye after evaluation match",
        )
        self.seat.finish()
        self.finished = True

    def start_game(self, position, moves):
        if self.finished:
            raise RuntimeError("cannot start a game after the seat shut down")
        key = id(position)
        if key in self.games:
            raise RuntimeError("position already has a seat game")
        opening = [(int(q), int(r)) for q, r in moves]
        state = {
            "key": key,
            "slot": self.next_slot,
            "opening": opening,
            "moves": opening.copy(),
            "sent": len(opening),
            "opened": False,
            "diagnostics": [],
        }
        self.next_slot += 1
        self.games[key] = state
        self.slots[state["slot"]] = state

    def after_move(self, position, move):
        state = self.games[id(position)]
        state["moves"].append((int(move[0]), int(move[1])))

    def _open(self, waiting) -> None:
        opening = [
            (position, state)
            for _, position, state in waiting
            if not state["opened"]
        ]
        if not opening:
            return
        slots = [state["slot"] for _, state in opening]
        context = f"open evaluation slots {slots}"
        self._ok(
            {
                "type": "open",
                "slots": [
                    {
                        "slot": state["slot"],
                        "side": f"p{position.current_player}",
                        "opening": [
                            wire_action(move) for move in state["opening"]
                        ],
                    }
                    for position, state in opening
                ],
            },
            request="open",
            context=context,
        )
        for _, state in opening:
            state["opened"] = True
            self.seat.open_slots.add(state["slot"])

    def _retire_refusal(self, state: dict) -> None:
        slot = state["slot"]
        self.games.pop(state["key"])
        self.slots.pop(slot)
        self.seat.open_slots.remove(slot)

    def _close_states(self, states: list[dict], context: str) -> None:
        if not states:
            return
        slots = [state["slot"] for state in states]
        self._ok(
            {"type": "close", "slots": slots},
            request="close",
            context=context,
        )
        for state in states:
            self.games.pop(state["key"])
            self.slots.pop(state["slot"])
            self.seat.open_slots.remove(state["slot"])

    def _decide(self, waiting, chosen) -> None:
        remaining = list(waiting)
        while remaining:
            slots = [state["slot"] for _, _, state in remaining]
            context = f"decide evaluation slots {slots}"
            response = self.seat.exchange(
                {
                    "type": "decide",
                    "slots": [
                        {
                            "slot": state["slot"],
                            "moves": [
                                wire_action(move)
                                for move in state["moves"][state["sent"] :]
                            ],
                            "zobrist": zobrist(position),
                        }
                        for _, position, state in remaining
                    ],
                },
                context,
            )
            if response.get("type") == "refuse":
                state = self.slots[response["slot"]]
                if (
                    response["cause"]["code"] != "restriction_exhausted"
                    or "restriction" not in self.seat.identity
                ):
                    self.seat.fail(
                        "slot refusal reports a participant fault: "
                        + json.dumps(response, ensure_ascii=False),
                        context,
                    )
                refused = next(
                    item for item in remaining if item[2] is state
                )
                chosen[refused[0]] = FORFEIT
                self._retire_refusal(state)
                remaining.remove(refused)
                continue
            try:
                decisions = validate_decided(response, slots)
                for (_, position, _), decision in zip(
                    remaining, decisions, strict=True
                ):
                    expected = zobrist(position)
                    if decision["zobrist"] != expected:
                        raise ValueError(
                            f"slot {decision['slot']} attested "
                            f"{decision['zobrist']}, expected {expected}"
                        )
            except ValueError as error:
                self.seat.invalid(error, context)

            illegal = []
            for (index, position, state), decision in zip(
                remaining, decisions, strict=True
            ):
                state["sent"] = len(state["moves"])
                state["diagnostics"].append(decision["diagnostics"])
                move = move_from_wire(decision["action"])
                if move not in set(position.legal_moves()):
                    chosen[index] = FORFEIT
                    illegal.append(state)
                else:
                    chosen[index] = move
            self._close_states(
                illegal,
                "close slots after illegal seat decisions",
            )
            return

    def __call__(self, positions, _rng):
        chosen = [None] * len(positions)
        waiting = [
            (index, position, self.games[id(position)])
            for index, position in enumerate(positions)
        ]
        try:
            self._open(waiting)
            self._decide(waiting, chosen)
            self._shutdown_if_done()
        except SeatError:
            self.seat.abort()
            raise
        return chosen

    def finish_game(self, position, _capped):
        state = self.games[id(position)]
        try:
            if state["opened"]:
                self._close_states(
                    [state],
                    f"close evaluation slot {state['slot']}",
                )
            else:
                self.games.pop(id(position))
                self.slots.pop(state["slot"])
            self._shutdown_if_done()
        except SeatError:
            self.seat.abort()
            raise
        return {"diagnostics": state["diagnostics"]}


@dataclass
class SeatOpponent:
    """One native subprocess seat at its welcome-bound strength."""

    participant: Participant
    _identity: dict | None = field(
        default=None, init=False, compare=False, repr=False
    )

    @property
    def name(self) -> str:
        if self._identity is None:
            raise RuntimeError(
                "seat opponent identity is unavailable before make_chooser"
            )
        return self._identity["name"]

    @property
    def config(self) -> dict:
        if self._identity is None:
            raise RuntimeError(
                "seat opponent config is unavailable before make_chooser"
            )
        return {
            "resolved_variant": self._identity["resolved_variant"],
            "digest": self._identity["digest"],
        }

    def make_chooser(self, _ply_cap):
        return _SeatChooser(self)


def shared_openings(
    rng: np.random.Generator, pairs: int, cut_range: tuple[int, int] = (2, 6)
) -> list[tuple[list[tuple[int, int]], int]]:
    """The seat-paired game schedule every match loop plays.

    ``pairs`` uniform-random prefixes, each played twice, returned as
    ``2 * pairs`` ``(opening, seat)`` games in pair-major order: games ``2i``
    and ``2i + 1`` are pair ``i`` — one shared prefix with the seats swapped —
    so opening difficulty and seat advantage cancel inside the pair instead of
    separating two games. ``seat`` is the seat the first chooser takes.

    Openings are non-terminal by the ``cut_range`` bound rather than by a
    check: at ten placements the leading player owns five stones, so no
    six-in-a-row is reachable.
    """
    import hexo_py

    if pairs < 1:
        raise ValueError(f"pairs must be >= 1, got {pairs}")
    lo, hi = cut_range
    if not 1 <= lo <= hi <= 10:
        raise ValueError(
            f"opening range must satisfy 1 <= lo <= hi <= 10: {cut_range}"
        )
    schedule = []
    for target in rng.integers(lo, hi + 1, size=pairs):
        position = hexo_py.Position()
        moves = []
        for _ in range(int(target)):
            move = position.nth_legal(int(rng.integers(position.legal_count)))
            position.advance(*move)
            moves.append(move)
        schedule.append((moves, 0))
        schedule.append((moves, 1))
    return schedule


def wilson(score: float, n: int, z: float = 1.96):
    """The Wilson interval, as rates, around a **total** score over ``n`` games."""
    p = score / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, centre - spread), min(1.0, centre + spread)


def elo(score: float) -> float:
    """Elo points from a score **rate** in ``[0, 1]``; ``±inf`` at the extremes."""
    if score <= 0.0:
        return -math.inf
    if score >= 1.0:
        return math.inf
    return -400.0 * math.log10(1.0 / score - 1.0)


def opponent_match(
    model_choose,
    opponent: Opponent,
    games: int,
    ply_cap: int,
    rng: np.random.Generator,
    opening_range: tuple[int, int] = (2, 6),
) -> tuple[dict, list[dict]]:
    """Seat-balanced paired games from shared openings against ``opponent``."""
    if games < 2 or games % 2:
        raise ValueError(f"games must be even and >= 2 (paired seats): {games}")

    import hexo_py

    opponent_choose = opponent.make_chooser(ply_cap)
    states = []
    for opening, seat in shared_openings(rng, games // 2, opening_range):
        position = hexo_py.Position.replay([tuple(move) for move in opening])
        state = {
            "pos": position,
            "moves": [tuple(move) for move in opening],
            "opening_len": len(opening),
            "seat": seat,
            "capped": False,
            "forfeit": False,
            "opponent_meta": {},
        }
        states.append(state)

    def apply(state, move):
        move = (int(move[0]), int(move[1]))
        before = getattr(opponent_choose, "before_move", None)
        if before is not None:
            before(state["pos"], move)
        if move not in set(state["pos"].legal_moves()):
            raise RuntimeError(f"evaluation chooser proposed illegal move {move}")
        state["pos"].advance(*move)
        state["moves"].append(move)
        after = getattr(opponent_choose, "after_move", None)
        if after is not None:
            after(state["pos"], move)

    wave_size = getattr(opponent_choose, "max_games_per_wave", games)
    started = time.monotonic()
    for wave_start in range(0, games, wave_size):
        wave = states[wave_start : wave_start + wave_size]
        start_game = getattr(opponent_choose, "start_game", None)
        if start_game is not None:
            for state in wave:
                start_game(state["pos"], state["moves"])
        live = list(range(len(wave)))

        def settle(indices):
            still = []
            for idx in indices:
                state = wave[idx]
                if state["forfeit"]:
                    continue  # The forfeiting chooser already dropped its state.
                if state["pos"].is_terminal:
                    pass
                elif len(state["moves"]) >= ply_cap:
                    state["capped"] = True
                else:
                    still.append(idx)
                    continue
                finish = getattr(opponent_choose, "finish_game", None)
                if finish is not None:
                    state["opponent_meta"] = finish(
                        state["pos"], state["capped"]
                    ) or {}
            return still

        live = settle(live)
        while live:
            model_group = [
                idx
                for idx in live
                if wave[idx]["pos"].current_player == wave[idx]["seat"]
            ]
            if model_group:
                chosen = model_choose(
                    [wave[idx]["pos"] for idx in model_group], rng
                )
                for idx, move in zip(model_group, chosen, strict=True):
                    apply(wave[idx], move)
            live = settle(live)
            if not live:
                break

            opponent_group = [
                idx
                for idx in live
                if not wave[idx]["pos"].is_terminal
                and wave[idx]["pos"].current_player != wave[idx]["seat"]
            ]
            if opponent_group:
                chosen = opponent_choose(
                    [wave[idx]["pos"] for idx in opponent_group], rng
                )
                for idx, move in zip(opponent_group, chosen, strict=True):
                    if move is FORFEIT:
                        wave[idx]["forfeit"] = True
                    else:
                        apply(wave[idx], move)
            live = settle(live)

    score, per_seat, capped, plies = 0.0, [0.0, 0.0], 0, 0
    all_depths = []
    per_game = []
    forfeits = 0
    for state in states:
        if state["forfeit"]:
            forfeits += 1
            game_score = 1.0  # An unplayable proposal forfeits the game.
        elif state["capped"]:
            capped += 1
            game_score = 0.5
        else:
            game_score = (
                1.0 if state["pos"].winner == state["seat"] else 0.0
            )
        depths = state["opponent_meta"].get("depths", [])
        all_depths.extend(depths)
        score += game_score
        per_seat[state["seat"]] += game_score
        plies += len(state["moves"])
        per_game.append(
            {
                "seat": state["seat"],
                "winner": state["seat"]
                if state["forfeit"]
                else None
                if state["capped"]
                else state["pos"].winner,
                "capped": state["capped"],
                "forfeit": state["forfeit"],
                "score": game_score,
                "opening_len": state["opening_len"],
                "depth_mean": float(np.mean(depths)) if depths else None,
                "moves": state["moves"],
            }
        )

    ci_lo, ci_hi = wilson(score, games)
    summary = {
        "score": score,
        "games": games,
        "capped": capped,
        "win_rate": score / games,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "elo": elo(score / games),
        "elo_lo": elo(ci_lo),
        "elo_hi": elo(ci_hi),
        "score_as_p0": per_seat[0],
        "score_as_p1": per_seat[1],
        "forfeits": forfeits,
        "opponent_retries": getattr(opponent_choose, "retries", 0),
        "opponent_name": opponent.name,
        "opponent_config": opponent.config,
        "opponent_depth_mean": (
            float(np.mean(all_depths)) if all_depths else float("nan")
        ),
        "avg_plies": plies / games,
        "seconds": time.monotonic() - started,
    }
    return summary, per_game
