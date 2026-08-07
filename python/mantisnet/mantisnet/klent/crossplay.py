"""Round-robin referee for independent subprocess seats.

The referee owns all authoritative positions, draws shared openings,
alternates seats, applies the ply cap, and adjudicates outcomes.

    python -m mantisnet.klent.crossplay \
        --participants participants.json --pairs 32 \
        --anchor baseline=0 --out crossplay.json
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import hexo_py
import numpy as np

from .headtohead import paired_statistics
from .opponents import shared_openings
from .seat import (
    Participant,
    Seat,
    SeatError,
    load_participants,
    move_from_wire,
    validate_decided,
    validate_ok,
    validate_welcome,
    wire_action,
    zobrist,
)

_BT_TOLERANCE = 1e-10
_BT_ITERATIONS = 200


@dataclass
class _Game:
    slot: int
    pairing: int
    pair: int
    a: str
    b: str
    a_seat: int
    opening: list[tuple[int, int]]
    position: Any
    moves: list[tuple[int, int]]
    sent_plies: dict[str, int]
    open_participants: set[str] = field(default_factory=set)
    plies: list[dict[str, Any]] = field(default_factory=list)
    winner: str | None = None
    capped: bool = False
    adjudication: dict[str, Any] | None = None

    @property
    def active(self) -> bool:
        return self.adjudication is None

    def participant_at(self, side: int) -> str:
        if side not in (0, 1):
            raise ValueError(f"invalid side {side}")
        return self.a if self.a_seat == side else self.b

    def side_of(self, participant: str) -> int:
        if participant == self.a:
            return self.a_seat
        if participant == self.b:
            return 1 - self.a_seat
        raise ValueError(f"{participant!r} is not in game {self.slot}")

    def score_a(self) -> float:
        if self.active:
            raise RuntimeError(f"game {self.slot} has no result")
        if self.capped:
            return 0.5
        return 1.0 if self.winner == self.a else 0.0


def _parse_anchors(values: Sequence[str]) -> dict[str, float]:
    anchors: dict[str, float] = {}
    for value in values:
        participant, separator, rating_text = value.partition("=")
        if not separator or not participant or not rating_text:
            raise ValueError(
                f"anchor must have the form PARTICIPANT=RATING, got {value!r}"
            )
        if participant in anchors:
            raise ValueError(f"anchor {participant!r} was specified more than once")
        try:
            rating = float(rating_text)
        except ValueError as error:
            raise ValueError(f"anchor rating is not a number: {value!r}") from error
        if not math.isfinite(rating):
            raise ValueError(f"anchor rating must be finite: {value!r}")
        anchors[participant] = rating
    if not anchors:
        raise ValueError("at least one --anchor PARTICIPANT=RATING is required")
    return anchors


def _components(adjacency: Sequence[set[int]]) -> list[list[int]]:
    unseen = set(range(len(adjacency)))
    components: list[list[int]] = []
    while unseen:
        root = min(unseen)
        stack, component = [root], []
        unseen.remove(root)
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbour in sorted(adjacency[node], reverse=True):
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)
        components.append(sorted(component))
    return components


def _reachable(starts: Iterable[int], adjacency: Sequence[set[int]]) -> set[int]:
    reached = set(starts)
    stack = list(reached)
    while stack:
        node = stack.pop()
        for neighbour in adjacency[node]:
            if neighbour not in reached:
                reached.add(neighbour)
                stack.append(neighbour)
    return reached


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _bt_likelihood(
    ratings: np.ndarray,
    comparisons: Sequence[tuple[int, int, float, float]],
) -> float:
    likelihood = 0.0
    for a, b, games, points_a in comparisons:
        difference = ratings[a] - ratings[b]
        likelihood += -points_a * float(np.logaddexp(0.0, -difference))
        likelihood += -(games - points_a) * float(
            np.logaddexp(0.0, difference)
        )
    return likelihood


def _bt_information(
    ratings: np.ndarray,
    comparisons: Sequence[tuple[int, int, float, float]],
    free: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    free_index = {participant: index for index, participant in enumerate(free)}
    gradient = np.zeros(len(free), dtype=float)
    information = np.zeros((len(free), len(free)), dtype=float)
    for a, b, games, points_a in comparisons:
        probability = _sigmoid(float(ratings[a] - ratings[b]))
        residual = points_a - games * probability
        weight = games * probability * (1.0 - probability)
        ai, bi = free_index.get(a), free_index.get(b)
        if ai is not None:
            gradient[ai] += residual
            information[ai, ai] += weight
        if bi is not None:
            gradient[bi] -= residual
            information[bi, bi] += weight
        if ai is not None and bi is not None:
            information[ai, bi] -= weight
            information[bi, ai] -= weight
    return gradient, information


def _fit_bt_core(
    ratings: np.ndarray,
    comparisons: Sequence[tuple[int, int, float, float]],
    free: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    if not free:
        return ratings, np.empty((0, 0), dtype=float)
    likelihood = _bt_likelihood(ratings, comparisons)
    converged = False
    for _iteration in range(_BT_ITERATIONS):
        gradient, information = _bt_information(
            ratings, comparisons, free
        )
        if float(np.max(np.abs(gradient))) <= _BT_TOLERANCE:
            converged = True
            break
        try:
            direction = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError as error:
            raise ArithmeticError("singular observed information") from error
        if not np.all(np.isfinite(direction)):
            raise ArithmeticError("non-finite Newton direction")
        directional_gain = float(gradient @ direction)
        step = 1.0
        accepted = False
        while step >= 2.0**-40:
            candidate = ratings.copy()
            candidate[list(free)] += step * direction
            candidate_likelihood = _bt_likelihood(candidate, comparisons)
            if candidate_likelihood >= likelihood + 1e-4 * step * directional_gain:
                ratings = candidate
                likelihood = candidate_likelihood
                accepted = True
                break
            step *= 0.5
        if not accepted:
            raise ArithmeticError("Newton line search did not improve likelihood")
        if float(np.max(np.abs(step * direction))) <= _BT_TOLERANCE:
            converged = True
            break
    if not converged:
        raise ArithmeticError("Bradley-Terry Newton fit did not converge")
    _gradient, information = _bt_information(ratings, comparisons, free)
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError as error:
        raise ArithmeticError("singular rating covariance") from error
    if not np.all(np.isfinite(covariance)) or np.any(np.diag(covariance) < 0):
        raise ArithmeticError("invalid rating covariance")
    return ratings, covariance


def fit_bradley_terry(
    participants: Sequence[str],
    pairwise_results: Sequence[Mapping[str, Any]],
    anchors: Mapping[str, float],
) -> dict[str, Any]:
    """Fit unregularized anchored Bradley-Terry ratings in natural log-odds.

    Each result carries ``a``, ``b``, ``games``, and ``score_a``, where
    ``score_a`` is a rate. A capped game has already contributed one half to
    that rate. Directed outcome reachability detects separation before Newton
    fitting, so an infinite maximum is reported as an absent rating rather than
    a clipped finite number.
    """
    ids = list(participants)
    if len(ids) < 2 or any(not isinstance(item, str) or not item for item in ids):
        raise ValueError("Bradley-Terry needs at least two nonempty participant ids")
    if len(set(ids)) != len(ids):
        raise ValueError("Bradley-Terry participant ids must be unique")
    if not anchors:
        raise ValueError("Bradley-Terry needs at least one fixed anchor")
    index = {participant: position for position, participant in enumerate(ids)}
    fixed: dict[int, float] = {}
    for participant, rating in anchors.items():
        if participant not in index:
            raise ValueError(f"unknown Bradley-Terry anchor {participant!r}")
        if isinstance(rating, bool) or not isinstance(rating, (int, float)):
            raise TypeError(f"anchor {participant!r} rating must be numeric")
        rating = float(rating)
        if not math.isfinite(rating):
            raise ValueError(f"anchor {participant!r} rating must be finite")
        fixed[index[participant]] = rating

    aggregated: dict[tuple[int, int], list[float]] = {}
    for row_number, row in enumerate(pairwise_results):
        try:
            a_id, b_id = row["a"], row["b"]
            games, score_a = row["games"], row["score_a"]
        except KeyError as error:
            raise ValueError(
                f"pairwise result {row_number} is missing {error.args[0]!r}"
            ) from error
        if a_id not in index or b_id not in index or a_id == b_id:
            raise ValueError(
                f"pairwise result {row_number} has invalid participants "
                f"{a_id!r}, {b_id!r}"
            )
        if type(games) is not int or games <= 0:
            raise ValueError(f"pairwise result {row_number}.games must be positive")
        if isinstance(score_a, bool) or not isinstance(score_a, (int, float)):
            raise TypeError(f"pairwise result {row_number}.score_a must be numeric")
        score_a = float(score_a)
        if not math.isfinite(score_a) or not 0.0 <= score_a <= 1.0:
            raise ValueError(
                f"pairwise result {row_number}.score_a must be within [0, 1]"
            )
        a, b = index[a_id], index[b_id]
        if a < b:
            key, points = (a, b), games * score_a
        else:
            key, points = (b, a), games * (1.0 - score_a)
        aggregate = aggregated.setdefault(key, [0.0, 0.0])
        aggregate[0] += games
        aggregate[1] += points

    comparisons = [
        (a, b, values[0], values[1])
        for (a, b), values in sorted(aggregated.items())
    ]
    undirected = [set() for _ in ids]
    directed = [set() for _ in ids]
    reverse = [set() for _ in ids]
    total_games = np.zeros(len(ids), dtype=float)
    total_points = np.zeros(len(ids), dtype=float)
    for a, b, games, points_a in comparisons:
        undirected[a].add(b)
        undirected[b].add(a)
        total_games[a] += games
        total_games[b] += games
        total_points[a] += points_a
        total_points[b] += games - points_a
        if points_a > 0.0:
            directed[a].add(b)
            reverse[b].add(a)
        if games - points_a > 0.0:
            directed[b].add(a)
            reverse[a].add(b)

    components = _components(undirected)
    component_of = {
        participant: component_index
        for component_index, component in enumerate(components)
        for participant in component
    }
    anchored_components = {component_of[participant] for participant in fixed}
    forward = _reachable(fixed, directed)
    backward = _reachable(fixed, reverse)
    entries: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    connected = len(components) == 1
    if not connected:
        named = [[ids[participant] for participant in group] for group in components]
        warnings.append(f"result graph is disconnected: {named}")

    estimated: list[int] = []
    statuses: dict[int, str] = {}
    for participant, participant_id in enumerate(ids):
        if participant in fixed:
            statuses[participant] = "fixed"
            continue
        if component_of[participant] not in anchored_components:
            statuses[participant] = "disconnected"
        elif participant not in forward and participant in backward:
            statuses[participant] = "unbounded_above"
        elif participant in forward and participant not in backward:
            statuses[participant] = "unbounded_below"
        elif participant not in forward and participant not in backward:
            statuses[participant] = "unbounded_both"
        else:
            statuses[participant] = "estimated"
            estimated.append(participant)
        if total_games[participant] > 0:
            if total_points[participant] == total_games[participant]:
                warnings.append(
                    f"{participant_id!r} won every recorded game; its free "
                    "rating is unbounded above"
                )
            elif total_points[participant] == 0:
                warnings.append(
                    f"{participant_id!r} lost every recorded game; its free "
                    "rating is unbounded below"
                )

    separated = [
        ids[participant]
        for participant, status in statuses.items()
        if status.startswith("unbounded")
    ]
    if separated:
        warnings.append(
            "directed outcomes separate these free ratings, which are absent: "
            f"{separated}"
        )
    disconnected = [
        ids[participant]
        for participant, status in statuses.items()
        if status == "disconnected"
    ]
    if disconnected:
        warnings.append(
            "no fixed anchor is connected to these participants, so their "
            f"ratings are absent: {disconnected}"
        )

    ratings = np.full(len(ids), np.nan, dtype=float)
    for participant, rating in fixed.items():
        ratings[participant] = rating
    for participant in estimated:
        component = components[component_of[participant]]
        component_anchors = [
            fixed[node] for node in component if node in fixed
        ]
        ratings[participant] = float(np.mean(component_anchors))
    core = set(fixed) | set(estimated)
    core_comparisons = [
        row for row in comparisons if row[0] in core and row[1] in core
    ]
    covariance: np.ndarray | None = None
    if estimated:
        try:
            ratings, covariance = _fit_bt_core(
                ratings, core_comparisons, estimated
            )
        except ArithmeticError as error:
            warnings.append(
                "finite Bradley-Terry coefficients could not be estimated: "
                f"{error}"
            )
            for participant in estimated:
                statuses[participant] = "unestimable"
                ratings[participant] = np.nan
            estimated = []
            covariance = None

    estimated_index = {
        participant: position for position, participant in enumerate(estimated)
    }
    for participant, participant_id in enumerate(ids):
        status = statuses[participant]
        if status == "fixed":
            rating: float | None = fixed[participant]
            standard_error: float | None = 0.0
        elif status == "estimated":
            rating = float(ratings[participant])
            assert covariance is not None
            standard_error = math.sqrt(
                float(covariance[estimated_index[participant], estimated_index[participant]])
            )
        else:
            rating, standard_error = None, None
        entries[participant_id] = {
            "rating": rating,
            "standard_error": standard_error,
            "fixed": status == "fixed",
            "status": status,
        }

    return {
        "model": "bradley-terry",
        "scale": "natural_log_odds",
        "connected": connected,
        "anchors": {participant: float(rating) for participant, rating in anchors.items()},
        "components": [[ids[node] for node in group] for group in components],
        "ratings": entries,
        "warnings": warnings,
    }


def _pairing_context(games: Sequence[_Game]) -> str:
    pairings = sorted({f"{game.a} vs {game.b}" for game in games})
    return "decide for pairing(s) " + ", ".join(pairings)


def _finish_refusal(game: _Game, loser: str, response: dict[str, Any]) -> None:
    game.winner = game.b if loser == game.a else game.a
    game.adjudication = {
        "type": "seat_refusal",
        "loser": loser,
        "response": response,
    }


def _require_restriction_exhaustion(
    seat: Seat,
    response: dict[str, Any],
    context: str,
) -> None:
    declared_restriction = (
        seat.identity is not None and "restriction" in seat.identity
    )
    if (
        response["cause"]["code"] != "restriction_exhausted"
        or not declared_restriction
    ):
        seat.fail(
            "slot refusal reports a participant fault: "
            + json.dumps(response, ensure_ascii=False),
            context,
        )


def _finish_illegal(
    game: _Game, loser: str, action: int, error: ValueError
) -> None:
    game.winner = game.b if loser == game.a else game.a
    game.adjudication = {
        "type": "illegal_action",
        "loser": loser,
        "action": action,
        "move_error": str(error),
    }


def _close_finished(games: Sequence[_Game], seats: Mapping[str, Seat]) -> None:
    grouped: dict[str, list[_Game]] = collections.defaultdict(list)
    for game in games:
        if game.active:
            continue
        for participant in sorted(game.open_participants):
            grouped[participant].append(game)
    for participant, participant_games in grouped.items():
        slots = [game.slot for game in participant_games]
        seat = seats[participant]
        response = seat.exchange(
            {"type": "close", "slots": slots},
            "close slots "
            + repr(slots)
            + " after pairing(s) "
            + ", ".join(sorted({f"{game.a} vs {game.b}" for game in participant_games})),
        )
        if response.get("type") == "refuse":
            seat.fail(
                "slot refusal while closing an already adjudicated game: "
                + json.dumps(response, ensure_ascii=False),
                "close",
            )
        try:
            validate_ok(response, "close")
        except ValueError as error:
            seat.invalid(error, "close")
        for game in participant_games:
            game.open_participants.remove(participant)
            seat.open_slots.remove(game.slot)


def _participant_record(seat: Seat) -> dict[str, Any]:
    assert seat.identity is not None
    return {
        "id": seat.participant.id,
        "command": list(seat.participant.command),
        "hello": seat.hello_message,
        "welcome": seat.identity,
    }


def _game_result(
    game: _Game, participant_records: Mapping[str, dict[str, Any]]
) -> dict[str, Any]:
    if game.active:
        raise RuntimeError(f"game {game.slot} did not finish")
    p0, p1 = game.participant_at(0), game.participant_at(1)
    return {
        "game": game.slot,
        "pair": game.pair,
        "opening": [list(move) for move in game.opening],
        "seat": game.a_seat,
        "seats": {
            "p0": participant_records[p0],
            "p1": participant_records[p1],
        },
        "winner": game.winner,
        "capped": game.capped,
        "score_a": game.score_a(),
        "moves": [list(move) for move in game.moves],
        "plies": game.plies,
        "adjudication": game.adjudication,
    }


def _pairing_report(
    pairing_games: Sequence[_Game],
    participant_records: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    a, b = pairing_games[0].a, pairing_games[0].b
    results = [_game_result(game, participant_records) for game in pairing_games]
    pairs = [
        results[index : index + 2] for index in range(0, len(results), 2)
    ]
    statistics = paired_statistics(pairs)
    per_pair = []
    for pair, games in enumerate(pairs):
        per_pair.append(
            {
                "pair": pair,
                "opening": games[0]["opening"],
                "games": [game["game"] for game in games],
                "seats": [game["seat"] for game in games],
                "scores": [game["score_a"] for game in games],
                "capped": [game["capped"] for game in games],
                "d": games[0]["score_a"] + games[1]["score_a"] - 1.0,
            }
        )
    return {
        "a": a,
        "b": b,
        "participants": {
            "a": participant_records[a],
            "b": participant_records[b],
        },
        "restrictions": {
            "a": participant_records[a]["welcome"].get("restriction"),
            "b": participant_records[b]["welcome"].get("restriction"),
        },
        "statistics": statistics,
        "pairs": per_pair,
        "games": results,
    }


def _matrix(
    participant_ids: Sequence[str], pairings: Sequence[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    matrix: dict[str, dict[str, Any]] = {
        participant: {opponent: None for opponent in participant_ids}
        for participant in participant_ids
    }
    for pairing_index, pairing in enumerate(pairings):
        a, b = pairing["a"], pairing["b"]
        games = pairing["games"]
        wins_a = sum(game["score_a"] == 1.0 for game in games)
        losses_a = sum(game["score_a"] == 0.0 for game in games)
        draws = sum(game["capped"] for game in games)
        common = {
            "games": len(games),
            "capped": draws,
            "pairing": pairing_index,
        }
        matrix[a][b] = {
            **common,
            "score": pairing["statistics"]["score"],
            "wins": wins_a,
            "draws": draws,
            "losses": losses_a,
        }
        matrix[b][a] = {
            **common,
            "score": 1.0 - pairing["statistics"]["score"],
            "wins": losses_a,
            "draws": draws,
            "losses": wins_a,
        }
    return matrix


def cross_play(
    participants: Sequence[Participant],
    *,
    pairs: int,
    anchors: Mapping[str, float],
    ply_cap: int = 512,
    seed: int = 0,
    opening_range: tuple[int, int] = (2, 6),
) -> dict[str, Any]:
    """Play every unordered participant pairing through one subprocess per seat."""
    participants = list(participants)
    if len(participants) < 2:
        raise ValueError("crossplay needs at least two participants")
    ids = [participant.id for participant in participants]
    if len(set(ids)) != len(ids):
        raise ValueError("crossplay participant ids must be unique")
    if pairs < 1:
        raise ValueError(f"pairs must be >= 1, got {pairs}")
    if not anchors:
        raise ValueError("crossplay needs at least one fixed rating anchor")
    unknown_anchors = sorted(set(anchors) - set(ids))
    if unknown_anchors:
        raise ValueError(f"unknown rating anchors: {unknown_anchors}")
    for participant, rating in anchors.items():
        if isinstance(rating, bool) or not isinstance(rating, (int, float)):
            raise TypeError(f"anchor {participant!r} rating must be numeric")
        if not math.isfinite(float(rating)):
            raise ValueError(f"anchor {participant!r} rating must be finite")
    lo, hi = opening_range
    if not 1 <= lo <= hi <= 10:
        raise ValueError(
            f"opening range must satisfy 1 <= lo <= hi <= 10: {opening_range}"
        )
    if ply_cap <= hi:
        raise ValueError(
            f"ply_cap must exceed the longest opening ({hi}), got {ply_cap}"
        )

    started = time.monotonic()
    seats: dict[str, Seat] = {}
    try:
        for participant in participants:
            seats[participant.id] = Seat(participant)
        for participant in participants:
            seat = seats[participant.id]
            seat.send(seat.hello_message, "hello")
        for participant in participants:
            seat = seats[participant.id]
            response = seat.receive()
            try:
                validate_welcome(response, participant)
            except ValueError as error:
                seat.invalid(error, "hello")
            seat.identity = response

        games: list[_Game] = []
        pairing_games: list[list[_Game]] = []
        slot = 0
        for a_index, a in enumerate(participants):
            for b_index in range(a_index + 1, len(participants)):
                b = participants[b_index]
                rng = np.random.default_rng([seed, a_index, b_index])
                schedule = shared_openings(rng, pairs, opening_range)
                current_pairing: list[_Game] = []
                for schedule_index, (opening, a_seat) in enumerate(schedule):
                    moves = [tuple(move) for move in opening]
                    game = _Game(
                        slot=slot,
                        pairing=len(pairing_games),
                        pair=schedule_index // 2,
                        a=a.id,
                        b=b.id,
                        a_seat=a_seat,
                        opening=moves.copy(),
                        position=hexo_py.Position.replay(moves),
                        moves=moves.copy(),
                        sent_plies={a.id: len(moves), b.id: len(moves)},
                    )
                    games.append(game)
                    current_pairing.append(game)
                    slot += 1
                pairing_games.append(current_pairing)

        for game in games:
            for side in (0, 1):
                participant = game.participant_at(side)
                seat = seats[participant]
                response = seat.exchange(
                    {
                        "type": "open",
                        "slots": [
                            {
                                "slot": game.slot,
                                "side": f"p{side}",
                                "opening": [
                                    wire_action(move) for move in game.opening
                                ],
                            }
                        ],
                    },
                    f"open slot {game.slot} for pairing {game.a} vs {game.b}",
                )
                if response.get("type") == "refuse":
                    seat.fail(
                        "slot refusal while opening a mirror: "
                        + json.dumps(response, ensure_ascii=False),
                        f"open slot {game.slot} for pairing {game.a} vs {game.b}",
                    )
                try:
                    validate_ok(response, "open")
                except ValueError as error:
                    seat.invalid(error, f"open slot {game.slot}")
                seat.open_slots.add(game.slot)
                game.open_participants.add(participant)
        _close_finished(games, seats)

        by_slot = {game.slot: game for game in games}
        while any(game.active for game in games):
            waiting: dict[str, list[_Game]] = collections.defaultdict(list)
            for game in games:
                if not game.active:
                    continue
                participant = game.participant_at(game.position.current_player)
                if participant not in game.open_participants:
                    raise RuntimeError(
                        f"active game {game.slot} has no open slot for {participant}"
                    )
                waiting[participant].append(game)

            requests: dict[str, tuple[list[_Game], list[int]]] = {}
            for participant in ids:
                participant_games = sorted(
                    waiting.get(participant, []), key=lambda game: game.slot
                )
                if not participant_games:
                    continue
                slots = [game.slot for game in participant_games]
                request = {
                    "type": "decide",
                    "slots": [
                        {
                            "slot": game.slot,
                            "moves": [
                                wire_action(move)
                                for move in game.moves[
                                    game.sent_plies[participant] :
                                ]
                            ],
                            "zobrist": zobrist(game.position),
                        }
                        for game in participant_games
                    ],
                }
                seats[participant].send(request, _pairing_context(participant_games))
                requests[participant] = (participant_games, slots)

            accepted: list[tuple[str, _Game, dict[str, Any]]] = []
            for participant in ids:
                if participant not in requests:
                    continue
                participant_games, slots = requests[participant]
                seat = seats[participant]
                response = seat.receive()
                if response.get("type") == "refuse":
                    refused_game = by_slot[response["slot"]]
                    _require_restriction_exhaustion(
                        seat,
                        response,
                        _pairing_context(participant_games),
                    )
                    _finish_refusal(refused_game, participant, response)
                    refused_game.open_participants.remove(participant)
                    seat.open_slots.remove(refused_game.slot)
                    continue
                try:
                    decisions = validate_decided(response, slots)
                    for game, decision in zip(
                        participant_games, decisions, strict=True
                    ):
                        expected = zobrist(game.position)
                        if decision["zobrist"] != expected:
                            raise ValueError(
                                f"slot {game.slot} attested "
                                f"{decision['zobrist']}, expected {expected}"
                            )
                except ValueError as error:
                    seat.invalid(error, _pairing_context(participant_games))
                for game, decision in zip(
                    participant_games, decisions, strict=True
                ):
                    game.sent_plies[participant] = len(game.moves)
                    accepted.append((participant, game, decision))

            for participant, game, decision in accepted:
                action = decision["action"]
                move = move_from_wire(action)
                try:
                    game.position.advance(*move)
                except ValueError as error:
                    _finish_illegal(game, participant, action, error)
                    continue
                game.moves.append(move)
                game.plies.append(
                    {
                        "participant": participant,
                        "seat": f"p{game.side_of(participant)}",
                        "action": action,
                        "zobrist": zobrist(game.position),
                        "diagnostics": decision["diagnostics"],
                    }
                )
                if game.position.is_terminal:
                    game.winner = game.participant_at(game.position.winner)
                    game.adjudication = {
                        "type": "line",
                        "winner": game.winner,
                    }
                elif len(game.moves) >= ply_cap:
                    game.capped = True
                    game.adjudication = {
                        "type": "ply_cap",
                        "limit": ply_cap,
                    }
            _close_finished(games, seats)

        for participant in ids:
            if seats[participant].open_slots:
                raise RuntimeError(
                    f"{participant} retained open slots "
                    f"{sorted(seats[participant].open_slots)}"
                )
        for participant in ids:
            seats[participant].send({"type": "bye"}, "bye")
        for participant in ids:
            seat = seats[participant]
            response = seat.receive()
            try:
                validate_ok(response, "bye")
            except ValueError as error:
                seat.invalid(error, "bye")
        for participant in ids:
            seats[participant].finish()

        participant_records = {
            participant: _participant_record(seats[participant])
            for participant in ids
        }
        pairings = [
            _pairing_report(current_games, participant_records)
            for current_games in pairing_games
        ]
        pairwise_ratings = [
            {
                "a": pairing["a"],
                "b": pairing["b"],
                "games": pairing["statistics"]["games"],
                "score_a": pairing["statistics"]["score"],
            }
            for pairing in pairings
        ]
        return {
            "versions": {
                "protocol": hexo_py.PROTOCOL_VERSION,
                "rules": hexo_py.RULES_VERSION,
                "action_order": hexo_py.ACTION_ORDER_VERSION,
            },
            "participants": [participant_records[participant] for participant in ids],
            "match": {
                "pairs_per_pairing": pairs,
                "games_per_pairing": 2 * pairs,
                "opening_range": list(opening_range),
                "ply_cap": ply_cap,
                "seed": seed,
                "seconds": time.monotonic() - started,
                "reproducibility": (
                    "the seed fixes referee openings only; the seat protocol "
                    "does not carry participant RNG seeds"
                ),
            },
            "pairings": pairings,
            "matrix": _matrix(ids, pairings),
            "ratings": fit_bradley_terry(ids, pairwise_ratings, anchors),
        }
    except BaseException:
        for seat in seats.values():
            seat.abort()
        raise


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--participants",
        type=Path,
        required=True,
        help="JSON participant list: id, argv command, and hello fields",
    )
    parser.add_argument(
        "--pairs",
        type=int,
        required=True,
        help="shared openings per pairing, each played from both seats",
    )
    parser.add_argument(
        "--anchor",
        action="append",
        default=[],
        metavar="PARTICIPANT=RATING",
        help="fixed Bradley-Terry log-odds rating; repeatable and required",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cap", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--opening-range",
        type=int,
        nargs=2,
        metavar=("LO", "HI"),
        default=(2, 6),
    )
    args = parser.parse_args(argv)
    if not args.out.parent.is_dir():
        parser.error(f"--out: no directory {args.out.parent} to write into")
    if args.out.is_dir():
        parser.error(f"--out: {args.out} is a directory")
    try:
        participants = load_participants(args.participants)
        anchors = _parse_anchors(args.anchor)
        result = cross_play(
            participants,
            pairs=args.pairs,
            anchors=anchors,
            ply_cap=args.cap,
            seed=args.seed,
            opening_range=tuple(args.opening_range),
        )
    except (OSError, TypeError, ValueError, SeatError) as error:
        parser.error(str(error))
    temporary = args.out.with_name(args.out.name + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.out)
    print(
        f"played {len(result['pairings'])} pairings among "
        f"{len(result['participants'])} participants"
    )
    for warning in result["ratings"]["warnings"]:
        print(f"WARNING: {warning}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
