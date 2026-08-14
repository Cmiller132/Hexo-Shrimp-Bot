"""Paired cross-run head-to-head: the statistics, the pairing, and the refusals.

The statistics tests are hand-computed against synthetic game rows, because the
whole point of the module is a number that is smaller than an unpaired one — a
property no end-to-end match can demonstrate on four games. The end-to-end tests
pin determinism, the pairing, and the manifest on two real checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from mantisnet.klent.headtohead import (
    head_to_head,
    main,
    paired_statistics,
    sign_test,
)
from mantisnet.klent.opponents import elo, shared_openings
from mantisnet.klent.run import _versions, save_checkpoint
from mantisnet.model import MantisConfig, MantisNet


def _game(score: float, seat: int, capped: bool = False) -> dict:
    """One synthetic ``play_match`` row, in the fields the statistics read."""
    return {"score_a": score, "seat": seat, "capped": capped}


def _pair(p0: float, p1: float, capped: tuple[bool, bool] = (False, False)) -> list[dict]:
    """One seat pair: A's score with A as P0, then with A as P1."""
    return [_game(p0, 0, capped[0]), _game(p1, 1, capped[1])]


@pytest.fixture(scope="session")
def checkpoints(tmp_path_factory) -> tuple[Path, Path]:
    """Two checkpoints in separate run directories, written by the driver's own
    ``save_checkpoint``. The weights are perturbed because the output layers
    initialize to zero: two untouched fresh models argmax to legal rank 0 and
    would play one line."""
    root = tmp_path_factory.mktemp("h2h")
    paths = []
    for index, seed in enumerate((11, 23), start=1):
        torch.manual_seed(seed)
        model = MantisNet(MantisConfig())
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.05)
        run = root / f"run{index}"
        run.mkdir()
        path = run / f"checkpoint_{1000 * index:06d}.pt"
        save_checkpoint(
            path,
            model,
            torch.optim.Adam(model.parameters()),
            1000 * index,
            np.random.default_rng(seed),
        )
        paths.append(path)
    return paths[0], paths[1]


def _retagged(source: Path, target: Path, **fields) -> Path:
    """``source`` rewritten with ``fields`` replacing entries of its versions dict."""
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    checkpoint["versions"] = {**checkpoint["versions"], **fields}
    torch.save(checkpoint, target)
    return target


def _match(path_a: Path, path_b: Path, **overrides) -> dict:
    settings = {
        "pairs": 2,
        "sims": 0,
        "tau": 0.1,
        "lam": 0.03,
        "ply_cap": 24,
        "device": "cpu",
        "seed": 7,
    }
    return head_to_head(path_a, path_b, **(settings | overrides))


# --- the pairing -----------------------------------------------------------


def test_shared_openings_plays_one_prefix_from_both_seats():
    schedule = shared_openings(np.random.default_rng(3), 4, (2, 6))

    assert len(schedule) == 8
    assert [seat for _opening, seat in schedule] == [0, 1] * 4
    for index in range(4):
        first, second = schedule[2 * index], schedule[2 * index + 1]
        assert first[0] == second[0]
        assert 2 <= len(first[0]) <= 6
    # Distinct prefixes across pairs are what makes the games distinct at all.
    assert len({tuple(opening) for opening, _seat in schedule}) == 4


# --- the statistics --------------------------------------------------------


def test_all_decisive_pairs_score_a_perfectly_and_sign_test_exactly():
    stats = paired_statistics([_pair(1.0, 1.0) for _ in range(4)])

    assert stats["pairs"] == 4 and stats["games"] == 8
    assert stats["a_wins"] == 8.0 and stats["score"] == 1.0
    assert stats["pair_counts"] == {"a_both": 4, "split": 0, "b_both": 0, "capped": 0}
    assert stats["decisive_pairs"] == 4
    assert stats["d_mean"] == 1.0 and stats["d_sd"] == 0.0
    assert stats["paired_se"] == 0.0
    # 2 * P(X = 4 | n = 4, p = ½) = 2/16.
    assert stats["sign_test_p"] == 0.125
    # A perfect score is +inf Elo, which strict JSON cannot hold.
    assert stats["elo"] is None and stats["elo_lo"] is None
    # Four identical pairs have no spread, so the sign test is the whole
    # quantified evidence and the warning says so.
    assert any("no spread" in warning for warning in stats["warnings"])


def test_all_split_pairs_leave_no_paired_variance_at_all():
    stats = paired_statistics([_pair(1.0, 0.0) for _ in range(4)])

    assert stats["a_wins"] == 4.0 and stats["score"] == 0.5
    assert stats["score_as_p0"] == 4.0 and stats["score_as_p1"] == 0.0
    assert stats["pair_counts"] == {"a_both": 0, "split": 4, "b_both": 0, "capped": 0}
    assert stats["d_mean"] == 0.0 and stats["d_sd"] == 0.0
    # The whole spread was the seat advantage, and the pairing removed it: the
    # eight games look like a 50% score with real noise, the four pairs like
    # four identical ties.
    assert stats["paired_se"] == 0.0
    assert stats["unpaired_se"] == pytest.approx(0.3779644730092272)
    assert stats["se_ratio"] is None
    # A zero sample spread is no variance estimate rather than a variance of
    # zero, so the interval it would imply — a point at 0 Elo — is absent.
    assert stats["elo"] == 0.0
    assert stats["elo_lo"] is None and stats["elo_hi"] is None
    assert any("no spread" in warning for warning in stats["warnings"])
    # No decisive pair is no evidence.
    assert stats["decisive_pairs"] == 0 and stats["sign_test_p"] == 1.0


def test_a_mixed_match_matches_the_hand_computed_paired_figures():
    # d = [+1, +1, 0, -1]: A sweeps two pairs, splits one, loses one.
    stats = paired_statistics(
        [_pair(1.0, 1.0), _pair(1.0, 1.0), _pair(1.0, 0.0), _pair(0.0, 0.0)]
    )

    assert stats["a_wins"] == 5.0 and stats["score"] == 0.625
    assert stats["score_as_p0"] == 3.0 and stats["score_as_p1"] == 2.0
    assert stats["pair_counts"] == {"a_both": 2, "split": 1, "b_both": 1, "capped": 0}
    assert stats["d_mean"] == 0.25
    # sd(d) = sqrt((0.75^2 + 0.75^2 + 0.25^2 + 1.25^2) / 3) = sqrt(11/12).
    assert stats["d_sd"] == pytest.approx(math.sqrt(11 / 12))
    assert stats["paired_se"] == pytest.approx(math.sqrt(11 / 12) / 2)
    # 2 * sd(five 1s and three 0s) / sqrt(8), the same estimand unpaired.
    assert stats["unpaired_se"] == pytest.approx(
        2 * math.sqrt(15 / 56) / math.sqrt(8)
    )
    assert stats["se_ratio"] == pytest.approx(
        stats["unpaired_se"] / stats["paired_se"]
    )
    # Two of three decisive pairs is the least extreme split there is: p = 1.
    assert stats["sign_test_p"] == 1.0
    assert stats["ci_lo"] < stats["score"] < stats["ci_hi"]
    assert stats["elo"] == pytest.approx(elo(0.625))
    # score + 1.96 * SE(score) exceeds 1 on four pairs, so the upper Elo bound is
    # genuinely unbounded — which is what an absent bound records.
    assert stats["elo_lo"] < stats["elo"] and stats["elo_hi"] is None
    assert stats["warnings"] == []


def test_sign_test_is_exact_and_two_sided():
    assert sign_test(0, 0) == 1.0
    assert sign_test(4, 4) == 2 / 16
    assert sign_test(0, 4) == 2 / 16
    assert sign_test(3, 5) == 1.0  # 2 * (10 + 5 + 1) / 32, clamped from 1.0
    assert sign_test(9, 10) == 22 / 1024
    assert sign_test(2, 3) == 1.0
    with pytest.raises(ValueError, match="within 0..decisive"):
        sign_test(3, 2)


def test_a_capped_game_is_flagged_and_is_not_a_decision():
    stats = paired_statistics(
        [_pair(1.0, 1.0), _pair(0.5, 1.0, capped=(True, False))]
    )

    assert stats["capped"] == 1
    assert stats["a_wins"] == 3.5 and stats["score"] == 0.875
    # The capped pair is counted apart: it is neither a sweep nor a split.
    assert stats["pair_counts"] == {"a_both": 1, "split": 0, "b_both": 0, "capped": 1}
    assert stats["decisive_pairs"] == 1
    assert stats["d_mean"] == 0.75  # (+1 and +0.5), so d left {-1, 0, +1}
    assert any("ply cap" in warning for warning in stats["warnings"])


def test_one_pair_has_no_paired_spread_and_says_so():
    stats = paired_statistics([_pair(1.0, 0.0)])

    assert stats["d_sd"] is None and stats["paired_se"] is None
    assert stats["se_ratio"] is None
    assert stats["elo_lo"] is None and stats["elo_hi"] is None
    assert any("at least two pairs" in warning for warning in stats["warnings"])


def test_paired_statistics_refuse_a_pair_that_is_not_two_games():
    with pytest.raises(ValueError, match="at least one pair"):
        paired_statistics([])
    with pytest.raises(ValueError, match=r"pairs \[1\]"):
        paired_statistics([_pair(1.0, 0.0), [_game(1.0, 0)]])


# --- end to end ------------------------------------------------------------


def test_head_to_head_is_deterministic_under_a_repeated_seed(checkpoints):
    first = _match(*checkpoints)
    second = _match(*checkpoints)

    # Wall time is the one field a rerun may not reproduce.
    for result in (first, second):
        result["match"].pop("seconds")
    assert first == second


def test_each_pair_plays_both_seats_from_its_own_shared_opening(checkpoints):
    result = _match(*checkpoints, pairs=3)

    assert result["pairs"] == 3 and result["games"] == 6
    assert result["a_wins"] == result["score_as_p0"] + result["score_as_p1"]
    assert bool(result["warnings"]) == (result["capped"] > 0)
    for row in result["per_pair"]:
        assert row["seats"] == [0, 1]
        assert 2 <= len(row["opening"]) <= 6
        assert row["d"] == row["scores"][0] + row["scores"][1] - 1.0
    assert len({tuple(map(tuple, row["opening"])) for row in result["per_pair"]}) == 3


def test_the_manifest_identifies_both_checkpoints_and_the_match(checkpoints):
    path_a, path_b = checkpoints
    result = _match(path_a, path_b, pairs=1, seed=5)

    for side, path, iteration in (("a", path_a, 1000), ("b", path_b, 2000)):
        assert result[side]["path"] == str(path)
        assert result[side]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert result[side]["iteration"] == iteration
        assert result[side]["versions"] == _versions()
        assert result[side]["versions"]["MODEL_REPR_VERSION"] == 7
    assert result["match"] == {
        "pairs": 1,
        "games": 2,
        "sims": 0,
        "tau": 0.1,
        "lam": 0.03,
        "temperature": 1.0,
        "opening_range": [2, 6],
        "ply_cap": 24,
        "device": "cpu",
        "seed": 5,
        "seconds": result["match"]["seconds"],
    }


def test_the_match_records_its_temperature_and_refuses_one_it_cannot_apply(
    checkpoints,
):
    # A score at one temperature says nothing about a score at another, so the
    # manifest carries it whether or not the match was searched. At sims == 0
    # there is no Gumbel to scale, and a request to scale it is an error rather
    # than a silently deterministic match.
    searched = _match(*checkpoints, pairs=1, sims=2, temperature=0.25)
    assert searched["match"]["temperature"] == 0.25

    try:
        _match(*checkpoints, pairs=1, sims=0, temperature=0.25)
    except ValueError as error:
        assert "sims > 0" in str(error)
    else:
        raise AssertionError("a temperature without a budget was accepted")


def test_the_cli_writes_strict_json_and_needs_coefficients_to_search(
    checkpoints, tmp_path
):
    path_a, path_b = checkpoints
    out = tmp_path / "h2h.json"
    main(
        [
            "--a", str(path_a), "--b", str(path_b),
            "--pairs", "1", "--sims", "0",
            "--cap", "20", "--device", "cpu", "--seed", "3",
            "--opening-range", "2", "3",
            "--out", str(out),
        ]
    )
    text = out.read_text(encoding="utf-8")
    assert "Infinity" not in text and "NaN" not in text
    written = json.loads(text)
    assert written["match"]["opening_range"] == [2, 3]
    assert written["games"] == 2
    # An unsearched match consulted no operating point, so it names none.
    assert written["match"]["tau"] is None and written["match"]["lam"] is None
    assert not (tmp_path / "h2h.tmp").exists()

    with pytest.raises(SystemExit):
        main(
            [
                "--a", str(path_a), "--b", str(path_b),
                "--pairs", "1", "--sims", "8", "--out", str(out),
            ]
        )
    # An unwritable destination is refused before the match, not after it.
    with pytest.raises(SystemExit):
        main(
            [
                "--a", str(path_a), "--b", str(path_b),
                "--pairs", "1", "--sims", "0", "--device", "cpu",
                "--out", str(tmp_path / "absent" / "h2h.json"),
            ]
        )


# --- refusals --------------------------------------------------------------


def test_head_to_head_refuses_a_missing_checkpoint(checkpoints, tmp_path):
    path_a, path_b = checkpoints
    absent = tmp_path / "absent.pt"
    with pytest.raises(FileNotFoundError, match=r"--a: no checkpoint at .*absent\.pt"):
        _match(absent, path_b)
    with pytest.raises(FileNotFoundError, match=r"--b: no checkpoint at .*absent\.pt"):
        _match(path_a, absent)


def test_head_to_head_refuses_an_unusable_match_shape(checkpoints):
    path_a, path_b = checkpoints
    with pytest.raises(ValueError, match="pairs must be >= 1, got 0"):
        _match(path_a, path_b, pairs=0)
    with pytest.raises(ValueError, match="sims must be >= 0, got -1"):
        _match(path_a, path_b, sims=-1)
    with pytest.raises(ValueError, match="needs both coefficients"):
        _match(path_a, path_b, sims=4, tau=None)
    # A cap at or below the longest opening ends every game on its opening.
    with pytest.raises(ValueError, match=r"must exceed the longest opening \(6\)"):
        _match(path_a, path_b, ply_cap=6)
    with pytest.raises(ValueError, match=r"must exceed the longest opening"):
        _match(path_a, path_b, ply_cap=0)
    with pytest.raises(ValueError, match=r"opening range must satisfy"):
        _match(path_a, path_b, opening_range=(2, 11))
    with pytest.raises(ValueError, match=r"opening range must satisfy"):
        _match(path_a, path_b, opening_range=(0, 4))


def test_head_to_head_refuses_a_representation_mismatch_and_names_the_bridge(
    checkpoints, tmp_path
):
    path_a, path_b = checkpoints
    older = _retagged(path_b, tmp_path / "v1.pt", MODEL_REPR_VERSION=1)
    with pytest.raises(ValueError, match=r"MODEL_REPR_VERSION") as refusal:
        _match(path_a, older)
    assert "cross-representation comparison is invalid" in str(refusal.value)


@pytest.mark.parametrize(
    "field, value",
    [("RULES_VERSION", 99), ("ACTION_ORDER_VERSION", 99), ("torch", "0.0.0")],
)
def test_head_to_head_refuses_an_incomparable_build(
    checkpoints, tmp_path, field, value
):
    path_a, path_b = checkpoints
    other = _retagged(path_b, tmp_path / f"{field}.pt", **{field: value})
    with pytest.raises(ValueError, match=field) as refusal:
        _match(path_a, other)
    # No conversion exists for these, so the refusal must not offer one.
    assert "graft" not in str(refusal.value)


def test_head_to_head_refuses_a_file_that_is_not_a_checkpoint(checkpoints, tmp_path):
    path_a, _path_b = checkpoints
    junk = tmp_path / "junk.pt"
    torch.save({"model": {}}, junk)
    with pytest.raises(ValueError, match="not a checkpoint"):
        _match(path_a, junk)


# --- driver_match: the in-driver instrument over the same statistics ---


def _driver_match(checkpoints, *, pairs=2, seed=7, **overrides):
    from mantisnet.klent.headtohead import driver_match
    from mantisnet.klent.run import load_model
    from mantisnet.klent.train import KlentConfig

    path_a, path_b = checkpoints
    settings = dict(
        cfg=KlentConfig(device="cpu"),
        pairs=pairs,
        sims=0,
        tau=0.1,
        lam=0.01,
        ply_cap=64,
        rng=np.random.default_rng([seed]),
    )
    settings.update(overrides)
    return driver_match(
        load_model(path_a),
        load_model(path_b),
        "h2h:run2/checkpoint_002000",
        {"checkpoint": str(path_b), "pairs": pairs},
        **settings,
    )


def test_driver_match_records_through_the_real_telemetry_writer(
    checkpoints, tmp_path
):
    """The result and games must fit ``write_eval_match`` as they are: the
    contract is the database row, not a key list."""
    from mantisnet.klent import telemetry as tel
    from mantisnet.klent.sealbot import record_match

    result, per_game, stats = _driver_match(checkpoints)
    assert result["games"] == len(per_game) == 4
    with tel.open_telemetry(tmp_path) as writer:
        match_id = record_match(
            writer, result, per_game, source="driver", iteration=50
        )
    import sqlite3

    with sqlite3.connect(tmp_path / "telemetry.db") as db:
        row = db.execute(
            "SELECT games, win_rate, score_as_p0 + score_as_p1, forfeits"
            " FROM eval_matches WHERE match_id = ?",
            (match_id,),
        ).fetchone()
        games = db.execute(
            "SELECT COUNT(*), SUM(model_seat) FROM games WHERE match = ?",
            (match_id,),
        ).fetchone()
    assert row == (4, result["win_rate"], result["score"], 0)
    # Pair-major seat swaps: two P0 games and two P1 games.
    assert games == (4, 2)


def test_driver_match_pairs_share_openings_and_swap_seats(checkpoints):
    result, per_game, stats = _driver_match(checkpoints, pairs=3)
    assert stats["pairs"] == 3 and result["games"] == 6
    for k in range(0, 6, 2):
        first, second = per_game[k], per_game[k + 1]
        assert (first["seat"], second["seat"]) == (0, 1)
        # One shared prefix per pair: the opening placements agree.
        length = first["opening_len"]
        assert second["opening_len"] == length
        assert first["moves"][:length] == second["moves"][:length]


def test_driver_match_is_deterministic_under_a_repeated_generator(checkpoints):
    one = _driver_match(checkpoints, seed=13)
    two = _driver_match(checkpoints, seed=13)
    assert [g["moves"] for g in one[1]] == [g["moves"] for g in two[1]]
    assert one[0]["score"] == two[0]["score"]
    assert one[2]["d_mean"] == two[2]["d_mean"]


def test_driver_match_refuses_a_single_pair(checkpoints):
    with pytest.raises(ValueError, match="pairs >= 2"):
        _driver_match(checkpoints, pairs=1)
