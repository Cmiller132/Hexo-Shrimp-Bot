"""Freeze and load replay-verified supervised corpora.

The source telemetry database is always opened through SQLite's read-only URI
mode.  Frozen positions remain move-prefix references; callers materialize the
move lists they need with :meth:`FrozenCorpus.moves_for` while preparing a
chunk, rather than constructing every prefix when the corpus is loaded.
"""

from __future__ import annotations

import hashlib
import json
import operator
import os
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hexo_py
import numpy as np

from ..klent import telemetry


FORMAT_VERSION = 1
SPLIT_NAMES = ("train", "val", "test")
SPLIT_IDS = {name: index for index, name in enumerate(SPLIT_NAMES)}
DEFAULT_FRACTIONS = (0.90, 0.05, 0.05)
DEFAULT_SAMPLE_COUNTS = (1_000_000, 100_000, 100_000)

_SELECTION_PREDICATE = "kind = 'selfplay' AND winner IS NOT NULL"
_GAME_ARRAYS = ("moves", "offsets", "winner", "source_game_id", "split")
_SAMPLE_FIELDS = ("game", "t", "rank", "mover", "z", "dist")
_SAMPLE_DTYPES = {
    "game": np.dtype("<i4"),
    "t": np.dtype("<i4"),
    "rank": np.dtype("<i4"),
    "mover": np.dtype("i1"),
    "z": np.dtype("i1"),
    "dist": np.dtype("<i4"),
}
_ARCHIVE_KEYS = frozenset(
    (*_GAME_ARRAYS, *(f"{split}_{field}" for split in SPLIT_NAMES for field in _SAMPLE_FIELDS))
)


@dataclass(frozen=True, slots=True)
class SampleSplit:
    """The six aligned sample arrays for one corpus split."""

    game: np.ndarray
    t: np.ndarray
    rank: np.ndarray
    mover: np.ndarray
    z: np.ndarray
    dist: np.ndarray

    def __len__(self) -> int:
        return int(self.game.shape[0])


@dataclass(frozen=True, slots=True)
class FrozenCorpus:
    """A validated corpus archive.

    ``game`` in a :class:`SampleSplit` is an index into the game-level arrays,
    not the telemetry database's primary key.  The latter is retained in
    :attr:`source_game_id` for provenance and diagnostics.
    """

    path: Path
    manifest: dict[str, Any]
    moves: np.ndarray
    offsets: np.ndarray
    winner: np.ndarray
    source_game_id: np.ndarray
    split: np.ndarray
    _samples: dict[str, SampleSplit]

    @property
    def name(self) -> str:
        return str(self.manifest["name"])

    @property
    def sha256(self) -> str:
        return str(self.manifest["corpus_sha256"])

    @property
    def n_games(self) -> int:
        return int(self.winner.shape[0])

    def split_samples(self, split: str) -> SampleSplit:
        """Return one split's aligned sample arrays."""

        try:
            return self._samples[split]
        except KeyError as exc:
            raise ValueError(
                f"corpus split must be one of {SPLIT_NAMES}, got {split!r}"
            ) from exc

    def moves_for(self, game: int) -> list[tuple[int, int]]:
        """Materialize one archived game's complete move sequence."""

        try:
            index = operator.index(game)
        except TypeError as exc:
            raise TypeError(f"game index must be an integer, got {game!r}") from exc
        if not 0 <= index < self.n_games:
            raise IndexError(f"game index {index} outside 0..{self.n_games - 1}")
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        return [(int(q), int(r)) for q, r in self.moves[start:end]]


@dataclass(frozen=True, slots=True)
class _SourceGame:
    source_game_id: int
    winner: int
    length: int
    moves: tuple[tuple[int, int], ...] | None


def _integer(name: str, value: Any, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer, got {value!r}") from exc
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _iteration_window(iters: tuple[int, int]) -> tuple[int, int]:
    if not isinstance(iters, (tuple, list)) or len(iters) != 2:
        raise ValueError(f"iters must be an inclusive (first, last) pair, got {iters!r}")
    first = _integer("first iteration", iters[0])
    last = _integer("last iteration", iters[1])
    if first > last:
        raise ValueError(f"iteration window starts after it ends: {first}..{last}")
    return first, last


def _fractions(values) -> tuple[float, float, float]:
    if not isinstance(values, (tuple, list)) or len(values) != len(SPLIT_NAMES):
        raise ValueError(
            f"fractions must contain train/val/test values, got {values!r}"
        )
    result = tuple(float(value) for value in values)
    if not all(np.isfinite(result)) or any(value < 0 for value in result):
        raise ValueError(f"split fractions must be finite and nonnegative: {result}")
    if not np.isclose(sum(result), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"split fractions must sum to 1, got {sum(result):.17g}")
    return result  # type: ignore[return-value]


def _open_source(run_dir: Path) -> sqlite3.Connection:
    db = run_dir / telemetry.DB_NAME
    if not db.is_file():
        raise FileNotFoundError(f"no telemetry database at {db}")
    # Path.as_uri quotes URI-significant path characters.  Do not use
    # telemetry.connect here: its writer-oriented connection sets WAL mode.
    uri = f"{db.resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def _schema_version(conn: sqlite3.Connection, db: Path) -> int:
    try:
        rows = conn.execute("SELECT version FROM schema_version").fetchall()
    except sqlite3.Error as exc:
        raise ValueError(f"{db} is not a versioned telemetry database: {exc}") from exc
    if len(rows) != 1:
        raise ValueError(
            f"{db} must contain exactly one telemetry schema version, found {len(rows)}"
        )
    found = int(rows[0][0])
    if found != telemetry.SCHEMA_VERSION:
        raise ValueError(
            f"{db} is telemetry schema v{found}, this build requires "
            f"v{telemetry.SCHEMA_VERSION}"
        )
    return found


def _select_games(
    conn: sqlite3.Connection,
    first: int,
    last: int,
    *,
    include_moves: bool,
) -> list[_SourceGame]:
    columns = "game_id, winner, length" + (", moves" if include_moves else "")
    rows = conn.execute(
        f"SELECT {columns} FROM games "
        "WHERE kind = 'selfplay' AND winner IS NOT NULL "
        "AND iteration BETWEEN ? AND ? ORDER BY game_id",
        (first, last),
    ).fetchall()
    if not rows:
        raise ValueError(
            f"empty corpus selection for {_SELECTION_PREDICATE}, iterations {first}..{last}"
        )

    games: list[_SourceGame] = []
    for row in rows:
        source_game_id = int(row["game_id"])
        winner = int(row["winner"])
        length = int(row["length"])
        if winner not in (0, 1):
            raise ValueError(f"game {source_game_id} has invalid winner {winner}")
        if length <= 0:
            raise ValueError(f"game {source_game_id} has invalid length {length}")
        moves = None
        if include_moves:
            try:
                unpacked = telemetry.unpack_moves(row["moves"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"game {source_game_id} has malformed moves: {exc}") from exc
            if len(unpacked) != length:
                raise ValueError(
                    f"game {source_game_id} stores length {length} but has "
                    f"{len(unpacked)} packed moves"
                )
            moves = tuple(unpacked)
        games.append(_SourceGame(source_game_id, winner, length, moves))
    return games


def _split_counts(n_games: int, fractions: tuple[float, float, float]) -> np.ndarray:
    exact = np.asarray(fractions, dtype=np.float64) * n_games
    counts = np.floor(exact).astype(np.int64)
    remainder = n_games - int(counts.sum())
    if remainder:
        # Stable sorting makes a fractional tie resolve train, then val, then
        # test.  This is recorded through the realized game counts.
        order = np.argsort(-(exact - counts), kind="stable")
        counts[order[:remainder]] += 1

    # Tiny corpora remain useful for smoke tests: when there are enough games,
    # every requested nonzero split gets one.  The donor is deterministic and
    # the exact realized counts remain part of the manifest.
    positive = np.flatnonzero(np.asarray(fractions) > 0)
    if n_games >= len(positive):
        for empty in positive[counts[positive] == 0]:
            donors = np.flatnonzero(counts > 1)
            donor = int(donors[np.argmax(counts[donors] - exact[donors])])
            counts[donor] -= 1
            counts[empty] += 1
    return counts


def _assign_splits(
    n_games: int,
    fractions: tuple[float, float, float],
    seed: int,
) -> np.ndarray:
    counts = _split_counts(n_games, fractions)
    shuffled = np.random.default_rng(seed).permutation(n_games)
    labels = np.empty(n_games, dtype=np.int8)
    start = 0
    for split_id, count in enumerate(counts):
        end = start + int(count)
        labels[shuffled[start:end]] = split_id
        start = end
    assert start == n_games
    return labels


def _empty_samples(size: int) -> SampleSplit:
    arrays = {
        field: np.zeros(size, dtype=dtype) for field, dtype in _SAMPLE_DTYPES.items()
    }
    return SampleSplit(**arrays)


def _sample_split(
    games: list[_SourceGame],
    labels: np.ndarray,
    split_id: int,
    target: int,
    rng: np.random.Generator,
) -> SampleSplit:
    game_indices = np.flatnonzero(labels == split_id)
    lengths = np.fromiter((games[int(i)].length for i in game_indices), dtype=np.int64)
    boundaries = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(lengths)))
    total = int(boundaries[-1])
    realized = min(target, total)
    samples = _empty_samples(realized)
    if realized == 0:
        return samples

    if realized == total:
        flat = np.arange(total, dtype=np.int64)
    else:
        flat = np.sort(rng.choice(total, size=realized, replace=False))
    local_game = np.searchsorted(boundaries[1:], flat, side="right")
    game = game_indices[local_game]
    t = flat - boundaries[local_game]
    if game.size and int(game.max()) > np.iinfo(np.int32).max:
        raise OverflowError("corpus contains more games than int32 sample indices allow")
    if t.size and int(t.max()) > np.iinfo(np.int32).max:
        raise OverflowError("a corpus game exceeds the int32 ply-index format")
    samples.game[:] = game.astype(np.int32)
    samples.t[:] = t.astype(np.int32)
    return samples


def _verify_samples(
    games: list[_SourceGame],
    samples_by_split: dict[str, SampleSplit],
) -> None:
    for split_name in SPLIT_NAMES:
        samples = samples_by_split[split_name]
        for game_index_raw in np.unique(samples.game):
            game_index = int(game_index_raw)
            game = games[game_index]
            assert game.moves is not None
            sample_rows = np.flatnonzero(samples.game == game_index)
            by_t = {int(samples.t[row]): int(row) for row in sample_rows}
            if len(by_t) != len(sample_rows):
                raise AssertionError(
                    f"sampling produced a duplicate position in game {game.source_game_id}"
                )

            pos = hexo_py.Position()
            for t, move in enumerate(game.moves):
                row = by_t.get(t)
                if row is not None:
                    mover = int(pos.current_player)
                    legal = pos.legal_moves()
                    try:
                        rank = legal.index(move)
                    except ValueError as exc:
                        raise ValueError(
                            f"game {game.source_game_id} ply {t}: stored move {move} "
                            "is absent from the engine legal order"
                        ) from exc
                    samples.rank[row] = rank
                    samples.mover[row] = mover
                    samples.z[row] = 1 if mover == game.winner else -1
                    samples.dist[row] = game.length - t
                try:
                    pos.advance(*move)
                except ValueError as exc:
                    raise ValueError(
                        f"game {game.source_game_id} replay failed at ply {t} "
                        f"for move {move}: {exc}"
                    ) from exc

            if not pos.is_terminal or int(pos.winner) != game.winner:
                raise ValueError(
                    f"game {game.source_game_id} replay winner mismatch: stored "
                    f"winner {game.winner}, terminal={pos.is_terminal}, "
                    f"engine winner={pos.winner}"
                )


def _counts_by_split(
    games: list[_SourceGame], labels: np.ndarray
) -> tuple[dict[str, int], dict[str, int]]:
    game_counts: dict[str, int] = {}
    ply_counts: dict[str, int] = {}
    lengths = np.fromiter((game.length for game in games), dtype=np.int64)
    for split_name, split_id in SPLIT_IDS.items():
        selected = labels == split_id
        game_counts[split_name] = int(selected.sum())
        ply_counts[split_name] = int(lengths[selected].sum())
    return game_counts, ply_counts


def _source_manifest(run_dir: Path, schema: int, first: int, last: int) -> dict[str, Any]:
    return {
        "run_dir": str(run_dir.resolve()),
        "telemetry_schema_version": schema,
        "iteration_window": [first, last],
        "selection_predicate": _SELECTION_PREDICATE,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_arrays(
    games: list[_SourceGame],
    labels: np.ndarray,
    samples_by_split: dict[str, SampleSplit],
) -> dict[str, np.ndarray]:
    lengths = np.fromiter((game.length for game in games), dtype=np.int64)
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(lengths)))
    moves = np.empty((int(offsets[-1]), 2), dtype=np.dtype("<i2"))
    for game_index, game in enumerate(games):
        assert game.moves is not None
        try:
            moves[int(offsets[game_index]) : int(offsets[game_index + 1])] = game.moves
        except (OverflowError, ValueError) as exc:
            raise ValueError(
                f"game {game.source_game_id} has a move outside the corpus int16 format"
            ) from exc
    try:
        source_ids = np.asarray(
            [game.source_game_id for game in games], dtype=np.dtype("<i8")
        )
    except (OverflowError, ValueError) as exc:
        raise ValueError("a source game id is outside the corpus int64 format") from exc

    arrays: dict[str, np.ndarray] = {
        "moves": moves,
        "offsets": offsets.astype(np.dtype("<i8"), copy=False),
        "winner": np.asarray([game.winner for game in games], dtype=np.int8),
        "source_game_id": source_ids,
        "split": labels.astype(np.int8, copy=False),
    }
    for split_name in SPLIT_NAMES:
        samples = samples_by_split[split_name]
        for field in _SAMPLE_FIELDS:
            arrays[f"{split_name}_{field}"] = getattr(samples, field)
    return arrays


def freeze(
    run_dir: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    iters: tuple[int, int],
    *,
    name: str | None = None,
    train_samples: int = DEFAULT_SAMPLE_COUNTS[0],
    val_samples: int = DEFAULT_SAMPLE_COUNTS[1],
    test_samples: int = DEFAULT_SAMPLE_COUNTS[2],
    seed: int = 0,
    fractions: tuple[float, float, float] = DEFAULT_FRACTIONS,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Freeze one inclusive self-play iteration window.

    A normal call writes ``manifest.json`` and ``corpus.npz`` into ``out_dir``
    and returns the manifest.  ``dry_run=True`` performs the read-only
    selection, split, and sampling-count plan, returns it with ``dry_run`` set,
    and creates no directories or files; the CLI prints that returned object.
    """

    run_path = Path(run_dir)
    output_path = Path(out_dir)
    corpus_name = output_path.name if name is None else str(name)
    if not corpus_name:
        raise ValueError("corpus name must not be empty")
    first, last = _iteration_window(iters)
    split_fractions = _fractions(fractions)
    targets = {
        "train": _integer("train_samples", train_samples),
        "val": _integer("val_samples", val_samples),
        "test": _integer("test_samples", test_samples),
    }
    sample_seed = _integer("seed", seed)

    if not dry_run and output_path.exists():
        if not output_path.is_dir():
            raise FileExistsError(f"corpus output path is not a directory: {output_path}")
        if any(output_path.iterdir()):
            raise FileExistsError(f"corpus output directory is not empty: {output_path}")

    db_path = run_path / telemetry.DB_NAME
    with closing(_open_source(run_path)) as conn:
        schema = _schema_version(conn, db_path)
        games = _select_games(conn, first, last, include_moves=not dry_run)

    labels = _assign_splits(len(games), split_fractions, sample_seed)
    game_counts, ply_counts = _counts_by_split(games, labels)
    realized = {
        split_name: min(targets[split_name], ply_counts[split_name])
        for split_name in SPLIT_NAMES
    }
    split_manifest = {
        "seed": sample_seed,
        "fractions": dict(zip(SPLIT_NAMES, split_fractions, strict=True)),
        "games": game_counts,
        "plies": ply_counts,
    }
    source_manifest = _source_manifest(run_path, schema, first, last)
    if dry_run:
        return {
            "dry_run": True,
            "name": corpus_name,
            "source": source_manifest,
            "split": split_manifest,
            "samples": {
                "seed": sample_seed,
                "requested": targets,
                "available": ply_counts,
                "realized": realized,
            },
        }

    rng = np.random.default_rng(sample_seed)
    samples_by_split = {
        split_name: _sample_split(
            games, labels, split_id, targets[split_name], rng
        )
        for split_name, split_id in SPLIT_IDS.items()
    }
    _verify_samples(games, samples_by_split)
    arrays = _archive_arrays(games, labels, samples_by_split)

    output_path.mkdir(parents=True, exist_ok=True)
    archive_path = output_path / "corpus.npz"
    manifest_path = output_path / "manifest.json"
    archive_tmp = output_path / ".corpus.npz.tmp"
    manifest_tmp = output_path / ".manifest.json.tmp"
    try:
        with archive_tmp.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        corpus_sha256 = _sha256(archive_tmp)
        manifest: dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "name": corpus_name,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": source_manifest,
            "split": split_manifest,
            "samples": {
                "seed": sample_seed,
                "requested": targets,
                "available": ply_counts,
                "realized": {name: len(samples_by_split[name]) for name in SPLIT_NAMES},
            },
            "RULES_VERSION": int(hexo_py.RULES_VERSION),
            "ACTION_ORDER_VERSION": int(hexo_py.ACTION_ORDER_VERSION),
            "corpus_sha256": corpus_sha256,
        }
        with manifest_tmp.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(archive_tmp, archive_path)
        os.replace(manifest_tmp, manifest_path)
    finally:
        archive_tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)
    return manifest


def _manifest_object(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read corpus manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"corpus manifest {path} must be a JSON object")
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"corpus format version {manifest.get('format_version')!r} != "
            f"this build {FORMAT_VERSION}"
        )
    for key, current in (
        ("RULES_VERSION", int(hexo_py.RULES_VERSION)),
        ("ACTION_ORDER_VERSION", int(hexo_py.ACTION_ORDER_VERSION)),
    ):
        found = manifest.get(key)
        if found != current:
            raise ValueError(f"corpus {key} {found!r} != this engine {current}")
    if not isinstance(manifest.get("name"), str) or not manifest["name"]:
        raise ValueError("corpus manifest has no nonempty name")
    expected_sha = manifest.get("corpus_sha256")
    if (
        not isinstance(expected_sha, str)
        or len(expected_sha) != 64
        or any(c not in "0123456789abcdef" for c in expected_sha)
    ):
        raise ValueError(f"corpus manifest has invalid corpus_sha256 {expected_sha!r}")
    return manifest


def _archive_object(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False, mmap_mode="r") as archive:
            keys = frozenset(archive.files)
            if keys != _ARCHIVE_KEYS:
                missing = sorted(_ARCHIVE_KEYS - keys)
                extra = sorted(keys - _ARCHIVE_KEYS)
                raise ValueError(
                    f"corpus archive fields differ: missing={missing}, extra={extra}"
                )
            arrays = {key: np.asarray(archive[key]) for key in _ARCHIVE_KEYS}
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("corpus archive fields"):
            raise
        raise ValueError(f"cannot load corpus archive {path}: {exc}") from exc
    return arrays


def _one_dimensional(
    arrays: dict[str, np.ndarray], key: str, dtype: np.dtype
) -> np.ndarray:
    value = arrays[key]
    if value.ndim != 1 or value.dtype != dtype:
        raise ValueError(
            f"corpus {key} must be a one-dimensional {dtype} array, got "
            f"shape {value.shape} dtype {value.dtype}"
        )
    return value


def _validate_loaded(
    manifest: dict[str, Any], arrays: dict[str, np.ndarray]
) -> tuple[dict[str, SampleSplit], int]:
    moves = arrays["moves"]
    if moves.ndim != 2 or moves.shape[1:] != (2,) or moves.dtype != np.dtype("<i2"):
        raise ValueError(
            "corpus moves must have shape (total_plies, 2) and dtype int16, "
            f"got shape {moves.shape} dtype {moves.dtype}"
        )
    offsets = _one_dimensional(arrays, "offsets", np.dtype("<i8"))
    if offsets.size < 2:
        raise ValueError("corpus must contain at least one game")
    if int(offsets[0]) != 0 or int(offsets[-1]) != len(moves):
        raise ValueError(
            f"corpus offsets must span 0..{len(moves)}, got "
            f"{int(offsets[0])}..{int(offsets[-1])}"
        )
    if np.any(offsets[1:] <= offsets[:-1]):
        raise ValueError("corpus offsets must give every game a positive length")
    n_games = len(offsets) - 1
    winner = _one_dimensional(arrays, "winner", np.dtype("i1"))
    source_ids = _one_dimensional(arrays, "source_game_id", np.dtype("<i8"))
    labels = _one_dimensional(arrays, "split", np.dtype("i1"))
    for key, value in (("winner", winner), ("source_game_id", source_ids), ("split", labels)):
        if len(value) != n_games:
            raise ValueError(f"corpus {key} has {len(value)} rows for {n_games} games")
    if np.any((winner < 0) | (winner > 1)):
        raise ValueError("corpus winner values must be 0 or 1")
    if len(np.unique(source_ids)) != n_games:
        raise ValueError("corpus source_game_id values must be unique")
    if np.any((labels < 0) | (labels >= len(SPLIT_NAMES))):
        raise ValueError("corpus split values must be 0, 1, or 2")

    samples_by_split: dict[str, SampleSplit] = {}
    for split_name, split_id in SPLIT_IDS.items():
        fields = {
            field: _one_dimensional(
                arrays, f"{split_name}_{field}", _SAMPLE_DTYPES[field]
            )
            for field in _SAMPLE_FIELDS
        }
        sizes = {len(value) for value in fields.values()}
        if len(sizes) != 1:
            raise ValueError(
                f"corpus {split_name} sample arrays have unequal lengths {sorted(sizes)}"
            )
        samples = SampleSplit(**fields)
        if len(samples):
            if np.any((samples.game < 0) | (samples.game >= n_games)):
                raise ValueError(f"corpus {split_name} has an out-of-range game index")
            if np.any(labels[samples.game] != split_id):
                raise ValueError(f"corpus {split_name} samples reference another split")
            starts = offsets[samples.game]
            lengths = offsets[samples.game + 1] - starts
            if np.any((samples.t < 0) | (samples.t >= lengths)):
                raise ValueError(f"corpus {split_name} has an out-of-range ply index")
            if np.any(samples.rank < 0):
                raise ValueError(f"corpus {split_name} has a negative legal rank")
            if np.any((samples.mover < 0) | (samples.mover > 1)):
                raise ValueError(f"corpus {split_name} mover values must be 0 or 1")
            if np.any((samples.z != -1) & (samples.z != 1)):
                raise ValueError(f"corpus {split_name} z values must be -1 or +1")
            if np.any(samples.dist != lengths - samples.t):
                raise ValueError(f"corpus {split_name} distance values are inconsistent")
            pairs = np.stack((samples.game, samples.t), axis=1)
            if len(np.unique(pairs, axis=0)) != len(samples):
                raise ValueError(f"corpus {split_name} contains duplicate samples")
        samples_by_split[split_name] = samples

    try:
        split_meta = manifest["split"]
        sample_meta = manifest["samples"]
        expected_games = split_meta["games"]
        expected_plies = split_meta["plies"]
        requested_samples = sample_meta["requested"]
        available_samples = sample_meta["available"]
        expected_samples = sample_meta["realized"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"corpus manifest count structure is malformed: {exc}") from exc
    if not all(
        isinstance(value, dict)
        for value in (
            expected_games,
            expected_plies,
            requested_samples,
            available_samples,
            expected_samples,
        )
    ):
        raise ValueError("corpus manifest game, ply, and sample counts must be objects")
    lengths = offsets[1:] - offsets[:-1]
    for split_name, split_id in SPLIT_IDS.items():
        selected = labels == split_id
        actual_games = int(selected.sum())
        actual_plies = int(lengths[selected].sum())
        actual_samples = len(samples_by_split[split_name])
        requested = requested_samples.get(split_name)
        if (
            isinstance(requested, bool)
            or not isinstance(requested, int)
            or requested < 0
        ):
            raise ValueError(
                f"corpus manifest {split_name} requested sample count "
                f"is invalid: {requested!r}"
            )
        for kind, expected, actual in (
            ("games", expected_games.get(split_name), actual_games),
            ("plies", expected_plies.get(split_name), actual_plies),
            ("available samples", available_samples.get(split_name), actual_plies),
            ("samples", expected_samples.get(split_name), actual_samples),
        ):
            if expected != actual:
                raise ValueError(
                    f"corpus manifest {split_name} {kind} count {expected!r} != {actual}"
                )
        if actual_samples != min(requested, actual_plies):
            raise ValueError(
                f"corpus manifest {split_name} realized samples {actual_samples} "
                f"!= min(requested={requested}, available={actual_plies})"
            )
    return samples_by_split, n_games


def load_corpus(path: str | os.PathLike[str]) -> FrozenCorpus:
    """Load a frozen corpus after verifying its digest and version pins."""

    corpus_path = Path(path)
    if not corpus_path.is_dir():
        raise FileNotFoundError(f"no corpus directory at {corpus_path}")
    entries = {entry.name for entry in corpus_path.iterdir()}
    expected_entries = {"manifest.json", "corpus.npz"}
    if entries != expected_entries:
        raise ValueError(
            f"corpus directory fields differ: missing="
            f"{sorted(expected_entries - entries)}, extra={sorted(entries - expected_entries)}"
        )
    manifest_path = corpus_path / "manifest.json"
    archive_path = corpus_path / "corpus.npz"
    manifest = _manifest_object(manifest_path)
    if not archive_path.is_file():
        raise FileNotFoundError(f"no corpus archive at {archive_path}")
    actual_sha = _sha256(archive_path)
    if actual_sha != manifest["corpus_sha256"]:
        raise ValueError(
            f"corpus sha256 mismatch: manifest {manifest['corpus_sha256']}, "
            f"archive {actual_sha}"
        )
    arrays = _archive_object(archive_path)
    samples, _ = _validate_loaded(manifest, arrays)
    for value in arrays.values():
        value.setflags(write=False)
    return FrozenCorpus(
        path=corpus_path,
        manifest=manifest,
        moves=arrays["moves"],
        offsets=arrays["offsets"],
        winner=arrays["winner"],
        source_game_id=arrays["source_game_id"],
        split=arrays["split"],
        _samples=samples,
    )
