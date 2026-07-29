"""The run's telemetry database: every iteration, game, and ply, queryable.

`metrics.jsonl` is the ``KLENT_FOR_HEXO.md`` §8 record and stays the run's
ground truth. This is
the substrate beside it — one `runs/<name>/telemetry.db` per run — holding
what the metrics row averaged away: which games were played and how they
went, what the policy believed at each ply, how a checkpoint scored against
each opponent, and what the machine was doing while it happened.

**π′ is not stored.** The improved policy is `|A_legal|` floats per ply,
kilobytes where the ply's row is tens of bytes, and it is reproducible
exactly from a checkpoint and a move prefix — which is what
`inspect.inspect_position` does. The database keeps the four scalars a
query needs (its KL to π_θ, its normalized entropy, its maximum, and its
value at the sampled move) and recomputes the rest on demand. That is the
storage decision this schema is built around; everything else follows from
having chosen scalars over arrays. Evaluation's model-side Gumbel simulation
count is recorded in ``config.json`` and ``invocations.jsonl``, not duplicated
in this telemetry schema.

Writes are one transaction per iteration into a WAL database, so a crash
loses at most the iteration in flight and never the file, and a dashboard
can read the run while it is still being written.

The schema is versioned and there are no migrations: a database this build
did not write is refused rather than adapted to, and the fix is to
regenerate it.

CLI::

    python -m mantisnet.klent.telemetry --run runs/<name> summary
    python -m mantisnet.klent.telemetry --run runs/<name> games --limit 20
    python -m mantisnet.klent.telemetry --run runs/<name> game 1234
"""

from __future__ import annotations

import itertools
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Covers every table, column, and packing below. Bumped, never migrated.
SCHEMA_VERSION = 2

DB_NAME = "telemetry.db"

# Per-ply scalars are stored quantized: integers in units of 1/_Q (v2). The
# five REALs were 40 bytes of a ~71-byte row and SQLite has no float32; at
# 1e-4 resolution a typical value is a 2-3 byte varint and no consumer's
# noise floor is anywhere near the rounding. Writers multiply, readers
# divide, and both live in this file — nothing outside sees an integer.
_Q = 10_000

# The five quantized columns, in schema order.
_PLY_SCALARS = ("v_hat", "kl", "norm_entropy", "pi_top1", "pi_chosen")

# The KLENT_FOR_HEXO.md §8 metrics that become their own iteration columns. Everything in the
# metrics row is also kept verbatim in ``metrics_json``; these are the ones a
# threshold query ("where did calibration collapse?") must not have to parse.
_ITERATION_METRICS = (
    ("f", "REAL"),
    ("acting_kl", "REAL"),
    ("acting_norm_entropy", "REAL"),
    ("won_length_mean", "REAL"),
    ("p0_win_rate", "REAL"),
    ("first_stone_win_rate", "REAL"),
    ("v_hat_winner_mean", "REAL"),
    ("v_hat_loser_mean", "REAL"),
    ("v_hat_mae", "REAL"),
    ("buffer_samples", "INTEGER"),
    ("policy_loss", "REAL"),
    ("q_loss", "REAL"),
    ("fit_steps", "INTEGER"),
    ("seconds", "REAL"),
    ("eval_score", "REAL"),
    ("eval_capped", "INTEGER"),
    ("eval_games", "INTEGER"),
    ("eval_seconds", "REAL"),
)

# What `hardware.HardwareSampler.drain` produces. GPU columns are empty on a
# CPU run — there is no sensor, which is not the same as a zero.
_HARDWARE_COLUMNS = (
    ("hw_samples", "INTEGER"),
    ("cpu_percent_mean", "REAL"),
    ("cpu_percent_max", "REAL"),
    ("threads_mean", "REAL"),
    ("threads_max", "INTEGER"),
    ("rss_mean", "REAL"),
    ("rss_max", "INTEGER"),
    ("sys_ram_used_mean", "REAL"),
    ("sys_ram_used_max", "INTEGER"),
    ("gpu_util_mean", "REAL"),
    ("gpu_util_max", "REAL"),
    ("gpu_power_w_mean", "REAL"),
    ("gpu_power_w_max", "REAL"),
    ("gpu_mem_used_mean", "REAL"),
    ("gpu_mem_used_max", "INTEGER"),
    ("gpu_temp_mean", "REAL"),
    ("gpu_temp_max", "REAL"),
    ("torch_alloc_max", "INTEGER"),
    ("torch_reserved_max", "INTEGER"),
)


def _columns(spec) -> str:
    return ",\n    ".join(f"{name} {kind}" for name, kind in spec)


_SCHEMA = f"""
CREATE TABLE schema_version (version INTEGER NOT NULL);

-- One row per process that trained into this run: the database's mirror of
-- invocations.jsonl, so a metric's meaning can be traced to the knobs live
-- when it was written.
CREATE TABLE runs (
    id              INTEGER PRIMARY KEY,
    created         TEXT NOT NULL,
    start_iteration INTEGER NOT NULL,
    iterations      INTEGER NOT NULL,
    config_json     TEXT NOT NULL,
    versions_json   TEXT NOT NULL
);

CREATE TABLE iterations (
    iteration    INTEGER PRIMARY KEY,
    run          INTEGER NOT NULL REFERENCES runs(id),
    {_columns(_ITERATION_METRICS)},
    samples_per_s REAL,
    games        INTEGER NOT NULL,
    plies        INTEGER NOT NULL,
    {_columns(_HARDWARE_COLUMNS)},
    metrics_json TEXT NOT NULL
);

-- Whoever a checkpoint was measured against. The opponent's own knobs live
-- in config_json rather than in columns here: SealBot is the yardstick
-- today and will not be the only one, and a stronger engine arriving must
-- not need a schema change to be recorded.
CREATE TABLE opponents (
    opponent_id INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    config_json TEXT NOT NULL,
    UNIQUE (name, config_json)
);

CREATE TABLE eval_matches (
    match_id   INTEGER PRIMARY KEY,
    created    TEXT NOT NULL,
    source     TEXT NOT NULL,   -- 'driver' (in-loop) | 'cli' (offline sweep)
    opponent   INTEGER NOT NULL REFERENCES opponents(opponent_id),
    iteration  INTEGER,         -- driver: iterations completed; cli: from the checkpoint name
    checkpoint TEXT,            -- cli only; the driver measures live weights
    games      INTEGER NOT NULL,
    score      REAL NOT NULL,
    win_rate   REAL NOT NULL,
    capped     INTEGER NOT NULL,
    ci_lo REAL, ci_hi REAL, elo REAL, elo_lo REAL, elo_hi REAL,
    score_as_p0 REAL, score_as_p1 REAL,
    forfeits INTEGER,          -- opponent's unplayable proposals, scored as wins
    opponent_retries INTEGER,  -- recovered re-consultations behind them
    opponent_depth_mean REAL,
    avg_plies REAL,
    seconds REAL
);

-- Every game the run played, self-play and evaluation alike, so one query
-- browses both. `moves` is the packing of `pack_moves`.
CREATE TABLE games (
    game_id     INTEGER PRIMARY KEY,
    kind        TEXT NOT NULL,   -- 'selfplay' | 'eval'
    iteration   INTEGER,         -- self-play: the iteration whose fit consumed it
    match       INTEGER REFERENCES eval_matches(match_id),
    game_index  INTEGER NOT NULL,
    winner      INTEGER,         -- NULL exactly when the game hit the cap
    length      INTEGER NOT NULL,
    capped      INTEGER NOT NULL,
    model_seat  INTEGER,         -- eval only: the seat the model took
    opening_len INTEGER,         -- eval only: shared random opening, in plies
    opponent_depth_mean REAL,    -- eval only: how deep the opponent searched
    moves       BLOB NOT NULL
);

CREATE UNIQUE INDEX games_selfplay_key ON games(iteration, game_index)
    WHERE kind = 'selfplay';
CREATE UNIQUE INDEX games_eval_key ON games(match, game_index)
    WHERE kind = 'eval';
-- Serves the iteration range of a browse that names no kind, and the
-- iteration sweeps `begin_run` deletes by. The game browser's own indexes
-- are `_GAME_INDEXES`, applied by `_index_games` rather than declared here.
CREATE INDEX games_iteration ON games(iteration);

-- Acting-time scalars, self-play only: an evaluation game plays argmax
-- π_θ and has no π′ to reduce.
-- The five scalar columns hold integers in units of 1/10000 (see _Q): the
-- write path quantizes, every read path in this file dequantizes.
CREATE TABLE plies (
    game_id         INTEGER NOT NULL REFERENCES games(game_id),
    t               INTEGER NOT NULL,
    mover           INTEGER NOT NULL,
    moves_remaining INTEGER NOT NULL,
    legal_count     INTEGER NOT NULL,
    rank            INTEGER NOT NULL,
    v_hat           INTEGER NOT NULL,
    kl              INTEGER NOT NULL,
    norm_entropy    INTEGER NOT NULL,
    pi_top1         INTEGER NOT NULL,
    pi_chosen       INTEGER NOT NULL,
    PRIMARY KEY (game_id, t)
) WITHOUT ROWID;

-- The A7 round-robin, replaced wholesale by each run of it, exactly as
-- crossplay.json is.
CREATE TABLE crossplay (
    checkpoint_a TEXT NOT NULL,
    checkpoint_b TEXT NOT NULL,
    games        INTEGER NOT NULL,
    score_a      REAL NOT NULL,
    capped       INTEGER NOT NULL,
    ply_cap      INTEGER NOT NULL,
    seed         INTEGER NOT NULL,
    created      TEXT NOT NULL,
    PRIMARY KEY (checkpoint_a, checkpoint_b)
);
"""


# --------------------------------------------------------------------------
# Move packing


def pack_moves(moves) -> bytes:
    """A move list to the `games.moves` blob.

    The packing, defined here and nowhere else: little-endian ``int16``
    ``(q, r)`` pairs in play order, four bytes a ply and nothing more, so
    ``len(blob) // 4`` is the game's length. Coordinates outside ``int16``
    are refused rather than wrapped — a real game stays within a few dozen
    steps of the origin, and a coordinate that does not is a bug worth
    hearing about.
    """
    arr = np.asarray(moves, dtype=np.int64).reshape(-1, 2)
    if arr.size and (arr.min() < -32768 or arr.max() > 32767):
        raise ValueError(f"move coordinates outside int16: {arr.min()}..{arr.max()}")
    return arr.astype("<i2").tobytes()


def unpack_moves(blob: bytes) -> list[tuple[int, int]]:
    """The inverse of :func:`pack_moves`."""
    if len(blob) % 4:
        raise ValueError(f"{len(blob)}-byte moves blob is not whole (q, r) pairs")
    return [(int(q), int(r)) for q, r in np.frombuffer(blob, dtype="<i2").reshape(-1, 2)]


# --------------------------------------------------------------------------
# The board's symmetries, for opening aggregation


def _d6_transforms():
    """The 12 symmetries of the board as maps on ``(q, r)``; index 0 is the
    identity.

    Generators are the 60-degree rotation ``(q, r) -> (-r, q + r)`` and the
    reflection ``(q, r) -> (r, q)``. Both permute the three window axes and
    preserve hex distance, so the rules are invariant under all twelve — a
    transformed game is legal placement for placement and ends with the same
    winner, which is what `tests/test_telemetry.py` holds them to.
    """

    def rot(m):
        return (-m[1], m[0] + m[1])

    def ref(m):
        return (m[1], m[0])

    out = []
    for base in (lambda m: m, ref):
        f = base
        for _ in range(6):
            out.append(f)
            f = (lambda g: lambda m: rot(g(m)))(f)
    return out


D6_TRANSFORMS = _d6_transforms()


def canonical_opening(moves, plies: int | None = None) -> tuple:
    """A move sequence's representative under the board's symmetries: the
    lexicographic minimum over all twelve transforms of it.

    Every game opens at the origin and the rules are invariant about it, so
    two openings differing by a rotation or a reflection are one opening;
    counting raw move lists would split each of them twelve ways. ``plies``
    truncates first, which is what an atlas over the first few placements
    wants.
    """
    seq = list(moves)[:plies] if plies is not None else list(moves)
    return min(tuple(t(m) for m in seq) for t in D6_TRANSFORMS)


# --------------------------------------------------------------------------
# Opening, versioning


def db_path(run_dir) -> Path:
    return Path(run_dir) / DB_NAME


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# What `search_games` needs to answer a page without reading the table:
# one index per entry of GAME_ORDERS, named after it. Each leads with
# `kind`, since every browse filters on it, and then repeats that order's
# sort key column for column, in the direction the order reads each one —
# so the whole ORDER BY comes off the index and only the page's rows are
# ever touched. Without them SQLite has no ordered path into `games` and
# sorts every matching row to hand back fifty: measured on a 575,342-game
# run, ~480 ms warm on a local disk and ~44 s through the deck's mount, per
# request.
#
# One index per order rather than a shared pair, because sharing depends on
# the planner turning an index around and sorting only the trailing term of
# the sort key inside each tie group. It will, sometimes: SQLite 3.45 does
# it for both reversed orders, 3.53 for one of them and a full sort for the
# other. Matching each order exactly costs ~14 MB apiece on a run that size
# and is a decision the planner cannot get wrong.
#
# `winner`, `capped`, and the length bounds stay out of them: the ordered
# scan tests those per row as it walks, which is free at fifty rows,
# whereas indexing them would mean an index per combination of filters. An
# iteration range conjoined with a *length* order is the one shape still
# answered by a sort — no index can lead with both columns — and there the
# sort is bounded by the range the filter selected.
#
# `games_search(kind, winner, length)` is dropped by the same list: it can
# satisfy a filter but never an order, so with these present the planner has
# no reason to pick it, and an index the planner never picks is weight on
# every insert.
_GAME_INDEXES = (
    "CREATE INDEX IF NOT EXISTS games_browse_recent"
    " ON games(kind, iteration DESC, game_index)",
    "CREATE INDEX IF NOT EXISTS games_browse_oldest"
    " ON games(kind, iteration, game_index)",
    "CREATE INDEX IF NOT EXISTS games_browse_longest"
    " ON games(kind, length DESC, iteration DESC, game_index)",
    "CREATE INDEX IF NOT EXISTS games_browse_shortest"
    " ON games(kind, length, iteration DESC, game_index)",
    "DROP INDEX IF EXISTS games_search",
)


def _index_games(conn: sqlite3.Connection) -> None:
    """Bring a database's `games` indexes up to `_GAME_INDEXES`.

    Called from `Telemetry.__init__` alone, which is the only path that
    holds a writable connection — the deck opens telemetry read-only, and a
    reader cannot create an index. Idempotent, because a database written
    before these indexes existed has to gain them the next time a writer
    opens it, and one written after must not be touched twice.

    This is deliberately not a schema version. An index is derived from
    rows already in the file and stores no format of its own, so a database
    with these and one without hold the same bytes and answer the same
    queries — SCHEMA_VERSION stays where it is, and every run database ever
    written stays readable by this build.
    """
    with conn:
        for statement in _GAME_INDEXES:
            conn.execute(statement)


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    existing = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'schema_version'"
    ).fetchone()
    if existing is None:
        with conn:
            conn.executescript(_SCHEMA)
            conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,)
            )
        return conn
    found = conn.execute("SELECT version FROM schema_version").fetchone()[0]
    if found != SCHEMA_VERSION:
        conn.close()
        raise ValueError(
            f"{path} is telemetry schema v{found}, this build writes "
            f"v{SCHEMA_VERSION} — there are no migrations; delete it and let "
            "the run rebuild, or read it with the build that wrote it"
        )
    return conn


def connect(run_dir) -> sqlite3.Connection:
    """A connection to an existing run's telemetry database, for reading.

    A run directory with no database is an error, not an empty result: the
    caller asked about a run that was never captured.
    """
    path = db_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"no telemetry database at {path}")
    return _connect(path)


def open_telemetry(run_dir) -> "Telemetry":
    """The writer for a run directory, creating the database if it is new."""
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    return Telemetry(db_path(run_dir))


# --------------------------------------------------------------------------
# The writer


class Telemetry:
    """One connection, one transaction per iteration.

    The connection belongs to the thread that opened it — sqlite3's default
    — so a stray write from the collection worker fails loudly instead of
    interleaving with the driver's transaction.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._conn = _connect(self.path)
        # The only moment a run's telemetry is open for writing, and so the
        # only moment a database written by an earlier build can be given
        # the game browser's indexes.
        _index_games(self._conn)
        self._run: int | None = None
        self._next_game_id = self._max_game_id() + 1

    def __enter__(self) -> "Telemetry":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    def _max_game_id(self) -> int:
        return self._conn.execute(
            "SELECT COALESCE(MAX(game_id), -1) FROM games"
        ).fetchone()[0]

    def _game_ids(self, n: int) -> range:
        """A contiguous block of ids, so a whole iteration's games and plies
        can be built before any of them is inserted."""
        base = self._next_game_id
        self._next_game_id += n
        return range(base, base + n)

    # -- the run driver ----------------------------------------------------

    def begin_run(self, config: dict, versions: dict, start_iteration: int) -> None:
        """Record this process, and drop the tail it is about to replay.

        A resume restarts at its checkpoint's iteration, which may be behind
        the last iteration recorded — those rows describe work about to be
        redone. They go, with their games and their plies, rather than
        colliding with the replay or double-counting it. An id from the
        discarded tail may then be handed out again, which is correct: the
        row it named no longer exists, and every id in the database still
        names exactly one game. Evaluation matches from the offline CLI are
        keyed by checkpoint rather than by the loop's position, and survive.
        """
        with self._conn:
            self._conn.execute(
                "DELETE FROM plies WHERE game_id IN"
                " (SELECT game_id FROM games WHERE kind = 'selfplay' AND iteration >= ?)",
                (start_iteration,),
            )
            self._conn.execute(
                "DELETE FROM games WHERE kind = 'selfplay' AND iteration >= ?",
                (start_iteration,),
            )
            self._conn.execute(
                "DELETE FROM games WHERE kind = 'eval' AND match IN"
                " (SELECT match_id FROM eval_matches"
                "  WHERE source = 'driver' AND iteration >= ?)",
                (start_iteration,),
            )
            self._conn.execute(
                "DELETE FROM eval_matches WHERE source = 'driver' AND iteration >= ?",
                (start_iteration,),
            )
            self._conn.execute(
                "DELETE FROM iterations WHERE iteration >= ?", (start_iteration,)
            )
            cur = self._conn.execute(
                "INSERT INTO runs (created, start_iteration, iterations,"
                " config_json, versions_json) VALUES (?, ?, ?, ?, ?)",
                (
                    _now(),
                    start_iteration,
                    config["iterations"],
                    json.dumps(config, sort_keys=True),
                    json.dumps(versions, sort_keys=True),
                ),
            )
            self._run = cur.lastrowid
        self._next_game_id = self._max_game_id() + 1

    def write_iteration(self, metrics: dict, episodes: list, hardware: dict) -> None:
        """One iteration's whole record — its metrics row, its games, and
        their plies — in one transaction.

        ``metrics`` is the row as it went to `metrics.jsonl` (NaN already
        None), so the two records cannot disagree.
        """
        if self._run is None:
            raise RuntimeError("begin_run() must precede write_iteration()")
        unknown = set(hardware) - {name for name, _ in _HARDWARE_COLUMNS}
        if unknown:
            raise ValueError(f"hardware columns not in the schema: {sorted(unknown)}")

        iteration = metrics["iteration"]
        ids = self._game_ids(len(episodes))
        plies = sum(len(e.ranks) for e in episodes)
        seconds, samples = metrics.get("seconds"), metrics.get("buffer_samples")
        per_s = samples / seconds if samples is not None and seconds else None

        names = [name for name, _ in _ITERATION_METRICS]
        values = [metrics.get(name) for name in names]
        names += ["samples_per_s", "games", "plies"]
        values += [per_s, len(episodes), plies]
        for name, _kind in _HARDWARE_COLUMNS:
            names.append(name)
            values.append(hardware.get(name))
        names += ["metrics_json"]
        values += [json.dumps(metrics, allow_nan=False, sort_keys=True)]

        with self._conn:
            self._conn.execute(
                f"INSERT INTO iterations (iteration, run, {', '.join(names)})"
                f" VALUES (?, ?, {', '.join('?' * len(names))})",
                [iteration, self._run, *values],
            )
            self._conn.executemany(
                "INSERT INTO games (game_id, kind, iteration, game_index, winner,"
                " length, capped, moves) VALUES (?, 'selfplay', ?, ?, ?, ?, ?, ?)",
                (
                    (
                        game_id,
                        iteration,
                        index,
                        ep.winner,
                        len(ep.moves),
                        int(ep.winner is None),
                        pack_moves(ep.moves),
                    )
                    for index, (game_id, ep) in enumerate(zip(ids, episodes))
                ),
            )
            self._conn.executemany(
                "INSERT INTO plies (game_id, t, mover, moves_remaining, legal_count,"
                " rank, v_hat, kl, norm_entropy, pi_top1, pi_chosen)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                _ply_rows(ids, episodes),
            )

    # -- evaluation --------------------------------------------------------

    def opponent(self, name: str, config: dict) -> int:
        """The id of an opponent with this name and these knobs, created on
        first sight. Identical knobs reuse the row, so a strength curve is a
        query over one opponent id rather than a string match."""
        blob = json.dumps(config, sort_keys=True)
        with self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO opponents (name, config_json) VALUES (?, ?)",
                (name, blob),
            )
        return self._conn.execute(
            "SELECT opponent_id FROM opponents WHERE name = ? AND config_json = ?",
            (name, blob),
        ).fetchone()[0]

    def write_eval_match(
        self,
        opponent_id: int,
        result: dict,
        per_game: list[dict],
        *,
        source: str,
        iteration: int | None = None,
        checkpoint: str | None = None,
        depth_mean: float | None = None,
    ) -> int:
        """One match against one opponent: its summary and its games.

        ``result`` carries the opponent-independent scalars every match
        produces; whatever made *this* opponent what it is belongs in its
        `opponents` row, not here.
        """
        if source not in ("driver", "cli", "deck"):
            raise ValueError(
                f"eval match source must be 'driver', 'cli', or 'deck': {source!r}"
            )
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO eval_matches (created, source, opponent, iteration,"
                " checkpoint, games, score, win_rate, capped, ci_lo, ci_hi, elo,"
                " elo_lo, elo_hi, score_as_p0, score_as_p1, forfeits,"
                " opponent_retries, opponent_depth_mean, avg_plies, seconds)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    _now(),
                    source,
                    opponent_id,
                    iteration,
                    checkpoint,
                    result["games"],
                    result["score"],
                    result["win_rate"],
                    result["capped"],
                    result["ci_lo"],
                    result["ci_hi"],
                    result["elo"],
                    result["elo_lo"],
                    result["elo_hi"],
                    result["score_as_p0"],
                    result["score_as_p1"],
                    # NULL for match kinds without the concept (a
                    # checkpoint-vs-checkpoint summary carries neither).
                    result.get("forfeits"),
                    result.get("opponent_retries"),
                    depth_mean,
                    result["avg_plies"],
                    result["seconds"],
                ),
            )
            match_id = cur.lastrowid
            ids = self._game_ids(len(per_game))
            self._conn.executemany(
                "INSERT INTO games (game_id, kind, iteration, match, game_index,"
                " winner, length, capped, model_seat, opening_len,"
                " opponent_depth_mean, moves)"
                " VALUES (?, 'eval', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    (
                        game_id,
                        iteration,
                        match_id,
                        index,
                        g["winner"],
                        len(g["moves"]),
                        int(g["capped"]),
                        g["seat"],
                        g["opening_len"],
                        g["depth_mean"],
                        pack_moves(g["moves"]),
                    )
                    for index, (game_id, g) in enumerate(zip(ids, per_game))
                ),
            )
        return match_id

    def write_crossplay(self, results: list[dict], *, ply_cap: int, seed: int) -> None:
        """The A7 matrix, replacing the last one — the same wholesale
        semantics `crossplay.json` has."""
        with self._conn:
            self._conn.execute("DELETE FROM crossplay")
            self._conn.executemany(
                "INSERT INTO crossplay (checkpoint_a, checkpoint_b, games, score_a,"
                " capped, ply_cap, seed, created) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    (
                        r["a"],
                        r["b"],
                        r["games"],
                        r["score_a"],
                        r["capped"],
                        ply_cap,
                        seed,
                        _now(),
                    )
                    for r in results
                ),
            )


def _ply_rows(ids, episodes):
    """Every episode's per-ply rows, streamed into ``executemany`` rather
    than materialised — an iteration is tens of thousands of them."""
    for game_id, ep in zip(ids, episodes):
        n = len(ep.ranks)
        lengths = {
            len(ep.movers),
            len(ep.moves_remaining),
            len(ep.improved),
            len(ep.v_hats),
            len(ep.kls),
            len(ep.norm_entropies),
            len(ep.pi_top1),
            len(ep.pi_chosen),
        }
        if lengths != {n}:
            raise ValueError(
                f"episode's per-ply records disagree on length: {n} ranks vs "
                f"{sorted(lengths)}"
            )
        # round() on a NaN raises: no per-ply scalar may be NaN, and the
        # quantization is where that would otherwise slip through as NULL.
        yield from zip(
            itertools.repeat(game_id, n),
            range(n),
            ep.movers,
            ep.moves_remaining,
            map(len, ep.improved),
            ep.ranks,
            (round(v * _Q) for v in ep.v_hats),
            (round(v * _Q) for v in ep.kls),
            (round(v * _Q) for v in ep.norm_entropies),
            (round(v * _Q) for v in ep.pi_top1),
            (round(v * _Q) for v in ep.pi_chosen),
        )


# --------------------------------------------------------------------------
# Reading: the queries a dashboard needs first


def convert_v1(run_dir) -> None:
    """Regenerate a v1 database as v2, deliberately and once.

    There are no migrations-on-open — a version mismatch refuses, always.
    But a finished run's telemetry is history that cannot be replayed, so
    "regenerate the data" for it means this: build a fresh v2 file, copy
    every table across (plies quantizing REAL to 1/_Q units; the two v2
    match columns are NULL — v1 never measured them), swap it in, and keep
    the original beside it as ``telemetry.db.v1.bak``. Never run it against
    a database a live process is writing.
    """
    path = db_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"no telemetry database at {path}")
    src = sqlite3.connect(path)
    try:
        found = src.execute("SELECT version FROM schema_version").fetchone()[0]
    finally:
        src.close()
    if found != 1:
        raise ValueError(f"convert regenerates v1 databases only; {path} is v{found}")

    fresh_path = path.with_suffix(".v2.tmp")
    fresh_path.unlink(missing_ok=True)
    conn = _connect(fresh_path)  # builds the v2 schema
    try:
        conn.execute("ATTACH DATABASE ? AS v1", (str(path),))
        with conn:
            conn.execute("INSERT INTO runs SELECT * FROM v1.runs")
            conn.execute("INSERT INTO iterations SELECT * FROM v1.iterations")
            conn.execute("INSERT INTO opponents SELECT * FROM v1.opponents")
            conn.execute(
                "INSERT INTO eval_matches (created, source, opponent, iteration,"
                " checkpoint, games, score, win_rate, capped, ci_lo, ci_hi, elo,"
                " elo_lo, elo_hi, score_as_p0, score_as_p1, opponent_depth_mean,"
                " avg_plies, seconds)"
                " SELECT created, source, opponent, iteration, checkpoint, games,"
                " score, win_rate, capped, ci_lo, ci_hi, elo, elo_lo, elo_hi,"
                " score_as_p0, score_as_p1, opponent_depth_mean, avg_plies,"
                " seconds FROM v1.eval_matches"
            )
            conn.execute("INSERT INTO games SELECT * FROM v1.games")
            conn.execute(
                "INSERT INTO plies SELECT game_id, t, mover, moves_remaining,"
                f" legal_count, rank, CAST(ROUND(v_hat * {_Q}) AS INTEGER),"
                f" CAST(ROUND(kl * {_Q}) AS INTEGER),"
                f" CAST(ROUND(norm_entropy * {_Q}) AS INTEGER),"
                f" CAST(ROUND(pi_top1 * {_Q}) AS INTEGER),"
                f" CAST(ROUND(pi_chosen * {_Q}) AS INTEGER) FROM v1.plies"
            )
            conn.execute("INSERT INTO crossplay SELECT * FROM v1.crossplay")
        conn.execute("DETACH DATABASE v1")
    finally:
        conn.close()
    backup = path.with_suffix(".db.v1.bak")
    path.replace(backup)
    fresh_path.replace(path)
    print(f"{path}: regenerated as schema v2; v1 original at {backup}")


def _rows(conn, sql: str, params=()) -> list[dict]:
    return [dict(r) for r in conn.execute(sql, params)]


def _iteration_range(iterations, table=""):
    """An inclusive ``(lo, hi)`` filter, either end None for open, qualified
    by ``table`` where the query joins more than one."""
    if iterations is None:
        return "", []
    column = f"{table}.iteration" if table else "iteration"
    lo, hi = iterations
    clauses, params = [], []
    if lo is not None:
        clauses.append(f"{column} >= ?")
        params.append(lo)
    if hi is not None:
        clauses.append(f"{column} <= ?")
        params.append(hi)
    return " AND ".join(clauses), params


def iteration_series(conn, columns, *, iterations=None) -> list[dict]:
    """Metric columns against iteration, in order — a line chart's data.

    Column names are checked against the table rather than interpolated
    blind, so a typo names itself instead of becoming a SQL error.
    """
    known = {r["name"] for r in conn.execute("PRAGMA table_info(iterations)")}
    unknown = [c for c in columns if c not in known]
    if unknown:
        raise ValueError(f"no such iteration columns: {unknown}; have {sorted(known)}")
    where, params = _iteration_range(iterations)
    return _rows(
        conn,
        f"SELECT iteration, {', '.join(columns)} FROM iterations"
        f"{' WHERE ' + where if where else ''} ORDER BY iteration",
        params,
    )


# The orders the game browser offers, by name. Every one ends in the game's
# natural key so paging is stable when the sort column ties. Adding one
# means checking `_GAME_INDEXES` still covers it: an order with no index
# behind it sorts every matching game to return a page of fifty.
GAME_ORDERS = {
    "recent": "iteration DESC, game_index",
    "oldest": "iteration, game_index",
    "longest": "length DESC, iteration DESC, game_index",
    "shortest": "length, iteration DESC, game_index",
}


def search_games(
    conn,
    *,
    kind=None,
    winner=None,
    capped=None,
    min_length=None,
    max_length=None,
    iterations=None,
    order="recent",
    limit=100,
    offset=0,
) -> list[dict]:
    """The game browser's query. Every filter is optional and they conjoin;
    the moves blob is left behind, since a list of games does not need it.

    ``winner=None`` means "any", which capped games (winner NULL) satisfy
    too — filter them with ``capped``. ``order`` is a name from
    :data:`GAME_ORDERS` rather than a SQL fragment: this is the query a web
    layer will hand user input to, and a name cannot be an injection.
    """
    if order not in GAME_ORDERS:
        raise ValueError(f"order must be one of {sorted(GAME_ORDERS)}: {order!r}")
    clauses, params = [], []
    for column, value in (("kind", kind), ("winner", winner)):
        if value is not None:
            clauses.append(f"{column} = ?")
            params.append(value)
    if capped is not None:
        clauses.append("capped = ?")
        params.append(int(capped))
    for column, value in (("length >= ?", min_length), ("length <= ?", max_length)):
        if value is not None:
            clauses.append(column)
            params.append(value)
    where, iter_params = _iteration_range(iterations)
    if where:
        clauses.append(where)
        params += iter_params
    return _rows(
        conn,
        "SELECT game_id, kind, iteration, match, game_index, winner, length,"
        " capped, model_seat, opening_len, opponent_depth_mean FROM games"
        + (" WHERE " + " AND ".join(clauses) if clauses else "")
        + f" ORDER BY {GAME_ORDERS[order]} LIMIT ? OFFSET ?",
        [*params, limit, offset],
    )


def fetch_game(conn, game_id: int) -> dict:
    """One game with its moves unpacked and its plies attached — the viewer's
    payload. A self-play game has a ply per move; an evaluation game has
    none, and the empty list is the honest answer."""
    row = conn.execute("SELECT * FROM games WHERE game_id = ?", (game_id,)).fetchone()
    if row is None:
        raise KeyError(f"no game {game_id}")
    game = dict(row)
    game["moves"] = unpack_moves(game["moves"])
    game["plies"] = [
        {**ply, **{k: ply[k] / _Q for k in _PLY_SCALARS}}
        for ply in _rows(
            conn, "SELECT * FROM plies WHERE game_id = ? ORDER BY t", (game_id,)
        )
    ]
    return game


def calibration(conn, *, by="v_hat", bucket=0.1, iterations=None) -> list[dict]:
    """v̂ against the outcome it was predicting, bucketed — the instrument
    for the overestimation bias in ``KLENT_FOR_HEXO.md`` §8.

    Self-play games that finished only: a capped game has no realized
    outcome to compare against. ``by`` chooses the axis — ``'v_hat'`` for a
    reliability diagram, ``'ply'`` for calibration against how far into the
    game the position was, ``'length'`` for how long its game ran — and
    ``bucket`` the bucket width on it.
    """
    # `floor` is a compile-time option in SQLite and this has to run against
    # whatever build the reader has, so the signed axis floors by hand: CAST
    # truncates toward zero, and a negative value not landing on a bucket
    # edge is one bucket short of it. The other two axes are non-negative,
    # where truncation is already the floor.
    v = f"(p.v_hat / {_Q}.0)"  # the stored integer back in value units
    axes = {
        "v_hat": (
            f"CAST({v} / ? AS INTEGER) - (CASE WHEN {v} < 0"
            f" AND CAST({v} / ? AS INTEGER) * ? <> {v} THEN 1 ELSE 0 END)",
            3,
        ),
        "ply": ("CAST(p.t / ? AS INTEGER)", 1),
        "length": ("CAST(g.length / ? AS INTEGER)", 1),
    }
    if by not in axes:
        raise ValueError(f"calibration axis must be one of {sorted(axes)}: {by!r}")
    expr, widths = axes[by]
    z = "(CASE WHEN p.mover = g.winner THEN 1.0 ELSE -1.0 END)"
    where, params = _iteration_range(iterations, "g")
    return _rows(
        conn,
        "SELECT bucket, bucket * ? AS bucket_lo, COUNT(*) AS plies,"
        " AVG(v_hat) AS v_hat_mean, AVG(z) AS outcome_mean,"
        " AVG(ABS(v_hat - z)) AS mae FROM ("
        f"  SELECT {expr} AS bucket, {v} AS v_hat, {z} AS z"
        "   FROM plies p JOIN games g ON g.game_id = p.game_id"
        "   WHERE g.kind = 'selfplay' AND g.winner IS NOT NULL"
        + (f" AND {where}" if where else "")
        + ") GROUP BY bucket ORDER BY bucket",
        [bucket, *[bucket] * widths, *params],
    )


# The mover's own assessment of the next position: v̂ is in the frame of
# whoever is to move, so a seat change between plies flips its sign before
# the two are comparable.
_SWING = (
    f"((CASE WHEN a.mover = b.mover THEN b.v_hat ELSE -b.v_hat END)"
    f" - a.v_hat) / {_Q}.0"
)


def blunders(conn, *, threshold=0.5, iterations=None, limit=50) -> list[dict]:
    """Plies where the position's value collapsed on the next one.

    A large negative swing is the mover watching its own assessment fall
    apart across one placement — either a real mistake or a spot where the
    Q head is unstable, and the policy debugger is how the two are told
    apart. Ordered by how hard the swing was.
    """
    where, params = _iteration_range(iterations, "g")
    return _rows(
        conn,
        f"SELECT a.game_id, g.iteration, a.t, a.mover,"
        f" a.v_hat / {_Q}.0 AS v_hat,"
        f" b.v_hat / {_Q}.0 AS v_hat_next, {_SWING} AS swing, a.rank,"
        f" a.legal_count, a.norm_entropy / {_Q}.0 AS norm_entropy,"
        f" a.pi_chosen / {_Q}.0 AS pi_chosen"
        " FROM plies a"
        " JOIN plies b ON b.game_id = a.game_id AND b.t = a.t + 1"
        " JOIN games g ON g.game_id = a.game_id"
        f" WHERE ABS({_SWING}) >= ?"
        + (f" AND {where}" if where else "")
        + f" ORDER BY ABS({_SWING}) DESC LIMIT ?",
        [threshold, *params, limit],
    )


def opening_atlas(conn, *, plies=4, kind="selfplay", iterations=None, limit=50):
    """The openings the run actually plays, symmetry-reduced.

    Canonicalization is query-time by :func:`canonical_opening` — storage
    keeps raw moves, because a stored canonical form would be a second
    representation to keep true. Games shorter than ``plies`` are skipped:
    a truncation that is not a prefix of length ``plies`` is a different
    key.
    """
    where, params = _iteration_range(iterations)
    rows = conn.execute(
        "SELECT moves, winner, length, capped FROM games WHERE kind = ?"
        + (f" AND {where}" if where else ""),
        [kind, *params],
    )
    atlas: dict[tuple, dict] = {}
    for row in rows:
        moves = unpack_moves(row["moves"])
        if len(moves) < plies:
            continue
        key = canonical_opening(moves, plies)
        entry = atlas.setdefault(
            key, {"opening": key, "games": 0, "p0_wins": 0, "p1_wins": 0,
                  "capped": 0, "total_length": 0}
        )
        entry["games"] += 1
        entry["capped"] += row["capped"]
        entry["total_length"] += row["length"]
        if row["winner"] == 0:
            entry["p0_wins"] += 1
        elif row["winner"] == 1:
            entry["p1_wins"] += 1
    out = sorted(atlas.values(), key=lambda e: -e["games"])[:limit]
    for entry in out:
        entry["mean_length"] = entry.pop("total_length") / entry["games"]
    return out


def strength_curve(conn, *, opponent_id=None) -> list[dict]:
    """Every evaluation match against every opponent, in iteration order —
    the strength curves, one per (opponent, source)."""
    clause = " WHERE m.opponent = ?" if opponent_id is not None else ""
    return _rows(
        conn,
        "SELECT m.*, o.name AS opponent_name, o.config_json AS opponent_config"
        " FROM eval_matches m JOIN opponents o ON o.opponent_id = m.opponent"
        + clause
        + " ORDER BY m.iteration, m.created",
        [opponent_id] if opponent_id is not None else [],
    )


def crossplay_matrix(conn) -> list[dict]:
    return _rows(conn, "SELECT * FROM crossplay ORDER BY checkpoint_a, checkpoint_b")


def summary(conn) -> dict:
    """What the run holds, in one round of counts."""
    counts = conn.execute(
        "SELECT (SELECT COUNT(*) FROM runs),"
        " (SELECT COUNT(*) FROM iterations),"
        " (SELECT MIN(iteration) FROM iterations),"
        " (SELECT MAX(iteration) FROM iterations),"
        " (SELECT COUNT(*) FROM games WHERE kind = 'selfplay'),"
        " (SELECT COUNT(*) FROM games WHERE kind = 'eval'),"
        " (SELECT COUNT(*) FROM plies),"
        " (SELECT COUNT(*) FROM eval_matches),"
        " (SELECT COUNT(*) FROM crossplay)"
    ).fetchone()
    last = conn.execute(
        "SELECT * FROM iterations ORDER BY iteration DESC LIMIT 1"
    ).fetchone()
    return {
        "invocations": counts[0],
        "iterations": counts[1],
        "iteration_first": counts[2],
        "iteration_last": counts[3],
        "selfplay_games": counts[4],
        "eval_games": counts[5],
        "plies": counts[6],
        "eval_matches": counts[7],
        "crossplay_pairs": counts[8],
        "opponents": _rows(conn, "SELECT * FROM opponents ORDER BY opponent_id"),
        "latest": dict(last) if last is not None else None,
    }


# --------------------------------------------------------------------------
# CLI


def _print_summary(conn, run_dir) -> None:
    s = summary(conn)
    size = db_path(run_dir).stat().st_size
    print(f"{db_path(run_dir)}  ({size / 1e6:.1f} MB, schema v{SCHEMA_VERSION})")
    print(
        f"  {s['invocations']} invocation(s), iterations "
        f"{s['iteration_first']}..{s['iteration_last']} ({s['iterations']} rows)"
    )
    print(
        f"  {s['selfplay_games']} self-play games, {s['plies']} plies"
        + (f" ({size / s['plies']:.0f} B/ply)" if s["plies"] else "")
    )
    print(
        f"  {s['eval_matches']} eval matches ({s['eval_games']} games), "
        f"{s['crossplay_pairs']} cross-play pairs"
    )
    for o in s["opponents"]:
        print(f"  opponent {o['opponent_id']}: {o['name']} {o['config_json']}")
    if s["latest"] is not None:
        keys = ("iteration", "f", "acting_norm_entropy", "buffer_samples", "seconds",
                "samples_per_s", "policy_loss", "q_loss", "v_hat_mae", "gpu_util_mean")
        shown = " ".join(
            f"{k}={s['latest'][k]:.4g}" if isinstance(s["latest"][k], float)
            else f"{k}={s['latest'][k]}"
            for k in keys
            if s["latest"][k] is not None
        )
        print(f"  latest: {shown}")


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, required=True, help="a run directory")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("summary", help="what the database holds")

    games = sub.add_parser("games", help="search finished games")
    games.add_argument("--kind", choices=("selfplay", "eval"))
    games.add_argument("--winner", type=int, choices=(0, 1))
    games.add_argument("--capped", type=int, choices=(0, 1))
    games.add_argument("--min-length", type=int)
    games.add_argument("--max-length", type=int)
    games.add_argument("--from-iteration", type=int)
    games.add_argument("--to-iteration", type=int)
    games.add_argument("--limit", type=int, default=20)

    one = sub.add_parser("game", help="one game's moves and plies")
    one.add_argument("game_id", type=int)

    sub.add_parser(
        "convert",
        help="regenerate a v1 database as v2 in place (v1 original kept as .v1.bak)",
    )

    args = ap.parse_args(argv)
    if args.command == "convert":
        convert_v1(args.run)
        return
    conn = connect(args.run)
    try:
        if args.command == "summary":
            _print_summary(conn, args.run)
        elif args.command == "games":
            rows = search_games(
                conn,
                kind=args.kind,
                winner=args.winner,
                capped=args.capped,
                min_length=args.min_length,
                max_length=args.max_length,
                iterations=(args.from_iteration, args.to_iteration),
                limit=args.limit,
            )
            print(f"{'game':>8} {'kind':<9} {'iter':>6} {'winner':>6} {'len':>5} capped")
            for r in rows:
                print(
                    f"{r['game_id']:>8} {r['kind']:<9} {r['iteration']!s:>6} "
                    f"{r['winner']!s:>6} {r['length']:>5} {r['capped']}"
                )
        else:
            game = fetch_game(conn, args.game_id)
            print(
                f"game {game['game_id']}: {game['kind']} iteration "
                f"{game['iteration']}, winner {game['winner']}, "
                f"{game['length']} plies, capped {game['capped']}"
            )
            for t, move in enumerate(game["moves"]):
                ply = game["plies"][t] if t < len(game["plies"]) else None
                line = f"  {t:>4} {str(move):>10}"
                if ply is not None:
                    # ASCII only: the console this lands on is cp1252.
                    line += (
                        f"  mover {ply['mover']}  v_hat {ply['v_hat']:+.3f}"
                        f"  KL {ply['kl']:.3f}  H/log|A| {ply['norm_entropy']:.3f}"
                        f"  pi' {ply['pi_chosen']:.3f}/{ply['pi_top1']:.3f}"
                        f"  rank {ply['rank']}/{ply['legal_count']}"
                    )
                print(line)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
