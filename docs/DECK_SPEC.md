# The Shrimp Control Deck — normative spec

The deck is a LAN-served dashboard over run telemetry, checkpoint inspection,
play, and match control. It consists of `mantisnet.deck`, a FastAPI service in
the Python project, and `frontend/`, a Vite + React single-page application.
The supported serving environment is Docker (`docker/`); there is no
Windows-native serving path.

The required screens are Play, Game history, Live run, and Model lab, with the
command palette and compare tray. Every displayed run or model value MUST come
from a backend response or persisted deck state.

## Ground rules

- **`telemetry.db` is read-only to the deck's query paths.** The schema and
  its query helpers (`mantisnet/klent/telemetry.py`: `iteration_series`,
  `search_games`, `fetch_game`, `calibration`, `blunders`, `opening_atlas`,
  `strength_curve`, `crossplay_matrix`, `summary`) are the substrate; the
  API wraps them, it does not re-derive SQL that exists there. The single
  exception is match recording (below), which goes through the existing
  writer path with a `busy_timeout` — `sealbot.record_match` for a SealBot
  match, `telemetry.write_eval_match` for a checkpoint one.
- **The training loop remains independent.** The deck consumes the
  run driver's artifacts — `telemetry.db`, `metrics.jsonl`, `config.json`,
  `invocations.jsonl`, checkpoints, and `status.json` —
  and controls a run only through the child-process boundary and the two
  sentinel files the driver honors (below). Deck code MUST NOT mutate training
  loop modules or their in-process state.
- **No auth, LAN-trusted.** The deck binds `0.0.0.0` and every control is
  live. Deployments MUST treat the bound network as trusted; the service does
  not implement authentication or authorization.
- **Explicit errors.** A missing run directory, unreadable database,
  version-mismatched checkpoint, or dead child process MUST surface as a
  structured HTTP error and event-stream error, not an empty panel.
- **One implementation per job.** The frontend is a Vite client-only SPA; no
  Cloudflare, vinext, Wrangler, RSC, or alternate application stack is
  supported.

## Architecture

```
browser ── :8000 ─┬─ /            static frontend build (frontend/dist)
                  ├─ /api/...     REST (JSON)
                  └─ /api/runs/{run}/events   SSE
                        │
                  mantisnet.deck (FastAPI + uvicorn, in the training image,
                        │          CUDA visible, spawns training children)
                        ├─ read:  runs/<name>/telemetry.db  (WAL)
                        ├─ read:  runs/<name>/status.json   (driver heartbeat)
                        ├─ write: runs/<name>/STOP, CHECKPOINT  (sentinels)
                        ├─ write: runs/deck.db  (deck-owned annotations)
                        └─ exec:  python -m mantisnet.klent.run ...  (children)
```

One service, one port. The API serves the built SPA as static files, so the
whole deck is `http://<host>:8000` on the LAN. `frontend/` development uses
Vite's dev server with a `/api` proxy to :8000; production is `npm run
build` output served by FastAPI.

### The run driver's side of the contract

The deck consumes these driver-owned interfaces and MUST NOT implement their
producer behavior.

- **`runs/<name>/status.json`** — the heartbeat, atomically replaced
  (write-then-rename) at most every second while a run is alive:

  ```json
  {
    "updated": "2026-07-28T21:04:11+00:00",
    "iteration": 241,
    "collect": {"iteration": 242, "finished": 3872, "quota": 4096,
                 "steps": 118, "slot_plies": [12, 31, 0, "..."]},
    "fit": {"iteration": 241, "chunk": 29, "chunks": 36},
    "eval": null
  }
  ```

  `slot_plies` is the live ply count of each collector slot (length =
  `envs`). `fit`/`collect`/`eval` are `null` when that lane is idle. A
  `status.json` older than ~10 s with a live process means the process is
  busy inside a phase, not dead; liveness comes from the process, staleness
  only downgrades the strip to "working".

- **Sentinels** — `runs/<name>/STOP`: finish the current iteration, write a
  checkpoint, exit 0. `runs/<name>/CHECKPOINT`: write a checkpoint at the
  next commit point, delete the file, continue. The deck creates these
  files; the driver consumes them.

### `mantisnet.deck` (`python/mantisnet/mantisnet/deck/`)

The FastAPI app and its non-HTTP services implement these seams:

- **Run registry.** A run is a directory under `runs/` containing
  `config.json`. State machine per run: `active` (a child process the deck
  spawned is alive, or `status.json` is fresh from an externally-launched
  process), `stopped`, `completed` (iteration count reached), `starved`.
  External runs (launched from a terminal) appear read-only-live: full
  telemetry and heartbeat, no process control except the sentinel files —
  which work regardless of who launched the run. Kill is available only for
  child processes launched by this deck.
- **Lifecycle.** `POST /api/runs` launches `python -m mantisnet.klent.run`
  as a child with a JSON body of knobs (name, iterations, games, envs,
  seed, lam_ret, eval cadence + sealbot depth/time, checkpoint cadence,
  init_from | resume). The deck captures the child's stdout/stderr into
  `runs/<name>/deck-console.log` and parses iteration/eval lines into the
  event stream. `POST /api/runs/{run}/stop` writes `STOP`;
  `.../checkpoint` writes `CHECKPOINT`; `.../kill` SIGTERMs the child
  (guarded by a `confirm: true` body field). Exactly one active training
  run at a time: launching while one is active is a 409, not a queue.
- **Telemetry queries.** Thin GET endpoints over the helpers listed above,
  each taking the run name plus that helper's own parameters (iteration
  windows, filters, pagination for game search). Game detail returns the
  unpacked move list plus the game's `plies` rows so the client can replay
  the board and scrub the per-ply scalars without a second round trip.
- **SSE.** `GET /api/runs/{run}/events`: typed events
  (`iteration`, `heartbeat`, `eval`, `checkpoint`, `log`, `lifecycle`),
  produced by polling the DB's max iteration (~1 s), the heartbeat file's
  mtime, the checkpoint directory, and the child's console stream. The
  client renders the event stream panel directly from these and refreshes
  panel queries on `iteration`.
- **Inference sessions.** A small LRU (≤ 2) of loaded checkpoints on the
  service's device. Loading is eager and MUST NOT call `torch.compile`.
  `inspect_position` is the one seam for position analysis. τ and λ come
  from the checkpoint's run `config.json`; a checkpoint outside a run
  directory requires both values in the request.
  Explicitly bounded: analysis batches are single positions; arena/match
  cohorts are at most 64. Match-launch responses MUST report whether training
  is active because the GPU may be shared with that process.
- **Play sessions.** Server-side authoritative state (`hexo_py` replay is
  the referee): create a session with two seats — human, a checkpoint
  (move mode argmax πθ | sample πθ | π′), SealBot (depth/time), or random
  — post human moves, get bot replies, query `inspect_position` for the
  current prefix for the overlays and candidate list. Illegal placements
  are a 400 with the reason, enforced by the engine, not the client.
- **Matches.** `POST /api/matches` runs a seat-paired set in a background
  thread: checkpoint vs checkpoint (`evaluate.play_match`) or checkpoint vs
  SealBot (`sealbot.sealbot_match`), progress over SSE, result recorded to
  the checkpoint's run `telemetry.db` as `source='deck'` — through
  `record_match` for a SealBot match, `write_eval_match` for a checkpoint one.
  Both kinds play `games / 2` shared random openings from both seats, so a
  request's game count is its count of distinct games.
  One match job at a time, same 409 rule as runs.
- **`runs/deck.db`** — the deck's own SQLite (WAL, schema-versioned and
  refused on mismatch like telemetry, no migrations): game tags and review
  notes (keyed run + game_id), saved lab probes (name + checkpoint + move
  prefix + module), play presets, and the match queue history. Deck state
  never touches `telemetry.db`.

### `frontend/`

- **Stack:** plain Vite + React 19 + TypeScript +
  Tailwind v4 + lucide-react, client-only SPA. `next`/`vinext`/`wrangler`/
  RSC machinery is not part of the application. `index.html` provides the
  mount point. `npm run build` emits `frontend/dist`;
  `npm run dev` proxies `/api` and the SSE path to :8000. Keep the
  rendered-shell test working against the Vite build.
- **Module structure.** Each screen has its own module and uses shared board,
  chart, and API-client components. Mock run or model data MUST NOT be present
  in the shipped application. A panel MUST either render live or persisted
  data or be absent.
- **Screens:**
  - **Live run** — run selector = API run list (state-badged). Status
    strip + pipeline panel from `status.json` heartbeat; charts from
    `iteration_series` with a window selector (the five metric tabs in the
    interface, plus per-iteration seconds); eval panel from the latest
    `eval_matches` row (CI, Elo, and seat split columns); losses/
    diagnostics/hardware panels from `iterations` columns; slot-cohort grid
    binned from `slot_plies` (legend: live ply bands + at-cap); artifact
    timeline from checkpoint files on disk + eval rows + cadence
    projection; manifest from `config.json` + `invocations.jsonl` (exact
    version strings, exact knobs); event stream = SSE. Controls: launch/
    resume form, checkpoint-now, stop-after-iteration, kill (confirm
    dialog) — all live against the lifecycle API.
  - **Game history** — table = `search_games` (kind, winner, length,
    iteration range, capped, pagination); filters map to query parameters;
    opening column via
    `canonical_opening`. Detail: board replay scrubber from the move blob
    with the per-ply v̂ / KL / entropy / π-chosen track under it (`plies`
    rows), eval games showing seat/opening/depth columns instead. Tags and
    notes persist to `deck.db`. Calibration, blunder list, and opening
    atlas panels from their namesake helpers. "Open in lab" hands the
    game's prefix to the lab.
  - **Play** — everything through play sessions: human seats click the
    engine-checked board, bot seats reply on their mode; overlays
    (πθ / Q / π′ / rank) and the candidate list from `inspect_position`,
    colored over the engine legal set; the KLENT position read (v̂, KL,
    H/log|A|, legal count) is the same response. Arena mode drives
    `/api/matches` with live progress and a link to the recorded result.
    Quick suites are the cross-play launcher and a SealBot set;
    πθ/Q-disagreement cases are saved lab probes.
  - **Model lab** — position editor (paste/edit a move list, or arrive via
    handoff) + checkpoint picker (runs' checkpoint files); policy/Q/π′/v̂
    readouts and board overlay from `inspect_position`; attention view
    from a deck endpoint that reruns the trunk with the reference SDPA
    path capturing per-block per-head weights over [token; stones] for the
    inspected position (the fused kernel never materializes weights — the
    capture path is the reference implementation, used only here); D6
    invariance check = `inspect_position` over the 12 transforms
    (`telemetry.D6_TRANSFORMS`) reporting max policy/Q deviation;
    representation/manifest panels from `MantisConfig` + `_versions()`.
    Saved probes persist to `deck.db`. Checkpoint compare = two inspect
    calls diffed per legal move.
- **Realtime behavior:** SSE-driven refresh on the live screen; history/
  lab/play refetch on navigation and on demand. The pause-telemetry toggle
  pauses the SSE subscription client-side.

## Docker

Compose (in `docker/compose.yaml`, beside the existing interactive `train`
service, same image):

- `deck`: runs `uvicorn` on :8000 (published), GPU visible, entrypoint
  builds `hexo_py` (maturin) and the SealBot extension if their artifacts
  are missing, then serves with the `unless-stopped` restart policy.
- `frontend-build`: one-shot `node:22` service that runs `npm ci && npm run
  build` in `/workspace/frontend` (node_modules in a named volume). Run it
  after frontend changes; `deck` serves whatever `frontend/dist` holds and
  refuses to start without one.

Training runs launched by the deck are children inside the `deck` service —
process control stays a plain child-process boundary, no Docker-in-Docker.
`runs/` stays on the repository bind mount so all runs share one path visible
from Windows.

## Conformance checks

From `python/mantisnet` in the container:

1. `pytest tests/ -q` MUST pass, including `tests/test_deck*.py` coverage of:
   run registry states, lifecycle sentinels (against a scripted fake run
   process), every query endpoint against a heuristic-run telemetry
   database, SSE event
   emission for a written iteration, play-session legality (an illegal move
   is a 400 with the engine's reason), and inspect parity (endpoint values
   equal a direct `inspect_position` call).
2. `npm run build` and the rendered-shell test MUST pass in `frontend/`.
3. `docker compose up deck` MUST serve the SPA and API on :8000. Every panel
   on all four screens MUST render backend or persisted data for a heuristic
   run launched through the API; no mock constants may be reachable in the
   shipped bundle.
4. `python/mantisnet/README.md`, `frontend/README.md`, and `docker/README.md`
   MUST describe the deck modules, artifacts, and services they expose.
