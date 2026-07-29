# Shrimp Control Deck frontend

## Purpose

`frontend/` is the React 19 single-page client for `mantisnet.deck`. It renders
live run control, game history, human play, and checkpoint inspection against
the deck's HTTP and SSE APIs. Vite builds a static `dist/` directory that the
FastAPI service serves.

## Public surface

The application has four persistent screens:

| Screen | Module | Contract |
| --- | --- | --- |
| Play | `src/screens/Play.tsx` | Engine-backed play sessions and arena jobs |
| Game history | `src/screens/History.tsx` | Game search, replay, reviews, and aggregates |
| Live run | `src/screens/LiveRun.tsx` | Heartbeat, metrics, SSE, sentinels, and kill |
| Model lab | `src/screens/Lab.tsx` | Position lines, checkpoint reads, D6, and attention |

The shared modules are:

| Path | Consumer-facing role |
| --- | --- |
| `src/App.tsx` | Run selection, screen routing, launch dialog, command palette |
| `src/api.ts` | JSON requests, posts, query encoding, and `useApi` |
| `src/types.ts` | Deck response and UI state types |
| `src/components/Board.tsx` | Hex board, legal masks, policy/Q overlays, move input |
| `src/components/Chart.tsx` | Multi-series SVG chart |
| `src/components/Replay.tsx` | Replay transport and deck key binding |
| `src/components/Pane.tsx` | Persistent mounted-screen visibility |
| `src/components/Ui.tsx` | Panels, notices, metrics, controls, and formatting |
| `src/lib/hex.ts` | Client-side move indexing and display geometry |
| `src/lib/inspect.ts` | Cancellable checkpoint inspection of a move line |

The npm scripts are:

| Script | Command |
| --- | --- |
| `dev` | Vite development server on all interfaces |
| `build` | Type-check, then create `dist/` |
| `test` | Build, then inspect the emitted HTML and bundle |
| `lint` | Type-check without emitting files |
| `preview` | Serve the production bundle locally |
| `prune-css` | Remove statically unreachable CSS rules |
| `lan-relay` | Bind a selected Windows LAN address and forward to loopback |

## Run / test

Start the deck API on port 8000, then run from `frontend/`:

```sh
npm ci
npm run dev
```

Vite proxies `/api` requests and SSE connections to `http://127.0.0.1:8000`.

Build and test:

```sh
npm run lint
npm run build
npm test
```

Inspect or apply the dead-CSS scan:

```sh
node scripts/prune-css.cjs
npm run prune-css
```

Use the Compose build from the repository root:

```sh
docker compose -f docker/compose.yaml run --rm frontend-build
```

Start the optional Windows LAN relay from `frontend/`:

```powershell
npm run lan-relay -- 192.168.68.62 8000 127.0.0.1 8000
```

## Connections

- `python/mantisnet/mantisnet/deck/app.py` implements the HTTP, SSE, and SPA
  routes.
- `python/mantisnet/mantisnet/deck/service.py` implements run and inference
  state.
- `python/mantisnet/mantisnet/deck/state.py` owns deck-local persistence.
- `docker/compose.yaml` builds the SPA and serves it from the `deck` service.
- `frontend/dist` is the static artifact consumed by the backend.
- The endpoint and interaction contract is
  [`docs/DECK_SPEC.md`](../docs/DECK_SPEC.md).

## Invariants & gotchas

- Node.js 22.13 or later is required by `package.json`.
- The frontend is a static client and contains no server runtime.
- API request and response types are centralized in `src/types.ts`.
- `api` and `useApi` treat non-success HTTP responses as structured failures.
- `useApi` aborts obsolete requests and prevents stale responses from replacing
  newer state.
- Screens mount on first use and remain mounted while hidden.
- Only the active pane binds document-level keyboard controls.
- Board legality comes from the server-provided legal set or legality mask.
- A legal mask carries no policy, action value, improved-policy value, or rank.
- The board uses engine canonical move coordinates; it does not submit policy
  indices as moves.
- Replay controls operate on an immutable move prefix and cursor.
- Edited model-lab lines invalidate readings derived from the prior line.
- Whole-line inspection is cancellable and uses bounded concurrency.
- Play mutations are accepted only at the live session position.
- Expensive history aggregates are manual requests and do not run on screen
  mount.
- Reviews, probes, presets, and match jobs persist through deck API calls.
- The dead-CSS scan is conservative for dynamically assembled class names.
- `npm test` always rebuilds before inspecting emitted artifacts.
- Production serving requires `dist/index.html`.
- Frontend source contains no bundled run dataset or model checkpoint.
