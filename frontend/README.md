# Shrimp Control Deck frontend

The client-only React 19 SPA for `mantisnet.deck`. It is plain Vite +
TypeScript + Tailwind v4's CSS toolchain + lucide-react: no Next, RSC, vinext,
Wrangler, or worker runtime remains.

The shell carries the run selector, the launch dialog and a command palette
(`⌘/Ctrl-K`) that filters over the four screens, every run on disk — switching
the active run is the header control most often reached for — and the deck-wide
actions. Screen-local work stays on its screen.

A screen mounts the first time it is opened and then stays mounted, hidden
(`components/Pane.tsx`): a hand-built line, a whole-line walk and an aggregate
that cost a minute of database time all survive a trip to another screen, and a
screen never opened never queries anything. Only the visible pane binds the
deck's document-level keys, and only its transport plays.

The four screens are real-data surfaces:

- **Play** creates engine-authoritative server sessions and posts human moves.
  The session's own `legal_moves` is the legality mask, so a game is playable
  with no checkpoint loaded — and a masked cell prints its coordinates and
  nothing else, because no π, Q or rank exists for it. Select a checkpoint and
  the same cells carry that checkpoint's π, Q and π′ heat, its top move (which
  only a read can name), and a ranked candidate table. The
  shared transport scrubs back through the game so far — the engine only accepts
  a placement from the live position, and the screen says so rather than failing
  the post. It also launches recorded arena jobs.
- **Game history** browses telemetry games and replays them on the shared
  transport, with the acting net's own stored per-ply trace charted under the
  board — v̂ in either seat's frame, the swing the blunder table ranks on, π
  top-1 against π played, entropy, KL and the legal count — and clicking the plot
  moves the same cursor. Its queries are the expensive ones on a real run, so no
  panel waits on another: the listing stages its filters and prints what the last
  page actually cost, a game opens by id without the listing at all, and
  calibration, the swing table and the D6-canonical opening atlas each load only
  when asked, against the browsed iteration window. Tags and notes persist, and a
  swing row opens its game at the offending ply.
- **Live run** follows heartbeat and typed SSE, queries iteration/evaluation
  series, renders collector cohorts and artifacts, and drives the sentinel and
  kill controls.
- **Model lab** is a position explorer. It holds a *line* and a cursor, not a
  static prefix: the line comes from a blank board, a telemetry game (by id, by
  an on-demand listing browse, or handed off from Game history), or a saved
  probe, and it is edited by clicking legal cells. Where the line on screen
  departs from the line its source arrived with is derived, not recorded, so the
  branch notice, the acting-net trace's cut-off and "restore the source line"
  agree with each other after any sequence of clicks and undos. The cursor's ply
  is inspected automatically, and the whole line through a cancellable walk —
  automatic for a game's own line, where the cost is bounded and known, explicit
  once the line has been edited, because an edit invalidates every reading. The
  chart under the board then carries this checkpoint's v̂, π, regret, entropy, KL
  and policy-rank at every ply beside the acting net's own stored trace, and
  clicking the plot moves the cursor. It also compares two checkpoints (Δπ on the
  board and in the candidate table), renders the 12-way D6 check as a verdict,
  the SDPA attention capture as a canvas heatmap, and saves, restores and deletes
  probes in `deck.db`.

`src/api.ts` is the one JSON client: `api()` honours an `AbortSignal`, and
`useApi(path, deps, {manual})` guards every response with an abort plus a
generation snapshot so a stale reply can never land on fresher state. The
aggregates that cost 60–110 s on a live run take `manual: true` and fetch only
when asked.

Four shared renderers, one implementation each. `src/components/Board.tsx` draws
the hex position — constant-radius cells, zoom/pan, a backdrop grid, per-position
normalised heat, top-k labels, and the played-vs-top marks. Its legal set arrives
as `legal` (candidates with a read behind them) or as `mask` (legality only);
they are different slots, so nothing on the board can imply a number that was
never measured. Its own arrow keys walk the candidate ring only while it holds
keyboard focus, leaving the transport's map alone after a click. `Chart.tsx` is
the multi-series plot for every screen: single y-axis, null-breaks, crosshair
tooltip, click-to-jump. `Replay.tsx` is the one replay transport (track,
first/prev/play/next/last, speed, keyboard) plus the `useDeckKeys` binder every
screen key goes through. `Ui.tsx` holds the panel, notice, metric, segmented,
progress and heat-legend primitives, and `format` — the one number formatter,
used by the board read-out and the chart axes as well.

Two libraries sit under them. `src/lib/hex.ts` is the game's rules as the client
needs them — the axes, the move key, which player placed ply *i*, the winning
six, and P1's turn bands — so no screen re-derives them. `src/lib/inspect.ts`
inspects a whole line against a checkpoint: the cursor's ply automatically, the
rest through an explicit, cancellable, bounded-concurrency walk that derives the
per-ply scalars — including the policy-sorted quality rank and the Q regret.

Each screen lives in `src/screens/`. `app/globals.css` carries the visual tokens,
the shell, and the wired module styles: `scripts/dead-css.cjs` is the single scan
that decides which classes `src/` cannot produce, `npm run prune-css` deletes
those rules, and the test suite fails if one reappears. The scan is one-sided by
design — it keeps a class assembled at runtime, so it proves a rule dead and
never proves one live. There are no mock datasets in the shipped source or
bundle.

## Development

Run the FastAPI service on port 8000, then:

```sh
npm ci
npm run dev
```

Vite proxies every `/api` request, including SSE, to port 8000. Production:

```sh
npm test
```

The build emits `dist/`. `npm test` compiles the SPA and then asserts two things
against what it emitted: that the bundle carries every deck surface — the four
screens, the palette, the lab's line/walk/diagnostics copy, history's aggregates,
the shared board, chart and transport — and carries no hand-typed JSON position
input, raw-response dump or removed worker layer; and that `app/globals.css`
names no class `src/` cannot produce. In the intended environment, use the
Compose `frontend-build` service so the `deck` service can serve that directory.

## WSL LAN access

Docker Desktop/WSL can publish the deck to Windows loopback while leaving the
LAN address unbound. On Windows, start the included TCP relay with the machine's
current LAN address:

```powershell
npm run lan-relay -- 192.168.68.62 8000 127.0.0.1 8000
```

The relay binds only that LAN address and forwards the complete HTTP, API, and
SSE connection to WSL's localhost relay. It avoids a static WSL IP and does not
collide with the `127.0.0.1:8000` listener. Keep the process running for LAN
access; restart it after a Windows reboot or LAN address change.
