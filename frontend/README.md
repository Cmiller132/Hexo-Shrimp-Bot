# Shrimp Control Deck frontend

The client-only React 19 SPA for `mantisnet.deck`. It is plain Vite +
TypeScript + Tailwind v4's CSS toolchain + lucide-react: no Next, RSC, vinext,
Wrangler, or worker runtime remains.

The four screens are real-data surfaces:

- **Play** creates engine-authoritative server sessions, posts human moves,
  renders checkpoint inspection overlays, and launches recorded arena jobs.
- **Game history** pages and filters telemetry games, replays move blobs with
  their ply scalars, persists tags/notes, and shows calibration, blunders, and
  the D6-canonical opening atlas.
- **Live run** follows heartbeat and typed SSE, queries iteration/evaluation
  series, renders collector cohorts and artifacts, and drives the sentinel and
  kill controls.
- **Model lab** edits move prefixes, compares checkpoints, captures reference
  SDPA attention, runs the 12-way D6 check, and saves probes in `deck.db`.

`src/api.ts` is the one JSON client. `src/components/Board.tsx` and
`Chart.tsx` are shared real-data renderers; each screen lives in
`src/screens/`. `app/globals.css` retains the design draft's visual tokens and
shell, with the wired module styles at its end. There are no mock datasets in
the shipped source or bundle.

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

The build emits `dist/`. The test compiles the SPA and verifies that the built
HTML and client bundle carry all four deck surfaces without the removed
vinext/worker layer. In the intended environment, use the Compose
`frontend-build` service so the `deck` service can serve that directory.
