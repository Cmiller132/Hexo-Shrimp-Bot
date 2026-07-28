# Shrimp Control Deck

Design draft for the Hexo Shrimp Bot frontend. It is intentionally driven by
realistic mock data and local UI state rather than backend APIs.

## Screens

- **Play** — human vs checkpoint, bot vs bot, baseline opponents, quick suites,
  search settings, board overlays, candidate moves, and a queued pairing.
- **Game history** — archive metrics, search and filters, a dense game table,
  replay controls, tags, notes, and position handoff to the lab.
- **Live run** — learner and actor health, promotion gates, losses, checkpoint
  lifecycle, utilization, worker status, and a live event stream.
- **Model lab** — position editing, checkpoint comparison, policy/value
  readouts, attention, activations, search, input planes, and saved probes.

The persistent compare tray and command palette are shared across screens.

## Run locally

Requires Node.js 22.13 or newer.

```bash
npm install
npm run dev -- --host 0.0.0.0
```

The `--host 0.0.0.0` flag makes the draft reachable from other devices on the
LAN. This repository does not add authentication or expose a public deployment.

## Validate

```bash
npm test
```

The production build and rendered shell test should both pass. Backend actions,
live telemetry, persistence, and destructive run controls are deliberately
inert in this draft.
