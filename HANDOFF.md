# Handoff — 2026-08-26

Resume map only. Operational state, no results, no hypotheses.

## Rulings in force

- **GPU frozen (2026-08-26).** The v3 screen queue was killed mid-cell on
  owner order; the GPU is free and stays free — no new GPU work of any kind
  (runs, benches, CUDA test lanes) without an explicit owner go-ahead.
  CPU-side work is unaffected.
- **Epistemic reset (2026-08-26).** `cellnodes-1`
  (`runs/cellnodes-1/checkpoint_000402.pt`) is the baseline to improve. The
  prior screen-ablation record is retired and `docs/ABLATIONS.md` deleted;
  nothing from it is citable as a finding. Evidence for strength claims is
  matches with sequential statistics, or deterministic interventions on
  trained checkpoints, at statistical significance.
- **Document bar.** Docs are slim contracts. Writes carry a high barrier:
  statistically significant, backed claims only — no hypotheses, no
  best-guess narratives, no run history.
- **Parked as provenance** (cite nothing from them): `claude/merged-sites` @
  `40a0f81`, `claude/site-attention` @ `0eb72e9`, the partial WSL
  `runs/lab/v3` screen data, `scripts/launch_confirm.sh` + `run/confirm-1`.

## Active program: attribution of cellnodes-1

Goal: determine what each stage of the trained model contributes to its
strength, then improve it against that baseline.

1. **Era environment.** Container `deck-repr7` mounts
   `.claude/worktrees/step13-run` (the tree the checkpoints were written by),
   the runs directory, `local/`, and SealBot. Current main cannot load the
   checkpoint faithfully. Faithfulness gate: reproduce a recorded
   `runs/cellnodes-1/metrics.jsonl` evaluation before any intervention
   counts.
2. **Instrument.** Stage×block eval-time knockouts of the checkpoint play
   SPRT paired matches against the intact model
   (`mantisnet.klent.opponents.opponent_match`, Gumbel chooser sims=32 — the
   run's own eval operating point). Ladder rungs: the strix anchor
   (`local/strix-eval/`, bridge host on WSL :9787) and `stack-939`. The
   solver classifies divergence points as in-horizon (exactly verified) or
   beyond-horizon. Raw-policy vs sims-32 matches separate policy-carried
   from search-recovered strength.
3. **Code.** Analysis harness lives in `local/attribution/` (untracked, the
   `local/` convention). A mantis-vs-mantis `Opponent` wrapper and the SPRT
   driver are to be written; everything else exists.

## Environment

- Strix bridge: `local/strix-eval/seat_bridge_host.py` on :9787; containers
  reach it at `STRIX_SEAT_ADDR=172.18.0.1:9787`.
- Deck containers all stopped (`restart=unless-stopped`, manual stops
  stick): `deck-repr7`, `docker-deck-1`, `deck-guard`, `deck-tunnel`,
  `deck-lan-8080`. `cellnodes1r` / `tactical1r2` hold resumable runs.
- WSL detached jobs need a live wsl.exe client (VM idle-death); one GPU
  consumer at a time.
