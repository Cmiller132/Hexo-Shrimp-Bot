# Handoff — 2026-08-24, end of day

State of the improve-and-de-bloat campaign at the stopping point. The repo is
the record: measured results live in `docs/ABLATIONS.md`; this file is the
resume map only.

## Rulings in force

- **confirm-1 is cancelled** (owner, 2026-08-24). Do not launch it; the old
  launch handoff is deleted. `scripts/launch_confirm.sh` and branch
  `run/confirm-1` remain as provenance only.
- **Site attention is retained as an open option** (owner, 2026-08-24 EOD).
  The screen was negative — S −5.48, full record in the ABLATIONS row — but
  the owner judges the screen may not capture its benefits. The knob is NOT
  merged and NOT deleted: it lives complete on branch `claude/site-attention`
  (tip `0eb72e9`, worktree `.claude/worktrees/site-attention`, pushed).
  Evidence for the option: seed 1 trained to the exact fixture mean, so the
  mechanism works; the 2-of-3 policy collapses are a training-stability
  failure under the fixed screen recipe, not an expressiveness ceiling. A
  retry means a stability change (e.g. lower LR or warmup on the joint
  softmax) and a fresh 3-seed screen — owner-initiated only.

## Repo state

- `main` is the safe tree: suite speedup `efac2b7`, ABLATIONS record
  `d174c90`, this handoff. Pushed to origin.
- `claude/site-attention` = main + 7 commits: the `site_attention` knob,
  the FlexAttention document-mask backend and its fixes, 12 green tests in
  `tests/test_site_attention.py`. The two suite commits on it are already
  cherry-picked to main; a future merge of the branch will see them as
  near-empty conflicts on `conftest.py`/`pyproject.toml` — take the branch
  side plus main's removal of nothing.
- Scored cells for the arm: WSL `/root/graft-v2/python/mantisnet/runs/lab/v2/
  site-attention/s{0,1,2}` (ext4 side, not in git). Verdict rerun:
  `cd /root/graft-v2/python/mantisnet && /root/graft-v2-venv/bin/python -m
  mantisnet.lab.screen verdict --fixture runs/lab/v2/fixture --arm
  site-attention=runs/lab/v2/site-attention`.

## Environment

- GPU free; no queue, no monitors, no containers training.
- Deck stack STOPPED by owner order. Restore:
  `docker start deck-repr7 docker-deck-1 deck-guard deck-tunnel deck-lan-8080`.
- Strix bridge was UP on :9787 at stop; it dies with reboots — before any
  launch, check the port and restart `local/strix-eval/seat_bridge_host.py`
  detached from WSL.
- WSL `/root/graft-v2` is checked out on `claude/site-attention`; venv
  `/root/graft-v2-venv` (no pip — `uv pip install --python …`). Never run
  pytest or benches from `/mnt/d`.
- Fast suite (2:46, all 513): from ext4,
  `CUDA_VISIBLE_DEVICES= pytest tests/ -n auto -m "not cuda_lane"` then
  `pytest tests/ -m cuda_lane`.

## Next up (owner list)

1. Wave-1 provable cuts: remove the stone-attention bias tables (the
   knock-out showed them decorative; also removes the `tl.atomic_add`
   screen-noise source), S9 dead block-3 stone rows, S4 table dedup.
2. Wave-2 dead-knob purge: `action_latents`/`line_pass`/`cell_adjacency`
   deletion; `window_attention` one v2 retest arm; `uncovered` scope delete
   decision.
3. Wave-3: stronger eval anchor FIRST, then corpus refresh, then the
   h/depth scale screen (VRAM work bought the headroom).
4. trigraft: dead-machinery deletion candidate (its 3×49 s tests are the
   largest remaining suite cost).
5. `cell_structure` keep/delete (R1: S +1.44, policy-only gain).
6. Site-attention stability retry — open option per the ruling above.
