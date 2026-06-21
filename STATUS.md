# Agent Inc. — Current Status (last updated: Sun Jun 21, ~08:35 PDT)

## ✅ The headline result (REAL, clean, demo-ready)
**RL trained the open model from 0.327 → 0.647 (+0.321) on all 30 scenarios.**
- Base Qwen3.5-4B baseline: **0.327** (clean, passed plausibility floor).
- RL-trained (`agent-inc-rl-v4`, head `step-000016`, 16 GRPO steps): **0.647**.
- **30/30 scenarios graded** (judge-health guard passed → uncontaminated).
- Generalized, not overfit: the after-eval includes the 24 scenarios it never trained on.
- Recorded in `results/calibration.json` (`post_rl_mean_reward: 0.647`), `results/rl_done.json`, `results/training_curve.jsonl`.

## Leaderboard (for the demo)
Claude Sonnet 4.6 **0.951** ≫ base Qwen3.5-4B **~0.33** → **Qwen+RL 0.647** (closed ~half the gap).

## Phase status
- Phase 0 (env + green run): ✅  · Phase 1 (30 scenarios + grader): ✅
- Phase 2 (multi-model leaderboard): ✅  · Phase 3 (RL train → improve): ✅ **(0.327→0.647)**
- Phase 4 (dashboard + demo): in progress — P2/teammate (issues #4-#7); dashboard auto-shows the RL runway now.

## State of the training run
- **Training is STOPPED** (we pivoted to the verdict eval). 16 checkpoints saved on HUD; head = `step-000016` (0.689 train-reward).
- `scripts/rl_train.py` is currently in **eval-only mode** (`STEPS = 16`).

## How to resume / re-run (if needed)
- **Resume training** (finish more steps): set `STEPS = 18` (or higher) in `scripts/rl_train.py`, then `bash scripts/rl_supervise.sh` (it resumes from the 16 checkpoints). Delete `results/rl_done.json` first.
- **Re-run the verdict eval**: keep `STEPS <= 16`, run `EXA_API_KEY= SIXTYFOUR_API_KEY= uv run python -u scripts/rl_train.py` (mocked tools, apples-to-apples with the baseline).
- **Watch status**: `bash scripts/rl_status.sh`.

## Hard-won gotchas (don't relearn these)
- Tinker pool (training backend) + Anthropic provider (the `claude-haiku-4-5` judge) both 503'd repeatedly under hackathon load. Guards: plausibility floor on baseline, graded-fraction guard on after-eval, retry+resume, single-instance lock. A deep connection-level **hang** once slipped past the per-call timeout → a watchdog/heartbeat is the remaining robustness gap.
- Integrity rule: NEVER cache/report a metric from a degraded pool. See `RESULTS_INTEGRITY.md`.

## Deadlines
First training run kicked off: ✅ (done long ago). **Submission due 1 PM Sun.** Top-10 present 2:30 PM.
