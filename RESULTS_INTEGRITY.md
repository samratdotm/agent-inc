# Result Integrity Rules (NON-NEGOTIABLE)

> Honesty over a good-looking number. A wrong number on stage is worse than a modest one.
> These rules bind every script, dashboard, and human on this project.

## The rules

1. **Never cache or report an empty/contaminated result.** A number measured while the
   Tinker pool (or any infra) is degraded — rollouts erroring, 5xx storms — is **not data**.
   It must never be saved as a baseline, an "after", or anything downstream depends on.

2. **The canonical scores are sacred.** `results/calibration.json` (the leaderboard + RL
   before/after) is the single source of truth the demo and submission cite. It is written
   **only** with a result that has passed validity checks. Nothing degraded or partial ever
   overwrites it. Bad numbers go to `results/quarantine.jsonl`, never the canonical file.

3. **A contaminated baseline is the worst case** — a low junk baseline fakes a huge
   "improvement" later. Reject any baseline below the plausibility floor
   (`MIN_PLAUSIBLE_BASELINE`, set from prior CLEAN baselines) and re-measure on a healthy pool.

4. **Before/after must be comparable.** Both ends measured under conditions that pass the
   same validity gate; a delta across a degraded vs. healthy pool is not a real delta.

5. **Quarantine, don't delete.** Rejected numbers are logged to `results/quarantine.jsonl`
   for transparency (so we can show we caught them), but they are inert — never referenced.

6. **Label real vs. sample/pending.** Any dashboard or slide must clearly mark a number as
   real / sample / pending. Never present sample or contaminated data as a measured result.

7. **When unsure, under-claim.** If we can't certify a number is clean, we say "pending a
   healthy pool" — we do not ship it.

## How this is enforced in code

- `scripts/rl_train.py`: plausibility floor on **both** baseline and after; `_quarantine()`
  records rejects; `calibration.json` is touched only with a clean before+after.
- `scripts/rl_supervise.sh` + `scripts/pool_probe.py`: health-gate launches; auto-retry on a
  healthier pool so we wait for clean data instead of caching junk.
