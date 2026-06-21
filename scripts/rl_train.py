"""Agent Inc. — RL run (HUD-native GRPO via Tinker), HARDENED + RESUMABLE.

Built to survive a degraded Tinker pool (503 on rollouts, 504 on the gradient call):
  - per-step retry with backoff (HUD says a failed step is safe to re-run; the
    server dedups), so a transient timeout no longer crashes the run;
  - resume from saved checkpoints — completed steps persist on HUD, so a relaunch
    continues from where it died instead of restarting (baseline is cached too);
  - a done-marker (results/rl_done.json) the supervisor watches.

    hud models fork Qwen/Qwen3.5-4B --name agent-inc-rl-v4   # once
    uv run python scripts/rl_train.py                         # (re)launchable
"""

import asyncio
import json
import statistics as stats
import time
from pathlib import Path

from hud import TrainingClient
from hud.agents import create_agent
from hud.eval import Job, LocalRuntime, Taskset

from env import all_scenarios, client_engagement

# ── knobs ─────────────────────────────────────────────────────────────────────
MODEL = "agent-inc-rl-v4"          # fresh fork
GROUP = 8
STEPS = 18
LR = 2e-5
MAX_CONCURRENT = 3                 # Tinker pool ceiling (degraded); retries cover the rest
TRAIN_IDS = [
    "easy_ticket_triage", "easy_data_analysis_sales_summary",
    "medium_market_research", "medium_competitive_landscape",
    "medium_gtm_segment_launch_plan", "hard_pricing_strategy",
]
EXPECTED_BATCH = len(TRAIN_IDS) * GROUP
STEP_MAX_WAIT = 900                # retry a wedged step up to ~15 min, then exit (supervisor relaunches)
# Prior CLEAN baselines for this base model were 0.26-0.39. A baseline below this floor
# means the eval ran on a degraded pool (rollouts failing) and is contaminated — reject it
# rather than cache a junk number that would fake a huge "improvement" later.
MIN_PLAUSIBLE_BASELINE = 0.20

REPO = Path(__file__).parent.parent
CURVE = REPO / "results" / "training_curve.jsonl"
CALIBRATION = REPO / "results" / "calibration.json"
STATE = REPO / "results" / "rl_state.json"
DONE = REPO / "results" / "rl_done.json"
RUNTIME = LocalRuntime("env.py")


def _mk(ids):
    tasks = []
    for sid in ids:
        t = client_engagement(scenario_id=sid)
        t.slug = sid
        tasks.append(t)
    return Taskset("agent-inc", tasks)


async def _retry(make_coro, what, max_wait=STEP_MAX_WAIT):
    """Await make_coro(); on ANY failure (503/504/timeout) wait+retry up to max_wait."""
    delay, waited = 15, 0
    while True:
        try:
            return await make_coro()
        except Exception as exc:  # noqa: BLE001 - degraded pool throws many shapes
            if waited >= max_wait:
                raise
            print(f"  [retry] {what}: {str(exc)[:160]} | waited {waited}s, sleeping {delay}s", flush=True)
            await asyncio.sleep(delay)
            waited += delay
            delay = min(delay * 2, 120)


def _quarantine(label: str, value: float, reason: str) -> None:
    """Record a rejected number for transparency — it NEVER touches calibration.json."""
    with (REPO / "results" / "quarantine.jsonl").open("a") as f:
        f.write(json.dumps({"label": label, "value": round(value, 4), "reason": reason}) + "\n")
    print(f"  [QUARANTINE] {label}={value:.3f}: {reason}", flush=True)


async def _eval_all(agent, label) -> float:
    ts = _mk([s["id"] for s in all_scenarios()])
    job = await _retry(
        lambda: ts.run(agent, runtime=RUNTIME, group=1, max_concurrent=MAX_CONCURRENT),
        f"eval:{label}",
    )
    rewards = [r.reward for r in job.runs if r.reward is not None]
    mean = stats.fmean(rewards) if rewards else 0.0
    print(f"[{label}] mean_reward={mean:.3f} over {len(rewards)} scenarios "
          f"(job https://hud.ai/jobs/{job.id})", flush=True)
    return mean


async def main() -> None:
    agent = create_agent(MODEL, max_steps=18,
                         completion_kwargs={"extra_body": {"return_token_ids": True}})
    trainer = TrainingClient(MODEL)
    train_ts = _mk(TRAIN_IDS)

    # ── resume: how many steps already landed, and the cached baseline ──────────
    done_steps = len(await _retry(lambda: trainer.checkpoints(), "checkpoints"))
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    if state.get("model") == MODEL and "baseline" in state:
        before = state["baseline"]
        print(f"[resume] {MODEL}: {done_steps} steps done, baseline={before:.3f}", flush=True)
    elif done_steps == 0:
        before = await _eval_all(agent, "before")
        if before < MIN_PLAUSIBLE_BASELINE:
            _quarantine("baseline", before, "below plausibility floor — degraded pool, rollouts failing")
            raise SystemExit(
                f"baseline {before:.3f} < {MIN_PLAUSIBLE_BASELINE} — NOT caching this junk number. "
                "Supervisor will retry on a healthier pool."
            )
        STATE.parent.mkdir(exist_ok=True)
        STATE.write_text(json.dumps({"model": MODEL, "baseline": before}))
        CURVE.write_text(json.dumps({"step": -1, "reward": before, "phase": "baseline"}) + "\n")
    else:
        raise SystemExit(f"{MODEL} has {done_steps} checkpoints but no saved baseline — "
                         "fork a fresh model so the before/after is clean.")

    job = await Job.start(MODEL, group=GROUP)
    for step in range(done_steps, STEPS):
        start = len(job.runs)
        t0 = time.monotonic()
        # rollouts: the openai client retries 503s per-call internally; wrap once more for safety
        await _retry(lambda: train_ts.run(agent, runtime=RUNTIME, group=GROUP,
                                          job=job, max_concurrent=MAX_CONCURRENT),
                     f"rollouts:step{step}")
        batch = job.runs[start:][-EXPECTED_BATCH:]            # trailing full batch (retry-safe)
        mean = stats.fmean([r.reward for r in batch if r.reward is not None]) if batch else 0.0
        # the 504 culprit — HUD dedups a re-run step, so retry is safe
        await _retry(lambda: trainer.step(batch, learning_rate=LR, group_size=GROUP),
                     f"trainer.step{step}")
        print(f"step {step:2d}: train_mean={mean:.3f} n={len(batch)} ({time.monotonic()-t0:.0f}s)", flush=True)
        with CURVE.open("a") as f:
            f.write(json.dumps({"step": step, "reward": mean, "phase": "train"}) + "\n")

    after = await _eval_all(agent, "after")
    if after < MIN_PLAUSIBLE_BASELINE:
        # An implausibly low 'after' almost always means a degraded pool, not a real
        # collapse. Do NOT finalize or touch calibration.json — re-eval clean on relaunch.
        _quarantine("after", after, "below plausibility floor — degraded pool; not finalizing")
        raise SystemExit(f"after {after:.3f} implausibly low — NOT writing results. Supervisor re-evals.")
    with CURVE.open("a") as f:
        f.write(json.dumps({"step": STEPS, "reward": after, "phase": "final"}) + "\n")

    # Canonical calibration.json is touched ONLY here, with a clean before AND after.
    if CALIBRATION.exists():
        cal = json.loads(CALIBRATION.read_text(encoding="utf-8"))
        for e in cal.get("leaderboard", []):
            if e.get("agent") == "openai_compatible":
                e["post_rl_mean_reward"] = round(after, 3)
                e["rl_baseline_mean_reward"] = round(before, 3)
        CALIBRATION.write_text(json.dumps(cal, indent=2) + "\n", encoding="utf-8")

    DONE.write_text(json.dumps({"model": MODEL, "before": before, "after": after,
                               "delta": round(after - before, 3)}))
    print(f"\n=== RL RESULT ===  before={before:.3f}  after={after:.3f}  delta={after-before:+.3f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
