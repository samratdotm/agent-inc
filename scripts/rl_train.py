"""Agent Inc. — the real RL run (HUD-native GRPO via Tinker).

Turns the proven smoke-test loop into a measured before/after improvement.

    1. baseline: eval the fresh fork on ALL 30 scenarios            -> "before"
    2. train:    GRPO loop on a TRAIN subset (group rollouts -> step) x STEPS
    3. after:    eval the trained model on ALL 30 scenarios          -> "after"
    4. write:    results/training_curve.jsonl + post_rl_mean_reward into calibration.json
                 (the dashboard's RL-runway bar auto-fills)

Fork a FRESH model first so the baseline is unambiguous (base weights, 0 steps):

    hud models fork Qwen/Qwen3.5-4B --name agent-inc-rl-v2
    uv run python scripts/rl_train.py
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

# ── knobs (scale here) ────────────────────────────────────────────────────────
MODEL = "agent-inc-rl-v2"          # the FRESH fork (base weights)
GROUP = 8                          # rollouts per task per step (the GRPO group)
STEPS = 12                         # optimizer steps
LR = 1e-5
# Train on a mixed subset with headroom; measure on ALL 30 (tests generalization).
TRAIN_IDS = [
    "easy_ticket_triage",
    "easy_data_analysis_sales_summary",
    "medium_market_research",
    "medium_competitive_landscape",
    "hard_pricing_strategy",
]
REPO = Path(__file__).parent.parent
CURVE = REPO / "results" / "training_curve.jsonl"
CALIBRATION = REPO / "results" / "calibration.json"
RUNTIME = LocalRuntime("env.py")


def _mk(ids):
    tasks = []
    for sid in ids:
        t = client_engagement(scenario_id=sid)
        t.slug = sid
        tasks.append(t)
    return Taskset("agent-inc", tasks)


async def _eval_all(agent, label) -> float:
    """Eval the current model on all 30 scenarios; return mean reward."""
    ts = _mk([s["id"] for s in all_scenarios()])
    job = await ts.run(agent, runtime=RUNTIME, group=1)
    rewards = [r.reward for r in job.runs if r.reward is not None]
    mean = stats.fmean(rewards) if rewards else 0.0
    print(f"[{label}] mean_reward={mean:.3f} over {len(rewards)} scenarios "
          f"(job https://hud.ai/jobs/{job.id})")
    return mean


async def main() -> None:
    agent = create_agent(
        MODEL, max_steps=18, completion_kwargs={"extra_body": {"return_token_ids": True}}
    )
    trainer = TrainingClient(MODEL)
    train_ts = _mk(TRAIN_IDS)

    before = await _eval_all(agent, "before")

    CURVE.parent.mkdir(exist_ok=True)
    with CURVE.open("w") as f:
        f.write(json.dumps({"step": -1, "reward": before, "phase": "baseline"}) + "\n")

    job = await Job.start("agent-inc-rl-v2", group=GROUP)
    print(f"train job: https://hud.ai/jobs/{job.id}")
    for step in range(STEPS):
        start = len(job.runs)
        t0 = time.monotonic()
        await train_ts.run(agent, runtime=RUNTIME, group=GROUP, job=job)
        batch = [r for r in job.runs[start:] if r.reward is not None]
        mean = stats.fmean([r.reward for r in batch]) if batch else 0.0
        await trainer.step(batch, learning_rate=LR, group_size=GROUP)
        print(f"step {step:2d}: train_mean={mean:.3f} n={len(batch)} ({time.monotonic()-t0:.0f}s)")
        with CURVE.open("a") as f:
            f.write(json.dumps({"step": step, "reward": mean, "phase": "train"}) + "\n")

    after = await _eval_all(agent, "after")
    with CURVE.open("a") as f:
        f.write(json.dumps({"step": STEPS, "reward": after, "phase": "final"}) + "\n")

    # Surface the headline on the Qwen leaderboard entry for the dashboard runway.
    if CALIBRATION.exists():
        cal = json.loads(CALIBRATION.read_text(encoding="utf-8"))
        for e in cal.get("leaderboard", []):
            if e.get("agent") == "openai_compatible":
                e["post_rl_mean_reward"] = round(after, 3)
                e["rl_baseline_mean_reward"] = round(before, 3)
        CALIBRATION.write_text(json.dumps(cal, indent=2) + "\n", encoding="utf-8")

    delta = after - before
    print(f"\n=== RL RESULT ===  before={before:.3f}  after={after:.3f}  delta={delta:+.3f}")
    print("training curve -> results/training_curve.jsonl ; calibration.json updated.")


if __name__ == "__main__":
    asyncio.run(main())
