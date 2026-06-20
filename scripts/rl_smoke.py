"""Agent Inc. — RL smoke test (HUD-native GRPO via Tinker).

De-risks the training path that must be running by 8 AM Sunday: fork already done
(`hud models fork Qwen/Qwen3.5-4B --name agent-inc-rl`), this script proves the
rollout -> group-relative advantage -> optim_step -> checkpoint loop end-to-end on
a couple of scenarios. Small on purpose. Scale up (more tasks/steps) once it's green.

    uv run python scripts/rl_smoke.py
"""

import asyncio

from hud import TrainingClient
from hud.agents import create_agent
from hud.eval import Job, LocalRuntime, Taskset

from env import client_engagement

MODEL = "agent-inc-rl"          # the trainable fork (gateway slug)
PICK = ["easy_ticket_triage", "medium_market_research"]  # room to improve + reward variance
GROUP = 4                        # rollouts per task per step (the GRPO group)
STEPS = 3                        # optimizer steps
LR = 1e-5


async def main() -> None:
    tasks = []
    for sid in PICK:
        t = client_engagement(scenario_id=sid)
        t.slug = sid
        tasks.append(t)
    taskset = Taskset("agent-inc-rl-smoke", tasks)

    # return_token_ids -> the trainable rollout carries token-level samples to train on.
    agent = create_agent(
        MODEL,
        max_steps=18,
        completion_kwargs={"extra_body": {"return_token_ids": True}},
    )
    trainer = TrainingClient(MODEL)

    job = await Job.start("agent-inc-rl-smoke", group=GROUP)
    print(f"job: https://hud.ai/jobs/{job.id}")

    for step in range(STEPS):
        start = len(job.runs)
        await taskset.run(agent, runtime=LocalRuntime("env.py"), group=GROUP, job=job)
        batch = [r for r in job.runs[start:] if r.reward is not None]
        rewards = [r.reward for r in batch]
        mean = sum(rewards) / len(rewards) if rewards else float("nan")
        result = await trainer.step(batch, learning_rate=LR, group_size=GROUP)
        loss = getattr(result, "loss", None)
        print(f"step {step}: n={len(batch)} mean_reward={mean:.3f} loss={loss}")

    cps = await trainer.checkpoints()
    head = await trainer.head()
    print(f"checkpoints={len(cps)} active_head={getattr(head, 'id', None)}")
    print("RL smoke test complete — the train->improve loop runs end-to-end.")


if __name__ == "__main__":
    asyncio.run(main())
