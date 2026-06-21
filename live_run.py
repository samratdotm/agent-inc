"""Live single-engagement runner for the dashboard (P2).

Runs ONE `client_engagement` through the HUD gateway and returns the scored
result (reward + per-criterion subscores). No streamlit import — callable and
testable on its own.

    from live_run import run_engagement
    res = run_engagement("easy_ticket_triage", "claude-sonnet-4-6")
    # -> {"reward": 1.0, "subscores": [{name,value,weight,contribution}, ...], ...}

Each call is a real eval: it costs HUD gateway credits and needs the gateway up.
Use a real gateway model id (e.g. claude-sonnet-4-6 / gemini-3.1-pro-preview),
NOT the agent alias ("claude"), which the gateway rejects.
"""

from __future__ import annotations

import asyncio
from typing import Any

DEFAULT_MODEL = "claude-sonnet-4-6"


async def _run_async(scenario_id: str, model: str, max_steps: int) -> dict[str, Any]:
    # Imported lazily so importing this module stays cheap (no hud/env at import).
    from hud.agents import create_agent
    from hud.eval import LocalRuntime, Taskset

    from env import client_engagement

    task = client_engagement(scenario_id=scenario_id)
    task.slug = scenario_id
    taskset = Taskset("agent-inc-live", [task])
    agent = create_agent(model, max_steps=max_steps)
    job = await taskset.run(agent, runtime=LocalRuntime("env.py"), group=1)

    run = job.runs[0]
    ev = run.evaluation if isinstance(run.evaluation, dict) else {}
    subs: list[dict[str, Any]] = []
    for s in ev.get("subscores") or []:
        value = float(s.get("value", 0.0))
        weight = float(s.get("weight", 0.0))
        subs.append(
            {"name": s.get("name", "?"), "value": value, "weight": weight, "contribution": round(value * weight, 3)}
        )
    return {
        "scenario_id": scenario_id,
        "model": model,
        "reward": float(run.reward) if run.reward is not None else None,
        "subscores": subs,
        "job_id": getattr(job, "id", None),
    }


def run_engagement(scenario_id: str, model: str = DEFAULT_MODEL, max_steps: int = 18) -> dict[str, Any]:
    """Run one engagement and return its scored result (blocking).

    Safe to call from a Streamlit button handler: falls back to a worker thread
    if an event loop is already running in this thread.
    """
    try:
        return asyncio.run(_run_async(scenario_id, model, max_steps))
    except RuntimeError:
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(lambda: asyncio.run(_run_async(scenario_id, model, max_steps))).result()
