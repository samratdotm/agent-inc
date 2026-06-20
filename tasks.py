"""Agent Inc. tasks — one client engagement per scenario in data/scenarios/.

    hud eval tasks.py claude                       # first scenario
    hud eval tasks.py claude --full                # the whole taskset
    hud eval tasks.py claude --task-ids easy_ticket_triage -y --max-steps 12
"""

from env import all_scenarios, client_engagement, env  # noqa: F401  (re-export env for `hud eval`)


def _build() -> list:
    tasks = []
    for scenario in all_scenarios():
        task = client_engagement(scenario_id=scenario["id"])
        task.slug = scenario["id"]
        tasks.append(task)
    return tasks


tasks = _build()
