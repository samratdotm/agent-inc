# Agent Inc.

**SWE-bench taught models to code. Agent Inc. teaches them to run a business.**

A HUD v6 RL environment where an agent acts as an **autonomous business operator**:
it reads a client brief, researches, makes a truthful + affordable + in-scope offer,
and submits a structured deliverable — and only gets paid for honest, complete work.
Built for the HUD × YC Frontier RL Environments Hackathon (Track: Autonomous Business).

## The loop

```
client_engagement(scenario):
  read_client_brief + read_company_capabilities
    → search_web / research_company        (Exa / Sixtyfour; mock-first without keys)
    → send_offer(scope, price, claims)      accepted iff affordable, relevant, honest
    → submit_deliverable(artifact)          structured JSON, checked for substance
```

Scenarios live in `data/scenarios/*.json` (multi-domain: workflow automation, market
research, pricing strategy, …) across `easy` / `medium` / `hard`. Add a scenario by
dropping in a new JSON file — no code change.

## Reward (hybrid, [0, 1])

| Criterion | Weight | How |
|---|---|---|
| completeness | 0.25 | deterministic: must-have content coverage (+ optional schema) |
| quality | 0.30 | LLM judge over the deliverable vs. the brief (needs `HUD_API_KEY`) |
| pricing | 0.20 | offer price lands in the scenario's budget band |
| efficiency | 0.15 | agent stayed within its tool-call budget |
| policy | 0.10 | honesty: no claims the business cannot back up |

The deterministic **0.70** runs offline with no key, so tests and RL reward shaping
work key-free; the **0.30** quality slice needs `HUD_API_KEY` for the judge. Pairing
the judge with deterministic checks is our answer to *"how do you know the judge isn't fooled?"*

## Run

Needs [uv](https://docs.astral.sh/uv/), Python 3.11–3.12, and the HUD CLI (`uv tool install hud-python`).

```bash
uv sync
hud set HUD_API_KEY=your-key            # or: cp .env.example .env && edit

hud eval tasks.py claude                # first scenario
hud eval tasks.py claude --full         # the whole taskset
hud eval tasks.py gemini --full         # leaderboard comparison
```

## Tests (offline, no keys)

```bash
uv run pytest tests/ -q
```

## Layout

```
env.py            tools, the deterministic verifier, and the client_engagement task
tasks.py          builds one task per scenario in data/scenarios/
data/scenarios/   the briefs (multi-domain, multi-difficulty) — the env is data-driven
tests/            offline tests proving good work beats bad work, no key required
_templates/       upstream HUD starters kept for reference (autonomous-businesses, deepresearch)
```
