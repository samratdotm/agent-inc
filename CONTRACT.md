# Agent Inc. — Interface Contract (v6, re-frozen 2026-06-20)

> The seam between **P1** (env/grader/eval/RL) and **P2** (live tool integrations/dashboard/demo).
> This supersedes the tool spec in `RL-Idea-extracted.md`, which was drafted before we saw the
> real HUD v6 API. Change a signature only by editing this file + pinging the other person.

## What changed from the original plan (and why)

| Original plan | Reality (v6) | Why |
|---|---|---|
| Tools via `@env.tool()` | Plain `async def` tools registered on a **FastMCP** server, served over a `Capability.mcp(...)` | That's how v6 templates (autonomous-businesses, deepresearch) actually expose tools |
| `price_project` + `check_calendar` tools | **dropped**; pricing is an arg of `send_offer`; calendar added no verifiable signal | Fewer tools, stronger signal |
| `write_deliverable` + `submit_deliverable` | merged into one `submit_deliverable(artifact)` | The artifact *is* the deliverable |
| Hand-rolled weighted grader | `hud.graders` `combine(SubScore...)` + `LLMJudgeGrader.grade(...)` | Use the SDK's primitives |
| Deliverable graded by LLM judge only | **deterministic substance check + LLM judge** | Defends the "judge is gameable" critique |
| Qwen needs custom wrapper | built-in `openai_compatible` agent type | `hud eval tasks.py openai_compatible -m <fireworks-model>` |

## Tools (P2 owns the live API bodies; signatures are FROZEN)

```python
async def read_client_brief() -> dict          # {brief, budget, must_have, deliverable_schema}
async def read_company_capabilities() -> dict   # {can_do, cannot_do}
async def search_web(query: str) -> list[dict]  # [{title, url, snippet}]   ← Exa (mock-first)
async def research_company(name: str, website: str = "") -> dict  # firmographics ← Sixtyfour (mock-first)
async def send_offer(scope: str, price: float, claims: list[str]) -> dict  # {accepted, reason}
async def submit_deliverable(artifact: str) -> dict  # {completeness, feedback}
async def get_business_state() -> dict          # {offer_sent, offer_accepted, deliverable_submitted, tool_calls}
```
Every tool calls `_bump()` first (efficiency accounting). Mock-first: `search_web`/`research_company`
return clearly-labelled mocks when `EXA_API_KEY` / `SIXTYFOUR_API_KEY` are unset — P2 swaps in the
live calls behind the same signatures (the live bodies already exist; they just need keys).

## Scenario schema (`data/scenarios/<id>.json`) — drives everything, no code change to add one

```json
{
  "id": "easy_ticket_triage", "difficulty": "easy|medium|hard", "domain": "string",
  "brief": "client request", "company_can_do": [..], "company_cannot_do": [..],
  "budget": 100, "budget_range": [50, 100],
  "must_have": [".."],            // -> completeness (deterministic token coverage)
  "reject_if_claims": [".."],      // -> policy (honesty)
  "deliverable_schema": {..},      // optional JSON schema gate
  "expected_topics": [".."], "reference": "ground truth for the judge", "tool_budget": 8
}
```

## Reward (0..1) — `combine()` of five SubScores

`completeness 0.25` (deterministic) · `quality 0.30` (LLM judge, needs `HUD_API_KEY`) ·
`pricing 0.20` · `efficiency 0.15` · `policy 0.10`. Deterministic 0.70 runs key-free.

## Result format (eval emits → dashboard reads)

HUD writes per-run results from `hud eval`; the dashboard (P2) reads them. The `EvaluationResult`
carries `.reward` (float) plus per-criterion `SubScore`s with `.value`/`.weight`/`.metadata`.
Exact on-disk path is set when we wire the eval runner in Phase 2 — P2: don't hardcode it yet.
