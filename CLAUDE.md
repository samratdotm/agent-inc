# Agent Inc. — project context for Claude

**What this is:** a HUD **v6** RL environment where an agent runs a business (read brief →
research → truthful priced offer → structured deliverable). For the HUD × YC Frontier RL
Hackathon, Autonomous Business track. *"SWE-bench taught models to code; we teach them to run a business."*

## ⚠️ Answering HUD-product questions: ground in docs, do NOT guess
The model's training data on HUD is thin/outdated — an early plan *guessed* the SDK and was wrong.
For anything about HUD's API/SDK/CLI: **verify against the installed package** (`hud-python` 0.6.6,
under `.../uv/tools/hud-python/...`) or **fetch `docs.hud.ai` / use the Context7 MCP**. Never answer HUD
specifics from memory.

## The REAL HUD v6 API (verified, corrects common wrong guesses)
- Env: `from hud import Environment; env = Environment(name="agent-inc")`.
- **Tools are plain `async def` functions served over a FastMCP server** registered in
  `@env.initialize` + `env.add_capability(Capability.mcp(...))` — **NOT** `@env.tool()` decorators.
- Tasks: `@env.template()` async generator — `yield prompt`, then `yield reward/EvaluationResult`.
- Grader: `from hud.graders import combine, SubScore, LLMJudgeGrader, EvaluationResult`.
- Eval CLI: `hud eval tasks.py <agent> -y --max-steps N --gateway` (agents: `claude`, `gemini`,
  `openai_compatible`; always pass `-y` non-interactively; `--gateway` routes via HUD using `HUD_API_KEY`).
- RL: `hud models fork Qwen/Qwen3.5-4B --name <m>` → `hud.TrainingClient(m).step(runs, learning_rate, group_size)`
  (GRPO / `importance_sampling`, on-policy, on the **Tinker** backend).

## Architecture
- `env.py` — 7 FastMCP tools + the deterministic verifier + `client_engagement` template + hybrid reward.
- `tasks.py` — builds one task per scenario in `data/scenarios/*.json` (data-driven; add a JSON, no code).
- `data/scenarios/` — 30 scenarios (10 easy / 10 medium / 10 hard, 9 business domains).
- `scripts/` — `rl_train.py` (hardened+resumable GRPO run), `rl_supervise.sh` (auto-relaunch + lock),
  `pool_probe.py` (Tinker health), `rl_status.sh` (one-glance status).
- `tests/` — offline, key-free (96 tests): good work ≥ 0.8, bad/dishonest ≤ 0.3.

## Reward (hybrid, [0,1]) — 70% deterministic + 30% LLM judge
completeness .25 · **quality .30 (LLM judge = `claude-haiku-4-5` via gateway)** · pricing .20 ·
efficiency .15 · policy/honesty .10. Delivery-gated: an undelivered offer can't coast.

## Models (don't confuse the roles)
- **Claude Sonnet 4.6** — leaderboard top / frontier reference (~0.95).
- **Qwen3.5-4B (Tinker fork)** — the open model we RL-train (on-policy GRPO, its OWN rollouts; NOT Claude traces).
- **Claude Haiku 4.5** — the LLM judge inside the grader (not a trace source).

## Result-integrity rules (NON-NEGOTIABLE — see `RESULTS_INTEGRITY.md`)
Honesty over a good number. Never cache/report a metric from a degraded pool. `results/calibration.json`
is the sacred canonical score — written only with validity-checked clean numbers; junk → `results/quarantine.jsonl`.
Reject any baseline below the plausibility floor. Label data real / sample / pending. When unsure, under-claim.

## Run / status
- `uv sync` · `uv run pytest tests/ -q` (offline) · `hud eval tasks.py claude -y --gateway`
- RL status anytime: `bash scripts/rl_status.sh`
- More: `README.md` (overview), `CONTRACT.md` (P1/P2 interface seam), `RESULTS_INTEGRITY.md` (honesty rules).
