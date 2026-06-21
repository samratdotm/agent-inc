# Agent Inc. — Demo Flow (submission day)

**Headline:** Qwen3.5-4B went from 0.327 → 0.647 (+0.321) after 16 GRPO steps on our environment.
Closed ~half the gap to Claude (0.951). Graded clean on all 30 scenarios including 24 it never trained on — so it generalised, not memorised.

---

## Step 1 — Show the dashboard

Launch:
```bash
uv run --with streamlit --with pandas streamlit run dashboard.py
```

Walk through three panels in order:

**Leaderboard** — point out the frontier ceiling (Claude 0.951, Gemini 0.910) and where Qwen base sat (0.327). This sets up the "how much room is there to improve?" story.

**RL runway bar** — show the before/after: Qwen base 0.327 → Qwen+RL 0.647. One bar tells the whole story. Mention: this is real, graded on all 30 scenarios, passed integrity checks.

**Training curve** — show the reward climbing over 16 steps (0.327 baseline → peaks at 0.689 step 14 → final eval 0.647). The upward trend is the model learning the business loop, not memorising.

---

## Step 2 — Run one scenario live in the terminal

Pick a scenario (suggest `easy_ticket_triage` — it's fast and clear):

```bash
hud eval tasks.py claude -y --gateway
```

Walk through what the agent does out loud as the output arrives:
1. Reads the client brief and company capabilities
2. Searches the web / researches the company (Exa / Sixtyfour)
3. Sends a priced offer — scope, price, claims
4. Submits a structured deliverable
5. Grader scores it across 5 criteria

---

## Step 3 — Explain the grader output

Five criteria, 0–1 each:

| Criterion | Weight | What it checks |
|---|---|---|
| completeness | 0.25 | did the deliverable cover the must-have topics? (deterministic) |
| quality | 0.30 | is the deliverable actually good? (LLM judge — Claude Haiku 4.5) |
| pricing | 0.20 | did the offer land in the client's budget band? |
| efficiency | 0.15 | did the agent stay within its tool-call budget? |
| policy | 0.10 | no false claims the business can't back up (honesty) |

Key point to make: 70% of the reward is deterministic Python — the judge can't be gamed. The 30% LLM judge is on top of that, not instead of it.

---

## Key talking points

- "SWE-bench taught models to code. Agent Inc. teaches them to run a business."
- The environment is data-driven — add a scenario by dropping a JSON file, no code change.
- RL training used GRPO (on-policy) — the model learned purely from its own attempts, not Claude traces.
- Result integrity was a hard engineering problem: the shared infra 503'd repeatedly. We built guards (plausibility floor, graded-fraction check) so a degraded server can't fake a good number.
- The improvement generalised: 6 scenarios trained on, 30 evaluated. The model learned the loop, not 6 answers.

---

## Backup plan (if HUD gateway is slow at demo time)

Don't run live — show the training curve and `results/rl_done.json` instead:
```bash
cat results/rl_done.json
# {"model": "agent-inc-rl-v4", "before": 0.327, "after": 0.647, "delta": 0.321}
```
That single line is the result. Explain what each field means.
