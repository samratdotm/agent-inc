# Agent Inc. — The Journey, the Debugging, and the Lessons

> A retrospective of building an RL environment + training an open model on it, at the HUD × YC
> Frontier RL Hackathon. Written to **learn from** — it includes the mistakes, not just the wins.
> To use in a fresh session: *"Read LEARNINGS.md and teach me about <topic>."*

---

## 0. The one-line outcome
We built a HUD v6 RL environment where an agent runs a business, then RL-trained Qwen3.5-4B on it:
**base 0.327 → RL-trained 0.647 (+0.321), graded clean on all 30 scenarios (incl. 24 it never trained on).**
Real, generalized improvement. The hard part wasn't the ML — it was surviving a degraded shared infra.

---

## 1. Building the environment (and the first lesson)

**What we did:** started from HUD's `autonomous-businesses` template + pulled the `deepresearch` template
for Exa/Sixtyfour tools. Built a data-driven env: 30 scenarios (JSON files), 7 FastMCP tools, a hybrid
grader, GRPO training.

**Lesson 1 — Verify the API against ground truth; don't trust the model's memory for new products.**
Our initial plan *guessed* the HUD SDK (`@env.tool()`, `@env.scenario`) and was **wrong**. The real v6 API
uses FastMCP tools over an `mcp` capability and `@env.template()`. We only got it right by **reading HUD's
actual template code + docs**. For any newer library/product, read the installed source or current docs —
the LLM's training data is stale.

**Lesson 2 — A grader is a product; reward *shaping* changes everything.**
First calibration: the weak model sat at exactly **0.45** on most scenarios. Inspecting the *distribution*
(not the mean) revealed why: it made a nice priced offer then **never delivered**, and our reward paid 0.45
for that. We added a **delivery gate** ("paid for delivery, not promises") → floor dropped to ~0.15, the
mean fell into the 20-50% target band, and the learning signal sharpened. **Always look at the reward
distribution, not just the average.**

---

## 2. The RL concepts (the part worth truly understanding)

- **On-policy RL (GRPO):** the model learns *only from its own attempts*. Each step it plays a task
  several times (a "group"), and the **"reference" is the group's own average** — attempts above average
  get reinforced, below get discouraged. **No teacher shows the right answer**; the reward (grader) only
  *scores* the model's own tries. RL doesn't inject new knowledge — it **amplifies good behavior the model
  already produces by chance.** (Analogy: practicing free throws alone with just a scoreboard.)
- **Reward variance is the fuel.** If every attempt in a group scores the same, every advantage is zero →
  *nothing learns*. This is why the "20-50% with variance" target matters — it's not bureaucracy, it's the
  precondition for GRPO to have a gradient.
- **Why you can't feed Claude traces into GRPO:** GRPO is on-policy — its math reinforces the *model's own*
  trajectories. Claude's traces are off-policy; feeding them breaks importance-sampling and destabilizes.
  The algorithm *designed* for someone-else's-outputs is **SFT** (imitation). RFT (reinforcement fine-tuning)
  = the RL category; HUD's GRPO and "Fireworks RFT" are both RFT, just different backends.
- **Generalization vs overfitting (the verdict question).** Training reward climbed 0.39→0.69 — but that was
  on the **6 training scenarios**. The honest test of "did it learn?" is the **after-eval on all 30** (incl.
  24 unseen). It hit **0.647** → it *generalized* (learned the loop), not memorized 6 cases. **A rising
  training-subset reward is necessary but not sufficient; always evaluate on held-out data.**
- **The grader's judge:** 70% of our reward is deterministic Python; 30% is an LLM judge =
  **`claude-haiku-4-5`** via the HUD gateway. Pairing a deterministic majority with the LLM judge is the
  defense against "the judge is gameable."

---

## 3. The infra saga (where the real debugging happened)

The shared HUD gateway buckled under hackathon load. We hit **four distinct failure modes**, and conflating
them wastes time — each has a different root cause and fix.

| # | Symptom | Root cause | What it taught us |
|---|---|---|---|
| 1 | `503 "no healthy keys for provider 'tinker'" / upstream_overloaded` | The shared **Tinker GPU pool** (runs/trains Qwen) saturated | Throttle `max_concurrent=3`; it's *capacity*, not auth |
| 2 | Baseline came back **0.107 / 0.000** | Eval ran *while the pool was failing rollouts* → contaminated | **Never cache a metric from degraded infra** (see §4) |
| 3 | `503 "no healthy keys for provider 'anthropic'"` during **grading** | The **Anthropic provider** (the LLM judge) saturated — a *different* resource than Tinker | Don't lump failures together; guard the *verdict* |
| 4 | `400 "no trainable turns"` + a 76-min **hang** | Degraded sampling returned **rollouts without token-data**; then a connection-level hang slipped past the SDK's 600s timeout *and* our retry | Per-call timeouts don't catch deep hangs → need a **watchdog** |

**Lesson 3 — Distinguish failure modes by their actual error text and endpoint.** "503" from `tinker` (training)
vs `anthropic` (judge) are unrelated bottlenecks. The user correctly pushed back when I lumped them — that was right.

**Lesson 4 — Concurrency affects *speed*, not model quality.** Tempting to "crank it up when it's back," but
higher concurrency on a degraded shared pool just re-storms it (and corrupts steps via failed rollouts). The
lever for a *better model* is hyperparameters (LR, steps), not throughput.

---

## 4. Result integrity (the most important engineering principle here)

A degraded pool doesn't just slow you down — it **fabricates wrong numbers**. A baseline measured while
rollouts fail comes back artificially low (0.107), which would later fake a *huge* "improvement." We made
honesty a hard, coded guardrail (`RESULTS_INTEGRITY.md`):

- **Plausibility floor:** reject any baseline below a threshold set from prior *clean* runs → it can't cache junk.
- **Graded-fraction guard:** the after-eval (30% LLM judge) must have ≥90% of rollouts actually graded, else
  reject + retry — so an Anthropic outage can't silently bias the verdict.
- **No fallback judge model:** swapping to a different judge would make before/after *incomparable* (judges
  aren't calibrated to each other). Retry the *same* judge instead — availability without the accuracy cost.
- **Quarantine, don't delete:** rejected numbers go to `results/quarantine.jsonl` (transparency), never the
  canonical `results/calibration.json`.

**Lesson 5 — When the infra is flaky, the failure isn't "slow," it's "wrong data." Guard the numbers, not
just the uptime. A wrong number on stage is worse than a modest true one.**

---

## 5. Resilience engineering (what kept a brutal run alive)

- **Hardened + resumable trainer:** per-step retry-with-backoff; resume from saved checkpoints (HUD persists
  them) so a crash loses *zero* completed steps; a done-marker.
- **Supervisor with a single-instance lock:** auto-relaunches the trainer until done. The lock matters — we
  once spawned *two* supervisors → two trainers on the same model → corrupted concurrent gradient steps.
- **Health-gating:** probe the pool before launching, so we don't burn attempts on a dead endpoint.
- **The gap we found:** a *deep hang* (no error thrown) slips past per-call timeouts AND exception-retry —
  the process sat idle 76 min. **Fix = a watchdog/heartbeat** that kills+relaunches on no-progress. (Still TODO.)
- **Local files lead the UI:** our `training_curve.jsonl` updated the instant a step finished; HUD's dashboard
  lagged 1-13+ min under load. **For real-time truth, watch the local artifacts, not the dashboard.**

---

## 6. Mistakes I (the assistant) made — and the meta-lesson

Documented honestly because catching them was the point:
- **Guessed "1-2 min sync lag"** for HUD's UI — it was 13+ min. *Don't state a number you didn't measure.*
- **Did a shallow `tail -120` log check** and declared "0 errors" — the errors had scrolled off. The user
  was right to ask "are you sure?" *Scan the full log by step/line, not the tail.*
- **Guessed "no timeout"** for the hang before checking — the SDK *does* have a 600s timeout; the real cause
  was a deeper connection hang. *Read the code before asserting a root cause.*

**Meta-lesson — when monitoring flaky systems, ground every claim in the full evidence; "looks fine from the
last few lines" is how you miss the failure.** Honesty over a confident-sounding guess.

---

## 7. Practical playbook (reusable next time)

1. **Mock-first:** every external tool returns labelled mock data without keys → the loop runs offline/free.
2. **De-risk early:** a tiny end-to-end RL pass on day 1 proves the pipeline before you depend on it.
3. **Mock the tools *during training*:** live Exa/Sixtyfour add latency + burn credits in the inner loop for
   ~zero learning benefit. Use live tools for the *demo/eval*, mocks for training.
4. **Checkpoints are free insurance:** they persist server-side; eval is read-only; you can always resume.
5. **Lean on the deployed checkpoint + eval for the demo** (HUD's own advice) — judges can't reproduce a
   live training run, and eval rollouts are robust to training-backend outages.

---

## 8. The outcome — and the honest reflection

**We were not selected (top-10) at the hackathon.** After a full-night build that produced a clean,
real RSI result, that stung. Captured here honestly, because the reflection is the most useful part.

**Why (most likely):** the *substance* was strong — a working env, a measured **0.327 → 0.647** RL
improvement, and result-integrity engineering. But hackathon judging happens **in the room, in minutes**,
on a 4-criterion rubric (Completion 0.2 / Originality 0.3 / **Design 0.2** / Technology 0.3). We had already
self-identified **Design/UX as our weakest axis** — and that's exactly the dimension decided by demo polish
and the live presentation, not by the depth of the result. **A strong result can lose to a slicker 90 seconds.**

**The big lesson — presentation is a first-class deliverable, not an afterthought.** Judges experience the
*demo*, not the repo. Our deepest work (the infra resilience, the integrity guards, the honest RL loop) was
largely *invisible* in a 90-second room. Invest in the demo UX + the narrative **as early as the engineering.**

**What we'd do differently:**
- Build the demo/dashboard **in parallel from day 1** (not as "Phase 4"), and rehearse the 90s pitch 3×.
- **Lead with the live "agent runs a business" moment + the one-glance before/after** (0.33→0.65) — make the
  win legible in *5 seconds*.
- Keep the infra-saga **out of the pitch** — it's a great blog post, a poor pitch beat.

**What still stands (more durable than a ribbon):**
- A working, MIT-licensed, public RL environment that trains a 4B open model to *run a business*.
- A real, honest, *generalized* RSI result (0.327 → 0.647) — produced on *failing* infra **without fabricating
  a number** (we refused to ship a contaminated baseline even when it would have looked better).
- Deep, documented learnings (this file): RL mechanics, on-policy GRPO, RSI, distributed-systems debugging,
  and result-integrity engineering.
- A genuinely portfolio/blog-worthy story: *"a 4B model learning to run a business — the honest 0.33→0.65,
  and surviving the infra meltdown."*

**The principle worth keeping:** refusing to ship a number we couldn't trust didn't win the bracket — but it's
the right way to do science, and it's a strength, not a cost. Optimize the *presentation* next time; never the *truth*.

## Pointers
- Result + how to resume: `STATUS.md` · Honesty rules: `RESULTS_INTEGRITY.md` · Project facts: `CLAUDE.md`
- The numbers: `results/calibration.json`, `results/training_curve.jsonl`, `results/rl_done.json`
- The code: `env.py` (env+grader), `scripts/rl_train.py` (trainer), `scripts/rl_supervise.sh` (supervisor)
