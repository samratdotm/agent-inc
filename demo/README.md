# Agent Inc. — demo site

A single, self-contained HTML demo of the whole project: the pitch, the architecture, a
gamified replay of an engagement, the RL training story, the leaderboard, the 30-scenario
universe, score distributions, and the result-integrity story. No server, no CDN, no network —
opens with a double-click and runs offline (projector-safe).

## Open it
```
open demo/index.html        # macOS — double-click also works (offline, projector-safe)
```

## Run a business scenario, scored live (the interactive panel)
The "Run it" section lets you insert a scenario and score it. It has two modes:

- **OFFLINE** (when opened as a plain file): scores a deliverable you provide with the
  **real deterministic grader** (the key-free 0.70) — reimplemented in JS, instant, no network.
  Tweak the price / claims / deliverable and the score reacts live. Verified to match `env.py`
  to the decimal (honest 0.700 · SOC2 lie 0.600 · no-delivery gate 0.150 · over-budget 0.500).
- **LIVE** (when served by the local server): a **real agent runs the engagement end-to-end**
  via the HUD gateway and is graded by the full grader — *including* the 0.30 LLM-judge quality
  slice — returning the reward, per-criterion bars, areas-to-improve, and wall-clock seconds.

Start the live server:
```
uv run python scripts/serve_demo.py     # then open http://127.0.0.1:7878/
```
Pick a scenario (or paste your own JSON following the schema), choose the model
(default: Qwen3.5-4B + RL), and hit **Run & score**. Each live run costs HUD gateway credits
and takes ~30–60s. `PORT=xxxx` overrides the port.

## Deploy the public site (Vercel)
The site is fully static, so it deploys to Vercel as-is. Deploy the **`demo/` folder** as the
project root (it contains `index.html`, `data.js`, `assets/`, `vercel.json`):

```
cd demo
npx vercel            # first run: log in + link the project (creates a preview URL)
npx vercel --prod     # promote to the production URL
```

On the deployed URL there is **no backend**, so the "Run a scenario" panel auto-detects this
(its `/api/health` probe fails) and runs in **offline mode** — the real in-browser deterministic
scorer. Everything else (charts, training curve, leaderboard, replay, receipts, Play mode) is
static and works fully. The **live agent run** stays local-only (it needs `HUD_API_KEY`, the
Python env, and ~50s per run, which Vercel's serverless functions aren't suited for).

## Refresh the data (after any new eval / RL run)
Every number is **baked from the canonical files** — nothing is hand-typed (RESULTS_INTEGRITY.md).
Regenerate `demo/data.js` from `results/*.json` + `data/scenarios/*.json`:
```
uv run python scripts/build_demo.py
```

## Screenshots ("Receipts")

**Captured live (already generated)** — real, unaltered command output rendered as terminal PNGs.
Regenerate anytime:
```
uv run python scripts/make_receipts.py
```
| filename | what it proves |
|---|---|
| `terminal-tests.png`    | the 96 offline tests passing (env + grader work, key-free) |
| `rl-status.png`         | RL status: baseline 0.327 → final 0.647, the 0.097 contamination caught |
| `grader-breakdown.png`  | the real grader: honest 0.70 · SOC2 lie 0.60 · no-delivery gate 0.15 |

**HUD platform jobs (you must be logged into hud.ai)** — the strongest third-party proof.
Drop PNGs into `demo/assets/` with these exact names; missing ones show a labelled placeholder.
Click any receipt in the page to enlarge it (lightbox).

| filename | what to capture | status |
|---|---|---|
| `hud-leaderboard.png`   | Claude Sonnet eval job (95% avg reward) — `3b76269a5ea8459c9d277287b89ba4e1` | ✅ added |
| `hud-gemini.png`        | Gemini 3.1 Pro eval job (91% avg reward) — `c3bc520e83804db686ddd639e24b469c` | ✅ added |
| `hud-qwen-baseline.png` | Qwen **base** eval job (33% avg reward) — `59b7587d0ca44371a9cbcc3aa9355fde` | ✅ added |
| `hud-qwen-rl.png`       | Qwen **+ RL** after-eval (65% avg reward, 30/30 graded) — `8ef60ebd990b4a0e956cc52079ebb043` | ✅ added |
| `hud-checkpoints.png`   | forked model `agent-inc-rl-v4` + its 16 GRPO checkpoints | ✅ added |

## Presenting (stage)
- Click **▶ PLAY** (top-right) or **Play the story** to enter guided mode: it auto-advances
  through every beat and triggers each section's animation. `→` / `space` advance, `←` back,
  `Esc` exits. Click a progress dot to jump.
- Click **Run engagement** in the Live-loop section to play the gamified replay on demand.

## Files
- `index.html` — the site (inlined CSS/JS, hand-rolled SVG charts).
- `data.js` — generated data blob (`window.AGENT_INC`); do not edit by hand.
- `assets/` — your screenshots.
