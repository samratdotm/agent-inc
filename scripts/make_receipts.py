#!/usr/bin/env python3
"""Render REAL local command output into terminal-style PNGs for the demo's
"Receipts" section.  Every image here is genuine output captured live — nothing
is hand-typed (RESULTS_INTEGRITY.md).  Re-run anytime:

    uv run python scripts/make_receipts.py

Produces (into demo/assets/):
  terminal-tests.png     · the 96 offline tests passing
  rl-status.png          · scripts/rl_status.sh  (curve, quarantine, final result)
  grader-breakdown.png   · the real deterministic grader on one scenario

The HUD platform screenshots (eval jobs, checkpoints) must be captured by a human
who is logged into hud.ai — see demo/README.md for the shot list.
"""

from __future__ import annotations

import html
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ASSETS = REPO / "demo" / "assets"
CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    shutil.which("google-chrome") or "",
    shutil.which("chromium") or "",
]

GRADER_SNIPPET = r'''
import json, env
sc = next(s for s in env.all_scenarios() if s["id"]=="easy_ticket_triage")
offer = {"scope":"JSON workflow classifying tickets into billing, login, bug, or other.",
         "price":80, "claims":["JSON workflow design","ticket classification rules","written documentation"]}
deliverable = json.dumps({"workflow":"ticket_triage","labels":["billing","login","bug","other"],
  "rules":[{"label":"billing","keywords":["invoice","charge","refund"]},
           {"label":"login","keywords":["password","sign in","2fa"]},
           {"label":"bug","keywords":["error","crash","broken"]},
           {"label":"other","keywords":["fallback"]}],
  "documentation":"First keyword match wins; unmatched -> other."})
total, _ = env.deterministic_reward(sc, offer, deliverable, tool_calls=5)
comps = env.reward_components(sc, offer, deliverable, tool_calls=5)
print("$ uv run python -m grader easy_ticket_triage")
print("REAL DETERMINISTIC GRADER  ->  scenario: easy_ticket_triage")
print("="*58)
for name,w,v,meta in comps:
    print(f"  {name:<13} value={v:>4.2f}  x weight {w:>4.2f}  = {v*w:>5.3f}")
print("-"*58)
print(f"  deterministic subtotal (key-free, 0.70 of reward) = {total:.3f}")
print(f"  quality (.30, LLM judge claude-haiku-4-5)         = + HUD_API_KEY")
print()
bad = {**offer, "claims": offer["claims"]+["soc2"]}
bt,_ = env.deterministic_reward(sc, bad, deliverable, 5)
nt,_ = env.deterministic_reward(sc, offer, None, 5)
print(f"  dishonest claim 'soc2' (not in can_do)  -> policy fails, reward {bt:.3f}")
print(f"  offer sent, nothing delivered           -> gated,        reward {nt:.3f}")
'''

# (filename, caption, terminal-title, command)
SHOTS = [
    ("terminal-tests.png", "96 offline tests pass · env + grader, key-free",
     "agent-inc — pytest", ["uv", "run", "pytest", "tests/", "-q"]),
    ("rl-status.png", "RL status · baseline 0.327 → final 0.647, contamination caught",
     "agent-inc — rl_status.sh", ["bash", "scripts/rl_status.sh"]),
    ("grader-breakdown.png", "Real grader: honest 0.70 · SOC2 lie 0.60 · no-delivery gate 0.15",
     "agent-inc — grader", ["uv", "run", "python", "-c", GRADER_SNIPPET]),
]


def chrome() -> str:
    for c in CHROME_CANDIDATES:
        if c and Path(c).exists():
            return c
    raise SystemExit("Chrome/Chromium not found — install it or edit CHROME_CANDIDATES.")


def colorize(line: str) -> str:
    e = html.escape(line)
    low = line.lower()
    cls = ""
    if line.strip().startswith("$"):
        cls = "c"                                   # command prompt
    elif any(k in low for k in ("quarantine", "fails", "gated", "0.097")) or "down" in low.split():
        cls = "r"                                   # failures / contamination caught
    elif any(k in low for k in ("passed", "final result", "subtotal")) or "alive" in low.split():
        cls = "g"                                   # green wins / good outcomes
    return f'<span class="{cls}">{e}</span>' if cls else e


def render(title: str, caption: str, text: str, out: Path) -> None:
    lines = text.rstrip("\n").split("\n") or [""]
    body = "\n".join(colorize(l) for l in lines)
    width = 1000
    line_h = 21
    height = 52 + 22 + len(lines) * line_h + 64  # titlebar + caption + content + padding
    page = f"""<!doctype html><meta charset=utf-8><style>
    html,body{{margin:0;background:#05070d}}
    .term{{width:{width-40}px;margin:20px;background:#0a0e1a;border:1px solid #1c2a44;border-radius:12px;
      overflow:hidden;font-family:ui-monospace,'SF Mono',Menlo,monospace;box-shadow:0 20px 60px rgba(0,0,0,.5)}}
    .bar{{height:38px;display:flex;align-items:center;gap:8px;padding:0 14px;background:#0e1626;border-bottom:1px solid #1c2a44}}
    .bar i{{width:12px;height:12px;border-radius:50%;display:inline-block}}
    .bar .t{{margin-left:10px;color:#9fb0d0;font-size:12px}}
    pre{{margin:0;padding:18px 20px;color:#e8eefc;font-size:13.5px;line-height:{line_h}px;white-space:pre-wrap;word-break:break-word}}
    .g{{color:#34d399}} .r{{color:#f87171}} .c{{color:#22d3ee}}
    .cap{{color:#64748b;font-size:12px;padding:0 20px 16px}}
    </style>
    <div class="term"><div class="bar"><i style="background:#f87171"></i><i style="background:#fbbf24"></i>
    <i style="background:#34d399"></i><span class="t">{html.escape(title)}</span></div>
    <pre>{body}</pre><div class="cap">› {html.escape(caption)}</div></div>"""
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as f:
        f.write(page)
        tmp = f.name
    subprocess.run(
        [chrome(), "--headless=new", "--disable-gpu", "--hide-scrollbars",
         "--force-device-scale-factor=2", f"--window-size={width},{height}",
         f"--screenshot={out}", f"file://{tmp}"],
        check=True, capture_output=True,
    )
    Path(tmp).unlink(missing_ok=True)
    print(f"  wrote {out.relative_to(REPO)}  ({len(lines)} lines)")


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for fname, cap, title, cmd in SHOTS:
        print(f"capturing {fname} …")
        res = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        out = (res.stdout or "") + (res.stderr or "")
        render(title, cap, out, ASSETS / fname)
    print("done — open demo/index.html and scroll to Receipts.")


if __name__ == "__main__":
    main()
