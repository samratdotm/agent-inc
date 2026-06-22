#!/usr/bin/env python3
"""Bake the canonical Agent Inc. data into demo/data.js for the demo site.

Integrity-first (see RESULTS_INTEGRITY.md): every number in the demo comes from a
canonical file — nothing is hand-typed — and each field carries a provenance tag
(``real`` vs ``sample``) so the UI can label it honestly. Re-run after any eval/RL
update:  uv run python scripts/build_demo.py

It reads:
  results/calibration.json      -> leaderboard + rl_runway        (real)
  results/training_curve.jsonl  -> per-step RL curve              (real)
  results/rl_done.json          -> before / after / delta         (real)
  results/quarantine.jsonl      -> the *caught* contamination     (rejected)
  data/scenarios/*.json         -> all 30 scenarios               (real)
  data/sample_runs.json         -> per-criterion breakdown        (sample)

and writes a single  window.AGENT_INC = {...}  blob to demo/data.js.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
SCENARIOS_DIR = REPO / "data" / "scenarios"
SAMPLE_RUNS = REPO / "data" / "sample_runs.json"
OUT = REPO / "demo" / "data.js"

# Criterion weights — the single source of truth is env.py; mirrored here so the
# generator has no import-time dependency on the FastMCP server. Kept in sync with
# dashboard_data.WEIGHTS.
WEIGHTS = {
    "completeness": 0.25,
    "quality": 0.30,
    "pricing": 0.20,
    "efficiency": 0.15,
    "policy": 0.10,
}
CRITERION_KIND = {  # how each criterion is computed — for the reward-engine panel
    "completeness": "deterministic",
    "quality": "llm-judge",
    "pricing": "deterministic",
    "efficiency": "deterministic",
    "policy": "deterministic",
}


def _read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def load_scenarios() -> list[dict]:
    out = []
    for p in sorted(SCENARIOS_DIR.glob("*.json")):
        s = _read_json(p, None)
        if not s:
            continue
        out.append(
            {
                "id": s.get("id", p.stem),
                "difficulty": s.get("difficulty", "?"),
                "domain": s.get("domain", "?"),
                "brief": s.get("brief", ""),
                "budget": s.get("budget"),
                "budget_range": s.get("budget_range", []),
                "must_have": s.get("must_have", []),
                "can_do": s.get("company_can_do", []),
                "cannot_do": s.get("company_cannot_do", []),
                "reject_if_claims": s.get("reject_if_claims", []),
                "tool_budget": s.get("tool_budget"),
                "reference": s.get("reference", ""),
            }
        )
    return out


def build() -> dict:
    calibration = _read_json(RESULTS / "calibration.json", {"leaderboard": []})
    curve = _read_jsonl(RESULTS / "training_curve.jsonl")
    rl_done = _read_json(RESULTS / "rl_done.json", {})
    quarantine = _read_jsonl(RESULTS / "quarantine.jsonl")
    scenarios = load_scenarios()
    sample = _read_json(SAMPLE_RUNS, {"runs": [], "source": ""})

    # leaderboard, best-first
    lb = []
    for e in calibration.get("leaderboard", []):
        lb.append(
            {
                "model": e.get("model", "?"),
                "agent": e.get("agent", ""),
                "mean_reward": e.get("mean_reward"),
                "std": e.get("std"),
                "success_rate": e.get("success_rate"),
                "role": e.get("role", ""),
                "post_rl_mean_reward": e.get("post_rl_mean_reward"),
                "rl_baseline_mean_reward": e.get("rl_baseline_mean_reward"),
            }
        )
    lb.sort(key=lambda r: (r["mean_reward"] or 0), reverse=True)

    # difficulty / domain rollups
    by_diff: dict[str, int] = {}
    by_domain: dict[str, int] = {}
    for s in scenarios:
        by_diff[s["difficulty"]] = by_diff.get(s["difficulty"], 0) + 1
        by_domain[s["domain"]] = by_domain.get(s["domain"], 0) + 1

    # featured scenario for the gamified replay (real scenario data)
    featured = next((s for s in scenarios if s["id"] == "easy_ticket_triage"), None)

    return {
        "generatedFrom": "results/*.json + data/scenarios/*.json (canonical)",
        "weights": WEIGHTS,
        "criterionKind": CRITERION_KIND,
        "leaderboard": {"provenance": "real", "rows": lb,
                        "runway": calibration.get("rl_runway", "")},
        "rl": {
            "provenance": "real",
            "model": rl_done.get("model"),
            "before": rl_done.get("before"),
            "after": rl_done.get("after"),
            "delta": rl_done.get("delta"),
            "curve": [
                {"step": c.get("step"), "reward": c.get("reward"), "phase": c.get("phase")}
                for c in curve
            ],
            "targetBand": calibration.get("target_band", ""),
        },
        "scenarios": {"provenance": "real", "items": scenarios,
                      "byDifficulty": by_diff, "byDomain": by_domain,
                      "total": len(scenarios), "domains": len(by_domain)},
        "featured": featured,
        "perCriterion": {"provenance": "sample", "source": sample.get("source", ""),
                         "runs": sample.get("runs", [])},
        "quarantine": {"provenance": "rejected", "rows": quarantine},
    }


def main() -> None:
    data = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(data, indent=2, ensure_ascii=False)
    OUT.write_text(
        "// AUTO-GENERATED by scripts/build_demo.py — do not edit by hand.\n"
        "// Every number here is baked from a canonical file (RESULTS_INTEGRITY.md).\n"
        f"window.AGENT_INC = {blob};\n",
        encoding="utf-8",
    )
    rl = data["rl"]
    print(f"wrote {OUT.relative_to(REPO)}")
    print(f"  scenarios : {data['scenarios']['total']} across {data['scenarios']['domains']} domains")
    print(f"  curve pts : {len(rl['curve'])}")
    print(f"  RL        : {rl['before']} -> {rl['after']}  (delta {rl['delta']})")
    print(f"  leaderboard rows: {len(data['leaderboard']['rows'])}")


if __name__ == "__main__":
    main()
