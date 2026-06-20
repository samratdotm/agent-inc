"""Data layer for the Agent Inc. dashboard (P2).

Pure functions — NO streamlit import — so they can be unit-tested and reused.

- The **leaderboard** and **RL runway** are read from the real Phase-1 calibration
  at ``results/calibration.json`` (written by P1's eval runs).
- The **per-scenario / per-criterion** breakdown reads a real per-run export when
  P1 wires it (CONTRACT.md: "don't hardcode the path yet"); until then it falls
  back to ``data/sample_runs.json`` so the dashboard renders end-to-end for the demo.

Swap one constant (``RUNS_FILE``) when the real per-run export lands.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).parent
CALIBRATION = REPO / "results" / "calibration.json"
SCENARIOS_DIR = REPO / "data" / "scenarios"
SAMPLE_RUNS = REPO / "data" / "sample_runs.json"

# Point this at P1's real per-run export when it exists; falls back to sample.
RUNS_FILE = SAMPLE_RUNS

# Criterion weights mirror env.py (kept local so the view has no env dependency).
WEIGHTS: dict[str, float] = {
    "completeness": 0.25,
    "quality": 0.30,
    "pricing": 0.20,
    "efficiency": 0.15,
    "policy": 0.10,
}

# The RL goal called out in calibration.json's "rl_runway" ("~0.5+").
RL_TARGET = 0.5


def load_calibration() -> dict[str, Any]:
    """The raw Phase-1 calibration blob (real data), or an empty shell if absent."""
    if CALIBRATION.exists():
        return json.loads(CALIBRATION.read_text(encoding="utf-8"))
    return {"leaderboard": [], "note": "results/calibration.json not found"}


def leaderboard_rows() -> list[dict[str, Any]]:
    """Normalized leaderboard rows, best-first. Real data from calibration.json."""
    cal = load_calibration()
    rows = [
        {
            "model": e.get("model", "?"),
            "agent": e.get("agent", ""),
            "mean_reward": float(e.get("mean_reward", 0.0)),
            "std": float(e.get("std", 0.0)),
            "success_rate": float(e.get("success_rate", 0.0)),
            "role": e.get("role", ""),
            "in_target_band": bool(e.get("in_target_band", False)),
        }
        for e in cal.get("leaderboard", [])
    ]
    rows.sort(key=lambda r: r["mean_reward"], reverse=True)
    return rows


def training_curve() -> dict[str, Any]:
    """Points for the RL before/after story.

    The 'after RL' point appears once a trained-model reward is exported (we look
    for ``post_rl_mean_reward`` on the Qwen leaderboard entry). Until then we show
    the base, the target line, and the frontier ceiling.
    """
    cal = load_calibration()
    by_agent = {e.get("agent"): e for e in cal.get("leaderboard", [])}
    qwen = by_agent.get("openai_compatible", {})
    claude = by_agent.get("claude", {})

    points: list[dict[str, Any]] = []
    if qwen:
        points.append({"label": "Qwen base", "reward": float(qwen.get("mean_reward", 0.0))})
    after = qwen.get("post_rl_mean_reward")
    if after is not None:
        points.append({"label": "Qwen + RL", "reward": float(after)})

    return {
        "points": points,
        "target": RL_TARGET,
        "frontier_ceiling": float(claude["mean_reward"]) if claude else None,
        "runway": cal.get("rl_runway", ""),
        "has_after": after is not None,
    }


def _scenario_meta() -> dict[str, dict[str, str]]:
    """id -> {difficulty, domain} for every scenario on disk (real)."""
    meta: dict[str, dict[str, str]] = {}
    for p in sorted(SCENARIOS_DIR.glob("*.json")):
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        meta[s.get("id", p.stem)] = {
            "difficulty": s.get("difficulty", "?"),
            "domain": s.get("domain", "?"),
        }
    return meta


def scenario_counts() -> dict[str, int]:
    """How many scenarios exist, by difficulty (real)."""
    counts: dict[str, int] = {}
    for m in _scenario_meta().values():
        counts[m["difficulty"]] = counts.get(m["difficulty"], 0) + 1
    counts["total"] = sum(v for k, v in counts.items() if k != "total")
    return counts


def load_runs() -> list[dict[str, Any]]:
    """Per-scenario, per-criterion runs (real export if present, else sample).

    Each row: {model, scenario_id, difficulty, domain, reward, subscores, source}
    where subscores maps criterion -> {value, weight}. Reward is derived from the
    subscores via WEIGHTS when not given explicitly.
    """
    if not RUNS_FILE.exists():
        return []
    raw = json.loads(RUNS_FILE.read_text(encoding="utf-8"))
    meta = _scenario_meta()
    source = raw.get("source", RUNS_FILE.name)
    rows: list[dict[str, Any]] = []
    for r in raw.get("runs", []):
        sid = r.get("scenario_id", "")
        subs = r.get("subscores", {})
        reward = r.get("reward")
        if reward is None:
            reward = sum(WEIGHTS.get(k, 0.0) * float(v.get("value", 0.0)) for k, v in subs.items())
        m = meta.get(sid, {})
        rows.append(
            {
                "model": r.get("model", "?"),
                "scenario_id": sid,
                "difficulty": m.get("difficulty", r.get("difficulty", "?")),
                "domain": m.get("domain", "?"),
                "reward": round(float(reward), 3),
                "subscores": subs,
                "source": source,
            }
        )
    return rows


def data_status() -> dict[str, Any]:
    """What's real vs sample right now — surfaced in the dashboard footer."""
    return {
        "leaderboard_real": CALIBRATION.exists(),
        "runs_are_sample": RUNS_FILE == SAMPLE_RUNS,
        "runs_file": RUNS_FILE.name,
    }
