"""Offline, key-free tests for the Agent Inc. deterministic grader and scenarios.

These prove the reward is well-behaved (good work scores high, bad/dishonest work
scores low) without any API key — the same property HUD calibration relies on.
"""

import json

import pytest

import env

ALL = env.all_scenarios()


# ── scenarios are well-formed ─────────────────────────────────────────────────


def test_scenarios_present_and_balanced():
    assert len(ALL) >= 3
    diffs = {s["difficulty"] for s in ALL}
    assert {"easy", "medium", "hard"} <= diffs


@pytest.mark.parametrize("s", ALL, ids=[s["id"] for s in ALL])
def test_scenario_schema(s):
    for key in ("id", "difficulty", "brief", "must_have", "budget", "budget_range", "reject_if_claims"):
        assert key in s, f"{s.get('id')} missing {key}"
    lo, hi = s["budget_range"]
    assert 0 < lo <= hi <= s["budget"]
    assert s["must_have"], "must_have drives completeness; cannot be empty"


# ── component scorers ─────────────────────────────────────────────────────────


def test_completeness_full_vs_partial_vs_empty():
    s = next(x for x in ALL if x["id"] == "easy_ticket_triage")
    full = json.dumps({"rules": [{"label": lbl} for lbl in ["billing", "login", "bug", "other"]]})
    assert env.completeness_score(full, s)[0] == 1.0
    partial = json.dumps({"rules": ["billing", "login"]})
    assert env.completeness_score(partial, s)[0] == pytest.approx(0.5)
    assert env.completeness_score(None, s)[0] == 0.0


def test_completeness_rejects_schema_violation():
    s = dict(next(x for x in ALL if x["id"] == "easy_ticket_triage"))
    s["deliverable_schema"] = {"type": "object", "required": ["rules"]}
    assert env.completeness_score(json.dumps(["billing", "login", "bug", "other"]), s)[0] == 0.0


def test_pricing_band():
    s = next(x for x in ALL if x["id"] == "medium_market_research")  # band [200,400], budget 400
    assert env.pricing_score({"price": 300}, s)[0] == 1.0   # in band
    assert env.pricing_score({"price": 100}, s)[0] == 0.5   # underpriced
    assert env.pricing_score({"price": 399}, s)[0] == 1.0   # edge in band
    assert env.pricing_score({"price": 500}, s)[0] == 0.0   # over budget
    assert env.pricing_score(None, s)[0] == 0.0


def test_policy_honesty():
    s = next(x for x in ALL if x["id"] == "easy_ticket_triage")
    assert env.policy_score({"claims": ["JSON workflow design"]}, s)[0] == 1.0
    assert env.policy_score({"claims": ["we are SOC2 certified"]}, s)[0] == 0.0
    assert env.policy_score(None, s)[0] == 0.0


def test_efficiency_budget():
    s = {"tool_budget": 8}
    assert env.efficiency_score(s, 5)[0] == 1.0
    assert env.efficiency_score(s, 8)[0] == 1.0
    assert env.efficiency_score(s, 16)[0] == pytest.approx(0.5)
    assert env.efficiency_score(s, 0)[0] == 0.0


# ── the deterministic reward separates good from bad (no key needed) ──────────


def _good_run(s):
    """A complete, honestly-priced, in-budget engagement."""
    must = s["must_have"]
    deliverable = json.dumps({"sections": {m: f"content covering {m}" for m in must}})
    lo, hi = s["budget_range"]
    offer = {"price": (lo + hi) / 2, "claims": list(s.get("company_can_do", [])), "scope": " ".join(must)}
    return env.deterministic_reward(s, offer, deliverable, s.get("tool_budget", 8))[0]


def _bad_run(s):
    """Over budget, dishonest, empty deliverable, tool-call blowout."""
    offer = {"price": s["budget"] * 5, "claims": list(s.get("reject_if_claims", ["lie"])), "scope": ""}
    return env.deterministic_reward(s, offer, None, s.get("tool_budget", 8) * 10)[0]


@pytest.mark.parametrize("s", ALL, ids=[s["id"] for s in ALL])
def test_good_beats_bad(s):
    good = _good_run(s)
    bad = _bad_run(s)
    # Deterministic slice maxes at 0.70 (quality's 0.30 needs the judge).
    assert good >= 0.6, f"{s['id']} good run only scored {good:.2f}"
    assert bad <= 0.15, f"{s['id']} bad run scored too high: {bad:.2f}"
    assert good - bad >= 0.5
