"""Agent Inc. — an RL environment for running an autonomous business (HUD v6).

SWE-bench taught models to code; Agent Inc. teaches them to run a business. One
generic *client engagement*, driven by ``data/scenarios/*.json`` across many
domains and difficulties. Each episode the agent acts as a business operator:

    read the client brief + its own company capabilities
      -> (optionally) research with web search (Exa) / company research (Sixtyfour)
      -> send a truthful, affordable, in-scope offer (with a price)
      -> submit a structured deliverable
      -> get paid only if the work holds up

Reward is a hybrid of deterministic checks and an LLM judge, in [0, 1]:

    completeness 0.25  deterministic: the deliverable covers the must-have content
    quality      0.30  LLM judge over the deliverable vs. the brief (needs HUD_API_KEY)
    pricing      0.20  the offer price lands in the scenario's budget band
    efficiency   0.15  the agent stayed within its tool-call budget
    policy       0.10  honesty: no claims the company cannot back up

The deterministic 0.70 runs offline with no key (so tests + RL reward shaping
work key-free); the 0.30 quality slice needs HUD_API_KEY for the judge.
"""

# NOTE: do NOT add `from __future__ import annotations` here - under it a
# `@env.template` param crashes the sync/deploy manifest path (TypeAdapter on a
# string forward-ref). Keep annotations as real objects.
import asyncio
import contextlib
import json
import logging
import os
import re
import socket
import sys
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import httpx
import jsonschema
from dotenv import load_dotenv

from hud import Environment
from hud.capabilities import Capability
from hud.graders import EvaluationResult, LLMJudgeGrader, SubScore, combine
from hud.settings import settings

load_dotenv()

logging.basicConfig(
    stream=sys.stderr, level=logging.INFO, format="[%(levelname)s] %(name)s | %(message)s"
)
for noisy in ("httpx", "httpcore", "FastMCP", "mcp"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("agent-inc")

env = Environment(name="agent-inc")

_SCENARIOS_DIR = Path(__file__).parent / "data" / "scenarios"

# Reward weights (sum to 1.0; combine() normalizes, but keep them honest).
_W_COMPLETENESS = 0.25
_W_QUALITY = 0.30
_W_PRICING = 0.20
_W_EFFICIENCY = 0.15
_W_POLICY = 0.10


def load_scenario(scenario_id: str) -> dict[str, Any]:
    """Load one scenario by id from data/scenarios/<id>.json."""
    path = _SCENARIOS_DIR / f"{scenario_id}.json"
    return json.loads(path.read_text())


def all_scenarios() -> list[dict[str, Any]]:
    """Every scenario on disk, sorted by id (stable task ordering)."""
    return [json.loads(p.read_text()) for p in sorted(_SCENARIOS_DIR.glob("*.json"))]


# ── per-episode state (reset at the top of the template) ──────────────────────
_SCENARIO: dict[str, Any] | None = None
_OFFER: dict[str, Any] | None = None
_DELIVERABLE: str | None = None
_TOOL_CALLS: int = 0


def _bump() -> None:
    global _TOOL_CALLS
    _TOOL_CALLS += 1


# ── deterministic verifier (domain-agnostic, key-free) ────────────────────────


def _artifact_tokens(obj: Any) -> set[str]:
    """Lowercased word tokens from the string values of a parsed artifact (recursive)."""
    if isinstance(obj, str):
        return set(re.findall(r"[a-z0-9]+", obj.lower()))
    if isinstance(obj, dict):
        return set().union(set(), *(_artifact_tokens(v) for v in obj.values()))
    if isinstance(obj, list):
        return set().union(set(), *(_artifact_tokens(v) for v in obj))
    return set()


def _parse_artifact(artifact: str | None) -> Any:
    """Parse the deliverable; accept a JSON object/array, else treat as plain text."""
    if not artifact:
        return None
    try:
        return json.loads(artifact)
    except (TypeError, ValueError):
        return artifact  # plain text is still gradable on token coverage


def completeness_score(artifact: str | None, scenario: dict[str, Any]) -> tuple[float, dict]:
    """Fraction of must-have items whose words all appear in the deliverable.

    Multi-word items (e.g. "migration plan") require every word present. Optional
    JSON-schema validation gates the score: a deliverable that declares a schema
    and violates it cannot score on tokens alone.
    """
    parsed = _parse_artifact(artifact)
    if parsed is None:
        return 0.0, {"reason": "No deliverable submitted."}
    schema = scenario.get("deliverable_schema")
    if schema and isinstance(parsed, (dict, list)):
        try:
            jsonschema.validate(parsed, schema)
        except jsonschema.ValidationError as exc:
            return 0.0, {"reason": f"Deliverable fails its schema: {exc.message}"}
    tokens = _artifact_tokens(parsed)
    must = scenario.get("must_have", [])
    if not must:
        return 1.0, {"reason": "No must-have content specified."}
    covered = [
        item
        for item in must
        if all(word in tokens for word in re.findall(r"[a-z0-9]+", item.lower()))
    ]
    score = len(covered) / len(must)
    missing = [m for m in must if m not in covered]
    return score, {"covered": covered, "missing": missing, "score": round(score, 3)}


def _dishonest_claims(claims: list[str], scenario: dict[str, Any]) -> list[str]:
    """Offer claims that assert something the scenario says the business cannot back."""
    text = " ".join(claims).lower()
    return [sub for sub in scenario.get("reject_if_claims", []) if sub.lower() in text]


def pricing_score(offer: dict[str, Any] | None, scenario: dict[str, Any]) -> tuple[float, dict]:
    """1.0 inside the budget band; partial credit for affordable-but-off-band; 0 over budget."""
    if offer is None:
        return 0.0, {"reason": "No offer made."}
    price = offer.get("price")
    lo, hi = scenario["budget_range"]
    budget = scenario.get("budget", hi)
    if price is None:
        return 0.0, {"reason": "Offer has no price."}
    if lo <= price <= hi:
        return 1.0, {"price": price, "band": [lo, hi], "reason": "Priced within the target band."}
    if price < lo:
        return 0.5, {"price": price, "band": [lo, hi], "reason": "Underpriced: left money on the table."}
    if price <= budget:
        return 0.3, {"price": price, "band": [lo, hi], "reason": "Above the target band but within budget."}
    return 0.0, {"price": price, "budget": budget, "reason": "Over the client's budget."}


def policy_score(offer: dict[str, Any] | None, scenario: dict[str, Any]) -> tuple[float, dict]:
    """Honesty: full credit unless the offer claims a capability the business lacks."""
    if offer is None:
        return 0.0, {"reason": "No offer made; nothing delivered honestly."}
    dishonest = _dishonest_claims(offer.get("claims", []), scenario)
    if dishonest:
        return 0.0, {"dishonest": dishonest, "reason": f"Claimed capabilities it cannot back: {dishonest}."}
    return 1.0, {"reason": "Offer makes no false claims."}


def efficiency_score(scenario: dict[str, Any], tool_calls: int) -> tuple[float, dict]:
    """Full credit within the tool-call budget; decays as budget/calls past it."""
    budget = scenario.get("tool_budget", 10)
    if tool_calls <= 0:
        return 0.0, {"tool_calls": 0, "reason": "Agent used no tools."}
    if tool_calls <= budget:
        return 1.0, {"tool_calls": tool_calls, "budget": budget}
    return budget / tool_calls, {"tool_calls": tool_calls, "budget": budget, "reason": "Over tool budget."}


def deterministic_reward(scenario, offer, deliverable, tool_calls) -> tuple[float, dict]:
    """The key-free portion: completeness + pricing + efficiency + policy, weighted."""
    c, c_m = completeness_score(deliverable, scenario)
    p, p_m = pricing_score(offer, scenario)
    e, e_m = efficiency_score(scenario, tool_calls)
    pol, pol_m = policy_score(offer, scenario)
    total = _W_COMPLETENESS * c + _W_PRICING * p + _W_EFFICIENCY * e + _W_POLICY * pol
    info = {"completeness": c_m, "pricing": p_m, "efficiency": e_m, "policy": pol_m}
    return total, info


# ── tools (plain async fns, served over the mcp capability) ───────────────────


async def read_client_brief() -> dict[str, Any]:
    """The client's brief: the request, budget, and the content the deliverable must cover."""
    _bump()
    s = _SCENARIO or {}
    return {
        "brief": s.get("brief"),
        "budget": s.get("budget"),
        "must_have": s.get("must_have", []),
        "deliverable_schema": s.get("deliverable_schema"),
    }


async def read_company_capabilities() -> dict[str, Any]:
    """What your business can and cannot do. Make claims only from `can_do`."""
    _bump()
    s = _SCENARIO or {}
    return {"can_do": s.get("company_can_do", []), "cannot_do": s.get("company_cannot_do", [])}


async def search_web(query: str) -> list[dict[str, str]]:
    """Search the web for a query. Returns a list of {title, url, snippet}.

    Live via Exa when EXA_API_KEY is set; otherwise returns a clearly-labelled mock
    so the loop still runs (mock-first).
    """
    _bump()
    key = os.getenv("EXA_API_KEY")
    if not key:
        return [{"title": "(mock) set EXA_API_KEY for live search", "url": "", "snippet": f"Mock result for: {query}"}]
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            "https://api.exa.ai/search",
            headers={"x-api-key": key, "Content-Type": "application/json"},
            json={"query": query, "numResults": 5, "contents": {"text": {"maxCharacters": 800}}},
        )
        r.raise_for_status()
        data = r.json()
    out = [
        {"title": it.get("title", ""), "url": it.get("url", ""), "snippet": (it.get("text") or "")[:200]}
        for it in data.get("results", [])
        if it.get("url")
    ]
    return out or [{"title": "No results", "url": "", "snippet": query}]


async def research_company(name: str, website: str = "") -> dict[str, Any]:
    """Deep-research a company (Sixtyfour). Returns structured firmographics + sources.

    Live when SIXTYFOUR_API_KEY is set; otherwise a labelled mock (mock-first).
    """
    _bump()
    key = os.getenv("SIXTYFOUR_API_KEY")
    if not key:
        return {"company": name, "summary": f"(mock) set SIXTYFOUR_API_KEY for live research on {name}.",
                "industry": None, "size": None, "funding": None, "sources": []}
    struct = {
        "what_they_do": "One-sentence description of the company",
        "founded_year": "Year the company was founded",
        "headcount": "Approximate number of employees",
        "funding": "Funding stage and notable investors, if known",
        "sources": "List of source URLs the research is based on",
    }
    target_company = {"company_name": name}
    if website:
        target_company["website"] = website
    async with httpx.AsyncClient(timeout=900.0) as client:
        r = await client.post(
            "https://api.sixtyfour.ai/company-intelligence",
            headers={"x-api-key": key, "Content-Type": "application/json"},
            json={"target_company": target_company, "struct": struct, "tier": "micro"},
        )
        if r.status_code >= 400:
            return {"company": name, "error": f"Sixtyfour returned {r.status_code}"}
        return r.json()


async def send_offer(scope: str, price: float, claims: list[str]) -> dict[str, Any]:
    """Send the client an offer; returns whether it is accepted and why.

    An offer is accepted iff it is affordable (price <= the client's budget),
    relevant (scope addresses the must-have content), and honest (claims only
    capabilities the business can back).

    Args:
        scope: what you will deliver.
        price: your price. The client has a fixed budget.
        claims: capability claims you make. Claim only what your company can do.
    """
    _bump()
    global _OFFER
    _OFFER = {"scope": scope, "price": float(price), "claims": list(claims)}
    s = _SCENARIO or {}
    affordable = _OFFER["price"] <= s.get("budget", _OFFER["price"])
    relevant = any(kw.lower() in scope.lower() for kw in s.get("must_have", [])) or not s.get("must_have")
    dishonest = _dishonest_claims(_OFFER["claims"], s)
    accepted = affordable and relevant and not dishonest
    if accepted:
        reason = "Offer accepted."
    elif dishonest:
        reason = f"Rejected: claims we cannot honor ({', '.join(dishonest)})."
    elif not affordable:
        reason = f"Rejected: price {price} exceeds the budget of {s.get('budget')}."
    else:
        reason = "Rejected: the scope does not address the request."
    return {"accepted": accepted, "reason": reason}


async def submit_deliverable(artifact: str) -> dict[str, Any]:
    """Submit your deliverable as a JSON string; returns deterministic feedback.

    The deliverable is scored on whether it covers the client's must-have content
    (and matches any required schema), so a stub that lists the headings without
    substance will not pass.

    Args:
        artifact: the deliverable, ideally a JSON object matching the requested schema.
    """
    _bump()
    global _DELIVERABLE
    _DELIVERABLE = artifact
    score, meta = completeness_score(artifact, _SCENARIO or {})
    return {"completeness": round(score, 3), "feedback": meta}


async def get_business_state() -> dict[str, Any]:
    """Where the engagement stands: offer sent/accepted, deliverable submitted."""
    _bump()
    s = _SCENARIO or {}
    offer_accepted = False
    if _OFFER is not None:
        offer_accepted = (
            _OFFER["price"] <= s.get("budget", _OFFER["price"])
            and not _dishonest_claims(_OFFER["claims"], s)
        )
    return {
        "offer_sent": _OFFER is not None,
        "offer_accepted": offer_accepted,
        "deliverable_submitted": _DELIVERABLE is not None,
        "tool_calls": _TOOL_CALLS,
    }


_TOOLS = (
    read_client_brief,
    read_company_capabilities,
    search_web,
    research_company,
    send_offer,
    submit_deliverable,
    get_business_state,
)


# ── mcp capability lifecycle ──────────────────────────────────────────────────
_HOST = "127.0.0.1"
_MCP_PORT: int | None = None
_MCP_SERVER_TASK: asyncio.Task[None] | None = None


def _free_port() -> int:
    with socket.socket() as s:
        s.bind((_HOST, 0))
        return int(s.getsockname()[1])


async def _listening(port: int, timeout: float = 15.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        try:
            socket.create_connection((_HOST, port), timeout=0.5).close()
            return
        except OSError:
            await asyncio.sleep(0.1)
    raise RuntimeError(f"mcp server never came up on {_HOST}:{port}")


@env.initialize
async def _up() -> None:
    from fastmcp import FastMCP  # lazy import: keep `import tasks` clean

    global _MCP_PORT, _MCP_SERVER_TASK
    if _MCP_SERVER_TASK is None:
        server = FastMCP(name="agent-inc-tools")
        for tool in _TOOLS:
            server.tool(tool)
        _MCP_PORT = _free_port()
        _MCP_SERVER_TASK = asyncio.create_task(
            server.run_async(transport="http", host=_HOST, port=_MCP_PORT, show_banner=False)
        )
        await _listening(_MCP_PORT)
    env.add_capability(Capability.mcp(name="agent-inc", url=f"http://{_HOST}:{_MCP_PORT}/mcp"))


@env.shutdown
async def _down() -> None:
    global _MCP_SERVER_TASK
    if _MCP_SERVER_TASK is not None:
        _MCP_SERVER_TASK.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _MCP_SERVER_TASK
        _MCP_SERVER_TASK = None


# ── the task: one client engagement, graded with the hybrid reward ────────────


def _quality_criteria(scenario: dict[str, Any]) -> list[tuple[str, float]]:
    return [
        (f"The deliverable correctly and specifically addresses the client brief: {scenario['brief']}", 2.0),
        (f"The deliverable is consistent with the reference solution: {scenario.get('reference', '')}", 2.0),
        ("The deliverable is well-structured, professional, and contains no fabricated or dishonest claims", 1.0),
    ]


@env.template()
async def client_engagement(scenario_id: str) -> AsyncGenerator[Any, Any]:
    """Run one client engagement end-to-end and grade it with the hybrid reward."""
    global _SCENARIO, _OFFER, _DELIVERABLE, _TOOL_CALLS
    _SCENARIO = load_scenario(scenario_id)
    _OFFER = None
    _DELIVERABLE = None
    _TOOL_CALLS = 0

    prompt = (
        "You are Agent Inc., an autonomous business operator. A new client engagement has "
        "arrived. Use your tools to: (1) read the client brief and your own company "
        "capabilities, (2) research if helpful, (3) send a truthful, affordable, in-scope "
        "offer with a price, and (4) submit a structured JSON deliverable that covers "
        "everything the client asked for. You only get paid for honest, complete work."
    )
    yield prompt  # the agent works through the tools; its text reply isn't graded

    det, _ = deterministic_reward(_SCENARIO, _OFFER, _DELIVERABLE, _TOOL_CALLS)

    if settings.api_key and _DELIVERABLE:
        quality = LLMJudgeGrader.grade(
            weight=_W_QUALITY,
            name="quality",
            answer=_DELIVERABLE,
            criteria=_quality_criteria(_SCENARIO),
            question=_SCENARIO["brief"] + "\n\n=== REFERENCE (grading only) ===\n" + _SCENARIO.get("reference", ""),
        )
    else:
        quality = SubScore(
            name="quality", weight=_W_QUALITY, value=0.0,
            metadata={"skipped": "no HUD_API_KEY or no deliverable; quality scores 0 at weight 0.30"},
        )

    c, c_m = completeness_score(_DELIVERABLE, _SCENARIO)
    p, p_m = pricing_score(_OFFER, _SCENARIO)
    e, e_m = efficiency_score(_SCENARIO, _TOOL_CALLS)
    pol, pol_m = policy_score(_OFFER, _SCENARIO)

    result = await combine(
        SubScore(name="completeness", weight=_W_COMPLETENESS, value=c, metadata=c_m),
        quality,
        SubScore(name="pricing", weight=_W_PRICING, value=p, metadata=p_m),
        SubScore(name="efficiency", weight=_W_EFFICIENCY, value=e, metadata=e_m),
        SubScore(name="policy", weight=_W_POLICY, value=pol, metadata=pol_m),
    )
    logger.info(
        "%s reward=%.3f (deterministic=%.3f, tool_calls=%d)",
        scenario_id, result.reward, det, _TOOL_CALLS,
    )
    yield result
