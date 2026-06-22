#!/usr/bin/env python3
"""Local server for the Agent Inc. demo's interactive 'Run a scenario' panel.

Serves demo/ as static files AND exposes POST /api/run, which runs ONE real
engagement end-to-end (agent via the HUD gateway + the full grader, including the
0.30 LLM-judge quality slice) and returns the reward, per-criterion subscores
with metadata, and wall-clock seconds.

    uv run python scripts/serve_demo.py            # then open the printed URL

Without this server the page still works: opened as a plain file it falls back to
the in-browser deterministic scorer (the key-free 0.70). This server adds the real
agent run + LLM judge. Each run costs HUD gateway credits and takes ~30s.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
DEMO = REPO / "demo"
SCN_DIR = REPO / "data" / "scenarios"
PORT = int(os.environ.get("PORT", "7878"))
DEFAULT_MODEL = "agent-inc-rl-v4"

# HUD_API_KEY lives in ~/.hud/.env (set via `hud set`); load it so the gateway works.
load_dotenv(Path.home() / ".hud" / ".env")
load_dotenv(REPO / ".env")
os.chdir(REPO)  # so LocalRuntime("env.py") and data/scenarios resolve


async def _run_async(scenario_id: str, model: str, max_steps: int = 18) -> dict:
    from hud.agents import create_agent
    from hud.eval import LocalRuntime, Taskset

    from env import client_engagement

    task = client_engagement(scenario_id=scenario_id)
    task.slug = scenario_id
    ts = Taskset("agent-inc-live", [task])
    agent = create_agent(model, max_steps=max_steps)
    job = await ts.run(agent, runtime=LocalRuntime("env.py"), group=1)

    run = job.runs[0]
    ev = run.evaluation if isinstance(run.evaluation, dict) else {}
    subs = []
    for s in ev.get("subscores") or []:
        v = float(s.get("value", 0.0))
        w = float(s.get("weight", 0.0))
        subs.append({
            "name": s.get("name", "?"), "value": v, "weight": w,
            "contribution": round(v * w, 3), "metadata": s.get("metadata") or {},
        })
    return {
        "scenario_id": scenario_id, "model": model,
        "reward": float(run.reward) if run.reward is not None else None,
        "subscores": subs, "job_id": getattr(job, "id", None),
    }


def _materialize(scenario: dict) -> tuple[str, Path]:
    """Write a custom scenario to a temp file so client_engagement can load it by id."""
    raw_id = scenario.get("id") or "custom"
    sid = "_live_" + re.sub(r"[^a-zA-Z0-9_]", "_", raw_id)[:40]
    sc = dict(scenario)
    sc["id"] = sid
    sc.setdefault("company_can_do", scenario.get("can_do", []))
    sc.setdefault("company_cannot_do", scenario.get("cannot_do", []))
    sc.setdefault("deliverable_schema", {"type": "object"})
    sc.setdefault("tool_budget", 12)
    sc.setdefault("difficulty", "custom")
    sc.setdefault("domain", "custom")
    sc.setdefault("budget_range", [0, sc.get("budget", 0)])
    sc.setdefault("reject_if_claims", [])
    sc.setdefault("must_have", [])
    sc.setdefault("reference", sc.get("brief", ""))
    path = SCN_DIR / f"{sid}.json"
    path.write_text(json.dumps(sc), encoding="utf-8")
    return sid, path


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(DEMO), **kw)

    def log_message(self, *a):  # quiet
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        # health check lets the page detect a live backend (else it stays offline)
        if self.path.rstrip("/") == "/api/health":
            return self._json({"ok": True, "default_model": DEFAULT_MODEL})
        return super().do_GET()

    def do_POST(self):
        if self.path.rstrip("/") != "/api/run":
            self.send_error(404)
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._json({"error": f"bad request: {e}"}, 400)

        model = body.get("model") or DEFAULT_MODEL
        created = None
        try:
            if body.get("scenario_id"):
                sid = body["scenario_id"]
                if not (SCN_DIR / f"{sid}.json").exists():
                    return self._json({"error": f"unknown scenario_id {sid!r}"}, 400)
            elif body.get("scenario"):
                sid, created = _materialize(body["scenario"])
            else:
                return self._json({"error": "provide scenario_id or scenario"}, 400)

            print(f"  ▶ running {sid} with {model} …", flush=True)
            t0 = time.time()
            res = asyncio.run(_run_async(sid, model))
            res["seconds"] = round(time.time() - t0, 2)
            res["scenario_id"] = body.get("scenario_id") or body.get("scenario", {}).get("id", sid)
            print(f"  ✓ {res['scenario_id']} → reward {res['reward']} in {res['seconds']}s", flush=True)
            self._json(res)
        except Exception as e:
            print(f"  ✗ run failed: {type(e).__name__}: {e}", flush=True)
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)
        finally:
            if created and created.exists():
                created.unlink()


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}/"
    print("=" * 60)
    print("  Agent Inc. demo server")
    print(f"  open:  {url}")
    print("  the 'Run a scenario' panel is now in LIVE mode (real agent run).")
    print("  Ctrl-C to stop.")
    print("=" * 60, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
