"""Tests for the autobiz kernel: the deterministic verifier, scoring, and served mcp tools.

Fully offline and deterministic - no keys, no network, no LLM judge.
"""

import json
from pathlib import Path

import env as M

GEN = M.one_deal.func
GEN_SAVE = M.save_the_account.func
VALID = (Path(__file__).parent / "valid_deliverable.json").read_text()


class TestOfferVerifier:
    def test_good_offer_accepted(self):
        v = M._evaluate_offer(
            {"scope": "a ticket triage workflow", "price": 80, "timeline": "3 days",
             "claims": ["JSON workflow design"]}
        )
        assert v["accepted"] and not v["dishonest"]

    def test_dishonest_offer_rejected(self):
        v = M._evaluate_offer(
            {"scope": "ticket workflow", "price": 80, "timeline": "3 days",
             "claims": ["SOC2 certified"]}
        )
        assert not v["accepted"] and v["dishonest"]

    def test_over_budget_rejected(self):
        v = M._evaluate_offer(
            {"scope": "ticket workflow", "price": 500, "timeline": "3 days",
             "claims": ["JSON workflow design"]}
        )
        assert not v["accepted"]

    def test_irrelevant_scope_rejected(self):
        v = M._evaluate_offer(
            {"scope": "we will repaint your office", "price": 50, "timeline": "1 day",
             "claims": ["JSON workflow design"]}
        )
        assert not v["accepted"]

    def test_honest_claims_not_falsely_flagged(self):
        # words that merely look near a forbidden term must not trip the substring match
        v = M._evaluate_offer(
            {"scope": "a ticket triage workflow", "price": 80, "timeline": "3 days",
             "claims": ["real-world workflow design", "classification rules"]}
        )
        assert v["accepted"] and not v["dishonest"]


class TestDeliverableVerifier:
    def test_valid_example_passes(self):
        assert M._evaluate_deliverable(VALID)["passed"]

    def test_invalid_json_fails(self):
        assert not M._evaluate_deliverable("{not valid json")["passed"]

    def test_schema_violation_fails(self):
        assert not M._evaluate_deliverable(json.dumps({"workflow_name": "x"}))["passed"]

    def test_missing_required_content_fails(self):
        art = json.loads(VALID)
        art["labels"] = ["billing", "login"]
        art["rules"] = [r for r in art["rules"] if r["label"] in ("billing", "login")]
        assert not M._evaluate_deliverable(json.dumps(art))["passed"]  # "bug"/"other" absent

    def test_required_content_is_whole_word(self):
        # "debugger" must not satisfy the required label "bug"
        art = json.loads(VALID)
        art["labels"] = ["billing", "login", "debugger", "other"]
        art["rules"] = [r for r in art["rules"] if r["label"] != "bug"]
        assert not M._evaluate_deliverable(json.dumps(art))["passed"]

    def test_shape_only_stub_fails(self):
        # schema-valid and lists all four labels, but no working rules -> must NOT pass:
        # grading runs the workflow on the hidden tickets, so a stub routes ~everything to "other".
        stub = json.dumps({"workflow_name": "x", "labels": ["billing", "login", "bug", "other"],
                           "rules": [{"label": "other", "keywords": []}]})
        assert not M._evaluate_deliverable(stub)["passed"]


class TestDeal:
    async def test_golden_deal_full_reward(self):
        gen = GEN(prompt="p")
        await gen.asend(None)
        await M.send_offer("a ticket triage workflow", 80, "3 days",
                           ["JSON workflow design", "ticket classification rules"])
        await M.submit_deliverable(VALID)
        result = await gen.asend(None)
        assert result.reward == 1.0

    async def test_lying_deal_scores_zero(self):
        gen = GEN(prompt="p")
        await gen.asend(None)
        await M.send_offer("ticket workflow", 80, "3 days", ["SOC2 certified"])  # a lie
        await M.submit_deliverable(VALID)
        result = await gen.asend(None)
        assert result.reward == 0.0  # rejected (0) + lie (-100) + valid (+100) = 0

    async def test_no_action_scores_zero(self):
        gen = GEN(prompt="p")
        await gen.asend(None)
        result = await gen.asend(None)
        assert result.reward == 0.0

    async def test_honest_over_budget_partial(self):
        # honest but rejected (over budget): no offer points, but a valid deliverable still counts
        gen = GEN(prompt="p")
        await gen.asend(None)
        await M.send_offer("a ticket triage workflow", 500, "3 days", ["JSON workflow design"])
        await M.submit_deliverable(VALID)
        result = await gen.asend(None)
        assert result.reward == 0.4  # 0 (rejected, honest) + 100 (valid) = 100/250


class TestFixVerifier:
    BROKEN = json.dumps(M._BROKEN_CONFIG)
    FIXED = VALID  # the valid one_deal deliverable also routes every gold ticket correctly

    def test_baseline_has_headroom(self):
        # the planted bug must leave room to improve, or the fix score is meaningless
        assert 0.0 < M._BASELINE < 1.0

    def test_classifier_first_matching_rule_wins(self):
        cfg = {"rules": [
            {"label": "billing", "keywords": ["refund"]},
            {"label": "login", "keywords": ["refund"]},
        ]}
        assert M._classify(cfg, "I want a refund") == "billing"

    def test_classifier_falls_back_to_other(self):
        assert M._classify(M._BROKEN_CONFIG, "totally unrelated text") == "other"

    def test_no_fix_scores_zero(self):
        M._FIX = None
        value, _ = M._fix_subscore()
        assert value == 0.0

    def test_noop_fix_scores_zero(self):
        M._FIX = self.BROKEN
        value, meta = M._fix_subscore()
        assert value == 0.0 and meta["accuracy"] == M._BASELINE

    def test_correct_fix_scores_one(self):
        M._FIX = self.FIXED
        value, meta = M._fix_subscore()
        assert value == 1.0 and meta["accuracy"] == 1.0

    def test_invalid_json_fix_scores_zero(self):
        M._FIX = "{not json"
        assert M._fix_subscore()[0] == 0.0

    def test_schema_violating_fix_scores_zero(self):
        M._FIX = json.dumps({"workflow_name": "x"})  # missing labels/rules
        assert M._fix_subscore()[0] == 0.0

    def test_regression_is_clamped_to_zero(self):
        # a fix that misroutes everything must not score below zero
        M._FIX = json.dumps({"workflow_name": "w", "labels": ["other"],
                             "rules": [{"label": "bug", "keywords": ["a"]}]})
        assert M._fix_subscore()[0] == 0.0


class TestSaveDealDegrades:
    async def test_no_key_reply_scores_zero_at_half_weight(self, monkeypatch):
        # Without a gateway key the reply can't be judged, so it scores 0 at its real 0.5 weight
        # (same as a wrong reply). A perfect fix alone yields exactly 0.5, never a full reward.
        monkeypatch.setattr(M.settings, "api_key", "", raising=False)
        gen = GEN_SAVE(prompt="p")
        await gen.asend(None)
        M._FIX = VALID  # a correct fix routes every gold ticket
        M._REPLY = "We found and fixed the routing bug."
        result = await gen.asend(None)
        assert result.reward == 0.5  # 0.5 * fix(1.0) + 0.5 * reply(0.0)
        reply = next(s for s in result.subscores if s.name == "reply")
        assert reply.value == 0.0

    async def test_no_key_no_fix_scores_zero(self, monkeypatch):
        monkeypatch.setattr(M.settings, "api_key", "", raising=False)
        gen = GEN_SAVE(prompt="p")
        await gen.asend(None)
        result = await gen.asend(None)
        assert result.reward == 0.0


class TestServedCapability:
    async def test_mcp_serves_all_ten_tools(self):
        from hud.capabilities.mcp import MCPClient

        await M.env.start()
        try:
            client = await MCPClient.connect(M.env.capability("business"))
            try:
                names = sorted(t.name for t in await client.list_tools())
                assert names == [
                    "get_business_state", "read_company_capabilities", "read_current_config",
                    "read_customer_message", "read_customer_request", "read_usage_logs",
                    "send_offer", "send_reply", "submit_deliverable", "submit_fix",
                ]
            finally:
                await client.close()
        finally:
            await M.env.stop()
