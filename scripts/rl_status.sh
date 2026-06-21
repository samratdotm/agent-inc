#!/bin/bash
# One-glance status of the RL run. Run anytime:  bash scripts/rl_status.sh
# Live follow (no refresh):  tail -f /tmp/rl_train_v4.log | grep -iE 'before|step|QUARANTINE|RESULT|supervisor'
cd /Users/samratmalisetti/Dev/agent-inc || exit 1
LOG=/tmp/rl_train_v4.log
echo "================ Agent Inc. RL status ================"
echo -n "supervisor: "; pgrep -f "bash scripts/rl_supervise.sh" >/dev/null && echo "ALIVE" || echo "DOWN"
echo -n "trainer:    "; pgrep -f "scripts/rl_train.py" >/dev/null && echo "ACTIVE (measuring/training)" || echo "between attempts"
echo -n "clean baseline cached: "; cat results/rl_state.json 2>/dev/null || echo "NOT YET (guards rejecting junk)"
echo
echo "-- training steps completed (curve) --"
[ -f results/training_curve.jsonl ] && tail -6 results/training_curve.jsonl || echo "  none yet"
echo
echo "-- last 3 baseline attempts --"
grep -iE '\[before\]|QUARANTINE.*baseline' "$LOG" 2>/dev/null | grep -vi 'Authlib' | tail -3
echo
echo "-- recent errors (last 80 log lines) --"
echo "  503/500/504/retry count: $(tail -80 "$LOG" 2>/dev/null | grep -ciE '503|500|504|Retrying')"
echo
echo -n "-- FINAL RESULT: "; [ -f results/rl_done.json ] && cat results/rl_done.json || echo "pending"
echo "====================================================="
