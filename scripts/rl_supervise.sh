#!/bin/bash
# Auto-relaunch supervisor for the hardened RL trainer.
# Runs the resumable trainer; whenever it exits (crash / pool-give-up / done),
# checks the done-marker — stops if complete, else waits and relaunches (the
# trainer resumes from its saved checkpoints). Survives a degraded Tinker pool.
cd /Users/samratmalisetti/Dev/agent-inc || exit 1
LOG=/tmp/rl_train_v4.log
DONE=results/rl_done.json
RELAUNCH_WAIT=120
LOCK=/tmp/agent_inc_rl_supervisor.lock

# Single-instance lock: two supervisors → two trainers → corrupted (concurrent) gradient steps.
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "[supervisor] another supervisor already running (pid $(cat "$LOCK")); exiting" >> "$LOG"
  exit 0
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

echo "[supervisor] started $(date)" >> "$LOG"
attempt=0
while true; do
  if [ -f "$DONE" ]; then
    echo "[supervisor] DONE — training complete $(date)" >> "$LOG"
    break
  fi
  # Health-gate: don't burn a baseline/step against a dead pool — wait for a pulse first.
  HEALTH=$(EXA_API_KEY= SIXTYFOUR_API_KEY= uv run python scripts/pool_probe.py 2>/dev/null | tail -1)
  if [ "$HEALTH" != "OK" ]; then
    echo "[supervisor] pool DOWN ($HEALTH) — waiting 90s before retry $(date)" >> "$LOG"
    sleep 90
    continue
  fi
  pkill -f "scripts/rl_train.py" 2>/dev/null  # clear any stray trainer before launching one
  attempt=$((attempt+1))
  echo "[supervisor] pool OK — launch attempt #$attempt $(date)" >> "$LOG"
  EXA_API_KEY= SIXTYFOUR_API_KEY= uv run python -u scripts/rl_train.py >> "$LOG" 2>&1
  rc=$?
  if [ -f "$DONE" ]; then
    echo "[supervisor] DONE after attempt #$attempt $(date)" >> "$LOG"
    break
  fi
  echo "[supervisor] trainer exited rc=$rc (pool likely degraded); relaunch in ${RELAUNCH_WAIT}s $(date)" >> "$LOG"
  sleep "$RELAUNCH_WAIT"
done
