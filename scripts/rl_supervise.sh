#!/bin/bash
# Auto-relaunch supervisor for the hardened RL trainer.
# Runs the resumable trainer; whenever it exits (crash / pool-give-up / done),
# checks the done-marker — stops if complete, else waits and relaunches (the
# trainer resumes from its saved checkpoints). Survives a degraded Tinker pool.
cd /Users/samratmalisetti/Dev/agent-inc || exit 1
LOG=/tmp/rl_train_v4.log
DONE=results/rl_done.json
RELAUNCH_WAIT=120

echo "[supervisor] started $(date)" >> "$LOG"
attempt=0
while true; do
  if [ -f "$DONE" ]; then
    echo "[supervisor] DONE — training complete $(date)" >> "$LOG"
    break
  fi
  attempt=$((attempt+1))
  echo "[supervisor] launch attempt #$attempt $(date)" >> "$LOG"
  EXA_API_KEY= SIXTYFOUR_API_KEY= uv run python -u scripts/rl_train.py >> "$LOG" 2>&1
  rc=$?
  if [ -f "$DONE" ]; then
    echo "[supervisor] DONE after attempt #$attempt $(date)" >> "$LOG"
    break
  fi
  echo "[supervisor] trainer exited rc=$rc (pool likely degraded); relaunch in ${RELAUNCH_WAIT}s $(date)" >> "$LOG"
  sleep "$RELAUNCH_WAIT"
done
