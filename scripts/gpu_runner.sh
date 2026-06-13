#!/bin/bash
# Per-GPU sequential runner — 每 GPU 一个独立进程，flock 原子取任务
# 基于 exp10/scripts/gpu_runner.sh
# 用法: bash scripts/gpu_runner.sh <GPU_ID> <TASK_FILE>
set -euo pipefail

GPU=$1
TASK_FILE=$2
LOCK_FILE="${TASK_FILE}.lock"
LOG="exp17/logs/gpu${GPU}_runner.log"

mkdir -p "$(dirname "$LOG")"

echo "[$(date +%H:%M)] GPU$GPU runner started" >> "$LOG"

while true; do
  # 原子弹出任务
  task=$(flock "$LOCK_FILE" sh -c "
    t=\$(head -1 '$TASK_FILE' 2>/dev/null)
    if [ -n \"\$t\" ]; then
      sed -i '1d' '$TASK_FILE'
    fi
    echo \"\$t\"
  ")

  if [ -z "$task" ]; then
    echo "[$(date +%H:%M)] GPU$GPU: queue empty, exiting" >> "$LOG"
    break
  fi

  cmd=$(echo "$task" | cut -d'|' -f1 | sed "s/GPU_ID/$GPU/g")
  name=$(echo "$task" | cut -d'|' -f2)

  echo "[$(date +%H:%M)] GPU$GPU: $name START" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU eval "$cmd"
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "[$(date +%H:%M)] GPU$GPU: $name DONE" | tee -a "$LOG"
  else
    echo "[$(date +%H:%M)] GPU$GPU: $name FAILED(rc=$rc)" | tee -a "$LOG"
  fi
done
