#!/bin/bash
# Per-GPU sequential runner — 原子取任务, 遇错继续, 不断循环
# 用法: bash scripts/gpu_runner.sh <GPU_ID> <TASK_FILE>
set -uo pipefail  # 不用 -e: 单任务失败不退出整个 worker

GPU=$1
TASK_FILE=$2
LOCK_FILE="${TASK_FILE}.lock"
RUNNER_LOG="${TASK_FILE%/*}/../logs/gpu${GPU}_runner.log" 2>/dev/null || RUNNER_LOG="logs/gpu${GPU}_runner.log"
mkdir -p "$(dirname "$RUNNER_LOG")"

echo "[$(date +%H:%M)] GPU$GPU started, task_file=$TASK_FILE" >> "$RUNNER_LOG"

consecutive_failures=0
while true; do
  # 原子弹出任务
  task=$(flock "$LOCK_FILE" sh -c "
    t=\$(head -1 '$TASK_FILE' 2>/dev/null)
    if [ -n \"\$t\" ]; then
      sed -i '1d' '$TASK_FILE'
    fi
    echo \"\$t\"
  " 2>/dev/null)

  if [ -z "$task" ]; then
    echo "[$(date +%H:%M)] GPU$GPU: queue empty, exiting" >> "$RUNNER_LOG"
    break
  fi

  cmd=$(echo "$task" | cut -d'|' -f1 | sed "s/GPU_ID/$GPU/g")
  name=$(echo "$task" | cut -d'|' -f2)

  echo "[$(date +%H:%M)] GPU$GPU: $name START" >> "$RUNNER_LOG"
  CUDA_VISIBLE_DEVICES=$GPU eval "$cmd" || true  # 失败不退出
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "[$(date +%H:%M)] GPU$GPU: $name DONE" >> "$RUNNER_LOG"
    consecutive_failures=0
  else
    echo "[$(date +%H:%M)] GPU$GPU: $name FAILED(rc=$rc)" >> "$RUNNER_LOG"
    consecutive_failures=$((consecutive_failures + 1))
    # 连续 3 次失败则退出
    if [ $consecutive_failures -ge 3 ]; then
      echo "[$(date +%H:%M)] GPU$GPU: 3 consecutive failures, exiting" >> "$RUNNER_LOG"
      break
    fi
  fi
done
