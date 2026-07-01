#!/bin/bash
# Simple reliable launcher: splits tasks across GPUs, each GPU runs sequentially
# Usage: bash scripts/launcher.sh TASK_FILE [GPUS]
#   GPUS: comma-separated list, default 0,1,2,3,4,5,6,7
set -e

TASK_FILE="${1:?Usage: bash scripts/launcher.sh TASK_FILE [GPUS]}"
GPUS="${2:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPU_LIST <<< "$GPUS"

# Read all tasks
mapfile -t TASKS < "$TASK_FILE"
TOTAL=${#TASKS[@]}
echo "=== launcher: $TOTAL tasks on GPUs: $GPUS ==="

# Kill old processes
pkill -f "train.py" 2>/dev/null || true
sleep 1

# Split tasks round-robin across GPUs
for gpu in "${GPU_LIST[@]}"; do
  GPU_DIR="$(dirname "$TASK_FILE")/gpu${gpu}"
  mkdir -p "$GPU_DIR"
  > "$GPU_DIR/tasks.sh"
  chmod +x "$GPU_DIR/tasks.sh"
done

for i in "${!TASKS[@]}"; do
  gpu_idx=$((i % ${#GPU_LIST[@]}))
  gpu="${GPU_LIST[$gpu_idx]}"
  task="${TASKS[$i]}"
  name=$(echo "$task" | cut -d'|' -f2)
  cmd=$(echo "$task" | cut -d'|' -f1 | sed "s/GPU_ID/$gpu/g")

  GPU_DIR="$(dirname "$TASK_FILE")/gpu${gpu}"
  # Skip if checkpoint exists
  cat >> "$GPU_DIR/tasks.sh" << 'TASK'
echo "[$(date +%H:%M)] START: NAME_PLACEHOLDER"
CMD_PLACEHOLDER
echo "[$(date +%H:%M)] DONE:  NAME_PLACEHOLDER"
TASK
  sed -i "s|NAME_PLACEHOLDER|$name|g" "$GPU_DIR/tasks.sh"
  cmd_escaped=$(echo "$cmd" | sed 's/&/\\&/g')
  sed -i "s|CMD_PLACEHOLDER|$cmd_escaped|g" "$GPU_DIR/tasks.sh"
done

# Launch per-GPU workers
for gpu in "${GPU_LIST[@]}"; do
  GPU_DIR="$(dirname "$TASK_FILE")/gpu${gpu}"
  TASK_COUNT=$(grep -c "START:" "$GPU_DIR/tasks.sh" 2>/dev/null || echo 0)
  if [ "$TASK_COUNT" -gt 0 ]; then
    nohup bash -c "CUDA_VISIBLE_DEVICES=$gpu bash $GPU_DIR/tasks.sh" \
      > "$GPU_DIR/runner.log" 2>&1 &
    echo "  GPU$gpu: $TASK_COUNT tasks (PID $!)"
  fi
done

echo "=== launched ==="
echo "Monitor: watch -n 30 'nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader'"
