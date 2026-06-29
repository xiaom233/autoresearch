#!/bin/bash
# exp_launcher.sh — 训练启动/清理/状态脚本 (v2)
#
# 用法:
#   清理:  bash scripts/exp_launcher.sh cleanup
#   启动:  bash scripts/exp_launcher.sh start TASK_FILE [GPUS]
#   状态:  bash scripts/exp_launcher.sh status
#
# 任务文件支持两种格式:
#   cmd|name           (gpu_runner.sh 格式, 含 GPU_ID 占位符)
#   {"id":"...","cmd":"..."}  (JSONL 格式)
#
# 示例:
#   bash scripts/exp_launcher.sh cleanup
#   bash scripts/exp_launcher.sh start exp38/scripts/phase5_tasks.txt 0,1,2,3,4,5,6,7

set -e

CMD="${1:-help}"
TASK_FILE="${2:-}"
GPUS="${3:-0,1,2,3,4,5,6,7}"

# ============================================================
# cleanup
# ============================================================
cleanup() {
    echo "=== 清理 GPU 进程 ==="
    pkill -9 -f "gpu_scheduler" 2>/dev/null || true
    pkill -9 -f "gpu_runner"    2>/dev/null || true
    sleep 1

    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u | while read pid; do
        [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null || true
    done
    sleep 1

    pkill -9 -f "train.py" 2>/dev/null || true
    sleep 2

    local remaining=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    echo "GPU 残留: $remaining"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null
    echo "=== 清理完成 ==="
}

# ============================================================
# status
# ============================================================
status() {
    echo "=== GPU ==="
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader 2>/dev/null
    echo ""
    echo "=== train.py 进程 ==="
    ps aux | grep "train.py" | grep -v grep | awk '{print $2, $11}' | while read pid cmd; do
        echo "  PID=$pid"
    done
    local count=$(ps aux | grep "train.py" | grep -v grep | wc -l)
    echo "  共 $count 个活跃训练"
}

# ============================================================
# start — 使用 gpu_runner.sh 排队启动
# ============================================================
start() {
    if [ -z "$TASK_FILE" ]; then
        echo "用法: bash scripts/exp_launcher.sh start TASK_FILE [GPUS]"
        exit 1
    fi

    if [ ! -f "$TASK_FILE" ]; then
        echo "❌ 任务文件不存在: $TASK_FILE"
        exit 1
    fi

    # 先清理
    cleanup

    IFS=',' read -ra GPU_LIST <<< "$GPUS"
    echo "=== 启动: $TASK_FILE ==="
    echo "GPU: $GPUS (${#GPU_LIST[@]} 张)"

    # 检测任务格式
    local first_line=$(head -1 "$TASK_FILE")
    local task_count=$(wc -l < "$TASK_FILE")

    if echo "$first_line" | grep -q '^{'; then
        # JSONL 格式 → 转换为 cmd|name 格式的临时文件
        local tmp_task=$(mktemp /tmp/exp_launcher_tasks.XXXXXX)
        .venv/bin/python3 -c "
import json, sys
for line in open('$TASK_FILE'):
    t = json.loads(line.strip())
    print(f\"{t['cmd']}|{t['id']}\")
" > "$tmp_task"
        echo "格式: JSONL → 转换了 $task_count 个任务"
        TASK_FILE="$tmp_task"
    else
        echo "格式: cmd|name ($task_count 个任务)"
    fi

    # 启动 per-GPU workers
    for gpu in "${GPU_LIST[@]}"; do
        nohup bash scripts/gpu_runner.sh "$gpu" "$TASK_FILE" > /dev/null 2>&1 &
        echo "  GPU$gpu: 已启动"
    done

    echo ""
    echo "=== 监控: ==="
    echo "  watch -n 30 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader'"
    echo "  bash scripts/exp_launcher.sh status"
    echo "  wc -l $TASK_FILE  # 剩余任务"
}

# ============================================================
case "$CMD" in
    cleanup) cleanup ;;
    start)   start ;;
    status)  status ;;
    *)       echo "用法: bash scripts/exp_launcher.sh {cleanup|start|status} [args]" ;;
esac
