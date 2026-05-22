#!/bin/bash
# setup_challenge.sh — 盲识别挑战的一站式初始化
#
# 此脚本封装了挑战生成和参数导出的全部流程，确保 Agent 在任何情况下
# 都不会接触到 .ground_truth.json 或退化 pipeline 的内容。
#
# 用法:
#   bash setup_challenge.sh --exp exp7                          # 生成新挑战到 exp7/
#   bash setup_challenge.sh --exp exp7 --seed 42                # 可复现
#   bash setup_challenge.sh --exp exp7 --target-only            # 无参考模式
#
# 输出（Agent 可见）:
#   challenge_id: blind_XXXXXXXX
#   target:      /path/to/degraded.png
#   ref:         /path/to/clean.png
#   export:      exp7/blind_challenge/
#   params:      exp7/degradation/params.json
#   READY
#
# 隔离保证:
#   1. blind_challenge.py 在子进程中运行（--quiet），pipeline 不输出
#   2. params.json 通过 --params-output 静默导出，不经过 stdout
#   3. .ground_truth.json 仅供 evaluate_blind_challenge.py 读取
#   4. Agent 在整个流程中无需打开 .ground_truth.json 或 degradation/params.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- 参数解析 ----
EXP_NAME=""
SEED=""
NUM_DEGS=""
TARGET_ONLY=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp)
            EXP_NAME="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --num-degs)
            NUM_DEGS="$2"
            shift 2
            ;;
        --target-only)
            TARGET_ONLY=true
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ -z "$EXP_NAME" ]; then
    echo "ERROR: --exp <name> is required (e.g. --exp exp7)"
    exit 1
fi

# ---- 创建目录结构 ----
mkdir -p "$EXP_NAME"/{blind_challenge,degradation,experiments,logs,model,results,summarize}

# ---- 构建 blind_challenge.py 参数 ----
BC_ARGS=(--output-dir "$EXP_NAME/blind_challenge")
BC_ARGS+=(--params-output "$EXP_NAME/degradation/params.json")
BC_ARGS+=(--quiet)

if [ -n "$SEED" ]; then
    BC_ARGS+=(--seed "$SEED")
fi

if [ -n "$NUM_DEGS" ]; then
    BC_ARGS+=(--num-degs "$NUM_DEGS")
fi
if [ "$TARGET_ONLY" = true ]; then
    BC_ARGS+=(--target-only)
fi

# ---- 子进程生成挑战（Agent 看不到子进程内部状态）----
# stdout 捕获后只输出公开信息
OUTPUT=$(.venv/bin/python3 blind_challenge.py "${BC_ARGS[@]}" 2>&1)
echo "$OUTPUT"

# ---- 快照 train.py ----
cp train.py "$EXP_NAME/model/train.py"

# ---- 验证隔离 ----
# 确保 .ground_truth.json 仅存在于 blind_challenge/ 目录（隐藏文件）
if [ -f "$EXP_NAME/blind_challenge/.ground_truth.json" ]; then
    echo "setup: OK (ground truth sealed in $EXP_NAME/blind_challenge/.ground_truth.json)"
else
    echo "setup: WARNING — ground truth file missing!"
fi

# 验证 degradation/params.json 不含 step/category 字段
if [ -f "$EXP_NAME/degradation/params.json" ]; then
    HAS_STEP=$(python3 -c "import json; d=json.load(open('$EXP_NAME/degradation/params.json')); print(any('step' in s for s in d.get('pipeline',[])))" 2>/dev/null || echo "True")
    if [ "$HAS_STEP" = "True" ]; then
        echo "setup: WARNING — params.json contains internal fields (step/category leak!)"
    fi
fi
