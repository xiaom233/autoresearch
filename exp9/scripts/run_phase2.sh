#!/bin/bash
# exp9/scripts/run_phase2.sh — Phase 2: 24 组 Fine-tune 变体实验
set -e

EXP_DIR="exp9/experiments"
DEG_DIR="exp9/degradation"
LOG_DIR="exp9/logs"
NUM_GPUS=8

# 阶段边界 (EPOCH_BUDGET=2)
HALF=7547   # fine-tune 中间点 (1 epoch)
P1=10047    # FtCurr 第二阶段 (HALF + 2500)
P2=12547    # FtCurr 第三阶段 (HALF + 5000, 仅三退化)

# 所有实验共享的基础 ENV
BASE="AR_EMBED_DIM=64 AR_EPOCH_BUDGET=2 AR_QUIET_PIPELINE=1"

# 实验定义: "名称|专属ENV"
declare -a EXPS=(
  # ===== D1 (blur→noise) =====
  "D1fc|AR_PARAMS_PATH=$DEG_DIR/D1_dual.json AR_VAL_PARAMS_PATH=$DEG_DIR/D1_dual.json AR_CURRICULUM_CONFIG=0:random,$HALF:$DEG_DIR/D1_phase1.json,$P1:$DEG_DIR/D1_dual.json"
  "D1fr|AR_PARAMS_PATH=$DEG_DIR/D1_dual.json AR_VAL_PARAMS_PATH=$DEG_DIR/D1_dual.json AR_CURRICULUM_CONFIG=0:random,$HALF:$DEG_DIR/D1_dual.json AR_REPLAY_RATIO=0.2"
  "D1ff|AR_PARAMS_PATH=$DEG_DIR/D1_dual.json AR_VAL_PARAMS_PATH=$DEG_DIR/D1_dual.json AR_CURRICULUM_CONFIG=0:random,$HALF:$DEG_DIR/D1_dual.json AR_FREEZE_STAGES=1"
  "D1fl|AR_PARAMS_PATH=$DEG_DIR/D1_dual.json AR_VAL_PARAMS_PATH=$DEG_DIR/D1_dual.json AR_CURRICULUM_CONFIG=0:random,$HALF:$DEG_DIR/D1_dual.json AR_PHASE2_LR_MULT=0.2"

  # ===== D2 (noise→blur) =====
  "D2fc|AR_PARAMS_PATH=$DEG_DIR/D2_dual.json AR_VAL_PARAMS_PATH=$DEG_DIR/D2_dual.json AR_CURRICULUM_CONFIG=0:random,$HALF:$DEG_DIR/D2_phase1.json,$P1:$DEG_DIR/D2_dual.json"
  "D2fr|AR_PARAMS_PATH=$DEG_DIR/D2_dual.json AR_VAL_PARAMS_PATH=$DEG_DIR/D2_dual.json AR_CURRICULUM_CONFIG=0:random,$HALF:$DEG_DIR/D2_dual.json AR_REPLAY_RATIO=0.2"
  "D2ff|AR_PARAMS_PATH=$DEG_DIR/D2_dual.json AR_VAL_PARAMS_PATH=$DEG_DIR/D2_dual.json AR_CURRICULUM_CONFIG=0:random,$HALF:$DEG_DIR/D2_dual.json AR_FREEZE_STAGES=1"
  "D2fl|AR_PARAMS_PATH=$DEG_DIR/D2_dual.json AR_VAL_PARAMS_PATH=$DEG_DIR/D2_dual.json AR_CURRICULUM_CONFIG=0:random,$HALF:$DEG_DIR/D2_dual.json AR_PHASE2_LR_MULT=0.2"

  # ===== D3 (comp→blur) =====
  "D3fc|AR_PARAMS_PATH=$DEG_DIR/D3_dual.json AR_VAL_PARAMS_PATH=$DEG_DIR/D3_dual.json AR_CURRICULUM_CONFIG=0:random,$HALF:$DEG_DIR/D3_phase1.json,$P1:$DEG_DIR/D3_dual.json"
  "D3fr|AR_PARAMS_PATH=$DEG_DIR/D3_dual.json AR_VAL_PARAMS_PATH=$DEG_DIR/D3_dual.json AR_CURRICULUM_CONFIG=0:random,$HALF:$DEG_DIR/D3_dual.json AR_REPLAY_RATIO=0.2"
  "D3ff|AR_PARAMS_PATH=$DEG_DIR/D3_dual.json AR_VAL_PARAMS_PATH=$DEG_DIR/D3_dual.json AR_CURRICULUM_CONFIG=0:random,$HALF:$DEG_DIR/D3_dual.json AR_FREEZE_STAGES=1"
  "D3fl|AR_PARAMS_PATH=$DEG_DIR/D3_dual.json AR_VAL_PARAMS_PATH=$DEG_DIR/D3_dual.json AR_CURRICULUM_CONFIG=0:random,$HALF:$DEG_DIR/D3_dual.json AR_PHASE2_LR_MULT=0.2"

  # ===== T1 (标准三退化) =====
  "T1fc|AR_PARAMS_PATH=$DEG_DIR/T1_triple.json AR_VAL_PARAMS_PATH=$DEG_DIR/T1_triple.json AR_CURRICULUM_CONFIG=0:random,$HALF:$DEG_DIR/T1_phase1.json,$P1:$DEG_DIR/T1_phase2.json,$P2:$DEG_DIR/T1_triple.json"
  "T1fr|AR_PARAMS_PATH=$DEG_DIR/T1_triple.json AR_VAL_PARAMS_PATH=$DEG_DIR/T1_triple.json AR_CURRICULUM_CONFIG=0:random,$HALF:$DEG_DIR/T1_triple.json AR_REPLAY_RATIO=0.2"
  "T1ff|AR_PARAMS_PATH=$DEG_DIR/T1_triple.json AR_VAL_PARAMS_PATH=$DEG_DIR/T1_triple.json AR_CURRICULUM_CONFIG=0:random,$HALF:$DEG_DIR/T1_triple.json AR_FREEZE_STAGES=1"
  "T1fl|AR_PARAMS_PATH=$DEG_DIR/T1_triple.json AR_VAL_PARAMS_PATH=$DEG_DIR/T1_triple.json AR_CURRICULUM_CONFIG=0:random,$HALF:$DEG_DIR/T1_triple.json AR_PHASE2_LR_MULT=0.2"

  # ===== T2 (噪声优先) =====
  "T2fc|AR_PARAMS_PATH=$DEG_DIR/T2_triple.json AR_VAL_PARAMS_PATH=$DEG_DIR/T2_triple.json AR_CURRICULUM_CONFIG=0:random,$HALF:$DEG_DIR/T2_phase1.json,$P1:$DEG_DIR/T2_phase2.json,$P2:$DEG_DIR/T2_triple.json"
  "T2fr|AR_PARAMS_PATH=$DEG_DIR/T2_triple.json AR_VAL_PARAMS_PATH=$DEG_DIR/T2_triple.json AR_CURRICULUM_CONFIG=0:random,$HALF:$DEG_DIR/T2_triple.json AR_REPLAY_RATIO=0.2"
  "T2ff|AR_PARAMS_PATH=$DEG_DIR/T2_triple.json AR_VAL_PARAMS_PATH=$DEG_DIR/T2_triple.json AR_CURRICULUM_CONFIG=0:random,$HALF:$DEG_DIR/T2_triple.json AR_FREEZE_STAGES=1"
  "T2fl|AR_PARAMS_PATH=$DEG_DIR/T2_triple.json AR_VAL_PARAMS_PATH=$DEG_DIR/T2_triple.json AR_CURRICULUM_CONFIG=0:random,$HALF:$DEG_DIR/T2_triple.json AR_PHASE2_LR_MULT=0.2"

  # ===== T3 (混合顺序) =====
  "T3fc|AR_PARAMS_PATH=$DEG_DIR/T3_triple.json AR_VAL_PARAMS_PATH=$DEG_DIR/T3_triple.json AR_CURRICULUM_CONFIG=0:random,$HALF:$DEG_DIR/T3_phase1.json,$P1:$DEG_DIR/T3_phase2.json,$P2:$DEG_DIR/T3_triple.json"
  "T3fr|AR_PARAMS_PATH=$DEG_DIR/T3_triple.json AR_VAL_PARAMS_PATH=$DEG_DIR/T3_triple.json AR_CURRICULUM_CONFIG=0:random,$HALF:$DEG_DIR/T3_triple.json AR_REPLAY_RATIO=0.2"
  "T3ff|AR_PARAMS_PATH=$DEG_DIR/T3_triple.json AR_VAL_PARAMS_PATH=$DEG_DIR/T3_triple.json AR_CURRICULUM_CONFIG=0:random,$HALF:$DEG_DIR/T3_triple.json AR_FREEZE_STAGES=1"
  "T3fl|AR_PARAMS_PATH=$DEG_DIR/T3_triple.json AR_VAL_PARAMS_PATH=$DEG_DIR/T3_triple.json AR_CURRICULUM_CONFIG=0:random,$HALF:$DEG_DIR/T3_triple.json AR_PHASE2_LR_MULT=0.2"
)

# 按 GPU 轮询分配
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
  SCRIPT="$LOG_DIR/gpu${gpu}_p2.sh"
  echo "#!/bin/bash" > "$SCRIPT"
  echo "set -e" >> "$SCRIPT"
  chmod +x "$SCRIPT"
done

for ((i=0; i<${#EXPS[@]}; i++)); do
  gpu=$((i % NUM_GPUS))
  IFS='|' read -r name env_str <<< "${EXPS[$i]}"
  echo "echo \"=== [$name] GPU${gpu} start \$(date) ===\"" >> "$LOG_DIR/gpu${gpu}_p2.sh"
  echo "CUDA_VISIBLE_DEVICES=$gpu $BASE $env_str AR_CKPT_PREFIX=$EXP_DIR/$name uv run train.py > $LOG_DIR/${name}.log 2>&1" >> "$LOG_DIR/gpu${gpu}_p2.sh"
  echo "echo \"=== [$name] GPU${gpu} done \$(date) ===\"" >> "$LOG_DIR/gpu${gpu}_p2.sh"
done

# 并行启动
echo "Launching 8 GPU workers (Phase 2)..."
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
  bash "$LOG_DIR/gpu${gpu}_p2.sh" &
done

wait
echo "All 24 Phase 2 experiments completed."
