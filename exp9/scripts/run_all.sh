#!/bin/bash
# exp9/scripts/run_all.sh — 18 组实验，轮询分配到 8 张 GPU
set -e

EXP_DIR="exp9/experiments"
DEG_DIR="exp9/degradation"
LOG_DIR="exp9/logs"
NUM_GPUS=8

# 阶段边界参数
HALF=7547    # fine-tune 中间切换点（1 epoch 后）
P1=5000      # curriculum 第一阶段边界
P2=10000     # curriculum 第二阶段边界（仅三退化）

# 实验定义: "名称|ENV_VARS"
declare -a EXPS=(
  # D1 (blur→noise)
  "D1d|AR_EPOCH_BUDGET=2 AR_PARAMS_PATH=$DEG_DIR/D1_dual.json AR_VAL_PARAMS_PATH=$DEG_DIR/D1_dual.json AR_CKPT_PREFIX=$EXP_DIR/D1d AR_QUIET_PIPELINE=1"
  "D1c|AR_EPOCH_BUDGET=2 AR_PARAMS_PATH=$DEG_DIR/D1_dual.json AR_VAL_PARAMS_PATH=$DEG_DIR/D1_dual.json AR_CURRICULUM_CONFIG=0:$DEG_DIR/D1_phase1.json,$P1:$DEG_DIR/D1_dual.json AR_CKPT_PREFIX=$EXP_DIR/D1c AR_QUIET_PIPELINE=1"
  "D1f|AR_EPOCH_BUDGET=2 AR_PARAMS_PATH=$DEG_DIR/D1_dual.json AR_VAL_PARAMS_PATH=$DEG_DIR/D1_dual.json AR_CURRICULUM_CONFIG=0:random,$HALF:$DEG_DIR/D1_dual.json AR_CKPT_PREFIX=$EXP_DIR/D1f AR_QUIET_PIPELINE=1"
  # D2 (noise→blur)
  "D2d|AR_EPOCH_BUDGET=2 AR_PARAMS_PATH=$DEG_DIR/D2_dual.json AR_VAL_PARAMS_PATH=$DEG_DIR/D2_dual.json AR_CKPT_PREFIX=$EXP_DIR/D2d AR_QUIET_PIPELINE=1"
  "D2c|AR_EPOCH_BUDGET=2 AR_PARAMS_PATH=$DEG_DIR/D2_dual.json AR_VAL_PARAMS_PATH=$DEG_DIR/D2_dual.json AR_CURRICULUM_CONFIG=0:$DEG_DIR/D2_phase1.json,$P1:$DEG_DIR/D2_dual.json AR_CKPT_PREFIX=$EXP_DIR/D2c AR_QUIET_PIPELINE=1"
  "D2f|AR_EPOCH_BUDGET=2 AR_PARAMS_PATH=$DEG_DIR/D2_dual.json AR_VAL_PARAMS_PATH=$DEG_DIR/D2_dual.json AR_CURRICULUM_CONFIG=0:random,$HALF:$DEG_DIR/D2_dual.json AR_CKPT_PREFIX=$EXP_DIR/D2f AR_QUIET_PIPELINE=1"
  # D3 (comp→blur)
  "D3d|AR_EPOCH_BUDGET=2 AR_PARAMS_PATH=$DEG_DIR/D3_dual.json AR_VAL_PARAMS_PATH=$DEG_DIR/D3_dual.json AR_CKPT_PREFIX=$EXP_DIR/D3d AR_QUIET_PIPELINE=1"
  "D3c|AR_EPOCH_BUDGET=2 AR_PARAMS_PATH=$DEG_DIR/D3_dual.json AR_VAL_PARAMS_PATH=$DEG_DIR/D3_dual.json AR_CURRICULUM_CONFIG=0:$DEG_DIR/D3_phase1.json,$P1:$DEG_DIR/D3_dual.json AR_CKPT_PREFIX=$EXP_DIR/D3c AR_QUIET_PIPELINE=1"
  "D3f|AR_EPOCH_BUDGET=2 AR_PARAMS_PATH=$DEG_DIR/D3_dual.json AR_VAL_PARAMS_PATH=$DEG_DIR/D3_dual.json AR_CURRICULUM_CONFIG=0:random,$HALF:$DEG_DIR/D3_dual.json AR_CKPT_PREFIX=$EXP_DIR/D3f AR_QUIET_PIPELINE=1"
  # T1 (标准三退化)
  "T1d|AR_EPOCH_BUDGET=2 AR_PARAMS_PATH=$DEG_DIR/T1_triple.json AR_VAL_PARAMS_PATH=$DEG_DIR/T1_triple.json AR_CKPT_PREFIX=$EXP_DIR/T1d AR_QUIET_PIPELINE=1"
  "T1c|AR_EPOCH_BUDGET=2 AR_PARAMS_PATH=$DEG_DIR/T1_triple.json AR_VAL_PARAMS_PATH=$DEG_DIR/T1_triple.json AR_CURRICULUM_CONFIG=0:$DEG_DIR/T1_phase1.json,$P1:$DEG_DIR/T1_phase2.json,$P2:$DEG_DIR/T1_triple.json AR_CKPT_PREFIX=$EXP_DIR/T1c AR_QUIET_PIPELINE=1"
  "T1f|AR_EPOCH_BUDGET=2 AR_PARAMS_PATH=$DEG_DIR/T1_triple.json AR_VAL_PARAMS_PATH=$DEG_DIR/T1_triple.json AR_CURRICULUM_CONFIG=0:random,$HALF:$DEG_DIR/T1_triple.json AR_CKPT_PREFIX=$EXP_DIR/T1f AR_QUIET_PIPELINE=1"
  # T2 (噪声优先)
  "T2d|AR_EPOCH_BUDGET=2 AR_PARAMS_PATH=$DEG_DIR/T2_triple.json AR_VAL_PARAMS_PATH=$DEG_DIR/T2_triple.json AR_CKPT_PREFIX=$EXP_DIR/T2d AR_QUIET_PIPELINE=1"
  "T2c|AR_EPOCH_BUDGET=2 AR_PARAMS_PATH=$DEG_DIR/T2_triple.json AR_VAL_PARAMS_PATH=$DEG_DIR/T2_triple.json AR_CURRICULUM_CONFIG=0:$DEG_DIR/T2_phase1.json,$P1:$DEG_DIR/T2_phase2.json,$P2:$DEG_DIR/T2_triple.json AR_CKPT_PREFIX=$EXP_DIR/T2c AR_QUIET_PIPELINE=1"
  "T2f|AR_EPOCH_BUDGET=2 AR_PARAMS_PATH=$DEG_DIR/T2_triple.json AR_VAL_PARAMS_PATH=$DEG_DIR/T2_triple.json AR_CURRICULUM_CONFIG=0:random,$HALF:$DEG_DIR/T2_triple.json AR_CKPT_PREFIX=$EXP_DIR/T2f AR_QUIET_PIPELINE=1"
  # T3 (混合顺序)
  "T3d|AR_EPOCH_BUDGET=2 AR_PARAMS_PATH=$DEG_DIR/T3_triple.json AR_VAL_PARAMS_PATH=$DEG_DIR/T3_triple.json AR_CKPT_PREFIX=$EXP_DIR/T3d AR_QUIET_PIPELINE=1"
  "T3c|AR_EPOCH_BUDGET=2 AR_PARAMS_PATH=$DEG_DIR/T3_triple.json AR_VAL_PARAMS_PATH=$DEG_DIR/T3_triple.json AR_CURRICULUM_CONFIG=0:$DEG_DIR/T3_phase1.json,$P1:$DEG_DIR/T3_phase2.json,$P2:$DEG_DIR/T3_triple.json AR_CKPT_PREFIX=$EXP_DIR/T3c AR_QUIET_PIPELINE=1"
  "T3f|AR_EPOCH_BUDGET=2 AR_PARAMS_PATH=$DEG_DIR/T3_triple.json AR_VAL_PARAMS_PATH=$DEG_DIR/T3_triple.json AR_CURRICULUM_CONFIG=0:random,$HALF:$DEG_DIR/T3_triple.json AR_CKPT_PREFIX=$EXP_DIR/T3f AR_QUIET_PIPELINE=1"
)

# 按 GPU 轮询分配，生成各 GPU 的 shell 脚本
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
  SCRIPT="$LOG_DIR/gpu${gpu}.sh"
  echo "#!/bin/bash" > "$SCRIPT"
  echo "set -e" >> "$SCRIPT"
  chmod +x "$SCRIPT"
done

for ((i=0; i<${#EXPS[@]}; i++)); do
  gpu=$((i % NUM_GPUS))
  IFS='|' read -r name env_str <<< "${EXPS[$i]}"
  echo "echo \"=== [$name] GPU${gpu} start \$(date) ===\"" >> "$LOG_DIR/gpu${gpu}.sh"
  echo "CUDA_VISIBLE_DEVICES=$gpu $env_str uv run train.py > $LOG_DIR/${name}.log 2>&1" >> "$LOG_DIR/gpu${gpu}.sh"
  echo "echo \"=== [$name] GPU${gpu} done \$(date) ===\"" >> "$LOG_DIR/gpu${gpu}.sh"
done

# 并行启动所有 GPU 脚本
echo "Launching 8 GPU workers..."
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
  bash "$LOG_DIR/gpu${gpu}.sh" &
done

wait
echo "All 18 experiments completed."
