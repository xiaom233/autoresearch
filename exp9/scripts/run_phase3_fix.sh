#!/bin/bash
# 重跑 l1+edge (已修复 edge_loss) 及排队中未执行实验
set -e
DEG="exp9/degradation"; EXP="exp9/experiments"; LOG="exp9/logs"
NUM_GPUS=8

declare -a FIXED=(
  # l1+edge (修复后重跑)
  "D1_l1edge|AR_EMBED_DIM=64 AR_LOSS_FN=l1+edge AR_PARAMS_PATH=$DEG/D1_dual.json AR_VAL_PARAMS_PATH=$DEG/D1_dual.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/D1_dual.json AR_EPOCH_BUDGET=2"
  "D2_l1edge|AR_EMBED_DIM=64 AR_LOSS_FN=l1+edge AR_PARAMS_PATH=$DEG/D2_dual.json AR_VAL_PARAMS_PATH=$DEG/D2_dual.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/D2_dual.json AR_EPOCH_BUDGET=2"
  "D3_l1edge|AR_EMBED_DIM=64 AR_LOSS_FN=l1+edge AR_PARAMS_PATH=$DEG/D3_dual.json AR_VAL_PARAMS_PATH=$DEG/D3_dual.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/D3_dual.json AR_EPOCH_BUDGET=2"
  "T1_l1edge|AR_EMBED_DIM=64 AR_LOSS_FN=l1+edge AR_PARAMS_PATH=$DEG/T1_triple.json AR_VAL_PARAMS_PATH=$DEG/T1_triple.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/T1_triple.json AR_EPOCH_BUDGET=2"
  "T2_l1edge|AR_EMBED_DIM=64 AR_LOSS_FN=l1+edge AR_PARAMS_PATH=$DEG/T2_triple.json AR_VAL_PARAMS_PATH=$DEG/T2_triple.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/T2_triple.json AR_EPOCH_BUDGET=2"
  "T3_l1edge|AR_EMBED_DIM=64 AR_LOSS_FN=l1+edge AR_PARAMS_PATH=$DEG/T3_triple.json AR_VAL_PARAMS_PATH=$DEG/T3_triple.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/T3_triple.json AR_EPOCH_BUDGET=2"
  # 排队中未执行
  "T1_mse|AR_EMBED_DIM=64 AR_LOSS_FN=mse AR_PARAMS_PATH=$DEG/T1_triple.json AR_VAL_PARAMS_PATH=$DEG/T1_triple.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/T1_triple.json AR_EPOCH_BUDGET=2"
  "T1_l1fft|AR_EMBED_DIM=64 AR_LOSS_FN=l1+fft AR_PARAMS_PATH=$DEG/T1_triple.json AR_VAL_PARAMS_PATH=$DEG/T1_triple.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/T1_triple.json AR_EPOCH_BUDGET=2"
  "T2_mse|AR_EMBED_DIM=64 AR_LOSS_FN=mse AR_PARAMS_PATH=$DEG/T2_triple.json AR_VAL_PARAMS_PATH=$DEG/T2_triple.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/T2_triple.json AR_EPOCH_BUDGET=2"
  "T2_l1fft|AR_EMBED_DIM=64 AR_LOSS_FN=l1+fft AR_PARAMS_PATH=$DEG/T2_triple.json AR_VAL_PARAMS_PATH=$DEG/T2_triple.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/T2_triple.json AR_EPOCH_BUDGET=2"
  "D3_mse|AR_EMBED_DIM=64 AR_LOSS_FN=mse AR_PARAMS_PATH=$DEG/D3_dual.json AR_VAL_PARAMS_PATH=$DEG/D3_dual.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/D3_dual.json AR_EPOCH_BUDGET=2"
  "T3_mse|AR_EMBED_DIM=64 AR_LOSS_FN=mse AR_PARAMS_PATH=$DEG/T3_triple.json AR_VAL_PARAMS_PATH=$DEG/T3_triple.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/T3_triple.json AR_EPOCH_BUDGET=2"
  "T3_huber|AR_EMBED_DIM=64 AR_LOSS_FN=huber AR_PARAMS_PATH=$DEG/T3_triple.json AR_VAL_PARAMS_PATH=$DEG/T3_triple.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/T3_triple.json AR_EPOCH_BUDGET=2"
  "T3_l1fft|AR_EMBED_DIM=64 AR_LOSS_FN=l1+fft AR_PARAMS_PATH=$DEG/T3_triple.json AR_VAL_PARAMS_PATH=$DEG/T3_triple.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/T3_triple.json AR_EPOCH_BUDGET=2"
)

echo "=== Fix: ${#FIXED[@]} experiments ==="

for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
  SCRIPT="$LOG/gpu${gpu}_fix.sh"
  echo "#!/bin/bash" > "$SCRIPT"; chmod +x "$SCRIPT"
done

for ((i=0; i<${#FIXED[@]}; i++)); do
  gpu=$((i % NUM_GPUS))
  IFS='|' read -r name env_str <<< "${FIXED[$i]}"
  echo "echo \"=== [$name] GPU${gpu} start \$(date) ===\"" >> "$LOG/gpu${gpu}_fix.sh"
  echo "CUDA_VISIBLE_DEVICES=$gpu $env_str AR_CKPT_PREFIX=$EXP/$name AR_QUIET_PIPELINE=1 uv run train.py > $LOG/${name}.log 2>&1" >> "$LOG/gpu${gpu}_fix.sh"
  echo "echo \"=== [$name] GPU${gpu} done \$(date) ===\"" >> "$LOG/gpu${gpu}_fix.sh"
done

for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
  n=$(grep -c "start" "$LOG/gpu${gpu}_fix.sh" 2>/dev/null || echo 0)
  echo "  GPU${gpu}: $n experiments"
  bash "$LOG/gpu${gpu}_fix.sh" &
done

wait
echo "Fix complete."
