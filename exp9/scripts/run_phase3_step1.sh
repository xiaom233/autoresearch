#!/bin/bash
# exp9/scripts/run_phase3_step1.sh — 专家训练 + 级联基线 + Loss 实验 (独立)
set -e

DEG="exp9/degradation"; EXP="exp9/experiments"; LOG="exp9/logs"
NUM_GPUS=8

# ============================================================
# 15 个独立专家训练 (精确步数)
# ============================================================
declare -a EXPS=(
  "D1_E1|AR_EMBED_DIM=48 AR_PARAMS_PATH=$DEG/D1_phase1.json AR_VAL_PARAMS_PATH=$DEG/D1_phase1.json AR_MAX_STEPS=5031"
  "D1_E2|AR_EMBED_DIM=48 AR_PARAMS_PATH=$DEG/E_noise_gaussian_RGB3.json AR_VAL_PARAMS_PATH=$DEG/E_noise_gaussian_RGB3.json AR_MAX_STEPS=5031"
  "D2_E1|AR_EMBED_DIM=48 AR_PARAMS_PATH=$DEG/D2_phase1.json AR_VAL_PARAMS_PATH=$DEG/D2_phase1.json AR_MAX_STEPS=5031"
  "D2_E2|AR_EMBED_DIM=48 AR_PARAMS_PATH=$DEG/E_blur_motion3.json AR_VAL_PARAMS_PATH=$DEG/E_blur_motion3.json AR_MAX_STEPS=5031"
  "D3_E1|AR_EMBED_DIM=48 AR_PARAMS_PATH=$DEG/D3_phase1.json AR_VAL_PARAMS_PATH=$DEG/D3_phase1.json AR_MAX_STEPS=5031"
  "D3_E2|AR_EMBED_DIM=48 AR_PARAMS_PATH=$DEG/E_blur_lens4.json AR_VAL_PARAMS_PATH=$DEG/E_blur_lens4.json AR_MAX_STEPS=5031"
  "T1_E1|AR_EMBED_DIM=48 AR_PARAMS_PATH=$DEG/T1_phase1.json AR_VAL_PARAMS_PATH=$DEG/T1_phase1.json AR_MAX_STEPS=3773"
  "T1_E2|AR_EMBED_DIM=48 AR_PARAMS_PATH=$DEG/E_noise_gaussian_RGB3.json AR_VAL_PARAMS_PATH=$DEG/E_noise_gaussian_RGB3.json AR_MAX_STEPS=3773"
  "T1_E3|AR_EMBED_DIM=48 AR_PARAMS_PATH=$DEG/D3_phase1.json AR_VAL_PARAMS_PATH=$DEG/D3_phase1.json AR_MAX_STEPS=3773"
  "T2_E1|AR_EMBED_DIM=48 AR_PARAMS_PATH=$DEG/T2_phase1.json AR_VAL_PARAMS_PATH=$DEG/T2_phase1.json AR_MAX_STEPS=3773"
  "T2_E2|AR_EMBED_DIM=48 AR_PARAMS_PATH=$DEG/E_blur_motion3.json AR_VAL_PARAMS_PATH=$DEG/E_blur_motion3.json AR_MAX_STEPS=3773"
  "T2_E3|AR_EMBED_DIM=48 AR_PARAMS_PATH=$DEG/E_comp_jpeg2000_3.json AR_VAL_PARAMS_PATH=$DEG/E_comp_jpeg2000_3.json AR_MAX_STEPS=3773"
  "T3_E1|AR_EMBED_DIM=48 AR_PARAMS_PATH=$DEG/T3_phase1.json AR_VAL_PARAMS_PATH=$DEG/T3_phase1.json AR_MAX_STEPS=3773"
  "T3_E2|AR_EMBED_DIM=48 AR_PARAMS_PATH=$DEG/E_comp_jpeg2.json AR_VAL_PARAMS_PATH=$DEG/E_comp_jpeg2.json AR_MAX_STEPS=3773"
  "T3_E3|AR_EMBED_DIM=48 AR_PARAMS_PATH=$DEG/E_noise_poisson5.json AR_VAL_PARAMS_PATH=$DEG/E_noise_poisson5.json AR_MAX_STEPS=3773"
)

# ============================================================
# 6 级联基线 + 24 Loss 实验
# ============================================================
LOSS_FNS="mse huber l1+edge l1+fft"
for deg in D1 D2 D3; do
  # 级联基线
  EXPS+=("${deg}_cb|AR_EMBED_DIM=48 AR_NUM_EXPERTS=2 AR_PARAMS_PATH=$DEG/${deg}_dual.json AR_VAL_PARAMS_PATH=$DEG/${deg}_dual.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/${deg}_dual.json AR_EPOCH_BUDGET=2")
  # Loss
  for loss in $LOSS_FNS; do
    EXPS+=("${deg}_${loss//+/}|AR_EMBED_DIM=64 AR_LOSS_FN=$loss AR_PARAMS_PATH=$DEG/${deg}_dual.json AR_VAL_PARAMS_PATH=$DEG/${deg}_dual.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/${deg}_dual.json AR_EPOCH_BUDGET=2")
  done
done
for deg in T1 T2 T3; do
  EXPS+=("${deg}_cb|AR_EMBED_DIM=48 AR_NUM_EXPERTS=3 AR_PARAMS_PATH=$DEG/${deg}_triple.json AR_VAL_PARAMS_PATH=$DEG/${deg}_triple.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/${deg}_triple.json AR_EPOCH_BUDGET=2")
  for loss in $LOSS_FNS; do
    EXPS+=("${deg}_${loss//+/}|AR_EMBED_DIM=64 AR_LOSS_FN=$loss AR_PARAMS_PATH=$DEG/${deg}_triple.json AR_VAL_PARAMS_PATH=$DEG/${deg}_triple.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/${deg}_triple.json AR_EPOCH_BUDGET=2")
  done
done

echo "=== Phase 3 Step 1: ${#EXPS[@]} experiments ==="

# ============================================================
# GPU 分配
# ============================================================
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
  SCRIPT="$LOG/gpu${gpu}_p3.sh"
  echo "#!/bin/bash" > "$SCRIPT"; echo "set -e" >> "$SCRIPT"; chmod +x "$SCRIPT"
done

for ((i=0; i<${#EXPS[@]}; i++)); do
  gpu=$((i % NUM_GPUS))
  IFS='|' read -r name env_str <<< "${EXPS[$i]}"
  if [[ "$name" == *_cb ]]; then
    runner="uv run python exp9/scripts/train_cascaded.py"
  else
    runner="uv run train.py"
  fi
  echo "echo \"=== [$name] GPU${gpu} start \$(date) ===\"" >> "$LOG/gpu${gpu}_p3.sh"
  echo "CUDA_VISIBLE_DEVICES=$gpu $env_str AR_CKPT_PREFIX=$EXP/$name AR_QUIET_PIPELINE=1 $runner > $LOG/${name}.log 2>&1" >> "$LOG/gpu${gpu}_p3.sh"
  echo "echo \"=== [$name] GPU${gpu} done \$(date) ===\"" >> "$LOG/gpu${gpu}_p3.sh"
done

for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
  n=$(grep -c "start" "$LOG/gpu${gpu}_p3.sh" 2>/dev/null || echo 0)
  echo "  GPU${gpu}: $n experiments"
  bash "$LOG/gpu${gpu}_p3.sh" &
done

wait
echo "=== Phase 3 Step 1 done ==="
echo "Run step 2 for joint fine-tuning."
