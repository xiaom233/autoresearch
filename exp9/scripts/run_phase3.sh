#!/bin/bash
# exp9/scripts/run_phase3.sh — Phase 3: 多专家 + Loss 耦合
set -e

DEG="exp9/degradation"
EXP="exp9/experiments"
LOG="exp9/logs"
TRAIN="uv run python exp9/scripts/train_cascaded.py"
TRAIN1="uv run train.py"  # 单专家训练用标准 train.py
NUM_GPUS=8
BASE="AR_EMBED_DIM=48 AR_QUIET_PIPELINE=1"

# ============================================================
# 阶段 1: 15 个独立专家训练
# ============================================================
declare -a EXPERT_EXPS=(
  # D1 (blur→noise): 2 experts, 各 5031 步
  "D1_E1|AR_PARAMS_PATH=$DEG/D1_phase1.json AR_VAL_PARAMS_PATH=$DEG/D1_phase1.json AR_MAX_STEPS=5031"
  "D1_E2|AR_PARAMS_PATH=$DEG/E_noise_gaussian_RGB3.json AR_VAL_PARAMS_PATH=$DEG/E_noise_gaussian_RGB3.json AR_MAX_STEPS=5031"
  # D2 (noise→blur): 2 experts, 各 5031 步
  "D2_E1|AR_PARAMS_PATH=$DEG/D2_phase1.json AR_VAL_PARAMS_PATH=$DEG/D2_phase1.json AR_MAX_STEPS=5031"
  "D2_E2|AR_PARAMS_PATH=$DEG/E_blur_motion3.json AR_VAL_PARAMS_PATH=$DEG/E_blur_motion3.json AR_MAX_STEPS=5031"
  # D3 (comp→blur): 2 experts, 各 5031 步
  "D3_E1|AR_PARAMS_PATH=$DEG/D3_phase1.json AR_VAL_PARAMS_PATH=$DEG/D3_phase1.json AR_MAX_STEPS=5031"
  "D3_E2|AR_PARAMS_PATH=$DEG/E_blur_lens4.json AR_VAL_PARAMS_PATH=$DEG/E_blur_lens4.json AR_MAX_STEPS=5031"
  # T1 (标准): 3 experts, 各 3773 步
  "T1_E1|AR_PARAMS_PATH=$DEG/T1_phase1.json AR_VAL_PARAMS_PATH=$DEG/T1_phase1.json AR_MAX_STEPS=3773"
  "T1_E2|AR_PARAMS_PATH=$DEG/E_noise_gaussian_RGB3.json AR_VAL_PARAMS_PATH=$DEG/E_noise_gaussian_RGB3.json AR_MAX_STEPS=3773"
  "T1_E3|AR_PARAMS_PATH=$DEG/D3_phase1.json AR_VAL_PARAMS_PATH=$DEG/D3_phase1.json AR_MAX_STEPS=3773"
  # T2 (噪声优先): 3 experts, 各 3773 步
  "T2_E1|AR_PARAMS_PATH=$DEG/T2_phase1.json AR_VAL_PARAMS_PATH=$DEG/T2_phase1.json AR_MAX_STEPS=3773"
  "T2_E2|AR_PARAMS_PATH=$DEG/E_blur_motion3.json AR_VAL_PARAMS_PATH=$DEG/E_blur_motion3.json AR_MAX_STEPS=3773"
  "T2_E3|AR_PARAMS_PATH=$DEG/E_comp_jpeg2000_3.json AR_VAL_PARAMS_PATH=$DEG/E_comp_jpeg2000_3.json AR_MAX_STEPS=3773"
  # T3 (混合): 3 experts, 各 3773 步
  "T3_E1|AR_PARAMS_PATH=$DEG/T3_phase1.json AR_VAL_PARAMS_PATH=$DEG/T3_phase1.json AR_MAX_STEPS=3773"
  "T3_E2|AR_PARAMS_PATH=$DEG/E_comp_jpeg2.json AR_VAL_PARAMS_PATH=$DEG/E_comp_jpeg2.json AR_MAX_STEPS=3773"
  "T3_E3|AR_PARAMS_PATH=$DEG/E_noise_poisson5.json AR_VAL_PARAMS_PATH=$DEG/E_noise_poisson5.json AR_MAX_STEPS=3773"
)

# ============================================================
# 阶段 2: 级联基线 (6 组) + 联合微调 (6 组) + Loss 实验 (24 组)
# ============================================================
LOSSES=("mse" "huber" "l1+edge" "l1+fft")

declare -a MAIN_EXPS=(
  # === D1 (blur→noise) ===
  "D1_cb|AR_NUM_EXPERTS=2 AR_PARAMS_PATH=$DEG/D1_dual.json AR_VAL_PARAMS_PATH=$DEG/D1_dual.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/D1_dual.json AR_EPOCH_BUDGET=2"
  "D1_me|AR_NUM_EXPERTS=2 AR_LOAD_EXPERTS=$EXP/D1_E1/checkpoints/D1_E1_step5030.pt,$EXP/D1_E2/checkpoints/D1_E2_step5030.pt AR_PARAMS_PATH=$DEG/D1_dual.json AR_VAL_PARAMS_PATH=$DEG/D1_dual.json AR_MAX_STEPS=5032"
  # D1 Loss 实验
  "D1_mse|AR_LOSS_FN=mse AR_PARAMS_PATH=$DEG/D1_dual.json AR_VAL_PARAMS_PATH=$DEG/D1_dual.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/D1_dual.json AR_EPOCH_BUDGET=2"
  "D1_huber|AR_LOSS_FN=huber AR_PARAMS_PATH=$DEG/D1_dual.json AR_VAL_PARAMS_PATH=$DEG/D1_dual.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/D1_dual.json AR_EPOCH_BUDGET=2"
  "D1_edge|AR_LOSS_FN=l1+edge AR_PARAMS_PATH=$DEG/D1_dual.json AR_VAL_PARAMS_PATH=$DEG/D1_dual.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/D1_dual.json AR_EPOCH_BUDGET=2"
  "D1_fft|AR_LOSS_FN=l1+fft AR_PARAMS_PATH=$DEG/D1_dual.json AR_VAL_PARAMS_PATH=$DEG/D1_dual.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/D1_dual.json AR_EPOCH_BUDGET=2"

  # === D2 (noise→blur) ===
  "D2_cb|AR_NUM_EXPERTS=2 AR_PARAMS_PATH=$DEG/D2_dual.json AR_VAL_PARAMS_PATH=$DEG/D2_dual.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/D2_dual.json AR_EPOCH_BUDGET=2"
  "D2_me|AR_NUM_EXPERTS=2 AR_LOAD_EXPERTS=$EXP/D2_E1/checkpoints/D2_E1_best.pt,$EXP/D2_E2/checkpoints/D2_E2_best.pt AR_PARAMS_PATH=$DEG/D2_dual.json AR_VAL_PARAMS_PATH=$DEG/D2_dual.json AR_TIME_BUDGET=1736"
  "D2_mse|AR_LOSS_FN=mse AR_PARAMS_PATH=$DEG/D2_dual.json AR_VAL_PARAMS_PATH=$DEG/D2_dual.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/D2_dual.json AR_EPOCH_BUDGET=2"
  "D2_huber|AR_LOSS_FN=huber AR_PARAMS_PATH=$DEG/D2_dual.json AR_VAL_PARAMS_PATH=$DEG/D2_dual.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/D2_dual.json AR_EPOCH_BUDGET=2"
  "D2_edge|AR_LOSS_FN=l1+edge AR_PARAMS_PATH=$DEG/D2_dual.json AR_VAL_PARAMS_PATH=$DEG/D2_dual.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/D2_dual.json AR_EPOCH_BUDGET=2"
  "D2_fft|AR_LOSS_FN=l1+fft AR_PARAMS_PATH=$DEG/D2_dual.json AR_VAL_PARAMS_PATH=$DEG/D2_dual.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/D2_dual.json AR_EPOCH_BUDGET=2"

  # === D3 (comp→blur) ===
  "D3_cb|AR_NUM_EXPERTS=2 AR_PARAMS_PATH=$DEG/D3_dual.json AR_VAL_PARAMS_PATH=$DEG/D3_dual.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/D3_dual.json AR_EPOCH_BUDGET=2"
  "D3_me|AR_NUM_EXPERTS=2 AR_LOAD_EXPERTS=$EXP/D3_E1/checkpoints/D3_E1_best.pt,$EXP/D3_E2/checkpoints/D3_E2_best.pt AR_PARAMS_PATH=$DEG/D3_dual.json AR_VAL_PARAMS_PATH=$DEG/D3_dual.json AR_TIME_BUDGET=1736"
  "D3_mse|AR_LOSS_FN=mse AR_PARAMS_PATH=$DEG/D3_dual.json AR_VAL_PARAMS_PATH=$DEG/D3_dual.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/D3_dual.json AR_EPOCH_BUDGET=2"
  "D3_huber|AR_LOSS_FN=huber AR_PARAMS_PATH=$DEG/D3_dual.json AR_VAL_PARAMS_PATH=$DEG/D3_dual.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/D3_dual.json AR_EPOCH_BUDGET=2"
  "D3_edge|AR_LOSS_FN=l1+edge AR_PARAMS_PATH=$DEG/D3_dual.json AR_VAL_PARAMS_PATH=$DEG/D3_dual.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/D3_dual.json AR_EPOCH_BUDGET=2"
  "D3_fft|AR_LOSS_FN=l1+fft AR_PARAMS_PATH=$DEG/D3_dual.json AR_VAL_PARAMS_PATH=$DEG/D3_dual.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/D3_dual.json AR_EPOCH_BUDGET=2"

  # === T1 (标准三退) ===
  "T1_cb|AR_NUM_EXPERTS=3 AR_PARAMS_PATH=$DEG/T1_triple.json AR_VAL_PARAMS_PATH=$DEG/T1_triple.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/T1_triple.json AR_EPOCH_BUDGET=2"
  "T1_me|AR_NUM_EXPERTS=3 AR_LOAD_EXPERTS=$EXP/T1_E1/checkpoints/T1_E1_best.pt,$EXP/T1_E2/checkpoints/T1_E2_best.pt,$EXP/T1_E3/checkpoints/T1_E3_best.pt AR_PARAMS_PATH=$DEG/T1_triple.json AR_VAL_PARAMS_PATH=$DEG/T1_triple.json AR_TIME_BUDGET=1302"
  "T1_mse|AR_LOSS_FN=mse AR_PARAMS_PATH=$DEG/T1_triple.json AR_VAL_PARAMS_PATH=$DEG/T1_triple.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/T1_triple.json AR_EPOCH_BUDGET=2"
  "T1_huber|AR_LOSS_FN=huber AR_PARAMS_PATH=$DEG/T1_triple.json AR_VAL_PARAMS_PATH=$DEG/T1_triple.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/T1_triple.json AR_EPOCH_BUDGET=2"
  "T1_edge|AR_LOSS_FN=l1+edge AR_PARAMS_PATH=$DEG/T1_triple.json AR_VAL_PARAMS_PATH=$DEG/T1_triple.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/T1_triple.json AR_EPOCH_BUDGET=2"
  "T1_fft|AR_LOSS_FN=l1+fft AR_PARAMS_PATH=$DEG/T1_triple.json AR_VAL_PARAMS_PATH=$DEG/T1_triple.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/T1_triple.json AR_EPOCH_BUDGET=2"

  # === T2 (噪声优先) ===
  "T2_cb|AR_NUM_EXPERTS=3 AR_PARAMS_PATH=$DEG/T2_triple.json AR_VAL_PARAMS_PATH=$DEG/T2_triple.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/T2_triple.json AR_EPOCH_BUDGET=2"
  "T2_me|AR_NUM_EXPERTS=3 AR_LOAD_EXPERTS=$EXP/T2_E1/checkpoints/T2_E1_best.pt,$EXP/T2_E2/checkpoints/T2_E2_best.pt,$EXP/T2_E3/checkpoints/T2_E3_best.pt AR_PARAMS_PATH=$DEG/T2_triple.json AR_VAL_PARAMS_PATH=$DEG/T2_triple.json AR_TIME_BUDGET=1302"
  "T2_mse|AR_LOSS_FN=mse AR_PARAMS_PATH=$DEG/T2_triple.json AR_VAL_PARAMS_PATH=$DEG/T2_triple.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/T2_triple.json AR_EPOCH_BUDGET=2"
  "T2_huber|AR_LOSS_FN=huber AR_PARAMS_PATH=$DEG/T2_triple.json AR_VAL_PARAMS_PATH=$DEG/T2_triple.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/T2_triple.json AR_EPOCH_BUDGET=2"
  "T2_edge|AR_LOSS_FN=l1+edge AR_PARAMS_PATH=$DEG/T2_triple.json AR_VAL_PARAMS_PATH=$DEG/T2_triple.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/T2_triple.json AR_EPOCH_BUDGET=2"
  "T2_fft|AR_LOSS_FN=l1+fft AR_PARAMS_PATH=$DEG/T2_triple.json AR_VAL_PARAMS_PATH=$DEG/T2_triple.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/T2_triple.json AR_EPOCH_BUDGET=2"

  # === T3 (混合) ===
  "T3_cb|AR_NUM_EXPERTS=3 AR_PARAMS_PATH=$DEG/T3_triple.json AR_VAL_PARAMS_PATH=$DEG/T3_triple.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/T3_triple.json AR_EPOCH_BUDGET=2"
  "T3_me|AR_NUM_EXPERTS=3 AR_LOAD_EXPERTS=$EXP/T3_E1/checkpoints/T3_E1_best.pt,$EXP/T3_E2/checkpoints/T3_E2_best.pt,$EXP/T3_E3/checkpoints/T3_E3_best.pt AR_PARAMS_PATH=$DEG/T3_triple.json AR_VAL_PARAMS_PATH=$DEG/T3_triple.json AR_TIME_BUDGET=1302"
  "T3_mse|AR_LOSS_FN=mse AR_PARAMS_PATH=$DEG/T3_triple.json AR_VAL_PARAMS_PATH=$DEG/T3_triple.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/T3_triple.json AR_EPOCH_BUDGET=2"
  "T3_huber|AR_LOSS_FN=huber AR_PARAMS_PATH=$DEG/T3_triple.json AR_VAL_PARAMS_PATH=$DEG/T3_triple.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/T3_triple.json AR_EPOCH_BUDGET=2"
  "T3_edge|AR_LOSS_FN=l1+edge AR_PARAMS_PATH=$DEG/T3_triple.json AR_VAL_PARAMS_PATH=$DEG/T3_triple.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/T3_triple.json AR_EPOCH_BUDGET=2"
  "T3_fft|AR_LOSS_FN=l1+fft AR_PARAMS_PATH=$DEG/T3_triple.json AR_VAL_PARAMS_PATH=$DEG/T3_triple.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/T3_triple.json AR_EPOCH_BUDGET=2"
)

echo "=== Phase 3: $((${#EXPERT_EXPS[@]} + ${#MAIN_EXPS[@]})) experiments ==="
echo "Experts: ${#EXPERT_EXPS[@]} (individual training)"
echo "Main: ${#MAIN_EXPS[@]} (6 cb + 6 me + 24 loss)"

# ============================================================
# 按 GPU 轮询分配
# ============================================================
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
  SCRIPT="$LOG/gpu${gpu}_p3.sh"
  echo "#!/bin/bash" > "$SCRIPT"
  echo "set -e" >> "$SCRIPT"
  chmod +x "$SCRIPT"
done

# 专家训练
for ((i=0; i<${#EXPERT_EXPS[@]}; i++)); do
  gpu=$((i % NUM_GPUS))
  IFS='|' read -r name env_str <<< "${EXPERT_EXPS[$i]}"
  echo "echo \"=== [$name] GPU${gpu} start \$(date) ===\"" >> "$LOG/gpu${gpu}_p3.sh"
  echo "CUDA_VISIBLE_DEVICES=$gpu $BASE $env_str AR_CKPT_PREFIX=$EXP/$name $TRAIN1 > $LOG/${name}.log 2>&1" >> "$LOG/gpu${gpu}_p3.sh"
  echo "echo \"=== [$name] GPU${gpu} done \$(date) ===\"" >> "$LOG/gpu${gpu}_p3.sh"
done

# 主实验 (级联基线用 train_cascaded, Loss 实验用 train.py)
for ((i=0; i<${#MAIN_EXPS[@]}; i++)); do
  gpu=$(( (${#EXPERT_EXPS[@]} + i) % NUM_GPUS ))
  IFS='|' read -r name env_str <<< "${MAIN_EXPS[$i]}"
  if [[ "$name" == *_cb ]] || [[ "$name" == *_me ]]; then
    runner="$TRAIN"  # 级联用 train_cascaded
    runner_base="AR_EMBED_DIM=48"
  else
    runner="$TRAIN1"  # Loss 用标准 train.py
    runner_base="AR_EMBED_DIM=64"
  fi
  echo "echo \"=== [$name] GPU${gpu} start \$(date) ===\"" >> "$LOG/gpu${gpu}_p3.sh"
  echo "CUDA_VISIBLE_DEVICES=$gpu $runner_base AR_QUIET_PIPELINE=1 $env_str AR_CKPT_PREFIX=$EXP/$name $runner > $LOG/${name}.log 2>&1" >> "$LOG/gpu${gpu}_p3.sh"
  echo "echo \"=== [$name] GPU${gpu} done \$(date) ===\"" >> "$LOG/gpu${gpu}_p3.sh"
done

# ============================================================
# 启动
# ============================================================
echo "Launching 8 GPU workers..."
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
  exp_count=$(grep -c "start" "$LOG/gpu${gpu}_p3.sh" 2>/dev/null || echo 0)
  echo "  GPU${gpu}: $exp_count experiments"
  bash "$LOG/gpu${gpu}_p3.sh" &
done

wait
echo "Phase 3 complete."
