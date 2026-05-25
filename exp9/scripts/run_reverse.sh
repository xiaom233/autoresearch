#!/bin/bash
# 逆序 Curric + FtCurr (修复顺序 = 退化施加的逆序: 从外到内剥离)
set -e
DEG="exp9/degradation"; EXP="exp9/experiments"; LOG="exp9/logs"
BASE="AR_EMBED_DIM=64 AR_EPOCH_BUDGET=2 AR_QUIET_PIPELINE=1"

# Curric(rev): 从最后一个退化开始, 逐步向外剥离
# FtCurr(rev): 盲预训练后, 逆序渐进
declare -a EXPS=(
  # === D1 blur→noise: 逆序 = noise → blur+noise ===
  "D1_cr|AR_PARAMS_PATH=$DEG/D1_dual.json AR_VAL_PARAMS_PATH=$DEG/D1_dual.json AR_CURRICULUM_CONFIG=0:$DEG/E_noise_gaussian_RGB3.json,5000:$DEG/D1_dual.json"
  "D1_fcr|AR_PARAMS_PATH=$DEG/D1_dual.json AR_VAL_PARAMS_PATH=$DEG/D1_dual.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/E_noise_gaussian_RGB3.json,10047:$DEG/D1_dual.json"

  # === D2 noise→blur: 逆序 = blur → noise+blur ===
  "D2_cr|AR_PARAMS_PATH=$DEG/D2_dual.json AR_VAL_PARAMS_PATH=$DEG/D2_dual.json AR_CURRICULUM_CONFIG=0:$DEG/E_blur_motion3.json,5000:$DEG/D2_dual.json"
  "D2_fcr|AR_PARAMS_PATH=$DEG/D2_dual.json AR_VAL_PARAMS_PATH=$DEG/D2_dual.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/E_blur_motion3.json,10047:$DEG/D2_dual.json"

  # === D3 comp→blur: 逆序 = blur → comp+blur ===
  "D3_cr|AR_PARAMS_PATH=$DEG/D3_dual.json AR_VAL_PARAMS_PATH=$DEG/D3_dual.json AR_CURRICULUM_CONFIG=0:$DEG/E_blur_lens4.json,5000:$DEG/D3_dual.json"
  "D3_fcr|AR_PARAMS_PATH=$DEG/D3_dual.json AR_VAL_PARAMS_PATH=$DEG/D3_dual.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/E_blur_lens4.json,10047:$DEG/D3_dual.json"

  # === T1 blur→noise→jpeg: 逆序 = jpeg → noise+jpeg → full ===
  "T1_cr|AR_PARAMS_PATH=$DEG/T1_triple.json AR_VAL_PARAMS_PATH=$DEG/T1_triple.json AR_CURRICULUM_CONFIG=0:$DEG/D3_phase1.json,5000:$DEG/T1_rev2.json,10000:$DEG/T1_triple.json"
  "T1_fcr|AR_PARAMS_PATH=$DEG/T1_triple.json AR_VAL_PARAMS_PATH=$DEG/T1_triple.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/D3_phase1.json,10047:$DEG/T1_rev2.json,12547:$DEG/T1_triple.json"

  # === T2 noise→blur→jpeg2000: 逆序 = jpeg2000 → blur+jpeg2000 → full ===
  "T2_cr|AR_PARAMS_PATH=$DEG/T2_triple.json AR_VAL_PARAMS_PATH=$DEG/T2_triple.json AR_CURRICULUM_CONFIG=0:$DEG/E_comp_jpeg2000_3.json,5000:$DEG/T2_rev2.json,10000:$DEG/T2_triple.json"
  "T2_fcr|AR_PARAMS_PATH=$DEG/T2_triple.json AR_VAL_PARAMS_PATH=$DEG/T2_triple.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/E_comp_jpeg2000_3.json,10047:$DEG/T2_rev2.json,12547:$DEG/T2_triple.json"

  # === T3 blur→jpeg→noise: 逆序 = noise → jpeg+noise → full ===
  "T3_cr|AR_PARAMS_PATH=$DEG/T3_triple.json AR_VAL_PARAMS_PATH=$DEG/T3_triple.json AR_CURRICULUM_CONFIG=0:$DEG/E_noise_poisson5.json,5000:$DEG/T3_rev2.json,10000:$DEG/T3_triple.json"
  "T3_fcr|AR_PARAMS_PATH=$DEG/T3_triple.json AR_VAL_PARAMS_PATH=$DEG/T3_triple.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/E_noise_poisson5.json,10047:$DEG/T3_rev2.json,12547:$DEG/T3_triple.json"
)

echo "=== Reverse Curric + FtCurr: ${#EXPS[@]} experiments ==="

# 按 GPU 分配 (12 组, 8 GPU)
for ((gpu=0; gpu<8; gpu++)); do
  SCRIPT="$LOG/gpu${gpu}_rev.sh"
  echo "#!/bin/bash" > "$SCRIPT"; chmod +x "$SCRIPT"
done

for ((i=0; i<${#EXPS[@]}; i++)); do
  gpu=$((i % 8))
  IFS='|' read -r name env_str <<< "${EXPS[$i]}"
  echo "echo \"=== [$name] GPU${gpu} start \$(date) ===\"" >> "$LOG/gpu${gpu}_rev.sh"
  echo "CUDA_VISIBLE_DEVICES=$gpu $BASE $env_str AR_CKPT_PREFIX=$EXP/$name uv run train.py > $LOG/${name}.log 2>&1" >> "$LOG/gpu${gpu}_rev.sh"
  echo "echo \"=== [$name] GPU${gpu} done \$(date) ===\"" >> "$LOG/gpu${gpu}_rev.sh"
done

echo "分配:"
for ((gpu=0; gpu<8; gpu++)); do
  n=$(grep -c "start" "$LOG/gpu${gpu}_rev.sh" 2>/dev/null || echo 0)
  names=$(grep "start" "$LOG/gpu${gpu}_rev.sh" | sed 's/.*\[\(.*\)\].*/\1/' | tr '\n' ' ')
  echo "  GPU${gpu}: $n ($names)"
done

for ((gpu=0; gpu<8; gpu++)); do
  bash "$LOG/gpu${gpu}_rev.sh" &
done

wait
echo "All reverse experiments done."
