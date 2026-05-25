#!/bin/bash
# Phase 5: 48 experiments — 12 degradations × 4 strategies (Direct/Ft/Curric(fwd)/Curric(rev))
set -e
DEG="exp9/degradation"; EXP="exp9/experiments"; LOG="exp9/logs"
BASE="AR_EMBED_DIM=64 AR_EPOCH_BUDGET=2 AR_QUIET_PIPELINE=1"

declare -a EXPS=()

# 添加实验: name|extra_env
add_exp() { EXPS+=("$1|$2"); }

for N in N1 N2 N3 N4 N5 N6 N7 N8; do
    T="${DEG}/${N}_dual.json"
    F1="${DEG}/${N}_fwd1.json"
    R1="${DEG}/${N}_rev1.json"
    # Direct
    add_exp "${N}d" "AR_PARAMS_PATH=$T AR_VAL_PARAMS_PATH=$T"
    # Ft
    add_exp "${N}f" "AR_PARAMS_PATH=$T AR_VAL_PARAMS_PATH=$T AR_CURRICULUM_CONFIG=0:random,7547:$T"
    # Curric(fwd): step1 → full
    add_exp "${N}cf" "AR_PARAMS_PATH=$T AR_VAL_PARAMS_PATH=$T AR_CURRICULUM_CONFIG=0:$F1,5000:$T"
    # Curric(rev): step2 → full
    add_exp "${N}cr" "AR_PARAMS_PATH=$T AR_VAL_PARAMS_PATH=$T AR_CURRICULUM_CONFIG=0:$R1,5000:$T"
done

for N in N9 N10 N11 N12; do
    T="${DEG}/${N}_triple.json"
    F1="${DEG}/${N}_fwd1.json"; F2="${DEG}/${N}_fwd2.json"
    R1="${DEG}/${N}_rev1.json"; R2="${DEG}/${N}_rev2.json"
    # Direct
    add_exp "${N}d" "AR_PARAMS_PATH=$T AR_VAL_PARAMS_PATH=$T"
    # Ft
    add_exp "${N}f" "AR_PARAMS_PATH=$T AR_VAL_PARAMS_PATH=$T AR_CURRICULUM_CONFIG=0:random,7547:$T"
    # Curric(fwd): step1 → step1+2 → full
    add_exp "${N}cf" "AR_PARAMS_PATH=$T AR_VAL_PARAMS_PATH=$T AR_CURRICULUM_CONFIG=0:$F1,5000:$F2,10000:$T"
    # Curric(rev): step3 → step3+2 → full
    add_exp "${N}cr" "AR_PARAMS_PATH=$T AR_VAL_PARAMS_PATH=$T AR_CURRICULUM_CONFIG=0:$R1,5000:$R2,10000:$T"
done

echo "=== Phase 5: ${#EXPS[@]} experiments ==="

# GPU 轮询分配
NUM_GPUS=8
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
  SCRIPT="$LOG/gpu${gpu}_p5.sh"
  echo "#!/bin/bash" > "$SCRIPT"; chmod +x "$SCRIPT"
done

for ((i=0; i<${#EXPS[@]}; i++)); do
  gpu=$((i % NUM_GPUS))
  IFS='|' read -r name env_str <<< "${EXPS[$i]}"
  echo "echo \"=== [$name] GPU${gpu} start \$(date) ===\"" >> "$LOG/gpu${gpu}_p5.sh"
  echo "CUDA_VISIBLE_DEVICES=$gpu $BASE $env_str AR_CKPT_PREFIX=$EXP/$name uv run train.py > $LOG/${name}.log 2>&1" >> "$LOG/gpu${gpu}_p5.sh"
  echo "echo \"=== [$name] GPU${gpu} done \$(date) ===\"" >> "$LOG/gpu${gpu}_p5.sh"
done

echo "分配:"
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
  n=$(grep -c "start" "$LOG/gpu${gpu}_p5.sh" 2>/dev/null || echo 0)
  echo "  GPU${gpu}: $n experiments"
done

for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
  bash "$LOG/gpu${gpu}_p5.sh" &
done
wait
echo "Phase 5 done."
