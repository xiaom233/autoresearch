#!/bin/bash
# Phase 7: 18组 Mixed Warmup + Phase 8: 16组 blur交叉验证 = 34组
set -e
DEG="exp9/degradation"; EXP="exp9/experiments"; LOG="exp9/logs"
BASE="AR_EMBED_DIM=64 AR_EPOCH_BUDGET=2 AR_QUIET_PIPELINE=1"
NUM_GPUS=8

# 双退化混合预热计划 (N1-N8)
MIX_DUAL="0:70:30,2500:40:60,5000:10:90,7547:0:100"
# 三退化混合预热计划 (M1-M10)
MIX_TRIPLE="0:50:30:20,2500:25:35:40,5000:5:20:75,7547:0:0:100"

declare -a EXPS=()

# === Phase 7: Mixed Warmup (18组) ===
for N in N1 N2 N3 N4 N5 N6 N7 N8; do
    T="${DEG}/${N}_dual.json"
    EXPS+=("${N}_mw|AR_PARAMS_PATH=$T AR_VAL_PARAMS_PATH=$T AR_MIXED_WARMUP=$MIX_DUAL")
done
for N in M1 M2 M3 M4 M5 M6 M7 M8 M9 M10; do
    T="${DEG}/${N}_triple.json"
    EXPS+=("${N}_mw|AR_PARAMS_PATH=$T AR_VAL_PARAMS_PATH=$T AR_MIXED_WARMUP=$MIX_TRIPLE")
done

# === Phase 8: blur交叉验证 (16组) ===
# M11: blur_zoom(3)+noise_gauss(3)+comp_jpeg(3)
EXPS+=("M11d|AR_PARAMS_PATH=$DEG/M11_triple.json AR_VAL_PARAMS_PATH=$DEG/M11_triple.json")
EXPS+=("M11f|AR_PARAMS_PATH=$DEG/M11_triple.json AR_VAL_PARAMS_PATH=$DEG/M11_triple.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/M11_triple.json")
EXPS+=("M11cf|AR_PARAMS_PATH=$DEG/M11_triple.json AR_VAL_PARAMS_PATH=$DEG/M11_triple.json AR_CURRICULUM_CONFIG=0:$DEG/M11_fwd1.json,5000:$DEG/M11_fwd2.json,10000:$DEG/M11_triple.json")
EXPS+=("M11cr|AR_PARAMS_PATH=$DEG/M11_triple.json AR_VAL_PARAMS_PATH=$DEG/M11_triple.json AR_CURRICULUM_CONFIG=0:$DEG/M11_rev1.json,5000:$DEG/M11_rev2.json,10000:$DEG/M11_triple.json")
# M12: blur_glass(3)+noise_gauss(3)+comp_jpeg(3)
EXPS+=("M12d|AR_PARAMS_PATH=$DEG/M12_triple.json AR_VAL_PARAMS_PATH=$DEG/M12_triple.json")
EXPS+=("M12f|AR_PARAMS_PATH=$DEG/M12_triple.json AR_VAL_PARAMS_PATH=$DEG/M12_triple.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/M12_triple.json")
EXPS+=("M12cf|AR_PARAMS_PATH=$DEG/M12_triple.json AR_VAL_PARAMS_PATH=$DEG/M12_triple.json AR_CURRICULUM_CONFIG=0:$DEG/M12_fwd1.json,5000:$DEG/M12_fwd2.json,10000:$DEG/M12_triple.json")
EXPS+=("M12cr|AR_PARAMS_PATH=$DEG/M12_triple.json AR_VAL_PARAMS_PATH=$DEG/M12_triple.json AR_CURRICULUM_CONFIG=0:$DEG/M12_rev1.json,5000:$DEG/M12_rev2.json,10000:$DEG/M12_triple.json")
# M13: blur_motion(3)+noise_impulse(3)+comp_jpeg(3)
EXPS+=("M13d|AR_PARAMS_PATH=$DEG/M13_triple.json AR_VAL_PARAMS_PATH=$DEG/M13_triple.json")
EXPS+=("M13f|AR_PARAMS_PATH=$DEG/M13_triple.json AR_VAL_PARAMS_PATH=$DEG/M13_triple.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/M13_triple.json")
EXPS+=("M13cf|AR_PARAMS_PATH=$DEG/M13_triple.json AR_VAL_PARAMS_PATH=$DEG/M13_triple.json AR_CURRICULUM_CONFIG=0:$DEG/M13_fwd1.json,5000:$DEG/M13_fwd2.json,10000:$DEG/M13_triple.json")
EXPS+=("M13cr|AR_PARAMS_PATH=$DEG/M13_triple.json AR_VAL_PARAMS_PATH=$DEG/M13_triple.json AR_CURRICULUM_CONFIG=0:$DEG/M13_rev1.json,5000:$DEG/M13_rev2.json,10000:$DEG/M13_triple.json")
# M14: blur_lens(3)+noise_poisson(3)+comp_jpeg(3)
EXPS+=("M14d|AR_PARAMS_PATH=$DEG/M14_triple.json AR_VAL_PARAMS_PATH=$DEG/M14_triple.json")
EXPS+=("M14f|AR_PARAMS_PATH=$DEG/M14_triple.json AR_VAL_PARAMS_PATH=$DEG/M14_triple.json AR_CURRICULUM_CONFIG=0:random,7547:$DEG/M14_triple.json")
EXPS+=("M14cf|AR_PARAMS_PATH=$DEG/M14_triple.json AR_VAL_PARAMS_PATH=$DEG/M14_triple.json AR_CURRICULUM_CONFIG=0:$DEG/M14_fwd1.json,5000:$DEG/M14_fwd2.json,10000:$DEG/M14_triple.json")
EXPS+=("M14cr|AR_PARAMS_PATH=$DEG/M14_triple.json AR_VAL_PARAMS_PATH=$DEG/M14_triple.json AR_CURRICULUM_CONFIG=0:$DEG/M14_rev1.json,5000:$DEG/M14_rev2.json,10000:$DEG/M14_triple.json")

echo "=== Phase 7+8: ${#EXPS[@]} experiments ==="

for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
  SCRIPT="$LOG/gpu${gpu}_p78.sh"
  echo "#!/bin/bash" > "$SCRIPT"; chmod +x "$SCRIPT"
done

for ((i=0; i<${#EXPS[@]}; i++)); do
  gpu=$((i % NUM_GPUS))
  IFS='|' read -r name env_str <<< "${EXPS[$i]}"
  echo "echo \"=== [$name] GPU${gpu} start \$(date) ===\"" >> "$LOG/gpu${gpu}_p78.sh"
  echo "CUDA_VISIBLE_DEVICES=$gpu $BASE $env_str AR_CKPT_PREFIX=$EXP/$name uv run train.py > $LOG/${name}.log 2>&1" >> "$LOG/gpu${gpu}_p78.sh"
  echo "echo \"=== [$name] GPU${gpu} done \$(date) ===\"" >> "$LOG/gpu${gpu}_p78.sh"
done

for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
  n=$(grep -c "start" "$LOG/gpu${gpu}_p78.sh" 2>/dev/null || echo 0)
  echo "  GPU${gpu}: $n experiments"
  bash "$LOG/gpu${gpu}_p78.sh" &
done
wait
echo "Phase 7+8 done."
