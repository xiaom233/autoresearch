#!/bin/bash
# Wait for E4 rain rule_R0 training, then evaluate and aggregate all rain baselines.
set -u
cd /data/zyli/projects/autoresearch
PY=/home/zyli/anaconda3/envs/dfpir/bin/python
LOG=rebuttal_runs/E4/rain/logs/rule_R0.log
OUT=rebuttal_runs/E4/rain/full_eval

while true; do
  if tr '\r' '\n' < "$LOG" 2>/dev/null | grep -q "Skipping final validation"; then break; fi
  if tr '\r' '\n' < "$LOG" 2>/dev/null | grep -q "Traceback"; then
    echo "[$(date +%H:%M)] rule_R0 training failed"; exit 1
  fi
  sleep 120
done

step=$(tr '\r' '\n' < "$LOG" | grep -o "Loaded best checkpoint: step [0-9]*" | grep -o "[0-9]*")
echo "[$(date +%H:%M)] rule_R0 best step=$step"
CK=rebuttal_runs/E4/rain/rule_R0/checkpoints/rule_R0_step${step}.pt
ls -l "$CK" || exit 1

CUDA_VISIBLE_DEVICES=0 $PY rebuttal_runs/scripts/eval_paired.py \
  --pairs-dir datasets/E4/rain/test --model-kind restorenet \
  --checkpoint "$CK" --meta rebuttal_runs/E4/meta_swin_sg.json \
  --controller E4-rain-rule --scenario rain \
  --out "$OUT/E4-rain-rule.jsonl" --tile 512 --overlap 32 --gpu-id 0 \
  > "$OUT/E4-rain-rule.log" 2>&1

.venv/bin/python3 rebuttal_runs/scripts/aggregate_e4_rain_full.py \
  > rebuttal_runs/E4/rain/full_eval/aggregate.log 2>&1
echo "[$(date +%H:%M)] RAIN_DONE"
