#!/bin/bash
# Wait for all 18 E6-A trainings, select best checkpoints, then test on GT-LQ.
set -u
cd /data/zyli/projects/autoresearch
PY=.venv/bin/python3
OUT=rebuttal_runs/E6_A

while true; do
  n=$(ls "$OUT"/logs/blind_*_rep*.log 2>/dev/null | wc -l)
  done_n=$(grep -l "Skipping final validation" "$OUT"/logs/blind_*_rep*.log 2>/dev/null | wc -l)
  echo "[$(date +%H:%M)] logs=$n finished=$done_n"
  if [ "$done_n" -ge 18 ]; then break; fi
  sleep 120
done

$PY rebuttal_runs/scripts/e6a_select.py
rm -f "$OUT"/test_metrics_gpu*.jsonl
for g in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g $PY rebuttal_runs/scripts/eval_checkpoint.py test \
    --selection "$OUT/selection.jsonl" --out "$OUT/test_metrics_gpu$g.jsonl" \
    --datasets DIV2K,LSDIR --no-lpips --gpu-id $g --num-gpus 8 \
    > "$OUT/logs/eval_gpu$g.log" 2>&1 &
done
wait
$PY rebuttal_runs/scripts/aggregate_e6a.py > "$OUT/aggregate.log" 2>&1
echo "[$(date +%H:%M)] E6A_DONE"
