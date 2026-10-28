#!/bin/bash
# Evaluate selected E6-LLM checkpoints on the fixed GT-LQ DIV2K/LSDIR test set.
set -u
cd /data/zyli/projects/autoresearch
PY=.venv/bin/python3
OUT=rebuttal_runs/E6_LLM

$PY rebuttal_runs/scripts/e6llm_select.py
rm -f "$OUT"/test_metrics_gpu*.jsonl
for g in 0 1 2 3 4 5 6 7; do
  $PY rebuttal_runs/scripts/eval_checkpoint.py test \
    --selection "$OUT/selection.jsonl" --out "$OUT/test_metrics_gpu$g.jsonl" \
    --datasets DIV2K,LSDIR --no-lpips --gpu-id $g --num-gpus 8 \
    > "$OUT/logs/eval_gpu$g.log" 2>&1 &
done
wait
$PY rebuttal_runs/scripts/aggregate_e6llm.py > "$OUT/logs/aggregate.log" 2>&1
echo "[$(date +%H:%M)] E6LLM_DONE"
