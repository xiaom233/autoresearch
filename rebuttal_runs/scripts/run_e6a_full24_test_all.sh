#!/bin/bash
# E6-A full24: evaluate R0 best (GPU 0-3), R0 last (GPU 4-7), then final R0/R1/R2 (8 GPU).
set -u
cd /data/zyli/projects/autoresearch || exit 1
PY=.venv/bin/python3
OUT=rebuttal_runs/E6A_full24
EVAL=rebuttal_runs/scripts/eval_checkpoint.py

for f in selection.jsonl selection_last.jsonl selection_final.jsonl; do
  [ -f "$OUT/$f" ] || { echo "missing $OUT/$f"; exit 1; }
done

# best (R0-only) on physical GPUs 0-3 (all GPUs visible; --device-id picks physical)
rm -f "$OUT"/test_metrics_gpu*.jsonl
for g in 0 1 2 3; do
  $PY "$EVAL" test --selection "$OUT/selection.jsonl" \
    --out "$OUT/test_metrics_gpu$g.jsonl" --datasets DIV2K,LSDIR --no-lpips \
    --gpu-id "$g" --device-id "$g" --num-gpus 4 \
    > "$OUT/logs/eval_best_gpu$g.log" 2>&1 &
done

# last (R0-only, fixed step) on physical GPUs 4-7, partition index 0-3
rm -f "$OUT"/test_metrics_last_gpu*.jsonl
for g in 0 1 2 3; do
  $PY "$EVAL" test --selection "$OUT/selection_last.jsonl" \
    --out "$OUT/test_metrics_last_gpu$((g+4)).jsonl" --datasets DIV2K,LSDIR --no-lpips \
    --gpu-id "$g" --device-id "$((g+4))" --num-gpus 4 \
    > "$OUT/logs/eval_last_gpu$((g+4)).log" 2>&1 &
done
wait
echo "[$(date -u +%FT%TZ)] eval best+last done"

# final (cross-round) on 8 GPUs
rm -f "$OUT"/test_metrics_final_gpu*.jsonl
for g in 0 1 2 3 4 5 6 7; do
  $PY "$EVAL" test --selection "$OUT/selection_final.jsonl" \
    --out "$OUT/test_metrics_final_gpu$g.jsonl" --datasets DIV2K,LSDIR --no-lpips \
    --gpu-id "$g" --device-id "$g" --num-gpus 8 \
    > "$OUT/logs/eval_final_gpu$g.log" 2>&1 &
done
wait
echo "[$(date -u +%FT%TZ)] eval final done"

$PY rebuttal_runs/scripts/aggregate_e6a_full24.py > "$OUT/logs/aggregate.log" 2>&1
echo "[$(date -u +%FT%TZ)] E6A24_TEST_ALL_DONE"
