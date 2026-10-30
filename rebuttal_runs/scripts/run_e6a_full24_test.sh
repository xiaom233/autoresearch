#!/bin/bash
# Evaluate E6-A full24 selected checkpoints on fixed GT-LQ DIV2K/LSDIR.
# Requires 48-row selection files (run e6a_full24_select.py first).
set -u
cd /data/zyli/projects/autoresearch || exit 1
PY=.venv/bin/python3
OUT=rebuttal_runs/E6A_full24

for kind in best last; do
  if [ "$kind" = "best" ]; then
    sel="$OUT/selection.jsonl"; suffix=""
  else
    sel="$OUT/selection_last.jsonl"; suffix="_last"
  fi
  if [ ! -f "$sel" ]; then
    echo "skip $kind: no $sel"
    continue
  fi
  n=$(wc -l < "$sel")
  if [ "$n" -lt 48 ]; then
    echo "skip $kind: only $n rows"
    continue
  fi
  rm -f "$OUT"/test_metrics${suffix}_gpu*.jsonl
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g $PY rebuttal_runs/scripts/eval_checkpoint.py test \
      --selection "$sel" --out "$OUT/test_metrics${suffix}_gpu$g.jsonl" \
      --datasets DIV2K,LSDIR --no-lpips --gpu-id "$g" --num-gpus 8 \
      > "$OUT/logs/eval_${kind}_gpu$g.log" 2>&1 &
  done
  wait
  echo "[$(date -u +%FT%TZ)] eval $kind done"
done

# final = best checkpoint across R0/R1/R2 (requires selection_final.jsonl)
if [ -f "$OUT/selection_final.jsonl" ]; then
  n=$(wc -l < "$OUT/selection_final.jsonl")
  if [ "$n" -ge 48 ]; then
    rm -f "$OUT"/test_metrics_final_gpu*.jsonl
    for g in 0 1 2 3 4 5 6 7; do
      CUDA_VISIBLE_DEVICES=$g $PY rebuttal_runs/scripts/eval_checkpoint.py test \
        --selection "$OUT/selection_final.jsonl" --out "$OUT/test_metrics_final_gpu$g.jsonl" \
        --datasets DIV2K,LSDIR --no-lpips --gpu-id "$g" --num-gpus 8 \
        > "$OUT/logs/eval_final_gpu$g.log" 2>&1 &
    done
    wait
    echo "[$(date -u +%FT%TZ)] eval final done"
  else
    echo "skip final: only $n rows"
  fi
fi

$PY rebuttal_runs/scripts/aggregate_e6a_full24.py > "$OUT/logs/aggregate.log" 2>&1
echo "[$(date -u +%FT%TZ)] E6A24_TEST_DONE"
