#!/bin/bash
# Re-evaluate the 7 nondeterministic scenarios on one shared fixed-LQ realization.
set -u
cd /data/zyli/projects/autoresearch || exit 1
PY=.venv/bin/python3
OUT=rebuttal_runs/E6A_full24
EVAL=rebuttal_runs/scripts/eval_e6a_full24_fixed_lq.py

for kind in best last final; do
  case "$kind" in
    best) sel="$OUT/selection.jsonl" ;;
    last) sel="$OUT/selection_last.jsonl" ;;
    final) sel="$OUT/selection_final.jsonl" ;;
  esac
  rm -f "$OUT"/test_metrics_fixed_${kind}_gpu*.jsonl
  for g in 0 1 2 3 4 5 6 7; do
    $PY "$EVAL" --selection "$sel" --out "$OUT/test_metrics_fixed_${kind}_gpu$g.jsonl" \
      --gpu-id "$g" --num-gpus 8 > "$OUT/logs/eval_fixed_${kind}_gpu$g.log" 2>&1 &
  done
  wait
  echo "[$(date -u +%FT%TZ)] fixed-lq $kind done"
done
echo "[$(date -u +%FT%TZ)] E6A24_FIXED_LQ_DONE"
