#!/bin/bash
# E1 training/evaluation stage (run after dirs exist).
set -uo pipefail
cd /data/zyli/projects/autoresearch || exit 1
LOG=rebuttal_runs/E1/stage.log
exec >>"$LOG" 2>&1
echo "E1 STAGE START $(date)"
mkdir -p rebuttal_runs/E1/logs rebuttal_runs/E1/experiments
PY=.venv/bin/python3

$PY rebuttal_runs/scripts/safe_launcher.py \
  rebuttal_runs/E1/tasks_R0.jsonl 0,1,2,3,4,5,6,7
echo "E1 R0 DONE $(date)"

$PY rebuttal_runs/scripts/rule_controller.py decide --round R0
if [ -s rebuttal_runs/E1/tasks_R1.jsonl ]; then
  $PY rebuttal_runs/scripts/safe_launcher.py \
    rebuttal_runs/E1/tasks_R1.jsonl 0,1,2,3,4,5,6,7
  echo "E1 R1 DONE $(date)"
  $PY rebuttal_runs/scripts/rule_controller.py decide --round R1
  if [ -s rebuttal_runs/E1/tasks_R2.jsonl ]; then
    $PY rebuttal_runs/scripts/safe_launcher.py \
      rebuttal_runs/E1/tasks_R2.jsonl 0,1,2,3,4,5,6,7
    echo "E1 R2 DONE $(date)"
  fi
fi

$PY rebuttal_runs/scripts/build_e1_manifest.py
mkdir -p rebuttal_runs/E1/support_metrics
for g in 0 1 2 3 4 5 6 7; do
  $PY rebuttal_runs/scripts/eval_checkpoint.py support \
    --manifest rebuttal_runs/E1/manifest.jsonl \
    --out rebuttal_runs/E1/support_metrics/support_gpu${g}.jsonl \
    --gpu-id $g --num-gpus 8 > rebuttal_runs/E1/support_metrics/support_gpu${g}.log 2>&1 &
done
wait
$PY rebuttal_runs/scripts/merge_e1_selection.py
mkdir -p rebuttal_runs/E1/test_metrics
for g in 0 1 2 3 4 5 6 7; do
  $PY rebuttal_runs/scripts/eval_checkpoint.py test \
    --selection rebuttal_runs/E1/test_selection.jsonl \
    --out rebuttal_runs/E1/test_metrics/rule_gpu${g}.jsonl \
    --gpu-id $g --num-gpus 8 > rebuttal_runs/E1/test_metrics/rule_gpu${g}.log 2>&1 &
done
wait
$PY rebuttal_runs/scripts/aggregate_results.py \
  --inputs 'rebuttal_runs/A0/test_metrics/*.jsonl' 'rebuttal_runs/E1/test_metrics/*.jsonl' \
  --out rebuttal_runs/E1/rule_vs_autoair.json > rebuttal_runs/E1/aggregate.log 2>&1 || true
echo "E1 STAGE COMPLETE $(date)" > rebuttal_runs/E1/STAGE_DONE
echo "E1 STAGE COMPLETE $(date)"
