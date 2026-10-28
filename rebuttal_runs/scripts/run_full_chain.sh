#!/bin/bash
# Unattended chain: wait for E2, evaluate E2, then train/evaluate E1.
set -uo pipefail
cd /data/zyli/projects/autoresearch || exit 1
LOG=rebuttal_runs/chain.log
exec >>"$LOG" 2>&1
echo "CHAIN START $(date)"

PY=.venv/bin/python3
DFPIR_PY=/home/zyli/anaconda3/envs/dfpir/bin/python

# 1) Wait for the E2 launcher to finish.
while pgrep -f '[s]afe_launcher.py rebuttal_runs/E2/tasks.jsonl' >/dev/null; do
  sleep 300
done
echo "E2 TRAINING DONE $(date)"

# 2) Evaluate E2 best/last + DFPIR zero-shot on the fixed test LQ.
$PY rebuttal_runs/scripts/build_e2_selection.py
mkdir -p rebuttal_runs/E2/test_metrics
for g in 0 1 2 3 4 5 6 7; do
  $DFPIR_PY rebuttal_runs/scripts/eval_dfpir.py \
    --selection rebuttal_runs/E2/test_selection.jsonl \
    --out rebuttal_runs/E2/test_metrics/dfpir_gpu${g}.jsonl \
    --gpu-id $g --num-gpus 8 > rebuttal_runs/E2/test_metrics/dfpir_gpu${g}.log 2>&1 &
done
wait
echo "E2 EVAL DONE $(date)"
$PY rebuttal_runs/scripts/aggregate_results.py \
  --inputs 'rebuttal_runs/A0/test_metrics/*.jsonl' 'rebuttal_runs/E2/test_metrics/*.jsonl' \
  --out rebuttal_runs/E2/e2_vs_autoair.json > rebuttal_runs/E2/aggregate.log 2>&1 || true

# 3) E1 R0 training.
$PY rebuttal_runs/scripts/safe_launcher.py \
  rebuttal_runs/E1/tasks_R0.jsonl 0,1,2,3,4,5,6,7
echo "E1 R0 DONE $(date)"

# 4) Decide and run R1 / R2.
$PY rebuttal_runs/scripts/rule_controller.py decide --round R0
if [ -s rebuttal_runs/E1/tasks_R1.jsonl ]; then
  $PY rebuttal_runs/scripts/safe_launcher.py \
    rebuttal_runs/E1/tasks_R1.jsonl 0,1,2,3,4,5,6,7
  $PY rebuttal_runs/scripts/rule_controller.py decide --round R1
  if [ -s rebuttal_runs/E1/tasks_R2.jsonl ]; then
    $PY rebuttal_runs/scripts/safe_launcher.py \
      rebuttal_runs/E1/tasks_R2.jsonl 0,1,2,3,4,5,6,7
  fi
fi
echo "E1 TRAINING DONE $(date)"

# 5) Support selection and fixed-LQ test evaluation for E1.
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
echo "CHAIN DONE $(date)" > rebuttal_runs/CHAIN_DONE
echo "CHAIN COMPLETE $(date)"
