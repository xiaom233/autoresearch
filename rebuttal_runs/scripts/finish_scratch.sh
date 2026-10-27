#!/bin/bash
# Wait for E2-scratch training, then evaluate best/last and aggregate.
set -u
cd /data/zyli/projects/autoresearch

while [ -s rebuttal_runs/E2_scratch/tasks_retry.txt ] || pgrep -f "rebuttal_runs/E2_scratch" >/dev/null 2>&1; do
  sleep 60
done
echo "SCRATCH_TRAIN_DONE $(date)" >> rebuttal_runs/E2_scratch/finish.log

.venv/bin/python3 rebuttal_runs/scripts/build_scratch_selection.py >> rebuttal_runs/E2_scratch/finish.log 2>&1
wc -l rebuttal_runs/E2_scratch/selection.jsonl >> rebuttal_runs/E2_scratch/finish.log 2>&1

for g in 0 1 2 3 4 5 6 7; do
  nohup /home/zyli/anaconda3/envs/dfpir/bin/python rebuttal_runs/scripts/eval_dfpir.py \
    --selection rebuttal_runs/E2_scratch/selection.jsonl \
    --out rebuttal_runs/E2_scratch/test_metrics_gpu${g}.jsonl \
    --gpu-id $g --num-gpus 8 \
    > rebuttal_runs/E2_scratch/eval_gpu${g}.log 2>&1 &
done
wait
echo "SCRATCH_EVAL_DONE $(date)" >> rebuttal_runs/E2_scratch/finish.log
.venv/bin/python3 rebuttal_runs/scripts/aggregate_scratch.py >> rebuttal_runs/E2_scratch/finish.log 2>&1
echo "SCRATCH_AGG_DONE $(date)" >> rebuttal_runs/E2_scratch/finish.log
