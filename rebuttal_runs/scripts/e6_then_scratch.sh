#!/bin/bash
# Wait for E6-B R0, run E6-B R1/R2, then run DFPIR-from-scratch (24 scenarios).
set -u
cd /data/zyli/projects/autoresearch
SIDS=blind_0006,blind_0008,blind_0010,blind_0013,blind_0017,blind_0024
PY=.venv/bin/python3

wait_no_train() {
  while pgrep -f "rebuttal_runs/E6/run" >/dev/null 2>&1 || pgrep -f "rebuttal_runs/E2_scratch" >/dev/null 2>&1; do
    sleep 30
  done
}

launch_txt() {
  local txt=$1
  for g in 0 1 2 3 4 5 6 7; do
    nohup bash scripts/gpu_runner.sh "$g" "$txt" > "rebuttal_runs/logs/$(basename $txt .txt)_gpu${g}.out" 2>&1 &
  done
}

# Wait until the current R0 queue is drained and no R0 training remains.
while [ -s rebuttal_runs/E6/tasks_R0_all.txt ] || pgrep -f "rebuttal_runs/E6/run" >/dev/null 2>&1; do
  sleep 60
done
echo "R0_DONE $(date)" >> rebuttal_runs/E6/orchestrator.log

# R1
for k in 1 2 3; do
  E6_SUPPORT_ROOT=/data/zyli/projects/autoresearch/rebuttal_runs/E6/support/set$k \
  E6_OUT=/data/zyli/projects/autoresearch/rebuttal_runs/E6/run/set$k \
  E6_SCENARIOS=$SIDS $PY rebuttal_runs/scripts/e6_rule_controller.py decide --round R0 \
    >> rebuttal_runs/E6/run/set${k}_decide_R1.log 2>&1
done
: > rebuttal_runs/E6/tasks_R1_all.txt
for k in 1 2 3; do
  [ -f rebuttal_runs/E6/run/set$k/tasks_R1.jsonl ] || continue
  $PY - "$k" <<'PY' >> rebuttal_runs/E6/tasks_R1_all.txt
import json,sys
k=sys.argv[1]
for line in open(f'rebuttal_runs/E6/run/set{k}/tasks_R1.jsonl'):
    if line.strip():
        t=json.loads(line); print(f'{t["cmd"]}|E6_set{k}_{t["id"]}')
PY
done
if [ -s rebuttal_runs/E6/tasks_R1_all.txt ]; then
  rm -f rebuttal_runs/E6/tasks_R1_all.txt.lock
  launch_txt rebuttal_runs/E6/tasks_R1_all.txt
  while [ -s rebuttal_runs/E6/tasks_R1_all.txt ] || pgrep -f "rebuttal_runs/E6/run" >/dev/null 2>&1; do sleep 60; done
fi
echo "R1_DONE $(date)" >> rebuttal_runs/E6/orchestrator.log

# R2
for k in 1 2 3; do
  E6_SUPPORT_ROOT=/data/zyli/projects/autoresearch/rebuttal_runs/E6/support/set$k \
  E6_OUT=/data/zyli/projects/autoresearch/rebuttal_runs/E6/run/set$k \
  E6_SCENARIOS=$SIDS $PY rebuttal_runs/scripts/e6_rule_controller.py decide --round R1 \
    >> rebuttal_runs/E6/run/set${k}_decide_R2.log 2>&1
done
: > rebuttal_runs/E6/tasks_R2_all.txt
for k in 1 2 3; do
  [ -f rebuttal_runs/E6/run/set$k/tasks_R2.jsonl ] || continue
  $PY - "$k" <<'PY' >> rebuttal_runs/E6/tasks_R2_all.txt
import json,sys
k=sys.argv[1]
for line in open(f'rebuttal_runs/E6/run/set{k}/tasks_R2.jsonl'):
    if line.strip():
        t=json.loads(line); print(f'{t["cmd"]}|E6_set{k}_{t["id"]}')
PY
done
if [ -s rebuttal_runs/E6/tasks_R2_all.txt ]; then
  rm -f rebuttal_runs/E6/tasks_R2_all.txt.lock
  launch_txt rebuttal_runs/E6/tasks_R2_all.txt
  while [ -s rebuttal_runs/E6/tasks_R2_all.txt ] || pgrep -f "rebuttal_runs/E6/run" >/dev/null 2>&1; do sleep 60; done
fi
echo "R2_DONE $(date)" >> rebuttal_runs/E6/orchestrator.log

# DFPIR from scratch, same T_s / DC* / support as E2.
wait_no_train
rm -f rebuttal_runs/E2_scratch/tasks.txt.lock
launch_txt rebuttal_runs/E2_scratch/tasks.txt
echo "SCRATCH_STARTED $(date)" >> rebuttal_runs/E6/orchestrator.log
while [ -s rebuttal_runs/E2_scratch/tasks.txt ] || pgrep -f "rebuttal_runs/E2_scratch" >/dev/null 2>&1; do sleep 60; done
echo "SCRATCH_DONE $(date)" >> rebuttal_runs/E6/orchestrator.log
