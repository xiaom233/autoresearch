#!/bin/bash
# Train one E6-A full24 R0 specialist (fixed Direct Swin+SimpleGate config).
# Usage: bash e6a_full24_train_one.sh <name>
# Waits for that run's predicted_params.json, then trains with seed 42.
set -uo pipefail
NAME="${1:?usage: e6a_full24_train_one.sh <name>}"
ROOT=/data/zyli/projects/autoresearch
cd "$ROOT" || exit 1
OUT=rebuttal_runs/E6A_full24
P="$OUT/challenges/$NAME/predicted_params.json"
LOG="$OUT/logs/$NAME.log"
CKPT="$OUT/experiments/$NAME"
mkdir -p "$OUT/logs"

# Wait up to 6 h for the blind-ID output of this run.
for _ in $(seq 1 720); do
  [ -f "$P" ] && break
  sleep 30
done
if [ ! -f "$P" ]; then
  echo "[$(date -u +%FT%TZ)] $NAME: predicted_params timeout" >> "$LOG"
  exit 2
fi

# Validate pipeline before spending GPU time.
.venv/bin/python3 - "$P" <<'PY' || exit 3
import json, sys
d = json.load(open(sys.argv[1]))
p = d.get("pipeline")
assert isinstance(p, list) and p, "empty pipeline"
for s in p:
    assert isinstance(s, dict) and s.get("function"), s
    assert 1 <= int(s.get("severity", 0)) <= 8, s
PY

# Resume safety: skip only a completed run.
if ls "$CKPT"/checkpoints/*.pt >/dev/null 2>&1 && grep -q "Skipping final validation" "$LOG" 2>/dev/null; then
  echo "[$(date -u +%FT%TZ)] $NAME: already complete, skip" >> "$LOG"
  exit 0
fi

START=$(date -u +%s)
echo "[$(date -u +%FT%TZ)] $NAME: TRAIN START (GPU=${CUDA_VISIBLE_DEVICES:-unset})" >> "$LOG"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
AR_PARAMS_PATH="$P" \
AR_VAL_PARAMS_PATH="$P" \
AR_CHALLENGE_DIR="$OUT/challenges/$NAME" \
AR_EPOCH_BUDGET=2 AR_LEARNING_RATE=0.0005 AR_BATCH_SIZE=16 AR_LOSS_FN=l1 \
AR_SEED=42 AR_SKIP_FINAL_VAL=1 AR_ATTENTION_TYPE=swin AR_EMBED_DIM=64 \
AR_WINDOW_SIZE=8 AR_USE_SIMPLE_GATE=1 AR_USE_GCM=0 AR_CKPT_PREFIX="$CKPT" \
.venv/bin/python3 train.py >> "$LOG" 2>&1
RC=$?
END=$(date -u +%s)
echo "[$(date -u +%FT%TZ)] $NAME: TRAIN END rc=$RC wall=$((END-START))s" >> "$LOG"

.venv/bin/python3 - "$OUT/challenges/$NAME/timing.json" "$NAME" "$START" "$END" "$RC" <<'PY' || true
import json, sys, datetime
path, name, start, end, rc = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
rec = {
    "scenario_run": name,
    "train_start_utc": datetime.datetime.utcfromtimestamp(start).isoformat() + "Z",
    "train_end_utc": datetime.datetime.utcfromtimestamp(end).isoformat() + "Z",
    "train_seconds": end - start,
    "gpu_hours": round((end - start) / 3600.0, 4),
    "exit_code": rc,
}
json.dump(rec, open(path, "w"), indent=2)
PY

exit $RC
