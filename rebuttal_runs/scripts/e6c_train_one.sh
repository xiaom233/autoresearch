#!/bin/bash
# Train one E6-B full24 R0 specialist using its strategy JSON.
set -uo pipefail
NAME="${1:?usage: e6c_train_one.sh <name>}"
ROOT=/data/zyli/projects/autoresearch
cd "$ROOT" || exit 1
OUT=rebuttal_runs/E6C_support_size
DIR="$OUT/challenges/$NAME"
P="$DIR/predicted_params.json"
S="$DIR/training_strategy.json"
LOG="$OUT/logs/${NAME}.log"
CKPT="$OUT/experiments/$NAME"
mkdir -p "$OUT/logs"

for _ in $(seq 1 720); do
  [ -f "$P" ] && [ -f "$S" ] && break
  sleep 30
done
[ -f "$P" ] && [ -f "$S" ] || { echo "[$(date -u +%FT%TZ)] $NAME: params/strategy timeout" >> "$LOG"; exit 2; }

.venv/bin/python3 - "$P" <<'PY' || exit 3
import json, sys
d = json.load(open(sys.argv[1]))
p = d.get("pipeline")
assert isinstance(p, list) and p, "empty pipeline"
for s in p:
    assert isinstance(s, dict) and s.get("function"), s
    assert 1 <= int(s.get("severity", 0)) <= 8, s
PY
ENVS=$(.venv/bin/python3 rebuttal_runs/scripts/e6b_full24_strategy_env.py "$S") || exit 4
eval "export $ENVS"

if ls "$CKPT"/checkpoints/*.pt >/dev/null 2>&1 && grep -q "TRAIN END" "$LOG" 2>/dev/null; then
  echo "[$(date -u +%FT%TZ)] $NAME: already complete, skip" >> "$LOG"
  exit 0
fi

START=$(date -u +%s)
echo "[$(date -u +%FT%TZ)] $NAME: TRAIN START (GPU=${CUDA_VISIBLE_DEVICES:-unset})" >> "$LOG"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
AR_PARAMS_PATH="$P" AR_VAL_PARAMS_PATH="$P" AR_CHALLENGE_DIR="$DIR" \
AR_SEED=42 AR_SKIP_FINAL_VAL=1 AR_QUIET_PIPELINE=1 \
AR_CKPT_PREFIX="$CKPT" \
.venv/bin/python3 train.py >> "$LOG" 2>&1
RC=$?
END=$(date -u +%s)
echo "[$(date -u +%FT%TZ)] $NAME: TRAIN END rc=$RC wall=$((END-START))s" >> "$LOG"

.venv/bin/python3 - "$DIR/timing.json" "$NAME" "$START" "$END" "$RC" <<'PY' || true
import json, sys, datetime
path, name, start, end, rc = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
json.dump({"scenario_set": name,
           "train_start_utc": datetime.datetime.utcfromtimestamp(start).isoformat() + "Z",
           "train_end_utc": datetime.datetime.utcfromtimestamp(end).isoformat() + "Z",
           "train_seconds": end - start, "gpu_hours": round((end - start) / 3600.0, 4),
           "exit_code": rc}, open(path, "w"), indent=2)
PY
exit $RC
