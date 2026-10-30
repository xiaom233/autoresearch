#!/bin/bash
# Train one reflected round (R1/R2) if its pipeline changed from the previous round.
# Usage: bash e6a_full24_train_reflect.sh <name> <R1|R2>
set -uo pipefail
NAME="${1:?usage: e6a_full24_train_reflect.sh <name> <R1|R2>}"
RND="${2:?usage: ... <name> <R1|R2>}"
ROOT=/data/zyli/projects/autoresearch
cd "$ROOT" || exit 1
OUT=rebuttal_runs/E6A_full24
DIR="$OUT/challenges/$NAME"
P="$DIR/predicted_params_${RND}.json"
if [ "$RND" = "R1" ]; then PREV="$DIR/predicted_params.json"; else PREV="$DIR/predicted_params_R1.json"; fi
LOG="$OUT/logs/${NAME}_${RND}.log"
CKPT="$OUT/experiments/${NAME}_${RND}"
mkdir -p "$OUT/logs"

for _ in $(seq 1 720); do
  [ -f "$P" ] && break
  sleep 30
done
if [ ! -f "$P" ]; then
  echo "[$(date -u +%FT%TZ)] $NAME $RND: params timeout" >> "$LOG"
  exit 2
fi

ACTION_FILE="$DIR/${NAME}_${RND}_action.json"
.venv/bin/python3 - "$P" "$PREV" "$ACTION_FILE" "$NAME" "$RND" <<'PY'
import json, sys
cur = json.load(open(sys.argv[1]))
prev = json.load(open(sys.argv[2]))
out = sys.argv[3]
def key(p):
    return [(s["function"], int(s["severity"])) for s in p.get("pipeline", [])]
c, p = key(cur), key(prev)
if not c:
    rec = {"name": sys.argv[4], "round": sys.argv[5], "action": "invalid",
           "reason": "empty pipeline"}
elif c == p:
    rec = {"name": sys.argv[4], "round": sys.argv[5], "action": "skip",
           "decision": cur.get("decision", "NO_CHANGE"),
           "reason": "pipeline identical to previous round",
           "pipeline": [[a, b] for a, b in c]}
else:
    rec = {"name": sys.argv[4], "round": sys.argv[5], "action": "train",
           "previous_pipeline": [[a, b] for a, b in p],
           "pipeline": [[a, b] for a, b in c]}
json.dump(rec, open(out, "w"), indent=2, ensure_ascii=False)
PY
ACTION=$(.venv/bin/python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['action'])" "$ACTION_FILE" 2>/dev/null)
case "$ACTION" in
  skip) echo "[$(date -u +%FT%TZ)] $NAME $RND: NO_CHANGE, skip training" >> "$LOG"; exit 0 ;;
  train) ;;  # pipeline changed -> train
  *) echo "[$(date -u +%FT%TZ)] $NAME $RND: invalid params" >> "$LOG"; exit 3 ;;
esac

if ls "$CKPT"/checkpoints/*.pt >/dev/null 2>&1 && grep -q "TRAIN END" "$LOG" 2>/dev/null; then
  echo "[$(date -u +%FT%TZ)] $NAME $RND: already complete, skip" >> "$LOG"
  exit 0
fi

START=$(date -u +%s)
echo "[$(date -u +%FT%TZ)] $NAME $RND: TRAIN START (GPU=${CUDA_VISIBLE_DEVICES:-unset})" >> "$LOG"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
AR_PARAMS_PATH="$P" \
AR_VAL_PARAMS_PATH="$P" \
AR_CHALLENGE_DIR="$DIR" \
AR_EPOCH_BUDGET=2 AR_LEARNING_RATE=0.0005 AR_BATCH_SIZE=16 AR_LOSS_FN=l1 \
AR_SEED=42 AR_SKIP_FINAL_VAL=1 AR_ATTENTION_TYPE=swin AR_EMBED_DIM=64 \
AR_WINDOW_SIZE=8 AR_USE_SIMPLE_GATE=1 AR_USE_GCM=0 AR_CKPT_PREFIX="$CKPT" \
.venv/bin/python3 train.py >> "$LOG" 2>&1
RC=$?
END=$(date -u +%s)
echo "[$(date -u +%FT%TZ)] $NAME $RND: TRAIN END rc=$RC wall=$((END-START))s" >> "$LOG"

.venv/bin/python3 - "$DIR/timing_${RND}.json" "$NAME" "$RND" "$START" "$END" "$RC" <<'PY' || true
import json, sys, datetime
path, name, rnd, start, end, rc = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
rec = {"scenario_run": name, "round": rnd,
       "train_start_utc": datetime.datetime.utcfromtimestamp(start).isoformat() + "Z",
       "train_end_utc": datetime.datetime.utcfromtimestamp(end).isoformat() + "Z",
       "train_seconds": end - start, "gpu_hours": round((end - start) / 3600.0, 4),
       "exit_code": rc}
json.dump(rec, open(path, "w"), indent=2)
PY
exit $RC
