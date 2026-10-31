#!/bin/bash
# Train one reflected E6-B round (R1/R2) if its pipeline changed.
# Usage: bash e6c_train_reflect.sh <name> <R1|R2>
set -uo pipefail
NAME="${1:?usage: e6c_train_reflect.sh <name> <R1|R2>}"
RND="${2:?usage: ... <name> <R1|R2>}"
ROOT=/data/zyli/projects/autoresearch
cd "$ROOT" || exit 1
OUT=rebuttal_runs/E6C_support_size
DIR="$OUT/challenges/$NAME"
P="$DIR/predicted_params_${RND}.json"
S="$DIR/training_strategy_${RND}.json"
if [ "$RND" = "R1" ]; then PREV="$DIR/predicted_params.json"; else PREV="$DIR/predicted_params_R1.json"; fi
LOG="$OUT/logs/${NAME}_${RND}.log"
CKPT="$OUT/experiments/${NAME}_${RND}"
mkdir -p "$OUT/logs"

for _ in $(seq 1 720); do
  [ -f "$P" ] && [ -f "$S" ] && break
  sleep 30
done
[ -f "$P" ] && [ -f "$S" ] || { echo "[$(date -u +%FT%TZ)] $NAME $RND: params/strategy timeout" >> "$LOG"; exit 2; }
.venv/bin/python3 - "$P" "$PREV" <<'PYNORM' || exit 5
import json, sys
for path in sys.argv[1:]:
    d = json.load(open(path)); ch = False; new = []
    for s in d.get("pipeline", []):
        if isinstance(s, dict): new.append(s)
        else:
            new.append({"function": s[0], "severity": int(s[1]), "verdict": s[2] if len(s) > 2 else "UNCERTAIN"}); ch = True
    if ch:
        d["pipeline"] = new; json.dump(d, open(path, "w"), ensure_ascii=False, indent=2)
PYNORM

.venv/bin/python3 - "$P" "$PREV" "$DIR/${NAME}_${RND}_action.json" "$NAME" "$RND" <<'PY'
import json, sys
cur = json.load(open(sys.argv[1])); prev = json.load(open(sys.argv[2]))
def key(p): return [(s["function"], int(s["severity"])) for s in p.get("pipeline", [])]
c, p = key(cur), key(prev)
if not c:
    rec = {"name": sys.argv[4], "round": sys.argv[5], "action": "invalid", "reason": "empty pipeline"}
elif c == p:
    rec = {"name": sys.argv[4], "round": sys.argv[5], "action": "skip",
           "decision": cur.get("decision", "NO_CHANGE"),
           "reason": "pipeline identical to previous round", "pipeline": [[a, b] for a, b in c]}
else:
    rec = {"name": sys.argv[4], "round": sys.argv[5], "action": "train",
           "previous_pipeline": [[a, b] for a, b in p], "pipeline": [[a, b] for a, b in c]}
json.dump(rec, open(sys.argv[3], "w"), indent=2, ensure_ascii=False)
PY
ACTION=$(.venv/bin/python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['action'])" "$DIR/${NAME}_${RND}_action.json" 2>/dev/null)
case "$ACTION" in
  skip) echo "[$(date -u +%FT%TZ)] $NAME $RND: NO_CHANGE, skip training" >> "$LOG"; exit 0 ;;
  train) ;;
  *) echo "[$(date -u +%FT%TZ)] $NAME $RND: invalid params" >> "$LOG"; exit 3 ;;
esac

ENVS=$(.venv/bin/python3 rebuttal_runs/scripts/e6b_full24_strategy_env.py "$S") || exit 4
eval "export $ENVS"
if ls "$CKPT"/checkpoints/*.pt >/dev/null 2>&1 && grep -q "TRAIN END" "$LOG" 2>/dev/null; then
  echo "[$(date -u +%FT%TZ)] $NAME $RND: already complete, skip" >> "$LOG"; exit 0
fi
START=$(date -u +%s)
echo "[$(date -u +%FT%TZ)] $NAME $RND: TRAIN START (GPU=${CUDA_VISIBLE_DEVICES:-unset})" >> "$LOG"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
AR_PARAMS_PATH="$P" AR_VAL_PARAMS_PATH="$P" AR_CHALLENGE_DIR="$DIR" \
AR_SEED=42 AR_SKIP_FINAL_VAL=1 AR_QUIET_PIPELINE=1 AR_CKPT_PREFIX="$CKPT" \
.venv/bin/python3 train.py >> "$LOG" 2>&1
RC=$?
END=$(date -u +%s)
echo "[$(date -u +%FT%TZ)] $NAME $RND: TRAIN END rc=$RC wall=$((END-START))s" >> "$LOG"
.venv/bin/python3 - "$DIR/timing_${RND}.json" "$NAME" "$RND" "$START" "$END" "$RC" <<'PY' || true
import json, sys, datetime
path, name, rnd, start, end, rc = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
json.dump({"scenario_set": name, "round": rnd,
           "train_start_utc": datetime.datetime.utcfromtimestamp(start).isoformat() + "Z",
           "train_end_utc": datetime.datetime.utcfromtimestamp(end).isoformat() + "Z",
           "train_seconds": end - start, "gpu_hours": round((end - start) / 3600.0, 4),
           "exit_code": rc}, open(path, "w"), indent=2)
PY
exit $RC
