#!/bin/bash
# Record wall-clock/cost metadata for a command.
# usage: timing_wrapper.sh OUT.jsonl STAGE SCENARIO REP ROUND -- command ...
set -u
OUT=${1:?out}
STAGE=${2:?stage}
SID=${3:?scenario}
REP=${4:?rep}
RND=${5:?round}
shift 5
[ "${1:-}" = "--" ] && shift
START_UTC=$(date -u +%FT%TZ)
START_EPOCH=$(date +%s.%N)
START_NS=$(date +%s%N)
"$@"
RC=$?
END_UTC=$(date -u +%FT%TZ)
END_EPOCH=$(date +%s.%N)
END_NS=$(date +%s%N)
WALL=$(awk -v a="$START_EPOCH" -v b="$END_EPOCH" 'BEGIN{printf "%.3f", b-a}')
CMD_JSON=$(printf '%s ' "$@" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))')
HOST=$(hostname)
LINE=$(python3 - "$OUT" "$STAGE" "$SID" "$REP" "$RND" "$START_UTC" "$END_UTC" "$WALL" "$RC" "$CMD_JSON" "$HOST" <<'PY'
import json,sys
out,stage,sid,rep,rnd,start,end,wall,rc,cmd,host=sys.argv[1:]
rec={"stage":stage,"scenario_id":sid,"rep":rep,"round":rnd,
     "start_utc":start,"end_utc":end,"wall_seconds":float(wall),
     "exit_code":int(rc),"command":json.loads(cmd),"host":host,
     "note":"wrapper timing; GPU training seconds must come from train.py training_seconds"}
print(json.dumps(rec,ensure_ascii=False))
PY
)
echo "$LINE" >> "$OUT"
echo "$LINE"
exit "$RC"
