#!/usr/bin/env python3
"""Queue E6-C NO_CHANGE R1 skips (deduplicated) and copy R0 strategy to R1."""
import glob
import json
from pathlib import Path

OUT = Path("rebuttal_runs/E6C_support_size")
QUEUE = OUT / "queue.txt"
queued = set(QUEUE.read_text().splitlines()) if QUEUE.exists() else set()
added = []
for f in sorted(glob.glob(str(OUT / "challenges/*/r1_decision.json"))):
    d = Path(f).parent
    n = d.name
    if json.load(open(f))["decision"] != "NO_CHANGE":
        continue
    s = d / "training_strategy_R1.json"
    if not s.exists():
        s.write_text((d / "training_strategy.json").read_text())
    line = f"bash rebuttal_runs/scripts/e6c_train_reflect.sh {n} R1|{n}_R1"
    if not (OUT / "logs" / f"{n}_R1.log").exists() and line not in queued:
        added.append(line)
        queued.add(line)
if added:
    with open(QUEUE, "a") as fh:
        fh.write("\n".join(added) + "\n")
print(f"added {len(added)}; queue={len(queued)}")
