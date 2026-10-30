#!/usr/bin/env python3
"""Append reflected-round training tasks to the live gpu_runner queue (deduplicated)."""
import fcntl
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUEUE = ROOT / "exp40/scripts/.exp_launcher_tasks.wzWpyE"


def main():
    if (len(sys.argv) - 1) % 2:
        sys.exit("usage: e6a_full24_enqueue.py <name> <R1|R2> [...]")
    pairs = list(zip(sys.argv[1::2], sys.argv[2::2]))
    lock_path = Path(str(QUEUE) + ".lock")
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        existing = QUEUE.read_text() if QUEUE.exists() else ""
        added = []
        with open(QUEUE, "a") as fq:
            for name, rnd in pairs:
                tag = f"{name}_{rnd}"
                if tag in existing:
                    print(f"[skip] {tag} already queued")
                    continue
                cmd = (f"bash rebuttal_runs/scripts/e6a_full24_train_reflect.sh "
                       f"{name} {rnd}")
                fq.write(f"{cmd}|{tag}\n")
                added.append(tag)
        fcntl.flock(lf, fcntl.LOCK_UN)
    print(f"appended {len(added)}: {added}")


if __name__ == "__main__":
    main()
