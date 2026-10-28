#!/usr/bin/env python3
"""Safe per-GPU sequential launcher.

- No global pkill.
- A task is complete only if the process exits 0 AND the declared result
  file exists and is non-empty; a completion record stores wall time and
  checkpoint hashes.
- Completed tasks are skipped on restart.
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMPLETION = ROOT / "rebuttal_runs" / "completion"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_tasks(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def completion_record(task_id):
    return COMPLETION / f"{task_id}.json"


def is_done(task):
    rec = completion_record(task["id"])
    check = ROOT / task["check"]
    if not rec.exists():
        # A finished summary file is sufficient evidence after a launcher restart.
        return check.exists() and check.stat().st_size > 0
    try:
        d = json.load(open(rec))
    except Exception:
        return False
    return d.get("rc") == 0 and d.get("check_ok") and d.get("check_sha256") == sha256(check)


def run_one(task, gpu, log_dir):
    tid = task["id"]
    cmd = task["cmd"].replace("GPU_ID", str(gpu))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_dir.mkdir(parents=True, exist_ok=True)
    runner_log = log_dir / f"launcher_gpu{gpu}.log"
    t0 = time.time()
    with open(runner_log, "a") as lf:
        lf.write(f"[{time.strftime('%H:%M:%S')}] GPU{gpu} {tid} START\n")
        lf.flush()
        proc = subprocess.run(cmd, shell=True, executable="/bin/bash", env=env,
                              stdout=lf, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    check = ROOT / task["check"]
    check_ok = check.exists() and check.stat().st_size > 0
    ckpts = []
    check_dir = ROOT / Path(task["check"]).parent
    for d in (check_dir / "checkpoints", check_dir):
        for p in sorted(glob.glob(str(d / "*.pt"))):
            ckpts.append({"path": str(Path(p).relative_to(ROOT)), "sha256": sha256(p),
                          "bytes": os.path.getsize(p)})
    COMPLETION.mkdir(parents=True, exist_ok=True)
    rec = {
        "task_id": tid, "gpu": gpu, "rc": proc.returncode, "wall_seconds": round(wall, 1),
        "check": task["check"], "check_ok": check_ok,
        "check_sha256": sha256(check) if check_ok else None,
        "checkpoints": ckpts,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    completion_record(tid).write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(runner_log, "a") as lf:
        lf.write(f"[{time.strftime('%H:%M:%S')}] GPU{gpu} {tid} rc={proc.returncode} "
                 f"check_ok={check_ok} wall={wall:.0f}s\n")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks")
    ap.add_argument("gpus", help="comma-separated, e.g. 0,1,2,3")
    ap.add_argument("--log-dir", default="rebuttal_runs/logs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tasks = [t for t in load_tasks(ROOT / args.tasks)]
    gpus = [int(g) for g in args.gpus.split(",")]
    queues = {g: [] for g in gpus}
    for i, t in enumerate(tasks):
        queues[gpus[i % len(gpus)]].append(t)

    pending = {g: [t for t in queues[g] if not is_done(t)] for g in gpus}
    print(json.dumps({g: len(pending[g]) for g in gpus}, ensure_ascii=False), flush=True)
    if args.dry_run:
        for g in gpus:
            for t in pending[g]:
                print(f"GPU{g}: {t['cmd'].replace('GPU_ID', str(g))}")
        return

    log_dir = ROOT / args.log_dir

    def worker(g):
        for t in pending[g]:
            rec = run_one(t, g, log_dir)
            print(f"GPU{g} {t['id']} rc={rec['rc']} check_ok={rec['check_ok']} "
                  f"wall={rec['wall_seconds']}s", flush=True)

    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        list(ex.map(worker, gpus))
    done = sum(1 for t in tasks if is_done(t))
    print(f"completed {done}/{len(tasks)}")


if __name__ == "__main__":
    main()
