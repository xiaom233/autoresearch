#!/usr/bin/env python3
"""Build the E1 manifest from finished rule-controller runs.

Includes standard R0/R1/R2 and corrected-mapping runs from E1_fix/, labelled
R0_fixed/R1_fixed so later selection can compare them on support PSNR.
"""
import hashlib
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
E1 = ROOT / "rebuttal_runs" / "E1"
E1FIX = ROOT / "rebuttal_runs" / "E1_fix"
CH = ROOT / "exp41" / "challenges" / "phase4"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def parse_meta(log):
    for line in open(log, errors="ignore"):
        if "EXP_META:" in line:
            raw = line.split("EXP_META: ", 1)[1].rstrip()
            if raw.endswith(" ==="):
                raw = raw[:-4]
            return json.loads(raw)
    return None


def parse_tail(log):
    out = {"training_seconds": None, "best_support_psnr": None}
    for line in open(log, errors="ignore"):
        m = re.search(r"training_seconds:\s+([0-9.]+)", line)
        if m:
            out["training_seconds"] = float(m.group(1))
        m = re.search(r"Loaded best checkpoint: step (\d+) \(Set14 PSNR_RGB=([0-9.]+)\)", line)
        if m:
            out["best_support_psnr"] = float(m.group(2))
    return out


def add_run(rows, sid, rnd_label, run, log, params):
    if not (run.is_dir() and log.exists() and params.exists()):
        return
    # Only include completed runs; prevents racing an in-flight fix task.
    if not (run / f"{run.name}_results.json").exists():
        return
    ckdir = run / "checkpoints"
    if not ckdir.is_dir():
        return
    pts = sorted(ckdir.glob("*.pt"),
                 key=lambda p: int(re.search(r"step(\d+)", p.name).group(1)))
    if not pts:
        return
    tail = parse_tail(log)
    rows.append({
        "scenario_id": sid, "round": rnd_label, "status": "ok",
        "run_dir": str(run.relative_to(ROOT)),
        "log": str(log.relative_to(ROOT)),
        "exp_meta": parse_meta(log),
        "params_path": str(params.relative_to(ROOT)),
        "params_sha256": sha256(params),
        "training_seconds": tail["training_seconds"],
        "best_support_psnr_logged": tail["best_support_psnr"],
        "checkpoints": [{"path": str(p.relative_to(ROOT)),
                         "step": int(re.search(r"step(\d+)", p.name).group(1)),
                         "sha256": sha256(p)} for p in pts],
        "support_pairs": [
            {"image_id": f"{sid}_{i}",
             "clean": str((CH / sid / f"clean_{i}.png").relative_to(ROOT)),
             "lq": str((CH / sid / f"degraded_{i}.png").relative_to(ROOT)),
             "clean_sha256": sha256(CH / sid / f"clean_{i}.png"),
             "lq_sha256": sha256(CH / sid / f"degraded_{i}.png")}
            for i in range(5)],
        "support_selection": "PENDING",
    })


def wait_for_fix_tasks(timeout=600):
    """Give in-flight E1_fix tasks a short grace period before building."""
    task_file = E1FIX / "tasks_all.jsonl"
    if not task_file.exists():
        return
    tasks = [json.loads(l) for l in open(task_file) if l.strip()]
    deadline = time.time() + timeout
    while time.time() < deadline:
        pending = [t["id"] for t in tasks if not (ROOT / t["check"]).exists()]
        if not pending:
            return
        print(f"[manifest] waiting for fix tasks: {pending}", flush=True)
        time.sleep(20)


def main():
    wait_for_fix_tasks()
    rows = []
    for sid in [f"blind_{i:04d}" for i in range(1, 25)]:
        for rnd in ("R0", "R1", "R2"):
            add_run(rows, sid, rnd,
                    E1 / "experiments" / f"{sid}_{rnd}",
                    E1 / "logs" / f"{sid}_{rnd}.log",
                    E1 / "params" / f"{sid}_{rnd}.json")
    if (E1FIX / "experiments").is_dir():
        for run in sorted((E1FIX / "experiments").glob("blind_*")):
            name = run.name
            if "_" not in name:
                continue
            sid, rnd = name.rsplit("_", 1)
            add_run(rows, sid, f"{rnd}_fixed", run,
                    E1FIX / "logs" / f"{name}.log",
                    E1FIX / "params" / f"{name}.json")
    with open(E1 / "manifest.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {E1 / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
