#!/usr/bin/env python3
"""Audit which DC/RS/CLR timings can be recovered from existing exp41 logs."""
import glob
import json
import re
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "rebuttal_runs/E7/existing_timing_coverage.json"


def birth_mtime(p):
    try:
        w, m = subprocess.check_output(["stat", "-c", "%W %Y", str(p)], text=True).split()
        return int(w), int(m)
    except Exception:
        st = Path(p).stat()
        return int(st.st_ctime), int(st.st_mtime)


def audit_training():
    out = {}
    for r in ["R0", "R1", "R2"]:
        vals = []
        for f in glob.glob(str(ROOT / f"exp41/logs/exp41_{r}_blind_*.log")):
            t = open(f, errors="ignore").read()
            m = re.search(r"training_seconds:\s+([0-9.]+)", t)
            if m:
                vals.append(float(m.group(1)))
        out[r] = {"n": len(vals),
                  "mean_s": float(np.mean(vals)) if vals else None,
                  "sum_s": float(np.sum(vals)) if vals else None,
                  "min_s": float(np.min(vals)) if vals else None,
                  "max_s": float(np.max(vals)) if vals else None,
                  "source": "EXP_META start + training_seconds"}
    allv = [v["sum_s"] for v in out.values() if v["sum_s"]]
    out["total_s"] = float(sum(allv))
    out["total_gpu_h"] = float(sum(allv) / 3600)
    out["per_scenario_gpu_h"] = float(sum(allv) / 3600 / 24)
    return out


def audit_reeval():
    out = {}
    for r in ["R0", "R1", "R2"]:
        rows = []
        for f in glob.glob(str(ROOT / f"exp41/logs/reeval_{r}_gpu*.log")):
            b, m = birth_mtime(f)
            n = len(re.findall(r"^\[\d+\] R\d+_blind_", open(f, errors="ignore").read(), re.M))
            if b > 0 and n:
                rows.append({"file": Path(f).name, "tasks": n,
                             "span_min": (m - b) / 60,
                             "per_task_min": (m - b) / 60 / n})
        out[r] = {"gpus": len(rows),
                  "total_tasks": int(sum(x["tasks"] for x in rows)),
                  "mean_per_task_min": float(np.mean([x["per_task_min"] for x in rows])) if rows else None,
                  "rows": rows}
    return out


def audit_blind_id():
    deltas = []
    for d in sorted(glob.glob(str(ROOT / "exp40/challenges/phase4/blind_*"))):
        d = Path(d)
        a, b = d / "degraded.png", d / "predicted_params.json"
        if a.exists() and b.exists():
            deltas.append((b.stat().st_mtime - a.stat().st_mtime) / 60)
    return {"n": len(deltas),
            "mean_min": float(np.mean(deltas)) if deltas else None,
            "median_min": float(np.median(deltas)) if deltas else None,
            "min_min": float(np.min(deltas)) if deltas else None,
            "max_min": float(np.max(deltas)) if deltas else None,
            "source": "exp40 degraded.png birth/mtime -> predicted_params.json birth; batch wall-clock proxy"}


def audit_reflection():
    p = ROOT / "exp41/challenges/phase4/blind_0001/reflection.json"
    d = json.load(open(p))
    stamps = {k: d[k].get("timestamp") for k in ["r1_summary", "r2_summary", "r3_summary", "r4_summary"]
              if k in d}
    return {"timestamps": stamps,
            "has_clock_time": any("T" in str(v) and ":" in str(v) for v in stamps.values()),
            "note": "r*_summary timestamps are date-only; no start/end or duration"}


def audit_strategy():
    files = glob.glob(str(ROOT / "exp41/challenges/phase4/blind_*/training_strategy.json"))
    internal = 0
    for f in files:
        d = json.load(open(f))
        internal += int(any("time" in str(k).lower() for k in d))
    return {"files": len(files), "files_with_internal_timestamp": internal,
            "note": "no agent start/end; exp41 mtimes are copy/rewrite times"}


def main():
    out = {"training": audit_training(),
           "support_eval": audit_reeval(),
           "blind_id_proxy": audit_blind_id(),
           "reflection": audit_reflection(),
           "strategy": audit_strategy(),
           "missing": ["LLM API start/end/tokens/cost", "DC candidate simulation time",
                       "strategy agent time", "R1/R2 reflection start/end",
                       "final test start/end", "offline development dedup manifest"],
           "note": "Only training and support-eval are recoverable; the rest is UNKNOWN. Do not backfill/fabricate."}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
