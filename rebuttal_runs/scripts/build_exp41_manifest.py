#!/usr/bin/env python3
"""A0: build a traceable manifest for exp41 R0-R2.

Reads only predicted params, training logs, checkpoints and support pairs.
Does not read degradation_gt / .gt_mappings / test results.
"""
import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "exp41"
OUT_DIR = ROOT / "rebuttal_runs" / "A0"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_meta(log_path):
    with open(log_path, errors="ignore") as f:
        for line in f:
            if "EXP_META:" in line:
                raw = line.split("EXP_META: ", 1)[1].rstrip()
                if raw.endswith(" ==="):
                    raw = raw[:-4]
                return json.loads(raw)
    return None


def parse_tail(log_path):
    out = {"training_seconds": None, "best_step": None, "best_psnr": None,
           "num_steps": None, "params_line": None, "paired_val": None,
           "quick_val": None}
    with open(log_path, errors="ignore") as f:
        for line in f:
            s = line.strip()
            m = re.match(r"training_seconds:\s+([0-9.]+)", s)
            if m:
                out["training_seconds"] = float(m.group(1))
            m = re.match(r"best_step:\s+(\d+)", s)
            if m:
                out["best_step"] = int(m.group(1))
            m = re.match(r"num_steps:\s+(\d+)", s)
            if m:
                out["num_steps"] = int(m.group(1))
            m = re.search(r"Loaded best checkpoint: step (\d+) \(Set14 PSNR_RGB=([0-9.]+)\)", s)
            if m:
                out["best_step"] = int(m.group(1))
                out["best_psnr"] = float(m.group(2))
            m = re.match(r"Params:\s+(.+)", s)
            if m:
                out["params_line"] = m.group(1).strip()
            m = re.match(r"PairedValDataset:\s+(\d+) image pairs from (.+)", s)
            if m:
                out["paired_val"] = {"pairs": int(m.group(1)), "dir": m.group(2).strip()}
            m = re.match(r"ValDataset:\s+(\d+) images from (\d+) dir", s)
            if m:
                out["quick_val"] = {"images": int(m.group(1)), "dirs": int(m.group(2))}
    return out


def list_checkpoints(run_dir):
    ck = run_dir / "checkpoints"
    out = []
    if not ck.is_dir():
        return out
    for p in sorted(ck.glob("*.pt"), key=lambda x: int(re.search(r"step(\d+)", x.name).group(1))):
        m = re.search(r"step(\d+)", p.name)
        out.append({"path": str(p.relative_to(ROOT)), "step": int(m.group(1)),
                    "sha256": sha256(p), "bytes": p.stat().st_size})
    return out


def support_pairs(bid):
    d = EXP / "challenges" / "phase4" / bid
    pairs = []
    for i in range(5):
        c = d / f"clean_{i}.png"
        lq = d / f"degraded_{i}.png"
        if not (c.exists() and lq.exists()):
            return None
        pairs.append({"image_id": f"{bid}_{i}",
                      "clean": str(c.relative_to(ROOT)),
                      "lq": str(lq.relative_to(ROOT)),
                      "clean_sha256": sha256(c),
                      "lq_sha256": sha256(lq)})
    return pairs


def resolve_params(meta, bid, round_name):
    p = ROOT / meta["params_path"]
    if not p.exists():
        # fall back to the per-round predicted params in challenge dir
        alt = EXP / "challenges" / "phase4" / bid / ("predicted_params.json" if round_name == "R0"
                                                     else f"predicted_params_{round_name}.json")
        return alt if alt.exists() else p
    return p


def pipeline_steps(params_path):
    with open(params_path) as f:
        d = json.load(f)
    return len(d.get("pipeline", []))


rows = []
snapshot = []
for i in range(1, 25):
    bid = f"blind_{i:04d}"
    pairs = support_pairs(bid)
    for r in ("R0", "R1", "R2"):
        log = EXP / "logs" / f"exp41_{r}_{bid}.log"
        run_dir = EXP / "experiments" / f"exp41_{r}_{bid}"
        if not log.exists():
            rows.append({"scenario_id": bid, "round": r, "status": "MISSING_LOG"})
            continue
        is_link = run_dir.is_symlink()
        real_dir = run_dir.resolve() if is_link else run_dir
        meta_log = log
        if is_link:
            # A symlinked round reuses the source run: take its architecture
            # and do not count its training time twice.
            meta_log = EXP / "logs" / f"{real_dir.name}.log"
        meta = parse_meta(meta_log) if meta_log.exists() else parse_meta(log)
        tail = parse_tail(meta_log) if meta_log.exists() else parse_tail(log)
        params = resolve_params(meta, bid, r) if meta else None
        ckpts = list_checkpoints(real_dir)
        row = {
            "scenario_id": bid,
            "round": r,
            "status": "ok" if meta and ckpts else "INCOMPLETE",
            "run_dir": str(run_dir.relative_to(ROOT)),
            "run_dir_is_symlink": is_link,
            "run_dir_resolved": str(real_dir.relative_to(ROOT)),
            "log": str(log.relative_to(ROOT)),
            "log_sha256": sha256(log),
            "exp_meta": meta,
            "params_path": str(params.relative_to(ROOT)) if params else None,
            "params_sha256": sha256(params) if params and params.exists() else None,
            "params_steps": pipeline_steps(params) if params and params.exists() else None,
            "training_seconds": None if is_link else tail["training_seconds"],
            "is_duplicate_training": bool(is_link),
            "best_step_logged": tail["best_step"],
            "best_set14_psnr": tail["best_psnr"],
            "num_steps": tail["num_steps"],
            "quick_val": tail["quick_val"],
            "paired_val": tail["paired_val"],
            "checkpoints": ckpts,
            "support_pairs": pairs,
            "support_pair_count": len(pairs) if pairs else 0,
            "support_selection": "PENDING",
            "test_evaluated": False,
            "notes": "",
        }
        if r == "R0" and ":R1:" in str(params):
            row["notes"] += "R0 params unexpected"
        rows.append(row)

with open(OUT_DIR / "exp41_selection_manifest.jsonl", "w") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

summary = {
    "total_rows": len(rows),
    "status_counts": {},
    "by_round": {},
    "training_seconds": {},
    "symlinks": [],
    "missing_support": [],
}
for row in rows:
    summary["status_counts"][row.get("status")] = summary["status_counts"].get(row.get("status"), 0) + 1
    r = row.get("round")
    d = summary["by_round"].setdefault(r, {"n": 0, "ok": 0, "seconds": [], "checkpoints": 0})
    d["n"] += 1
    if row.get("status") == "ok":
        d["ok"] += 1
        d["checkpoints"] += len(row.get("checkpoints", []))
        if row.get("training_seconds"):
            d["seconds"].append(row["training_seconds"])
    if row.get("run_dir_is_symlink"):
        summary["symlinks"].append({"scenario_id": row["scenario_id"], "round": r,
                                    "target": row.get("run_dir_resolved")})
    if row.get("support_pair_count") != 5:
        summary["missing_support"].append({"scenario_id": row["scenario_id"], "round": r})
for r, d in summary["by_round"].items():
    secs = d.pop("seconds")
    d["seconds_sum"] = round(sum(secs), 1) if secs else None
    d["seconds_mean"] = round(sum(secs) / len(secs), 1) if secs else None
summary["exp41_r0_r2_total_gpu_seconds"] = round(
    sum(row.get("training_seconds") or 0 for row in rows), 1)
summary["exp41_r0_r2_total_gpu_hours"] = round(
    sum(row.get("training_seconds") or 0 for row in rows) / 3600, 2)
summary["git_head"] = os.popen("git rev-parse HEAD").read().strip()
summary["git_dirty"] = os.popen("git status --porcelain").read().strip().splitlines()
with open(OUT_DIR / "exp41_manifest_summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(json.dumps(summary, indent=2, ensure_ascii=False))
