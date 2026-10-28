#!/usr/bin/env python3
"""Select E6-A checkpoints by support PSNR (agent run-to-run, R0 only)."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
E6A = ROOT / "rebuttal_runs/E6_A"
SCENARIOS = ["blind_0006", "blind_0008", "blind_0010",
             "blind_0013", "blind_0017", "blind_0024"]
REPS = [1, 2, 3]


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def parse_log(p):
    txt = p.read_text(errors="ignore")
    m = re.search(r"=== EXP_META: (\{.*?\}) ===", txt, re.S)
    meta = json.loads(m.group(1)) if m else None
    m = re.search(r"Loaded best checkpoint: step (\d+) \(Set14 PSNR_RGB=([0-9.]+)\)", txt)
    if not m:
        return None, None, meta
    return int(m.group(1)), float(m.group(2)), meta


def main():
    rows = []
    for sid in SCENARIOS:
        for rep in REPS:
            name = f"{sid}_rep{rep}"
            log = E6A / "logs" / f"{name}.log"
            step, psnr, meta = parse_log(log) if log.exists() else (None, None, None)
            if step is None or meta is None:
                print(f"[warn] incomplete {name}")
                continue
            ck = E6A / "experiments" / name / "checkpoints" / f"{name}_step{step}.pt"
            if not ck.exists():
                print(f"[warn] missing {ck}")
                continue
            pred = json.loads((E6A / "challenges" / name / "predicted_params.json").read_text())
            rows.append({
                "run_id": f"E6A_{name}",
                "scenario_id": sid,
                "rep": rep,
                "controller": "E6A-agent",
                "round": "R0",
                "checkpoint": str(ck.relative_to(ROOT)),
                "checkpoint_sha256": sha_file(ck),
                "checkpoint_step": step,
                "support_psnr": psnr,
                "predicted_pipeline": [[s["function"], int(s["severity"])]
                                       for s in pred["pipeline"]],
                "predicted_verdict": pred.get("analysis", {}).get("verdict"),
                "predicted_ci": pred.get("analysis", {}).get("ci_pass_rate"),
                "exp_meta": meta,
            })
    out = E6A / "selection.jsonl"
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
