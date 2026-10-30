#!/usr/bin/env python3
"""Select E6-A full24 checkpoints by support PSNR (best) or fixed last step."""
import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "rebuttal_runs/E6A_full24"
SCENARIOS = [f"blind_{i:04d}" for i in range(1, 25)]
REPS = [1, 2]


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def parse_log(p, kind="best"):
    txt = p.read_text(errors="ignore")
    if "TRAIN END" not in txt:
        return None, None, None
    m = re.search(r"=== EXP_META: (\{.*?\}) ===", txt, re.S)
    meta = json.loads(m.group(1)) if m else None
    if kind == "last":
        matches = re.findall(r"\[Ckpt @ step (\d+)/\d+\] Set14: PSNR_RGB=([0-9.]+|nan)", txt)
        if not matches:
            return None, None, meta
        step, psnr = matches[-1]
        return int(step), float(psnr) if psnr != "nan" else None, meta
    m = re.search(r"Loaded best checkpoint: step (\d+) \(Set14 PSNR_RGB=([0-9.]+|nan)\)", txt)
    if not m:
        return None, None, meta
    return int(m.group(1)), float(m.group(2)) if m.group(2) != "nan" else None, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["best", "last"], default="best")
    args = ap.parse_args()
    rows = []
    missing = []
    for sid in SCENARIOS:
        for rep in REPS:
            name = f"{sid}_rep{rep}"
            log = OUT / "logs" / f"{name}.log"
            step, psnr, meta = parse_log(log, args.kind) if log.exists() else (None, None, None)
            if step is None or meta is None:
                missing.append(name)
                continue
            ck = OUT / "experiments" / name / "checkpoints" / f"{name}_step{step}.pt"
            if not ck.exists():
                missing.append(f"{name}:missing {ck.name}")
                continue
            pred = json.loads((OUT / "challenges" / name / "predicted_params.json").read_text())
            rows.append({
                "run_id": f"E6A24_{name}",
                "scenario_id": sid,
                "rep": rep,
                "controller": "E6A-full24-agent",
                "round": "R0",
                "selection_kind": args.kind,
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
    out = OUT / ("selection_last.jsonl" if args.kind == "last" else "selection.jsonl")
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {out}")
    if missing:
        print(f"[warn] incomplete/missing ({len(missing)}): {missing}")


if __name__ == "__main__":
    main()
