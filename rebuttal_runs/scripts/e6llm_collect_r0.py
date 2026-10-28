#!/usr/bin/env python3
"""Collect E6-LLM R0 support PSNR / best checkpoint / NaN status."""
import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
E = ROOT / "rebuttal_runs/E6_LLM"


def parse(name):
    log = E / "logs" / f"{name}_R0.log"
    if not log.exists():
        return None
    txt = log.read_text(errors="ignore")
    m = re.search(r"=== EXP_META: (\{.*?\}) ===", txt, re.S)
    meta = json.loads(m.group(1)) if m else None
    m = re.search(r"Loaded best checkpoint: step (\d+) \(Set14 PSNR_RGB=([0-9.]+|nan)\)", txt)
    step, psnr = (int(m.group(1)), m.group(2)) if m else (None, None)
    done = "Skipping final validation" in txt
    nan = "loss: nan" in txt or "PSNR_RGB=nan" in txt
    ck = None
    if step is not None:
        p = E / "runs" / name / "R0" / "checkpoints" / f"R0_step{step}.pt"
        if p.exists():
            ck = str(p.relative_to(ROOT))
    return {"name": name, "done": done, "nan": nan, "best_step": step,
            "support_psnr": None if psnr in (None, "nan") else float(psnr),
            "checkpoint": ck, "exp_meta": meta}


if __name__ == "__main__":
    rows = []
    for d in sorted((E / "challenges").iterdir()):
        if not (d / "predicted_params.json").exists():
            continue
        r = parse(d.name)
        if r:
            rows.append(r)
    json.dump(rows, open(E / "r0_collect.json", "w"), indent=2, ensure_ascii=False)
    print(f"{'name':24s} done nan best support_psnr")
    for r in rows:
        print(f"{r['name']:24s} {str(r['done']):5s} {str(r['nan']):5s} "
              f"{str(r['best_step']):>6s} {r['support_psnr']}")
