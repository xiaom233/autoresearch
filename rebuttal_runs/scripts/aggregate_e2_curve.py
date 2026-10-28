#!/usr/bin/env python3
"""Aggregate E2 T1/T2/T3 snapshot test metrics into a budget curve."""
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load(pattern):
    out = []
    for p in glob.glob(pattern):
        out += [json.loads(l) for l in open(p) if l.strip()]
    return out


def mean_by(rows):
    g = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        g[r["controller"]][r["scenario_id"]][r["dataset"]].append(r)
    out = {}
    for ctrl, by_s in g.items():
        per = {}
        for ds in ("DIV2K", "LSDIR"):
            vals = []
            for sid, by_ds in by_s.items():
                rs = by_ds.get(ds, [])
                if rs:
                    vals.append(float(np.mean([x["psnr_rgb"] for x in rs])))
            per[ds] = {
                "n": len(vals),
                "psnr": float(np.mean(vals)) if vals else None,
                "ssim": float(np.mean([np.mean([x["ssim_rgb"] for x in by_s[sid][ds]])
                                      for sid in by_s if ds in by_s[sid]])) if vals else None,
                "lpips": float(np.mean([np.mean([x["lpips_vgg"] for x in by_s[sid][ds]])
                                       for sid in by_s if ds in by_s[sid]])) if vals else None,
            }
        out[ctrl] = per
    return out


def main():
    snap = load(str(ROOT / "rebuttal_runs/E2/test_metrics_snapshots/*.jsonl"))
    base = load(str(ROOT / "rebuttal_runs/A0/test_metrics/*.jsonl"))
    e2 = load(str(ROOT / "rebuttal_runs/E2/test_metrics_labeled.jsonl"))
    out = {
        "snapshots": mean_by(snap),
        "autoair": mean_by(base).get("AutoIR"),
        "dfpir_original": mean_by(e2).get("DFPIR-original"),
        "dfpir_ft_best": mean_by(e2).get("DFPIR-ft-best"),
        "dfpir_ft_last": mean_by(e2).get("DFPIR-ft-last"),
        "notes": ["T1/T2/T3 are snapshots of one continuous DFPIR-ft run per scenario.",
                  "Macro average over 24 scenarios; test LQ fixed."],
    }
    (ROOT / "rebuttal_runs/E2/budget_curve.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
