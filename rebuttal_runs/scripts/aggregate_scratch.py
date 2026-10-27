#!/usr/bin/env python3
"""Aggregate DFPIR-from-scratch vs AutoIR / DFPIR-ft on fixed LQ."""
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCR = ROOT / "rebuttal_runs/E2_scratch"
SUP = json.load(open(ROOT / "rebuttal_runs/A0/support_test_overlap.json"))["per_scenario_support_images"]


def load(pat):
    rows = []
    for p in glob.glob(str(ROOT / pat)):
        rows += [json.loads(l) for l in open(p) if l.strip()]
    return rows


def macro(rows, ctrl, exclude_support=False):
    g = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("controller") != ctrl:
            continue
        if exclude_support and r["dataset"] == "DIV2K" and r["image_id"] in SUP.get(r["scenario_id"], []):
            continue
        g[r["dataset"]][r["scenario_id"]].append(r)
    out = {}
    for ds in ("DIV2K", "LSDIR"):
        per = {}
        for sid, rs in g[ds].items():
            per[sid] = {"psnr": float(np.mean([x["psnr_rgb"] for x in rs])),
                        "ssim": float(np.mean([x["ssim_rgb"] for x in rs])),
                        "lpips": float(np.mean([x["lpips_vgg"] for x in rs])),
                        "n": len(rs)}
        out[ds] = {"n_scenarios": len(per),
                   "psnr": float(np.mean([v["psnr"] for v in per.values()])) if per else None,
                   "ssim": float(np.mean([v["ssim"] for v in per.values()])) if per else None,
                   "lpips": float(np.mean([v["lpips"] for v in per.values()])) if per else None,
                   "per_scenario": per}
    return out


def paired(a, b, ds="DIV2K"):
    A, B = a[ds]["per_scenario"], b[ds]["per_scenario"]
    ids = sorted(set(A) & set(B))
    d = np.array([B[s]["psnr"] - A[s]["psnr"] for s in ids])
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(d), (10000, len(d)))
    m = d[idx].mean(axis=1)
    return {"n": len(ids), "delta": float(d.mean()),
            "ci": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))],
            "wins": int((d > 0).sum())}


def main():
    auto = load("rebuttal_runs/A0/test_metrics/*.jsonl")
    ft = load("rebuttal_runs/E2/test_metrics_labeled.jsonl")
    scratch = load("rebuttal_runs/E2_scratch/test_metrics_gpu*.jsonl")
    controllers = ["AutoIR", "DFPIR-ft-best", "DFPIR-ft-last",
                   "DFPIR-scratch-best", "DFPIR-scratch-last"]
    rows = auto + ft + scratch
    full = {c: macro(rows, c) for c in controllers}
    sup = {c: macro(rows, c, True) for c in controllers}
    res = {"full": full, "support_excluded": sup,
           "paired_vs_AutoIR": {}, "paired_vs_ft_best": {},
           "n_rows": {c: sum(1 for r in rows if r.get("controller") == c) for c in controllers},
           "notes": ["24 scenarios; DIV2K 100 + LSDIR 250 fixed LQ.",
                     "scratch = random init, same T_s/DC*/support as E2-ft.",
                     "support_excluded removes each scenario's 5 DIV2K support images."]}
    for c in ["DFPIR-ft-best", "DFPIR-ft-last", "DFPIR-scratch-best", "DFPIR-scratch-last"]:
        if c in full:
            res["paired_vs_AutoIR"][c] = {ds: paired(full["AutoIR"], full[c], ds) for ds in ("DIV2K", "LSDIR")}
    if "DFPIR-ft-best" in full:
        for c in ["DFPIR-scratch-best", "DFPIR-scratch-last"]:
            if c in full:
                res["paired_vs_ft_best"][c] = {ds: paired(full["DFPIR-ft-best"], full[c], ds)
                                               for ds in ("DIV2K", "LSDIR")}
    (SCR / "scratch_summary.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    md = ["# DFPIR from scratch vs AutoIR / DFPIR-ft", ""]
    md += ["| Controller | DIV2K PSNR | LSDIR PSNR | DIV2K SSIM |", "|---|---:|---:|---:|"]
    for c in controllers:
        if c in full:
            md.append(f"| {c} | {full[c]['DIV2K']['psnr']:.3f} | {full[c]['LSDIR']['psnr']:.3f} | {full[c]['DIV2K']['ssim']:.4f} |")
    md += ["", "Paired vs AutoIR (scenario-level):", ""]
    for c, dsres in res["paired_vs_AutoIR"].items():
        for ds, v in dsres.items():
            md.append(f"- {c} {ds}: {v['delta']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}], wins {v['wins']}/{v['n']}")
    md += ["", "Paired vs DFPIR-ft best:", ""]
    for c, dsres in res["paired_vs_ft_best"].items():
        for ds, v in dsres.items():
            md.append(f"- {c} {ds}: {v['delta']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}], wins {v['wins']}/{v['n']}")
    (SCR / "scratch_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
