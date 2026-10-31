#!/usr/bin/env python3
"""Aggregate E6-C support-size ablation: size1/size3 (E6C) vs size5 (E6-B setA).

All three sizes are nested: size1 subset size3 subset E6-B setA (5 images).
Macro = 24 scenarios equal weight; paired scenario bootstrap for size deltas.
"""
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
E6C = ROOT / "rebuttal_runs/E6C_support_size"
E6B = ROOT / "rebuttal_runs/E6B_full24"
SCEN = [f"blind_{i:04d}" for i in range(1, 25)]
SIZES = ["size1", "size3", "size5"]


def gt_steps(sid):
    return len(json.load(open(ROOT / f"exp41/degradation_gt/{sid}_params.json"))["pipeline"])


def load(pattern, key_fn):
    by = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(str(pattern))):
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            key = key_fn(r)
            if key is None:
                continue
            by[key][r["dataset"]].append(r["psnr_rgb"])
    return by


def main():
    c = load(E6C / "test_metrics_final_gpu*.jsonl",
             lambda r: (r["scenario_id"], r["support_set"]))
    missing_c = [(s, z) for s in SCEN for z in ("size1", "size3") if not c[(s, z)]["DIV2K"]]
    b = load(E6B / "test_metrics_final_gpu*.jsonl",
             lambda r: (r["scenario_id"], "size5") if r.get("support_set") == "setA" else None)
    missing_b = [s for s in SCEN if not b[(s, "size5")]["DIV2K"]]
    if missing_c or missing_b:
        raise SystemExit(f"missing metrics: E6C {missing_c}, E6B {missing_b}")

    table = {}
    for sid in SCEN:
        rec = {"gt_steps": gt_steps(sid), "sizes": {}}
        for z in SIZES:
            src = c if z != "size5" else b
            rec["sizes"][z] = {ds: float(np.mean(src[(sid, z)][ds])) for ds in ("DIV2K", "LSDIR")}
        table[sid] = rec

    out = {"note": "E6-C support-size ablation, nested size1 subset size3 subset E6-B setA(size5); "
                   "24 scenarios equal weight; hybrid fixed-LQ test.",
           "macro": {}, "paired": {}, "stratified_by_gt_steps": {}, "scenarios": table}

    def macro(z, ds):
        vals = [table[s]["sizes"][z][ds] for s in SCEN]
        return {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)),
                "median": float(np.median(vals)), "min": float(np.min(vals)),
                "max": float(np.max(vals))}

    for ds in ("DIV2K", "LSDIR"):
        out["macro"][ds] = {z: macro(z, ds) for z in SIZES}

    rng = np.random.default_rng(20261031)
    for ds in ("DIV2K", "LSDIR"):
        out["paired"][ds] = {}
        for a, bb in (("size3", "size1"), ("size5", "size3"), ("size5", "size1")):
            d = np.array([table[s]["sizes"][a][ds] - table[s]["sizes"][bb][ds] for s in SCEN])
            boots = [float(np.mean(d[rng.integers(0, len(d), len(d))])) for _ in range(10000)]
            out["paired"][ds][f"{a}_minus_{bb}"] = {
                "mean": float(np.mean(d)), "median": float(np.median(d)),
                "sd": float(np.std(d, ddof=1)),
                "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                "wins": int(np.sum(d > 0)), "n": len(d)}

    for gs in (1, 2, 3):
        group = [s for s in SCEN if table[s]["gt_steps"] == gs]
        out["stratified_by_gt_steps"][str(gs)] = {"n_scenarios": len(group)}
        for ds in ("DIV2K", "LSDIR"):
            out["stratified_by_gt_steps"][str(gs)][ds] = {
                z: float(np.mean([table[s]["sizes"][z][ds] for s in group])) for z in SIZES}

    (E6C / "e6c_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    lines = ["# E6-C support 数量消融（1 vs 3 vs 5）", "",
             "24 场景；嵌套 support（size1 ⊂ size3 ⊂ E6-B setA=size5）；固定 GT-LQ test；"
             "宏平均 = 24 场景等权。", "",
             "| 数据集 | size1 | size3 | size5 |", "|---|---:|---:|---:|"]
    for ds in ("DIV2K", "LSDIR"):
        m = out["macro"][ds]
        lines.append(f"| {ds} | {m['size1']['mean']:.3f} | {m['size3']['mean']:.3f} | {m['size5']['mean']:.3f} |")
    lines += ["", "## 配对差（场景级 10k bootstrap）", "",
              "| 数据集 | size3−size1 | size5−size3 | size5−size1 |", "|---|---:|---:|---:|"]
    for ds in ("DIV2K", "LSDIR"):
        p = out["paired"][ds]
        def fmt(k):
            x = p[k]
            return f"{x['mean']:+.3f} [{x['ci95'][0]:+.3f},{x['ci95'][1]:+.3f}] ({x['wins']}/24)"
        lines.append(f"| {ds} | {fmt('size3_minus_size1')} | {fmt('size5_minus_size3')} | {fmt('size5_minus_size1')} |")
    lines += ["", "## 分层（GT 步数）", ""]
    for gs in ("1", "2", "3"):
        s = out["stratified_by_gt_steps"][gs]
        lines.append(f"- {gs} 步（n={s['n_scenarios']}）：DIV2K {s['DIV2K']['size1']:.3f} / "
                     f"{s['DIV2K']['size3']:.3f} / {s['DIV2K']['size5']:.3f}；LSDIR "
                     f"{s['LSDIR']['size1']:.3f} / {s['LSDIR']['size3']:.3f} / {s['LSDIR']['size5']:.3f}")
    (E6C / "e6c_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
