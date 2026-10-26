#!/usr/bin/env python3
"""Merge the fixed-LQ re-evaluation of 7 scenarios with the 17 deterministic ones."""
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "rebuttal_runs/E1E2_fixed_lq"
SIDS = sorted(p.name for p in FIX.glob("blind_*") if (p / "index.json").exists())
SUP = json.load(open(ROOT / "rebuttal_runs/A0/support_test_overlap.json"))[
    "per_scenario_support_images"]
CAL = {f"{i:04d}" for i in range(801, 811)}


def load(pattern):
    rows = []
    for p in glob.glob(str(ROOT / pattern)):
        rows += [json.loads(l) for l in open(p) if l.strip()]
    return rows


def fixed_rows():
    rows = []
    for sid in SIDS:
        p = FIX / sid / "metrics.jsonl"
        rows += [json.loads(l) for l in open(p) if l.strip()]
    return rows


def replace(rows, fixed, ctrl):
    keep = [r for r in rows if r["scenario_id"] not in SIDS]
    add = [r for r in fixed if r["controller"] == ctrl]
    return keep + add


def macro(rows, ctrl, exclude_support=False, exclude_cal=False):
    g = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("controller") != ctrl:
            continue
        if r["dataset"] == "DIV2K":
            if exclude_support and r["image_id"] in SUP.get(r["scenario_id"], []):
                continue
            if exclude_cal and r["image_id"] in CAL:
                continue
        g[r["dataset"]][r["scenario_id"]].append(r)
    out = {}
    for ds in ("DIV2K", "LSDIR"):
        vals, per = [], {}
        for sid, rs in g[ds].items():
            v = float(np.mean([x["psnr_rgb"] for x in rs]))
            vals.append(v)
            per[sid] = v
        out[ds] = {"n": len(vals), "psnr": float(np.mean(vals)) if vals else None,
                   "per_scenario": per}
    return out


def paired(base, other, ds="DIV2K", seed=42, n=10000):
    a, b = base[ds]["per_scenario"], other[ds]["per_scenario"]
    ids = sorted(set(a) & set(b))
    d = np.array([b[s] - a[s] for s in ids])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(axis=1)
    return {"n": len(ids), "delta": float(d.mean()),
            "ci": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))],
            "wins": int((d > 0).sum())}


def main():
    fixed = fixed_rows()
    a0 = load("rebuttal_runs/A0/test_metrics/*.jsonl")
    e1 = load("rebuttal_runs/E1/test_metrics/*.jsonl")
    e2 = load("rebuttal_runs/E2/test_metrics_labeled.jsonl")
    snaps = load("rebuttal_runs/E2/test_metrics_snapshots/*.jsonl")

    auto = replace(a0, fixed, "AutoIR")
    rule = replace(e1, fixed, "rule")
    auto_full = macro(auto, "AutoIR")
    rule_full = macro(rule, "rule")
    auto_sup = macro(auto, "AutoIR", True)
    rule_sup = macro(rule, "rule", True)
    auto_both = macro(auto, "AutoIR", True, True)
    rule_both = macro(rule, "rule", True, True)

    res = {"fixed_scenarios": SIDS,
           "e1": {
               "full": {"AutoIR": auto_full, "rule": rule_full},
               "support_excluded": {"AutoIR": auto_sup, "rule": rule_sup},
               "support_calib_excluded": {"AutoIR": auto_both, "rule": rule_both},
               "paired_full": {ds: paired(auto_full, rule_full, ds) for ds in ("DIV2K", "LSDIR")},
               "paired_support_excluded": {ds: paired(auto_sup, rule_sup, ds) for ds in ("DIV2K", "LSDIR")},
           }}

    e2_full = {}
    e2_sup = {}
    for ctrl in ["DFPIR-original", "DFPIR-ft-best", "DFPIR-ft-last"]:
        rows = replace(e2, fixed, ctrl)
        e2_full[ctrl] = macro(rows, ctrl)
        e2_sup[ctrl] = macro(rows, ctrl, True)
    e2_full["AutoIR"] = auto_full
    e2_sup["AutoIR"] = auto_sup
    budget = {}
    for ctrl in ["DFPIR-ft-T1", "DFPIR-ft-T2", "DFPIR-ft-T3"]:
        rows = replace(snaps, fixed, ctrl)
        budget[ctrl] = {"full": macro(rows, ctrl), "support_excluded": macro(rows, ctrl, True)}
    res["e2"] = {
        "full": e2_full, "support_excluded": e2_sup,
        "paired_full": {c: {ds: paired(auto_full, e2_full[c], ds)
                            for ds in ("DIV2K", "LSDIR")}
                        for c in ["DFPIR-original", "DFPIR-ft-best", "DFPIR-ft-last"]},
        "paired_support_excluded": {c: {ds: paired(auto_sup, e2_sup[c], ds)
                                       for ds in ("DIV2K", "LSDIR")}
                                   for c in ["DFPIR-original", "DFPIR-ft-best", "DFPIR-ft-last"]},
        "budget": budget,
    }

    # per-scenario comparison for the 7 fixed scenarios: old vs new
    per = {}
    for sid in SIDS:
        per[sid] = {}
        for ctrl in ["AutoIR", "rule", "DFPIR-original", "DFPIR-ft-best", "DFPIR-ft-last"]:
            old = a0 if ctrl == "AutoIR" else e1 if ctrl == "rule" else e2
            oldv = [r["psnr_rgb"] for r in old
                    if r["scenario_id"] == sid and r["controller"] == ctrl]
            newv = [r["psnr_rgb"] for r in fixed
                    if r["scenario_id"] == sid and r["controller"] == ctrl]
            per[sid][ctrl] = {"old": float(np.mean(oldv)) if oldv else None,
                              "new": float(np.mean(newv)) if newv else None}
    res["per_scenario_old_vs_new"] = per

    out = ROOT / "rebuttal_runs/E1E2_fixed_lq_summary.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    print("E1 macro (full / support-excluded):")
    for ctrl in ["AutoIR", "rule"]:
        f = res["e1"]["full"][ctrl]["DIV2K"]["psnr"]
        s = res["e1"]["support_excluded"][ctrl]["DIV2K"]["psnr"]
        print(f"  {ctrl:6s} DIV2K {f:.3f} / {s:.3f}   LSDIR {res['e1']['full'][ctrl]['LSDIR']['psnr']:.3f}")
    for ds, p in res["e1"]["paired_full"].items():
        print(f"  rule-AutoIR {ds}: {p['delta']:+.3f} [{p['ci'][0]:+.3f},{p['ci'][1]:+.3f}] wins {p['wins']}/{p['n']}")
    print("E2 macro (full / support-excluded):")
    for ctrl in ["DFPIR-original", "DFPIR-ft-best", "DFPIR-ft-last"]:
        f = res["e2"]["full"][ctrl]["DIV2K"]["psnr"]
        s = res["e2"]["support_excluded"][ctrl]["DIV2K"]["psnr"]
        p = res["e2"]["paired_support_excluded"][ctrl]["DIV2K"]
        print(f"  {ctrl:16s} DIV2K {f:.3f} / {s:.3f}   LSDIR {res['e2']['full'][ctrl]['LSDIR']['psnr']:.3f} "
              f"  vs AutoIR(sup-excl) {p['delta']:+.3f} [{p['ci'][0]:+.3f},{p['ci'][1]:+.3f}]")
    print("Budget curve (full DIV2K / LSDIR):")
    for ctrl in ["DFPIR-ft-T1", "DFPIR-ft-T2", "DFPIR-ft-T3"]:
        b = res["e2"]["budget"][ctrl]["full"]
        print(f"  {ctrl:14s} {b['DIV2K']['psnr']:.3f} / {b['LSDIR']['psnr']:.3f}")


if __name__ == "__main__":
    main()
