#!/usr/bin/env python3
"""Aggregate E6-B support-set stability results (T9)."""
import json
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
E6 = ROOT / "rebuttal_runs/E6"
SCENARIOS = ["blind_0006", "blind_0008", "blind_0010",
             "blind_0013", "blind_0017", "blind_0024"]


def load_rows():
    rows = []
    for p in glob.glob(str(E6 / "test_metrics_gpu*.jsonl")):
        rows += [json.loads(l) for l in open(p) if l.strip()]
    return rows


def pipe_key(p):
    if not p:
        return []
    if isinstance(p[0], dict):
        return [(x["function"], int(x["severity"])) for x in p]
    return [(x[0], int(x[1])) for x in p]


def type_key(p):
    if not p:
        return []
    if isinstance(p[0], dict):
        return [x["function"] for x in p]
    return [x[0] for x in p]


def main():
    sel = {r["run_id"]: r for r in map(json.loads, open(E6 / "selection.jsonl"))}
    rows = load_rows()
    by_run = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_run[r["run_id"]][r["dataset"]].append(r)

    out = {"scenarios": {}, "macro": {}, "n_metrics": len(rows),
           "selection": {}, "notes": [
               "Per scenario/support-set, metrics are macro averages over DIV2K 100 and LSDIR 250.",
               "Support-set spread = max-min across three sets; not a confidence interval.",
               "Selected pipeline is from the rule controller decisions; GT is only used for post-hoc match scoring."]}
    for k in [1, 2, 3]:
        base = E6 / "run" / f"set{k}"
        for sid in SCENARIOS:
            dec = json.load(open(base / "decisions" / f"{sid}.json"))
            run_id = f"E6_set{k}_{sid}"
            s = sel[run_id]
            rnd = s["round"].lower()
            pipe = dec.get(f"pipeline_{rnd.upper()}") or dec.get("pipeline_R0")
            gt = json.load(open(ROOT / f"exp41/degradation_gt/{sid}_params.json"))["pipeline"]
            rec = {"support_set": f"set{k}", "round": s["round"],
                   "support_psnr": s["support_psnr"],
                   "checkpoint": s["checkpoint"],
                   "pipeline": pipe,
                   "pipeline_exact_gt": pipe_key(pipe) == pipe_key(gt),
                   "pipeline_type_gt": type_key(pipe) == type_key(gt),
                   "step_count_match_gt": len(pipe) == len(gt),
                   "datasets": {}}
            for ds in ("DIV2K", "LSDIR"):
                rs = by_run[run_id][ds]
                rec["datasets"][ds] = {
                    "n": len(rs),
                    "psnr": float(np.mean([x["psnr_rgb"] for x in rs])),
                    "ssim": float(np.mean([x["ssim_rgb"] for x in rs])),
                    "lpips": float(np.mean([x["lpips_vgg"] for x in rs])),
                }
            out["scenarios"].setdefault(sid, {})[f"set{k}"] = rec

    mac = {"per_set": {}, "spread": {}}
    for ds in ("DIV2K", "LSDIR"):
        for k in [1, 2, 3]:
            vals = []
            for sid in SCENARIOS:
                vals.append(out["scenarios"][sid][f"set{k}"]["datasets"][ds]["psnr"])
            mac["per_set"][f"set{k}_{ds}"] = {
                "mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)),
                "min": float(np.min(vals)), "max": float(np.max(vals))}
        spreads = []
        for sid in SCENARIOS:
            v = [out["scenarios"][sid][f"set{k}"]["datasets"][ds]["psnr"] for k in [1, 2, 3]]
            spreads.append(max(v) - min(v))
        mac["spread"][ds] = {
            "mean_range": float(np.mean(spreads)), "max_range": float(np.max(spreads)),
            "per_scenario": {sid: float(max(out["scenarios"][sid][f"set{k}"]["datasets"][ds]["psnr"] for k in [1, 2, 3]) -
                                        min(out["scenarios"][sid][f"set{k}"]["datasets"][ds]["psnr"] for k in [1, 2, 3]))
                             for sid in SCENARIOS}}
    out["macro"] = mac
    out["selection"] = {
        "selected_rounds": {sid: {k: out["scenarios"][sid][f"set{k}"]["round"] for k in [1, 2, 3]} for sid in SCENARIOS},
        "exact_gt": {sid: {k: out["scenarios"][sid][f"set{k}"]["pipeline_exact_gt"] for k in [1, 2, 3]} for sid in SCENARIOS},
        "type_gt": {sid: {k: out["scenarios"][sid][f"set{k}"]["pipeline_type_gt"] for k in [1, 2, 3]} for sid in SCENARIOS},
        "step_match": {sid: {k: out["scenarios"][sid][f"set{k}"]["step_count_match_gt"] for k in [1, 2, 3]} for sid in SCENARIOS},
    }
    (E6 / "e6_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    md = ["# E6-B support-set stability (T9)", "",
          "Each scenario uses 3 disjoint DIV2K_train support sets of 5 pairs. Test is the fixed GT-LQ DIV2K/LSDIR set.", "",
          "| Scenario | Set1 PSNR | Set2 PSNR | Set3 PSNR | mean | SD | range | selected rounds |",
          "|---|---:|---:|---:|---:|---:|---:|---|"]
    for sid in SCENARIOS:
        vals = [out["scenarios"][sid][f"set{k}"]["datasets"]["DIV2K"]["psnr"] for k in [1, 2, 3]]
        rounds = "/".join(out["scenarios"][sid][f"set{k}"]["round"] for k in [1, 2, 3])
        md.append(f"| {sid} | " + " | ".join(f"{x:.3f}" for x in vals) +
                  f" | {np.mean(vals):.3f} | {np.std(vals, ddof=1):.3f} | {max(vals)-min(vals):.3f} | {rounds} |")
    md += ["", "Macro over six scenarios (mean of scenario means):", ""]
    for key, v in mac["per_set"].items():
        md.append(f"- {key}: {v['mean']:.3f} ± {v['sd']:.3f} dB (min {v['min']:.3f}, max {v['max']:.3f})")
    md += ["", "Support-set spread per dataset:", ""]
    for ds, v in mac["spread"].items():
        md.append(f"- {ds}: mean range {v['mean_range']:.3f} dB, max range {v['max_range']:.3f} dB")
    md += ["", "Selected pipeline exact GT match:", ""]
    for sid in SCENARIOS:
        md.append(f"- {sid}: " + ", ".join(f"set{k}={out['scenarios'][sid][f'set{k}']['pipeline_exact_gt']}" for k in [1, 2, 3]))
    (E6 / "e6_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:30]))
    print(f"\nwrote {E6/'e6_summary.json'}")


if __name__ == "__main__":
    main()
