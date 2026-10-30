#!/usr/bin/env python3
"""Update E1's LLM/AutoIR comparison with the recent E6-A full24 final checkpoints.

Keeps the original E1/A0 files untouched; writes new e1_newllm_* files.
LQ: E1 rule/AutoIR use fixed_lq_revalidation values; E6-A uses fixed-LQ values
for the 7 nondeterministic scenarios and main values for the other 17.
"""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
E1 = ROOT / "rebuttal_runs/E1"
E6A = ROOT / "rebuttal_runs/E6A_full24"
SCEN = [f"blind_{i:04d}" for i in range(1, 25)]
FIXED7 = {"blind_0003", "blind_0005", "blind_0017", "blind_0018",
          "blind_0020", "blind_0022", "blind_0023"}


def gt_steps(sid):
    return len(json.load(open(ROOT / f"exp41/degradation_gt/{sid}_params.json"))["pipeline"])


def main():
    e1 = json.load(open(E1 / "e1_final_summary.json"))
    fixed = e1["fixed_lq_revalidation"]["data"]["full"]
    rule = fixed["rule"]
    auto = fixed["AutoIR"]
    e6 = json.load(open(E6A / "e6a_full24_summary.json"))["final"]["scenarios"]
    sens = json.load(open(E6A / "fixed_lq_sensitivity.json"))["final"]

    rows, table = [], {}
    for sid in SCEN:
        rec = {"gt_steps": gt_steps(sid)}
        for ds in ("DIV2K", "LSDIR"):
            rec[f"rule_{ds}"] = rule[ds]["per_scenario"][sid]
            rec[f"auto_old_{ds}"] = auto[ds]["per_scenario"][sid]
            vals = {}
            for rep in ("rep1", "rep2"):
                if sid in FIXED7:
                    v = sens[sid][rep][ds]["fixed"]
                else:
                    v = e6[sid]["reps"][rep][ds]
                vals[rep] = v
            rec[f"new_rep1_{ds}"] = vals["rep1"]
            rec[f"new_rep2_{ds}"] = vals["rep2"]
            rec[f"new_mean_{ds}"] = (vals["rep1"] + vals["rep2"]) / 2
        table[sid] = rec
        rows.append({"scenario": sid, "gt_steps": rec["gt_steps"],
                     "rule_DIV2K": rec["rule_DIV2K"], "auto_old_DIV2K": rec["auto_old_DIV2K"],
                     "new_rep1_DIV2K": rec["new_rep1_DIV2K"],
                     "new_rep2_DIV2K": rec["new_rep2_DIV2K"],
                     "new_mean_DIV2K": rec["new_mean_DIV2K"],
                     "rule_LSDIR": rec["rule_LSDIR"], "auto_old_LSDIR": rec["auto_old_LSDIR"],
                     "new_rep1_LSDIR": rec["new_rep1_LSDIR"],
                     "new_rep2_LSDIR": rec["new_rep2_LSDIR"],
                     "new_mean_LSDIR": rec["new_mean_LSDIR"]})

    def macro(key, ds):
        return float(np.mean([table[s][f"{key}_{ds}"] for s in SCEN]))

    def paired(newkey, oldkey, ds):
        d = np.array([table[s][f"{newkey}_{ds}"] - table[s][f"{oldkey}_{ds}"] for s in SCEN])
        rng = np.random.default_rng(20261030)
        boots = [float(np.mean(d[rng.integers(0, len(d), len(d))])) for _ in range(10000)]
        return {"mean_delta": float(np.mean(d)), "median_delta": float(np.median(d)),
                "sd": float(np.std(d, ddof=1)),
                "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                "wins": int(sum(d > 0)), "n": len(d)}

    out = {"note": "E1 new LLM comparison: E6-A full24 final checkpoints (new Skill agents, "
                   "Oct 2026) vs E1 rule; old A0/exp41 AutoIR retained as historical. "
                   "E6-A uses fixed Direct Swin+SG, R0/R1 selected by support PSNR; "
                   "E1 rule uses rule controller R0-R2. LQ: fixed values used where available.",
           "macro": {}, "paired": {}, "scenarios": table}
    for ds in ("DIV2K", "LSDIR"):
        out["macro"][ds] = {k: macro(k, ds) for k in
                            ("rule", "auto_old", "new_rep1", "new_rep2", "new_mean")}
        for newk in ("new_rep1", "new_rep2", "new_mean"):
            out["paired"][f"{newk}_minus_rule_{ds}"] = paired(newk, "rule", ds)
            out["paired"][f"{newk}_minus_auto_old_{ds}"] = paired(newk, "auto_old", ds)
        out["paired"][f"new_mean_minus_new_rep1_{ds}"] = paired("new_mean", "new_rep1", ds)
    out["stratified_by_gt_steps"] = {}
    for gs in (1, 2, 3):
        group = [s for s in SCEN if table[s]["gt_steps"] == gs]
        out["stratified_by_gt_steps"][str(gs)] = {"n_scenarios": len(group)}
        for ds in ("DIV2K", "LSDIR"):
            out["stratified_by_gt_steps"][str(gs)][ds] = {
                k: float(np.mean([table[s][f"{k}_{ds}"] for s in group]))
                for k in ("rule", "auto_old", "new_rep1", "new_rep2", "new_mean")}
    (E1 / "e1_newllm_summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(E1 / "e1_newllm_scenarios.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    lines = ["# E1 新 LLM 对照（E6-A full24 final checkpoints）", "",
             "E1 rule 为规则控制器；旧 AutoIR = A0/exp41 六月链（历史）；"
             "new LLM = E6-A full24 final（新 Skill agents，R0/R1 selected，fixed Direct Swin+SG）。",
             "",
             "| 口径 | DIV2K macro | LSDIR macro |",
             "|---|---:|---:|"]
    for k, label in (("rule", "E1 rule"), ("auto_old", "旧 AutoIR（A0/exp41）"),
                     ("new_rep1", "new LLM rep1"), ("new_rep2", "new LLM rep2"),
                     ("new_mean", "new LLM rep1/rep2 mean")):
        lines.append(f"| {label} | {out['macro']['DIV2K'][k]:.3f} | {out['macro']['LSDIR'][k]:.3f} |")
    lines += ["", "| 配对差 | DIV2K mean [95% CI] | wins | LSDIR mean [95% CI] | wins |",
              "|---|---:|---:|---:|---:|"]
    for newk, label in (("new_rep1", "new LLM rep1 − rule"),
                        ("new_rep2", "new LLM rep2 − rule"),
                        ("new_mean", "new LLM mean − rule")):
        a = out["paired"][f"{newk}_minus_rule_DIV2K"]
        b = out["paired"][f"{newk}_minus_rule_LSDIR"]
        lines.append(f"| {label} | {a['mean_delta']:+.3f} "
                     f"[{a['ci95'][0]:+.3f},{a['ci95'][1]:+.3f}] | {a['wins']}/24 | "
                     f"{b['mean_delta']:+.3f} [{b['ci95'][0]:+.3f},{b['ci95'][1]:+.3f}] | "
                     f"{b['wins']}/24 |")
    lines += ["", "旧 AutoIR 仍是历史基线，不与 new LLM 混为同一版本。"]
    (E1 / "e1_newllm_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
