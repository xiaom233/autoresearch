#!/usr/bin/env python3
"""Aggregate E6-LLM support-set variation and compare with the rule E6-B."""
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
E = ROOT / "rebuttal_runs/E6_LLM"
SCENARIOS = ["blind_0006", "blind_0008", "blind_0010",
             "blind_0013", "blind_0017", "blind_0024"]


def load_metrics():
    rows = []
    for p in glob.glob(str(E / "test_metrics_gpu*.jsonl")):
        rows += [json.loads(l) for l in open(p) if l.strip()]
    return rows


def main():
    sel = {r["run_id"]: r for r in map(json.loads, open(E / "selection.jsonl"))}
    rows = load_metrics()
    by_run = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_run[r["run_id"]][r["dataset"]].append(r["psnr_rgb"])

    rule = json.load(open(ROOT / "rebuttal_runs/E6/e6_summary.json"))
    out = {"scenarios": {}, "macro": {}, "n_metrics": len(rows),
           "scope_notes": [
               "E6-LLM: 6 scenarios x 3 disjoint support sets; per chain Skill blind ID + strategy agent + Direct R0 + Skill R1/R2 reflection (support-only triggers) + support-PSNR selection + fixed GT-LQ test.",
               "set1 blind IDs are reused from E6-A rep1 (same support); set2/set3 are new Skill agent runs.",
               "Test is the fixed GT-LQ DIV2K 100 / LSDIR 250 set (same protocol as E6-B); GT is evaluation-only.",
               "Comparison to E6-B uses the rule-controller support sets; both use the same test LQ.",
           ]}
    diffs = {"DIV2K": [], "LSDIR": []}
    for k in [1, 2, 3]:
        for sid in SCENARIOS:
            name = f"{sid}_set{k}"
            run_id = f"E6LLM_{name}"
            if run_id not in sel:
                continue
            d = by_run[run_id].get("DIV2K")
            l = by_run[run_id].get("LSDIR")
            rec = {"name": name, "round": sel[run_id]["round"],
                   "pipeline": sel[run_id]["predicted_pipeline"],
                   "support_psnr": sel[run_id]["support_psnr"],
                   "DIV2K": float(np.mean(d)) if d else None,
                   "LSDIR": float(np.mean(l)) if l else None,
                   "n_DIV2K": len(d or []), "n_LSDIR": len(l or [])}
            out["scenarios"].setdefault(sid, {})[f"set{k}"] = rec
            rule_rec = rule["scenarios"][sid][f"set{k}"]["datasets"]
            rec["rule_DIV2K"] = rule_rec["DIV2K"]["psnr"]
            rec["rule_LSDIR"] = rule_rec["LSDIR"]["psnr"]
            rec["delta_DIV2K"] = (rec["DIV2K"] - rec["rule_DIV2K"]) if rec["DIV2K"] is not None else None
            rec["delta_LSDIR"] = (rec["LSDIR"] - rec["rule_LSDIR"]) if rec["LSDIR"] is not None else None
            if rec["delta_DIV2K"] is not None:
                diffs["DIV2K"].append(rec["delta_DIV2K"])
            if rec["delta_LSDIR"] is not None:
                diffs["LSDIR"].append(rec["delta_LSDIR"])

    for ds in ["DIV2K", "LSDIR"]:
        means = []
        sds = []
        for sid in SCENARIOS:
            vals = [out["scenarios"][sid][f"set{k}"][ds] for k in [1, 2, 3]
                    if f"set{k}" in out["scenarios"].get(sid, {})
                    and out["scenarios"][sid][f"set{k}"][ds] is not None]
            if vals:
                means.append(float(np.mean(vals)))
                sds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        out["macro"][ds] = {"mean_of_scenario_means": float(np.mean(means)) if means else None,
                            "mean_within_scenario_sd": float(np.mean(sds)) if sds else None,
                            "scenario_sd": float(np.std(means, ddof=1)) if len(means) > 1 else None}
    for ds in ["DIV2K", "LSDIR"]:
        arr = np.array(diffs[ds])
        if len(arr):
            rng = np.random.default_rng(42)
            idx = rng.integers(0, len(arr), (10000, len(arr)))
            boot = arr[idx].mean(axis=1)
            out.setdefault("vs_rule", {})[ds] = {
                "n": len(arr), "delta": float(arr.mean()),
                "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                "wins": int((arr > 0).sum()),
                "note": "cell-level bootstrap (18 scenario/set cells); cells within a scenario are not independent"}
        # scenario-level paired comparison (6 scenarios; primary inference)
        sc_deltas = []
        for sid in SCENARIOS:
            rec = out["scenarios"].get(sid, {})
            vals = [rec[f"set{k}"][ds] for k in [1, 2, 3]
                    if f"set{k}" in rec and rec[f"set{k}"][ds] is not None]
            rvals = [rule["scenarios"][sid][f"set{k}"]["datasets"][ds]["psnr"] for k in [1, 2, 3]]
            if len(vals) == 3:
                sc_deltas.append(float(np.mean(vals) - np.mean(rvals)))
        if len(sc_deltas) == len(SCENARIOS):
            arr6 = np.array(sc_deltas)
            rng = np.random.default_rng(42)
            idx = rng.integers(0, len(arr6), (10000, len(arr6)))
            boot = arr6[idx].mean(axis=1)
            out.setdefault("vs_rule_scenario", {})[ds] = {
                "n_scenarios": len(arr6), "delta": float(arr6.mean()),
                "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                "wins": int((arr6 > 0).sum()),
                "per_scenario": {sid: float(v) for sid, v in zip(SCENARIOS, arr6)}}
        ranges = []
        for sid in SCENARIOS:
            rec = out["scenarios"].get(sid, {})
            vals = [rec[f"set{k}"][ds] for k in [1, 2, 3]
                    if f"set{k}" in rec and rec[f"set{k}"][ds] is not None]
            if len(vals) == 3:
                ranges.append(float(max(vals) - min(vals)))
        out["macro"][ds]["mean_within_scenario_range"] = float(np.mean(ranges)) if ranges else None
    json.dump(out, open(E / "e6llm_summary.json", "w"), indent=2, ensure_ascii=False)

    lines = ["# E6-LLM support-set variation (full LLM flow)",
             "",
             "6 scenarios x 3 support sets; Skill blind ID + strategy agent + Direct R0 + Skill R1/R2 reflection + support selection; fixed GT-LQ test.",
             "",
             "| Scenario | set1 | set2 | set3 | mean | SD | range | rule set1 | rule set2 | rule set3 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for sid in SCENARIOS:
        rec = out["scenarios"][sid]
        vals = [rec[f"set{k}"]["DIV2K"] for k in [1, 2, 3] if f"set{k}" in rec and rec[f"set{k}"]["DIV2K"] is not None]
        rule_vals = [rec[f"set{k}"]["rule_DIV2K"] for k in [1, 2, 3] if f"set{k}" in rec]
        if not vals:
            continue
        lines.append(f"| {sid} | " + " | ".join(f"{v:.3f}" for v in vals) +
                     f" | {np.mean(vals):.3f} | {np.std(vals, ddof=1):.3f} | {max(vals)-min(vals):.3f} | " +
                     " | ".join(f"{v:.3f}" for v in rule_vals) + " |")
    m = out["macro"]
    lines += ["", f"DIV2K macro: {m['DIV2K']['mean_of_scenario_means']:.3f} dB; "
              f"mean within-scenario SD {m['DIV2K']['mean_within_scenario_sd']:.3f}; "
              f"mean range {m['DIV2K'].get('mean_within_scenario_range')}.",
              f"LSDIR macro: {m['LSDIR']['mean_of_scenario_means']:.3f} dB; "
              f"mean within-scenario SD {m['LSDIR']['mean_within_scenario_sd']:.3f}; "
              f"mean range {m['LSDIR'].get('mean_within_scenario_range')}.",
              "", "vs rule E6-B:"]
    for ds in ["DIV2K", "LSDIR"]:
        v6 = out.get("vs_rule_scenario", {}).get(ds)
        v = out.get("vs_rule", {}).get(ds)
        if v6:
            lines.append(f"- {ds} (primary, 6-scenario paired bootstrap): {v6['delta']:+.3f} dB, "
                         f"CI [{v6['ci'][0]:+.3f}, {v6['ci'][1]:+.3f}], LLM wins {v6['wins']}/{v6['n_scenarios']} scenarios.")
        if v:
            lines.append(f"  - cell-level (18 cells, not independent): {v['delta']:+.3f} dB, "
                         f"CI [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}], wins {v['wins']}/{v['n']}.")
    (E / "e6llm_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
