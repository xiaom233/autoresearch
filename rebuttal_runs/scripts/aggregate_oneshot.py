#!/usr/bin/env python3
"""Aggregate the 2x2 controller x reflection comparison.

Arms:
  rule-oneshot     : E1 rule R0 prediction, one training, no reflection
  rule-full        : E1 rule R0->R1/R2 reflection, support-selected
  AutoIR-oneshot   : AutoIR/LLM R0 prediction, one training, no reflection
  AutoIR-full      : AutoIR/LLM reflection, support-selected
"""
import glob
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "rebuttal_runs/E1_oneshot"
SUP = json.load(open(ROOT / "rebuttal_runs/A0/support_test_overlap.json"))[
    "per_scenario_support_images"]


def load(pattern):
    rows = []
    for p in glob.glob(str(ROOT / pattern)):
        rows += [json.loads(l) for l in open(p) if l.strip()]
    return rows


def macro(rows, ctrl, exclude_support=False, sids=None):
    g = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("controller") != ctrl:
            continue
        if sids is not None and r["scenario_id"] not in sids:
            continue
        if exclude_support and r["dataset"] == "DIV2K" and \
                r["image_id"] in SUP.get(r["scenario_id"], []):
            continue
        g[r["dataset"]][r["scenario_id"]].append(r)
    out = {}
    for ds in ("DIV2K", "LSDIR"):
        vals = {"psnr": [], "ssim": [], "lpips": [], "per": {}}
        for sid, rs in g[ds].items():
            p = float(np.mean([x["psnr_rgb"] for x in rs]))
            vals["psnr"].append(p)
            if all("ssim_rgb" in x for x in rs):
                vals["ssim"].append(float(np.mean([x["ssim_rgb"] for x in rs])))
            if all("lpips_vgg" in x for x in rs):
                vals["lpips"].append(float(np.mean([x["lpips_vgg"] for x in rs])))
            vals["per"][sid] = p
        out[ds] = {
            "n_scenarios": len(vals["psnr"]),
            "n_ssim": len(vals["ssim"]),
            "n_lpips": len(vals["lpips"]),
            "psnr": float(np.mean(vals["psnr"])) if vals["psnr"] else None,
            "ssim": float(np.mean(vals["ssim"])) if vals["ssim"] else None,
            "lpips": float(np.mean(vals["lpips"])) if vals["lpips"] else None,
            "per_scenario_psnr": vals["per"],
        }
    return out


def paired(base, other, ds, n_boot=10000, seed=42):
    a, b = base[ds]["per_scenario_psnr"], other[ds]["per_scenario_psnr"]
    ids = sorted(set(a) & set(b))
    if not ids:
        return None
    d = np.array([b[s] - a[s] for s in ids])
    rng = random.Random(seed)
    boots = sorted(float(np.mean([d[rng.randrange(len(d))] for _ in d]))
                   for _ in range(n_boot))
    return {
        "n": len(ids), "delta": float(d.mean()),
        "ci95": [boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]],
        "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
        "worst": float(d.min()), "best": float(d.max()),
    }


def main():
    rule_one = load("rebuttal_runs/E1_oneshot/test_metrics/div2k_gpu*.jsonl")
    auto_one = rule_one
    patch = load("rebuttal_runs/E1_oneshot/test_metrics/fixed_lq_patch_gpu*.jsonl")
    if patch:
        patch_map = {(r["controller"], r["scenario_id"], r["dataset"], r["image_id"]): r
                     for r in patch}
        def replace_patch(rows):
            return [patch_map.get((r.get("controller"), r["scenario_id"],
                                   r["dataset"], r["image_id"]), r) for r in rows]
        rule_one = replace_patch(rule_one)
        auto_one = rule_one
    rule_full = load("rebuttal_runs/E1/test_metrics/rule_gpu*.jsonl")
    auto_full = load("rebuttal_runs/A0/test_metrics/*.jsonl")
    arms = {
        "rule-oneshot": rule_one,
        "AutoIR-oneshot": auto_one,
        "rule-full": rule_full,
        "AutoIR-full": auto_full,
    }
    ctrl = {
        "rule-oneshot": "rule-oneshot",
        "AutoIR-oneshot": "AutoIR-oneshot",
        "rule-full": "rule",
        "AutoIR-full": "AutoIR",
    }
    full = {name: macro(rows, ctrl[name]) for name, rows in arms.items()}
    sup = {name: macro(rows, ctrl[name], True) for name, rows in arms.items()}
    common = sorted(set.intersection(*[
        set(full[name]["DIV2K"]["per_scenario_psnr"]) for name in full]))
    full_common = {name: macro(rows, ctrl[name], False, set(common))
                   for name, rows in arms.items()}
    sup_common = {name: macro(rows, ctrl[name], True, set(common))
                  for name, rows in arms.items()}
    # Full arms: use the fixed-LQ paired values for the 7 nondeterministic
    # scenarios; the one-shot rows for the 6 common ones are patched above.
    fixed = json.load(open(ROOT / "rebuttal_runs/E1E2_fixed_lq_summary.json"))
    fixed_sids = set(fixed["fixed_scenarios"])
    for name in ("rule-full", "AutoIR-full"):
        c = ctrl[name]
        for ds in ("DIV2K", "LSDIR"):
            per = full_common[name][ds]["per_scenario_psnr"]
            for sid in fixed_sids:
                v = fixed.get("e1", {}).get("full", {}).get(c, {}).get(ds, {}).get("per_scenario", {}).get(sid)
                if v is not None and sid in per:
                    per[sid] = float(v)
            full_common[name][ds]["psnr"] = float(np.mean(list(per.values()))) if per else None
    comps = {
        "rule-full - rule-oneshot": ("rule-oneshot", "rule-full"),
        "AutoIR-full - AutoIR-oneshot": ("AutoIR-oneshot", "AutoIR-full"),
        "AutoIR-full - rule-oneshot": ("rule-oneshot", "AutoIR-full"),
        "AutoIR-oneshot - rule-oneshot": ("rule-oneshot", "AutoIR-oneshot"),
        "AutoIR-full - rule-full": ("rule-full", "AutoIR-full"),
    }
    paired_full = {name: {ds: paired(full_common[a], full_common[b], ds) for ds in ("DIV2K", "LSDIR")}
                   for name, (a, b) in comps.items()}
    paired_sup = {name: {ds: paired(sup_common[a], sup_common[b], ds) for ds in ("DIV2K", "LSDIR")}
                  for name, (a, b) in comps.items()}
    res = {"full": full, "full_common": full_common, "common_scenarios": common,
           "support_excluded": sup, "support_excluded_common": sup_common,
           "paired_full": paired_full, "paired_support_excluded": paired_sup,
           "notes": [
               "Scenario-level macro; paired bootstrap resamples scenarios.",
               "rule-oneshot is E1 R0 (R0_fixed preferred); AutoIR-oneshot is A0 per-round R0.",
               "rule-full is E1 selected R1/R0; AutoIR-full is A0 selected R0/R1/R2.",
               "E1-oneshot blind_0020 uses a stable Swin+SG NaN fallback if present.",
           ]}
    (OUT / "oneshot_summary.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    md = ["# E1 one-shot / no-reflection comparison", "",
          "| Controller | DIV2K PSNR | DIV2K SSIM | DIV2K LPIPS | LSDIR PSNR | LSDIR SSIM | LSDIR LPIPS |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ["rule-oneshot", "rule-full", "AutoIR-oneshot", "AutoIR-full"]:
        f = full_common[name]
        def fmt(v, key="ssim"):
            if v is None or f['DIV2K'].get(f"n_{key}", 0) != f['DIV2K']['n_scenarios']:
                return "—"
            return f"{v:.4f}"
        ls_psnr = "—" if f['LSDIR']['psnr'] is None else f"{f['LSDIR']['psnr']:.3f}"
        md.append(f"| {name} | {f['DIV2K']['psnr']:.3f} | {fmt(f['DIV2K']['ssim'], 'ssim')} | "
                  f"{fmt(f['DIV2K']['lpips'], 'lpips')} | {ls_psnr} | "
                  f"{fmt(f['LSDIR']['ssim'], 'ssim')} | {fmt(f['LSDIR']['lpips'], 'lpips')} |")
    md += ["", "Paired scenario-level comparisons (full test):", ""]
    for name, per in paired_full.items():
        for ds, v in per.items():
            if v:
                md.append(f"- {name} {ds}: {v['delta']:+.3f} "
                          f"[{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}], "
                          f"wins {v['wins']}/{v['n']}, worst {v['worst']:+.3f}, best {v['best']:+.3f}")
    (OUT / "oneshot_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
