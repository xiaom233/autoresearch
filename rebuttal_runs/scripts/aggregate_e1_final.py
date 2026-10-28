#!/usr/bin/env python3
"""Aggregate E1 rule vs AutoIR on fixed-LQ test, with exclusion variants."""
import glob
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CAL = {f"{i:04d}" for i in range(801, 811)}
SUP = json.load(open(ROOT / "rebuttal_runs/A0/support_test_overlap.json"))
SUPPORT = {k: set(v) for k, v in SUP["per_scenario_support_images"].items()}


def load(pattern):
    out = []
    for p in glob.glob(pattern):
        out += [json.loads(l) for l in open(p) if l.strip()]
    return out


def group(rows, exclude_support=False, exclude_cal=False):
    g = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        if r["dataset"] == "DIV2K":
            if exclude_support and r["image_id"] in SUPPORT.get(r["scenario_id"], set()):
                continue
            if exclude_cal and r["image_id"] in CAL:
                continue
        g[r["controller"]][r["scenario_id"]][r["dataset"]].append(r)
    return g


def summarize(rows, exclude_support=False, exclude_cal=False):
    g = group(rows, exclude_support, exclude_cal)
    out = {}
    for ctrl, by_s in g.items():
        per_ds = {}
        for ds in ("DIV2K", "LSDIR"):
            vals = {"psnr": [], "ssim": [], "lpips": []}
            for sid, by_ds in by_s.items():
                rs = by_ds.get(ds, [])
                if not rs:
                    continue
                vals["psnr"].append(float(np.mean([x["psnr_rgb"] for x in rs])))
                vals["ssim"].append(float(np.mean([x["ssim_rgb"] for x in rs])))
                vals["lpips"].append(float(np.mean([x["lpips_vgg"] for x in rs])))
            per_ds[ds] = {
                "n_scenarios": len(vals["psnr"]),
                "psnr": float(np.mean(vals["psnr"])) if vals["psnr"] else None,
                "ssim": float(np.mean(vals["ssim"])) if vals["ssim"] else None,
                "lpips": float(np.mean(vals["lpips"])) if vals["lpips"] else None,
                "per_scenario_psnr": {},
            }
        out[ctrl] = per_ds
    # fill per-scenario psnr
    for ctrl, by_s in g.items():
        for ds in ("DIV2K", "LSDIR"):
            for sid, by_ds in by_s.items():
                rs = by_ds.get(ds, [])
                if rs:
                    out[ctrl][ds]["per_scenario_psnr"][sid] = float(
                        np.mean([x["psnr_rgb"] for x in rs]))
    return out


def paired(ref, other, ds, n_boot=5000, seed=42):
    a = ref[ds]["per_scenario_psnr"]
    b = other[ds]["per_scenario_psnr"]
    sids = sorted(set(a) & set(b))
    if not sids:
        return None
    diffs = np.array([b[s] - a[s] for s in sids])
    rng = random.Random(seed)
    boots = sorted(float(np.mean([diffs[rng.randrange(len(diffs))] for _ in diffs]))
                   for _ in range(n_boot))
    return {
        "n_scenarios": len(sids),
        "delta_psnr": float(np.mean(diffs)),
        "ci95": [boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]],
        "wins": int(np.sum(diffs > 0)),
        "losses": int(np.sum(diffs < 0)),
        "worst": float(np.min(diffs)),
        "best": float(np.max(diffs)),
    }


def main():
    auto = load(str(ROOT / "rebuttal_runs/A0/test_metrics/*.jsonl"))
    rule = load(str(ROOT / "rebuttal_runs/E1/test_metrics/*.jsonl"))
    rows = auto + rule
    full = summarize(rows)
    sup = summarize(rows, exclude_support=True)
    both = summarize(rows, exclude_support=True, exclude_cal=True)
    comp = {}
    if "AutoIR" in full and "rule" in full:
        comp = {ds: paired(full["AutoIR"], full["rule"], ds)
                for ds in ("DIV2K", "LSDIR")}
    out = {
        "full": full, "support_excluded": sup,
        "support_plus_calib_excluded": both,
        "rule_minus_autoIR_paired": comp,
        "notes": [
            "Macro average over scenarios; paired bootstrap resamples scenarios.",
            "support_excluded removes each scenario's 5 support images from DIV2K.",
            "support_plus_calib_excluded also removes 0801-0810.",
            "LSDIR has no support/calibration overlap.",
        ],
    }
    path = ROOT / "rebuttal_runs/E1/e1_final_summary.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    md = ["# E1 rule vs AutoIR (fixed LQ)", "",
          "## Full test", "", "| Controller | Dataset | PSNR | SSIM | LPIPS |",
          "|---|---|---:|---:|---:|"]
    for ctrl, per in full.items():
        for ds, m in per.items():
            md.append(f"| {ctrl} | {ds} | {m['psnr']:.3f} | {m['ssim']:.4f} | {m['lpips']:.4f} |")
    md += ["", "## DIV2K support-excluded", "", "| Controller | PSNR | SSIM | LPIPS |",
           "|---|---:|---:|---:|"]
    for ctrl, per in sup.items():
        m = per["DIV2K"]
        md.append(f"| {ctrl} | {m['psnr']:.3f} | {m['ssim']:.4f} | {m['lpips']:.4f} |")
    md += ["", "## DIV2K support+calibration-excluded", "", "| Controller | PSNR | SSIM | LPIPS |",
           "|---|---:|---:|---:|"]
    for ctrl, per in both.items():
        m = per["DIV2K"]
        md.append(f"| {ctrl} | {m['psnr']:.3f} | {m['ssim']:.4f} | {m['lpips']:.4f} |")
    md += ["", "## Paired rule − AutoIR", ""]
    for ds, c in comp.items():
        if c:
            md.append(f"- {ds}: ΔPSNR={c['delta_psnr']:+.3f} "
                      f"[{c['ci95'][0]:+.3f},{c['ci95'][1]:+.3f}], wins {c['wins']}/{c['n_scenarios']}, "
                      f"worst {c['worst']:+.3f}, best {c['best']:+.3f}")
    (ROOT / "rebuttal_runs/E1/e1_final_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
