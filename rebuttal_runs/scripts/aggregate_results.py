#!/usr/bin/env python3
"""Aggregate per-image test metrics into per-scenario/dataset tables.

- Macro average over scenarios (equal weight), not over images.
- Reports full test and DIV2K-overlap-excluded (0801-0810) variants.
- Paired scenario bootstrap for controller differences.
"""
import argparse
import glob
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OVERLAP_STEMS = {f"{i:04d}" for i in range(801, 811)}


def load_rows(patterns):
    rows = []
    for pat in patterns:
        for p in glob.glob(pat):
            for line in open(p):
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def mean(xs):
    return float(np.mean(xs)) if xs else None


def group_rows(rows, exclude_div2k_overlap=False):
    g = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        if exclude_div2k_overlap and r["dataset"] == "DIV2K" and r["image_id"] in OVERLAP_STEMS:
            continue
        key = (r["controller"], r["scenario_id"], r["dataset"])
        g[r["controller"]][r["scenario_id"]][r["dataset"]].append(r)
    return g


def summarize(rows, exclude=False):
    g = group_rows(rows, exclude)
    out = {}
    for ctrl, by_s in g.items():
        per_ds = {}
        for ds in ("DIV2K", "LSDIR"):
            scen_means = {"psnr": [], "ssim": [], "lpips": []}
            for sid, by_ds in by_s.items():
                if ds not in by_ds:
                    continue
                rs = by_ds[ds]
                scen_means["psnr"].append(mean([x["psnr_rgb"] for x in rs]))
                scen_means["ssim"].append(mean([x["ssim_rgb"] for x in rs]))
                scen_means["lpips"].append(mean([x["lpips_vgg"] for x in rs]))
            per_ds[ds] = {
                "n_scenarios": len(scen_means["psnr"]),
                "psnr": mean(scen_means["psnr"]),
                "ssim": mean(scen_means["ssim"]),
                "lpips": mean(scen_means["lpips"]),
                "per_scenario_psnr": dict(zip(
                    [sid for sid in by_s if ds in by_s[sid]],
                    [mean([x["psnr_rgb"] for x in by_s[sid][ds]]) for sid in by_s if ds in by_s[sid]])),
            }
        out[ctrl] = per_ds
    return out


def paired_bootstrap(ref, other, ds, n_boot=5000, seed=42):
    a = ref[ds]["per_scenario_psnr"]
    b = other[ds]["per_scenario_psnr"]
    sids = sorted(set(a) & set(b))
    if not sids:
        return None
    diffs = np.array([b[s] - a[s] for s in sids])
    rng = random.Random(seed)
    boots = []
    n = len(diffs)
    for _ in range(n_boot):
        boots.append(float(np.mean([diffs[rng.randrange(n)] for _ in range(n)])))
    boots.sort()
    return {
        "n_scenarios": n,
        "delta_psnr": float(np.mean(diffs)),
        "ci95": [boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]],
        "wins": int(np.sum(diffs > 0)),
        "losses": int(np.sum(diffs < 0)),
        "worst_delta": float(np.min(diffs)),
        "best_delta": float(np.max(diffs)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="glob patterns of per-image JSONL")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference", default="AutoIR")
    args = ap.parse_args()
    rows = load_rows(args.inputs)
    full = summarize(rows, exclude=False)
    clean = summarize(rows, exclude=True)
    comps = {}
    if args.reference in full:
        for ctrl in full:
            if ctrl == args.reference:
                continue
            comps[ctrl] = {}
            for ds in ("DIV2K", "LSDIR"):
                r = paired_bootstrap(full[args.reference], full[ctrl], ds)
                if r:
                    comps[ctrl][ds] = r
    result = {
        "full_test": full,
        "div2k_overlap_excluded": clean,
        "paired_vs_" + args.reference: comps,
        "overlap_excluded_images": sorted(OVERLAP_STEMS),
        "notes": [
            "Per-scenario means first, then equal-weight macro average across scenarios.",
            "Paired bootstrap resamples scenarios, not images.",
            "DIV2K overlap-excluded removes 0801-0810 (used by current calibration scripts).",
        ],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    md = ["# Test metrics (fixed LQ, VGG LPIPS, Gaussian SSIM)", "",
          "## Full test", "",
          "| Controller | Dataset | Scenarios | PSNR | SSIM | LPIPS |",
          "|---|---|---:|---:|---:|---:|"]
    for ctrl, per in full.items():
        for ds, m in per.items():
            md.append(f"| {ctrl} | {ds} | {m['n_scenarios']} | {m['psnr']:.3f} | "
                      f"{m['ssim']:.4f} | {m['lpips']:.4f} |")
    md += ["", "## DIV2K overlap-excluded (0801-0810 removed)", "",
           "| Controller | Dataset | Scenarios | PSNR | SSIM | LPIPS |",
           "|---|---|---:|---:|---:|---:|"]
    for ctrl, per in clean.items():
        for ds, m in per.items():
            md.append(f"| {ctrl} | {ds} | {m['n_scenarios']} | {m['psnr']:.3f} | "
                      f"{m['ssim']:.4f} | {m['lpips']:.4f} |")
    md += ["", f"## Paired vs {args.reference} (full test)", ""]
    for ctrl, per in comps.items():
        for ds, c in per.items():
            md.append(f"- {ctrl} vs {args.reference} / {ds}: ΔPSNR={c['delta_psnr']:+.3f} dB "
                      f"[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}], "
                      f"wins {c['wins']}/{c['n_scenarios']}, worst {c['worst_delta']:+.3f}")
    Path(args.out).with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
