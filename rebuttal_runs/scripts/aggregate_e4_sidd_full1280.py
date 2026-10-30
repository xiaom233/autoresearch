#!/usr/bin/env python3
"""Aggregate SIDD full 1,280 validation blocks (40 scenes x 32).

Merges the original 1,120-block test rows (keys 160..1279) with the newly
evaluated 160 support-scene blocks (keys 0..159, image_id clean_1120..1279).
Reports full1280, the old test1120 subset, balanced40/35, per-scene macro,
and the 5-block support overlap.
"""
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SIDD = ROOT / "datasets/E4/sidd"
FULL = ROOT / "rebuttal_runs/E4/sidd/full_eval"
EXTRA = ROOT / "rebuttal_runs/E4/sidd/full_eval_1280_extra"
METHODS = ["M_blind", "DFPIR-official-clip25", "DFPIR-blinded",
           "E4-SIDD-skill", "E4-SIDD-rule", "X-Restormer", "PromptIR",
           "DCPT-5D", "R2R-5D", "AirNet",
           "DATPRL-IR-5Task", "DATPRL-IR-CDD11"]
SUPPORT_KEYS = {"ValidationBlocksSrgb_0", "ValidationBlocksSrgb_32",
                "ValidationBlocksSrgb_64", "ValidationBlocksSrgb_96",
                "ValidationBlocksSrgb_128"}


def load(paths):
    rows = {}
    dup = 0
    for p in paths:
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["image_id"] in rows:
                dup += 1
            rows[r["image_id"]] = r
    return rows, dup


def stats(rows, ids):
    present = [i for i in ids if i in rows]
    if not present:
        return {"n": 0, "psnr": None, "ssim": None, "lpips": None,
                "missing": len(ids)}
    return {
        "n": len(present),
        "psnr": float(np.mean([rows[i]["psnr_rgb"] for i in present])),
        "ssim": float(np.mean([rows[i]["ssim_rgb"] for i in present])),
        "lpips": float(np.mean([rows[i]["lpips_vgg"] for i in present])),
        "missing": len(ids) - len(present),
    }


def main():
    old = json.load(open(SIDD / "test_full_manifest.json"))
    miss = json.load(open(SIDD / "test_support_scenes/manifest.json"))["pairs"]
    scene_of = {}
    key_of = {}
    for m in old["test"]:
        i = f"clean_{m['index']}"
        scene_of[i] = m["scene"]; key_of[i] = m["key"]
    for m in miss:
        i = f"clean_{m['index']}"
        scene_of[i] = m["scene"]; key_of[i] = m["key"]

    full_ids = [f"clean_{i}" for i in range(1280)]
    test_ids = [f"clean_{i}" for i in range(1120)]
    overlap_ids = [f"clean_{i}" for i in range(1280) if key_of.get(f"clean_{i}") in SUPPORT_KEYS]
    no_overlap_ids = [i for i in full_ids if i not in set(overlap_ids)]
    balanced40 = []
    balanced35 = []
    for scene in range(40):
        idx = ((scene - 5) * 32 if scene >= 5 else 1120 + scene * 32)
        balanced40.append(f"clean_{idx}")
        if scene >= 5:
            balanced35.append(f"clean_{idx}")

    out = {"methods": {}, "notes": [
        "full1280: all 40 scenes x 32 = 1,280 blocks (keys 0..1279).",
        "test1120: original scene-disjoint test (keys 160..1279, scenes 5..39).",
        "support_scene160: keys 0..159 (scenes 0..4), never tested before.",
        "support overlap: 5 blocks were used as support (keys 0/32/64/96/128); "
        "they are included in full1280; no_overlap1275 excludes them.",
        "balanced40: block 0 of each of 40 scenes; balanced35: block 0 of test scenes only.",
        "DFPIR-official-clip25 uses the official CLIP prompt sigma=25.",
    ],
             "support_overlap_ids": overlap_ids,
             "n_full": len(full_ids), "n_test1120": len(test_ids),
             "n_support_scene": 160}
    for m in METHODS:
        old_rows, d1 = load(sorted(FULL.glob(f"{m}_gpu*.jsonl")))
        new_rows, d2 = load(sorted(EXTRA.glob(f"{m}_gpu*.jsonl")))
        rows = {**old_rows, **new_rows}
        rec = {"n_rows": len(rows), "duplicates": d1 + d2,
               "full1280": stats(rows, full_ids),
               "test1120": stats(rows, test_ids),
               "support_scene160": stats(rows, [f"clean_{i}" for i in range(1120, 1280)]),
               "no_overlap1275": stats(rows, no_overlap_ids),
               "support_overlap5": stats(rows, overlap_ids),
               "balanced40": stats(rows, balanced40),
               "balanced35": stats(rows, balanced35)}
        by_scene = defaultdict(list)
        for i in full_ids:
            if i in rows:
                by_scene[scene_of[i]].append(rows[i]["psnr_rgb"])
        rec["per_scene_full"] = {
            "n_scenes": len(by_scene),
            "psnr_macro": float(np.mean([np.mean(v) for v in by_scene.values()])),
            "psnr_min_scene": float(min(np.mean(v) for v in by_scene.values())),
            "psnr_max_scene": float(max(np.mean(v) for v in by_scene.values())),
        }
        if rec["full1280"]["psnr"] is not None and rec["test1120"]["psnr"] is not None:
            rec["delta_full_minus_test1120"] = rec["full1280"]["psnr"] - rec["test1120"]["psnr"]
        else:
            rec["delta_full_minus_test1120"] = None
        out["methods"][m] = rec

    (ROOT / "rebuttal_runs/E4/sidd/full_eval_1280_summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# SIDD validation full 1,280 blocks (40 scenes x 32)", "",
             "Merged: original test1120 (keys 160..1279) + new support-scene 160 (keys 0..159).",
             "Support overlap: 5 blocks (keys 0/32/64/96/128).", "",
             "| Method | full1280 PSNR | SSIM | LPIPS | test1120 PSNR | Δ(full−1120) | "
             "balanced40 | per-scene macro | missing |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in METHODS:
        r = out["methods"][m]
        f = r["full1280"]; t = r["test1120"]; b = r["balanced40"]
        lines.append(
            f"| {m} | {f['psnr']:.3f} | {f['ssim']:.4f} | {f['lpips']:.4f} | "
            f"{t['psnr']:.3f} | {r['delta_full_minus_test1120']:+.3f} | "
            f"{b['psnr']:.3f} | {r['per_scene_full']['psnr_macro']:.3f} | "
            f"{f['missing']} |")
    lines += ["", "`no_overlap1275` excludes the 5 support blocks; see JSON for detail."]
    (ROOT / "rebuttal_runs/E4/sidd/full_eval_1280_summary.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
