#!/usr/bin/env python3
"""Aggregate full SIDD (1120 blocks) results + balanced 35-scene subset."""
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FULL = ROOT / "rebuttal_runs/E4/sidd/full_eval"
SIDD = ROOT / "datasets/E4/sidd"


def load(method):
    rows = {}
    for p in sorted(FULL.glob(f"{method}_gpu*.jsonl")):
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows[r["image_id"]] = r
    return rows


def summarize(rows, ids):
    ps = [rows[i]["psnr_rgb"] for i in ids]
    ss = [rows[i]["ssim_rgb"] for i in ids]
    lp = [rows[i]["lpips_vgg"] for i in ids]
    return {"n": len(ids),
            "psnr": float(np.mean(ps)),
            "ssim": float(np.mean(ss)),
            "lpips": float(np.mean(lp))}


def main():
    full_manifest = json.load(open(SIDD / "test_full_manifest.json"))
    key_to_idx = {m["key"]: m["index"] for m in full_manifest["test"]}
    scene_of = {f"clean_{m['index']}": m["scene"] for m in full_manifest["test"]}

    orig = json.load(open(SIDD / "manifest.json"))
    balanced_ids = [f"clean_{key_to_idx[k]}" for k in orig["test_keys"]]

    methods = ["M_blind", "DFPIR-official", "DFPIR-blinded",
               "E4-SIDD-skill", "E4-SIDD-rule"]
    data = {m: load(m) for m in methods}

    all_ids = [f"clean_{i}" for i in range(len(full_manifest["test"]))]
    out = {"full": {}, "balanced35": {}, "per_scene_full": {}, "notes": [
        "full: all 1120 blocks of 35 test scenes (32 blocks/scene).",
        "balanced35: block 0 of each test scene (the original 35-image subset).",
        "PSNR/SSIM are RGB; LPIPS is VGG.",
    ]}
    for m in methods:
        rows = data[m]
        present = [i for i in all_ids if i in rows]
        out["full"][m] = summarize(rows, present)
        out["full"][m]["missing"] = len(all_ids) - len(present)
        bal = [i for i in balanced_ids if i in rows]
        out["balanced35"][m] = summarize(rows, bal)
        # per-scene macro (mean of 32-block means, scenes weighted equally)
        by_scene = defaultdict(list)
        for i in present:
            by_scene[scene_of[i]].append(rows[i]["psnr_rgb"])
        out["per_scene_full"][m] = {
            "n_scenes": len(by_scene),
            "psnr_macro": float(np.mean([np.mean(v) for v in by_scene.values()])),
            "psnr_min_scene": float(min(np.mean(v) for v in by_scene.values())),
            "psnr_max_scene": float(max(np.mean(v) for v in by_scene.values())),
        }

    (ROOT / "rebuttal_runs/E4/sidd/full_eval_summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{'method':<16} {'full PSNR':>10} {'full SSIM':>10} {'full LPIPS':>11} "
          f"{'bal35 PSNR':>11} {'bal35 SSIM':>11} {'bal35 LPIPS':>12}")
    for m in methods:
        f, b = out["full"][m], out["balanced35"][m]
        print(f"{m:<16} {f['psnr']:>10.3f} {f['ssim']:>10.4f} {f['lpips']:>11.4f} "
              f"{b['psnr']:>11.3f} {b['ssim']:>11.4f} {b['lpips']:>12.4f}")
    print()
    for m in methods:
        p = out["per_scene_full"][m]
        print(f"{m:<16} per-scene macro PSNR={p['psnr_macro']:.3f} "
              f"(min {p['psnr_min_scene']:.2f}, max {p['psnr_max_scene']:.2f}, "
              f"{p['n_scenes']} scenes)")


if __name__ == "__main__":
    main()
