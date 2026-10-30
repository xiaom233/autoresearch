#!/usr/bin/env python3
"""Compute SIDD full-1280 LQ baseline and scene-level bootstrap vs M_blind."""
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SIDD = ROOT / "datasets/E4/sidd"
FULL = ROOT / "rebuttal_runs/E4/sidd/full_eval"
EXTRA = ROOT / "rebuttal_runs/E4/sidd/full_eval_1280_extra"
METHODS = ["DFPIR-official-clip25", "DFPIR-blinded", "E4-SIDD-skill",
           "E4-SIDD-rule", "X-Restormer", "PromptIR", "DCPT-5D",
           "R2R-5D", "AirNet", "DATPRL-IR-5Task", "DATPRL-IR-CDD11"]


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 100.0 if mse == 0 else 10.0 * math.log10(255.0 ** 2 / mse)


def load_rows():
    rows = {}
    for m in ["M_blind"] + METHODS:
        rows[m] = {}
        for base in (FULL, EXTRA):
            for p in glob.glob(str(base / f"{m}_gpu*.jsonl")):
                for line in open(p):
                    if line.strip():
                        r = json.loads(line)
                        rows[m][r["image_id"]] = r["psnr_rgb"]
    return rows


def main():
    man = json.load(open(SIDD / "test_all_manifest.json"))
    scene_of = {f"clean_{m['index']}": m["scene"] for m in man["test"]}
    old = json.load(open(SIDD / "test_full_manifest.json"))
    old_ids = {f"clean_{m['index']}" for m in old["test"]}

    # LQ identity baseline
    lq = {}
    for base, ids in ((SIDD / "test_full", range(0, 1120)),
                      (SIDD / "test_support_scenes", range(1120, 1280))):
        for i in ids:
            c = np.array(Image.open(base / f"clean_{i}.png").convert("RGB"))
            d = np.array(Image.open(base / f"degraded_{i}.png").convert("RGB"))
            lq[f"clean_{i}"] = psnr(d, c)
    rows = load_rows()
    rows["LQ-identity"] = lq

    def scene_means(method, ids):
        by = defaultdict(list)
        for i in ids:
            if i in rows[method]:
                by[scene_of[i]].append(rows[method][i])
        return {s: float(np.mean(v)) for s, v in by.items()}

    full_ids = [f"clean_{i}" for i in range(1280)]
    out = {"lq_identity": {
        "full1280": float(np.mean([lq[i] for i in full_ids])),
        "test1120": float(np.mean([lq[i] for i in full_ids if i in old_ids]))},
        "full1280": {}, "test1120": {}}
    rng = np.random.default_rng(20261030)
    for subset, ids in (("full1280", full_ids),
                        ("test1120", [i for i in full_ids if i in old_ids])):
        base = scene_means("M_blind", ids)
        scenes = sorted(base)
        for m in ["M_blind"] + METHODS:
            sm = scene_means(m, ids)
            scenes_m = sorted(set(base) & set(sm))
            diff = np.array([sm[s] - base[s] for s in scenes_m])
            boots = []
            for _ in range(10000):
                idx = rng.integers(0, len(diff), len(diff))
                boots.append(float(np.mean(diff[idx])))
            out[subset][m] = {
                "n_scenes": len(scenes_m),
                "mean_psnr": float(np.mean(list(sm.values()))),
                "delta_vs_M_blind": float(np.mean(diff)),
                "ci95": [float(np.percentile(boots, 2.5)),
                         float(np.percentile(boots, 97.5))],
                "wins": int(sum(d > 0 for d in diff)),
            }
    (ROOT / "rebuttal_runs/E4/sidd/full_eval_1280_bootstrap.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# SIDD full 1,280 — LQ baseline and scene bootstrap", "",
             f"LQ identity: full1280 {out['lq_identity']['full1280']:.3f} dB; "
             f"test1120 {out['lq_identity']['test1120']:.3f} dB.", "",
             "| Method | full1280 PSNR | Δ vs M_blind | 95% CI | wins | old1120 Δ |",
             "|---|---:|---:|---:|---:|---:|"]
    for m in ["M_blind"] + METHODS:
        f = out["full1280"][m]; o = out["test1120"][m]
        ci = f" [{f['ci95'][0]:+.3f},{f['ci95'][1]:+.3f}]" if m != "M_blind" else ""
        lines.append(f"| {m} | {f['mean_psnr']:.3f} | {f['delta_vs_M_blind']:+.3f} | "
                     f"{ci} | {f['wins']}/{f['n_scenes']} | {o['delta_vs_M_blind']:+.3f} |")
    (ROOT / "rebuttal_runs/E4/sidd/full_eval_1280_bootstrap.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
