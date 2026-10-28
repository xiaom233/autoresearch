#!/usr/bin/env python3
"""Aggregate E4 rain (Rain100H 95 pairs) real paired evaluation."""
import glob
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RAIN = ROOT / "rebuttal_runs/E4/rain/full_eval"
METHODS = ["M_blind", "DFPIR-official", "DFPIR-blinded", "E4-rain-rule"]
REFERENCE = ["PromptIR", "AirNet"]


def load(m):
    rows = {}
    for p in sorted(RAIN.glob(f"{m}*.jsonl")):
        for l in open(p):
            if l.strip():
                r = json.loads(l)
                rows[r["image_id"]] = r
    return rows


def ident_psnr():
    vals = {}
    d = ROOT / "datasets/E4/rain/test"
    for c in sorted(d.glob("clean_*.png")):
        idx = c.stem.split("_")[1]
        a = np.asarray(Image.open(d / f"degraded_{idx}.png").convert("RGB"), dtype=np.float32)
        b = np.asarray(Image.open(c).convert("RGB"), dtype=np.float32)
        mse = float(np.mean((a - b) ** 2))
        vals[c.stem] = 100.0 if mse == 0 else float(10 * np.log10(255 ** 2 / mse))
    return vals


def boot(d, seed=42):
    d = np.array(d)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (10000, len(d)))
    m = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ident = ident_psnr()
    ids = sorted(ident, key=lambda s: int(s.split("_")[1]))
    out = {"n": len(ids), "dataset": "Rain100H test (dronefreak/Rain13K)",
           "methods": {"identity/AutoIR-empty": {
               "n": len(ids), "psnr": float(np.mean([ident[i] for i in ids]))}},
           "paired_vs_identity": {}, "paired_vs_M_blind": {}}
    data = {}
    for m in METHODS + REFERENCE:
        rows = load(m)
        if not rows:
            continue
        data[m] = rows
        present = [i for i in ids if i in rows]
        bucket = "methods" if m in METHODS else "reference_methods"
        out.setdefault(bucket, {})
        out[bucket][m] = {
            "n": len(present),
            "psnr": float(np.mean([rows[i]["psnr_rgb"] for i in present])),
            "ssim": float(np.mean([rows[i]["ssim_rgb"] for i in present])),
            "lpips": float(np.mean([rows[i]["lpips_vgg"] for i in present])),
            "missing": len(ids) - len(present),
        }
        d = [rows[i]["psnr_rgb"] - ident[i] for i in present]
        mean, lo, hi = boot(d)
        out["paired_vs_identity"][m] = {
            "n": len(d), "delta": mean, "ci": [lo, hi],
            "wins": int(np.sum(np.array(d) > 0))}
    if "M_blind" in data:
        base = data["M_blind"]
        for m, rows in data.items():
            if m == "M_blind":
                continue
            common = [i for i in ids if i in base and i in rows]
            d = [rows[i]["psnr_rgb"] - base[i]["psnr_rgb"] for i in common]
            mean, lo, hi = boot(d)
            out["paired_vs_M_blind"][m] = {
                "n": len(common), "delta": mean, "ci": [lo, hi],
                "wins": int(np.sum(np.array(d) > 0))}

    json.dump(out, open(RAIN / "summary.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
