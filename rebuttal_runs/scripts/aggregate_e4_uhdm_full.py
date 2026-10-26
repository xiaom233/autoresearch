#!/usr/bin/env python3
"""Aggregate the full UHDM 495-pair evaluation and compare with the old 50 subset."""
import glob
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
FULL = ROOT / "rebuttal_runs/E4/uhdm/full_eval"
SUB = ROOT / "rebuttal_runs/E4/uhdm"
METHODS = ["M_blind", "DFPIR-official", "DFPIR-blinded", "E4-UHDM-rule"]


def load_full(m):
    rows = {}
    for p in sorted(FULL.glob(f"{m}_gpu*.jsonl")):
        for l in open(p):
            if l.strip():
                r = json.loads(l)
                rows[r["image_id"]] = r
    return rows


def load_old(m, path):
    return {json.loads(l)["image_id"]: json.loads(l)
            for l in open(path) if l.strip()}


def psnr_identity():
    vals = []
    man = json.load(open(ROOT / "datasets/E4/uhdm/test_full/manifest.json"))
    for p in man["test"]:
        a = np.asarray(Image.open(ROOT / p["degraded_source"]).convert("RGB"), dtype=np.float32)
        b = np.asarray(Image.open(ROOT / p["clean_source"]).convert("RGB"), dtype=np.float32)
        mse = float(np.mean((a - b) ** 2))
        vals.append(100.0 if mse == 0 else float(10 * np.log10(255 ** 2 / mse)))
    return vals


def boot(d, seed=42):
    d = np.array(d)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (10000, len(d)))
    m = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    out = {"n_full": 495, "methods": {}, "identity": {}, "subset_vs_full": {}}
    ids = None
    for m in METHODS:
        rows = load_full(m)
        if ids is None:
            ids = sorted(rows, key=lambda s: int(s.split("_")[1]))
        present = [i for i in ids if i in rows]
        out["methods"][m] = {
            "n": len(present),
            "psnr": float(np.mean([rows[i]["psnr_rgb"] for i in present])),
            "ssim": float(np.mean([rows[i]["ssim_rgb"] for i in present])),
            "lpips": float(np.mean([rows[i]["lpips_vgg"] for i in present])),
            "missing": len(ids) - len(present),
        }
    ident = psnr_identity()
    out["identity"] = {"n": len(ident), "psnr": float(np.mean(ident))}

    # paired vs M_blind
    base = load_full("M_blind")
    out["paired_vs_M_blind"] = {}
    for m in METHODS[1:]:
        rows = load_full(m)
        common = [i for i in ids if i in base and i in rows]
        d = [rows[i]["psnr_rgb"] - base[i]["psnr_rgb"] for i in common]
        mean, lo, hi = boot(d)
        out["paired_vs_M_blind"][m] = {
            "n": len(common), "delta": mean, "ci": [lo, hi],
            "wins": int(np.sum(np.array(d) > 0))}
    # old 50-subset vs full for methods with old files
    old_files = {
        "M_blind": SUB / "baselines/m_blind.jsonl",
        "DFPIR-official": SUB / "baselines_official/dfpir_official.jsonl",
        "DFPIR-blinded": SUB / "baselines/dfpir_zero.jsonl",
        "E4-UHDM-rule": SUB / "rule_eval.jsonl",
    }
    sub_man = json.load(open(ROOT / "datasets/E4/uhdm/manifest.json"))["test"]
    full_man = json.load(open(ROOT / "datasets/E4/uhdm/test_full/manifest.json"))["test"]
    src2full = {p["source_index"]: p["index"] for p in full_man}
    old_id2src = {f"clean_{i}": int(Path(x["degraded_source"]).stem.split("_")[0])
                  for i, x in enumerate(sub_man)}
    for m, p in old_files.items():
        if not p.exists():
            continue
        old = load_old(m, p)
        full = load_full(m)
        pairs = []
        for oid, src in old_id2src.items():
            fid = f"clean_{src2full[src]}"
            if oid in old and fid in full:
                pairs.append((oid, fid))
        out["subset_vs_full"][m] = {
            "n_common": len(pairs),
            "subset_psnr": float(np.mean([old[o]["psnr_rgb"] for o, f in pairs])),
            "full_psnr_same_images": float(np.mean([full[f]["psnr_rgb"] for o, f in pairs])),
            "delta_full_minus_subset": float(np.mean(
                [full[f]["psnr_rgb"] - old[o]["psnr_rgb"] for o, f in pairs])),
        }
    out["subset_vs_full_note"] = "mapped by official source index; test_full clean_j = source j+5"
    (ROOT / "rebuttal_runs/E4/uhdm/full_eval_summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
