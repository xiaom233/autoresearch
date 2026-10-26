#!/usr/bin/env python3
"""Materialize one fixed LQ realization for the 7 nondeterministic scenarios.

Reason: noise_impulse (skimage fresh RNG) and blur_glass (numba private RNG)
ignore np.random.seed, so earlier per-method evaluations used different LQ
pixels for these scenarios. Saving the LQ once makes every later evaluation
compare on identical pixels.
"""
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rebuttal_runs/scripts"))
sys.path.insert(0, str(ROOT))
from eval_checkpoint import load_rgb, make_lq  # noqa: E402

SCENARIOS = ["blind_0003", "blind_0005", "blind_0017", "blind_0018",
             "blind_0020", "blind_0022", "blind_0023"]
DATASETS = {
    "DIV2K": ROOT / "datasets/DIV2K/DIV2K_valid_HR",
    "LSDIR": ROOT / "datasets/LSDIR/val1/HR/val",
}
OUT = ROOT / "rebuttal_runs/E1E2_fixed_lq"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    for sid in SCENARIOS:
        pipe = json.load(open(ROOT / f"exp41/degradation_gt/{sid}_params.json"))["pipeline"]
        base = OUT / sid
        index = []
        for ds, d in DATASETS.items():
            outdir = base / ds
            outdir.mkdir(parents=True, exist_ok=True)
            for i, p in enumerate(sorted(glob.glob(str(d / "*.png")))):
                image_id = Path(p).stem
                clean = load_rgb(p)
                lq = make_lq(clean, sid, f"{ds}/{image_id}", pipe, "v1")
                lq_path = outdir / f"degraded_{i}.png"
                Image.fromarray(lq).save(lq_path, compress_level=1)
                index.append({
                    "index": i, "dataset": ds, "image_id": image_id,
                    "clean_path": str(Path(p).relative_to(ROOT)),
                    "lq_path": str(lq_path.relative_to(ROOT)),
                    "lq_sha256": sha(lq.tobytes()),
                    "image_sha256": sha(clean.tobytes()),
                })
        (base / "index.json").write_text(json.dumps({
            "scenario_id": sid, "pipeline": pipe,
            "note": "one fixed LQ realization shared by every method",
            "pairs": index}, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[fixed-lq] {sid}: {len(index)} pairs", flush=True)


if __name__ == "__main__":
    main()
