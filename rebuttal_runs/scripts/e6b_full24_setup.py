#!/usr/bin/env python3
"""Materialize E6-B full24: 24 scenarios x 2 random 5-image support sets.

Sampling: for each (scenario, setA/setB), independently sample 5 images from
DIV2K_train_HR with a recorded seed (overlap allowed and reported). Degraded
images use the scenario's GT pipeline; only the saved PNGs and hashes are
exposed to agents. GT params live in support_plan_gt.json (chmod 600).
"""
import hashlib
import json
import os
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rebuttal_runs/scripts"))
from eval_checkpoint import load_rgb, make_lq  # noqa: E402

OUT = ROOT / "rebuttal_runs/E6B_full24"
POOL = ROOT / "datasets/DIV2K/DIV2K_train_HR"
SCENARIOS = [f"blind_{i:04d}" for i in range(1, 25)]
SETS = ["setA", "setB"]
SEED = 20261030
LQ_SALT = "e6b-full-v1"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def stable_seed(*parts):
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:16], 16) % (2 ** 32)


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    challenges = OUT / "challenges"
    if challenges.exists() and any(challenges.iterdir()):
        sys.exit(f"refusing to overwrite non-empty {challenges}")
    challenges.mkdir(parents=True, exist_ok=True)

    imgs = sorted(POOL.glob("*.png"))
    if len(imgs) < 5:
        sys.exit(f"support pool too small: {len(imgs)}")
    plan = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "pool": str(POOL.relative_to(ROOT)),
        "sampling": "per (scenario,set), random.Random(stable_seed(SEED,sid,set)).sample(pool,5); overlap allowed",
        "scenarios": {},
    }
    overlap = {}
    for sid in SCENARIOS:
        pipe = json.load(open(ROOT / f"exp41/degradation_gt/{sid}_params.json"))["pipeline"]
        plan["scenarios"][sid] = {"pipeline": pipe, "sets": {}}
        ids = {}
        for set_name in SETS:
            rng = random.Random(stable_seed(SEED, sid, set_name))
            picked = rng.sample(imgs, 5)
            ids[set_name] = {p.stem for p in picked}
            d = challenges / f"{sid}_{set_name}"
            d.mkdir(parents=True, exist_ok=False)
            pairs = []
            for i, src in enumerate(picked):
                clean = load_rgb(src)
                lq = make_lq(clean, sid, f"{set_name}/{src.stem}", pipe, LQ_SALT)
                cpath = d / f"clean_{i}.png"
                lpath = d / f"degraded_{i}.png"
                link_or_copy(src, cpath)
                Image.fromarray(lq).save(lpath, compress_level=1)
                pairs.append({
                    "index": i, "image_id": src.stem,
                    "source": str(src.relative_to(ROOT)),
                    "clean_sha256": sha(cpath), "degraded_sha256": sha(lpath),
                })
            link_or_copy(d / "clean_0.png", d / "clean.png")
            link_or_copy(d / "degraded_0.png", d / "degraded.png")
            manifest = {
                "scenario_id": sid, "set": set_name,
                "sampling_seed": stable_seed(SEED, sid, set_name),
                "pairs": pairs,
                "aliases": {"clean.png": pairs[0]["clean_sha256"],
                            "degraded.png": pairs[0]["degraded_sha256"]},
            }
            (d / "support_manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
            plan["scenarios"][sid]["sets"][set_name] = pairs
            print(f"[ok] {sid}_{set_name}: {[p.stem for p in picked]}", flush=True)
        overlap[sid] = len(ids["setA"] & ids["setB"])
    (OUT / "support_plan_gt.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    os.chmod(OUT / "support_plan_gt.json", 0o600)
    ov = {
        "per_scenario_setA_setB_overlap": overlap,
        "mean_overlap": float(np.mean(list(overlap.values()))),
        "scenarios_with_overlap": [s for s, v in overlap.items() if v],
        "note": "random independent draws; overlap allowed and reported",
    }
    (OUT / "support_overlap.json").write_text(
        json.dumps(ov, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] 48 challenge dirs; mean setA∩setB overlap {ov['mean_overlap']:.2f}/5; "
          f"overlap scenarios {ov['scenarios_with_overlap']}")


if __name__ == "__main__":
    main()
