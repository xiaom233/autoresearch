#!/usr/bin/env python3
"""Pre-register E6-B scenarios and materialize 3 independent support sets."""
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from x_distortion import add_distortion  # noqa: E402

SCENARIOS = ["blind_0006", "blind_0008", "blind_0010",
             "blind_0013", "blind_0017", "blind_0024"]
OUT = ROOT / "rebuttal_runs/E6"
POOL = ROOT / "datasets/DIV2K/DIV2K_train_HR"
SEED = 20261026


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def stable_seed(*parts):
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:16], 16) % (2 ** 32)


def load_rgb(p):
    return np.array(Image.open(p).convert("RGB"), dtype=np.uint8)


def make_lq(clean, key, pipeline):
    seed = stable_seed(key, "e6-support-v1")
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    img = clean.copy()
    for step in pipeline:
        img = add_distortion(img, severity=int(step["severity"]),
                             distortion_name=step["function"])
    return img


def main():
    imgs = sorted(POOL.glob("*.png"))
    if len(imgs) < 90:
        raise RuntimeError(f"need 90 DIV2K train images, got {len(imgs)}")
    rng = random.Random(SEED)
    order = imgs[:]
    rng.shuffle(order)
    # 6 scenarios x 15 images, then split into 3 disjoint support sets
    report = {"seed": SEED, "pool": str(POOL.relative_to(ROOT)),
              "scenarios": SCENARIOS,
              "selection": "global shuffled DIV2K_train_HR; 15 unique images per scenario; 3 disjoint sets of 5",
              "sets": {}}
    cursor = 0
    for si, sid in enumerate(SCENARIOS):
        pipeline = json.load(open(ROOT / f"exp41/degradation_gt/{sid}_params.json"))["pipeline"]
        selected = order[cursor:cursor + 15]
        cursor += 15
        report["sets"][sid] = {"pipeline": pipeline, "groups": {}}
        for k in range(1, 4):
            group = selected[(k - 1) * 5:k * 5]
            d = OUT / "support" / f"set{k}" / sid
            d.mkdir(parents=True, exist_ok=True)
            meta = []
            for i, p in enumerate(group):
                clean = load_rgb(p)
                lq = make_lq(clean, f"{sid}/set{k}/{p.stem}", pipeline)
                cpath = d / f"clean_{i}.png"
                lpath = d / f"degraded_{i}.png"
                Image.fromarray(clean).save(cpath)
                Image.fromarray(lq).save(lpath)
                meta.append({"index": i, "image_id": p.stem,
                             "clean_sha256": sha_file(cpath),
                             "degraded_sha256": sha_file(lpath)})
            report["sets"][sid]["groups"][f"set{k}"] = meta
            print(f"[support] {sid} set{k}: {[m['image_id'] for m in meta]}", flush=True)
    (OUT / "support_manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[support] wrote {OUT/'support_manifest.json'}")


if __name__ == "__main__":
    main()
