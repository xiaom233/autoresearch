#!/usr/bin/env python3
"""Extract the 160 SIDD validation blocks omitted from the 1,120 test set.

Existing test_full = keys 160..1279 (scenes 5..39), indexed 0..1119.
Missing = keys 0..159 (scenes 0..4, 32 blocks/scene), saved as clean_1120..1279
so per-image rows do not collide. Also writes a full 1,280 manifest.
Run with /tmp/gdown_env/bin/python (has lmdb).
"""
import hashlib
import io
import json
from pathlib import Path

import lmdb
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SIDD = ROOT / "datasets/E4/sidd"
OUT = SIDD / "test_support_scenes"


def decode(v):
    try:
        return Image.open(io.BytesIO(v)).convert("RGB")
    except Exception:
        a = np.frombuffer(v, dtype=np.uint8)
        for shape in [(256, 256, 3), (512, 512, 3)]:
            if a.size == np.prod(shape):
                return Image.fromarray(a.reshape(shape))
        raise


def sha_img(img):
    return hashlib.sha256(np.array(img, dtype=np.uint8).tobytes()).hexdigest()


def main():
    if OUT.exists() and any(OUT.iterdir()):
        print(f"exists: {OUT} ({len(list(OUT.glob('*.png')))} pngs)")
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        inp = next((SIDD / "lmdb").rglob("input_crops.lmdb"))
        gt = next((SIDD / "lmdb").rglob("gt_crops.lmdb"))
        e_in = lmdb.open(str(inp), readonly=True, lock=False, readahead=False)
        e_gt = lmdb.open(str(gt), readonly=True, lock=False, readahead=False)
        meta = []
        with e_in.begin() as ti, e_gt.begin() as tg:
            for flat in range(160):
                key = f"ValidationBlocksSrgb_{flat}".encode()
                idx = 1120 + flat
                img_in = decode(ti.get(key))
                img_gt = decode(tg.get(key))
                img_in.save(OUT / f"degraded_{idx}.png")
                img_gt.save(OUT / f"clean_{idx}.png")
                meta.append({"index": idx, "key": key.decode(),
                             "scene": flat // 32, "block": flat % 32,
                             "degraded_sha256": sha_img(img_in),
                             "clean_sha256": sha_img(img_gt)})
        (OUT / "manifest.json").write_text(
            json.dumps({"n": len(meta), "pairs": meta}, indent=2, ensure_ascii=False),
            encoding="utf-8")
        print(f"extracted {len(meta)} pairs -> {OUT}")

    # Full 1,280 manifest
    old = json.load(open(SIDD / "test_full_manifest.json"))
    miss = json.load(open(OUT / "manifest.json"))["pairs"]
    entries = [{"index": m["index"], "key": m["key"], "scene": m["scene"],
                "split": "test1120"} for m in old["test"]]
    entries += [{"index": m["index"], "key": m["key"], "scene": m["scene"],
                 "split": "support_scene"} for m in miss]
    entries.sort(key=lambda x: x["index"])
    support_keys = old.get("support_keys", [])
    overlap = [x for x in entries if x["key"] in set(support_keys)]
    full = {
        "n_total": len(entries), "n_scenes": len({x["scene"] for x in entries}),
        "blocks_per_scene": 32, "test": entries,
        "support_keys": support_keys,
        "support_overlap_test": [x["key"] for x in overlap],
        "n_support_overlap": len(overlap),
        "note": "test1120 = keys 160..1279 (scenes 5..39); support_scene = keys 0..159 (scenes 0..4). "
                "Five support keys (one block per support scene) are included in the full 1,280 test; "
                "the other 155 blocks of those scenes were never used as support.",
    }
    (SIDD / "test_all_manifest.json").write_text(
        json.dumps(full, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"full manifest: {full['n_total']} blocks, {full['n_scenes']} scenes, "
          f"support overlap {full['n_support_overlap']} ({full['support_overlap_test']})")


if __name__ == "__main__":
    main()
