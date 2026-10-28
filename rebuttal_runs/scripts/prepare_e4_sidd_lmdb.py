#!/usr/bin/env python3
"""Extract real SIDD validation pairs from NAFNet's lmdb package.

Five support pairs come from five distinct scenes; test pairs come from the
remaining scenes (one block per scene), so support/test content is disjoint.
Run with /tmp/gdown_env/bin/python (has lmdb/numpy/Pillow).
"""
import argparse
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path

import lmdb
import numpy as np
from PIL import Image


def decode(v):
    try:
        return Image.open(io.BytesIO(v)).convert("RGB")
    except Exception:
        a = np.frombuffer(v, dtype=np.uint8)
        # try common raw shapes
        for shape in [(256, 256, 3), (512, 512, 3)]:
            if a.size == np.prod(shape):
                return Image.fromarray(a.reshape(shape))
        raise


def sha_img(img):
    return hashlib.sha256(np.array(img, dtype=np.uint8).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()
    out = Path(args.out_root)
    raw = out / "lmdb"
    raw.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip) as z:
        z.extractall(raw)
    inp = next(raw.rglob("input_crops.lmdb"))
    gt = next(raw.rglob("gt_crops.lmdb"))
    e_in = lmdb.open(str(inp), readonly=True, lock=False, readahead=False)
    e_gt = lmdb.open(str(gt), readonly=True, lock=False, readahead=False)
    with e_in.begin() as ti, e_gt.begin() as tg:
        keys_in = [k.decode() if isinstance(k, bytes) else str(k) for k, _ in ti.cursor()]
        keys_gt = set(k.decode() if isinstance(k, bytes) else str(k) for k, _ in tg.cursor())
        keys = [k for k in keys_in if k in keys_gt]
        print("keys", len(keys), "sample", keys[:5])
        # NAFNet keys are ValidationBlocksSrgb_<flat_index>; the official
        # validation layout is 40 scenes x 32 blocks, scene-major.
        def flat_num(k):
            try:
                return int(k.rsplit("_", 1)[-1])
            except ValueError:
                return 0

        def scene(k):
            if "_" in k and k.rsplit("_", 1)[-1].isdigit():
                return str(flat_num(k) // 32)
            return k.split("-")[0] if "-" in k else k
        by_scene = {}
        for k in keys:
            by_scene.setdefault(scene(k), []).append(k)
        scenes = sorted(by_scene, key=lambda s: int(s) if s.isdigit() else s)
        print("scenes", len(scenes), scenes[:10])
        support_scenes = scenes[:5]
        test_scenes = scenes[5:]
        support_keys = [sorted(by_scene[s], key=flat_num)[0] for s in support_scenes]
        test_keys = [sorted(by_scene[s], key=flat_num)[0] for s in test_scenes]

        def save(keys, sub):
            d = out / sub
            d.mkdir(parents=True, exist_ok=True)
            meta = []
            for i, k in enumerate(keys):
                img_in = decode(ti.get(k.encode() if isinstance(keys_in[0], str) else k))
                img_gt = decode(tg.get(k.encode() if isinstance(keys_in[0], str) else k))
                img_in.save(d / f"degraded_{i}.png")
                img_gt.save(d / f"clean_{i}.png")
                meta.append({"index": i, "key": k, "scene": scene(k),
                             "degraded_sha256": sha_img(img_in),
                             "clean_sha256": sha_img(img_gt)})
            return meta

        support_meta = save(support_keys, "support")
        test_meta = save(test_keys, "test")
    report = {
        "source": "NAFNet SIDD validation lmdb",
        "support_keys": [m["key"] for m in support_meta],
        "test_keys": [m["key"] for m in test_meta],
        "support_scenes": sorted({m["scene"] for m in support_meta}),
        "test_scenes": sorted({m["scene"] for m in test_meta}),
        "scene_disjoint": not (set(m["scene"] for m in support_meta) &
                               set(m["scene"] for m in test_meta)),
        "support": support_meta, "test": test_meta,
    }
    (out / "manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in ["support_keys", "test_keys", "scene_disjoint"]},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
