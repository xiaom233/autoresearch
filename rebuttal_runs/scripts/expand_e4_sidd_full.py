#!/usr/bin/env python3
"""Expand SIDD E4 test to all 32 blocks of the 35 non-support scenes."""
import io
import json
import zipfile
from pathlib import Path

import lmdb
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "datasets/E4/sidd"
ZIP = OUT / "sidd_val_lmdb.zip"


def decode(v):
    try:
        return Image.open(io.BytesIO(v)).convert("RGB")
    except Exception:
        a = np.frombuffer(v, dtype=np.uint8)
        for shape in [(256, 256, 3), (512, 512, 3)]:
            if a.size == np.prod(shape):
                return Image.fromarray(a.reshape(shape))
        raise


def main():
    raw = OUT / "lmdb"
    raw.mkdir(parents=True, exist_ok=True)
    if not any(raw.rglob("input_crops.lmdb")):
        with zipfile.ZipFile(ZIP) as z:
            z.extractall(raw)
    inp = next(raw.rglob("input_crops.lmdb"))
    gt = next(raw.rglob("gt_crops.lmdb"))
    e_in = lmdb.open(str(inp), readonly=True, lock=False, readahead=False)
    e_gt = lmdb.open(str(gt), readonly=True, lock=False, readahead=False)
    with e_in.begin() as ti, e_gt.begin() as tg:
        keys = []
        for k, _ in ti.cursor():
            ks = k.decode() if isinstance(k, bytes) else str(k)
            if tg.get(k) is not None:
                keys.append(ks)
        def flat(k):
            return int(k.rsplit("_", 1)[-1])
        keys.sort(key=flat)
        # support: first block of scenes 0-4; test: all blocks of scenes 5-39
        support = []
        test = []
        for k in keys:
            scene = flat(k) // 32
            if scene < 5:
                if flat(k) % 32 == 0:
                    support.append((k, scene))
            else:
                test.append((k, scene))
        out = OUT / "test_full"
        out.mkdir(parents=True, exist_ok=True)
        meta = []
        for i, (k, scene) in enumerate(test):
            kb = k.encode()
            decode(ti.get(kb)).save(out / f"degraded_{i}.png")
            decode(tg.get(kb)).save(out / f"clean_{i}.png")
            meta.append({"index": i, "key": k, "scene": scene})
        report = {"support_keys": [k for k, _ in support],
                  "n_test": len(test), "n_test_scenes": len({s for _, s in test}),
                  "blocks_per_scene": 32, "test": meta}
        (OUT / "test_full_manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"test_full: {len(test)} blocks, {len({s for _, s in test})} scenes")


if __name__ == "__main__":
    main()
