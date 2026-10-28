#!/usr/bin/env python3
"""Map each scenario's 5 support images to DIV2K test filenames.

The challenge generator samples support images from DIV2K_valid_HR, which is
also the DIV2K test set. This records the exact per-scenario overlap so final
aggregation can exclude the support images.
"""
import glob
import hashlib
import json
import os
from collections import Counter

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VAL = os.path.join(ROOT, "datasets/DIV2K/DIV2K_valid_HR")
CH = os.path.join(ROOT, "exp41/challenges/phase4")
OUT = os.path.join(ROOT, "rebuttal_runs/A0/support_test_overlap.json")
CALIB = {f"{i:04d}" for i in range(801, 811)}


def pixel_hash(path):
    a = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    return hashlib.sha256(a.tobytes()).hexdigest()


def main():
    valmap = {}
    for p in sorted(glob.glob(os.path.join(VAL, "*.png"))):
        valmap[pixel_hash(p)] = os.path.splitext(os.path.basename(p))[0]
    per = {}
    usage = Counter()
    for d in sorted(glob.glob(os.path.join(CH, "blind_*"))):
        sid = os.path.basename(d)
        hits = []
        for i in range(5):
            name = valmap.get(pixel_hash(os.path.join(d, f"clean_{i}.png")))
            hits.append(name)
            if name:
                usage[name] += 1
        per[sid] = hits
    all_names = [x for v in per.values() for x in v if x]
    report = {
        "source": "exp41 challenges clean_i sampled from datasets/DIV2K/DIV2K_valid_HR",
        "div2k_test_images": len(valmap),
        "scenarios": len(per),
        "support_per_scenario": 5,
        "scenarios_all_5_matched": sum(1 for v in per.values() if all(v)),
        "unique_div2k_test_images_used_as_support": len(usage),
        "total_support_draws": len(all_names),
        "support_usage_counts": dict(usage.most_common()),
        "per_scenario_support_images": per,
        "per_scenario_div2k_test_after_excluding_support": {
            sid: len(valmap) - sum(1 for x in v if x) for sid, v in per.items()},
        "support_images_also_in_calibration_0801_0810": sorted(set(all_names) & CALIB),
        "calibration_images": sorted(CALIB),
        "note": "LSDIR support/test overlap is zero because support is drawn only from DIV2K_valid.",
    }
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: report[k] for k in [
        "div2k_test_images", "scenarios_all_5_matched", "unique_div2k_test_images_used_as_support",
        "total_support_draws", "support_images_also_in_calibration_0801_0810"]}, indent=2))
    print("saved", OUT)


if __name__ == "__main__":
    main()
