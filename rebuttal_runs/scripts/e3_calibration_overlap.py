#!/usr/bin/env python3
"""E3: static calibration-space audit and overlap with the exp41 evaluation.

Reads calibration scripts (constants only) and exp41 GT chains on the
evaluation side. Only aggregate overlap is written; no per-scenario GT is
exported to the decision side.
"""
import ast
import glob
import itertools
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def consts(path, names):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in names:
                    out[t.id] = ast.literal_eval(node.value)
    return out


def chains_exact(gt):
    return {tuple(gt)}


def main():
    out_dir = ROOT / "rebuttal_runs" / "E3"
    out_dir.mkdir(parents=True, exist_ok=True)

    b = consts(ROOT / "scripts/calibrate_blur_kernel.py",
               {"BLUR_TYPES", "BLUR_SEVS", "NOISE_TYPES", "NOISE_SEVS", "ORDERS", "SEEDS", "CLEAN_SRC"})
    j = consts(ROOT / "scripts/calibrate_jpeg_thresholds.py",
               {"JPEG_SEVS", "NOISE_TYPES", "NOISE_SEVS", "BLUR_TYPES", "BLUR_SEVS", "ORDERS", "SEEDS", "VAL_DIR"})
    j["IMAGES_FULL"] = sorted(glob.glob(str(ROOT / j["VAL_DIR"] / "*.png")))[:10]
    j["IMAGES"] = j["IMAGES_FULL"][:5]

    blur_chains = set()
    for bt, bs, nt, ns in itertools.product(b["BLUR_TYPES"], b["BLUR_SEVS"], b["NOISE_TYPES"], b["NOISE_SEVS"]):
        blur_chains.add(((bt, bs), (nt, ns)))
        blur_chains.add(((nt, ns), (bt, bs)))
    assert len(blur_chains) == 128, len(blur_chains)

    jpeg_chains = set()
    for js in j["JPEG_SEVS"]:
        jpeg_chains.add((("compression_jpeg", js),))
    for js, nt, ns in itertools.product(j["JPEG_SEVS"], j["NOISE_TYPES"], j["NOISE_SEVS"]):
        jpeg_chains.add(((nt, ns), ("compression_jpeg", js)))
    for js, bt, bs in itertools.product(j["JPEG_SEVS"], j["BLUR_TYPES"], j["BLUR_SEVS"]):
        jpeg_chains.add(((bt, bs), ("compression_jpeg", js)))
    for js, bt, bs, nt, ns, order in itertools.product(
            j["JPEG_SEVS"], j["BLUR_TYPES"], j["BLUR_SEVS"], j["NOISE_TYPES"], j["NOISE_SEVS"], j["ORDERS"]):
        pair = ((bt, bs), (nt, ns)) if order == "blur_first" else ((nt, ns), (bt, bs))
        jpeg_chains.add(pair + (("compression_jpeg", js),))
    assert len(jpeg_chains) == 1320, len(jpeg_chains)

    exp41 = {}
    for p in sorted((ROOT / "exp41/degradation_gt").glob("*_params.json")):
        d = json.load(open(p))
        exp41[p.stem.replace("_params", "")] = tuple((s["function"], int(s["severity"])) for s in d["pipeline"])

    # evaluation-side aggregate overlap only
    b_inter = [sid for sid, c in exp41.items() if c in blur_chains]
    j_inter = [sid for sid, c in exp41.items() if c in jpeg_chains]
    u_inter = sorted(set(b_inter) | set(j_inter))

    # image-level overlap: calibration scripts use DIV2K_valid_HR images
    div2k = sorted(Path(p).stem for p in glob.glob(str(ROOT / "datasets/DIV2K/DIV2K_valid_HR/*.png")))
    blur_img = Path(b["CLEAN_SRC"]).stem
    jpeg_imgs = [Path(p).stem for p in j["IMAGES_FULL"]] + [Path(p).stem for p in j["IMAGES"]]
    jpeg_imgs = sorted(set(jpeg_imgs))
    calib_imgs = sorted(set([blur_img] + jpeg_imgs))
    img_overlap = sorted(set(div2k) & set(calib_imgs))

    report = {
        "generator_version": {
            "blind_challenge_last_commit": "d80790f (2026-06-29)",
            "changes_after_that_until_HEAD": False,
            "exp41_challenge_mtime": "2026-07-14",
            "conclusion": "current generator matches exp41 generation",
        },
        "space": json.load(open(ROOT / "rebuttal_plan/degradation_space_counts.json")),
        "calibration": {
            "blur": {
                "unique_ordered_chains": len(blur_chains),
                "cases": len(blur_chains) * len(b["SEEDS"]),
                "seeds": len(b["SEEDS"]),
                "image": blur_img,
                "note": "128 configs x 5 seeds, one 256x256 center crop",
            },
            "jpeg": {
                "unique_ordered_chains": len(jpeg_chains),
                "cases": (len(j["JPEG_SEVS"]) * 10 * len(j["SEEDS"])
                          + len(j["JPEG_SEVS"]) * len(j["NOISE_TYPES"]) * len(j["NOISE_SEVS"]) * 10 * len(j["SEEDS"])
                          + len(j["JPEG_SEVS"]) * len(j["BLUR_TYPES"]) * len(j["BLUR_SEVS"]) * 10 * len(j["SEEDS"])
                          + len(j["JPEG_SEVS"]) * len(j["BLUR_TYPES"]) * len(j["BLUR_SEVS"])
                          * len(j["NOISE_TYPES"]) * len(j["NOISE_SEVS"]) * 5 * len(j["SEEDS"]) * len(j["ORDERS"])),
                "seeds": len(j["SEEDS"]),
                "images": jpeg_imgs,
                "note": "1320 unique configs; 14,400 cases by declared image/seed plan",
            },
            "historic_3840_manifest": None,
        },
        "chain_overlap_exp41": {
            "blur_calibration": len(b_inter),
            "jpeg_calibration": len(j_inter),
            "union": len(u_inter),
            "exp41_scenarios": len(exp41),
            "reported_3840_upper_bound_pct": round(3840 / 144615 * 100, 2),
        },
        "image_overlap": {
            "test_div2k_images": len(div2k),
            "calibration_images": calib_imgs,
            "overlap_count": len(img_overlap),
            "overlap_images": img_overlap,
            "note": "calibration images 0801-0810 are inside the reported DIV2K test split",
        },
        "limitations": [
            "3840-case manifest not recovered; its unique-config count is UNKNOWN.",
            "The current calibration scripts use DIV2K test images; calibration/test image independence is not established.",
            "Calibration thresholds may have been inspected on those images; we cannot rule out test-informed tuning.",
            "Large symbolic space does not prove calibration/evaluation independence.",
        ],
    }
    out = out_dir / "E3_calibration_overlap.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
