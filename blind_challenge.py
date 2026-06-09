#!/usr/bin/env python3
"""
Generate a blind degradation identification challenge.

This script runs as a SUBPROCESS — the main agent must NOT see its internal
state. It picks a random clean image, applies a random degradation pipeline,
and saves:
  - blind_challenge/degraded.png      ← public (agent analyzes this)
  - blind_challenge/clean.png         ← public (reference for the skill)
  - blind_challenge/.ground_truth.json ← HIDDEN (agent must NOT read this)

The agent then uses image-degradation-simulator skill to identify the
degradation blindly, saving the prediction as:
  - blind_challenge/predicted_params.json

Finally, evaluate_blind_challenge.py compares prediction vs ground truth.

Usage:
    uv run blind_challenge.py                  # random image, random degradation
    uv run blind_challenge.py --seed 42        # reproducible challenge
    uv run blind_challenge.py --num-degs 2     # force exactly 2 degradations
"""

import argparse
import json
import os
import random
import sys
import numpy as np
from PIL import Image

# Project imports
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from x_distortion import add_distortion

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = "blind_challenge"
DEGRADED_NAME = "degraded.png"
CLEAN_NAME = "clean.png"
GROUND_TRUTH_NAME = ".ground_truth.json"      # hidden — agent must NOT read
PREDICTED_NAME = "predicted_params.json"       # agent writes this

# Clean image sources — pick randomly from validation sets
VAL_DIRS = [
    "datasets/Set5/GTmod4",
    "datasets/Set14/GTmod4",
    "datasets/B100/GTmod4",
    "datasets/Urban100/GTmod4",
    "datasets/Manga109/GTmod4",
    "datasets/DIV2K/DIV2K_valid_HR",
]

# Degradation categories matching train.py's random pipeline
DEG_CATEGORIES = {
    "blur": [
        "blur_gaussian", "blur_motion", "blur_glass",
        "blur_lens", "blur_zoom", "blur_jitter",
    ],
    "noise": [
        "noise_gaussian_RGB", "noise_gaussian_YCrCb", "noise_speckle",
        "noise_spatially_correlated", "noise_poisson", "noise_impulse",
    ],
    "compression": [
        "compression_jpeg", "compression_jpeg_2000",
    ],
}


def scan_images(directory):
    """Return sorted list of absolute image paths in a directory."""
    exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    paths = []
    try:
        for entry in os.scandir(directory):
            if entry.is_file():
                ext = os.path.splitext(entry.name)[1].lower()
                if ext in exts:
                    paths.append(entry.path)
    except (FileNotFoundError, PermissionError):
        pass
    return sorted(paths)


def pick_random_image(val_dirs):
    """Pick a random clean image from available validation directories."""
    all_paths = []
    for d in val_dirs:
        all_paths.extend(scan_images(d))
    if not all_paths:
        # Fallback: use prepare's scandir
        from prepare import scandir as _scandir
        for d in val_dirs:
            all_paths.extend(_scandir(d, full_path=True))
    if not all_paths:
        raise FileNotFoundError(f"No images found in: {val_dirs}")
    return random.choice(all_paths)


def generate_random_pipeline(num_degs=None):
    """Generate a random degradation pipeline (blur/noise/compression categories).

    Args:
        num_degs: if None, randomly choose 1/2/3 with equal probability.
                  if int, use exactly that many degradations.

    Returns:
        list of dicts: [{"step": 1, "category": "blur", "function": "...",
                         "severity": 3}, ...]
    """
    if num_degs is None:
        n = random.choices([1, 2, 3], weights=[0.33, 0.33, 0.34])[0]
    else:
        n = max(1, min(num_degs, len(DEG_CATEGORIES)))

    cats = list(DEG_CATEGORIES.keys())
    pipeline = []
    used_cats = set()    # 禁止同一类别出现两次
    used_funcs = set()   # 禁止同一函数出现两次
    for step_idx in range(n):
        available = [c for c in cats if c not in used_cats]
        if not available:
            break  # 没有未使用的类别了
        cat = random.choice(available)
        funcs = [f for f in DEG_CATEGORIES[cat] if f not in used_funcs]
        func = random.choice(funcs)
        sev = random.randint(1, 5)
        pipeline.append({
            "step": step_idx + 1,
            "category": cat,
            "function": func,
            "severity": sev,
        })
        used_cats.add(cat)
        used_funcs.add(func)
    return pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Generate a blind degradation identification challenge")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--num-degs", type=int, default=None,
                        help="Force exact number of degradations (1-3)")
    parser.add_argument("--clean-image", type=str, default=None,
                        help="Use specific clean image (otherwise picks randomly)")
    parser.add_argument("--ref-image", type=str, default=None,
                        help="Clean REFERENCE image for skill (MUST be different from --clean-image)")
    parser.add_argument("--same-image", action="store_true",
                        help="Same-image mode: clean reference = original undegraded image (enables pixel-level calibration)")
    parser.add_argument("--target-only", action="store_true",
                        help="Target-only mode: NO clean reference provided (hardest test)")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--export-to", type=str, default=None,
                        help="复制挑战文件到指定目录（如 exp3/blind_challenge/）")
    parser.add_argument("--params-output", type=str, default=None,
                        help="静默导出训练兼容的 params.json（仅 function+severity，不含 step/category）")
    parser.add_argument("--quiet", action="store_true",
                        help="静默模式：仅输出 challenge_id 和路径，不打印 seed")
    args = parser.parse_args()

    # Seed everything
    seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
    random.seed(seed)
    np.random.seed(seed)

    # Pick target image (to be degraded)
    if args.clean_image:
        target_path = args.clean_image
    else:
        target_path = pick_random_image(VAL_DIRS)

    # Pick reference image (for skill's simulation — must be DIFFERENT)
    if args.same_image:
        ref_path = target_path  # same source, undegraded
    elif args.target_only:
        ref_path = None
    elif args.ref_image:
        ref_path = args.ref_image
        # Same-image mode: using the same source as reference is valid.
        # This enables pixel-level calibration of content-dependent metrics.
    else:
        # Pick a DIFFERENT random image from a DIFFERENT dataset
        ref_path = pick_random_image(VAL_DIRS)
        attempts = 0
        while ref_path == target_path and attempts < 20:
            ref_path = pick_random_image(VAL_DIRS)
            attempts += 1
        if ref_path == target_path:
            print("ERROR: Could not find a different reference image.")
            sys.exit(1)

    # Generate degradation pipeline
    pipeline = generate_random_pipeline(args.num_degs)

    # Load and degrade target image
    target_img = np.array(Image.open(target_path).convert("RGB"), dtype=np.uint8)
    degraded = target_img.copy()
    for step in pipeline:
        degraded = add_distortion(degraded, severity=step["severity"],
                                  distortion_name=step["function"])

    # Setup output directory
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Save degraded image (public)
    degraded_path = os.path.join(out_dir, DEGRADED_NAME)
    Image.fromarray(degraded).save(degraded_path)

    # Save clean reference (public — DIFFERENT image from target, for skill simulation)
    if ref_path:
        ref_img = np.array(Image.open(ref_path).convert("RGB"), dtype=np.uint8)
        clean_out_path = os.path.join(out_dir, CLEAN_NAME)
        Image.fromarray(ref_img).save(clean_out_path)
    else:
        clean_out_path = "(target-only mode — no reference provided)"

    # Save ground truth (HIDDEN — agent must NOT read this file)
    ground_truth = {
        "challenge_id": f"blind_{seed:08d}",
        "target_source": os.path.basename(target_path),
        "reference_source": os.path.basename(ref_path) if ref_path else "none (target-only)",
        "pipeline": pipeline,
        "mode": "target-only" if args.target_only else "cross-image",
    }
    truth_path = os.path.join(out_dir, GROUND_TRUTH_NAME)
    with open(truth_path, 'w') as f:
        json.dump(ground_truth, f, indent=2)

    # Export to target directory if requested
    if args.export_to:
        import shutil
        # Skip if export-to is the same as output-dir (already saved above)
        if os.path.abspath(args.export_to) != os.path.abspath(out_dir):
            os.makedirs(args.export_to, exist_ok=True)
            for fname in [DEGRADED_NAME, CLEAN_NAME, GROUND_TRUTH_NAME]:
                src = os.path.join(out_dir, fname)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(args.export_to, fname))
        # Also copy to blind_challenge/ for skill access (only if different dir)
        skill_dir = "blind_challenge"
        if os.path.abspath(out_dir) != os.path.abspath(skill_dir):
            os.makedirs(skill_dir, exist_ok=True)
            for fname in [DEGRADED_NAME, CLEAN_NAME, GROUND_TRUTH_NAME]:
                src = os.path.join(out_dir, fname)
                if os.path.exists(src):
                    dst = os.path.join(skill_dir, fname)
                    if os.path.abspath(src) != os.path.abspath(dst):
                        shutil.copy(src, dst)

    # Export training-compatible params.json (silently, without step/category)
    if args.params_output:
        params_out_dir = os.path.dirname(args.params_output)
        if params_out_dir:
            os.makedirs(params_out_dir, exist_ok=True)
        training_params = {
            "pipeline": [
                {"function": step["function"], "severity": step["severity"]}
                for step in pipeline
            ]
        }
        with open(args.params_output, 'w') as f:
            json.dump(training_params, f)
        # NEVER print pipeline contents — this path is recorded in public info below

    # Print public info ONLY — NEVER print degradation details, num_degs, or pipeline info
    if not args.quiet:
        print(f"seed:           {seed}")
    print(f"challenge_id: {ground_truth['challenge_id']}")
    print(f"target: {os.path.abspath(degraded_path)}")
    print(f"ref:    {os.path.abspath(clean_out_path) if ref_path else 'none (target-only)'}")
    print(f"export: {os.path.abspath(args.export_to) if args.export_to else 'none'}")
    if args.params_output:
        print(f"params: {os.path.abspath(args.params_output)}")
    print(f"READY")


if __name__ == "__main__":
    main()
