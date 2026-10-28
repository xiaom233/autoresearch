#!/usr/bin/env python3
"""验证 run_full_analysis.py 新增 3 个信号的有效性.

优化: 多进程并行 + 图像缩放到 MAX_SIZE (保持统计量分布)

用法:
    .venv/bin/python3 scripts/validate_new_signals.py --num-images 10 --workers 8
"""
import argparse
import json
import os
import sys
import random
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

MAX_SIZE = 512  # 最大边长, 保持噪声/blur/JPEG 统计特征同时加速 FFT/DCT

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(THIS_DIR, '..')
SKILL_DIR = os.path.join(ROOT, '.claude/skills/image-degradation-simulator/scripts')


def resize_if_needed(img, max_size=MAX_SIZE):
    """等比缩放至 max_size, 长边不超过 max_size."""
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img
    scale = max_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    # PIL 高质量缩放
    from PIL import Image as PILImage
    if len(img.shape) == 3:
        pil = PILImage.fromarray(img)
        pil = pil.resize((new_w, new_h), PILImage.LANCZOS)
        return np.array(pil, dtype=np.uint8)
    else:
        pil = PILImage.fromarray(img)
        pil = pil.resize((new_w, new_h), PILImage.LANCZOS)
        return np.array(pil, dtype=np.uint8)


def pick_images(n, source_dir="datasets/DIV2K/DIV2K_valid_HR"):
    """随机选取 n 张图像."""
    exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    paths = []
    for entry in os.scandir(source_dir):
        if entry.is_file():
            ext = os.path.splitext(entry.name)[1].lower()
            if ext in exts:
                paths.append(entry.path)
    if len(paths) < n:
        return paths
    return random.sample(paths, n)


# ================================================================
# Worker 函数 (在子进程中运行)
# ================================================================

def _worker_single(args):
    """实验 1 worker: (img_idx, img_path, label, deg_fn, sev)"""
    img_idx, img_path, label, deg_fn, sev = args

    # 每个子进程独立导入 (避免 pickle 问题)
    sys.path.insert(0, ROOT)
    sys.path.insert(0, SKILL_DIR)
    from x_distortion import add_distortion
    from run_full_analysis import run_full_analysis
    from PIL import Image

    clean = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    clean = resize_if_needed(clean)
    degraded = add_distortion(clean.copy(), severity=sev, distortion_name=deg_fn)

    tmp_clean = f"/tmp/vsig1_c_{os.getpid()}_{img_idx}.png"
    tmp_degraded = f"/tmp/vsig1_d_{os.getpid()}_{img_idx}.png"
    Image.fromarray(clean).save(tmp_clean)
    Image.fromarray(degraded).save(tmp_degraded)

    try:
        report = run_full_analysis(tmp_degraded, tmp_clean)
        s2 = report.get("step2_noise", {})
        s4 = report.get("step4_global", {})
        signals = {
            "variance_ratio": s4.get("signals", {}).get("variance_ratio"),
            "residual_intensity_corr": s2.get("signals", {}).get("residual_intensity_corr"),
            "laplacian_energy_ratio": s4.get("signals", {}).get("laplacian_energy_ratio"),
            "gamma_suspect_from_noise": s2.get("signals", {}).get("gamma_suspect_from_noise", False),
            "gamma_suspect": s4.get("signals", {}).get("gamma_suspect", False),
            "oversharpen_detected_raw": s4.get("signals", {}).get("oversharpen_detected_raw", False),
            "brightness_detected": "BRIGHTNESS" in s4.get("verdict", ""),
            "contrast_from_variance": s4.get("signals", {}).get("contrast_from_variance", False),
            "noise_verdict": s2.get("verdict", "NO_NOISE"),
            "global_verdict": s4.get("verdict", "NO_GLOBAL"),
        }
    except Exception as e:
        signals = {"error": str(e)}
    finally:
        for f in [tmp_clean, tmp_degraded]:
            if os.path.exists(f):
                os.remove(f)

    return (label, sev, img_idx, signals)


def _worker_composite(args):
    """实验 2/3 worker: (img_idx, img_path, label, degradations)"""
    img_idx, img_path, label, degradations = args

    sys.path.insert(0, ROOT)
    sys.path.insert(0, SKILL_DIR)
    from x_distortion import add_distortion
    from run_full_analysis import run_full_analysis
    from PIL import Image

    clean = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    clean = resize_if_needed(clean)
    degraded = clean.copy()
    for fn, sev in degradations:
        degraded = add_distortion(degraded, severity=sev, distortion_name=fn)

    tmp_clean = f"/tmp/vsig_c_{os.getpid()}_{img_idx}.png"
    tmp_degraded = f"/tmp/vsig_d_{os.getpid()}_{img_idx}.png"
    Image.fromarray(clean).save(tmp_clean)
    Image.fromarray(degraded).save(tmp_degraded)

    try:
        report = run_full_analysis(tmp_degraded, tmp_clean)
        s2 = report.get("step2_noise", {})
        s4 = report.get("step4_global", {})
        signals = {
            "variance_ratio": s4.get("signals", {}).get("variance_ratio"),
            "residual_intensity_corr": s2.get("signals", {}).get("residual_intensity_corr"),
            "laplacian_energy_ratio": s4.get("signals", {}).get("laplacian_energy_ratio"),
            "gamma_suspect_from_noise": s2.get("signals", {}).get("gamma_suspect_from_noise", False),
            "gamma_suspect": s4.get("signals", {}).get("gamma_suspect", False),
            "oversharpen_detected_raw": s4.get("signals", {}).get("oversharpen_detected_raw", False),
            "brightness_detected": "BRIGHTNESS" in s4.get("verdict", ""),
            "contrast_from_variance": s4.get("signals", {}).get("contrast_from_variance", False),
            "noise_verdict": s2.get("verdict", "NO_NOISE"),
            "global_verdict": s4.get("verdict", "NO_GLOBAL"),
        }
    except Exception as e:
        signals = {"error": str(e)}
    finally:
        for f in [tmp_clean, tmp_degraded]:
            if os.path.exists(f):
                os.remove(f)

    return (label, img_idx, signals)


# ================================================================
# 实验 1: 单退化 — 信号基线校准
# ================================================================

EXP1_TESTS = [
    ("contrast_weaken", "contrast_weaken_stretch", range(1, 6),
     "variance_ratio", lambda v: v is not None and v < 0.90),
    ("contrast_strengthen", "contrast_strengthen_stretch", range(1, 6),
     "variance_ratio", lambda v: v is not None and v > 1.12),
    ("brightness_brighten_shift", "brightness_brighten_shfit_RGB", range(1, 6),
     "variance_ratio", lambda v: v is not None and 0.85 < v < 1.18),
    ("brightness_darken_shift", "brightness_darken_shfit_RGB", range(1, 6),
     "variance_ratio", lambda v: v is not None and 0.85 < v < 1.18),
    ("gamma_brighten", "brightness_brighten_gamma_RGB", range(1, 6),
     "residual_intensity_corr", lambda v: v is not None and abs(v) > 0.20),
    ("gamma_darken", "brightness_darken_gamma_RGB", range(1, 6),
     "residual_intensity_corr", lambda v: v is not None and abs(v) > 0.20),
    ("YCrCb_noise", "noise_gaussian_YCrCb", range(1, 6),
     "residual_intensity_corr", lambda v: v is not None and abs(v) < 0.20),
    ("oversharpen", "oversharpen", range(1, 6),
     "laplacian_energy_ratio", lambda v: v is not None and v > 2.0),
    ("gaussian_noise_ctrl", "noise_gaussian_RGB", range(1, 6),
     "laplacian_energy_ratio", lambda v: v is not None and v < 2.5),
]


def run_exp1_parallel(images, workers=8):
    """实验 1: 多进程并行."""
    tasks = []
    for label, deg_fn, sevs, sig_name, check_fn in EXP1_TESTS:
        for sev in sevs:
            for i, img_path in enumerate(images):
                tasks.append((i, img_path, label, deg_fn, sev))

    print(f"  实验 1: {len(tasks)} 个任务, {workers} 进程并行...")
    results_by_key = defaultdict(list)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker_single, t): t for t in tasks}
        done = 0
        for future in as_completed(futures):
            label, sev, img_idx, signals = future.result()
            results_by_key[(label, sev)].append(signals)
            done += 1
            if done % 50 == 0:
                print(f"    exp1: {done}/{len(tasks)}")

    # Aggregate
    output = []
    for label, deg_fn, sevs, sig_name, check_fn in EXP1_TESTS:
        for sev in sevs:
            signals_list = results_by_key[(label, sev)]
            values = [s[sig_name] for s in signals_list if s.get(sig_name) is not None and "error" not in s]
            correct = sum(1 for v in values if check_fn(v))
            avg_val = np.mean(values) if values else float('nan')
            acc = correct / len(images) * 100 if images else 0
            output.append({
                "label": label, "function": deg_fn, "severity": sev,
                "signal": sig_name, "accuracy": round(acc, 1),
                "avg_value": round(float(avg_val), 4), "n_valid": len(values),
                "errors": sum(1 for s in signals_list if "error" in s)
            })
            print(f"  {label:30s} sev={sev} {sig_name}={avg_val:.4f}  acc={acc:.0f}%")

    return output


# ================================================================
# 实验 2/3: 复合退化 (共享 worker)
# ================================================================

def run_composite_parallel(images, tests, exp_name, workers=8):
    """通用复合退化测试."""
    tasks = []
    for label, degradations, check_fn in tests:
        for i, img_path in enumerate(images):
            tasks.append((i, img_path, label, degradations))

    print(f"  {exp_name}: {len(tasks)} 个任务, {workers} 进程...")
    results_by_label = defaultdict(list)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker_composite, t): t for t in tasks}
        done = 0
        for future in as_completed(futures):
            label, img_idx, signals = future.result()
            results_by_label[label].append(signals)
            done += 1
            if done % 20 == 0:
                print(f"    {exp_name}: {done}/{len(tasks)}")

    return results_by_label


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Validate new blind ID signals")
    parser.add_argument("--num-images", type=int, default=10,
                        help="Number of test images (default: 10)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers (default: 8)")
    parser.add_argument("--max-size", type=int, default=MAX_SIZE,
                        help=f"Max image dimension (default: {MAX_SIZE})")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    # 注意: resize 使用模块级 MAX_SIZE (默认512). 如需修改请改源码.
    max_size = args.max_size  # reserved for future use
    print(f"选取 {args.num_images} 张图像, 最大边长 {max_size}px, {args.workers} 进程")
    images = pick_images(args.num_images, "datasets/DIV2K/DIV2K_valid_HR")
    print(f"实际选取: {len(images)} 张\n")

    all_results = {}

    # --- 实验 1: 单退化 ---
    print("=" * 60)
    print("实验 1: 单退化 — 新信号基线校准")
    print("=" * 60)
    all_results["exp1_single"] = run_exp1_parallel(images, args.workers)

    print("\n--- 实验 1 汇总 ---")
    by_signal = defaultdict(list)
    for r in all_results["exp1_single"]:
        by_signal[r["signal"]].append(r["accuracy"])
    for sig, accs in sorted(by_signal.items()):
        print(f"  {sig:30s}: avg acc={np.mean(accs):.1f}%")

    # --- 实验 2: 复合退化 ---
    EXP2_TESTS = [
        ("JPEG+gamma", [("compression_jpeg", 2), ("brightness_brighten_gamma_RGB", 3)],
         lambda s: not s.get("error") and (s.get("gamma_suspect_from_noise") or s.get("gamma_suspect"))),
        ("JPEG+contrast_wk", [("compression_jpeg", 4), ("contrast_weaken_stretch", 1)],
         lambda s: not s.get("error") and s.get("contrast_from_variance") and not s.get("brightness_detected")),
        ("JPEG+oversharpen", [("compression_jpeg", 2), ("oversharpen", 2)],
         lambda s: not s.get("error") and s.get("oversharpen_detected_raw")),
    ]

    print("\n" + "=" * 60)
    print("实验 2: 复合退化 — 复现 exp27 失败案例")
    print("=" * 60)
    exp2_results = run_composite_parallel(images, EXP2_TESTS, "exp2", args.workers)
    all_results["exp2_composite"] = []

    for label, degradations, check_fn in EXP2_TESTS:
        signals_list = exp2_results[label]
        correct = sum(1 for s in signals_list if check_fn(s))
        acc = correct / len(signals_list) * 100 if signals_list else 0

        agg = {}
        for key in ["variance_ratio", "residual_intensity_corr", "laplacian_energy_ratio",
                     "gamma_suspect_from_noise", "gamma_suspect", "oversharpen_detected_raw",
                     "noise_verdict", "global_verdict"]:
            vals = [s[key] for s in signals_list if s.get(key) is not None and "error" not in s]
            if not vals:
                agg[key] = None
            elif isinstance(vals[0], (bool, np.bool_)):
                agg[key] = round(sum(1 for v in vals if v) / len(vals), 3)
            elif isinstance(vals[0], (int, float, np.integer, np.floating)):
                agg[key] = round(float(np.mean(vals)), 4)
            else:
                agg[key] = str(vals[0])  # string verdict, just record one

        old_error = {"JPEG+gamma": "YCrCb noise", "JPEG+contrast_wk": "brightness_shift",
                      "JPEG+oversharpen": "YCrCb noise"}[label]
        all_results["exp2_composite"].append({
            "label": label, "old_error": old_error,
            "correction_rate": round(acc, 1), "signals": agg, "n": len(signals_list)
        })
        print(f"  {label}: correction={acc:.0f}% (was: {old_error}) signals={agg}")

    # --- 实验 3: 负面测试 ---
    EXP3_TESTS = [
        ("JPEG+Gaussian", [("compression_jpeg", 3), ("noise_gaussian_RGB", 3)],
         lambda s: not s.get("contrast_from_variance") and not s.get("gamma_suspect")),
        ("JPEG+impulse", [("compression_jpeg", 2), ("noise_impulse", 2)],
         lambda s: not s.get("gamma_suspect_from_noise")),
        ("JPEG+blur_gauss", [("compression_jpeg", 1), ("blur_gaussian", 3)],
         lambda s: not s.get("oversharpen_detected_raw")),
    ]

    print("\n" + "=" * 60)
    print("实验 3: 负面测试 — 确保不引入新 FP")
    print("=" * 60)
    exp3_results = run_composite_parallel(images, EXP3_TESTS, "exp3", args.workers)
    all_results["exp3_negative"] = []

    for label, degradations, fp_check in EXP3_TESTS:
        signals_list = exp3_results[label]
        passed = sum(1 for s in signals_list if not s.get("error") and fp_check(s))
        fp_rate = (1 - passed / len(signals_list)) * 100 if signals_list else 0
        all_results["exp3_negative"].append({
            "label": label, "fp_rate": round(fp_rate, 1),
            "passed": passed, "n": len(signals_list)
        })
        print(f"  {label}: FP_rate={fp_rate:.0f}%")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("验证总结")
    print("=" * 60)
    by_signal = defaultdict(list)
    for r in all_results["exp1_single"]:
        by_signal[r["signal"]].append(r["accuracy"])
    for sig, accs in sorted(by_signal.items()):
        print(f"  {sig:30s}: avg acc={np.mean(accs):.1f}%")
    for r in all_results["exp2_composite"]:
        print(f"  [{r['label']}] correction_rate={r['correction_rate']:.0f}% (was: {r['old_error']})")
    for r in all_results["exp3_negative"]:
        print(f"  [{r['label']}] FP_rate={r['fp_rate']:.0f}%")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        # Convert numpy types for JSON
        def convert(obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, (np.bool_,)): return bool(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return obj
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2, default=convert)
        print(f"\n结果已保存: {args.output}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
