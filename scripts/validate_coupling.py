#!/usr/bin/env python3
"""新信号耦合验证: 全面测试每个信号在退化耦合下的鲁棒性."""
import argparse, json, os, sys, random, numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

MAX_SIZE = 512
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, '..'))
sys.path.insert(0, os.path.join(ROOT, '..', '.claude/skills/image-degradation-simulator/scripts'))


def resize_if_needed(img):
    h, w = img.shape[:2]
    if max(h, w) <= MAX_SIZE: return img
    scale = MAX_SIZE / max(h, w)
    from PIL import Image as PILImage
    pil = PILImage.fromarray(img)
    pil = pil.resize((int(w*scale), int(h*scale)), PILImage.LANCZOS)
    return np.array(pil, dtype=np.uint8)


def pick_images(n):
    exts = {'.jpg', '.jpeg', '.png'}
    paths = []
    for entry in os.scandir("datasets/DIV2K/DIV2K_valid_HR"):
        if entry.is_file() and os.path.splitext(entry.name)[1].lower() in exts:
            paths.append(entry.path)
    return random.sample(paths, min(n, len(paths)))


# ================================================================
# Test Matrix (纯数据, 无 lambda)
# ================================================================
# check_key + check_op 定义验证条件:
#   check_op: "is_true" / "is_false" / "abs_gt_X" / "abs_lt_X" / "lt_X" / "gt_X"

COUPLING_PAIRS = [
    ("blur", "blur_gaussian"), ("JPEG", "compression_jpeg"),
    ("noise", "noise_gaussian_RGB"), ("impulse", "noise_impulse"),
    ("brightness", "brightness_brighten_shfit_RGB"),
    ("contrast_wk", "contrast_weaken_stretch"),
    ("gamma", "brightness_brighten_gamma_RGB"),
    ("oversharpen", "oversharpen"),
]

ALL_TESTS = []

def add_tests(signal_name, primary_fn, primary_sev, couplings, expected, check_key, check_op, category):
    """批量添加测试."""
    for cname, cfn in couplings:
        ALL_TESTS.append({
            "label": f"{primary_fn.split('_')[0]}+{cname}",
            "degradations": [(primary_fn, primary_sev), (cfn, 3)],
            "expected": expected,
            "signal": signal_name,
            "check_key": check_key,
            "check_op": check_op,
            "category": category
        })


# --- Signal 1: variance_ratio / contrast_from_variance ---
contrast_couplings = [p for p in COUPLING_PAIRS if p[1] not in ["contrast_weaken_stretch"]]
# recall: contrast_weaken + X → contrast_from_variance should be True
add_tests("variance_ratio", "contrast_weaken_stretch", 3, contrast_couplings,
          True, "contrast_from_variance", "is_true", "recall")
# recall: contrast_strengthen + X
str_couplings = [p for p in COUPLING_PAIRS if p[1] not in ["contrast_strengthen_stretch", "oversharpen"]]
add_tests("variance_ratio", "contrast_strengthen_stretch", 3, str_couplings,
          True, "contrast_from_variance", "is_true", "recall")
# precision: brightness (NO contrast) + X → should NOT trigger
bright_couplings = [p for p in COUPLING_PAIRS if p[1] not in ["brightness_brighten_shfit_RGB",
    "contrast_weaken_stretch", "contrast_strengthen_stretch"]]
add_tests("variance_ratio", "brightness_brighten_shfit_RGB", 2, bright_couplings,
          False, "contrast_from_variance", "is_false", "precision")
# precision: noise + X → should NOT trigger contrast
noise_couplings = [("blur", "blur_gaussian"), ("JPEG", "compression_jpeg"), ("gamma", "brightness_brighten_gamma_RGB")]
add_tests("variance_ratio", "noise_gaussian_RGB", 3, noise_couplings,
          False, "contrast_from_variance", "is_false", "precision")

# --- Signal 2: residual_intensity_corr / gamma_suspect ---
gamma_couplings = [p for p in COUPLING_PAIRS if p[1] not in ["brightness_brighten_gamma_RGB"]]
# recall: gamma + X → gamma_suspect should be True (either from noise or global side)
# We check the combined OR of both gamma signals
add_tests("residual_intensity_corr", "brightness_brighten_gamma_RGB", 3, gamma_couplings,
          True, "gamma_any", "is_true", "recall")
# precision: YCrCb noise (NO gamma) + X → should NOT trigger
ycbcr_couplings = [p for p in COUPLING_PAIRS if p[1] not in ["noise_gaussian_YCrCb",
    "brightness_brighten_gamma_RGB", "oversharpen"]]
add_tests("residual_intensity_corr", "noise_gaussian_YCrCb", 3, ycbcr_couplings,
          False, "gamma_any", "is_false", "precision")
# precision: JPEG (NO gamma) + X → should NOT trigger
jpeg_couplings2 = [("blur", "blur_gaussian"), ("noise", "noise_gaussian_RGB"), ("contrast_wk", "contrast_weaken_stretch")]
add_tests("residual_intensity_corr", "compression_jpeg", 3, jpeg_couplings2,
          False, "gamma_any", "is_false", "precision")

# --- Signal 3: laplacian_energy_ratio / oversharpen_detected_raw ---
os_couplings = [p for p in COUPLING_PAIRS if p[1] not in ["oversharpen"]]
# recall: oversharpen + X → oversharpen_detected_raw should be True
add_tests("laplacian_energy_ratio", "oversharpen", 3, os_couplings,
          True, "oversharpen_detected_raw", "is_true", "recall")
# precision: noise (NO oversharpen) + X → should NOT trigger (clustering check)
noise_os_couplings = [p for p in COUPLING_PAIRS
    if p[1] not in ["oversharpen", "noise_gaussian_RGB", "noise_impulse"]]
add_tests("laplacian_energy_ratio", "noise_gaussian_RGB", 3, noise_os_couplings,
          False, "oversharpen_detected_raw", "is_false", "precision")
# precision: blur (NO oversharpen) + X → should NOT trigger
blur_os_couplings = [p for p in COUPLING_PAIRS
    if p[1] not in ["oversharpen", "blur_gaussian"]]
add_tests("laplacian_energy_ratio", "blur_gaussian", 3, blur_os_couplings,
          False, "oversharpen_detected_raw", "is_false", "precision")


# ================================================================
# Worker
# ================================================================
def _worker(args_tuple):
    img_idx, img_path, test_item = args_tuple
    sys.path.insert(0, os.path.join(ROOT, '..'))
    sys.path.insert(0, os.path.join(ROOT, '..', '.claude/skills/image-degradation-simulator/scripts'))
    from PIL import Image
    from x_distortion import add_distortion
    from run_full_analysis import run_full_analysis

    clean = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    clean = resize_if_needed(clean)
    degraded = clean.copy()
    for fn, sev in test_item["degradations"]:
        degraded = add_distortion(degraded, severity=sev, distortion_name=fn)

    tc = f"/tmp/vc2_c_{os.getpid()}_{img_idx}.png"
    td = f"/tmp/vc2_d_{os.getpid()}_{img_idx}.png"
    Image.fromarray(clean).save(tc); Image.fromarray(degraded).save(td)

    try:
        report = run_full_analysis(td, tc)
        s2 = report.get("step2_noise", {})
        s4 = report.get("step4_global", {})
        signals = {
            "variance_ratio": s4.get("signals", {}).get("variance_ratio"),
            "contrast_from_variance": s4.get("signals", {}).get("contrast_from_variance", False),
            "residual_intensity_corr": s2.get("signals", {}).get("residual_intensity_corr"),
            "gamma_suspect_from_noise": s2.get("signals", {}).get("gamma_suspect_from_noise", False),
            "gamma_suspect": s4.get("signals", {}).get("gamma_suspect", False),
            "gamma_any": (s2.get("signals", {}).get("gamma_suspect_from_noise", False) or
                          s4.get("signals", {}).get("gamma_suspect", False)),
            "laplacian_energy_ratio": s4.get("signals", {}).get("laplacian_energy_ratio"),
            "oversharpen_detected_raw": s4.get("signals", {}).get("oversharpen_detected_raw", False),
            "oversharpen_vs_noise": s4.get("signals", {}).get("oversharpen_vs_noise"),
            "noise_verdict": s2.get("verdict", "?"),
            "global_verdict": s4.get("verdict", "?"),
        }
    except Exception as e:
        signals = {"error": str(e)}
    finally:
        for f in [tc, td]:
            if os.path.exists(f): os.remove(f)

    # Evaluate check condition
    check_ok = False
    if "error" not in signals:
        key = test_item["check_key"]
        op = test_item["check_op"]
        val = signals.get(key)
        if op == "is_true": check_ok = bool(val)
        elif op == "is_false": check_ok = not bool(val)
        elif op.startswith("abs_gt_"):
            thresh = float(op.split("_")[-1])
            check_ok = val is not None and abs(float(val)) > thresh
        elif op.startswith("abs_lt_"):
            thresh = float(op.split("_")[-1])
            check_ok = val is not None and abs(float(val)) < thresh
        elif op.startswith("lt_"):
            thresh = float(op.split("_")[-1])
            check_ok = val is not None and float(val) < thresh
        elif op.startswith("gt_"):
            thresh = float(op.split("_")[-1])
            check_ok = val is not None and float(val) > thresh

    return (test_item["label"], test_item["category"], test_item["expected"],
            test_item["signal"], check_ok, signals)


# ================================================================
# Main
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-images", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    random.seed(42); np.random.seed(42)
    images = pick_images(args.num_images)
    print(f"图像: {len(images)} 张, 进程: {args.workers}, 测试组合: {len(ALL_TESTS)}")

    tasks = []
    for test_item in ALL_TESTS:
        for i, img_path in enumerate(images):
            tasks.append((i, img_path, test_item))

    print(f"总任务: {len(tasks)}")

    results = defaultdict(list)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, t): t for t in tasks}
        done = 0
        for future in as_completed(futures):
            label, cat, expected, sig_name, check_ok, signals = future.result()
            results[label].append((expected, check_ok, sig_name, cat, signals))
            done += 1
            if done % 40 == 0:
                print(f"  {done}/{len(tasks)}")

    # Analyze
    print(f"\n{'='*80}")
    print(f"{'Label':38s} {'Cat':9s} {'Acc':>6s}  {'Signal values'}")
    print(f"{'='*80}")

    summary = defaultdict(lambda: {"recall": [], "precision": []})

    for label in sorted(results.keys()):
        items = results[label]
        total = sum(1 for _, _, _, _, s in items if "error" not in s)
        correct = sum(1 for _, ok, _, _, s in items if ok and "error" not in s)
        acc = correct / max(total, 1) * 100
        cat = items[0][3]
        sig_name = items[0][2]

        summary[sig_name][cat].append(acc)

        # Signal averages
        sig_vals = {}
        for key in ["variance_ratio", "residual_intensity_corr", "laplacian_energy_ratio"]:
            vals = [s[key] for _, _, _, _, s in items if s.get(key) is not None and "error" not in s]
            if vals and isinstance(vals[0], (int, float, np.integer, np.floating)):
                sig_vals[key] = round(float(np.mean(vals)), 4)

        status = "✅" if (cat == "recall" and acc >= 50) or (cat == "precision" and acc >= 60) else "⚠️"
        if (cat == "recall" and acc < 30) or (cat == "precision" and acc < 40):
            status = "🔴"

        vals_str = " ".join(f"{k}={v}" for k, v in sig_vals.items())
        print(f"  {status} {label:35s} {cat:9s} {acc:5.0f}%  {vals_str}")

    # Summary
    print(f"\n{'='*80}")
    print(f"汇总")
    print(f"{'='*80}")
    for sig in ["variance_ratio", "residual_intensity_corr", "laplacian_energy_ratio"]:
        r = summary[sig]["recall"]
        p = summary[sig]["precision"]
        r_avg = np.mean(r) if r else 0
        p_avg = np.mean(p) if p else 0
        r_fail = sum(1 for v in r if v < 50)
        p_fail = sum(1 for v in p if v < 60)
        print(f"  {sig:30s} recall={r_avg:.0f}% ({len(r)}组, fail:{r_fail})  "
              f"precision={p_avg:.0f}% ({len(p)}组, fail:{p_fail})")

    # Save
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        out = {"summary": {sig: {"recall_avg": float(np.mean(summary[sig]["recall"])) if summary[sig]["recall"] else 0,
                                 "precision_avg": float(np.mean(summary[sig]["precision"])) if summary[sig]["precision"] else 0}
                           for sig in summary}}
        with open(args.output, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"\n结果: {args.output}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
