#!/usr/bin/env python3
"""
全局退化分析器 — Histogram 匹配 + 逐通道均值 + 噪声鲁棒检测。

解决的问题: 原有的 Cr_variance 只能检测饱和度, 且被噪声干扰后失效。
            brightness/contrast/gamma 均无对应检测工具。

方法:
  1. 大 bin 直方图 (32 bins) → 对噪声鲁棒, 捕捉全局色调映射
  2. 逐通道均值偏移 → brightness 检测
  3. Percentile 散度 → contrast 检测
  4. Mid-tone mass 偏移 → gamma 检测
  5. Cr/Cb 方差 + Y 通道 → saturation 检测

用法:
  # 单张分析
  .venv/bin/python3 .claude/skills/image-degradation-simulator/scripts/global_degradation_analyzer.py \
    --target degraded.png --clean clean.png

  # 批量测试
  .venv/bin/python3 .claude/skills/image-degradation-simulator/scripts/global_degradation_analyzer.py \
    --test --num-images 10
"""

import argparse
import json
import os
import sys
from collections import Counter

# Ensure project root is in path for x_distortion imports
# Try relative path first, then fall back to cwd
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, '..', '..', '..', '..'))
if not os.path.exists(os.path.join(_project_root, 'x_distortion')):
    _project_root = os.getcwd()
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# 使用 PIL/Pillow 做通用图像读取，numpy 做计算
try:
    import numpy as np
    from PIL import Image
except ImportError:
    print("ERROR: numpy and Pillow required", file=sys.stderr)
    sys.exit(1)

# ============================================================
# 核心分析函数
# ============================================================

def load_image(path):
    """加载图像，返回 RGB numpy array [0,255]"""
    img = Image.open(path).convert('RGB')
    return np.array(img, dtype=np.float64)


def compute_histogram_features(img, n_bins=32):
    """
    噪声鲁棒的直方图特征 (大 bin → 噪声在 bin 内平均掉)。

    Returns dict:
      - per_channel_mean: [R_mean, G_mean, B_mean]
      - per_channel_std: [R_std, G_std, B_std]
      - percentile_5: [R_p5, G_p5, B_p5]
      - percentile_95: [R_p95, G_p95, B_p95]
      - mid_tone_mass: [R_mid, G_mid, B_mid]  (128±32 范围内的像素比例)
      - histogram_bins: 32-bin histogram per channel
    """
    h, w, _ = img.shape
    features = {}

    # Per-channel statistics
    features['per_channel_mean'] = [float(img[:,:,c].mean()) for c in range(3)]
    features['per_channel_std'] = [float(img[:,:,c].std()) for c in range(3)]

    # Percentiles (robust to outliers/noise)
    features['percentile_5'] = [float(np.percentile(img[:,:,c], 5)) for c in range(3)]
    features['percentile_95'] = [float(np.percentile(img[:,:,c], 95)) for c in range(3)]
    features['percentile_25'] = [float(np.percentile(img[:,:,c], 25)) for c in range(3)]
    features['percentile_75'] = [float(np.percentile(img[:,:,c], 75)) for c in range(3)]

    # Mid-tone mass: proportion of pixels in [96, 160] (around 128)
    mid_lo, mid_hi = 96, 160
    features['mid_tone_mass'] = [
        float(((img[:,:,c] >= mid_lo) & (img[:,:,c] <= mid_hi)).mean())
        for c in range(3)
    ]

    # Large-bin histogram (32 bins, robust to noise)
    features['hist_32bin'] = [
        np.histogram(img[:,:,c], bins=n_bins, range=(0, 255))[0].astype(float).tolist()
        for c in range(3)
    ]

    return features


def compute_histogram_similarity(hist1, hist2):
    """计算两个直方图的相似度 (0-1, 1=完全相同)"""
    # 使用直方图交集 (histogram intersection)
    intersection = sum(min(h1, h2) for h1, h2 in zip(hist1, hist2))
    total = sum(hist2)
    return intersection / total if total > 0 else 0


def detect_brightness(target_features, clean_features, threshold=3.0):
    """
    检测 brightness 变化。
    特征: RGB 三通道均值同时偏移 (同方向同幅度)。
    对噪声鲁棒: 均值不受零均值噪声影响 (大数定律)。
    """
    mean_shifts = [
        target_features['per_channel_mean'][c] - clean_features['per_channel_mean'][c]
        for c in range(3)
    ]
    avg_shift = sum(mean_shifts) / 3

    # 三通道同向偏移
    signs = [1 if s > threshold else -1 if s < -threshold else 0 for s in mean_shifts]
    same_direction = all(s == signs[0] for s in signs) and signs[0] != 0

    return {
        'detected': abs(avg_shift) > threshold and same_direction,
        'avg_shift': round(avg_shift, 2),
        'per_channel_shift': [round(s, 2) for s in mean_shifts],
        'same_direction': same_direction,
    }


def detect_contrast(target_features, clean_features, threshold=0.10):
    """
    检测 contrast 变化。
    特征: percentile 散度 (P95-P5) 与 clean 的比值。
    对噪声鲁棒: percentile 不受少量极端值影响。
    """
    ratios = []
    for c in range(3):
        t_spread = target_features['percentile_95'][c] - target_features['percentile_5'][c]
        c_spread = clean_features['percentile_95'][c] - clean_features['percentile_5'][c]
        if c_spread > 0:
            ratios.append(t_spread / c_spread)

    avg_ratio = sum(ratios) / len(ratios) if ratios else 1.0

    # contrast weaken: ratio < 1-threshold, contrast strengthen: ratio > 1+threshold
    weaken = avg_ratio < (1.0 - threshold)
    strengthen = avg_ratio > (1.0 + threshold)

    return {
        'detected': weaken or strengthen,
        'type': 'weaken' if weaken else ('strengthen' if strengthen else 'none'),
        'spread_ratio': round(avg_ratio, 3),
        'per_channel_ratio': [round(r, 3) for r in ratios],
    }


def detect_gamma(target_features, clean_features, threshold=0.05):
    """
    检测 gamma 变化。
    特征: mid-tone mass 偏移。
      - gamma < 1 (brighten): 暗部拉伸, 亮部压缩 → mid_tone_mass 上升
      - gamma > 1 (darken):  暗部压缩, 亮部拉伸 → mid_tone_mass 下降
    对噪声鲁棒: 大 bin 下的像素计数不受零均值噪声影响。
    """
    mid_shift = target_features['mid_tone_mass'][1] - clean_features['mid_tone_mass'][1]  # G channel

    # Also check percentile ratio (P25/P50/P75 shape)
    p25_ratio = (target_features['percentile_25'][1] - clean_features['percentile_25'][1])
    p75_ratio = (target_features['percentile_75'][1] - clean_features['percentile_75'][1])

    # gamma brighten: P25 shift up > P75 shift up (暗部提亮更明显)
    # gamma darken: P75 shift down > P25 shift down (亮部压暗更明显)
    gamma_type = 'none'
    if abs(mid_shift) > threshold:
        if mid_shift > 0 and p25_ratio > p75_ratio:
            gamma_type = 'brighten'  # gamma < 1
        elif mid_shift < 0 and p75_ratio < p25_ratio:
            gamma_type = 'darken'    # gamma > 1

    return {
        'detected': gamma_type != 'none',
        'type': gamma_type,
        'mid_tone_shift': round(mid_shift, 4),
        'p25_shift': round(p25_ratio, 2),
        'p75_shift': round(p75_ratio, 2),
    }


def detect_saturation(target_features, clean_features, threshold=0.05):
    """
    检测 saturation 变化。
    特征: 逐通道 std 比值 + 通道间相关性变化。
      - 饱和度降低: 各通道 std 同比例下降 + 通道间相关性上升
      - 饱和度上升: 各通道 std 同比例上升 + 通道间相关性下降
    在 HSV/YCrCb 空间检测更准确, 但 RGB 空间也可近似。
    """
    std_ratios = [
        target_features['per_channel_std'][c] / max(clean_features['per_channel_std'][c], 1.0)
        for c in range(3)
    ]
    avg_std_ratio = sum(std_ratios) / 3

    weaken = avg_std_ratio < (1.0 - threshold)
    strengthen = avg_std_ratio > (1.0 + threshold)

    return {
        'detected': weaken or strengthen,
        'type': 'weaken' if weaken else ('strengthen' if strengthen else 'none'),
        'std_ratio': round(avg_std_ratio, 3),
        'per_channel_ratio': [round(r, 3) for r in std_ratios],
    }


def detect_global_degradations(target_path, clean_path):
    """
    综合分析: 检测所有全局退化类型。

    Returns dict:
      - brightness: {detected, avg_shift, ...}
      - contrast: {detected, type, spread_ratio, ...}
      - gamma: {detected, type, mid_tone_shift, ...}
      - saturation: {detected, type, std_ratio, ...}
      - summary: 检测到的全局退化列表
    """
    target = load_image(target_path)
    clean = load_image(clean_path)

    tf = compute_histogram_features(target)
    cf = compute_histogram_features(clean)

    results = {
        'brightness': detect_brightness(tf, cf),
        'contrast': detect_contrast(tf, cf),
        'gamma': detect_gamma(tf, cf),
        'saturation': detect_saturation(tf, cf),
    }

    # Summary
    detected = []
    if results['brightness']['detected']:
        sign = 'brighten' if results['brightness']['avg_shift'] > 0 else 'darken'
        detected.append(f"brightness_{sign}")
    if results['contrast']['detected']:
        detected.append(f"contrast_{results['contrast']['type']}")
    if results['gamma']['detected']:
        detected.append(f"gamma_{results['gamma']['type']}")
    if results['saturation']['detected']:
        detected.append(f"saturation_{results['saturation']['type']}")

    results['summary'] = detected
    results['any_detected'] = len(detected) > 0

    return results


# ============================================================
# 测试框架
# ============================================================

def apply_single_degradation(img, func_name, severity):
    """对图像应用单个退化函数。返回退化后的 numpy array。"""
    import importlib
    # Map function names to modules
    module_map = {
        'brightness_brighten_shfit_HSV': 'brightness',
        'brightness_brighten_shfit_RGB': 'brightness',
        'brightness_brighten_gamma_RGB': 'brightness',
        'brightness_brighten_gamma_HSV': 'brightness',
        'brightness_darken_shfit_HSV': 'brightness',
        'brightness_darken_shfit_RGB': 'brightness',
        'brightness_darken_gamma_RGB': 'brightness',
        'brightness_darken_gamma_HSV': 'brightness',
        'contrast_weaken_scale': 'contrast',
        'contrast_weaken_stretch': 'contrast',
        'contrast_strengthen_scale': 'contrast',
        'contrast_strengthen_stretch': 'contrast',
        'saturate_weaken_HSV': 'saturate',
        'saturate_weaken_YCrCb': 'saturate',
        'saturate_strengthen_HSV': 'saturate',
        'saturate_strengthen_YCrCb': 'saturate',
        'gamma_correct_RGB': 'brightness',
        'noise_gaussian_RGB': 'noise',
        'noise_speckle': 'noise',
        'noise_impulse': 'noise',
        'blur_gaussian': 'blur',
        'blur_lens': 'blur',
        'blur_motion': 'blur',
    }

    module_name = module_map.get(func_name)
    if module_name is None:
        raise ValueError(f"Unknown function: {func_name}")

    mod = importlib.import_module(f'x_distortion.{module_name}')
    func = getattr(mod, func_name)
    return func(img.copy(), severity)


def apply_pipeline(img, pipeline):
    """按顺序应用退化管线。"""
    result = img.copy()
    for step in pipeline:
        func_name = step.get('function', step[0] if isinstance(step, (list, tuple)) else step)
        severity = step.get('severity', step[1] if isinstance(step, (list, tuple)) else 1)
        result = apply_single_degradation(result, func_name, severity)
    return result


def run_tests(num_images=10):
    """
    系统测试: 对纯全局退化和复合退化测试检测精度。

    测试矩阵:
      - 纯全局: brightness × 5, contrast × 4, gamma × 2, saturation × 4 = 15 种
      - 全局+noise: 15 × 2 noise types × 3 sev
      - 全局+blur: 15 × 3 blur types
      - 全局+blur+noise: 15 × 2 combos
    """
    import os, glob, tempfile

    # Find clean images
    img_dir = 'datasets/DIV2K/DIV2K_valid_HR'
    if not os.path.isdir(img_dir):
        img_dir = 'datasets/Set5/GTmod4'
    images = sorted(glob.glob(os.path.join(img_dir, '*.png')))[:num_images]

    if not images:
        print("ERROR: No test images found")
        return

    # Define test degradations (using verified function names)
    global_funcs = {
        'brightness': [
            'brightness_brighten_shfit_HSV',
            'brightness_brighten_shfit_RGB',
            'brightness_brighten_gamma_RGB',
            'brightness_darken_shfit_HSV',
            'brightness_darken_shfit_RGB',
            'brightness_darken_gamma_RGB',
        ],
        'contrast': [
            'contrast_weaken_scale',
            'contrast_weaken_stretch',
            'contrast_strengthen_scale',
            'contrast_strengthen_stretch',
        ],
        'saturation': [
            'saturate_weaken_HSV',
            'saturate_weaken_YCrCb',
            'saturate_strengthen_HSV',
            'saturate_strengthen_YCrCb',
        ],
        'gamma': [
            'brightness_brighten_gamma_RGB',  # gamma < 1
            'brightness_darken_gamma_RGB',    # gamma > 1
            'brightness_brighten_gamma_HSV',
            'brightness_darken_gamma_HSV',
        ],
    }

    noise_funcs = ['noise_gaussian_RGB', 'noise_speckle', 'noise_impulse']
    blur_funcs = ['blur_gaussian', 'blur_lens', 'blur_motion']

    # Flatten global functions with severities
    all_global = []
    for category, funcs in global_funcs.items():
        for fn in funcs:
            for sev in [1, 3, 5]:
                all_global.append({
                    'function': fn,
                    'severity': sev,
                    'category': category,
                })

    print(f"Testing {len(all_global)} global degradation variants × {num_images} images")
    print(f"  Pure global: {len(all_global)}")
    print(f"  Global + noise: {len(all_global) * 3}")
    print(f"  Global + blur: {len(all_global) * 3}")
    print(f"  Global + blur + noise: {len(all_global) * 6}")
    print()

    results = {
        'pure_global': [],
        'global_noise': [],
        'global_blur': [],
        'global_blur_noise': [],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        for img_path in images:
            clean = np.array(Image.open(img_path).convert('RGB'))

            for g in all_global:
                # --- Pure global ---
                degraded = apply_single_degradation(clean.copy(), g['function'], g['severity'])
                t_path = os.path.join(tmpdir, 'target.png')
                c_path = os.path.join(tmpdir, 'clean.png')
                Image.fromarray(degraded.astype(np.uint8)).save(t_path)
                Image.fromarray(clean.astype(np.uint8)).save(c_path)

                r = detect_global_degradations(t_path, c_path)
                correct = g['category'] in ','.join(r['summary'])
                results['pure_global'].append({
                    'function': g['function'], 'severity': g['severity'],
                    'category': g['category'], 'detected': r['summary'], 'correct': correct,
                })

                # --- Global + noise ---
                for nf in noise_funcs[:2]:  # gaussian, speckle
                    for ns in [1, 3]:
                        degraded_n = apply_single_degradation(degraded.copy(), nf, ns)
                        Image.fromarray(degraded_n.astype(np.uint8)).save(t_path)
                        r_n = detect_global_degradations(t_path, c_path)
                        correct_n = g['category'] in ','.join(r_n['summary'])
                        results['global_noise'].append({
                            'function': g['function'], 'severity': g['severity'],
                            'category': g['category'], 'noise': f'{nf}({ns})',
                            'detected': r_n['summary'], 'correct': correct_n,
                        })

                # --- Global + blur ---
                for bf in blur_funcs[:2]:  # gaussian, lens
                    degraded_b = apply_single_degradation(degraded.copy(), bf, 3)
                    Image.fromarray(degraded_b.astype(np.uint8)).save(t_path)
                    r_b = detect_global_degradations(t_path, c_path)
                    correct_b = g['category'] in ','.join(r_b['summary'])
                    results['global_blur'].append({
                        'function': g['function'], 'severity': g['severity'],
                        'category': g['category'], 'blur': f'{bf}(3)',
                        'detected': r_b['summary'], 'correct': correct_b,
                    })

                # --- Global + blur + noise ---
                for bf, nf in [('blur_gaussian', 'noise_gaussian_RGB'), ('blur_lens', 'noise_speckle')]:
                    degraded_bn = apply_single_degradation(degraded.copy(), bf, 3)
                    degraded_bn = apply_single_degradation(degraded_bn, nf, 2)
                    Image.fromarray(degraded_bn.astype(np.uint8)).save(t_path)
                    r_bn = detect_global_degradations(t_path, c_path)
                    correct_bn = g['category'] in ','.join(r_bn['summary'])
                    results['global_blur_noise'].append({
                        'function': g['function'], 'severity': g['severity'],
                        'category': g['category'], 'combo': f'{bf}(3)+{nf}(2)',
                        'detected': r_bn['summary'], 'correct': correct_bn,
                    })

    # Print results
    for scenario, data in results.items():
        total = len(data)
        correct = sum(1 for d in data if d['correct'])
        rate = correct / total * 100 if total > 0 else 0

        # Per-category breakdown
        cat_stats = {}
        for d in data:
            cat = d['category']
            if cat not in cat_stats:
                cat_stats[cat] = {'total': 0, 'correct': 0}
            cat_stats[cat]['total'] += 1
            if d['correct']:
                cat_stats[cat]['correct'] += 1

        cat_detail = ' | '.join(
            f"{c}={cat_stats[c]['correct']}/{cat_stats[c]['total']} ({cat_stats[c]['correct']/max(cat_stats[c]['total'],1)*100:.0f}%)"
            for c in sorted(cat_stats)
        )

        print(f"{scenario:>20s}: {correct}/{total} ({rate:.1f}%)  [{cat_detail}]")

    return results


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Global Degradation Analyzer')
    parser.add_argument('--target', help='Degraded image path')
    parser.add_argument('--clean', help='Clean reference image path')
    parser.add_argument('--test', action='store_true', help='Run systematic tests')
    parser.add_argument('--num-images', type=int, default=10, help='Test images count')
    parser.add_argument('--json', action='store_true', help='Output JSON')

    args = parser.parse_args()

    if args.test:
        run_tests(args.num_images)
        return

    if args.target and args.clean:
        results = detect_global_degradations(args.target, args.clean)

        if args.json:
            print(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            print("=== 全局退化分析 ===")
            print(f"Target: {args.target}")
            print(f"Clean:  {args.clean}")
            print()
            for deg_type in ['brightness', 'contrast', 'gamma', 'saturation']:
                r = results[deg_type]
                status = '✅ DETECTED' if r['detected'] else '   (none)'
                detail = ''
                if deg_type == 'brightness':
                    detail = f"shift={r['avg_shift']}"
                elif deg_type == 'contrast':
                    detail = f"type={r['type']} ratio={r['spread_ratio']}"
                elif deg_type == 'gamma':
                    detail = f"type={r['type']} mid_shift={r['mid_tone_shift']}"
                elif deg_type == 'saturation':
                    detail = f"type={r['type']} ratio={r['std_ratio']}"
                print(f"  {deg_type:12s}: {status}  {detail}")
            print()
            if results['any_detected']:
                print(f"检测到: {', '.join(results['summary'])}")
            else:
                print("未检测到全局退化 (可能是 blur/noise/compression 类型)")
        return

    parser.print_help()


if __name__ == '__main__':
    main()
