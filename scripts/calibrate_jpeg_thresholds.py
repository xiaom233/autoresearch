#!/usr/bin/env python3
"""JPEG压缩信号大规模校准 — block_boundary + dct_zero_ratio 阈值优化
覆盖: 5 JPEG sev × 所有blur/noise类型 × 两种顺序 × 5图 × 2种子
"""
import json, os, sys, numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, '.')
from x_distortion import add_distortion

sys.path.insert(0, '.claude/skills/image-degradation-simulator/scripts')
from run_full_analysis import compute_block_boundary, compute_dct_zero_ratio

VAL_DIR = 'datasets/DIV2K/DIV2K_valid_HR'
IMAGES = sorted([os.path.join(VAL_DIR, f) for f in os.listdir(VAL_DIR) if f.endswith('.png')])[:5]
IMAGES_FULL = sorted([os.path.join(VAL_DIR, f) for f in os.listdir(VAL_DIR) if f.endswith('.png')])[:10]

JPEG_SEVS = [1, 2, 3, 4, 5]
NOISE_TYPES = ['noise_gaussian_RGB', 'noise_gaussian_YCrCb', 'noise_speckle', 'noise_poisson', 'noise_impulse']
NOISE_SEVS = [1, 3, 5]
BLUR_TYPES = ['blur_gaussian', 'blur_motion', 'blur_lens', 'blur_glass']
BLUR_SEVS = [2, 4]
ORDERS = ['blur_first', 'noise_first']
SEEDS = [42, 123]

OUT_DIR = '/tmp/jpeg_calibration'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Generate cases ──
cases = []

# 1) Pure JPEG: 10 images
for img_path in IMAGES_FULL:
    clean = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
    h, w = clean.shape[:2]
    clean_crop = clean[h//2-128:h//2+128, w//2-128:w//2+128]
    for jpeg_sev in JPEG_SEVS:
        for seed in SEEDS:
            cases.append({
                'jpeg_sev': jpeg_sev, 'seed': seed,
                'blur_type': None, 'blur_sev': 0,
                'noise_type': None, 'noise_sev': 0, 'order': 'none',
                'clean': clean_crop,
            })

# 2) noise→JPEG: 10 images
for img_path in IMAGES_FULL:
    clean = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
    h, w = clean.shape[:2]
    clean_crop = clean[h//2-128:h//2+128, w//2-128:w//2+128]
    for jpeg_sev in JPEG_SEVS:
        for nt in NOISE_TYPES:
            for ns in NOISE_SEVS:
                for seed in SEEDS:
                    cases.append({
                        'jpeg_sev': jpeg_sev, 'seed': seed,
                        'blur_type': None, 'blur_sev': 0,
                        'noise_type': nt, 'noise_sev': ns, 'order': 'noise_first',
                        'clean': clean_crop,
                    })

# 3) blur→JPEG: 10 images
for img_path in IMAGES_FULL:
    clean = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
    h, w = clean.shape[:2]
    clean_crop = clean[h//2-128:h//2+128, w//2-128:w//2+128]
    for jpeg_sev in JPEG_SEVS:
        for bt in BLUR_TYPES:
            for bs in BLUR_SEVS:
                for seed in SEEDS:
                    cases.append({
                        'jpeg_sev': jpeg_sev, 'seed': seed,
                        'blur_type': bt, 'blur_sev': bs,
                        'noise_type': None, 'noise_sev': 0, 'order': 'blur_first',
                        'clean': clean_crop,
                    })

# 4-5) blur+noise+JPEG (both orders): 5 images (reduced to manage total)
for img_path in IMAGES:
    clean = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
    h, w = clean.shape[:2]
    clean_crop = clean[h//2-128:h//2+128, w//2-128:w//2+128]
    for jpeg_sev in JPEG_SEVS:
        for bt in BLUR_TYPES:
            for bs in BLUR_SEVS:
                for nt in NOISE_TYPES:
                    for ns in NOISE_SEVS:
                        for seed in SEEDS:
                            for order in ORDERS:
                                cases.append({
                                    'jpeg_sev': jpeg_sev, 'seed': seed,
                                    'blur_type': bt, 'blur_sev': bs,
                                    'noise_type': nt, 'noise_sev': ns, 'order': order,
                                    'clean': clean_crop,
                                })

n_pure = len(JPEG_SEVS) * len(IMAGES_FULL) * len(SEEDS)
n_noise = len(JPEG_SEVS) * len(NOISE_TYPES) * len(NOISE_SEVS) * len(IMAGES_FULL) * len(SEEDS)
n_blur = len(JPEG_SEVS) * len(BLUR_TYPES) * len(BLUR_SEVS) * len(IMAGES_FULL) * len(SEEDS)
n_mixed = len(JPEG_SEVS) * len(BLUR_TYPES) * len(BLUR_SEVS) * len(NOISE_TYPES) * len(NOISE_SEVS) * len(IMAGES) * len(SEEDS) * len(ORDERS)

print(f"Cases: {len(cases)}")
print(f"  纯JPEG: {n_pure}")
print(f"  noise→JPEG (5种noise×3sev×10图×2种子): {n_noise}")
print(f"  blur→JPEG (4种blur×2sev×10图×2种子): {n_blur}")
print(f"  blur+noise→JPEG 双顺序 (4blur×2bs×5noise×3ns×5图×2种子×2顺序): {n_mixed}")
print(f"  退化顺序: 物理顺序(blur→noise→JPEG) + 逆序(noise→blur→JPEG)")

# ── Generate & compute ──
def process_case(case):
    np.random.seed(case['seed'])
    img = case['clean'].copy()
    if case['order'] == 'blur_first':
        if case['blur_type']:
            img = add_distortion(img, severity=case['blur_sev'], distortion_name=case['blur_type'])
        if case['noise_type']:
            img = add_distortion(img, severity=case['noise_sev'], distortion_name=case['noise_type'])
    else:
        if case['noise_type']:
            img = add_distortion(img, severity=case['noise_sev'], distortion_name=case['noise_type'])
        if case['blur_type']:
            img = add_distortion(img, severity=case['blur_sev'], distortion_name=case['blur_type'])
    if case['jpeg_sev'] > 0:
        img = add_distortion(img, severity=case['jpeg_sev'], distortion_name='compression_jpeg')
    gray = img.mean(axis=2)
    return {
        'jpeg_sev': case['jpeg_sev'],
        'blur_type': case['blur_type'] or 'none',
        'blur_sev': case['blur_sev'],
        'noise_type': case['noise_type'] or 'none',
        'noise_sev': case['noise_sev'],
        'order': case['order'],
        'block_boundary': round(compute_block_boundary(img), 5),
        'dct_zero_ratio': round(compute_dct_zero_ratio(gray), 5),
    }

print("Processing...")
results = []
with ProcessPoolExecutor(max_workers=16) as ex:
    futures = {ex.submit(process_case, c): c for c in cases}
    done = 0
    for f in as_completed(futures):
        results.append(f.result()); done += 1
        if done % 1000 == 0:
            print(f"  {done}/{len(cases)}")

# ── Analysis ──
results_by_sev = {s: [] for s in JPEG_SEVS}
no_jpeg_bb = []

for r in results:
    results_by_sev[r['jpeg_sev']].append(r['block_boundary'])

# noise/blur-only baseline (no JPEG)
for img_path in IMAGES[:3]:
    clean = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
    h, w = clean.shape[:2]
    clean_crop = clean[h//2-128:h//2+128, w//2-128:w//2+128]
    for nt in NOISE_TYPES:
        for ns in [1,3,5]:
            for seed in SEEDS:
                np.random.seed(seed)
                img = add_distortion(clean_crop.copy(), severity=ns, distortion_name=nt)
                no_jpeg_bb.append(compute_block_boundary(img))
    for bt in BLUR_TYPES:
        for bs in BLUR_SEVS:
            for seed in SEEDS:
                np.random.seed(seed)
                img = add_distortion(clean_crop.copy(), severity=bs, distortion_name=bt)
                no_jpeg_bb.append(compute_block_boundary(img))

no_jpeg_bb = np.array(no_jpeg_bb)

# ── Report ──
print(f"\n{'='*60}")
print("JPEG block_boundary 校准报告")
print(f"{'='*60}")
print(f"样本: {len(cases)}")
print()

print("=== 按JPEG severity分组 (全部) ===")
for sev in JPEG_SEVS:
    vals = results_by_sev[sev]
    print(f"  sev={sev}: mean={np.mean(vals):.4f}  median={np.median(vals):.4f}  std={np.std(vals):.4f}  "
          f"p5={np.percentile(vals,5):.4f}  p95={np.percentile(vals,95):.4f}")

print(f"\n=== 无JPEG基线 (噪声+模糊, 无压缩) ===")
print(f"  n={len(no_jpeg_bb)}  mean={no_jpeg_bb.mean():.4f}  max={no_jpeg_bb.max():.4f}  "
      f"p99={np.percentile(no_jpeg_bb,99):.4f}  p999={np.percentile(no_jpeg_bb,99.9):.4f}")

print(f"\n=== 阈值扫描 (FP vs Recall) ===")
best_threshold = None
best_f1 = 0
for threshold in [1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.10, 1.12, 1.15, 1.20, 1.30, 1.50, 2.00]:
    fp = np.mean(no_jpeg_bb > threshold)
    recalls = {sev: np.mean(np.array(results_by_sev[sev]) > threshold) for sev in JPEG_SEVS}
    avg_recall = np.mean(list(recalls.values()))
    f1 = 2 * (1-fp) * avg_recall / ((1-fp) + avg_recall + 1e-10)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
    print(f"  bb>{threshold:.2f}: FP={fp:.3%}  "
          f"R1={recalls[1]:.1%} R2={recalls[2]:.1%} R3={recalls[3]:.1%} R4={recalls[4]:.1%} R5={recalls[5]:.1%}  "
          f"avgR={avg_recall:.1%}  F1={f1:.3f}")

print(f"\n  🏆 最佳阈值: block_boundary > {best_threshold:.2f} (F1={best_f1:.3f})")

# 噪声影响
print(f"\n=== 噪声对 block_boundary 的衰减 (sev=3 JPEG) ===")
for nt in ['none'] + NOISE_TYPES:
    for ns in ([0] if nt == 'none' else NOISE_SEVS):
        subset = [r['block_boundary'] for r in results
                  if r['jpeg_sev']==3 and r['noise_type']==nt
                  and r['noise_sev']==ns and r['blur_type']=='none']
        if subset:
            label = f"{nt}:{ns}" if nt != 'none' else '无噪声'
            print(f"  {label:<35} mean={np.mean(subset):.4f}  median={np.median(subset):.4f}  "
                  f">1.05={np.mean(np.array(subset)>1.05):.1%}")

# 模糊影响
print(f"\n=== 模糊对 block_boundary 的衰减 (sev=3 JPEG, 无噪声) ===")
for bt in ['none'] + BLUR_TYPES:
    for bs in ([0] if bt == 'none' else BLUR_SEVS):
        subset = [r['block_boundary'] for r in results
                  if r['jpeg_sev']==3 and r['blur_type']==bt
                  and r['blur_sev']==bs and r['noise_type']=='none']
        if subset:
            label = f"{bt}:{bs}" if bt != 'none' else '无模糊'
            print(f"  {label:<25} mean={np.mean(subset):.4f}  median={np.median(subset):.4f}")

# 顺序影响
print(f"\n=== 退化顺序影响 (blur+noise+JPEG, sev=3) ===")
for order in ORDERS:
    subset = [r['block_boundary'] for r in results
              if r['jpeg_sev']==3 and r['blur_type']!='none' and r['noise_type']!='none'
              and r['order']==order]
    if subset:
        print(f"  {order}: mean={np.mean(subset):.4f}  median={np.median(subset):.4f}")

# DCT
print(f"\n=== DCT zero_ratio 校准 ===")
for sev in JPEG_SEVS:
    dct_vals = [r['dct_zero_ratio'] for r in results if r['jpeg_sev']==sev]
    print(f"  sev={sev}: mean={np.mean(dct_vals):.4f}  median={np.median(dct_vals):.4f}  p5={np.percentile(dct_vals,5):.4f}")

no_jpeg_dct = []
for img_path in IMAGES[:3]:
    clean = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
    h, w = clean.shape[:2]
    clean_crop = clean[h//2-128:h//2+128, w//2-128:w//2+128]
    no_jpeg_dct.append(compute_dct_zero_ratio(clean_crop.mean(axis=2)))
print(f"  no-JPEG baseline: mean={np.mean(no_jpeg_dct):.4f}")

print(f"\n=== DCT 阈值扫描 ===")
for threshold in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 0.77, 0.80, 0.85]:
    fp = np.mean(np.array(no_jpeg_dct) > threshold)
    recalls = {sev: np.mean(np.array([r['dct_zero_ratio'] for r in results if r['jpeg_sev']==sev]) > threshold)
               for sev in JPEG_SEVS}
    avg_recall = np.mean(list(recalls.values()))
    print(f"  dct>{threshold:.2f}: FP={fp:.1%}  "
          f"R1={recalls[1]:.1%} R2={recalls[2]:.1%} R3={recalls[3]:.1%} R4={recalls[4]:.1%} R5={recalls[5]:.1%}")

json.dump({'results': results, 'no_jpeg_bb': no_jpeg_bb.tolist(), 'best_threshold': best_threshold},
          open(os.path.join(OUT_DIR, 'results.json'), 'w'), indent=2)
print(f"\nSaved: {OUT_DIR}/results.json")
