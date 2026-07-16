#!/usr/bin/env python3
"""
SKILL.md v13 强制检测脚本 — 锁定 Step 1→4 执行顺序。

用法:
  .venv/bin/python3 .claude/skills/image-degradation-simulator/scripts/run_full_analysis.py \
    --target expN/challenges/phase4/blind_XXXX/degraded.png \
    --clean expN/challenges/phase4/blind_XXXX/clean.png \
    --output /tmp/analysis.json

输出 structured_report.json，Agent 必须基于此 JSON 做决策，不能跳过步骤。
"""

import argparse
import json
import os
import sys
import warnings
import numpy as np
from PIL import Image
from collections import OrderedDict

warnings.filterwarnings("ignore")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def to_native(obj):
    """Recursively convert numpy types to Python native types."""
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.bool_,)): return bool(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

# ============================================================
# Utilities
# ============================================================

def load_image(path):
    return np.array(Image.open(path).convert('RGB'), dtype=np.float64)

def compute_residual(target, clean):
    return target - clean

# ============================================================
# Step 1: Compression Detection (0% FP)
# ============================================================

def compute_block_boundary(img):
    """8x8 block boundary discontinuity ratio. >1.2 suggests JPEG."""
    h, w = img.shape[:2]
    all_scores = []
    for orientation in ['h', 'v']:
        for ch in range(3):
            ch_data = img[:,:,ch].astype(np.float64)
            b_diffs = []; i_diffs = []
            for y in range(0, h-8, 8):
                for x in range(0, w-8, 8):
                    block = ch_data[y:y+8, x:x+8]
                    if orientation == 'h' and y+8 < h:
                        b_diffs.append(abs(block[7,:].mean() - ch_data[y+8, x:x+8].mean()))
                    if orientation == 'v' and x+8 < w:
                        b_diffs.append(abs(block[:,7].mean() - ch_data[y:y+8, x+8].mean()))
                    if orientation == 'h':
                        i_diffs.append(abs(block[3,:].mean() - block[4,:].mean()))
                    else:
                        i_diffs.append(abs(block[:,3].mean() - block[:,4].mean()))
            if b_diffs and i_diffs:
                all_scores.append(np.mean(b_diffs) / max(np.mean(i_diffs), 0.01))
    return np.mean(all_scores) if all_scores else 1.0


def compute_dct_zero_ratio(img_gray):
    """DCT AC coefficient zero ratio. JPEG quantization → many near-zero AC coefficients."""
    h, w = img_gray.shape
    if h < 128 or w < 128:
        return 0.0  # too small for reliable DCT
    zeros = 0; total = 0
    try:
        from scipy.fftpack import dct
        for y in range(0, h-7, 8):
            for x in range(0, w-7, 8):
                block = img_gray[y:y+8, x:x+8].astype(np.float64)
                block -= block.mean()
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                ac = dct_block.flatten()[1:]  # skip DC
                zeros += (np.abs(ac) < 0.5).sum()
                total += len(ac)
    except ImportError:
        return 0.0
    return zeros / max(total, 1)


def step1_compression(target, clean):
    """Compression detection via 8x8 block boundary + unique_G ratio."""
    result = {"verdict": "NO_COMPRESSION", "confidence": "high", "signals": {}}

    # Signal 1: Block boundary (JPEG grid artifact)
    block_boundary = compute_block_boundary(target)
    result["signals"]["block_boundary"] = round(block_boundary, 4)

    # Signal 2: unique_G ratio (target/clean)
    ug_target = min(len(np.unique(target[:,:,c])) for c in range(3))
    ug_clean = min(len(np.unique(clean[:,:,c])) for c in range(3))
    ug_ratio = ug_target / max(ug_clean, 1)
    result["signals"]["unique_G_target"] = ug_target
    result["signals"]["unique_G_clean"] = ug_clean
    result["signals"]["unique_G_ratio"] = round(ug_ratio, 4)

    # Signal 3: Per-channel unique count anomaly
    unique_per_ch = [len(np.unique(target[:,:,c])) for c in range(3)]
    result["signals"]["unique_per_channel"] = unique_per_ch

    # Signal 4: DCT zero ratio (JPEG quantization → many near-zero AC coefficients)
    gray = target.mean(axis=2) if len(target.shape)==3 else target
    dct_zeros = compute_dct_zero_ratio(gray)
    result["signals"]["dct_zero_ratio"] = round(dct_zeros, 4)

    # Multi-signal voting (calibrated from 290 synthetic tests)
    bb_detected = block_boundary > 1.05   # 14400-case校准: 0% FP, 99.5% R1, 100% R2+
    dct_detected = dct_zeros > 0.50       # 14400-case校准: 0% FP, R1=84%, R2+=90%+
    dct_degraded = dct_zeros > 0.20       # DCT fails in composite
    ug_detected = ug_ratio < 0.3 and ug_target < 50  # Only severe quantization

    # Voting: require 2/3 signals for HIGH confidence
    signals_positive = sum([bb_detected, dct_detected, ug_detected])

    if signals_positive >= 2:
        result["verdict"] = "COMPRESSION_DETECTED"
        result["subtype"] = "JPEG"
        result["confidence"] = "high"
    elif bb_detected:
        # Block boundary alone is the strongest signal (0% FP)
        result["verdict"] = "COMPRESSION_DETECTED"
        result["subtype"] = "JPEG"
        result["confidence"] = "high"
    elif dct_detected and not dct_degraded:
        # DCT alone, but not degraded → likely JPEG in pure setting
        result["verdict"] = "COMPRESSION_DETECTED"
        result["subtype"] = "JPEG"
        result["confidence"] = "medium"
    elif dct_detected:
        # DCT detected but possibly degraded → needs verification
        result["verdict"] = "MILD_COMPRESSION_POSSIBLE"
        result["subtype"] = "JPEG"
        result["confidence"] = "low"
        result["note"] = "DCT signal present but may be degraded by noise/blur"
        result["should_still_test"] = True
        result["suggested_action"] = "PSNR-test JPEG at severity 1-3 despite weak signal"
    elif ug_detected:
        result["verdict"] = "COMPRESSION_DETECTED"
        result["subtype"] = "QUANTIZATION_OR_JPEG2000"
        result["confidence"] = "medium"

    # Always suggest testing if verdict is not high-confidence YES
    if result["verdict"] == "NO_COMPRESSION":
        result["should_still_test"] = True
        result["suggested_action"] = "Test 1-2 JPEG candidates via PSNR — mild JPEG can evade block_boundary detection"
    elif result["confidence"] != "high":
        result["should_still_test"] = True

    result["_for_step4"] = {
        "block_uniformity": round(block_boundary, 4),
        "jpeg_detected": result["verdict"] != "NO_COMPRESSION"
    }

    return result


# ============================================================
# Step 2: Noise Detection
# ============================================================

def haar_dwt2_1d(img_channel):
    """Simple Haar wavelet decomposition."""
    h, w = img_channel.shape
    h2, w2 = h // 2, w // 2
    LL = (img_channel[0::2, 0::2] + img_channel[1::2, 0::2] +
          img_channel[0::2, 1::2] + img_channel[1::2, 1::2]) / 4
    LH = (img_channel[0::2, 0::2] - img_channel[1::2, 0::2] +
          img_channel[0::2, 1::2] - img_channel[1::2, 1::2]) / 4
    HL = (img_channel[0::2, 0::2] + img_channel[1::2, 0::2] -
          img_channel[0::2, 1::2] - img_channel[1::2, 1::2]) / 4
    HH = (img_channel[0::2, 0::2] - img_channel[1::2, 0::2] -
          img_channel[0::2, 1::2] + img_channel[1::2, 1::2]) / 4
    return LL, LH, HL, HH

def estimate_noise_sigma(img):
    """Wavelet MAD estimator on HH subband."""
    sigma_per_channel = []
    for c in range(3):
        _, _, _, HH = haar_dwt2_1d(img[:,:,c])
        mad = np.median(np.abs(HH - np.median(HH)))
        sigma_per_channel.append(mad / 0.6745)
    return sigma_per_channel

def step2_noise(target, clean):
    """噪声分类 v16 — 基于数学指纹 (来源: 统计特征分析)。

    六种噪声的物理边界:
    1. Impulse: 残差中绝大多数像素为0 (稀疏性)
    2. Spatially_Correlated: 相邻像素残差高度相关 (>0.3, 3x3 box blur)
    3. YCrCb: RGB 通道间残差互相关 (>0.12, YCrCb→RGB 线性组合)
    4. Gaussian_RGB: 残差方差与亮度无关 (corr < 0.25)
    5. Poisson: 残差方差 ∝ 亮度 (corr > 0.25, var/I ≈ constant)
    6. Speckle: 残差方差 ∝ 亮度² (var/I 仍与亮度相关)
    """
    result = {"verdict": "NO_NOISE", "confidence": "high", "signals": {}}
    residual = compute_residual(target, clean)
    R = residual.astype(np.float64) / 255.0  # 归一化到 [0,1]
    I = clean.astype(np.float64) / 255.0
    h, w = R.shape[:2]

    # Wavelet noise floor
    sigma = estimate_noise_sigma(residual)
    avg_sigma = np.mean(sigma)
    result["signals"]["noise_prior_sigma"] = round(avg_sigma, 2)

    # ================================================================
    # 1. Impulse 检测 — 稀疏性指纹 (残差中大部分像素≈0)
    # ================================================================
    zero_threshold = 1.1 / 255.0
    zero_ratio = np.mean(np.abs(R) < zero_threshold)
    result["signals"]["impulse_zero_ratio"] = round(zero_ratio, 4)
    # 即使 severity=1 (amount=0.01), 也有 99% 像素未改变
    if zero_ratio > 0.85:
        result["verdict"] = "IMPULSE"
        result["confidence"] = "high" if zero_ratio > 0.95 else "medium"
        result["estimated_severity"] = 1 if zero_ratio > 0.98 else (2 if zero_ratio > 0.94 else 3)
        result["_for_step4"] = {"sigma": 0.0, "can_fake_contrast": False,
                                 "can_fake_saturation": False, "can_fake_brightness": False}
        return result

    # Compute spatial_corr signal
    r_g = R[:, :, 1]
    sc_h = np.corrcoef(r_g[:, :-1].flatten(), r_g[:, 1:].flatten())[0, 1] if r_g.shape[1] > 1 else 0
    sc_v = np.corrcoef(r_g[:-1, :].flatten(), r_g[1:, :].flatten())[0, 1] if r_g.shape[0] > 1 else 0
    spatial_corr = (sc_h + sc_v) / 2.0
    result["signals"]["spatial_corr"] = round(spatial_corr, 4)

    # ================================================================
    # 2. Spatially_Correlated — 暂时关闭
    #    原因: JPEG 8×8 块 + blur 产生 spatial_corr 0.68-0.85，
    #          与真 SC 噪声 (0.53-0.70) 不可区分 (同图模式)
    #    exp27 证明: 3/16 的 SC 检测来自 JPEG+blur FP
    # ================================================================
    # if spatial_corr > 0.72:
    #     result["verdict"] = "SPATIALLY_CORRELATED"
    #     ...
    # elif spatial_corr > 0.50 and avg_sigma >= 1.4:
    #     result["verdict"] = "SPATIALLY_CORRELATED"
    #     ...

    # ================================================================
    # 3. YCrCb — 通道间相关指纹 (sc低 + cc高 → YCrCb)
    # ================================================================
    r_R = R[:, :, 0].flatten()
    r_G = R[:, :, 1].flatten()
    r_B = R[:, :, 2].flatten()
    corr_matrix = np.corrcoef([r_R, r_G, r_B])
    cross_ch_corr = (abs(corr_matrix[0, 1]) + abs(corr_matrix[0, 2]) + abs(corr_matrix[1, 2])) / 3.0
    result["signals"]["cross_channel_corr"] = round(cross_ch_corr, 4)
    # RGB 独立加噪: <0.05; YCrCb 转 RGB: >0.15
    # ⚠️ 同图模式 FP 风险: blur/JPEG 也产生跨通道相关 (exp28: 3例FP)
    #   blur: RGB通道同步平滑 → cc升高
    #   JPEG: YCrCb色彩空间操作 → cc升高
    #   前置条件: 如果 blur 或 JPEG 被检测到, YCrCb 置信度必须降级
    if cross_ch_corr > 0.12:
        # 二次判定: 残差是否随像素强度变化? (gamma=确定性函数 YCrCb=随机噪声)
        # exp28校准: ric阈值从0.25降至0.15 (0009 ric=0.09未触发, 0010 ric=0.21未触发)
        R_mag = np.abs(R).mean(axis=2).flatten()
        I_flat = I.mean(axis=2).flatten()
        # 降采样加速 (大图上采样)
        if len(I_flat) > 50000:
            idx = np.random.choice(len(I_flat), 50000, replace=False)
            R_mag_sample = R_mag[idx]
            I_sample = I_flat[idx]
        else:
            R_mag_sample, I_sample = R_mag, I_flat
        residual_intensity_corr = np.corrcoef(I_sample, R_mag_sample)[0, 1]
        result["signals"]["residual_intensity_corr"] = round(residual_intensity_corr, 4)

        if abs(residual_intensity_corr) > 0.15:
            # 残差与强度相关 → 确定性变换 (gamma/contrast), 非随机噪声
            # exp28: 阈值从0.25降至0.15 (0009 ric=0.09未触发, 0010 ric=0.21→0.21>0.15触发)
            result["signals"]["gamma_suspect_from_noise"] = True
            result["verdict"] = "GAUSSIAN_YCrCb"
            result["confidence"] = "low"  # 降级 — 可能是 gamma FP
            result["estimated_severity"] = 2 if avg_sigma > 10 else 3
            result["_for_step4"] = {"sigma": round(avg_sigma, 2), "can_fake_contrast": avg_sigma > 8,
                                     "can_fake_saturation": avg_sigma > 5, "can_fake_brightness": False}
            return result
        else:
            result["verdict"] = "GAUSSIAN_YCrCb"
            result["confidence"] = "high" if cross_ch_corr > 0.20 else "medium"
            result["estimated_severity"] = 2 if avg_sigma > 10 else (3 if avg_sigma > 20 else (4 if avg_sigma > 30 else 5))
            result["_for_step4"] = {"sigma": round(avg_sigma, 2), "can_fake_contrast": avg_sigma > 8,
                                     "can_fake_saturation": avg_sigma > 5, "can_fake_brightness": False}
            return result

    # ================================================================
    # 4. Gaussian/Poisson/Speckle — 亮度-方差关系指纹
    # ================================================================
    I_gray = I.mean(axis=2)
    R_gray = np.abs(R).mean(axis=2)

    # 自适应 patch 大小 (最小8px, 保证足够样本)
    patch_size = max(8, min(16, min(h, w) // 20))
    means, variances = [], []
    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):
            patch_I = I_gray[y:y+patch_size, x:x+patch_size]
            patch_R = R_gray[y:y+patch_size, x:x+patch_size]
            m = np.mean(patch_I)
            if 0.05 < m < 0.95:
                means.append(m)
                variances.append(np.var(patch_R))

    if len(means) >= 10:
        means = np.array(means)
        variances = np.array(variances)
        corr_mean_var = np.corrcoef(means, variances)[0, 1]
        result["signals"]["intensity_var_corr"] = round(corr_mean_var, 4)

        if np.mean(variances) < 1e-6:
            result["signals"]["gm_ratio_usable"] = True
            return result

        if corr_mean_var < 0.15:
            result["verdict"] = "GAUSSIAN_RGB"
            result["confidence"] = "high" if abs(corr_mean_var) < 0.10 else "medium"
        else:
            scaled_var = variances / (means + 1e-6)
            corr_scaled = np.corrcoef(means, scaled_var)[0, 1]
            result["signals"]["intensity_scaled_var_corr"] = round(corr_scaled, 4)

            if corr_scaled < 0.3:
                result["verdict"] = "POISSON"
                result["confidence"] = "high" if corr_mean_var > 0.5 else "medium"
            else:
                result["verdict"] = "SPECKLE"
                result["confidence"] = "high" if corr_scaled > 0.5 else "medium"
    else:
        # 无足够平坦区域 → 回退 (SC 已关闭)
        if avg_sigma < 1.9:
            result["signals"]["gm_ratio_usable"] = True
            return result
        else:
            result["verdict"] = "GAUSSIAN_RGB"
            result["confidence"] = "low"

    # Severity estimation
    sev_map = {2:1, 5:2, 10:3, 20:4, 35:5}
    if result["verdict"] == "POISSON":
        sev_map = {5:1, 9:2, 11:3, 15:4, 17:5}
    sev = 1
    for threshold, s in sorted(sev_map.items()):
        if avg_sigma > threshold: sev = s
    result["estimated_severity"] = sev

    # For Step4 cross-validation
    result["_for_step4"] = {
        "sigma": round(avg_sigma, 2),
        "can_fake_contrast": avg_sigma > 8,
        "can_fake_saturation": avg_sigma > 5,
        "can_fake_brightness": False
    }

    return result


# ============================================================
# Step 3: Blur Detection (with MTF signals)
# ============================================================

def compute_gm_ratio(img):
    """Gradient magnitude ratio (simplified)."""
    gx = np.abs(np.diff(img, axis=1))
    gy = np.abs(np.diff(img, axis=0))
    h, w = img.shape[:2]
    gx_full = np.pad(gx, ((0,0),(0,1),(0,0)), mode='edge')
    gy_full = np.pad(gy, ((0,1),(0,0),(0,0)), mode='edge')
    gm = np.sqrt(gx_full**2 + gy_full**2).mean()
    return gm


def compute_jitter_signal(target, clean):
    """Jitter blur: shuffle_pixels swaps nearby pixels → high change% + low magnitude.
    Returns (nonzero_pct, residual_std, jitter_score). jitter_score > 0.5 suggests jitter."""
    diff = np.abs(target.astype(float) - clean.astype(float))
    nonzero_pct = (diff.max(axis=2) > 5).mean()
    residual_std = diff.std()
    # Jitter: high change% (>50%) + low magnitude (std < 30) + not noise (noise has higher std)
    jitter_score = nonzero_pct * (1.0 - min(residual_std / 60.0, 1.0))
    return round(nonzero_pct, 4), round(residual_std, 1), round(jitter_score, 4)


def compute_speckle_severity(clean, residual):
    """Speckle severity: combined sigma + vm_slope signal.
    Returns (sigma, vm_slope, estimated_severity)."""
    sigma = residual.std()
    flat_mask = np.std(clean, axis=2) < 20
    vm_slope = 0.0
    if flat_mask.sum() > 100:
        intensities = clean[flat_mask].mean(axis=1)
        variances = residual[flat_mask].var(axis=1)
        bins = np.digitize(intensities, np.percentile(intensities, [25, 50, 75]))
        bin_vars = [variances[bins == i].mean() for i in range(4) if (bins == i).sum() > 5]
        bin_means = [intensities[bins == i].mean() for i in range(4) if (bins == i).sum() > 5]
        if len(bin_means) >= 3:
            coeffs = np.polyfit(bin_means, bin_vars, 1)
            vm_slope = coeffs[0]
    # Severity lookup: (sigma, vm_slope) → sev (calibrated on speckle sev 1-5)
    sev = 1
    if sigma > 50 and vm_slope > 10: sev = 4
    elif sigma > 40 and vm_slope > 7: sev = 3
    elif sigma > 25 and vm_slope > 3: sev = 2
    elif sigma > 10 and vm_slope > 0.5: sev = 1
    return round(sigma, 1), round(vm_slope, 4), sev


def compute_extreme_spatial_clustering(target, clean):
    """Distinguish oversharpen (clustered extremes near edges) vs impulse (random extremes).
    Returns (extreme_pct, spatial_clustering). clustering > 0.4 → oversharpen, < 0.2 → impulse."""
    residual = target.astype(float) - clean.astype(float)
    extreme_mask = np.abs(residual).max(axis=2) > 50
    extreme_pct = extreme_mask.mean()
    if extreme_pct < 0.001:
        return 0.0, 0.0
    ext_map = extreme_mask.astype(float)
    h, w = ext_map.shape
    if h > 2 and w > 2:
        right_corr = np.corrcoef(ext_map[:, :-1].flatten(), ext_map[:, 1:].flatten())[0, 1]
        down_corr = np.corrcoef(ext_map[:-1, :].flatten(), ext_map[1:, :].flatten())[0, 1]
        clustering = (right_corr + down_corr) / 2
    else:
        clustering = 0.0
    return round(extreme_pct * 100, 2), round(float(clustering), 4)

def compute_radial_mtf(img):
    """Radial MTF analysis."""
    h, w = img.shape[:2]
    gray = img.mean(axis=2) if len(img.shape)==3 else img
    fft = np.abs(np.fft.fftshift(np.fft.fft2(gray - gray.mean())))
    cy, cx = h//2, w//2
    max_r = min(cy, cx)
    radial = []
    for r in range(1, max_r, 2):
        y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
        mask = (np.sqrt(x**2 + y**2) >= r) & (np.sqrt(x**2 + y**2) < r+2)
        radial.append(fft[mask].mean())
    return np.array(radial)


def compute_spectral_slope(img):
    """Log-power vs log-frequency slope (mid-frequencies 10-100 px).
    More negative = more blur. Noise adds flat offset → slope barely changes.
    Pure: -1.1 (no blur) to -1.8 (sev5). Noise: -1.08 (sev5, no blur).
    Blur+Noise sev5+sev5: -1.60 (blur dominates)."""
    gray = img.mean(axis=2) if len(img.shape) == 3 else img
    h, w = gray.shape
    fft = np.abs(np.fft.fftshift(np.fft.fft2(gray - gray.mean())))
    cy, cx = h // 2, w // 2
    max_r = min(cy, cx) - 5
    freqs, powers = [], []
    for r in range(5, max_r, 3):
        y, x = np.ogrid[-cy:h - cy, -cx:w - cx]
        mask = (np.sqrt(x**2 + y**2) >= r) & (np.sqrt(x**2 + y**2) < r + 3)
        if mask.sum() > 10:
            freqs.append(r)
            powers.append(fft[mask].mean())
    freqs = np.array(freqs)
    powers = np.array(powers)
    mid = (freqs > 10) & (freqs < min(100, max_r - 10))
    if mid.sum() < 5:
        return 0.0
    log_f = np.log(freqs[mid])
    log_p = np.log(powers[mid] + 1)
    slope, _ = np.polyfit(log_f, log_p, 1)
    return float(slope)

def step3_blur(target, clean, noise_info):
    """Blur detection using gm_ratio + spectral_slope (noise-robust) + MTF signals."""
    result = {"verdict": "NO_BLUR", "confidence": "medium", "signals": {}}
    gm_ratio_usable = noise_info.get("signals", {}).get("gm_ratio_usable", True)
    has_noise = noise_info.get("verdict", "NO_NOISE") != "NO_NOISE"

    target_gm = compute_gm_ratio(target)
    clean_gm = compute_gm_ratio(clean)
    gm_ratio = target_gm / max(clean_gm, 1e-6)
    result["signals"]["gm_ratio"] = round(gm_ratio, 4)
    result["signals"]["gm_ratio_usable"] = gm_ratio_usable

    # Spectral slope: noise-robust blur signal
    # Blur steepens slope (more negative); noise barely affects it (±0.03)
    target_slope = compute_spectral_slope(target)
    clean_slope = compute_spectral_slope(clean)
    slope_ratio = target_slope / min(clean_slope, -0.01)  # > 1 when target has steeper slope (more blurred)
    result["signals"]["spectral_slope_target"] = round(target_slope, 4)
    result["signals"]["spectral_slope_clean"] = round(clean_slope, 4)
    result["signals"]["spectral_slope_ratio"] = round(slope_ratio, 4)

    # gm_ratio threshold: pure=0.62, with noise=0.72
    # BUT noise completely destroys gm_ratio (2.0+ even for sev=1 noise + sev=5 blur).
    # When noise present, rely on spectral_slope instead.
    blur_threshold = 0.72 if has_noise else 0.62
    gm_detects_blur = gm_ratio < blur_threshold
    # Spectral slope detects blur: slope_ratio > 1.15 (target significantly steeper than clean)
    # Calibrated: pure blur sev≥2 → slope_ratio > 1.12; blur+noise sev≥2 → slope_ratio > 1.05
    slope_detects_blur = slope_ratio > 1.08
    # When noise present, trust spectral_slope over gm_ratio
    blur_detected = slope_detects_blur if has_noise else gm_detects_blur

    if not blur_detected:
        result["verdict"] = "BLUR_UNLIKELY"
        result["confidence"] = "low" if has_noise else "medium"
        result["should_still_test"] = True
        if has_noise:
            result["note"] = f"gm_ratio={gm_ratio:.3f} (noise-destroyed), spectral_slope_ratio={slope_ratio:.3f} (threshold=1.08)"
        else:
            result["note"] = f"gm_ratio={gm_ratio:.3f} > {blur_threshold}"
        result["suggested_action"] = "Test 2-3 blur candidates via PSNR — noise/blur coupling can mask blur in both gm and spectral signals"
        return result

    # Blur detected — simplified sub-type classification
    # Only MOTION has a reliable distinguishing signal (anisotropy on residual FFT).
    # Lens/Glass/Zoom/Gaussian MTF signals are content-dominated → unreliable.
    # Gaussian ↔ Lens 几乎可互换 (REFLECTION_MECHANISM: 1-4dB gap), Agent 靠 PSNR 区分.

    # Motion detection: anisotropy on residual FFT (not target — content-dominated)
    residual = target.astype(float) - clean.astype(float)
    residual_gray = residual.mean(axis=2)
    h, w = residual_gray.shape
    cy_r, cx_r = h//2, w//2
    fft_res = np.abs(np.fft.fftshift(np.fft.fft2(residual_gray - residual_gray.mean())))
    angles = np.linspace(0, np.pi, 36)
    res_energy = []
    for a in angles:
        x, y = np.cos(a), np.sin(a)
        vals = []
        for rad in range(5, min(h, w)//4, 3):
            px, py = int(cx_r + rad*x), int(cy_r + rad*y)
            if 0 <= px < w and 0 <= py < h:
                vals.append(fft_res[py, px])
        res_energy.append(np.mean(vals) if vals else 0)
    res_energy = np.array(res_energy)
    res_anisotropy = res_energy.max() / res_energy.mean() if res_energy.mean() > 0 else 1.0
    result["signals"]["residual_anisotropy"] = round(res_anisotropy, 4)

    # Also compute legacy MTF signals as hints for the Agent (not for classification)
    target_mtf = compute_radial_mtf(target)
    clean_mtf = compute_radial_mtf(clean)
    kernel_mtf = target_mtf / (clean_mtf + 1e-6)
    mtf_diff = np.abs(np.diff(kernel_mtf))
    result["signals"]["mtf_roughness"] = round(mtf_diff.std() / (mtf_diff.mean() + 1e-6), 4)

    # --- Simplified classification (v15: Glass 分数唯一可用附加信号) ---
    # Motion: 残差各向异性受图像内容主导(FP>80%) → 不可自动区分, Agent PSNR 测试
    # Lens/Gaussian: 数学上不可区分 (disk vs gaussian 核差异被图像内容淹没)
    # Glass: glass_score > 0.25 (Laplacian/gradient 比, pixel shuffle 高频残留)
    #   Gaussian(2-4): 0.04-0.13, Lens(2-4): 0.08-0.20, Glass(2-4): 0.19-0.41
    gray_target = target.mean(axis=2)
    lap = np.abs(np.diff(gray_target, 2, axis=0)[:, :-2]) + np.abs(np.diff(gray_target, 2, axis=1)[:-2, :])
    grad = np.abs(np.diff(gray_target, axis=0)[:, :-1]) + np.abs(np.diff(gray_target, axis=1)[:-1, :])
    glass_score = lap.var() / (grad.var() + 0.01)
    result["signals"]["glass_score"] = round(glass_score, 4)

    # 两阶段模糊分类: 梯度域联合核提取 → PCA 统计特征决策树
    # exp28 验证: lens 4/4, zoom 4/4, motion≥3 3/3, gaussian 4/4
    blur_subtype = "blur_gaussian"
    subtype_confidence = "low"
    try:
        from blur_kernel_recovery import identify_blur_type
        # Save temp files for identify_blur_type (it reads from disk)
        import tempfile, os as _os
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tf, \
             tempfile.NamedTemporaryFile(suffix='.png', delete=False) as cf:
            Image.fromarray(target.clip(0,255).astype(np.uint8)).save(tf.name)
            clean_img = clean if len(clean.shape) == 2 else clean.mean(axis=2)
            Image.fromarray(clean_img.clip(0,255).astype(np.uint8)).save(cf.name)
            tf.close(); cf.close()
            blur_subtype, subtype_confidence, kfeatures = identify_blur_type(tf.name, cf.name)
            _os.unlink(tf.name); _os.unlink(cf.name)
        result["signals"]["blur_pca_ratio"] = kfeatures.get("pca_ratio", 0)
        result["signals"]["blur_density"] = kfeatures.get("density", 0)
        result["signals"]["blur_flatness"] = kfeatures.get("flatness", 0)
        result["signals"]["blur_glass_noise_floor"] = kfeatures.get("glass_noise_floor", 0)
        result["signals"]["blur_kernel_method"] = "gradient_joint_wiener_pca"
    except Exception as e:
        result["signals"]["blur_kernel_error"] = str(e)[:100]

    # 🔴 多图校准 (10 images, 3840 cases):
    #    PCA>2.5: 41% recall, 0% FP. PRIMARY signal.
    #    FFT>2.2: 5% recall, 0.2% FP. Fallback when kernel corrupted.
    #    noise→blur: PCA=56%. blur→noise: PCA=3% (kernel destroyed).
    pca_val = kfeatures.get('pca_ratio', 0)
    if blur_subtype in ["blur_motion", "blur_lens"] and subtype_confidence in ["high", "medium"]:
        result["verdict"] = blur_subtype.upper().replace("BLUR_", "") + "_BLUR"
        result["confidence"] = subtype_confidence
        result["blur_subtype"] = blur_subtype
        result["subtype_confidence"] = subtype_confidence
        result["note"] = f"PCA: {blur_subtype} (pca={pca_val:.2f}>2.5, 41%recall 0%FP)"
    elif pca_val > 2.5:
        # PCA 检测到 motion (0% FP guaranteed across 2880 non-motion cases)
        result["verdict"] = "MOTION_BLUR"
        result["blur_subtype"] = "blur_motion"
        result["subtype_confidence"] = "high" if pca_val > 3.5 else "medium"
        result["confidence"] = result["subtype_confidence"]
        result["note"] = f"PCA: pca={pca_val:.2f}>2.5 → motion (0% FP). FFT={res_anisotropy:.2f}"
    elif res_anisotropy > 2.2:
        # FFT 备选: 5% recall, 0.2% FP. 仅在 PCA 失败时使用 (kernel corrupted by noise)
        result["verdict"] = "MOTION_BLUR"
        result["blur_subtype"] = "blur_motion"
        result["subtype_confidence"] = "low"
        result["confidence"] = "low"
        result["note"] = f"FFT: {res_anisotropy:.2f}>2.2 → motion (5%recall 0.2%FP). PCA failed (pca={pca_val:.1f}, kernel likely corrupted by noise)."
    elif glass_score > 0.25:
        result["verdict"] = "GLASS_BLUR"
        result["blur_subtype"] = "blur_glass"
        result["subtype_confidence"] = "medium" if glass_score > 0.35 else "low"
        result["note"] = f"glass_score={glass_score:.3f}>0.25 (pixel shuffle高频残留)."
    else:
        result["verdict"] = "GAUSSIAN_BLUR"
        result["blur_subtype"] = "blur_gaussian"
        result["subtype_confidence"] = "medium" if gm_ratio_usable else "low"
        result["note"] = f"Cepstral PCA subtype={blur_subtype}({subtype_confidence}), fallback to Gaussian. Agent应PSNR测试替代子类型."

    # Severity estimation (prefer gm_ratio when usable, fall back to spectral slope)
    if gm_ratio_usable:
        if gm_ratio < 0.35: sev = 5
        elif gm_ratio < 0.50: sev = 4
        elif gm_ratio < 0.65: sev = 3
        elif gm_ratio < 0.78: sev = 2
        else: sev = 1
        result["estimated_severity"] = sev
        result["signals"]["severity_source"] = "gm_ratio"
    elif slope_ratio > 1.01:
        # Spectral-slope-based severity (noise-robust, calibrated on blur+noise combos)
        if slope_ratio > 1.50: sev = 5
        elif slope_ratio > 1.30: sev = 4
        elif slope_ratio > 1.18: sev = 3
        elif slope_ratio > 1.08: sev = 2
        else: sev = 1
        result["estimated_severity"] = sev
        result["signals"]["severity_source"] = "spectral_slope"
    else:
        result["estimated_severity"] = None
        result["signals"]["severity_source"] = "none"

    return result


# ============================================================
# Step 4: Global Degradation (with cross-validation)
# ============================================================

def step4_global(target, clean, step1_info, step2_info):
    """Global degradation detection with false-positive cross-validation."""
    result = {"verdict": "NO_GLOBAL", "signals": {}, "false_positive_checks": {}}

    # --- Brightness ---
    mean_shift = (target.mean() - clean.mean()) / 255.0
    mean_shifts_rgb = [(target[:,:,c].mean() - clean[:,:,c].mean()) / 255.0 for c in range(3)]
    all_same_dir = all(s > 0.01 for s in mean_shifts_rgb) or all(s < -0.01 for s in mean_shifts_rgb)

    brightness_detected = abs(mean_shift) > 0.014 and all_same_dir  # calibrated: 82% recall, 3.3% FP
    result["signals"]["brightness_mean_shift"] = round(mean_shift * 255, 1)
    result["signals"]["brightness_detected_raw"] = brightness_detected

    # Cross-validate: JPEG can cause false brightness
    if brightness_detected and step1_info.get("verdict") == "COMPRESSION_DETECTED":
        block_uniformity = step1_info.get("_for_step4", {}).get("block_uniformity", 1.0)
        result["false_positive_checks"]["brightness_from_jpeg"] = block_uniformity > 0.1
        if block_uniformity > 0.1:
            brightness_detected = False  # JPEG artifact, exclude

    # --- Clipping detection (高 sev brightness/contrast 导致像素裁剪) ---
    # 裁剪会改变方差, 可能导致 contrast FP. 裁剪本身容易识别: 大量像素在 0 或 255.
    clipped_frac = np.mean((target <= 1) | (target >= 254))  # float64 [0,255] range
    result["signals"]["clipped_fraction"] = round(float(clipped_frac), 4)
    heavy_clipping = clipped_frac > 0.05  # >5% 像素被裁剪
    result["signals"]["heavy_clipping"] = heavy_clipping

    # --- Contrast ---
    # variance_ratio: 区分 contrast (方差变,均值不变) vs brightness (均值变,方差不变)
    t_var = target.var()
    c_var = clean.var()
    variance_ratio = t_var / max(c_var, 1)
    result["signals"]["variance_ratio"] = round(variance_ratio, 4)

    t_spread = sum(np.percentile(target[:,:,c], 95) - np.percentile(target[:,:,c], 5) for c in range(3))
    c_spread = sum(np.percentile(clean[:,:,c], 95) - np.percentile(clean[:,:,c], 5) for c in range(3))
    spread_ratio = t_spread / max(c_spread, 1)

    # 二阶判定: contrast_weaken→方差减小, contrast_strengthen→方差增大
    # 校准 (10图×5sev): c_weaken 0.08-0.40, c_strengthen(sev≥2) 1.84-2.74, brightness 0.91-0.97
    contrast_from_variance = variance_ratio < 0.55 or variance_ratio > 1.80
    result["signals"]["contrast_from_variance"] = contrast_from_variance
    # spread_ratio 作为补充 (percentile-based, 对极值更鲁棒)
    contrast_detected = contrast_from_variance or (spread_ratio < 0.80 or spread_ratio > 1.20)
    result["signals"]["contrast_spread_ratio"] = round(spread_ratio, 4)
    result["signals"]["contrast_detected_raw"] = contrast_detected

    # --- Saturation (must be detected BEFORE contrast cross-validation) ---
    # YCrCb space
    def rgb_to_ycrcb(img):
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        y = 0.299*r + 0.587*g + 0.114*b
        cb = 128 - 0.168736*r - 0.331264*g + 0.5*b
        cr = 128 + 0.5*r - 0.418688*g - 0.081312*b
        return y, cb, cr

    _, cb_t, cr_t = rgb_to_ycrcb(target)
    _, cb_c, cr_c = rgb_to_ycrcb(clean)
    chroma_ratio = (cr_t.std() + cb_t.std()) / max(cr_c.std() + cb_c.std(), 1)
    saturation_detected = chroma_ratio < 0.85 or chroma_ratio > 1.15
    result["signals"]["saturation_chroma_ratio"] = round(chroma_ratio, 4)
    result["signals"]["saturation_detected_raw"] = saturation_detected

    # Cross-validate: 高 sev brightness 的像素裁剪会降低方差 → 可能误判 contrast
    if contrast_from_variance and brightness_detected and heavy_clipping:
        # 裁剪解释了方差变化 → 排除 contrast FP
        result["false_positive_checks"]["contrast_from_clipping"] = True
        # 仅当 spread_ratio 也异常时才保留 contrast (spread 对 clipping 更鲁棒)
        if not (spread_ratio < 0.75 or spread_ratio > 1.25):
            contrast_detected = False
            result["signals"]["contrast_from_variance"] = False
            result["signals"]["contrast_clipping_corrected"] = True

    # Cross-validate: saturation can fake contrast (exp28 0004: saturation→方差↓→contrast FP)
    if contrast_detected and saturation_detected:
        result["false_positive_checks"]["contrast_from_saturation"] = True
        # saturation改变chroma也会改变方差分布 → 排除contrast
        if not (spread_ratio < 0.70 or spread_ratio > 1.30):
            contrast_detected = False
            result["signals"]["contrast_from_variance"] = False
            result["signals"]["contrast_saturation_corrected"] = True

    # Cross-validate: noise can fake contrast
    if contrast_detected and step2_info.get("verdict", "NO_NOISE") != "NO_NOISE":
        sigma = step2_info.get("_for_step4", {}).get("sigma", 0)
        if sigma > 5:
            noise_spread_boost = sigma * 2 * 3  # ~2σ * 3 channels
            clean_spread_est = c_spread + noise_spread_boost
            noise_corrected_ratio = t_spread / max(clean_spread_est, 1)
            result["signals"]["contrast_noise_corrected"] = round(noise_corrected_ratio, 4)
            result["false_positive_checks"]["contrast_from_noise"] = abs(noise_corrected_ratio - 1) < 0.15
            if abs(noise_corrected_ratio - 1) < 0.15:
                contrast_detected = False  # noise artifact, exclude

    # Cross-validate: noise can fake saturation
    if saturation_detected and step2_info.get("verdict", "NO_NOISE") != "NO_NOISE":
        sigma = step2_info.get("_for_step4", {}).get("sigma", 0)
        if sigma > 3:
            # Noise adds ~sigma to each of Cr and Cb
            chroma_clean_est = cr_c.std() + cb_c.std() + 2 * sigma
            noise_corrected_chroma = (cr_t.std() + cb_t.std()) / max(chroma_clean_est, 1)
            result["false_positive_checks"]["saturation_from_noise"] = abs(noise_corrected_chroma - 1) < 0.15
            if abs(noise_corrected_chroma - 1) < 0.15:
                saturation_detected = False  # noise artifact, exclude

    # --- Cross-validate: brightness vs contrast vs gamma ---
    # 关键区分 (同图模式):
    #   brightness: mean_shift ✓, variance_ratio ≈ 1.0
    #   contrast:   mean_shift ✗, variance_ratio ≠ 1.0
    #   gamma:      mean_shift ✓, variance_ratio ≠ 1.0 (两者都变)
    #   ⚠️ exp28 0013: heavy_clipping 也会导致方差变化 → gamma FP
    if brightness_detected and contrast_detected:
        if heavy_clipping:
            # 裁剪解释了方差变化 → 不是 gamma, 保持 brightness
            # exp28 0013: clipping 20%+ → vr=0.29 → gamma_suspect FP
            result["false_positive_checks"]["gamma_from_clipping"] = True
            result["signals"]["gamma_suspect"] = False  # 抑制
            # 保留 brightness, 丢弃 contrast (来自裁剪)
            contrast_detected = False
            result["signals"]["contrast_from_variance"] = False
            result["signals"]["brightness_clip_explains_variance"] = True
        else:
            # 两者都触发 → 可能是 gamma (均值+方差都变)
            result["false_positive_checks"]["brightness_contrast_both"] = True
            result["signals"]["gamma_suspect"] = True
            # 降级 brightness confidence — 让 Agent 测试 gamma 替代
            brightness_detected = False  # 只用 contrast, gamma 留给 Agent PSNR 测试
            result["signals"]["brightness_masked_by_gamma_suspect"] = True
    elif brightness_detected and not contrast_detected:
        # 均值变但方差不变 → 纯 brightness, 高置信
        result["false_positive_checks"]["brightness_pure"] = abs(variance_ratio - 1.0) < 0.12
    elif not brightness_detected and contrast_detected:
        # 方差变但均值不变 → 纯 contrast, 相对可靠
        result["false_positive_checks"]["contrast_pure"] = True

    # --- Oversharpen check (暂时关闭: blur/noise 耦合时召回率低) ---
    # from scipy.ndimage import laplace as nd_laplace
    # if len(target.shape) == 3:
    #     t_lap = np.array([np.var(nd_laplace(target[:,:,c].astype(np.float64))) for c in range(3)]).mean()
    #     c_lap = np.array([np.var(nd_laplace(clean[:,:,c].astype(np.float64))) for c in range(3)]).mean()
    # else:
    #     t_lap = np.var(nd_laplace(target.astype(np.float64)))
    #     c_lap = np.var(nd_laplace(clean.astype(np.float64)))
    # lap_energy_ratio = t_lap / max(c_lap, 0.01)
    # result["signals"]["laplacian_energy_ratio"] = round(lap_energy_ratio, 4)
    # oversharpen_detected = False
    # if lap_energy_ratio > 3.0:
    #     extreme_pct, clustering = compute_extreme_spatial_clustering(target, clean)
    #     if clustering > 0.30:
    #         oversharpen_detected = True
    # result["signals"]["oversharpen_detected_raw"] = oversharpen_detected
    result["signals"]["oversharpen_detected_raw"] = False  # 暂时关闭

    # --- Final verdict ---
    detections = []
    if brightness_detected:
        direction = "brighten" if mean_shift > 0 else "darken"
        severity = min(5, max(1, int(abs(mean_shift) / 0.07) + 1))
        detections.append(f"BRIGHTNESS_{direction}_sev{severity}")
    if contrast_detected:
        direction = "weaken" if spread_ratio < 1 else "strengthen"
        severity = min(5, max(1, int(abs(spread_ratio - 1) / 0.08) + 1))
        detections.append(f"CONTRAST_{direction}_sev{severity}")
    if saturation_detected:
        direction = "weaken" if chroma_ratio < 1 else "strengthen"
        severity = min(5, max(1, int(abs(chroma_ratio - 1) / 0.08) + 1))
        detections.append(f"SATURATION_{direction}_sev{severity}")
    # if oversharpen_detected:  # 暂时关闭
    #     severity = min(5, max(1, int((lap_energy_ratio - 1) / 2) + 1))
    #     detections.append(f"OVERSHARPEN_sev{severity}")

    result["verdict"] = "; ".join(detections) if detections else "NO_GLOBAL"
    # Confidence: brightness is high-confidence, contrast/saturation are low-confidence (calibration F1 < 0.5)
    if brightness_detected and not (contrast_detected or saturation_detected):
        result["confidence"] = "high"
    elif brightness_detected:
        result["confidence"] = "medium"
    else:
        result["confidence"] = "low"  # contrast/saturation are unreliable signals

    return result


# ============================================================
# Main: Run all steps
# ============================================================

def run_full_analysis(target_path, clean_path):
    target = load_image(target_path)
    clean = load_image(clean_path)

    report = OrderedDict()

    # Step 1: Compression
    report["step1_compression"] = step1_compression(target, clean)

    # Step 2: Noise
    report["step2_noise"] = step2_noise(target, clean)

    # Step 3: Blur (with noise info)
    report["step3_blur"] = step3_blur(target, clean, report["step2_noise"])

    # Step 4: Global (with cross-validation from step1+step2)
    report["step4_global"] = step4_global(
        target, clean,
        report["step1_compression"],
        report["step2_noise"]
    )

    # --- Summary ---
    detections = []
    recommended_search = []

    if report["step1_compression"]["verdict"] != "NO_COMPRESSION":
        detections.append(f"compression: {report['step1_compression']['verdict']}")
        recommended_search.append("compression candidates: JPEG, JPEG2000")
    else:
        recommended_search.append("compression: none detected")

    if report["step2_noise"]["verdict"] != "NO_NOISE":
        noise_label = f"noise: {report['step2_noise']['verdict']} (sev~{report['step2_noise']['estimated_severity']})"
        if report["step2_noise"].get("subjective_guess"):
            noise_label += f" [subjective: {report['step2_noise']['subjective_guess'][:80]}...]"
        detections.append(noise_label)
        guess_types = ["POISSON"]  # SC 已关闭
        if report['step2_noise']['verdict'] in guess_types:
            recommended_search.append(f"noise candidates: {report['step2_noise']['verdict']} (speculative) + GAUSSIAN_RGB, SPECKLE (verify via §B stats)")
        else:
            recommended_search.append(f"noise candidates: {report['step2_noise']['verdict']}+related types")
    else:
        recommended_search.append("noise: none detected")

    if report["step3_blur"]["verdict"] != "NO_BLUR":
        detections.append(f"blur: {report['step3_blur']['verdict']}")
        gm_usable = report["step3_blur"]["signals"]["gm_ratio_usable"]
        fft_a = report["step3_blur"]["signals"].get("residual_anisotropy", 0)
        pca = report["step3_blur"]["signals"].get("blur_pca_ratio", 0)
        motion_hint = ""
        if fft_a > 1.7:
            motion_hint = f" [FFT={fft_a:.2f}>1.7→motion(94%recall),PCA={pca:.1f}]"
        recommended_search.append(f"blur: {report['step3_blur']['verdict']}{motion_hint} (gm_ratio_usable={gm_usable})")
    else:
        recommended_search.append("blur: none detected")

    if report["step4_global"]["verdict"] != "NO_GLOBAL":
        detections.append(f"global: {report['step4_global']['verdict']}")
        fp_checks = report["step4_global"]["false_positive_checks"]
        recommended_search.append(f"global candidates (FP checks: {fp_checks})")
    else:
        recommended_search.append("global: none detected")

    # --- Failure Risk Analysis ---
    risks = []
    reflections = []

    s1 = report["step1_compression"]; s2 = report["step2_noise"]
    s3 = report["step3_blur"]; s4 = report["step4_global"]

    has_noise = s2["verdict"] != "NO_NOISE"
    has_blur = s3["verdict"] != "NO_BLUR"
    has_comp = s1["verdict"] != "NO_COMPRESSION" and "MILD" not in s1["verdict"]
    gm = s3["signals"]["gm_ratio"]
    bb = s1["signals"]["block_boundary"]
    dct = s1["signals"]["dct_zero_ratio"]

    # --- Reflection strategies (来源: 7-case reflection experiment, 71% success) ---
    # JPEG: 3/3 success, 2/3 pixel-perfect match. PSNR verification is reliable for compression.
    # Blur: 2/2 improvement but marginal (+0.1~2.0dB). PSNR gap between blur subtypes <3dB.
    # Noise: PSNR-based reflection fails. Must use statistical checks instead.
    # Global: PSNR-based reflection fails. Already well-identified (90% recall, 2% FP).

    # Risk 1: Missing compression — HIGH priority, PSNR verification works
    if has_noise and not has_comp and bb < 1.05:
        risks.append("MISS_comp_BY_NOISE")
        reflections.append("PSNR test JPEG(sev 1-5): 3/3 success in reflection experiment, 2x pixel-perfect match found")
    elif not has_comp:
        risks.append("MISS_comp_POSSIBLE")
        reflections.append("PSNR test 2-3 JPEG candidates — compression is deterministic, PSNR verification reliable")

    # Risk 2: Missing blur — MEDIUM priority, PSNR works but gains marginal
    if has_noise and not has_blur:
        risks.append("MISS_blur_BY_NOISE")
        reflections.append("PSNR test 3 blur candidates (gaussian/lens/glass sev 1-3). Gains marginal (+0.1~2.0dB) but may find match")
    elif not has_blur and gm < 0.85:
        risks.append("MISS_blur_MILD")
        reflections.append("PSNR test 3 blur candidates at sev 1-2. Mild blur detectable by PSNR")

    # Risk 3: Noise type may be wrong — use statistical checks, NOT PSNR
    if has_noise and s2["confidence"] in ["low", "medium"]:
        risks.append("NOISE_TYPE_UNCERTAIN")
        reflections.append("Re-run §B 6-step statistical check. Do NOT use PSNR for noise type selection (experiment: 0/1 PSNR success)")
    # Risk 3b: Speculative noise type (poisson/spatially_correlated) — needs extra verification
    if has_noise and s2.get("confidence") == "speculative":
        risks.append("NOISE_TYPE_SPECULATIVE")
        reflections.append("Subjective noise guess based on auxiliary signals (var_slope/spatial_corr). Verify via §B stats + PSNR test against GAUSSIAN_RGB alternative.")

    # Risk 4: Global FP check — PSNR verification
    if s4["verdict"] != "NO_GLOBAL" and s4["confidence"] in ["low", "medium"]:
        risks.append("FP_global_risk")
        reflections.append("PSNR verify: test if adding global degradation improves over non-global. Global is already 90% accurate")

    # Risk 5: Blur FP
    if has_blur and has_noise and s3["confidence"] in ["low", "medium"]:
        risks.append("FP_blur_risk")
        reflections.append("PSNR test: verify if blur candidate actually improves PSNR vs noise-only baseline")

    # Risk 6: Compression FP
    if has_comp and s1["confidence"] in ["low", "medium"]:
        risks.append("FP_comp_risk")
        reflections.append("PSNR test compression vs non-compression candidates — low confidence detection")

    # Risk 7: YCrCb noise → FP (exp28: 3例FP来自blur/JPEG跨通道相关)
    if has_noise and s2["verdict"] == "GAUSSIAN_YCrCb":
        if s2["signals"].get("gamma_suspect_from_noise"):
            risks.append("FP_YCrCb_GAMMA_SUSPECT")
            reflections.append("YCrCb noise detected but residual correlated with intensity → may be gamma. Test gamma candidates as alternative.")
        elif has_blur or has_comp:
            # blur/JPEG 产生的跨通道相关, 非真实 YCrCb 噪声
            risks.append("FP_YCrCb_BLUR_JPEG")
            reflections.append(f"YCrCb noise detected but blur={'YES' if has_blur else 'NO'} JPEG={'YES' if has_comp else 'NO'} → cross_ch_corr may come from blur/JPEG, not real YCrCb. Test GAUSSIAN_RGB/IMPULSE/POISSON as replacements. Agent should also test removing noise entirely.")
    # Risk 7b: brightness+contrast both detected → gamma suspect from Step4 side
    if s4["signals"].get("gamma_suspect"):
        risks.append("GAMMA_SUSPECT_FROM_GLOBAL")
        reflections.append("Both brightness and contrast signals triggered → suspect gamma (deterministic nonlinear transform). Test gamma candidates against brightness+contrast.")
        # exp28 0005: gamma_suspect + sigma>5 → 噪声被gamma信号掩盖, 强制测试
        if has_noise and float(s2.get('signals', {}).get('noise_prior_sigma', 0) or 0) > 5:
            risks.append("NOISE_MASKED_BY_GAMMA")
            reflections.append(f"sigma={s2['signals']['noise_prior_sigma']:.1f}>5 strongly suggests noise present. Gamma may not explain all variance. Force-test noise+gaussian_RGB and noise+gamma combinations.")

    # Risk 7c: sigma>5 but noise verdict absent or overridden → forced noise reminder
    sigma_val = float(s2['signals'].get('noise_prior_sigma', 0) or 0)
    if sigma_val > 5 and (not has_noise or s2['confidence'] in ['low', 'medium', 'speculative']):
        risks.append("NOISE_EVIDENCE_STRONG")
        reflections.append(f"noise_prior sigma={sigma_val:.1f}>5 strongly indicates noise. Even if type uncertain, Agent MUST test at least: noise_gaussian_RGB and noise_impulse candidates.")

    # # Risk 8: Oversharpen (暂时关闭)
    # if s4["signals"].get("oversharpen_detected_raw"):
    #     ...

    report["failure_risks"] = risks
    report["recommended_reflection"] = reflections

    # --- 🔴 交叉验证: JPEG 块结构 → spatial_corr 假阳性 (exp25 根因) ---
    # JPEG 8x8 块边界产生 0.5+ 的空间相关性, 与 SC 噪声不可区分。
    # 当 JPEG 被检测到且噪声被判为 SPATIALLY_CORRELATED 时, 大概率是 JPEG 假阳性。
    if (s1["verdict"] != "NO_COMPRESSION" and "MILD" not in s1.get("verdict", "") and
        s2["verdict"] == "SPATIALLY_CORRELATED" and s2.get("confidence") in ["speculative", "low"]):
        report["cross_validation"] = {
            "trigger": "JPEG + SPATIALLY_CORRELATED",
            "action": "spatial_corr 信号可能来自 JPEG 8x8 块结构而非真实 SC 噪声",
            "recommendation": "Agent 应优先测试 GAUSSIAN_RGB/IMPULSE/SPECKLE 替代 SC",
            "affected_signal": "spatial_corr",
            "confidence": "high"
        }
        risks.append("JPEG_SC_FALSE_POSITIVE")
        reflections.append("JPEG detected + SC noise (speculative) → high probability JPEG block artifact. Test GAUSSIAN_RGB/IMPULSE/SPECKLE as replacements.")

    # --- Architecture recommendation (来源: finetune_strategy.md exp9/10/12) ---
    has_contrast = any('contrast' in s.lower() for s in detections)
    has_motion = any('motion' in s.lower() for s in detections)
    has_brightness = any('brightness' in s.lower() for s in detections)
    is_multi_step = len(detections) >= 2
    is_uncertain = report["summary"].get("confidence", "medium") != "high" if False else is_multi_step

    arch = {"attention_type": "swin", "window_size": 8, "extra_flags": []}

    if has_contrast:
        # contrast → Swin mandatory (MDTA/OCAB crash -9.78 dB)
        arch["attention_type"] = "swin"
        arch["note"] = "Swin required: contrast detected (MDTA/OCAB would crash)"
    elif has_motion and not has_contrast:
        # motion blur → OCAB + ws=16 (+1.72 dB over Swin)
        arch["attention_type"] = "ocab"
        arch["window_size"] = 16
        arch["note"] = "OCAB+ws16: motion blur detected (+1.72 dB over Swin)"
    elif is_multi_step or is_uncertain:
        # Uncertain/multi-step → DualBranch (robustness #1)
        arch["extra_flags"] = ["AR_USE_DUAL_BRANCH=1"]
        arch["note"] = "DualBranch: multi-step/uncertain, best robustness (10.1 dB gap)"
    else:
        # Simple single-step → can try optimization
        arch["extra_flags"] = ["AR_USE_DUAL_BRANCH=1"]
        arch["note"] = "DualBranch: default for single-step"

    report["recommended_architecture"] = arch

    report["summary"] = OrderedDict([
        ("detected_degradations", detections),
        ("recommended_search_order", recommended_search),
        ("total_expected_steps", max(1, len(detections))),
        ("confidence", "high" if len(detections) <= 1 else "medium"),
        ("failure_risks", risks),
        ("reflection_strategies", reflections),
        ("recommended_architecture", arch),
    ])

    # 🔴 AGENT_REMINDER: 在做出最终预测决策前，Agent 必须重新阅读 SKILL.md。
    # exp28 教训: 代码正确识别了风险和建议 (should_still_test, reflection_strategies),
    # 但 Agent 未执行这些建议 → 盲识别失败。此提醒强制 Agent 在保存预测前重读协议。
    # PSNR test budget counter from thinking_process
    psnr_count = report.get("psnr_test_count", 0)
    budget_warning = ""
    if psnr_count > 20:
        budget_warning = f"⚠️ PSNR 测试已达 {psnr_count}/25，剩余预算不足 {25-psnr_count} 次。必须使用信号驱动二元对比协议，禁止枚举。"

    # ── 信号可靠性标注（附带校准数据来源）──
    report["signal_reliability"] = {
        "compression": {
            "block_boundary": {"accuracy": "0% FP (exp17: 35/35, exp18: 10/10)", "rule": ">1.1 → JPEG confirmed. DO NOT override with PSNR."},
            "unique_G": {"accuracy": "0% FP for compression presence (exp18: 32/32)", "rule": "<200 → compression EXISTS. But JPEG may not be the cause — test JPEG2000 if PSNR fails."},
        },
        "blur": {
            "hybrid_dt": {"accuracy": "83.4% (3840-case). gaussian=94% lens=79% motion=82% glass=79%. FP(glass)=140",
                          "rule": "Hybrid mode: glass_detected→depth=4, else→depth=3. Best of both worlds."},
            "depth3": {"accuracy": "72.9%, gaussian=94% (safe default), FP=898"},
            "depth4": {"accuracy": "75.1%, glass=79% (aggressive), FP(glass)=723"},
        },
        "noise": {
            "dt_depth4_blur_first": {"accuracy": "87.4% (1920-case, GT-stripped). gaussian=97% speckle=82% poisson=80% impulse=90%",
                          "rule": "Trust when PSNR confirms blur→noise order. Use peeled_noise_check.py §B features."},
            "dt_depth4_noise_first": {"accuracy": "54.7% ≈ random. Blur convolution Gaussianizes noise (CLT).",
                          "rule": "When noise→blur confirmed, mark noise type UNCERTAIN. Classification impossible."},
        },
    }

    # ── 反思触发条件 ──
    reflection_triggers = []
    s3_verdict = s3.get("verdict", "")
    s2_verdict = s2.get("verdict", "")
    if s3_verdict in ["GAUSSIAN_BLUR", "MOTION_BLUR"]:
        reflection_triggers.append({
            "trigger": "DT-PSNR 矛盾检查",
            "condition": "IF DT predicts gaussian/motion BUT PSNR test shows alternative blur type >5dB better",
            "action": "Trust PSNR. Replace blur type. Save DT prediction as alternative.",
            "note": "Known DT failure mode: gaussian↔lens MTF indistinguishable, motion↔gaussian in weak directional signal"
        })
    if s2_verdict != "NO_NOISE" and float(s2.get("signals", {}).get("noise_prior_sigma", 0) or 0) > 5:
        reflection_triggers.append({
            "trigger": "noise→blur 顺序检测",
            "condition": "PSNR test order: IF noise→blur wins → noise type unreliable (54.7% ≈ random)",
            "action": "Mark noise type UNCERTAIN. Use PSNR-best noise type but set verdict=UNCERTAIN.",
            "note": "Blur convolution Gaussianizes noise. Noise type classification only valid in blur→noise order."
        })
    if s2_verdict != "NO_NOISE":
        reflection_triggers.append({
            "trigger": "peeled §B extreme_pct 假阳性",
            "condition": "extreme_pct>50% AND noise_prior sigma<5 → likely JPEG/blur artifact, not real impulse",
            "action": "Ignore §B IMPULSE diagnosis. Trust noise_prior type. Test gaussian_RGB and noise_prior type only.",
            "note": "Known FP: JPEG/blur residuals inflate extreme_pct to 50%+ even without impulse noise."
        })
    report["reflection_triggers"] = reflection_triggers

    # ── 生成 Agent 可直接执行的候选管线 ──
    candidates = []
    sigma = s2["signals"].get("noise_prior_sigma", 0)
    fft_a = s3["signals"].get("residual_anisotropy", 0)
    pca = s3["signals"].get("blur_pca_ratio", 0)
    gm = s3["signals"].get("gm_ratio", 0)
    slope = s3["signals"].get("spectral_slope_ratio", 1.0)
    use_signal_verify = sigma > 2
    verify_tool = "verify_signals.py" if use_signal_verify else "compare_degradation.py"

    # Compression candidates
    if s1["verdict"] != "NO_COMPRESSION" or s1.get("should_still_test"):
        bb = s1["signals"].get("block_boundary", 0)
        sev_est = 1 if bb < 1.2 else (2 if bb < 1.5 else (3 if bb < 2.0 else 4))
        candidates.append({
            "step": "compression", "signal": f"block_boundary={bb:.3f}",
            "pipeline": f"compression_jpeg:{sev_est}",
            "verify_with": verify_tool
        })

    # Blur candidates
    btype = s3.get("blur_subtype", "blur_gaussian")
    sev_src = s3["signals"].get("severity_source", "none")
    sev_est = s3.get("estimated_severity", 2) or 2
    if s3["verdict"] != "NO_BLUR" and s3["verdict"] != "BLUR_UNLIKELY":
        if pca > 2.5:
            candidates.append({
                "step": "blur", "signal": f"PCA={pca:.1f}>2.5 (0% FP)",
                "pipeline": f"blur_motion:{sev_est}",
                "alternatives": [f"blur_motion:{sev_est-1}", f"blur_motion:{sev_est+1}"],
                "verify_with": verify_tool
            })
        elif fft_a > 2.2:
            candidates.append({
                "step": "blur", "signal": f"FFT={fft_a:.2f}>2.2 (fallback)",
                "pipeline": f"blur_motion:{sev_est}",
                "verify_with": verify_tool
            })
        elif s3["signals"].get("glass_score", 0) > 0.25 and sigma < 3:
            candidates.append({
                "step": "blur", "signal": f"glass_score={s3['signals']['glass_score']:.2f}>0.25",
                "pipeline": f"blur_glass:{sev_est}",
                "alternatives": [f"blur_gaussian:{sev_est}"],
                "verify_with": verify_tool
            })
        else:
            candidates.append({
                "step": "blur", "signal": f"DT subtype={btype}, gm={gm:.2f}, slope={slope:.3f}",
                "pipeline": f"blur_gaussian:{sev_est}",
                "alternatives": [f"blur_lens:{sev_est}"],
                "verify_with": verify_tool,
                "note": "gaussian↔lens MTF不可区分, PSNR测试两者"
            })
    elif s3["verdict"] == "BLUR_UNLIKELY" and s3.get("should_still_test"):
        candidates.append({
            "step": "blur", "signal": "BLUR_UNLIKELY but should_still_test",
            "pipeline": f"blur_gaussian:1",
            "optional": True,
            "verify_with": verify_tool
        })

    # Noise candidates
    if s2["verdict"] != "NO_NOISE":
        ntype = s2["verdict"]
        nsev = s2.get("estimated_severity", 2)
        noise_fn_map = {
            "GAUSSIAN_RGB": "noise_gaussian_RGB", "GAUSSIAN_YCrCb": "noise_gaussian_YCrCb",
            "IMPULSE": "noise_impulse", "SPECKLE": "noise_speckle",
            "POISSON": "noise_poisson", "SPATIALLY_CORRELATED": "noise_spatially_correlated",
        }
        nfn = noise_fn_map.get(ntype, "noise_gaussian_RGB")
        candidates.append({
            "step": "noise", "signal": f"sigma={sigma:.1f}, type={ntype}",
            "pipeline": f"{nfn}:{nsev}",
            "alternatives": [f"noise_gaussian_RGB:{nsev}"],
            "verify_with": verify_tool,
            "note": "PSNR不可用于噪声验证 — 使用信号验证" if use_signal_verify else ""
        })

    # Global candidates
    global_parts = []
    if s4["verdict"] != "NO_GLOBAL":
        for det in s4["verdict"].split("; "):
            if "BRIGHTNESS" in det:
                direction = "brighten" if "brighten" in det.lower() else "darken"
                sev_str = det.split("sev")[-1] if "sev" in det else "1"
                try: gsev = int(sev_str)
                except: gsev = 1
                fn = f"brightness_{direction}_shfit_HSV"
                global_parts.append(f"{fn}:{gsev}")
            elif "CONTRAST" in det:
                direction = "strengthen" if "strengthen" in det.lower() else "weaken"
                sev_str = det.split("sev")[-1] if "sev" in det else "1"
                try: gsev = int(sev_str)
                except: gsev = 1
                global_parts.append(f"contrast_{direction}_scale:{gsev}")
            elif "SATURATION" in det:
                direction = "strengthen" if "strengthen" in det.lower() else "weaken"
                sev_str = det.split("sev")[-1] if "sev" in det else "1"
                try: gsev = int(sev_str)
                except: gsev = 1
                global_parts.append(f"saturate_{direction}_HSV:{gsev}")
        if global_parts:
            candidates.append({
                "step": "global", "signal": f"variance_ratio={s4['signals'].get('variance_ratio',0):.3f}, mean_shift={s4['signals'].get('brightness_mean_shift',0):.1f}",
                "pipeline": ",".join(global_parts),
                "verify_with": verify_tool,
                "fp_checks": s4.get("false_positive_checks", {})
            })

    report["agent_instructions"] = {
        "verification_mode": "SIGNAL" if use_signal_verify else "PSNR",
        "reason": f"sigma={sigma:.1f} {'≥' if use_signal_verify else '<'} 2 → {'PSNR不可靠, 必须用信号验证' if use_signal_verify else 'PSNR可用'}",
        "verdict_allowed": "UNCERTAIN or POOR only (sigma>2 → NO LIKELY/GOOD)" if use_signal_verify else "GOOD/LIKELY/UNCERTAIN/POOR",
        "candidates": candidates,
        "execution_order": [
            "1. 按顺序测试每个候选: apply_multi.py → verify_signals.py",
            "2. 解读 verify_signals.py 输出:",
            "   MATCH   → 保存, 进入下一步",
            "   PARTIAL → 【severity 扫描 ±1】: 同类型, sev±1 各测1次 → 选最佳",
            "   WEAK    → 【severity 扫描 ±2】: 同类型, sev±2 各测1次 → 若改善→保留; 否则换子类型",
            "   MISMATCH→ 【severity 扫描 ±2】: 先扫 severity, 仍无改善→换子类型 (≤2种替代)",
            "3. 🔴 severity 扫描是强制步骤, 不可跳过。禁止在 WEAK 时直接换类型而不扫 severity。",
            "4. 每类退化最多: 1个主候选 + 4个severity扫描 + 2个替代类型 = 7次测试",
            "5. 所有步骤完成 → 合并 pipeline → 最终 verify_signals.py 确认",
            "6. 保存 predicted_params.json (verdict 遵守允许值)",
        ],
        "severity_scan_rules": {
            "when": "verify_signals 输出 PARTIAL / WEAK / MISMATCH",
            "how": "保持函数名不变, severity 从 est-2 扫到 est+2 (限制 1-5 范围内)",
            "stop": "任一 severity 达到 MATCH → 立即停止扫描",
            "fallback": "全部 severity 未达 MATCH → 选 score 最高的, 然后尝试换子类型",
        },
        "forbidden": [
            "❌ sigma>2 时使用 compare_degradation.py PSNR 判断候选好坏",
            "❌ sigma>2 时设置 verdict=LIKELY 或 GOOD",
            "❌ 跳过 verify_signals.py 直接用 PSNR 排名",
            "❌ WEAK/MISMATCH 时跳过 severity 扫描直接换类型",
            "❌ 测试超过 2 个 alternatives 不终止",
        ] if use_signal_verify else [
            "❌ PSNR<30 时声称 LIKELY",
            "❌ 测试超过 3 个 severity 候选",
        ]
    }

    return report


def main():
    parser = argparse.ArgumentParser(description="SKILL.md v13 forced analysis pipeline")
    parser.add_argument("--target", required=True, help="Path to degraded image")
    parser.add_argument("--clean", required=True, help="Path to clean reference image")
    parser.add_argument("--output", help="Output JSON file (default: stdout)")
    parser.add_argument("--json", action="store_true", help="Force JSON output to stdout")
    args = parser.parse_args()

    report = run_full_analysis(args.target, args.clean)
    report = to_native(report)

    json_output = json.dumps(report, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    # ── 文字摘要 (Agent 原生可读, 不需要解析JSON) ──
    s1 = report["step1_compression"]; s2 = report["step2_noise"]
    s3 = report["step3_blur"]; s4 = report["step4_global"]
    sr = report["signal_reliability"]

    fft_a = s3["signals"].get("residual_anisotropy", 0)
    pca = s3["signals"].get("blur_pca_ratio", 0)
    gm = s3["signals"].get("gm_ratio", 0)
    slope = s3["signals"].get("spectral_slope_ratio", 0)
    sigma = s2["signals"].get("noise_prior_sigma", 0)

    blur_decision = "NO_BLUR"
    if s3["verdict"] != "NO_BLUR":
        if pca > 2.5:
            blur_decision = f"MOTION (PCA={pca:.1f}>2.5, 41%recall 0%FP, primary signal. FFT={fft_a:.2f})"
        elif fft_a > 2.2:
            blur_decision = f"MOTION (FFT={fft_a:.2f}>2.2, 5%recall 0.2%FP, fallback. PCA={pca:.1f} failed—kernel corrupted by noise)"
        elif s3["signals"].get("glass_score",0) > 0.25 and sigma < 3:
            blur_decision = f"GLASS (score={s3['signals']['glass_score']:.2f}>0.25, sigma={sigma:.1f}<3)"
        else:
            blur_decision = f"GAUSSIAN (lens alternative, MTF不可区分. PCA={pca:.1f})"

    comp_decision = s1["verdict"]
    if s1.get("should_still_test"):
        comp_decision += " (should_still_test)"

    # Build candidates text from report (not local scope)
    ai = report.get("agent_instructions", {})
    candidates = ai.get("candidates", [])
    verify_mode = ai.get("verification_mode", "PSNR")
    verify_tool = "verify_signals.py" if verify_mode == "SIGNAL" else "compare_degradation.py"
    verdict_allowed = ai.get("verdict_allowed", "UNCERTAIN")
    cand_lines = []
    for c in candidates:
        line = f'  [{c["step"]}] {c["pipeline"]}  <- {c.get("signal","")}'
        cand_lines.append(line)
        if c.get("alternatives"):
            cand_lines.append(f'       alternatives: {c["alternatives"]}')
        if c.get("note"):
            cand_lines.append(f'       note: {c["note"]}')
    cand_text = chr(10).join(cand_lines)
    constraints_text = chr(10).join(f'  {f}' for f in ai.get('forbidden', []))

    reason_text = 'PSNR不可靠, 用 verify_signals.py 验证每个候选' if verify_mode == 'SIGNAL' else 'PSNR可用, 用 compare_degradation.py 验证'

    print(f"""
{'='*60}
🔴 SIGNAL-DRIVEN DIAGNOSIS
{'='*60}
Compression: {comp_decision}
  block_boundary={s1['signals'].get('block_boundary',0):.3f} (>1.05=JPEG, 0%FP, 14400-case校准)

Blur: {s3['verdict']} -> {blur_decision}
  gm_ratio={gm:.3f} slope={slope:.3f} FFT_anisotropy={fft_a:.2f} pca={pca:.1f}

Noise: {s2['verdict']}  sigma={sigma:.1f}

Global: {s4['verdict']}
  variance_ratio={s4['signals'].get('variance_ratio',0):.3f} mean_shift={s4['signals'].get('brightness_mean_shift',0):.1f}

🔴 MODE: {verify_mode}  |  Verdict must be: {verdict_allowed}
   Reason: sigma={sigma:.1f} -> {reason_text}
{'='*60}
🔴 CANDIDATES (test in order, verify with {verify_tool}):
{cand_text}
{'='*60}
🔴 CONSTRAINTS:
{constraints_text}
{'='*60}
""")

    if args.output:
        with open(args.output, 'w') as f:
            f.write(json_output)
        print(f"Report saved to {args.output}", file=sys.stderr)
    if args.json:
        print(json_output)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(json_output)
        print(f"Report saved to {args.output}")
    elif args.json:
        print(json_output)
    else:
        print(json_output)


if __name__ == "__main__":
    main()
