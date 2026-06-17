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
    bb_detected = block_boundary > 1.05   # 1.5% deviation threshold (no natural image triggers)
    dct_detected = dct_zeros > 0.77       # 87% recall, 12.8% FP
    dct_degraded = dct_zeros > 0.15       # DCT fails in composite (4.3σ degradation)
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
    """Noise detection via wavelet + §B 6-step check on residual."""
    result = {"verdict": "NO_NOISE", "confidence": "high", "signals": {}}
    residual = compute_residual(target, clean)

    # Wavelet noise estimation
    sigma = estimate_noise_sigma(residual)
    avg_sigma = np.mean(sigma)
    result["signals"]["noise_prior_sigma"] = round(avg_sigma, 2)
    result["signals"]["sigma_per_channel"] = [round(s, 2) for s in sigma]

    if avg_sigma < 1.9:   # calibrated: 83% recall, 7.3% FP
        result["signals"]["gm_ratio_usable"] = True
        return result

    result["signals"]["gm_ratio_usable"] = False

    # §B Step 1: Impulse (calibrated threshold: >2% extreme pixels)
    extreme_pct = ((np.abs(residual) > 3 * np.std(residual)).sum(axis=2) > 0).mean()
    result["signals"]["impulse_extreme_pct"] = round(extreme_pct * 100, 2)

    # §B Step 2: Speckle (var/mean slope)
    flat_mask = np.std(clean, axis=2) < 30
    if flat_mask.sum() > 100:
        flat_intensities = clean[flat_mask].mean(axis=1)
        flat_variances = residual[flat_mask].var(axis=1)
        if len(flat_intensities) > 10:
            bins = np.digitize(flat_intensities, np.percentile(flat_intensities, [20,40,60,80]))
            bin_vars = [flat_variances[bins==i].mean() for i in range(6) if (bins==i).sum()>5]
            bin_means = [flat_intensities[bins==i].mean() for i in range(6) if (bins==i).sum()>5]
            if len(bin_vars) >= 3:
                coeffs = np.polyfit(bin_means, bin_vars, 1)
                result["signals"]["speckle_vm_slope"] = round(coeffs[0], 5)
                result["signals"]["poisson_var_slope"] = round(coeffs[0], 3)

    # §B Step 4: Spatial correlation
    if residual.shape[0] > 4 and residual.shape[1] > 4:
        spatial_corrs = []
        for c in range(3):
            ch_r = residual[:,:,c]
            # Adjacent pixel correlation
            c1 = np.corrcoef(ch_r[:-1,:].flatten(), ch_r[1:,:].flatten())[0,1]
            c2 = np.corrcoef(ch_r[:,:-1].flatten(), ch_r[:,1:].flatten())[0,1]
            spatial_corrs.extend([c1, c2])
        result["signals"]["spatial_corr"] = round(np.mean(spatial_corrs), 4)

    # §B Step 5: RGB ratio (YCrCb vs RGB)
    rgb_stds = [residual[:,:,c].std() for c in range(3)]
    if min(rgb_stds) > 0:
        result["signals"]["rgb_ratio"] = round(max(rgb_stds) / min(rgb_stds), 2)

    # Classify noise type
    if extreme_pct > 0.02:  # 2% threshold — noise alone won't trigger this
        result["verdict"] = "IMPULSE"
    elif result["signals"].get("speckle_vm_slope", 0) > 0.01:
        result["verdict"] = "SPECKLE"
    elif result["signals"].get("poisson_var_slope", 0) > 0.9 and result["signals"].get("speckle_vm_slope", 0) < 0.005:
        result["verdict"] = "POISSON"
    elif result["signals"].get("spatial_corr", 0) > 0.15:
        result["verdict"] = "SPATIALLY_CORRELATED"
    elif result["signals"].get("rgb_ratio", 1) > 1.4:
        result["verdict"] = "GAUSSIAN_YCrCb"
    else:
        result["verdict"] = "GAUSSIAN_RGB"

    # Severity estimation
    sev_map = {2:1, 5:2, 10:3, 20:4, 35:5}
    sev = 1
    for threshold, s in sorted(sev_map.items()):
        if avg_sigma > threshold: sev = s
    result["estimated_severity"] = sev

    # Speckle-specific severity signal
    if result["verdict"] == "SPECKLE":
        sp_sigma, sp_vm, sp_sev = compute_speckle_severity(clean, residual)
        result["signals"]["speckle_sigma"] = sp_sigma
        result["signals"]["speckle_vm_slope"] = sp_vm
        result["signals"]["speckle_est_severity"] = sp_sev

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

def step3_blur(target, clean, noise_info):
    """Blur detection using gm_ratio + MTF signals."""
    result = {"verdict": "NO_BLUR", "confidence": "medium", "signals": {}}
    gm_ratio_usable = noise_info.get("signals", {}).get("gm_ratio_usable", True)

    target_gm = compute_gm_ratio(target)
    clean_gm = compute_gm_ratio(clean)
    gm_ratio = target_gm / max(clean_gm, 1e-6)
    result["signals"]["gm_ratio"] = round(gm_ratio, 4)
    result["signals"]["gm_ratio_usable"] = gm_ratio_usable

    # Calibrated thresholds:
    #   pure: gm_ratio < 0.62 (95% recall, 9.3% FP)
    #   with noise: gm_ratio < 0.72 (noise increases gradient, signal shifts ~5%)
    has_noise = noise_info.get("verdict", "NO_NOISE") != "NO_NOISE"
    blur_threshold = 0.72 if has_noise else 0.62

    if gm_ratio > blur_threshold:
        result["verdict"] = "BLUR_UNLIKELY"
        result["confidence"] = "medium" if has_noise else "high"
        result["should_still_test"] = True  # Agent should still PSNR-test blur candidates
        result["note"] = f"gm_ratio={gm_ratio:.3f} > {blur_threshold}" + (" (noise present, signal may be masked)" if has_noise else "")
        result["suggested_action"] = "Test 2-3 blur candidates via PSNR despite gm_ratio signal — noise/blur coupling can mask blur"
        return result

    # Blur detected — use MTF for sub-type
    target_mtf = compute_radial_mtf(target)
    clean_mtf = compute_radial_mtf(clean)
    kernel_mtf = target_mtf / (clean_mtf + 1e-6)

    # --- Lens: MTF zeros (Bessel function) ---
    mtf_smoothed = np.convolve(kernel_mtf, np.ones(5)/5, mode='same')
    local_mins = []
    for i in range(1, len(mtf_smoothed)-1):
        if mtf_smoothed[i] < mtf_smoothed[i-1] and mtf_smoothed[i] < mtf_smoothed[i+1]:
            if mtf_smoothed[i] < 0.7 * max(mtf_smoothed[i-1], mtf_smoothed[i+1]):
                local_mins.append(i)
    result["signals"]["mtf_local_minima"] = len(local_mins)
    result["signals"]["mtf_has_zeros"] = len(local_mins) > 0

    # --- Zoom: spatial variation ---
    h, w = target.shape[:2]
    cy, cx = h//2, w//2
    center_crop = target[cy-cy//4:cy+cy//4, cx-cx//4:cx+cx//4]
    corner_crops = [
        target[:h//4, :w//4],
        target[:h//4, -w//4:],
        target[-h//4:, :w//4],
        target[-h//4:, -w//4:]
    ]
    center_gm = compute_gm_ratio(center_crop)
    corner_gm = np.mean([compute_gm_ratio(c) for c in corner_crops])
    zoom_ratio = corner_gm / max(center_gm, 1e-6)
    result["signals"]["zoom_center_corner_ratio"] = round(zoom_ratio, 4)

    # --- Glass: MTF roughness ---
    mtf_diff = np.abs(np.diff(kernel_mtf))
    mtf_roughness = mtf_diff.std() / (mtf_diff.mean() + 1e-6)
    result["signals"]["mtf_roughness"] = round(mtf_roughness, 4)

    # --- Motion: angular FFT ---
    fft_2d = np.abs(np.fft.fftshift(np.fft.fft2(target.mean(axis=2) - target.mean(axis=2).mean())))
    angles = np.linspace(0, np.pi, 36)
    angular_energy = []
    for a in angles:
        x = np.cos(a); y = np.sin(a)
        r = np.arange(5, min(h,w)//4)
        energy = 0
        for rad in r[::5]:
            px = int(cx + rad*x); py = int(cy + rad*y)
            if 0 <= px < w and 0 <= py < h:
                energy += fft_2d[py, px]
        angular_energy.append(energy)
    angular_energy = np.array(angular_energy)
    if angular_energy.mean() > 0:
        anisotropy = angular_energy.max() / angular_energy.mean()
    else:
        anisotropy = 1.0
    result["signals"]["angular_anisotropy"] = round(anisotropy, 4)

    # --- Oversharpen vs Impulse discrimination ---
    # Only usable when no blur is detected (blur edge artifacts mimic oversharpen clustering)
    extreme_pct, extreme_cluster = compute_extreme_spatial_clustering(target, clean)
    result["signals"]["extreme_spatial_clustering"] = extreme_cluster
    if not has_noise:
        # Pure case: clustering > 0.4 → oversharpen, < 0.15 → impulse-like random
        if extreme_cluster > 0.4:
            result["signals"]["oversharpen_vs_impulse"] = "OVERSHARPEN_LIKELY"
            result["oversharpen_hint"] = True
        elif extreme_cluster < 0.15 and extreme_pct > 1.0:
            result["signals"]["oversharpen_vs_impulse"] = "IMPULSE_LIKELY"
            result["impulse_hint"] = True

    # --- Classification ---
    if anisotropy > 4.0:
        result["verdict"] = "MOTION_BLUR"
        result["confidence"] = "high"
    elif result["signals"]["mtf_has_zeros"]:
        result["verdict"] = "LENS_BLUR"
        result["confidence"] = "medium"
    elif zoom_ratio > 1.3:
        result["verdict"] = "ZOOM_BLUR"
        result["confidence"] = "medium"
    elif mtf_roughness > 0.5:
        result["verdict"] = "GLASS_BLUR"
        result["confidence"] = "low"
    elif gm_ratio > 0.85:
        result["verdict"] = "JITTER_BLUR"
        result["confidence"] = "low"
    else:
        result["verdict"] = "GAUSSIAN_BLUR"
        result["confidence"] = "medium" if gm_ratio_usable else "low"

    # Severity estimation from gm_ratio (only if usable)
    if gm_ratio_usable:
        # gm_ratio → severity mapping (calibrated)
        if gm_ratio < 0.35: sev = 5
        elif gm_ratio < 0.50: sev = 4
        elif gm_ratio < 0.65: sev = 3
        elif gm_ratio < 0.78: sev = 2
        else: sev = 1
        result["estimated_severity"] = sev
    else:
        result["estimated_severity"] = None

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

    # --- Contrast ---
    t_spread = sum(np.percentile(target[:,:,c], 95) - np.percentile(target[:,:,c], 5) for c in range(3))
    c_spread = sum(np.percentile(clean[:,:,c], 95) - np.percentile(clean[:,:,c], 5) for c in range(3))
    spread_ratio = t_spread / max(c_spread, 1)
    contrast_detected = spread_ratio < 0.80 or spread_ratio > 1.20  # wide threshold: low F1 signal
    result["signals"]["contrast_spread_ratio"] = round(spread_ratio, 4)
    result["signals"]["contrast_detected_raw"] = contrast_detected

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

    # --- Saturation ---
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
        detections.append(f"noise: {report['step2_noise']['verdict']} (sev~{report['step2_noise']['estimated_severity']})")
        recommended_search.append(f"noise candidates: {report['step2_noise']['verdict']}+related types")
    else:
        recommended_search.append("noise: none detected")

    if report["step3_blur"]["verdict"] != "NO_BLUR":
        detections.append(f"blur: {report['step3_blur']['verdict']}")
        gm_usable = report["step3_blur"]["signals"]["gm_ratio_usable"]
        recommended_search.append(f"blur candidates: {report['step3_blur']['verdict']}+related (gm_ratio_usable={gm_usable})")
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

    # Risk 1a: Blur missed because noise elevates gradient (gm_ratio still below 1)
    if has_noise and not has_blur and gm < 1.0:
        risks.append("MISS_blur_BY_NOISE")
        reflections.append("Train(noise_only) → model denoise → re-run Step3 blur detection on residual")
    # Risk 1b: Noise gradient dominates — blur completely invisible (gm > 1)
    if has_noise and not has_blur and gm >= 1.0:
        risks.append("MISS_blur_NOISE_DOMINATES")
        reflections.append("Train(noise_only) → model denoise → re-run Step3 (only way to detect blur under strong noise)")

    # Risk 2: Mild blur undetected (gm_ratio 0.62-0.85, no noise)
    if not has_noise and not has_blur and 0.60 < gm < 0.85:
        risks.append("MISS_blur_MILD")
        reflections.append("Train(blur_candidate) → val_psnr_db confirmation → if PSNR improves, blur exists")

    # Risk 3: Compression missed because noise fills DCT/blurs blocks
    if has_noise and not has_comp and bb < 1.05:
        risks.append("MISS_comp_BY_NOISE")
        reflections.append("Train(compression_candidate) → check 8×8 block pattern in model residual")

    # Risk 4: Compression is false positive (low confidence)
    if has_comp and s1["confidence"] in ["low", "medium"]:
        risks.append("FP_comp_risk")
        reflections.append("PSNR verify compression candidates — low confidence, may be FP")

    # Risk 5: Blur is false positive (noise+texture misidentified)
    if has_blur and has_noise and s3["confidence"] in ["low", "medium"]:
        risks.append("FP_blur_risk")
        reflections.append(f"Verify gm_ratio={gm:.3f} vs noise+blur threshold — may be noise artifact")

    # Risk 6: Global detection has false positive risk
    if s4["confidence"] in ["low", "medium"] and s4["verdict"] != "NO_GLOBAL":
        risks.append("FP_global_risk")
        reflections.append("Cross-validate global signals with PSNR testing — low F1 signal")

    report["failure_risks"] = risks
    report["recommended_reflection"] = reflections

    report["summary"] = OrderedDict([
        ("detected_degradations", detections),
        ("recommended_search_order", recommended_search),
        ("total_expected_steps", max(1, len(detections))),
        ("confidence", "high" if len(detections) <= 1 else "medium"),
        ("failure_risks", risks),
        ("reflection_strategies", reflections),
    ])

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
