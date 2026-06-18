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

    # --- Always compute auxiliary signals (may trigger noise entry even when sigma low) ---

    # §B Step 1: Impulse (calibrated threshold: >2% extreme pixels)
    extreme_pct = ((np.abs(residual) > 3 * np.std(residual)).sum(axis=2) > 0).mean()
    result["signals"]["impulse_extreme_pct"] = round(extreme_pct * 100, 2)

    # §B Step 2/3: Speckle (vm_slope) + Poisson (var_slope)
    # var_slope: slope of variance vs intensity (Poisson: var ∝ intensity)
    # vm_slope: slope of variance/mean ratio vs intensity (Speckle: var/mean ∝ intensity)
    flat_mask = np.std(clean, axis=2) < 30
    if flat_mask.sum() > 100:
        flat_intensities = clean[flat_mask].mean(axis=1)
        flat_variances = residual[flat_mask].var(axis=1)
        if len(flat_intensities) > 10:
            bins = np.digitize(flat_intensities, np.percentile(flat_intensities, [20,40,60,80]))
            bin_vars = [flat_variances[bins==i].mean() for i in range(6) if (bins==i).sum()>5]
            bin_means = [flat_intensities[bins==i].mean() for i in range(6) if (bins==i).sum()>5]
            if len(bin_vars) >= 3:
                # Poisson var_slope: variance vs intensity
                var_coeffs = np.polyfit(bin_means, bin_vars, 1)
                result["signals"]["poisson_var_slope"] = round(var_coeffs[0], 3)
                # Speckle vm_slope: variance/mean ratio vs intensity
                bin_vm_ratios = [v / max(m, 1) for v, m in zip(bin_vars, bin_means)]
                vm_coeffs = np.polyfit(bin_means, bin_vm_ratios, 1)
                result["signals"]["speckle_vm_slope"] = round(vm_coeffs[0], 6)

    # §B Step 2.5: Poisson dark-vs-bright variance ratio (Poisson 独特签名)
    # Poisson 加噪: Poisson(img*c)/c → 暗区方差≈0, 亮区方差=img/c
    # 暗区 (intensity<50) vs 亮区 (intensity>180) 方差比:
    #   Gaussian ~1.0-1.2, Poisson ~4-6, Speckle ~22-33
    if residual.shape[0] > 4 and residual.shape[1] > 4:
        gray_clean = clean.mean(axis=2)
        flat_gray = np.std(clean, axis=2) < 15  # 灰色平坦区域
        dark_mask = (gray_clean < 50) & flat_gray
        bright_mask = (gray_clean > 180) & flat_gray
        if dark_mask.sum() > 100 and bright_mask.sum() > 100:
            dark_var = residual[dark_mask].var()
            bright_var = residual[bright_mask].var()
            db_ratio = bright_var / max(dark_var, 0.01)
            result["signals"]["poisson_db_ratio"] = round(db_ratio, 2)

    # §B Step 4: Spatial correlation (on HH wavelet subband → noise-dominated)
    # Full-residual spatial_corr conflates blur (spatially correlated) with noise.
    # HH subband contains finest details → isolates noise spatial structure.
    if residual.shape[0] > 4 and residual.shape[1] > 4:
        spatial_corrs = []
        hh_spatial_corrs = []
        for c in range(3):
            ch_r = residual[:,:,c]
            # Full residual spatial correlation (legacy)
            c1 = np.corrcoef(ch_r[:-1,:].flatten(), ch_r[1:,:].flatten())[0,1]
            c2 = np.corrcoef(ch_r[:,:-1].flatten(), ch_r[:,1:].flatten())[0,1]
            spatial_corrs.extend([c1, c2])
            # HH subband spatial correlation (noise-isolated)
            _, _, _, HH = haar_dwt2_1d(ch_r)
            if HH.shape[0] > 2 and HH.shape[1] > 2:
                h1 = np.corrcoef(HH[:-1,:].flatten(), HH[1:,:].flatten())[0,1]
                h2 = np.corrcoef(HH[:,:-1].flatten(), HH[:,1:].flatten())[0,1]
                hh_spatial_corrs.extend([h1, h2])
        result["signals"]["spatial_corr"] = round(np.mean(spatial_corrs), 4)
        result["signals"]["spatial_corr_hh"] = round(np.mean(hh_spatial_corrs), 4) if hh_spatial_corrs else 0

    # §B Step 5: RGB ratio (YCrCb vs RGB)
    rgb_stds = [residual[:,:,c].std() for c in range(3)]
    if min(rgb_stds) > 0:
        result["signals"]["rgb_ratio"] = round(max(rgb_stds) / min(rgb_stds), 2)

    # --- Auto-detect same-image vs cross-image mode ---
    # Same-image: clean is undegraded version of target → high correlation
    # Cross-image: clean is a different image → low correlation, residual dominated by content
    is_same_image = False
    try:
        # Coarse-scale correlation to reduce noise impact
        from PIL.Image import Resampling
        t_small = np.array(Image.fromarray(target.astype(np.uint8)).resize((64, 64), Resampling.LANCZOS))
        c_small = np.array(Image.fromarray(clean.astype(np.uint8)).resize((64, 64), Resampling.LANCZOS))
        corr = np.corrcoef(t_small.ravel(), c_small.ravel())[0, 1]
        is_same_image = corr > 0.8  # Same image: corr > 0.85 even with strong degradation
    except Exception:
        pass  # Conservative: assume cross-image if detection fails

    # --- Noise detection gate ---
    # sigma < 1.9 AND no auxiliary signals → exit (low FP: 83% recall, 7.3% FP)
    # Exception: spatial_corr/impulse can indicate noise even when wavelet sigma is low,
    # but ONLY in same-image mode (cross-image residual dominated by content).
    # Spatial correlation for noise typing:
    # - Spatially_correlated noise: sc ~0.5-0.6 (pure), drops in composite but still >0.15
    # - Blur also inflates sc (0.2-0.4) → false SC detections when blur+noise present
    # - Sigma-dependent threshold: strong noise + high sc → likely blur+noise artifact
    #   Weak noise + high sc → likely real SC noise
    sc = result["signals"].get("spatial_corr", 0)
    ep = result["signals"].get("impulse_extreme_pct", 0)
    # Cross-image: residual dominated by content → spatial_corr unreliable
    if is_same_image:
        if avg_sigma > 15:      sc_threshold = 0.40  # Strong noise: blur+noise artifact risk
        elif avg_sigma > 5:     sc_threshold = 0.25
        else:                   sc_threshold = 0.15  # Low sigma: classic SC signature
        ep_threshold = 2
    else:
        sc_threshold = 0.50
        ep_threshold = 8
    if avg_sigma < 1.9 and (sc < sc_threshold) and (ep < ep_threshold):
        result["signals"]["gm_ratio_usable"] = True
        result["signals"]["_same_image"] = is_same_image
        return result

    result["signals"]["gm_ratio_usable"] = avg_sigma < 1.9
    result["signals"]["_same_image"] = is_same_image
    # Low-sigma subjective detection (wavelet-blind noise types)
    if avg_sigma <= 5:
        _sc_t = 0.30 if is_same_image else 0.50
        _ep_t = 5 if is_same_image else 15
        if sc > _sc_t and is_same_image:
            result["verdict"] = "SPATIALLY_CORRELATED"
            result["confidence"] = "speculative"
            result["subjective_guess"] = f"sigma={avg_sigma:.1f}≤5 but spatial_corr={sc:.4f}>{_sc_t} → spatially structured noise (wavelet-blind)"
            result["estimated_severity"] = 3 if sc > 0.45 else (2 if sc > 0.3 else 1)
            result["_for_step4"] = {"sigma": round(avg_sigma, 2), "can_fake_contrast": False,
                                     "can_fake_saturation": False, "can_fake_brightness": False}
            return result
        if ep > _ep_t:
            result["verdict"] = "IMPULSE"
            result["confidence"] = "speculative"
            result["subjective_guess"] = f"sigma={avg_sigma:.1f}≤5 but extreme_pct={ep:.1f}%>{_ep_t}% → sparse impulse noise (wavelet-blind, mode={'same' if is_same_image else 'cross'})"
            result["estimated_severity"] = 2 if ep > 10 else 1
            result["_for_step4"] = {"sigma": round(avg_sigma, 2), "can_fake_contrast": False,
                                     "can_fake_saturation": False, "can_fake_brightness": False}
            return result

    # Histogram-based noise classification (skew+kurtosis, 56-case validated)
    # gaussian_RGB=100%, impulse=72%, YCrCb=66% in pure+composite
    # Only classify when sigma > 5 (below that, not real noise — 41% FP otherwise)
    if avg_sigma > 5:
        flat_mask = np.std(clean, axis=2) < 15
        flat_r = residual[flat_mask] if flat_mask.mean() > 0.05 else residual.reshape(-1, 3)
        skew_vals, kurt_vals = [], []
        for c in range(3):
            ch = flat_r[:,c]
            if ch.std() > 1:
                try:
                    from scipy import stats
                    skew_vals.append(stats.skew(ch))
                    kurt_vals.append(stats.kurtosis(ch))
                except ImportError: pass
        avg_skew = np.mean(skew_vals) if skew_vals else 0
        avg_kurt = np.mean(kurt_vals) if kurt_vals else 0
        result["signals"]["noise_hist_skew"] = round(float(avg_skew), 3)
        result["signals"]["noise_hist_kurt"] = round(float(avg_kurt), 2)

        # Classify: Tier 1 histogram (56-case validated) + Tier 1.5 db_ratio (物理信号优先)
        # ⚠️ 复合退化中直方图(skew/kurt/rgb_ratio)可能被 blur/saturate/JPEG 扭曲。
        # db_ratio (暗区vs亮区绝对方差比) 是直接物理测量, 复合退化中更可靠。
        # db_ratio: Gaussian~1.1, Poisson~4-6, Speckle~22-33
        subjective_reason = None
        db = result["signals"].get("poisson_db_ratio", 0)
        vs = result["signals"].get("poisson_var_slope", 0)
        vm = result["signals"].get("speckle_vm_slope", 0)

        # Tier 1.5: db_ratio 物理信号 (在直方图之前, 复合退化更鲁棒)
        # ⚠️ 复合退化中 blur 会压低 Speckle 的 db_ratio (从 >15 到 2-10),
        # 与 Poisson 的 db_ratio (4-6) 重叠。用 vm_slope 区分:
        #   vm_slope > 0.001 → var/mean ∝ I (Speckle 乘性)
        #   vm_slope ≈ 0 → var ∝ I (Poisson 加性, var/mean ≈ constant)
        if db > 15 or (db > 2.0 and vm > 0.01):
            result["verdict"] = "SPECKLE"
            result["confidence"] = "high" if db > 15 else "medium"
            subjective_reason = f"db_ratio={db:.1f}>15" if db > 15 else f"db_ratio={db:.1f} in [2,15] + vm_slope={vm:.4f}>0.01 → Speckle (乘性噪声, var/mean∝I)"
        elif db > 2.0 and vs > 0.3:
            # Poisson 物理签名: 暗区方差小、亮区方差大 (var∝I)
            result["verdict"] = "POISSON"
            result["confidence"] = "medium" if db > 3.5 else "speculative"
            subjective_reason = f"db_ratio={db:.1f} in [2,15] + var_slope={vs:.2f}>0.3 → signal-dependent additive noise (Poisson-like, var∝I)"
        elif result["signals"].get("rgb_ratio", 1) > 1.4:
            result["verdict"] = "GAUSSIAN_YCrCb"
            result["confidence"] = "high" if abs(avg_skew) < 0.3 else "medium"
        elif avg_kurt > 5 or extreme_pct > 5:
            result["verdict"] = "IMPULSE"
            result["confidence"] = "high" if avg_kurt > 20 else "medium"
        elif avg_skew < -0.5:
            result["verdict"] = "SPECKLE"
            result["confidence"] = "high" if vm > 0.01 else "medium"
        # --- Tier 2: subjective guesses (直方图歧义, 辅助信号) ---
        elif vs > 0.5:
            # Signal-dependent variance (var ∝ intensity) — Poisson or Speckle signature.
            # Disambiguate via db_ratio (dark/bright absolute variance ratio) + vm_slope:
            #   Gaussian: db_ratio ~1.1, cv_ratio ~6.5
            #   Poisson:  db_ratio ~5,   cv_ratio ~2.8
            #   Speckle:  db_ratio ~25,  cv_ratio ~1.3
            vm = result["signals"].get("speckle_vm_slope", 0)
            db = result["signals"].get("poisson_db_ratio", 0)
            if db > 15:
                result["verdict"] = "SPECKLE"
                result["confidence"] = "medium" if vm > 0.001 else "low"
                subjective_reason = f"db_ratio={db:.1f}>15 → multiplicative noise (Speckle-like, bright var >> dark var)"
            elif db > 2.0:
                result["verdict"] = "POISSON"
                result["confidence"] = "medium" if db > 3.5 else "speculative"
                subjective_reason = f"db_ratio={db:.1f} in [2,15] + var_slope={result['signals']['poisson_var_slope']:.2f}>0.5 → signal-dependent additive noise (Poisson-like, var∝I)"
            elif vm > 0.001:
                result["verdict"] = "SPECKLE"
                result["confidence"] = "low"
                subjective_reason = f"var_slope={result['signals']['poisson_var_slope']:.3f}>0.5 + vm_slope={vm:.5f}>0.001 → speckle (low-sev, histogram missed)"
            else:
                result["verdict"] = "POISSON"
                result["confidence"] = "speculative"
                subjective_reason = f"var_slope={result['signals']['poisson_var_slope']:.3f}>0.5 + vm_slope≈0 + db_ratio={db:.1f} → signal-dependent noise (Poisson-like). NOT histogram-verified."
        elif sc > sc_threshold:
            result["verdict"] = "SPATIALLY_CORRELATED"
            result["confidence"] = "speculative"
            subjective_reason = f"spatial_corr={sc:.4f}>{sc_threshold} (sigma={avg_sigma:.1f}) → spatially structured noise. NOT histogram-verified."
        elif abs(avg_skew) < 0.3 and avg_kurt < 3:
            result["verdict"] = "GAUSSIAN_RGB"
            result["confidence"] = "high"
        else:
            result["verdict"] = "GAUSSIAN_RGB"
            result["confidence"] = "low"
        if subjective_reason:
            result["subjective_guess"] = subjective_reason

    # Severity estimation (type-specific maps calibrated on wavelet MAD sigma)
    v = result["verdict"]
    if v == "POISSON":
        # Poisson wavelet sigma: sev1=8.9, sev3=12.4, sev5=19.3
        sev_map = {5:1, 9:2, 11:3, 15:4, 17:5}
    elif v == "SPATIALLY_CORRELATED":
        # Spatially_correlated wavelet sigma: sev1=1.5, sev3=2.2, sev5=3.3
        sev_map = {0:1, 1.8:2, 2.5:3, 3.0:4, 3.1:5}
    elif v == "IMPULSE" and result.get("confidence") == "speculative":
        # Low-sigma impulse (wavelet-blind): severity from extreme_pct
        ep = result["signals"].get("impulse_extreme_pct", 0)
        result["estimated_severity"] = 1 if ep < 5 else (2 if ep < 15 else (3 if ep < 30 else 4))
        sev_map = None  # Already set
    else:
        # Gaussian-calibrated (works for GAUSSIAN_RGB, GAUSSIAN_YCrCb, SPECKLE, IMPULSE)
        sev_map = {2:1, 5:2, 10:3, 20:4, 35:5}
    if sev_map is not None:
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

    if glass_score > 0.25:
        result["verdict"] = "GLASS_BLUR"
        result["confidence"] = "medium" if glass_score > 0.35 else "low"
        result["note"] = f"glass_score={glass_score:.3f}>0.25 (pixel shuffle高频残留). Motion/Lens/Gaussian需Agent PSNR区分."
    else:
        result["verdict"] = "GAUSSIAN_BLUR"
        result["confidence"] = "medium" if gm_ratio_usable else "low"
        result["note"] = "Glass/Lens/Motion/Zoom不可靠区分 — Agent应PSNR测试替代子类型"

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
        noise_label = f"noise: {report['step2_noise']['verdict']} (sev~{report['step2_noise']['estimated_severity']})"
        if report["step2_noise"].get("subjective_guess"):
            noise_label += f" [subjective: {report['step2_noise']['subjective_guess'][:80]}...]"
        detections.append(noise_label)
        guess_types = ["POISSON", "SPATIALLY_CORRELATED"]
        if report['step2_noise']['verdict'] in guess_types:
            recommended_search.append(f"noise candidates: {report['step2_noise']['verdict']} (speculative) + GAUSSIAN_RGB, SPECKLE (verify via §B stats)")
        else:
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
