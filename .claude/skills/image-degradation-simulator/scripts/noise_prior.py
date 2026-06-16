#!/usr/bin/env python3
"""
Wavelet-based noise prior estimation.
Uses Haar DWT MAD estimator on HH subband for robust sigma estimation.
Works without clean reference (directly on degraded image) but is more
accurate with a clean reference (residual = degraded - clean).

Usage:
    python noise_prior.py --target degraded.png [--clean clean.png] [--output report.json]

Output: JSON with estimated sigma, noise type classification, and severity mapping.
"""
import argparse
import json
import sys
import numpy as np
from PIL import Image


# ======================================================================
# Pure-numpy 2D Haar Discrete Wavelet Transform
# ======================================================================

def haar_dwt2(img):
    """Single-level 2D Haar DWT. Returns LL, LH, HL, HH subbands."""
    h, w = img.shape
    h_even = h - (h % 2)
    w_even = w - (w % 2)
    img = img[:h_even, :w_even]

    # Row-wise: average and difference
    avg_rows = (img[:, 0::2] + img[:, 1::2]) / np.sqrt(2)
    diff_rows = (img[:, 0::2] - img[:, 1::2]) / np.sqrt(2)

    # Column-wise on averages
    LL = (avg_rows[0::2, :] + avg_rows[1::2, :]) / np.sqrt(2)
    LH = (avg_rows[0::2, :] - avg_rows[1::2, :]) / np.sqrt(2)

    # Column-wise on differences
    HL = (diff_rows[0::2, :] + diff_rows[1::2, :]) / np.sqrt(2)
    HH = (diff_rows[0::2, :] - diff_rows[1::2, :]) / np.sqrt(2)

    return LL, LH, HL, HH


# ======================================================================
# Noise Sigma Estimation
# ======================================================================

def estimate_noise_sigma_mad(img):
    """Robust noise sigma estimation via MAD of HH subband.

    Formula: sigma = MAD(HH) / 0.6745
    0.6745 = MAD of standard normal distribution.

    Returns sigma in [0,255] pixel intensity units.
    """
    gray = np.mean(img, axis=2) if img.ndim == 3 else img
    LL, LH, HL, HH = haar_dwt2(gray)
    hh_flat = HH.flatten()
    mad = np.median(np.abs(hh_flat - np.median(hh_flat)))
    sigma = mad / 0.6745
    return float(sigma)


def estimate_noise_per_channel(img):
    """Estimate noise sigma separately for R, G, B channels."""
    if img.ndim == 3 and img.shape[2] >= 3:
        sigmas = {}
        for i, ch in enumerate(['R', 'G', 'B']):
            ch_data = img[:, :, i]
            LL, LH, HL, HH = haar_dwt2(ch_data)
            mad = np.median(np.abs(HH.flatten() - np.median(HH.flatten())))
            sigmas[ch] = round(float(mad / 0.6745), 2)
        return sigmas
    return {"gray": estimate_noise_sigma_mad(img)}


# ======================================================================
# Sigma-Intensity Curve (for noise type discrimination)
# ======================================================================

def compute_sigma_intensity_curve(residual):
    """Sigma vs pixel intensity curve.

    Bins residual by intensity and computes std per bin.
    The shape discriminates noise types:
      - Flat curve ~ 0 slope → Gaussian (signal-independent)
      - Positive slope → Poisson-like (variance ∝ intensity)
      - Quadratic → Speckle (variance ∝ intensity²)
    """
    if residual.ndim == 3:
        gray = np.mean(residual, axis=2)
    else:
        gray = residual

    n_bins = 10
    bins = np.linspace(0, 255, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    sigma_per_bin = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (gray >= bins[i]) & (gray < bins[i+1])
        if mask.sum() > 10:
            sigma_per_bin[i] = gray[mask].std()

    # Linear fit: sigma = a * intensity + b
    valid = sigma_per_bin > 0
    if valid.sum() > 3:
        coeffs = np.polyfit(bin_centers[valid], sigma_per_bin[valid], 1)
        slope = float(coeffs[0])
        intercept = float(coeffs[1])

        # R-squared
        pred = np.polyval(coeffs, bin_centers[valid])
        ss_res = np.sum((sigma_per_bin[valid] - pred) ** 2)
        ss_tot = np.sum((sigma_per_bin[valid] - sigma_per_bin[valid].mean()) ** 2)
        r_squared = float(1 - ss_res / (ss_tot + 1e-8))
    else:
        slope, intercept, r_squared = 0.0, 0.0, 0.0

    return {
        "intensity_bins": bin_centers.tolist(),
        "sigma_per_bin": sigma_per_bin.tolist(),
        "linear_fit": {
            "slope": round(slope, 4),
            "intercept": round(intercept, 2),
            "r_squared": round(r_squared, 3),
        },
    }


# ======================================================================
# Noise Type Classification
# ======================================================================

def classify_noise_type(sigma_est, channel_sigmas, intensity_curve):
    """Classify noise type from wavelet sigma + channel ratios + intensity curve.

    Decision order (matching SKILL.md §B 6-step protocol):
    1. No significant noise → "no_noise"
    2. Channel imbalance → "gaussian_YCrCb"
    3. Sigma grows with intensity → "poisson" or "speckle"
    4. Default → "gaussian_RGB"
    """
    if sigma_est < 2.0:
        return {
            "predicted_type": "no_noise",
            "confidence": "high",
            "sigma_estimate": round(sigma_est, 2),
            "evidence": [f"Sigma={sigma_est:.1f} < 2.0 threshold"],
        }

    evidence = [f"Wavelet MAD sigma={sigma_est:.1f}"]
    conf = "medium"

    # Check channel imbalance for YCrCb detection
    if channel_sigmas and len(channel_sigmas) >= 3:
        ch_values = [channel_sigmas.get(ch, 0) for ch in ['R', 'G', 'B']]
        if min(ch_values) > 0:
            ch_ratio = max(ch_values) / min(ch_values)
            if ch_ratio > 1.4:
                evidence.append(f"Channel sigma ratio={ch_ratio:.1f} > 1.4 → YCrCb")
                return {
                    "predicted_type": "noise_gaussian_YCrCb",
                    "confidence": "high" if ch_ratio > 2.0 else "medium",
                    "sigma_estimate": round(sigma_est, 2),
                    "evidence": evidence,
                }

    # Check sigma-intensity slope
    curve = intensity_curve.get("linear_fit", {})
    slope = curve.get("slope", 0)
    r2 = curve.get("r_squared", 0)

    if r2 > 0.7:
        if abs(slope) > 0.05:
            evidence.append(f"Sigma-intensity slope={slope:.4f}, R²={r2:.2f} → Poisson/Speckle")
            return {
                "predicted_type": "noise_poisson",
                "confidence": "high",
                "sigma_estimate": round(sigma_est, 2),
                "evidence": evidence,
            }

    # Default: Gaussian RGB
    evidence.append(f"Slope={slope:.4f} (flat) → Gaussian RGB")
    return {
        "predicted_type": "noise_gaussian_RGB",
        "confidence": "high" if sigma_est > 8.0 else "medium",
        "sigma_estimate": round(sigma_est, 2),
        "evidence": evidence,
    }


# ======================================================================
# Sigma → Severity Mapping
# ======================================================================

# Calibrated sigma values from x_distortion (pixel intensity [0,255] scale)
NOISE_SIGMA_CALIBRATION = {
    "noise_gaussian_RGB": [12.75, 25.5, 38.25, 51.0, 63.75],
    "noise_gaussian_YCrCb": [12.75, 15.3, 17.85, 20.4, 22.95],
    "noise_speckle": [35.7, 53.55, 71.4, 89.25, 107.1],
    "noise_spatially_correlated": [20.4, 28.05, 35.7, 45.9, 56.1],
    "noise_poisson": None,  # Poisson uses variance = I/c parameter, not sigma
}

# Impulse noise is detected via extreme_pct, not sigma
IMPULSE_CALIBRATION = {
    "noise_impulse": [0.01, 0.03, 0.05, 0.07, 0.10],  # extreme_pct per severity
}


def map_sigma_to_severity(sigma_est, noise_type):
    """Map estimated sigma to closest severity level (1-5).

    Returns the severity level whose calibrated sigma is closest to the estimate.
    If estimated sigma is beyond the max, returns 5 with out_of_range=True.
    """
    if noise_type not in NOISE_SIGMA_CALIBRATION:
        return {"severity": 1, "closeness": 0.0, "out_of_range": False}

    calib = NOISE_SIGMA_CALIBRATION[noise_type]
    if calib is None:
        return {"severity": 2, "closeness": 0.0, "out_of_range": False,
                "note": "Poisson severity mapping requires intensity-dependent analysis"}

    if sigma_est <= calib[0] / 2:
        return {"severity": 1, "closeness": 0.0, "out_of_range": False,
                "note": "Estimated sigma below sev=1 range"}

    # Find closest severity
    diffs = [abs(sigma_est - s) for s in calib]
    best_idx = int(np.argmin(diffs))
    closeness = 1.0 - diffs[best_idx] / (calib[best_idx] + 1e-8)
    severity = best_idx + 1

    # Check if beyond max
    if sigma_est > calib[-1] * 1.3:
        return {"severity": 5, "closeness": round(closeness, 2), "out_of_range": True,
                "note": f"Sigma={sigma_est:.0f} significantly exceeds sev=5 range ({calib[-1]:.0f})"}

    return {"severity": severity, "closeness": round(closeness, 2), "out_of_range": False}


# ======================================================================
# Main Entry Point
# ======================================================================

def estimate_noise_from_image(target_path, clean_path=None):
    """Full noise estimation pipeline.

    Args:
        target_path: Path to degraded image
        clean_path: Optional path to clean reference (same-image mode)

    Returns:
        dict with sigma_estimate, noise_type, severity_estimate, method used
    """
    target = np.array(Image.open(target_path).convert("RGB"), dtype=np.float32)

    if clean_path:
        # Same-image mode: estimate from residual (more accurate)
        clean = np.array(Image.open(clean_path).convert("RGB"), dtype=np.float32)
        residual = target - clean
        sigma_est = estimate_noise_sigma_mad(residual + 128.0)  # shift for wavelet
        method = "residual_mad"
        # Check for blur: if gradient ratio < 0.5, blur is present and
        # wavelet sigma will be underestimated (blur removes HH energy)
        grad_t = np.mean(np.abs(np.gradient(np.mean(target, axis=2))))
        grad_c = np.mean(np.abs(np.gradient(np.mean(clean, axis=2))))
        blur_ratio = grad_t / (grad_c + 1e-8)
        blur_present = blur_ratio < 0.5
    else:
        residual = None
        sigma_est = estimate_noise_sigma_mad(target)
        method = "image_mad"
        blur_present = False  # can't determine without clean reference

    channel_sigmas = estimate_noise_per_channel(residual + 128.0 if residual is not None else target)
    intensity_curve = compute_sigma_intensity_curve(residual if residual is not None else target)

    noise_type = classify_noise_type(sigma_est, channel_sigmas, intensity_curve)
    severity_est = map_sigma_to_severity(sigma_est, noise_type["predicted_type"])

    result = {
        "sigma_estimate": round(sigma_est, 2),
        "sigma_per_channel": channel_sigmas,
        "intensity_curve": intensity_curve,
        "noise_type": noise_type,
        "severity_estimate": severity_est,
        "method": method,
    }
    if clean_path:
        result["blur_present"] = blur_present
        if blur_present:
            result["warning"] = (
                "Blur detected (gradient ratio < 0.5). Wavelet sigma may be "
                "underestimated because blur attenuates high-frequency noise. "
                "Consider this a lower bound estimate."
            )
    return result


def main():
    parser = argparse.ArgumentParser(description="Wavelet-based noise prior estimation")
    parser.add_argument("--target", required=True, help="Path to degraded image")
    parser.add_argument("--clean", default=None, help="Path to clean reference (optional)")
    parser.add_argument("--output", default=None, help="Output JSON path (prints to stdout if omitted)")
    args = parser.parse_args()

    result = estimate_noise_from_image(args.target, args.clean)

    if args.output:
        json.dump(result, open(args.output, "w"), indent=2)
        print(f"Saved to {args.output}")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
