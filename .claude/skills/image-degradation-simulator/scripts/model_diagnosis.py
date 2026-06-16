#!/usr/bin/env python3
"""
Failed model diagnosis: analyze residual patterns to identify what the model
failed to fix. Reuses analyze_degradation.py modules to detect remaining
degradation patterns in the residual (model_output - target).

Usage:
    python model_diagnosis.py --target degraded.png --predicted params.json \
        [--clean clean.png] [--output diagnosis.json]
"""
import argparse
import json
import sys
import os
import numpy as np
from PIL import Image

# Import analysis modules from sibling script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from analyze_degradation import (
    compute_compression_analysis,
    compute_noise_analysis,
    compute_sharpening_analysis,
    compute_gradient_analysis,
    compute_frequency_analysis,
)


def load_image(path):
    """Load image as float32 RGB array [0,255]."""
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32)


def compute_residuals(target, clean=None):
    """Compute residual images.

    Args:
        target: Degraded image
        clean: Optional clean reference (same-image mode)

    Returns:
        dict with residual arrays
    """
    residuals = {}
    if clean is not None:
        # Same-image: residual = target - clean shows the degradation
        residuals["degradation"] = target - clean
        # Shift for analysis (some modules expect 0-255 range)
        residuals["degradation_shifted"] = target - clean + 128.0
    return residuals


def analyze_residual_patterns(residual, target=None, clean=None):
    """Run residual through analysis modules to detect remaining degradation.

    Each module checks for a specific degradation signature in the residual.
    Returns structured pattern detection results.
    """
    # Shift residual to 0-255 range for analysis modules
    res = np.clip(residual + 128.0, 0, 255)

    patterns = {}

    # JPEG block detection
    comp = compute_compression_analysis(res)
    patterns["jpeg"] = {
        "detected": comp.get("block_boundary_ratio", 0) > 1.05,
        "block_boundary_ratio": round(comp.get("block_boundary_ratio", 0), 4),
        "unique_G": comp.get("unique_G", 256),
        "unique_R": comp.get("unique_R", 256),
        "unique_B": comp.get("unique_B", 256),
    }

    # Noise detection
    noise = compute_noise_analysis(res)
    patterns["noise"] = {
        "detected": noise.get("flat_region_variance", 0) > 50,
        "flat_region_variance": round(noise.get("flat_region_variance", 0), 2),
        "impulse_total_pct": round(noise.get("impulse_total_pct", 0), 4),
        "adjacent_pixel_diff_h": round(noise.get("adjacent_pixel_diff_h", 0), 4),
    }

    # Sharpening detection
    sharp = compute_sharpening_analysis(res)
    patterns["sharpening"] = {
        "detected": sharp.get("overshoot_ratio", 0) > 0.3,
        "overshoot_ratio": round(sharp.get("overshoot_ratio", 0), 4),
        "zero_crossing_density": round(sharp.get("zero_crossing_density", 0), 4),
    }

    # Structural blur in residual
    grad = compute_gradient_analysis(res)
    patterns["blur"] = {
        "detected": grad.get("gradient_magnitude_mean", 0) > 30,
        "gradient_mag_mean": round(grad.get("gradient_magnitude_mean", 0), 2),
        "h_v_ratio": round(grad.get("directional_h_v_ratio", 0), 4),
    }

    # Frequency analysis
    freq = compute_frequency_analysis(res)
    patterns["frequency"] = {
        "hf_lf_ratio": round(freq.get("hf_lf_ratio", 0), 4),
        "high_freq_in_residual": freq.get("hf_lf_ratio", 0) > 0.5,
    }

    return patterns


def classify_failures(patterns):
    """Classify failure modes from residual pattern analysis.

    Returns list of failure dicts, each with:
        failure: what was missed
        confidence: high/medium/low
        suggestion: what to add/change in the pipeline
    """
    failures = []

    # JPEG: block boundaries in residual + unique colors reduced
    if patterns["jpeg"]["detected"]:
        sev_hint = "1-2" if patterns["jpeg"]["unique_G"] > 200 else "3-5"
        failures.append({
            "failure": "missed_compression_jpeg",
            "confidence": "high" if patterns["jpeg"]["block_boundary_ratio"] > 1.2 else "medium",
            "evidence": f"Residual block_boundary={patterns['jpeg']['block_boundary_ratio']:.2f}",
            "suggestion": f"Add compression_jpeg severity {sev_hint} to pipeline",
        })

    # JPEG2000: low unique colors without 8x8 blocks → JPEG2000
    if (patterns["jpeg"]["unique_G"] < 200 and
            patterns["jpeg"]["block_boundary_ratio"] < 1.05):
        failures.append({
            "failure": "missed_compression_jpeg2000",
            "confidence": "low",
            "evidence": f"Residual unique_G={patterns['jpeg']['unique_G']} without JPEG blocks",
            "suggestion": "Consider adding compression_jpeg_2000",
        })

    # Noise: high flat variance in residual
    if patterns["noise"]["detected"]:
        noise_type = "gaussian_RGB"
        if patterns["noise"]["impulse_total_pct"] > 0.3:
            noise_type = "impulse"
        failures.append({
            "failure": "missed_noise",
            "confidence": "high" if patterns["noise"]["flat_region_variance"] > 200 else "medium",
            "evidence": (
                f"Residual flat_var={patterns['noise']['flat_region_variance']:.0f}, "
                f"impulse={patterns['noise']['impulse_total_pct']:.2f}%"
            ),
            "suggestion": f"Add {noise_type} noise (severity TBD by noise_prior.py)",
        })

    # Sharpening: overshoot in residual
    if patterns["sharpening"]["detected"]:
        failures.append({
            "failure": "missed_oversharpen",
            "confidence": "medium",
            "evidence": f"Residual overshoot={patterns['sharpening']['overshoot_ratio']:.2f}",
            "suggestion": "Add oversharpen to pipeline",
        })

    # Blur: high gradient in residual → model output still has edges
    if patterns["blur"]["detected"]:
        directional = abs(patterns["blur"]["h_v_ratio"] - 1.0) > 0.3
        blur_hint = "motion" if directional else "gaussian/lens/glass"
        failures.append({
            "failure": "wrong_blur_type_or_severity",
            "confidence": "medium",
            "evidence": (
                f"Residual gradient={patterns['blur']['gradient_mag_mean']:.1f}, "
                f"h_v_ratio={patterns['blur']['h_v_ratio']:.2f}"
            ),
            "suggestion": f"Adjust blur type (try {blur_hint}) or severity",
        })

    if not failures:
        failures.append({
            "failure": "no_clear_pattern",
            "confidence": "low",
            "evidence": "No significant degradation pattern detected in residual",
            "suggestion": "Predicted pipeline may be correct (PSNR limited by noise randomness)",
        })

    return failures


def per_channel_analysis(residual):
    """Analyze R, G, B residuals separately."""
    if residual.ndim != 3 or residual.shape[2] < 3:
        return {}
    ch_stats = {}
    for i, ch in enumerate(['R', 'G', 'B']):
        ch_data = residual[:, :, i]
        ch_stats[ch] = {
            "mean": round(float(ch_data.mean()), 2),
            "std": round(float(ch_data.std()), 2),
            "min": round(float(ch_data.min()), 2),
            "max": round(float(ch_data.max()), 2),
        }
    # Cross-channel comparison
    stds = [ch_stats[ch]["std"] for ch in ['R', 'G', 'B']]
    ch_stats["cross_channel"] = {
        "max_std_ratio": round(max(stds) / (min(stds) + 1e-8), 2),
        "interpretation": (
            "Ratio > 1.4 → YCrCb noise (channel imbalance). "
            "Ratio ≈ 1.0 → RGB noise (equal per-channel)."
        ),
    }
    return ch_stats


def diagnose(target_path, clean_path=None, predicted_params_path=None, output_path=None):
    """Full diagnosis pipeline: load images, analyze residual, classify failures.

    Args:
        target_path: Path to degraded (target) image
        clean_path: Path to clean reference (same-image mode, recommended)
        predicted_params_path: Path to the predicted degradation params JSON
        output_path: Optional path to save JSON output

    Returns:
        dict with failures, residual_analysis, per_channel
    """
    target = load_image(target_path)
    clean = load_image(clean_path) if clean_path else None

    # Compute residuals
    residuals = compute_residuals(target, clean)

    # Use degradation residual if available, otherwise target itself
    res = residuals.get("degradation_shifted", target)

    # Analyze residual patterns
    patterns = analyze_residual_patterns(res, target, clean)

    # Classify failures
    failures = classify_failures(patterns)

    # Per-channel analysis
    ch_analysis = per_channel_analysis(residuals.get("degradation", target))

    # Suggested corrections
    suggestions = [f["suggestion"] for f in failures if f["failure"] != "no_clear_pattern"]

    result = {
        "failures": failures,
        "residual_analysis": patterns,
        "per_channel": ch_analysis,
        "suggested_corrections": suggestions,
        "has_clean_reference": clean_path is not None,
    }

    if output_path:
        json.dump(result, open(output_path, "w"), indent=2, ensure_ascii=False)

    return result


def main():
    parser = argparse.ArgumentParser(description="Diagnose failed model from residual patterns")
    parser.add_argument("--target", required=True, help="Path to degraded (target) image")
    parser.add_argument("--clean", default=None, help="Path to clean reference (same-image mode)")
    parser.add_argument("--predicted", default=None, help="Path to predicted_params.json")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    result = diagnose(args.target, args.clean, args.predicted, args.output)

    if args.output:
        print(f"Saved to {args.output}")
        print(f"Failures: {len(result['failures'])}")
        for f in result['failures']:
            print(f"  - {f['failure']} ({f['confidence']}): {f['suggestion']}")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
