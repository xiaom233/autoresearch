#!/usr/bin/env python3
"""
Compare two images using degradation-specific metrics.
Works with DIFFERENT images (cross-image) — no pixel-level comparison.

Usage:
    python compare_degradation.py --target /path/to/target.png --simulated /path/to/sim.png [--clean /path/to/clean.png]
"""

import argparse
import json
import sys
import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, '/mnt/zyli/projects/X-Distortion-main')
# Reuse analysis functions
from x_distortion.helper import *  # noqa - needed for analyze_degradation imports

# Import analyze functions
sys.path.insert(0, '/mnt/zyli/projects/X-Distortion-main/.claude/skills/image-degradation-simulator/scripts')
from analyze_degradation import (
    compute_basic_stats, compute_gradient_analysis, compute_noise_analysis,
    compute_compression_analysis, compute_ycrcb_analysis, compute_sharpening_analysis,
    compute_saturation_analysis, compute_radial_analysis, compute_frequency_analysis,
    compute_multiscale_analysis, compute_cross_channel_correlation
)


def load_image(path):
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32)


def metric_diff(name, target_val, sim_val, independent=True):
    """Format a single metric comparison."""
    diff = target_val - sim_val
    pct = (abs(diff) / (abs(target_val) + 1e-8)) * 100
    return {
        "metric": name,
        "target": round(target_val, 4) if isinstance(target_val, float) else target_val,
        "simulated": round(sim_val, 4) if isinstance(sim_val, float) else sim_val,
        "difference": round(diff, 4) if isinstance(diff, float) else diff,
        "pct_error": round(pct, 2) if isinstance(pct, float) else pct,
        "content_independent": independent
    }


def compare(target_path, simulated_path, clean_path=None):
    """Compare degradation characteristics between target and simulated images."""
    target = load_image(target_path)
    simulated = load_image(simulated_path)

    t_grad = compute_gradient_analysis(target)
    s_grad = compute_gradient_analysis(simulated)
    t_noise = compute_noise_analysis(target)
    s_noise = compute_noise_analysis(simulated)
    t_comp = compute_compression_analysis(target)
    s_comp = compute_compression_analysis(simulated)
    t_ycrcb = compute_ycrcb_analysis(target)
    s_ycrcb = compute_ycrcb_analysis(simulated)
    t_sharp = compute_sharpening_analysis(target)
    s_sharp = compute_sharpening_analysis(simulated)
    t_freq = compute_frequency_analysis(target)
    s_freq = compute_frequency_analysis(simulated)

    # Content-INDEPENDENT metrics (trust these)
    ci_metrics = [
        metric_diff("impulse_total_pct", t_noise["impulse_total_pct"], s_noise["impulse_total_pct"]),
        metric_diff("impulse_pct_zero", t_noise["impulse_pct_zero"], s_noise["impulse_pct_zero"]),
        metric_diff("impulse_pct_255", t_noise["impulse_pct_255"], s_noise["impulse_pct_255"]),
        metric_diff("block_boundary_ratio", t_comp["block_boundary_ratio"], s_comp["block_boundary_ratio"]),
        metric_diff("unique_R", t_comp["unique_R"], s_comp["unique_R"]),
        metric_diff("unique_G", t_comp["unique_G"], s_comp["unique_G"]),
        metric_diff("unique_B", t_comp["unique_B"], s_comp["unique_B"]),
        metric_diff("overshoot_ratio", t_sharp["overshoot_ratio"], s_sharp["overshoot_ratio"]),
        metric_diff("zero_crossing_density", t_sharp["zero_crossing_density"], s_sharp["zero_crossing_density"]),
        metric_diff("hf_lf_ratio", t_freq["hf_lf_ratio"], s_freq["hf_lf_ratio"]),
    ]

    # Content-DEPENDENT metrics (informational only, don't trust for cross-image)
    cd_metrics = [
        metric_diff("gradient_magnitude_mean", t_grad["gradient_magnitude_mean"], s_grad["gradient_magnitude_mean"], False),
        metric_diff("laplacian_variance", t_grad["laplacian_variance"], s_grad["laplacian_variance"], False),
        metric_diff("directional_h_v_ratio", t_grad["directional_h_v_ratio"], s_grad["directional_h_v_ratio"], False),
        metric_diff("flat_region_variance", t_noise["flat_region_variance"], s_noise["flat_region_variance"], False),
        metric_diff("saturation_mean", compute_saturation_analysis(target)["saturation_mean"],
                    compute_saturation_analysis(simulated)["saturation_mean"], False),
        metric_diff("Y_local_std_median", t_ycrcb["Y_local_std_median"], s_ycrcb["Y_local_std_median"], False),
        metric_diff("Cr_local_std_median", t_ycrcb["Cr_local_std_median"], s_ycrcb["Cr_local_std_median"], False),
        metric_diff("Cb_local_std_median", t_ycrcb["Cb_local_std_median"], s_ycrcb["Cb_local_std_median"], False),
        metric_diff("gradient_radial_ratio", compute_radial_analysis(target)["gradient_radial_ratio"],
                    compute_radial_analysis(simulated)["gradient_radial_ratio"], False),
    ]

    # Summary
    ci_passed = sum(1 for m in ci_metrics if m["pct_error"] < 20)
    ci_total = len(ci_metrics)

    # Direction assessment
    if clean_path:
        clean = load_image(clean_path)
        c_grad = compute_gradient_analysis(clean)
        c_noise = compute_noise_analysis(clean)
        t_imp = t_noise["impulse_total_pct"]
        s_imp = s_noise["impulse_total_pct"]
        c_imp = c_noise["impulse_total_pct"]

        # Did the simulation move the clean image in the right direction?
        direction = {}
        for name, t_val, s_val, c_val in [
            ("impulse", t_imp, s_imp, c_imp),
            ("gradient_magnitude", t_grad["gradient_magnitude_mean"], s_grad["gradient_magnitude_mean"], c_grad["gradient_magnitude_mean"]),
            ("laplacian_variance", t_grad["laplacian_variance"], s_grad["laplacian_variance"], c_grad["laplacian_variance"]),
        ]:
            t_diff = abs(t_val - c_val)
            s_diff = abs(s_val - c_val)
            if s_diff < t_diff * 1.05:  # within 5% of target
                direction[name] = "toward_target"
            elif abs(s_val - c_val) < 0.01:
                direction[name] = "unchanged"
            else:
                direction[name] = "overshoot" if abs(s_val - c_val) > abs(t_val - c_val) * 1.5 else "toward_target"
    else:
        direction = None

    return {
        "content_independent_metrics": ci_metrics,
        "content_dependent_metrics": cd_metrics,
        "ci_pass_rate": f"{ci_passed}/{ci_total}",
        "direction_vs_clean": direction,
        "verdict": "GOOD" if ci_passed >= ci_total * 0.7 else "NEEDS_WORK" if ci_passed >= ci_total * 0.4 else "POOR",
        "interpretation": {
            "GOOD": "Content-independent metrics closely match. Degradation type and severity are likely correct.",
            "NEEDS_WORK": "Some content-independent metrics don't match. Consider adjusting severity or checking for missing/extra degradations.",
            "POOR": "Major content-independent metric mismatches. The degradation type or pipeline is likely wrong."
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Compare degradation characteristics between two images")
    parser.add_argument("--target", required=True)
    parser.add_argument("--simulated", required=True)
    parser.add_argument("--clean", default=None, help="Optional clean reference for direction check")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    result = compare(args.target, args.simulated, args.clean)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Comparison saved to: {args.output}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
