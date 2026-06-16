#!/usr/bin/env python3
"""
Analyze a degraded image to identify distortion types and severity.
Outputs a JSON report with quantitative evidence for each degradation category.

Usage:
    python analyze_degradation.py --target /path/to/degraded.png [--clean /path/to/clean.png]
"""

import argparse
import json
import sys
import numpy as np
from PIL import Image
from scipy import ndimage


def compute_basic_stats(img):
    """Per-channel and overall statistics."""
    gray = np.mean(img, axis=2)
    stats = {
        "mean": float(np.mean(img)),
        "std": float(np.std(img)),
        "min": float(np.min(img)),
        "max": float(np.max(img)),
        "per_channel": {
            "R": {"mean": float(np.mean(img[:,:,0])), "std": float(np.std(img[:,:,0]))},
            "G": {"mean": float(np.mean(img[:,:,1])), "std": float(np.std(img[:,:,1]))},
            "B": {"mean": float(np.mean(img[:,:,2])), "std": float(np.std(img[:,:,2]))},
        }
    }
    return stats


def compute_gradient_analysis(img):
    """Gradient magnitude for blur detection. Also directional ratio for motion blur."""
    gray = np.mean(img, axis=2)
    gx = ndimage.sobel(gray, axis=0)
    gy = ndimage.sobel(gray, axis=1)
    grad_mag = np.sqrt(gx**2 + gy**2)

    # Directional analysis: horizontal vs vertical gradient
    h_grad = np.mean(np.abs(gx))  # vertical edges (horizontal gradient)
    v_grad = np.mean(np.abs(gy))  # horizontal edges (vertical gradient)
    h_v_ratio = h_grad / (v_grad + 1e-8)

    # Laplacian variance (sharpness measure)
    laplacian = ndimage.laplace(gray)
    lap_var = float(np.var(laplacian))

    return {
        "gradient_magnitude_mean": float(np.mean(grad_mag)),
        "gradient_magnitude_std": float(np.std(grad_mag)),
        "laplacian_variance": lap_var,
        "directional_h_grad": float(h_grad),
        "directional_v_grad": float(v_grad),
        "directional_h_v_ratio": float(h_v_ratio),
        "interpretation": "h_v_ratio ≈ 1.0 → isotropic blur (Gaussian/Lens). h_v_ratio significantly ≠ 1.0 → directional blur (Motion)."
    }


def compute_noise_analysis(img):
    """Noise type detection: impulse, Gaussian, spatially correlated."""
    gray = np.mean(img, axis=2)
    h, w = gray.shape

    # Impulse noise: pixel extremes
    pct_zero = float(np.sum(img <= 1) / img.size * 100)
    pct_255 = float(np.sum(img >= 254) / img.size * 100)

    # Flat region noise: local variance in smooth areas
    flat_mask = ndimage.laplace(gray) < np.percentile(np.abs(ndimage.laplace(gray)), 20)
    flat_variance = float(np.var(gray[flat_mask])) if np.sum(flat_mask) > 0 else 0.0

    # Adjacent pixel differences for noise grain
    adj_h = np.mean(np.abs(gray[:, 1:] - gray[:, :-1]))
    adj_v = np.mean(np.abs(gray[1:, :] - gray[:-1, :]))

    return {
        "impulse_pct_zero": round(pct_zero, 4),
        "impulse_pct_255": round(pct_255, 4),
        "impulse_total_pct": round(pct_zero + pct_255, 4),
        "flat_region_variance": round(flat_variance, 4),
        "adjacent_pixel_diff_h": round(float(adj_h), 4),
        "adjacent_pixel_diff_v": round(float(adj_v), 4),
        "interpretation": {
            "impulse": "If pct_zero + pct_255 > 0.5%, likely impulse/salt&pepper noise.",
            "gaussian": "If flat_region_variance elevated (>1.0) but no impulse extremes, likely Gaussian noise.",
            "spatially_correlated": "If adjacent differences vary across regions, likely correlated noise."
        }
    }


def compute_compression_analysis(img):
    """JPEG compression and quantization detection."""
    unique_per_channel = {}
    for ch_idx, ch_name in enumerate(['R', 'G', 'B']):
        unique_per_channel[ch_name] = int(len(np.unique(img[:,:,ch_idx])))

    # Total unique RGB combinations
    # Sample pixels for efficiency on large images
    flat = img.reshape(-1, 3)
    if flat.shape[0] > 100000:
        idx = np.random.choice(flat.shape[0], 100000, replace=False)
        flat = flat[idx]
    unique_rgb = len(set(tuple(p) for p in flat))

    # 8x8 block boundary analysis for JPEG detection
    gray = np.mean(img, axis=2)
    h, w = gray.shape
    blockiness = 0.0
    if h >= 8 and w >= 8:
        h_blocks = gray[:, 7::8].astype(float)
        h_neighbors = gray[:, 8::8].astype(float)
        if h_blocks.shape[1] > 1 and h_neighbors.shape[1] > 1:
            h_block_diff = np.mean(np.abs(h_blocks[:, :h_neighbors.shape[1]] - h_neighbors[:, :h_blocks.shape[1]]))
            h_inner_diff = np.mean(np.abs(h_blocks[:, 1:] - h_blocks[:, :-1]))
            if h_inner_diff > 0:
                blockiness = float(h_block_diff / h_inner_diff)

    return {
        "unique_R": unique_per_channel["R"],
        "unique_G": unique_per_channel["G"],
        "unique_B": unique_per_channel["B"],
        "unique_RGB_estimate": unique_rgb,
        "block_boundary_ratio": round(blockiness, 4),
        "interpretation": {
            "jpeg": "If unique_G < 255 or unique_RGB low and block_boundary_ratio > 1.1, likely JPEG compression.",
            "quantization": "If unique_R/G/B all < 32, likely color quantization (posterization).",
            "jpeg2000": "If quality loss without 8x8 blocking, possible JPEG2000."
        }
    }


def compute_jpeg2000_analysis(img):
    """JPEG2000-specific detection via wavelet ringing artifacts.

    JPEG2000 uses wavelet compression (not DCT), so:
    - No 8x8 block boundaries (unlike JPEG)
    - Ringing/oscillation artifacts near edges (Gibbs phenomenon from wavelet quantization)
    - High-frequency energy distribution differs from JPEG

    Detection method:
    1. Edge-adjacent ringing: measure oscillation amplitude near strong edges
    2. High-frequency spatial distribution: JPEG2000 ringing is concentrated near edges,
       while noise is uniform
    """
    gray = np.mean(img.astype(float), axis=2)

    # Edge detection
    gy, gx = np.gradient(gray)
    edge_mag = np.sqrt(gx**2 + gy**2)
    edge_mask = edge_mag > np.percentile(edge_mag, 90)

    # Ringing detection: Laplacian oscillation near edges
    laplacian = ndimage.laplace(gray)

    # Dilate edge mask to capture ringing region
    from scipy.ndimage import binary_dilation
    edge_region = binary_dilation(edge_mask, iterations=3)
    non_edge_region = ~edge_region

    # Ringing: ratio of high-frequency energy near edges vs away from edges
    edge_hf = np.std(np.abs(laplacian[edge_region])) if np.sum(edge_region) > 0 else 0
    non_edge_hf = np.std(np.abs(laplacian[non_edge_region])) if np.sum(non_edge_region) > 0 else 1e-8
    ringing_ratio = edge_hf / non_edge_hf if non_edge_hf > 0 else 1.0

    # Oscillation count: number of sign changes in laplacian near edges
    edge_laplacian = laplacian[edge_region]
    sign_changes = np.sum(np.abs(np.diff(np.sign(edge_laplacian)))) / 2 if len(edge_laplacian) > 1 else 0
    oscillation_density = sign_changes / len(edge_laplacian) if len(edge_laplacian) > 0 else 0

    # Cross-validation: jitter blur also creates edge-adjacent HF (multiple edges)
    # JPEG2000 should also show compression artifacts (bytes_per_pixel or unique color change)
    # Jitter blur: gradient normal or increased, no compression artifacts
    # Use oscillation_density threshold: jitter has higher oscillation (>0.5) than JPEG2000 ringing (0.3-0.5)

    return {
        "ringing_ratio": round(float(ringing_ratio), 4),
        "oscillation_density": round(float(oscillation_density), 4),
        "interpretation": {
            "jpeg2000_ringing": "ringing_ratio > 2.0 AND oscillation_density 0.3-0.5 → wavelet ringing (JPEG2000).",
            "jitter_not_jpeg2000": "ringing_ratio > 2.0 AND oscillation_density > 0.5 → likely jitter blur (edge duplication), not JPEG2000.",
            "noise_not_ringing": "ringing_ratio ≈ 1.0 → HF energy uniform (likely noise, not JPEG2000 ringing).",
        }
    }


def compute_noise_subtype_analysis(img):
    """Sub-type noise discrimination: Poisson, speckle, spatially correlated.

    Poisson noise: variance proportional to signal intensity (photon counting).
    Speckle noise: multiplicative, variance highest in bright regions.
    Spatially correlated: adjacent pixel differences show spatial structure.
    """
    gray = np.mean(img.astype(float), axis=2)
    h, w = gray.shape

    # Signal-dependent noise: divide image into intensity bins, measure variance per bin
    n_bins = 10
    bin_edges = np.linspace(0, 255, n_bins + 1)
    bin_variances = []
    bin_means = []
    for i in range(n_bins):
        mask = (gray >= bin_edges[i]) & (gray < bin_edges[i+1])
        if np.sum(mask) > 100:
            bin_variances.append(np.var(gray[mask]))
            bin_means.append(np.mean(gray[mask]))

    # Poisson: variance ∝ mean. Spearman correlation between mean and variance
    signal_var_corr = 0.0
    if len(bin_means) > 3:
        from scipy.stats import spearmanr
        signal_var_corr, _ = spearmanr(bin_means, bin_variances)

    # Speckle: ratio of variance to mean^2 (speckle contrast)
    speckle_contrast = np.mean(bin_variances) / (np.mean(bin_means)**2 + 1e-8) if bin_means else 0

    # Spatial correlation: autocorrelation of adjacent differences
    adj_diff = np.abs(gray[1:, 1:] - gray[:-1, :-1])
    # Measure spatial clustering of high-diff regions
    high_diff_mask = adj_diff > np.percentile(adj_diff, 80)
    spatial_cluster = float(np.mean(high_diff_mask)) if high_diff_mask.size > 0 else 0

    # Poisson: additional check - ratio of var/mean across bins (Poisson has var≈mean)
    var_mean_ratios = []
    if bin_means and bin_variances:
        var_mean_ratios = [v/(m+1e-8) for v,m in zip(bin_variances, bin_means) if m > 0]
    avg_var_mean_ratio = float(np.mean(var_mean_ratios)) if var_mean_ratios else 0

    return {
        "signal_var_correlation": round(float(signal_var_corr), 4),
        "var_mean_ratio": round(avg_var_mean_ratio, 4),
        "speckle_contrast": round(float(speckle_contrast), 6),
        "spatial_cluster_ratio": round(float(spatial_cluster), 4),
        "interpretation": {
            "poisson": "signal_var_correlation > 0.5 OR var_mean_ratio > 5 → variance linked to signal intensity (Poisson/shot noise characteristic).",
            "speckle": "speckle_contrast > 0.01 → multiplicative noise pattern (speckle).",
            "spatially_correlated": "spatial_cluster_ratio > 0.3 → noise has spatial structure, not independent."
        }
    }


def compute_blur_subtype_analysis(img, directional_h_v_ratio, gradient_radial_ratio):
    """Combined blur subtype classification based on directional and radial metrics.

    Key differentiators:
    - Gaussian: isotropic (h_v_ratio ≈ clean), uniform across frame (radial ≈ 1.0)
    - Motion: directional (h_v_ratio significantly ≠ clean)
    - Lens: radial falloff (radial_ratio > 1.3, center sharper)
    - Jitter: may increase gradient magnitude (multiple overlapping edges)
    - Glass: irregular distortion, radial_ratio 1.1-1.3
    - Zoom: radial_ratio < 0.9 (center blur) or radial with isotropic direction
    """
    gray = np.mean(img.astype(float), axis=2)

    # Jitter detection: measures edge duplication
    # Jitter creates multiple slightly-offset edges → increased zero-crossing density
    # without the overshoot of sharpening
    laplacian = ndimage.laplace(gray)
    zero_crossings = np.sum(np.abs(np.diff(np.sign(laplacian[::2, ::2])))) / 2  # subsampled for speed
    zc_density = zero_crossings / (gray.size / 4)

    return {
        "zero_crossing_density": round(float(zc_density), 4),
        "directional_h_v_ratio": round(float(directional_h_v_ratio), 4),
        "gradient_radial_ratio": round(float(gradient_radial_ratio), 4),
        "blur_subtype_hints": {
            "gaussian": "h_v_ratio stable vs clean + radial ≈ 1.0 + gradient decreased",
            "motion": "h_v_ratio significantly changed vs clean (>15%)",
            "lens": "radial_ratio > 1.3 (center sharper than periphery)",
            "jitter": "gradient may increase + zero_crossing elevated + no directional change",
            "glass": "radial_ratio 1.1-1.3 + irregular local variance",
            "zoom": "radial_ratio < 0.9 or radial with isotropic direction"
        }
    }


def compute_ycrcb_analysis(img):
    """YCrCb decomposition for noise type and JPEG chroma subsampling detection.
    Y=luma, Cr/Cb=chroma. JPEG chroma subsampling suppresses Cb/Cr variance."""
    import cv2
    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    ycrcb = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2YCR_CB).astype(np.float32)

    # Local patch std for each channel (16x16 patches)
    local_std = {}
    for ch_idx, ch_name in enumerate(['Y', 'Cr', 'Cb']):
        channel = ycrcb[:, :, ch_idx]
        h, w = channel.shape
        patch_stds = []
        for y in range(0, h - 16, 16):
            for x in range(0, w - 16, 16):
                patch = channel[y:y+16, x:x+16]
                patch_stds.append(float(np.std(patch)))
        local_std[ch_name] = {
            "median": round(float(np.median(patch_stds)), 4),
            "mean": round(float(np.mean(patch_stds)), 4),
        }

    # Chroma subsampling signature: suppressed Cb/Cr variance → JPEG
    cr_var = float(np.var(ycrcb[:, :, 1]))
    cb_var = float(np.var(ycrcb[:, :, 2]))

    return {
        "Y_local_std_median": local_std["Y"]["median"],
        "Cr_local_std_median": local_std["Cr"]["median"],
        "Cb_local_std_median": local_std["Cb"]["median"],
        "Cr_variance": round(cr_var, 4),
        "Cb_variance": round(cb_var, 4),
        "interpretation": {
            "gaussian_noise": "Elevated Y local std + suppressed Cr/Cb → Gaussian noise + JPEG chroma subsampling.",
            "jpeg_subsampling": "Low Cb/Cr variance (< ~5) relative to luma → JPEG 4:2:0 chroma subsampling.",
            "noise_only": "Elevated Y + proportionate Cr/Cb elevation → noise without JPEG subsampling.",
        }
    }


def compute_multiscale_analysis(img):
    """Multi-scale variance for pixelate detection.
    Pixelation shows abrupt variance change at the pixelation scale."""
    gray = np.mean(img, axis=2)
    h, w = gray.shape

    scales = []
    for scale in [2, 4, 8, 16, 32]:
        if h >= scale * 2 and w >= scale * 2:
            # Downsample then upsample
            from PIL import Image as PILImage
            small_h, small_w = h // scale, w // scale
            pil_img = PILImage.fromarray(gray.astype(np.uint8))
            down = np.array(pil_img.resize((small_w, small_h), PILImage.NEAREST), dtype=np.float32)
            up = np.array(PILImage.fromarray(down.astype(np.uint8)).resize((w, h), PILImage.NEAREST), dtype=np.float32)
            residual_var = float(np.var(gray - up))
            scales.append({"scale": scale, "residual_variance": round(residual_var, 4)})

    return {
        "scales": scales,
        "interpretation": "Abrupt spike in residual variance at a specific scale → pixelation at that scale. Smooth roll-off → natural image or other degradation."
    }


def compute_cross_channel_correlation(img):
    """Cross-channel correlation for noise type discrimination."""
    flat = img.reshape(-1, 3).astype(np.float32)
    if flat.shape[0] > 50000:
        idx = np.random.choice(flat.shape[0], 50000, replace=False)
        flat = flat[idx]

    corr_matrix = np.corrcoef(flat.T)
    return {
        "R_G_correlation": round(float(corr_matrix[0, 1]), 4),
        "R_B_correlation": round(float(corr_matrix[0, 2]), 4),
        "G_B_correlation": round(float(corr_matrix[1, 2]), 4),
        "interpretation": {
            "high_correlation": "High cross-channel correlation (>0.8) → brightness/contrast changes or natural image structure.",
            "low_correlation": "Low cross-channel correlation → independent per-channel noise (Gaussian RGB) or JPEG chroma subsampling.",
            "ycrb_noise": "If noise is in YCrCb, Cr/Cb channels will show different correlation pattern from RGB noise."
        }
    }


def compute_file_info(target_path):
    """File size and format info."""
    import os
    size_bytes = os.path.getsize(target_path)
    size_kb = round(size_bytes / 1024, 2)
    img = Image.open(target_path)
    pixel_count = img.size[0] * img.size[1]
    bytes_per_pixel = round(size_bytes / pixel_count, 4)
    return {
        "file_size_kb": size_kb,
        "bytes_per_pixel": bytes_per_pixel,
        "interpretation": "JPEG: < 1 byte/pixel. PNG/uncompressed: > 2 bytes/pixel. Lower bytes/pixel → stronger compression."
    }


def compute_sharpening_analysis(img):
    """Oversharpen detection: Laplacian overshoot, halo artifacts at edges.
    Oversharpening creates unnatural edge enhancement visible as bright/dark halos."""
    gray = np.mean(img, axis=2)
    laplacian = ndimage.laplace(gray)

    # Zero-crossing density: oversharpened images have exaggerated zero-crossings
    zero_crossings = (laplacian[:-1, :] * laplacian[1:, :] < 0).sum() + \
                     (laplacian[:, :-1] * laplacian[:, 1:] < 0).sum()
    zc_density = float(zero_crossings) / (gray.size * 2)

    # Laplacian absolute mean: higher → more edge enhancement
    lap_abs_mean = float(np.mean(np.abs(laplacian)))

    # Overshoot detection: ratio of Laplacian extremes to gradient magnitude
    gx = ndimage.sobel(gray, axis=0)
    gy = ndimage.sobel(gray, axis=1)
    grad_mag = np.sqrt(gx**2 + gy**2)
    overshoot_ratio = float(np.std(laplacian) / (np.mean(grad_mag) + 1e-8))

    return {
        "zero_crossing_density": round(zc_density, 6),
        "laplacian_abs_mean": round(lap_abs_mean, 4),
        "overshoot_ratio": round(overshoot_ratio, 4),
        "interpretation": {
            "oversharpen": "High overshoot_ratio (>0.5) + high zero_crossing_density → likely oversharpened. Edge halos visible as alternating bright/dark bands.",
            "normal": "Low overshoot_ratio (<0.3) → natural edges or blurred (no sharpening).",
            "note": "This method works on any single image — no clean reference needed."
        }
    }


def compute_saturation_analysis(img):
    """HSV saturation channel analysis for saturation modification detection."""
    import cv2
    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
    s_channel = hsv[:, :, 1]

    # Saturation statistics
    s_mean = float(np.mean(s_channel))
    s_std = float(np.std(s_channel))
    s_median = float(np.median(s_channel))

    # Saturation histogram skew
    s_hist, _ = np.histogram(s_channel, bins=50, range=(0, 255))
    s_skew = float(np.sum((np.arange(50) - 25)**3 * s_hist) / (np.sum(s_hist) * 12.5**3 + 1e-8))

    return {
        "saturation_mean": round(s_mean, 4),
        "saturation_std": round(s_std, 4),
        "saturation_median": round(s_median, 4),
        "saturation_skew": round(s_skew, 4),
        "interpretation": {
            "strengthened": "High saturation_mean (>100) + positive skew → saturation boosted.",
            "weakened": "Low saturation_mean (<30) + negative skew → desaturated, possibly near grayscale.",
            "jpeg_subsampling_note": "JPEG chroma subsampling may slightly reduce saturation_std — distinguish from deliberate desaturation by checking compression metrics."
        }
    }


def compute_radial_analysis(img):
    """Center vs periphery gradient comparison for lens/zoom blur detection.
    Lens blur: soft edges, sharp center. Zoom blur: radial streaking pattern."""
    gray = np.mean(img, axis=2)
    h, w = gray.shape
    cy, cx = h // 2, w // 2

    # Define center region (inner 40%) and periphery (outer 30%)
    y, x = np.ogrid[:h, :w]
    max_dist = np.sqrt(cy**2 + cx**2)
    dist = np.sqrt((y - cy)**2 + (x - cx)**2) / max_dist

    center_mask = dist < 0.4
    periphery_mask = dist > 0.7

    # Gradient magnitude in center vs periphery
    gx = ndimage.sobel(gray, axis=0)
    gy = ndimage.sobel(gray, axis=1)
    grad_mag = np.sqrt(gx**2 + gy**2)

    center_grad = float(np.mean(grad_mag[center_mask])) if center_mask.sum() > 0 else 0
    periphery_grad = float(np.mean(grad_mag[periphery_mask])) if periphery_mask.sum() > 0 else 0
    radial_ratio = center_grad / (periphery_grad + 1e-8)

    # Laplacian variance in center vs periphery
    laplacian = np.abs(ndimage.laplace(gray))
    center_lap = float(np.mean(laplacian[center_mask])) if center_mask.sum() > 0 else 0
    periphery_lap = float(np.mean(laplacian[periphery_mask])) if periphery_mask.sum() > 0 else 0
    lap_radial_ratio = center_lap / (periphery_lap + 1e-8)

    return {
        "center_gradient_mean": round(center_grad, 4),
        "periphery_gradient_mean": round(periphery_grad, 4),
        "gradient_radial_ratio": round(radial_ratio, 4),
        "laplacian_radial_ratio": round(lap_radial_ratio, 4),
        "interpretation": {
            "lens_blur": "radial_ratio > 1.3 → center sharper than periphery, likely lens blur or vignette blur.",
            "zoom_blur": "radial_ratio < 1.0 but h_v_ratio ≈ 1.0 → uniform blur with possible radial streaking, check visually.",
            "uniform_blur": "radial_ratio ≈ 1.0 → uniform blur across frame (Gaussian, motion).",
            "vignette": "Check brightness radial profile — darkening at edges combined with radial blur suggests vignette."
        }
    }


def compute_noise_vs_sharpen_analysis(img):
    """Distinguish noise from oversharpening. Both elevate overshoot_ratio, but:
    - Oversharpen: Laplacian extremes cluster at edges, flat regions are clean
    - Noise: Laplacian extremes are uniformly distributed across the image"""
    gray = np.mean(img, axis=2)
    laplacian = np.abs(ndimage.laplace(gray))

    # Edge mask: top 15% gradient magnitude pixels
    gx = ndimage.sobel(gray, axis=0)
    gy = ndimage.sobel(gray, axis=1)
    grad_mag = np.sqrt(gx**2 + gy**2)
    edge_threshold = np.percentile(grad_mag, 85)
    edge_mask = grad_mag > edge_threshold
    flat_mask = ~edge_mask

    # Laplacian variance in edge vs flat regions
    edge_lap_var = float(np.var(laplacian[edge_mask])) if edge_mask.sum() > 100 else 0
    flat_lap_var = float(np.var(laplacian[flat_mask])) if flat_mask.sum() > 100 else 0
    edge_flat_ratio = edge_lap_var / (flat_lap_var + 1e-8)

    # Spatial autocorrelation of Laplacian extremes (top 5%)
    lap_threshold = np.percentile(laplacian, 95)
    extreme_mask = laplacian > lap_threshold
    extreme_pixels = np.argwhere(extreme_mask)
    if len(extreme_pixels) > 10:
        # Mean nearest-neighbor distance between extreme pixels
        from scipy.spatial import cKDTree
        if len(extreme_pixels) > 1000:
            idx = np.random.choice(len(extreme_pixels), 1000, replace=False)
            extreme_pixels = extreme_pixels[idx]
        tree = cKDTree(extreme_pixels)
        distances, _ = tree.query(extreme_pixels, k=2)
        nn_mean = float(np.mean(distances[:, 1])) if distances.shape[1] > 1 else 0
        nn_std = float(np.std(distances[:, 1])) if distances.shape[1] > 1 else 0
    else:
        nn_mean, nn_std = 0, 0

    return {
        "edge_lap_variance": round(edge_lap_var, 4),
        "flat_lap_variance": round(flat_lap_var, 4),
        "edge_flat_ratio": round(edge_flat_ratio, 4),
        "extreme_nn_distance_mean": round(nn_mean, 2),
        "extreme_nn_distance_std": round(nn_std, 2),
        "verdict": "oversharpen" if (nn_mean < 5.0) else "noise" if (nn_mean > 6.0) else "uncertain",
        "interpretation": {
            "oversharpen": f"nn_distance < 5.5 with low std → Laplacian extremes cluster along edges (oversharpen halos). Current: nn_mean={nn_mean:.1f} nn_std={nn_std:.1f}",
            "noise": f"nn_distance > 6.0 → Laplacian extremes randomly scattered (noise grain, not edge halos). Current: nn_mean={nn_mean:.1f}",
            "extreme_clustering": f"Low nn_distance + low nn_std → clustered at edges = oversharpen. High nn_distance + high nn_std → random = noise.",
            "usage": "When overshoot_ratio > 0.5, check this verdict to determine if the source is oversharpen or noise."
        }
    }


def compute_frequency_analysis(img):
    """FFT-based frequency analysis for blur/compression detection."""
    gray = np.mean(img, axis=2)
    f = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.log(np.abs(f) + 1)
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    # Low vs high frequency energy
    r_low = min(h, w) * 0.1
    y, x = np.ogrid[:h, :w]
    mask_low = (y - cy)**2 + (x - cx)**2 <= r_low**2
    low_energy = float(magnitude[mask_low].mean())
    high_energy = float(magnitude[~mask_low].mean())

    # 8x8 frequency peaks (JPEG signature)
    peak_8x8_h = float(np.mean(magnitude[cy, cx-4:cx+4:1]))
    peak_8x8_v = float(np.mean(magnitude[cy-4:cy+4:1, cx]))

    return {
        "low_freq_energy": round(low_energy, 4),
        "high_freq_energy": round(high_energy, 4),
        "hf_lf_ratio": round(high_energy / (low_energy + 1e-8), 4),
        "interpretation": "Low hf_lf_ratio → blur (loss of high frequencies). Regular peaks at 8-pixel intervals → JPEG compression."
    }


# ======================================================================
# MTF (Modulation Transfer Function) analysis for blur type classification
# ======================================================================

def compute_mtf_analysis(img):
    """Radial average of FFT power spectrum → MTF curve.

    The MTF (Modulation Transfer Function) describes how much contrast
    is preserved at each spatial frequency. Different blur types have
    distinct MTF shapes:
      - Gaussian: smooth exp(-f^2) rolloff
      - Motion: directional sinc oscillations
      - Lens/Defocus: jinc function with zeros

    Returns mtf50_freq and mtf10_freq — content-independent blur severity metrics.
    """
    gray = np.mean(img, axis=2)
    h, w = gray.shape
    f = np.fft.fftshift(np.fft.fft2(gray))
    ps = np.abs(f) ** 2

    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy)**2 + (x - cx)**2)

    # Radial binning: 50 bins from DC to Nyquist
    r_max = min(cy, cx)
    n_bins = 50
    bin_edges = np.linspace(0, r_max, n_bins + 1)
    mtf_curve = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (r >= bin_edges[i]) & (r < bin_edges[i+1])
        if mask.any():
            mtf_curve[i] = ps[mask].mean()

    # Normalize to DC=1
    mtf_curve = mtf_curve / (mtf_curve[0] + 1e-8)

    # Find MTF50 and MTF10 frequencies
    freq_axis = (bin_edges[:-1] + bin_edges[1:]) / 2 / r_max
    mtf50_freq = float(freq_axis[np.argmin(np.abs(mtf_curve - 0.5))])
    mtf10_freq = float(freq_axis[np.argmin(np.abs(mtf_curve - 0.1))])

    return {
        "mtf_curve": mtf_curve[:10].tolist(),  # first 10 bins for compactness
        "mtf50_freq": round(mtf50_freq, 4),
        "mtf10_freq": round(mtf10_freq, 4),
        "power_at_nyquist": round(float(mtf_curve[-1]), 4),
        "n_bins": n_bins,
    }


def compute_angular_fft(img):
    """Angular binning of FFT magnitude → directional blur detection.

    Motion blur creates a line of suppressed frequencies perpendicular
    to the motion direction. This function bins the FFT magnitude by
    angle to find the direction of minimum energy (the blur direction).
    Returns anisotropy_ratio: > 2.0 strongly suggests motion blur.
    """
    gray = np.mean(img, axis=2)
    h, w = gray.shape
    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.abs(f)

    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy)**2 + (x - cx)**2)
    theta = np.arctan2(y - cy, x - cx)  # [-π, π]

    # Exclude DC region (r < 5 pixels)
    valid = r > 5

    # Angular binning: 36 bins of 10 degrees each
    n_angular = 36
    theta_bins = np.linspace(-np.pi, np.pi, n_angular + 1)
    angular_energy = np.zeros(n_angular)
    for i in range(n_angular):
        mask = valid & (theta >= theta_bins[i]) & (theta < theta_bins[i+1])
        if mask.any():
            angular_energy[i] = mag[mask].mean()

    # Normalize
    angular_energy = angular_energy / (angular_energy.max() + 1e-8)

    # Anisotropy: ratio of max to min energy
    anisotropy_ratio = float(angular_energy.max() / (angular_energy.min() + 1e-8))

    # Dominant direction: angle of MINIMUM energy (blur suppresses frequencies
    # perpendicular to the motion direction)
    min_idx = int(np.argmin(angular_energy))
    dominant_direction = float(np.degrees(theta_bins[min_idx] + theta_bins[min_idx+1]) / 2)

    return {
        "angular_energy": angular_energy.tolist(),
        "anisotropy_ratio": round(anisotropy_ratio, 3),
        "dominant_direction_deg": round(dominant_direction, 1),
        "interpretation": f"anisotropy_ratio={anisotropy_ratio:.1f}: "
            f"{'Strongly suggests motion blur at ~' + str(round(dominant_direction)) + '°' if anisotropy_ratio > 2.0 else 'Isotropic blur (Gaussian/Lens/Glass)'}",
    }


def compute_ps_ratio(target, clean):
    """Power spectrum ratio: target / clean → isolate blur kernel frequency response.

    Only meaningful in same-image mode (target and clean are the same scene).
    Removes image content from the frequency analysis, leaving only the
    degradation's effect on the power spectrum.
    """
    t_gray = np.mean(target, axis=2)
    c_gray = np.mean(clean, axis=2)

    ft = np.fft.fftshift(np.fft.fft2(t_gray))
    fc = np.fft.fftshift(np.fft.fft2(c_gray))

    ps_t = np.abs(ft) ** 2
    ps_c = np.abs(fc) ** 2 + 1e-8  # avoid division by zero

    h, w = t_gray.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy)**2 + (x - cx)**2)

    # Radial average of ratio
    r_max = min(cy, cx)
    n_bins = 30
    bin_edges = np.linspace(0, r_max, n_bins + 1)
    ps_ratio_radial = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (r >= bin_edges[i]) & (r < bin_edges[i+1])
        if mask.any():
            ps_ratio_radial[i] = (ps_t[mask].mean() / ps_c[mask].mean())

    # MTF50 from the ratio
    freq_axis = (bin_edges[:-1] + bin_edges[1:]) / 2 / r_max
    if ps_ratio_radial.max() > 0:
        ps_ratio_radial_norm = ps_ratio_radial / ps_ratio_radial[0]
        mtf50_idx = np.argmin(np.abs(ps_ratio_radial_norm - 0.5))
        mtf50_from_ratio = float(freq_axis[mtf50_idx])
    else:
        mtf50_from_ratio = 0.0

    return {
        "ps_ratio_radial": ps_ratio_radial[:10].tolist(),
        "mtf50_from_ratio": round(mtf50_from_ratio, 4),
        "requires_same_image": True,
    }


def classify_blur_from_mtf(mtf_data, angular_data, grad_data, radial_data, clean_mtf=None):
    """Synthesize MTF + angular FFT + spatial metrics → blur type.

    Uses MTF curve shape comparison when clean reference MTF is available.
    Falls back to spatial metrics when clean reference is unavailable.

    Decision priority:
    1. anisotropy > 4.0 + h_v_ratio significantly ≠ 1.0 → motion blur
    2. MTF curve correlation with known blur types → gaussian/lens/zoom
    3. Spatial gradient metrics as tiebreaker
    """
    anisotropy = angular_data.get("anisotropy_ratio", 1.0) if angular_data else 1.0
    mtf50 = mtf_data.get("mtf50_freq", 0) if mtf_data else 0
    radial_ratio = radial_data.get("gradient_radial_ratio", 1.0) if radial_data else 1.0
    h_v_ratio = grad_data.get("directional_h_v_ratio", 1.0) if grad_data else 1.0
    mtf_curve = mtf_data.get("mtf_curve", []) if mtf_data else []

    candidates = []
    h_v_is_directional = abs(h_v_ratio - 1.0) > 0.3

    # 1. Motion blur: requires BOTH angular anisotropy AND directional h_v
    if anisotropy > 4.0 and h_v_is_directional:
        candidates.append({
            "type": "blur_motion",
            "confidence": "high" if anisotropy > 6.0 else "medium",
            "score": anisotropy,
            "evidence": f"Angular anisotropy={anisotropy:.1f} + h_v={h_v_ratio:.2f}",
        })

    # 2. Compare MTF curve with clean reference to isolate blur kernel
    if clean_mtf and mtf_curve:
        clean_curve = clean_mtf.get("mtf_curve", [])
        if clean_curve and len(clean_curve) == len(mtf_curve):
            # The blur kernel MTF ≈ target_MTF / clean_MTF
            kernel_mtf = [t / (c + 1e-8) for t, c in zip(mtf_curve, clean_curve)]
            # Normalize
            if kernel_mtf[0] > 0:
                kernel_mtf = [k / kernel_mtf[0] for k in kernel_mtf]

            # MTF curve shape heuristics:
            # - Gaussian: smooth monotonic decay, no zero crossings
            # - Lens/Defocus: faster initial drop then flattens (more low-freq attenuation)
            # - Zoom: similar to Gaussian but steeper at high freqs
            mid_val = kernel_mtf[len(kernel_mtf)//2] if len(kernel_mtf) > 2 else 0.5
            high_val = kernel_mtf[-1] if kernel_mtf else 0

            # Compute "gaussian-ness": how smoothly the curve decays
            # A pure Gaussian MTF follows exp(-f^2). Check if values
            # form a smooth convex curve.
            diffs = [kernel_mtf[i] - kernel_mtf[i+1] for i in range(len(kernel_mtf)-1)]
            is_smooth = all(d > 0 for d in diffs)  # monotonically decreasing

            if is_smooth:
                # Gaussian: gradual decay
                if mid_val > 0.3:
                    candidates.append({
                        "type": "blur_gaussian",
                        "confidence": "high" if high_val > 0.05 else "medium",
                        "score": mid_val + high_val,
                        "evidence": f"MTF kernel: smooth decay, mid={mid_val:.2f}, high={high_val:.2f}",
                    })
                # Lens/Defocus: fast initial drop
                else:
                    candidates.append({
                        "type": "blur_lens",
                        "confidence": "medium",
                        "score": 1.0 - mid_val,
                        "evidence": f"MTF kernel: fast drop, mid={mid_val:.2f}",
                    })

    # 3. If no MTF-based candidates, fall back to spatial metrics
    if not candidates:
        if radial_ratio > 1.5 and anisotropy < 4.0:
            candidates.append({
                "type": "blur_lens", "confidence": "low",
                "score": radial_ratio,
                "evidence": f"Radial ratio={radial_ratio:.1f} (spatial only, low confidence)",
            })
        candidates.append({
            "type": "blur_gaussian", "confidence": "low",
            "score": 0.5,
            "evidence": f"Default isotropic (aniso={anisotropy:.1f}, h_v={h_v_ratio:.2f})",
        })

    # Best guess: highest confidence or highest score
    candidates.sort(key=lambda c: (c["confidence"] == "high", c.get("score", 0)), reverse=True)

    # Severity from MTF50
    severity_hint = None
    if mtf50 > 0:
        if mtf50 > 0.8: severity_hint = 1
        elif mtf50 > 0.5: severity_hint = 2
        elif mtf50 > 0.3: severity_hint = 3
        elif mtf50 > 0.15: severity_hint = 4
        else: severity_hint = 5

    return {
        "candidates": candidates[:3],
        "best_guess": candidates[0]["type"] if candidates else "unknown",
        "anisotropy_ratio": round(anisotropy, 3),
        "mtf50_freq": round(mtf50, 4),
        "severity_hint": severity_hint,
        "method": "mtf_comparison" if clean_mtf else "spatial_fallback",
    }


def analyze(target_path, clean_path=None):
    """Run full degradation analysis."""
    target = np.array(Image.open(target_path).convert("RGB"), dtype=np.float32)

    report = {
        "target_image": target_path,
        "target_size": list(Image.open(target_path).size),
        "file_info": compute_file_info(target_path),
        "basic_stats": compute_basic_stats(target),
        "gradient": compute_gradient_analysis(target),
        "noise": compute_noise_analysis(target),
        "compression": compute_compression_analysis(target),
        "ycrcb": compute_ycrcb_analysis(target),
        "cross_channel": compute_cross_channel_correlation(target),
        "multiscale": compute_multiscale_analysis(target),
        "frequency": compute_frequency_analysis(target),
        "sharpening": compute_sharpening_analysis(target),
        "saturation": compute_saturation_analysis(target),
        "radial": compute_radial_analysis(target),
        "noise_vs_sharpen": compute_noise_vs_sharpen_analysis(target),
        "jpeg2000": compute_jpeg2000_analysis(target),
        "noise_subtype": compute_noise_subtype_analysis(target),
        "mtf": compute_mtf_analysis(target),
        "angular_fft": compute_angular_fft(target),
    }

    if clean_path:
        clean = np.array(Image.open(clean_path).convert("RGB"), dtype=np.float32)
        clean_report = {
            "basic_stats": compute_basic_stats(clean),
            "gradient": compute_gradient_analysis(clean),
            "noise": compute_noise_analysis(clean),
            "compression": compute_compression_analysis(clean),
        }

        # Compute ratios (target/clean) for key metrics
        report["ratios_vs_clean"] = {
            "gradient_magnitude_ratio": round(report["gradient"]["gradient_magnitude_mean"] /
                                              (clean_report["gradient"]["gradient_magnitude_mean"] + 1e-8), 4),
            "laplacian_variance_ratio": round(report["gradient"]["laplacian_variance"] /
                                              (clean_report["gradient"]["laplacian_variance"] + 1e-8), 4),
            "hf_energy_preserved": "requires frequency comparison with clean",
            "unique_colors_change": {
                "target_unique_RGB": report["compression"]["unique_RGB_estimate"],
                "clean_unique_RGB": clean_report["compression"]["unique_RGB_estimate"],
            },
        }
        # PS ratio (requires same-image)
        report["ps_ratio"] = compute_ps_ratio(target, clean)
        report["mtf_clean"] = compute_mtf_analysis(clean)
        report["clean_reference"] = clean_report

    # Add degradation summary with content-independence classification
    report["degradation_summary"] = generate_degradation_summary(report)

    # Blur subtype analysis — now MTF-enhanced: combines spatial + frequency
    report["blur_subtype"] = compute_blur_subtype_analysis(
        target,
        report["gradient"]["directional_h_v_ratio"],
        report["radial"]["gradient_radial_ratio"]
    )
    # MTF-based blur classification
    report["blur_mtf"] = classify_blur_from_mtf(
        report.get("mtf", {}),
        report.get("angular_fft", {}),
        report.get("gradient", {}),
        report.get("radial", {}),
        clean_mtf=report.get("mtf_clean", None) if clean_path else None,
    )
    if clean_path:
        report["blur_subtype"]["directional_h_v_ratio_clean"] = round(float(clean_report["gradient"]["directional_h_v_ratio"]), 4)
        report["blur_subtype"]["directional_change_pct"] = round(
            abs(report["gradient"]["directional_h_v_ratio"] - clean_report["gradient"]["directional_h_v_ratio"])
            / (clean_report["gradient"]["directional_h_v_ratio"] + 1e-8) * 100, 2
        )

    return report


def generate_degradation_summary(report):
    """Generate a structured degradation summary classifying findings by evidence strength.
    Separates content-independent (trustworthy) from content-dependent (needs calibration) evidence."""

    summary = {
        "degradations_detected": [],
        "degradations_ruled_out": [],
        "metric_classification": {
            "content_independent": {
                "description": "These metrics detect structural anomalies — trust them for degradation TYPE identification even in cross-image scenarios.",
                "findings": []
            },
            "content_dependent": {
                "description": "These metrics vary with image content — DO NOT use alone to claim a degradation is present. Use only for severity calibration after type is confirmed.",
                "findings": []
            }
        },
        "recommended_investigation_order": []
    }

    # --- Content-INDEPENDENT checks ---
    # These detect structural anomalies no natural image exhibits, regardless of content.
    ci = summary["metric_classification"]["content_independent"]["findings"]

    # Impulse noise: pixel extremes — no natural image has clusters of 0/255
    # Caveat: heavy JPEG at low quality can also push pixels to 0/255, causing false positives
    imp = report["noise"]["impulse_total_pct"]
    if imp > 0.5:
        ci.append({"metric": "impulse_total_pct", "value": imp, "flag": "IMPULSE_NOISE",
                   "reason": f"{imp:.2f}% of pixels at 0/255. Salt-and-pepper noise detected (>0.5% threshold). CAVEAT: heavy JPEG compression can also create extreme pixels — verify with visual inspection: impulse noise shows isolated random dots, JPEG extremes appear in block-aligned clusters."})
    else:
        ci.append({"metric": "impulse_total_pct", "value": imp, "flag": "NO_IMPULSE",
                   "reason": f"{imp:.2f}% pixels at extremes. No impulse noise detected."})

    # JPEG: block boundary ratio — natural images don't have 8×8 DCT block boundaries
    bbr = report["compression"]["block_boundary_ratio"]
    if bbr > 1.1:
        ci.append({"metric": "block_boundary_ratio", "value": bbr, "flag": "JPEG_COMPRESSION",
                   "reason": f"8×8 block boundary ratio={bbr:.2f} > 1.1. JPEG compression confirmed."})
    else:
        ci.append({"metric": "block_boundary_ratio", "value": bbr, "flag": "NO_JPEG_BLOCKS",
                   "reason": f"Block boundary ratio={bbr:.2f} ≤ 1.1. No JPEG 8×8 blocking detected."})

    # JPEG via G-channel reduction: JPEG uses YCrCb where G dominates Y
    # G-only reduction with full R/B is JPEG signature. All-channel reduction = blur or quantization.
    uR = report["compression"]["unique_R"]
    uG = report["compression"]["unique_G"]
    uB = report["compression"]["unique_B"]
    if uG < 250 and uR >= 250 and uB >= 250:
        ci.append({"metric": "unique_G_selective", "value": uG, "flag": "JPEG_G_REDUCTION",
                   "reason": f"G channel reduced to {uG}/256 while R={uR} B={uB} remain full. JPEG chroma subsampling signature."})
    elif uG < 256 or uR < 256 or uB < 256:
        # All-channel or uncertain reduction — note it but don't claim JPEG
        ci.append({"metric": "unique_colors_reduced", "value": f"R={uR} G={uG} B={uB}", "flag": "PALETTE_REDUCTION",
                   "reason": f"Color palette reduced (R={uR} G={uG} B={uB}). Could be JPEG (check block_boundary_ratio), quantization (all channels < 32), or blur side-effect (all channels reduced similarly)."})

    # Oversharpen: overshoot ratio
    osr = report["sharpening"]["overshoot_ratio"]
    if osr > 0.5:
        ci.append({"metric": "overshoot_ratio", "value": osr, "flag": "OVERSHARPEN",
                   "reason": f"Laplacian overshoot ratio={osr:.2f} > 0.5. Edge halos likely."})

    # Pixelate: multi-scale variance spike — natural images have smooth roll-off
    ms = report["multiscale"]["scales"]
    if len(ms) >= 3:
        vars_list = [s["residual_variance"] for s in ms]
        max_jump = max([vars_list[i] / (vars_list[i-1] + 1e-8) for i in range(1, len(vars_list))])
        if max_jump > 5.0:
            ci.append({"metric": "multiscale_variance_jump", "value": round(max_jump, 2), "flag": "PIXELATE",
                       "reason": f"Abrupt {max_jump:.1f}x variance jump at specific scale. Pixelation likely."})

    # Compression level from file size
    bpp = report["file_info"]["bytes_per_pixel"]
    if bpp < 0.8:
        ci.append({"metric": "bytes_per_pixel", "value": bpp, "flag": "HEAVY_COMPRESSION",
                   "reason": f"{bpp:.2f} BPP indicates strong compression."})

    # --- Content-DEPENDENT checks ---
    cd = summary["metric_classification"]["content_dependent"]["findings"]

    gmm = report["gradient"]["gradient_magnitude_mean"]
    cd.append({"metric": "gradient_magnitude_mean", "value": gmm,
               "warning": "Absolute sharpness — varies with image content. Do NOT use to claim blur without clean reference."})

    fvar = report["noise"]["flat_region_variance"]
    cd.append({"metric": "flat_region_variance", "value": fvar,
               "warning": "DANGER: High values may be natural texture, not noise. Never claim Gaussian noise from this alone."})

    sm = report["saturation"]["saturation_mean"]
    cd.append({"metric": "saturation_mean", "value": sm,
               "warning": "Varies with scene content. Colorful photo ≠ saturation boost."})

    rr = report["radial"]["gradient_radial_ratio"]
    cd.append({"metric": "gradient_radial_ratio", "value": rr,
               "warning": "Center-composed images naturally have higher center sharpness. Not necessarily lens blur."})

    h_v = report["gradient"]["directional_h_v_ratio"]
    cd.append({"metric": "directional_h_v_ratio", "value": h_v,
               "warning": "Natural images range from 0.5 to 1.5. Only use for motion blur detection when comparing target vs clean of the SAME image. Cross-image: rely on visual inspection for directional smear."})

    # --- Build detected degradations ---
    # Only include content-independent detections here.
    # Blur type (motion vs isotropic) is content-dependent — determined by visual inspection, not metrics alone.
    flags = {f["flag"] for f in ci}
    if "IMPULSE_NOISE" in flags:
        summary["degradations_detected"].append({"type": "noise_impulse", "confidence": "high",
                                                  "evidence": "content-independent: pixel extremes > 0.5%"})
    if "JPEG_COMPRESSION" in flags:
        summary["degradations_detected"].append({"type": "compression_jpeg", "confidence": "high",
                                                  "evidence": "content-independent: 8×8 block boundary ratio > 1.1"})
    if "JPEG_G_REDUCTION" in flags:
        summary["degradations_detected"].append({"type": "compression_jpeg", "confidence": "high",
                                                  "evidence": "content-independent: G-channel selective reduction (JPEG chroma subsampling)"})
    if "PALETTE_REDUCTION" in flags and "JPEG_COMPRESSION" not in flags and "JPEG_G_REDUCTION" not in flags:
        summary["degradations_detected"].append({"type": "palette_reduction", "confidence": "low",
                                                  "evidence": "Color palette reduced. Could be quantization (if severe: all < 32), blur side-effect (all channels similarly reduced), or mild JPEG2000. Cannot determine type from this alone."})
    if "OVERSHARPEN" in flags:
        summary["degradations_detected"].append({"type": "oversharpen", "confidence": "high",
                                                  "evidence": "content-independent: Laplacian overshoot ratio > 0.5"})
    if "PIXELATE" in flags:
        summary["degradations_detected"].append({"type": "pixelate", "confidence": "high",
                                                  "evidence": "content-independent: multi-scale variance spike > 5x"})

    # --- Build ruled out ---
    if "NO_IMPULSE" in flags:
        summary["degradations_ruled_out"].append({"type": "noise_impulse", "reason": "impulse_total_pct below 0.5% threshold"})
    if "NO_JPEG_BLOCKS" in flags:
        summary["degradations_ruled_out"].append({"type": "compression_jpeg (blocking)", "reason": "no 8×8 block boundaries detected"})

    # --- Recommended investigation order ---
    summary["recommended_investigation_order"] = [
        "1. Start with content-independent detections above — these are trustworthy regardless of image content.",
        "2. For each detected type, use visual inspection to confirm before simulating.",
        "3. For severity: content-dependent metrics can guide initial guess, but visual comparison is the final arbiter.",
        "4. If clean reference exists (even different content): compare content-independent metrics between target and clean to verify.",
        "5. NEVER claim a degradation based solely on content-dependent metric values."
    ]

    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze degraded image for distortion identification")
    parser.add_argument("--target", required=True, help="Path to degraded target image")
    parser.add_argument("--clean", default=None, help="Path to clean reference image (optional)")
    parser.add_argument("--output", default=None, help="Output JSON path (prints to stdout if not specified)")
    args = parser.parse_args()

    report = analyze(args.target, args.clean)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Analysis saved to: {args.output}")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
