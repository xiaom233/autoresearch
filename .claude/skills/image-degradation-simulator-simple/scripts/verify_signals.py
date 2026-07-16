#!/usr/bin/env python3
"""
信号级验证脚本 — 对比 target 和 simulated 的诊断信号，替代 PSNR。

当 noise 存在时 (sigma > 2)，PSNR 因 seed 失配而不可靠。
此脚本对比 run_full_analysis.py 使用的同一套诊断信号，
判断 simulated 是否在信号层面匹配 target。

用法:
  .venv/bin/python3 verify_signals.py \
    --target <degraded.png> --simulated <simulated.png> --clean <clean.png> \
    --pipeline "blur_gaussian:3,noise_gaussian_RGB:2"

输出:
  - 每类退化的信号对比表 (target值 vs simulated值 vs 差值 vs 阈值)
  - 整体判定: MATCH / PARTIAL / MISMATCH
  - Agent 可直接读取的文字摘要
"""

import argparse, json, os, sys, warnings, numpy as np
from PIL import Image
from collections import OrderedDict

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', '..', '..', '..'))

# ── Reuse signal extraction from run_full_analysis ──
from run_full_analysis import (
    compute_block_boundary,
    compute_dct_zero_ratio,
    compute_gm_ratio,
    compute_spectral_slope,
    estimate_noise_sigma,
    compute_residual,
)


def load_gray(path):
    img = np.array(Image.open(path).convert('RGB'), dtype=np.float64)
    return img.mean(axis=2)


def load_rgb(path):
    return np.array(Image.open(path).convert('RGB'), dtype=np.float64)


def compute_variance_ratio(target, clean):
    t_var = target.var()
    c_var = clean.var()
    return t_var / max(c_var, 1)


def compute_mean_shift(target, clean):
    return (target.mean() - clean.mean()) / 255.0


def compute_chroma_ratio(target, clean):
    def rgb_to_crcb(img):
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        cb = 128 - 0.168736*r - 0.331264*g + 0.5*b
        cr = 128 + 0.5*r - 0.418688*g - 0.081312*b
        return cr, cb
    cr_t, cb_t = rgb_to_crcb(target)
    cr_c, cb_c = rgb_to_crcb(clean)
    return (cr_t.std() + cb_t.std()) / max(cr_c.std() + cb_c.std(), 1)


def compute_wiener_kernel_corr(target_gray, clean_gray):
    """提取 target 和 simulated 的 Wiener 核，计算相关系数。
    返回 (corr, target_kernel, sim_kernel)。"""
    try:
        from blur_kernel_recovery import estimate_kernel_gradient_joint
        h_target = estimate_kernel_gradient_joint(target_gray, clean_gray)
        h_sim = estimate_kernel_gradient_joint(target_gray, clean_gray)  # will be recomputed
        # Actually: extract kernel from target vs clean, and from target vs clean
        # Both use the same clean as reference
        k_t = estimate_kernel_gradient_joint(target_gray, clean_gray, crop_size=31)
        k_t_flat = k_t.ravel()
        # For simulated, we need clean → pipeline → simulated
        # But we already have the simulated image passed in
        return float(np.corrcoef(k_t_flat, k_t_flat)[0,1]), k_t, k_t  # placeholder
    except Exception as e:
        return 0.0, None, None


def compute_wiener_kernel_corr_pair(target_gray, simulated_gray, clean_gray):
    """提取 target 的核和 simulated 的核，计算相关系数。"""
    try:
        from blur_kernel_recovery import estimate_kernel_gradient_joint
        k_target = estimate_kernel_gradient_joint(target_gray, clean_gray, crop_size=31)
        k_sim = estimate_kernel_gradient_joint(simulated_gray, clean_gray, crop_size=31)
        k_t = k_target.ravel()
        k_s = k_sim.ravel()
        corr = np.corrcoef(k_t, k_s)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0
    except Exception as e:
        return 0.0


def compute_residual_autocorr_fwhm(residual_gray):
    """残差自相关峰的半高宽。纯噪声应 < 3px，有结构则 > 5px。"""
    try:
        r = residual_gray - residual_gray.mean()
        # 2D autocorrelation via FFT
        fft = np.fft.fft2(r)
        psd = np.abs(fft) ** 2
        ac = np.fft.fftshift(np.real(np.fft.ifft2(psd)))
        ac /= ac.max()
        # Find FWHM along center row
        cy = ac.shape[0] // 2
        row = ac[cy, :]
        above_half = row > 0.5
        indices = np.where(above_half)[0]
        if len(indices) >= 2:
            return float(indices[-1] - indices[0])
        return 0.0
    except Exception:
        return 0.0


def compute_sb6_on_residual(residual):
    """在残差上计算 §B 6步值。residual = target - simulated (或 peeled)。"""
    res = residual.astype(np.float64)
    h, w = res.shape[:2]
    ph, pw = 16, 16

    # extreme_pct
    extreme = np.sum((res <= 1) | (res >= 254)) / res.size
    zero_ratio = np.mean(np.abs(res) < 1e-6)

    # vm_slope
    means_list, vars_list = [], []
    for y in range(0, h - ph, ph):
        for x in range(0, w - pw, pw):
            patch = res[y:y+ph, x:x+pw].reshape(-1)
            means_list.append(np.mean(patch))
            vars_list.append(np.var(patch))
    means_arr = np.array(means_list)
    vars_arr = np.array(vars_list)
    mean_sq = means_arr ** 2
    vm_slope = np.cov(mean_sq, vars_arr)[0, 1] / max(np.var(mean_sq), 1e-10) if len(means_arr) > 5 else 0

    # var_slope
    r_gray = res.mean(axis=2) if len(res.shape) == 3 else res
    flat = r_gray.ravel()
    bins = np.linspace(flat.min(), flat.max(), 20)
    bin_means, bin_vars = [], []
    for j in range(len(bins) - 1):
        mask = (flat >= bins[j]) & (flat < bins[j+1])
        if mask.sum() > 100:
            bin_means.append(flat[mask].mean())
            ch_vars = [res[:,:,c].ravel()[mask].var() for c in range(min(3, res.shape[2]))] if len(res.shape) == 3 else [res.ravel()[mask].var()]
            bin_vars.append(np.mean(ch_vars))
    var_slope = float(np.polyfit(bin_means, bin_vars, 1)[0]) if len(bin_means) > 2 else 0

    # spatial_corr
    r2d = res.mean(axis=2) if len(res.shape) == 3 else res
    r2d_flat = r2d.ravel()
    # simple neighbor correlation
    if r2d.shape[1] > 1:
        sc_h = np.corrcoef(r2d[:, :-1].ravel(), r2d[:, 1:].ravel())[0, 1]
    else:
        sc_h = 0
    if r2d.shape[0] > 1:
        sc_v = np.corrcoef(r2d[:-1, :].ravel(), r2d[1:, :].ravel())[0, 1]
    else:
        sc_v = 0
    spatial_corr = (sc_h + sc_v) / 2.0

    # rgb_ratio and cross_ch_corr
    if len(res.shape) == 3 and res.shape[2] >= 3:
        ch_stds = [res[:,:,i].std() for i in range(3)]
        rgb_ratio = max(ch_stds) / max(min(ch_stds), 1e-6)
        ch_flat = [res[:,:,i].ravel() for i in range(3)]
        ccc = max(abs(np.corrcoef(ch_flat[i], ch_flat[j])[0,1]) for i in range(3) for j in range(i+1,3))
    else:
        ch_stds = [res.std()]*3
        rgb_ratio = 1.0
        ccc = 0

    return {
        "extreme_pct": round(float(extreme * 100), 2),
        "zero_ratio": round(float(zero_ratio), 4),
        "vm_slope": round(float(vm_slope), 4),
        "var_slope": round(float(var_slope), 4),
        "spatial_corr": round(float(spatial_corr), 4),
        "rgb_ratio": round(float(rgb_ratio), 4),
        "cross_channel_corr": round(float(ccc), 4),
        "ch_stds": [round(float(s), 1) for s in ch_stds],
        "residual_std": round(float(res.std()), 1),
    }


# ── Thresholds (calibrated from 3840 synthetic cases) ──
THRESHOLDS = {
    "blur": {
        "spectral_slope_diff": {"max": 0.08, "desc": "噪声偏移仅±0.03"},
        "wiener_kernel_corr": {"min": 0.85, "desc": "核形状应高度相关"},
        "gm_ratio_diff_pct": {"max": 15, "desc": "仅低噪声时可用"},
    },
    "noise": {
        "sigma_diff_pct": {"max": 20, "desc": "噪声强度匹配"},
        "sb6_extreme_pct_diff": {"max": 5, "desc": "impulse 信号"},
        "sb6_vm_slope_diff": {"max": 0.01, "desc": "speckle 信号"},
        "sb6_var_slope_diff": {"max": 0.5, "desc": "poisson 信号"},
        "sb6_spatial_corr_diff": {"max": 0.1, "desc": "SC 噪声信号"},
        "sb6_rgb_ratio_diff": {"max": 0.3, "desc": "YCrCb 信号"},
        "residual_autocorr_fwhm": {"max": 4, "desc": "残差应无结构 (<3px=纯噪声)"},
    },
    "compression": {
        "block_boundary_diff_pct": {"max": 8, "desc": "噪声同向衰减8×8块"},
        "dct_zero_ratio_diff": {"max": 0.08, "desc": "DCT零系数比例"},
    },
    "global": {
        "mean_shift_diff": {"max": 0.005, "desc": "噪声均值为零 (0-1 scale)"},
        "variance_ratio_diff_pct": {"max": 8, "desc": "方差比匹配"},
        "chroma_ratio_diff_pct": {"max": 8, "desc": "色度比匹配"},
    },
}


def verify(target_path, simulated_path, clean_path, pipeline_str):
    """主验证逻辑。返回 (verdict, details_dict)。"""
    target = load_rgb(target_path)
    simulated = load_rgb(simulated_path)
    clean = load_rgb(clean_path)

    # Ensure same size
    if target.shape != simulated.shape:
        simulated = np.array(Image.open(simulated_path).convert('RGB').resize(
            (target.shape[1], target.shape[0]), Image.LANCZOS), dtype=np.float64)

    target_gray = target.mean(axis=2)
    simulated_gray = simulated.mean(axis=2)
    clean_gray = clean.mean(axis=2)

    # Parse pipeline
    steps = []
    for s in pipeline_str.split(","):
        s = s.strip()
        if ":" in s:
            fn, sev = s.rsplit(":", 1)
            steps.append({"function": fn, "severity": int(sev)})
    has_blur = any(s["function"].startswith("blur_") for s in steps)
    has_noise = any(s["function"].startswith("noise_") for s in steps)
    has_compression = any(s["function"].startswith("compression_") for s in steps)
    has_global = any(s["function"].startswith(("brightness_", "contrast_", "saturate_", "quantization_")) for s in steps)

    results = OrderedDict()
    matches = []
    mismatches = []

    # ── Blur signals ──
    if has_blur:
        blur = OrderedDict()
        # spectral_slope
        slope_t = compute_spectral_slope(target)
        slope_s = compute_spectral_slope(simulated)
        slope_diff = abs(slope_t - slope_s)
        blur["spectral_slope"] = {"target": round(slope_t, 4), "simulated": round(slope_s, 4),
                                   "diff": round(slope_diff, 4), "threshold": "<0.08",
                                   "match": slope_diff < 0.08}
        if blur["spectral_slope"]["match"]:
            matches.append("blur:spectral_slope")
        else:
            mismatches.append(f"blur:spectral_slope (diff={slope_diff:.3f})")

        # Wiener kernel correlation
        kernel_corr = compute_wiener_kernel_corr_pair(target_gray, simulated_gray, clean_gray)
        blur["wiener_kernel_corr"] = {"value": round(kernel_corr, 4), "threshold": ">0.85",
                                       "match": kernel_corr > 0.85}
        if blur["wiener_kernel_corr"]["match"]:
            matches.append("blur:wiener_kernel")
        else:
            mismatches.append(f"blur:wiener_kernel (corr={kernel_corr:.3f})")

        results["blur"] = blur

    # ── Noise signals ──
    if has_noise:
        noise = OrderedDict()
        # noise_prior sigma
        sigma_t = np.mean(estimate_noise_sigma(target))
        sigma_s = np.mean(estimate_noise_sigma(simulated))
        sigma_diff_pct = abs(sigma_t - sigma_s) / max(sigma_t, 0.1) * 100
        noise["sigma"] = {"target": round(float(sigma_t), 1), "simulated": round(float(sigma_s), 1),
                          "diff_pct": round(sigma_diff_pct, 1), "threshold": "<20%",
                          "match": sigma_diff_pct < 20}
        if noise["sigma"]["match"]:
            matches.append("noise:sigma")
        else:
            mismatches.append(f"noise:sigma (diff={sigma_diff_pct:.0f}%)")

        # §B on target residual (target - simulated = noise residual)
        noise_residual = target - simulated
        sb6 = compute_sb6_on_residual(noise_residual)

        # The §B values on residual should show NO structure if pipeline is correct
        # For pure noise residual: extreme_pct should be small (no JPEG artifacts)
        # spatial_corr should be small (no blur artifacts)
        noise["sb6_extreme_pct"] = {"value": sb6["extreme_pct"],
                                     "note": "噪声残差应无极端值(无JPEG残留)",
                                     "match": sb6["extreme_pct"] < 5}
        noise["sb6_spatial_corr"] = {"value": sb6["spatial_corr"],
                                      "note": "噪声残差应无空间相关(无blur残留)",
                                      "match": sb6["spatial_corr"] < 0.15}
        noise["sb6_rgb_ratio"] = {"value": sb6["rgb_ratio"],
                                   "note": ">1.4=YCrCb噪声, <1.3=RGB噪声"}

        # Residual autocorr
        residual_gray = noise_residual.mean(axis=2) if len(noise_residual.shape) == 3 else noise_residual
        fwhm = compute_residual_autocorr_fwhm(residual_gray)
        noise["residual_autocorr_fwhm"] = {"value": round(fwhm, 1), "threshold": "<4px",
                                            "note": "<3px=纯噪声, >5px=有结构残留",
                                            "match": fwhm < 4}
        if noise["residual_autocorr_fwhm"]["match"]:
            matches.append("noise:clean_residual")
        else:
            mismatches.append(f"noise:structured_residual (FWHM={fwhm:.1f}px)")

        results["noise"] = noise

    # ── Compression signals ──
    if has_compression:
        comp = OrderedDict()
        bb_t = compute_block_boundary(target)
        bb_s = compute_block_boundary(simulated)
        bb_diff_pct = abs(bb_t - bb_s) / max(bb_t, 0.01) * 100
        comp["block_boundary"] = {"target": round(bb_t, 4), "simulated": round(bb_s, 4),
                                   "diff_pct": round(bb_diff_pct, 1), "threshold": "<8%",
                                   "match": bb_diff_pct < 8}
        if comp["block_boundary"]["match"]:
            matches.append("compression:block_boundary")
        else:
            mismatches.append(f"compression:block_boundary (diff={bb_diff_pct:.0f}%)")

        # DCT zero ratio
        dct_t = compute_dct_zero_ratio(target_gray)
        dct_s = compute_dct_zero_ratio(simulated_gray)
        dct_diff = abs(dct_t - dct_s)
        comp["dct_zero_ratio"] = {"target": round(dct_t, 4), "simulated": round(dct_s, 4),
                                   "diff": round(dct_diff, 4), "threshold": "<0.08",
                                   "match": dct_diff < 0.08}
        results["compression"] = comp

    # ── Global signals ──
    if has_global:
        glbl = OrderedDict()
        ms_t = compute_mean_shift(target, clean)
        ms_s = compute_mean_shift(simulated, clean)
        ms_diff = abs(ms_t - ms_s)
        glbl["mean_shift"] = {"target": round(ms_t, 4), "simulated": round(ms_s, 4),
                              "diff": round(ms_diff, 4), "threshold": "<0.005 (0-1 scale)",
                              "match": ms_diff < 0.005}
        if glbl["mean_shift"]["match"]:
            matches.append("global:mean_shift")
        else:
            mismatches.append(f"global:mean_shift (diff={ms_diff:.4f})")

        vr_t = compute_variance_ratio(target, clean)
        vr_s = compute_variance_ratio(simulated, clean)
        vr_diff_pct = abs(vr_t - vr_s) / max(abs(vr_t - 1), 0.01) * 100 if abs(vr_t - 1) > 0.01 else abs(vr_t - vr_s) * 100
        glbl["variance_ratio"] = {"target": round(vr_t, 4), "simulated": round(vr_s, 4),
                                   "diff_pct": round(vr_diff_pct, 1), "threshold": "<8%",
                                   "match": vr_diff_pct < 8}
        if glbl["variance_ratio"]["match"]:
            matches.append("global:variance_ratio")
        else:
            mismatches.append(f"global:variance_ratio (diff={vr_diff_pct:.0f}%)")

        results["global"] = glbl

    # ── Overall verdict ──
    total_checks = len(matches) + len(mismatches)
    if total_checks == 0:
        verdict = "NO_CHECKS"
    elif len(mismatches) == 0:
        verdict = "MATCH"
    elif len(matches) >= len(mismatches) * 2:
        verdict = "PARTIAL"
    elif len(matches) > 0:
        verdict = "WEAK"
    else:
        verdict = "MISMATCH"

    return verdict, matches, mismatches, results


def main():
    parser = argparse.ArgumentParser(description="Signal-level verification (替代 PSNR)")
    parser.add_argument("--target", required=True, help="Degraded target image")
    parser.add_argument("--simulated", required=True, help="Simulated degradation image")
    parser.add_argument("--clean", required=True, help="Clean reference image")
    parser.add_argument("--pipeline", required=True, help="Pipeline string, e.g. 'blur_gaussian:3,noise_gaussian_RGB:2'")
    parser.add_argument("--output", help="Save JSON output")
    parser.add_argument("--json", action="store_true", help="JSON to stdout")
    args = parser.parse_args()

    verdict, matches, mismatches, results = verify(
        args.target, args.simulated, args.clean, args.pipeline)

    # ── Text summary for Agent ──
    match_count = len(matches)
    mismatch_count = len(mismatches)
    print(f"""
{'='*60}
🔴 SIGNAL VERIFICATION (not PSNR)
Pipeline: {args.pipeline}
Verdict:  {verdict} ({match_count}✓ / {mismatch_count}✗)
{'='*60}
""")
    if matches:
        print("  ✅ Matched signals:")
        for m in matches:
            print(f"     {m}")
    if mismatches:
        print("  ❌ Mismatched signals:")
        for m in mismatches:
            print(f"     {m}")
    print(f"\n  Verdict: {verdict}")
    print(f"  → MATCH:      所有信号匹配 → 退化参数正确")
    print(f"  → PARTIAL:    多数信号匹配 → 类型正确,severity可能需微调")
    print(f"  → WEAK:       少数信号匹配 → 类型可能错误")
    print(f"  → MISMATCH:   无信号匹配 → 退化参数错误,需重做")
    print(f"{'='*60}\n")

    # ── JSON output ──
    output_data = {
        "pipeline": args.pipeline,
        "verdict": verdict,
        "matches": matches,
        "mismatches": mismatches,
        "signals": results,
    }
    json_str = json.dumps(output_data, indent=2, ensure_ascii=False, default=str)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(json_str)
    if args.json:
        print(json_str)


if __name__ == "__main__":
    main()
