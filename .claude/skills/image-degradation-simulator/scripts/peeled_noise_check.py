#!/usr/bin/env python3
"""
剥离后噪声诊断 — 在剥离确定性退化后的残差上运行 §B 6步检查。
用法:
  python peeled_noise_check.py --target <degraded.png> --clean <clean.png> \
      --pipeline "compression_jpeg:3,blur_gaussian:2"

  输出: 格式化的 §B 6步数值表 + noise_prior 估计 + 噪声类型建议
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
import numpy as np
from PIL import Image

# ── helpers ──
def load(path):
    return np.array(Image.open(path).convert("RGB"), dtype=np.float64)

def save(arr, path):
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(path)

def apply_pipeline(clean, pipeline_str):
    """正向施加退化管线"""
    from x_distortion import add_distortion
    img = clean.copy().astype(np.uint8)
    for step in pipeline_str.split(","):
        step = step.strip()
        if not step: continue
        fn, sev = step.rsplit(":", 1)
        img = add_distortion(img, severity=int(sev), distortion_name=fn)
    return img.astype(np.float64)

# ── §B 6-step diagnostic ──
def compute_sb6(residual):
    """在纯噪声残差上计算 §B 6步检查值。residual = target - simulated_deterministic"""
    res = residual.astype(np.float64)
    h, w, c = res.shape

    # 1. impulse: extreme pixel fraction (exact 0 or 255)
    extreme = np.sum((res <= 0) | (res >= 255)) / res.size
    # Also check zero_ratio for v16-style impulse detection
    zero_ratio = np.mean(np.abs(res) < 1e-6)

    # 2. speckle: variance vs mean-squared slope (vm_slope)
    # Divide into 16x16 patches
    ph, pw = 16, 16
    means, vars_ = [], []
    for y in range(0, h - ph, ph):
        for x in range(0, w - pw, pw):
            patch = res[y:y+ph, x:x+pw].reshape(-1)
            means.append(np.mean(patch))
            vars_.append(np.var(patch))
    means, vars_ = np.array(means), np.array(vars_)
    # vm_slope = cov(mean^2, var) / var(mean^2)
    mean_sq = means ** 2
    vm_slope = np.cov(mean_sq, vars_)[0, 1] / max(np.var(mean_sq), 1e-10)

    # 3. poisson: variance vs mean slope (var_slope)
    # Bin pixels by intensity, compute variance per bin
    r_gray = res.mean(axis=2)
    flat_gray = r_gray.ravel()
    bins = np.linspace(flat_gray.min(), flat_gray.max(), 20)
    bin_means, bin_vars = [], []
    for j in range(len(bins) - 1):
        mask = (flat_gray >= bins[j]) & (flat_gray < bins[j+1])
        if mask.sum() > 100:
            bin_means.append(flat_gray[mask].mean())
            # Compute variance per channel on masked pixels, then average
            ch_vars = [res[:,:,c].ravel()[mask].var() for c in range(3)]
            bin_vars.append(np.mean(ch_vars))
    if len(bin_means) > 2:
        var_slope = np.polyfit(bin_means, bin_vars, 1)[0]
    else:
        var_slope = 0

    # Also intensity-variance correlation (v16 style) - per channel average
    ivc_vals = [np.corrcoef(flat_gray, res[:,:,c].ravel())[0,1] for c in range(3)]
    ivc = np.mean([v for v in ivc_vals if not np.isnan(v)]) if ivc_vals else 0

    # 4. spatial correlation
    from scipy.ndimage import convolve
    r_gray = res.mean(axis=2)
    neighbor = convolve(r_gray, np.ones((3,3))/9)
    spatial_corr = np.corrcoef(r_gray.flat, neighbor.flat)[0, 1]

    # 5. YCrCb: RGB channel std ratio
    ch_stds = [res[:,:,i].std() for i in range(3)]
    rgb_ratio = max(ch_stds) / max(min(ch_stds), 1e-6)

    # 6. cross-channel correlation
    ch_flat = [res[:,:,i].ravel() for i in range(3)]
    cross_ch_corr = max(abs(np.corrcoef(ch_flat[i], ch_flat[j])[0,1])
                       for i in range(3) for j in range(i+1,3))

    return {
        "extreme_pct": round(float(extreme * 100), 2),
        "zero_ratio": round(float(zero_ratio), 4),
        "vm_slope": round(float(vm_slope), 4),
        "var_slope": round(float(var_slope), 4),
        "intensity_var_corr": round(float(ivc), 4),
        "spatial_corr": round(float(spatial_corr), 4),
        "rgb_ratio": round(float(rgb_ratio), 4),
        "cross_channel_corr": round(float(cross_ch_corr), 4),
        "ch_stds": [round(float(s), 1) for s in ch_stds],
        "residual_std": round(float(res.std()), 1),
    }

def classify_noise(sb):
    """根据 §B 6步数值推断噪声类型，返回 (type, confidence, reason)"""
    # Ordered per SKILL.md §B protocol
    if sb["extreme_pct"] > 0.3:
        return ("IMPULSE", "high", f"extreme_pct={sb['extreme_pct']}% > 0.3%")
    if sb["vm_slope"] > 0.005 and sb["vm_slope"] < 1.0:
        return ("SPECKLE", "high" if sb["vm_slope"] > 0.01 else "medium",
                f"vm_slope={sb['vm_slope']:.4f} > 0.005")
    if sb["var_slope"] > 1.0 and abs(sb["vm_slope"]) < 0.005:
        return ("POISSON", "high", f"var_slope={sb['var_slope']:.1f} > 1.0 + vm_slope≈0")
    if sb["rgb_ratio"] > 1.4:
        if sb.get("cross_channel_corr", 0) > 0.12:
            return ("GAUSSIAN_YCrCb", "high", f"rgb_ratio={sb['rgb_ratio']:.2f} > 1.4 + cross_ch_corr={sb['cross_ch_corr']:.3f}")
    # Default
    if sb["vm_slope"] < 0.005 and sb["var_slope"] < 1.0 and sb["rgb_ratio"] < 1.4:
        return ("GAUSSIAN_RGB", "medium", "all special patterns negative — default gaussian_RGB")
    return ("GAUSSIAN_RGB", "low", "ambiguous — recommend PSNR test against alternatives")

# ── main ──
def main():
    parser = argparse.ArgumentParser(description="Peeled noise diagnostic — §B on stripped residual")
    parser.add_argument("--target", required=True, help="Degraded target image")
    parser.add_argument("--clean", required=True, help="Clean source image")
    parser.add_argument("--pipeline", required=True,
                       help="Deterministic pipeline to strip, e.g. 'compression_jpeg:3,blur_gaussian:2'")
    parser.add_argument("--output", default=None, help="Save peeled residual to file")
    parser.add_argument("--json", default=None, help="Save §B results as JSON")
    args = parser.parse_args()

    target = load(args.target)
    clean = load(args.clean)

    # Apply deterministic pipeline
    simulated = apply_pipeline(clean, args.pipeline)

    # Compute peeled residual
    residual = target - simulated

    if args.output:
        save(residual + 128, args.output)  # offset for visibility
        print(f"Peeled residual saved: {args.output}")

    # Run noise_prior
    import subprocess, tempfile
    tmp_target = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_clean = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    Image.fromarray(np.clip(residual + 128, 0, 255).astype(np.uint8)).save(tmp_target.name)
    Image.fromarray(clean.astype(np.uint8)).save(tmp_clean.name)

    np_result = subprocess.run([
        ".venv/bin/python3",
        os.path.join(os.path.dirname(__file__), "noise_prior.py"),
        "--target", tmp_target.name,
        "--clean", tmp_clean.name,
    ], capture_output=True, text=True)
    os.unlink(tmp_target.name); os.unlink(tmp_clean.name)

    # Compute §B
    sb = compute_sb6(residual)
    noise_type, confidence, reason = classify_noise(sb)

    # ── formatted output ──
    print(f"\n{'='*60}")
    print(f"🔴 §B 6-STEP NOISE DIAGNOSTIC (on peeled residual)")
    print(f"{'='*60}")
    print(f"Pipeline stripped: {args.pipeline}")
    print(f"Residual std: {sb['residual_std']:.1f}")
    print(f"")
    print(f"  [1] extreme_pct:  {sb['extreme_pct']:.2f}%  {'← IMPULSE' if sb['extreme_pct'] > 0.3 else ''}")
    print(f"      zero_ratio:   {sb['zero_ratio']:.4f}")
    print(f"  [2] vm_slope:     {sb['vm_slope']:.4f}     {'← SPECKLE' if sb['vm_slope'] > 0.005 else ''}")
    print(f"  [3] var_slope:    {sb['var_slope']:.4f}     {'← POISSON' if sb['var_slope'] > 1.0 else ''}")
    print(f"      ivc:          {sb['intensity_var_corr']:.4f}")
    print(f"  [4] spatial_corr: {sb['spatial_corr']:.4f}")
    print(f"  [5] rgb_ratio:    {sb['rgb_ratio']:.4f}     {'← YCrCb' if sb['rgb_ratio'] > 1.4 else ''}")
    print(f"      ch_stds:      R={sb['ch_stds'][0]:.1f} G={sb['ch_stds'][1]:.1f} B={sb['ch_stds'][2]:.1f}")
    print(f"  [6] cross_ch_corr:{sb['cross_channel_corr']:.4f}")
    print(f"")
    print(f"  🏆 NOISE TYPE: {noise_type} (confidence={confidence})")
    print(f"     Reason: {reason}")
    print(f"")
    print(f"{'='*60}")
    print(f"  Recommended actions:")
    if noise_type == "IMPULSE":
        print(f"    → Test noise_impulse severity 1-3, use extreme_pct to calibrate")
    elif noise_type == "SPECKLE":
        print(f"    → Test noise_speckle severity 1-4, use vm_slope magnitude to calibrate")
    elif noise_type == "POISSON":
        print(f"    → Test noise_poisson severity 1-5, use residual_std to calibrate")
    elif noise_type == "GAUSSIAN_YCrCb":
        print(f"    → Test noise_gaussian_YCrCb severity 1-5, use ch_stds to calibrate")
    else:
        print(f"    → noise_gaussian_RGB is default. If PSNR < 30, test noise_impulse:1-3 as alternative")
    print(f"    → If sigma<5 (noise_prior output below), consider NO_NOISE")
    print(f"{'='*60}\n")

    # noise_prior output
    for line in np_result.stdout.strip().split("\n"):
        if "sigma" in line.lower() or "channel" in line.lower() or "gaussian" in line.lower():
            print(f"  [noise_prior] {line.strip()}")

    if args.json:
        json.dump(sb, open(args.json, 'w'), indent=2)

if __name__ == "__main__":
    main()
