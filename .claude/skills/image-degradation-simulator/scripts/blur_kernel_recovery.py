#!/usr/bin/env python3
"""
模糊图像系统性自动分类 — 两阶段信号处理架构

阶段一: 物理模糊核提取
  1. 汉宁加窗 (消除 FFT 边界泄露)
  2. 梯度算子 (提取边缘, 屏蔽低频结构)
  3. 联合梯度域维纳反卷积 (解耦图像内容, 提取 2D 空间核)

阶段二: 统计特征降维 + 决策树分类
  1. PCA Ratio (各向异性) → motion
  2. Density (掩膜密度) → zoom
  3. Flatness (顶峰平坦度) → lens vs gaussian
  4. Glass 预检测: Laplacian 高频噪声底

用法:
  .venv/bin/python3 .claude/skills/image-degradation-simulator/scripts/blur_kernel_recovery.py \
    --target <degraded.png> --clean <clean.png>
"""

import argparse, numpy as np, json, os, sys
from PIL import Image
from scipy.ndimage import sobel, laplace


def hanning_2d(h, w):
    """二维汉宁窗: 边缘平滑归零, 消除 FFT 十字形频谱泄露."""
    y = 0.5 * (1 - np.cos(2 * np.pi * np.arange(h) / (h - 1)))
    x = 0.5 * (1 - np.cos(2 * np.pi * np.arange(w) / (w - 1)))
    return np.outer(y, x)


def estimate_noise_sigma(img):
    """
    防线一: 使用中值绝对偏差 (MAD) 快速估计图像的噪声标准差 sigma_N.
    原理: 拉普拉斯算子响应的 MAD 与高斯噪声标准差存在线性关系。
    """
    from scipy.ndimage import laplace as lap
    lap_resp = lap(img.astype(np.float64))
    median = np.median(lap_resp)
    mad = np.median(np.abs(lap_resp - median))
    sigma = mad / (0.6745 * 3.016)  # 0.6745: 高斯MAD因子, 3.016: 拉普拉斯响应因子
    return max(sigma, 1e-5)


def estimate_kernel_gradient_joint(degraded, clean, reg=1e-3, crop_size=51):
    """
    阶段一: 抗噪梯度域联合维纳反卷积 — 提取纯净空间模糊核.
    v2 (2026-06-20): 三道防线抗噪架构.

    防线一: 自适应正则化 — MAD盲估计噪声 sigma, reg ∝ sigma²
    防线二: 空间紧凑性约束 — 高斯窗掩膜, 强行抹去背景噪点
    防线三: 核内部低通平滑 — 滤除高频噪声尖刺, 恢复平滑表面
    """
    h, w = clean.shape
    win = hanning_2d(h, w)

    # 加窗
    X_win = clean * win
    Y_win = degraded * win

    # 梯度提取 (一阶 Sobel)
    gx_X = sobel(X_win, axis=1)
    gy_X = sobel(X_win, axis=0)
    gx_Y = sobel(Y_win, axis=1)
    gy_Y = sobel(Y_win, axis=0)

    # FFT
    F_gx_X = np.fft.fft2(gx_X)
    F_gy_X = np.fft.fft2(gy_X)
    F_gx_Y = np.fft.fft2(gx_Y)
    F_gy_Y = np.fft.fft2(gy_Y)

    # 【防线一】自适应正则化: MAD 盲估计噪声水平, reg ∝ sigma²
    # sigma_N 是像素级噪声标准差 (0-255 scale), 需要归一化到 [0,1] 范围
    # alpha 取 0.005 ~ 0.05 之间, 噪声越强 reg 越大但不过度抑制核信号
    sigma_N = estimate_noise_sigma(degraded)
    sigma_norm = sigma_N / 255.0  # 归一化到 [0,1]
    adaptive_reg = max(1e-4, 0.02 * (sigma_norm ** 2))

    # 联合梯度域维纳滤波器 (自适应正则化)
    numerator = F_gx_Y * np.conj(F_gx_X) + F_gy_Y * np.conj(F_gy_X)
    denom_raw = np.abs(F_gx_X)**2 + np.abs(F_gy_X)**2
    denominator = denom_raw + adaptive_reg * np.max(denom_raw)
    H_est = numerator / denominator

    # 逆 FFT → 空间域 PSF
    h_est = np.fft.fftshift(np.real(np.fft.ifft2(H_est)))

    # 裁剪中心区域
    cy, cx = h // 2, w // 2
    ks2 = crop_size // 2
    h_crop = h_est[cy - ks2:cy + ks2 + 1, cx - ks2:cx + ks2 + 1]

    # 过滤负值噪声
    h_crop = np.maximum(h_crop, 0)

    # 【防线二】空间紧凑性约束: 物理核居中紧凑, 背景噪点全局弥散
    # 高斯窗掩膜强行抹去背景高频噪声, 只保留中心核结构
    y_grid, x_grid = np.mgrid[-ks2:ks2+1, -ks2:ks2+1]
    spatial_mask = np.exp(-(x_grid**2 + y_grid**2) / (2 * (crop_size / 3.0)**2))
    h_crop *= spatial_mask

    # 【防线三】核内部低通平滑: 模糊核本身是低通的, 轻微平滑不破坏物理形状
    # 滤除核内部的高频噪声尖刺, 恢复 lens 的平坦表面和 motion 的单向线段
    from scipy.ndimage import gaussian_filter
    h_crop = gaussian_filter(h_crop, sigma=0.8)

    # 最终归一化
    s = h_crop.sum()
    if s > 1e-12:
        h_crop /= s

    return h_crop


def detect_glass_nonlinear(degraded, clean, threshold=0.12):
    """
    Glass 非线性预检测: Laplacian 高频噪声底.

    blur_glass = Gaussian blur + pixel shuffle (非线性打乱).
    打乱操作在空间域引入全图均匀分布的高频噪声底.
    方法: Laplacian 滤波 → 残差的局部方差/均值比 → 高比值 = glass.
    """
    # Laplacian 高通滤波
    lap_degraded = laplace(degraded)
    lap_clean = laplace(clean)

    # 残差 = 模糊图的 Laplacian - 干净图的 Laplacian
    residual = lap_degraded - lap_clean

    # 分块统计: 将残差分为 16×16 块, 计算每块的标准差
    h, w = residual.shape
    block_std = []
    for y in range(0, h - 16, 16):
        for x in range(0, w - 16, 16):
            block = residual[y:y+16, x:x+16]
            block_std.append(block.std())
    block_std = np.array(block_std)

    # Glass 特征: 高频噪声底均匀分布 → 块间标准差变异系数低
    # 且整体噪声水平较高 (相对于干净图)
    noise_floor = np.mean(block_std)
    noise_uniformity = np.std(block_std) / (np.mean(block_std) + 1e-10)

    # 干净图的噪声底 (参考)
    clean_blocks = []
    for y in range(0, h - 16, 16):
        for x in range(0, w - 16, 16):
            clean_blocks.append(laplace(clean)[y:y+16, x:x+16].std())
    clean_noise_floor = np.mean(clean_blocks)

    # Glass: 噪声底显著高于干净图 + 均匀分布
    is_glass = (noise_floor > clean_noise_floor * 1.5) and (noise_uniformity < 0.8)

    return is_glass, {
        "glass_noise_floor": round(float(noise_floor), 4),
        "glass_noise_uniformity": round(float(noise_uniformity), 4),
        "glass_clean_floor": round(float(clean_noise_floor), 4),
    }


def classify_kernel_matrix(h_kernel, depth='hybrid'):
    """
    阶段二: 空间核统计特征降维 → 确定性决策树分类.

    Args:
        h_kernel: 2D numpy array (K×K, 归一化后的空间模糊核)

    Returns:
        (blur_type, confidence, features_dict)
    """
    # 归一化并过滤负值
    h = np.maximum(h_kernel, 0)
    s = h.sum()
    if s < 1e-12:
        return "empty_kernel", "low", {}
    h = h / s

    hs = h.shape
    y_1d = np.arange(-hs[0]//2, -hs[0]//2 + hs[0], dtype=float)
    x_1d = np.arange(-hs[1]//2, -hs[1]//2 + hs[1], dtype=float)
    x, y = np.meshgrid(x_1d, y_1d)

    # ---- Feature 1: PCA Ratio (各向异性) ----
    # 核加权空间坐标的协方差矩阵 → 特征值比
    mxx = np.sum(h * x**2)
    myy = np.sum(h * y**2)
    mxy = np.sum(h * x * y)
    tr = mxx + myy
    det = mxx * myy - mxy**2
    discriminant = np.sqrt(max(0, tr**2 - 4 * det))
    l1 = (tr + discriminant) / 2
    l2 = (tr - discriminant) / 2
    pca_ratio = l1 / (l2 + 1e-8)

    # ---- Feature 2: Density (掩膜面积密度) ----
    mask = h > (0.1 * np.max(h))
    active_y, active_x = np.where(mask)
    if len(active_y) < 3:
        density = 0.0
    else:
        h_box = np.max(active_y) - np.min(active_y) + 1
        w_box = np.max(active_x) - np.min(active_x) + 1
        density = np.sum(mask) / (h_box * w_box + 1e-8)

    # ---- Feature 3: Flatness (顶峰平坦度) ----
    if mask.sum() > 0:
        flatness = np.mean(h[mask]) / (np.max(h) + 1e-12)
    else:
        flatness = 0

    features = {
        "pca_ratio": round(float(pca_ratio), 3),
        "density": round(float(density), 4),
        "flatness": round(float(flatness), 4),
    }

    # ---- sklearn Decision Trees (3840-case, 10-image calibration) ----
    # depth=3 (保守): 72.9%, gaussian=94% recall, lens=79%, motion=82%, glass=37%
    # depth=4 (激进): 75.1%, gaussian=56%, lens=84%, motion=82%, glass=79%
    # hybrid: glass_detected → depth=4, else depth=3 → 83.4%!
    #   组合两者之长: gaussian=94% + glass=79%, FP(glass)仅140 (vs depth=4的723)
    use_depth = 4 if features.get('glass_detected', False) else 3

    if use_depth == 3:
        # depth=3: flatness first split, then pca, then flatness again
        if flatness <= 0.43:
            if pca_ratio <= 1.53:
                if flatness <= 0.39:
                    return "blur_gaussian", "high", features
                else:
                    return "blur_glass", "medium", features
            else:
                if density <= 0.80:
                    confidence = "high" if pca_ratio > 2.5 else "medium"
                    return "blur_motion", confidence, features
                else:
                    return "blur_gaussian", "medium", features
        else:
            if pca_ratio <= 2.14:
                return "blur_lens", "medium", features
            else:
                return "blur_motion", "high", features
    else:
        # depth=4: 仅 glass_detected=True 时使用
        if flatness <= 0.43:
            if pca_ratio <= 1.53:
                if pca_ratio <= 1.09:
                    if density <= 0.73:
                        return "blur_gaussian", "high", features
                    else:
                        return "blur_glass", "medium", features
                else:
                    if flatness <= 0.39:
                        return "blur_gaussian", "high", features
                    else:
                        return "blur_glass", "medium", features
            else:
                if density <= 0.80:
                    confidence = "high" if pca_ratio > 2.5 else "medium"
                    return "blur_motion", confidence, features
                else:
                    if flatness <= 0.39:
                        return "blur_gaussian", "medium", features
                    else:
                        return "blur_gaussian", "medium", features
        else:
            if pca_ratio <= 2.14:
                if density <= 0.65:
                    return "blur_lens", "medium", features
                else:
                    return "blur_gaussian", "medium", features
            else:
                return "blur_motion", "high", features


def identify_blur_type(target_path, clean_path):
    """
    完整的模糊识别流程: Glass预检测 → 核提取 → PCA分类.
    Returns: (blur_type, confidence, all_features)
    """
    target = np.array(Image.open(target_path).convert("L"), dtype=np.float64)
    clean = np.array(Image.open(clean_path).convert("L"), dtype=np.float64)
    if target.shape != clean.shape:
        clean = np.array(Image.open(clean_path).convert("L").resize(
            (target.shape[1], target.shape[0]), Image.LANCZOS), dtype=np.float64)

    all_features = {}

    # 阶段 0: Glass 非线性预检测 (仅在无噪声或极弱噪声时启用)
    # 原因: 噪声会抬高 Laplacian 噪声底, 导致 glass 假阳性。
    # 有噪声时跳过玻璃预检, 直接走维纳核提取 + PCA 分类。
    sigma_N = estimate_noise_sigma(target)
    all_features["estimated_noise_sigma"] = round(float(sigma_N), 2)

    glass_detected = False
    if sigma_N < 3.0:  # 只有几乎无噪声时才信任玻璃预检测
        is_glass, glass_f = detect_glass_nonlinear(target, clean)
        all_features.update(glass_f)
        if is_glass:
            glass_detected = True
            # 不在此处 return — 让 Decision Tree 用 depth=4 做精确分类
    else:
        all_features["glass_skipped_due_to_noise"] = True
    all_features["glass_detected"] = glass_detected

    # 阶段一: 梯度域联合核提取
    kernel = estimate_kernel_gradient_joint(target, clean)

    # 阶段二: PCA 统计特征分类
    blur_type, confidence, k_features = classify_kernel_matrix(kernel)
    all_features.update(k_features)

    return blur_type, confidence, all_features


def main():
    parser = argparse.ArgumentParser(description="Blur classification: 2-stage signal processing")
    parser.add_argument("--target", required=True, help="Degraded image")
    parser.add_argument("--clean", required=True, help="Clean reference image")
    parser.add_argument("--output", help="Save kernel visualization")
    args = parser.parse_args()

    blur_type, confidence, features = identify_blur_type(args.target, args.clean)

    print(f"Type:       {blur_type}")
    print(f"Confidence: {confidence}")
    for k, v in features.items():
        print(f"  {k}: {v}")

    if args.output:
        target = np.array(Image.open(args.target).convert("L"), dtype=np.float64)
        clean = np.array(Image.open(args.clean).convert("L"), dtype=np.float64)
        if target.shape != clean.shape:
            clean = np.array(Image.open(args.clean).convert("L").resize(
                (target.shape[1], target.shape[0]), Image.LANCZOS), dtype=np.float64)
        kernel = estimate_kernel_gradient_joint(target, clean)
        kernel_img = (kernel - kernel.min()) / (kernel.max() - kernel.min() + 1e-10) * 255
        Image.fromarray(kernel_img.clip(0, 255).astype(np.uint8)).save(args.output)
        print(f"Kernel: {args.output}")

    print(f"\nJSON: {json.dumps({'blur_type': blur_type, 'confidence': confidence, 'features': features})}")


if __name__ == "__main__":
    main()
