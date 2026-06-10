# exp1 — 退化管线下的架构与训练策略探索

## 目标

在固定退化管线 `blur_motion(3) → noise_spatially_correlated(2) → compression_jpeg(4)` 下最大化验证 PSNR。首次系统探索模型架构和训练超参。

## 实验规模

Phase 1: 32 组 | Phase 5: 72 组 | 总计 104 组

## 核心结论

1. **恒定 LR 优于余弦调度**（+0.44 dB）——短训练下模型未收敛，衰减 LR 减少有效步数
2. **更宽优于更深**——EMBED_DIM=96 > 增加层数
3. **改进效果不叠加**——组合优化不如单点优化
4. **window_size=16 是最大免费增益**（+0.50 dB，零参数成本）
5. **Sobel 边缘 loss 与 bfloat16 AMP 不兼容**，导致崩溃
6. **梯度裁剪 (max_norm=1.0)** 提供 +0.33 dB

## 最佳结果

Y3 配置 (window_size=16, embed_dim=96, heads=8, clip, cosine LR): **21.59 dB**（+2.71 vs M_blind 18.88 dB）
