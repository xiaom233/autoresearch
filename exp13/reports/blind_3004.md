# blind_3004 — 3退化

## 退化管线

| 来源 | Pipeline |
|------|----------|
| GT (ground truth) | noise_gaussian_YCrCb(1) → blur_jitter(1) → compression_jpeg_2000(1) |
| 盲识别预测 | oversharpen(2) → brightness_darken_shfit_RGB(1) → compression_jpeg(1) |

**识别质量**: NEEDS_WORK (CI pass 6/10) | 函数匹配 0/3

## 训练结果 (EPOCH=2, VAL=GT)

| 模型 | 训练参数 | PSNR | Δ vs M_blind |
|------|---------|:--:|:--:|
| M_blind (盲基线) | 随机退化 | 25.4946 | — |
| Spec (从零训练) | PARAMS=predicted | 18.21 | -7.28 |
| Ft (微调) | PARAMS=predicted + LOAD_CKPT | 18.41 | -7.08 |

## 分析

- **Spec vs Ft**: Δ = -0.20 dB
- **最佳模型**: M_blind(GT) (25.4946 dB)
- **识别匹配度**: 0/3 函数匹配
- Spec ≈ Ft (-0.20 dB)，盲识别质量高时两种策略等价
- ⚠️ 盲识别需要改进 (CI=6/10)，影响了下游训练效果
