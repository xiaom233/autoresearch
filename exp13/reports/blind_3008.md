# blind_3008 — 3退化

## 退化管线

| 来源 | Pipeline |
|------|----------|
| GT (ground truth) | noise_gaussian_RGB(2) → compression_jpeg(1) → blur_gaussian(1) |
| 盲识别预测 | compression_jpeg(3) → noise_gaussian_YCrCb(2) → blur_gaussian(1) |

**识别质量**: GOOD (CI pass 8/10) | 函数匹配 2/3

## 训练结果 (EPOCH=2, VAL=GT)

| 模型 | 训练参数 | PSNR | Δ vs M_blind |
|------|---------|:--:|:--:|
| M_blind (盲基线) | 随机退化 | 26.1663 | — |
| Spec (从零训练) | PARAMS=predicted | 28.06 | +1.89 |
| Ft (微调) | PARAMS=predicted + LOAD_CKPT | 28.14 | +1.97 |

## 分析

- **Spec vs Ft**: Δ = -0.08 dB
- **最佳模型**: Ft(pred) (28.14 dB)
- **识别匹配度**: 2/3 函数匹配
- Spec ≈ Ft (-0.08 dB)，盲识别质量高时两种策略等价
